"""Allocation-driven SISO links using the common simulation environment."""
import numpy as np
import torch
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.channel.tr38901 import PanelArray, UMi
from sionna.phy.ofdm import ResourceGrid

from .. import utils as ut
from ..modules.environment import SimulationEnvironment
from ..modules.transmission import SisoPhyAbstraction
from . import parameters as anomaly_params
from .parameters import StepResult


class AnomalyTrackingScenario(SimulationEnvironment):
    """Generate UE geometry through the base environment, without ray tracing.

    Call setup_scenario(), then setup_modules(), then step(allocation).
    UEs are stationary by default; explicit mobility and restored trajectories
    use the base environment's geometry update. One occupied slot is one batch
    example with a single UE/AP antenna and a single transport block.
    """

    shared_type = anomaly_params.SharedParams
    wireless_type = anomaly_params.WirelessParams

    def __init__(self, *args, **kwargs):
        if kwargs.get("load_scene", False) or kwargs.get("scene_name") is not None:
            raise ValueError("AnomalyTrackingScenario uses load_scene=False and no RT scene")
        kwargs["load_scene"] = False
        super().__init__(*args, **kwargs)
        self._scenario_ready = False
        self._modules_ready = False

    def ue_random_positioning(self, height=anomaly_params.UE_HEIGHT,
                              min_dist=np.ones(2), margin=0.95):
        """Use the shared placement algorithm with the anomaly UE height."""
        return super().ue_random_positioning(height, min_dist, margin)

    def setup_scenario(self, ap_pos=None, antenna_spacing=None, init_ue_pos=None,
                       ue_vel_mean=anomaly_params.VEL_MAG,
                       ue_vel_cov=anomaly_params.VEL_STD):
        """Generate or restore base geometry; default AP height is 10 m (UMi).

        A zero velocity covariance must remain exactly zero, including the
        vertical component, so the default geometry is stationary.
        """
        super().setup_scenario(
            ap_pos=10.0 if ap_pos is None else ap_pos,
            antenna_spacing=antenna_spacing, init_ue_pos=init_ue_pos,
            ue_vel_mean=ue_vel_mean, ue_vel_cov=ue_vel_cov)
        if not self._load_env and np.all(np.asarray(ue_vel_cov) == 0):
            self.geo.ue_vel_cov.fill(0)
        self._scenario_ready = True

    def setup_modules(self, ra_algorithm=None, transmission_module=None,
                      tracker_module=None, performance=None, *, device="cpu"):
        """Wire the shared module slots and create a geometry-based UMi channel.

        Allocation is supplied externally, so no allocator/tracker is required.
        Optional injected modules are retained using the base setup_modules API.
        """
        if not self._scenario_ready:
            raise RuntimeError("Call setup_scenario() before setup_modules()")
        self.device = torch.device(device)
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self.T, fft_size=self.F,
            subcarrier_spacing=self.wireless.sc_spacing,
            num_tx=1, num_streams_per_tx=1,
            cyclic_prefix_length=int(round(self.wireless.cp_time * self.wireless.bw)),
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[0] if self.T < 6 else [2, self.T - 3],
            device=str(self.device))
        array_kwargs = dict(num_rows_per_panel=1, num_cols_per_panel=1,
                            polarization="single", polarization_type="V",
                            antenna_pattern="omni", carrier_frequency=self.wireless.carrier,
                            device=str(self.device))
        self.channel_model = UMi(
            carrier_frequency=self.wireless.carrier, o2i_model="low",
            ut_array=PanelArray(**array_kwargs), bs_array=PanelArray(**array_kwargs),
            direction=self.wireless.direction, enable_pathloss=True,
            enable_shadow_fading=self.wireless.enable_shadow_fading,
            device=str(self.device))
        self.channel_generator = GenerateOFDMChannel(
            self.channel_model, self.resource_grid,
            normalize_channel=False, device=str(self.device))
        if transmission_module is None:
            transmission_module = SisoPhyAbstraction(
                self.resource_grid, self.device, bler_target=self.wireless.bler_target,
                direction=self.wireless.direction, mcs_table_index=self.wireless.mcs_table_index)
        super().setup_modules(ra_algorithm, transmission_module, tracker_module, performance)
        self._topology_batch_size = None
        self._modules_ready = True

    def _set_channel_topology(self, ue_ids):
        """One actual AP/UE pair per occupied slot, from the current frame."""
        slots = ue_ids.size
        # System-level model buffers need resetting when the occupied-slot batch
        # size changes between frames. Available on supported Sionna 2.x models.
        if self._topology_batch_size != slots:
            self.channel_model.reset_topology()
        self._topology_batch_size = slots
        def tensor(value):
            return torch.as_tensor(np.array(value, copy=True), dtype=torch.float32,
                                   device=self.device)

        ut_loc = tensor(self.geo.ue_location[self.frame, ue_ids])[:, None, :]
        bs_loc = tensor(np.broadcast_to(self.geo.ap_location[0], (slots, 1, 3)))
        velocity = tensor(self.geo.ue_velocity[self.frame, ue_ids])[:, None, :]
        # Fixed vertically polarized omnidirectional elements. The RT array's
        # +Z-boresight rotation is not a PHY PanelArray orientation (+X boresight).
        self.channel_model.set_topology(
            ut_loc=ut_loc, bs_loc=bs_loc,
            ut_orientations=torch.zeros_like(ut_loc),
            bs_orientations=torch.zeros_like(bs_loc),
            ut_velocities=velocity,
            in_state=torch.zeros((slots, 1), dtype=torch.bool, device=self.device),
            los=self.wireless.force_los)

    def _estimate_channel(self, channel):
        """Add independent circular complex Gaussian estimation error.

        Variance is absolute E[abs(error)**2] per CFR coefficient, after path
        loss, and defaults to zero. Zero variance consumes no random samples.
        """
        variance = self.wireless.channel_estimation_error_variance
        if not np.isfinite(variance) or variance < 0:
            raise ValueError("channel_estimation_error_variance must be finite and nonnegative")
        if variance == 0 or channel.numel() == 0:
            return channel.clone()
        error = (self.rng.standard_normal(tuple(channel.shape))
                 + 1j * self.rng.standard_normal(tuple(channel.shape))) * np.sqrt(variance / 2)
        return channel + torch.as_tensor(error, dtype=channel.dtype, device=self.device)

    @torch.no_grad()
    def step(self, allocation) -> StepResult:
        """Advance geometry, then decode one transport block per occupied slot.

        Allocation contains ordered zero-based UE IDs (-1 = idle), up to the
        configured frame's slot count. Collisions must be resolved by the MAC.
        """
        if not self._modules_ready:
            raise RuntimeError("Call setup_scenario() and setup_modules() before step()")
        allocation = np.asarray(allocation)
        if allocation.ndim != 1 or (allocation.size and (
                not np.issubdtype(allocation.dtype, np.integer)
                or np.issubdtype(allocation.dtype, np.bool_))):
            raise ValueError("allocation must be a 1D sequence of integer UE IDs")
        if allocation.size > self.S:
            raise ValueError("allocation exceeds num_slots")
        if np.any((allocation < -1) | (allocation >= self.U)):
            raise ValueError("UE IDs must lie in [0,num_ue); use -1 for idle")
        if self.frame + 1 >= self.K:
            raise RuntimeError("All configured frames have already been simulated")
        allocation = allocation.astype(np.int64, copy=True)
        slot_indices = np.flatnonzero(allocation >= 0)
        ue_ids = allocation[slot_indices]
        # The base lifecycle starts at -1. Frame zero is written before sampling.
        self.frame += 1
        self.move_ue()
        shape = (ue_ids.size, 1, 1, 1, 1, self.T, self.F)
        if ue_ids.size:
            self._set_channel_topology(ue_ids)
            channel = self.channel_generator(batch_size=ue_ids.size)
            if tuple(channel.shape) != shape:
                raise ValueError(f"Channel generator returned {tuple(channel.shape)}, expected {shape}")
        else:
            channel = torch.empty(shape, dtype=torch.complex64, device=self.device)
        channel_estimate = self._estimate_channel(channel)
        # Both available power and total-band noise are divided by F, giving
        # per-RE quantities with the same wideband SNR as the shared parameters.
        if self.wireless.direction == "uplink":
            power = np.asarray(self.wireless.ue_power)[ue_ids]
            noise = self.wireless.nob_ap
        else:
            power = np.full(ue_ids.size, np.asarray(self.wireless.ap_power)[0])
            noise = self.wireless.nob_ue
        transmission = self.transmission_module(
            channel, tx_power=power / self.F, no=noise / self.F,
            channel_estimate=channel_estimate,
            error_variance=self.wireless.channel_estimation_error_variance)
        return StepResult(allocation, slot_indices, ue_ids, channel, transmission,
                          channel_estimate, self.wireless.channel_estimation_error_variance,
                          self.frame)


def run_episode(episode, allocations, env_kwargs=None, device="cpu") -> list[StepResult]:
    """Run an external allocation sequence through the shared environment lifecycle.

    ``allocations`` yields one vector per frame. No MAC decisions or semantic
    tracking metrics are fabricated here. Save the returned StepResults in the
    caller, together with env.c/wireless/geo when managing a scenario directly.
    """
    env_kwargs = dict(env_kwargs or {})
    ut.validate_kwargs(
        env_kwargs, AnomalyTrackingScenario.shared_type, AnomalyTrackingScenario.wireless_type,
        SimulationEnvironment, AnomalyTrackingScenario.setup_scenario,
        exclude={"self", "episode", "shared_params", "wireless_params", "c"}, name="env_kwargs")
    constructor = ut.filter_call_kwargs(SimulationEnvironment, **env_kwargs)
    constructor.setdefault("load_env", False)
    env = AnomalyTrackingScenario(
        episode, ut.filter_call_kwargs(AnomalyTrackingScenario.shared_type, **env_kwargs),
        ut.filter_call_kwargs(AnomalyTrackingScenario.wireless_type, **env_kwargs), **constructor)
    env.setup_scenario(**ut.filter_call_kwargs(env.setup_scenario, **env_kwargs))
    env.setup_modules(device=device)
    return [env.step(allocation) for allocation in allocations]


if __name__ == "__main__":
    results = run_episode(1, [np.array([0, -1, 1])] * 2,
                          env_kwargs={"num_ue": 3, "num_slots": 3, "num_frames": 2})
    print([result.decoded for result in results])
