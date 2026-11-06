"""Allocation-driven SISO links using the common simulation environment."""
import numpy as np
import torch
from shapely.geometry import MultiPolygon
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.channel.tr38901 import PanelArray, UMi
from sionna.phy import config as sionna_config
from phy_mac_isac.anomaly_tracking.allocation import TDDResourceAllocation
from .. import utils as ut
from ..modules.environment import SimulationEnvironment
from ..modules.transmission import SisoPhyAbstraction
from . import parameters as anomaly_params


class AnomalyTrackingScenario(SimulationEnvironment):
    """Generate UE geometry through the base environment, without ray tracing.

    Call setup_scenario(), then setup_modules(), then step(allocation).
    UEs are stationary by default; explicit mobility and restored trajectories
    use the base environment's geometry update. One occupied slot is one batch
    example with a single UE/AP antenna and a single transport block.
    """

    shared_type = anomaly_params.SharedParams
    wireless_type = anomaly_params.WirelessParams
    metrics_type = anomaly_params.Metrics

    def __init__(self, *args, **kwargs):
        if kwargs.get("load_scene", False) or kwargs.get("scene_name") is not None:
            raise ValueError("AnomalyTrackingScenario uses load_scene=False and no RT scene")
        kwargs["load_scene"] = False
        super().__init__(*args, **kwargs)

        # Internal Attribute
        self.channel_model = None
        self.channel_generator = None
        self._topology_batch_size = None
        self._perfect_csi = kwargs.get("perfect_csi", True)

    def _build_base_scene(self,
                          scene_name: str | None,
                          area_coordinates: list | np.ndarray | None = None,
                          default_room_side: float = anomaly_params.ROOM_SIDE,
                          ) -> tuple[None, list | np.ndarray, str | None]:
        """Build the base propagation scene with the new default room side."""
        return super()._build_base_scene(scene_name, area_coordinates, default_room_side)

    def ue_random_positioning(self,
                              height: float | np.ndarray = anomaly_params.UE_HEIGHT,
                              min_dist: np.ndarray = 10 * np.ones(2),
                              margin: float = 0.99,
                              exclusion_centers: np.ndarray | None = None,
                              exclusion_radius: float = 10.):
        """First generate the cluster center randomly. Then, place the UEs around them randomly.
        Finally, place the unclustered UEs randomly throughout the whole environment.
        """
        # Check if height is scalar and convert to array if necessary
        if np.isscalar(height):
            height = height * np.ones((self.U, 1))

        # Sample the center first
        cluster_centroids, cells = ut.random_2d_positioning(
            self.geo.environment_shape,
            n=self.c.num_clusters,
            min_dist=min_dist,
            margin=margin,
            exclusion_centers=exclusion_centers,
            exclusion_radius=exclusion_radius,
            rng=self.rng
        )

        # Now sample the device of each cluster
        coords = []
        for (centroid, cell) in zip(cluster_centroids, cells):
            coords.append(
                ut.random_2d_positioning(
                    cell,
                    n=self.c.cluster_size,
                    min_dist=np.ones(2),
                    margin=1.,
                    exclusion_centers=exclusion_centers,
                    exclusion_radius=exclusion_radius,
                    rng=self.rng
                )[0]
            )

        clustered_coords = np.vstack(coords)

        # Check how many UE are left
        unclustered_ues = self.c.num_ue - clustered_coords.shape[0]
        if unclustered_ues:
            # Now sample the unclustered device randomly
            unclustered_coords = ut.random_2d_positioning(
                self.geo.environment_shape,
                n=unclustered_ues,
                min_dist=np.ones(2),
                margin=margin,
                exclusion_centers=clustered_coords,
                exclusion_radius=1,
                rng=self.rng
            )[0]
        else:
            unclustered_coords = []

        coords = np.vstack([clustered_coords, unclustered_coords])

        # The following can be useful for visualization and to check which user is in which cluster
        self.geo.update(semantic_regions=MultiPolygon(cells))

        return np.hstack((coords, height))

    def setup_scenario(self,
                       ap_pos: float | np.ndarray | None = None,
                       antenna_spacing: float | None = None,
                       init_ue_pos: np.ndarray | None = None,
                       ue_vel_mean: np.ndarray | float= 0.,
                       ue_vel_cov: np.ndarray | float = 0.,
                       ue_ap_min_distance: float = 10.):
        """Generate or restore base geometry; default AP height is 10 m (UMi)."""
        super().setup_scenario(
            ap_pos=np.array([[0., 0., 10.0]]) if ap_pos is None else ap_pos,
            antenna_spacing=antenna_spacing,
            init_ue_pos=init_ue_pos,
            ue_vel_mean=ue_vel_mean,
            ue_vel_cov=ue_vel_cov,
            ue_ap_min_distance=ue_ap_min_distance)

    def setup_modules(self,
                      ra_module = None,
                      transmission_module = None,
                      tracker_module = None,
                      performance = None,
                      device: torch.device | str = "cpu"):
        """Wire the shared module slots and create a geometry-based UMi channel.

        Allocation is supplied externally, so no allocator/tracker is required.
        Optional injected modules are retained using the base setup_modules API.
        """
        # Put default modules
        if ra_module is None:
            ra_module = TDDResourceAllocation(self.c, self.wireless)
        if transmission_module is None:
            transmission_module = SisoPhyAbstraction(self.wireless, ra_module, device=device)

        super().setup_modules(ra_module, transmission_module, tracker_module, performance)

        self.channel_model = UMi(
            carrier_frequency=self.wireless.carrier, o2i_model="low",
            ut_array=self._omni_antenna(self.geo.antenna_spacing),
            bs_array=self._omni_antenna(self.geo.antenna_spacing),
            direction=self.wireless.direction,
            enable_pathloss=True,
            enable_shadow_fading=self.wireless.enable_shadow_fading,
        )

        self.channel_generator = GenerateOFDMChannel(
            self.channel_model,
            self.ra_module.rg,
            normalize_channel=False
        )

        if not self._perfect_csi:
            self.set_ls_estimation_error()

    def _set_channel_topology(self, ue_ids):
        """One actual AP/UE pair per occupied slot, from the current frame."""
        slots = ue_ids.size
        # Sionna 2.x system-level model buffers need resetting when the occupied-slot batch size changes between frames.
        if self._topology_batch_size != slots:
            self.channel_model.reset_topology()
        self._topology_batch_size = slots
        def tensor(value):
            return torch.as_tensor(np.array(value, copy=True), dtype=torch.float32)

        # The important thing here is that the UE location and velocity is assumed per batch (1 UE per batch)
        ut_loc = tensor(self.geo.ue_location[self.frame, ue_ids])[:, None, :]
        bs_loc = tensor(np.broadcast_to(self.geo.ap_location[0], (slots, 1, 3)))
        velocity = tensor(self.geo.ue_velocity[self.frame, ue_ids])[:, None, :]
        # Fixed vertically polarized omnidirectional elements.
        self.channel_model.set_topology(
            ut_loc=ut_loc,
            bs_loc=bs_loc,
            ut_orientations=torch.zeros_like(ut_loc),   # It is fine for iso antennas
            bs_orientations=torch.zeros_like(bs_loc),   # It is fine for iso antennas
            ut_velocities=velocity,
            in_state=torch.zeros((slots, 1), dtype=torch.bool),
            los=self.wireless.force_los
        )

    def set_ls_estimation_error(self):
        """Compute the variance of the least-square channel estimation assuming no pilot contamination"""
        variance = np.mean(self.wireless.nob_ap /  self.wireless.ue_power / len(self.ra_module.pilot_ofdm_symbol_indices()))
        self.wireless.update(ce_error_variance=variance)

    def _estimate_channel(self, channel):
        """Add independent circular complex Gaussian estimation error.

        Variance is absolute E[abs(error)**2] per CFR coefficient, after path
        loss, and defaults to zero. Zero variance consumes no random samples.
        """
        variance = self.wireless.ce_error_variance
        if not np.isfinite(variance) or variance < 0:
            raise ValueError("channel_estimation_error_variance must be finite and nonnegative")
        if variance == 0 or channel.numel() == 0:
            return channel.clone()
        error = (self.rng.standard_normal(tuple(channel.shape)) + 1j * self.rng.standard_normal(tuple(channel.shape))) * np.sqrt(variance / 2)
        return channel + torch.as_tensor(error, dtype=channel.dtype)

    @torch.no_grad()
    def step(self, allocation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance geometry, then decode one transport block per occupied slot.

        Allocation contains ordered zero-based UE IDs (-1 = idle), up to the
        configured frame's slot count. Collisions must be resolved by the MAC.
        """
        # Check module readiness
        if not self._modules_ready:
            raise RuntimeError("Call setup_scenario() and setup_modules() before step()")

        # Check frames
        if self.frame + 1 >= self.K:
            raise RuntimeError("All configured frames have already been simulated")

        # Check allocation vector
        allocation = np.asarray(allocation)
        if allocation.ndim != 1 or not np.issubdtype(allocation.dtype, np.integer):
            raise ValueError("allocation must be a 1D sequence of integer UE IDs")
        if np.any((allocation < -1) | (allocation >= self.U)):
            raise ValueError("UE IDs must lie in [0,num_ue); use -1 for idle")

        # Start allocation toward channel generation
        slot_indices = np.flatnonzero(allocation >= 0)
        ue_idx = allocation[slot_indices]

        # Frame starts now (The first value of frame is -1, i.e., the initialization frame.)
        self.frame += 1
        self.move_ue()      # By default, movement is not considered

        ## Generate channel
        # Channel shape desired
        shape = (ue_idx.size, 1, 1, 1, 1, self.T, self.F)
        if ue_idx.size:
            self._set_channel_topology(ue_idx)
            channel = self.channel_generator(batch_size=ue_idx.size)
            if tuple(channel.shape) != shape:
                raise ValueError(f"Channel generator returned {tuple(channel.shape)}, expected {shape}")
        else:
            channel = torch.empty(shape, dtype=torch.complex64)

        # Compute channel estimation
        channel_estimate = self._estimate_channel(channel)

        # Get transmitted and noise power
        if self.wireless.direction == "uplink":
            power = np.asarray(self.wireless.ue_power)[ue_idx]
            noise = self.wireless.nob_ap
        else:
            power = np.full(ue_idx.size, np.asarray(self.wireless.ap_power)[0])
            noise = self.wireless.nob_ue

        # Simulate transmission through PhyAbstraction
        mcs_index, decoded_bits, harq_feedback, sinr_eff, tbler, bler  = self.transmission_module(channel, power, noise, channel_estimate=channel_estimate)

        self.metrics.update(self.frame, ue_idx,
                            allocation=allocation,
                            mcs_index=mcs_index,
                            decoded_bits=decoded_bits,
                            harq_feedback=harq_feedback,
                            sinr_eff=sinr_eff,
                            tbler=tbler,
                            bler=bler)

        return self.metrics.decoded(self.frame), self.metrics.successful_ues(self.frame)

    def _omni_antenna(self, antenna_spacing: float):
        return PanelArray(
            num_rows_per_panel=self.M_v,
            num_cols_per_panel=self.M_h,
            polarization="single",
            polarization_type="V",
            element_vertical_spacing=antenna_spacing,
            element_horizontal_spacing=antenna_spacing,
            antenna_pattern="omni",
            carrier_frequency=self.wireless.carrier,
        )


def generate_environment(episode,
                         env_kwargs: dict | None = None,
                         device: str | torch.device | None = None) -> AnomalyTrackingScenario:
    """Run an external allocation sequence through the shared environment lifecycle.

    ``allocations`` yields one vector per frame. No MAC decisions or semantic
    tracking metrics are fabricated here. Save the returned StepResults in the
    caller, together with env.c/wireless/geo when managing a scenario directly.
    """
    ### Set up machine variables
    device = torch.device("cpu" if device is None else device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({device}), but CUDA is unavailable")

        # Take cuda device index for Sionna
        index = device.index if device.index is not None else torch.cuda.current_device()
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is out of range")

        # Reset device just to be sure it is consistent
        device = torch.device("cuda", index)

    # Set all
    sionna_config.device = str(device)
    torch.set_default_device(device)

    # Take the kwargs
    env_kwargs = env_kwargs or {}

    # Validate the complete argument dictionary before routing it to each stage.
    ut.validate_kwargs(
        env_kwargs,
        AnomalyTrackingScenario.shared_type,
        AnomalyTrackingScenario.wireless_type,
        SimulationEnvironment,
        AnomalyTrackingScenario.setup_scenario,
        exclude={"self", "episode", "shared_params", "wireless_params", "c"},
        name="env_kwargs"
    )
    # --- Build environment ---
    sha_kwargs = ut.filter_call_kwargs(AnomalyTrackingScenario.shared_type, **env_kwargs)
    wir_kwargs = ut.filter_call_kwargs(SimulationEnvironment.wireless_type, **env_kwargs)
    ini_kwargs = ut.filter_call_kwargs(SimulationEnvironment, **env_kwargs)
    env = AnomalyTrackingScenario(episode, sha_kwargs, wir_kwargs, **ini_kwargs)
    # Setup scenario
    env.setup_scenario(**ut.filter_call_kwargs(env.setup_scenario, **env_kwargs))
    # Setup default modules
    env.setup_modules(device=device)

    return env


if __name__ == "__main__":
    results = []
    num_ue = 50
    num_slots = 10
    num_clusters = 10
    cluster_size = 4
    env_kwargs = {"num_ue": num_ue, "num_slots": num_slots, "num_frames": 10,
                  "num_clusters": num_clusters, "cluster_size": cluster_size}
    env = generate_environment(1, env_kwargs=env_kwargs, device="cuda")
    for _ in range(env.c.num_frames):
        allocation = env.rng.integers(-1, num_ue, size=num_slots)
        results.append(env.step(allocation))
        print(results[-1])
