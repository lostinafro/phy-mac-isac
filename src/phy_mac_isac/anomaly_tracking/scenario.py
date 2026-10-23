"""Allocation-driven TDD slots with one SISO UE per occupied slot."""

from dataclasses import dataclass
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from sionna.phy import config as sionna_config
from .. import utils as ut
from ..modules import params
from . import parameters as anomaly_params
from .allocation import TDDResourceAllocation
from ..modules.transmission import SEuplinkOFDMpostMRC, SisoTransmissionResult, SisoPhyAbstraction
from ..modules.trackers import LinearKalmanTracker
from ..modules.theory import DMimoFarFieldOMA
from ..modules.environment import SimulationEnvironment


@dataclass(frozen=True)
class StepResult:
    allocation: np.ndarray             # [R], including idle slots (-1)
    slot_indices: np.ndarray           # [S], occupied positions in allocation
    ue_ids: np.ndarray                 # [S], corresponding UE identities
    channel: torch.Tensor             # [S,1,1,1,1,T,F]
    transmission: SisoTransmissionResult

    @property
    def decoded(self) -> np.ndarray:
        """[R] success flags in the original scheduler order; idle is False."""
        result = np.zeros(self.allocation.size, dtype=bool)
        result[self.slot_indices] = self.transmission.decoded.cpu().numpy()
        return result

    @property
    def successful_ues(self) -> np.ndarray:
        return self.allocation[self.decoded]


class AnomalyTrackingScenario(SimulationEnvironment):
    """
    Configures and manages the setup of the SemanticISAC scenario.
    """
    shared_type = anomaly_params.SharedParams
    wireless_type = anomaly_params.WirelessParams

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("load_scene", False)
        super().__init__(*args, **kwargs)

    def step(self, allocation) -> StepResult:
        """Evaluate one allocation without running or modifying the MAC scheduler.

        The manager's (push_budget,pull_budget) tuple is NOT an allocation.
        Push contention/collisions must be resolved outside this single-UE PHY.
        Feedback is a sampled TB ACK/NACK, not a waveform decoder CRC.
        """
        allocation = np.asarray(allocation)
        if allocation.ndim != 1 or (allocation.size and (
                not np.issubdtype(allocation.dtype, np.integer)
                or np.issubdtype(allocation.dtype, np.bool_))):
            raise ValueError("allocation must be a 1D sequence of integer UE IDs")
        if np.any((allocation < -1) | (allocation >= self.config.num_ues)):
            raise ValueError("UE IDs must lie in [0,num_ues); use -1 for idle")
        allocation = allocation.astype(np.int64, copy=True)
        slot_indices = np.flatnonzero(allocation >= 0)
        ue_ids = allocation[slot_indices]
        slots = ue_ids.size
        shape = (slots, 1, 1, 1, 1, self.config.num_ofdm_symbols,
                 self.config.num_subcarriers)
        if slots:
            h = torch.as_tensor(self.channel_generator(batch_size=slots), device=self.device)
            if tuple(h.shape) != shape:
                raise ValueError(f"Channel generator returned {tuple(h.shape)}, expected {shape}")
            ids = torch.as_tensor(ue_ids, device=self.device)
            h = h * self.link_gain[ids].sqrt().reshape(-1, 1, 1, 1, 1, 1, 1)
        else:
            h = torch.empty(shape, dtype=torch.complex64, device=self.device)
        transmission = self.transmission(
            h, tx_power=self.config.tx_power_w / self.config.num_subcarriers,
            no=self.config.noise_power_per_subcarrier)
        self.frame += 1
        return StepResult(allocation, slot_indices, ue_ids, h, transmission)


def run_episode(episode: int,
                env_kwargs: dict | None = None,
                step_kwargs: dict | None = None,
                device: str | torch.device | None = None,
                save: bool = True,
                save_dirpath: str = params.DATA_FOLDER,
                quiet: bool = False,
                msg: str = "") -> dict:
    """Run one full simulation episode: build the environment, wire up the modules, step
    through every frame, then save the whole run's Shared/Geometric/Estimation/Metrics
    parameters to a single .npz.

    :param episode: int, episode index -- seeds the environment's RNG, and disambiguates
        this run's output filename (must be the first positional argument).
    :param env_kwargs: dict | None, kwargs forwarded to `SimulationEnvironment.__init__`
        (num_ue, num_ap, area_coordinates, lo_ppm, ue_power, ...). `episode` is filled in
        automatically -- don't include it here.
    :param step_kwargs: dict | None, kwargs forwarded to every `env.step(...)` call
        (cto, cfo, sync).

    :param device: str | torch.device, the device to use for Sionna computations.
    :param save: bool, if False, skip saving (useful for quick interactive runs).
    :param save_dirpath: str, directory the output .npz is written to.
    :param quiet: bool, print a per-frame progress bar.
    :param msg: str, message to display before the progress bar.

    :returns: dict, small scalar summary (mean over frames) of a few key metrics, plus the
        saved file path under "save_path" (None if save=False). Useful for a parameter-sweep
        script to aggregate across episodes without re-reading every .npz.
    """
    ### Set up machine variables
    if device is None:
        device = "cpu"

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({device}), but CUDA is unavailable")

    # Take the kwargs
    env_kwargs = env_kwargs or {}
    step_kwargs = step_kwargs or {}

    # Validate the complete argument dictionary before routing it to each stage.
    # Dataclass types come from the selected scenario so subclass fields are supported.
    ut.validate_kwargs(
        env_kwargs,
        AnomalyTrackingScenario.shared_type,
        AnomalyTrackingScenario.wireless_type,
        SimulationEnvironment,
        AnomalyTrackingScenario.setup_scenario,
        exclude={"self", "episode", "shared_params", "wireless_params", "c"},
        name="env_kwargs",
    )
    ut.validate_kwargs(
        step_kwargs,
        AnomalyTrackingScenario.step,
        exclude={"self"},
        name="step_kwargs",
    )

    # --- Build environment ---
    sha_kwargs = ut.filter_call_kwargs(AnomalyTrackingScenario.shared_type, **env_kwargs)
    wir_kwargs = ut.filter_call_kwargs(AnomalyTrackingScenario.wireless_type, **env_kwargs)
    ini_kwargs = ut.filter_call_kwargs(SimulationEnvironment, **env_kwargs)
    ini_kwargs.setdefault("env_dirpath", save_dirpath)
    env = AnomalyTrackingScenario(episode, sha_kwargs, wir_kwargs, **ini_kwargs)

    set_kwargs = ut.filter_call_kwargs(env.setup_scenario, **env_kwargs)
    env.setup_scenario(**set_kwargs)

    # --- Build and wire up modules ---
    ra_module = TDDResourceAllocation(env.c, env.wireless, env.geo)
    tx_module = SEuplinkOFDMpostMRC(env.wireless, ra_module, device)

    trans_mat = np.eye(3)[np.newaxis, :, :].repeat(env.U, axis=0)
    obs_mat = np.eye(3)[np.newaxis, :, :].repeat(env.U, axis=0)
    proc_noise_cov = env.geo.ue_vel_cov * env.wireless.frame_time ** 2
    obs_noise_cov = np.eye(3)[np.newaxis, :, :].repeat(env.U, axis=0)
    control_mat = np.eye(3)[np.newaxis, :, :].repeat(env.U, axis=0)
    tracker = LinearKalmanTracker(trans_mat, obs_mat, proc_noise_cov, obs_noise_cov, control_mat)

    performance = DMimoFarFieldOMA(env.c, env.wireless, env.geo, env.estimate)

    env.setup_modules(ra_module, tx_module, tracker, performance)

    # --- Run the simulation ---
    frame_iter = ut.std_progressbar(range(env.c.num_frames), desc=msg) if not quiet else range(env.c.num_frames)
    for _ in frame_iter:
        env.step(**step_kwargs)

    # --- Save the whole run: Estimation, and Metrics parameters ---
    save_path = None
    if save:
        ra_method_label = next(key for key, val in ra.available_methods.items() if val == ra_method)
        filename = os.path.join(save_dirpath, f"{ra_method_label}_ep{episode:04d}.npz")
        save_path = params.save_params_npz(filename, c=env.c, wireless=env.wireless, geo=env.geo,
                                      estimate=env.estimate, metrics=env.metrics)

    # --- Small scalar summary for quick cross-episode aggregation ---
    summary = {
        "save_path": save_path,
        "avg_se_achievable": np.nanmean(env.metrics.se_achievable[:-1], axis=0),
        "avg_se_set": np.nanmean(env.metrics.se_set[:-1], axis=0),
        "avg_queue": np.nanmean(env.metrics.queue[:-1], axis=0),
        "avg_misclassification": np.nanmean(env.metrics.misclassification[:-1], axis=0),
        "avg_risk_predicted": np.nanmean(env.metrics.risk_predicted[:-1], axis=0),
        "avg_risk_hat": np.nanmean(env.metrics.risk_hat[:-1], axis=0),
    }
    return summary


results_labels = ["avg_se_achievable", "avg_se_set", "avg_queue", "avg_misclassification", "avg_risk_predicted", "avg_risk_hat"]


def load_summary_from_npz(filepath: str, results_label: list[str]) -> dict:
    """Reconstruct a run_episode-style summary dict from an already-saved .npz,
    for episodes being skipped because they're already done."""
    with np.load(filepath) as data:
        summary = {"save_path": filepath}
        for label in results_label:
            # Remove the avg_ from the label to catch the right field
            field = label.replace("avg_", "", 1)
            # Raise error if the field is not found
            if field not in data:
                raise KeyError(f"Field '{field}' not found in {filepath}. Available keys: {list(data.keys())}")
            # Add the avg_ back to the label to match the original format while averaging the data
            summary[label] = np.nanmean(data[field][:-1], axis=0)
    return summary


def common_parser():
    parser = ArgumentParser(description="Common parser to run simulations.")

    parser.add_argument("--ra_method", type=str, default="", help="Method to use for RA simulation")
    parser.add_argument('--parallel', action='store_true', help="Enable parallel processing")
    parser.add_argument('--savesubdir', type=str, default="", help="Optional name of the subdirectory to save results in")
    parser.add_argument('--quiet', action='store_true', help="Quiet outputs")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite results  if already present in the savesubdir")
    parser.add_argument("--concurrent", type=int, default=1, help="Set the number of concurrent scripts run")
    parser.add_argument("--cpu", action='store_true', help="Set the processing on the CPU only")

    args = parser.parse_args()
    if args.cpu:
        sionna_config.device = "cpu"
        device = "cpu"
    else:
        sionna_config.device = "cuda:0"
        device = "cuda:0"

    return args.ra_method, args.parallel, args.savesubdir, args.quiet, args.overwrite, args.concurrent, device


if __name__ == "__main__":
    num_ue = 100
    env_args = {"debugging": True, "num_frames": 100, "num_ue": num_ue}
    summary = run_episode(1, env_kwargs=env_args)
    print(summary)
