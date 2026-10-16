import threading
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from sionna.phy import config as sionna_config
from .. import utils as ut
from ..modules import params
from . import parameters as semantic_params
from ..modules import resource_allocation as ra
from ..modules.transmission import SEuplinkOFDMpostMRC
from ..modules.trackers import LinearKalmanTracker
from ..modules.theory import DMimoFarFieldOMA
from ..modules.environment import SimulationEnvironment


class SemanticISACScenario(SimulationEnvironment):
    """
    Configures and manages the setup of the SemanticISAC scenario.
    """
    shared_type = semantic_params.SharedParams
    wireless_type = semantic_params.WirelessParams
    metrics_type = semantic_params.Metrics

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("rm_dirpath", semantic_params.RM_FOLDER)
        super().__init__(*args, **kwargs)

        # Specific subclass attribute
        self.se_margin = np.ones((self.c.num_frames, self.U)) * .9

    def step(self, cto: bool = True, cfo: bool = True, sync: bool = False):
        """Evaluates operations in a frame"""
        ### Update frame and geometry
        # Start by updating the frame number
        self.frame += 1

        # Move the UEs to a new position under Brownian motion
        new_pos, new_vel = self.move_ue()

        ## Take channel instance through Sionna.rt
        # Check if synchronization has been performed during the previous frame (all UEs)
        if sync:
            # Resample LO offset in the current frame for CTO and CFO baselines
            cto_base, cfo_base = self.wireless.sample_lo_offset("gaussian", store=True, rng=self.rng)
            self.frame_sync = self.frame
        else:
            cto_base, cfo_base = self.wireless.cto_base, self.wireless.cfo_base
        # Check if CTO or CFO are neglected
        if not cto:
            cto_base = np.zeros_like(cto_base)
        if not cfo:
            cfo_base = np.zeros_like(cfo_base)

        # Generate channel realization for the whole frame
        paths = self.path_solver(self.scene, seed=self.episode, **ut.filter_kwargs(self.path_solver, **self.c.rays))
        channel = self.compute_cfr(paths, cfo_base, cto_base)  # [L, M, U, S, T, F]

        ### Prior
        # Update through Kalman equations
        ue_pos_hat_pri, ue_pos_cov_pri = self.tracker_module.prior(self.estimate.ue_pos_hat_post[self.frame - 1],
                                                                   self.estimate.ue_pos_cov_post[self.frame - 1],
                                                                   u=self.geo.ue_vel_mean * self.wireless.frame_time)

        self.estimate.update(self.frame,
                             ue_pos_hat_prio=ue_pos_hat_pri,
                             ue_pos_cov_prio=ue_pos_cov_pri)

        # Generate data at the UE and check the queue
        delta_queue = self.wireless.queue_max - self.metrics.queue[self.frame - 1]
        if self._load_env:  # Take the already stored values
            new_data = self.wireless.gen_bit[self.frame]
        else:   # Generate new data and save it in the wireless paramer
            new_data = self.wireless.sample_new_data(rng=self.rng)
            self.wireless.update(frame=self.frame, gen_bit=new_data)
        queue = self.metrics.queue[self.frame - 1] + new_data

        ### Perform allocation based on prior
        # Evaluate common SNR related to eFIM
        snr = self.wireless.ue_power / self.wireless.nob_ap * len(self.ra_method.pilot_ofdm_symbol_indices())
        # Average energy over antennas (far-field approximation), symbols and subcarriers in a slot (quasi-static fading)
        beta = np.mean(np.abs(channel) ** 2, axis=(1, 4, 5))  # (L, U, S)

        # Compute eFIM and rate assuming perfect knowledge (oracle) (shape (S, U, d, d))
        oracle_efim = self.performance.eval_efim(snr, self.geo.ue_location[self.frame], beta.transpose())
        oracle_se = self.performance.eval_achievable_se(beta.transpose(), len(self.ra_method.pilot_ofdm_symbol_indices()))

        # Compute predicted eFIM and rate with estimated values (shape (U, d, d))
        # Using 3dB margin to account for UL/DL mismatches
        beta_hat = self.estimate_gain_dl(self.estimate.ue_pos_hat_prio[self.frame]) / 2
        predicted_efim = np.squeeze(self.performance.eval_efim(snr, self.estimate.ue_pos_hat_prio[self.frame], beta_hat[..., np.newaxis].transpose()), axis=0)
        predicted_se = np.squeeze(self.performance.eval_achievable_se(beta_hat[..., np.newaxis].transpose(), len(self.ra_method.pilot_ofdm_symbol_indices())), axis=0)

        # Compute allocation matrix based on the predictions
        kwargs = {"efim": predicted_efim, "covariance_pri": ue_pos_cov_pri, "delta_queue": delta_queue}
        ue_slot_map = self.ra_method(self.frame, **kwargs)
        # Take indexing variables
        ue_active_per_slot, slot_idx, ue_idx = self.ra_method.scheduled_ues(ue_slot_map)
        not_selected = ~np.isin(np.arange(self.U), ue_idx)
        # Compute approximate semantic risk through oracle eFIM
        risk_hat = np.zeros(self.U)
        risk0, risk1, _ = self.ra_method.eval_semantic_risk(self.frame, oracle_efim, ue_pos_cov_pri)
        risk_hat[ue_idx] = risk1[slot_idx, ue_idx]
        risk_hat[not_selected] = risk0[not_selected]
        # Compute predicted risk
        risk_predicted_full = self.ra_method.output_risks()
        risk_predicted = np.zeros(self.U)
        risk_predicted[ue_idx] = risk_predicted_full[ue_idx, 1]
        risk_predicted[not_selected] = risk_predicted_full[not_selected, 0]

        ### Simulating abstract transmission for CF-mMIMO operations
        # Transpose to put the slots as batches for the transmission module (needed for GPU enhancement)
        # Also, merge the dimensions on the antennas to consider a single receiver with distributed placement and
        # add a single UE stream dimension
        channel_cf = np.moveaxis(channel.copy(), 3, 0)  # [S, L, M, U, T, F]
        channel_cf = channel_cf.reshape(self.S, 1, self.L * self.M, self.U, 1, self.T, self.F)    # [S, 1, L*M, U, 1, T, F]

        # Apply the allocation mask
        channel_cf *= self.ra_method.broadcast_slot_ue_mask(ue_slot_map) * self.ra_method.broadcast_ap_ue_mask()

        # Perform (abstracted) usual centralized CF-mMIMO transmission
        sinr_eff, se_eff = self.transmission_module(channel_cf)

        ### Update queues
        # Checking SE achievability
        se_achievable = se_eff[slot_idx, ue_idx]
        se_per_scheduled_ue = predicted_se[ue_idx] * self.se_margin[self.frame, ue_idx]
        se_per_scheduled_ue[se_per_scheduled_ue > se_achievable] = 0.
        # Re-init margin for UE with achievable SE while reducing margin for UE with unachievable SE
        if self.frame < self.c.num_frames - 1:
            self.se_margin[self.frame + 1:, ue_idx[se_per_scheduled_ue > 0.]] /= self.se_margin[0, ue_idx[se_per_scheduled_ue > 0.]]
            self.se_margin[self.frame + 1:, ue_idx[se_per_scheduled_ue == 0.]] *= self.se_margin[0, ue_idx[se_per_scheduled_ue == 0.]]

        # Taking the amount of bits transmitted and update the queue
        bits_tx = np.asarray(np.floor(se_per_scheduled_ue * self.wireless.bw * self.wireless.slot_time), dtype=np.int32)
        queue[ue_idx] = np.clip(queue[ue_idx] - bits_tx, 0, None)

        # Create the observation for the scheduled UEs (assuming asymptotic performance are achievable)
        asymptotic_cov = np.linalg.inv(oracle_efim[ue_slot_map])
        obs_vec = ut.sample_multivariate_gaussian(cov=asymptotic_cov, mean=self.geo.ue_location[self.frame, ue_idx],
                                                  rng=self.rng)

        ### Posterior
        # Compute posterior for UE observed
        ue_pos_hat_post, ue_pos_cov_post = self.tracker_module.posterior(obs_vec, ue_pos_hat_pri,
                                                                         ue_pos_cov_pri, ue_idx,
                                                                         obs_cov=asymptotic_cov)
        self.estimate.update(self.frame, ue_idx,
                             ue_pos_hat_post=ue_pos_hat_post,
                             ue_pos_cov_post=ue_pos_cov_post,
                             ue_pos_hat_meas=obs_vec,
                             ue_pos_cov_meas=asymptotic_cov,
                             )

        # Compute which UEs are misclassified
        misclassified = self.geo.ue_region[self.frame] != self.estimate.ue_region_hat[self.frame]

        ### Evaluate Metrics
        self.metrics.update(self.frame, ue_idx,
                            queue=queue,
                            delta_queue=delta_queue,
                            large_scale=beta.mean(axis=-1),
                            large_scale_hat=beta_hat,
                            sinr=sinr_eff.mean(axis=-1)[slot_idx, ue_idx],
                            se_set=se_per_scheduled_ue,
                            se_achievable=se_achievable,
                            se_oracle=oracle_se[ue_slot_map],
                            efim_oracle=oracle_efim,
                            efim_predicted=predicted_efim,
                            misclassification=misclassified,
                            risk_predicted_full=risk_predicted_full,
                            risk_predicted=risk_predicted,
                            risk_hat=risk_hat,
                            ue_slot_map=ue_slot_map,
                            )
        # Print errors
        every = self.c.num_frames // 100
        if self.debug and (not (self.frame % every) or self.frame == self.c.num_frames - 1):
            if self.frame == 0:
                frame_range = np.array([-1, 0])
            else:
                frame_range = np.arange(self.frame - every, self.frame) + 1
            # Visualize
            # self.scene_fig = self.show_environment(paths=paths, fig=getattr(self, "scene_fig", None))
            # self.scene_fig.canvas.draw_idle()
            # plt.pause(0.001)
            try:
                self.env_fig = self.show_estimation(frame=self.frame, fig=getattr(self, "env_fig", None))
                self.env_fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception as e:
                print(f"Probably the figure dies in some way. Was it you?: {e}")

            try:
                self.channel_fig = self.plot_channel(channel, fig=getattr(self, "channel_fig", None))
                self.channel_fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception as e:
                print(f"Probably the figure dies in some way. Was it you?: {e}")

            # Print starts
            print(f"----------------------- FRAME {self.frame:03d} -------------------------")

            # UE movement
            for u in range(min(10, self.U)):
                msg1 = [f"{float(a):.2f}" for a in self.geo.ue_location[frame_range[0], u]]
                msg2 = [f"{float(a):.2f}" for a in new_pos[u]]
                msg3 = [f"{float(a):.2f}" for a in new_vel[u]]
                print(f"UE {u}: old_pos={msg1}, new_pos={msg2}, new_vel={msg3}")

            # Allocated UEs
            msg = ' '.join([f"{u:02d}" for u in np.sort(ue_idx)])
            print(f"Allocated UEs are: {msg}")
            base_x = 0.1
            self.env_fig.text(x=base_x, y=0.9, s=f"Allocated UEs: {msg}")
            try:
                msg = [f"{u:02d}" for u in np.sort(np.nonzero(self.ra_method.forced)[0])]
                self.env_fig.text(x=3 * base_x, y=0.9, s=f"Forced by comm: {' '.join(msg)}")
            except ValueError:
                pass

            # Test quasi-constant beta
            beta_full = np.abs(channel) ** 2
            mask = np.nonzero(np.any(np.abs(beta_full - beta[:, np.newaxis, :, :, np.newaxis, np.newaxis]) > 1e-6, axis=(1, 4, 5)))
            if len(mask[0]) > 0:
                for idx in range(len(mask[0])):
                    print(f"Not quasi-constant beta for AP {mask[0][idx]}, UE {mask[1][idx]} and slot {mask[2][idx]}")

            # Compute absolute and relative error between RadioMap and RayTracer as metrics
            beta_gt = beta.mean(axis=-1)
            print(f"Avg. abs RM error is: {np.mean(np.abs(beta_gt - beta_hat)) / np.mean(beta_gt) * 100:.2f} %")

            # Comm values
            # Check issues in the oracle if any
            unachievable_err = np.nonzero(se_achievable < oracle_se[slot_idx, ue_idx])[0]
            if len(unachievable_err) > 0:
                print(f"ERROR: Unachievable oracle rate for UE {unachievable_err}")

            print("\nCOMMS:")
            msg = [f"{float(s):.2f}" for s in self.metrics.se_achievable[self.frame]]
            print(f"UEs achievable SE is: \t{' '.join(msg)}")
            msg = [f"{float(s):.2f}" for s in self.metrics.se_set[self.frame]]
            print(f"UEs transmitted SE is: \t{' '.join(msg)}")
            msg = [f"{int(q):04d}" for q in self.metrics.queue[self.frame]]
            print(f"UEs queue status is: \t{' '.join(msg)}")
            queue_exceeded = np.nonzero(queue > self.wireless.queue_max)[0]
            if len(queue_exceeded) > 0:
                msg = [f"{int(u):02d}" for u in queue_exceeded]
                print(f"Queue exceeded for UE \t{' '.join(msg)}")
            else:
                print(f"Queue within limits for all UEs")
            msg = [f"{int(q):04d}" for q in self.metrics.queue[frame_range].mean(0)]
            print(f"UEs avg. frames {frame_range[0]:02d}/{self.frame:02d} queue status is: \t{' '.join(msg)}")
            msg = [f"{float(q) * 100:02.1f}%" for q in (self.metrics.queue[:frame_range[-1]] < self.wireless.queue_max).mean(0)]
            print(f"UEs queue reliability up to frame {self.frame:03d}: \t{' '.join(msg)}")

            # Cartesian errors
            print("\nLOCA:")
            msg = [f"{float(peb):.2f}" for peb in np.sqrt(np.linalg.trace(np.linalg.inv(predicted_efim)))]
            print(f"Per UE predic PEB is: \t{' '.join(msg)}")
            msg = [f"{peb:.2f}" for peb in np.sqrt(np.linalg.trace(np.linalg.inv(oracle_efim[0])))]
            print(f"Per UE oracle PEB is: \t{' '.join(msg)}")
            msg = [f"{peb:.2f}" for peb in np.sqrt(np.linalg.trace(ue_pos_cov_pri))]
            print(f"Per UE priori PEB is: \t{' '.join(msg)}")
            msg = [f"{peb:.2f}" for peb in np.sqrt(np.linalg.trace(ue_pos_cov_post))]
            print(f"Per UE poster PEB is: \t{' '.join(msg)}")
            msg = [f"{rse:.2f}" for rse in np.linalg.norm(self.geo.ue_location[self.frame] - self.estimate.ue_pos_hat_post[self.frame], axis=-1)]
            print(f"Per UE poste RMSE is: \t{' '.join(msg)}")

            # Semantic errors
            print("\nSEMA:")
            msg = ' '.join([f"{int(risk):02d}" for risk in np.nonzero(self.metrics.misclassification[self.frame])[0]])
            print(f"UEs misclassified in this frame are: \t{msg}")
            self.env_fig.text(x=base_x * 6, y=0.9, s=f"UEs misclassified: {msg}")
            msg = [f"{float(risk):.2e}" for risk in self.metrics.risk_predicted[self.frame]]
            print(f"UEs semantic risk predicted was: \t{' '.join(msg)}")
            msg = [f"{float(risk):.3f}" for risk in self.metrics.risk_predicted_full[self.frame, :, 0]]
            self.env_fig.text(x=base_x, y=0.87, s=f"hat_r0: {' '.join(msg)}")
            msg = [f"{float(risk):.3f}" for risk in self.metrics.risk_predicted_full[self.frame, :, 1]]
            self.env_fig.text(x=base_x, y=0.84, s=f"hat_r1: {' '.join(msg)}")
            msg = [f"{float(risk):.3f}" for risk in self.metrics.risk_hat[self.frame]]
            print(f"UEs semantic risk approxim. was: \t{' '.join(msg)}")
            msg = [f"{float(risk):.3f}" for risk in self.metrics.risk_hat[:frame_range[-1]].mean(0)]
            print(f"UEs avg risk appr up to frame {self.frame:03d}: \t{' '.join(msg)}")
            self.env_fig.text(x=base_x, y=0.81, s=f"riskh: {' '.join(msg)}")
            msg = [f"{float(risk):.3f}" for risk in self.metrics.misclassification[:frame_range[-1]].mean(0)]
            print(f"UEs avg misprob up to frame {self.frame:03d}: \t{' '.join(msg)}")
            self.env_fig.text(x=base_x, y=0.78, s=f"mispr: {' '.join(msg)}")

            os.makedirs(self.env_folder, exist_ok=True)
            ra_method_label = next(key for key, val in ra.available_methods.items() if val == type(self.ra_method))
            filename = os.path.join(self.env_folder, f"{ra_method_label}_f{self.frame:02d}_ep{self.episode:02d}.png")
            self.env_fig.savefig(filename, dpi=300)

            print("----------------------- END FRAME -------------------------\n")

        # Check feasibility of the current allocation
        if np.any(np.sum(ue_slot_map, axis=-1) > 1):
            # print("Allocation is not OMA")
            return False
        if np.any(np.nonzero(queue > self.wireless.queue_max)):
            # print("Queue is exceeding its maximum for at least one UE")
            return False
        return True


def run_episode(episode: int,
                ra_method: type[ra.ResourceAllocationBase],
                ra_kwargs: dict | None = None,
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
    :param ra_method: type[ResourceAllocationBase], a class from resource_allocation.py
        (not an instance -- it's constructed here once env.c/wireless/geo/estimate exist),
        e.g. `ra.RoundRobinAllocation` (must be the second positional argument).
    :param ra_kwargs: dict | None, extra kwargs forwarded to `ra_method`'s constructor.
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
    ra_kwargs = ra_kwargs or {}
    env_kwargs = env_kwargs or {}
    step_kwargs = step_kwargs or {}

    # --- Build environment ---
    sha_kwargs = ut.filter_kwargs(params.SharedParams, **env_kwargs)
    wir_kwargs = ut.filter_kwargs(params.WirelessParams, **env_kwargs)
    ini_kwargs = ut.filter_kwargs(SimulationEnvironment, **env_kwargs)
    env = SemanticISACScenario(episode, sha_kwargs, wir_kwargs, env_dirpath=save_dirpath, **ini_kwargs)

    set_kwargs = ut.filter_kwargs(env.setup_scenario, **env_kwargs)
    env.setup_scenario(**set_kwargs)

    # --- Build and wire up modules ---
    ra_module = ra_method(env.c, env.wireless, env.geo, env.estimate, **ra_kwargs)
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
    num_ue = 12
    avg_datasize, queue_max = semantic_params.arrival_rate_init(avg_se=2, num_ue=num_ue)
    env_args = {"debugging": True, "num_frames": 100, "num_ue": num_ue, "traffic_avg": avg_datasize,
                "queue_max": queue_max, "num_regions": 32, "ue_vel_mean": 5}
    step_args = {"cfo": False}
    summary = run_episode(1, ra.RoundRobinAllocation, env_kwargs=env_args, step_kwargs=step_args)
    print(summary)