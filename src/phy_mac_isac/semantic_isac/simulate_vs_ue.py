import os
import sys
if "--cpu" in sys.argv[1:]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datetime import datetime
import time
import numpy as np
import pandas as pd
import warnings
from ..modules import resource_allocation as ra
from ..utils import check_data, get_processing_resources, run_simulations
from . import parameters as semantic_params
from .scenario import run_episode, results_labels, load_summary_from_npz, common_parser


warnings.filterwarnings("ignore", category=RuntimeWarning)

### MAIN SIMULATION PARAMETERS ###
num_ue_vec = np.array([6, 12, 18, 27, 36])
num_episodes = semantic_params.NUM_EP


if __name__ == "__main__":
    ra_methods, parallel, data_folder_suffix, quiet, overwrite, concurrent, episode_device = common_parser()

    # Test all methods if unspecified
    if ra_methods == "":
        ra_methods = ra.available_methods
    else:
        ra_methods = [ra_methods]

    # Data folder for today
    today = datetime.now().strftime("%Y-%m-%d")
    data_folder = os.path.join(semantic_params.DATA_FOLDER, today, data_folder_suffix)

    # Compute datasize and queue_max
    # Check maximum load for non-trivial/non_explosive queues
    avg_datasize, queue_max = semantic_params.arrival_rate_init(avg_se=2, num_ue=num_ue_vec)

    # Cycling through methods
    for ra_label in ra_methods:
        # Set RA method
        ra_method = ra.available_methods[ra_label]

        # Saving parameters
        prefix = ra_label + "_vs_ue"
        data_shape = (len(num_ue_vec), len(results_labels))
        avg_results, filename = check_data(data_shape, prefix, data_folder, overwrite_flag=overwrite)

        # Start processing
        for m, num_ue in enumerate(num_ue_vec):
            # Printing info about variable under test
            print(f"Running {prefix} with U={num_ue}:", end=" ")

            ## CPU/GPU Performances
            # Setting up maximum no. of workers and GPU semaphore to avoid oversubscription of CPU/GPU memory
            cpu_slots, gpu_slots = get_processing_resources(concurrent, episode_device, num_ue)
            max_workers = min(cpu_slots, gpu_slots) if gpu_slots else cpu_slots

            # Define save folder per episode
            save_ep_dirpath = os.path.join(data_folder, prefix + f"_U{num_ue:03d}")
            os.makedirs(save_ep_dirpath, exist_ok=True)

            ## Check which episodes have already been run
            results = []
            episode_kwargs = []
            for ep in range(num_episodes):
                out_path = os.path.join(save_ep_dirpath, f"{ra_label}_ep{ep:04d}.npz")
                if not overwrite and os.path.exists(out_path):
                    try:
                        results.append(load_summary_from_npz(out_path, results_labels))
                        continue
                    except Exception as e:
                        print(f"\tWarning: couldn't reuse cached episode {ep} ({out_path}): {e} -- re-running it.")
                episode_kwargs.append(dict(episode=ep,
                                           ra_method=ra_method,
                                           env_kwargs={"num_ue": int(num_ue),
                                                       "debugging": False,
                                                       "traffic_avg": avg_datasize[m],
                                                       "queue_max": queue_max[m],
                                                       "ue_vel_mean": semantic_params.VEL_MAG},
                                           step_kwargs={"cfo": False},
                                           save_dirpath=save_ep_dirpath,
                                           device=episode_device,
                                           quiet=quiet,
                                           msg=f"RA: {ra_label} - U={num_ue:02d} - Episode: {ep:02d}/{num_episodes - 1:02d}"))

            print(f"{len(results)} episodes already done, {len(episode_kwargs)} remaining.")

            if len(episode_kwargs) == 0:
                continue

            # Start simulation on the missing episodes
            start_time = time.time()
            results.extend(run_simulations(run_episode,
                                           episode_kwargs,
                                           device=episode_device,
                                           parallel=parallel,
                                           max_workers=max_workers,
                                           torch_threads=1))

            # Average the scalar summaries across episodes
            avg_results[m] = np.nanmean(np.array([[r[label] for label in results_labels] for r in results]), axis=(0, -1))

            avg_df = pd.DataFrame(avg_results, columns=results_labels)
            avg_df.insert(0, "num_ue", num_ue_vec)
            avg_df.to_csv(filename, index=False)

            elapsed = time.time() - start_time
            print(f"\t...done in {elapsed:.3f} seconds")
