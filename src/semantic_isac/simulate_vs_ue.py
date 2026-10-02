import os
from datetime import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import warnings
from src.modules import resource_allocation as ra
import utils as ut
import parameters as p
from scenario import run_episode, results_labels


warnings.filterwarnings("ignore", category=RuntimeWarning)

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

 ### MAIN SIMULATION PARAMETERS ###
num_ue_vec = np.array([6, 12, 18, 27, 36])
num_episodes = p.NUM_EP


if __name__ == "__main__":
    ra_methods, parallel, data_folder_suffix, quiet, overwrite, concurrent, on_cpu = p.common_parser()
    # if not on_cpu:
    #     free_gpus = ut.select_available_gpus()
    #     if not free_gpus:
    #         print("No idle GPUs found -- falling back to CPU.")
    #         on_gpu = False
    #         os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #     else:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in free_gpus)
    #         print(f"Using GPUs: {free_gpus}")
    # else:
    #     free_gpus = None

    # Test all methods if unspecified
    if ra_methods == "":
        ra_methods = ra.available_methods
    else:
        ra_methods = [ra_methods]

    # Data folder for today
    today = datetime.now().strftime("%Y-%m-%d")
    data_folder = os.path.join(p.DATA_FOLDER, today, data_folder_suffix)

    # Compute datasize and queue_max
    # Check maximum load for non-trivial/non_explosive queues
    avg_datasize, queue_max = p.arrival_rate_init(avg_se=2, num_ue=num_ue_vec)

    # Cycling through methods
    for ra_label in ra_methods:
        # Set RA method
        ra_method = ra.available_methods[ra_label]

        # Saving parameters
        prefix = ra_label + "_vs_ue"
        data_shape = (len(num_ue_vec), len(results_labels))
        avg_results, filename = ut.check_data(data_shape, prefix, data_folder, overwrite_flag=overwrite)

        # Start processing
        for m, num_ue in enumerate(num_ue_vec):
            # Printing info about variable under test
            print(f"Running {prefix} with U={num_ue}:", end=" ")

            ## CPU/GPU Performances
            # Setting up maximum no. of workers and GPU semaphore to avoid oversubscription of CPU/GPU memory
            cpu_slots, gpu_slots = ut.get_processing_resources(concurrent, on_cpu, num_ue)
            cpu_workers = min(cpu_slots, num_episodes)
            # Generate GPU semaphore
            gpu_semaphore = threading.Semaphore(gpu_slots) if not on_cpu else None

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
                episode_kwargs.append(dict(episode=ep, ra_method=ra_method,
                                           env_kwargs={"num_ue": int(num_ue),
                                                       "debugging": False,
                                                       "traffic_avg": avg_datasize[m],
                                                       "queue_max": queue_max[m],
                                                       "ue_vel_mean": p.VEL_MAG},
                                           step_kwargs={"cfo": False},
                                           metrics_dirpath=save_ep_dirpath,
                                           gpu_guard=gpu_semaphore, quiet=quiet,
                                           msg=f"RA: {ra_label} - U={num_ue:02d} - Episode: {ep:02d}/{num_episodes - 1:02d}"))

            print(f"{len(results)} episodes already done, {len(episode_kwargs)} remaining.")

            if len(episode_kwargs) == 0:
                continue

            # Start simulation on the missing episodes
            start_time = time.time()
            if episode_kwargs:
                if parallel:
                    import torch
                    torch.set_num_threads(cpu_slots)
                    with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
                        futures = [executor.submit(run_episode, **kw) for kw in episode_kwargs]
                        results.extend(f.result() for f in futures)
                else:
                    for ep_kw in episode_kwargs:
                        results.append(run_episode(**ep_kw))

            # Average the scalar summaries across episodes
            avg_results[m] = np.nanmean(np.array([[r[label] for label in results_labels] for r in results]), axis=(0, -1))

            avg_df = pd.DataFrame(avg_results, columns=results_labels)
            avg_df.insert(0, "num_ue", num_ue_vec)
            avg_df.to_csv(filename, index=False)

            elapsed = time.time() - start_time
            print(f"\t...done in {elapsed:.3f} seconds")
