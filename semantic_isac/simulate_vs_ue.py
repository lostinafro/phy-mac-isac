import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import pandas as pd
import warnings
import resource_allocation as ra
import utils as ut
import parameters as p
from environment import run_episode


warnings.filterwarnings("ignore", category=RuntimeWarning)
results_label = ["avg_se_achievable", "avg_queue", "avg_misclassification", "avg_risk_hat"]

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


if __name__ == "__main__":
    ra_label, parallel, data_folder, debug, overwrite, concurrent, on_gpu = p.common_parser()

    ### MAIN SIMULATION PARAMETERS ###
    num_ue_vec = np.array([5, 10, 15, 20, 30])
    num_episodes = int(1e3)

    # Saving parameters
    prefix = ra_label + "_vs_ue"
    data_shape = (len(num_ue_vec), len(results_label))
    avg_results, filename = ut.check_data(data_shape, prefix, data_folder, overwrite_flag=overwrite)

    # Set RA method
    ra_method = ra.available_methods[ra_label]

    # Start processing
    for m, num_ue in enumerate(num_ue_vec):
        if overwrite or np.any(np.isnan(avg_results[m])):
            # Info about variable under test
            print(f"{prefix} with U={num_ue}")

            ## CPU/GPU Performances
            # Setting up maximum no. of workers and GPU semaphore to avoid oversubscription of CPU/GPU memory
            cpu_slots, gpu_slots = ut.get_processing_resources(concurrent, on_gpu, num_ue)
            cpu_workers = min(cpu_slots, num_episodes)
            # Generate GPU semaphore
            gpu_semaphore = threading.Semaphore(gpu_slots) if on_gpu else None

            # Define save folder per episode
            save_folder_per_ep = os.path.join(data_folder, prefix + f"_U{num_ue:03d}")
            os.makedirs(save_folder_per_ep, exist_ok=True)

            ## Check which episodes have already been run
            results = []
            episode_kwargs = []
            for ep in range(num_episodes):
                out_path = os.path.join(save_folder_per_ep, f"{ra_label}_ep{ep:04d}.npz")
                if not overwrite and os.path.exists(out_path):
                    try:
                        results.append(load_summary_from_npz(out_path, results_label))
                        continue
                    except Exception as e:
                        print(f"\tWarning: couldn't reuse cached episode {ep} ({out_path}): {e} -- re-running it.")
                episode_kwargs.append(dict(episode=ep, ra_method=ra_method,
                                           env_kwargs={"num_ue": int(num_ue)},
                                           save_folder=save_folder_per_ep,
                                           gpu_guard=gpu_semaphore, verbose=True))

            print(f"\t{len(results)}/{num_episodes} episodes already done, running {len(episode_kwargs)} more.")

            # Start simulation on the missing episodes
            start_time = time.time()
            if episode_kwargs:
                if parallel:
                    torch.set_num_threads(cpu_slots)
                    with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
                        futures = [executor.submit(run_episode, **kw) for kw in episode_kwargs]
                        results.extend(f.result() for f in futures)
                else:
                    for ep_kw in episode_kwargs:
                        print(f"\tEpisode: {ep_kw['episode']:02d}/{num_episodes - 1:02d} (U={num_ue})")
                        results.append(run_episode(**ep_kw))

            # Average the scalar summaries across episodes
            avg_results[m] = np.array([[r[label] for label in results_label] for r in results]).mean(axis=(0, -1))

            avg_df = pd.DataFrame(avg_results, columns=results_label)
            avg_df.insert(0, "num_ue", num_ue_vec)
            avg_df.to_csv(filename, index=False)

            elapsed = time.time() - start_time
            print(f"\t...done in {elapsed:.3f} seconds")
        else:
            print(f"\tnum_ue={num_ue} ...already done!")
            continue