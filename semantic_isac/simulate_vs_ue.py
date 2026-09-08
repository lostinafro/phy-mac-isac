import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import warnings
import resource_allocation as ra
import utils as ut
import parameters as p
from environment import run_episode


_worker_gpu_semaphore = None
def init_worker(semaphore):
    global _worker_gpu_semaphore
    _worker_gpu_semaphore = semaphore

warnings.filterwarnings("ignore", category=RuntimeWarning)
results_label = ["avg_se_achievable", "avg_queue", "avg_misclassification", "avg_risk_hat"]


if __name__ == "__main__":
    ra_label, parallel, data_folder, debug, overwrite, concurrent, on_gpu = p.common_parser()

    ### MAIN SIMULATION PARAMETERS ###
    num_ue_vec = np.array([5, 10, 20, 30, 40])
    num_episodes = int(1e3)

    # Saving parameters
    prefix = ra_label + "_vs_ue"
    data_shape = (len(num_ue_vec), len(results_label))
    avg_results, filename = ut.check_data(data_shape, prefix, data_folder, overwrite_flag=overwrite)

    # Set RA method
    ra_method = ra.available_methods[ra_label]

    # Setting up maximum no. of workers and GPU semaphore to avoid oversubscription of CPU/GPU memory
    cpu_workers, threads_per_worker, gpu_semaphore, manager = ut.get_processing_resources(concurrent, num_episodes,
                                                                                 on_gpu, num_ue_vec.max())

    for m, num_ue in enumerate(num_ue_vec):
        if overwrite or np.any(np.isnan(avg_results[m])):
            # Info about variable under test
            print(f"{prefix} with U={num_ue}")

            start_time = time.time()
            episode_kwargs = [
                dict(episode=ep,
                     ra_method=ra_method,
                     env_kwargs={"num_ue": int(num_ue)},
                     save_folder=os.path.join(data_folder, prefix + f"_U{num_ue:03d}"),
                     gpu_guard=gpu_semaphore,
                     torch_threads=threads_per_worker,
                     verbose=True)
                for ep in range(num_episodes)
            ]

            if parallel:
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=cpu_workers, mp_context=ctx) as executor:
                    futures = [executor.submit(run_episode, **kw) for kw in episode_kwargs]
                    results = [f.result() for f in futures]
            else:
                results = []
                for ep, kw in enumerate(episode_kwargs):
                    print(f"\tEpisode: {ep:02d}/{num_episodes - 1:02d} (U={num_ue})")
                    results.append(run_episode(**kw))

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
    # Close the shared manager
    manager.shutdown()