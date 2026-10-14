# System: folders, data files, GPU discovery, progress bars
import os
import sys
import GPUtil
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import date


def check_data(data_shape: tuple, prefix: str, folder: str, overwrite_flag: bool = False):
    filename = os.path.join(folder, prefix + '.csv')

    if os.path.exists(filename) and not overwrite_flag:
        results = pd.read_csv(filename).iloc[:, 1:].to_numpy()
    else:
        results = np.full(data_shape, np.nan)
    return results, filename


## Folder management
uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])


def standard_output_dir(subdirname: str) -> str:
    basedir = os.path.join(os.path.expanduser('~'), 'OneDrive/plots')
    if not os.path.exists(basedir):
        basedir = os.path.join(uppath(__file__, 2), 'plots')
        if not os.path.exists(basedir):
            os.mkdir(basedir)
    subdir = os.path.join(basedir, subdirname)
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    output_dir = os.path.join(subdir, str(date.today()))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir


## Progress bars
def std_progressbar(total_iteration, **kwargs):
    return tqdm(total_iteration, file=sys.stdout, leave=False, ascii=True, **kwargs)


def fixed_progressbar(total_iteration, **kwargs):
    return tqdm(total_iteration, file=sys.stdout, leave=True, ascii=True, **kwargs)


## Hardware resources
def get_processing_resources(num_scripts: int,
                             device: str,
                             num_ue: int = 1,
                             mib_vram_per_ue: int | float = 200,
                             mib_vram_baseline: int | float = 210) -> tuple[int, int]:
    """Return CPU workers and GPU episode slots for the selected device.

    ``device`` is the logical Torch device, such as "cpu" or "cuda:0".
    GPU slots are calculated from free memory on that device only.
    ``num_scripts`` reserves an equal share for simultaneously running scripts.
    """
    # TODO: the amount of GPU VRAM (in MiB) required per episode instead of per UE
    if num_scripts < 1:
        raise ValueError("num_scripts must be at least 1")
    if num_ue < 0:
        raise ValueError("num_ue cannot be negative")

    available_cpus = os.cpu_count() or 1
    cpu_slots = max(1, (available_cpus // 2) // num_scripts)

    # Do not query or initialize CUDA for a CPU run.
    if not str(device).startswith("cuda"):
        return cpu_slots, 0

    import torch

    selected_device = torch.device(device)
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({device}), but is unavailable")

    vram_per_episode_mib = mib_vram_baseline + mib_vram_per_ue * num_ue
    if vram_per_episode_mib <= 0:
        raise ValueError("Estimated VRAM per episode must be positive")

    free_bytes, _total_bytes = torch.cuda.mem_get_info(selected_device)
    free_mib = free_bytes / (1024 ** 2)

    gpu_slots = max(0, int(free_mib // (num_scripts * vram_per_episode_mib)))
    return cpu_slots, gpu_slots


def select_available_gpus(max_utilization_pct: float = 10.0,
                           max_others_mem_mib: float = 1500.0) -> list[int]:
    """Identify which GPUs on this machine are effectively idle (not meaningfully used by
    other users' processes), so a shared-cluster run only considers those for its own budget.

    :param max_utilization_pct: float, skip a GPU if its compute utilization is above this
        (e.g. someone else is actively running something on it).
    :param max_others_mem_mib: float, skip a GPU if more than this much VRAM is already
        occupied (by anyone) -- catches GPUs that are memory-reserved but momentarily idle.

    :returns: list[int], physical GPU indices considered free to use. Empty list if none
        qualify (caller should fall back to CPU-only in that case).
    """
    gpus = GPUtil.getGPUs()
    available = []
    for gpu in gpus:
        if gpu.load * 100 <= max_utilization_pct and gpu.memoryUsed <= max_others_mem_mib:
            available.append(gpu.id)
        else:
            print(f"GPU {gpu.id} busy (util={gpu.load*100:.0f}%, used={gpu.memoryUsed:.0f} MiB) -- excluding.")
    return available