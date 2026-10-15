"""Shared execution utilities for simulation episodes."""

import os
import threading
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Iterable
import math


def _initialize_worker(device: str, torch_threads: int) -> None:
    """Configure the device and CPU threads for a simulation worker."""
    import torch
    from sionna.phy import config

    selected_device = torch.device(device)

    if selected_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({device}), but is unavailable")
        torch.cuda.set_device(selected_device)

    torch.set_num_threads(torch_threads)
    config.device = str(selected_device)


def _run_one(episode_function: Callable, kwargs: dict) -> dict:
    """Invoke an episode function with its keyword arguments."""
    return episode_function(**kwargs)


def run_simulations(episode_function: Callable[..., dict],
                    episode_kwargs: Iterable[dict],
                    *,
                    device: str,
                    parallel: bool = False,
                    max_workers: int = 1,
                    torch_threads: int = 1) -> list[dict]:
    """Run independent simulation episodes sequentially or in processes.

    :param episode_function: Module-level function that runs one episode.
        It must accept the keyword arguments supplied in ``episode_kwargs``.
    :param episode_kwargs: Keyword arguments for each episode. Arguments must
        be serializable when parallel execution is enabled.
    :param device: Device configured for every worker, e.g. "cpu" or "cuda:0".
        Episode arguments must select the same device.
    :param parallel: Whether to run episodes in separate worker processes.
    :param max_workers: Maximum number of simultaneous episode processes.
        The caller must account for CPU and GPU resource limits.
    :param torch_threads: Number of PyTorch CPU threads used per worker.

    :returns: Episode results in the same order as the input arguments.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if torch_threads < 1:
        raise ValueError("torch_threads must be at least 1")

    jobs = list(episode_kwargs)
    if not jobs:
        return []

    if not parallel:
        _initialize_worker(device, torch_threads)
        return [episode_function(**kwargs) for kwargs in jobs]

    workers = min(max_workers, len(jobs))

    # Spawn isolates Sionna's configuration and supports CUDA workers.
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=mp.get_context("spawn"),
                             initializer=_initialize_worker,
                             initargs=(device, torch_threads)) as executor:
        execute = partial(_run_one, episode_function)
        return list(executor.map(execute, jobs, chunksize=1))


def _profile_vram_worker(connection,
                         probe_function: Callable,
                         probe_kwargs: dict,
                         device: str,
                         poll_interval: float) -> None:
    """Run a probe and measure GPU memory inside an isolated process."""
    try:
        import torch
        import pynvml as nvml

        _initialize_worker(device, torch_threads=1)
        selected_device = torch.device(device)

        # Use the UUID to respect CUDA_VISIBLE_DEVICES and logical numbering.
        gpu_uuid = str(torch.cuda.get_device_properties(selected_device).uuid)

        nvml.nvmlInit()
        try:
            handle = nvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
            pid = os.getpid()

            stop = threading.Event()
            peak_bytes = [0]
            sampling_errors = []

            def sample_memory() -> None:
                processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for process in processes:
                    if process.pid == pid:
                        used = process.usedGpuMemory
                        if used == nvml.NVML_VALUE_NOT_AVAILABLE:
                            raise RuntimeError("NVML cannot report this process's GPU memory")
                        peak_bytes[0] = max(peak_bytes[0], used)

            def monitor() -> None:
                try:
                    while not stop.wait(poll_interval):
                        sample_memory()
                except Exception:
                    sampling_errors.append(traceback.format_exc())

            torch.cuda.synchronize(selected_device)
            torch.cuda.reset_peak_memory_stats(selected_device)
            sample_memory()

            sampler = threading.Thread(target=monitor, daemon=True)
            sampler.start()

            try:
                probe_function(**probe_kwargs)
                torch.cuda.synchronize(selected_device)
            finally:
                stop.set()
                sampler.join()

            sample_memory()

            if sampling_errors:
                raise RuntimeError(sampling_errors[0])
            if peak_bytes[0] == 0:
                raise RuntimeError("NVML did not observe GPU memory usage")

            mib = 1024 ** 2
            connection.send({
                "ok": True,
                "gpu_uuid": gpu_uuid,
                "peak_process_mib": peak_bytes[0] / mib,
                "peak_torch_allocated_mib": (
                    torch.cuda.max_memory_allocated(selected_device) / mib
                ),
                "peak_torch_reserved_mib": (
                    torch.cuda.max_memory_reserved(selected_device) / mib
                ),
            })
        finally:
            nvml.nvmlShutdown()

    except Exception:
        connection.send({
            "ok": False,
            "error": traceback.format_exc(),
        })
    finally:
        connection.close()


def profile_episode_vram(probe_function: Callable,
                         probe_kwargs: dict,
                         *,
                         device: str = "cuda:0",
                         poll_interval: float = 0.02,
                         safety_factor: float = 1.2,
                         timeout: float = 600) -> dict:
    """Measure a simulation probe's GPU memory in a spawned process.

    :param probe_function: Module-level function that builds and runs the probe.
    :param probe_kwargs: Serializable keyword arguments passed to the function.
    :param device: Logical CUDA device used by the probe.
    :param poll_interval: Seconds between NVML memory samples.
    :param safety_factor: Multiplier applied to the observed process peak.
    :param timeout: Maximum number of seconds allowed for the probe.

    :returns: Measured peaks and a conservative suggested episode budget in MiB.
    """
    if not device.startswith("cuda"):
        raise ValueError("VRAM profiling requires a CUDA device")
    if poll_interval <= 0 or timeout <= 0:
        raise ValueError("poll_interval and timeout must be positive")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least 1")

    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)

    process = context.Process(
        target=_profile_vram_worker,
        args=(
            sender,
            probe_function,
            probe_kwargs,
            device,
            poll_interval,
        ),
    )
    process.start()
    sender.close()

    try:
        if not receiver.poll(timeout):
            raise TimeoutError("The VRAM profiling process timed out")

        try:
            report = receiver.recv()
        except EOFError as exc:
            raise RuntimeError(
                "The VRAM profiling process exited without a report"
            ) from exc

        if not report.pop("ok"):
            raise RuntimeError(report["error"])

        report["suggested_episode_mib"] = math.ceil(report["peak_process_mib"] * safety_factor)
        return report
    finally:
        receiver.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join()