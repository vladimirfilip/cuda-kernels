"""Shared CUDA timing helper."""

import statistics

import torch


def cuda_time_ms(fn, warmup: int = 25, iters: int = 200) -> float:
    """Median ms per call of ``fn`` (a no-arg callable launching GPU work)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)
