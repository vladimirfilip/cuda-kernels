"""Shared CUDA timing helper.

One implementation, used by every benchmark and by the notebooks, so numbers
from different slices are directly comparable and cannot drift apart.
"""

import statistics

import torch


def cuda_time_ms(fn, warmup: int = 25, iters: int = 200) -> float:
    """Median milliseconds per call of ``fn``, a no-arg callable launching GPU work.

    Three deliberate choices:

    * **Warm-up before timing.** The first launches pay context creation, module
      loading and (for Triton) JIT compilation. Including them measures the
      compiler, not the kernel.
    * **CUDA events, not wall clock.** Kernel launches are asynchronous, so
      ``time.perf_counter()`` around a launch measures the enqueue, not the work.
      Events are recorded on the stream and timed by the device.
    * **Median, not mean.** On a shared or thermally-throttled GPU the tail is
      long and one-sided; a mean chases the worst sample, a median does not.

    Note that this synchronises once per iteration, so it reports *latency* of a
    single call rather than the throughput of a pipelined sequence. For kernels
    small enough to be launch-bound the two differ substantially, which is the
    honest thing to report when comparing against a library kernel that has the
    same overhead.
    """
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
