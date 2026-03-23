# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Timing utilities for benchmarking SOL ExecBench kernel solutions.

Uses GPU events (CUDA or XPU) for L2-cache-cleared, warmed-up timing.
Derived from triton.testing.do_bench (MIT licence); argument cloning and
synchronization follow sol-bench conventions for accurate measurements.
"""

# This file includes code derived from Triton (https://github.com/openai/triton)
#
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from typing import Any, List, Literal, Optional, Union

import torch


def _quantile(a: list[float], q: Sequence[float]) -> list[float]:
    """Compute quantiles of a list of values."""
    n = len(a)
    a = sorted(a)

    def get_quantile(q: float) -> float:
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(qi) for qi in q]


def _summarize_statistics(
    times: list[float],
    quantiles: Sequence[float] | None,
    return_mode: Literal["min", "max", "mean", "median", "all"],
) -> Union[float, list[float]]:
    """Summarize timing statistics based on return mode."""
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            return ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)
    raise ValueError(f"Unknown return_mode: {return_mode}")


def _get_empty_cache_for_benchmark(device: str = "cuda") -> torch.Tensor:
    """Create a 256 MB buffer for clearing L2 cache before benchmark runs."""
    cache_size = 256 * 1024 * 1024
    return torch.empty(int(cache_size // 4), dtype=torch.int, device=device)


def _clear_cache(cache: torch.Tensor) -> None:
    """Clear the cache buffer by zeroing it."""
    cache.zero_()


def clone_args(args: list[Any]) -> list[Any]:
    """Clone tensor arguments to prevent cross-iteration data contamination.

    Returns fresh copies of all tensor arguments so each benchmark iteration
    starts with independent data.  Non-tensor arguments are passed through.
    """
    return [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]


def _tensor_bytes(obj: Any) -> int:
    """Return the total bytes of all tensors in *obj* (flat or nested list)."""
    if isinstance(obj, torch.Tensor):
        return obj.nelement() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(item) for item in obj)
    return 0


def do_bench(
    fn: Callable[..., Any],
    warmup: int = 10,
    rep: int = 100,
    grad_to_none: list[torch.Tensor] | None = None,
    quantiles: Sequence[float] | None = None,
    return_mode: Literal["min", "max", "mean", "median", "all"] = "mean",
    setup: Callable[[], Any] | None = None,
    device: str = "cuda",
) -> Union[float, list[float]]:
    """Benchmark the runtime of the provided function.

    Derived from triton.testing.do_bench (MIT licence), with fixes from
    sol-bench: explicit synchronization before each start event, L2 cache
    clearing during warmup, and a setup callback for argument cloning.

    Parameters
    ----------
    fn : Callable[..., Any]
        The function to benchmark.  If *setup* is provided, *fn* receives
        the return value of *setup* as its sole argument.
    warmup : int
        Number of warmup iterations (default: 10).
    rep : int
        Number of timed iterations (default: 100).
    grad_to_none : list[torch.Tensor] | None
        Tensors whose ``.grad`` should be cleared before each timed iteration.
    quantiles : Sequence[float] | None
        If provided, return the specified quantiles instead of *return_mode*.
    return_mode : {"min", "max", "mean", "median", "all"}
        How to summarize the timing results (default: ``"mean"``).
    setup : Callable[[], Any] | None
        Called before each timed iteration; its return value is passed to *fn*.
        Setup time is **not** included in measurements.
    device : str
        CUDA device for cache-clearing buffer (default: ``"cuda"``).

    Returns
    -------
    float | list[float]
        Benchmark result(s) in milliseconds.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    from .device_compat import create_events, gpu_synchronize

    cache = _get_empty_cache_for_benchmark(device)
    start_events = create_events(rep, device)
    end_events = create_events(rep, device)

    # Pin all setup results so CPython cannot recycle their id() values.
    # Without this, adversarial code can cache results keyed on tensor id()
    # and return them instantly on subsequent iterations.
    _pinned = []
    _pinned_bytes = 0
    # 15GB limit to prevent VRAM cannibalization when running many iterations
    _max_pinned_bytes = 15 * 1024 * 1024 * 1024

    def _get_args():
        nonlocal setup, _pinned, _pinned_bytes, _max_pinned_bytes
        args = setup()
        _pinned_bytes += _tensor_bytes(args)
        while _pinned_bytes > _max_pinned_bytes and _pinned:
            _pinned_bytes -= _tensor_bytes(_pinned.pop(0))
        _pinned.append(args)
        return args

    for _ in range(warmup):
        _clear_cache(cache)
        if setup is not None:
            args = _get_args()
            fn(args)
        else:
            fn()
    gpu_synchronize(device)

    # Timed iterations
    for i in range(rep):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        _clear_cache(cache)
        if setup is not None:
            args = _get_args()
            gpu_synchronize(device)
            start_events[i].record()
            fn(args)
        else:
            gpu_synchronize(device)
            start_events[i].record()
            fn()
        end_events[i].record()

    gpu_synchronize(device)
    del _pinned
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return _summarize_statistics(times, quantiles, return_mode)


def time_runnable(
    fn: Any,
    setup: Optional[Callable[[], List[Any]]],
    device: str,
    warmup: int = 10,
    rep: int = 100,
) -> float:
    """Time the execution of a callable using CUDA events.

    Each timed iteration receives freshly prepared arguments via the ``setup``
    callback so in-place kernels cannot exploit shared state across iterations.
    Setup time is excluded from measurements.

    Parameters
    ----------
    fn : callable
        The function to benchmark.
    setup : Callable[[], List[Any]]
        Called before each iteration to produce the argument list for *fn*.
        Setup time is excluded from measurements.
    device : str
        The CUDA device to run the benchmark on (e.g. ``"cuda:0"``).
    warmup : int
        Number of warmup iterations (default: 10).
    rep : int
        Number of timed iterations (default: 100).

    Returns
    -------
    float
        Mean execution time in milliseconds.
    """
    from .device_compat import device_context

    with device_context(device):
        return do_bench(
            fn=lambda prepared_args: fn(*prepared_args),
            warmup=warmup,
            rep=rep,
            setup=setup,
            device=device,
        )
