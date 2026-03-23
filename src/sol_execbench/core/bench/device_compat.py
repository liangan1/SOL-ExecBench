# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device compatibility layer using torch.accelerator unified API.

Wraps ``torch.accelerator`` and ``torch.Event`` so the rest of the codebase
never branches on ``cuda`` vs ``xpu`` directly.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, List

import torch
import torch.accelerator as acc


def detect_device() -> str:
    """Return the first available accelerator device, or ``"cpu"``."""
    if acc.is_available():
        return f"{acc.current_accelerator()}:0"
    return "cpu"


def gpu_synchronize(device: str) -> None:
    """Synchronize the accelerator for *device*."""
    d = torch.device(device)
    if d.type in ("cuda", "xpu", "mps"):
        acc.synchronize(d)


def create_events(n: int, device: str) -> List[Any]:
    """Create *n* timing events for *device* using ``torch.Event``."""
    d = torch.device(device)
    return [torch.Event(d.type, enable_timing=True) for _ in range(n)]


def device_context(device: str):
    """Return a device context manager for *device*."""
    d = torch.device(device)
    if d.type == "cuda":
        return torch.cuda.device(device)
    if d.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.device(device)
    return nullcontext()


def get_device_name(device: str) -> str:
    """Return the human-readable name of the GPU at *device*."""
    d = torch.device(device)
    idx = d.index or 0
    if d.type == "cuda":
        return torch.cuda.get_device_name(idx)
    if d.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.get_device_name(idx)
    return ""


def is_gpu_available() -> bool:
    """Return True if any accelerator backend is available."""
    return acc.is_available()


def get_oom_errors() -> tuple:
    """Return a tuple of OOM exception classes for the available backends."""
    classes: list[type] = []
    try:
        if torch.cuda.is_available():
            classes.append(torch.cuda.OutOfMemoryError)
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "OutOfMemoryError"):
            classes.append(torch.xpu.OutOfMemoryError)
    except Exception:
        pass
    return tuple(classes) if classes else (RuntimeError,)
