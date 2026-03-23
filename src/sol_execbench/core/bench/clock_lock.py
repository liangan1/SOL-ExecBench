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

"""GPU clock locking for stable SOL ExecBench benchmark timing.

Clock locking pins the GPU to a fixed frequency, eliminating the 10-30% latency
variance caused by GPU boost clock fluctuations under thermal pressure.

Supports NVIDIA GPUs via ``nvidia-smi`` and Intel GPUs via the xe driver sysfs
interface.

Entry points:

- ``probe_clock_lock_available()`` — probes whether clock locking is available
  for the current GPU backend.

- ``lock_clocks(device_name)`` — locks GPU and DRAM clocks.  Called once at
  Docker entrypoint or server startup.

- ``unlock_clocks()`` — resets GPU and DRAM clocks.  Best-effort cleanup.

- ``are_clocks_locked()`` — reads ``SOL_EXECBENCH_CLOCKS_LOCKED`` env var to check
  whether clocks were locked at startup.
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import time

from .config.device_config import get_clock_preset

logger = logging.getLogger(__name__)

# Seconds to wait after locking before verifying actual clock frequencies.
VERIFY_DELAY_S = 3


# ---------------------------------------------------------------------------
# Intel GPU sysfs helpers (xe driver)
# ---------------------------------------------------------------------------

def _find_intel_gpu_freq_dir() -> str | None:
    """Find the sysfs freq0 directory for the first Intel discrete GPU (xe driver).

    Returns the path to ``.../tile0/gt0/freq0`` or None if not found.
    """
    # xe driver exposes frequency control at:
    # /sys/class/drm/cardN/device/tile0/gt0/freq0/{min_freq,max_freq,cur_freq}
    pattern = "/sys/class/drm/card*/device/tile0/gt0/freq0/cur_freq"
    matches = sorted(glob.glob(pattern))
    for match in matches:
        freq_dir = os.path.dirname(match)
        # Verify it's an xe device (not i915 integrated)
        device_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(freq_dir))))
        driver_link = os.path.join(device_dir, "driver")
        try:
            driver_target = os.readlink(driver_link)
            if "xe" in driver_target:
                return freq_dir
        except OSError:
            continue
    return None


def _read_sysfs(path: str) -> int | None:
    """Read an integer value from a sysfs file."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _write_sysfs(path: str, value: int) -> bool:
    """Write an integer value to a sysfs file (requires appropriate permissions)."""
    try:
        with open(path, "w") as f:
            f.write(str(value))
        return True
    except OSError as e:
        logger.warning(f"Failed to write {value} to {path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def probe_clock_lock_available() -> bool:
    """Probe whether GPU clock locking is available.

    For NVIDIA: Runs ``sudo -n nvidia-smi -lgc 1`` and immediately resets.
    For Intel: Checks if xe driver sysfs freq interface is writable.

    Returns ``True`` if clock locking is possible, ``False`` otherwise.
    """
    # Try NVIDIA first
    try:
        probe = subprocess.run(
            ["sudo", "-n", "nvidia-smi", "-lgc", "1"],
            capture_output=True,
        )
        if probe.returncode == 0:
            subprocess.run(["sudo", "-n", "nvidia-smi", "-rgc"], capture_output=True)
            return True
    except FileNotFoundError:
        pass

    # Try Intel xe sysfs
    freq_dir = _find_intel_gpu_freq_dir()
    if freq_dir:
        min_freq_path = os.path.join(freq_dir, "min_freq")
        return os.access(min_freq_path, os.W_OK)

    return False


def lock_clocks(device_name: str) -> bool:
    """Lock GPU clocks for the given device.

    Dispatches to NVIDIA (nvidia-smi) or Intel (xe sysfs) based on the device
    name.  GPU frequency can be overridden via ``SOL_EXECBENCH_GPU_CLK_MHZ``.

    Parameters
    ----------
    device_name : str
        GPU device name (e.g. ``"NVIDIA B200"`` or ``"Intel(R) Arc(TM) B580"``).

    Returns
    -------
    bool
        True if clocks were locked successfully.
    """
    if "Intel" in device_name or "Arc" in device_name:
        return _lock_clocks_intel(device_name)
    return _lock_clocks_nvidia(device_name)


def _lock_clocks_nvidia(device_name: str) -> bool:
    """Lock clocks on NVIDIA GPU via nvidia-smi."""
    preset = get_clock_preset(device_name)
    gpu_mhz_str = os.environ.get("SOL_EXECBENCH_GPU_CLK_MHZ")
    dram_mhz_str = os.environ.get("SOL_EXECBENCH_DRAM_CLK_MHZ")

    gpu_mhz = (
        int(gpu_mhz_str) if gpu_mhz_str else (preset.gpu_clk_mhz if preset else None)
    )
    dram_mhz = (
        int(dram_mhz_str) if dram_mhz_str else (preset.dram_clk_mhz if preset else None)
    )

    if gpu_mhz is None:
        logger.warning(
            f"No GPU clock preset for '{device_name}' and SOL_EXECBENCH_GPU_CLK_MHZ not set"
        )
        return False

    if dram_mhz is None:
        logger.warning(
            f"No DRAM clock preset for '{device_name}' and SOL_EXECBENCH_DRAM_CLK_MHZ not set"
        )
        return False

    try:
        subprocess.run(
            ["sudo", "-n", "nvidia-smi", "-lgc", str(gpu_mhz)],
            check=True,
            capture_output=True,
        )
        logger.info(f"GPU clocks locked to {gpu_mhz} MHz")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Failed to lock GPU clocks: {e}")
        return False

    try:
        subprocess.run(
            ["sudo", "-n", "nvidia-smi", "-lmc", str(dram_mhz)],
            check=True,
            capture_output=True,
        )
        logger.info(f"DRAM clocks locked to {dram_mhz} MHz")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Failed to lock DRAM clocks: {e}")
        subprocess.run(["sudo", "-n", "nvidia-smi", "-rgc"], capture_output=True)
        return False

    logger.info(f"Waiting {VERIFY_DELAY_S}s for clocks to stabilize...")
    time.sleep(VERIFY_DELAY_S)
    if not verify_clocks(gpu_mhz, dram_mhz):
        logger.warning("Clock verification failed after locking — unlocking")
        unlock_clocks()
        return False

    return True


def _lock_clocks_intel(device_name: str) -> bool:
    """Lock clocks on Intel GPU via xe driver sysfs.

    Sets min_freq = max_freq = target frequency to pin the GPU clock.
    """
    freq_dir = _find_intel_gpu_freq_dir()
    if not freq_dir:
        logger.warning("Intel GPU xe sysfs freq interface not found")
        return False

    preset = get_clock_preset(device_name)
    gpu_mhz_str = os.environ.get("SOL_EXECBENCH_GPU_CLK_MHZ")
    gpu_mhz = (
        int(gpu_mhz_str) if gpu_mhz_str else (preset.gpu_clk_mhz if preset else None)
    )

    if gpu_mhz is None:
        # Default to rp0 (max hardware frequency)
        rp0 = _read_sysfs(os.path.join(freq_dir, "rp0_freq"))
        if rp0:
            gpu_mhz = rp0
        else:
            logger.warning(f"No clock preset for '{device_name}' and cannot read rp0_freq")
            return False

    min_path = os.path.join(freq_dir, "min_freq")
    max_path = os.path.join(freq_dir, "max_freq")

    # Set max first, then min (to avoid min > max error)
    if not _write_sysfs(max_path, gpu_mhz):
        return False
    if not _write_sysfs(min_path, gpu_mhz):
        return False

    logger.info(f"Intel GPU clocks locked to {gpu_mhz} MHz via sysfs")

    logger.info(f"Waiting {VERIFY_DELAY_S}s for clocks to stabilize...")
    time.sleep(VERIFY_DELAY_S)

    cur = _read_sysfs(os.path.join(freq_dir, "cur_freq"))
    if cur is not None and abs(cur - gpu_mhz) > 100:
        logger.warning(
            f"Intel GPU clock verification failed — expected {gpu_mhz} MHz, got {cur} MHz"
        )
        _unlock_clocks_intel()
        return False

    logger.info(f"Intel GPU clock verified: cur_freq = {cur} MHz")
    return True


def verify_clocks(
    expected_gpu_mhz: int, expected_dram_mhz: int, tolerance_mhz: int = 50
) -> bool:
    """Verify current GPU/DRAM clocks match expected frequencies via nvidia-smi.

    Parameters
    ----------
    expected_gpu_mhz : int
        Expected GPU core clock in MHz.
    expected_dram_mhz : int
        Expected DRAM (memory) clock in MHz.
    tolerance_mhz : int
        Allowed deviation in MHz (default 50).

    Returns
    -------
    bool
        True if all GPUs report clocks within tolerance.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.current.graphics,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"nvidia-smi clock query failed: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        logger.warning("nvidia-smi not found — cannot verify clocks")
        return False

    lines = result.stdout.strip().splitlines()
    if not lines:
        logger.warning("nvidia-smi returned no GPU data")
        return False

    for i, line in enumerate(lines):
        parts = line.split(",")
        if len(parts) < 2:
            logger.warning(f"GPU {i}: unexpected nvidia-smi output: {line!r}")
            return False
        try:
            actual_gpu = int(parts[0].strip())
            actual_dram = int(parts[1].strip())
        except ValueError:
            logger.warning(f"GPU {i}: could not parse clock values: {line!r}")
            return False

        if abs(actual_gpu - expected_gpu_mhz) > tolerance_mhz:
            logger.warning(
                f"GPU {i}: GPU clock mismatch — expected {expected_gpu_mhz} MHz, got {actual_gpu} MHz"
            )
            return False
        if abs(actual_dram - expected_dram_mhz) > tolerance_mhz:
            logger.warning(
                f"GPU {i}: DRAM clock mismatch — expected {expected_dram_mhz} MHz, got {actual_dram} MHz"
            )
            return False

        logger.info(
            f"GPU {i}: clocks verified — GPU {actual_gpu} MHz, DRAM {actual_dram} MHz"
        )

    return True


def unlock_clocks() -> None:
    """Reset GPU and DRAM clocks.  Best-effort — errors are logged but not raised."""
    # NVIDIA
    try:
        subprocess.run(["sudo", "-n", "nvidia-smi", "-rgc"], capture_output=True)
        logger.info("NVIDIA GPU clocks unlocked")
    except Exception as e:
        logger.debug(f"nvidia-smi -rgc failed (expected on non-NVIDIA): {e}")

    try:
        subprocess.run(["sudo", "-n", "nvidia-smi", "-rmc"], capture_output=True)
        logger.info("NVIDIA DRAM clocks unlocked")
    except Exception as e:
        logger.debug(f"nvidia-smi -rmc failed (expected on non-NVIDIA): {e}")

    # Intel
    _unlock_clocks_intel()


def _unlock_clocks_intel() -> None:
    """Reset Intel GPU clocks to hardware defaults via xe sysfs."""
    freq_dir = _find_intel_gpu_freq_dir()
    if not freq_dir:
        return

    rpn = _read_sysfs(os.path.join(freq_dir, "rpn_freq"))  # min hardware freq
    rp0 = _read_sysfs(os.path.join(freq_dir, "rp0_freq"))  # max hardware freq
    if rpn is not None:
        _write_sysfs(os.path.join(freq_dir, "min_freq"), rpn)
    if rp0 is not None:
        _write_sysfs(os.path.join(freq_dir, "max_freq"), rp0)
    logger.info("Intel GPU clocks reset to defaults")


def are_clocks_locked() -> bool:
    """Check whether clocks were locked at startup.

    Reads the ``SOL_EXECBENCH_CLOCKS_LOCKED`` environment variable set by the Docker
    entrypoint or server startup.
    """
    return os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED", "0") == "1"
