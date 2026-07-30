# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import contextlib
import time
from functools import partial
from typing import Literal, Optional, Tuple, Union

import torch

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
except ImportError:
    pynvml = None

from tensorrt_llm.logger import logger

if psutil is None:
    logger.warning(
        "A required package 'psutil' is not installed. Will not "
        "monitor the host memory usages. Please install the package "
        "first, e.g, 'pip install psutil'."
    )

if pynvml is None:
    logger.warning(
        "A required package 'pynvml' is not installed. Will not "
        "monitor the device memory usages. Please install the package "
        "first, e.g, 'pip install nvidia-ml-py>=12'."
    )


class Timer:
    def __init__(self):
        self._start_times = {}
        self._total_elapsed_times = {}

    def start(self, tag):
        self._start_times[tag] = time.time()

    def stop(self, tag) -> float:
        elapsed_time = time.time() - self._start_times[tag]
        if tag not in self._total_elapsed_times:
            self._total_elapsed_times[tag] = 0
        self._total_elapsed_times[tag] += elapsed_time
        return elapsed_time

    def elapsed_time_in_sec(self, tag) -> float:
        if tag not in self._total_elapsed_times:
            return None
        return self._total_elapsed_times[tag]

    def reset(self, tag=None) -> None:
        if tag is None:
            self._start_times.clear()
            self._total_elapsed_times.clear()
        else:
            self._start_times.pop(tag, None)
            self._total_elapsed_times.pop(tag, None)

    def summary(self):
        logger.info("Profile Results")
        for tag, elapsed_time in self._total_elapsed_times.items():
            logger.info(f" - {tag.ljust(30, '.')}: {elapsed_time:.6f} (sec)")


_default_timer = Timer()


def start(tag):
    _default_timer.start(tag)


def stop(tag):
    return _default_timer.stop(tag)


def elapsed_time_in_sec(tag):
    return _default_timer.elapsed_time_in_sec(tag)


def reset(tag=None):
    _default_timer.reset(tag=tag)


def summary():
    _default_timer.summary()


MemUnitType = Literal["GiB", "MiB", "KiB"]


@contextlib.contextmanager
def pynvml_context():
    has_pynvml = pynvml is not None
    if has_pynvml:
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            has_pynvml = False

    try:
        yield
    finally:
        if has_pynvml:
            pynvml.nvmlShutdown()


if pynvml is not None:
    with pynvml_context():
        _device_get_memory_info_fn = partial(
            pynvml.nvmlDeviceGetMemoryInfo,
            version=pynvml.nvmlMemory_v2,
        )


def host_memory_info(pid: Optional[int] = None) -> Tuple[int, int, int]:
    if psutil is not None:
        process = psutil.Process(pid)
        # USS reports the amount of memory that would be freed if the process
        # was terminated right now.
        #   https://psutil.readthedocs.io/en/latest/index.html#psutil.Process.memory_full_info
        vmem = psutil.virtual_memory()
        total_mem = vmem.total
        free_mem = vmem.available
        alloc_mem = process.memory_full_info().uss
        return alloc_mem, free_mem, total_mem
    return 0, 0, 0  # used, free, total


def device_memory_info(device: Optional[Union[torch.device, int]] = None) -> Tuple[int, int, int]:
    if pynvml is not None:
        if device is None:
            device = torch.cuda.current_device()
        index = device.index if isinstance(device, torch.device) else device
        with pynvml_context():
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem_info = _device_get_memory_info_fn(handle)
        return mem_info.used, mem_info.free, mem_info.total
    return 0, 0, 0  # used, free, total


def bytes_to_target_unit(mem_bytes: int, unit: MemUnitType) -> float:
    units = {"GiB": 1 << 30, "MiB": 1 << 20, "KiB": 1 << 10}
    _rename_map = {"GB": "GiB", "MB": "MiB", "KB": "KiB"}
    if unit not in units:
        unit = _rename_map[unit]
    return float(mem_bytes) / units[unit]


def _format(mem_bytes: int, unit: MemUnitType) -> str:
    mem_usage = bytes_to_target_unit(mem_bytes, unit)
    return f"{mem_usage:.4f} ({unit})"


def _print_mem_message(msg: str, tag: Optional[str] = None):
    if tag:
        msg = f"{tag} - {msg}"
    logger.info(f"[MemUsage] {msg}")


def print_host_memory_usage(tag: Optional[str] = None, unit: MemUnitType = "GiB"):
    if psutil is None:
        return
    alloc_mem, _, _ = host_memory_info()
    msg = f"Allocated Host Memory {_format(alloc_mem, unit)}"
    _print_mem_message(msg, tag)


def print_device_memory_usage(
    tag: Optional[str] = None,
    unit: MemUnitType = "GiB",
    device: Optional[Union[torch.device, int]] = None,
):
    alloc_mem, _, _ = device_memory_info(device)
    msg = f"Allocated Device Memory {_format(alloc_mem, unit)}"
    _print_mem_message(msg, tag)


def print_memory_usage(
    tag: Optional[str] = None,
    unit: MemUnitType = "GiB",
    device: Optional[Union[torch.device, int]] = None,
):
    alloc_host_mem, _, _ = host_memory_info()
    alloc_device_mem, _, _ = device_memory_info(device=device)
    msg = (
        f"Allocated Memory: Host {_format(alloc_host_mem, unit)} "
        f"Device {_format(alloc_device_mem, unit)}"
    )
    _print_mem_message(msg, tag)
