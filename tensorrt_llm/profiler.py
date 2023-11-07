# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time
from functools import partial
from typing import Literal, Optional, Union

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
except ImportError:
    pynvml = None
import torch

from tensorrt_llm.logger import logger


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

    def reset(self):
        self._start_times.clear()
        self._total_elapsed_times.clear()

    def summary(self):
        logger.info('Profile Results')
        for tag, elapsed_time in self._total_elapsed_times.items():
            logger.info(f' - {tag.ljust(30, ".")}: {elapsed_time:.6f} (sec)')


_default_timer = Timer()


def start(tag):
    _default_timer.start(tag)


def stop(tag):
    return _default_timer.stop(tag)


def elapsed_time_in_sec(tag):
    return _default_timer.elapsed_time_in_sec(tag)


def reset():
    _default_timer.reset()


def summary():
    _default_timer.summary()


_pynvml_initialized = False


def initialize_pynvml():
    global _pynvml_initialized
    if pynvml is not None and not _pynvml_initialized:
        pynvml.nvmlInit()
        _pynvml_initialized = True


def finalize_pynvml():
    global _pynvml_initialized
    if pynvml is not None and _pynvml_initialized:
        pynvml.nvmlInvmlShutdownnit()
        _pynvml_initialized = False


class MemoryMonitor:

    TAG = '[MemUsage]'
    UnitType = Literal['GiB', 'MiB', 'KiB']
    units = {'GiB': 1 << 30, 'MiB': 1 << 20, 'KiB': 1 << 10}
    # For convenience.
    _rename_map = {'GB': 'GiB', 'MB': 'MiB', 'KiB': 'KB'}

    _maybe_warned = False

    def __init__(self):
        # bytes
        self._peak_host_memory = 0
        self._peak_device_memory = 0
        self._check_required_packages()

        self.device_handles = {}
        initialize_pynvml()

        if pynvml.__version__ < '11.5.0':
            logger.warning(f'Found pynvml=={pynvml.__version__}. Please use '
                           f'pynvml>=11.5.0 to get accurate memory usage')
            # Support legacy pynvml. Note that an old API could return
            # wrong GPU memory usage.
            self._device_mem__fn = pynvml.nvmlDeviceGetMemoryInfo
        else:
            self._device_mem__fn = partial(pynvml.nvmlDeviceGetMemoryInfo,
                                           version=pynvml.nvmlMemory_v2)

    @classmethod
    def _check_required_packages(cls):
        if cls._maybe_warned:
            return
        if psutil is None:
            # Warning once.
            logger.warning(
                "A required package 'psutil' is not installed. Will not "
                "monitor the host memory usages. Please install the package "
                "first, e.g, 'pip install psutil'.")
            return
        if pynvml is None:
            # Warning once.
            logger.warning(
                "A required package 'psutil' is not installed. Will not "
                "monitor the host memory usages. Please install the package "
                "first, e.g, 'pip install pynvml>=11.5.0'.")
        cls._maybe_warned = True

    def host_memory_info(self) -> int:
        process = psutil.Process()
        # USS reports the amount of memory that would be freed if the process
        # was terminated right now.
        #   https://psutil.readthedocs.io/en/latest/index.html#psutil.Process.memory_full_info
        vmem = psutil.virtual_memory()
        total_mem = vmem.total
        free_mem = vmem.available
        alloc_mem = process.memory_full_info().uss
        if alloc_mem > self._peak_host_memory:
            self._peak_host_memory = alloc_mem
        return alloc_mem, free_mem, total_mem

    def device_memory_info(
        self,
        device: Optional[Union[torch.device, int]] = None,
    ) -> int:
        index = torch._utils._get_device_index(device, optional=True)
        if index not in self.device_handles:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.device_handles[index] = handle
        mem_info = self._device_mem__fn(self.device_handles[index])
        if mem_info.used > self._peak_device_memory:
            self._peak_device_memory = mem_info.used
        return mem_info.used, mem_info.free, mem_info.total

    @staticmethod
    def _normalize_unit_name(unit: str):
        # Rename GB -> GiB.
        return {'GB': 'GiB', 'MB': 'MiB', 'KiB': 'KB'}[unit]

    @classmethod
    def _format(cls, mem_bytes: int, unit: UnitType) -> str:
        if unit not in cls.units:
            unit = cls._rename_map[unit]
        mem_usage = float(mem_bytes) / cls.units[unit]
        return f'{mem_usage:.4f} ({unit})'

    @classmethod
    def _print_message(cls, msg: str, tag: Optional[str] = None):
        if tag:
            msg = f'{tag} - {msg}'
        logger.info(f'{cls.TAG} {msg}')

    def print_host_memory_usage(self,
                                tag: Optional[str] = None,
                                unit: UnitType = 'GiB'):
        if psutil is None:
            return
        alloc_mem, _, _ = self.host_memory_info()
        msg = f'Allocated Host Memory {self._format(alloc_mem, unit)}'
        self._print_message(msg, tag)

    def print_device_memory_usage(
        self,
        tag: Optional[str] = None,
        unit: UnitType = 'GB',
        device: Optional[Union[torch.device, int]] = None,
    ):
        alloc_mem, _, _ = self.device_memory_info(device)
        msg = f'Allocated Device Memory {self._format(alloc_mem, unit)}'
        self._print_message(msg, tag)

    def print_memory_usage(
        self,
        tag: Optional[str] = None,
        unit: UnitType = 'GiB',
        device: Optional[Union[torch.device, int]] = None,
    ):
        alloc_host_mem, _, _ = self.host_memory_info()
        alloc_device_mem, _, _ = self.device_memory_info(device=device)
        msg = f'Allocated Memory: Host {self._format(alloc_host_mem, unit)} '\
              f'Device {self._format(alloc_device_mem, unit)}'
        self._print_message(msg, tag)

    def print_peak_memory_usage(self, unit: UnitType = 'GiB'):
        self._print_message(
            f'Peak Memory Usage: '
            f'Host {self._format(self._peak_host_memory, unit)} '
            f'Device {self._format(self._peak_device_memory, unit)}')


if psutil is not None and pynvml is not None:
    _default_memory_monitor = MemoryMonitor()
else:
    _default_memory_monitor = None


def host_memory_info():
    if _default_memory_monitor is not None:
        return _default_memory_monitor.host_memory_info()


def device_memory_info(device: Optional[Union[torch.device, int]] = None):
    if _default_memory_monitor is not None:
        return _default_memory_monitor.device_memory_info(device)


def print_host_memory_usage(tag: Optional[str] = None,
                            unit: MemoryMonitor.UnitType = 'GiB'):
    if _default_memory_monitor is not None:
        _default_memory_monitor.print_host_memory_usage(tag=tag, unit=unit)


def print_device_memory_usage(tag: Optional[str] = None,
                              unit: MemoryMonitor.UnitType = 'GiB'):
    if _default_memory_monitor is not None:
        _default_memory_monitor.print_device_memory_usage(tag=tag, unit=unit)


def print_memory_usage(tag: Optional[str] = None,
                       unit: MemoryMonitor.UnitType = 'GiB'):
    if _default_memory_monitor is not None:
        _default_memory_monitor.print_memory_usage(tag=tag, unit=unit)


def print_peak_memory_usage(unit: MemoryMonitor.UnitType = 'GiB'):
    if _default_memory_monitor is not None:
        _default_memory_monitor.print_peak_memory_usage(unit=unit)
