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
import time
from functools import partial
from typing import Literal, Optional, Tuple, Union

# isort: off
import torch
import tensorrt as trt
# isort: on

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
except ImportError:
    pynvml = None
import traceback

from tensorrt_llm.logger import logger

from ._common import _is_building

if psutil is None:
    logger.warning("A required package 'psutil' is not installed. Will not "
                   "monitor the host memory usages. Please install the package "
                   "first, e.g, 'pip install psutil'.")
if pynvml is None:
    logger.warning(
        "A required package 'psutil' is not installed. Will not "
        "monitor the device memory usages. Please install the package "
        "first, e.g, 'pip install pynvml>=11.5.0'.")
elif pynvml.__version__ < '11.5.0':
    logger.warning(f'Found pynvml=={pynvml.__version__}. Please use '
                   f'pynvml>=11.5.0 to get accurate memory usage')
    # Support legacy pynvml. Note that an old API could return
    # wrong GPU memory usage.
    _device_get_memory_info_fn = pynvml.nvmlDeviceGetMemoryInfo
else:
    _device_get_memory_info_fn = partial(pynvml.nvmlDeviceGetMemoryInfo,
                                         version=pynvml.nvmlMemory_v2)


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


MemUnitType = Literal['GiB', 'MiB', 'KiB']


class PyNVMLContext:

    def __enter__(self):
        if pynvml is not None:
            pynvml.nvmlInit()

    def __exit__(self, type, value, traceback):
        if pynvml is not None:
            pynvml.nvmlShutdown()


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


def device_memory_info(
        device: Optional[Union[torch.device,
                               int]] = None) -> Tuple[int, int, int]:
    if pynvml is not None:
        if device is None:
            device = torch.cuda.current_device()
        index = device.index if isinstance(device, torch.device) else device
        with PyNVMLContext():
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem_info = _device_get_memory_info_fn(handle)
        return mem_info.used, mem_info.free, mem_info.total
    return 0, 0, 0  # used, free, total


def bytes_to_target_unit(mem_bytes: int, unit: MemUnitType) -> float:
    units = {'GiB': 1 << 30, 'MiB': 1 << 20, 'KiB': 1 << 10}
    _rename_map = {'GB': 'GiB', 'MB': 'MiB', 'KB': 'KiB'}
    if unit not in units:
        unit = _rename_map[unit]
    return float(mem_bytes) / units[unit]


def _format(mem_bytes: int, unit: MemUnitType) -> str:
    mem_usage = bytes_to_target_unit(mem_bytes, unit)
    return f'{mem_usage:.4f} ({unit})'


def _print_mem_message(msg: str, tag: Optional[str] = None):
    if tag:
        msg = f'{tag} - {msg}'
    logger.info(f'[MemUsage] {msg}')


def print_host_memory_usage(tag: Optional[str] = None,
                            unit: MemUnitType = 'GiB'):
    if psutil is None:
        return
    alloc_mem, _, _ = host_memory_info()
    msg = f'Allocated Host Memory {_format(alloc_mem, unit)}'
    _print_mem_message(msg, tag)


def print_device_memory_usage(
    tag: Optional[str] = None,
    unit: MemUnitType = 'GiB',
    device: Optional[Union[torch.device, int]] = None,
):
    alloc_mem, _, _ = device_memory_info(device)
    msg = f'Allocated Device Memory {_format(alloc_mem, unit)}'
    _print_mem_message(msg, tag)


def print_memory_usage(
    tag: Optional[str] = None,
    unit: MemUnitType = 'GiB',
    device: Optional[Union[torch.device, int]] = None,
):
    alloc_host_mem, _, _ = host_memory_info()
    alloc_device_mem, _, _ = device_memory_info(device=device)
    msg = f'Allocated Memory: Host {_format(alloc_host_mem, unit)} '\
            f'Device {_format(alloc_device_mem, unit)}'
    _print_mem_message(msg, tag)


@_is_building
def check_gpt_mem_usage(engine, kv_dtype, use_gpt_attention_plugin,
                        paged_kv_cache, max_batch_size, max_beam_width,
                        max_input_len, max_output_len, local_num_kv_heads,
                        head_size, num_layers):
    # Get the amount of memory
    runtime = trt.Runtime(logger.trt_logger)
    # 1. TensorRT engine activation memory
    activation_size = 0
    try:
        cuda_engine = runtime.deserialize_cuda_engine(engine)
        assert cuda_engine is not None
        activation_size = cuda_engine.device_memory_size / 1024 / 1024
        del cuda_engine
    except Exception:
        logger.warning(
            f'Exception when deserializing engine: {traceback.format_exc()}')
        logger.warning(f'Activation memory size will be regarded as 0.')
    logger.info(f'Activation memory size: {activation_size:.2f} MiB')

    # 2. Weights
    weights_size = bytes_to_target_unit(engine.nbytes, 'MiB')
    logger.info(f'Weights memory size: {weights_size:.2f} MiB')

    # 3. Estimated max KV Cache size
    kv_cache_size = max_batch_size * max_beam_width * 2 * local_num_kv_heads * (
        max_input_len +
        max_output_len) * head_size * num_layers * kv_dtype.itemsize
    # without plugin, we need two set of kv cache buffers,
    # one for inputs, and the other for outputs.
    if not use_gpt_attention_plugin:
        kv_cache_size *= 2
    kv_cache_size = bytes_to_target_unit(kv_cache_size, 'MiB')
    logger.info(f'Max KV Cache memory size: {kv_cache_size:.2f} MiB')

    # Estimated total amount of memory
    est_memory_size = activation_size + weights_size + kv_cache_size
    logger.info(
        f'Estimated max memory usage on runtime: {est_memory_size:.2f} MiB')
    _, _, total_mem = device_memory_info(torch.cuda.current_device())
    total_mem = bytes_to_target_unit(total_mem, 'MiB')
    if est_memory_size > total_mem:
        logger.warning(
            f'Engine is successfully built, but GPU Memory ({total_mem:.2f} MB)'
            ' may not be enough when inferencing on max shape.')
        if paged_kv_cache:
            logger.warning(f'Since paged_kv_cache is enabled, the max KV Cache '
                           'memory size is a estimate for very extreme cases, '
                           'it\'s possible that most cases won\'t meet OOM.')
        else:
            logger.warning(f'Enabling `--paged_kv_cache` could help reduce the '
                           'GPU memory usage on runtime.')

    return est_memory_size
