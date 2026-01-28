# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import gc
from contextlib import contextmanager
from typing import Literal, Tuple, Union

import torch

from .logger import ad_logger

Number = Union[int, float]
ByteUnit = Literal["B", "KB", "MB", "GB", "TB"]


@contextmanager
def cuda_memory_tracker(logger=ad_logger):
    """
    Context manager to track CUDA memory allocation differences.

    Logs a warning if there is an increase in memory allocation after the
    code block, which might indicate a potential memory leak.
    """
    mem_before = torch.cuda.memory_allocated()
    try:
        yield
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        mem_after = torch.cuda.memory_allocated()
        leaked = mem_after - mem_before
        if leaked > 0:
            logger.warning(f"Potential memory leak detected, leaked memory: {leaked} bytes")


def bytes_to(bytes: int, *more_bytes: int, unit: ByteUnit) -> Union[Number, Tuple[Number, ...]]:
    units = {"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30, "TB": 1 << 40}
    bytes_converted = (bytes,) + more_bytes
    unit = unit.upper()
    if unit != "B":
        bytes_converted = tuple(float(x) / units[unit.upper()] for x in bytes_converted)
    return bytes_converted if more_bytes else bytes_converted[0]


def get_mem_info(
    empty_cache: bool = True, unit: ByteUnit = "B"
) -> Tuple[Number, Number, Number, Number, Number]:
    """Get the memory information of the current device.

    Args:
        empty_cache: Whether to empty the memory cache.
        unit: The unit of the memory information. Defaults to bytes.

    Returns:
        A tuple of the
            - total memory,
            - free memory,
            - reserved memory,
            - allocated memory,
            - fragmented memory
        in the specified unit.
    """
    if empty_cache:
        # Clear the memory cache to get the exact free memory
        torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info()
    res_mem = torch.cuda.memory_reserved()
    alloc_mem = torch.cuda.memory_allocated()
    frag_mem = res_mem - alloc_mem
    return bytes_to(total_mem, free_mem, res_mem, alloc_mem, frag_mem, unit=unit)
