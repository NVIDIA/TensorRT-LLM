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
from typing import Tuple

import torch

from .logger import ad_logger


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


def get_mem_info_in_mb(empty_cache: bool = True) -> Tuple[int, int]:
    if empty_cache:
        # Clear the memory cache to get the exact free memory
        torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info()
    MB = 1024**2
    return free_mem // MB, total_mem // MB
