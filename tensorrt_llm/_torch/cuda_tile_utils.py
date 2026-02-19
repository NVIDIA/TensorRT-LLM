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

import functools
import importlib.metadata
import os
import platform
import shutil

import torch

from ..logger import logger

IS_CUDA_TILE_AVAILABLE = False


@functools.lru_cache()
def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def ceil_div(a, b):
    return (a + b - 1) // b


if platform.system() != "Windows":
    try:
        import cuda.tile  # noqa: F401
    except ImportError:
        logger.warning("cuda-tile package not found, TileIR kernels will not be available")
    else:
        if (cc := torch.cuda.get_device_properties()) and (cc.major, cc.minor) < (10, 0):
            logger.warning(
                f"TileIR requires compute capability 10.0 or higher, but the current device has "
                f"{cc.major}.{cc.minor}. TileIR kernels will not be available"
            )
        elif shutil.which("tileiras") is not None:
            IS_CUDA_TILE_AVAILABLE = True
        # For systems without tileiras installed, try to locate from nvidia-cuda-tileiras package.
        elif tileiras_files := importlib.metadata.files("nvidia-cuda-tileiras"):
            for pkg_file in tileiras_files:
                if pkg_file.name == "tileiras":
                    tileiras_dir = pkg_file.locate().parent
                    os.environ["PATH"] = f"{os.environ['PATH']}:{tileiras_dir}"
                    break
            assert shutil.which("tileiras") is not None
            IS_CUDA_TILE_AVAILABLE = True
        else:
            logger.warning("tileiras compiler not found, TileIR kernels will not be available")
