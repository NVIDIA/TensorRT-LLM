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


def _add_trt_llm_dll_directory():
    import platform
    on_windows = platform.system() == "Windows"
    if on_windows:
        import os
        import sysconfig
        from pathlib import Path
        os.add_dll_directory(
            Path(sysconfig.get_paths()['purelib']) / "tensorrt_llm" / "libs")


_add_trt_llm_dll_directory()

import sys

# Need to import xgrammar before tensorrt_llm library,
# otherwise `MemoryError: std::bad_alloc` pattern error will be raised.
import xgrammar  # noqa

import tensorrt_llm.functional as functional
import tensorrt_llm.math_utils as math_utils
import tensorrt_llm.models as models
import tensorrt_llm.quantization as quantization
import tensorrt_llm.runtime as runtime
import tensorrt_llm.tools as tools

from ._common import _init, default_net, default_trtnet, precision
from ._mnnvl_utils import MnnvlMemory, MnnvlMoe, MoEAlltoallInfo
from ._utils import (default_gpus_per_node, local_mpi_rank, local_mpi_size,
                     mpi_barrier, mpi_comm, mpi_rank, mpi_world_size,
                     set_mpi_comm, str_dtype_to_torch, str_dtype_to_trt,
                     torch_dtype_to_trt)
from .auto_parallel import AutoParallelConfig, auto_parallel
from .builder import BuildConfig, Builder, BuilderConfig, build
from .disaggregated_params import DisaggregatedParams
from .functional import Tensor, constant
from .llmapi import LLM, LlmArgs
from .llmapi.llm_args import LlmArgs, TorchLlmArgs, TrtLlmArgs
from .logger import logger
from .mapping import Mapping
from .models.automodel import AutoConfig, AutoModelForCausalLM
from .module import Module
from .network import Network, net_guard
from .parameter import Parameter
from .python_plugin import PluginBase
from .sampling_params import SamplingParams
from .version import __version__

__all__ = [
    'AutoConfig',
    'AutoModelForCausalLM',
    'logger',
    'str_dtype_to_trt',
    'torch_dtype_to_trt',
    'str_dtype_to_torch',
    'default_gpus_per_node',
    'local_mpi_rank',
    'local_mpi_size',
    'mpi_barrier',
    'mpi_comm',
    'mpi_rank',
    'set_mpi_comm',
    'mpi_world_size',
    'constant',
    'default_net',
    'default_trtnet',
    'precision',
    'net_guard',
    'Network',
    'Mapping',
    'MnnvlMemory',
    'MnnvlMoe',
    'MoEAlltoallInfo',
    'PluginBase',
    'Builder',
    'BuilderConfig',
    'build',
    'BuildConfig',
    'Tensor',
    'Parameter',
    'runtime',
    'Module',
    'functional',
    'models',
    'auto_parallel',
    'AutoParallelConfig',
    'quantization',
    'tools',
    'LLM',
    'LlmArgs',
    'TorchLlmArgs',
    'TrtLlmArgs',
    'SamplingParams',
    'DisaggregatedParams',
    'KvCacheConfig',
    'math_utils',
    '__version__',
]

_init()

print(f"[TensorRT-LLM] TensorRT-LLM version: {__version__}")

sys.stdout.flush()
