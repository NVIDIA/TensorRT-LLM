# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This module is the package's compatibility surface: the re-exports below and
# the `__all__` at the bottom are what `from tensorrt_llm import X` resolves to,
# and later tasks of this layout migration extend that surface rather than
# replace it. The bootstrap mechanics live in `_bootstrap.py`, which keeps its
# module scope to the standard library so that importing it here cannot pull in
# torch ahead of the environment preparation it performs.
import sys

from ._bootstrap import _init, _prepare_environment

# Phase 1: DLL search path, Python-library preload and vendored triton_kernels
# precedence. Must run before torch and before any TensorRT-LLM shared object.
_prepare_environment()

# Need to import torch before tensorrt_llm library, otherwise some shared binary files
# cannot be found for the public PyTorch, raising errors like:
# ImportError: libc10.so: cannot open shared object file: No such file or directory
import torch  # noqa

import tensorrt_llm._torch.models as torch_models
import tensorrt_llm.math_utils as math_utils
import tensorrt_llm.models as models
import tensorrt_llm.quantization as quantization
import tensorrt_llm.runtime as runtime
import tensorrt_llm.tools as tools

from ._mnnvl_utils import MnnvlMemory, MnnvlMoe, MoEAlltoallInfo
from ._utils import (default_gpus_per_node, local_mpi_rank, local_mpi_size,
                     mpi_barrier, mpi_comm, mpi_rank, mpi_world_size,
                     set_mpi_comm, str_dtype_to_torch)
from .executor.params.disaggregation import DisaggregatedParams
from .executor.params.sampling import SamplingParams
from .llmapi import LLM, AsyncLLM, MultimodalEncoder
from .llmapi.llm_args import LlmArgs, TorchLlmArgs
from .logger import logger
from .mapping import Mapping
from .models.automodel import AutoConfig, AutoModelForCausalLM
from .version import __version__
from .visual_gen import (ExtraParamSchema, VisualGen, VisualGenArgs,
                         VisualGenMetrics, VisualGenOutput, VisualGenParams,
                         VisualGenResult)

__all__ = [
    'AutoConfig',
    'AutoModelForCausalLM',
    'logger',
    'str_dtype_to_torch',
    'default_gpus_per_node',
    'local_mpi_rank',
    'local_mpi_size',
    'mpi_barrier',
    'mpi_comm',
    'mpi_rank',
    'set_mpi_comm',
    'mpi_world_size',
    'torch_models',
    'Mapping',
    'MnnvlMemory',
    'MnnvlMoe',
    'MoEAlltoallInfo',
    'runtime',
    'models',
    'quantization',
    'tools',
    'LLM',
    'AsyncLLM',
    'MultimodalEncoder',
    'LlmArgs',
    'TorchLlmArgs',
    'SamplingParams',
    'VisualGenArgs',
    'ExtraParamSchema',
    'VisualGenMetrics',
    'VisualGenOutput',
    'VisualGenResult',
    'DisaggregatedParams',
    'KvCacheConfig',
    'math_utils',
    'VisualGen',
    'VisualGenParams',
    '__version__',
]

_init()

print(f"[TensorRT-LLM] TensorRT LLM version: {__version__}")

sys.stdout.flush()
