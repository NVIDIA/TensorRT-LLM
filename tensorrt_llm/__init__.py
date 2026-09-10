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

# The package's public surface is loaded lazily (PEP 562): importing
# tensorrt_llm no longer executes the whole product tree (previously ~99% of
# product modules ran at import time through the eager chain below). Accessing
# any public name (tensorrt_llm.LLM, tensorrt_llm.models, ...) imports just
# what that name needs. The TYPE_CHECKING block keeps the original imports
# visible to static tooling.
import importlib
from typing import TYPE_CHECKING

# Need to import torch before tensorrt_llm library, otherwise some shared binary files
# cannot be found for the public PyTorch, raising errors like:
# ImportError: libc10.so: cannot open shared object file: No such file or directory
import torch  # noqa

from .logger import logger
from .version import __version__

if TYPE_CHECKING:
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
    from .disaggregated_params import DisaggregatedParams
    from .llmapi import LLM, AsyncLLM, KvCacheConfig, MultimodalEncoder
    from .llmapi.llm_args import LlmArgs, TorchLlmArgs
    from .mapping import Mapping
    from .models.automodel import AutoConfig, AutoModelForCausalLM
    from .sampling_params import SamplingParams
    from .visual_gen import (ExtraParamSchema, VisualGen, VisualGenArgs,
                             VisualGenMetrics, VisualGenOutput, VisualGenParams,
                             VisualGenResult)

# Public name -> (source module, attribute); attribute None = the module itself.
_LAZY_ATTRS = {
    'torch_models': ('tensorrt_llm._torch.models', None),
    'math_utils': ('tensorrt_llm.math_utils', None),
    'models': ('tensorrt_llm.models', None),
    'quantization': ('tensorrt_llm.quantization', None),
    'runtime': ('tensorrt_llm.runtime', None),
    'tools': ('tensorrt_llm.tools', None),
    'MnnvlMemory': ('tensorrt_llm._mnnvl_utils', 'MnnvlMemory'),
    'MnnvlMoe': ('tensorrt_llm._mnnvl_utils', 'MnnvlMoe'),
    'MoEAlltoallInfo': ('tensorrt_llm._mnnvl_utils', 'MoEAlltoallInfo'),
    'default_gpus_per_node': ('tensorrt_llm._utils', 'default_gpus_per_node'),
    'local_mpi_rank': ('tensorrt_llm._utils', 'local_mpi_rank'),
    'local_mpi_size': ('tensorrt_llm._utils', 'local_mpi_size'),
    'mpi_barrier': ('tensorrt_llm._utils', 'mpi_barrier'),
    'mpi_comm': ('tensorrt_llm._utils', 'mpi_comm'),
    'mpi_rank': ('tensorrt_llm._utils', 'mpi_rank'),
    'mpi_world_size': ('tensorrt_llm._utils', 'mpi_world_size'),
    'set_mpi_comm': ('tensorrt_llm._utils', 'set_mpi_comm'),
    'str_dtype_to_torch': ('tensorrt_llm._utils', 'str_dtype_to_torch'),
    'DisaggregatedParams':
    ('tensorrt_llm.disaggregated_params', 'DisaggregatedParams'),
    'LLM': ('tensorrt_llm.llmapi', 'LLM'),
    'AsyncLLM': ('tensorrt_llm.llmapi', 'AsyncLLM'),
    'MultimodalEncoder': ('tensorrt_llm.llmapi', 'MultimodalEncoder'),
    'KvCacheConfig': ('tensorrt_llm.llmapi', 'KvCacheConfig'),
    'LlmArgs': ('tensorrt_llm.llmapi.llm_args', 'LlmArgs'),
    'TorchLlmArgs': ('tensorrt_llm.llmapi.llm_args', 'TorchLlmArgs'),
    'Mapping': ('tensorrt_llm.mapping', 'Mapping'),
    'AutoConfig': ('tensorrt_llm.models.automodel', 'AutoConfig'),
    'AutoModelForCausalLM':
    ('tensorrt_llm.models.automodel', 'AutoModelForCausalLM'),
    'SamplingParams': ('tensorrt_llm.sampling_params', 'SamplingParams'),
    'ExtraParamSchema': ('tensorrt_llm.visual_gen', 'ExtraParamSchema'),
    'VisualGen': ('tensorrt_llm.visual_gen', 'VisualGen'),
    'VisualGenArgs': ('tensorrt_llm.visual_gen', 'VisualGenArgs'),
    'VisualGenMetrics': ('tensorrt_llm.visual_gen', 'VisualGenMetrics'),
    'VisualGenOutput': ('tensorrt_llm.visual_gen', 'VisualGenOutput'),
    'VisualGenParams': ('tensorrt_llm.visual_gen', 'VisualGenParams'),
    'VisualGenResult': ('tensorrt_llm.visual_gen', 'VisualGenResult'),
}


def __getattr__(name):
    entry = _LAZY_ATTRS.get(name)
    if entry is not None:
        module_name, attr = entry
        module = importlib.import_module(module_name)
        value = module if attr is None else getattr(module, attr)
        globals()[name] = value  # cache: subsequent access skips __getattr__
        return value
    # Fall back to plain submodules (tensorrt_llm.functional, .profiler, ...)
    # that used to be reachable as attributes via the eager import chain.
    try:
        return importlib.import_module(f'.{name}', __name__)
    except ModuleNotFoundError as e:
        # Only translate "no such submodule" into AttributeError. A
        # ModuleNotFoundError raised *inside* an existing submodule (missing
        # dependency) is a real error and must propagate unchanged.
        if e.name != f'{__name__}.{name}':
            raise
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(__all__) | set(globals()) | set(_LAZY_ATTRS))


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
