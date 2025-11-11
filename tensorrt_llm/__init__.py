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


def _preload_python_lib():
    """
    Preload Python library.

    On Linux, the python executable links to libpython statically,
    so the dynamic library `libpython3.x.so` is not loaded.
    When using virtual environment on top of non-system Python installation,
    our libraries installed under `$VENV_PREFIX/lib/python3.x/site-packages/`
    have difficulties loading `$PREFIX/lib/libpython3.x.so.1.0` on their own,
    since venv does not symlink `libpython3.x.so` into `$VENV_PREFIX/lib/`,
    and the relative path from `$VENV_PREFIX` to `$PREFIX` is arbitrary.

    We preload the libraries here since the Python executable under `$PREFIX/bin`
    can easily find the library.
    """
    import platform
    on_linux = platform.system() == "Linux"
    if on_linux:
        import sys
        from ctypes import cdll
        v_major, v_minor, *_ = sys.version_info
        pythonlib = f'libpython{v_major}.{v_minor}.so'
        _ = cdll.LoadLibrary(pythonlib + '.1.0')
        _ = cdll.LoadLibrary(pythonlib)


_preload_python_lib()

import sys

# Need to import torch before tensorrt_llm library, otherwise some shared binary files
# cannot be found for the public PyTorch, raising errors like:
# ImportError: libc10.so: cannot open shared object file: No such file or directory
import torch  # noqa

import tensorrt_llm._torch.models as torch_models
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
from .builder import BuildConfig, Builder, BuilderConfig, build
from .disaggregated_params import DisaggregatedParams
from .functional import Tensor, constant
from .llmapi import LLM, MultimodalEncoder
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
    'torch_models',
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
    'quantization',
    'tools',
    'LLM',
    'MultimodalEncoder',
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

print(f"[TensorRT-LLM] TensorRT LLM version: {__version__}")

sys.stdout.flush()
