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
import tensorrt_llm.functional as functional
import tensorrt_llm.models as models
import tensorrt_llm.quantization as quantization
import tensorrt_llm.runtime as runtime
import tensorrt_llm.tools as tools

from ._common import _init, default_net, default_trtnet, precision
# Disable flake8 on the line below because mpi_rank is not used in tensorrt_llm project
# but may be called in dependencies (such as examples)
from ._utils import mpi_rank  # NOQA
from ._utils import mpi_world_size, str_dtype_to_torch, str_dtype_to_trt  # NOQA
from .builder import Builder, BuilderConfig
from .functional import Tensor, constant
from .hlapi.llm import LLM, ModelConfig
from .logger import logger
from .mapping import Mapping
from .module import Module
from .network import Network, net_guard
from .parameter import Parameter
from .version import __version__

__all__ = [
    'logger',
    'str_dtype_to_trt',
    'str_dtype_to_torch'
    'mpi_rank',
    'mpi_world_size',
    'constant',
    'default_net',
    'default_trtnet',
    'precision',
    'net_guard',
    'Network',
    'Mapping',
    'Builder',
    'BuilderConfig',
    'Tensor',
    'Parameter',
    'runtime',
    'Module',
    'functional',
    'models',
    'quantization',
    'tools',
    'LLM',
    'ModelConfig',
    '__version__',
]

_init(log_level="error")
