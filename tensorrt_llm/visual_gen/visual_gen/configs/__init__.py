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

from visual_gen.utils.logger import get_logger

from .diffusion_cache import TeaCacheConfig
from .op_manager import (
    AttentionOpManager,
    LinearOpManager,
    SparseVideogenConfig,
    SparseVideogenConfig2,
    attention_op_context,
    linear_op_context,
)
from .parallel import (
    DiTParallelConfig,
    RefinerDiTParallelConfig,
    T5ParallelConfig,
    VAEParallelConfig,
    dit_parallel_config_context,
    get_dit_parallel_config,
)
from .pipeline import PipelineConfig

logger = get_logger(__name__)

__all__ = [
    "TeaCacheConfig",
    "DiTParallelConfig",
    "RefinerDiTParallelConfig",
    "T5ParallelConfig",
    "VAEParallelConfig",
    "PipelineConfig",
    "AttentionOpManager",
    "LinearOpManager",
    "SparseVideogenConfig",
    "SparseVideogenConfig2",
    "get_dit_parallel_config",
    "linear_op_context",
    "attention_op_context",
    "dit_parallel_config_context",
]
