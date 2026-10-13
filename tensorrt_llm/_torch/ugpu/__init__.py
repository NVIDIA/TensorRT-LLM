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
"""
uGPU Localization: Policy and Runtime layers.

This package provides a clean separation of concerns for uGPU execution:
- layout: Logical/padded shape and partition-slice metadata
- policy: Planning decisions (enable/disable, partition count, backend selection)
- runtime: Stream/mempool/event management
"""

from tensorrt_llm._torch.ugpu.layout import (
    PartitionedTensorLayout,
    make_bf16_linear_output_layout,
    make_nvfp4_linear_output_layout,
)
from tensorrt_llm._torch.ugpu.policy import (
    LinearPartitionPlan,
    PartitionPlan,
    UgpuExecutionPlanner,
    UgpuPolicy,
)
from tensorrt_llm._torch.ugpu.runtime import UgpuRuntime

__all__ = [
    "UgpuPolicy",
    "PartitionPlan",
    "LinearPartitionPlan",
    "PartitionedTensorLayout",
    "make_bf16_linear_output_layout",
    "make_nvfp4_linear_output_layout",
    "UgpuExecutionPlanner",
    "UgpuRuntime",
]
