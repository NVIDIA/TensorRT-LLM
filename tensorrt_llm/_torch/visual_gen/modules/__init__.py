# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Visual Generation Modules

This module provides modular neural network components for visual generation models.
"""

from .attention import Attention, QKVMode
from .tp_sequence_parallel import (
    RowNorm,
    TokenShardPlan,
    TPSequenceParallel,
    quantize_nvfp4,
    regroup_swizzled_sf,
    static_nvfp4_input_scale,
)

__all__ = [
    "Attention",
    "QKVMode",
    "RowNorm",
    "TokenShardPlan",
    "TPSequenceParallel",
    "quantize_nvfp4",
    "regroup_swizzled_sf",
    "static_nvfp4_input_scale",
]
