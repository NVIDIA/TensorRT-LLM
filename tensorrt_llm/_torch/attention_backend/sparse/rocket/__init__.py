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

from .backend import RocketTrtllmAttention, RocketVanillaAttention
from .cache_manager import RocketKVCacheManager
from .kernels import (
    triton_rocket_batch_to_flatten,
    triton_rocket_paged_kt_cache_bmm,
    triton_rocket_qk_split,
    triton_rocket_reduce_scores,
    triton_rocket_update_kt_cache_ctx,
    triton_rocket_update_kt_cache_gen,
)
from .metadata import RocketTrtllmAttentionMetadata, RocketVanillaAttentionMetadata

__all__ = [
    "RocketTrtllmAttention",
    "RocketVanillaAttention",
    "RocketKVCacheManager",
    "RocketTrtllmAttentionMetadata",
    "RocketVanillaAttentionMetadata",
    "triton_rocket_qk_split",
    "triton_rocket_batch_to_flatten",
    "triton_rocket_update_kt_cache_gen",
    "triton_rocket_update_kt_cache_ctx",
    "triton_rocket_paged_kt_cache_bmm",
    "triton_rocket_reduce_scores",
]
