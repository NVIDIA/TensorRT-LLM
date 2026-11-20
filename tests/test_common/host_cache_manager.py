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

from unittest.mock import Mock, patch

from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


def make_host_cache_manager(*, max_batch_size: int = 1) -> KVCacheManagerV2:
    """Initialize the cache manager for tests without device or pinned host buffers."""
    native_cache = Mock(buffer_grouping=[], pool_group_descs=[])
    native_cache.get_layer_group_id.return_value = 0
    max_seq_len = 1024
    with (
        patch.object(kv_cache_manager_v2, "IndexMapper", spec_set=True),
        patch.object(KVCacheManagerV2, "_allocate_cache_pools", return_value=native_cache),
        patch.object(KVCacheManagerV2, "get_num_available_tokens", return_value=max_seq_len),
        patch.object(KVCacheManagerV2, "_prepare_page_table_tensor"),
        patch.object(KVCacheManagerV2, "_log_kv_cache_pool_lifecycle_mapping"),
        patch.object(KVCacheManagerV2, "_reserve_guard_page"),
    ):
        return KVCacheManagerV2(
            KvCacheConfig(
                max_gpu_total_bytes=16 << 20,
                host_cache_size=0,
                enable_block_reuse=False,
            ),
            CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=128,
            tokens_per_block=32,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=Mapping(),
            execution_stream=Mock(),
        )
