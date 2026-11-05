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
"""Unit tests for MambaHybridCacheManagerV2 under helix context parallelism."""

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_mamba_hybrid_quota_to_tokens_under_helix(cp_size: int) -> None:
    """A rank-local byte quota converts to GLOBAL tokens in whole ledger
    blocks; recurrent state is charged per request slot, not per token."""
    phys, attn_bytes_per_token, state_bytes_per_slot = 32, 10, 1000
    num_slots = 4 + 1  # resident sequences + reserved dummy slots
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr._has_cp_helix = True
    mgr._helix_cp_size = cp_size
    mgr.tokens_per_block = phys
    mgr._ledger_tokens_per_block = phys * cp_size
    mgr.kv_cache_config = None
    mgr._num_reserved_dummy_slots = 1
    mgr._max_resident_sequences = lambda: 4
    mgr._num_ssm_snapshots_for_capacity = lambda max_tokens, config: 0
    mgr._mamba_state_bytes_per_slot = lambda: state_bytes_per_slot
    mgr._attention_cache_bytes_per_token = lambda: attn_bytes_per_token
    mgr._get_quota_from_max_tokens_impl = lambda tokens: tokens * attn_bytes_per_token

    # State takes 5 slots; the rest buys 7 full physical pages plus a
    # partial page that is never usable.
    quota = num_slots * state_bytes_per_slot + 7 * phys * attn_bytes_per_token + 100
    tokens = mgr._get_max_tokens_from_quota(quota)

    assert tokens == 7 * phys * cp_size
    assert mgr._get_quota_from_max_tokens(tokens) <= quota
    assert mgr._get_quota_from_max_tokens(tokens + 1) > quota


def test_block_reuse_rejected_for_recurrent_state_under_helix() -> None:
    """Recurrent-state snapshot points are not aligned to ledger blocks, so a
    reused prefix could restore the state of a later token."""
    import torch

    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import CpType, Mapping

    with pytest.raises(ValueError, match="recurrent state layers under helix"):
        MambaHybridCacheManagerV2(
            mamba_d_state=128,
            mamba_d_conv=4,
            mamba_num_heads=2,
            mamba_n_groups=2,
            mamba_head_dim=64,
            mamba_num_layers=1,
            mamba_layer_mask=[True, False],
            mamba_cache_dtype=torch.bfloat16,
            mamba_ssm_cache_dtype=torch.float32,
            kv_cache_config=KvCacheConfig(enable_block_reuse=True),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=576,
            tokens_per_block=32,
            max_seq_len=1024,
            max_batch_size=1,
            mapping=Mapping(world_size=2, rank=0, cp_size=2, cp_config={"cp_type": CpType.HELIX}),
            dtype=DataType.HALF,
            layer_mask=[False, True],
            is_disagg=False,
        )
