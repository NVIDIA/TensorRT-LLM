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

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MambaStateConfig

pytestmark = pytest.mark.cpu_only

TOKENS_PER_BLOCK = 32


def _helix_manager(cp_size: int, **mamba_state: object) -> MambaHybridCacheManagerV2:
    """Bare V2 hybrid manager carrying only what snapshot placement reads."""
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr._has_cp_helix = True
    mgr.tokens_per_block = TOKENS_PER_BLOCK
    mgr._ledger_tokens_per_block = TOKENS_PER_BLOCK * cp_size
    mgr.enable_block_reuse = True
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True, mamba_state_config=MambaStateConfig(**mamba_state)
    )
    return mgr


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


@pytest.mark.parametrize(
    ("cp_size", "expected"),
    [
        (2, [64, 192, 256, 384, 448, 576, 640, 768, 832, 960]),
        (4, [128, 256, 384, 512, 640, 768, 896]),
    ],
)
def test_snapshot_points_land_on_ledger_blocks_under_helix(cp_size: int, expected: list) -> None:
    """A reused prefix ends on a ledger block, so every recurrent-state
    snapshot moves down onto one; points below the first block are dropped."""
    mgr = _helix_manager(
        cp_size, periodic_snapshot_interval=96, additional_snapshot_offsets_from_end=[0, 10]
    )
    request = SimpleNamespace(py_request_id=1, prompt_len=1000, expect_snapshot_points=[])

    mgr.prepare_expect_snapshot_points([request])

    assert request.expect_snapshot_points == expected


@pytest.mark.parametrize("cp_size", [2, 4])
def test_waiter_snapshot_point_lands_on_ledger_block_under_helix(cp_size: int) -> None:
    """A prefix a waiting request asks for moves down onto a ledger block; one
    shorter than a ledger block cannot be reused, so the request does not wait."""
    mgr = _helix_manager(cp_size)
    mgr.kv_cache_map = {}
    mgr.num_extra_kv_tokens = 0
    owner = SimpleNamespace(
        py_request_id=1, prompt_len=1000, context_current_position=0, expect_snapshot_points=[]
    )

    assert mgr.request_prefix_snapshot(owner, 300)
    assert owner.expect_snapshot_points == [256]

    assert not mgr.request_prefix_snapshot(owner, 50)
    assert owner.expect_snapshot_points == [256]
