# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA cache manager's gen-init admission gate (``K31GenInitDecodeHeadroom``), which keeps decode headroom, and the
manager's per-pool-group growth and free slots it reads. CPU only."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_cache_manager import (
    K31GenInitDecodeHeadroom,
    K31HybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only


# --- The gen-init admission gate --------------------------------------------------------------------


def _request(request_id, state, prompt_len, num_tokens=None):
    return SimpleNamespace(
        py_request_id=request_id,
        state_value=state.value,
        prompt_len=prompt_len,
        py_max_new_tokens=None,
        py_draft_tokens=None,
        get_num_tokens=lambda beam: prompt_len if num_tokens is None else num_tokens,
    )


def _headroom_pass(available, allocation_extra_tokens=0):
    """A pass over one resident generation request (10 blocks held, 16 blocks of growth) and one arriving
    disaggregated gen-init request (a 100-block prompt, 16 blocks of growth), on a pool group whose growth takes
    2 slots per block."""
    manager = SimpleNamespace(
        tokens_per_block=64,
        _kv_alloc_extra_tokens=allocation_extra_tokens,
        token_growth_slots_per_block=lambda: {0: 2},
        get_available_primary_slots=lambda: [available],
        is_request_active=lambda request_id: True,
        kv_cache_map={0: SimpleNamespace(capacity=640, is_active=True)},
    )
    resident = _request(0, LlmRequestState.GENERATION_IN_PROGRESS, 600, num_tokens=640)
    arriving = _request(1, LlmRequestState.DISAGG_GENERATION_INIT, 6400)
    return K31GenInitDecodeHeadroom(1024).begin_pass(manager, [resident, arriving]), arriving


def test_admits_while_the_growth_of_every_resident_request_fits():
    # (100 + 16 + 16) blocks x 2 slots per block.
    gate, arriving = _headroom_pass(available=264)
    assert gate.reserved_blocks == 16
    token = gate.admits(arriving)
    assert token == (100, 16)
    gate.commit(token)
    assert gate.reserved_blocks == 32
    # The committed prompt took 200 slots; the next arrival must fit in the rest.
    assert gate.admits(arriving) is None


def test_defers_when_the_pool_cannot_keep_the_headroom():
    gate, arriving = _headroom_pass(available=263)
    assert gate.admits(arriving) is None


def test_the_manager_counts_full_attention_growth_per_pool_group():
    """Full-attention layer groups grow with tokens; sliding-window groups and the KDA state do not."""
    manager = object.__new__(K31HybridCacheManager)
    manager.max_seq_len = 4096
    manager._stats_life_cycle_metadata = lambda: {
        0: (0, 4096, "attention"),
        1: (0, 4096, "attention"),
        2: (1, 4096, "attention"),
        3: (1, 128, "attention"),
        4: (2, None, "ssm"),
    }
    manager._get_storage_statistics = lambda level: [
        SimpleNamespace(available=n) for n in (7, 8, 9)
    ]
    assert manager.token_growth_slots_per_block() == {0: 2, 1: 1}
    assert manager.get_available_primary_slots() == [7, 8, 9]
