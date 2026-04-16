# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MambaCacheManager padding-slot behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    CppMambaCacheManager,
    PythonMambaCacheManager,
)
from tensorrt_llm.mapping import Mapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_mgr(max_batch_size=4, max_draft_len=2):
    # Pool size mirrors MambaHybridCacheManager's +max_draft_len+1 headroom.
    pool = max_batch_size + max_draft_len + 1
    return PythonMambaCacheManager(
        d_state=8,
        d_conv=4,
        num_heads=4,
        n_groups=1,
        head_dim=8,
        num_layers=2,
        max_batch_size=pool,
        spec_state_size=max_batch_size,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        dtype=torch.float16,
        ssm_cache_dtype=torch.float16,
        speculative_num_draft_tokens=max_draft_len,
    )


@skip_no_cuda
def test_padding_slot_not_held_by_parked_real():
    """get_state_indices must not hand the padding position a slot
    owned by a live request outside the current batch (REPORT Bug #2).
    The fix uses mamba_cache_index.values() as the exclusion set."""
    mgr = _make_mgr(max_batch_size=4, max_draft_len=2)
    # Four real requests claim slots; pool has max_batch_size+max_draft_len+1 = 7 slots.
    mgr._prepare_mamba_cache_blocks([100, 101, 102, 103])
    # Current batch has only two of them; 102 and 103 are "parked".
    request_ids = [100, 101, CUDA_GRAPH_DUMMY_REQUEST_ID]
    indices = mgr.get_state_indices(request_ids, [False, False, True])
    # Padding slot must NOT equal mamba_cache_index[102] or [103].
    parked_slots = {mgr.mamba_cache_index[102], mgr.mamba_cache_index[103]}
    assert indices[2] not in parked_slots, (
        f"padding slot {indices[2]} overlaps parked real's slot (parked slots: {parked_slots})"
    )


@skip_no_cuda
def test_update_mamba_states_uses_passed_state_indices():
    """update_mamba_states must scatter using the caller-supplied
    state_indices tensor (e.g. mamba_metadata.state_indices)."""
    mgr = _make_mgr()
    mgr._prepare_mamba_cache_blocks([100, 101, 102])

    ssm, conv = mgr.mamba_cache.temporal, mgr.mamba_cache.conv
    ssm.zero_()
    conv.zero_()
    mgr.mamba_cache.intermediate_ssm.fill_(7.0)
    mgr.mamba_cache.intermediate_conv_window.fill_(7.0)

    # Caller passes slots [slot_R1, slot_R2, slot_R3, 0] for a padded
    # batch. Slot 0 belongs to the padding dummy.
    state_indices = torch.tensor(
        [mgr.mamba_cache_index[100], mgr.mamba_cache_index[101], mgr.mamba_cache_index[102], 0],
        dtype=torch.int32,
        device="cuda",
    )
    attn = SimpleNamespace(num_seqs=4, num_contexts=0)
    mgr.update_mamba_states(
        attn,
        torch.tensor([1, 1, 1, 1], dtype=torch.int32, device="cuda"),
        state_indices=state_indices,
    )

    for rid in [100, 101, 102]:
        slot = mgr.mamba_cache_index[rid]
        assert torch.all(ssm[:, slot] == 7.0)
        assert torch.all(conv[:, slot] == 7.0)


@skip_no_cuda
def test_update_mamba_states_fallback_uses_own_state_indices():
    """When the caller doesn't pass state_indices (e.g. AutoDeploy
    SimpleNamespace), update_mamba_states falls back to
    self.state_indices — pre-regression behavior."""
    mgr = _make_mgr()
    mgr._prepare_mamba_cache_blocks([100, 101, 102])

    ssm, conv = mgr.mamba_cache.temporal, mgr.mamba_cache.conv
    ssm.zero_()
    conv.zero_()
    mgr.mamba_cache.intermediate_ssm.fill_(5.0)
    mgr.mamba_cache.intermediate_conv_window.fill_(5.0)

    attn = SimpleNamespace(num_seqs=3, num_contexts=0)
    mgr.update_mamba_states(
        attn,
        torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda"),
    )

    for rid in [100, 101, 102]:
        slot = mgr.mamba_cache_index[rid]
        assert torch.all(ssm[:, slot] == 5.0)
        assert torch.all(conv[:, slot] == 5.0)


def test_cpp_add_dummy_requests_filters_sentinel():
    """CppMambaCacheManager.add_dummy_requests must drop the exact
    sentinel; per-draft-len dummies (sentinel-K) pass through and
    get a permanent slot via the C++ allocator."""
    stub = SimpleNamespace(mamba_impl=MagicMock(), speculative_num_draft_tokens=None)
    CppMambaCacheManager.add_dummy_requests(stub, [100, CUDA_GRAPH_DUMMY_REQUEST_ID, 101])
    stub.mamba_impl.allocate_cache_blocks.assert_called_once_with([100, 101])


def test_cpp_add_dummy_requests_noop_when_only_sentinel():
    stub = SimpleNamespace(mamba_impl=MagicMock(), speculative_num_draft_tokens=None)
    CppMambaCacheManager.add_dummy_requests(stub, [CUDA_GRAPH_DUMMY_REQUEST_ID])
    stub.mamba_impl.allocate_cache_blocks.assert_not_called()
