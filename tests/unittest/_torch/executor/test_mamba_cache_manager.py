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
    owned by a live request outside the current batch. Padding entries
    all reuse the pre-allocated slot of their dummy request (added via
    add_dummy_requests), which is distinct from every real request's
    slot."""
    mgr = _make_mgr(max_batch_size=4, max_draft_len=2)
    # Four real requests claim slots; pool has max_batch_size+max_draft_len+1 = 7 slots.
    mgr._prepare_mamba_cache_blocks([100, 101, 102, 103])
    # Pre-allocate the padding dummy's slot (what _get_padded_batch does
    # via kv_cache_manager.add_dummy_requests before get_state_indices).
    mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])
    # Current batch has only two reals; 102 and 103 are "parked".
    request_ids = [100, 101, CUDA_GRAPH_DUMMY_REQUEST_ID]
    indices = mgr.get_state_indices(request_ids, [False, False, True])
    real_slots = {mgr.mamba_cache_index[r] for r in [100, 101, 102, 103]}
    assert indices[2] not in real_slots, (
        f"padding slot {indices[2]} overlaps a real request's slot (real slots: {real_slots})"
    )
    # Padding should reuse the dummy's reserved slot, not allocate a new one.
    assert indices[2] == mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID]


@skip_no_cuda
def test_padding_survives_overlap_scheduler_pressure():
    """Regression for the overlap-scheduler + attention-dp + CUDA-graph
    padding combo: prior-iter completions linger in mamba_cache_index
    until _process_previous_batch runs, so get_state_indices must not
    require N unused pool slots to serve N padding entries."""
    mgr = _make_mgr(max_batch_size=4, max_draft_len=0)
    # Fill the pool with "live" real requests (simulates completed
    # requests from prior iter that haven't been freed yet).
    mgr._prepare_mamba_cache_blocks([100, 101, 102, 103])
    # Pre-allocate the padding dummy's slot.
    mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])
    # Current batch: 1 real request + 3 padding entries (attention-dp
    # pushed padded_batch_size to 4 on this rank even though only 1 real
    # gen request is scheduled here).
    request_ids = [100] + [CUDA_GRAPH_DUMMY_REQUEST_ID] * 3
    is_padding = [False] + [True] * 3
    indices = mgr.get_state_indices(request_ids, is_padding)
    # All padding entries share the dummy's slot.
    dummy_slot = mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID]
    assert indices[0] == mgr.mamba_cache_index[100]
    assert indices[1:] == [dummy_slot] * 3


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
def test_update_mamba_states_uses_self_state_indices_when_passed():
    """Regression for the pre-patch behavior where update_mamba_states
    implicitly read self.state_indices: passing it explicitly as the
    caller-supplied tensor must produce the same writes."""
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
        state_indices=mgr.state_indices,
    )

    for rid in [100, 101, 102]:
        slot = mgr.mamba_cache_index[rid]
        assert torch.all(ssm[:, slot] == 5.0)
        assert torch.all(conv[:, slot] == 5.0)


def test_cpp_add_dummy_requests_allocates_sentinel():
    """CppMambaCacheManager.add_dummy_requests must allocate a permanent
    slot for every id — including the raw CUDA-graph sentinel — so
    get_state_indices can return it without racing parked live requests
    for a free slot."""
    stub = SimpleNamespace(mamba_impl=MagicMock())
    CppMambaCacheManager.add_dummy_requests(stub, [100, CUDA_GRAPH_DUMMY_REQUEST_ID, 101])
    stub.mamba_impl.allocate_cache_blocks.assert_called_once_with(
        [100, CUDA_GRAPH_DUMMY_REQUEST_ID, 101]
    )


def test_cpp_add_dummy_requests_noop_on_empty_list():
    stub = SimpleNamespace(mamba_impl=MagicMock())
    CppMambaCacheManager.add_dummy_requests(stub, [])
    stub.mamba_impl.allocate_cache_blocks.assert_not_called()
