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
    # +1 headroom matches MixedMambaHybridCacheManager.pool_size.
    pool = max_batch_size + 1
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
    """Padding must not resolve to a slot owned by a parked real
    request outside the current batch."""
    mgr = _make_mgr(max_batch_size=4, max_draft_len=2)
    mgr._prepare_mamba_cache_blocks([100, 101, 102, 103])
    mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])
    # 102 and 103 are parked outside the current batch.
    request_ids = [100, 101, CUDA_GRAPH_DUMMY_REQUEST_ID]
    indices = mgr.get_state_indices(request_ids, [False, False, True])
    real_slots = {mgr.mamba_cache_index[r] for r in [100, 101, 102, 103]}
    assert indices[2] not in real_slots
    assert indices[2] == mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID]


@skip_no_cuda
def test_padding_survives_overlap_scheduler_pressure():
    """Under the overlap scheduler, prior-iter completions linger in
    mamba_cache_index, so N padding entries must not need N free
    slots."""
    mgr = _make_mgr(max_batch_size=4, max_draft_len=0)
    mgr._prepare_mamba_cache_blocks([100, 101, 102, 103])
    mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])
    # 1 real + 3 padding (attention-dp padded_batch_size=4 on this rank).
    request_ids = [100] + [CUDA_GRAPH_DUMMY_REQUEST_ID] * 3
    is_padding = [False] + [True] * 3
    indices = mgr.get_state_indices(request_ids, is_padding)
    dummy_slot = mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID]
    assert indices[0] == mgr.mamba_cache_index[100]
    assert indices[1:] == [dummy_slot] * 3


@skip_no_cuda
def test_all_draft_len_sentinels_share_one_slot():
    """All per-draft-len sentinels must collapse to a single slot, so
    the pool needs only +1 headroom regardless of max_draft_len."""
    max_batch_size, max_draft_len = 4, 3
    mgr = _make_mgr(max_batch_size=max_batch_size, max_draft_len=max_draft_len)
    mgr._prepare_mamba_cache_blocks([100, 101, 102, 103])

    sentinels = [CUDA_GRAPH_DUMMY_REQUEST_ID - k for k in range(max_draft_len + 1)]
    mgr.add_dummy_requests(sentinels)

    shared = mgr.mamba_cache_index[sentinels[0]]
    real_slots = {mgr.mamba_cache_index[r] for r in [100, 101, 102, 103]}
    assert shared not in real_slots
    for s in sentinels:
        assert mgr.mamba_cache_index[s] == shared
    assert mgr.mamba_cache_free_blocks == []


@skip_no_cuda
def test_padding_slot_is_permanent():
    """free_resources drops a sentinel's index entry but the shared
    slot stays reserved for the next batch."""
    mgr = _make_mgr(max_batch_size=4, max_draft_len=2)
    sentinels = [CUDA_GRAPH_DUMMY_REQUEST_ID - k for k in range(3)]
    mgr.add_dummy_requests(sentinels)
    shared = mgr.mamba_cache_index[sentinels[0]]

    def _fake(rid):
        return SimpleNamespace(py_request_id=rid)

    for s in sentinels:
        mgr.free_resources(_fake(s))
        assert s not in mgr.mamba_cache_index
        assert shared not in mgr.mamba_cache_free_blocks

    assert mgr._padding_slot == shared


@skip_no_cuda
def test_update_mamba_states_mtp_path():
    """MTP forward path: update_mamba_states must scatter using the
    caller-supplied Mamba2Metadata-style state_indices tensor (partition
    order, padded to the captured batch size)."""
    mgr = _make_mgr()
    mgr._prepare_mamba_cache_blocks([100, 101, 102])

    ssm, conv = mgr.mamba_cache.temporal, mgr.mamba_cache.conv
    ssm.zero_()
    conv.zero_()
    mgr.mamba_cache.intermediate_ssm.fill_(7.0)
    mgr.mamba_cache.intermediate_conv_window.fill_(7.0)

    # Simulate mamba_metadata.state_indices — full captured batch of 4
    # (3 reals + 1 padding dummy on slot 0).
    state_indices = torch.tensor(
        [
            mgr.mamba_cache_index[100],
            mgr.mamba_cache_index[101],
            mgr.mamba_cache_index[102],
            0,
        ],
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
def test_update_mamba_states_autodeploy_path():
    mgr = _make_mgr()
    mgr._prepare_mamba_cache_blocks([200, 201, 202])

    ssm, conv = mgr.mamba_cache.temporal, mgr.mamba_cache.conv
    ssm.zero_()
    conv.zero_()
    mgr.mamba_cache.intermediate_ssm.fill_(3.0)
    mgr.mamba_cache.intermediate_conv_window.fill_(3.0)

    # Mimic csi.get_arg("slot_idx", truncate=True): int64, truncated
    # to current_length. One prefill (200), two generation (201, 202).
    state_indices = torch.tensor(
        [
            mgr.mamba_cache_index[200],
            mgr.mamba_cache_index[201],
            mgr.mamba_cache_index[202],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    # num_contexts=1 means only indices [1:] are scattered (the gen slice).
    attn = SimpleNamespace(num_seqs=3, num_contexts=1)
    mgr.update_mamba_states(
        attn,
        torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda"),
        state_indices=state_indices,
    )

    # Only 201 and 202 (the generation slice) should have been written.
    assert torch.all(ssm[:, mgr.mamba_cache_index[200]] == 0.0)
    for rid in [201, 202]:
        slot = mgr.mamba_cache_index[rid]
        assert torch.all(ssm[:, slot] == 3.0)
        assert torch.all(conv[:, slot] == 3.0)


@skip_no_cuda
def test_non_mtp_pytorch_prepare_and_get_state_indices_flow():
    mgr = _make_mgr(max_batch_size=4, max_draft_len=0)
    # Simulate a non-MTP step: mix of context + generation requests,
    # plus a CUDA-graph padding dummy.
    context_ids = [400]
    gen_ids = [401, 402]
    mgr._prepare_mamba_cache_blocks(context_ids + gen_ids)
    mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])

    # All reals have distinct slots.
    reals = set(mgr.mamba_cache_index[r] for r in context_ids + gen_ids)
    assert len(reals) == 3
    # Dummy's slot is distinct from every real.
    assert mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID] not in reals

    # What Mamba2Metadata.prepare would do: resolve indices for the
    # current padded batch (3 reals + 1 padding).
    request_ids = context_ids + gen_ids + [CUDA_GRAPH_DUMMY_REQUEST_ID]
    is_padding = [False, False, False, True]
    indices = mgr.get_state_indices(request_ids, is_padding)
    assert indices == [
        mgr.mamba_cache_index[400],
        mgr.mamba_cache_index[401],
        mgr.mamba_cache_index[402],
        mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID],
    ]


def test_cpp_add_dummy_requests_noop_on_empty_list():
    stub = SimpleNamespace(mamba_impl=MagicMock())
    CppMambaCacheManager.add_dummy_requests(stub, [])
    stub.mamba_impl.allocate_cache_blocks.assert_not_called()


@skip_no_cuda
def test_cpp_get_state_indices_resolves_sentinel_to_reserved_slot():
    """End-to-end C++ path: add_dummy_requests + getStateIndices must
    resolve the CUDA-graph sentinel to its reserved slot, distinct from
    every live request's slot — guards the C++ mCacheIndex lookup, not
    just the Python forwarder."""
    mgr = CppMambaCacheManager(
        d_state=8,
        d_conv=4,
        num_heads=4,
        n_groups=1,
        head_dim=8,
        num_layers=2,
        max_num_sequences=8,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        dtype=torch.float16,
        ssm_cache_dtype=torch.float16,
    )
    mgr.add_dummy_requests([100, 101, CUDA_GRAPH_DUMMY_REQUEST_ID])

    request_ids = [100, 101, CUDA_GRAPH_DUMMY_REQUEST_ID]
    is_padding = [False, False, True]
    indices = mgr.get_state_indices(request_ids, is_padding)

    sentinel_slot = indices[2]
    real_slots = {indices[0], indices[1]}
    assert sentinel_slot not in real_slots, (
        f"sentinel slot {sentinel_slot} aliases a real request's slot {real_slots}"
    )
    # Resolve again — reserved slot must be stable across calls.
    assert mgr.get_state_indices(request_ids, is_padding) == indices
