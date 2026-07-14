# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MambaCacheManager padding-slot behavior and
CppMambaHybridCacheManager PP-sharding edge cases."""

import os
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    MIN_REPLAY_HISTORY_SIZE,
    CppMambaHybridCacheManager,
    PythonMambaCacheManager,
    _get_mamba_hybrid_pool_size,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings.internal.batch_manager import LinearCacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.mapping import Mapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_mgr(
    max_batch_size=4, max_draft_len=2, enable_attention_dp=False, use_replay_state_update=False
):
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, enable_attention_dp=enable_attention_dp)
    pool = _get_mamba_hybrid_pool_size(max_batch_size, mapping)
    return PythonMambaCacheManager(
        d_state=8,
        d_conv=4,
        num_heads=4,
        n_groups=1,
        head_dim=8,
        num_layers=2,
        max_batch_size=pool,
        spec_state_size=max_batch_size,
        mapping=mapping,
        dtype=torch.float16,
        ssm_cache_dtype=torch.float16,
        speculative_num_draft_tokens=max_draft_len,
        use_replay_state_update=use_replay_state_update,
    )


@skip_no_cuda
@pytest.mark.parametrize("enable_attention_dp", [False, True])
def test_python_mamba_resource_count_excludes_reserved_dummy_slots(enable_attention_dp):
    max_batch_size = 4
    mgr = _make_mgr(
        max_batch_size=max_batch_size,
        max_draft_len=2,
        enable_attention_dp=enable_attention_dp,
    )

    assert mgr.get_max_resource_count() == max_batch_size
    assert len(mgr.mamba_cache_free_blocks) == max_batch_size


@skip_no_cuda
def test_replay_inactive_without_spec_config():
    mgr = _make_mgr(
        max_batch_size=2,
        max_draft_len=None,
        use_replay_state_update=True,
    )

    assert mgr.use_replay_state_update is False
    assert mgr.get_replay_state_update_metadata() is None


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
def test_replay_update_mamba_states_uses_history_window():
    """Replay path accumulates PNAT until layer kernels write a checkpoint."""
    mgr = _make_mgr(max_batch_size=4, max_draft_len=5, use_replay_state_update=True)
    assert mgr.replay_step_width == 6
    assert mgr.replay_history_size == MIN_REPLAY_HISTORY_SIZE
    assert mgr.mamba_cache.prev_num_accepted_tokens.dtype == torch.int32
    assert mgr.mamba_cache.cache_buf_idx.dtype == torch.int32
    assert mgr.mamba_cache.old_x.shape[3] == MIN_REPLAY_HISTORY_SIZE
    assert mgr.mamba_cache.old_B.shape[3] == MIN_REPLAY_HISTORY_SIZE
    assert mgr.mamba_cache.old_dt.shape[4] == MIN_REPLAY_HISTORY_SIZE
    assert mgr.mamba_cache.old_dA_cumsum.shape[4] == MIN_REPLAY_HISTORY_SIZE

    mgr._prepare_mamba_cache_blocks([100, 101])
    slot_appended = mgr.mamba_cache_index[100]
    slot_checkpointed = mgr.mamba_cache_index[101]

    mgr.mamba_cache.prev_num_accepted_tokens[slot_appended] = 7
    mgr.mamba_cache.prev_num_accepted_tokens[slot_checkpointed] = 13
    mgr.mamba_cache.cache_buf_idx[slot_appended] = 0
    mgr.mamba_cache.cache_buf_idx[slot_checkpointed] = 1
    mgr.mamba_cache.conv.zero_()
    mgr.mamba_cache.intermediate_conv_window.zero_()
    mgr.mamba_cache.intermediate_conv_window[:, 0, 2] = 11.0
    mgr.mamba_cache.intermediate_conv_window[:, 1, 2] = 13.0

    state_indices = torch.tensor(
        [slot_appended, slot_checkpointed], dtype=torch.int32, device="cuda"
    )
    attn = SimpleNamespace(num_seqs=2, num_contexts=0)
    mgr.update_mamba_states(
        attn,
        torch.tensor([3, 3], dtype=torch.int32, device="cuda"),
        state_indices=state_indices,
    )

    assert mgr.mamba_cache.prev_num_accepted_tokens[slot_appended].item() == 10
    assert mgr.mamba_cache.prev_num_accepted_tokens[slot_checkpointed].item() == 3
    assert mgr.mamba_cache.cache_buf_idx[slot_appended].item() == 0
    assert mgr.mamba_cache.cache_buf_idx[slot_checkpointed].item() == 0
    assert torch.all(mgr.mamba_cache.conv[:, slot_appended] == 11.0)
    assert torch.all(mgr.mamba_cache.conv[:, slot_checkpointed] == 13.0)


@skip_no_cuda
def test_replay_update_mamba_states_skips_dummy_slots():
    mgr = _make_mgr(max_batch_size=2, max_draft_len=5, use_replay_state_update=True)
    mgr._prepare_mamba_cache_blocks([100])
    mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])

    real_slot = mgr.mamba_cache_index[100]
    dummy_slot = mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID]
    mgr.mamba_cache.prev_num_accepted_tokens[real_slot] = 13
    mgr.mamba_cache.prev_num_accepted_tokens[dummy_slot] = 13
    mgr.mamba_cache.cache_buf_idx[real_slot] = 1
    mgr.mamba_cache.cache_buf_idx[dummy_slot] = 1

    state_indices = torch.tensor(
        mgr.get_state_indices([100, CUDA_GRAPH_DUMMY_REQUEST_ID], [False, True]),
        dtype=torch.int32,
        device="cuda",
    )
    attn = SimpleNamespace(num_seqs=2, num_contexts=0)
    mgr.update_mamba_states(
        attn,
        torch.tensor([3, 3], dtype=torch.int32, device="cuda"),
        state_indices=state_indices,
    )

    assert mgr.mamba_cache.prev_num_accepted_tokens[real_slot].item() == 3
    assert mgr.mamba_cache.prev_num_accepted_tokens[dummy_slot].item() == 13
    assert mgr.mamba_cache.cache_buf_idx[real_slot].item() == 0
    assert mgr.mamba_cache.cache_buf_idx[dummy_slot].item() == 1


@skip_no_cuda
def test_attention_dp_dummy_has_reserved_slot_with_batch_size_one():
    mgr = _make_mgr(max_batch_size=1, max_draft_len=0, enable_attention_dp=True)
    mgr._prepare_mamba_cache_blocks([100])

    mgr.add_dummy_requests([ATTENTION_DP_DUMMY_REQUEST_ID])

    assert mgr.mamba_cache_free_blocks == []
    assert mgr.mamba_cache_index[100] != mgr._attention_dp_dummy_slot
    assert mgr.mamba_cache_index[ATTENTION_DP_DUMMY_REQUEST_ID] == mgr._attention_dp_dummy_slot

    mgr.free_resources(SimpleNamespace(py_request_id=ATTENTION_DP_DUMMY_REQUEST_ID))
    assert mgr._attention_dp_dummy_slot not in mgr.mamba_cache_free_blocks


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


# ---------------------------------------------------------------------------
# CppMambaHybridCacheManager: recurrent-state snapshot pool sizing
#
# Sized in KVCacheManager._calculate_max_num_blocks_for_linear_attention.
# Mirrors the MixedMambaCacheManager fix where each kind of padding sentinel
# (CUDA-graph dummy, plus one per draft length under spec decoding) must not
# evict live recurrent state. Wanli's fix made all sentinels share one slot
# in the Python manager (#13489); the C++ hybrid path instead reserves a
# dedicated slot per sentinel kind in the underlying pool — same invariant,
# different mechanism. These tests guard the pool sizing.
# ---------------------------------------------------------------------------


def _build_hybrid_with_mamba_layer(
    spec_config=None,
    max_batch_size=4,
    enable_block_reuse=False,
    mamba_state_cache_interval=256,
    is_estimating_kv_cache=False,
):
    """Construct a real CppMambaHybridCacheManager with one mamba layer +
    one full-attention layer so the parent KVCacheManager goes through the
    linear-attention pool sizing path."""
    # Layer 0: mamba; Layer 1: full attention. Single rank, no MPI.
    mamba_mask = [True, False]
    attn_mask = [False, True]
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    # Cap max_tokens to keep the real C++ pool allocation tiny.
    kv_cache_config = KvCacheConfig(
        max_tokens=512,
        enable_block_reuse=enable_block_reuse,
        mamba_state_cache_interval=mamba_state_cache_interval,
    )
    return CppMambaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=1,
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float16,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=max_batch_size,
        mapping=mapping,
        spec_config=spec_config,
        layer_mask=attn_mask,
        is_estimating_kv_cache=is_estimating_kv_cache,
    )


@skip_no_cuda
def test_cpp_hybrid_recurrent_pool_reserves_cuda_graph_padding_slot():
    """Without spec decoding, the recurrent-state snapshot pool must
    have at least max_batch_size + 1 slots — one extra for the
    CUDA-graph padding sentinel (CUDA_GRAPH_DUMMY_REQUEST_ID). Without
    it, the padding sentinel evicts live recurrent state under load."""
    max_batch_size = 4
    mgr = _build_hybrid_with_mamba_layer(spec_config=None, max_batch_size=max_batch_size)
    recurrent_primary, _ = mgr.blocks_per_window[LinearCacheType.RECURRENT_STATES.value]
    assert recurrent_primary >= max_batch_size + 1, (
        f"recurrent-state pool has {recurrent_primary} slots, "
        f"need >= max_batch_size + 1 = {max_batch_size + 1} to host the "
        f"CUDA-graph padding sentinel without evicting live state"
    )


@skip_no_cuda
def test_cpp_hybrid_recurrent_pool_reserves_draft_len_sentinel_slots():
    """With spec decoding, CUDAGraphRunner._get_padded_batch issues a
    distinct dummy request id for each runtime_draft_len in
    [0, max_draft_len], so the recurrent-state snapshot pool must reserve
    one slot per draft length on top of the CUDA-graph padding slot."""
    max_batch_size, max_draft_len = 4, 2
    spec_config = MTPDecodingConfig(max_draft_len=max_draft_len)
    mgr = _build_hybrid_with_mamba_layer(spec_config=spec_config, max_batch_size=max_batch_size)
    recurrent_primary, _ = mgr.blocks_per_window[LinearCacheType.RECURRENT_STATES.value]
    expected_min = max_batch_size + 1 + max_draft_len
    assert recurrent_primary >= expected_min, (
        f"recurrent-state pool has {recurrent_primary} slots, "
        f"need >= max_batch_size + 1 + max_draft_len = {expected_min} so "
        f"per-draft-len sentinels don't collide with live state"
    )


def _build_hybrid_with_mamba_layer_pp(
    spec_config=None, max_batch_size=4, enable_block_reuse=False, pp_size=2
):
    """Same as ``_build_hybrid_with_mamba_layer`` but with ``pp_size`` >= 1.

    Uses ``world_size = pp_size`` and ``rank = 0`` so the real C++ KVCacheManager
    still goes through its single-process path while the Python pool-sizing
    code sees ``mapping.pp_size > 1``. Constructs ``pp_size * 2`` total layers
    (alternating mamba/attn) so that each PP slice has both a mamba and an
    attention layer — otherwise some ranks would hit a slope=0 edge case in
    the affine memory model when block reuse is disabled.
    """
    pairs = pp_size  # one (mamba, attn) pair per PP stage so every rank has both kinds
    mamba_mask = [True, False] * pairs
    attn_mask = [False, True] * pairs
    mamba_num_layers = sum(mamba_mask)
    num_layers = sum(attn_mask)
    mapping = Mapping(world_size=pp_size, rank=0, tp_size=1, pp_size=pp_size)
    kv_cache_config = KvCacheConfig(max_tokens=512, enable_block_reuse=enable_block_reuse)
    return CppMambaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=mamba_num_layers,
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float16,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_layers,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=max_batch_size,
        mapping=mapping,
        spec_config=spec_config,
        layer_mask=attn_mask,
    )


# Skip when running under pytest --run-ray (sets TLLM_DISABLE_MPI=1). In that
# mode ``Mapping`` resolves to ``DeviceMeshTopology``, whose ``pp_rank``
# requires torch.distributed initialisation that isn't available in this
# single-process unit test. The pp-sharding behaviour exercised here is
# orthogonal to the Ray orchestrator.
_skip_under_ray = pytest.mark.skipif(
    os.environ.get("TLLM_DISABLE_MPI") == "1",
    reason="pp_size>1 helper builds Mapping with world_size>1 which needs "
    "torch.distributed under TLLM_DISABLE_MPI=1 (Ray) sessions",
)


@_skip_under_ray
@skip_no_cuda
@pytest.mark.parametrize("pp_size", [2, 4])
def test_cpp_hybrid_recurrent_pool_scales_with_pp_size(pp_size):
    """With pipeline parallelism, multiple microbatches are in-flight on the
    same rank concurrently, each holding up to ``max_batch_size`` sequences'
    Mamba state. The recurrent-state pool must therefore size for
    ``max_batch_size * pp_size`` live slots (plus the CUDA-graph padding
    sentinel). Without this scaling, the first inference batch under PP>1
    trips ``No free block found`` once requests beyond the first microbatch
    enter the pool (cf. TestNemotronV3Super::test_nvfp4_parallelism[TP4_PP2]).
    """
    max_batch_size = 4
    mgr = _build_hybrid_with_mamba_layer_pp(
        spec_config=None, max_batch_size=max_batch_size, pp_size=pp_size
    )
    recurrent_primary, _ = mgr.blocks_per_window[LinearCacheType.RECURRENT_STATES.value]
    expected_min = max_batch_size * pp_size + 1
    assert recurrent_primary >= expected_min, (
        f"recurrent-state pool has {recurrent_primary} slots with pp_size={pp_size}, "
        f"need >= max_batch_size * pp_size + 1 = {expected_min} so concurrent "
        f"in-flight microbatches don't exhaust live-state slots"
    )


@skip_no_cuda
def test_cpp_hybrid_recurrent_pool_floor_with_block_reuse():
    """With block reuse enabled, the block-reuse branch must not drop the
    live-state + CUDA-graph-padding floor.

    With max_batch_size=4, mamba_state_cache_interval=256, max_tokens=512:
      naive: max_snapshots = 512 // 256 = 2  (drops live-state floor!)
      fixed: max_snapshots = max(2, 4 + 1) = 5
    """
    max_batch_size = 4
    mgr = _build_hybrid_with_mamba_layer(
        spec_config=None,
        max_batch_size=max_batch_size,
        enable_block_reuse=True,
        mamba_state_cache_interval=256,
    )
    recurrent_primary, _ = mgr.blocks_per_window[LinearCacheType.RECURRENT_STATES.value]
    assert recurrent_primary >= max_batch_size + 1, (
        f"recurrent-state pool has {recurrent_primary} slots with block reuse enabled, "
        f"need >= max_batch_size + 1 = {max_batch_size + 1} to prevent the padding "
        f"sentinel from evicting live recurrent state"
    )


@skip_no_cuda
def test_cpp_hybrid_dry_run_recurrent_pool_additive_with_block_reuse():
    """Dry-run path (is_estimating_kv_cache=True) under block reuse must
    keep the live-state floor *plus* room for snapshots, not collapse to
    max(snapshots, live). With max_batch_size=4, interval=256, max_tokens=512:
      old:  max_snapshots = max(512//256, 4)         = 4   (no headroom for snapshots)
      new:  max_snapshots = 4 + 512//256             = 6   (live + snapshots)
    """
    max_batch_size = 4
    mgr = _build_hybrid_with_mamba_layer(
        spec_config=None,
        max_batch_size=max_batch_size,
        enable_block_reuse=True,
        mamba_state_cache_interval=256,
        is_estimating_kv_cache=True,
    )
    recurrent_primary, _ = mgr.blocks_per_window[LinearCacheType.RECURRENT_STATES.value]
    # 4 live state slots + 2 reuse snapshots = 6.
    expected_min = max_batch_size + (512 // 256)
    assert recurrent_primary >= expected_min, (
        f"dry-run recurrent-state pool has {recurrent_primary} slots, "
        f"need >= live_state + reuse_snapshots = {expected_min}; the old "
        f"max(reuse, live) formula dropped reuse headroom"
    )


# ---------------------------------------------------------------------------
# CppMambaHybridCacheManager: rank with zero local mamba layers
#
# Regression test for the early-exit path added when a rank ends up with no
# mamba layers (e.g. under PP sharding when all mamba layers fall on other
# ranks). On that path, the constructor must:
#   - call the real parent KVCacheManager with the union layer_mask and
#     num_layers=num_layers (not mamba_num_layers + num_layers),
#   - skip allocating any mamba-only state, and
#   - leave self.requests = [] so the guards on prepare_resources /
#     update_mamba_states / _setup_state_indices can no-op without touching
#     uninitialized state.
#
# We exercise the same Python branch with world_size=1 (so the real C++
# KVCacheManager init doesn't need MPI) and a layer mask that contains zero
# mamba layers.
# ---------------------------------------------------------------------------


def _build_zero_mamba_hybrid():
    """Construct a real CppMambaHybridCacheManager whose this-rank slice has
    no mamba layers. world_size=1 / pp_size=1 keeps the real parent
    KVCacheManager off the MPI path."""
    # [other, other, full_attn, full_attn]
    mamba_mask = [False, False, False, False]
    attn_mask = [False, False, True, True]
    mamba_num_layers = sum(mamba_mask)  # 0
    num_layers = sum(attn_mask)  # 4

    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    # Cap KV pool size so the real C++ allocator only takes a tiny slice of
    # GPU memory; we don't actually use the cache.
    kv_cache_config = KvCacheConfig(max_tokens=128)

    mgr = CppMambaHybridCacheManager(
        # mamba cache parameters — values are unused on the early-exit path
        # but must be type-valid.
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=mamba_num_layers,
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float16,
        # kv cache parameters
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_layers,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=2,
        mapping=mapping,
        spec_config=None,
        layer_mask=attn_mask,
    )
    return mgr


@skip_no_cuda
def test_cpp_hybrid_zero_local_mamba_layers():
    """End-to-end: real parent KVCacheManager + real early-exit. Verifies
    early-exit invariants on the manager state AND that the three guarded
    methods no-op without raising on uninitialized mamba-only state."""
    mgr = _build_zero_mamba_hybrid()

    # Early-exit indicators.
    assert mgr.local_num_mamba_layers == 0
    assert mgr.mamba_pp_layers == []
    assert mgr.requests == []
    assert mgr.pp_layers == [2, 3]

    # Parent KVCacheManager was really initialized. self.impl is the C++
    # KVCacheManagerCpp object; blocks_per_window is set up by it.
    assert hasattr(mgr, "impl")
    assert hasattr(mgr, "blocks_per_window")
    # Parent saw num_layers = num_layers (4), not mamba_num_layers + num_layers.
    # On the early-exit branch, num_layers is forwarded as-is.
    assert mgr.num_layers == 4
    assert mgr.num_local_layers == 2

    # No mamba-only state was allocated.
    for attr in (
        "ssm_state_shape",
        "conv_state_shape",
        "mamba_layer_offsets",
        "cuda_state_indices",
        "host_block_offsets",
        "recurrent_states_pool_index",
    ):
        assert not hasattr(mgr, attr), f"{attr} must not be set on the zero-mamba early-exit path"
    # Parent must not have been told to treat this as linear attention.
    assert mgr.is_linear_attention is False

    # Guards on the three mamba-only methods must turn them into no-ops
    # instead of crashing on the missing state above.
    empty_batch = ScheduledRequests()
    mgr.prepare_resources(empty_batch)  # super() runs, then guard returns
    mgr.update_mamba_states(attn_metadata=None, num_accepted_tokens=None, state_indices=None)
    mgr._setup_state_indices()
