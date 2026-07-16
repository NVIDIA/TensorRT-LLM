# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MambaCacheManager padding-slot behavior and
CppMambaHybridCacheManager PP-sharding edge cases."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID, LlmRequest
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    MIN_REPLAY_HISTORY_SIZE,
    CppMambaHybridCacheManager,
    MixedMambaHybridCacheManager,
    PythonMambaCacheManager,
    V2MambaHybridCacheManager,
    _get_mamba_hybrid_pool_size,
    calc_context_stop_positions,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings.internal.batch_manager import LinearCacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import GpuCacheTierConfig, LayerId
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager as RuntimeKVCacheManager

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_hybrid_cache_manager_factory_defaults_to_v2(monkeypatch):
    monkeypatch.delenv("TRTLLM_USE_CPP_MAMBA", raising=False)
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    config = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    model_config = SimpleNamespace(
        pretrained_config=config,
        sparse_attention_config=None,
        get_num_mamba_layers=lambda: 1,
    )

    assert (
        get_kv_cache_manager_cls(model_config, KvCacheConfig(enable_block_reuse=False))
        is V2MambaHybridCacheManager
    )
    assert (
        get_kv_cache_manager_cls(model_config, KvCacheConfig(enable_block_reuse=True))
        is V2MambaHybridCacheManager
    )


def test_hybrid_cache_manager_factory_honors_cpp_preference_with_block_reuse(monkeypatch):
    monkeypatch.delenv("TRTLLM_USE_CPP_MAMBA", raising=False)
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.setenv("TLLM_MAMBA_MANAGER_PREFERENCE", "CPP")
    config = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    model_config = SimpleNamespace(
        pretrained_config=config,
        sparse_attention_config=None,
        get_num_mamba_layers=lambda: 1,
    )

    assert (
        get_kv_cache_manager_cls(model_config, KvCacheConfig(enable_block_reuse=True))
        is CppMambaHybridCacheManager
    )


def test_hybrid_cache_manager_factory_keeps_legacy_disagg_route(monkeypatch):
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    config = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    model_config = SimpleNamespace(
        pretrained_config=config,
        sparse_attention_config=None,
        get_num_mamba_layers=lambda: 1,
    )

    assert (
        get_kv_cache_manager_cls(
            model_config,
            KvCacheConfig(enable_block_reuse=False),
            is_disagg=True,
        )
        is CppMambaHybridCacheManager
    )
    assert (
        get_kv_cache_manager_cls(
            model_config,
            KvCacheConfig(enable_block_reuse=False),
            is_disagg=True,
            cache_transceiver_config=SimpleNamespace(transceiver_runtime="PYTHON"),
        )
        is MixedMambaHybridCacheManager
    )


def test_v2_hybrid_incompatibility_falls_back_to_cpp_manager():
    config = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    model_config = SimpleNamespace(
        pretrained_config=config,
        sparse_attention_config=None,
    )
    creator = object.__new__(KvCacheCreator)
    creator._kv_connector_manager = None
    creator._max_beam_width = 2

    assert (
        creator._fallback_if_unsupported_kv_cache_manager_v2(
            V2MambaHybridCacheManager, model_config, KvCacheConfig()
        )
        is CppMambaHybridCacheManager
    )


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


def test_calc_context_stop_positions_returns_snapshot_points():
    assert calc_context_stop_positions(128, 32, 64) == [64, 128]
    assert calc_context_stop_positions(70, 32, 256) == []
    assert calc_context_stop_positions(70, 32, 0) == []
    assert calc_context_stop_positions(70, 32, None) == []


def test_v2_hybrid_prepare_expect_snapshot_points():
    mgr = object.__new__(V2MambaHybridCacheManager)
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        mamba_state_cache_interval=64,
    )
    mgr.tokens_per_block = 32
    mgr._mamba_state_cache_interval = 64
    request = SimpleNamespace(prompt_len=150)

    mgr.prepare_expect_snapshot_points([request])

    assert request.expect_snapshot_points == [64, 128]


def test_request_block_reuse_commit_limit_uses_snapshot_points():
    request = SimpleNamespace(prompt_len=150, expect_snapshot_points=[137])

    assert LlmRequest.block_reuse_commit_limit(request) == 137


def test_request_block_reuse_commit_limit_defaults_to_prompt_len():
    request = SimpleNamespace(prompt_len=150, expect_snapshot_points=[])

    assert LlmRequest.block_reuse_commit_limit(request) == 150


def test_v2_block_reuse_commit_saves_ssm_snapshot_at_snapshot_point():
    mgr = object.__new__(KVCacheManagerV2)
    mgr.enable_block_reuse = True
    mgr.is_draft = False
    mgr._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]
    mgr._mark_context_position_as_history = MagicMock()

    token_ids = list(range(150))
    request = SimpleNamespace(
        prompt_len=150,
        context_current_position=137,
        context_remaining_length=13,
        expect_snapshot_points=[137],
        is_dummy_request=False,
        is_dummy=False,
        py_request_id=0,
        get_tokens=lambda beam_idx: token_ids,
        block_reuse_commit_limit=lambda: 137,
    )
    kv_cache = SimpleNamespace(
        num_committed_tokens=0,
        commit=MagicMock(),
        stop_committing=MagicMock(),
    )

    mgr.try_commit_blocks(request, kv_cache)

    kv_cache.commit.assert_called_once_with(token_ids[:137])
    kv_cache.stop_committing.assert_not_called()
    mgr._mark_context_position_as_history.assert_called_once_with(request, kv_cache)


def test_v2_hybrid_add_dummy_requests_forwards_encoder_output_lens(mocker):
    mgr = object.__new__(V2MambaHybridCacheManager)
    base_add_dummy_requests = mocker.patch.object(
        KVCacheManagerV2, "add_dummy_requests", return_value=[]
    )

    mgr.add_dummy_requests([123], encoder_output_lens=[17])

    assert base_add_dummy_requests.call_args.kwargs["encoder_output_lens"] == [17]


def test_v2_hybrid_prepare_expect_snapshot_points_clears_when_reuse_disabled():
    mgr = object.__new__(V2MambaHybridCacheManager)
    mgr.kv_cache_config = KvCacheConfig(enable_block_reuse=False)
    mgr.tokens_per_block = 32
    mgr._mamba_state_cache_interval = 64
    request = SimpleNamespace(prompt_len=150)

    mgr.prepare_expect_snapshot_points([request])

    assert request.expect_snapshot_points == []


@skip_no_cuda
def test_v2_hybrid_pool_ratio_controls_allocated_memory():
    def allocated_memory(pool_ratio):
        mgr = object.__new__(V2MambaHybridCacheManager)
        mgr.kv_cache_type = CacheTypeCpp.SELF
        mgr.head_dim_per_layer = [64, 64]
        mgr.pp_layers = [0, 1]
        mgr._mamba_layer_mask = [True, False]
        mgr.ssm_bytes = 64
        mgr.conv_bytes = 32
        mgr.max_attention_window_vec = [128, 128]
        mgr.max_batch_size = 2
        mgr.max_seq_len = 128
        mgr._mamba_state_cache_interval = 64
        mgr.enable_swa_scratch_reuse = False
        mgr.num_extra_kv_tokens = 0
        mgr.get_layer_bytes_per_token = lambda **kwargs: 8

        config = mgr._build_cache_config(
            KvCacheConfig(
                pool_ratio=pool_ratio,
                enable_partial_reuse=False,
            ),
            tokens_per_block=32,
            vocab_size=1024,
            cache_tiers=[GpuCacheTierConfig(quota=64 << 20)],
        )
        runtime_manager = RuntimeKVCacheManager(config)
        try:
            statistics = runtime_manager._storage.get_statistics()
            allocated_bytes = [
                int(stats.total) * sum(int(size) for size in stats.slot_size)
                for stats in statistics
            ]
            return allocated_bytes, list(runtime_manager._current_gpu_ratio)
        finally:
            runtime_manager.shutdown()

    low_mamba_allocation, low_actual_ratio = allocated_memory([0.25, 0.75])
    high_mamba_allocation, high_actual_ratio = allocated_memory([0.75, 0.25])

    assert low_actual_ratio == pytest.approx([0.25, 0.75])
    assert high_actual_ratio == pytest.approx([0.75, 0.25])
    assert high_mamba_allocation[0] > low_mamba_allocation[0]
    assert high_mamba_allocation[1] < low_mamba_allocation[1]


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
    dtype=DataType.HALF,
    mamba_layer_mask=None,
    attention_layer_mask=None,
    mamba_ssm_cache_dtype=torch.float16,
    use_replay_state_update=False,
):
    """Construct a real CppMambaHybridCacheManager with one mamba layer +
    one full-attention layer so the parent KVCacheManager goes through the
    linear-attention pool sizing path."""
    # Layer 0: mamba; Layer 1: full attention. Single rank, no MPI.
    mamba_mask = mamba_layer_mask or [True, False]
    attn_mask = attention_layer_mask or [False, True]
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
        mamba_num_layers=sum(mamba_mask),
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=sum(attn_mask),
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=max_batch_size,
        mapping=mapping,
        spec_config=spec_config,
        layer_mask=attn_mask,
        is_estimating_kv_cache=is_estimating_kv_cache,
        dtype=dtype,
        use_replay_state_update=use_replay_state_update,
    )


def _build_v2_hybrid_with_mamba_layer(
    max_batch_size=4,
    num_mamba_layers=1,
    spec_config=None,
    use_replay_state_update=False,
    model_type="nemotron_hybrid",
    enable_block_reuse=False,
    enable_partial_reuse=True,
    enable_attention_dp=False,
):
    """Construct a real V2MambaHybridCacheManager."""
    num_attention_layers = 1
    mamba_mask = [True] * num_mamba_layers + [False] * num_attention_layers
    attn_mask = [False] * num_mamba_layers + [True] * num_attention_layers
    mapping = Mapping(
        world_size=1,
        rank=0,
        tp_size=1,
        pp_size=1,
        enable_attention_dp=enable_attention_dp,
    )
    kv_cache_config = KvCacheConfig(
        max_tokens=512,
        enable_block_reuse=enable_block_reuse,
        enable_partial_reuse=enable_partial_reuse,
    )
    return V2MambaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=num_mamba_layers,
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float16,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_attention_layers,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=max_batch_size,
        mapping=mapping,
        spec_config=spec_config,
        layer_mask=attn_mask,
        vocab_size=1024,
        use_replay_state_update=use_replay_state_update,
        model_type=model_type,
    )


def _make_wide_spec_config(max_draft_len=2, tokens_per_gen_step=5):
    """Spec config whose per-step token width is wider than draft depth.

    This mirrors parallel-draft style metadata closely enough for cache-manager
    sizing without constructing the full speculative worker stack.
    """
    return SimpleNamespace(
        max_draft_len=max_draft_len,
        max_total_draft_tokens=tokens_per_gen_step - 1,
        tokens_per_gen_step=tokens_per_gen_step,
        spec_dec_mode=SimpleNamespace(use_one_engine=lambda: False),
    )


def _assert_replay_layer_cache_uses_history_size(layer_cache, history_size):
    assert layer_cache.old_x is not None
    assert layer_cache.old_B is not None
    assert layer_cache.old_dt is not None
    assert layer_cache.old_dA_cumsum is not None
    assert layer_cache.cache_buf_idx is not None
    assert layer_cache.prev_num_accepted_tokens is not None
    assert layer_cache.old_x.dim() == 5
    cache_size = layer_cache.temporal.shape[0]
    assert layer_cache.old_x.shape[0] == cache_size
    assert layer_cache.old_B.shape[0] == cache_size
    assert layer_cache.old_dt.shape[0] == cache_size
    assert layer_cache.old_dA_cumsum.shape[0] == cache_size
    assert layer_cache.cache_buf_idx.shape[0] == cache_size
    assert layer_cache.prev_num_accepted_tokens.shape[0] == cache_size
    assert layer_cache.old_x.shape[1] == 2
    assert layer_cache.old_B.shape[1] == 2
    assert layer_cache.old_dt.shape[1] == 2
    assert layer_cache.old_dA_cumsum.shape[1] == 2
    assert layer_cache.old_x.shape[2] == history_size
    assert layer_cache.old_B.shape[2] == history_size
    assert layer_cache.old_dt.shape[-1] == history_size
    assert layer_cache.old_dA_cumsum.shape[-1] == history_size


@skip_no_cuda
def test_v2_hybrid_allocates_mamba_state_and_dummy_indices():
    mgr = _build_v2_hybrid_with_mamba_layer(max_batch_size=4)
    try:
        assert mgr.local_num_mamba_layers == 1
        assert len(mgr.all_ssm_states) == 1
        assert len(mgr.all_conv_states) == 1
        assert mgr.all_ssm_states[0].shape[1:] == torch.Size([4, 8, 8])
        assert mgr.all_conv_states[0].shape[1:] == torch.Size([48, 3])
        assert mgr.get_max_resource_count() == 4
        assert mgr.blocks_in_primary_pool > 0
        assert isinstance(mgr.check_invalid_values_in_kv_cache(), bool)

        requests = mgr.add_dummy_requests([123], token_nums=[8], is_gen=False)

        assert len(requests) == 1
        indices = mgr.get_state_indices([123], [False])
        assert len(indices) == 1
        assert indices[0] >= 0
        assert mgr.cuda_state_indices[0].item() == indices[0]
        assert mgr.get_ssm_states(0).data_ptr() == mgr.all_ssm_states[0].data_ptr()
        assert mgr.get_conv_states(0).data_ptr() == mgr.all_conv_states[0].data_ptr()
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_dummy_indices_keep_cuda_buffer_address():
    max_batch_size = 4
    mgr = _build_v2_hybrid_with_mamba_layer(max_batch_size=max_batch_size, enable_attention_dp=True)
    try:
        stale_request = mgr.add_dummy_requests([123], token_nums=[8], is_gen=False)[0]
        state_indices_ptr = mgr.cuda_state_indices.data_ptr()

        # Model a transfer-pending prior batch. The new ADP dummy is the only
        # request participating in the next forward pass.
        mgr.requests = [stale_request] * max_batch_size
        new_requests = mgr.add_dummy_requests([456], token_nums=[8], is_gen=False)

        assert len(new_requests) == 1
        assert len(mgr.requests) == max_batch_size + 1
        assert mgr.cuda_state_indices.data_ptr() == state_indices_ptr
        assert mgr.cuda_state_indices[0].item() == mgr.get_state_indices([456], [False])[0]
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_free_resources_drops_stale_state_index_mapping():
    mgr = _build_v2_hybrid_with_mamba_layer()
    try:
        request = mgr.add_dummy_requests([123], token_nums=[8], is_gen=False)[0]
        request_id = request.py_request_id
        assert request_id in mgr._request_id_to_state_index

        mgr.requests.clear()
        mgr.free_resources(request)

        assert request_id not in mgr._request_id_to_state_index
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_uses_upstream_min_snapshot_policy():
    mgr = _build_v2_hybrid_with_mamba_layer(
        enable_block_reuse=True,
        enable_partial_reuse=True,
    )
    try:
        assert mgr.block_reuse_policy is BlockReusePolicy.PER_REQUEST
        assert mgr.kv_cache_config.enable_partial_reuse
        assert mgr.kv_cache_manager_py_config.commit_min_snapshot
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_mamba_state_views_use_logical_slots():
    mgr = _build_v2_hybrid_with_mamba_layer(max_batch_size=4, num_mamba_layers=2)
    try:
        assert len(mgr.all_ssm_states) == 2
        assert len(mgr.all_conv_states) == 2

        ssm_slots = mgr.all_ssm_states[0].shape[0]
        conv_slots = mgr.all_conv_states[0].shape[0]
        assert all(t.shape[0] == ssm_slots for t in mgr.all_ssm_states)
        assert all(t.shape[0] == conv_slots for t in mgr.all_conv_states)
        assert ssm_slots == conv_slots

        for local_layer_idx, ssm_state, conv_state in zip(
            mgr.mamba_local_layer_ids, mgr.all_ssm_states, mgr.all_conv_states
        ):
            layer_id = LayerId(local_layer_idx)
            ssm_scale = mgr.impl.get_page_index_scale(layer_id, Role.SSM_STATE)
            conv_scale = mgr.impl.get_page_index_scale(layer_id, Role.CONV_STATE)
            assert ssm_state.stride(0) == mgr.ssm_count * ssm_scale
            assert conv_state.stride(0) == mgr.conv_count * conv_scale
            assert (
                ssm_state.shape[0]
                == (mgr.impl.get_page_index_upper_bound(layer_id, Role.SSM_STATE) + ssm_scale - 1)
                // ssm_scale
            )
            assert (
                conv_state.shape[0]
                == (mgr.impl.get_page_index_upper_bound(layer_id, Role.CONV_STATE) + conv_scale - 1)
                // conv_scale
            )

        mgr.add_dummy_requests([123, 456], token_nums=[8, 8], is_gen=False)
        indices = mgr.get_state_indices([123, 456], [False, False])
        assert all(0 <= index < ssm_slots for index in indices)
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_cpp_hybrid_replay_buffers_size_by_tokens_per_gen_step():
    spec_config = _make_wide_spec_config(max_draft_len=2, tokens_per_gen_step=5)
    mgr = _build_hybrid_with_mamba_layer(
        spec_config=spec_config,
        max_batch_size=4,
        use_replay_state_update=True,
    )
    try:
        replay_metadata = mgr.get_replay_state_update_metadata()
        assert mgr.use_replay_state_update is True
        assert replay_metadata is not None
        assert replay_metadata.replay_step_width == spec_config.tokens_per_gen_step
        assert replay_metadata.replay_history_size == max(
            MIN_REPLAY_HISTORY_SIZE, spec_config.tokens_per_gen_step
        )
        layer_cache = mgr.mamba_layer_cache(0)
        _assert_replay_layer_cache_uses_history_size(
            layer_cache, replay_metadata.replay_history_size
        )
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_replay_buffers_size_by_tokens_per_gen_step():
    spec_config = _make_wide_spec_config(max_draft_len=2, tokens_per_gen_step=5)
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=4,
        spec_config=spec_config,
        use_replay_state_update=True,
    )
    try:
        replay_metadata = mgr.get_replay_state_update_metadata()
        assert mgr.use_replay_state_update is True
        assert replay_metadata is not None
        assert replay_metadata.replay_step_width == spec_config.tokens_per_gen_step
        assert replay_metadata.replay_history_size == spec_config.tokens_per_gen_step
        layer_cache = mgr.mamba_layer_cache(0)
        _assert_replay_layer_cache_uses_history_size(
            layer_cache, replay_metadata.replay_history_size
        )
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_replay_bookkeeping_matches_checkpoint_predicate(monkeypatch):
    spec_config = _make_wide_spec_config(max_draft_len=2, tokens_per_gen_step=5)
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=4,
        spec_config=spec_config,
        use_replay_state_update=True,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.mamba_cache_manager._promote_mamba_state_triton",
        lambda *args, **kwargs: None,
    )
    try:
        slot = torch.tensor([0], dtype=torch.int32, device="cuda")
        attn_metadata = SimpleNamespace(num_seqs=1, num_contexts=0)

        mgr.update_mamba_states(
            attn_metadata,
            torch.tensor([3], dtype=torch.int32, device="cuda"),
            state_indices=slot,
        )
        assert mgr.prev_num_accepted_tokens[0].item() == 3
        assert mgr.cache_buf_idx[0].item() == 0

        mgr.update_mamba_states(
            attn_metadata,
            torch.tensor([2], dtype=torch.int32, device="cuda"),
            state_indices=slot,
        )
        assert mgr.prev_num_accepted_tokens[0].item() == 2
        assert mgr.cache_buf_idx[0].item() == 1
    finally:
        mgr.shutdown()


@skip_no_cuda
@pytest.mark.parametrize(
    "mamba_ssm_cache_dtype",
    [torch.float16, torch.float32, torch.bfloat16],
)
def test_cpp_hybrid_passes_per_window_pool_dtypes_for_nvfp4_kv_cache(
    mamba_ssm_cache_dtype,
):
    mgr = _build_hybrid_with_mamba_layer(
        dtype=DataType.NVFP4,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
    )
    recurrent_pool_dtype = torch_dtype_to_binding(mamba_ssm_cache_dtype)

    expected_dtypes = [
        (LinearCacheType.RECURRENT_STATES.value, recurrent_pool_dtype),
        (128, DataType.NVFP4),
    ]
    assert [
        (config.window_size, config.dtype) for config in mgr.pool_configurations
    ] == expected_dtypes
    assert [
        (config.window_size, config.dtype) for config in mgr.impl.pool_configurations
    ] == expected_dtypes
    assert mgr._layer_to_pool_idx == {0: 0, 1: 1}
    assert mgr.recurrent_states_pool_index == 0
    assert mgr.impl.get_recurrent_states_pool().dtype == mamba_ssm_cache_dtype

    compact_scale_pointers = mgr.impl.get_block_scale_pool_pointers()
    assert mgr.impl.get_block_pool_pointers().shape == (2, 2)
    assert compact_scale_pointers.shape == (1, 2)
    assert mgr.kv_cache_pool_pointers.shape == (2, 2, 2)
    assert torch.count_nonzero(mgr.kv_cache_pool_pointers[0, :, 1]) == 0
    assert torch.equal(mgr.kv_cache_pool_pointers[1, :, 1], compact_scale_pointers[0])


@skip_no_cuda
def test_cpp_hybrid_merges_compact_scale_rows_with_unmanaged_layers():
    mgr = _build_hybrid_with_mamba_layer(
        dtype=DataType.NVFP4,
        mamba_layer_mask=[True, False, True, False],
        attention_layer_mask=[False, False, False, True],
    )

    assert mgr.pp_layers == [0, 2, 3]
    assert mgr.kv_cache_pool_mapping[:, 0].tolist() == [0, 0, 1]
    compact_scale_pointers = mgr.impl.get_block_scale_pool_pointers()
    assert compact_scale_pointers.shape == (1, 2)
    assert mgr.kv_cache_pool_pointers.shape == (2, 2, 2)
    assert torch.count_nonzero(mgr.kv_cache_pool_pointers[0, :, 1]) == 0
    assert torch.equal(mgr.kv_cache_pool_pointers[1, :, 1], compact_scale_pointers[0])


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
    assert all(
        config.window_size != LinearCacheType.RECURRENT_STATES.value
        for config in mgr.pool_configurations
    )
    assert all(
        config.window_size != LinearCacheType.RECURRENT_STATES.value
        for config in mgr.impl.pool_configurations
    )

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
