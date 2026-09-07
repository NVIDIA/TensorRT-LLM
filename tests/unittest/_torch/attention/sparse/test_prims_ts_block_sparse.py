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

import math
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from utils.util import isSM100Family

from tensorrt_llm._torch.attention.backends import prims_ts
from tensorrt_llm._torch.attention.backends.fmha import prims_ts_block_sparse as block_sparse_fmha
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.sparse.params import (
    BlockSparseForwardInputs,
    SparseRuntimeParams,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.functional import PositionEmbeddingType

pytestmark = pytest.mark.cpu_only

_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not isSM100Family(),
    reason="PrimTS block-sparse attention requires SM100 or SM103",
)


def _bsr_inputs(*, kv_valid_bits: torch.Tensor | None = None):
    return BlockSparseForwardInputs(
        q_block_size=64,
        kv_block_size=64,
        max_blocks_per_row=2,
        block_indptr=torch.tensor([[[0, 2]], [[2, 4]]], dtype=torch.int32),
        block_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        kv_valid_bits=kv_valid_bits,
    )


def _bitmask_inputs(*, proxy: bool):
    summaries = {
        "k_summary": torch.zeros((2, 4, 1, 128), dtype=torch.bfloat16),
        "v_summary": torch.zeros((2, 4, 1, 128), dtype=torch.bfloat16),
    }
    return BlockSparseForwardInputs(
        q_block_size=64,
        kv_block_size=64,
        exact_block_bits=torch.ones((2, 1, 1, 1), dtype=torch.uint32),
        **(summaries if proxy else {}),
    )


def _set_block_sparse_inputs(
    forward_args: AttentionForwardArgs,
    block_sparse_inputs,
) -> None:
    forward_args.sparse_runtime_params = SparseRuntimeParams(
        block_sparse_inputs=block_sparse_inputs
    )


def _get_block_sparse_inputs(forward_args: AttentionForwardArgs):
    block_sparse_inputs = forward_args.sparse_runtime_params.block_sparse_inputs
    assert block_sparse_inputs is not None
    return block_sparse_inputs


def _pack_token_mask(mask: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(32, dtype=torch.int64, device=mask.device)
    weights = torch.ones_like(shifts).bitwise_left_shift_(shifts)
    return (mask.view(1, -1, 32).to(torch.int64) * weights).sum(dim=-1).to(torch.uint32)


def _proxy_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_summary: torch.Tensor,
    v_summary: torch.Tensor,
    exact_block: int,
) -> torch.Tensor:
    block_size = 64
    exact_tokens = torch.arange(
        exact_block * block_size,
        (exact_block + 1) * block_size,
        device=q.device,
    )
    proxy_blocks = [block for block in range(k_summary.shape[1]) if block != exact_block]
    q_rows = q[0, :, 0].float()
    exact_logits = q_rows @ k[0, exact_tokens, 0].float().T
    proxy_logits = q_rows @ k_summary[0, proxy_blocks, 0].float().T
    logits = torch.cat((exact_logits, proxy_logits), dim=1) / math.sqrt(q.shape[-1])
    weights = torch.exp(logits - logits.amax(dim=1, keepdim=True))
    exact_weights, proxy_weights = weights.split((block_size, len(proxy_blocks)), dim=1)
    numerator = exact_weights @ v[0, exact_tokens, 0].float()
    numerator += proxy_weights @ v_summary[0, proxy_blocks, 0].float()
    denominator = exact_weights.sum(dim=1, keepdim=True)
    denominator += proxy_weights.sum(dim=1, keepdim=True) * block_size
    return (numerator / denominator).to(q.dtype)[None, :, None]


class _Attention:
    def __init__(self) -> None:
        self.sparse_params = None
        self.num_heads = 2
        self.num_kv_heads = 1
        self.head_dim = 128
        self.is_mla_enable = False
        self.kv_lora_rank = None
        self.qk_rope_head_dim = None
        self.qk_nope_head_dim = None
        self.v_head_dim = None
        self.q_scaling = 1.0
        self.quant_mode = 0
        self.local_layer_idx = 0
        self.position_embedding_type = PositionEmbeddingType.learned_absolute
        self.attention_chunk_size = 0


def _contiguous_case():
    attention = _Attention()
    fmha = block_sparse_fmha.PrimsTSBlockSparseFmha(attention)
    q = torch.zeros((128, 256), dtype=torch.bfloat16)
    k = torch.zeros((512, 128), dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    metadata = SimpleNamespace(
        is_cross=False,
        num_sparse_topk=0,
        helix_position_offsets=None,
        kv_cache_manager=None,
        seq_lens=torch.tensor([64, 64], dtype=torch.int32),
    )
    args = AttentionForwardArgs(
        output=torch.empty_like(q),
        attention_input_type=AttentionInputType.context_only,
        attention_mask=PredefinedAttentionMask.FULL,
        sparse_runtime_params=SparseRuntimeParams(block_sparse_inputs=_bsr_inputs()),
    )
    return attention, fmha, q, k, v, metadata, args


def _paged_metadata():
    batch_size, max_pages, page_size = 2, 4, 64
    key_pages = torch.arange(batch_size * max_pages, dtype=torch.int32).view(batch_size, max_pages)
    block_offsets = torch.stack((key_pages, key_pages + 8), dim=1).unsqueeze(0)
    manager = Mock(spec=KVCacheManager)
    manager.dtype = torch.bfloat16
    manager.num_pools = manager.num_local_layers = 1
    manager.host_kv_cache_block_offsets = block_offsets
    return SimpleNamespace(
        is_cross=False,
        num_sparse_topk=0,
        helix_position_offsets=None,
        num_contexts=0,
        num_generations=batch_size,
        seq_lens=torch.ones(batch_size, dtype=torch.int32),
        beam_width=1,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        tokens_per_block=page_size,
        max_seq_len=max_pages * page_size,
        kv_layout="HND",
        kv_lens_runtime=torch.tensor([129, 193], dtype=torch.int32),
        kv_cache_block_offsets=block_offsets,
        host_kv_cache_pool_pointers=torch.tensor([[1234, 5678]], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
        kv_cache_manager=manager,
    )


def _paged_case():
    attention = _Attention()
    fmha = block_sparse_fmha.PrimsTSBlockSparseFmha(attention)
    fmha._multi_processor_count = 1
    metadata = _paged_metadata()
    q = torch.zeros((2, 512), dtype=torch.bfloat16)
    args = AttentionForwardArgs(
        output=torch.empty((2, 256), dtype=q.dtype),
        attention_input_type=AttentionInputType.generation_only,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_window_size=metadata.max_seq_len,
        is_fused_qkv=True,
        sparse_runtime_params=SparseRuntimeParams(block_sparse_inputs=_bsr_inputs()),
    )
    return attention, fmha, q, metadata, args


def test_block_sparse_route_mode_is_derived_from_payload() -> None:
    bsr = _bsr_inputs()
    exact = _bitmask_inputs(proxy=False)
    proxy = _bitmask_inputs(proxy=True)

    assert (bsr.sparse_format, bsr.use_proxy_routes) == ("bsr", False)
    assert (exact.sparse_format, exact.use_proxy_routes) == ("bitmask", False)
    assert (proxy.sparse_format, proxy.use_proxy_routes) == ("bitmask", True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"block_indices": None}, "block_indptr and block_indices"),
        ({"max_blocks_per_row": None}, "max_blocks_per_row"),
        (
            {"exact_block_bits": torch.ones((1, 1, 1, 1), dtype=torch.uint32)},
            "exactly one route representation",
        ),
        ({"k_summary": torch.empty(0)}, "k_summary and v_summary"),
    ],
)
def test_block_sparse_payload_rejects_ambiguous_combinations(overrides, message) -> None:
    kwargs = {
        "q_block_size": 64,
        "kv_block_size": 64,
        "max_blocks_per_row": 1,
        "block_indptr": torch.tensor([[[0, 1]]], dtype=torch.int32),
        "block_indices": torch.tensor([0], dtype=torch.int32),
    }
    kwargs.update(overrides)

    with pytest.raises((TypeError, ValueError), match=message):
        BlockSparseForwardInputs(**kwargs)


def test_block_sparse_support_is_phase_specific_and_paged_proxy_is_rejected(
    monkeypatch,
) -> None:
    _attention, contiguous, q, k, v, metadata, args = _contiguous_case()
    monkeypatch.setattr(contiguous, "_common_unsupported_reason", Mock(return_value=None))
    assert contiguous.is_supported(q, k, v, metadata, args, phase=FmhaPhase.CONTEXT)
    assert not contiguous.is_supported(q, k, v, metadata, args, phase=FmhaPhase.GENERATION)

    _attention, paged, q, metadata, args = _paged_case()
    _set_block_sparse_inputs(args, _bitmask_inputs(proxy=True))
    _supported, reason = paged._is_supported_with_reason(
        q, None, None, metadata, args, phase=FmhaPhase.GENERATION
    )
    assert not _supported
    assert reason == "paged block-sparse attention only supports BSR exact routes"


def test_contiguous_proxy_routes_reject_causal_mask_before_planning(monkeypatch) -> None:
    _attention, fmha, q, k, v, metadata, args = _contiguous_case()
    _set_block_sparse_inputs(args, _bitmask_inputs(proxy=True))
    args.attention_mask = PredefinedAttentionMask.CAUSAL
    monkeypatch.setattr(fmha, "_common_unsupported_reason", Mock(return_value=None))

    supported, reason = fmha._is_supported_with_reason(
        q,
        k,
        v,
        metadata,
        args,
        phase=FmhaPhase.CONTEXT,
    )

    assert not supported
    assert reason == "block-sparse proxy routes require mask_type='dense'"


@pytest.mark.parametrize("paged", [False, True])
def test_block_sparse_support_rejects_invalid_static_kernel_profile(
    monkeypatch,
    paged,
) -> None:
    if paged:
        attention, fmha, q, metadata, args = _paged_case()
        attention.head_dim = 64
        q = torch.zeros((2, 256), dtype=torch.bfloat16)
        args.output = torch.empty((2, 128), dtype=q.dtype)
        monkeypatch.setattr(fmha, "_common_unsupported_reason", Mock(return_value=None))
        supported, reason = fmha._is_supported_with_reason(
            q,
            None,
            None,
            metadata,
            args,
            phase=FmhaPhase.GENERATION,
        )
    else:
        attention, fmha, _q, _k, _v, metadata, args = _contiguous_case()
        attention.head_dim = 64
        q = torch.zeros((128, 128), dtype=torch.bfloat16)
        k = torch.zeros((512, 64), dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        args.output = torch.empty_like(q)
        monkeypatch.setattr(fmha, "_common_unsupported_reason", Mock(return_value=None))
        supported, reason = fmha._is_supported_with_reason(
            q,
            k,
            v,
            metadata,
            args,
            phase=FmhaPhase.CONTEXT,
        )

    assert not supported
    assert reason == "block-sparse requires head_dim=128"


def test_contiguous_wrappers_cache_static_profile_and_keep_routes_live(monkeypatch) -> None:
    _attention, fmha, q, k, v, _metadata, args = _contiguous_case()
    wrapper = Mock()
    factory = Mock(return_value=wrapper)
    monkeypatch.setattr(block_sparse_fmha, "_BlockSparseTSWrapper", factory)

    bsr_inputs = [
        _bsr_inputs(),
        BlockSparseForwardInputs(
            q_block_size=64,
            kv_block_size=64,
            max_blocks_per_row=2,
            block_indptr=torch.tensor([[[0, 1]], [[1, 4]]], dtype=torch.int32),
            block_indices=torch.tensor([3, 1, 0, 2], dtype=torch.int32),
        ),
    ]
    for inputs in bsr_inputs:
        _set_block_sparse_inputs(args, inputs)
        fmha._forward_contiguous(q, k, v, args)

    proxy_inputs = [_bitmask_inputs(proxy=True), _bitmask_inputs(proxy=True)]
    for inputs in proxy_inputs:
        _set_block_sparse_inputs(args, inputs)
        fmha._forward_contiguous(q, k, v, args)

    assert factory.call_count == 2
    assert wrapper.plan.call_count == 2
    assert wrapper.plan.call_args_list[0].kwargs["sparse_format"] == "bsr"
    assert wrapper.plan.call_args_list[0].kwargs["use_proxy_routes"] is False
    assert wrapper.plan.call_args_list[1].kwargs["sparse_format"] == "bitmask"
    assert wrapper.plan.call_args_list[1].kwargs["use_proxy_routes"] is True
    assert wrapper.plan.call_args_list[1].kwargs["max_blocks_per_row"] == 4
    assert wrapper.run.call_count == 4

    for call, inputs in zip(wrapper.run.call_args_list[:2], bsr_inputs):
        assert call.kwargs["block_indptr"] is inputs.block_indptr
        assert call.kwargs["block_indices"] is inputs.block_indices
    for call, inputs in zip(wrapper.run.call_args_list[2:], proxy_inputs):
        assert call.kwargs["exact_block_bits"] is inputs.exact_block_bits
        assert call.kwargs["k_summary"] is inputs.k_summary
        assert call.kwargs["v_summary"] is inputs.v_summary


def test_block_sparse_plan_key_includes_attention_head_topology() -> None:
    inputs = _bitmask_inputs(proxy=True)
    q = torch.empty((128, 256), dtype=torch.bfloat16)
    first_attention = _Attention()
    second_attention = _Attention()
    second_attention.num_heads = 4
    first = block_sparse_fmha.PrimsTSBlockSparseFmha(first_attention)
    second = block_sparse_fmha.PrimsTSBlockSparseFmha(second_attention)

    def _key(fmha):
        return fmha._make_plan_key(
            q,
            inputs,
            batch_size=1,
            seq_len_q=128,
            kv_capacity=256,
            page_size=None,
            mask_type="dense",
        )

    assert _key(first) != _key(second)


def test_block_sparse_plan_cache_is_shared_only_when_explicitly_bound() -> None:
    first = block_sparse_fmha.PrimsTSBlockSparseFmha(_Attention())
    second = block_sparse_fmha.PrimsTSBlockSparseFmha(_Attention())

    assert first._contiguous_wrappers is not second._contiguous_wrappers
    assert first._paged_wrappers is not second._paged_wrappers

    cache_state = {}
    first.bind_plan_cache(cache_state)
    second.bind_plan_cache(cache_state)

    assert first._contiguous_wrappers is second._contiguous_wrappers
    assert first._paged_wrappers is second._paged_wrappers
    assert cache_state == {
        "contiguous_wrappers": {},
        "paged_wrappers": {},
    }


def test_paged_wrapper_uses_zero_copy_padded_row_stride_block_tables(monkeypatch) -> None:
    attention, fmha, q, metadata, args = _paged_case()
    wrapper = Mock()
    monkeypatch.setattr(block_sparse_fmha, "_BlockSparsePagedTSWrapper", Mock(return_value=wrapper))
    monkeypatch.setattr(block_sparse_fmha, "get_kv_page_offset", Mock(return_value=8))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", Mock(return_value=False))
    q_processed = torch.zeros((2, 2, 128), dtype=torch.bfloat16)
    kv_pool = torch.empty((16, 1, 64, 128), dtype=torch.bfloat16)
    block_tables = metadata.kv_cache_block_offsets[0]
    empty = torch.empty(0, dtype=torch.uint8)
    preprocessed = (q_processed, kv_pool, block_tables, None, 1.0, 1.0) + (
        empty,
        None,
        1,
        256,
        -1,
        False,
    )
    monkeypatch.setattr(fmha, "_run_generation_preprocess", Mock(return_value=preprocessed))
    params = FmhaParams(
        attn=attention,
        meta=metadata,
        fwd=args,
        workspace=torch.empty(0, dtype=torch.uint8),
        qkv_input=q,
        context_buf=args.output,
        sequence_lengths=torch.tensor([129, 193], dtype=torch.int32),
        input_seq_length=1,
        tokens_per_block=64,
        num_requests=2,
    )
    expected_block_tables = block_tables[:2, 0, :]
    snapshots = []

    def snapshot(*_args, **kwargs):
        snapshots.append(
            (
                kwargs["seq_lens_kv"].clone(),
                kwargs["block_tables"],
                kwargs["block_tables"].clone(),
                kwargs["block_indptr"],
                kwargs["block_indices"],
            )
        )

    wrapper.run.side_effect = snapshot
    first_inputs = _get_block_sparse_inputs(args)
    fmha.run_generation(params)
    block_tables[:, 0].add_(10)
    params.sequence_lengths = torch.tensor([130, 194], dtype=torch.int32)
    _set_block_sparse_inputs(
        args,
        BlockSparseForwardInputs(
            q_block_size=64,
            kv_block_size=64,
            max_blocks_per_row=2,
            block_indptr=torch.tensor([[[0, 1]], [[1, 4]]], dtype=torch.int32),
            block_indices=torch.tensor([3, 2, 1, 0], dtype=torch.int32),
        ),
    )
    fmha.run_generation(params)

    wrapper.plan.assert_called_once()
    assert wrapper.run.call_count == 2
    torch.testing.assert_close(snapshots[0][0], torch.tensor([129, 193], dtype=torch.int32))
    torch.testing.assert_close(snapshots[1][0], torch.tensor([130, 194], dtype=torch.int32))
    assert snapshots[0][1].data_ptr() == expected_block_tables.data_ptr()
    assert snapshots[1][1].data_ptr() == expected_block_tables.data_ptr()
    assert snapshots[0][1].shape == (2, 4)
    assert snapshots[0][1].stride() == (8, 1)
    torch.testing.assert_close(snapshots[0][2], torch.arange(8, dtype=torch.int32).view(2, 4))
    torch.testing.assert_close(snapshots[1][2], torch.arange(8, dtype=torch.int32).view(2, 4) + 10)
    assert snapshots[0][3] is first_inputs.block_indptr
    assert snapshots[1][3] is _get_block_sparse_inputs(args).block_indptr


def test_paged_block_tables_remain_live_across_graph_replay(monkeypatch) -> None:
    attention, fmha, q, metadata, args = _paged_case()
    wrapper = Mock()
    monkeypatch.setattr(block_sparse_fmha, "_BlockSparsePagedTSWrapper", Mock(return_value=wrapper))
    monkeypatch.setattr(block_sparse_fmha, "get_kv_page_offset", Mock(return_value=8))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", Mock(return_value=True))
    q_processed = torch.zeros((2, 2, 128), dtype=torch.bfloat16)
    kv_pool = torch.empty((16, 1, 64, 128), dtype=torch.bfloat16)
    block_tables = metadata.kv_cache_block_offsets[0]
    empty = torch.empty(0, dtype=torch.uint8)
    preprocessed = (q_processed, kv_pool, block_tables, None, 1.0, 1.0) + (
        empty,
        None,
        1,
        256,
        -1,
        False,
    )
    monkeypatch.setattr(fmha, "_run_generation_preprocess", Mock(return_value=preprocessed))
    params = FmhaParams(
        attn=attention,
        meta=metadata,
        fwd=args,
        workspace=torch.empty(0, dtype=torch.uint8),
        qkv_input=q,
        context_buf=args.output,
        sequence_lengths=torch.tensor([129, 193], dtype=torch.int32),
        input_seq_length=1,
        tokens_per_block=64,
        num_requests=2,
    )
    seen = []

    def snapshot(*_args, **kwargs):
        seen.append((kwargs["block_tables"].data_ptr(), kwargs["block_tables"].clone()))

    wrapper.run.side_effect = snapshot
    fmha.run_generation(params)
    block_tables[:, 0, :].add_(10)
    block_tables[:, 1, :].fill_(-1)
    fmha.run_generation(params)

    assert seen[0][0] == seen[1][0] == block_tables.data_ptr()
    torch.testing.assert_close(seen[0][1], torch.arange(8, dtype=torch.int32).view(2, 4))
    torch.testing.assert_close(seen[1][1], torch.arange(8, dtype=torch.int32).view(2, 4) + 10)


def test_prepare_workspace_checks_capture_before_resize(monkeypatch) -> None:
    _attention, fmha, _q, _metadata, _args = _paged_case()
    query_device = torch.device("cuda:1")
    q = SimpleNamespace(
        device=query_device,
        dtype=torch.bfloat16,
        shape=(2, 512),
    )
    metadata = SimpleNamespace(
        kv_cache_manager=object(),
        kv_cache_block_offsets=SimpleNamespace(device=query_device, shape=(1, 2, 4)),
        max_num_requests=2,
        tokens_per_block=64,
        num_generations=2,
    )
    workspace = torch.empty(0, dtype=torch.uint8)
    monkeypatch.setattr(
        fmha,
        "_get_generation_workspace_layout",
        Mock(return_value={"total_size": 16}),
    )
    fmha._multi_processor_count = 1
    device_scope = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch.cuda, "device", device_scope)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", Mock(return_value=True))

    with pytest.raises(RuntimeError, match="workspace must be sized"):
        fmha.prepare_workspace(q, None, None, metadata, _args, workspace)

    device_scope.assert_called_once_with(query_device)
    assert workspace.numel() == 0


def test_prepare_workspace_skips_generation_layout_for_contiguous_requests(monkeypatch) -> None:
    _attention, fmha, q, _k, _v, metadata, args = _contiguous_case()
    layout = Mock()
    monkeypatch.setattr(fmha, "_get_generation_workspace_layout", layout)
    monkeypatch.setattr(torch.cuda, "device", Mock(return_value=nullcontext()))
    fmha._multi_processor_count = 1
    workspace = torch.empty(0, dtype=torch.uint8)

    fmha.prepare_workspace(q, None, None, metadata, args, workspace)

    layout.assert_not_called()
    assert workspace.numel() == 0


@_REQUIRES_PRIMTS_GPU
@torch.no_grad()
def test_real_gpu_raw_routes_and_token_mask_match_reference() -> None:
    torch.manual_seed(1234)
    q = torch.randn((1, 128, 1, 128), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 256, 1, 128), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    token_mask = torch.ones(256, device="cuda", dtype=torch.bool)
    token_mask[[1, 63, 64, 95, 129, 190, 255]] = False
    inputs = BlockSparseForwardInputs(
        q_block_size=64,
        kv_block_size=64,
        max_blocks_per_row=3,
        block_indptr=torch.tensor([[[0, 2, 5]]], device="cuda", dtype=torch.int32),
        block_indices=torch.tensor([0, 2, 0, 1, 3], device="cuda", dtype=torch.int32),
        kv_valid_bits=_pack_token_mask(token_mask),
    )
    sm_scale = 128**-0.5

    key_blocks = torch.arange(256, device="cuda") // 64
    allowed = torch.zeros((128, 256), device="cuda", dtype=torch.bool)
    for row, selected_blocks in enumerate(((0, 2), (0, 1, 3))):
        selected = torch.tensor(selected_blocks, device="cuda")
        allowed[row * 64 : (row + 1) * 64] = torch.isin(key_blocks, selected) & token_mask
    scores = (q[0, :, 0].float() @ k[0, :, 0].float().T) * sm_scale
    expected = (
        torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1) @ v[0, :, 0].float()
    ).to(q.dtype)[None, :, None, :]

    actual = prims_ts.block_sparse_attention(
        q,
        k,
        v,
        block_indptr=inputs.block_indptr,
        block_indices=inputs.block_indices,
        q_block_size=inputs.q_block_size,
        kv_block_size=inputs.kv_block_size,
        kv_valid_bits=inputs.kv_valid_bits,
        sm_scale=sm_scale,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@_REQUIRES_PRIMTS_GPU
@torch.no_grad()
def test_real_gpu_proxy_adapter_replays_live_routes_and_summaries() -> None:
    torch.manual_seed(20260901)
    q = torch.randn((1, 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 192, 1, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k_blocks = k.float().view(1, 3, 64, 1, 128)
    v_blocks = v.float().view(1, 3, 64, 1, 128)
    initial_k_summary = k_blocks.mean(dim=2).to(k.dtype)
    initial_v_summary = v_blocks.sum(dim=2).to(v.dtype)
    live_k_summary = initial_k_summary.clone()
    live_v_summary = initial_v_summary.clone()
    live_exact_bits = torch.tensor([[[[1]]]], device="cuda", dtype=torch.uint32)

    attention = _Attention()
    attention.num_heads = attention.num_kv_heads = 1
    fmha = block_sparse_fmha.PrimsTSBlockSparseFmha(attention)
    output = torch.empty_like(q).view(64, 128)
    args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.context_only,
        attention_mask=PredefinedAttentionMask.FULL,
        sparse_runtime_params=SparseRuntimeParams(
            block_sparse_inputs=BlockSparseForwardInputs(
                q_block_size=64,
                kv_block_size=64,
                exact_block_bits=live_exact_bits,
                k_summary=live_k_summary,
                v_summary=live_v_summary,
            ),
        ),
    )
    metadata = SimpleNamespace(
        is_cross=False,
        kv_cache_manager=None,
        seq_lens=torch.tensor([64], dtype=torch.int32),
    )
    flat_q, flat_k, flat_v = (tensor.flatten(0, 2) for tensor in (q, k, v))

    fmha.forward(flat_q, flat_k, flat_v, metadata, args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fmha.forward(flat_q, flat_k, flat_v, metadata, args)

    graph.replay()
    torch.cuda.synchronize()
    expected = _proxy_reference(q, k, v, live_k_summary, live_v_summary, exact_block=0)
    torch.testing.assert_close(output.view_as(q), expected, rtol=2e-2, atol=2e-2)

    live_exact_bits.fill_(1 << 2)
    live_k_summary.copy_((initial_k_summary.float() * 0.5 + 0.125).to(k.dtype))
    live_v_summary.copy_((initial_v_summary.float() * -0.25).to(v.dtype))
    graph.replay()
    torch.cuda.synchronize()
    expected = _proxy_reference(q, k, v, live_k_summary, live_v_summary, exact_block=2)
    torch.testing.assert_close(output.view_as(q), expected, rtol=2e-2, atol=2e-2)


@_REQUIRES_PRIMTS_GPU
@torch.no_grad()
def test_real_gpu_paged_routes_use_live_length_below_capacity() -> None:
    torch.manual_seed(7)
    q = torch.randn((1, 64, 1, 128), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((4, 1, 64, 128), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    page_indices = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([160], device="cuda", dtype=torch.int32)
    inputs = BlockSparseForwardInputs(
        q_block_size=64,
        kv_block_size=64,
        max_blocks_per_row=2,
        block_indptr=torch.tensor([[[0, 2]]], device="cuda", dtype=torch.int32),
        block_indices=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
    )
    sm_scale = 128**-0.5

    actual = prims_ts.block_sparse_attention_with_paged_kv_cache(
        q,
        (k_cache, v_cache),
        block_tables=page_indices.view(1, 4),
        seq_lens_kv=seq_lens_kv,
        block_indptr=inputs.block_indptr,
        block_indices=inputs.block_indices,
        max_seq_len_kv=256,
        q_block_size=inputs.q_block_size,
        kv_block_size=inputs.kv_block_size,
        sm_scale=sm_scale,
    )

    logical_k = k_cache.index_select(0, page_indices.long()).reshape(256, 1, 128)
    logical_v = v_cache.index_select(0, page_indices.long()).reshape(256, 1, 128)
    allowed = torch.zeros(256, device="cuda", dtype=torch.bool)
    allowed[:64] = True
    allowed[128:160] = True
    scores = (q[0, :, 0].float() @ logical_k[:, 0].float().T) * sm_scale
    expected = (
        torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1) @ logical_v[:, 0].float()
    ).to(q.dtype)[None, :, None, :]

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


# Block-sparse FMHA support matrix (generic PrimTS block-sparse kernels):
#
#   GPU architecture       SM100 and SM103
#   Compute phase          Contiguous prefill with separate Q/K/V and no KV cache;
#                          fixed-query generation over a paged HND KV cache
#   Attention type         MHA, MQA, and GQA; num_heads % num_kv_heads == 0
#   Q/K/V head dimension   128
#   Model dtype            BF16 or FP16; Q, K/V, and output share one dtype
#   Route format           BSR (block_indptr [B, Hkv, num_q_blocks + 1] plus flat
#                          block IDs) or a packed block bitmask, optionally with
#                          K/V block summaries for proxy routes (contiguous only)
#   KV block size          8, 16, 32, or a positive multiple of 64 (contiguous);
#                          a positive multiple of 64 (paged)
#   KV-cache layout        Paged HND; page size 64 or 128 (a page holds at least
#                          one 64-token route fragment)
#   Attention semantics    Dense or causal; proxy routes require dense

_REAL_GPU_DTYPES = (torch.float16, torch.bfloat16)
# (num_heads, num_kv_heads): MHA, GQA, and MQA head topologies.
_REAL_GPU_HEAD_TOPOLOGIES = ((1, 1), (8, 8), (8, 2), (8, 1))
_REAL_GPU_KV_BLOCK_SIZES = (64, 128)
_REAL_GPU_PAGE_SIZES = (64, 128)


def _bsr_routes_per_kv_head(
    num_kv_heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    num_selected: int,
) -> tuple[torch.Tensor, torch.Tensor, list[list[list[int]]]]:
    """Build head-dependent, increasing BSR routes and the selection per (head, q block)."""
    selections: list[list[list[int]]] = []
    indptr = torch.zeros((1, num_kv_heads, num_q_blocks + 1), dtype=torch.int32)
    indices: list[int] = []
    for head_idx in range(num_kv_heads):
        head_selection = []
        indptr[0, head_idx, 0] = len(indices)
        for q_block in range(num_q_blocks):
            start = (head_idx + q_block) % num_kv_blocks
            selected = sorted({(start + step) % num_kv_blocks for step in range(num_selected)})
            indices.extend(selected)
            indptr[0, head_idx, q_block + 1] = len(indices)
            head_selection.append(selected)
        selections.append(head_selection)
    return indptr.cuda(), torch.tensor(indices, dtype=torch.int32, device="cuda"), selections


def _reference_block_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selections: list[list[list[int]]],
    *,
    q_block_size: int,
    kv_block_size: int,
    seq_len_kv: int,
) -> torch.Tensor:
    """Dense reference with a per-(kv head, q block) block mask; Q/K/V are [1, S, H, D]."""
    num_heads, num_kv_heads = q.shape[2], k.shape[2]
    heads_per_kv = num_heads // num_kv_heads
    key_blocks = torch.arange(seq_len_kv, device=q.device) // kv_block_size
    outputs = []
    for head_idx in range(num_heads):
        kv_head = head_idx // heads_per_kv
        allowed = torch.zeros((q.shape[1], seq_len_kv), device=q.device, dtype=torch.bool)
        for q_block, selected in enumerate(selections[kv_head]):
            rows = slice(q_block * q_block_size, (q_block + 1) * q_block_size)
            allowed[rows] = torch.isin(key_blocks, torch.tensor(selected, device=q.device))
        scores = (q[0, :, head_idx].float() @ k[0, :seq_len_kv, kv_head].float().T) * (
            q.shape[-1] ** -0.5
        )
        probs = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        outputs.append(probs @ v[0, :seq_len_kv, kv_head].float())
    return torch.stack(outputs, dim=1).to(q.dtype)[None]


@_REQUIRES_PRIMTS_GPU
@torch.no_grad()
@pytest.mark.parametrize("dtype", _REAL_GPU_DTYPES, ids=lambda d: str(d).removeprefix("torch."))
@pytest.mark.parametrize("num_heads,num_kv_heads", _REAL_GPU_HEAD_TOPOLOGIES)
@pytest.mark.parametrize("kv_block_size", _REAL_GPU_KV_BLOCK_SIZES)
def test_real_gpu_contiguous_routes_match_reference_across_topologies(
    dtype: torch.dtype, num_heads: int, num_kv_heads: int, kv_block_size: int
) -> None:
    torch.manual_seed(7)
    seq_len_q, seq_len_kv, q_block_size = 128, 512, 64
    q = torch.randn((1, seq_len_q, num_heads, 128), device="cuda", dtype=dtype)
    k = torch.randn((1, seq_len_kv, num_kv_heads, 128), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    num_kv_blocks = seq_len_kv // kv_block_size
    block_indptr, block_indices, selections = _bsr_routes_per_kv_head(
        num_kv_heads, seq_len_q // q_block_size, num_kv_blocks, num_selected=2
    )

    actual = prims_ts.block_sparse_attention(
        q,
        k,
        v,
        block_indptr=block_indptr,
        block_indices=block_indices,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )

    expected = _reference_block_sparse(
        q,
        k,
        v,
        selections,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        seq_len_kv=seq_len_kv,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@_REQUIRES_PRIMTS_GPU
@torch.no_grad()
@pytest.mark.parametrize("dtype", _REAL_GPU_DTYPES, ids=lambda d: str(d).removeprefix("torch."))
@pytest.mark.parametrize("num_heads,num_kv_heads", _REAL_GPU_HEAD_TOPOLOGIES)
@pytest.mark.parametrize("kv_block_size", _REAL_GPU_KV_BLOCK_SIZES)
@pytest.mark.parametrize("page_size", _REAL_GPU_PAGE_SIZES)
def test_real_gpu_paged_routes_match_reference_across_topologies(
    dtype: torch.dtype, num_heads: int, num_kv_heads: int, kv_block_size: int, page_size: int
) -> None:
    torch.manual_seed(11)
    max_seq_len_kv, seq_len_kv = 512, 400
    num_pages = max_seq_len_kv // page_size
    q = torch.randn((1, 1, num_heads, 128), device="cuda", dtype=dtype)
    k_cache = torch.randn((num_pages, num_kv_heads, page_size, 128), device="cuda", dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    page_indices = torch.randperm(num_pages, device="cuda").to(torch.int32)
    num_kv_blocks = -(-seq_len_kv // kv_block_size)
    block_indptr, block_indices, selections = _bsr_routes_per_kv_head(
        num_kv_heads, 1, num_kv_blocks, num_selected=3
    )

    actual = prims_ts.block_sparse_attention_with_paged_kv_cache(
        q,
        (k_cache, v_cache),
        block_tables=page_indices.view(1, num_pages),
        seq_lens_kv=torch.tensor([seq_len_kv], device="cuda", dtype=torch.int32),
        block_indptr=block_indptr,
        block_indices=block_indices,
        max_seq_len_kv=max_seq_len_kv,
        q_block_size=64,
        kv_block_size=kv_block_size,
    )

    logical_k = k_cache.index_select(0, page_indices.long()).permute(0, 2, 1, 3)
    logical_k = logical_k.reshape(1, max_seq_len_kv, num_kv_heads, 128)
    logical_v = v_cache.index_select(0, page_indices.long()).permute(0, 2, 1, 3)
    logical_v = logical_v.reshape(1, max_seq_len_kv, num_kv_heads, 128)
    expected = _reference_block_sparse(
        q,
        logical_k,
        logical_v,
        selections,
        q_block_size=64,
        kv_block_size=kv_block_size,
        seq_len_kv=seq_len_kv,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_paged_support_rejects_pages_smaller_than_a_route_fragment() -> None:
    page_size = 32
    _attention, fmha, q, metadata, args = _paged_case()
    metadata.tokens_per_block = page_size
    metadata.max_seq_len = 4 * page_size
    metadata.kv_lens_runtime = torch.tensor([page_size + 1, 3 * page_size], dtype=torch.int32)
    args = AttentionForwardArgs(
        output=args.output,
        attention_input_type=args.attention_input_type,
        attention_mask=args.attention_mask,
        attention_window_size=metadata.max_seq_len,
        is_fused_qkv=True,
        sparse_runtime_params=args.sparse_runtime_params,
    )

    reason = fmha._paged_unsupported_reason(q, metadata, args)

    assert reason == "atom_size must not exceed page_size"
