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

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from utils.util import isSM100Family

from tensorrt_llm._torch.attention_backend import prims_ts
from tensorrt_llm._torch.attention_backend.fmha import prims_ts as prims_ts_fmha_module
from tensorrt_llm._torch.attention_backend.fmha import (
    prims_ts_block_sparse as block_sparse_fmha_module,
)
from tensorrt_llm._torch.attention_backend.fmha.prims_ts_block_sparse import (
    PrimsTSBlockSparseFmha,
    PrimsTSBlockSparseRuntime,
)
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention_backend.sparse.block_sparse import (
    BlockSparseForwardInputs,
    BlockSparseParams,
    BlockSparseRouteBuilder,
    BlockSparseRoutes,
    pack_kv_token_mask,
)
from tensorrt_llm.functional import PositionEmbeddingType

_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not isSM100Family(),
    reason="PrimTS block-sparse attention requires SM100 or SM103",
)


def _qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.zeros((1, 128, 2, 128), dtype=torch.float16)
    k = torch.zeros((1, 256, 2, 128), dtype=torch.float16)
    return q, k, k.clone()


def _raw_routes(indices: torch.Tensor) -> BlockSparseRoutes:
    return BlockSparseRoutes(
        block_indptr=torch.tensor([[[0, 1, 4], [4, 5, 8]]], dtype=torch.int32),
        block_indices=indices,
        max_blocks_per_row=3,
    )


def _routes(batch_size: int = 2) -> BlockSparseRoutes:
    starts = torch.arange(batch_size, dtype=torch.int32).mul_(2)
    return BlockSparseRoutes(
        block_indptr=torch.stack((starts, starts + 2), dim=-1).view(batch_size, 1, 2),
        block_indices=torch.arange(2 * batch_size, dtype=torch.int32).remainder_(4),
        max_blocks_per_row=2,
    )


class _Attention:
    def __init__(self, params: BlockSparseParams) -> None:
        self.sparse_params = params
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
        self.predicted_tokens_per_seq = 1
        self.position_embedding_type = PositionEmbeddingType.learned_absolute
        self.attention_chunk_size = 0
        self.rope_dim = self.head_dim
        self.rope_params = SimpleNamespace(
            dim=self.head_dim,
            theta=10000.0,
            scale_type=0,
            scale=1.0,
            max_positions=4096,
        )
        self.rotary_inv_freq = None
        self.rotary_cos_sin = None
        self.attention_metadata_state = None


def _paged_metadata(*, query_lengths: tuple[int, ...] = (1, 1)) -> SimpleNamespace:
    batch_size = len(query_lengths)
    max_pages = 4
    page_size = 64
    kv_lengths = torch.tensor([129, 193], dtype=torch.int32)[:batch_size]
    return SimpleNamespace(
        is_cross=False,
        num_contexts=0,
        num_generations=batch_size,
        num_ctx_tokens=0,
        seq_lens=torch.tensor(query_lengths, dtype=torch.int32),
        beam_width=1,
        max_num_requests=batch_size,
        tokens_per_block=page_size,
        max_seq_len=max_pages * page_size,
        kv_layout="HND",
        kv_lens_runtime=kv_lengths,
        kv_lens_cuda_runtime=kv_lengths,
        kv_cache_block_offsets=torch.arange(batch_size * 2 * max_pages, dtype=torch.int32).view(
            1, batch_size, 2, max_pages
        ),
        host_kv_cache_pool_pointers=torch.tensor([[1234, 5678]], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            dtype=torch.bfloat16,
            impl=SimpleNamespace(),
            enable_swa_scratch_reuse=False,
            num_local_layers=1,
            num_pools=1,
            blocks_in_primary_pool=max_pages,
            kv_offset=torch.tensor([max_pages], dtype=torch.int32),
        ),
        effective_workspace=torch.empty(4096, dtype=torch.uint8),
        cache_indirection=None,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        spec_decoding_generation_lengths=None,
        spec_decoding_position_offsets_for_cpp=None,
    )


def _paged_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    query_lengths: tuple[int, ...] = (1, 1),
    position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.learned_absolute,
) -> tuple[
    _Attention, PrimsTSBlockSparseFmha, Mock, torch.Tensor, SimpleNamespace, AttentionForwardArgs
]:
    runtime = Mock()
    monkeypatch.setattr(block_sparse_fmha_module, "PrimsTSBlockSparseRuntime", lambda: runtime)
    monkeypatch.setattr(
        block_sparse_fmha_module,
        "_get_prims_ts_block_sparse_metadata_unsupported_reason",
        lambda *args, **kwargs: None,
    )
    attention = _Attention(BlockSparseParams(q_block_size=64, kv_block_size=64))
    attention.position_embedding_type = position_embedding_type
    fmha = PrimsTSBlockSparseFmha(attention)
    metadata = _paged_metadata(query_lengths=query_lengths)
    q = torch.zeros(
        (sum(query_lengths), (2 + 2 * 1) * 128),
        dtype=torch.bfloat16,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((sum(query_lengths), 2 * 128), dtype=q.dtype),
        attention_input_type=AttentionInputType.generation_only,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_window_size=metadata.max_seq_len,
        is_fused_qkv=True,
        block_sparse_inputs=BlockSparseForwardInputs(routes=_routes(len(query_lengths))),
    )
    return attention, fmha, runtime, q, metadata, forward_args


def test_vendored_block_sparse_api_keeps_routes_and_paged_metadata_live() -> None:
    assert {
        "BlockSparseTSWrapper",
        "BlockSparsePagedTSWrapper",
        "block_sparse_attention",
        "block_sparse_attention_with_paged_kv_cache",
    } <= set(prims_ts.__all__)

    plan = inspect.signature(prims_ts.BlockSparseTSWrapper.plan).parameters
    assert {"device", "max_blocks_per_row", "use_kv_valid_bits"} <= plan.keys()
    assert {"block_indptr", "block_indices", "kv_valid_bits"}.isdisjoint(plan)
    run = inspect.signature(prims_ts.BlockSparseTSWrapper.run).parameters
    assert {"block_indptr", "block_indices", "kv_valid_bits"} <= run.keys()

    paged_plan = inspect.signature(prims_ts.BlockSparsePagedTSWrapper.plan).parameters
    assert {"batch_size", "device", "max_seq_len_kv"} <= paged_plan.keys()
    assert {"paged_kv_indptr", "paged_kv_indices", "seq_lens_kv"}.isdisjoint(paged_plan)
    paged_run = inspect.signature(prims_ts.BlockSparsePagedTSWrapper.run).parameters
    assert {
        "paged_kv_indptr",
        "paged_kv_indices",
        "seq_lens_kv",
        "block_indptr",
        "block_indices",
    } <= paged_run.keys()


def test_uniform_routes_sort_reuse_indptr_and_fail_capture_miss(monkeypatch) -> None:
    builder = BlockSparseRouteBuilder()
    storage = torch.tensor([[[[5, 99, 1, 99, 3, 99], [4, 99, 2, 99, 0, 99]]]], dtype=torch.int32)
    selected = storage[..., ::2]
    first = builder.from_uniform_selected_blocks(selected)
    second = builder.from_uniform_selected_blocks(selected + 8)

    assert not selected.is_contiguous()
    assert first.block_indptr is second.block_indptr
    torch.testing.assert_close(first.block_indptr, torch.tensor([[[0, 3, 6]]], dtype=torch.int32))
    torch.testing.assert_close(
        first.block_indices, torch.tensor([1, 3, 5, 0, 2, 4], dtype=torch.int32)
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention_backend.sparse.block_sparse._is_current_stream_capturing",
        lambda device: True,
    )
    with pytest.raises(RuntimeError, match="cache miss during CUDA Graph capture"):
        builder.from_uniform_selected_blocks(torch.zeros((2, 1, 1, 1), dtype=torch.int32))


def test_pack_kv_token_mask_broadcasts_lsb_first() -> None:
    mask = torch.ones(35, dtype=torch.bool)
    mask[[0, 34]] = False

    packed = pack_kv_token_mask(mask, batch_size=2)

    assert packed.dtype == torch.uint32 and packed.is_contiguous()
    torch.testing.assert_close(
        packed,
        torch.tensor([[0xFFFFFFFE, 0b011], [0xFFFFFFFE, 0b011]], dtype=torch.uint32),
    )


def test_contiguous_runtime_reuses_plan_for_live_metadata_and_fails_capture_miss(
    monkeypatch,
) -> None:
    wrappers: list[Mock] = []

    def wrapper_factory() -> Mock:
        wrapper = Mock()

        def run(q, k, v, block_indptr, block_indices, *, out=None, **kwargs):
            del k, v, block_indptr, block_indices, kwargs
            target = q + 1
            if out is None:
                return target
            out.copy_(target)
            return out

        wrapper.run.side_effect = run
        wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr(block_sparse_fmha_module, "_BlockSparseTSWrapper", wrapper_factory)
    q, k, v = _qkv()
    routes_a = _raw_routes(torch.tensor([2, 0, 1, 3, 1, 0, 2, 3], dtype=torch.int32))
    routes_b = _raw_routes(torch.tensor([0, 1, 2, 3, 3, 0, 1, 2], dtype=torch.int32))
    mask_a = pack_kv_token_mask(torch.ones(256, dtype=torch.bool), batch_size=1)
    mask_b = mask_a.clone()
    mask_b[:, -1] = 0
    runtime = PrimsTSBlockSparseRuntime()

    for routes, mask in ((routes_a, mask_a), (routes_b, mask_b)):
        torch.testing.assert_close(
            runtime.run_contiguous(
                q,
                k,
                v,
                routes=routes,
                q_block_size=64,
                kv_block_size=64,
                kv_valid_bits=mask,
            ),
            q + 1,
        )

    assert len(wrappers) == 1
    wrapper = wrappers[0]
    wrapper.plan.assert_called_once()
    assert wrapper.run.call_count == 2
    for call, routes, mask in zip(
        wrapper.run.call_args_list, (routes_a, routes_b), (mask_a, mask_b), strict=True
    ):
        assert call.args[3] is routes.block_indptr
        assert call.args[4] is routes.block_indices
        assert call.kwargs["kv_valid_bits"] is mask

    monkeypatch.setattr(
        block_sparse_fmha_module, "_is_current_stream_capturing", lambda device: True
    )
    with pytest.raises(RuntimeError, match="plan cache miss during CUDA Graph capture"):
        PrimsTSBlockSparseRuntime().run_contiguous(
            q,
            k,
            v,
            routes=routes_a,
            q_block_size=64,
            kv_block_size=64,
            kv_valid_bits=mask_a,
        )


@pytest.mark.parametrize(
    ("value", "error"),
    [(True, TypeError), (-1, ValueError)],
)
def test_block_sparse_routes_reject_invalid_capacity(value, error) -> None:
    routes = _routes()
    with pytest.raises(error, match="max_blocks_per_row"):
        BlockSparseRoutes(routes.block_indptr, routes.block_indices, value)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("q_block_size", True, TypeError, "Python integer"),
        ("q_block_size", 0, ValueError, "positive"),
        ("kv_block_size", 64.0, TypeError, "Python integer"),
        ("kv_block_size", -1, ValueError, "positive"),
    ],
)
def test_block_sparse_params_reject_invalid_block_sizes(field, value, error, message) -> None:
    kwargs = {"q_block_size": 64, "kv_block_size": 64, field: value}
    with pytest.raises(error, match=field + ".*" + message):
        BlockSparseParams(**kwargs)


def test_contiguous_fmha_writes_directly_and_requires_uniform_fixed_q(monkeypatch) -> None:
    runtime = Mock()

    def run(q, k, v, *, out, **kwargs):
        del k, v, kwargs
        out.copy_(q + 1)
        return out

    runtime.run_contiguous.side_effect = run
    monkeypatch.setattr(block_sparse_fmha_module, "PrimsTSBlockSparseRuntime", lambda: runtime)
    attention = _Attention(BlockSparseParams(q_block_size=64, kv_block_size=64))
    fmha = PrimsTSBlockSparseFmha(attention)
    routes = _routes()
    q = torch.zeros((128, 2 * 128), dtype=torch.float16)
    k = torch.zeros((512, 128), dtype=torch.float16)
    v = torch.zeros_like(k)
    output = torch.empty_like(q)
    args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.context_only,
        attention_mask=PredefinedAttentionMask.FULL,
        is_fused_qkv=False,
        block_sparse_inputs=BlockSparseForwardInputs(routes=routes),
    )
    metadata = SimpleNamespace(
        is_cross=False,
        kv_cache_manager=None,
        seq_lens=torch.tensor([64, 64], dtype=torch.int32),
    )

    assert fmha.forward(q, k, v, metadata, args) is None
    call = runtime.run_contiguous.call_args
    assert call.args[0].shape == (2, 64, 2, 128)
    assert call.args[0].data_ptr() == q.data_ptr()
    assert call.args[1].shape == (2, 256, 1, 128)
    assert call.kwargs["routes"] is routes
    assert call.kwargs["out"].data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, torch.ones_like(output))

    metadata.seq_lens = torch.tensor([32, 96], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="uniform.*query"):
        fmha.forward(q, k, v, metadata, args)
    runtime.run_contiguous.assert_called_once()


def test_paged_runtime_reuses_plan_for_live_page_route_and_lengths(monkeypatch) -> None:
    wrappers: list[Mock] = []

    def wrapper_factory() -> Mock:
        wrapper = Mock()

        def run(q, *args, out=None, **kwargs):
            del args, kwargs
            if out is None:
                return q + 1
            out.copy_(q + 1)
            return out

        wrapper.run.side_effect = run
        wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr(block_sparse_fmha_module, "_BlockSparsePagedTSWrapper", wrapper_factory)
    runtime = PrimsTSBlockSparseRuntime()
    q = torch.zeros((2, 1, 2, 128), dtype=torch.float16)
    k_cache = torch.zeros((8, 1, 64, 128), dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    indptr = torch.tensor([0, 4, 8], dtype=torch.int32)
    page_indices = torch.arange(8, dtype=torch.int32)
    lengths = torch.tensor([129, 193], dtype=torch.int32)
    routes_a = _routes()
    routes_b = BlockSparseRoutes(
        block_indptr=routes_a.block_indptr,
        block_indices=torch.tensor([1, 3, 0, 2], dtype=torch.int32),
        max_blocks_per_row=2,
    )
    output = torch.empty_like(q)
    common = dict(
        q_block_size=64,
        kv_block_size=64,
        page_size=64,
        max_seq_len_kv=256,
        out=output,
    )

    runtime.run_paged(
        q,
        (k_cache, v_cache),
        paged_kv_indptr=indptr,
        paged_kv_indices=page_indices,
        seq_lens_kv=lengths,
        routes=routes_a,
        **common,
    )
    page_indices.copy_(page_indices.flip(0))
    lengths.copy_(torch.tensor([65, 255], dtype=torch.int32))
    runtime.run_paged(
        q,
        (k_cache, v_cache),
        paged_kv_indptr=indptr,
        paged_kv_indices=page_indices,
        seq_lens_kv=lengths,
        routes=routes_b,
        **common,
    )

    assert len(wrappers) == 1
    wrapper = wrappers[0]
    wrapper.plan.assert_called_once()
    plan_values = (*wrapper.plan.call_args.args, *wrapper.plan.call_args.kwargs.values())
    assert all(
        all(value is not live for value in plan_values)
        for live in (indptr, page_indices, lengths, routes_a.block_indptr, routes_a.block_indices)
    )
    assert wrapper.run.call_count == 2
    for call, routes in zip(wrapper.run.call_args_list, (routes_a, routes_b), strict=True):
        assert call.args[2] is indptr
        assert call.args[3] is page_indices
        assert call.args[4] is lengths
        assert call.args[5] is routes.block_indptr
        assert call.args[6] is routes.block_indices


def test_paged_fmha_fixed_q_v2_capacity_live_metadata_and_rope(monkeypatch) -> None:
    attention, fmha, runtime, q, metadata, args = _paged_case(monkeypatch, query_lengths=(2, 2))
    prepared_wrapper = Mock()
    runtime.ensure_paged_plan.return_value = prepared_wrapper
    metadata.max_seq_len = 193
    args.attention_window_size = 193
    metadata.kv_cache_manager.impl.get_page_index_upper_bound = Mock(return_value=8)

    routes = args.block_sparse_inputs.routes
    bits = torch.full((2, 8), 0xFFFFFFFF, dtype=torch.uint32)
    args.block_sparse_inputs = BlockSparseForwardInputs(routes=routes, kv_valid_bits=bits)
    q_processed = torch.zeros((4, 2, 128), dtype=torch.bfloat16)
    kv_pool = torch.empty((8, 1, 64, 128), dtype=torch.bfloat16)
    selected_tables = metadata.kv_cache_block_offsets[0]

    def preprocess_after_plan(*_args):
        runtime.ensure_paged_plan.assert_called_once()
        return (
            q_processed,
            kv_pool,
            selected_tables,
            None,
            1.0,
            1.0,
            torch.empty(64, dtype=torch.uint8),
            None,
            2,
            256,
            -1,
            False,
        )

    preprocess = Mock(side_effect=preprocess_after_plan)
    monkeypatch.setattr(prims_ts_fmha_module.thop, "trtllm_gen_generation_preprocess", preprocess)

    live_inv_freq = torch.tensor([3.0])
    live_cos_sin = torch.tensor([4.0])
    attention.rotary_inv_freq = live_inv_freq
    attention.rotary_cos_sin = live_cos_sin
    fmha.forward(q, None, None, metadata, args)

    preprocess.assert_called_once()
    assert preprocess.call_args.args[5] is metadata.kv_cache_block_offsets
    assert preprocess.call_args.args[11] is live_inv_freq
    assert preprocess.call_args.args[12] is live_cos_sin
    runtime.ensure_paged_plan.assert_called_once()
    plan = runtime.ensure_paged_plan.call_args.kwargs
    assert plan["seq_len_q"] == 2
    assert plan["max_seq_len_kv"] == 4 * 64
    runtime.run_paged.assert_not_called()
    prepared_wrapper.run.assert_called_once()
    run = prepared_wrapper.run.call_args
    assert run.args[4] is metadata.kv_lens_cuda_runtime
    assert run.args[5] is routes.block_indptr
    assert run.args[6] is routes.block_indices
    assert run.kwargs["kv_valid_bits"] is bits
    assert run.kwargs["out"].data_ptr() == args.output.data_ptr()


@pytest.mark.parametrize(
    ("case", "reason_fragment"),
    [
        ("varlen_q", "uniform"),
        ("mixed", "generation-only"),
        ("position", "position embedding type"),
        ("helix", "Helix"),
        ("softmax_stats", "softmax statistics"),
    ],
)
def test_paged_fmha_rejects_variable_q_and_unsupported_semantics(
    monkeypatch, case, reason_fragment
) -> None:
    position = (
        PositionEmbeddingType.alibi
        if case == "position"
        else PositionEmbeddingType.learned_absolute
    )
    _, fmha, runtime, q, metadata, args = _paged_case(
        monkeypatch,
        query_lengths=(1, 3) if case == "varlen_q" else (1, 1),
        position_embedding_type=position,
    )
    if case == "mixed":
        args.attention_input_type = AttentionInputType.mixed
    elif case == "helix":
        metadata.helix_position_offsets = torch.ones(1)
    elif case == "softmax_stats":
        args.softmax_stats_tensor = torch.ones(1)

    preprocess = Mock()
    monkeypatch.setattr(
        prims_ts_fmha_module.thop,
        "trtllm_gen_generation_preprocess",
        preprocess,
    )

    with pytest.raises(RuntimeError, match=reason_fragment):
        fmha.forward(q, None, None, metadata, args)

    preprocess.assert_not_called()
    runtime.ensure_paged_plan.assert_not_called()


@_REQUIRES_PRIMTS_GPU
@torch.no_grad()
def test_real_gpu_raw_routes_and_token_mask_match_reference() -> None:
    torch.manual_seed(1234)
    q = torch.randn((1, 128, 1, 128), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 256, 1, 128), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    routes = BlockSparseRoutes(
        block_indptr=torch.tensor([[[0, 2, 5]]], device="cuda", dtype=torch.int32),
        block_indices=torch.tensor([0, 2, 0, 1, 3], device="cuda", dtype=torch.int32),
        max_blocks_per_row=3,
    )
    token_mask = torch.ones(256, device="cuda", dtype=torch.bool)
    token_mask[[1, 63, 64, 95, 129, 190, 255]] = False
    valid_bits = pack_kv_token_mask(token_mask, batch_size=1)
    sm_scale = 128**-0.5

    key_blocks = torch.arange(256, device="cuda") // 64
    allowed = torch.zeros((128, 256), device="cuda", dtype=torch.bool)
    for row, selected_blocks in enumerate(((0, 2), (0, 1, 3))):
        selected = torch.tensor(selected_blocks, device="cuda")
        allowed[row * 64 : (row + 1) * 64] = torch.isin(key_blocks, selected) & token_mask
    scores = (q[0, :, 0].float() @ k[0, :, 0].float().T) * sm_scale
    probabilities = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
    expected = (probabilities @ v[0, :, 0].float()).to(q.dtype)[None, :, None, :]

    actual = PrimsTSBlockSparseRuntime().run_contiguous(
        q,
        k,
        v,
        routes=routes,
        q_block_size=64,
        kv_block_size=64,
        kv_valid_bits=valid_bits,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@_REQUIRES_PRIMTS_GPU
@torch.no_grad()
def test_real_gpu_paged_routes_match_reference() -> None:
    torch.manual_seed(7)
    q = torch.randn((1, 64, 1, 128), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((4, 1, 64, 128), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    page_indices = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([256], device="cuda", dtype=torch.int32)
    routes = BlockSparseRoutes(
        block_indptr=torch.tensor([[[0, 2]]], device="cuda", dtype=torch.int32),
        block_indices=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        max_blocks_per_row=2,
    )
    sm_scale = 128**-0.5

    actual = PrimsTSBlockSparseRuntime().run_paged(
        q,
        (k_cache, v_cache),
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=page_indices,
        seq_lens_kv=seq_lens_kv,
        routes=routes,
        q_block_size=64,
        kv_block_size=64,
        page_size=64,
        max_seq_len_kv=256,
        sm_scale=sm_scale,
    )

    logical_k = k_cache.index_select(0, page_indices.long()).reshape(256, 1, 128)
    logical_v = v_cache.index_select(0, page_indices.long()).reshape(256, 1, 128)
    allowed = torch.zeros(256, device="cuda", dtype=torch.bool)
    allowed[:64] = True
    allowed[128:192] = True
    scores = (q[0, :, 0].float() @ logical_k[:, 0].float().T) * sm_scale
    expected = (
        torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1) @ logical_v[:, 0].float()
    ).to(q.dtype)[None, :, None, :]
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
