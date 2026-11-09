# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call-owned MiniMax-M3 combine buffers and safe BF16 fallback contracts."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import msa_combine_mxfp8 as m
from tensorrt_llm._torch.attention.backends.fmha import msa_prefill
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
    MiniMaxM3BackendForwardArgs,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_utils
from tensorrt_llm._torch.modules import linear
from tensorrt_llm._torch.utils import MxFp8QuantizedTensor

pytestmark = pytest.mark.cpu_only


class _Sparse:
    def sparse_atten_nvfp4_kv_func(self, q, *, out=None, o_fp8=None, sf_out=None, write_bf16=True):
        pass


class _OldSparse:
    def sparse_atten_nvfp4_kv_func(self, q, *, out=None):
        pass


def _out(rows: int = 200, heads: int = 32) -> torch.Tensor:
    return torch.zeros((rows, heads, 128), dtype=torch.bfloat16)


def _metadata(generations: int = 0) -> Mock:
    metadata = Mock(spec_set=["num_generations"])
    metadata.num_generations = generations
    return metadata


def test_engages_only_for_explicit_pure_context_in_place_and_supported(monkeypatch) -> None:
    monkeypatch.setattr(m, "ENABLED", True)
    out, metadata = _out(), _metadata()
    request = MiniMaxM3BackendForwardArgs(mxfp8_output=True)
    assert m.combine_kwargs(_Sparse(), out, metadata, in_place=True) is None
    kw = m.combine_kwargs(_Sparse(), out, metadata, in_place=True, request=request)
    assert kw is not None and kw["write_bf16"] is True
    assert kw["o_fp8"].shape == (200, 32 * 128) and kw["o_fp8"].dtype == torch.float8_e4m3fn
    assert kw["sf_out"].numel() == 256 * 32 * 128 // 32
    assert int(kw["sf_out"][-128 * 32 * 128 // 32 :].sum()) == 0
    for sparse, output, meta, in_place in (
        (_Sparse(), out, _metadata(3), True),
        (_Sparse(), out, metadata, False),
        (_OldSparse(), out, metadata, True),
        (_Sparse(), out.view(200, -1), metadata, True),
        (_Sparse(), _out().float(), metadata, True),
        (_Sparse(), out.transpose(0, 1), metadata, True),
    ):
        assert m.combine_kwargs(sparse, output, meta, in_place, request) is None
    assert m.combine_kwargs(_Sparse(), out, metadata, in_place=True) is None


@pytest.mark.parametrize(
    "enabled,requested,eager", [(False, True, True), (True, False, True), (True, True, False)]
)
def test_disabled_unrequested_or_compiling_declines(monkeypatch, enabled, requested, eager) -> None:
    monkeypatch.setattr(m, "ENABLED", enabled)
    monkeypatch.setattr(linear, "ctx_eager_only", lambda: eager)
    request = MiniMaxM3BackendForwardArgs(mxfp8_output=requested)
    assert m.combine_kwargs(_Sparse(), _out(), _metadata(), True, request) is None


@pytest.mark.parametrize(
    "skip,proven,write", [(False, True, True), (True, False, True), (True, True, False)]
)
def test_bf16_skip_only_when_enabled_and_proven(monkeypatch, skip, proven, write) -> None:
    monkeypatch.setattr(m, "ENABLED", True)
    monkeypatch.setattr(m, "SKIP_BF16", skip)
    request = MiniMaxM3BackendForwardArgs(mxfp8_output=True, mxfp8_skip_bf16=proven)
    out = _out()
    kwargs = m.combine_kwargs(_Sparse(), out, _metadata(), True, request)
    assert kwargs["write_bf16"] is write
    result = m.quantized_output(out, kwargs)
    act, sf = result.quantized()
    assert act is kwargs["o_fp8"] and sf is kwargs["sf_out"]
    assert result.is_sf_swizzled
    if write:
        assert result.unquantized().data_ptr() == out.data_ptr()
        assert result.unquantized().shape == act.shape
    else:
        with pytest.raises(ValueError, match="no unquantized hidden-state view"):
            result.unquantized()


def test_interleaved_requests_retain_their_own_payloads(monkeypatch) -> None:
    monkeypatch.setattr(m, "ENABLED", True)
    requests = [MiniMaxM3BackendForwardArgs(mxfp8_output=True) for _ in range(2)]
    payloads = []
    for request, rows in zip(requests, (200, 128)):
        out = _out(rows)
        kwargs = m.combine_kwargs(_Sparse(), out, _metadata(), True, request)
        request.mxfp8_result = m.quantized_output(out, kwargs)
        payloads.append((kwargs["o_fp8"], kwargs["sf_out"]))
    for request, (act, sf) in zip(reversed(requests), reversed(payloads)):
        first, second = request.mxfp8_result.quantized()
        assert first is act and second is sf
    assert requests[0].mxfp8_result is not requests[1].mxfp8_result
    assert requests[0].mxfp8_result.quantized()[0] is payloads[0][0]


@pytest.mark.parametrize(
    "use_cutlass,decode_graph,expected",
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_projection_capability_requires_a_consuming_method(
    use_cutlass, decode_graph, expected
) -> None:
    method = object.__new__(linear.MXFP8LinearMethod)
    method.use_cutlass = use_cutlass
    projection = Mock(spec_set=["quant_method"])
    projection.quant_method = method
    with linear.flashinfer_mxfp8_decode_graph_capture() if decode_graph else nullcontext():
        assert m.o_proj_accepts_mxfp8(projection) is expected
    projection.quant_method = object()
    assert not m.o_proj_accepts_mxfp8(projection)


def test_carrier_row_splits_retain_bf16_views_and_reject_swizzled_scales() -> None:
    hidden = torch.arange(128, dtype=torch.float32).reshape(4, 32).to(torch.bfloat16)
    act = hidden.to(torch.float8_e4m3fn)
    scales = torch.ones(4, 1, dtype=torch.uint8)
    carrier = MxFp8QuantizedTensor(act, scales, unquantized_hidden_states=hidden)
    chunks = carrier.split([1, 3])
    assert [chunk.shape[0] for chunk in chunks] == [1, 3]
    for chunk, expected in zip(chunks, hidden.split([1, 3])):
        assert chunk.unquantized().data_ptr() == expected.data_ptr()
        torch.testing.assert_close(chunk.unquantized(), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="token dimension"):
        carrier.split(1, dim=1)
    carrier.is_sf_swizzled = True
    with pytest.raises(ValueError, match="row-major"):
        carrier.split(1)


def test_fmha_publishes_quantized_result_and_clears_a_declined_reuse(monkeypatch) -> None:
    monkeypatch.setattr(m, "ENABLED", True)
    monkeypatch.setattr(m, "SKIP_BF16", True)
    monkeypatch.setattr(msa_prefill, "_SM107_NVFP4_SPARSE_OUT_IN_PLACE", True)
    monkeypatch.setattr(msa_prefill, "get_sm_version", lambda: 107)
    monkeypatch.setattr(msa_prefill._nvfp4_sparse_takes_out, "_cached", None, raising=False)
    calls = []

    class SparseKernel:
        def build_k2q_csr(self, *args, **kwargs):
            raise AssertionError("prepared identity CSR should be reused")

        def sparse_atten_nvfp4_kv_func(
            self, q, *args, out=None, o_fp8=None, sf_out=None, write_bf16=True, **kwargs
        ):
            calls.append((o_fp8, sf_out, write_bf16))
            if o_fp8 is not None:
                o_fp8.fill_(1)
                sf_out.fill_(127)
            if write_bf16:
                out.fill_(3)
            return out

    library = Mock(spec_set=["sparse"])
    library.sparse = SparseKernel()
    monkeypatch.setattr(msa_utils, "require_msa_module", lambda: library)
    manager = Mock(spec_set=["get_buffers", "get_block_scale_buffers", "is_nvfp4_layer"])
    manager.get_buffers.return_value = torch.zeros((1, 2, 1, 128, 64), dtype=torch.uint8)
    manager.get_block_scale_buffers.return_value = torch.zeros((1, 2, 1, 128, 8), dtype=torch.uint8)
    manager.is_nvfp4_layer.return_value = True
    metadata = Mock(
        spec_set=[
            "kv_cache_manager",
            "num_contexts",
            "num_generations",
            "_msa_live_batch",
            "_msa_max_q_len",
            "_msa_max_kv_len_all",
            "_msa_total_k",
            "_msa_total_k_rows",
            "_msa_context_prefix_bounds",
            "msa_cu_q_lens",
            "msa_cu_kv_lens",
            "_msa_prefill_identity",
            "_msa_identity_csr",
            "msa_block_table",
            "msa_seq_lens_cuda",
            "msa_prefill_gqa_plan",
        ]
    )
    metadata.kv_cache_manager = manager
    metadata.num_contexts, metadata.num_generations = 1, 0
    metadata._msa_live_batch = 1
    metadata._msa_max_q_len = metadata._msa_max_kv_len_all = metadata._msa_total_k = 2
    metadata._msa_total_k_rows = 1
    metadata._msa_context_prefix_bounds = (2, 2, 2, 1)
    metadata.msa_cu_q_lens = metadata.msa_cu_kv_lens = torch.tensor([0, 2], dtype=torch.int32)
    metadata._msa_prefill_identity = object()
    metadata._msa_identity_csr = (
        metadata._msa_prefill_identity,
        (1, 2),
        (torch.tensor([0, 2], dtype=torch.int32), torch.tensor([0, 1], dtype=torch.int32), None),
    )
    metadata.msa_block_table = torch.zeros((1, 1), dtype=torch.int32)
    metadata.msa_seq_lens_cuda = torch.tensor([2], dtype=torch.int32)
    metadata.msa_prefill_gqa_plan = None
    attn = Mock(
        spec_set=[
            "head_dim",
            "num_heads",
            "layer_idx",
            "q_scaling",
            "_msa_nvfp4_dequant_scales",
            "_msa_nvfp4_dequant_scale_source_ptr",
        ]
    )
    attn.head_dim, attn.num_heads, attn.layer_idx, attn.q_scaling = 128, 1, 3, 1.0
    attn._msa_nvfp4_dequant_scales = attn._msa_nvfp4_dequant_scale_source_ptr = None
    request = MiniMaxM3BackendForwardArgs(mxfp8_output=True, mxfp8_skip_bf16=True)
    output = torch.full((2, 128), float("nan"), dtype=torch.bfloat16)
    forward = AttentionForwardArgs(
        output=output,
        sparse_backend_args=request,
        kv_scale_quant_orig=torch.ones(3),
    )
    forward.sparse_runtime_params.sparse_attn_indices = torch.zeros(
        (2, 1, msa_utils.MSA_REQUIRED_TOPK), dtype=torch.int32
    )
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=forward,
        workspace=torch.empty(0),
        query_input=torch.zeros((2, 128), dtype=torch.bfloat16),
        output=output,
        num_tokens=2,
    )
    # All three forwarding/publication functions execute; only the external MSA library is replaced.
    fmha = object.__new__(msa_prefill.MsaPrefillFmha)
    fmha.run_context(params)
    result = request.mxfp8_result
    assert isinstance(result, MxFp8QuantizedTensor)
    assert result.fp8_tensor is calls[0][0] and result.scaling_factor is calls[0][1]
    assert calls[0][2] is False and result.unquantized_hidden_states is None
    assert output.isnan().all(), "the skipped BF16 buffer must not be used as a fallback"
    monkeypatch.setattr(m, "ENABLED", False)
    fmha.run_context(params)
    assert request.mxfp8_result is None, "a declined call must clear the earlier result"
    assert calls[1] == (None, None, True)
    torch.testing.assert_close(output, torch.full_like(output, 3), rtol=0, atol=0)
