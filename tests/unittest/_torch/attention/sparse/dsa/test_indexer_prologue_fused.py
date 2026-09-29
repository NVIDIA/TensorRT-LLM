# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused DSA indexer prologue (TRTLLM_DSA_INDEXER_FUSED_PROLOGUE) vs the per-kernel chain.

One Triton launch replaces k_norm + flashinfer RoPE + fused_cat_fp8 (Q, K) + the
weight scaling; its five outputs must be bit-identical to the chain's, eagerly and
under CUDA-graph replay. fused_cat_fp8/fp4 read the token rows at stride
n_heads * head_dim, so the chain densifies a column-sliced Q first;
test_unfused_strided_q_matches_contiguous covers that precondition.
"""

import dataclasses

import pytest
import torch
from utils.util import skip_pre_blackwell, skip_pre_hopper

from tensorrt_llm._torch.attention.backends.interface import (
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import Indexer
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

H, D, R = 32, 128, 64

# FP8 E4M3 conversion in Triton needs sm90+.
pytestmark = skip_pre_hopper


@pytest.fixture(autouse=True)
def _fused_gate_on(monkeypatch):
    # The Indexer reads the gate at construction; force the fused path on.
    monkeypatch.setenv("TRTLLM_DSA_INDEXER_FUSED_PROLOGUE", "1")
    # The model forward runs under inference_mode (in-place RoPE on split views).
    with torch.inference_mode():
        yield


def _make_indexer(interleave, ln_dtype, indexer_k_dtype="fp8"):
    cfg = DeepSeekSparseAttentionConfig(
        index_head_dim=D, index_n_heads=H, index_topk=2048, indexer_k_dtype=indexer_k_dtype
    )
    sp = dataclasses.replace(cfg.to_sparse_params(), indexer_rope_interleave=interleave)
    rope = RopeParams(dim=R, theta=8000000.0, max_positions=8192)
    pe = PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox, rope=rope, is_neox=not interleave
    )
    mla = MLAParams(hidden_size=6144, q_lora_rank=2048, qk_rope_head_dim=R)
    idx = Indexer(
        None, pe, mla, True, sp, torch.bfloat16, layer_idx=0, aux_stream=torch.cuda.Stream()
    )
    idx.k_norm = idx.k_norm.cuda().to(ln_dtype)
    with torch.no_grad():
        idx.k_norm.weight.copy_(1.0 + 0.2 * torch.randn(D))
        idx.k_norm.bias.copy_(0.1 * torch.randn(D))
    return idx


def _positions(bs, next_n):
    """``next_n`` consecutive positions per request, each from a random context length."""
    ctx = torch.randint(80, 900, (bs,))
    pos = torch.cat([torch.arange(c, c + next_n) for c in ctx.tolist()])
    return pos.to(torch.int32).cuda().view(1, -1)


def _inputs(n, k_dtype):
    """Fused ``wk | weights_proj`` output, Q, and the indexer_k / weights views of the former."""
    fused = torch.randn(n, D + H, device="cuda").to(k_dtype)
    fused[:, :D] *= 3.0
    indexer_k, weights = fused.split([D, H], dim=-1)
    q = (torch.randn(n, H * D, device="cuda") * 2.0).to(torch.bfloat16)
    return fused, q, indexer_k, weights


def _run_unfused(idx, indexer_k, weights, q_in, pos):
    """The per-kernel chain with ``q_in`` standing in for ``wq_b(qr)`` (built weight-less here)."""
    idx._modules.pop("wq_b", None)
    idx.wq_b = lambda qr: q_in
    return idx._prologue_unfused(indexer_k, weights, None, pos)


def _run_fused(idx, indexer_k, weights, q_in, pos):
    return idx._prologue_fused(indexer_k, weights, q_in, pos)


def _bits(t):
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def _assert_bitwise_equal(ref, got):
    assert len(ref) == len(got) == 5
    for a, b in zip(ref, got):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(_bits(a), _bits(b))


@pytest.mark.parametrize("bs,next_n", [(1, 6), (1, 1), (2, 6), (4, 6), (16, 1)])
@pytest.mark.parametrize("interleave", [True, False])
@pytest.mark.parametrize("ue8m0", [True, False])
@pytest.mark.parametrize(
    "ln_dtype,k_dtype", [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)]
)
def test_fused_prologue_bitwise(bs, next_n, interleave, ue8m0, ln_dtype, k_dtype):
    torch.manual_seed(1000 * bs + next_n)
    idx = _make_indexer(interleave, ln_dtype)
    assert idx._use_fused_prologue
    idx.scale_fmt = "ue8m0" if ue8m0 else "none"
    pos = _positions(bs, next_n)
    _, q, indexer_k, weights = _inputs(bs * next_n, k_dtype)
    # The chain applies RoPE in place: give it its own copy of Q.
    ref = _run_unfused(idx, indexer_k, weights, q.clone(), pos)
    got = _run_fused(idx, indexer_k, weights, q, pos)
    _assert_bitwise_equal(ref, got)


def test_fused_prologue_cuda_graph_replay():
    torch.manual_seed(5)
    idx = _make_indexer(True, torch.float32)
    pos = _positions(1, 6)
    fused, q, indexer_k, weights = _inputs(6, torch.bfloat16)
    for _ in range(2):
        _run_fused(idx, indexer_k, weights, q, pos)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = _run_fused(idx, indexer_k, weights, q, pos)
    for _ in range(3):
        fused.copy_(torch.randn_like(fused, dtype=torch.float32).to(fused.dtype))
        new_q = (torch.randn_like(q, dtype=torch.float32) * 2).to(q.dtype)
        q.copy_(new_q)
        graph.replay()
        torch.cuda.synchronize()
        got = [t.clone() for t in outs]
        _assert_bitwise_equal(_run_unfused(idx, indexer_k, weights, new_q, pos), got)


def test_unfused_strided_q_matches_contiguous():
    """fused_cat_fp8 precondition: the chain densifies a column-sliced Q.

    fused_cat_fp8 reads the token rows at stride n_heads * head_dim. Without the
    densify in Indexer._qk_projection_and_rope, a Q whose token stride is wider
    (here a column slice of a [n, 2048 + n_heads * head_dim] tensor) quantizes
    the wrong memory for every token after the first.
    """
    torch.manual_seed(7)
    idx = _make_indexer(True, torch.float32)
    pos = _positions(4, 6)
    _, _, indexer_k, weights = _inputs(24, torch.bfloat16)
    wide = (torch.randn(24, 2048 + H * D, device="cuda") * 2.0).to(torch.bfloat16)
    assert not wide[:, 2048:].is_contiguous()
    ref = _run_unfused(idx, indexer_k, weights, wide[:, 2048:].contiguous(), pos)
    _assert_bitwise_equal(ref, _run_unfused(idx, indexer_k, weights, wide.clone()[:, 2048:], pos))
    # The fused kernel takes the token stride from the tensor itself.
    _assert_bitwise_equal(ref, _run_fused(idx, indexer_k, weights, wide[:, 2048:], pos))


@skip_pre_blackwell
def test_fp4_indexer_keeps_unfused_chain():
    """The fused kernel is FP8-only: an FP4 indexer is not eligible."""
    idx = _make_indexer(True, torch.float32, indexer_k_dtype="fp4")
    assert idx.use_fp4 and not idx._use_fused_prologue


def test_gate_off_keeps_unfused_chain(monkeypatch):
    monkeypatch.setenv("TRTLLM_DSA_INDEXER_FUSED_PROLOGUE", "0")
    assert not _make_indexer(True, torch.float32)._use_fused_prologue
