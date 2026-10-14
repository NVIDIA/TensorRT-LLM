# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused DSA indexer prologue (TRTLLM_DSA_INDEXER_FUSED_PROLOGUE) vs the per-kernel chain.

The fused Triton kernel replaces k_norm + flashinfer RoPE + fused_cat_fp8 (Q, K) +
weight scaling + indexer_k_cache_scatter_op. Outputs and the whole indexer K cache
must be bit-identical for contiguous Q tensors and column slices with gaps
between tokens. Both the unfused chain and fused kernel preserve those strides.
"""

import dataclasses

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.backends.sparse.dsa import Indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa import kernels as dsa_kernels
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm.functional import PositionEmbeddingType

from .test_dsa_indexer import _create_mock_metadata, create_dsa_cache_manager

H, D, R, TPB = 32, 128, 64, 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 E4M3 Triton conversion needs sm90+",
)


@pytest.fixture(autouse=True)
def _inference_mode(monkeypatch):
    # Exercise the fused path regardless of TRTLLM_DSA_INDEXER_FUSED_PROLOGUE.
    monkeypatch.setattr(dsa_kernels, "FUSED_PROLOGUE_ENABLED", True)
    # The model forward runs under inference_mode (in-place RoPE on split views).
    with torch.inference_mode():
        yield


def _make_indexer(cfg, interleave, ln_dtype):
    rope = RopeParams(dim=R, theta=8000000.0, max_positions=8192)
    pe = PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox, rope=rope, is_neox=not interleave
    )

    class _MLAParams:
        hidden_size = 6144
        q_lora_rank = 2048
        qk_rope_head_dim = R

    sp = cfg.to_sparse_params()
    sp = dataclasses.replace(sp, indexer_rope_interleave=interleave)
    idx = Indexer(
        None,
        pe,
        _MLAParams(),
        True,
        sp,
        torch.bfloat16,
        layer_idx=0,
        aux_stream=torch.cuda.Stream(),
    )
    idx.k_norm = idx.k_norm.cuda().to(ln_dtype)
    with torch.no_grad():
        idx.k_norm.weight.copy_(1.0 + 0.2 * torch.randn(D))
        idx.k_norm.bias.copy_(0.1 * torch.randn(D))
    return idx


def _setup(bs, next_n, interleave, ln_dtype):
    ctx = torch.randint(80, 900, (bs,), dtype=torch.int32)
    final = ctx + next_n
    cm, cfg = create_dsa_cache_manager(
        batch_size=bs, head_dim=D, tokens_per_block=TPB, max_seq_len=1024 + 64
    )
    rids = list(range(bs))
    cm.add_dummy_requests(
        request_ids=rids, token_nums=final.tolist(), is_gen=False, prepare_resource=True
    )
    n = bs * next_n
    md = _create_mock_metadata(
        rids,
        bs,
        num_contexts=0,
        num_generations=bs,
        seq_lens=torch.tensor([next_n] * bs, dtype=torch.int32),
        kv_lens=final.clone(),
        num_cached_tokens=ctx.tolist(),
        cache_manager=cm,
        num_ctx_tokens=0,
        num_tokens=n,
        max_draft_tokens=next_n - 1,
        use_cute_dsl_paged_mqa_logits=True,
        indexer_head_dim=D,
    )
    Indexer.prepare(md)
    idx = _make_indexer(cfg, interleave, ln_dtype)
    pos = torch.cat([torch.arange(c, c + next_n) for c in ctx.tolist()]).to(torch.int32).cuda()
    return cm, md, idx, pos.view(1, -1), n


def _inputs(n, k_dtype):
    fused = torch.randn(n, D + H, device="cuda").to(k_dtype)
    fused[:, :D] *= 3.0
    indexer_k, weights = fused.split([D, H], dim=-1)
    wide = (torch.randn(n, 2048 + H * D, device="cuda") * 2.0).to(torch.bfloat16)
    return fused, wide, indexer_k, weights


def _run_old(idx, md, indexer_k, weights, q_in, pos):
    out = idx._prologue_chain(indexer_k, weights, q_in, pos, torch.bfloat16)
    idx._update_k_cache(out[1][: md.num_tokens], out[2][: md.num_tokens], md)
    return out


def _run_fused(idx, md, indexer_k, weights, q_in, pos):
    out = idx._prologue_fused(indexer_k, weights, q_in, pos, md)
    assert idx._k_cache_prewritten_for is out[1]
    idx._k_cache_prewritten_for = None
    return out


def _bits(t):
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


@pytest.mark.parametrize("bs,next_n", [(1, 6), (1, 1), (2, 6), (4, 6), (16, 1)])
@pytest.mark.parametrize("interleave", [True, False])
@pytest.mark.parametrize("ue8m0", [True, False])
@pytest.mark.parametrize(
    "ln_dtype,k_dtype", [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)]
)
def test_fused_prologue_bitwise(bs, next_n, interleave, ue8m0, ln_dtype, k_dtype):
    torch.manual_seed(1000 * bs + next_n)
    cm, md, idx, pos, n = _setup(bs, next_n, interleave, ln_dtype)
    idx.scale_fmt = "ue8m0" if ue8m0 else "none"
    try:
        kc = cm.get_indexer_k_cache_buffers(0).view(torch.uint8)
        _, wide, indexer_k, weights = _inputs(n, k_dtype)
        assert idx._fused_prologue_eligible(wide[:, 2048:], indexer_k, weights, torch.bfloat16)
        kc.fill_(0xA5)
        # reference: the original chain on a contiguous Q copy (see
        # test_old_chain_strided_q_reads_wrong_rows for the strided layout)
        ref = _run_old(idx, md, indexer_k, weights, wide.clone()[:, 2048:].contiguous(), pos)
        cache_ref = kc.clone()
        kc.fill_(0xA5)
        got = _run_fused(idx, md, indexer_k, weights, wide[:, 2048:], pos)
        for a, b in zip(ref, got):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert torch.equal(_bits(a), _bits(b))
        assert torch.equal(cache_ref, kc)
        assert int((kc != 0xA5).sum()) > 0
    finally:
        cm.shutdown()


def test_fused_prologue_cuda_graph_replay():
    torch.manual_seed(5)
    cm, md, idx, pos, n = _setup(1, 6, True, torch.float32)
    try:
        kc = cm.get_indexer_k_cache_buffers(0).view(torch.uint8)
        fused, wide, indexer_k, weights = _inputs(n, torch.bfloat16)
        for _ in range(2):
            _run_fused(idx, md, indexer_k, weights, wide[:, 2048:], pos)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), torch.cuda.graph(graph):
            outs = _run_fused(idx, md, indexer_k, weights, wide[:, 2048:], pos)
        for _ in range(3):
            fused.copy_(torch.randn_like(fused, dtype=torch.float32).to(fused.dtype))
            new_wide = (torch.randn_like(wide, dtype=torch.float32) * 2).to(wide.dtype)
            wide.copy_(new_wide)
            kc.fill_(0x5A)
            graph.replay()
            torch.cuda.synchronize()
            got = [t.clone() for t in outs] + [kc.clone()]
            kc.fill_(0x5A)
            ref = list(
                _run_old(idx, md, indexer_k, weights, new_wide.clone()[:, 2048:].contiguous(), pos)
            )
            ref.append(kc.clone())
            for a, b in zip(ref, got):
                assert torch.equal(_bits(a), _bits(b))
    finally:
        cm.shutdown()


def test_prewritten_handoff_skips_scatter_once():
    idx = Indexer.__new__(Indexer)
    torch.nn.Module.__init__(idx)
    calls = []
    idx._update_k_cache = lambda *a: calls.append(a)
    k_full = torch.empty(4, 128, dtype=torch.float8_e4m3fn)
    idx._k_cache_prewritten_for = k_full
    idx.update_k_cache_unless_prewritten(k_full, k_full[:2], None, None)
    assert calls == [] and idx._k_cache_prewritten_for is None
    idx.update_k_cache_unless_prewritten(k_full, k_full[:2], None, None)
    assert len(calls) == 1
    idx._k_cache_prewritten_for = torch.empty(1)
    idx.update_k_cache_unless_prewritten(k_full, k_full[:2], None, None)
    assert len(calls) == 2


def test_strided_q_matches_contiguous_reference():
    """Fused and unfused projections must handle gaps between Q tokens."""
    torch.manual_seed(11)
    cm, md, idx, pos, n = _setup(1, 6, True, torch.float32)
    try:
        _, wide, indexer_k, weights = _inputs(n, torch.bfloat16)
        strided = _run_old(idx, md, indexer_k, weights, wide.clone()[:, 2048:], pos)
        contig = _run_old(idx, md, indexer_k, weights, wide.clone()[:, 2048:].contiguous(), pos)
        fused = _run_fused(idx, md, indexer_k, weights, wide[:, 2048:], pos)
        for strided_tensor, contiguous_tensor, fused_tensor in zip(strided, contig, fused):
            assert torch.equal(_bits(strided_tensor), _bits(contiguous_tensor))
            assert torch.equal(_bits(fused_tensor), _bits(contiguous_tensor))
    finally:
        cm.shutdown()
