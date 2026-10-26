# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""catalog/spec_decode/dspark_qk_norm_rope: the drafter block's q / k RMSNorm + partial NeoX RoPE in one launch is bit
for bit the stock lines of DFlashForCausalLM.dflash_forward's flashinfer-RoPE branch (q_norm, k_norm, then
flashinfer_apply_rope_with_cos_sin_cache_inplace), at the M3.1 TP8 and TP4 rank head counts, 16 seeds, rows 3..56,
real-scale activations with tiny / huge / all-zero heads, eager and in a CUDA graph. DRAFT_APPEND_FUSE
(``qk_norm_rope_append``): the same launch's K / V pages are bit for bit flashinfer.page.append_paged_kv_cache's
(HND), with blocks crossing page boundaries, slots re-written, int32 / int64 indices, eager and in a CUDA graph."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.dspark_qk_norm_rope import (
    append_supported,
    qk_norm_rope,
    qk_norm_rope_append,
)
from tensorrt_llm._torch.custom_ops import (
    flashinfer_apply_rope_with_cos_sin_cache_inplace as fi_rope,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")

HD, ROT, EPS, MAX_POS = 128, 64, 1e-6, 1 << 20


def _cache():
    inv = 1.0 / (1e7 ** (torch.arange(0, ROT, 2, device="cuda", dtype=torch.float32) / ROT))
    ang = torch.arange(MAX_POS, device="cuda", dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=1).contiguous()


_CS = None


def _cs():
    global _CS
    if _CS is None:
        _CS = _cache()
    return _CS


def _norms(seed):
    g = torch.Generator(device="cuda").manual_seed(1000 + seed)
    qn = RMSNorm(hidden_size=HD, eps=EPS, dtype=torch.bfloat16, device="cuda")
    kn = RMSNorm(hidden_size=HD, eps=EPS, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        qn.weight.copy_((torch.randn(HD, generator=g, device="cuda") * 0.3 + 1).bfloat16())
        kn.weight.copy_((torch.randn(HD, generator=g, device="cuda") * 0.3 + 1).bfloat16())
    return qn, kn


def _qkv(T, hq, hkv, seed):
    """Real-scale qkv rows: per-head scales spanning 1e-3..3e2, some all-zero heads, a few outlier channels."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    width = (hq + 2 * hkv) * HD
    x = torch.randn(T, hq + 2 * hkv, HD, generator=g, device="cuda")
    scale = torch.exp(
        torch.empty(T, hq + 2 * hkv, 1, device="cuda").uniform_(-7.0, 5.7, generator=g)
    )
    x = x * scale
    x[:, :, :4] *= 20.0  # outlier channels
    if T > 1:
        x[1, 0] = 0.0  # an all-zero q head
        x[T - 1, hq] = 0.0  # an all-zero k head
    pos = torch.randint(0, MAX_POS, (T,), generator=g, device="cuda", dtype=torch.int32)
    return x.reshape(T, width).bfloat16(), pos


def _stock(qkv, hq, hkv, qn, kn, pos):
    qs, ks = hq * HD, hkv * HD
    q = qn(qkv[:, :qs].reshape(-1, HD)).view(-1, qs)
    k = kn(qkv[:, qs : qs + ks].reshape(-1, HD)).view(-1, ks)
    fi_rope(pos, q, k, HD, _cs(), True)
    return q, k


@pytest.mark.parametrize("hq,hkv", [(8, 1), (16, 1)], ids=["tp8", "tp4"])
def test_bitwise_to_the_stock_lines(hq, hkv):
    worst = 0
    for seed in range(16):
        qn, kn = _norms(seed)
        for T in (3, 4, 7, 8, 12, 24, 28, 32, 56):
            qkv, pos = _qkv(T, hq, hkv, 100 * seed + T)
            rq, rk = _stock(qkv, hq, hkv, qn, kn, pos)
            got = qk_norm_rope(qkv, hq * HD, hkv * HD, HD, qn, kn, pos, _cs(), True)
            assert got is not None
            torch.cuda.synchronize()
            bad = int((got[0] != rq).sum() + (got[1] != rk).sum())
            worst = max(worst, bad)
            assert bad == 0, (seed, T, bad)
    assert worst == 0


def test_in_a_cuda_graph():
    hq, hkv = 8, 1
    qn, kn = _norms(0)
    qkv, pos = _qkv(32, hq, hkv, 1)
    qk_norm_rope(qkv, hq * HD, hkv * HD, HD, qn, kn, pos, _cs(), True)  # compile outside capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            got = qk_norm_rope(qkv, hq * HD, hkv * HD, HD, qn, kn, pos, _cs(), True)
    torch.cuda.current_stream().wait_stream(s)
    q2, p2 = _qkv(32, hq, hkv, 2)
    qkv.copy_(q2), pos.copy_(p2)
    g.replay()
    rq, rk = _stock(qkv, hq, hkv, qn, kn, pos)
    torch.cuda.synchronize()
    assert torch.equal(got[0], rq) and torch.equal(got[1], rk)


def test_unsupported_inputs_keep_the_stock_path():
    hq, hkv = 8, 1
    qn, kn = _norms(0)
    qkv, pos = _qkv(4, hq, hkv, 3)
    assert (
        qk_norm_rope(qkv, hq * HD, hkv * HD, HD, qn, kn, pos, _cs(), False) is None
    )  # interleaved RoPE
    assert qk_norm_rope(qkv.float(), hq * HD, hkv * HD, HD, qn, kn, pos, _cs(), True) is None
    assert qk_norm_rope(qkv, hq * HD, hkv * HD, HD, qn, kn, pos.long(), _cs(), True) is None
    gem = RMSNorm(hidden_size=HD, eps=EPS, dtype=torch.bfloat16, device="cuda", use_gemma=True)
    assert qk_norm_rope(qkv, hq * HD, hkv * HD, HD, gem, kn, pos, _cs(), True) is None


PS = 32  # the draft pool's page size


def _paged(B, K, hkv, pps, ctx, seed, idx_dtype=torch.int32):
    """A random-filled HND cache [pages, 2, hkv, PS, HD], per-request block tables, and the append rows of a K-token
    block per request at ctx[b] .. ctx[b] + K - 1 (as dflash_forward builds them)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = B * pps + 3
    cache = torch.randn(pages, 2, hkv, PS, HD, generator=g, device="cuda").bfloat16()
    perm = torch.randperm(pages, generator=g, device="cuda")[: B * pps]
    kv_indices = perm.to(idx_dtype).contiguous()
    kv_indptr = torch.arange(0, (B + 1) * pps, pps, device="cuda", dtype=torch.int32)
    bidx = torch.arange(B, device="cuda", dtype=idx_dtype).repeat_interleave(K).contiguous()
    apos = (torch.tensor(ctx, device="cuda", dtype=torch.int64).view(-1, 1)
            + torch.arange(K, device="cuda")).reshape(-1).to(idx_dtype).contiguous()  # fmt: skip
    return cache, bidx, apos, kv_indices, kv_indptr


def _stock_append(qkv, hq, hkv, qn, kn, pos, cache, bidx, apos, kv_indices, kv_indptr):
    import flashinfer

    q, k = _stock(qkv, hq, hkv, qn, kn, pos)
    qs, ks = hq * HD, hkv * HD
    v = qkv[:, qs + ks : qs + 2 * ks].reshape(-1, hkv, HD)
    seq = torch.bincount(bidx.long(), minlength=kv_indptr.numel() - 1)
    last = ((apos.view(seq.numel(), -1)[:, -1].long() % PS) + 1).to(torch.int32)
    i32 = lambda t: t.to(torch.int32).contiguous()  # noqa: E731 -- flashinfer takes int32 index tensors
    flashinfer.page.append_paged_kv_cache(k.reshape(-1, hkv, HD).contiguous(), v, i32(bidx), i32(apos), cache,
                                          i32(kv_indices), kv_indptr, last, kv_layout="HND")  # fmt: skip
    return q, k


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize("hq,hkv", [(8, 1), (16, 1)], ids=["tp8", "tp4"])
def test_append_pages_bitwise_to_flashinfer(hq, hkv, idx_dtype):
    K, pps = 7, 4
    for seed in range(6):
        qn, kn = _norms(seed)
        for B, ctx in (
            (1, [29]),
            (1, [0]),
            (2, [30, 61]),
            (3, [5, 95, 31]),
        ):  # blocks crossing pages, page 0, ...
            T = B * K
            qkv, pos = _qkv(T, hq, hkv, 500 * seed + 10 * B + ctx[0])
            cache, bidx, apos, kvi, kvp = _paged(B, K, hkv, pps, ctx, seed, idx_dtype)
            ref_cache = cache.clone()
            rq, rk = _stock_append(qkv, hq, hkv, qn, kn, pos, ref_cache, bidx, apos, kvi, kvp)
            assert append_supported(qkv, hkv * HD, cache, bidx, apos, kvi, kvp)
            got = qk_norm_rope_append(qkv, hq * HD, hkv * HD, HD, qn, kn, pos, _cs(), True, cache, bidx, apos, kvi,
                                      kvp)  # fmt: skip
            assert got is not None
            torch.cuda.synchronize()
            assert torch.equal(got[0], rq) and torch.equal(got[1], rk), (seed, B, ctx)
            assert torch.equal(cache.view(torch.int16), ref_cache.view(torch.int16)), (seed, B, ctx)
            # Slot reuse: the same rows written again with new values (a rejected draft's slots are re-appended).
            qkv2, _ = _qkv(T, hq, hkv, 777 + seed)
            _stock_append(qkv2, hq, hkv, qn, kn, pos, ref_cache, bidx, apos, kvi, kvp)
            qk_norm_rope_append(
                qkv2, hq * HD, hkv * HD, HD, qn, kn, pos, _cs(), True, cache, bidx, apos, kvi, kvp
            )
            torch.cuda.synchronize()
            assert torch.equal(cache.view(torch.int16), ref_cache.view(torch.int16)), (
                "reuse",
                seed,
                B,
                ctx,
            )


def test_append_in_a_cuda_graph():
    hq, hkv, K, B, pps = 8, 1, 7, 1, 4
    qn, kn = _norms(0)
    qkv, pos = _qkv(B * K, hq, hkv, 5)
    cache, bidx, apos, kvi, kvp = _paged(B, K, hkv, pps, [29], 0)
    qk_norm_rope_append(
        qkv, hq * HD, hkv * HD, HD, qn, kn, pos, _cs(), True, cache, bidx, apos, kvi, kvp
    )
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            got = qk_norm_rope_append(qkv, hq * HD, hkv * HD, HD, qn, kn, pos, _cs(), True, cache, bidx, apos, kvi,
                                      kvp)  # fmt: skip
    torch.cuda.current_stream().wait_stream(s)
    for rep, c in enumerate((29, 60, 3, 95)):
        q2, p2 = _qkv(B * K, hq, hkv, 40 + rep)
        qkv.copy_(q2), pos.copy_(p2)
        _, _, a2, _, _ = _paged(B, K, hkv, pps, [c], 0)
        apos.copy_(a2)
        ref_cache = cache.clone()
        rq, rk = _stock_append(qkv, hq, hkv, qn, kn, pos, ref_cache, bidx, apos, kvi, kvp)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(got[0], rq) and torch.equal(got[1], rk), rep
        assert torch.equal(cache.view(torch.int16), ref_cache.view(torch.int16)), rep


def test_append_unsupported_keeps_the_stock_path():
    hq, hkv, K = 8, 1, 7
    qn, kn = _norms(0)
    qkv, pos = _qkv(K, hq, hkv, 9)
    cache, bidx, apos, kvi, kvp = _paged(1, K, hkv, 4, [10], 0)
    args = (qkv, hq * HD, hkv * HD, HD, qn, kn, pos, _cs(), True)
    assert qk_norm_rope_append(*args, cache.float(), bidx, apos, kvi, kvp) is None  # fp32 cache
    assert (
        qk_norm_rope_append(*args, cache.transpose(2, 3), bidx, apos, kvi, kvp) is None
    )  # NHD / strided
    assert qk_norm_rope_append(*args, cache, bidx[:-1], apos, kvi, kvp) is None  # rows mismatch
    assert qk_norm_rope_append(*args, cache, bidx.float(), apos, kvi, kvp) is None
    before = cache.clone()
    assert qk_norm_rope_append(qkv, hq * HD, hkv * HD, HD, qn, kn, pos, _cs(), False, cache, bidx, apos, kvi,
                               kvp) is None  # interleaved RoPE: nothing written  # fmt: skip
    assert torch.equal(cache.view(torch.int16), before.view(torch.int16))
