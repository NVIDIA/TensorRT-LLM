# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context chunks of the target's attention layers attend as stock's `forward_fused_context` (the shared-bank FP8
context path, as the dep2 target): layers 0 (window only), 2 (ratio-2 owner), 3 (reuse), 20 and 24 (candidate source and
consumer) on chunks of one request of 1100 rows (stock's fused query from 1024 rows) and of two requests with history,
take that path, agree with the plain-torch oracle (`_torch_forward`, test-only: bf16 dequantized projections, the torch
indexer and attention) within the attention gate and with the decode-kernel path (the one a mixed step's rows take)
within the E4M3 rounding of Q / K / P that separates the two engine paths (measured 0.055-0.060 on the dep2 target)."""

import importlib
import itertools
from types import SimpleNamespace as NS

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_compress import (
    csa2_compress_pairs,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer import (
    csa2_indexer_topk,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_pack_rows import (
    csa2_pack_rows,
    csa2_unpack_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope import (
    csa2_rope_tail,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rows_io import (
    csa2_row_addr,
    csa2_scatter_rows,
    csa2_scatter_rows_footer,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_sparse_attn import (
    csa2_sparse_attn,
)

TARGET = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)
att = importlib.import_module(f"{TARGET}.attention")
rope_mod = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope"
)
DEV, MAXSEQ = "cuda", 16384
RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
CFG = NS(
    rms_norm_eps=1e-6,
    hidden_size=5120,
    head_dim=512,
    qk_rope_head_dim=64,
    num_attention_heads=64,
    q_lora_rank=1280,
    o_lora_rank=1024,
    o_groups=8,
    sliding_window=128,
    compress_ratios=RATIOS,
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    candidate_source_layer_id=20,
    candidate_block_size=8,
    candidate_topk_blocks=2048,
    index_topk=512,
    index_n_heads=32,
    index_head_dim=128,
)


_COLS = MAXSEQ // 128 + 1  # page-table columns of a request
_MAXREQ = 80  # requests of the widest step below


class _Kvm:
    """Random pools and page tables in which every request owns its pages (a step's row writes never collide); the
    index pages carry sane UE8M0 scales (finite scores without ties), the fp8 rows no NaN bytes."""

    def __init__(self, nreq):
        self.g = torch.Generator(device=DEV).manual_seed(7)
        self.kvm, self.pools = NS(max_seq_len=MAXSEQ), {}
        for layer, role in itertools.product(
            (2, 20), ("SWA", "GLOBAL", "INDEX", "COMPRESSOR_KV", "COMPRESSOR_SCORE")
        ):
            self.pool(
                layer, role
            )  # every pool in a fixed order, whatever order (and stream) the layer asks in
        self.swa_tables = torch.stack(
            [self.step_table(layer, "SWA", nreq) for layer in range(len(RATIOS))]
        ).contiguous()

    def pool(self, layer, role):
        r = max(RATIOS[layer], 1)
        pages = _MAXREQ * _COLS
        shape, dt = {
            "SWA": ((pages, 128, 528), torch.uint8),
            "GLOBAL": ((pages, 128 // r, 288), torch.uint8),
            "INDEX": ((pages, 128 // r, 68), torch.uint8),
            "COMPRESSOR_KV": ((pages, 128, 512), torch.float32),
            "COMPRESSOR_SCORE": ((pages, 128, 512), torch.float32),
        }[role]
        key = (role, shape)
        if key not in self.pools:
            if (
                role == "INDEX"
            ):  # native 64-row footer pages: 64 x 64 data bytes, then 64 x 4 UE8M0 scale bytes
                p = torch.randint(0, 256, shape, dtype=dt, device=DEV, generator=self.g)
                scales = p.view(-1, 64 * 68)[:, 64 * 64 :]
                scales.copy_(
                    torch.randint(124, 130, scales.shape, dtype=dt, device=DEV, generator=self.g)
                )
            elif dt == torch.uint8:
                p = torch.randint(0, 100, shape, dtype=dt, device=DEV, generator=self.g)
                tail = {"GLOBAL": (256, 56), "SWA": (512, 120)}[role]
                p[..., tail[0] :] = tail[1]
            else:
                p = torch.randn(shape, device=DEV, generator=self.g)
            self.pools[key] = p
        return self.pools[key]

    def step_table(self, layer, role, nreq):
        return torch.arange(nreq * _COLS, device=DEV, dtype=torch.int32).view(nreq, _COLS)


def _gen_ctx(nreq, qlen):
    bases = 3000 + (torch.arange(nreq, device=DEV) * 997) % 12000  # a different history per request
    pos = (bases.repeat_interleave(qlen) + torch.arange(qlen, device=DEV).repeat(nreq)).int()
    rid = torch.arange(nreq, device=DEV).repeat_interleave(qlen)
    cs = pos - (torch.arange(nreq * qlen, device=DEV) % qlen).int()
    rope = {
        0: rope_mod.csa2_rope_table(64, MAXSEQ + 8, 0, 10000.0, 1.0, 32, 1, DEV),
        1: rope_mod.csa2_rope_table(64, MAXSEQ + 8, 65536, 160000.0, 16.0, 32, 1, DEV),
    }
    return att.StepCtx(
        pos=pos,
        rid=rid,
        cs=cs,
        ce=cs + qlen,
        segs=[],
        kvm=_Kvm(nreq),
        rope=rope,
        nreq=nreq,
        paged=True,
    )


def _layer(L):
    torch.manual_seed(L)
    m = att._Attention(CFG, L, NS(enable_attention_dp=True, world_size=1, tp_size=1, tp_rank=0)).to(
        DEV
    )
    for _, p in m.named_parameters():
        if p.dtype == torch.float8_e4m3fn:
            p.data.copy_(torch.randn(p.shape, device=p.device).to(p.dtype))
        elif p.dtype == torch.uint8:
            p.data.fill_(121)
        elif p.dtype.is_floating_point:
            p.data.normal_(0, 0.02)
    m.attn_sink.data.normal_(0, 1)
    m.wo_a_scale.data.fill_(2.0**-6)
    for nm in ("q_norm", "kv_norm", "c_norm", "i_knorm"):
        if hasattr(m, nm):
            getattr(m, nm).data.fill_(1.0)
    return m


def _ctx(lens, bases):
    """A context step of several requests: rows `lens[r]` starting at position `bases[r]`."""
    pos = torch.cat([b + torch.arange(n, device=DEV) for n, b in zip(lens, bases)]).int()
    rid = torch.cat([torch.full((n,), r, device=DEV) for r, n in enumerate(lens)])
    cs = torch.cat([torch.full((n,), b, device=DEV) for n, b in zip(lens, bases)]).int()
    ce = torch.cat([torch.full((n,), b + n, device=DEV) for n, b in zip(lens, bases)]).int()
    rope = {
        0: rope_mod.csa2_rope_table(64, MAXSEQ + 8, 0, 10000.0, 1.0, 32, 1, DEV),
        1: rope_mod.csa2_rope_table(64, MAXSEQ + 8, 65536, 160000.0, 16.0, 32, 1, DEV),
    }
    t0, segs = 0, []
    for r, (n, b) in enumerate(zip(lens, bases)):
        segs.append((t0, t0 + n, r, b + n))
        t0 += n
    return att.StepCtx(
        pos=pos,
        rid=rid,
        cs=cs,
        ce=ce,
        segs=segs,
        kvm=_Kvm(len(lens)),
        rope=rope,
        nreq=len(lens),
        paged=False,
    )


# ---- the plain-torch oracle (bf16 dequantized projections,
# torch indexer and attention) -------------------------------
def _bf16(m, name):
    w, s = getattr(m, name), getattr(m, name + "_s")
    return (w.float() * torch.exp2(s.float() - 127).repeat_interleave(32, 1)).to(torch.bfloat16)


def _rope(m, ctx, x, pos, inverse=False):
    return csa2_rope_tail(x, ctx.rope[1 if m.ratio else 0], pos, m.rd, inverse)


def _write_compressed(m, ctx, x, pos):
    L = m.layer
    keep = torch.ones(pos.shape, dtype=torch.bool, device=pos.device)
    if m.ratio == 2:
        kv = cublas_mm(x.to(torch.float32), m.c_wkv.t(), None, torch.float32)
        sc = cublas_mm(x.to(torch.float32), m.c_wgate.t(), None, torch.float32)
        lat = csa2_compress_pairs(
            kv,
            sc,
            pos,
            ctx.rid,
            ctx.cs,
            ctx.ce,
            att._pool(ctx, L, att.COMP_KV),
            att._pool(ctx, L, att.COMP_SCORE),
            att._table(ctx, L, att.COMP_KV),
            att._table(ctx, L, att.COMP_SCORE),
        )
        lat = flashinfer_rmsnorm(lat.to(torch.bfloat16), m.c_norm, m.eps)
        row, first = torch.div(pos, 2, rounding_mode="floor"), pos - torch.remainder(pos, 2)
    else:
        lat = flashinfer_rmsnorm(cublas_mm(x, m.c_wkv.t()), m.c_norm, m.eps)
        row, first = pos, pos
    addr = csa2_row_addr(att._table(ctx, L, att.GLOBAL), ctx.rid, row, att._WINDOW_ROWS // m.ratio)
    k = flashinfer_rmsnorm(cublas_mm(lat, m.i_wk.t()), m.i_knorm, m.eps)
    k = csa2_rope_tail(k, ctx.rope[1], first, m.rd)
    csa2_scatter_rows_footer(att._pool(ctx, L, att.INDEX), addr, csa2_pack_rows(k, "index"), keep)
    csa2_scatter_rows(
        att._pool(ctx, L, att.GLOBAL), addr, csa2_pack_rows(_rope(m, ctx, lat, first), "main"), keep
    )


def _index_topk(m, ctx, x, qr, pos):
    q = cublas_mm(qr, _bf16(m, "i_wq_b").t()).view(-1, m.index_heads, m.index_hd)
    q = csa2_unpack_rows(
        csa2_pack_rows(csa2_rope_tail(q, ctx.rope[1], pos, m.rd), "index"), "index", q.dtype
    )
    w = m._index_weights(x)
    idx, cands = csa2_indexer_topk(
        q,
        w,
        pos,
        ctx.segs,
        att._pool(ctx, m.owner, att.INDEX),
        att._table(ctx, m.owner, att.GLOBAL),
        m.ratio,
        m.index_topk,
        m.cand_block,
        m.cand_topk,
        ctx.cands if m.uses_cands else None,
        m.is_cand_source,
    )
    if m.is_cand_source:
        ctx.cands = cands
    ctx.topk = idx


def _torch_forward(m, x, ctx):
    L, pos, n = m.layer, ctx.pos, x.shape[0]
    qr = flashinfer_rmsnorm(cublas_mm(x, _bf16(m, "wq_a").t()), m.q_norm, m.eps)
    q = _rope(m, ctx, cublas_mm(qr, _bf16(m, "wq_b").t()).view(n, m.hl, m.hd), pos)
    kv = flashinfer_rmsnorm(cublas_mm(x, _bf16(m, "wkv").t()), m.kv_norm, m.eps)
    kv_rows = csa2_pack_rows(
        _rope(m, ctx, kv, pos), "swa"
    )  # rotated, fp8 rows as the pool holds them
    kv = csa2_unpack_rows(kv_rows, "swa")
    addr = csa2_row_addr(att._table(ctx, L, att.SWA), ctx.rid, pos, att._WINDOW_ROWS)
    csa2_scatter_rows(att._pool(ctx, L, att.SWA), addr, kv_rows, pos >= ctx.ce - m.window)
    main_pool = main_table = topk = None
    if m.ratio:
        if m.is_owner:
            _write_compressed(m, ctx, x, pos)
        if m.is_index_source:
            _index_topk(m, ctx, x, qr, pos)
        main_pool, main_table, topk = (
            att._pool(ctx, m.owner, att.GLOBAL),
            att._table(ctx, m.owner, att.GLOBAL),
            ctx.topk,
        )
    a = csa2_sparse_attn(
        q,
        kv,
        m.attn_sink,
        pos,
        ctx.cs,
        ctx.rid,
        att._pool(ctx, L, att.SWA),
        att._table(ctx, L, att.SWA),
        m.window,
        m.scale,
        main_pool,
        main_table,
        topk,
        m.ratio,
    )
    a = _rope(m, ctx, a, pos, inverse=True).reshape(n, m.gl, -1)
    wo_a = (m.wo_a.float() * m.wo_a_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(
        torch.bfloat16
    )
    parts = [
        cublas_mm(a[:, g].contiguous(), wo_a[g * m.o_rank : (g + 1) * m.o_rank].t())
        for g in range(m.gl)
    ]
    return cublas_mm(torch.cat(parts, 1), _bf16(m, "wo_b").t())


def _run(L, lens, bases, fused=True, shared=True):
    """Layers chain so that reuse layers and candidate consumers see their source's state. `fused`: the target's
    forward, else the torch oracle; `shared=False`: the step's rows marked as a mixed step's (one row short of all
    context), whose rows attend on the decode kernel as stock's mixed steps do."""
    ctx = _ctx(lens, bases)
    if fused and not shared:
        ctx.nctx = sum(lens) - 1
    x = torch.randn(
        sum(lens), 5120, device=DEV, generator=torch.Generator(device=DEV).manual_seed(3)
    ).bfloat16()
    for layer in {0: [0], 2: [2], 3: [2, 3], 20: [20], 24: [20, 24]}[L]:
        m = _layer(layer)
        out = m(x, ctx) if fused else _torch_forward(m, x, ctx)
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("L", [0, 2, 3, 20, 24])
@pytest.mark.parametrize("lens, bases", [([1100], [0]), ([500, 90], [700, 40])])
def test_context_attends_as_stock(L, lens, bases, monkeypatch):
    calls = []
    fn = att.csa2_sparse_context_o_a_input
    monkeypatch.setattr(
        att, "csa2_sparse_context_o_a_input", lambda *a: (calls.append(1), fn(*a))[1]
    )
    got = _run(L, lens, bases)
    assert calls, "a context chunk takes the shared-bank FP8 context path"
    dec = _run(L, lens, bases, shared=False)
    ref = _run(L, lens, bases, fused=False)
    assert torch.isfinite(got).all()
    if L == 24:  # a candidate consumer picks among near-equal random scores: only finite and sane
        norm, ref_norm = got.float().norm(), ref.float().norm()
        assert (
            0.1 * ref_norm < norm < 3 * ref_norm
        )  # sane, and the consumer's output is not dropped
        return
    rel = float((got.float() - ref.float()).norm() / ref.float().norm())
    pair = float((got.float() - dec.float()).norm() / dec.float().norm())
    print(f"L{L} lens{lens}: shared vs torch {rel:.4f}, shared vs decode kernel {pair:.4f}")
    assert rel < 0.35, (
        rel
    )  # the gate of the target's attention tests (MXFP8 projections over random data)
    assert pair < 0.08, pair
