# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One DSv4.1 attention layer of the dep2 target: the target's forward (the engine's fused kernels: `csa2_store_rows`,
`csa2_sparse_decode`, `csa2_indexer_topk_decode` ...) against a plain-torch oracle of the same math (`_torch_forward`,
test-only). Same random weights and inputs, layers 0 (window only), 2
(ratio-2 owner), 3 (reuse), 20 and 24 (candidate source and consumer), with the decode, long-chunk and fresh-chunk
shapes. The gate (`_gate`) is a relative error of 5% of the output norm for layer 0 at <= 16 rows: bf16 weights and
quantized rows differ by rounding steps between the two paths. Layers 2, 3 and 20 attend compressed rows picked by an
indexer (their own or, for 3, their source's) among near-equal random scores, and every cell past 16 rows quantizes the
projections' activations to MXFP8: those are gated at 35%, which catches a plumbing error but not a small numerical
drift (the op-level tests bound each op). A candidate consumer picks among near-equal random scores, so it is gated
only on being finite and sane (the op-level tests bound it).
"""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog.comm.allgather import allgather as cat_allgather
from tensorrt_llm._torch.modeling_v2.catalog.comm.reducescatter import (
    reducescatter as cat_reducescatter,
)
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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SM100 family",
)
att = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.attention"
)
rope_mod = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope"
)
DEV, MAXSEQ = "cuda", 8192
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
    candidate_topk_blocks=64,
    index_topk=64,
    index_n_heads=32,
    index_head_dim=128,
)


class _Kvm:
    def __init__(self, nreq):
        g = torch.Generator(device=DEV).manual_seed(7)
        self.kvm, self.pools, self.tables, self.g = NS(max_seq_len=MAXSEQ), {}, {}, g
        # Every request owns its own pages, as the cache manager's: a pool smaller than the requests' tables would alias
        # one request's pages with another's, and a context chunk (which writes every row's SWA slot) would then race
        # with another request's reads of the same physical rows in the same step.
        self.pages = max(64, nreq * (MAXSEQ // 128 + 1))
        # the target's own stacked SWA page tables ([layers, requests, blocks]), as `step_table` serves them
        self.swa_tables = torch.stack(
            [self.step_table(layer, "SWA", nreq) for layer in range(len(RATIOS))]
        ).contiguous()

    def pool(self, layer, role):
        r = max(RATIOS[layer], 1)
        pages = self.pages
        shape, dt = {
            "SWA": ((pages, 128, 528), torch.uint8),
            "GLOBAL": ((max(128, pages), 128 // r, 288), torch.uint8),
            "INDEX": ((max(128, pages), 128 // r, 68), torch.uint8),
            "COMPRESSOR_KV": ((pages, 128, 512), torch.float32),
            "COMPRESSOR_SCORE": ((pages, 128, 512), torch.float32),
        }[role]
        key = (role, shape)
        if key not in self.pools:
            if dt == torch.uint8:
                p = torch.randint(0, 100, shape, dtype=dt, device=DEV, generator=self.g)
                if role == "GLOBAL":
                    p[..., 256:] = 56
                if role == "SWA":
                    p[..., 512:] = 120
            else:
                p = torch.randn(shape, device=DEV, generator=self.g)
            self.pools[key] = p
        return self.pools[key]

    def step_table(self, layer, role, nreq):
        n = self.pool(layer, role).shape[0]
        return (torch.arange(nreq * (MAXSEQ // 128 + 1), device=DEV, dtype=torch.int32) % n).view(
            nreq, -1
        )


def _ctx(nreq, qlen, base, paged):
    n = nreq * qlen
    pos = (base + torch.arange(qlen, device=DEV)).repeat(nreq).int()
    rid = torch.arange(nreq, device=DEV).repeat_interleave(qlen)
    cs = pos - (torch.arange(n, device=DEV) % qlen).int()
    rope = {
        0: rope_mod.csa2_rope_table(64, MAXSEQ + 8, 0, 10000.0, 1.0, 32, 1, DEV),
        1: rope_mod.csa2_rope_table(64, MAXSEQ + 8, 65536, 160000.0, 16.0, 32, 1, DEV),
    }
    segs = (
        []
        if paged
        else [(r * qlen, (r + 1) * qlen, r, int(pos[(r + 1) * qlen - 1]) + 1) for r in range(nreq)]
    )
    return att.StepCtx(
        pos=pos,
        rid=rid,
        cs=cs,
        ce=cs + qlen,
        segs=segs,
        kvm=_Kvm(nreq),
        rope=rope,
        nreq=nreq,
        paged=paged,
    )


def _layer(L, tp8=False):
    torch.manual_seed(L)
    mapping = (
        NS(enable_attention_dp=False, world_size=8, tp_size=8, tp_rank=0)
        if tp8
        else NS(enable_attention_dp=True, world_size=1, tp_size=1, tp_rank=0)
    )
    m = att._Attention(CFG, L, mapping).to(DEV)
    if tp8:
        m.reduce_output = False  # this rank's partial sum: no collective on one GPU
    for name, p in m.named_parameters():
        if p.dtype == torch.float8_e4m3fn:  # MXFP8 projections: weights ~ N(0, 1) x scale 2^-6
            p.data.copy_(torch.randn(p.shape, device=p.device).to(p.dtype))
        elif p.dtype == torch.uint8:
            p.data.fill_(121)
        elif p.dtype.is_floating_point:
            p.data.normal_(0, 0.02)
    m.attn_sink.data.normal_(0, 1)
    m.wo_a_scale.data.fill_(
        2.0**-6
    )  # the grouped o_a's 128x128 block multipliers are powers of two
    for nm in ("q_norm", "kv_norm", "c_norm", "i_knorm"):
        if hasattr(m, nm):
            getattr(m, nm).data.fill_(1.0)
    return m


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
    if att._publish_mask(ctx) is not None:
        keep = att._publish_mask(ctx)
    k = flashinfer_rmsnorm(cublas_mm(lat, m.i_wk.t()), m.i_knorm, m.eps)
    k = csa2_rope_tail(k, ctx.rope[1], first, m.rd)
    csa2_scatter_rows_footer(att._pool(ctx, L, att.INDEX), addr, csa2_pack_rows(k, "index"), keep)
    csa2_scatter_rows(
        att._pool(ctx, L, att.GLOBAL), addr, csa2_pack_rows(_rope(m, ctx, lat, first), "main"), keep
    )


def _index_topk(m, ctx, x, qr, pos):
    o = m.owner
    q = cublas_mm(qr, _bf16(m, "i_wq_b").t()).view(-1, m.index_heads, m.index_hd)
    q = csa2_unpack_rows(
        csa2_pack_rows(csa2_rope_tail(q, ctx.rope[1], pos, m.rd), "index"), "index", q.dtype
    )
    proj = cublas_mm(x, m.i_wproj.t())
    w = torch.mul(
        proj,
        m.index_hd**-0.5 * m.index_heads**-0.5,
        out=torch.empty(proj.shape, dtype=torch.float32, device=proj.device),
    )
    cands = ctx.cands if m.uses_cands else None
    pool, table = att._pool(ctx, o, att.INDEX), att._table(ctx, o, att.GLOBAL)
    segs = ctx.segs
    if ctx.paged:  # a generation step: each request's rows are consecutive, its span one past its last position
        rid = ctx.rid.tolist()
        bounds = [t for t in range(1, len(rid)) if rid[t] != rid[t - 1]]
        segs = [
            (t0, t1, rid[t0], int(pos[t1 - 1]) + 1)
            for t0, t1 in zip([0, *bounds], [*bounds, len(rid)])
        ]
    idx, cands = csa2_indexer_topk(
        q,
        w,
        pos,
        segs,
        pool,
        table,
        m.ratio,
        m.index_topk,
        m.cand_block,
        m.cand_topk,
        cands,
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
        if m.is_owner and not ctx.published:
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
        True,
        ctx.floor,
    )
    a = _rope(m, ctx, a, pos, inverse=True).reshape(n, m.gl, -1)
    wo_a = m._wo_a_dense()
    parts = [
        cublas_mm(a[:, g].contiguous(), wo_a[g * m.o_rank : (g + 1) * m.o_rank].t())
        for g in range(m.gl)
    ]
    out = cublas_mm(torch.cat(parts, 1), _bf16(m, "wo_b").t())
    if m.tp == 1 or not m.reduce_output:
        return out
    ws = len(m.group)
    rows = -(-n // ws) * ws
    outp = out if rows == n else torch.nn.functional.pad(out, [0, 0, 0, rows - n])
    red = cat_allgather(cat_reducescatter(outp.contiguous(), None, m.group), None, m.group)
    return red if rows == n else torch.split(red, [n, rows - n], 0)[0]


def _gate(rows, indexer=False):
    """<= 16 rows run the low-M kernel (exact FP8-weight products): bf16-rounding gate. More rows also quantize every
    projection's activation to MXFP8 (3-bit mantissa) as the engine does; on random data a chain of five such
    projections feeding a softmax moves the output by ~20%, so only a plumbing error (O(1)) is caught; the op test
    (`test_modeling_v2_csa2_mxfp8_linear`) bounds each projection at 3% and GSM8K bounds the model."""
    if indexer:  # the index keys are picked among near-equal random scores: a rounding step upstream moves the picks
        return 0.35
    return 0.05 if rows <= 16 else 0.35


def _run(L, nreq, qlen, base, paged, fused, tp8=False, shared=False):
    """Layers 20 -> L chain so that candidate consumers and reuse layers see their source's picks. `fused`: the target's
    forward, else the torch oracle. Context chunks attend with the shared-bank FP8 context path (eager steps, `shared`)
    or with the decode kernel (a graph bucket's body: every row `live`)."""
    ctx = _ctx(nreq, qlen, base, paged)
    if fused and not paged and not shared:
        ctx.live = torch.ones(nreq * qlen, dtype=torch.bool, device=DEV)
    x = torch.randn(
        nreq * qlen, 5120, device=DEV, generator=torch.Generator(device=DEV).manual_seed(3)
    ).bfloat16()
    chain = {0: [0], 2: [2], 3: [2, 3], 20: [20], 24: [20, 24]}[L]
    for layer in chain:
        m = _layer(layer, tp8)
        out = m(x, ctx) if fused else _torch_forward(m, x, ctx)
    return out


@pytest.mark.parametrize("L", [0, 2, 3, 20, 24])
@pytest.mark.parametrize(
    "nreq,qlen,base,paged",
    [(6, 4, 3000, True), (4, 4, 3000, True), (1, 600, 700, False), (2, 300, 0, False)],
)
def test_fused_layer_matches_torch(L, nreq, qlen, base, paged):
    ref = _run(L, nreq, qlen, base, paged, fused=False)
    got = _run(L, nreq, qlen, base, paged, fused=True)
    assert torch.isfinite(got).all()
    if L == 24:
        # A candidate consumer picks its keys among near-equal random scores, so any rounding difference upstream
        # changes which rows are attended: only the op-level tests
        # (csa2_indexer_decode) can bound it. Here: finite and sane.
        norm, ref_norm = got.float().norm(), ref.float().norm()
        assert (
            0.1 * ref_norm < norm < 3 * ref_norm
        )  # sane, and the consumer's output is not dropped
        return
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < _gate(nreq * qlen, L != 0), float(rel)


@pytest.mark.parametrize("L", [0, 2, 3])
@pytest.mark.parametrize("nreq,qlen,base", [(1, 600, 700), (2, 300, 0)])
def test_tp8_context_chunk_uses_the_fused_kernels(L, nreq, qlen, base):
    """8 heads per rank (TP8): a context chunk runs the 64-head kernel on a head-padded query, not the torch path."""
    ref = _run(L, nreq, qlen, base, False, fused=False, tp8=True)
    got = _run(L, nreq, qlen, base, False, fused=True, tp8=True)
    assert torch.isfinite(got).all()
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < _gate(nreq * qlen), float(rel)


@pytest.mark.parametrize("L", [0, 2, 3, 20])
@pytest.mark.parametrize("nreq,qlen,base", [(1, 600, 700), (2, 300, 0), (3, 150, 1000)])
def test_shared_context_matches_torch_and_decode_kernel(L, nreq, qlen, base):
    """The shared-bank FP8 context path (stock's `forward_fused_context` with the O-A input from the kernel) against the
    torch oracle (the same plumbing gate as the decode kernel's) and against the decode-kernel path (bf16 attention):
    the two engine paths differ by the context kernel's E4M3 Q / K / P and the O-A input's E4M3 rounding."""
    ref = _run(L, nreq, qlen, base, False, fused=False)
    dec = _run(L, nreq, qlen, base, False, fused=True)
    got = _run(L, nreq, qlen, base, False, fused=True, shared=True)
    assert torch.isfinite(got).all()
    rel = float((got.float() - ref.float()).norm() / ref.float().norm())
    rel_dec = float((dec.float() - ref.float()).norm() / ref.float().norm())
    rel_pair = float((got.float() - dec.float()).norm() / dec.float().norm())
    print(
        f"L{L} nreq{nreq} qlen{qlen} base{base}: shared vs torch {rel:.4f}, decode vs torch {rel_dec:.4f}, "
        f"shared vs decode {rel_pair:.4f}"
    )
    assert rel < _gate(nreq * qlen, L != 0), rel
    assert rel_pair < 0.08, rel_pair  # measured 0.055-0.060 over these cells


def _ctx_chunks(lens, bases):
    """A context step of several requests: (rows `lens[r]` starting at position `bases[r]`)."""
    pos = torch.cat([base + torch.arange(n, device=DEV) for n, base in zip(lens, bases)]).int()
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
    kvm = _Kvm(len(lens))
    for role in (
        "SWA",
        "GLOBAL",
        "INDEX",
    ):  # the pools are random on first use: fix their contents whatever the call order
        kvm.pool(20, role)
    return att.StepCtx(
        pos=pos, rid=rid, cs=cs, ce=ce, segs=segs, kvm=kvm, rope=rope, nreq=len(lens), paged=False
    )


def test_bounded_replay_boundary_matches_truncated_full_layer():
    """The boundary owner (layer 20) run over the last-window rows only, after `publish_global` of every row, gives
    those rows what the whole layer gives them with the window truncated at the replay floor (stock's contract), on the
    shared-bank context path (its staging scale is a power of two, so staging other rows does not move it)."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import DecoderReplayPlan

    modeling = importlib.import_module(
        "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling"
    )
    lens, bases, window = [500, 90], [700, 40], 140
    keep = [
        min(n, window) for n in lens
    ]  # the last `window` rows of each chunk; a short chunk keeps all of it
    floors = [max(0, b + n - window) for n, b in zip(lens, bases)]  # prompt end = chunk end here
    x = torch.randn(
        sum(lens), 5120, device=DEV, generator=torch.Generator(device=DEV).manual_seed(5)
    ).bfloat16()
    attn = _layer(20)

    full_ctx = _ctx_chunks(lens, bases)
    full_ctx.floor = torch.tensor(floors, dtype=torch.int32, device=DEV)[full_ctx.rid]
    want = attn(x, full_ctx)

    rows = torch.cat(
        [
            torch.arange(sum(lens[:r]) + n - k, sum(lens[:r]) + n)
            for r, (n, k) in enumerate(zip(lens, keep))
        ]
    )
    plan = DecoderReplayPlan(
        rows=rows.to(DEV), replay_seq_lens=torch.tensor(keep), replay_num_cached=[],
        saved_seq_lens=torch.tensor(lens), saved_num_cached=[], replay_prompt_lens=[], saved_prompt_lens=[],
        num_encoder_tokens=sum(lens), swa_floors=floors,
    )  # fmt: skip
    ctx = _ctx_chunks(lens, bases)
    boundary = NS(publish_global=lambda h, pre_mix, c: attn.publish_global(h, c))
    h_sub, _, dctx, kept = modeling.ModelingV2Core._enter_decoder(
        None, boundary, x, torch.zeros(sum(lens), 4, device=DEV), ctx, plan
    )
    assert dctx.published and dctx.pos.numel() == sum(keep) and torch.equal(kept.cpu(), rows)
    assert dctx.segs == [
        (0, keep[0], 0, bases[0] + lens[0]),
        (keep[0], sum(keep), 1, bases[1] + lens[1]),
    ]
    assert torch.equal(
        dctx.cs.cpu(),
        torch.tensor(
            [bases[0] + lens[0] - keep[0]] * keep[0] + [bases[1] + lens[1] - keep[1]] * keep[1]
        ).int(),
    )
    got = attn(h_sub, dctx)
    ref = want.index_select(0, rows.to(DEV))
    assert torch.isfinite(got).all()
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.03, float(rel)
