# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The headtp drafter layout: each CP rank attends with its own 6 heads over a replicated ring.

* weights.py, head-parallel: ``q_b`` and ``kv_b`` hold exactly heads ``[6 r, 6 r + 6)`` of the
  whole load, in the views ``_draft_views`` reads (``k_b`` / ``v_b_t`` per head), and every other
  tensor is the helix layouts' (``o_proj``'s 6 heads' columns among them). CPU, a small drafter
  of the same head count.
* ``_draft_attention_headtp`` on ranks 0 and 15 at the served dims: against the FP32 composition of
  its ops' references over the whole weights' heads of that rank (q_b, W_UK, the block RoPE, the
  window over the ring, W_UV), with the block latents left in the ring bitwise the block RoPE's
  reference; two requests on other slots, one with history past the ring. Both query paths: q_b then
  the W_UK product, and the absorbed query weight (one GEMM, the RoPE in place).
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31."
    "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
)
_OPS = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec."


def _mod(name):
    return importlib.import_module(_PKG + "." + name)


# ---- weights ------------------------------------------------------------------------------------

_SMALL = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 96,
    "q_lora_rank": 32,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 8,
    "v_head_dim": 8,
    "intermediate_size": 256,
    "vocab_size": 64,
    "rms_norm_eps": 1e-5,
    "sliding_window": 1024,
}


def _checkpoint(s, dflash, captures):
    hidden, heads, layers = s["hidden_size"], s["num_attention_heads"], s["num_hidden_layers"]
    q_rank, latent, rope = s["q_lora_rank"], s["kv_lora_rank"], s["qk_rope_head_dim"]
    nope, value, inter = s["qk_nope_head_dim"], s["v_head_dim"], s["intermediate_size"]
    taps, groups = dflash["conv_kernel_size"], hidden // dflash["conv_group_size"]
    rank = dflash["selector_rank"]
    shapes = {
        "fc.weight": (hidden, captures * hidden),
        "hidden_norm.weight": (hidden,),
        "norm.weight": (hidden,),
        "candidate_selector.predecessor_codebook": (s["vocab_size"], rank),
        "candidate_selector.successor_codebook": (s["vocab_size"], rank),
        "candidate_selector.hidden_projection.weight": (rank, hidden),
    }
    for j in range(layers):
        shapes.update(
            {
                f"layers.{j}.self_attn.kv_a_proj_with_mqa.weight": (latent + rope, hidden),
                f"layers.{j}.self_attn.kv_a_layernorm.weight": (latent,),
                f"layers.{j}.self_attn.kv_b_proj.weight": (heads * (nope + value), latent),
                f"layers.{j}.self_attn.q_a_proj.weight": (q_rank, hidden),
                f"layers.{j}.self_attn.q_a_layernorm.weight": (q_rank,),
                f"layers.{j}.self_attn.q_b_proj.weight": (heads * (nope + rope), q_rank),
                f"layers.{j}.self_attn.o_proj.weight": (hidden, heads * value),
                f"layers.{j}.mlp.gate_proj.weight": (inter, hidden),
                f"layers.{j}.mlp.up_proj.weight": (inter, hidden),
                f"layers.{j}.mlp.down_proj.weight": (hidden, inter),
                f"layers.{j}.input_layernorm.weight": (hidden,),
                f"layers.{j}.post_attention_layernorm.weight": (hidden,),
                f"layers.{j}.attention_conv.base_kernel": (2, taps, hidden),
                f"layers.{j}.attention_conv.kernel_projection.weight": (2 * taps * groups, hidden),
                f"layers.{j}.mlp_conv.base_kernel": (2, taps, hidden),
                f"layers.{j}.mlp_conv.kernel_projection.weight": (2 * taps * groups, hidden),
            }
        )
    g = torch.Generator().manual_seed(11)
    return {k: torch.randn(v, generator=g).bfloat16() for k, v in shapes.items()}


@pytest.mark.parametrize("rank", [0, 7, 15])
def test_head_parallel_load_is_the_ranks_heads(rank, monkeypatch):
    weights = _mod("weights")
    modeling = _mod("modeling")
    monkeypatch.setattr(weights, "DRAFT_SHAPE", _SMALL)
    ckpt = _checkpoint(_SMALL, weights.DRAFT_DFLASH, weights.DRAFT_CAPTURES)
    loader = weights.FlatCheckpointLoader(ckpt, {}, device="cpu", rank=rank)
    config = SimpleNamespace(**_SMALL)
    whole = loader.dflash2(config, cp_rank=rank)
    tp = loader.dflash2(config, cp_rank=rank, head_parallel=True)
    assert whole.keys() == tp.keys()
    heads, nope, rope = 96, _SMALL["qk_nope_head_dim"], _SMALL["qk_rope_head_dim"]
    latent, layers = _SMALL["kv_lora_rank"], _SMALL["num_hidden_layers"]
    mine = slice(6 * rank, 6 * rank + 6)
    view_w = modeling._draft_views(whole, layers, nope, latent)
    view_tp = modeling._draft_views(tp, layers, nope, latent)
    for j in range(layers):
        for key in whole:
            if key in (f"d_l{j}_q_b", f"d_l{j}_kv_b"):
                continue
            if key.startswith("d_l") and not key.startswith(f"d_l{j}_"):
                continue
            assert torch.equal(whole[key], tp[key]), key
        q_whole = whole[f"d_l{j}_q_b"].view(heads, nope + rope, -1)
        assert torch.equal(tp[f"d_l{j}_q_b"], q_whole[mine].reshape(-1, q_whole.shape[-1]))
        lw, lt = view_w.layers[j], view_tp.layers[j]
        assert lt.q_b.is_contiguous() and lt.q_b.shape == (6 * (nope + rope), _SMALL["q_lora_rank"])
        assert torch.equal(lt.k_b, lw.k_b[mine]) and torch.equal(lt.v_b_t, lw.v_b_t[mine])
        # o_proj's input columns are the same 6 heads' in both layouts.
        assert torch.equal(lt.o, lw.o) and lt.o.shape[1] == 6 * _SMALL["v_head_dim"]
        # ...and they are heads [6 r, 6 r + 6) of the checkpoint's o_proj.
        o_ck = ckpt[f"layers.{j}.self_attn.o_proj.weight"].view(_SMALL["hidden_size"], heads, -1)
        assert torch.equal(lt.o, o_ck[:, mine].reshape(_SMALL["hidden_size"], -1))


# ---- the attention ------------------------------------------------------------------------------

Q_RANK, HEADS, NOPE, ROPE, LAT, VAL = 1536, 96, 128, 64, 512, 128
BLOCK, W, PAGE = 8, 1024, 64


def _rel(got, want):
    return (torch.linalg.vector_norm(got.float() - want) / torch.linalg.vector_norm(want)).item()


def test_the_absorbed_query_weight_is_w_uk_times_q_b():
    """``_absorbed_query_weight``: per head the FP64 ``W_UK^T q_b_nope`` rounded once to BF16, then q_b's rope
    rows unchanged (CPU, the served per-rank dims)."""
    modeling = _mod("modeling")
    g = torch.Generator().manual_seed(3)
    q_b = (0.03 * torch.randn(6 * (NOPE + ROPE), Q_RANK, generator=g)).bfloat16()
    k_b = (0.05 * torch.randn(6, NOPE, LAT, generator=g)).bfloat16()
    w = modeling._absorbed_query_weight(q_b, k_b)
    assert w.shape == (6 * (LAT + ROPE), Q_RANK) and w.dtype == torch.bfloat16 and w.is_contiguous()
    q = q_b.view(6, NOPE + ROPE, Q_RANK)
    want = torch.einsum("hnc,hnq->hcq", k_b.double(), q[:, :NOPE].double()).bfloat16()
    w = w.view(6, LAT + ROPE, Q_RANK)
    assert torch.equal(w[:, :LAT], want)
    assert torch.equal(w[:, LAT:], q[:, NOPE:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")
@pytest.mark.parametrize("absorbed", [False, True])
@pytest.mark.parametrize("rank", [0, 15])
@torch.inference_mode()
def test_headtp_attention_matches_the_fp32_composition(rank, absorbed):
    modeling = _mod("modeling")
    ring_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_ring")
    decode_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_decode")
    rope_op = importlib.import_module(_OPS + "mla_norm_rope").dflash2_mla_block_rope
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(rank)

    def rnd(*shape, scale=1.0):
        return (scale * torch.randn(shape, generator=g, device=dev)).bfloat16()

    # Whole weights; this rank's view as weights.py loads it (q_b / kv_b rows of its heads).
    q_b = rnd(HEADS * (NOPE + ROPE), Q_RANK, scale=0.03)
    kv_b = rnd(HEADS, NOPE + VAL, LAT, scale=0.05)
    mine = slice(6 * rank, 6 * rank + 6)
    hidden = 64
    w = {
        "d_kv_a": rnd(LAT + ROPE, hidden),
        "d_kv_a_norm": (1.0 + rnd(1, LAT, scale=0.1)),
        "d_fc": rnd(hidden, hidden),
        "d_final_norm": rnd(hidden),
        "d_l0_q_kv_a": rnd(Q_RANK + LAT + ROPE, hidden),
        "d_l0_q_a_norm": rnd(Q_RANK),
        "d_l0_q_b": q_b.view(HEADS, NOPE + ROPE, Q_RANK)[mine].reshape(-1, Q_RANK).contiguous(),
        "d_l0_kv_b": torch.cat(
            (kv_b[mine, :NOPE].reshape(-1, LAT), kv_b[mine, NOPE:].reshape(-1, LAT))
        ),
    }
    for key in ("in_norm", "post_norm", "o", "gate_up", "down") + tuple(
        f"{s}_conv_{p}" for s in ("attn", "mlp") for p in ("base", "proj")
    ):
        w[f"d_l0_{key}"] = rnd(1)
    L = modeling._draft_views(w, 1, NOPE, LAT).layers[0]
    assert L.k_b.shape == (6, NOPE, LAT)

    # Two requests on slots 3 and 1 (batch rows 0 and 1): one past the ring, one short.
    num_ctx = torch.tensor([5000, 700], dtype=torch.int64, device=dev)
    slot_table = torch.tensor([3, 1], dtype=torch.int64, device=dev)
    rows_idx = torch.arange(2, device=dev)
    ring = ring_mod.build_ring(1, 5, W, BLOCK, BLOCK, PAGE, LAT + ROPE, torch.float8_e4m3fn, dev)
    ring.data.view(torch.uint8).copy_(
        (0.3 * torch.randn(ring.data.shape, generator=g, device=dev))
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    ctx_positions = (num_ctx - BLOCK).view(-1, 1) + torch.arange(BLOCK, device=dev)
    step = ring_mod.ring_step(ring, slot_table, rows_idx, ctx_positions.reshape(-1), num_ctx)
    rows = 2 * BLOCK
    positions = (num_ctx.view(-1, 1) + torch.arange(BLOCK, device=dev)).reshape(-1)
    angles = (
        0.01
        * torch.arange(6000, device=dev).view(-1, 1)
        * torch.rand(1, ROPE // 2, generator=g, device=dev)
    )
    freqs = torch.polar(torch.ones_like(angles), angles).to(torch.complex64).contiguous()
    ws = decode_mod.headtp_decode_workspace(4, BLOCK, 6, W, dev)  # room for more requests
    d = SimpleNamespace(
        ring=ring,
        headtp=decode_mod.headtp_ranked_workspace(4, BLOCK, 6, dev),
        freqs=freqs,
        softmax_scale=0.0723,
        kv_scale=1.0,
        slot_table=slot_table,
        q_abs=(modeling._absorbed_query_weight(L.q_b, L.k_b),) if absorbed else None,
    )
    st = SimpleNamespace(num_ctx=num_ctx, positions=positions, block=BLOCK)
    qa = rnd(rows, Q_RANK)
    q_kv = torch.cat((qa, rnd(rows, LAT + ROPE)), dim=1)
    eps = 1e-5
    before = ring.pools[0][:, 0, 0].clone()

    got = modeling._draft_attention_headtp(L, 0, qa, q_kv, st, d, step, eps)
    torch.cuda.synchronize()
    assert got.shape == (rows, 6 * VAL)

    # The FP32 composition over the whole weights' heads of this rank.
    q = (qa.float() @ q_b.float().T).view(rows, HEADS, NOPE + ROPE)[:, mine]
    query = torch.zeros(rows, 6, LAT + ROPE, device=dev)
    query[..., :LAT] = torch.einsum("rhn,hnc->rhc", q[..., :NOPE], kv_b[mine, :NOPE].float())
    query, pool = rope_op.reference(
        q.bfloat16(),
        query.bfloat16(),
        q_kv[:, Q_RANK:],
        L.kv_a_norm,
        eps,
        freqs,
        positions,
        before,
        step.block_pages,
        step.block_offsets,
    )
    # The block latents in the ring: bitwise the block RoPE's reference.
    assert torch.equal(ring.pools[0][:, 0, 0].view(torch.uint8), pool.view(torch.uint8))
    ctx = decode_mod.rubin_ull_dflash2_headtp_decode.reference(
        query.bfloat16(), pool.view(-1, LAT + ROPE), step.slots, num_ctx, ring.rows,
        d.softmax_scale, 1.0, ws,
    )  # fmt: skip
    want = torch.einsum("rhc,hvc->rhv", ctx, kv_b[mine, NOPE:].float()).reshape(rows, -1)
    assert _rel(got, want) < 0.02, _rel(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")
@torch.inference_mode()
def test_the_ranked_merge_is_helix_mla_rounding_order():
    """The served headtp window (helix_mla's rounding order): per helix page group (page ``g`` of
    64 positions on CP rank ``g % 16``) the normalized partial rounded to BF16 and its log-sum-exp,
    W_UV per group, an FP32 merge rounded once; against that order in torch (bitwise-close), and
    against the fused decode."""
    decode_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_decode")
    ring_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_ring")
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(7)
    heads, cp, page = 6, decode_mod.RANKED_CP, decode_mod.RANKED_PAGE
    starts = torch.tensor([0, 63, 1017, 1088, 5000], dtype=torch.int64, device=dev)
    groups = starts.shape[0]
    rows = groups * BLOCK
    rr = ring_mod.ring_rows(W, BLOCK, BLOCK, PAGE)
    slots = torch.tensor([4, 0, 2, 1, 3], dtype=torch.int64, device=dev)
    ring = (0.5 * torch.randn(5 * rr, LAT + ROPE, generator=g, device=dev)).to(torch.float8_e4m3fn)
    query = (0.2 * torch.randn(rows, heads, LAT + ROPE, generator=g, device=dev)).bfloat16()
    v_b_t = (0.05 * torch.randn(heads, LAT, VAL, generator=g, device=dev)).bfloat16()
    scale = 0.0723
    v_b_t_view = (
        v_b_t.transpose(1, 2).contiguous().transpose(1, 2)
    )  # the drafter's W_UV is a transposed view
    rws = decode_mod.headtp_ranked_workspace(groups, BLOCK, heads, dev)
    part, lse = decode_mod.headtp_decode_ranked(
        query, ring, slots, starts, rr, scale, 1.0, W, v_b_t_view, rws
    )
    # The groups' normalized BF16 partials before W_UV: the same launch with a one-hot 512-wide W_UV
    # (every output is one exact product).
    eye = torch.eye(LAT, device=dev).bfloat16().expand(heads, LAT, LAT)
    rws_eye = decode_mod.headtp_ranked_workspace(groups, BLOCK, heads, dev, v=LAT)
    splits = decode_mod.RANKED_TILES[3]
    assert part.shape == (splits, cp, rows, heads, VAL) and part.dtype == (
        torch.bfloat16 if splits == 1 else torch.float32
    )
    o_split, lse_eye = decode_mod.headtp_decode_ranked(
        query, ring, slots, starts, rr, scale, 1.0, W, eye, rws_eye
    )
    assert torch.equal(lse_eye, lse)
    # Each split holds its column range of the one-hot products (exact), zeros elsewhere: their sum is the
    # BF16 partial (one split: the BF16 partial itself).
    o_rank = o_split.float().sum(0)
    assert torch.equal(o_rank, o_rank.bfloat16().float())
    # The merge alone on identical partials: the Triton LSE merge is torch's, bitwise.
    got = decode_mod.headtp_ranked_lse_merge(part, lse, BLOCK)
    assert torch.equal(got, decode_mod.headtp_ranked_merge(part, lse)), (
        "the merge differs from torch's"
    )
    # W_UV + merge against the torch reference on the same partials (W_UV per group, BF16, FP32 merge).
    ref_part = torch.einsum("crhl,hlv->crhv", o_rank.float(), v_b_t.float()).bfloat16()
    ref = decode_mod.headtp_ranked_merge(ref_part, lse)

    # Then only the FP32 order inside the 512-long dot differs: a group's BF16 partial may round 1 ulp
    # the other way, so an output is within one ulp of the largest group partial plus its own ulp
    # (BF16 ulp of x: |x| = m 2**e, m in [0.5, 1), ulp = 2**(e - 8)).
    def bf16_ulp(x):
        return torch.ldexp(torch.ones_like(x), torch.frexp(x).exponent - 8)

    peak_part = ref_part.float().abs().amax(0).reshape(ref.shape)
    bound = bf16_ulp(peak_part) + bf16_ulp(ref.float())
    diff = (got.float() - ref.float()).abs()
    off = (got != ref).float().mean().item()
    print(
        f"ranked Triton merge vs torch: {off:.4%} of outputs differ, max "
        f"{(diff / bf16_ulp(ref.float())).max().item():.1f} ulp of the output, "
        f"{(diff / bf16_ulp(peak_part)).max().item():.2f} ulp of the largest group partial"
    )
    assert bool((diff <= bound).all()), (diff / bound).max()
    assert off < 0.01, off

    # The order in torch: FP32 softmax per (row, group) over the group's visible keys.
    want_o = torch.zeros(cp, rows, heads, LAT, device=dev)
    want_lse = torch.full((cp, rows, heads), float("-inf"), device=dev)
    for r in range(rows):
        b, j = divmod(r, BLOCK)
        end = int(starts[b]) + j + 1
        pos = torch.arange(max(0, end - W), end, device=dev)
        keys = ring[int(slots[b]) * rr + pos % rr].float()
        logits = query[r].float() @ keys.T * scale
        for c in range(cp):
            sel = (pos // page) % cp == c
            if sel.any():
                lg = logits[:, sel]
                want_lse[c, r] = lg.logsumexp(-1)
                want_o[c, r] = lg.softmax(-1) @ keys[sel, :LAT]
    assert torch.equal(torch.isfinite(lse), torch.isfinite(want_lse)), "page groups differ"
    live = torch.isfinite(want_lse)
    assert (lse[live] - want_lse[live]).abs().max() < 2e-3
    assert _rel(o_rank.float()[live], want_o[live]) < 0.01
    want_part = torch.einsum("crhl,hlv->crhv", want_o.bfloat16().float(), v_b_t.float()).bfloat16()
    want = decode_mod.headtp_ranked_merge(want_part, want_lse)
    assert _rel(got.float(), want.float()) < 0.01, _rel(got.float(), want.float())

    # The fused decode (the served order) of the same window: the same attention up to rounding.
    ws = decode_mod.headtp_decode_workspace(groups, BLOCK, heads, W, dev)
    ctx = decode_mod.rubin_ull_dflash2_headtp_decode(query, ring, slots, starts, rr, scale, 1.0, ws)
    fused = torch.einsum("rhl,hlv->rhv", ctx.float(), v_b_t.float()).reshape(rows, -1)
    assert _rel(got.float(), fused) < 0.02, _rel(got.float(), fused)


# ---- the context overlap's join -----------------------------------------------------------------


def _late_store(side, ring_mod, ring, latents, step, cycles):
    """The context stream's verify store, late: ``cycles`` of sleep first; returns its event."""
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        torch.cuda._sleep(cycles)
        ring_mod.ring_store(ring, latents, step.verify_pages, step.verify_offsets)
        done = torch.cuda.Event()
        done.record(side)
    return done


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")
@torch.inference_mode()
def test_the_join_orders_the_verify_store_before_the_block_rope():
    """Under the context overlap the verify store runs on the context stream and layer 0 joins it
    right before the block RoPE (``_draft_attention_headtp(join=...)``). Two steps of one request
    back to back across the ring's wrap: accept 7 (all 8 verify rows kept, the block right after
    them), then accept 0 (one kept: the block overwrites 7 of the 8 verify rows just stored). With
    the store delayed on its stream, the joined call equals the serial one (store, then the call)
    bitwise in the output and the ring; the same delayed store joined after the call instead (the
    join moved past the block RoPE) differs, in both steps."""
    modeling = _mod("modeling")
    ring_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_ring")
    decode_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_decode")
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(7)

    def rnd(*shape, scale=1.0):
        return (scale * torch.randn(shape, generator=g, device=dev)).bfloat16()

    hidden = 64
    w = {
        "d_kv_a": rnd(LAT + ROPE, hidden),
        "d_kv_a_norm": (1.0 + rnd(1, LAT, scale=0.1)),
        "d_fc": rnd(hidden, hidden),
        "d_final_norm": rnd(hidden),
        "d_l0_q_kv_a": rnd(Q_RANK + LAT + ROPE, hidden),
        "d_l0_q_a_norm": rnd(Q_RANK),
        "d_l0_q_b": rnd(6 * (NOPE + ROPE), Q_RANK, scale=0.03),
        "d_l0_kv_b": rnd(6 * (NOPE + VAL), LAT, scale=0.05),
    }
    for key in ("in_norm", "post_norm", "o", "gate_up", "down") + tuple(
        f"{s}_conv_{p}" for s in ("attn", "mlp") for p in ("base", "proj")
    ):
        w[f"d_l0_{key}"] = rnd(1)
    L = modeling._draft_views(w, 1, NOPE, LAT).layers[0]

    def ring_like(src=None):
        ring = ring_mod.build_ring(
            1, 3, W, BLOCK, BLOCK, PAGE, LAT + ROPE, torch.float8_e4m3fn, dev
        )
        if src is None:
            ring.data.view(torch.uint8).copy_(
                (0.3 * torch.randn(ring.data.shape, generator=g, device=dev))
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
        else:
            ring.data.copy_(src.data)
        return ring

    serial = ring_like()
    joined, late = ring_like(serial), ring_like(serial)
    R = serial.rows
    angles = (
        0.01
        * torch.arange(2 * R + 64, device=dev).view(-1, 1)
        * torch.rand(1, ROPE // 2, generator=g, device=dev)
    )
    freqs = torch.polar(torch.ones_like(angles), angles).to(torch.complex64).contiguous()
    ranked_ws = decode_mod.headtp_ranked_workspace(1, BLOCK, 6, dev)
    slot_table = torch.tensor([2], dtype=torch.int64, device=dev)
    rows_idx = torch.arange(1, device=dev)
    side = torch.cuda.Stream()
    eps = 1e-5
    cycles = 200_000_000  # ~0.1 s: the store lands long after an unjoined call has finished

    committed = R - 3  # the verify rows [R - 3, R + 5) wrap
    for accepted in (8, 1):  # accept 7, then accept 0
        num_ctx = torch.tensor([committed + accepted], dtype=torch.int64, device=dev)
        ctx_positions = committed + torch.arange(BLOCK, device=dev)
        positions = num_ctx + torch.arange(BLOCK, device=dev)
        latents = rnd(BLOCK, 1, LAT + ROPE)
        qa = rnd(BLOCK, Q_RANK)
        q_kv = torch.cat((qa, rnd(BLOCK, LAT + ROPE)), dim=1)
        st = SimpleNamespace(num_ctx=num_ctx, positions=positions, block=BLOCK)
        out = {}
        for name, ring in (("serial", serial), ("joined", joined), ("late", late)):
            d = SimpleNamespace(
                ring=ring, headtp=ranked_ws, freqs=freqs, softmax_scale=0.0723, kv_scale=1.0,
                slot_table=slot_table,
            )  # fmt: skip
            step = ring_mod.ring_step(ring, slot_table, rows_idx, ctx_positions, num_ctx)
            if name == "serial":
                ring_mod.ring_store(ring, latents, step.verify_pages, step.verify_offsets)
                out[name] = modeling._draft_attention_headtp(L, 0, qa, q_kv, st, d, step, eps)
            elif name == "joined":
                join = _late_store(side, ring_mod, ring, latents, step, cycles)
                out[name] = modeling._draft_attention_headtp(
                    L, 0, qa, q_kv, st, d, step, eps, join=join
                )
            else:
                done = _late_store(side, ring_mod, ring, latents, step, cycles)
                out[name] = modeling._draft_attention_headtp(L, 0, qa, q_kv, st, d, step, eps)
                torch.cuda.current_stream().wait_event(done)
            torch.cuda.synchronize()
        assert torch.equal(out["joined"], out["serial"]), accepted
        assert torch.equal(joined.data.view(torch.uint8), serial.data.view(torch.uint8)), accepted
        # The check bites: the store joined past the call reads stale keys (and, at accept 0,
        # leaves rejected verify rows over the block's own latents).
        assert not torch.equal(out["late"], out["serial"]), accepted
        if accepted == 1:
            assert not torch.equal(late.data.view(torch.uint8), serial.data.view(torch.uint8))
        late.data.copy_(serial.data)
        committed += accepted


# ---- the Rubin row of the ranked partial ---------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")
def test_the_rubin_ranked_tiles_are_bitwise_the_defaults():
    """Each Rubin row of the ranked partial's tiles (modeling._DRAFT_RANKED_TILES_RUBIN by requests, and the rest
    row) writes the same partials and log-sum-exps as RANKED_TILES at the row's
    split count bit for bit (the QK dot over the 576 columns in other chunks, the same FP32 sums; the cap only
    moves registers), so the merged rows are the same too."""
    decode_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_decode")
    ring_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_ring")
    modeling = _mod("modeling")
    rows_rubin = [
        *modeling._DRAFT_RANKED_TILES_RUBIN.values(),
        modeling._DRAFT_RANKED_TILES_RUBIN_REST,
    ]
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(11)
    heads = 6
    starts = torch.tensor([0, 63, 1017, 1088, 190000], dtype=torch.int64, device=dev)
    groups = starts.shape[0]
    rr = ring_mod.ring_rows(W, BLOCK, BLOCK, PAGE)
    slots = torch.tensor([4, 0, 2, 1, 3], dtype=torch.int64, device=dev)
    ring = (0.5 * torch.randn(5 * rr, LAT + ROPE, generator=g, device=dev)).to(torch.float8_e4m3fn)
    query = (
        0.2 * torch.randn(groups * BLOCK, heads, LAT + ROPE, generator=g, device=dev)
    ).bfloat16()
    v_b_t = (0.05 * torch.randn(heads, LAT, VAL, generator=g, device=dev)).bfloat16()
    # The drafter's W_UV is a transposed view.
    v_b_t = v_b_t.transpose(1, 2).contiguous().transpose(1, 2)

    def bits(t):
        return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)

    for rubin in rows_rubin:
        outs = []
        for tiles in (tuple(decode_mod.RANKED_TILES[:3]) + (rubin[3], None), rubin):
            rws = decode_mod.headtp_ranked_workspace(groups, BLOCK, heads, dev)
            part, lse = decode_mod.headtp_decode_ranked(
                query, ring, slots, starts, rr, 0.0723, 1.0, W, v_b_t, rws, tiles
            )
            outs.append(
                (part.clone(), lse.clone(), decode_mod.headtp_ranked_lse_merge(part, lse, BLOCK))
            )
        for got, want in zip(outs[1], outs[0]):
            assert torch.equal(bits(got), bits(want)), rubin


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")
def test_the_column_splits_only_reorder_the_fp32_sums():
    """Splitting the latent columns over CTAs (tiles[3]) changes only the order of each group's FP32 W_UV sum:
    one split rounds the whole sum once (the order before the split), two add the two halves' sums. The
    log-sum-exps are bitwise the same; the merged rows within one BF16 ulp of the largest group partial plus
    their own, 1 and 2 requests (8 and 16 rows) at the GB300 tiles and every Rubin row."""
    decode_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_decode")
    ring_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_ring")
    modeling = _mod("modeling")
    rows_rubin = [
        *modeling._DRAFT_RANKED_TILES_RUBIN.values(),
        modeling._DRAFT_RANKED_TILES_RUBIN_REST,
    ]
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(5)
    heads = 6
    rr = ring_mod.ring_rows(W, BLOCK, BLOCK, PAGE)
    ring = (0.5 * torch.randn(5 * rr, LAT + ROPE, generator=g, device=dev)).to(torch.float8_e4m3fn)
    v_b_t = (0.05 * torch.randn(heads, LAT, VAL, generator=g, device=dev)).bfloat16()
    v_b_t = v_b_t.transpose(1, 2).contiguous().transpose(1, 2)

    def bf16_ulp(x):
        return torch.ldexp(torch.ones_like(x), torch.frexp(x).exponent - 8)

    for starts in ((5000,), (63, 190000)):
        groups = len(starts)
        st = torch.tensor(starts, dtype=torch.int64, device=dev)
        slots = torch.arange(groups, dtype=torch.int64, device=dev) + 2
        query = (
            0.2 * torch.randn(groups * BLOCK, heads, LAT + ROPE, generator=g, device=dev)
        ).bfloat16()
        for tiles in (decode_mod.RANKED_TILES, *rows_rubin):
            got = {}
            for splits in (1, 2):
                t = tuple(tiles[:3]) + (splits, tiles[4])
                rws = decode_mod.headtp_ranked_workspace(groups, BLOCK, heads, dev, splits=splits)
                part, lse = decode_mod.headtp_decode_ranked(
                    query, ring, slots, st, rr, 0.0723, 1.0, W, v_b_t, rws, t
                )
                merged = decode_mod.headtp_ranked_lse_merge(part, lse, BLOCK)
                assert torch.equal(merged, decode_mod.headtp_ranked_merge(part, lse)), (starts, t)
                got[splits] = (part.sum(0).bfloat16().float(), lse.clone(), merged.float())
            assert torch.equal(got[1][1], got[2][1]), (starts, tiles)
            peak = got[1][0].abs().amax(0).reshape(got[1][2].shape)
            bound = bf16_ulp(peak) + bf16_ulp(got[1][2])
            assert bool(((got[2][2] - got[1][2]).abs() <= bound).all()), (starts, tiles)


@pytest.mark.parametrize(
    "capability, sms, want",
    [((10, 7), 212, True), ((10, 3), 152, False), ((10, 3), 212, False), ((10, 7), 152, False)],
)
def test_the_rubin_rows_key_on_sm107_and_its_sm_count(monkeypatch, capability, sms, want):
    """The Rubin rows (modeling._rubin_rows) need compute capability 10.7 and more SMs than GB300_SMS: a part
    with more SMs alone does not take them."""
    modeling = _mod("modeling")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sms),
    )
    modeling._device_sms.cache_clear()
    modeling._rubin_rows.cache_clear()
    try:
        assert modeling._rubin_rows(torch.device("cuda", 0)) is want
    finally:
        modeling._device_sms.cache_clear()
        modeling._rubin_rows.cache_clear()


# ---- the ranked pair under PDL ---------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")
@pytest.mark.parametrize("absorbed", [False, True])
@torch.inference_mode()
def test_the_pdl_ranked_pair_equals_the_serialized_one(monkeypatch, absorbed):
    """The ranked pair launches with PDL: the partial waits for the block RoPE before any load, the merge for
    the partial, and each releases its successor after its wait. One served request (block 8, 6 heads, history
    past the ring) captured in a CUDA graph and replayed with fresh block inputs each time (new block latents
    over the previous replay's rows): the output and the ring equal, bitwise, the same call with each kernel
    finished before the next starts (a synchronize after the block RoPE and after the partial)."""
    modeling = _mod("modeling")
    ring_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_ring")
    decode_mod = importlib.import_module(_OPS + "rubin_ull_dflash2_headtp_decode")
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(11)

    def rnd(*shape, scale=1.0):
        return (scale * torch.randn(shape, generator=g, device=dev)).bfloat16()

    hidden = 64
    w = {
        "d_kv_a": rnd(LAT + ROPE, hidden),
        "d_kv_a_norm": (1.0 + rnd(1, LAT, scale=0.1)),
        "d_fc": rnd(hidden, hidden),
        "d_final_norm": rnd(hidden),
        "d_l0_q_kv_a": rnd(Q_RANK + LAT + ROPE, hidden),
        "d_l0_q_a_norm": rnd(Q_RANK),
        "d_l0_q_b": rnd(6 * (NOPE + ROPE), Q_RANK, scale=0.03),
        "d_l0_kv_b": rnd(6 * (NOPE + VAL), LAT, scale=0.05),
    }
    for key in ("in_norm", "post_norm", "o", "gate_up", "down") + tuple(
        f"{s}_conv_{p}" for s in ("attn", "mlp") for p in ("base", "proj")
    ):
        w[f"d_l0_{key}"] = rnd(1)
    L = modeling._draft_views(w, 1, NOPE, LAT).layers[0]

    def ring_like(src=None):
        ring = ring_mod.build_ring(
            1, 4, W, BLOCK, BLOCK, PAGE, LAT + ROPE, torch.float8_e4m3fn, dev
        )
        if src is None:
            ring.data.view(torch.uint8).copy_(
                (0.3 * torch.randn(ring.data.shape, generator=g, device=dev))
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
        else:
            ring.data.copy_(src.data)
        return ring

    early = ring_like()
    serial = ring_like(early)
    angles = (
        0.01
        * torch.arange(6000, device=dev).view(-1, 1)
        * torch.rand(1, ROPE // 2, generator=g, device=dev)
    )
    freqs = torch.polar(torch.ones_like(angles), angles).to(torch.complex64).contiguous()
    slot_table = torch.tensor([2], dtype=torch.int64, device=dev)
    num_ctx = torch.tensor([5000], dtype=torch.int64, device=dev)
    ctx_positions = num_ctx - BLOCK + torch.arange(BLOCK, device=dev)
    step = ring_mod.ring_step(
        early, slot_table, torch.arange(1, device=dev), ctx_positions, num_ctx
    )
    st = SimpleNamespace(
        num_ctx=num_ctx, positions=num_ctx + torch.arange(BLOCK, device=dev), block=BLOCK
    )
    eps = 1e-5

    q_abs = (modeling._absorbed_query_weight(L.q_b, L.k_b),) if absorbed else None

    def rt(ring):
        return SimpleNamespace(
            ring=ring, headtp=decode_mod.headtp_ranked_workspace(1, BLOCK, 6, dev), freqs=freqs,
            softmax_scale=0.0723, kv_scale=1.0, slot_table=slot_table, q_abs=q_abs,
        )  # fmt: skip

    d_early, d_serial = rt(early), rt(serial)
    qa, q_kv = rnd(BLOCK, Q_RANK), rnd(BLOCK, Q_RANK + LAT + ROPE)

    def call(d):
        return modeling._draft_attention_headtp(L, 0, qa, q_kv, st, d, step, eps)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        call(d_early)  # compiles; the ring rows it writes are overwritten by every call
    torch.cuda.current_stream().wait_stream(side)
    serial.data.copy_(early.data)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_early = call(d_early)

    rope, decode = modeling.dflash2_mla_block_rope, modeling.headtp_decode_ranked

    def rope_serial(*args):
        rope(*args)
        torch.cuda.synchronize()

    def decode_serial(*args):
        res = decode(*args)
        torch.cuda.synchronize()
        return res

    for _ in range(16):
        qa.copy_(rnd(BLOCK, Q_RANK))
        q_kv.copy_(rnd(BLOCK, Q_RANK + LAT + ROPE))
        graph.replay()
        with monkeypatch.context() as mp:
            mp.setattr(modeling, "dflash2_mla_block_rope", rope_serial)
            mp.setattr(modeling, "headtp_decode_ranked", decode_serial)
            want = call(d_serial)
        torch.cuda.synchronize()
        assert torch.equal(out_early, want)
        assert torch.equal(early.data.view(torch.uint8), serial.data.view(torch.uint8))
