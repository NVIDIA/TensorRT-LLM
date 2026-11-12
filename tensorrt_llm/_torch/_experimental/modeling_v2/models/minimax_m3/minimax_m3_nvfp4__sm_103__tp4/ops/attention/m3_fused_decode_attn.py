# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused MSA short-sequence decode attention of this target's sparse layers, and the plain-torch
composition it fuses (shared with the megakernel and c1 entries, which wrap the same attention)."""

from __future__ import annotations

import math
from typing import NamedTuple, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Per-rank attention geometry at tp4: 16 query heads, one KV head, one index head, head_dim 128.
NUM_HEADS = 16
HEAD_DIM = 128
#: The packed QKV(+index) projection row: [Q x16 | K | V | idxQ | idxK] x 128.
PACKED_WIDTH = (NUM_HEADS + 4) * HEAD_DIM
#: Partial NeoX RoPE over the first 64 dims of every head.
ROTARY_DIM = 64
#: One MSA block is one 128-token KV page; identity selection covers at most 16 of them.
PAGE = 128
MAX_BLOCKS = 16
MAX_SEQ_LEN = MAX_BLOCKS * PAGE
#: NVFP4 block: 16 elements per E4M3 scale.
NVFP4_BLOCK = 16
#: MXFP8 block of the o_proj hand-off: 32 elements per UE8M0 scale.
MXFP8_BLOCK = 32

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E4M3_MAX = 448.0


def e2m1_values(device) -> torch.Tensor:
    """The 16 E2M1 codes' values (sign in bit 3)."""
    pos = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)
    return torch.cat([pos, -pos])


def e4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 to E4M3 the way ``__nv_fp8_e4m3(float)`` does (RN-even, saturating to +-448), as raw bytes."""
    return x.clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_value(b: torch.Tensor) -> torch.Tensor:
    return b.view(torch.float8_e4m3fn).float()


def e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """``cvt.rn.satfinite.e2m1x2.f32`` per element: round-to-nearest-even onto {0, .5, 1, 1.5, 2, 3, 4, 6},
    saturating at 6, sign in bit 3 (a negative value that rounds to zero keeps its sign)."""
    a = x.abs()
    # Ties go to the even code: 0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0, 3.5 -> 4.0, 5 -> 4.
    code = (
        (a > 0.25).int()
        + (a >= 0.75).int()
        + (a > 1.25).int()
        + (a >= 1.75).int()
        + (a > 2.5).int()
        + (a >= 3.5).int()
        + (a > 5.0).int()
    )
    return (code | (torch.signbit(x).int() << 3)).to(torch.uint8)


def nvfp4_quantize(
    x: torch.Tensor, second_level_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The MiniMax-M3 producer's NVFP4 recipe for rows of 128: (packed data [N, 64] uint8, E4M3 scales [N, 8]).

    Elements are rounded to bf16 first; each 16-element block's scale is ``e4m3(sls * amax / 6)`` and the codes are
    ``e2m1(x / (scale / sls))``. Element ``2i`` is the low nibble of byte ``i``.
    """
    n = x.shape[0]
    xb = x.to(torch.bfloat16).float().view(n, HEAD_DIM // NVFP4_BLOCK, NVFP4_BLOCK)
    sls = second_level_scale.float().reshape(())
    amax = xb.abs().amax(-1)
    scale = e4m3_bytes(sls * (amax * (1.0 / 6.0)))
    out_scale = torch.where(
        amax != 0, 1.0 / (e4m3_value(scale) * (1.0 / sls)), torch.zeros_like(amax)
    )
    codes = e2m1_codes(xb * out_scale[..., None]).view(n, HEAD_DIM)
    data = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return data, scale


def nvfp4_dequantize(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[N, 64] packed E2M1 + [N, 8] E4M3 block scales -> fp32 [N, 128] (the per-tensor scale not applied)."""
    table = e2m1_values(data.device)
    lo = table[(data & 0xF).long()]
    hi = table[(data >> 4).long()]
    values = torch.stack([lo, hi], dim=-1).reshape(data.shape[0], HEAD_DIM)
    return values * e4m3_value(scale).repeat_interleave(NVFP4_BLOCK, dim=-1)


def v_scale_rows(scale_plane: torch.Tensor) -> torch.Tensor:
    """A page's V scale plane in the 4x4 token-quad order [token // 4][column][token % 4] -> [128, 8] row-major."""
    return (
        scale_plane.reshape(PAGE // 4, HEAD_DIM // NVFP4_BLOCK, 4)
        .permute(0, 2, 1)
        .reshape(PAGE, -1)
    )


def v_scale_plane(rows: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`v_scale_rows`: [128, 8] row-major scales -> the plane's [128, 8] storage."""
    return rows.reshape(PAGE // 4, 4, HEAD_DIM // NVFP4_BLOCK).permute(0, 2, 1).reshape(PAGE, -1)


def norm_rope(
    x: torch.Tensor, weight: torch.Tensor, eps: float, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Per-head Gemma RMSNorm (``x * rsqrt(mean(x^2) + eps) * (1 + w)``) then NeoX RoPE over the first 64 dims.

    x: fp32 [N, 128]; cos / sin: fp32 [N, 32] for each row's position."""
    y = x * (torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float()))
    x1, x2 = y[:, : ROTARY_DIM // 2], y[:, ROTARY_DIM // 2 : ROTARY_DIM]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, y[:, ROTARY_DIM:]], dim=-1)


def swizzled_scale_offsets(rows: int, k_blocks: int, device) -> torch.Tensor:
    """Byte offset of scale (row, k-block) in the 128x4-swizzled UE8M0 layout: [rows, k_blocks] int64."""
    r = torch.arange(rows, device=device)[:, None]
    k = torch.arange(k_blocks, device=device)[None, :]
    k_tiles = (k_blocks + 3) // 4
    return (
        (r // 128) * k_tiles * 512 + (k // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + k % 4
    )


def mxfp8_rows(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The merge's MXFP8 hand-off recipe on bf16 rows [T, K]: (E4M3 bytes [T, K], UE8M0 exponents [T, K / 32]).

    Per 32-element block: ``e = ceil_exp(amax / 448)`` (biased, at most 254), values ``e4m3(x * 2^(127 - e))``."""
    t, k = x.shape
    xf = x.float().view(t, k // MXFP8_BLOCK, MXFP8_BLOCK)
    amax = xf.abs().amax(-1)
    bits = (amax * (1.0 / _E4M3_MAX)).view(torch.int32)
    ex = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    ex = ex.clamp(max=254)
    scale = torch.where(amax != 0, ((254 - ex) << 23).view(torch.float32), torch.zeros_like(amax))
    q8 = e4m3_bytes(xf * scale[..., None]).view(t, k)
    return q8, ex.to(torch.uint8)


def mxfp8_dequantize(q8: torch.Tensor, ex: torch.Tensor) -> torch.Tensor:
    """E4M3 bytes [T, K] with UE8M0 exponents [T, K / 32] -> fp64 values."""
    scale = torch.exp2(ex.double() - 127.0).repeat_interleave(MXFP8_BLOCK, dim=-1)
    return e4m3_value(q8).double() * scale


def unswizzle_scales(swizzled: torch.Tensor, rows: int, k_blocks: int) -> torch.Tensor:
    """The [rows, k_blocks] UE8M0 bytes of a 128x4-swizzled scale buffer."""
    flat = swizzled.reshape(-1).view(torch.uint8)
    return flat[swizzled_scale_offsets(rows, k_blocks, flat.device)]


def mx_gemm(
    a: torch.Tensor, a_exp: torch.Tensor, w: torch.Tensor, w_exp: torch.Tensor
) -> torch.Tensor:
    """bf16 ``dequant(a) @ dequant(w)^T`` of MXFP8 operands (E4M3 bytes, row-major UE8M0 exponents), accumulated in
    fp64 and rounded once."""
    return (mxfp8_dequantize(a, a_exp) @ mxfp8_dequantize(w, w_exp).T).to(torch.bfloat16)


class AttnState(NamedTuple):
    """What the fused attention leaves behind: its bf16 output and the three caches after the new tokens' insert."""

    output: torch.Tensor
    kv_data: torch.Tensor
    kv_scale: torch.Tensor
    idx_k: torch.Tensor


def query_rows(packed, position_ids, q_norm_weight, eps, rotary_cos_sin) -> torch.Tensor:
    """fp32 [T, 16, 128]: the query heads of packed rows after Gemma RMSNorm + NeoX RoPE (before E4M3)."""
    rows = packed.float().view(packed.shape[0], NUM_HEADS + 4, HEAD_DIM)
    pos_ids = position_ids.long()
    cos = rotary_cos_sin[pos_ids, 0].float()
    sin = rotary_cos_sin[pos_ids, 1].float()
    return torch.stack(
        [norm_rope(rows[:, h], q_norm_weight, eps, cos, sin) for h in range(NUM_HEADS)], dim=1
    )


def attend_cached(
    q,
    kv_data,
    kv_scale,
    block_table,
    seq_lens,
    k_global_scale,
    v_global_scale,
    sm_scale,
    decode_query_len,
) -> torch.Tensor:
    """bf16 [T, 2048]: the identity-selection decode of fp32 queries ``q`` [T, 16, 128] over caches that already
    hold the step's tokens. q is rounded to E4M3 (no scale), the operand the kernel's tensor core reads; request
    ``r``'s token ``t`` sits at ``pos = seq_lens[r] - qlen + t`` and attends positions ``[0, pos]`` through
    ``block_table``: scores ``q . dequant(K) * sm_scale * k_global_scale``, softmax, values
    ``dequant(V) * v_global_scale``. fp64 throughout, one bf16 rounding at the end (the kernel's split-K partials
    and their LSE merge are exact algebra)."""
    q8 = e4m3_value(e4m3_bytes(q)).double()
    t_total = q.shape[0]
    qlen = int(decode_query_len)
    out = torch.empty(t_total, NUM_HEADS, HEAD_DIM, dtype=torch.float64, device=q.device)
    k_gs = float(k_global_scale.reshape(-1)[0])
    v_gs = float(v_global_scale.reshape(-1)[0])
    for r, seq_len in enumerate(seq_lens.tolist()):
        pages = block_table[r, : math.ceil(seq_len / PAGE)].tolist()
        kd = torch.cat([nvfp4_dequantize(kv_data[p, 0, 0], kv_scale[p, 0, 0]) for p in pages])
        vd = torch.cat(
            [nvfp4_dequantize(kv_data[p, 1, 0], v_scale_rows(kv_scale[p, 1, 0])) for p in pages]
        )
        kd, vd = kd.double() * k_gs, vd.double() * v_gs
        for t in range(qlen):
            row = r * qlen + t
            n = seq_len - qlen + t + 1
            p = torch.softmax(q8[row] @ kd[:n].T * sm_scale, dim=-1)
            out[row] = p @ vd[:n]
    return out.to(torch.bfloat16).view(t_total, NUM_HEADS * HEAD_DIM)


def reference_attention(
    packed,
    kv_data,
    kv_scale,
    idx_k,
    out_cache_loc,
    position_ids,
    block_table,
    seq_lens,
    kv_quant_scale,
    k_global_scale,
    v_global_scale,
    q_norm_weight,
    k_norm_weight,
    index_k_norm_weight,
    rotary_cos_sin,
    eps,
    sm_scale,
    decode_query_len,
) -> AttnState:
    """The composition the fused kernel replaces, in plain torch, on copies of the caches.

    1. Per token, q heads / k / index-k: Gemma RMSNorm + NeoX RoPE at ``position_ids``; v as is.
    2. k / v -> NVFP4 (``kv_quant_scale[1]`` / ``[2]``) into the token's ``out_cache_loc`` slot (K scales linear, V
       scales in 4x4 token quads); index-k -> E4M3 into the index cache.
    3. :func:`attend_cached` over the caches after the insert.
    """
    kv_data, kv_scale, idx_k = kv_data.clone(), kv_scale.clone(), idx_k.clone()
    idx_bytes = idx_k.view(torch.uint8)
    t_total = packed.shape[0]
    rows = packed.float().view(t_total, NUM_HEADS + 4, HEAD_DIM)
    pos_ids = position_ids.long()
    cos = rotary_cos_sin[pos_ids, 0].float()
    sin = rotary_cos_sin[pos_ids, 1].float()
    q = query_rows(packed, position_ids, q_norm_weight, eps, rotary_cos_sin)
    k = norm_rope(rows[:, NUM_HEADS], k_norm_weight, eps, cos, sin)
    v = rows[:, NUM_HEADS + 1]
    ik = norm_rope(rows[:, NUM_HEADS + 3], index_k_norm_weight, eps, cos, sin)
    k_data, k_sf = nvfp4_quantize(k, kv_quant_scale[1])
    v_data, v_sf = nvfp4_quantize(v, kv_quant_scale[2])
    ik_bytes = e4m3_bytes(ik)
    for t, slot in enumerate(out_cache_loc[:t_total].tolist()):
        page, row = divmod(slot, PAGE)
        kv_data[page, 0, 0, row] = k_data[t]
        kv_data[page, 1, 0, row] = v_data[t]
        kv_scale[page, 0, 0, row] = k_sf[t]
        vrows = v_scale_rows(kv_scale[page, 1, 0])
        vrows[row] = v_sf[t]
        kv_scale[page, 1, 0] = v_scale_plane(vrows)
        idx_bytes[page, 0, row] = ik_bytes[t]
    output = attend_cached(
        q,
        kv_data,
        kv_scale,
        block_table,
        seq_lens,
        k_global_scale,
        v_global_scale,
        sm_scale,
        decode_query_len,
    )
    return AttnState(output, kv_data, kv_scale, idx_k)


def check_attention_inputs(
    packed_rows,
    kv_data,
    kv_scale,
    idx_k,
    out_cache_loc,
    position_ids,
    block_table,
    seq_lens,
    q_norm_weight,
    k_norm_weight,
    index_k_norm_weight,
    rotary_cos_sin,
    decode_query_len,
) -> None:
    """The inputs the attention kernels accept and answer wrongly (shared by the three entries' ``is_valid``)."""
    assert (
        kv_data.dim() == 5
        and tuple(kv_data.shape[1:]) == (2, 1, PAGE, HEAD_DIM // 2)
        and kv_data.stride(4) == 1
        and kv_data.stride(3) == HEAD_DIM // 2
    ), "kv_data must be [pages, 2, 1, 128, 64] with contiguous head blocks"
    assert (
        kv_scale.dim() == 5
        and tuple(kv_scale.shape[1:]) == (2, 1, PAGE, HEAD_DIM // NVFP4_BLOCK)
        and kv_scale.stride(4) == 1
        and kv_scale.stride(3) == HEAD_DIM // NVFP4_BLOCK
    ), "kv_scale must be [pages, 2, 1, 128, 8] with contiguous head blocks"
    assert (
        idx_k.dim() == 4 and tuple(idx_k.shape[1:]) == (1, PAGE, HEAD_DIM) and idx_k.stride(3) == 1
    ), "idx_k must be [pages, 1, 128, 128] with contiguous rows"
    for name, w in (
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("index_k_norm_weight", index_k_norm_weight),
    ):
        assert w.is_contiguous(), f"{name} must be contiguous (read as 4-element vectors)"
    assert rotary_cos_sin.is_contiguous() and tuple(rotary_cos_sin.shape[1:]) == (
        2,
        ROTARY_DIM // 2,
    ), "rotary_cos_sin must be contiguous [max_pos, 2, 32]"
    for name, t in (
        ("out_cache_loc", out_cache_loc),
        ("position_ids", position_ids),
        ("seq_lens", seq_lens),
    ):
        assert t.is_contiguous(), f"{name} must be contiguous"
    qlen = int(decode_query_len)
    num_reqs = seq_lens.shape[0]
    assert packed_rows == num_reqs * qlen, "one row per (request, query token)"
    assert position_ids.numel() == packed_rows, "one position per token"
    lens = seq_lens.tolist()
    assert all(qlen <= n <= MAX_SEQ_LEN for n in lens), (
        f"every seq_len must be in [decode_query_len, {MAX_SEQ_LEN}]: identity selection covers 16 blocks of 128, "
        "a longer request is silently cut to them"
    )
    assert block_table.shape[1] >= math.ceil(max(lens) / PAGE), (
        "block_table must hold every attended block (missing columns read page 0)"
    )
    assert int(position_ids.max()) < rotary_cos_sin.shape[0], "position past the rotary table"
    table = block_table.tolist()
    slots = out_cache_loc.tolist()
    for r, n in enumerate(lens):
        for t in range(qlen):
            pos = n - qlen + t
            want = table[r][pos // PAGE] * PAGE + pos % PAGE
            assert slots[r * qlen + t] == want, (
                "out_cache_loc must be the slot block_table gives the token's position seq_len - qlen + t: the "
                "attention reads that page's other rows from the cache it loaded before the insert"
            )


class M3FusedDecodeAttn(OpWrapper):
    """One launch after the packed QKV GEMM of a sparse layer in short-sequence decode.

    Fuses q / k / index-k Gemma RMSNorm + NeoX partial RoPE, the NVFP4 K/V and FP8 index-K cache insert, the
    block-sparse decode over every attended block (selection is the identity: at most 16 blocks of 128 tokens) and
    the split-K LSE merge, writing the bf16 output and its MXFP8 copy for the o_proj GEMM.

    ``(tokens_per_cluster, blocks_per_cta)`` is the launch geometry; the target passes
    ``ops.m3_fused_attn.default_geometry(qlen, num_requests)``. Returns None: the cells read ``output``,
    ``mxfp8_data`` / ``mxfp8_scale`` and the three caches.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Every compiled geometry, each at a step shape the target routes to it (decode_query_len x requests, at most
    # 16 tokens). seq_lens is a per-request list; positions are seq_len - qlen + t.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="plain decode, one request, a few blocks: geometry (1, 1, 1)",
            spec=dict(qlen=1, seq_lens=(300,), geometry=(1, 1)),
        ),
        Cell(
            why="plain decode, 6 requests: the widest one-block-per-CTA batch; full 16 blocks and a page's 1st row",
            spec=dict(qlen=1, seq_lens=(2048, 129, 1, 700, 1500, 256), geometry=(1, 1)),
        ),
        Cell(
            why="plain decode, 8 requests: KV blocks paired per CTA (8-CTA clusters), geometry (1, 1, 2)",
            spec=dict(qlen=1, seq_lens=(2048, 33, 128, 129, 1024, 1900, 640, 7), geometry=(1, 2)),
        ),
        Cell(
            why="plain decode at the 16-token cap: 16 paired-block clusters",
            spec=dict(qlen=1, seq_lens=tuple(range(97, 2048, 125))[:16], geometry=(1, 2)),
        ),
        Cell(
            why="Eagle3 verify of 1 draft (qlen 2), 2 requests: geometry (2, 1, 1); a draft crossing a page",
            spec=dict(qlen=2, seq_lens=(129, 900), geometry=(1, 1)),
        ),
        Cell(
            why="qlen 2 at 8 requests: geometry (2, 1, 2)",
            spec=dict(qlen=2, seq_lens=(2, 2048, 130, 257, 640, 1000, 1500, 64), geometry=(1, 2)),
        ),
        Cell(
            why="qlen 3 at 5 requests (15 tokens): geometry (3, 1, 1)",
            spec=dict(qlen=3, seq_lens=(3, 2048, 130, 385, 777), geometry=(1, 1)),
        ),
        Cell(
            why="qlen 4 at 4 requests: geometry (4, 1, 1)",
            spec=dict(qlen=4, seq_lens=(4, 2048, 131, 1030), geometry=(1, 1)),
        ),
        Cell(
            why="qlen 5 at 3 requests: geometry (5, 1, 1)",
            spec=dict(qlen=5, seq_lens=(5, 1100, 2048), geometry=(1, 1)),
        ),
        Cell(
            why="qlen 6, one request (the c1 shape when c1 does not apply): geometry (6, 1, 1)",
            spec=dict(qlen=6, seq_lens=(1283,), geometry=(1, 1)),
        ),
        Cell(
            why="qlen 6 at 2 requests: geometry (6, 1, 2)",
            spec=dict(qlen=6, seq_lens=(6, 2048), geometry=(1, 2)),
        ),
        Cell(
            why="qlen 7, two requests: geometry (7, 1, 1)",
            spec=dict(qlen=7, seq_lens=(132, 2000), geometry=(1, 1)),
        ),
        Cell(
            why="qlen 8 at 2 requests (the 16-token cap): geometry (8, 1, 1)",
            spec=dict(qlen=8, seq_lens=(8, 1029), geometry=(1, 1)),
        ),
        # Every other (query rows per request, requests) the decode dispatch sends here (dispatch.FUSED_DECODE_ATTN):
        # lengths spread over the 16-block window, the geometry the target picks for the shape.
        *(
            Cell(
                why=f"qlen {qlen} at {reqs} requests: a decode shape between the ones above",
                spec=dict(
                    qlen=qlen,
                    seq_lens=tuple(
                        qlen + (r * 389 + 97 * qlen) % (2049 - qlen) for r in range(reqs)
                    ),
                    geometry=geometry,
                ),
            )
            for qlen, reqs, geometry in (
                (1, 2, (1, 1)),
                (1, 3, (1, 1)),
                (1, 4, (1, 1)),
                (1, 5, (1, 1)),
                (1, 7, (1, 2)),
                (1, 9, (1, 2)),
                (1, 10, (1, 2)),
                (1, 11, (1, 2)),
                (1, 12, (1, 2)),
                (1, 13, (1, 2)),
                (1, 14, (1, 2)),
                (1, 15, (1, 2)),
                (2, 1, (1, 1)),
                (2, 3, (1, 1)),
                (2, 4, (1, 2)),
                (2, 5, (1, 2)),
                (2, 6, (1, 2)),
                (2, 7, (1, 2)),
                (3, 1, (1, 1)),
                (3, 2, (1, 1)),
                (3, 3, (1, 1)),
                (3, 4, (1, 1)),
                (4, 1, (1, 1)),
                (4, 2, (1, 1)),
                (4, 3, (1, 1)),
                (5, 1, (1, 1)),
                (5, 2, (1, 1)),
                (7, 1, (1, 1)),
                (8, 1, (1, 1)),
            )
        ),
    )

    note = """
    Only the variants the target launches are compiled: the geometries above, each with the MXFP8 hand-off (the
    target always passes it). Q is quantized to E4M3 with no scale (`kv_quant_scale[0]` is not read): its operand
    on the tensor core is the E4M3 value, which `reference` reproduces.

    Not certified: a KV pool whose pages interleave other layers (any `stride(0)` / `stride(1)` that is a multiple
    of 16 is accepted; the cells use a contiguous pool), `position_ids` other than `seq_len - qlen + t` (RoPE reads
    `position_ids`, placement reads `seq_lens`; the target passes equal values), and `use_pdl=False` timing.
    """

    def __call__(
        self,
        packed: torch.Tensor,
        kv_data: torch.Tensor,
        kv_scale: torch.Tensor,
        idx_k: torch.Tensor,
        out_cache_loc: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_quant_scale: torch.Tensor,
        k_global_scale: torch.Tensor,
        v_global_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        index_k_norm_weight: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        eps: float,
        sm_scale: float,
        decode_query_len: int,
        tokens_per_cluster: int,
        blocks_per_cta: int,
        output: torch.Tensor,
        mxfp8_data: torch.Tensor,
        mxfp8_scale: torch.Tensor,
        use_pdl: bool,
    ) -> None:
        torch.ops.trtllm.minimax_m3_fused_decode_attn(
            packed,
            kv_data,
            kv_scale,
            idx_k,
            out_cache_loc,
            position_ids,
            block_table,
            seq_lens,
            kv_quant_scale,
            k_global_scale,
            v_global_scale,
            q_norm_weight,
            k_norm_weight,
            index_k_norm_weight,
            rotary_cos_sin,
            eps,
            sm_scale,
            decode_query_len,
            tokens_per_cluster,
            blocks_per_cta,
            output,
            mxfp8_data,
            mxfp8_scale,
            use_pdl,
        )

    def reference(
        self,
        packed: torch.Tensor,
        kv_data: torch.Tensor,
        kv_scale: torch.Tensor,
        idx_k: torch.Tensor,
        out_cache_loc: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_quant_scale: torch.Tensor,
        k_global_scale: torch.Tensor,
        v_global_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        index_k_norm_weight: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        eps: float,
        sm_scale: float,
        decode_query_len: int,
        tokens_per_cluster: int,
        blocks_per_cta: int,
        output: torch.Tensor,
        mxfp8_data: torch.Tensor,
        mxfp8_scale: torch.Tensor,
        use_pdl: bool,
    ) -> AttnState:
        """The bf16 output and the caches after the insert, as :func:`reference_attention` computes them.

        The geometry, the output buffers and ``use_pdl`` do not change what is computed and are ignored; the
        MXFP8 hand-off is :func:`mxfp8_rows` of the bf16 output, which the test checks on the op's own output.
        """
        return reference_attention(
            packed,
            kv_data,
            kv_scale,
            idx_k,
            out_cache_loc,
            position_ids,
            block_table,
            seq_lens,
            kv_quant_scale,
            k_global_scale,
            v_global_scale,
            q_norm_weight,
            k_norm_weight,
            index_k_norm_weight,
            rotary_cos_sin,
            eps,
            sm_scale,
            decode_query_len,
        )

    def is_valid(
        self,
        packed: torch.Tensor,
        kv_data: torch.Tensor,
        kv_scale: torch.Tensor,
        idx_k: torch.Tensor,
        out_cache_loc: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_quant_scale: torch.Tensor,
        k_global_scale: torch.Tensor,
        v_global_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        index_k_norm_weight: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        eps: float,
        sm_scale: float,
        decode_query_len: int,
        tokens_per_cluster: int,
        blocks_per_cta: int,
        output: torch.Tensor,
        mxfp8_data: torch.Tensor,
        mxfp8_scale: torch.Tensor,
        use_pdl: bool,
    ) -> None:
        check_attention_inputs(
            packed.shape[0],
            kv_data,
            kv_scale,
            idx_k,
            out_cache_loc,
            position_ids,
            block_table,
            seq_lens,
            q_norm_weight,
            k_norm_weight,
            index_k_norm_weight,
            rotary_cos_sin,
            decode_query_len,
        )
        k_blocks = output.shape[1] // MXFP8_BLOCK
        needed = math.ceil(packed.shape[0] / 128) * ((k_blocks + 3) // 4) * 512
        assert mxfp8_scale.numel() >= needed, (
            "mxfp8_scale must hold the swizzled scales of every output row tile"
        )


m3_fused_decode_attn = M3FusedDecodeAttn()
