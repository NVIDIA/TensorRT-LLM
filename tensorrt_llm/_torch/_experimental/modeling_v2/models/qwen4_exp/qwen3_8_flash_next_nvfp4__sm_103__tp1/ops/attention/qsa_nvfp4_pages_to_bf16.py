# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A batch's pages of an NVFP4 paged K/V pool, dequantized into a bf16 scratch pool, and the
block table that addresses them there.

Tier 5 (Triton written for this target). The target's context launches over an NVFP4 pool
read their requests' pages through this scratch with the bf16 row-tile kernel.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .qsa_paged_sparse_gqa_nvfp4 import _e2m1_to_f32

#: Values per e4m3 block scale, e2m1 values per byte, tokens per V scale group.
_SF_VEC = 16
_PER_BYTE = 2
_V_SF_QUAD = 4
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: This checkpoint's K/V geometry and the engine's QSA table (`width` pages a slot);
#: `slot_pages` role pages of one lifecycle slot hold the call's planes, layer `layer`'s K
#: and V among them (the trunk pool holds 24).
_CHECKPOINT = dict(
    kv_heads=2,
    head_dim=256,
    tokens_per_block=32,
    width=320,
    slots=40,
    global_k=1.0,
    global_v=1.0,
    slot_pages=2,
    layer=0,
    holes=False,
)


def _spec(kv_lens, **overrides) -> dict:
    """`kv_lens` is one KV length per listed request, in order."""
    return {**_CHECKPOINT, "kv_lens": tuple(kv_lens), **overrides}


@triton.jit
def _dequantize_head_page(
    data,
    scales,
    page,
    head,
    global_scale,
    valid,
    d_stride_page,
    d_stride_head,
    d_stride_token,
    s_stride_page,
    s_stride_head,
    GROUPED: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """[tpb, D] fp32 values of one head page: e2m1 x e4m3 block scale x second-level
    scale, in the sparse GQA kernels' order."""
    tokens = tl.arange(0, TOKENS_PER_BLOCK)[:, None]
    dims = tl.arange(0, HEAD_DIM)[None, :]
    packed = tl.load(
        data + page * d_stride_page + head * d_stride_head + tokens * d_stride_token + dims // 2,
        mask=valid,
        other=0,
    ).to(tl.int32)
    codes = (packed >> ((dims % 2) * 4)) & 0xF
    if GROUPED:
        # V: `[tpb / 4, D / 16, 4]` per head page.
        offsets = (tokens // 4) * (HEAD_DIM // 4) + (dims // 16) * 4 + tokens % 4
    else:
        # K: token-major `[tpb, D / 16]` per head page.
        offsets = tokens * (HEAD_DIM // 16) + dims // 16
    block_scales = (
        tl.load(scales + page * s_stride_page + head * s_stride_head + offsets, mask=valid, other=0)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    return _e2m1_to_f32(codes) * block_scales * tl.load(global_scale)


@triton.jit
def _nvfp4_pages_to_bf16_kernel(
    k_cache,
    v_cache,
    k_scales,
    v_scales,
    kv_scale_quant_orig,
    block_table,
    slots,
    counts,
    offsets,
    table,
    k_out,
    v_out,
    num_pool_pages,
    listed,
    d_stride_page,
    d_stride_head,
    d_stride_token,
    s_stride_page,
    s_stride_head,
    bt_stride,
    table_stride,
    o_stride_page,
    o_stride_head,
    o_stride_token,
    NUM_KV_HEADS: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    request = tl.program_id(0)
    page = tl.program_id(1)
    slot = tl.load(slots + request).to(tl.int64)
    dst = tl.load(offsets + request).to(tl.int64) + page
    listed_page = (page < tl.load(counts + request)) & (dst < listed)
    src = tl.load(block_table + slot * bt_stride + page, mask=listed_page, other=-1).to(tl.int64)
    resident = listed_page & (src >= 0) & (src < num_pool_pages)
    safe = tl.where(resident, src, 0)
    tl.store(
        table + slot * table_stride + page,
        tl.where(resident, dst, -1).to(tl.int32),
        mask=listed_page,
    )
    tokens = tl.arange(0, TOKENS_PER_BLOCK)[:, None]
    dims = tl.arange(0, HEAD_DIM)[None, :]
    out_offsets = dst * o_stride_page + tokens * o_stride_token + dims
    for head in tl.static_range(NUM_KV_HEADS):
        k = _dequantize_head_page(
            k_cache,
            k_scales,
            safe,
            head,
            kv_scale_quant_orig + 1,
            resident,
            d_stride_page,
            d_stride_head,
            d_stride_token,
            s_stride_page,
            s_stride_head,
            False,
            TOKENS_PER_BLOCK,
            HEAD_DIM,
        )
        v = _dequantize_head_page(
            v_cache,
            v_scales,
            safe,
            head,
            kv_scale_quant_orig + 2,
            resident,
            d_stride_page,
            d_stride_head,
            d_stride_token,
            s_stride_page,
            s_stride_head,
            True,
            TOKENS_PER_BLOCK,
            HEAD_DIM,
        )
        out = out_offsets + head * o_stride_head
        tl.store(k_out + out, k.to(k_out.dtype.element_ty), mask=listed_page)
        tl.store(v_out + out, v.to(v_out.dtype.element_ty), mask=listed_page)


def _dequantize_pages(data, scales, pages, global_scale, grouped: bool) -> torch.Tensor:
    """[n, Hkv, tpb, D] fp32 values of `pages` (in the pool) of one role plane."""
    heads, tokens_per_block, packed = data.shape[1], data.shape[2], data.shape[3]
    head_dim, groups = packed * _PER_BYTE, packed * _PER_BYTE // _SF_VEC
    raw = data[pages].long()  # [n, H, tpb, D/2]
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=data.device)
    lut = torch.cat([lut, -lut])
    values = torch.stack([lut[raw & 0xF], lut[raw >> 4]], dim=-1).flatten(-2)
    flat = scales[pages].reshape(pages.shape[0], heads, tokens_per_block * groups)
    tokens = torch.arange(tokens_per_block, device=data.device)[:, None]
    dims = torch.arange(head_dim, device=data.device)[None, :]
    if grouped:
        columns = (tokens // _V_SF_QUAD) * (_V_SF_QUAD * groups) + (dims // _SF_VEC) * _V_SF_QUAD
        columns = columns + tokens % _V_SF_QUAD
    else:
        columns = tokens * groups + dims // _SF_VEC
    sf = flat[:, :, columns.flatten()].reshape(pages.shape[0], heads, tokens_per_block, head_dim)
    return values * sf.view(torch.float8_e4m3fn).float() * global_scale


class QsaNvfp4PagesToBf16(OpWrapper):
    """K and V of listed requests' pages of an NVFP4 HND paged pool, as bf16 scratch pools,
    and the block table that addresses them there.

    Request `r` lists its first `counts[r]` table pages, `block_table[slots[r], j]` for `j
    < counts[r]`, at scratch pages `offsets[r] + j` (those below `listed`, the scratch's
    page count)::

        k_out[offsets[r] + j, h, t, d] = bf16(e2m1(K code at (s, h, t, d)) x
                                              e4m3(K scale of (t, d // 16)) x g_k)

    with `s = block_table[slots[r], j]` and `g_k = kv_scale_quant_orig[1]`, `v_out` likewise
    from the V planes and `g_v = kv_scale_quant_orig[2]`. K's block scales are token-major
    `[tpb, D/16]` per head page, V's in four-token groups `[tpb/4, D/16, 4]`. The product is
    formed in fp32 in the sparse GQA kernels' order and rounded once to bf16, so a bf16
    kernel over the scratch reads the values those kernels dequantize. `table[slots[r], j]`
    becomes `offsets[r] + j`, or -1 where the pool's entry is unmapped or out of the pool
    (whose scratch page is then zeros); the call writes no other entry of `table`, which the
    caller fills with -1. Returns `(k_out, v_out, table)`.

    Fusion boundary: the listing's dequantize and table; choosing the requests and their
    page counts (and `offsets`, their exclusive prefix sum) is the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one request, one page", spec=_spec((17,))),
        Cell(why="con 1 prefill: one 8192-token request", spec=_spec((8192,))),
        Cell(why="max_num_tokens: two 8192-token requests", spec=_spec((8192, 8192))),
        Cell(
            why="three requests, a chunked continuation and one filling the table",
            spec=_spec((5000 + 3192, 3000, 10240)),
        ),
        Cell(why="many short requests", spec=_spec(tuple(33 + 61 * i for i in range(30)))),
        Cell(why="holes: unmapped and out-of-pool entries", spec=_spec((4000, 2000), holes=True)),
        Cell(
            why="second-level scales g_k 0.37, g_v 2.5: the product rounds to bf16",
            spec=_spec((1500, 700), global_k=0.37, global_v=2.5),
        ),
        Cell(
            why="the engine's trunk layout: layer 11's planes among the 24 role pages",
            spec=_spec((3000, 900), slot_pages=24, layer=11),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. Call site: the target's
    context/mixed forward over an NVFP4 pool, once per attention layer, for the step's
    context requests (`_nvfp4_context_plan`: slots from the context rows' request
    indices, counts `ceil(kv_len / tpb)` from the device KV lengths, `listed` from the
    host ones); the row-tile kernel then reads the scratch through `table`.

    One Triton kernel, a program per (request, table page), both K/V heads and both
    roles: a head page is 32 tokens x 256 dims, 8 warps. Programs past a request's
    count store nothing.

    Preconditions: uint8 planes `[pages, Hkv, tpb, D/2]` (data) and `[pages, Hkv, tpb,
    D/16]` (scales), K's and V's at equal strides, unit last-dim stride and contiguous
    head pages (the QSA manager's role planes); `kv_scale_quant_orig` an fp32 [3];
    `table` an int32 tensor of `block_table`'s shape. Bitwise deterministic.
    """

    def __call__(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale_cache: torch.Tensor,
        v_scale_cache: torch.Tensor,
        kv_scale_quant_orig: torch.Tensor,
        block_table: torch.Tensor,
        slots: torch.Tensor,
        counts: torch.Tensor,
        offsets: torch.Tensor,
        table: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        requests, listed = slots.shape[0], k_out.shape[0]
        if requests and listed:
            _nvfp4_pages_to_bf16_kernel[(requests, block_table.shape[1])](
                k_cache,
                v_cache,
                k_scale_cache,
                v_scale_cache,
                kv_scale_quant_orig,
                block_table,
                slots,
                counts,
                offsets,
                table,
                k_out,
                v_out,
                k_cache.shape[0],
                listed,
                k_cache.stride(0),
                k_cache.stride(1),
                k_cache.stride(2),
                k_scale_cache.stride(0),
                k_scale_cache.stride(1),
                block_table.stride(0),
                table.stride(0),
                k_out.stride(0),
                k_out.stride(1),
                k_out.stride(2),
                NUM_KV_HEADS=k_cache.shape[1],
                TOKENS_PER_BLOCK=k_cache.shape[2],
                HEAD_DIM=k_cache.shape[3] * _PER_BYTE,
                num_warps=8,
            )
        return k_out, v_out, table

    def reference(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale_cache: torch.Tensor,
        v_scale_cache: torch.Tensor,
        kv_scale_quant_orig: torch.Tensor,
        block_table: torch.Tensor,
        slots: torch.Tensor,
        counts: torch.Tensor,
        offsets: torch.Tensor,
        table: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_ref, v_ref, table_ref = k_out.clone(), v_out.clone(), table.clone()
        listed = k_out.shape[0]
        for r in range(slots.shape[0]):
            slot, base = int(slots[r]), int(offsets[r])
            for j in range(min(int(counts[r]), block_table.shape[1])):
                dst = base + j
                if dst >= listed:
                    break
                src = int(block_table[slot, j])
                resident = 0 <= src < k_cache.shape[0]
                table_ref[slot, j] = dst if resident else -1
                page = torch.tensor([src if resident else 0], device=k_cache.device)
                for out, data, sc, g, grouped in (
                    (k_ref, k_cache, k_scale_cache, kv_scale_quant_orig[1], False),
                    (v_ref, v_cache, v_scale_cache, kv_scale_quant_orig[2], True),
                ):
                    vals = _dequantize_pages(data, sc, page, g, grouped)[0]
                    out[dst] = (vals if resident else torch.zeros_like(vals)).to(out.dtype)
        return k_ref, v_ref, table_ref

    def is_valid(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale_cache: torch.Tensor,
        v_scale_cache: torch.Tensor,
        kv_scale_quant_orig: torch.Tensor,
        block_table: torch.Tensor,
        slots: torch.Tensor,
        counts: torch.Tensor,
        offsets: torch.Tensor,
        table: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
    ) -> None:
        pages, heads, tokens_per_block, packed = k_cache.shape
        head_dim = packed * _PER_BYTE
        assert tokens_per_block % _V_SF_QUAD == 0, "tokens_per_block: a multiple of 4"
        assert head_dim % _SF_VEC == 0 and math.log2(head_dim).is_integer(), "head_dim"
        for name, plane, width in (
            ("k_cache", k_cache, packed),
            ("v_cache", v_cache, packed),
            ("k_scale_cache", k_scale_cache, head_dim // _SF_VEC),
            ("v_scale_cache", v_scale_cache, head_dim // _SF_VEC),
        ):
            assert plane.dtype == torch.uint8, f"{name}: uint8"
            assert tuple(plane.shape) == (pages, heads, tokens_per_block, width), f"{name}: shape"
            assert plane.stride(3) == 1, f"{name}: unit last-dim stride"
        assert v_cache.stride() == k_cache.stride(), "K and V data planes at equal strides"
        assert v_scale_cache.stride() == k_scale_cache.stride(), "K and V scales at equal strides"
        assert k_scale_cache.stride(2) == head_dim // _SF_VEC, "scales: contiguous head pages"
        assert kv_scale_quant_orig.dtype == torch.float32 and kv_scale_quant_orig.shape == (3,)
        assert block_table.dim() == 2 and block_table.dtype == torch.int32
        assert block_table.stride(1) == 1, "block_table: row-contiguous"
        assert table.shape == block_table.shape and table.dtype == torch.int32
        assert table.stride(1) == 1, "table: row-contiguous"
        for name, t in (("slots", slots), ("counts", counts), ("offsets", offsets)):
            assert t.dim() == 1 and t.shape == slots.shape, f"{name}: one per request"
            assert t.dtype in (torch.int32, torch.int64), f"{name}: integer"
        for out in (k_out, v_out):
            assert out.dtype == torch.bfloat16, "out: bf16"
            assert out.shape[1:] == (heads, tokens_per_block, head_dim), (
                "out: [listed, Hkv, tpb, D]"
            )
            assert out.is_contiguous(), "out: contiguous"


qsa_nvfp4_pages_to_bf16 = QsaNvfp4PagesToBf16()
