# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse path's paged K/V append into an NVFP4 pool: quantize and scatter in one launch.

This target's local variant (`__r37`) of the tp1 target's
`ops/attention/qsa_paged_kv_store_nvfp4`.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Values per e4m3 block scale, and e2m1 values per stored byte.
_SF_VEC = 16
_PER_BYTE = 2
#: Tokens of a V scale group: V's scales are stored `[tpb / 4, D / 16, 4]` per head page.
_V_SF_QUAD = 4
#: e2m1's largest magnitude: a block's scale maps its amax onto it.
_E2M1_MAX = 6.0
#: What `cvt.rn.satfinite.e2m1x2.f32` makes of a NaN of either sign: +6. A NaN reaches
#: it as 0 x inf, from a zero in a block whose nonzero amax rounds its scale to 0.
_E2M1_NAN_CODE = 0x7
#: The round-to-nearest-even decision points between neighbouring e2m1 magnitudes:
#: `(threshold, inclusive)` steps the code up when `|v| > t`, or `|v| >= t` where the
#: upper neighbour is the even code.
_E2M1_STEPS = (
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
)

#: This checkpoint's K/V geometry and the engine's QSA table. `heads_q` places k and v as
#: head slices of the packed `[T, 24 + 2 + 2, 256]` QKV row; `global_k` / `global_v` are
#: `kv_scale_orig_quant[1]` / `[2]`; `slot_pages` role pages of one lifecycle slot hold
#: the call's planes, layer `layer`'s K and V among them (the trunk pool holds 24).
_CKPT = dict(
    hkv=2,
    head_dim=256,
    heads_q=24,
    tokens_per_block=32,
    slots=32,
    width=260,
    row_dtype=torch.bfloat16,
    request_dtype=torch.int32,
    position_dtype=torch.int64,
    global_k=1.0,
    global_v=1.0,
    values="randn",
    hole=None,
    thop_append=False,
    slot_pages=2,
    layer=0,
)

_DECODE = ((0, 1), (31, 1), (32, 1), (33, 1), (100, 1), (255, 1), (7, 1), (64, 1))


def _spec(segments, pages, **overrides) -> dict:
    """`segments` is `(first position, rows)` per request, in row order."""
    return {**_CKPT, "segments": segments, "pages": pages, **overrides}


@triton.jit
def _rcp_approx(x):
    """`rcp.approx.ftz.f32`, the reciprocal the engine's FP4 quantizer uses."""
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _e2m1x2(lo, hi):
    """One byte of two e2m1 codes, `lo` in the low nibble (`cvt.rn.satfinite`)."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; .reg .b16 s; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; "
        "mov.b16 s, {t, t}; cvt.u32.u16 $0, s; }",
        "=r,f,f",
        [lo, hi],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _quantize_row(
    src,
    stride_dim,
    global_scale,
    HEAD_DIM: tl.constexpr,
):
    """One head row: (packed e2m1 bytes [D/2], e4m3 scale bytes [D/16])."""
    pairs = tl.arange(0, HEAD_DIM // 2)
    lo = tl.load(src + (2 * pairs) * stride_dim).to(tl.float32)
    hi = tl.load(src + (2 * pairs + 1) * stride_dim).to(tl.float32)
    pair_max = tl.maximum(tl.abs(lo), tl.abs(hi))
    block_max = tl.max(tl.reshape(pair_max, [HEAD_DIM // 16, 8]), axis=1)
    sf_value = global_scale * (block_max * _rcp_approx(tl.full([HEAD_DIM // 16], 6.0, tl.float32)))
    sf_code = sf_value.to(tl.float8e4nv)
    sf_rounded = sf_code.to(tl.float32)
    out_scale = tl.where(
        block_max != 0.0,
        _rcp_approx(sf_rounded * _rcp_approx(tl.zeros_like(sf_rounded) + global_scale)),
        0.0,
    )
    lo_scaled = tl.reshape(
        tl.reshape(lo, [HEAD_DIM // 16, 8]) * out_scale[:, None], [HEAD_DIM // 2]
    )
    hi_scaled = tl.reshape(
        tl.reshape(hi, [HEAD_DIM // 16, 8]) * out_scale[:, None], [HEAD_DIM // 2]
    )
    packed = (_e2m1x2(lo_scaled, hi_scaled) & 0xFF).to(tl.uint8)
    return packed, sf_code.to(tl.uint8, bitcast=True)


@triton.jit
def _qsa_paged_kv_store_nvfp4_kernel(
    k,
    v,
    k_cache,
    v_cache,
    k_scale_cache,
    v_scale_cache,
    request_indices,
    logical_positions,
    block_table,
    kv_scale_orig_quant,
    num_pages,
    num_requests,
    table_width,
    k_stride_row,
    k_stride_head,
    k_stride_dim,
    v_stride_row,
    v_stride_head,
    v_stride_dim,
    kc_stride_page,
    kc_stride_head,
    kc_stride_token,
    vc_stride_page,
    vc_stride_head,
    vc_stride_token,
    ks_stride_page,
    ks_stride_head,
    vs_stride_page,
    vs_stride_head,
    bt_stride_request,
    bt_stride_page,
    HEAD_DIM: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    request = tl.load(request_indices + row).to(tl.int64)
    logical = tl.load(logical_positions + row).to(tl.int64)
    column = logical // TOKENS_PER_BLOCK
    token = logical % TOKENS_PER_BLOCK
    lookup = (request >= 0) & (request < num_requests) & (logical >= 0) & (column < table_width)
    page = tl.load(
        block_table + request * bt_stride_request + column * bt_stride_page, mask=lookup, other=-1
    ).to(tl.int64)
    valid = (page >= 0) & (page < num_pages)
    page = tl.maximum(page, 0)

    k_data, k_sf = _quantize_row(
        k + row * k_stride_row + head * k_stride_head,
        k_stride_dim,
        tl.load(kv_scale_orig_quant + 1),
        HEAD_DIM,
    )
    v_data, v_sf = _quantize_row(
        v + row * v_stride_row + head * v_stride_head,
        v_stride_dim,
        tl.load(kv_scale_orig_quant + 2),
        HEAD_DIM,
    )
    pairs = tl.arange(0, HEAD_DIM // 2)
    blocks = tl.arange(0, HEAD_DIM // 16)
    tl.store(
        k_cache + page * kc_stride_page + head * kc_stride_head + token * kc_stride_token + pairs,
        k_data,
        mask=valid,
    )
    tl.store(
        v_cache + page * vc_stride_page + head * vc_stride_head + token * vc_stride_token + pairs,
        v_data,
        mask=valid,
    )
    # K scales token-major within the head page; V scales in four-token groups.
    tl.store(
        k_scale_cache
        + page * ks_stride_page
        + head * ks_stride_head
        + token * (HEAD_DIM // 16)
        + blocks,
        k_sf,
        mask=valid,
    )
    tl.store(
        v_scale_cache
        + page * vs_stride_page
        + head * vs_stride_head
        + (token // 4) * (4 * (HEAD_DIM // 16))
        + blocks * 4
        + token % 4,
        v_sf,
        mask=valid,
    )


def _store_nvfp4(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    block_table: torch.Tensor,
    tokens_per_block: int,
    kv_scale_orig_quant: torch.Tensor,
) -> None:
    """Quantize rows of K and V to NVFP4 and store them into their pages (see the entry)."""
    if k.ndim != 3 or v.shape != k.shape:
        raise ValueError("NVFP4 K/V store: k and v must be matching [rows, heads, dim]")
    rows, heads, head_dim = k.shape
    if k.dtype not in (torch.bfloat16, torch.float16) or v.dtype != k.dtype:
        raise ValueError("NVFP4 K/V store: k and v must both be BF16 or both FP16")
    if head_dim < 32 or head_dim & (head_dim - 1):
        raise ValueError("NVFP4 K/V store: head_dim must be a power of two of at least 32")
    if tokens_per_block <= 0 or tokens_per_block % _V_SF_QUAD:
        raise ValueError("NVFP4 K/V store: tokens_per_block must be a positive multiple of 4")
    data_shape = (k_cache.shape[0], heads, tokens_per_block, head_dim // _PER_BYTE)
    scale_shape = (k_cache.shape[0], heads, tokens_per_block, head_dim // _SF_VEC)
    for name, cache, shape in (
        ("k_cache", k_cache, data_shape),
        ("v_cache", v_cache, data_shape),
        ("k_scale_cache", k_scale_cache, scale_shape),
        ("v_scale_cache", v_scale_cache, scale_shape),
    ):
        if cache.dtype != torch.uint8 or tuple(cache.shape) != shape:
            raise ValueError(
                f"NVFP4 K/V store: {name} must be uint8 {shape}, got {cache.dtype} "
                f"{tuple(cache.shape)}"
            )
        if cache.stride(3) != 1:
            raise ValueError(f"NVFP4 K/V store: {name} must be contiguous along its last dim")
    for name, cache in (("k_scale_cache", k_scale_cache), ("v_scale_cache", v_scale_cache)):
        if cache.stride(2) != head_dim // _SF_VEC:
            raise ValueError(f"NVFP4 K/V store: each head page of {name} must be contiguous")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("NVFP4 K/V store: block_table must be a 2-D int32 tensor")
    for name, meta in (
        ("request_indices", request_indices),
        ("logical_positions", logical_positions),
    ):
        if meta.shape != (rows,) or meta.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"NVFP4 K/V store: {name} must be one int32/int64 entry per row")
        if not meta.is_contiguous():
            raise ValueError(f"NVFP4 K/V store: {name} must be contiguous")
    if (
        kv_scale_orig_quant.dtype != torch.float32
        or kv_scale_orig_quant.shape != (3,)
        or not kv_scale_orig_quant.is_contiguous()
    ):
        raise ValueError("NVFP4 K/V store: kv_scale_orig_quant must be a contiguous fp32 [3]")
    operands = (k, v, k_cache, v_cache, k_scale_cache, v_scale_cache, request_indices)
    operands += (logical_positions, block_table, kv_scale_orig_quant)
    if not all(t.is_cuda and t.device == k.device for t in operands):
        raise ValueError("NVFP4 K/V store: every operand must be on k's CUDA device")
    if rows == 0:
        return
    _qsa_paged_kv_store_nvfp4_kernel[(rows, heads)](
        k,
        v,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        request_indices,
        logical_positions,
        block_table,
        kv_scale_orig_quant,
        k_cache.shape[0],
        block_table.shape[0],
        block_table.shape[1],
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        k_scale_cache.stride(0),
        k_scale_cache.stride(1),
        v_scale_cache.stride(0),
        v_scale_cache.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        HEAD_DIM=head_dim,
        TOKENS_PER_BLOCK=tokens_per_block,
        num_warps=4,
    )


def _e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    """fp32 -> e2m1 codes 0..15, round to nearest even, saturating at 6; a NaN
    (0 x inf, in a block whose scale underflows to zero) converts to +6."""
    magnitude = values.abs()
    code = torch.zeros(values.shape, dtype=torch.int32, device=values.device)
    for threshold, inclusive in _E2M1_STEPS:
        code += (magnitude >= threshold if inclusive else magnitude > threshold).to(torch.int32)
    code |= torch.signbit(values).to(torch.int32) << 3
    return torch.where(torch.isnan(values), torch.full_like(code, _E2M1_NAN_CODE), code)


def _quantize_reference(rows: torch.Tensor, global_scale: torch.Tensor):
    """[n, D] rows -> (packed bytes [n, D/2], e4m3 scale bytes [n, D/16]), in plain torch."""
    n, head_dim = rows.shape
    x = rows.float().reshape(n, head_dim // _SF_VEC, _SF_VEC)
    block_max = x.abs().amax(dim=-1)
    sixth = torch.tensor(_E2M1_MAX, dtype=torch.float32, device=rows.device).reciprocal()
    scale_code = (global_scale * (block_max * sixth)).to(torch.float8_e4m3fn)
    rounded = scale_code.float()
    out_scale = torch.where(
        block_max != 0,
        (rounded * global_scale.reciprocal()).reciprocal(),
        torch.zeros_like(rounded),
    )
    codes = _e2m1_codes(x * out_scale[..., None]).reshape(n, head_dim // _PER_BYTE, _PER_BYTE)
    packed = (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8)
    return packed, scale_code.view(torch.uint8)


class QsaPagedKvStoreNvfp4(OpWrapper):
    """Quantize every row's K and V to NVFP4 and store them into the request's page.

    A Triton kernel defined in this file, one launch over `(rows, Hkv)`. For row t
    and KV head h, with `page = block_table[request_indices[t], logical_positions[t]
    // tpb]` and `s = logical_positions[t] % tpb`, each 16-value block `b` of
    `k[t, h]` (and alike for v, with `kv_scale_orig_quant[2]`) is stored as::

        g        = kv_scale_orig_quant[1]
        sf       = e4m3(g * (amax_b * rcp(6)))                  # one byte
        code_i   = e2m1(k[t, h, i] * rcp(float(sf) * rcp(g)))   # 0 when amax_b == 0
        k_cache[page, h, s, i // 2]            nibble i % 2 (low nibble = even i)
        k_scale_cache[page, h, s, b]           = sf             # token-major
        v_scale_cache[page, h, (s // 4) * 4 * (D / 16) + 4 * b + s % 4]   # four-token groups

    `rcp` is `rcp.approx.ftz.f32`, `e4m3` and `e2m1` round to nearest even and
    saturate. This is the attention op's own NVFP4 append (`cvt_warp_fp16_to_fp4`
    behind `quantizeAndWriteFP4KVCache`), the same bits for the same row: a
    token's stored value does not depend on which path wrote it. A row that
    resolves to no slot -- negative request, request past the table, negative
    position, page column past the table, a table entry that is negative or past
    the pool -- is skipped. Returns the four caches, written in place.

    Fusion boundary: quantization and the scatter. The projection, norm and RoPE
    of K are the caller's; the QSA index plane is not touched.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: 8 rows at aligned, mid-page and page-crossing positions, with zeros, "
            "-0.0 and a 900.0 outlier inside blocks, and a row on an unallocated (-1) page: "
            "zero scales, saturation and the skip",
            spec=_spec(_DECODE, pages=69, slots=8, values="edge", hole="unallocated_page"),
        ),
        Cell(
            why="prefill: 1024 rows, 4 requests x 256 walking 8 pages each, in the engine's "
            "trunk layout: layer 11's planes among the 24 role pages of a slot",
            spec=_spec(((0, 256),) * 4, pages=160, slot_pages=24, layer=11),
        ),
        Cell(
            why="the attention op's own append over the same rows (context, chunk, decode "
            "P=1 and P=3) at unit second-level scales: data and both scale layouts bitwise equal",
            spec=_spec(((0, 115), (0, 101), (0, 36)), pages=96, thop_append=True),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. The target passes the
    step's `qsa_req_idx_per_token` (int32) and `qsa_logical_positions` (int64),
    `qsa_block_table` unsliced (raw pool page ids, `-1` unallocated), the layer's
    K and V planes of the QSA manager's `get_buffers(i, "HND")` (uint8 `[slots,
    Hkv, tpb, D/2]`), the two planes of `get_block_scale_buffers(i, "HND")`
    (uint8 `[slots, Hkv, tpb, D/16]`), and its own `kv_scale_orig_quant` -- the
    same fp32 `[3]` tensor its dense attention call binds, so both writers apply
    one second-level scale.

    Preconditions:
      - destination slots are unique across the batch: two rows naming one
        (page, slot) both store, unordered.
      - the dense path's reader assumes this layout: e2m1 nibbles HND per page,
        K scales linear, V scales in four-token groups. The attention op's
        append, run over the same rows, leaves bitwise the same pool (a cell),
        at the unit second-level scales the target binds.
    The callable rejects loudly (ValueError; the test pins the non-contiguous
    metadata refusal): non-matching or non-3-D k/v, a row
    dtype outside bf16/fp16, a head_dim that is not a power of two >= 32, a
    tokens_per_block that is not a multiple of 4, caches that are not uint8 of
    the stated shapes or whose last dim or scale head page is strided, a non-2-D
    or non-int32 table, metadata that is not one contiguous int32/int64 entry per
    row, a `kv_scale_orig_quant` that is not a contiguous fp32 `[3]`, operands on
    another device. `rows == 0` returns before launching (not certified).

    Every offset is formed in 64 bits: rows, heads, requests, positions and pages
    are widened before any product. CUDA-graph capturable; a replay follows rows,
    pages and positions rewritten in place (the test replays 8 MTP verify blocks of
    8 rows bitwise equal to eager).

    Compiled variants: the page count (`num_pages`, the pool's slot count), the
    table's height (`num_requests`) and width and every stride are plain-int
    arguments, and Triton compiles the kernel once per class of each -- equal to
    1, a multiple of 16, anything else -- besides once per head_dim and
    tokens_per_block. A served call puts the width and each stride in one fixed
    class (the k/v head slices of the packed QKV row, the planes' slot, head and
    token strides, the contiguous table's) and varies the page count and the
    table height. The cells run (other, other) and (multiple of 16, multiple of
    16); the other combinations, a table height of 1 among them, are not
    certified.

    Not finite values: a NaN or +-inf in a block is not certified (the e2m1 and
    e4m3 conversions saturate, the block maximum is the kernel's `max`).

    Not certified either: second-level scales other than 1.0, int32 positions
    or int64 request indices, and of the rows that resolve to no slot any but
    an unallocated (`-1`) table entry (the reference skips each).
    """

    def __call__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale_cache: torch.Tensor,
        v_scale_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        kv_scale_orig_quant: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _store_nvfp4(
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            request_indices=request_indices,
            logical_positions=logical_positions,
            block_table=block_table,
            tokens_per_block=tokens_per_block,
            kv_scale_orig_quant=kv_scale_orig_quant,
        )
        return k_cache, v_cache, k_scale_cache, v_scale_cache

    def reference(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale_cache: torch.Tensor,
        v_scale_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        kv_scale_orig_quant: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The four caches after the store, as fresh tensors; the inputs are untouched.

        Quantization in fp32 torch with IEEE reciprocals where the kernel uses
        `rcp.approx.ftz`; the two agree on every operand the cells drive (6.0, the
        unit second-level scale, and the e4m3 scale values their rows produce), which
        the cells' bitwise gate is what pins.
        """
        out = [
            t.clone(memory_format=torch.contiguous_format)
            for t in (k_cache, v_cache, k_scale_cache, v_scale_cache)
        ]
        rows, heads, head_dim = k.shape
        requests = request_indices.long()
        positions = logical_positions.long()
        columns = positions.div(tokens_per_block, rounding_mode="floor")
        table_rows, table_width = block_table.shape
        ok = (requests >= 0) & (requests < table_rows) & (positions >= 0) & (columns < table_width)
        pages = block_table[
            requests.clamp(0, table_rows - 1), columns.clamp(0, table_width - 1)
        ].long()
        ok &= (pages >= 0) & (pages < k_cache.shape[0])
        page, slot = pages[ok], (positions % tokens_per_block)[ok]
        n = int(page.numel())
        if n == 0:
            return tuple(out)
        groups = head_dim // _SF_VEC
        quad_base = (slot // _V_SF_QUAD) * (_V_SF_QUAD * groups) + slot % _V_SF_QUAD
        for src, data, scales, global_scale, linear in (
            (k, out[0], out[2], kv_scale_orig_quant[1], True),
            (v, out[1], out[3], kv_scale_orig_quant[2], False),
        ):
            packed, sf = _quantize_reference(src[ok].reshape(n * heads, head_dim), global_scale)
            data[page, :, slot, :] = packed.reshape(n, heads, head_dim // _PER_BYTE)
            sf = sf.reshape(n, heads, groups)
            if linear:
                scales[page, :, slot, :] = sf
            else:
                flat = scales.view(scales.shape[0], heads, tokens_per_block * groups)
                offsets = quad_base[:, None] + _V_SF_QUAD * torch.arange(groups, device=sf.device)
                for h in range(heads):
                    flat[page[:, None], h, offsets] = sf[:, h]
        return tuple(out)


qsa_paged_kv_store_nvfp4 = QsaPagedKvStoreNvfp4()
