# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather + dequantize native MXFP8 rows (E4M3 codes x UE8M0 per 32) from a flat or paged pool into
bf16, via the Triton op `torch.ops.trtllm.mxfp8_dequantize`
(`tensorrt_llm/_torch/custom_ops/mxfp8_dequantize.py`)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops.mxfp8_dequantize  # noqa: F401 — registers torch.ops.trtllm.mxfp8_dequantize
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_BLOCK = 32
_W_KV = 512  # DeepSeek-V4.1 window-KV head_dim (448 nope + 64 RoPE, all quantized)

#: Rows of the full-byte-range identity read: decode-sized through a prefill-sized pool.
_IDENTITY_ROWS = (1, 6, 128, 4096)
#: Widths other than the window KV's 512 (448 is 14 blocks, not a power of two).
_OTHER_WIDTHS = (32, 128, 448, 1024)
#: Token counts whose checkpoint-recipe rows round-trip to the bf16 container.
_RECIPE_ROWS = (1, 6, 32, 4096, 8192)
_GATHER_PAGE_SIZES = (1, 64, 128, 256)
_GATHER_SLOT_DTYPES = (torch.int32, torch.int64)
_GATHER_ROWS = (1, 128, 1024, 8192)


def _num_slots(t: torch.Tensor) -> int:
    return t.shape[0] if t.dim() == 2 else t.shape[0] * t.shape[1]


def _slot_rows(t: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Rows of a [slots, C] or [pages, page_size, C] tensor at the given slots."""
    if t.dim() == 2:
        return t[slots]
    return t[slots // t.shape[1], slots % t.shape[1]]


class Mxfp8Dequantize(OpWrapper):
    """Read native MXFP8 rows out of a pool, decode them to bf16, and write each one to a slot of `out`.

    The pool is E4M3 codes plus one UE8M0 byte per 32 codes: DeepSeek-V4.1's SWA window-KV layout,
    512 + 16 = 528 B per slot. For output row `r`::

        s = src_slots[r] if src_slots is not None else r
        d = dst_slots[r] if dst_slots is not None else r
        if d < 0 or d >= out_slots:  nothing is written for row r
        elif s < 0 or s >= slots:    out[d, :] = +0.0
        else:                        out[d, i] = bf16_rne(e4m3(codes[s, i]) * 2^(scales[s, i // 32] - 127))

    `rows` is `len(src_slots)`, else `len(dst_slots)`, else the pool's slot count. Slot `s` of a
    paged `[pages, page_size, W]` tensor is `(s // page_size, s % page_size)`, for the pool and
    for `out` alike; leading strides are free, so a 528 B layout is two views of one uint8 buffer.

    Fusion boundary: the slot gather, the E4M3 x UE8M0 decode, the bf16 rounding and the optional
    slot scatter are inside the call. Deriving the slot lists from block tables, remapping the
    sparse-attention index table, the attention itself, and writing the pool
    (`mxfp8_fake_quantize`'s native form) are not. Writes `out` in place, only at the
    destination slots of the rows, and returns `out`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"identity {n} rows x 512: every e4m3 code x UE8M0 byte 0..254"
                + (", bytes 0 / 1 / 254 pinned" if n >= 16 else ""),
                spec=dict(kind="identity", rows=n, width=_W_KV, pin_extremes=n >= 16),
            )
            for n in _IDENTITY_ROWS
        )
        + tuple(
            Cell(
                why=f"identity 257 rows x {w}: width {w}, every code x byte 0..254",
                spec=dict(kind="identity", rows=257, width=w, pin_extremes=False),
            )
            for w in _OTHER_WIDTHS
        )
        + tuple(
            Cell(
                why=f"recipe rows T={t}: equals the bf16 container"
                + (", the 1e-4 floor byte 105 present" if t >= 32 else ""),
                spec=dict(kind="recipe", rows=t),
            )
            for t in _RECIPE_ROWS
        )
        + tuple(
            Cell(
                why=f"gather {rows} rows, page_size {ps}, {str(sd).removeprefix('torch.')} slots"
                + (": 1/16 invalid" if rows >= 16 else "")
                + ", from a 40960-slot pool",
                spec=dict(kind="gather", rows=rows, page_size=ps, slot_dtype=sd),
            )
            for ps in _GATHER_PAGE_SIZES
            for sd in _GATHER_SLOT_DTYPES
            for rows in _GATHER_ROWS
        )
        + tuple(
            Cell(
                why=f"528 B rows {'paged' if paged else 'flat'}, {str(sd).removeprefix('torch.')}"
                " scales: two strided views of one buffer, 4096 rows",
                spec=dict(kind="interleaved", paged=paged, scale_dtype=sd),
            )
            for paged in (False, True)
            for sd in (torch.uint8, torch.float8_e8m0fnu)
        )
        + tuple(
            Cell(
                why=f"page footer, page_size {ps}: code rows then scale rows per page",
                spec=dict(kind="footer", page_size=ps),
            )
            for ps in (128, 256)
        )
        + (
            Cell(
                why="strided src_slots [::2], out the left half of [777, 1024]",
                spec=dict(kind="strided"),
            ),
            Cell(
                why="shadow pool: src = dst over 512 slots, -1 skipped, the rest kept",
                spec=dict(kind="shadow"),
            ),
            Cell(
                why="scatter: invalid dst skipped, invalid src -> zero row, int64/int32",
                spec=dict(kind="scatter"),
            ),
            Cell(
                why="dst_slots only into paged [16, 128, 512]: row r reads slot r",
                spec=dict(kind="dst_only"),
            ),
            Cell(
                why="zero rows: an empty src_slots launches nothing, out untouched",
                spec=dict(kind="zero_rows"),
            ),
        )
    )

    note = """
    Exactness: `e4m3(code) * 2^k` has at most 4 significant bits and is exact in fp32 for every
    byte 0..254; the one rounding is the bf16 RNE conversion, exact whenever the product is a bf16
    normal (every byte the fake-quantize recipe produces, 105..247). Byte 0 is the fp32-subnormal
    scale 2^-127, not zero. Byte 255 makes the whole 32-block NaN; codes 0x7F / 0xFF are NaN for
    that element; code 0x80 is -0.0. Dequantizing `mxfp8_fake_quantize`'s native output
    reproduces its bf16-container output bit for bit.

    Call forms the target uses: compact gather (`out` [rows, 512], `src_slots` the referenced ring
    slots); shadow pool (`out` a bf16 pool of the native pool's geometry, `src_slots = dst_slots`;
    slots not named keep their bytes); fixed-size window buffer (`out` one persistent zero-filled
    [max_num_tokens * 128, 512] buffer, `src_slots` the window half of the attention's int32
    [T, 128] index table flattened, `dst_slots` None; rows at and past 128 * T keep what an
    earlier call left).

    Metadata consumed: none. The slot addresses come only from `src_slots`, `dst_slots` and the
    tensors' geometry.

    Preconditions the op raises ValueError on, leaving `out` untouched: codes CUDA,
    float8_e4m3fn or uint8, 2-D or 3-D, last dim a positive multiple of 32 with unit stride;
    scales uint8 or float8_e8m0fnu, exactly `codes.shape[:-1] + (W // 32,)` with unit last stride,
    on the codes device; out bfloat16 with last dim W at unit stride, on the codes device; slot
    lists int32/int64, 1-D, on the codes device, equal lengths when both given; without
    `dst_slots`, `out` needs at least `rows` slots. Unchecked: two rows with the same valid
    destination and different sources race (identical sources are fine and certified); `out`
    must not overlap the pool.

    CUDA graphs: make one eager call of the same variant before capturing (Triton compiles on
    first use). A variant is W, whether each slot list is given, and the class of `out`'s slot
    count (1, a multiple of 16, any other), which Triton specializes on; rows are grid axis 0 and
    never a kernel argument. A shadow or a fixed-size `out` therefore compiles one variant.

    Certified by the rest of this entry's test, beyond the cells, on sm_103: native pools of the
    live DeepSeek-V4.1 cache manager (`TRTLLM_DSV41_KV_STORAGE=native`, `get_v41_native_kv_views`,
    `[B, 128, 528]` interleaved, strides (67584, 528, 1)) up to B = 1,648,320 pages read compact
    (the measured ceiling over the scanned engine configurations including PP2 stages, at 0.5 x
    total memory; past byte offset 2^36) and up to 1,098,720 pages into a shadow (past 2^37),
    with a wrap control showing a mod-2^32 offset would be seen; every row count 1..16,384 compact
    and into a shadow; per-query window reads up to 1,048,576 rows; the fixed-size window buffer
    at 262,144 / 524,288 / 1,048,576 / 2,097,152 rows (max_num_tokens 2048 / 4096 / 8192 /
    16,384; independent of max_batch_size and max_seq_len), every row count up to the buffer
    (sampled at 2,097,152); renumbered reads over random multi-request steps on live managers at
    every max_batch_size 1..32 and at (16,384, 256); CUDA-graph capture at every graph batch
    1..32, 40..128 step 8, 192 and 256 with next_n 1 and 6, adding no compiled variant after one
    eager warm-up.

    Not certified: duplicate destinations with different sources; `out` aliasing the pool; a
    first-ever call inside a capture; torch.compile (a fake is registered, not exercised); a
    native pool above 1,648,320 pages or a shadow above 1,098,720; more than 16,384 rows into a
    rows-sized out or a shadow other than the per-query counts; live-manager steps at positions
    above 32,781; a fixed window buffer of another size; page sizes other than 1, 64, 128, 256;
    widths other than 32, 128, 448, 512, 1024.
    """

    def raw_call(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        out: torch.Tensor,
        src_slots: Optional[torch.Tensor] = None,
        dst_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch.ops.trtllm.mxfp8_dequantize(codes, scales, out, src_slots, dst_slots)
        return out

    def reference(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        out: torch.Tensor,
        src_slots: Optional[torch.Tensor] = None,
        dst_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The expected `out` after the call: a copy of `out` with the rows' destinations filled.

        `out` is read for its prior contents (every slot the call does not write keeps them) and is
        not mutated; pass a snapshot taken before the op ran. The decode is torch's
        float8_e4m3fn -> fp64 cast and `torch.ldexp(code, byte - 127)` in fp64, then fp64 -> fp32
        (exact) -> bf16 RNE: independent of the kernel's exponent-bit construction. Byte 255
        decodes to NaN; NaN payloads are not modelled, so cells carry no NaN encodings.
        """
        w = codes.shape[-1]
        num_slots = _num_slots(codes)
        expected = out.clone()
        out_slots = _num_slots(expected)
        if src_slots is not None:
            rows = src_slots.shape[0]
        elif dst_slots is not None:
            rows = dst_slots.shape[0]
        else:
            rows = num_slots
        dev = codes.device
        src = src_slots.long() if src_slots is not None else torch.arange(rows, device=dev)
        dst = dst_slots.long() if dst_slots is not None else torch.arange(rows, device=dev)
        written = (dst >= 0) & (dst < out_slots)
        src, dst = src[written], dst[written]
        valid = (src >= 0) & (src < num_slots)
        s = src.clamp(0, max(num_slots - 1, 0))
        c = _slot_rows(codes.view(torch.uint8), s).contiguous().view(torch.float8_e4m3fn)
        b = _slot_rows(scales.view(torch.uint8), s).long()
        v = c.float().double().view(-1, w // _BLOCK, _BLOCK)
        vals = torch.ldexp(v, (b - 127).double().unsqueeze(-1))
        vals = torch.where((b == 255).unsqueeze(-1), torch.nan, vals)
        vals = vals.view(-1, w).float().to(torch.bfloat16)
        vals[~valid] = 0.0
        if expected.dim() == 2:
            expected[dst] = vals
        else:
            expected[dst // expected.shape[1], dst % expected.shape[1]] = vals
        return expected

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Bitwise, bf16 compared as int16, over all of `out`.

        Each element is one e4m3 value times a power of two rounded once to bf16, so there is no
        tolerance to choose; untouched slots must keep their prior bits. On T = 4096 recipe rows
        the gate fails for each deliberately wrong decode of the same bytes: exponent bias 126 or
        128, an e5m2 decode, one scale byte per 64 codes, a slot off by one.
        """
        assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        a = out.reshape(-1).view(torch.int16)
        e = ref.reshape(-1).view(torch.int16)
        bad = torch.zeros((), dtype=torch.int64, device=a.device)
        for i in range(0, a.numel(), 1 << 26):  # chunked: a full bool reduction copies to int64
            bad += (a[i : i + (1 << 26)] != e[i : i + (1 << 26)]).sum()
        assert int(bad) == 0, f"{int(bad)} of {out.numel()} elements differ bitwise"


mxfp8_dequantize = Mxfp8Dequantize()
