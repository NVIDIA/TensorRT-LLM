# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather native MXFP8 rows (E4M3 codes x UE8M0 per 32) from a flat or paged pool and transcode them to
E4M3 with one per-tensor scale, via the Triton op `torch.ops.trtllm.mxfp8_to_e4m3`
(`tensorrt_llm/_torch/custom_ops/mxfp8_to_e4m3.py`)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops.mxfp8_to_e4m3  # noqa: F401 — registers torch.ops.trtllm.mxfp8_to_e4m3
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_BLOCK = 32
_E4M3_MAX = 448.0

#: Per-tensor factors of the compact gather: powers of two around the target's bound, and two
#: non-powers of two (one extra fp32 rounding before the E4M3 rounding).
_COMPACT_INV_SCALES = tuple(2.0**k for k in (-2, 0, 3, 6, 7, 8)) + (0.75, 1.0 / 3.0)
#: Query-row counts of the target's window read on the engine-sized pool.
_ENGINE_QUERIES = (1, 32, 192, 2048, 8192)
#: SWA pool pages of an engine runtime-value dump.
_ENGINE_POOL_PAGES = 188160


def _num_slots(t: torch.Tensor) -> int:
    return t.shape[0] if t.dim() == 2 else t.shape[0] * t.shape[1]


def _slot_rows(t: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Rows of a [slots, C] or [pages, page_size, C] tensor at the given slots."""
    if t.dim() == 2:
        return t[slots]
    return t[slots // t.shape[1], slots % t.shape[1]]


class Mxfp8ToE4m3(OpWrapper):
    """Gather native MXFP8 rows from a pool and write them as E4M3 with ONE per-tensor factor.

    For output row `r`::

        s = src_slots[r] if src_slots is not None else r
        d = dst_slots[r] if dst_slots is not None else r
        v[i] = e4m3(codes[s, i]) * 2 ^ (scales[s, i // 32] - 127)  # exact in fp32
        out[d, i] = e4m3_rne(clamp(v[i] * inv_scale, -448, 448))  # NaN stays NaN

    A source slot that is negative or `>= slots` gives a row of `+0.0` codes; a destination slot
    that is negative or `>= out slots` is skipped. Slot `s` of a paged `[pages, page_size, W]`
    tensor is `(s // page_size, s % page_size)`; leading strides are free, so DeepSeek-V4.1's
    528 B window row is passed as two views (the first 512 bytes, the last 16).

    Fusion boundary: the gather, the dequantization, the rescale and the E4M3 cast are one launch.
    The caller picks `inv_scale` (`1 / kv_scale`); nothing here computes an amax. Element for
    element this is `e4m3(mxfp8_dequantize(...) * inv_scale)` and writes half the bytes. Writes
    `out` in place, only at the destination slots of the rows, and returns `out`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"compact gather inv_scale {s:g}, "
                + ("paged [320, 128, 528]" if paged else "flat [40960, 528]")
                + ": 300 queries x 128 window columns, -1 tails",
                spec=dict(kind="compact", inv_scale=s, paged=paged, queries=300),
            )
            for s in _COMPACT_INV_SCALES
            for paged in (True, False)
        )
        + tuple(
            Cell(
                why=f"engine pool T={t}: [188160, 128, 528] first and last pages into "
                "[8192 * 128, 512]",
                spec=dict(
                    kind="engine_pool", queries=t, inv_scale=128.0, pool_pages=_ENGINE_POOL_PAGES
                ),
            )
            for t in _ENGINE_QUERIES
        )
        + tuple(
            Cell(
                why=f"identity inv_scale {s:g}: no slot lists, normal range exact",
                spec=dict(kind="identity", inv_scale=s, pool_pages=64),
            )
            for s in (1.0, 16.0, 128.0)
        )
        + (
            Cell(
                why="dst_slots scatter, inv_scale 64: -1 and >= out_slots skipped",
                spec=dict(kind="scatter", inv_scale=64.0),
            ),
        )
    )

    note = """
    The target's call is the compact window read: `src_slots` the window half of the attention's
    index table flattened (T queries x 128 columns, -1 for an empty column), `out` the first
    `128 * T` rows of a `[max_num_tokens * 128, 512]` float8_e4m3fn buffer, `dst_slots` None.

    A power-of-two `inv_scale` is exact in E4M3's normal range: `out / inv_scale == v` for every
    result with magnitude in [2^-6, 448] (measured on sm_103: none inexact at `inv_scale` 1, 16,
    128). Results below 2^-6 round to E4M3 subnormals (error <= 2^-10 / inv_scale); results above
    448 saturate to +-448. A non-power-of-two `inv_scale` adds one fp32 rounding (the multiply).
    UE8M0 byte 0 is 2^-127 (fp32 subnormal); byte 255 makes the whole 32-block NaN codes; e4m3 NaN
    codes stay NaN; signed zeros are kept.

    Metadata consumed: none. The caller supplies the slot lists.

    Preconditions: the op raises ValueError on a wrong dtype, shape, stride or device, and on a
    non-positive `inv_scale`. `codes` float8_e4m3fn / uint8, `[slots, W]` or
    `[pages, page_size, W]`, W % 32 == 0, unit last-dim stride; `scales` uint8 / float8_e8m0fnu,
    `codes.shape[:-1] + (W / 32,)`; `out` float8_e4m3fn with unit last-dim stride; slot lists
    int32 / int64 of one length; without `dst_slots`, `out` needs at least `rows` slots. The
    caller chooses `inv_scale` so the values it needs do not saturate (for DeepSeek-V4.1's normed
    window KV the target derives a power-of-two bound from the norm weights,
    |x| <= sqrt(512) * max|w|). Graph-capturable: no host synchronization, no allocation; one
    Triton program per output row, no autotuning.

    Not certified: W other than 512; page geometries other than the cells'; `inv_scale` values that
    make the fp32 product overflow or underflow fp32 (not reachable from a UE8M0 byte and a scale
    in 2^-8 .. 2^8); duplicate valid destinations with different sources.
    """

    def raw_call(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        out: torch.Tensor,
        inv_scale: float,
        src_slots: Optional[torch.Tensor] = None,
        dst_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch.ops.trtllm.mxfp8_to_e4m3(codes, scales, out, inv_scale, src_slots, dst_slots)
        return out

    def reference(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        out: torch.Tensor,
        inv_scale: float,
        src_slots: Optional[torch.Tensor] = None,
        dst_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The expected `out` after the call: a copy of `out` with the rows' destinations filled.

        `out` is read for its prior contents (every slot the call does not write keeps them) and is
        not mutated; pass a snapshot taken before the op ran. The decode is the float8_e4m3fn ->
        fp64 cast and `torch.ldexp(code, byte - 127)` in fp64, exact in fp32; the one fp32 rounding
        is the multiply by the fp32 `inv_scale` (none for a power of two), then the clamp to +-448
        and torch's RNE cast to E4M3 -- independent of the kernel's exponent-bit construction.
        Byte 255 decodes to NaN; NaN code payloads are not modelled, so cells carry no NaN
        encodings.
        """
        w = codes.shape[-1]
        num_slots = _num_slots(codes)
        expected = out.view(torch.uint8).clone()
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
        v = torch.ldexp(
            c.float().double().view(-1, w // _BLOCK, _BLOCK), (b - 127).double().unsqueeze(-1)
        )
        v = torch.where((b == 255).unsqueeze(-1), torch.nan, v)
        y = v.float() * torch.tensor(inv_scale, dtype=torch.float32, device=dev)
        q = y.clamp(-_E4M3_MAX, _E4M3_MAX).view(-1, w).to(torch.float8_e4m3fn).view(torch.uint8)
        q[~valid] = 0
        if expected.dim() == 2:
            expected[dst] = q
        else:
            expected[dst // expected.shape[1], dst % expected.shape[1]] = q
        return expected.view(out.dtype)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Bitwise on the uint8 codes over all of `out`.

        Every output is one E4M3 rounding of an exact fp32 product (two roundings for a
        non-power-of-two `inv_scale`, the same two on both sides), so there is no tolerance to
        choose; untouched slots must keep their prior bytes. Controls on recipe rows each change
        more than 30% of the codes: UE8M0 bias 126, `inv_scale` ignored, codes read as E5M2.
        """
        assert out.dtype == ref.dtype == torch.float8_e4m3fn, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        a = out.view(torch.uint8).reshape(-1)
        e = ref.view(torch.uint8).reshape(-1)
        bad = torch.zeros((), dtype=torch.int64, device=a.device)
        for i in range(0, a.numel(), 1 << 26):  # chunked: a full bool reduction copies to int64
            bad += (a[i : i + (1 << 26)] != e[i : i + (1 << 26)]).sum()
        assert int(bad) == 0, f"{int(bad)} of {out.numel()} codes differ"


mxfp8_to_e4m3 = Mxfp8ToE4m3()
