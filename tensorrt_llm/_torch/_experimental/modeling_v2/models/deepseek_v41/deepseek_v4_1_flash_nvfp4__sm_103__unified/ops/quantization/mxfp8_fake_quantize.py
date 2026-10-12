# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 fake quantization (E4M3 x UE8M0 per 32, amax floor 1e-4) written into slots of a
bf16 container pool, or as native e4m3 codes + UE8M0 bytes, via the Triton op
`torch.ops.trtllm.mxfp8_fake_quantize` (`tensorrt_llm/_torch/custom_ops/mxfp8_fake_quantize.py`)."""

from __future__ import annotations

from typing import Optional, Union

import torch

import tensorrt_llm._torch.custom_ops.mxfp8_fake_quantize  # noqa: F401 — registers torch.ops.trtllm.mxfp8_fake_quantize
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_BLOCK = 32
_E4M3_MAX = 448.0
_AMAX_FLOOR = 1e-4
_W_KV = 512  # DeepSeek-V4.1 window-KV head_dim (448 nope + 64 RoPE, all quantized)

#: Decode (1, 5 / 6 = DSpark draft block / verify), batch, window and prefill row counts.
_CONTAINER_ROWS = (1, 5, 6, 32, 128, 4096, 8192)
#: Widths other than the window KV's 512 (448 is 14 blocks, not a power of two).
_OTHER_WIDTHS = (32, 128, 448, 1024)
#: The two tokens_per_block values the DeepSeek-V4 cache manager accepts, and their pool pages.
_POOL_PAGES = {128: 300, 256: 150}
_SCATTER_ROWS = (1, 6, 32, 4096, 8192)
_SLOT_DTYPES = (torch.int32, torch.int64)
_NATIVE_ROWS = (1, 6, 4096)


def _num_slots(t: torch.Tensor) -> int:
    return t.shape[0] if t.dim() == 2 else t.shape[0] * t.shape[1]


def _write_slots(t: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """Write `rows` to the given slots of a [slots, C] or [pages, page_size, C] tensor."""
    if t.dim() == 2:
        t[slots] = rows
    else:
        t[slots // t.shape[1], slots % t.shape[1]] = rows


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.uint8)


class Mxfp8FakeQuantize(OpWrapper):
    """MXFP8-fake-quantize every bf16 row of `x` and write each row to one slot of `out`.

    For row `r` and 32-element block `b` along the last dim::

        amax      = max(max |x[r, 32b : 32b+32]|, 1e-4)            # fp32; the floor is fp32(1e-4)
        e         = ceil(log2(amax / 448))                         # exact; scale 2^e >= 2^-22
        code[r,i] = e4m3_rne(clamp(x[r,i] / 2^e, -448, 448))       # the one rounding
        byte[r,b] = e + 127                                        # UE8M0
        value     = e4m3(code) * 2^e                               # exact in bf16

    This is the checkpoint reference's `act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu,
    inplace=True)` on the post-RoPE window KV. The dtype of `out` selects the form: a bfloat16
    `out` (the bf16 container) receives the values; a float8_e4m3fn / uint8 `out` (native)
    receives the codes and `out_scale` (uint8 / float8_e8m0fnu, column b = block b) the bytes.
    Both forms hold the same value set. Row r goes to slot `slot_mapping[r]` (skipped when
    negative or >= slots), or to slot r when `slot_mapping` is None; slot `s` of a paged
    `[pages, page_size, W]` tensor is `(s // page_size, s % page_size)`.

    Fusion boundary: the per-block amax, the scale, the e4m3 rounding, the dequantization (bf16
    form) and the slot scatter are inside the call. kv_norm, RoPE, deriving `slot_mapping` from
    block tables, and any read-back of a native pool (`mxfp8_dequantize`) are not.

    Writes in place and returns what it wrote: `out` in the bf16-container form (`out_scale`
    None), `(out, out_scale)` in the native form.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"bf16 container T={t} x 512, identity: decode to prefill rows",
                spec=dict(kind="container", rows=t, width=_W_KV),
            )
            for t in _CONTAINER_ROWS
        )
        + tuple(
            Cell(
                why=f"bf16 container 257 x {w}, identity: width {w}",
                spec=dict(kind="container", rows=257, width=w),
            )
            for w in _OTHER_WIDTHS
        )
        + (
            Cell(
                why="in place, out is x (T=1000): rows read whole before written",
                spec=dict(kind="in_place", rows=1000),
            ),
            Cell(
                why="x a column slice of [300, 1024]: row-strided input",
                spec=dict(kind="x_column_slice", rows=300),
            ),
            Cell(
                why="x [300, 1, 512]: leading dims viewed as one row dim",
                spec=dict(kind="x_3d", rows=300),
            ),
        )
        + tuple(
            Cell(
                why=f"scatter T={t}, {str(sd).removeprefix('torch.')} slots, into bf16 "
                f"[{_POOL_PAGES[ps]}, {ps}, 512]" + (": 1/8 of rows -1" if t >= 8 else ""),
                spec=dict(kind="scatter", rows=t, page_size=ps, slot_dtype=sd),
            )
            for ps in _POOL_PAGES
            for t in _SCATTER_ROWS
            for sd in _SLOT_DTYPES
        )
        + (
            Cell(
                why="pool the left half of [40, 128, 1024]: strided paged out",
                spec=dict(kind="pool_left_half", rows=777),
            ),
            Cell(
                why="flat pool [5120, 640][:, 64:576], int32 slots: column slice",
                spec=dict(kind="pool_column_slice", rows=777),
            ),
            Cell(
                why="out-of-range slots 256, 257, 1e9, -1, -1e9 skipped",
                spec=dict(kind="out_of_range", rows=8),
            ),
        )
        + tuple(
            Cell(
                why=f"native 528 B interleaved rows, T={t}: codes then UE8M0 bytes",
                spec=dict(kind="native_interleaved", rows=t),
            )
            for t in _NATIVE_ROWS
        )
        + tuple(
            Cell(
                why=f"native page footer, page_size {ps}, T=2000: fp8 codes, e8m0 scales",
                spec=dict(kind="native_footer", rows=2000, page_size=ps),
            )
            for ps in (128, 256)
        )
        + (
            Cell(
                why="native identity T=4096, fp8 codes + uint8 scales: floor byte 105",
                spec=dict(kind="native_identity", rows=4096),
            ),
            Cell(
                why="zero rows, mapped: no launch, the pool untouched",
                spec=dict(kind="zero_rows", mapped=True),
            ),
            Cell(
                why="zero rows, identity: no launch, the pool untouched",
                spec=dict(kind="zero_rows", mapped=False),
            ),
        )
    )

    note = """
    The kernel's scale exponent is a bit-level ceil-log2 of `fp32(amax * fp32(1/448))`; for every
    bf16 amax and at the floor it equals the mathematical `ceil(log2(amax / 448))`, including
    `amax = 7 * 2^k` where `amax / 448` is an exact power of two. (A float64 `torch.log2` is not a
    valid way to compute it: `ceil(log2(56 / 448))` came out -2, one too high.) The scale is
    rounded up, so nothing saturates; the floor block's byte is 105 (2^-22). Native dequantization
    `e4m3(code) * 2^(byte - 127)` is bit-identical to the bf16 container.

    Metadata consumed: none. Slot addresses come only from `slot_mapping` and `out`'s geometry.

    Preconditions the op raises ValueError on: `x` not bf16, not CUDA, rank < 2, last dim not a
    positive multiple of 32, non-unit inner stride; `out` last dim != W or non-unit stride, not
    2-D/3-D, a dtype other than bf16 / float8_e4m3fn / uint8, or on another device; a native
    `out` without `out_scale`, a bf16 `out` with one, `out_scale` of the wrong dtype or shape;
    `slot_mapping` not int32/int64 or not one entry per row; without `slot_mapping`, `out` with
    fewer slots than rows. `x`'s leading dims must be viewable as one row dim (row strides
    free). Unchecked: `x` must be finite (a non-finite row changes no other row, but what is
    written for it is not certified); two rows mapped to one slot race; `out` may be `x` itself
    only with `slot_mapping=None`, any other overlap is a race.

    CUDA graphs: make one eager call of the same variant before capturing (Triton compiles on
    first use). A variant is the width, identity vs mapped, bf16 vs native, whether the slot
    map's stride is 1 and its pointer 16-byte aligned, and, in the identity form only, the class
    of the row count (1, a multiple of 16, any other) since there the slot count is the row count.
    The mapped form compiles one variant whatever T: rows are grid axis 0, never an argument.

    Certified by the rest of this entry's test, beyond the cells, on sm_103: every row count
    1..8192 in the DSpark drafter's identity call (`out` the first T rows of a longer buffer, the
    rest untouched) and CUDA-graph capture at every draft block N = 5G; every row count 1..32768
    in the paged-scatter call; bf16 SWA pools `[B, 128, 512]` up to B = 958,720 pages and native
    `[B, 128, 528]` pools of the live DeepSeek-V4.1 cache manager (native KV storage,
    `get_v41_native_kv_views`, strides (67584, 528, 1)) up to B = 1,648,320 pages (the
    measured maxima over the scanned engine configurations including PP2 stages, at 0.5 x total
    memory), with far-end slots at every byte offset 2^31..2^36, pool-size invariance and a wrap
    control; CUDA-graph capture at T = G and 6G for every graph batch G in 1..32, 40..128 step 8,
    192, 256; the DSpark drafter's stride-3 stage-ring writes (captured at M = G and 5G, eager
    at every M 1..1280, seeding n 1..128); native writes over live single- and multi-request
    steps (up to 16384 tokens and 256 requests, positions up to 32,781), round-tripped through
    `torch.ops.trtllm.mxfp8_dequantize`.

    Not certified: non-finite values; duplicate slots; a first-ever call inside a capture; a
    captured row count outside those listed; an identity-form capture whose `out` has more slots
    than T; more than 32768 rows in one call; torch.compile (a fake is registered, not
    exercised); a native pool above 1,648,320 pages or a bf16 pool above 958,720; page sizes
    other than 1, 128, 256; widths other than 32, 128, 448, 512, 1024.
    """

    def raw_call(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        slot_mapping: Optional[torch.Tensor] = None,
        out_scale: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        torch.ops.trtllm.mxfp8_fake_quantize(x, out, slot_mapping, out_scale)
        return out if out_scale is None else (out, out_scale)

    def reference(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        slot_mapping: Optional[torch.Tensor] = None,
        out_scale: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """The expected `out` (and `out_scale`) after the call: copies with the rows' slots filled.

        `out` and `out_scale` are read for their prior contents (every slot the call does not write
        keeps them) and are not mutated; pass snapshots taken before the op ran (for the in-place
        form, a snapshot of `x` as both). The scale exponent comes from `torch.frexp` of the fp64
        quotient `amax / 448` (mantissa 0.5 means an exact power of two), not from the kernel's
        IEEE-bit trick; the codes are torch's float8_e4m3fn RNE cast of an exact power-of-two
        scaling.
        """
        w = x.shape[-1]
        xf = x.reshape(-1, w).float().view(-1, w // _BLOCK, _BLOCK)
        rows = xf.shape[0]
        m, e = torch.frexp(xf.abs().amax(-1).clamp_min(_AMAX_FLOOR).double() / _E4M3_MAX)
        e = torch.where(m == 0.5, e - 1, e).to(torch.int32)
        s = torch.exp2(e.double()).float().unsqueeze(-1)
        q = (xf / s).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        if slot_mapping is not None:
            slots = slot_mapping.long()
        else:
            slots = torch.arange(rows, device=x.device)
        live = (slots >= 0) & (slots < _num_slots(out))
        dst = slots[live]
        if out.dtype == torch.bfloat16:
            expected = out.clone()
            _write_slots(expected, dst, (q.float() * s).view(rows, w).to(torch.bfloat16)[live])
            return expected
        expected = out.view(torch.uint8).clone()
        _write_slots(expected, dst, q.view(rows, w).view(torch.uint8)[live])
        expected_scale = out_scale.view(torch.uint8).clone()
        _write_slots(expected_scale, dst, (e + 127).to(torch.uint8)[live])
        return expected.view(out.dtype), expected_scale.view(out_scale.dtype)

    def compare(
        self,
        out: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        ref: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Bitwise over all of `out` (and `out_scale`): bf16 as int16, codes and bytes as uint8.

        The op computes one exact value per element, so there is no tolerance to choose; untouched
        slots must keep their prior bits. On T = 4096 rows covering every path of the recipe the
        gate sees each plausible wrong recipe differ from the op: no 1e-4 floor, `floor` instead of
        `ceil` for the exponent, block 64, and the scale floored at 2^-13 instead of amax at 1e-4
        (SGLang's V4.1 store; on real window KV only the synthetic data discriminates the floors).
        """
        outs = out if isinstance(out, tuple) else (out,)
        refs = ref if isinstance(ref, tuple) else (ref,)
        assert len(outs) == len(refs), (len(outs), len(refs))
        for name, o, r in zip(("out", "out_scale"), outs, refs):
            assert o.dtype == r.dtype, (name, o.dtype, r.dtype)
            assert o.shape == r.shape, (name, o.shape, r.shape)
            a, b = _bits(o).reshape(-1), _bits(r).reshape(-1)
            bad = torch.zeros((), dtype=torch.int64, device=a.device)
            for i in range(0, a.numel(), 1 << 26):  # chunked: a full bool reduction copies to int64
                bad += (a[i : i + (1 << 26)] != b[i : i + (1 << 26)]).sum()
            assert int(bad) == 0, f"{name}: {int(bad)} of {o.numel()} elements differ bitwise"


mxfp8_fake_quantize = Mxfp8FakeQuantize()
