# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 fake quantization (E2M1 x E4M3 per 16, amax floor 6*2^-9, no global scale) written into
slots of a bf16 container pool, or as native packed E2M1 bytes + E4M3 block-scale bytes, via the
trtllm Triton op.

The op is `torch.ops.trtllm.nvfp4_fake_quantize`, registered in
`tensorrt_llm/_torch/custom_ops/nvfp4_fake_quantize.py`.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops.nvfp4_fake_quantize  # noqa: F401 — registers torch.ops.trtllm.nvfp4_fake_quantize
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_BLOCK = 16
_FLOOR = 6 * 2.0**-9  # the model's amax floor for E4M3-scaled FP4
_E4M3_MAX = 448.0

_FORMS = ("core", "native")
_ROWS = (1, 5, 6, 64, 1024, 8192)
_WIDTHS = (16, 48, 128, 1024)


class Nvfp4FakeQuantize(OpWrapper):
    """Fake-quantize bf16 rows to NVFP4 and write each row to a slot of a pool, in one launch.

    For row `r` and block `b` (elements `16b .. 16b+15` of the last dim)::

        amax  = max(max_i |x[r, 16b+i]|, 6 * 2^-9)          # fp32
        scale = e4m3_rne(min(amax / 6, 448))                # IEEE fp32 division
        y_i   = x[r, 16b+i] / scale                         # IEEE fp32 division
        q_i   = sign(y_i) * E2M1_rne(|y_i|)                 # ties to even, saturates at 6

    The dtype of `out` picks the form. Core (bf16 `out` `[..., W]`): `q_i * scale`, exact
    in bf16. Native (uint8 `out` `[..., W/2]` plus `out_scale` `[..., W/16]`, float8_e4m3fn
    or uint8): the packed codes (`code[2j] | code[2j+1] << 4`, sign in bit 3, kept when
    `q_i` is zero) and the scale bytes. Row `r` goes to slot `slot_mapping[r]`, skipped
    when negative or `>= slots`, or to slot `r` when `slot_mapping` is None; a paged `out`
    `[pages, page_size, C]` maps slot `s` to `(s // page_size, s % page_size)`.

    Fusion boundary: the amax, the scale, the quantization, the dequantization (core form)
    and the scatter into slots are inside the call; RoPE (applied before it), the compressor
    and any read of the pool are not. Writes `out` (and `out_scale`) in place and returns
    `out` in the core form, `(out, out_scale)` in the native form.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"{form} random rows t={t}: [t, 512] dense, slot r = row r",
                spec=dict(kind="random", form=form, t=t, width=512),
            )
            for form in _FORMS
            for t in _ROWS
        )
        + tuple(
            Cell(why=f"{form} {why}", spec=dict(kind=kind, form=form))
            for form in _FORMS
            for kind, why in (
                ("ties", "exact midpoints: every E4M3 scale, +-1 bf16 ulp"),
                ("floor", "amax floor: blocks at/below 6*2^-9, zero, -0.0"),
                ("every_amax", "every bf16 amax: one block each, 448 saturates"),
            )
        )
        + tuple(
            Cell(
                why=f"{form} width {w}: [33, {w}] random multi-scale rows",
                spec=dict(kind="random", form=form, t=33, width=w),
            )
            for form in _FORMS
            for w in _WIDTHS
        )
        + (
            Cell(
                why="core leading dims: x [96, 1, 512] into out [96, 512]",
                spec=dict(kind="leading_dims", form="core"),
            ),
            Cell(
                why="core row stride: x a [96, 640][:, :512] column slice",
                spec=dict(kind="row_stride", form="core"),
            ),
        )
        + tuple(
            Cell(
                why=f"bf16 pool {layout} {idx}: 37 rows into 4096 slots, 3 out of range",
                spec=dict(kind="bf16_pool", form="core", layout=layout, idx_dtype=idx),
            )
            for layout in ("flat", "paged64", "paged128")
            for idx in ("int32", "int64")
        )
        + tuple(
            Cell(
                why=f"native pool {layout}: 37 rows into 4096 slots, 3 out of range",
                spec=dict(kind="native_pool", form="native", layout=layout, idx_dtype="int32"),
            )
            for layout in ("split", "row288", "paged_row288")
        )
        + (
            Cell(
                why="zero rows: an empty batch and slot map write nothing",
                spec=dict(kind="zero_rows", form="core"),
            ),
        )
    )

    note = """
    Cells. Dense calls with slot r = row r in both forms over random multi-scale rows (the
    low end below the floor), every E4M3 scale at exact E2M1 midpoints and one bf16 ulp
    either side, the floor region and every positive finite bf16 amax; widths 16, 48, 128,
    1024; `x` with leading dims and as a column slice; 37 rows into 4096-slot pools (bf16
    flat / 64-row / 128-row pages with int32 and int64 slots; native as split pools, one
    288-byte row, and pages of 288-byte rows), three of them at -1, slots + 5 and -2^30 or
    -2^40; an empty batch.

    Value set: the DeepSeek-V4.1 compressed KV, the model's
    `fp4_act_quant(x, 16, inplace, scale_dtype=float8_e4m3fn)`, measured bit-exact on sm_103
    against the checkpoint's own kernel (every E4M3 scale at its midpoints, the floor region,
    every bf16 amax) and against the golden compressed KV (5,158,912 elements), both forms.
    Scale byte 0x01 at or below the floor, 0x7E (448) for amax >= 2688, never NaN. FlashInfer's
    NVFP4 KV quantize and `trtllm.fp4_quantize` are a different value set: they multiply by a
    reciprocal (3,608 golden elements differ, all exact midpoints) and flush blocks with
    amax <= 6*2^-10 to zero. FlashInfer's `nvfp4_kv_dequantize` rejects the strided 288-byte
    row views (it needs contiguous inputs).

    Metadata consumed: none; the slot map is an argument. DeepSeek-V4.1's target writes the
    native form as `(latent, codes, md.v41_compress_slot_mapping[r][:n], scales)` with `codes`
    / `scales` the cache manager's `[..., :256]` / `[..., 256:]` views of its 288-byte rows.

    Preconditions: `x` bf16, W a positive multiple of 16, unit last-dim stride, leading dims
    viewable as rows; `out` `[slots, C]` or `[pages, page_size, C]`, unit last-dim stride;
    `slot_mapping` int32/int64 `[rows]`; `out_scale` exactly with a uint8 `out`; without a
    slot map `out` has at least `rows` slots. The op raises ValueError otherwise. Use -1 for
    "no slot". `out` may be `x` itself when `slot_mapping` is None (certified at 256 rows).

    CUDA graphs: capture after one eager call has compiled the variant -- (W, slot map or not,
    form), and in the slot-r form the class of the row count (1, a multiple of 16, other).
    Capture is certified at every graph batch G of 1..32, 40..128 step 8, 192, 256 and at 6G
    (ratio 1) / 3G (ratio 2) rows, replay bitwise equal to eager; a first-ever call captured
    is not.

    Envelope, in the test's far-end and sweep cases: bf16 views up to `[2118144, 64, 512]` /
    `[1059584, 128, 512]` and native views up to `[6480780, 64, 288]` / `[3245959, 128, 288]`
    (the PP2 stage-alone cache managers at 0.5 x total memory), rows on both sides of every
    2^31..2^36 offset, nothing else written; the result does not depend on the view's page
    count; every row count 1..16384; multi-request steps on live managers up to 256 requests
    and position 32,781. All offsets are 64-bit.

    Not certified: non-finite input; two rows naming one slot; overlap of `x` and `out` other
    than exact in place; widths other than 16, 48, 128, 512, 1024; more than 16384 rows; a
    capture at a row count outside the list above; views larger than the envelope (a KV quota
    above 0.5 x total memory, pipeline splits other than PP2's even split, unsampled
    max_batch_size); page sizes other than 1, 64, 128; torch.compile.
    """

    def raw_call(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        slot_mapping: Optional[torch.Tensor] = None,
        out_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        torch.ops.trtllm.nvfp4_fake_quantize(x, out, slot_mapping, out_scale)
        return out if out_scale is None else (out, out_scale)

    def reference(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        slot_mapping: Optional[torch.Tensor] = None,
        out_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The whole of `out` (and `out_scale`) after the call: quantized rows in the slots
        the op writes, the prior contents everywhere else.

        The model's recipe in torch ops: fp32 amax clamped to the floor, `amax / 6` cast by
        torch to float8_e4m3fn (round-to-nearest-even, clamped to 448 first), `x / scale` in
        fp32, E2M1 rounding by `torch.round` (half to even) on the three binades, the sign of
        `y` kept. Nothing is mutated: the expected buffers are contiguous copies of `out` /
        `out_scale` with the written slots replaced.
        """
        width = x.shape[-1]
        xf = x.reshape(-1, width).float().unflatten(-1, (-1, _BLOCK))
        amax = xf.abs().amax(-1, keepdim=True).clamp_min(_FLOOR)
        s8 = (amax / 6.0).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
        s = s8.float()
        y = (xf / s).clamp(-6.0, 6.0)
        a = y.abs()
        r = torch.where(
            a < 2,
            torch.round(a * 2) / 2,
            torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2),
        )

        rows = xf.shape[0]
        flat_out = out.clone(memory_format=torch.contiguous_format)
        dest = flat_out.view(-1, out.shape[-1])
        if slot_mapping is None:
            src = torch.arange(rows, device=x.device)
            slots = src
        else:
            slots = slot_mapping.long()
            src = torch.nonzero((slots >= 0) & (slots < dest.shape[0])).flatten()
            slots = slots[src]

        if out_scale is None:
            deq = (torch.copysign(r, y) * s).flatten(-2).to(torch.bfloat16)
            dest[slots] = deq[src]
            return flat_out

        code = torch.where(r < 2, r * 2, torch.where(r == 6, 7.0, r + 2)).to(torch.int32)
        nib = (code | (torch.signbit(y).to(torch.int32) << 3)).flatten(-2)
        packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8)
        flat_scale = out_scale.clone(memory_format=torch.contiguous_format)
        dest[slots] = packed[src]
        scale_bytes = s8.squeeze(-1).view(torch.uint8)
        flat_scale.view(-1, out_scale.shape[-1])[slots] = scale_bytes[src].view(out_scale.dtype)
        return flat_out, flat_scale

    def compare(
        self,
        out: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        ref: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Bitwise on every element of `out` (and `out_scale`), written or not.

        `E2M1 x E4M3` is exact in bf16 and both sides make the same IEEE divisions and the
        same round-to-nearest-even, so a differing element is a wrong scale, tie, nibble or
        slot, and a differing element outside the written slots is a write the op must not
        make. Controls on 1024 random rows plus the tie and floor sets (538,624 elements):
        a power-of-two scale changes 233,906 of them, no amax floor 42,579, ties toward zero
        2,919, `x * (1 / scale)` 190.
        """
        assert isinstance(out, tuple) == isinstance(ref, tuple), (type(out), type(ref))
        pairs = zip(out, ref) if isinstance(out, tuple) else ((out, ref),)
        for got, want in pairs:
            assert got.dtype == want.dtype, (got.dtype, want.dtype)
            assert got.shape == want.shape, (got.shape, want.shape)
            raw = torch.int16 if got.element_size() == 2 else torch.uint8
            bad = (got.view(raw) != want.view(raw)).sum().item()
            assert bad == 0, f"{bad} of {got.numel()} elements of {tuple(got.shape)} differ"


nvfp4_fake_quantize = Nvfp4FakeQuantize()
