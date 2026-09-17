# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization: packed e2m1 data plus one e4m3 scale per 16 elements."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

_E4M3_MAX = 448.0
_E2M1_MAX = 6.0

#: Midpoints between consecutive e2m1 magnitudes. A value landing exactly on one
#: is a rounding tie; `_TIE_UP[i]` is True when rounding *up* yields the even
#: code, which is the direction round-to-nearest-even takes.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)

#: The kernel builds its per-block output scale through two `rcp.approx.ftz.f32`
#: reciprocals, each ~2^-23 relative. A value within a few fp32 ulps of a
#: midpoint can therefore round to either neighbour, and no reference can say
#: which. Everything further out than this must match bit for bit. Largest
#: deviation observed on this machine: 7.9e-8 relative, ~12x inside the window.
TIE_WINDOW = 2.0**-20


def _pad_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


class _Fp4Quantize(OpWrapper):
    """Quantize to FP4: `(packed e2m1 data [..., K/2] uint8, scale bytes, 1-D)`.

    One scale per `sf_vec_size` contiguous elements of the last dim. NVFP4 --
    the only form any shipped target uses -- is `sf_vec_size=16` with
    `sf_use_ue8m0=False`, which makes the scale an e4m3 byte derived from the
    block maximum and the caller's `global_scale`.

    `is_sf_swizzled_layout` selects the 128x4 swizzled scale order (True) or the
    row-major linear order (False); the buffer is 1-D either way, so the layout
    is something a consumer has to be told, not something its shape shows.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1 is the only caller and reaches this op from both layouts: the
    # swizzled one feeding nvfp4_gemm in the dense and shared-expert MLPs, the
    # linear one feeding the routed-expert path. Widths are its hidden size and
    # its intermediate.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, hidden 7168, swizzled scales -- the dense MLP's input",
            spec=dict(shape=(1, 7168), dtype=torch.bfloat16, vec=16, swizzled=True),
        ),
        Cell(
            why="prefill chunk, hidden 7168, swizzled",
            spec=dict(shape=(2048, 7168), dtype=torch.bfloat16, vec=16, swizzled=True),
        ),
        Cell(
            why="linear scales -- the routed-expert path asks for this order",
            spec=dict(shape=(2048, 7168), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="the activation width, 2 x moe_intermediate, between the two GEMMs",
            spec=dict(shape=(512, 4096), dtype=torch.bfloat16, vec=16, swizzled=True),
        ),
        Cell(
            why="a row count that is not a multiple of the swizzle's 128-row tile",
            spec=dict(shape=(37, 7168), dtype=torch.bfloat16, vec=16, swizzled=True),
        ),
    )

    note = """
    `sf_use_ue8m0=True` selects power-of-two scales instead of e4m3. No shipped
    target uses it and this entry does not certify it.
    """

    def __call__(
        self,
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor],
        sf_vec_size: int,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.fp4_quantize(
            input, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled_layout
        )

    def reference(
        self,
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor],
        sf_vec_size: int,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Native-torch NVFP4, with the scales in the requested layout.

        Per block: `vecmax = max|x|`, the scale is `e4m3(global_scale * vecmax /
        6)`, and the data is the input times `global_scale / scale` rounded to
        e2m1. torch's fp32 -> e4m3 cast emits NaN above 464 where the kernel's
        saturates, so the scale is clamped first -- between 448 and 464 both
        round to 448 anyway.
        """
        packed, sf, _ = self._quantize(input, global_scale, sf_vec_size)
        if not is_sf_swizzled_layout:
            return packed, sf.reshape(-1)
        return packed, self._swizzle(sf)

    def scaled(
        self, input: torch.Tensor, global_scale: Optional[torch.Tensor], sf_vec_size: int
    ) -> torch.Tensor:
        """The pre-rounding product, which is what `near_tie` reads.

        Exposed because a caller cannot reconstruct it from the op's outputs:
        rounding is the step that loses it.
        """
        return self._quantize(input, global_scale, sf_vec_size)[2]

    @staticmethod
    def near_tie(scaled: torch.Tensor) -> torch.Tensor:
        """Elements whose rounding this entry refuses to predict.

        Inside `TIE_WINDOW` of an e2m1 midpoint the kernel's approximate
        reciprocals decide the direction, so a code that differs there is not a
        defect. Outside it, a difference is.
        """
        mid = torch.tensor(_E2M1_MIDPOINTS, device=scaled.device, dtype=torch.float32)
        a = scaled.abs().unsqueeze(-1)
        return ((a - mid).abs() <= TIE_WINDOW * a).any(-1)

    def is_valid(
        self,
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor],
        sf_vec_size: int,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> None:
        # The kernel loads one scalar from global_scale and ignores every
        # element past the first, so a per-token [num_tokens] tensor is silently
        # applied as global_scale[0] to every row -- observed to return a
        # valid-looking result, never to raise.
        assert global_scale is None or global_scale.numel() == 1, (
            "global_scale must hold exactly one element; extra elements are "
            "silently ignored and global_scale[0] is applied to every row"
        )
        assert input.shape[-1] % sf_vec_size == 0, (
            f"the quantized dim must be a whole number of {sf_vec_size}-element "
            f"blocks; got {input.shape[-1]}"
        )
        assert input.is_contiguous(), "input must be contiguous; the kernel reads it flat"

    # ── the arithmetic, shared by reference() and scaled() ──────────────────

    def _quantize(
        self, input: torch.Tensor, global_scale: Optional[torch.Tensor], vec: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gs = 1.0 if global_scale is None else global_scale.float().reshape(())
        k = input.shape[-1]
        rows = input.numel() // k
        flat = input.reshape(rows, k)
        vecmax = flat.abs().reshape(rows, k // vec, vec).amax(-1).float()
        sf = (gs * (vecmax / _E2M1_MAX)).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
        out_scale = torch.where(vecmax != 0, gs / sf.float(), torch.zeros_like(vecmax))
        scaled = (flat.float().reshape(rows, k // vec, vec) * out_scale.unsqueeze(-1)).reshape(
            rows, k
        )
        codes = self._e2m1_codes(scaled)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        return packed, sf.view(torch.uint8), scaled

    @staticmethod
    def _e2m1_codes(v: torch.Tensor) -> torch.Tensor:
        """fp32 -> 4-bit e2m1 codes (bit 3 = sign), round-to-nearest-even."""
        mid = torch.tensor(_E2M1_MIDPOINTS, device=v.device, dtype=torch.float32)
        tie_up = torch.tensor(_TIE_UP, device=v.device, dtype=torch.bool)
        a = v.abs().unsqueeze(-1)
        greater = (a > mid).sum(-1)  # strict: ties provisionally round down
        up = ((a == mid) & tie_up).any(-1)  # ties whose even side is up
        code = (greater + up).to(torch.uint8)
        negative = (v < 0) | ((v == 0) & torch.signbit(v))
        return code | (negative.to(torch.uint8) << 3)

    @staticmethod
    def _swizzle(sf: torch.Tensor) -> torch.Tensor:
        """Scatter [rows, cols] scale bytes into the 128x4-swizzled buffer."""
        rows, cols = sf.shape
        padded_cols, padded_rows = _pad_up(cols, 4), _pad_up(rows, 128)
        r = torch.arange(rows, device=sf.device).view(-1, 1)
        c = torch.arange(cols, device=sf.device).view(1, -1)
        index = (
            (c % 4)
            + (c // 4) * (4 * 128)
            + (r % 32) * 16
            + ((r % 128) // 32) * 4
            + (r // 128) * (128 * padded_cols)
        )
        out = torch.zeros(padded_rows * padded_cols, dtype=torch.uint8, device=sf.device)
        out[index.reshape(-1)] = sf.reshape(-1)
        return out


fp4_quantize = _Fp4Quantize()
