# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 quantization: e4m3 data plus one UE8M0 scale per 32 elements."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

_E4M3_MAX = 448.0
_BLOCK = 32


def _pad_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


class _Mxfp8Quantize(OpWrapper):
    """Quantize to MXFP8: `(e4m3 data [..., pad_up(K, alignment)], uint8 scales)`.

    One scale per 32 contiguous elements of the last dim. The scale is the
    smallest power of two at or above `amax / 448`, carried as its E8M0 byte
    (`exponent + 127`) rather than a float.

    `swizzled_layout` selects the 128x4 swizzled scale order (True) or the
    row-major linear order (False). The scale tensor is 1-D either way; the
    layout is what a consumer kernel expects, not something the shape shows.
    """

    ARCHS = frozenset({Arch.SM_103})

    # gpt-oss is the only shipped caller, and it asks for the *linear* layout
    # with the 512-element alignment its MoE FC1 needs -- hidden 2880 pads to
    # 3072. The swizzled cell is here because the argument exists and the two
    # layouts are separate code paths in the kernel, not because a target uses
    # it today.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="gpt-oss hidden 2880, linear scales, aligned to 512 -> padded 3072",
            spec=dict(shape=(16, 2880), dtype=torch.bfloat16, swizzled=False, alignment=512),
        ),
        Cell(
            why="decode-shaped: one token through the same padding",
            spec=dict(shape=(1, 2880), dtype=torch.bfloat16, swizzled=False, alignment=512),
        ),
        Cell(
            why="prefill-shaped, and enough rows to cross the swizzle's 128-row tile",
            spec=dict(shape=(2048, 2880), dtype=torch.bfloat16, swizzled=False, alignment=512),
        ),
        Cell(
            why="the swizzled scale order, the other of the kernel's two paths",
            spec=dict(shape=(160, 2880), dtype=torch.bfloat16, swizzled=True, alignment=512),
        ),
    )

    note = """
    Blocks whose amax is below `448 * 2^-127` take a flush-to-zero path in the
    kernel that `reference` does not model; no shipped target produces one,
    since the residual stream it quantizes never gets that small.

    Leading dims are collapsed: a [B, T, K] input quantizes as [B*T, K], and
    under the swizzled layout the collapsed rows are padded as one matrix.
    """

    def __call__(
        self,
        input: torch.Tensor,
        swizzled_layout: bool = True,
        alignment: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.mxfp8_quantize(input, swizzled_layout, alignment)

    def reference(
        self,
        input: torch.Tensor,
        swizzled_layout: bool = True,
        alignment: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Native-torch MXFP8, returning the scales in the requested layout.

        Columns beyond the input's K, up to the aligned width, are zero -- an
        all-zero block quantizes to scale byte 0 and zero data.
        """
        k = input.shape[-1]
        rows = input.numel() // k
        padded_k = _pad_up(k, alignment)

        padded = torch.zeros(rows, padded_k, dtype=torch.float32, device=input.device)
        padded[:, :k] = input.reshape(rows, k).float()
        blocks = padded.view(rows, padded_k // _BLOCK, _BLOCK)
        amax = blocks.abs().amax(dim=-1)

        # E8M0 of amax/448 rounded toward +inf: take the fp32 biased exponent
        # and bump it whenever any mantissa bit is set, i.e. whenever the value
        # is not already a power of two. amax is exact in fp32 for bf16 and
        # fp16 inputs, so this is the exact round-up rather than an estimate.
        bits = (amax / _E4M3_MAX).view(torch.int32)
        byte = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) > 0).to(torch.int32)).clamp(0, 254)
        scale = torch.exp2(byte.to(torch.float32) - 127.0).unsqueeze(-1)
        data = (blocks / scale).view(rows, padded_k).to(torch.float8_e4m3fn)

        byte = byte.to(torch.uint8)
        if not swizzled_layout:
            return data, byte.reshape(-1)
        return data, self._swizzle(byte)

    @staticmethod
    def _swizzle(byte: torch.Tensor) -> torch.Tensor:
        """Scatter [rows, cols] scale bytes into the 128x4-swizzled buffer."""
        rows, cols = byte.shape
        padded_cols = _pad_up(cols, 4)
        padded_rows = _pad_up(rows, 128)
        r = torch.arange(rows, device=byte.device).view(-1, 1)
        c = torch.arange(cols, device=byte.device).view(1, -1)
        index = (
            (c % 4)
            + (c // 4) * (4 * 128)
            + (r % 32) * 16
            + ((r % 128) // 32) * 4
            + (r // 128) * (128 * padded_cols)
        )
        out = torch.zeros(padded_rows * padded_cols, dtype=torch.uint8, device=byte.device)
        out[index.reshape(-1)] = byte.reshape(-1)
        return out

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Bit-exact.

        Both sides compute the same power-of-two scale and the same e4m3
        rounding, so any difference is a defect rather than accumulated error.
        fp8 and uint8 both compare as bytes.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8)), (
            "quantization is deterministic; a mismatch is a defect, not drift"
        )

    def is_valid(
        self,
        input: torch.Tensor,
        swizzled_layout: bool = True,
        alignment: int = 32,
    ) -> None:
        # A last dim that is not a whole number of blocks leaves a partial
        # block the kernel reads past the end of, and an alignment that is not
        # a multiple of the block size pads to a width the scale count cannot
        # describe. Neither is rejected by the op.
        assert alignment % _BLOCK == 0, (
            f"alignment must be a multiple of the {_BLOCK}-element scale block; got {alignment}"
        )
        assert input.shape[-1] % _BLOCK == 0, (
            f"the quantized dim must be a whole number of {_BLOCK}-element blocks; "
            f"got {input.shape[-1]}"
        )
        assert input.is_contiguous(), "input must be contiguous; the kernel reads it flat"


mxfp8_quantize = _Mxfp8Quantize()
