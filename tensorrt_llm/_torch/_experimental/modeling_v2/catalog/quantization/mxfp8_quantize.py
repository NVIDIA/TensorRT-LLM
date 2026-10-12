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


# DeepSeek-V4.1's dense FP8 linear inputs, (K, which linear): each quantized
# with the default call (swizzled, alignment 32) and handed unchanged to the
# MXFP8 GEMM. The wo_b input is 8 / tp * 1024 per rank. Every row count
# 1..32768 is certified at each K by the row sweep in the test; the cells hold
# the sweep's two ends.
_DSV41_K = (
    (5120, "hidden (wqkv_a, shared w1 || w3)"),
    (1280, "q_lora (wq_b, indexer wq_b)"),
    (2048, "wo_b under attention TP4"),
    (8192, "wo_b under attention DP"),
    (2304, "shared w2"),
    (6144, "Engram wkv"),
    (4096, "wo_b under attention TP2"),
    (1024, "wo_b under attention TP8"),
)
_DSV41_ROWS = (1, 32768)
# The DSpark drafter's two calls the core target does not make, (tag, K,
# swizzled, alignment, what): every row count 1..1280 is certified, the cells
# hold the two ends.
_DSV41_DSPARK = (
    ("MoE in", 5120, False, 512, "the MXFP4 MoE runner's linear-layout input"),
    ("main_proj", 15360, True, 32, "main_proj's input"),
)
_DSV41_DSPARK_ROWS = (1, 1280)

_DSV41_CELLS = tuple(
    Cell(
        why=f"deepseek-v4.1 K={k} M={m}: {site} input, swizzled, alignment 32, "
        f"{'one row' if m == 1 else 'top'} of the row sweep 1..32768",
        spec=dict(
            shape=(m, k), dtype=torch.bfloat16, swizzled=True, alignment=32, data="block_scaled"
        ),
    )
    for k, site in _DSV41_K
    for m in _DSV41_ROWS
) + tuple(
    Cell(
        why=f"deepseek-v4.1 DSpark {tag} M={m}: {what}, K {k}, "
        f"{'swizzled' if swizzled else 'linear'}, alignment {alignment}, "
        f"{'one row' if m == 1 else 'top'} of the row sweep 1..1280",
        spec=dict(
            shape=(m, k),
            dtype=torch.bfloat16,
            swizzled=swizzled,
            alignment=alignment,
            data="block_scaled",
        ),
    )
    for tag, k, swizzled, alignment, what in _DSV41_DSPARK
    for m in _DSV41_DSPARK_ROWS
)


class Mxfp8Quantize(OpWrapper):
    """Quantize to MXFP8: `(e4m3 data [..., pad_up(K, alignment)], uint8 scales)`.

    One scale per 32 contiguous elements of the last dim. The scale is the
    smallest power of two at or above `amax / 448`, carried as its E8M0 byte
    (`exponent + 127`) rather than a float.

    `swizzled_layout` selects the 128x4 swizzled scale order (True) or the
    row-major linear order (False). The scale tensor is 1-D either way; the
    layout is what a consumer kernel expects, not something the shape shows.
    """

    ARCHS = frozenset({Arch.SM_103})

    # gpt-oss asks for the *linear* layout with the 512-element alignment its
    # MoE FC1 needs -- hidden 2880 pads to 3072. The swizzled cell at its width
    # is here because the two layouts are separate code paths in the kernel;
    # DeepSeek-V4.1, whose cells follow, is the target that takes the swizzled
    # layout.
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
        # DeepSeek-V4.1: see `_DSV41_K` and `_DSV41_DSPARK`. `data` names the
        # input's value pattern (the test's `_build` documents it).
        *_DSV41_CELLS,
    )

    note = """
    Blocks whose amax is below `448 * 2^-127` take a flush-to-zero path in the
    kernel that `reference` does not model; no shipped target produces one,
    since the residual stream it quantizes never gets that small.

    Leading dims are collapsed: a [B, T, K] input quantizes as [B*T, K], and
    under the swizzled layout the collapsed rows are padded as one matrix.

    DeepSeek-V4.1: every dense FP8 linear of the core target calls it as
    `mxfp8_quantize(x)` (swizzled, alignment 32) on contiguous bf16 `[M, K]`
    at the eight K of `_DSV41_K` and feeds both outputs unchanged to the MXFP8
    GEMM. Metadata none. The row count is the rank's token count, and the
    bounded decoder replay of layers 21-39 can reach 32768 rows uncut; every
    M 1..32768 is certified at each K, bit-exact in every data byte, every
    scale byte at its 128x4 offset and every row-padding byte. All eight K are
    multiples of 128, so no scale column is padded; rows pad to 128. The
    DSpark drafter adds its MoE input (linear, alignment 512, K 5120, the
    layout the MXFP4 MoE runner consumes) and main_proj's input (swizzled,
    alignment 32, K 15360), each certified at every M 1..1280. CUDA graphs,
    after an eager call at each shape: the eight K at the 114 per-rank row
    counts {G, 5G, 6G} over the 46 captured batch sizes up to 256, plus 1935;
    the two drafter calls at 176 counts (every M 1..160 plus G and 5G).
    Data in the certification has block maxima spread over 2^-24..2^24 so a
    misplaced scale byte is visible. Not certified for V4.1: fp16 input; more
    than 32768 rows; another K in the swizzled layout; an alignment above 32
    at the eight K; the linear layout beyond the MoE-input call; capture at
    other row counts, at a batch size outside that list, or with no eager call
    first.
    """

    def raw_call(
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


mxfp8_quantize = Mxfp8Quantize()
