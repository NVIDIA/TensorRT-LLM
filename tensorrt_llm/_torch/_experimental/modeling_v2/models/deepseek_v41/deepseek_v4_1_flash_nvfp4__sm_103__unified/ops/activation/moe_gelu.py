# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact (erf) GELU over the rows of a tile-laid-out 2-D tensor via the MoE activation kernel.

The op is `torch.ops.trtllm.moe_gelu`, registered in
`cpp/tensorrt_llm/thop/moe/cuteDsl/cuteDslMoeUtilsOp.cpp`.
"""

from __future__ import annotations

import math

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The DeepSeek-V4.1 vision aligner applies GELU to w1's output, [C, 5120].
_WIDTH = 5120

#: Row counts run against the reference with one full tile, at N(0, 1) and N(0, 3^2):
#: decode-sized to multi-image prefill-sized.
_DENSE_ROWS = (1, 9, 100, 1024, 4096, 8192)

#: The aligner's own row counts: the golden cases' C and the MMMU maximum.
_ALIGNER_ROWS = (
    (182, "golden-case row count"),
    (546, "golden-case row count"),
    (601, "golden-case row count"),
    (986, "golden-case row count"),
    (1902, "the largest C over the MMMU val documents"),
    (2048, "the aligner's row-count ceiling"),
)


class MoeGelu(OpWrapper):
    """Exact GELU `0.5 * x * (1 + erf(x / sqrt(2)))`, fp32 math, rounded to `x.dtype`.

    `x` is `[num_tiles * tile_tokens_dim, width]`. Row `r` of tile
    `t = r // tile_tokens_dim` is computed only when
    `t < num_non_exiting_tiles[0]` and `r < tile_idx_to_mn_limit[t]`; every
    other output row is left unwritten. For a plain dense activation over all
    `rows` rows use one tile: `tile_idx_to_mn_limit = [rows]` (int32),
    `num_non_exiting_tiles = [1]` (int32), `tile_tokens_dim = rows`.

    Fusion boundary: only the activation is inside the call. The caller owns the
    `w1` projection and its bias that produce `x`, and the `w2` projection that
    consumes the output; no gating, bias or quantization here. Returns a newly
    allocated contiguous tensor of `x`'s shape and dtype; `x` is not mutated.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"rows={rows} N(0,{sigma:g}^2): [rows, 5120] bf16, one full tile",
                spec=dict(kind="randn", rows=rows, limit=rows, sigma=sigma),
            )
            for rows in _DENSE_ROWS
            for sigma in (1.0, 3.0)
        )
        + tuple(
            Cell(
                why=f"aligner C={rows} N(0,3^2): {reason}",
                spec=dict(kind="randn", rows=rows, limit=rows, sigma=3.0),
            )
            for rows, reason in _ALIGNER_ROWS
        )
        + (
            Cell(
                why="every bf16 bit pattern: [13, 5120], +-inf/NaN/+-0 included",
                spec=dict(kind="bits", rows=13, limit=13),
            ),
            Cell(
                why="limit 37 < rows 64: rows at or past the limit unwritten",
                spec=dict(kind="randn", rows=64, limit=37, sigma=3.0),
            ),
        )
    )

    note = """
    Metadata consumed: the tile metadata, read on the device by the kernel.
    `num_non_exiting_tiles[0] * tile_tokens_dim` bounds the rows walked and
    `tile_idx_to_mn_limit[t]` is the exclusive end row of tile `t`. The vision
    aligner builds the single-tile values host-side once per request, `[C]` and
    `[1]` int32 on CUDA with `tile_tokens_dim = C`; no runtime or attention
    metadata is read.

    Preconditions: `x` 2-D, contiguous, bf16 on CUDA with `width % 8 == 0`;
    `x.shape[0] == tile_tokens_dim * tile_idx_to_mn_limit.size(0)`; both metadata
    tensors int32 on CUDA, `num_non_exiting_tiles` one element. The kernel
    ignores strides (see `is_valid`). Rows outside the metadata are not written
    and the output is a fresh `torch::empty`: a stale limit one row short (left
    over from another request's C) leaves the last row holding whatever the
    allocator block held, with no error, so the limit tensor must be rebuilt or
    rewritten for every C. Under CUDA graphs the metadata is read at replay from
    the captured addresses: a graph is valid only for the C its metadata holds.
    The op raises RuntimeError on a row count that is not
    `tile_tokens_dim * num_tiles`, non-int32 limits, `width % 8 != 0`, an fp32
    input and a non-2-D input.

    Numerics: the erf form, not the tanh approximation (they differ at bf16
    only for x < -2.3). Measured on sm_103 over all 65536 bf16 bit patterns the
    output is bit-identical to `F.gelu(x, approximate="none")` on the bf16
    tensor, which is what the aligner computes; against fp64 erf GELU the 115
    mismatching patterns all lie in x in [-8.3125, -4.25], where fp32
    `1 + erf(...)` cancels, at most 2.4e-7 absolute. `+inf -> inf`,
    `-inf -> NaN`, `NaN -> NaN`, `+-0 -> +-0`, as `F.gelu`.

    One launch of one ahead-of-time compiled kernel at every row count 1..2048,
    with the PDL launch attribute; no autotuning or tactic choice. Every row
    count 1..2048 in the aligner's call form returns exactly the first rows of
    the 2048-row result.

    Not certified: fp16 input, widths other than 5120, multi-tile layouts,
    zero rows, rows 2049..4095, 4097..8191 and above 8192, CUDA-graph capture at
    row counts other than 182, 546, 601, 986 and 1902, and PDL behaviour under
    a producer kernel that triggers early completion.
    """

    def raw_call(
        self,
        x: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        tile_tokens_dim: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.moe_gelu(
            x, tile_idx_to_mn_limit, num_non_exiting_tiles, tile_tokens_dim
        )

    def reference(
        self,
        x: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        tile_tokens_dim: int,
    ) -> torch.Tensor:
        """Exact GELU in fp64, rounded once to `x.dtype`, on the rows the metadata selects.

        Rows the op leaves unwritten (tile at or past `num_non_exiting_tiles[0]`,
        or row at or past its tile's limit) are NaN here: `compare` then matches
        them only where the output block held NaN before the call, which is how
        the test shows those rows were left alone.
        """
        xd = x.double()
        ref = (0.5 * xd * (1.0 + torch.erf(xd / math.sqrt(2.0)))).to(x.dtype)
        rows = torch.arange(x.shape[0], device=x.device)
        tile = rows // tile_tokens_dim
        limits = tile_idx_to_mn_limit.to(device=x.device, dtype=torch.int64)
        written = (tile < int(num_non_exiting_tiles.reshape(-1)[0].item())) & (
            rows < limits[tile.clamp_max(limits.numel() - 1)]
        )
        return ref.masked_fill(~written.unsqueeze(-1), float("nan"))

    def is_valid(
        self,
        x: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        tile_tokens_dim: int,
    ) -> None:
        # The kernel addresses row r at `data_ptr + r * width` and ignores the
        # tensor's strides, so a non-contiguous view is read as if it were
        # contiguous: on sm_103 a row-strided view (row stride width + 256) gave a
        # bit-exact fraction of 0.01607971265912056 and a transposed view
        # 0.0014892577892169356 against the reference of the view, with no error.
        assert x.is_contiguous(), (
            f"x must be contiguous; got strides {x.stride()} for shape {tuple(x.shape)} "
            f"-- the kernel ignores strides and would read the wrong elements"
        )

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """`assert_close` at its default bf16 tolerance against the fp64 erf reference, NaN equal to NaN.

        Nothing is loosened: measured on sm_103 every case passes the default
        bf16 band (bit-exact fraction 0.998 over all bit patterns, 0.938 on
        N(0, 3^2)), while the tanh-approximated GELU fails it on 123 of the
        66560 bit-pattern elements and 51261 of 5242880 N(0, 1) elements.
        `equal_nan` is what the bit-pattern cell needs (`-inf` and NaN inputs
        give NaN) and what the unwritten-rows convention of `reference` uses.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        torch.testing.assert_close(out, ref, equal_nan=True)


moe_gelu = MoeGelu()
