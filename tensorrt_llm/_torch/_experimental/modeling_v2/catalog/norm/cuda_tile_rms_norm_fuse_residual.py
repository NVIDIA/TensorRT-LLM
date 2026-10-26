# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place residual add + (zero-centred) RMS normalization through the CUDA Tile op
`torch.ops.trtllm.cuda_tile_rms_norm_fuse_residual_`."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: Step-5-Preview's residual stream: hidden 4096, fp32 activations and fp32 norm
#: weights, RMSNorm eps 1e-5, zero-centred `(1 + w)` weights.
HIDDEN = 4096
EPS = 1e-5

#: (static_persistent, gather) -> the kernel the op launches.
PATHS = {
    (True, True): "persistent",
    (False, False): "tiled",
    (False, True): "gather",
}


def _cell(rows: int, static_persistent: bool, gather: bool, why: str) -> Cell:
    return Cell(
        why=why,
        spec=dict(
            rows=rows,
            hidden=HIDDEN,
            dtype=torch.float32,
            eps=EPS,
            use_gemma=True,
            static_persistent=static_persistent,
            gather=gather,
        ),
    )


class CudaTileRmsNormFuseResidual(OpWrapper):
    """`residual += x; x = residual / rms(residual) * (1 + w)` (or `* w` without
    `use_gemma`), in place, returns `(x, residual)` -- the same two tensors.

    The sum is formed in fp32 and stored rounded once to `residual.dtype`; the
    normalization reduces in fp32 and the store rounds once to `x.dtype`.
    `static_persistent=True` launches the static-persistent kernel (one weight
    load per program, `gather` ignored); `static_persistent=False` launches
    one program per row, tiled loads (`gather=False`) or pointer gathers
    (`gather=True`).

    Fusion boundary: add, normalization and the weight multiply. No output
    quantization, no bias.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, hidden, dtype (x, residual and weight share it), eps,
    # use_gemma, static_persistent, gather. Values are drawn over the
    # checkpoint's measured ranges (see the test's module docstring).
    CELLS: tuple[Cell, ...] = (
        _cell(1, True, True, "step5 persistent, 1 decode row fp32 x 4096"),
        _cell(37, True, True, "step5 persistent, 37 rows: not a multiple of the 4-row tile"),
        _cell(128, True, True, "step5 persistent, 128 rows: decode graph ceiling"),
        _cell(8192, True, True, "step5 persistent, 8192-row prefill: max_num_tokens ceiling"),
        _cell(1, False, False, "step5 tiled, 1 decode row fp32 x 4096"),
        _cell(128, False, False, "step5 tiled, 128 rows: decode graph ceiling"),
        _cell(8192, False, False, "step5 tiled, 8192-row prefill: max_num_tokens ceiling"),
        _cell(1, False, True, "step5 gather, 1 decode row fp32 x 4096"),
        _cell(128, False, True, "step5 gather, 128 rows: decode graph ceiling"),
        _cell(8192, False, True, "step5 gather, 8192-row prefill: max_num_tokens ceiling"),
    )

    note = """
    Registered only when the CUDA Tile toolchain is present
    (`IS_CUDA_TILE_AVAILABLE`: `cuda.tile` importable, `tileiras` found,
    compute capability >= 10.0); the op has no non-tile fallback inside it, so
    a call that returns ran a cuTile kernel. Each (dtype, shape class, eps,
    use_gemma, path) compiles once on first call (about 0.2-0.9 s measured on
    GB300); warm up outside CUDA-graph capture. Refuses SM 107 loudly.

    Metadata consumed: none. Writes both `x` and `residual`; the non-persistent
    paths also allocate a scratch fp32 [M] rstd per call. `weight` is
    `.contiguous()`d first (a strided weight is accepted and correct).

    Certified (all ten cells): contiguous 2-D fp32 x, residual and weight,
    hidden 4096, eps 1e-5, use_gemma=True, on the persistent path
    (static_persistent=True, gather=True as the engine's RMSNorm passes it) at
    1, 37, 128 and 8192 rows, and on the tiled and gather paths at 1, 128 and
    8192 rows. The residual output is bit-exact (fp32 + fp32 rounded once);
    the normalized output is within 5 fp32 ulp (2^-23) per element relative to
    an fp64 reference, measured <= 3.13 ulp over synthetic rows at 1..8192 and
    over the oracle's real residual stream entering layers 4 and 88. The
    persistent and non-persistent paths are not bitwise equal to each other
    (<= 2.7 ulp apart); tiled and gather are bitwise equal. Rows past a view's
    end are untouched at 1, 3, 37 and 8191 rows on every path. Capture and
    replay of a 128-row call reproduce eager bitwise on every path.

    Silent wrong answers (refused by `is_valid`, measured on GB300):
    - `x` and `residual` the same buffer: returns norm(2x), residual lost.
    - a weight shorter than the row: the persistent and gather paths read the
      missing tail as w = 0 (scale 1 under use_gemma), the tiled path returns
      other wrong values.
    - a residual with fewer rows than `x`: the persistent path reads the
      missing rows as 0, the tiled path returns non-finite values there.
    - the tiled path (static_persistent=False, gather=False) when the row width
      is not a multiple of its tile `min(2048 // x.element_size(),
      next_pow2(N))`: the ragged last tile corrupts the normalized output
      (fp32 N 4000), while the persistent and gather paths are correct there.
    A longer weight uses its first N values, correctly; a residual with more
    rows or a wider row stride is read correctly and its extra rows are not
    written.

    Loud failures (left to the op): non-contiguous `x` or `residual`
    ("x must be contiguous for in-place operation"); a 3-D `x` ("too many
    values to unpack (expected 2)"); 0 rows ("Failed to launch cuTile kernel:
    invalid argument"); a CPU weight ("Input array is not on a CUDA device").

    Accepted, not certified: bf16/fp16 activations; mixed dtypes. Measured
    with fp32 x and a bf16 residual: the residual stores the bf16-rounded sum;
    the persistent path normalizes the unrounded fp32 sum, while the
    non-persistent paths match neither that nor the normalized bf16 sum (their
    second pass reloads the rounded residual). A bf16 weight is widened
    correctly. Also uncertified: N other than 4096 (N 4000 measured correct on
    the persistent and gather paths), use_gemma=False, eps other than 1e-5,
    and row counts other than the cells' (2, 3 and 8191 measured within the
    band).

    Hard limits (computed; the envelope stays far below them): element offsets
    are `row * N + col`, which crosses 2^31 at 2^31 / 4096 = 524288 rows (the
    envelope's ceiling is 8192). The tiled and gather paths launch one program
    per row (grid x = M, under 2^31 - 1). The fp32 sum of squares overflows
    when a row's mean square exceeds 3.4e38 / 4096, i.e. row RMS above about
    2.9e17; the checkpoint's residual stream measured max |x| 9375 and row RMS
    <= 166. No index is carried through fp32 (N enters only as the divisor of
    the mean, exact at 4096).
    """

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        static_persistent: bool,
        gather: bool,
        use_gemma: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes `x` (normalized) and `residual` (the sum) in place and returns
        both, in that order."""
        torch.ops.trtllm.cuda_tile_rms_norm_fuse_residual_(
            x, residual, weight, eps, static_persistent, gather, use_gemma
        )
        return x, residual

    def reference(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        static_persistent: bool,
        gather: bool,
        use_gemma: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two values the op writes, as (x_out, residual_out), in fp64 and
        rounded once each. `static_persistent` and `gather` choose a kernel,
        not a result, and are ignored. Returns rather than mutates."""
        h = x.double() + residual.double()
        normed = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
        scale = weight.double() + 1.0 if use_gemma else weight.double()
        return (normed * scale).to(x.dtype), h.to(residual.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        static_persistent: bool,
        gather: bool,
        use_gemma: bool,
    ) -> None:
        # Only inputs the op takes and answers wrongly; see the note.
        assert x.data_ptr() != residual.data_ptr(), (
            "x and residual must be distinct buffers; the op writes both"
        )
        n = x.shape[-1]
        assert weight.numel() >= n, (
            f"weight has {weight.numel()} elements for rows of {n}; the op reads "
            f"the missing tail as padding"
        )
        assert residual.shape[0] >= x.shape[0], (
            f"residual has {residual.shape[0]} rows for {x.shape[0]} rows of x; "
            f"the op reads the missing rows as padding"
        )
        if not static_persistent and not gather:
            tile = min(2048 // x.element_size(), 1 << max(n - 1, 0).bit_length())
            assert n % tile == 0, (
                f"the tiled path (static_persistent=False, gather=False) needs the "
                f"row width {n} to be a multiple of its tile {tile}; use the "
                f"persistent or gather path"
            )


cuda_tile_rms_norm_fuse_residual = CudaTileRmsNormFuseResidual()
