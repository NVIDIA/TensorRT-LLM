# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMS normalization over the last dim, plain or zero-centred, through the cuTile op `trtllm::cuda_tile_rms_norm`."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

EPS = 1e-5
HIDDEN = 4096
HEAD_DIM = 192
INDEXER_DIM = 256


def _cells() -> tuple[Cell, ...]:
    cells: list[Cell] = []
    # The residual-stream sites: [T, 4096], one row per token.
    for rows in (1, 37, 128, 8192):
        cells.append(
            Cell(
                why=f"fp32 residual {rows} x 4096, fp32 weight, layer-0 input norm",
                spec=dict(rows=rows, width=HIDDEN, dtype=torch.float32, family="input_ln0"),
            )
        )
    for rows in (1, 128, 8192):
        cells.append(
            Cell(
                why=f"bf16 residual {rows} x 4096, fp32 weight, final norm",
                spec=dict(rows=rows, width=HIDDEN, dtype=torch.bfloat16, family="final_norm"),
            )
        )
    # The per-head sites: [T * H, D], one row per (token, head).
    head_sites = (
        ("q_norm", HEAD_DIM, torch.float32, (("dp", 64), ("tp4", 16))),
        ("k_norm", HEAD_DIM, torch.float32, (("dp", 4), ("tp4", 1))),
        ("idx_q_norm", INDEXER_DIM, torch.bfloat16, (("dp", 16), ("tp4", 4))),
    )
    for family, width, weight_dtype, splits in head_sites:
        for split, heads in splits:
            for tokens in (1, 128, 8192):
                rows = tokens * heads
                cells.append(
                    Cell(
                        why=f"{family} {split} {tokens} tok x {heads} heads = {rows} x {width} bf16",
                        spec=dict(
                            rows=rows,
                            width=width,
                            dtype=torch.bfloat16,
                            weight_dtype=weight_dtype,
                            family=family,
                        ),
                    )
                )
    # A decode batch whose head rows do not fill the kernel's 16-row tile.
    cells.append(
        Cell(
            why="k_norm dp 37 tok x 4 heads = 148 rows, a ragged last tile",
            spec=dict(
                rows=148,
                width=HEAD_DIM,
                dtype=torch.bfloat16,
                weight_dtype=torch.float32,
                family="k_norm",
            ),
        )
    )
    return tuple(cells)


class CudaTileRmsNorm(OpWrapper):
    """`x / sqrt(mean(x^2, -1) + eps) * (1 + w)` (or `* w` with `use_gemma=False`)
    over the last dim of a 2-D `x`, one op call, new contiguous tensor in `x`'s dtype.

    `x` and `weight` are widened to fp32 on load; the sum of squares, the
    `1 + w` and the products run in fp32 and only the store rounds. Fusion
    boundary: the norm and its scale only -- no residual add
    (`cuda_tile_rms_norm_fuse_residual_`), no quantization. `x` is not mutated.

    The op is `trtllm::cuda_tile_rms_norm`, a cuTile (`cuda.tile`) kernel JIT
    compiled per constant set (dtype, width, tile size, `eps`, `use_gemma`).
    `static_persistent=True` launches `rms_norm_kernel_static_persistent`
    (one tile of whole rows per program, programs capped at the SM count) and
    ignores `gather`; `static_persistent=False` launches one program per row,
    `rms_norm_kernel_gather` or `rms_norm_kernel` by `gather`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, width, dtype (x and out), weight_dtype (default fp32),
    # family -- which checkpoint weight family the weight is drawn over (see
    # the test). Every cell is eps 1e-5, use_gemma=True and
    # static_persistent=True, gather=True: the zero-centred form at the call
    # TensorRT-LLM's own RMSNorm module makes.
    CELLS: tuple[Cell, ...] = _cells()

    note = """
    Registered only where `IS_CUDA_TILE_AVAILABLE` (the `cuda.tile` package
    and the `tileiras` compiler present, compute capability >= 10.0); the op
    raises on SM 107. There is no non-tile fallback behind the name: the
    certified cells launch `rms_norm_kernel_static_persistent_*` and nothing
    else (`test_one_cutile_kernel_per_call`).

    Metadata consumed: none. One output allocation per call; the
    non-persistent kernels also allocate a scratch fp32 [M] rstd.

    Certified, all with eps 1e-5, use_gemma=True, static_persistent=True,
    gather=True, contiguous 2-D `x`:
    - fp32 x, fp32 weight, width 4096 at 1, 37, 128 and 8192 rows (a
      zero-centred norm on an fp32 residual stream, fp32 out);
    - bf16 x, fp32 weight, width 4096 at 1, 128 and 8192 rows;
    - bf16 x, fp32 weight, width 192 at 1, 4, 16, 64, 128, 148, 512, 2048,
      8192, 32768, 131072 and 524288 rows (per-head q/k norms: T tokens x
      64/16 q heads or 4/1 kv heads; 524288 = 8192 x 64 is the envelope);
    - bf16 x, bf16 weight, width 256 at 4, 16, 512, 2048, 32768 and 131072
      rows (a per-head indexer q norm, 16/4 heads).
    Not certified: static_persistent=False (both kernels measured correct
    at every certified shape, but slower at large M: 524288 x 192 bf16
    takes ~270 us with gather and ~1850 us without, against ~104 us
    persistent); use_gemma=False (measured within one bf16 rounding at
    64 x 4096, not a cell); fp16 x; fp32 x with a bf16 weight; eps other
    than 1e-5; widths other than 192, 256 and 4096; row counts between and
    beyond the cells.

    Accuracy (measured on the cells, gated in the test): a bf16 result is
    one of the two bf16 neighbours of the fp64 value, the correctly rounded
    one on all but <= 3.6e-5 of elements; an fp32 result is within 4 fp32
    spacings of the once-rounded fp64 value (relative RMS <= 0.6 spacing).
    Two launches agree bitwise. The fp32 weight matters: the same q-norm
    weights rounded to bf16 before the call put 23% of elements off the
    correctly rounded result, up to 32 spacings.

    Layout: a non-contiguous `x` (a head slice of a wider projection) or
    `weight` is `.contiguous()`d inside the op -- correct, one extra copy
    kernel. The output is always contiguous with `x`'s shape.

    Loud failures (left to the op): a non-2-D `x` raises "too many values to
    unpack (expected 2)"; zero rows raise "Failed to launch cuTile kernel:
    invalid argument" (every mode), so a caller with an empty batch skips
    the call.

    Silent: `weight` of the wrong length. Every mode accepts a weight shorter
    or longer than the normalized width and returns a full-shape result
    from tile loads that pad or ignore the difference; the non-persistent
    tiled kernel (`gather=False`) answers wrongly at width 192 with a 191- or
    193-long weight. `is_valid` refuses it.

    Compilation: each new constant set compiles on its first call (each
    distinct `eps` is a new kernel). CUDA graphs: capture after a warm-up
    call and replay on refilled inputs reproduce the eager result bitwise
    (`test_cuda_graph_capture_and_replay`).

    Hard limits (computed, not probed: the envelope stays far below): 2^31
    elements is 11184810 rows at width 192, 8388608 at 256 and 524288 at
    4096; the envelope reaches 524288 x 192 (2^26.6 elements) and 8192 x
    4096. The persistent grid is min(SM count, row tiles); the
    non-persistent grid is M, under CUDA's 2^31 - 1 grid-x limit. The width
    is carried to the kernel as a float (N * 1.0), exact below 2^24.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        static_persistent: bool,
        gather: bool,
        use_gemma: bool,
    ) -> torch.Tensor:
        return torch.ops.trtllm.cuda_tile_rms_norm(
            x=x,
            weight=weight,
            eps=eps,
            static_persistent=static_persistent,
            gather=gather,
            use_gemma=use_gemma,
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        static_persistent: bool,
        gather: bool,
        use_gemma: bool,
    ) -> torch.Tensor:
        """fp64 norm, rounded once to `x.dtype`; `static_persistent` and `gather` are ignored.

        The scale `1 + w` is formed in fp32, as the model defines it (an fp32
        weight plus one, then the multiply); everything else is fp64.
        """
        scale = weight.float() + 1.0 if use_gemma else weight.float()
        xd = x.double()
        normed = xd * torch.rsqrt(xd.square().mean(dim=-1, keepdim=True) + eps)
        return (normed * scale.double()).to(x.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        static_persistent: bool,
        gather: bool,
        use_gemma: bool,
    ) -> None:
        # The op reads the weight through a tile as wide as the row and never
        # checks its length; every other mistake here it rejects loudly.
        assert weight.dim() == 1 and weight.shape[0] == x.shape[-1], (
            f"weight must be 1-D and as long as the normalized width {x.shape[-1]}; "
            f"got shape {tuple(weight.shape)}, which the op accepts without reading it whole"
        )


cuda_tile_rms_norm = CudaTileRmsNorm()
