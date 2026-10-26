# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LayerNorm over the last dim -- mean subtraction, biased variance, weight and
bias -- in one Triton kernel, through the FLA module callable
`tensorrt_llm._torch.modules.fla.layernorm_gated.layernorm_fn`.

Tier-2 module callable, not a `torch.ops.trtllm.*` op: the registered norm
family is RMS-norm throughout, and `fused_adaptive_layernorm` refuses every
width but 5120. One call reshapes to 2-D, allocates the output and launches
`_layer_norm_fwd_1pass_kernel` once.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modules.fla.layernorm_gated import layernorm_fn as _layernorm_fn

from .._op import Arch, Cell, OpWrapper

#: A vision tower's norm1 / norm2 / merger.norm: width 1152 with a bias, eps
#: 1e-6. The merger norm runs before the 4-to-1 merge, so it too sees 1152.
WIDTH = 1152
EPS = 1e-6

#: The sparse-attention indexer's key norm: one 256-wide key head per token,
#: weight and bias stored fp32 and rounded to bf16 for the call, eps 1e-5.
INDEXER_K_WIDTH = 256
#: A perception-encoder vision tower's ln_pre / ln_1 / ln_2: width 1536, bf16
#: weight and bias, eps 1e-5.
VISION_1536_WIDTH = 1536
EPS_1E5 = 1e-5


class LayerNorm(OpWrapper):
    """`(x - mean) / sqrt(var + eps) * weight + bias` per row of the last dim,
    variance over `N` (not `N - 1`), new contiguous tensor in `x`'s dtype.

    Every operand is widened to fp32 on load, the reduction and the affine run
    in fp32, and only the store rounds. `bias=None` drops the additive term (a
    compile-time branch of the kernel, not a zero bias).

    Fusion boundary: the norm and its affine only -- no residual, no gate, no
    output quantization. The callable's `z`, `group_size`, `norm_before_gate`
    and `is_rms_norm` are fixed at "no gate, one group, LayerNorm"; the gated
    and RMS forms are different computations, not options of this entry.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: shape, dtype (weight and bias share it). Weight and bias are drawn
    # over the checkpoint's measured ranges. Optional keys: `eps` (default
    # EPS), `affine` naming the checkpoint tensor family whose ranges the draw
    # uses (default the 1152-wide vision norms), and `row_offset`, the half
    # width of a uniform per-row offset added to `x` so every row has a mean
    # the kernel must subtract (default 0: zero-mean rows).
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"vision norm, {rows} patch rows x 1152 bf16",
                spec=dict(shape=(rows, WIDTH), dtype=torch.bfloat16),
            )
            for rows in (1, 1024)
        )
        + (
            Cell(
                why="vision norm at the tower call ceiling, 32768 patch rows x 1152 bf16",
                spec=dict(shape=(32768, WIDTH), dtype=torch.bfloat16),
            ),
        )
        + tuple(
            Cell(
                why=f"indexer k, {label} x 256 bf16, eps 1e-5",
                spec=dict(
                    shape=(rows, INDEXER_K_WIDTH),
                    dtype=torch.bfloat16,
                    eps=EPS_1E5,
                    affine="indexer_k_norm",
                    row_offset=4.0,
                ),
            )
            for rows, label in (
                (1, "1 decode row"),
                (128, "128 decode rows"),
                (8192, "8192-row prefill chunk (rank ceiling)"),
            )
        )
        + tuple(
            Cell(
                why=f"vision ln 1536, {label}, bf16, eps 1e-5",
                spec=dict(
                    shape=(rows, VISION_1536_WIDTH),
                    dtype=torch.bfloat16,
                    eps=EPS_1E5,
                    affine="vision_ln_1536",
                    row_offset=4.0,
                ),
            )
            for rows, label in (
                (2704, "2704 rows: one overview tile"),
                (13072, "13072 rows: 8 crops + overview (ceiling)"),
            )
        )
    )

    note = """
    Metadata consumed: none. Inputs are not mutated; two scratch fp32 [M]
    tensors (mean, rstd) are allocated per call and discarded.

    Layout: `x.stride(-1) == 1` is read in place through its row stride; any
    other `x`, and non-contiguous `weight`/`bias`, are silently `.contiguous()`d
    first (one extra copy). `weight`/`bias` dtype is free.

    Certified, contiguous 2-D bf16 `x` with a bf16 weight and bias:
    - width 1152, eps 1e-6, at 1, 1024 and 32768 rows (a 1152-wide vision
      tower's norms);
    - width 256, eps 1e-5, at 1, 128 and 8192 rows (the sparse indexer's key
      norm; its fp32 checkpoint weight and bias rounded to bf16 by the caller);
    - width 1536, eps 1e-5, at 2704 and 13072 rows (a 1536-wide vision tower's
      ln_pre / ln_1 / ln_2).
    Not certified: strided or 3-D `x`, `bias=None`, fp16/fp32 activations,
    weight/bias in another dtype than `x` (an fp32 weight is accepted and
    widened on load, but no cell drives it), widths other than 256/1152/1536,
    rows beyond 8192 at width 256 or beyond 13072 at width 1536.

    Variance formula: the kernel is two-pass in registers, var = E[(x -
    mean)^2] in fp32. A one-pass var = E[x^2] - mean^2 in fp32 (the form a
    fused norm+rope indexer kernel may use) is algebraically the same and
    differs only in fp32 rounding. At the width-256 cells (per-row means up
    to 4 sigma) the two bf16 results are bitwise equal on >= 99.99% of
    elements and differ by <= 0.81 * 2^-8 of the row maximum elsewhere; the
    kernel matches the fp64 reference on >= 99.997% of elements, the
    one-pass form on fewer (gated by
    `test_one_pass_variance_formula_agrees_within_one_ulp`). They diverge
    further only under cancellation (|mean| >> std), where the one-pass form
    is the inaccurate side.

    Loud failures (left to the callable): weight or bias length != N raises a
    bare AssertionError with no message; `next_pow2(N) * x.element_size() >
    64KB` raises "This layer norm doesn't support feature dim >= 64KB" (N =
    65536 in bf16); CPU tensors fail before or at the Triton launch.

    Hard limits (computed, not probed: the envelope stays far below): the
    kernel addresses `row * stride_x_row` from an int32 program id, so x's
    element offset crosses 2^31 at 2^31 / N rows -- 8388608 rows at width
    256, 1864135 at 1152, 1398101 at 1536 (the envelope's ceilings are 8192,
    32768 and 13072). The grid's first dimension is M, under CUDA's 2^31 - 1
    grid-x limit at every such M. No fp32-carried index.

    Accuracy, measured beyond the cells' default gate and not certified: at
    a 1152-wide tower's affine (weights to 17.5, biases to -25.6) the bf16
    result is the correctly rounded fp64 value on >= 99.9989% of elements;
    where it differs, by <= 2^-7 of the summed term magnitude. A constant row
    returns `bias` bitwise. `eps` sits inside the sqrt.

    CUDA graphs: capture and replay of a width-256 call after a warm-up call
    (which compiles the Triton kernel) reproduce the eager result bitwise on
    refilled inputs (`test_cuda_graph_capture_and_replay`), as the decode
    graph replays the indexer key norm.

    Accepted but not certified: an fp8-e4m3 `x` returns fp8-e4m3 (output is
    `empty_like(x)`), within one e4m3 ulp of the exact value; fp64 `x`.
    Forward only: the autograd Function has no backward. Measured, not
    certified: one kernel launch, runs unchanged under `inference_mode`.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        return _layernorm_fn(x, weight, bias, z=None, eps=eps)

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        """fp64 from primitive ops (not `F.layer_norm`), rounded once to `x.dtype`."""
        xd = x.double()
        mean = xd.mean(dim=-1, keepdim=True)
        var = (xd - mean).square().mean(dim=-1, keepdim=True)
        out = (xd - mean) / torch.sqrt(var + eps) * weight.double()
        if bias is not None:
            out = out + bias.double()
        return out.to(x.dtype)


layer_norm = LayerNorm()
