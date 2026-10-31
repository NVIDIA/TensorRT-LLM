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

#: The vision tower's norm1 / norm2 / merger.norm: width 1152 with a bias, eps
#: 1e-6. The merger norm runs before the 4-to-1 merge, so it too sees 1152.
WIDTH = 1152
EPS = 1e-6


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
    # over the checkpoint's measured ranges.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"vision norm, {rows} patch rows x 1152 bf16",
            spec=dict(shape=(rows, WIDTH), dtype=torch.bfloat16),
        )
        for rows in (1, 1024)
    ) + (
        Cell(
            why="vision norm at the tower call ceiling, 32768 patch rows x 1152 bf16",
            spec=dict(shape=(32768, WIDTH), dtype=torch.bfloat16),
        ),
    )

    note = """
    Source: the in-tree Python callable `layernorm_fn`
    (`tensorrt_llm/_torch/modules/fla/layernorm_gated.py`: `LayerNormFn.apply`
    -> `_layer_norm_fwd` -> `_layer_norm_fwd_1pass_kernel`). A Triton kernel,
    JIT-compiled on first use, not in the wheel. Measured on GB300: the JIT path
    ran and compiled `GPUTarget(backend='cuda', arch=103)`, PTX `.target
    sm_103a`, at BLOCK_N 2048 and 8 warps for width 1152; `M` and both row
    strides are passed as `i32`. One program per row.

    Metadata consumed: none. Inputs are not mutated; two scratch fp32 [M]
    tensors (mean, rstd) are allocated per call and discarded.

    Layout: `x.stride(-1) == 1` is read in place through its row stride; any
    other `x`, and non-contiguous `weight`/`bias`, are silently `.contiguous()`d
    first (one extra copy). `weight`/`bias` dtype is free.

    Certified: contiguous bf16 `[rows, 1152]` `x` at 1, 1024 and 32768 rows
    with a bf16 weight and bias, as the vision tower calls it. Not certified:
    strided or 3-D `x`, `bias=None`, fp16/fp32 activations, weight/bias in
    another dtype than `x`.

    Loud failures (left to the callable): weight or bias length != N raises a
    bare AssertionError with no message; `next_pow2(N) * x.element_size() >
    64KB` raises "This layer norm doesn't support feature dim >= 64KB" (N =
    65536 in bf16); CPU tensors fail before or at the Triton launch.

    Hard limit: the row offsets `row * stride_x_row` and `row * stride_y_row`
    are `i32` products. At width 1152 the first row whose offset reaches 2^31
    is 1864136, so an `x` of >= 1864137 rows wraps. In a fresh process this
    was measured as a CUDA illegal memory access. With other allocations
    mapped below `x`, the wrapped read could land in them and be silent; that
    was not measured and is not guarded. The envelope never reaches it:
    `VisionTower` cuts a request's frames into calls of at most 32768 rows.

    Accuracy at the cells against an fp64 reference, measured on GB300 beyond
    the default gate:
      rows    differs from the correctly rounded fp64   worst |err| / row max
      1       0 / 1152                                  0.907 bf16 ulp
      1024    37 / 1179648 (3.1e-5)                     0.949 bf16 ulp
      32768   1096 / 37748736 (2.9e-5)                  0.995 bf16 ulp
    Cosine 0.9999986 and relative L2 1.65e-3 at every cell (the bf16 output
    rounding itself). The elements that differ are outputs where `x_hat * w`
    and `b` cancel toward zero: up to 205 ulp of their own tiny magnitude,
    and each within torch's bf16 tolerance. A constant row returns `bias`
    bitwise. `eps` sits inside the sqrt.

    What the default bf16 gate tells apart, as negative controls against the
    same output: an unbiased (N - 1) variance, a missing mean subtraction and
    a missing bias each fail it at every cell (8.6e-2, 0.79 and 0.999 of the
    elements then differ from the correctly rounded value). An eps of 1e-5,
    or eps outside the sqrt, moves 1.7e-3 and 2.5e-4 of the elements and fails
    the gate only at some cells, so the gate does not certify the eps
    placement. `test_the_gate_discriminates_a_wrong_normalization` holds the
    first three. At residual-stream-like inputs (scale 8, offset 3; not a
    cell) the result differs on 3.4e-5 of the elements, worst 0.984 bf16 ulp
    of the row max.

    Accepted but not certified: an fp8-e4m3 `x` returns fp8-e4m3 (output is
    `empty_like(x)`), within one e4m3 ulp of the exact value; fp64 `x`.
    Forward only: the autograd Function has no backward. Measured, not
    certified: one kernel launch, runs unchanged under `inference_mode`,
    CUDA-graph capturable after a warm-up call compiles the Triton kernel (a
    1024-row capture replayed on refilled input is bitwise the eager call).
    The call site is eager-only. 32768 rows take about 61 us per call.
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
