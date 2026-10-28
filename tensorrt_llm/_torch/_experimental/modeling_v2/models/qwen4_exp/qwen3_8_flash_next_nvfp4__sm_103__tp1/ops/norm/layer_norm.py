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

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.fla.layernorm_gated import layernorm_fn as _layernorm_fn

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

    Accuracy, measured beyond the cells' default gate and not certified: at
    the checkpoint's affine (weights to 17.5, biases to -25.6) the bf16 result
    is the correctly rounded fp64 value on >= 99.9989% of elements; where it
    differs, by <= 2^-7 of the summed term magnitude. A constant row returns
    `bias` bitwise. `eps` sits inside the sqrt.

    Accepted but not certified: an fp8-e4m3 `x` returns fp8-e4m3 (output is
    `empty_like(x)`), within one e4m3 ulp of the exact value; fp64 `x`.
    Forward only: the autograd Function has no backward. Measured, not
    certified: one kernel launch, runs unchanged under `inference_mode`,
    CUDA-graph capturable after a warm-up call compiles the Triton kernel.
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
