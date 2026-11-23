# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LayerNorm over the last dim -- mean subtraction, biased variance, weight and
bias -- in one Triton kernel, through the FLA module callable
`tensorrt_llm._torch.modules.fla.layernorm_gated.layernorm_fn`.

Adapted from `models/qwen4_exp/ops/norm/layer_norm.py`, which wraps the same
callable; the width, eps, dtype and row counts here are this checkpoint's.

The Hy4-Flash vision tower normalizes with `nn.LayerNorm` at width 1152
(`vit.layers.*.input_layernorm` and `vit.layers.*.post_attention_layernorm`,
27 blocks x 2), eps 1e-5 (`vision_config.rms_norm_eps`, which
`HYV4VLVisionConfig.attribute_map` maps `layer_norm_eps` onto), with a bf16
weight and bias. Its merger norm `vit.perceive.norm` is an RMSNorm with a
weight only and belongs to the RMSNorm entry, not this one.

Tier-2 module callable, not a `torch.ops.trtllm.*` op: the registered norm
family is RMS-norm throughout, and `fused_adaptive_layernorm` checks
`D == 5120` (`cpp/tensorrt_llm/thop/fusedAdaptiveLayerNormOp.cpp`). One call
reshapes to 2-D, allocates the output and launches
`_layer_norm_fwd_1pass_kernel` once.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.fla.layernorm_gated import layernorm_fn as _layernorm_fn

#: The vision tower's width: `vision_config.hidden_size` 1152.
WIDTH = 1152


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

    # The vision tower's rows: `max_image_size` 2048 over `patch_size` 16 is a
    # 128 x 128 patch grid, the largest frame the processor produces; 512 px is
    # the smallest, a 32 x 32 grid. One row is the op's own smallest input.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="vision norm, one patch row x 1152 bf16",
            spec=dict(shape=(1, WIDTH), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="vision norm, a 512-pixel frame: 32 x 32 patches",
            spec=dict(shape=(1024, WIDTH), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="vision norm at the tower ceiling, 2048 px: 128 x 128 patches",
            spec=dict(shape=(16384, WIDTH), dtype=torch.bfloat16, eps=1e-5),
        ),
    )

    note = """
    Source: the in-tree Python callable `layernorm_fn`
    (`tensorrt_llm/_torch/modules/fla/layernorm_gated.py`: `LayerNormFn.apply`
    -> `_layer_norm_fwd` -> `_layer_norm_fwd_1pass_kernel`). A Triton kernel,
    JIT-compiled on first use for the device it runs on, not in the wheel; the
    cells below were driven on sm_103 with BLOCK_N 2048 and 8 warps for width
    1152. `M` and both row strides are passed as `i32`. One program per row.

    Origin: adapted from `models/qwen4_exp/ops/norm/layer_norm.py` (same
    callable, same kernel); the kernel file itself is adapted from
    flash-linear-attention and SGLang and keeps their attribution.

    Measured on NVIDIA GB300 (sm_103), torch 2.14.0a0, with this checkpoint's
    own affine ranges and the test's fixed seeds, against the fp64 reference:
    at 1 row the output is bitwise the reference; at 1024 rows the worst
    element is 0.381 ulp of its row's max (max |diff| 0.0625, 32 of 1179648
    elements differ bitwise, relative RMS 0.0022 ulp, cosine 0.99999994); at
    16384 rows 1.369 ulp of row max (max |diff| 0.25, 627 of 18874368 differ,
    relative RMS 0.0032 ulp, cosine 1.000000000). No element of any cell
    fails torch's default bf16 band. The test's distribution band (relative
    RMS <= 0.05 ulp of the reference's RMS, cosine >= 1 - 1e-5) sits 16x above
    the worst cell. The wrong normalizations it must refuse separate from that
    band at 1024 rows: biased -> unbiased variance (ddof=1) moves 7703
    elements past the element band and reaches 0.28 ulp relative RMS,
    dropping the mean 508538 (9.33 ulp worst, cosine 0.999822), dropping the
    bias 1173164 (166 ulp worst, cosine 0.636963). The whole test file runs in
    2.08 s; its first cell takes 1.21 s of that, the first-call compile, and
    the 1024- and 16384-row cells 0.20 s and under 5 ms. The Triton cache
    holds `_layer_norm_fwd_1pass_kernel.cubin` for this device afterwards.

    Metadata consumed: none. Inputs are not mutated; two scratch fp32 [M]
    tensors (mean, rstd) are allocated per call and discarded.

    Layout: `x.stride(-1) == 1` is read in place through its row stride; any
    other `x`, and non-contiguous `weight`/`bias`, are silently `.contiguous()`d
    first (one extra copy). A 3-D `x` is flattened to 2-D and restored.
    `weight`/`bias` dtype is free.

    Certified: contiguous bf16 `[rows, 1152]` `x` at 1, 1024 and 16384 rows
    with a bf16 weight and bias and eps 1e-5, as the vision tower calls it.
    Not certified: strided or 3-D `x`, `bias=None`, fp16/fp32 activations,
    weight/bias in another dtype than `x`, eps other than 1e-5, and rows above
    16384 (a frame set packed past one max-size frame).

    Loud failures (left to the callable): weight or bias length != N raises a
    bare AssertionError with no message; a feature dim above
    `65536 // x.element_size()` elements (32768 in bf16) raises "This layer
    norm doesn't support feature dim >= 64KB."; CPU tensors fail before or at
    the Triton launch.

    Hard limit: the row offsets `row * stride_x_row` and `row * stride_y_row`
    are `i32` products. At width 1152 the first row whose offset reaches 2^31
    is 1864136, so an `x` of >= 1864137 rows wraps. Not measured here; the
    envelope never reaches it: the tower cuts a request's patches into calls of
    at most 16384 rows.

    Accepted but not certified: an fp8-e4m3 `x` returns fp8-e4m3 (the output is
    `empty_like(x)`); fp64 `x`. Forward only: the autograd Function has no
    backward. CUDA-graph capture: measured at 1024 rows, a captured call
    replayed on refilled input is bitwise the eager call; the vision tower
    calls this eagerly in prefill, so capture is not a call-site requirement.
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
