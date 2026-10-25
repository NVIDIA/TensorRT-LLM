# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision patch embedding: one 3-D convolution per pixel patch.

Tier-3 composed: one `torch.nn.functional.conv3d` call between two views that
launch nothing. No registered op is a 3-D convolution, and the CuTe DSL
implicit-GEMM conv3d kernels in `visual_gen/kernels/blackwell/conv/` refuse this
checkpoint's three input channels (bf16 needs `in_channels % 8 == 0`;
`can_implement` swallows the refusal and returns False).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: This checkpoint: in_channels 3, kernel (temporal_patch_size 2, patch_size
#: 16, 16), out_channels 1152 -- weight [1152, 3, 2, 16, 16], a 1536-pixel row.
IN_CHANNELS = 3
KERNEL = (2, 16, 16)
EMBED_DIM = 1152


class PatchEmbedConv3d(OpWrapper):
    """`out[n, k] = bias[k] + sum_{c,t,i,j} pixel[n, c, t, i, j] * weight[k, c, t, i, j]`.

    One whole `[in_channels, kt, kh, kw]` patch per row of `pixel_values`, flat
    `[N, C*kt*kh*kw]` (what the image processor emits, channel-slowest) or
    explicit `[N, C, kt, kh, kw]`; the kernel extent is read from `weight`.
    Cross-correlation (no kernel flip), like `nn.Conv3d`. Returns a fresh
    contiguous `[N, out_channels]` in the input dtype.

    Fusion boundary: the patch projection and its bias. Patchification, the
    position embedding added after it, and the row order (rows pass through in
    the order they arrive -- the merge-block order the merger needs) are the
    caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: num_patches, dtype. The checkpoint's geometry, flat contiguous
    # rows and a bias: what the vision tower passes.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"checkpoint geometry, {n} patches bf16",
            spec=dict(num_patches=n, dtype=torch.bfloat16),
        )
        for n in (64, 4096)
    ) + (
        Cell(
            why="tower call ceiling: 8192 tokens x merge 2x2 = 32768 patches bf16",
            spec=dict(num_patches=32768, dtype=torch.bfloat16),
        ),
    )

    note = """
    Metadata consumed: none. `pixel_values` is not mutated. All three operands
    share one dtype and device -- torch raises "... should be the same"
    otherwise. Several images, and a video's frame groups, are just more rows;
    there is no batch axis.

    fp32 runs in TF32 unless the caller sets `torch.backends.cudnn.allow_tf32 =
    False` (global state the caller owns): measured 5.84e-4 from an exact fp32
    matmul with it on (99.99% of outputs differ), 3.58e-6 with it off. bf16 and
    fp16 are unaffected.

    At this shape the op is a GEMM: one output position per row, and the
    `[1152, 1536]` reshape of the filter is a view of the same bytes. Not timed.
    CUDA-graph capturable; the output is a fresh allocation.

    Not batch-invariant: cuDNN picks its kernel by row count, so a row's last bits
    depend on how many rows share the call. Check rows against `reference`, never
    against a call of a different size.

    Not certified: dtypes other than bf16 (the fp32 TF32 behavior above is
    measured, not certified); other geometries, `bias=None` and non-contiguous
    pixel rows; patch counts below 64, between the cells, and past 32768
    (cuDNN picks its kernel by row count).
    """

    def __call__(
        self,
        pixel_values: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out_channels, in_channels, kt, kh, kw = weight.shape
        patches = pixel_values.view(-1, in_channels, kt, kh, kw)
        return F.conv3d(patches, weight, bias).view(-1, out_channels)

    def reference(
        self,
        pixel_values: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """True-fp32 matmul of the flattened rows against the flattened filter."""
        rows = pixel_values.reshape(pixel_values.shape[0], -1).float()
        filters = weight.reshape(weight.shape[0], -1).float()
        with true_fp32_matmul():
            out = rows @ filters.t()
        if bias is not None:
            out = out + bias.float()
        return out.to(pixel_values.dtype)

    def is_valid(
        self,
        pixel_values: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        # The view reinterprets anything whose element count divides: a whole
        # [1, 3, 2, 64, 64] volume becomes 16 "patches" -- a [16, 1152] result
        # of the right shape from the wrong pixels, 2.52 from the strided conv
        # the caller meant -- and [N, 3072] rows become 2N, without an error.
        _, in_channels, kt, kh, kw = weight.shape
        patch_numel = in_channels * kt * kh * kw
        if pixel_values.dim() == 2:
            one_patch_per_row = pixel_values.shape[1] == patch_numel
        elif pixel_values.dim() == 5:
            one_patch_per_row = tuple(pixel_values.shape[1:]) == (in_channels, kt, kh, kw)
        else:
            one_patch_per_row = False
        assert one_patch_per_row, (
            f"each row must be exactly one patch: pixel_values must be "
            f"[num_patches, {patch_numel}] or "
            f"[num_patches, {in_channels}, {kt}, {kh}, {kw}] for this weight; got "
            f"{tuple(pixel_values.shape)}"
        )


patch_embed_conv3d = PatchEmbedConv3d()
