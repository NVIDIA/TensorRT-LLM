# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision patch embedding: one 3-D convolution per pixel patch, made as the GEMM it is.

With one whole patch per row and the stride equal to the kernel, the convolution
has one output position per row: `out = rows @ filters.T + bias`, where `rows` and
`filters` are views of `pixel_values` and `weight`. `__call__` is one
`torch.ops.trtllm.cublas_mm` call (the C++ thop in `thop/cublasScaledMM.cpp`,
cuBLASLt, bias fused into the epilogue) between views that launch nothing.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
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
    # rows and a bias: what the vision tower passes. The tower is replicated,
    # so these are every member's shapes (rank-invariant).
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
    Source: `torch.ops.trtllm.cublas_mm`, compiled into the engine's torch-op
    library (no JIT path). One kernel launch per call; cuBLASLt picks the
    kernel by row count (`nvjet_sm103` 64x8 tiles at 64 rows, 128x256 at 4096
    and 32768, each with the bias in its epilogue).

    Metadata consumed: none. `pixel_values` is not mutated. Several images,
    and a video's frame groups, are just more rows; there is no batch axis.

    Measured on GB300 against `reference` at the three cells: 99.91-99.92% of
    outputs bitwise equal, the rest one-ulp rounding flips of the element
    (at most 4.4 bf16 ulps at the output RMS); relative L2 <= 9.4e-5; the worst
    element at 0.49 of torch's default bf16 band. Each cell also holds
    `reference`, in fp32, to an fp32 `F.conv3d` of the 5-D patches (TF32 off):
    max |d| 2.0e-6 at 64 patches, 0 at 4096 and 32768 -- the flatten order
    `__call__` and `reference` share is `nn.Conv3d`'s.

    The same patches through `torch.nn.functional.conv3d` (cuDNN) are seven
    GPU operations: three memsets, NCHW-to-NHWC transposes of both operands, an
    implicit-GEMM fprop and a separate bf16 bias add that rounds a second time
    (relative L2 2.85e-3 against `reference`). Eager time per call, this op vs
    that one: 64 vs 117 us at 64 patches, 65 vs 489 us at 4096, 121 vs 2963 us
    at 32768.

    Row-count invariance is not claimed: the first 64 rows of a 4096-row call
    measured bitwise equal to a 64-row call, but the kernel changes with the
    row count. Check rows against `reference`, never against a call of a
    different size. CUDA-graph capturable (replayed at 64 and 32768 patches);
    the output is a fresh allocation.

    Silent, refused by `is_valid` (each measured unguarded): a row that is not
    exactly one patch; row-strided pixel rows (2-D, or 5-D with a batch
    stride) and a weight sliced along its output channels, read as if dense
    (10^5 times outside the band); a bias in another dtype (fp32 or fp16 with
    bf16 input) or strided (5000 times outside); a bias shorter than
    `out_channels`, read past its end without an error; fp32 input with a
    bias, which drops the bias. Loud, not guarded: a channels-last or
    transposed operand fails the view, and a weight dtype unlike the input's
    fails in cuBLASLt (CUBLAS_STATUS_NOT_SUPPORTED).

    Not certified: dtypes other than bf16 (fp16 with an fp16 bias measured
    inside the band, not a cell); other geometries; `bias=None`; patch counts
    below 64, between the cells, and past 32768.
    """

    def __call__(
        self,
        pixel_values: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out_channels, in_channels, kt, kh, kw = weight.shape
        rows = pixel_values.view(-1, in_channels * kt * kh * kw)
        filters = weight.view(out_channels, -1)
        return torch.ops.trtllm.cublas_mm(rows, filters.t(), bias, None, 0, None)

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
        # of the right shape from the wrong pixels -- and [N, 3072] rows become
        # 2N, without an error.
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
        # cuBLASLt reads the rows as dense row-major and the filter as dense
        # column-major: a row-strided view, or a weight sliced along its output
        # channels, still views to 2-D and is read as if dense.
        assert pixel_values.is_contiguous(), (
            "pixel_values must be contiguous; a strided view is read as dense rows"
        )
        assert weight.is_contiguous(), "weight must be contiguous; a strided view is read as dense"
        if bias is not None:
            # The epilogue reads the bias as `out_channels` dense elements of
            # the output dtype, and drops it altogether for fp32 input.
            assert bias.dtype == pixel_values.dtype, (
                f"bias dtype {bias.dtype} must equal the input dtype {pixel_values.dtype}"
            )
            assert bias.shape == (weight.shape[0],) and bias.is_contiguous(), (
                f"bias must be a contiguous [{weight.shape[0]}] tensor; got "
                f"{tuple(bias.shape)} with stride {bias.stride()}"
            )
            assert pixel_values.dtype != torch.float32, (
                "the bias is silently dropped for fp32 input"
            )


patch_embed_conv3d = PatchEmbedConv3d()
