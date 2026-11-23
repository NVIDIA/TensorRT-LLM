# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision patch embedding: a stride-equals-kernel 2-D convolution made as the GEMM it is.

With one whole kernel window per row and the stride equal to the kernel, the
convolution has one output position per row: `out = rows @ filters.T + bias`,
where `rows` and `filters` are views of `pixel_values` and `weight`. `__call__`
is one `torch.ops.trtllm.cublas_mm` call (the C++ thop in
`thop/cublasScaledMM.cpp`, cuBLASLt, bias fused into the epilogue) between views
that launch nothing.

Adapted from `models/qwen4_exp/ops/vision/patch_embed_conv3d.py`, which is the
same contraction for a 3-D kernel; this checkpoint's vision tower has 4-D
weights only (`vit.embeddings.patch_embedding`, `vit.perceive.proj.0`).
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul


class PatchEmbedConv(OpWrapper):
    """`out[n, k] = bias[k] + sum_{c,i,j} window[n, c, i, j] * weight[k, c, i, j]`.

    One whole `[in_channels, kh, kw]` kernel window per row of `pixel_values`,
    flat `[N, in_channels*kh*kw]` (channel-slowest, the order the image
    processor emits a patch in) or explicit `[N, in_channels, kh, kw]`; the
    kernel extent is read from `weight`. Cross-correlation (no kernel flip),
    like `nn.Conv2d` with the stride equal to the kernel and no padding.
    Returns a fresh contiguous `[N, out_channels]` in the input dtype.

    Both of this checkpoint's vision convolutions are this contraction: the
    patch embedding (`vit.embeddings.patch_embedding`, `[1152, 3, 16, 16]`, one
    16x16 patch per row, stride 16) and the merger's 2x2 projection
    (`vit.perceive.proj.0`, `[4608, 1152, 2, 2]`, one 2x2 block per row, stride
    2). The row order is the caller's: the merger's rows must be its 2x2 blocks
    flattened `(channel, kh, kw)` -- a gather from the patch grid, not a
    concatenation of patch rows.

    Fusion boundary: the patch projection and its bias. Patchification, the
    position embedding added after it, the merger's norm/GELU and its 1x1
    `proj.2`, and the row order are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: num_rows (GEMM rows = kernel windows), out_channels, in_channels,
    # kernel, dtype. The checkpoint's two geometries at the images its
    # processor is configured for: min_pixels 512x512 and 1024x1024, plus the
    # envelope -- max_image_size 2048x2048 = max_vit_seq_len 16384 patches =
    # img_max_token_num 4096 merged tokens. The vision tower is replicated, so
    # these are every member's shapes (rank-invariant).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="patch embed, 512x512 image: 1024 patches bf16",
            spec=dict(
                num_rows=1024,
                out_channels=1152,
                in_channels=3,
                kernel=16,
                dtype=torch.bfloat16,
            ),
        ),
        Cell(
            why="patch embed, 1024x1024 image: 4096 patches bf16",
            spec=dict(
                num_rows=4096,
                out_channels=1152,
                in_channels=3,
                kernel=16,
                dtype=torch.bfloat16,
            ),
        ),
        Cell(
            why="patch embed envelope: 16384 patches (max_vit_seq_len)",
            spec=dict(
                num_rows=16384,
                out_channels=1152,
                in_channels=3,
                kernel=16,
                dtype=torch.bfloat16,
            ),
        ),
        Cell(
            why="merger 2x2, 512x512 image: 256 blocks bf16",
            spec=dict(
                num_rows=256,
                out_channels=4608,
                in_channels=1152,
                kernel=2,
                dtype=torch.bfloat16,
            ),
        ),
        Cell(
            why="merger 2x2, 1024x1024 image: 1024 blocks bf16",
            spec=dict(
                num_rows=1024,
                out_channels=4608,
                in_channels=1152,
                kernel=2,
                dtype=torch.bfloat16,
            ),
        ),
        Cell(
            why="merger envelope: 4096 blocks from 16384 patches",
            spec=dict(
                num_rows=4096,
                out_channels=4608,
                in_channels=1152,
                kernel=2,
                dtype=torch.bfloat16,
            ),
        ),
    )

    note = """
    Source: `torch.ops.trtllm.cublas_mm`, compiled into the engine's torch-op
    library (`cpp/tensorrt_llm/thop/cublasScaledMM.cpp`,
    `TORCH_LIBRARY_FRAGMENT(trtllm)`, cuBLASLt); no JIT path. One kernel launch
    per call, with the bias in cuBLASLt's epilogue.

    Metadata consumed: none. `pixel_values` is not mutated. Several images'
    patches are just more rows; there is no batch axis.

    Measured on sm_103 against `reference`, bitwise-equal share / relative L2 /
    worst element as a fraction of torch's bf16 band (rtol 1.6e-2, atol 1e-5):
    patch embed 1024 rows 99.961% / 5.38e-05 / 0.487, 4096 rows 99.960% /
    5.79e-05 / 0.483, 16384 rows 99.962% / 5.79e-05 / 0.486; merger 256 rows
    99.753% / 1.56e-04 / 0.487, 1024 rows 99.754% / 1.58e-04 / 0.727, 4096 rows
    99.754% / 1.57e-04 / 0.688. Every cell is inside the default band, so the
    test compares with `torch.testing.assert_close`'s default and no wider one.

    `torch.nn.functional.conv2d` (cuDNN) on the same windows adds the bias
    separately in bf16, rounding a second time; it is used in the test only as
    an fp32 flatten-order oracle, never as the band.

    Row-count invariance is not claimed: cuBLASLt picks the kernel by row
    count. Check rows against `reference`, never against a call of a different
    size. CUDA-graph capturable; the output is a fresh allocation.

    Silent, refused by `is_valid` (each measured unguarded): a row that is not
    exactly one kernel window; row-strided pixel rows (2-D, or 4-D with a batch
    stride) and a weight sliced along its output channels, read as if dense; a
    bias in another dtype or strided, or shorter than `out_channels`, read as
    dense output-dtype elements; fp32 input with a bias, which drops the bias.
    Loud, not guarded: a channels-last or transposed operand fails the view,
    and a weight dtype unlike the input's fails in cuBLASLt.

    Not certified: dtypes other than bf16; other geometries (including the
    merger's 1x1 `proj.2`, a plain GEMM this target's cublas_mm entry serves);
    `bias=None`; row counts below 256, between the cells, and above 16384 (a
    request concatenating several max-size images could exceed it); the
    `cublas_mm` `out_dtype`/`output_buffer_kind`/`group` arguments, left at
    their defaults.
    """

    def __call__(
        self,
        pixel_values: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out_channels, in_channels, kh, kw = weight.shape
        rows = pixel_values.view(-1, in_channels * kh * kw)
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
        # The view reinterprets anything whose element count divides: two
        # patches in one row become twice the rows -- a result of the right
        # shape from the wrong pixels -- without an error.
        _, in_channels, kh, kw = weight.shape
        patch_numel = in_channels * kh * kw
        if pixel_values.dim() == 2:
            one_patch_per_row = pixel_values.shape[1] == patch_numel
        elif pixel_values.dim() == 4:
            one_patch_per_row = tuple(pixel_values.shape[1:]) == (in_channels, kh, kw)
        else:
            one_patch_per_row = False
        assert one_patch_per_row, (
            f"each row must be exactly one kernel window: pixel_values must be "
            f"[num_rows, {patch_numel}] or "
            f"[num_rows, {in_channels}, {kh}, {kw}] for this weight; got "
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


patch_embed_conv = PatchEmbedConv()
