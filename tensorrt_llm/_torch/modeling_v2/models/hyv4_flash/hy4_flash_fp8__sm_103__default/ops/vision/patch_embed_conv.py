# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision patch embedding: a stride-equals-kernel 2-D convolution as one `cublas_mm` GEMM with fused bias."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul


class PatchEmbedConv(OpWrapper):
    """`out[n, k] = bias[k] + sum_{c,i,j} window[n, c, i, j] * weight[k, c, i, j]`, a new `[N, out_channels]`.

    Each row of `pixel_values` is one whole kernel window, flat `[N, C*kh*kw]` (channel-slowest) or
    `[N, C, kh, kw]`; serves the patch embedding (16x16) and the merger's 2x2 projection.
    """

    ARCHS = frozenset({Arch.SM_103})

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
        # Each violation below gives a right-shaped answer from the wrong values, without an error.
        _, in_channels, kh, kw = weight.shape
        patch_numel = in_channels * kh * kw
        if pixel_values.dim() == 2:
            one_patch_per_row = pixel_values.shape[1] == patch_numel
        elif pixel_values.dim() == 4:
            one_patch_per_row = tuple(pixel_values.shape[1:]) == (in_channels, kh, kw)
        else:
            one_patch_per_row = False
        assert one_patch_per_row, (
            f"each row must be exactly one kernel window [{in_channels}, {kh}, {kw}]; got {tuple(pixel_values.shape)}"
        )
        assert pixel_values.is_contiguous(), "pixel_values must be contiguous"
        assert weight.is_contiguous(), "weight must be contiguous"
        if bias is not None:
            assert bias.dtype == pixel_values.dtype, (
                f"bias dtype {bias.dtype} must equal the input dtype {pixel_values.dtype}"
            )
            assert bias.shape == (weight.shape[0],) and bias.is_contiguous(), (
                f"bias must be a contiguous [{weight.shape[0]}] tensor; got {tuple(bias.shape)}"
            )
            assert pixel_values.dtype != torch.float32, (
                "the bias is silently dropped for fp32 input"
            )


patch_embed_conv = PatchEmbedConv()
