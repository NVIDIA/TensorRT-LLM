# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill's conv output to q / k / v: L2-normalize q and k per head and go token-major, in one launch."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import fused_kda_post_conv


class KdaPostConv(OpWrapper):
    """Split a channel-major `[3 * heads * head_dim, tokens]` conv output into token-major q, k, v.

    Returns three new `[1, tokens, heads, head_dim]` tensors in `packed.dtype`; q and k are
    L2-normalized along the head dim, v is copied as is.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{heads} heads at TP{96 // heads}, {tokens} tokens across 16-token blocks",
                spec=dict(heads=heads, tokens=tokens, dtype=torch.bfloat16, kind="random"),
            )
            for heads in (6, 24)
            for tokens in (1, 15, 16, 300, 4096)
        ),
        *(
            Cell(
                why=f"four heads with {dtype}, outside the bf16 Kimi K3 surface",
                spec=dict(heads=4, tokens=77, dtype=dtype, kind="random"),
            )
            for dtype in (torch.float16, torch.float32)
        ),
        Cell(
            why="zero and tiny heads exercise epsilon inside the square root",
            spec=dict(heads=6, tokens=4, dtype=torch.bfloat16, kind="tiny"),
        ),
        Cell(
            why="zero tokens return empty tensors without a launch",
            spec=dict(heads=6, tokens=0, dtype=torch.bfloat16, kind="random"),
        ),
    )
    note = """
    No metadata or persistent state. Triton compiles on the first call per shape
    class; tokens is not a specialization key. The input is not mutated. Only
    normalization and relayout are fused; convolution and the delta rule are
    separate entries. Kimi K3 uses head_dim 128 and the default l2_norm_eps.
    """

    def __call__(
        self,
        packed: torch.Tensor,
        num_heads: int,
        head_dim: int,
        l2_norm_eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return fused_kda_post_conv(packed, num_heads, head_dim, l2_norm_eps)

    def reference(
        self,
        packed: torch.Tensor,
        num_heads: int,
        head_dim: int,
        l2_norm_eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = packed.shape[1]
        q, k, v = packed.float().view(3, num_heads, head_dim, tokens).permute(0, 3, 1, 2)

        def norm(t):
            return t / torch.sqrt(t.pow(2).sum(-1, keepdim=True) + l2_norm_eps)

        return (
            norm(q).to(packed.dtype).unsqueeze(0).contiguous(),
            norm(k).to(packed.dtype).unsqueeze(0).contiguous(),
            v.to(packed.dtype).unsqueeze(0).contiguous(),
        )


kda_post_conv = KdaPostConv()
