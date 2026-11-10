# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5 MoE router logits (256 experts at hidden 6144, FP32 from BF16) for up to 32 tokens, via
the trtllm glm_router_gemm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

HIDDEN_SIZE = 6144
NUM_EXPERTS = 256
MAX_TOKENS = 32


class GlmRouterGemm(OpWrapper):
    """`hidden_states @ weight.T` in FP32 from BF16 operands: the router logits `[tokens, 256]`.

    `weight` is the gate weight as stored, `[256, 6144]` BF16 row-major. Through16 tokens there
    is one CTA per expert;17-32 use eight-CTA clusters. Before
    its grid-dependency wait each CTA prefetches its weight row into L2; right after the wait,
    before any logit is written, it lets its dependents launch.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The c1 decode step routes the MTP layer's BF16 MoE here once per draft step (7 calls); the
    # 75 MoE layers of the target compute their router logits inside the routing kernel.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP draft steps: one row (draft step 0 after its row select, steps 1-6)",
            spec=dict(tokens=1),
        ),
        Cell(
            why="8 rows: a verify-width step routed without the row select",
            spec=dict(tokens=8),
        ),
        Cell(why="16 tokens: the largest GEMV instance", spec=dict(tokens=16)),
        *(
            Cell(why=f"TP4 tensorcore router: {tokens} rows", spec=dict(tokens=tokens))
            for tokens in (17, 18, 24, 32)
        ),
    )

    note = """
    PDL (`TRTLLM_ENABLE_PDL`): the kernel reads only `weight` before its grid-dependency wait (an
    L2 prefetch), so no kernel launched earlier in the step may write it. Its dependents launch
    as soon as it passes the wait, before the logits exist: each must wait on its own grid
    dependency before it reads the logits or writes `hidden_states`. At c1 the dependent is the
    low-M BF16 MoE's FC1 kernel, which waits before it routes; its FC2 kernel reads the logits
    before its own wait, which holds because FC1 releases it only after FC1's wait.

    Preconditions, enforced loudly by the op: CUDA BF16 2-D `hidden_states` and `weight` on one
    device, contiguous and 16-byte aligned; hidden 6144; 256 experts; 1 to 32 tokens.

    Numerics: per thread an FP32 FMA chain over its 48 products, then a warp butterfly and the four
    warp sums in order; deterministic, and bitwise the logits of the shared
    `dsv3_router_gemm_op` through 16 tokens. Above 16, an eight-CTA cluster uses
    BF16 tensor cores and a fixed warp-then-cluster reduction order; FP32 logits
    approximate the same product, but are not bitwise the GEMV reduction.
    """

    def __call__(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.glm_router_gemm(hidden_states, weight)

    def reference(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """`hidden_states @ weight.T` accumulated in FP64 and rounded to FP32 once.

        The op's output is FP32, so an FP32-accumulated reference would carry rounding of the
        same size as the kernel's; FP64 keeps the reference the more accurate side.
        """
        return (hidden_states.double() @ weight.double().t()).float()

    def is_valid(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> None:
        # The shape gates of the kernel, which a target checks to route anything else to the
        # dense router fallback. The op raises on them as well.
        assert hidden_states.dim() == 2 and hidden_states.shape[1] == HIDDEN_SIZE, (
            f"hidden_states must be [tokens, {HIDDEN_SIZE}]"
        )
        # One CTA per expert row of the [256, 6144] gate weight.
        assert tuple(weight.shape) == (NUM_EXPERTS, HIDDEN_SIZE), (
            f"weight must be [{NUM_EXPERTS}, {HIDDEN_SIZE}], got {tuple(weight.shape)}"
        )
        # One kernel instance per token count.
        assert 1 <= hidden_states.shape[0] <= MAX_TOKENS, (
            f"the kernel takes 1 to {MAX_TOKENS} tokens, got {hidden_states.shape[0]}"
        )


router_gemm = GlmRouterGemm()
