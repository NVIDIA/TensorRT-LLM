# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``x @ weight^T`` with an fp32 output from bf16 operands, on the shared ``trtllm::dsv3_router_gemm_op``: Kimi K3's
router logits (896 experts at hidden 7168) and the decode path's fp32 GEMMs."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class Dsv3RouterGemm(OpWrapper):
    """Return ``hidden_states @ weight^T`` as a new fp32 ``[M, N]`` tensor. ``weight`` is ``[N, K]`` row-major (the
    op reads it as the column-major ``[K, N]`` view)."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3 router logits, M={m}: the op's 896 x 7168 tensor-core kernel",
                spec=dict(m=m, n=896, k=7168),
            )
            for m in (1, 8, 16)
        ),
        *(
            Cell(
                why=f"K3 router logits, M={m}: above 16 rows the op falls back to cuBLAS",
                spec=dict(m=m, n=896, k=7168),
            )
            for m in (17, 64, 2048)
        ),
        Cell(
            why="decode MoE latent up (moe_up): 3584 -> 7168 on cuBLAS, fp32 out",
            spec=dict(m=8, n=7168, k=3584),
        ),
    )

    note = """
    bf16 operands, fp32 accumulation and output, no bias. Up to 16 rows at
    N=896, K=7168 the op runs its MMA kernel (one fixed reduction order); every
    other shape runs cuBLAS. A weight that is not contiguous with a 16-byte
    aligned start is copied into that layout first, so the call is correct but
    slower; `is_valid` rejects a non-contiguous weight because the target never
    passes one and the decode path's fp32 GEMM refuses it too.
    Deterministic per shape; cuBLAS and the MMA kernel need not agree bit for
    bit with each other or with the reference, which sums in another order.
    """

    def __call__(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.dsv3_router_gemm_op(
            hidden_states, weight.t(), bias=None, out_dtype=torch.float32
        )

    def reference(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """fp64 accumulation, rounded to fp32 once: the reference stays the more accurate side."""
        with true_fp32_matmul():
            return (hidden_states.double() @ weight.double().t()).float()

    def is_valid(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> None:
        assert hidden_states.dim() == 2 and hidden_states.stride(1) == 1, (
            "hidden_states must be a row-major 2-D tensor"
        )
        assert weight.is_contiguous(), "weight must be a contiguous [N, K] tensor"


dsv3_router_gemm = Dsv3RouterGemm()
