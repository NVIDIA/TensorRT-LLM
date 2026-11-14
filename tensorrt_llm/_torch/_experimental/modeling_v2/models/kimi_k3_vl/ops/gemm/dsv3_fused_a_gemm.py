# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``x @ weight^T`` in bf16 on the shared ``trtllm::dsv3_fused_a_gemm_op``: Kimi K3's latent MoE projections
(7168 -> 3584 down, 3584 -> 7168 up)."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class Dsv3FusedAGemm(OpWrapper):
    """Return ``hidden_states @ weight^T`` as a new bf16 ``[M, N]`` tensor. ``weight`` is ``[N, K]`` row-major (the
    op reads it as the column-major ``[K, N]`` view)."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"K3 latent {name}, M={m}: decode, wide decode and prefill rows",
            spec=dict(m=m, n=n, k=k),
        )
        for name, n, k in (("down", 3584, 7168), ("up", 7168, 3584))
        for m in (1, 9, 64, 65, 2048)
    )

    note = """
    bf16 operands, fp32 accumulation, one bf16 rounding, no bias. The op's own
    low-latency kernel covers only the DeepSeek-V3 (7168 -> 2112) and GLM
    (6144 -> 2624) shapes up to 16 rows; Kimi K3's latent shapes always run
    its cuBLAS path (`cublas_mm_out`). The target calls this op rather than
    another cuBLAS entry so the GEMM stays the one the built-in module runs.
    Deterministic per shape; the reference sums in another order, so the two
    agree within bf16 rounding, not bit for bit.
    """

    def __call__(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(hidden_states, weight.t(), None, None)

    def reference(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """fp32 accumulation (TF32 off), rounded to bf16 once."""
        with true_fp32_matmul():
            return (hidden_states.float() @ weight.float().t()).to(hidden_states.dtype)


dsv3_fused_a_gemm = Dsv3FusedAGemm()
