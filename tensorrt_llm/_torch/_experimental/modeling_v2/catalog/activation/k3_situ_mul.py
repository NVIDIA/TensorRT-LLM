# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiTU-gated multiply (SituAndMul) of a gate_up output for M <= 8 tokens via the CTM (CuTe DSL) kernel."""

import torch

import tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv.op  # noqa: F401  (registers torch.ops.trtllm.k3_situ_mul)


def k3_situ_mul(
    gu: torch.Tensor, beta: float = 1.0, linear_beta: float | None = None
) -> torch.Tensor:
    """Return `SituAndMul(beta, linear_beta)(gu)`, gate half first, in one k3_situ_mul call."""
    # The op hands the kernel `linear_beta or 1.0`, so 0.0 would run as linear_beta=1.0 (up half
    # tanh(u)) with no error, where the formula gives 0 * tanh(u / 0).
    assert linear_beta is None or linear_beta != 0.0, (
        "linear_beta=0.0 would silently run as 1.0; pass None to leave the up half unscaled"
    )
    return torch.ops.trtllm.k3_situ_mul(gu, beta=beta, linear_beta=linear_beta)
