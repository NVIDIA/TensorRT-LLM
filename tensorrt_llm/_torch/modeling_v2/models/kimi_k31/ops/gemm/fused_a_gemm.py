# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MoE latent down / up projection through the engine's dsv3_fused_a_gemm_op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

HIDDEN = 7168
#: The routed experts' latent width (``moe_hidden_size``) is read off the weights; Kimi K3.1's.
LATENT = 3584


def _proj(tokens: int, k: int, n: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(tokens=tokens, k=k, n=n))


class K31FusedAGemm(OpWrapper):
    """``hidden_states @ weight_t``: BF16 ``[tokens, K] x [K, N] -> [tokens, N]``, FP32 accumulation.

    ``weight_t`` is the transposed view of an ``nn.Linear`` weight (``projection.weight.t()``,
    column-major ``[K, N]``). For the Kimi K3.1 latent shapes the engine op runs its cuBLAS GEMM
    (``cublas_mm_out``; its own min-latency kernel serves only the DeepSeek / GLM fused-A shapes).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _proj(1, HIDDEN, LATENT, "latent down projection, one decode row"),
        _proj(8, HIDDEN, LATENT, "latent down projection, a DFlash2 verify step"),
        _proj(2954, HIDDEN, LATENT, "latent down projection, a prefill chunk"),
        _proj(1, LATENT, HIDDEN, "latent up projection, one decode row"),
        _proj(2954, LATENT, HIDDEN, "latent up projection, a prefill chunk"),
        _proj(96, HIDDEN, LATENT, "latent down projection, the DEP16 and Helix CP 8 verify step"),
        _proj(96, LATENT, HIDDEN, "latent up projection, the DEP16 and Helix CP 8 verify step"),
    )

    note = """
    The op takes no bias (it raises on one) and BF16 output by default. The latent projections are
    replicated (not tensor-parallel), so every Mapping calls the same shapes.
    """

    def __call__(self, hidden_states: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(hidden_states, weight_t, None, None)

    def reference(self, hidden_states: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        """The FP32 product rounded once to BF16 (TF32 off)."""
        with true_fp32_matmul():
            return (hidden_states.float() @ weight_t.float()).to(hidden_states.dtype)

    def is_valid(self, hidden_states: torch.Tensor, weight_t: torch.Tensor) -> None:
        # The op checks the strides it needs (row-major A, column-major B) itself. It does not
        # check K: a weight with another inner size than the activation's is read with the
        # activation's.
        assert hidden_states.dim() == 2 and weight_t.dim() == 2, "both operands must be 2D"
        assert hidden_states.shape[1] == weight_t.shape[0], "hidden_states [M, K] x weight_t [K, N]"


k31_fused_a_gemm = K31FusedAGemm()
