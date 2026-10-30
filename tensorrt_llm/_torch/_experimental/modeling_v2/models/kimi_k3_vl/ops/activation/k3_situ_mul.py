# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiTU-gated multiply (SituAndMul) of a gate_up output for M <= 8 tokens via the CTM (CuTe DSL) kernel."""

import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_ctm_gemv.op  # noqa: F401  (registers torch.ops.trtllm.k3_situ_mul)
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class K3SituMul(OpWrapper):
    """Return `SituAndMul(beta, linear_beta)(gu)`, gate half first, in one k3_situ_mul call."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{name} caps at M={m}: TP16 dense MLP row-count independence",
            spec=dict(rows=m, width=4224, beta=beta, linear_beta=linear_beta),
        )
        for name, beta, linear_beta in (("K3", 4.0, 25.0), ("default", 1.0, None))
        for m in range(1, 9)
    )
    note = """
    Activation only: the caller owns gate_up and down projections; no quantization.
    Inputs are not mutated; the output is new, contiguous bf16. The kernel uses
    fp32 IEEE division, exp and tanh, whose implementations differ from torch's.
    Identical calls and a row at different token counts have identical bits.

    No metadata or module state is consumed; the current CUDA stream is used.
    A result-neutral process cache holds compiled kernels by
    (K, linear_beta is not None, PDL). Warm each key eagerly before graph capture;
    a cold capture raises RuntimeError. M and cap values are runtime arguments.
    TRTLLM_ENABLE_PDL is read per call: unset or 1 enables PDL. Each CTA releases
    dependents, then waits for its predecessor before reading gu. A PDL consumer
    must wait before reading the output; non-PDL consumers retain stream ordering.
    The grid is ceil(K / 1024) by 8 CTAs of 128 threads, eight columns per thread;
    rows past M exit. Other widths and caps are accepted but not certified.
    Width zero passes the op's check; K >= 8 is the supported geometry.
    cutlass and cuda.bindings are required, otherwise import fails loudly.
    """

    def __call__(
        self, gu: torch.Tensor, beta: float = 1.0, linear_beta: float | None = None
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_situ_mul(gu, beta=beta, linear_beta=linear_beta)

    def reference(
        self, gu: torch.Tensor, beta: float = 1.0, linear_beta: float | None = None
    ) -> torch.Tensor:
        k = gu.shape[1] // 2
        g, u = gu[:, :k].float(), gu[:, k:].float()
        a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
        if linear_beta is not None:
            u = linear_beta * torch.tanh(u / linear_beta)
        return (a * u).to(gu.dtype)

    def is_valid(
        self, gu: torch.Tensor, beta: float = 1.0, linear_beta: float | None = None
    ) -> None:
        # The op hands the kernel `linear_beta or 1.0`, so 0.0 would run as linear_beta=1.0 (up half
        # tanh(u)) with no error, where the formula gives 0 * tanh(u / 0).
        assert linear_beta is None or linear_beta != 0.0, (
            "linear_beta=0.0 would silently run as 1.0; pass None to leave the up half unscaled"
        )


k3_situ_mul = K3SituMul()
