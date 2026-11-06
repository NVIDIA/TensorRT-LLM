# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One rank's slice of the BF16 dense MLP for up to 8 decode rows, via the trtllm
glm_dense_mlp_small_m op (two weight-streaming kernels)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: GLM-5.2's hidden size, the contraction of gate/up and the output width of down.
HIDDEN = 6144
#: The rank-local intermediate size of the dense layers 0-2 at TP16: 12288 / 16.
INTER_PER_RANK = 768


class GlmDenseMlpSmallM(OpWrapper):
    """`out = (silu(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T` on one rank's slice, 1-8 rows.

    The slice is TP16's: `gate_up_weight` [2 * 768, 6144] stacks this rank's
    768 gate_proj rows over the same 768 up_proj rows (gate first), and
    `down_weight` [6144, 768] holds the same 768 input columns of down_proj.
    `out` [M, 6144] is the rank-local partial: the sum over the 16 ranks, which
    the post-MLP all-reduce forms, is the MLP output. Returns a new tensor; no
    input is modified.

    Rounding points are the unfused chain's: the gate and up sums round to
    bf16 (the bf16 gate_up GEMM output), SwiGLU runs in fp32 and rounds once to
    bf16 `h`, and the down sum rounds to bf16. Sums are fp32 in another order.

    Two kernels on the current stream: gate/up + SwiGLU (192 CTAs in 2-CTA
    clusters that split the 6144 contraction) and down (96 CTAs of 64 output
    columns). Under programmatic dependent launch each CTA bulk-copies all of
    its weights into shared memory before its grid-dependency wait, and reads
    `x` (gate/up) or `h` (down) only after it. gate/up releases down at entry;
    down releases its own dependents once its weight copies are issued, so the
    kernel after the op must wait before it reads `out`.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The target calls the op on the dense layers 0-2 of a generation-only
    # forward: rows = requests x (draft length + 1), at most 8. Operand scales
    # are the checkpoint's: post_attention_layernorm's gamma averages 0.021 on
    # these layers (x has unit RMS before it), gate/up weights 0.014 std, down
    # weights 0.013 std. `saturated` scales gate/up so the gate sums reach the
    # sigmoid's tails.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1 verify forward: MTP K=7's 8 rows through a TP16 dense layer",
            spec=dict(rows=8, x_scale=0.021, gate_up_std=0.014, down_std=0.013),
        ),
        Cell(
            why="decode without speculation: 1 row, 7 of the 8 MMA token columns empty",
            spec=dict(rows=1, x_scale=0.021, gate_up_std=0.014, down_std=0.013),
        ),
        Cell(
            why="MTP K=3: 4 rows, the empty token columns start at a thread's pair",
            spec=dict(rows=4, x_scale=0.021, gate_up_std=0.014, down_std=0.013),
        ),
        Cell(
            why="saturated SwiGLU: gate sums to +-40, the fast exp and division in sigmoid's tails",
            spec=dict(rows=8, x_scale=1.0, gate_up_std=0.13, down_std=0.036),
        ),
    )

    note = """
    Shapes: hidden 6144, rank-local intermediate 768, 1-8 rows; BF16,
    contiguous, 16-byte aligned operands on an sm_90 or newer device. The op
    raises on anything else, so the caller routes other shapes (more rows,
    context forwards, another TP) to a generic gated MLP.
    `torch.ops.trtllm.glm_dense_mlp_small_m_supported(rows, hidden, inter)`
    answers the shape question on the host.

    The weight layout is the caller's to get right and no check can see it: a
    gate_up_weight that stacks up over gate, or that takes rows
    [r * 1536, (r + 1) * 1536) of a pre-fused [gate; up] tensor, gives a
    plausible wrong MLP.

    The weights are read before the grid-dependency wait, so they must not be
    written by any kernel still running ahead of the op on the stream.
    """

    def __call__(
        self, x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.trtllm.glm_dense_mlp_small_m(x, gate_up_weight, down_weight)

    def reference(
        self, x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor
    ) -> torch.Tensor:
        """The rank partial with the op's rounding points, fp32 sums.

        The intermediate bf16 roundings are part of what the op computes (the
        unfused bf16 GEMMs round there too), so the reference rounds at the
        same points instead of carrying fp32 through; without them the band
        would measure those roundings, not the kernel.

        TF32 off: torch's default fp32 matmul on this hardware is a TF32
        product, which would make the reference the inaccurate side.
        """
        inter = down_weight.shape[1]
        with true_fp32_matmul():
            gate_up = x.float() @ gate_up_weight.float().t()
        gate = gate_up[:, :inter].to(torch.bfloat16).float()
        up = gate_up[:, inter:].to(torch.bfloat16).float()
        h = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        with true_fp32_matmul():
            return (h.float() @ down_weight.float().t()).to(x.dtype)

    def is_valid(
        self, x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor
    ) -> None:
        """Nothing to refuse: every input the op accepts it computes as stated.

        The op raises on every shape, dtype, layout, alignment and device it
        does not serve, including pre-sm_90 devices where the kernels have no
        body. What is left is the weight layout (see `note`), which no check on
        the tensors can see.
        """
        return None


dense_mlp_small_m = GlmDenseMlpSmallM()
