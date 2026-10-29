# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draft tokens through the MTP layer's BF16 MoE, routing included, via the glm_lowm_bf16_moe op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class GlmLowmBf16Moe(OpWrapper):
    """The routed half of the MTP layer's MoE for one to six tokens, from the router logits to the weighted expert sum.

    Routing is the n_group == 1 case of `noaux_tc_op`: experts by `sigmoid(logits) + bias` (sigmoid in the tanh
    form), the top 8 in descending order with the lower expert id first on ties, weights from the unbiased scores
    normalized over the 8 and scaled by `routed_scaling_factor`. Then per routed expert `e_k`:
    `act_k = bf16(silu(bf16(W1[e_k] x)) * bf16(W3[e_k] x))` and the output is
    `bf16(sum_k bf16(w_k * (W2[e_k] act_k)))`, the sum in FP32 in the routing order.

    `w3_w1` is `[256, 2 * I, H]` with the up rows (W3) first and the gate rows (W1) second, `w2` is `[256, H, I]`,
    all experts local (tensor-parallel slices, no expert parallelism). Returns `[M, H]`; the shared experts are not
    part of the op.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The MTP layer (layer 78) is BF16 throughout; at tp16 an expert's intermediate slice is 2048 / 16 = 128. Its
    # MoE runs on one row on every draft step (step 0 after the row select). The checkpoint's correction bias is
    # FP32 and its routed scaling factor 2.5. The value cells drive the routing's three regimes.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="draft step at tp16: one token, hidden 6144, expert slice 128, random routing",
            spec=dict(hidden=6144, inter=128, routing="random"),
        ),
        Cell(
            why="tied routing keys at the top-8 boundary: the lower expert id wins",
            spec=dict(hidden=6144, inter=128, routing="ties"),
        ),
        Cell(
            why="selected scores summing below 2^-13: the weights take the FP64 expression",
            spec=dict(hidden=6144, inter=128, routing="tiny"),
        ),
        *(
            Cell(
                why=f"TP4 draft batch {tokens}: intermediate512, {routing} routing",
                spec=dict(tokens=tokens, hidden=6144, inter=512, routing=routing),
            )
            for tokens in (1, 2, 3)
            for routing in ("random", "ties", "tiny")
        ),
        *(
            Cell(
                why=f"TP8 draft batch {tokens}: intermediate256, {routing} routing",
                spec=dict(tokens=tokens, hidden=6144, inter=256, routing=routing),
            )
            for tokens in (1, 2)
            for routing in ("random", "ties", "tiny")
        ),
    )

    note = """
    The kernels serve hidden sizes that are multiples of 2048 up to 8192 and expert slices that are multiples of
    64 up to 512, exactly 256 experts and top 8. Multi-token calls serve 2--6 rows with
    hidden size at most 6144; the target uses the wide512 slice only for draft batches2/3. A
    BF16 correction bias is accepted as well; the checkpoint's is FP32, so no cell drives it.

    The order of the two halves of `w3_w1` is not visible in the shapes: a gate-first tensor gives silently wrong
    output. The target builds `w3_w1` up-first from the checkpoint's `up_proj` and `gate_proj`.

    Under PDL the FC2 kernel reads the router logits before its grid-dependency wait: the logits must be complete
    when the FC1 kernel's wait returns, which holds for any producer earlier in stream order.
    """

    def __call__(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        w3_w1: torch.Tensor,
        w2: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
    ) -> torch.Tensor:
        return torch.ops.trtllm.glm_lowm_bf16_moe(
            x, router_logits, e_score_correction_bias, w3_w1, w2, top_k, routed_scaling_factor
        )

    def reference(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        w3_w1: torch.Tensor,
        w2: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
    ) -> torch.Tensor:
        """Plain torch with the op's rounding points.

        BF16 after FC1, after SwiGLU and after each expert's scaled FC2 output, the routing weights in FP64 as
        `noaux_tc_op` forms them, the expert sum in FP32 in the routing order. Products and sums are FP32 (no
        TF32).
        """
        rows = []
        for row in range(x.shape[0]):
            inter = w2.shape[2]
            scores = 0.5 * torch.tanh(0.5 * router_logits.float()[row]) + 0.5
            choice = scores + e_score_correction_bias.float()
            ids = torch.sort(choice, descending=True, stable=True).indices[:top_k]
            selected = scores[ids]
            total = selected.sum().double()
            weights = (selected.double() * routed_scaling_factor / (total + 1e-20)).float()
            xf = x.float()[row]
            out = torch.zeros(w2.shape[1], dtype=torch.float32, device=x.device)
            with true_fp32_matmul():
                for k in range(top_k):
                    expert = int(ids[k])
                    fc1 = w3_w1[expert].float() @ xf
                    up = fc1[:inter].to(torch.bfloat16).float()
                    gate = fc1[inter:].to(torch.bfloat16).float()
                    act = (gate * (1.0 / (1.0 + torch.exp(-gate))) * up).to(torch.bfloat16).float()
                    term = (weights[k] * (w2[expert].float() @ act)).to(torch.bfloat16).float()
                    out = out + term
            rows.append(out.to(torch.bfloat16))
        return torch.stack(rows)

    def is_valid(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        w3_w1: torch.Tensor,
        w2: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
    ) -> None:
        # Nothing to add: the op refuses every token count, expert count, top-k, shape, dtype, layout and
        # alignment it does not serve, loudly, and the one silent mistake left -- a gate-first `w3_w1` -- is not
        # visible in the tensors (see `note`).
        del x, router_logits, e_score_correction_bias, w3_w1, w2, top_k, routed_scaling_factor


glm_lowm_bf16_moe = GlmLowmBf16Moe()
