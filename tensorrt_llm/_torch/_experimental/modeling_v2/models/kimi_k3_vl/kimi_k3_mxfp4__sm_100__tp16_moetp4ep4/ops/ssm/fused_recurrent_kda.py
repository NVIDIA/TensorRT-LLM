# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLA's token-by-token KDA (``fla.ops.kda.fused_recurrent_kda``) from dense V-first states: one step of Kimi K3's
sequential KDA verify."""

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class FusedRecurrentKda(OpWrapper):
    """The gated delta rule over T tokens of each of B rows from ``initial_state`` [B, H, V, K] (V-first, fp32):
    returns new ``(o [B, T, H, V], final_state [B, H, V, K])``; ``initial_state`` is not written.

    Q/K L2 normalization (q scaled by ``K ** -0.5``), the beta sigmoid and the lower-bound gate
    ``exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias)))`` run in the kernel, on the raw inputs.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"TP16, {rows} requests: one sequential-verify step from the gathered states",
                spec=dict(rows=rows, heads=6, tokens=1),
            )
            for rows in (1, 4, 8)
        ),
        Cell(
            why="TP4 heads, two tokens per row",
            spec=dict(rows=2, heads=24, tokens=2),
        ),
    )
    note = """
    Kimi K3's sequential verify calls it once per verify token with q / k / v
    the conv outputs [B, 1, H, 128] bf16 (raw, not normalized), g the f_b
    gate [B, 1, H, 128] bf16 (dt_bias not added), beta the fp32 [B, 1, H]
    logits, A_log [H] and dt_bias [H 128] the module's bf16 parameters,
    initial_state the dense fp32 [B, H, 128, 128] states gathered from the
    pool rows, lower_bound -5. initial_state is read only; the final state
    is a new fp32 tensor.

    FLA is imported on first call. The kernel's launch configuration is fixed
    (fla-core 0.5.2 does not autotune it), so equal inputs give equal bits.
    Output band 2e-2 (bf16), state band 1e-3 (fp32).
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        initial_state: torch.Tensor,
        lower_bound: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Imported on the first call, so that importing the target does not need FLA.
        from fla.ops.kda import fused_recurrent_kda as fla_fused_recurrent_kda

        return fla_fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=lower_bound,
            state_v_first=True,
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        initial_state: torch.Tensor,
        lower_bound: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heads, dim = q.shape[-2:]
        qf = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6) * dim**-0.5
        kf = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
        vf = v.float()
        gate = g.float() + dt_bias.float().reshape(heads, dim)
        rate = A_log.float().exp().reshape(-1, 1)
        decay = (lower_bound * torch.sigmoid(rate * gate)).exp()  # [B, T, H, K]
        b = beta.float().sigmoid()
        state = initial_state.float().clone()  # [B, H, V, K]
        out = torch.empty_like(vf)
        for t in range(q.shape[1]):
            state = state * decay[:, t].unsqueeze(-2)
            err = vf[:, t] - (state * kf[:, t].unsqueeze(-2)).sum(-1)
            state = state + b[:, t, :, None, None] * err.unsqueeze(-1) * kf[:, t].unsqueeze(-2)
            out[:, t] = (state * qf[:, t].unsqueeze(-2)).sum(-1)
        return out.to(v.dtype), state


fused_recurrent_kda = FusedRecurrentKda()
