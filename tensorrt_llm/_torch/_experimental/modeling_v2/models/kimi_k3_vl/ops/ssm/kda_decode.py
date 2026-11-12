# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-token KDA decode of B requests: the causal conv, the gated delta rule on each request's state slot and the
gated RMS norm, the conv and state pools updated in place."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class KdaDecode(OpWrapper):
    """Writes the decode output into ``output`` [B, 1, HV, 128] bf16; updates ``state`` at ``ssm_state_indices``
    and, with ``update_conv_cache``, the conv pools there. Returns None."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=(
                f"{num} requests at six heads on real manager pools: "
                f"{'four-CTA cluster' if num <= 5 else 'legacy compact-heads'} dispatch"
            ),
            spec=dict(num=num),
        )
        for num in range(1, 9)
    )
    note = """
    Caller-owned pools only. The V2 hybrid manager uses conv_state_layout=q_k_v,
    bf16 conv windows and fp32 V-first SSM states, with per-slot coalescing
    strides. The conv arguments are sections of one packed pool when
    update_conv_cache=True, otherwise dense per-request windows. Only the
    named layer's slots are changed; other slots, layers and padding survive.
    ssm_state_indices is the only batch metadata.

    One call per layer per step, in step order; each batch names every slot
    once. Swapping steps silently computes another history. Tests drive three
    layers, repeated steps, CUDA-graph replay with rewritten inputs, and a
    swapped-order negative control.

    K=V=128, width 4, bf16 conv inputs/gates/output and fp32 recurrent state.
    apply_onorm, use_lower_bound and apply_beta_sigmoid must all be True;
    the op rejects other combinations and invalid layouts before launch.
    Kimi K3 uses six heads, lower_bound=-5, scale=128**-0.5, onorm_eps=1e-5.
    On SM100/SM103 B*H <= 32 selects four-CTA clusters, <=144 legacy compact
    heads, then architecture-specific kernels; other architectures use legacy
    kernels. Only SM100 B=1..8 is certified here.

    run_kda_decode_fusion_cuda fills optional arguments with cached dummies;
    this entry exposes the raw schema. k3_kda_decode_attn additionally fuses
    the projection for up to eight requests. Output and state comparison
    bands are 2e-2 and 1e-3 respectively; conv windows copy raw inputs exactly.
    """

    def __call__(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        w_q_t: torch.Tensor,
        w_k_t: torch.Tensor,
        w_v_t: torch.Tensor,
        bias_q: torch.Tensor,
        bias_k: torch.Tensor,
        bias_v: torch.Tensor,
        conv_state_q: torch.Tensor,
        conv_state_k: torch.Tensor,
        conv_state_v: torch.Tensor,
        a_log: torch.Tensor,
        g: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: torch.Tensor,
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        ssm_state_indices: Optional[torch.Tensor],
        state: torch.Tensor,
        apply_onorm: bool,
        update_conv_cache: bool,
        use_lower_bound: bool,
        apply_beta_sigmoid: bool,
        lower_bound: float,
        scale: float,
        onorm_eps: float,
        output: torch.Tensor,
    ) -> None:
        torch.ops.trtllm.kda_decode(
            x_q,
            x_k,
            x_v,
            w_q_t,
            w_k_t,
            w_v_t,
            bias_q,
            bias_k,
            bias_v,
            conv_state_q,
            conv_state_k,
            conv_state_v,
            a_log,
            g,
            dt_bias,
            beta,
            onorm_g,
            onorm_weight,
            ssm_state_indices,
            state,
            apply_onorm,
            update_conv_cache,
            use_lower_bound,
            apply_beta_sigmoid,
            lower_bound,
            scale,
            onorm_eps,
            output=output,
        )

    def reference(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        w_q_t: torch.Tensor,
        w_k_t: torch.Tensor,
        w_v_t: torch.Tensor,
        bias_q: torch.Tensor,
        bias_k: torch.Tensor,
        bias_v: torch.Tensor,
        conv_state_q: torch.Tensor,
        conv_state_k: torch.Tensor,
        conv_state_v: torch.Tensor,
        a_log: torch.Tensor,
        g: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: torch.Tensor,
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        ssm_state_indices: Optional[torch.Tensor],
        state: torch.Tensor,
        apply_onorm: bool,
        update_conv_cache: bool,
        use_lower_bound: bool,
        apply_beta_sigmoid: bool,
        lower_bound: float,
        scale: float,
        onorm_eps: float,
        output: torch.Tensor,
    ) -> None:
        heads, head_dim = x_q.shape[-2:]
        slots = range(x_q.shape[1]) if ssm_state_indices is None else ssm_state_indices.tolist()
        for row, slot in enumerate(slots):
            acts = []
            for x, weight, bias, conv in zip(
                (x_q, x_k, x_v),
                (w_q_t, w_k_t, w_v_t),
                (bias_q, bias_k, bias_v),
                (conv_state_q, conv_state_k, conv_state_v),
            ):
                conv_row = slot if update_conv_cache else row
                full = torch.cat([conv[conv_row].float(), x[0, row].float().reshape(-1, 1)], dim=-1)
                act = (full * weight.float().t()).sum(-1) + bias.float()
                acts.append(torch.nn.functional.silu(act).reshape(heads, head_dim))
                if update_conv_cache:
                    conv[conv_row] = full[:, 1:].to(conv.dtype)
            q, k, v = acts
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * scale
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            b = beta[0, row].float()
            if apply_beta_sigmoid:
                b = b.sigmoid()
            gate = g[0, row].float() + dt_bias.float().reshape(heads, head_dim)
            rate = a_log.float().exp().unsqueeze(-1)
            gate = (
                lower_bound * torch.sigmoid(rate * gate)
                if use_lower_bound
                else -rate * torch.nn.functional.softplus(gate)
            )
            st = state[slot].float() * gate.exp().unsqueeze(-2)
            residual = (v - (st * k.unsqueeze(-2)).sum(-1)) * b.unsqueeze(-1)
            st = st + residual.unsqueeze(-1) * k.unsqueeze(-2)
            state[slot] = st
            o = (st * q.unsqueeze(-2)).sum(-1)
            if apply_onorm:
                rms = torch.rsqrt(o.square().mean(-1, keepdim=True) + onorm_eps)
                o = o * rms * onorm_weight.float() * onorm_g[0, row].float().sigmoid()
            output[row, 0] = o.to(output.dtype)


kda_decode = KdaDecode()
