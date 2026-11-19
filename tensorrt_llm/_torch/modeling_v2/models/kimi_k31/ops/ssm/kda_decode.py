# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-token KDA decode of B requests: the causal conv, the gated delta rule on each request's state slot and the
gated RMS norm, the conv and state pools updated in place."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class KdaDecode(OpWrapper):
    """Writes the decode output into `output` [B, 1, HV, 128] bf16; updates `state` and the conv
    windows at `ssm_state_indices`. Returns None."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{heads} KDA heads per rank, B={batch}: indexed fp32 pool, packed conv pool updated in place",
                spec=dict(heads=heads, batch=batch, pool_slots=batch + 4),
            )
            for heads in (96, 48, 24, 12, 6, 3)
            for batch in (1, 4, 8)
        ),
        Cell(
            why="larger decode batch B=64 at 12 heads (legacy compact-heads dispatch)",
            spec=dict(heads=12, batch=64, pool_slots=70),
        ),
        Cell(
            why="larger decode batch B=32 at 96 heads (attention TP1 / DP)",
            spec=dict(heads=96, batch=32, pool_slots=36),
        ),
        Cell(
            why="sm_107 DEP16 GEN: 96 KDA heads per rank (attention DP), B=12",
            spec=dict(heads=96, batch=12, pool_slots=16),
        ),
        Cell(
            why="sm_107 Helix CP 8 GEN (KDA TP8): 12 heads, B=16",
            spec=dict(heads=12, batch=16, pool_slots=20),
        ),
    )
    note = """
    The raw trtllm::kda_decode schema; run_kda_decode_fusion_cuda, which the
    target's KDA layer reaches today, only validates, makes views contiguous
    and fills the optional arguments: zero bf16 biases ([H * 128]) and, when
    unused, a dummy onorm_g / unit onorm_weight. Here every argument is
    explicit. The target's call: x_q/x_k/x_v [1, B, H, 128] bf16 column
    views of the fused qkvg GEMM output (head/channel axes packed, token
    stride free); conv weights w_*_t [4, H * 128] bf16; conv_state_{q,k,v}
    the three section views of the packed [slots, 3D, 3] bf16 conv pool
    (exactly that packed layout); fp32 a_log [H], dt_bias [H * 128],
    onorm_weight [128]; g [1, B, H, 128] and beta
    [1, B, H] bf16 raw logits (beta may be a strided column view); onorm_g
    [1, B, H, 128] bf16; the fp32 V-first state pool [slots, H, 128, 128]
    indexed by int32 ssm_state_indices (slot stride may be padded);
    lower_bound=-5, scale=128**-0.5, onorm_eps = the layer's rms_norm_eps.
    The kernel runs with the gated output norm, the conv-window update, the
    lower-bound gate and the beta sigmoid.

    In place: output (inplace-only op, never allocates), state rows and the
    selected conv windows (shifted by the new raw input, bitwise). Approximate
    exp / rcp: output within 2e-2 and state within 1e-3 of an fp32
    evaluation, relative to the largest magnitude.
    Graph-capturable (no allocation, no host sync). ARCHS lists only the
    architectures the cells were driven on: sm_103 (GB300) and sm_107 (VR200).
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
        ssm_state_indices: torch.Tensor,
        state: torch.Tensor,
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
            True,  # apply_onorm
            True,  # update_conv_cache
            True,  # use_lower_bound
            True,  # apply_beta_sigmoid
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
        ssm_state_indices: torch.Tensor,
        state: torch.Tensor,
        lower_bound: float,
        scale: float,
        onorm_eps: float,
        output: torch.Tensor,
    ) -> None:
        heads, head_dim = x_q.shape[-2:]
        slots = ssm_state_indices.tolist()
        for row, slot in enumerate(slots):
            acts = []
            for x, weight, bias, conv in zip(
                (x_q, x_k, x_v),
                (w_q_t, w_k_t, w_v_t),
                (bias_q, bias_k, bias_v),
                (conv_state_q, conv_state_k, conv_state_v),
            ):
                full = torch.cat([conv[slot].float(), x[0, row].float().reshape(-1, 1)], dim=-1)
                act = (full * weight.float().t()).sum(-1) + bias.float()
                acts.append(torch.nn.functional.silu(act).reshape(heads, head_dim))
                conv[slot] = full[:, 1:].to(conv.dtype)
            q, k, v = acts
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * scale
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            b = beta[0, row].float().sigmoid()
            gate = g[0, row].float() + dt_bias.float().reshape(heads, head_dim)
            rate = a_log.float().exp().unsqueeze(-1)
            gate = lower_bound * torch.sigmoid(rate * gate)
            st = state[slot].float() * gate.exp().unsqueeze(-2)
            residual = (v - (st * k.unsqueeze(-2)).sum(-1)) * b.unsqueeze(-1)
            st = st + residual.unsqueeze(-1) * k.unsqueeze(-2)
            state[slot] = st
            o = (st * q.unsqueeze(-2)).sum(-1)
            rms = torch.rsqrt(o.square().mean(-1, keepdim=True) + onorm_eps)
            o = o * rms * onorm_weight.float() * onorm_g[0, row].float().sigmoid()
            output[row, 0] = o.to(output.dtype)

    def is_valid(
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
        ssm_state_indices: torch.Tensor,
        state: torch.Tensor,
        lower_bound: float,
        scale: float,
        onorm_eps: float,
        output: torch.Tensor,
    ) -> None:
        slots = ssm_state_indices.tolist()
        assert len(set(slots)) == len(slots), (
            "ssm_state_indices must name each slot once: the kernel updates rows in place"
        )
        assert all(0 <= slot < state.shape[0] for slot in slots), "state slot out of range"


kda_decode = KdaDecode()
