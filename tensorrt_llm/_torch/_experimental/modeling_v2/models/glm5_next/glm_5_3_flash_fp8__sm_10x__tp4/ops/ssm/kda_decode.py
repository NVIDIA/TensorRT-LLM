# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-token KDA decode of B requests: the width-4 conv, the gated delta rule on each request's state row and
the gated output RMS norm, with the conv windows and state rows updated in place."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- loads the library that registers trtllm::kda_decode
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: KDA's K = V.
_HEAD_DIM = 128
#: The q / k L2-norm epsilon the kernel adds to the sum of squares.
_QK_NORM_EPS = 1e-6


class KdaDecode(OpWrapper):
    """Write the decode output of B requests into ``output``; update their conv windows and state rows.

    Per request ``b`` on row ``r`` (``ssm_state_indices[b]``, or ``b`` itself when it is None) and head ``h``: q, k
    and v are ``silu(bias + sum_j w_t[j] * [window | x][j])`` over the three-tap window ``conv_state_*[r]``
    (oldest first) and the new input, which then replaces the oldest tap. q and k are L2-normalized with
    ``rsqrt(sum(x^2) + 1e-6)`` (q then scaled); ``beta = sigmoid(beta_logit)``; the per-key decay is
    ``exp(lower_bound * sigmoid(exp(a_log[h]) * (g + dt_bias)))``. With the V-first ``[V, K]`` state:
    ``S <- S * decay``, ``S <- S + beta * (v - S k) k^T``, ``o = S q``, and the output is
    ``o * rsqrt(mean(o^2) + onorm_eps) * onorm_weight * sigmoid(onorm_g)``.

    The op's mode flags are fixed to the forms the kernels implement: output norm, lower-bound gate and beta
    sigmoid on, and the conv windows updated in place (``update_conv_cache``).

    Args:
        x_q, x_k, x_v: bf16 ``[1, B, H, 128]`` new conv inputs; head and channel axes dense, the row stride free
            (column sections of the fused projection).
        w_q_t, w_k_t, w_v_t: contiguous bf16 ``[4, H * 128]`` conv taps (tap 3 multiplies the new input).
        bias_q, bias_k, bias_v: contiguous bf16 ``[H * 128]`` conv biases (zeros for a bias-free conv).
        conv_state_q, conv_state_k, conv_state_v: bf16 ``[rows, H * 128, 3]`` sections of one packed
            ``[rows, 3 * H * 128, 3]`` conv pool (equal slot strides), updated in place.
        a_log: contiguous fp32 ``[H]``.
        g: bf16 ``[1, B, H, 128]`` raw forget-gate logits, row stride free.
        dt_bias: contiguous fp32 ``[H * 128]``.
        beta: bf16 ``[1, B, H]`` beta logits with a unit head stride.
        onorm_g: bf16 ``[1, B, H, 128]`` output-gate logits, row stride free.
        onorm_weight: contiguous fp32 ``[128]``.
        ssm_state_indices: int32 ``[B]`` unique rows of the pools, or None for rows ``0..B-1``.
        state: fp32 ``[rows, H, 128, 128]`` V-first state pool, updated in place; rows dense, slot stride a
            multiple of 4 elements, 16-byte aligned.
        lower_bound: the gate bound; scale: the q scale (``128 ** -0.5``); onorm_eps: the output norm epsilon.
        output: contiguous bf16 ``[B, 1, H, 128]``, written.

    Returns:
        ``output``.
    """

    ARCHS = frozenset({Arch.SM_100})
    # GLM-5.3-Flash's KDA at TP4 (16 of 64 heads). The op's own edges here; the decode steps the shipped target
    # runs are in this entry's test. On sm_100 with 16 heads the kernel follows B * H: B <= 2 takes the four-CTA
    # cluster kernel, 3 <= B <= 9 the legacy compact-heads one, 42 <= B <= 45 the legacy many-heads one,
    # 39 <= B <= 54 otherwise and 86 <= B <= 156 the single-CTA cp.async one, other batches the two-stage
    # bulk-copy one.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="two requests on scattered pool rows: the four-CTA cluster kernel's largest batch",
            spec=dict(batch=2, layout="indexed", pool=12),
        ),
        Cell(
            why="39 requests: the single-CTA cp.async kernel (624 <= B * H <= 865)",
            spec=dict(batch=39, layout="indexed", pool=48),
        ),
        Cell(
            why="44 requests: the legacy many-heads kernel (672 <= B * H <= 720)",
            spec=dict(batch=44, layout="indexed", pool=48),
        ),
    )
    note = """
    Fused: conv, SiLU, q/k norm, gate and beta activation, the recurrence and the
    gated RMS norm; the projections are the caller's. The op takes the conv
    windows and states only by row: the batch metadata is ssm_state_indices.
    With it, the rows are pool slots (a slot named twice races). Without it,
    request b uses row b of state and of the conv sections, the dense form a
    caller uses on fp32 staging copies of a bf16 pool's rows.

    The state must be fp32; the conv windows hold raw bf16 inputs, so their update
    is a bit copy. All math is fp32 CUDA-core code with fast exp / rsqrt; the
    output rounds once to bf16.

    One call per layer per step, in step order: swapping two steps of a request
    computes another history without any error. CUDA-graph safe: nothing
    allocates and nothing reads the device on the host.

    bias tensors are explicit: the op has no bias-free form, so a bias-free conv
    passes zeros. The op rejects the other flag combinations and malformed layouts
    itself (RuntimeError); is_valid states what it does not check.
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
        ssm_state_indices: torch.Tensor | None,
        state: torch.Tensor,
        lower_bound: float,
        scale: float,
        onorm_eps: float,
        output: torch.Tensor,
    ) -> torch.Tensor:
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
        return output

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
        ssm_state_indices: torch.Tensor | None,
        state: torch.Tensor,
        lower_bound: float,
        scale: float,
        onorm_eps: float,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """``(output, state, conv_state_q, conv_state_k, conv_state_v)`` as the call leaves them; mutates nothing.

        fp32 throughout (elementwise products and sums, no matmul); the output rounds once to bf16, the state
        rows stay fp32 and the conv windows take the raw inputs. ``output``'s prior content is ignored.
        """
        batch, heads = x_q.shape[1], x_q.shape[2]
        if ssm_state_indices is None:
            rows = torch.arange(batch, device=state.device)
        else:
            rows = ssm_state_indices.long()
        acts, windows = [], []
        for x, w_t, bias, conv in (
            (x_q, w_q_t, bias_q, conv_state_q),
            (x_k, w_k_t, bias_k, conv_state_k),
            (x_v, w_v_t, bias_v, conv_state_v),
        ):
            full = torch.cat([conv[rows].float(), x[0].reshape(batch, -1, 1).float()], dim=-1)
            act = (full * w_t.float().t()).sum(-1) + bias.float()
            acts.append(torch.nn.functional.silu(act).reshape(batch, heads, _HEAD_DIM))
            windows.append(full[..., 1:])
        q, k, v = acts
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + _QK_NORM_EPS) * scale
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + _QK_NORM_EPS)
        rate = a_log.float().exp().reshape(heads, 1)
        gate = lower_bound * torch.sigmoid(
            rate * (g[0].float() + dt_bias.float().reshape(heads, _HEAD_DIM))
        )
        b = beta[0].float().sigmoid()  # [B, H]
        s = state[rows].float() * gate.exp().unsqueeze(-2)  # [B, H, V, K]
        err = (v - (s * k.unsqueeze(-2)).sum(-1)) * b.unsqueeze(-1)
        s = s + err.unsqueeze(-1) * k.unsqueeze(-2)
        o = (s * q.unsqueeze(-2)).sum(-1)  # [B, H, V]
        o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + onorm_eps)
        o = o * onorm_weight.float() * onorm_g[0].float().sigmoid()
        new_state = state.clone()
        new_state[rows] = s.to(state.dtype)
        new_convs = []
        for conv, window in zip((conv_state_q, conv_state_k, conv_state_v), windows):
            updated = conv.clone()
            updated[rows] = window.to(conv.dtype)
            new_convs.append(updated)
        return (o.unsqueeze(1).to(output.dtype), new_state, *new_convs)

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
        ssm_state_indices: torch.Tensor | None,
        state: torch.Tensor,
        lower_bound: float,
        scale: float,
        onorm_eps: float,
        output: torch.Tensor,
    ) -> None:
        # The op checks dtypes, shapes, strides and the state's alignment itself; it never reads the slots.
        if ssm_state_indices is not None:
            slots = ssm_state_indices.tolist()
            assert len(set(slots)) == len(slots), (
                "ssm_state_indices must name each row once: two requests on a row race on its state"
            )
            assert all(0 <= slot < state.shape[0] for slot in slots), (
                "ssm_state_indices must lie in the pool: the kernel indexes the state and conv rows unchecked"
            )


kda_decode = KdaDecode()
