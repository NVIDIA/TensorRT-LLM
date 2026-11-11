# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA speculative verify of B requests, 1 + num_spec new tokens each: replay of the previous round's accepted
drafts, then the conv, the gated delta rule and the gated output RMS norm of the new tokens, with the state and
conv window committed after the first (golden) token and the new drafts cached for the next round."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops  # noqa: F401 -- registers trtllm::kda_mtp_decode

from .._op import Arch, Cell, OpWrapper

#: KDA's K = V, and the committed conv window (conv width 4 - 1).
_HEAD_DIM = 128
_WINDOW = 3
#: The q / k L2-norm epsilon the kernel adds to the sum of squares.
_QK_NORM_EPS = 1e-6
#: The alignment the op promises its CuTe kernels for every tensor operand but the metadata and beta cache.
_ALIGN = 16


class KdaMtpDecode(OpWrapper):
    """Verify ``1 + num_spec`` packed new tokens per request in one launch; return the normalized output.

    Request ``n`` owns slot ``s = ssm_state_indices[n]`` and rows ``n * (1 + num_spec) + j`` of the token inputs.
    Its slot holds the state after its last golden token, the raw conv inputs ``cs_*[s, :, 0:3]`` before it and,
    from the previous round, ``num_spec`` drafts: their raw inputs ``cs_*[s, :, 3:3 + num_spec]`` and their
    normalized k, log-decay g, v and beta in ``k_cache`` / ``g_cache`` / ``v_cache`` / ``beta_cache``. Of those,
    the first ``a = num_accepted_tokens[s]`` were accepted.

    The kernel replays the a accepted drafts on the state (``S <- S * exp(g)``, ``S <- S + beta (v - S k) k^T``
    from the caches), then runs the new tokens: q, k, v are ``silu(sum_j w[:, j] * x[t + j])`` over the raw
    sequence ``[window | accepted drafts | new tokens]``, q and k L2-normalized with ``rsqrt(sum(x^2) + 1e-6)``
    (q then scaled), ``beta = sigmoid(beta_logit)``, ``g = lower_bound * sigmoid(exp(A_log[h]) * (g_raw +
    dt_bias))``, then the same recurrence and ``o = S q``. The output is
    ``o * rsqrt(mean(o^2) + onorm_eps) * onorm_weight * sigmoid(onorm_g)``.

    In place, at slot s: the state after the golden token (rounded to the pool dtype), the conv window of the
    three raw inputs ending at the golden token, and for draft ``j >= 1`` its raw inputs in conv column
    ``2 + j`` and its k, g, v and beta in cache row ``j - 1``. ``num_accepted_tokens`` is read, not written:
    the runtime records the next round's count after sampling.

    The op's mode is fixed to the runtime's verify: packed token rows (no ``cu_seqlens``), slot-indexed accepted
    counts, the gated RMS norm fused, no metadata hints.

    Args:
        x_q, x_k, x_v: bf16 ``[1, B * (1 + num_spec), H, 128]`` new conv inputs; head and channel axes dense, the
            row stride free (column sections of the fused projection).
        w_q, w_k, w_v: contiguous fp32 ``[H * 128, 4]`` conv taps.
        cs_q, cs_k, cs_v: bf16 ``[slots, H * 128, >= 3 + num_spec]`` replay conv caches, dim-contiguous (stride 1
            along channels), updated in place.
        g: bf16 ``[1, T, H, 128]`` raw forget-gate logits; beta: bf16 ``[1, T, H]`` beta logits, unit head stride.
        A_log: fp32 ``[H]``; dt_bias: fp32 ``[H * 128]``.
        recurrent_state: fp32 or bf16 ``[slots, H, 128, 128]`` V-first pool, updated in place.
        k_cache, v_cache: bf16 (or fp32) ``[slots, num_spec, H * 128]``; g_cache: fp32 ``[slots, num_spec, H * 128]``;
            beta_cache: bf16 (or fp32) ``[slots, num_spec, H]``; all updated in place.
        ssm_state_indices: int32 ``[B]`` unique slots.
        num_accepted_tokens: int32 ``[>= slots]``, slot-indexed (``prev_num_accepted_tokens``), each used entry in
            ``[0, num_spec]``.
        num_spec: drafts per round (at most 7).
        lower_bound: the gate bound; scale: the q scale (``128 ** -0.5``).
        onorm_g: bf16 ``[1, T, H, 128]`` output-gate logits; onorm_weight: fp32 ``[128]``; onorm_eps: its epsilon.

    Returns:
        A new contiguous bf16 ``[1, B * (1 + num_spec), H, 128]`` output.
    """

    ARCHS = frozenset({Arch.SM_100})
    # GLM-5.3-Flash's KDA at TP4 (16 of 64 heads). The op's own edges here; the MTP-3 verify steps the shipped
    # target runs are in this entry's test. With 16 heads the packed fused-norm verify runs the cluster-pair
    # kernel while B * H <= 64, one CTA per (head, request) while B * H fits the SMs, the two-CTA-per-SM kernel
    # beyond.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one draft per round, every request accepting it: the shortest replay, cluster-pair kernel",
            spec=dict(batch=3, num_spec=1, pending=(1, 1, 1), state_dtype=torch.float32, pool=8),
        ),
        Cell(
            why="seven drafts, accepted counts 0..7: the longest chunk the kernel holds (2 * 7 + 1 tokens)",
            spec=dict(
                batch=8,
                num_spec=7,
                pending=(0, 1, 2, 3, 4, 5, 6, 7),
                state_dtype=torch.bfloat16,
                pool=12,
            ),
        ),
    )
    note = """
    Fused: conv, SiLU, q/k norm, gate and beta activation, replay, recurrence and
    the gated RMS norm (on the bf16-rounded raw output, as the kernel's epilogue
    does); projections are the caller's. The recurrence runs on tensor cores in
    3xTF32 (within about 2e-6 of fp32); gates, exponents and the norm are fp32 with
    fast exp / rcp / rsqrt. The committed state is rounded to nearest for a bf16
    pool. Conv caches hold raw bf16 inputs and are bit copies.

    State protocol: after the call each used slot holds the state as of its golden
    token, with the new drafts pending in the caches. The runtime records how many
    were accepted in num_accepted_tokens before the next verify; nothing promotes
    states on the host. The caches must be seeded from the live conv pool after a
    prefill or a plain decode advanced it, and pending drafts committed before a
    slot is released or moved.

    Each slot appears once per call. The kernel clamps a count above num_spec to
    num_spec and does not guard a negative one (is_valid refuses both).
    Every tensor operand is taken as 16-byte aligned except the int32 metadata
    and the beta cache (whose alignment the op derives); the op does not check it.
    Operands reach the kernels through DLPack, which refuses tensors that require
    grad: pass parameters detached. The first call per layout compiles (CuTe
    DSL); warm it outside CUDA-graph capture. The batch size is a runtime extent.
    """

    def __call__(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        recurrent_state: torch.Tensor,
        k_cache: torch.Tensor,
        g_cache: torch.Tensor,
        v_cache: torch.Tensor,
        beta_cache: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_spec: int,
        lower_bound: float,
        scale: float,
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
    ) -> torch.Tensor:
        return torch.ops.trtllm.kda_mtp_decode(
            x_q=x_q,
            x_k=x_k,
            x_v=x_v,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            cs_q=cs_q,
            cs_k=cs_k,
            cs_v=cs_v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            recurrent_state=recurrent_state,
            k_cache=k_cache,
            g_cache=g_cache,
            v_cache=v_cache,
            beta_cache=beta_cache,
            ssm_state_indices=ssm_state_indices,
            cu_seqlens=None,
            num_spec=num_spec,
            num_accepted_tokens=num_accepted_tokens,
            lower_bound=lower_bound,
            scale=scale,
            zero_accepted_hint=False,
            regular_metadata_hint=False,
            packed_token_layout=True,
            onorm_g=onorm_g,
            onorm_weight=onorm_weight,
            onorm_eps=onorm_eps,
            fuse_output_norm=True,
        )

    def reference(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        recurrent_state: torch.Tensor,
        k_cache: torch.Tensor,
        g_cache: torch.Tensor,
        v_cache: torch.Tensor,
        beta_cache: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_spec: int,
        lower_bound: float,
        scale: float,
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
    ) -> tuple[torch.Tensor, ...]:
        """``(out, recurrent_state, cs_q, cs_k, cs_v, k_cache, g_cache, v_cache, beta_cache)`` as the call leaves
        them; mutates nothing.

        Token by token in fp32 (elementwise products and sums, no matmul). Every store rounds once to its buffer's
        dtype; ``out`` is normalized from the fp32 raw output (the kernel rounds that to bf16 first, a
        difference of at most 2^-9 relative that the test's band covers).
        """
        heads = A_log.shape[0]
        steps = 1 + num_spec
        rate = A_log.float().exp().reshape(heads, 1)
        bias = dt_bias.float().reshape(heads, _HEAD_DIM)
        weights = [w.float() for w in (w_q, w_k, w_v)]  # [H * 128, 4]
        tokens = [x[0].float().reshape(x.shape[1], -1) for x in (x_q, x_k, x_v)]  # [T, H * 128]
        out = torch.zeros(x_v.shape[1:], dtype=torch.float32, device=x_v.device)  # [T, H, V]
        state_after = recurrent_state.clone()
        convs_after = [cs.clone() for cs in (cs_q, cs_k, cs_v)]
        caches_after = [cache.clone() for cache in (k_cache, g_cache, v_cache, beta_cache)]
        counts = num_accepted_tokens.tolist()

        def step(state, key, log_decay, value, b):
            state = state * log_decay.exp().unsqueeze(-2)
            err = (value - (state * key.unsqueeze(-2)).sum(-1)) * b.unsqueeze(-1)
            return state + err.unsqueeze(-1) * key.unsqueeze(-2)

        for n, slot in enumerate(ssm_state_indices.tolist()):
            accepted = min(counts[slot], num_spec)
            state = recurrent_state[slot].float()  # [H, V, K]
            for t in range(accepted):
                state = step(
                    state,
                    k_cache[slot, t].float().reshape(heads, _HEAD_DIM),
                    g_cache[slot, t].float().reshape(heads, _HEAD_DIM),
                    v_cache[slot, t].float().reshape(heads, _HEAD_DIM),
                    beta_cache[slot, t].float(),
                )
            # The raw conv inputs in order: the committed window, the accepted drafts, the new tokens.
            rows = slice(n * steps, (n + 1) * steps)
            raw = [
                torch.cat([cs[slot, :, : _WINDOW + accepted].float().t(), x[rows]], dim=0)
                for cs, x in zip((cs_q, cs_k, cs_v), tokens)
            ]
            for j in range(steps):
                t = accepted + j
                conv = [
                    torch.nn.functional.silu(
                        sum(seq[t + tap] * w[:, tap] for tap in range(_WINDOW + 1))
                    ).reshape(heads, _HEAD_DIM)
                    for seq, w in zip(raw, weights)
                ]
                q, key, value = conv
                q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + _QK_NORM_EPS) * scale
                key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + _QK_NORM_EPS)
                row = n * steps + j
                log_decay = lower_bound * torch.sigmoid(rate * (g[0, row].float() + bias))
                b = beta[0, row].float().sigmoid()
                state = step(state, key, log_decay, value, b)
                o = (state * q.unsqueeze(-2)).sum(-1)  # [H, V]
                o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + onorm_eps)
                out[row] = o * onorm_weight.float() * onorm_g[0, row].float().sigmoid()
                if j == 0:
                    state_after[slot] = state.to(state_after.dtype)
                    for cs_after, seq in zip(convs_after, raw):
                        cs_after[slot, :, :_WINDOW] = (
                            seq[t + 1 : t + 1 + _WINDOW].t().to(cs_after.dtype)
                        )
                else:
                    for cs_after, seq in zip(convs_after, raw):
                        cs_after[slot, :, _WINDOW + j - 1] = seq[t + _WINDOW].to(cs_after.dtype)
                    for cache, value_row in zip(caches_after, (key, log_decay, value, b)):
                        cache[slot, j - 1] = value_row.reshape(cache.shape[2:]).to(cache.dtype)
        return (out.unsqueeze(0).to(x_v.dtype), state_after, *convs_after, *caches_after)

    def is_valid(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        recurrent_state: torch.Tensor,
        k_cache: torch.Tensor,
        g_cache: torch.Tensor,
        v_cache: torch.Tensor,
        beta_cache: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_spec: int,
        lower_bound: float,
        scale: float,
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
    ) -> None:
        # The op checks shapes, dtypes, unit strides and the vectorized layouts itself; it never reads the
        # metadata on the host and takes the operand alignment on trust.
        slots = ssm_state_indices.tolist()
        assert len(set(slots)) == len(slots), (
            "ssm_state_indices must name each slot once: two requests on a slot race on its state and caches"
        )
        assert all(0 <= slot < recurrent_state.shape[0] for slot in slots), (
            "ssm_state_indices must lie in the pool: the kernel indexes every pool with them unchecked"
        )
        # A count tensor shorter than the pool is the op's own (loud) refusal; only the values are checked here.
        counts = num_accepted_tokens.tolist()
        assert all(0 <= counts[slot] <= num_spec for slot in slots if slot < len(counts)), (
            f"num_accepted_tokens of a used slot must be in [0, {num_spec}]: a larger count is clamped and "
            "a negative one shifts the request onto other requests' token rows"
        )
        operands = dict(
            x_q=x_q,
            x_k=x_k,
            x_v=x_v,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            cs_q=cs_q,
            cs_k=cs_k,
            cs_v=cs_v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            recurrent_state=recurrent_state,
            k_cache=k_cache,
            g_cache=g_cache,
            v_cache=v_cache,
            onorm_g=onorm_g,
            onorm_weight=onorm_weight,
        )
        misaligned = [name for name, t in operands.items() if t.data_ptr() % _ALIGN]
        assert not misaligned, (
            f"{misaligned} must start 16-byte aligned: the kernels assume it unchecked"
        )


kda_mtp_decode = KdaMtpDecode()
