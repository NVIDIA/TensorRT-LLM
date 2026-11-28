# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA multi-token speculative verify of one Kimi K3.1 linear-attention layer, packed rows with
the gated RMSNorm epilogue, via the family's ``kda_mtp_decode_v2.k31_kda_mtp_verify``."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.ssm import kda_mtp_decode_v2


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _kda_step(state, k, gate, v, beta):
    """One delta-rule step on a K-first ``[N, H, K, V]`` fp32 state: decay by ``exp(gate)`` along
    K, then ``S += k (x) beta * (v - S^T k)``."""
    state = state * gate.exp().unsqueeze(-1)
    predicted = (state * k.unsqueeze(-1)).sum(dim=-2)
    v_new = (v - predicted) * beta.unsqueeze(-1)
    return state + k.unsqueeze(-1) * v_new.unsqueeze(-2)


class KdaMtpVerify(OpWrapper):
    """Verify ``1 + num_spec`` new rows per request against the KDA recurrence of its state slot.

    Per request ``n`` (state slot ``s = ssm_state_indices[n]``, ``a = num_accepted_tokens[s]``
    drafts of the previous round accepted): replay the ``a`` cached drafts (their k / gate / v /
    beta from the replay caches, their raw conv inputs from the conv caches' pending columns),
    then run the request's new rows ``n * (1 + num_spec) + j``: causal conv (width 4) + SiLU of
    q / k / v, L2-normalized q (times ``scale``) and k, gate ``lower_bound * sigmoid((g + dt_bias)
    * exp(A_log))``, ``beta = sigmoid(beta)``, the delta-rule step, ``o = S^T q``. After the golden
    row ``j = 0`` the state and the conv windows are committed in place; rows ``j >= 1`` become the
    next round's pending drafts (replay caches and conv pending columns ``j - 1``). The returned
    core is the gated RMSNorm of ``bf16(o)``: ``o * rsqrt(mean(o^2) + onorm_eps) * onorm_weight *
    sigmoid(onorm_g)`` over each head's V, bf16 ``[1, T_total, H, V]``.

    Mutates ``recurrent_state`` (fp32, V-first ``[pool, H, V, K]``), ``cs_q / cs_k / cs_v``
    (bf16 ``[pool, H * 128, >= 3 + num_spec]``, dim-contiguous), ``k_cache / v_cache`` (bf16),
    ``g_cache`` (fp32) and ``beta_cache`` (bf16), all at the slots of ``ssm_state_indices`` only.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    # Kimi K3.1 KDA: 96 heads of K = V = 128, conv width 4. Heads per rank are 96 (attention TP1 /
    # attention DP), 24 (TP4), 12 (TP8) and 6 (TP16); DFlash2 verifies 4 or 7 drafts.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP16 decode, one request: the cluster-pair form (6 tiles), 7 drafts",
            spec=dict(N=1, H=6, num_spec=7),
        ),
        Cell(
            why="TP8 batch 8: 96 tiles, single-CTA form in one wave, 4 drafts",
            spec=dict(N=8, H=12, num_spec=4),
        ),
        Cell(
            why="TP4 batch 8: 192 tiles, more tiles than SMs (waves), 7 drafts",
            spec=dict(N=8, H=24, num_spec=7),
        ),
        Cell(
            why="attention DP, one request: 96 heads per rank, 4 drafts",
            spec=dict(N=1, H=96, num_spec=4),
        ),
        Cell(
            why="attention DP batch 8: 768 tiles, 7 drafts",
            spec=dict(N=8, H=96, num_spec=7),
        ),
        Cell(
            why="DEP16 bar (c512): 14 requests of DFlash2 width 5, two-CTA kernel, PDL launch",
            spec=dict(N=14, H=96, num_spec=4, pool=20, use_pdl=True),
        ),
        Cell(
            why="DEP16 bar (c512): 14 requests of DFlash2 width 8, two-CTA kernel",
            spec=dict(N=14, H=96, num_spec=7, pool=16),
        ),
        Cell(
            why="DEP16 c256 / c512 graph sizes (16 requests of width 5), two-CTA kernel loading the state evict-last",
            spec=dict(N=16, H=96, num_spec=4, pool=20, state_evict_last=True),
        ),
        Cell(
            why="TP8 batch 64: 768 tiles, 4 drafts, slots spread over a larger pool",
            spec=dict(N=64, H=12, num_spec=4, pool=96),
        ),
        Cell(
            why="sm_107 DEP16 GEN b12k7: 96 heads per rank, 1152 tiles",
            spec=dict(N=12, H=96, num_spec=7),
        ),
        Cell(
            why="sm_107 Helix CP 8 GEN b16k5: 12 heads (KDA TP8), 5 drafts",
            spec=dict(N=16, H=12, num_spec=5),
        ),
        Cell(
            why="sm_107 TEP8 GEN b1k7: 12 heads, one request, 7 drafts",
            spec=dict(N=1, H=12, num_spec=7),
        ),
    )

    note = """
    The latency-scheduled verify kernel ``kda_decode_mtp_v2_kernel``
    on the engine ``trtllm::kda_mtp_decode`` runtime contract with ``packed_token_layout=True``,
    ``fuse_output_norm=True`` and ``cu_seqlens=None``. On that contract its output, committed
    state, conv windows and replay caches are bitwise equal to the engine op's
    ``kda_decode_mtp_kernel``. Above one wave of tiles (more (head, request) tiles than SMs:
    attention DP) it runs the two-CTA-per-SM variant
    ``kda_decode_mtp_2cta_kernel`` (same bits; DEP16 bar shape 96 heads x 14 requests: 79.7 ->
    60.3 us per call on GB300). ``use_pdl`` launches that variant as a programmatic dependent
    (state, replay entries, conv windows and weights loaded before ``griddepcontrol.wait``,
    dependent released right after it); same bits. ``state_evict_last`` makes that variant load the
    state with an L2 evict-last cache policy (attention-DP perf target: the state lines then outlive
    the weight streams around the verify); same bits.

    Contract the op rejects loudly: K = V = 128, conv width 4, ``1 <= num_spec <= 7``, an fp32
    state, bf16 projections and conv caches with 16-byte-vectorizable rows, contiguous conv weights
    ``[H * 128, 4]``, fp32 gate cache, SM100 / SM103 / SM107.

    The projections may be column sections of one fused per-token projection (row stride wider
    than ``H * 128``), as the target's fused verify passes them.
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
        num_spec: int,
        num_accepted_tokens: torch.Tensor,
        lower_bound: float,
        scale: Optional[float],
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
        use_pdl: bool = False,
        state_evict_last: bool = False,
    ) -> torch.Tensor:
        return kda_mtp_decode_v2.k31_kda_mtp_verify(
            x_q,
            x_k,
            x_v,
            w_q,
            w_k,
            w_v,
            cs_q,
            cs_k,
            cs_v,
            g,
            beta,
            A_log,
            dt_bias,
            recurrent_state,
            k_cache,
            g_cache,
            v_cache,
            beta_cache,
            ssm_state_indices,
            num_spec,
            num_accepted_tokens,
            lower_bound,
            scale,
            onorm_g,
            onorm_weight,
            onorm_eps,
            use_pdl,
            state_evict_last,
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
        num_spec: int,
        num_accepted_tokens: torch.Tensor,
        lower_bound: float,
        scale: Optional[float],
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
        use_pdl: bool = False,
        state_evict_last: bool = False,
    ) -> torch.Tensor:
        """Sequential fp32 recurrence, one token step at a time, batched over requests and heads.

        Applies the op's in-place updates to its own arguments (pass clones). Elementwise
        products and sums only, so no TF32 matmul is involved.
        """
        f32 = torch.float32
        N = ssm_state_indices.shape[0]
        H = A_log.shape[0]
        K = x_q.shape[-1]
        V = x_v.shape[-1]
        W = w_q.shape[1]
        M = num_spec
        T = M + 1
        scale = K**-0.5 if scale is None else scale
        slots = ssm_state_indices.long()
        accepted = num_accepted_tokens[slots].long()

        def rows(x, d):
            return x[0].to(f32).reshape(N, T, H, d)

        xq, xk, xv = rows(x_q, K), rows(x_k, K), rows(x_v, V)
        gate_in = rows(g, K)
        beta_in = beta[0].to(f32).reshape(N, T, H)
        onorm_in = rows(onorm_g, V)

        def window(cs, d):
            return cs[slots].to(f32).reshape(N, H, d, cs.shape[-1])

        win = {"q": window(cs_q, K), "k": window(cs_k, K), "v": window(cs_v, V)}
        weights = {
            "q": w_q.to(f32).reshape(H, K, W),
            "k": w_k.to(f32).reshape(H, K, W),
            "v": w_v.to(f32).reshape(H, V, W),
        }
        hist = {name: w[..., : W - 1] for name, w in win.items()}

        kc = k_cache[slots].to(f32).reshape(N, M, H, K)
        gc = g_cache[slots].to(f32).reshape(N, M, H, K)
        vc = v_cache[slots].to(f32).reshape(N, M, H, V)
        bc = beta_cache[slots][..., :H].to(f32)
        state = recurrent_state[slots].to(f32).transpose(-1, -2)  # K-first [N, H, K, V]

        def shift(h, raw):
            return torch.cat([h[..., 1:], raw.unsqueeze(-1)], dim=-1)

        # Replay the accepted drafts of the previous round (no output).
        for i in range(M):
            active = (accepted > i).view(N, 1, 1, 1)
            stepped = _kda_step(state, kc[:, i], gc[:, i], vc[:, i], bc[:, i])
            state = torch.where(active, stepped, state)
            for name in hist:
                shifted = shift(hist[name], win[name][..., W - 1 + i])
                hist[name] = torch.where(active, shifted, hist[name])

        decay_scale = A_log.to(f32).exp().view(H, 1)
        bias = dt_bias.to(f32).view(H, K)
        out = torch.empty(N, T, H, V, dtype=f32, device=x_q.device)
        new_k, new_g, new_v, new_b = [], [], [], []
        committed_state = committed_hist = None
        for j in range(T):
            raw = {"q": xq[:, j], "k": xk[:, j], "v": xv[:, j]}
            conv = {
                name: (
                    torch.cat([hist[name], raw[name].unsqueeze(-1)], dim=-1) * weights[name]
                ).sum(-1)
                for name in raw
            }
            q_t = torch.nn.functional.normalize(_silu(conv["q"]), p=2, dim=-1) * scale
            k_t = torch.nn.functional.normalize(_silu(conv["k"]), p=2, dim=-1)
            v_t = _silu(conv["v"])
            gk_t = lower_bound * torch.sigmoid((gate_in[:, j] + bias) * decay_scale)
            b_t = torch.sigmoid(beta_in[:, j])
            state = _kda_step(state, k_t, gk_t, v_t, b_t)
            out[:, j] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
            for name in hist:
                hist[name] = shift(hist[name], raw[name])
            if j == 0:
                committed_state = state.clone()
                committed_hist = {name: h.clone() for name, h in hist.items()}
            else:
                new_k.append(k_t)
                new_g.append(gk_t)
                new_v.append(v_t)
                new_b.append(b_t)

        # In-place commits at the requests' slots.
        recurrent_state[slots] = committed_state.transpose(-1, -2).to(recurrent_state.dtype)
        for name, cs, d in (("q", cs_q, K), ("k", cs_k, K), ("v", cs_v, V)):
            cs[slots, :, : W - 1] = committed_hist[name].reshape(N, H * d, W - 1).to(cs.dtype)
            pending = (
                torch.stack([{"q": xq, "k": xk, "v": xv}[name][:, j] for j in range(1, T)], dim=-1)
                .reshape(N, H * d, M)
                .to(cs.dtype)
            )
            cs[slots, :, W - 1 : W - 1 + M] = pending
        k_cache[slots, :M] = torch.stack(new_k, dim=1).reshape(N, M, H * K).to(k_cache.dtype)
        g_cache[slots, :M] = torch.stack(new_g, dim=1).reshape(N, M, H * K).to(g_cache.dtype)
        v_cache[slots, :M] = torch.stack(new_v, dim=1).reshape(N, M, H * V).to(v_cache.dtype)
        beta_rows = beta_cache[slots]
        beta_rows[..., :H] = torch.stack(new_b, dim=1).to(beta_cache.dtype)
        beta_cache[slots] = beta_rows

        # Gated RMSNorm of the bf16-rounded core, per head over V.
        core = out.to(torch.bfloat16).to(f32)
        normed = core * torch.rsqrt(core.pow(2).mean(dim=-1, keepdim=True) + onorm_eps)
        normed = normed * onorm_weight.to(f32) * torch.sigmoid(onorm_in)
        return normed.to(torch.bfloat16).reshape(1, N * T, H, V)

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
        num_spec: int,
        num_accepted_tokens: torch.Tensor,
        lower_bound: float,
        scale: Optional[float],
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
        use_pdl: bool = False,
        state_evict_last: bool = False,
    ) -> None:
        # Device-side values the kernel trusts: a slot outside the pool or a repeated slot is read
        # and committed without a check, and an accepted count above num_spec replays rows past the
        # replay caches' last pending draft.
        pool = recurrent_state.shape[0]
        slots = ssm_state_indices.long()
        assert bool(((slots >= 0) & (slots < pool)).all()), (
            "ssm_state_indices must lie in [0, pool); an out-of-range slot is read and committed "
            "out of bounds"
        )
        assert torch.unique(slots).numel() == slots.numel(), (
            "ssm_state_indices must be distinct; two requests on one slot race on the in-place "
            "state commit"
        )
        accepted = num_accepted_tokens[slots]
        assert bool(((accepted >= 0) & (accepted <= num_spec)).all()), (
            f"num_accepted_tokens at the requests' slots must lie in [0, {num_spec}]; a larger count "
            "replays rows past the cached drafts"
        )


kda_mtp_verify = KdaMtpVerify()
