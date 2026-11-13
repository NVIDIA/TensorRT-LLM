# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA speculative verify of N requests of 1 + num_spec tokens from the manager's replay caches: replay each
request's accepted drafts, then the new tokens, the gated-norm core output, committing after the golden token."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops  # noqa: F401 — registers the op
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class KdaMtpDecode(OpWrapper):
    """Return the gated-norm core bf16 [1, N T, H, 128] of N requests of T = 1 + ``num_spec`` packed tokens.

    Request n (state slot ``s = ssm_state_indices[n]``, replay slot ``r = replay_state_indices[n]``, ``s`` when
    ``replay_state_indices`` is None) first replays its ``p = num_accepted_tokens[r]`` accepted drafts of the
    previous round from the replay caches (no output), then runs its T new tokens from rows ``n T ..``: causal conv
    over the window ``cs_*[r][:, p : p + 3]`` + SiLU, q / k L2 norm (q times ``scale``), the beta sigmoid, the
    lower-bound gate, the delta rule, ``rmsnorm(o) * onorm_weight * sigmoid(onorm_g)``. After the golden token the
    state ``recurrent_state[s]`` and the conv window of ``cs_*[r]`` (columns 0..2) are committed in place; the T - 1
    drafts' raw inputs go to columns 3.. of ``cs_*[r]``, their normalized k, log decay, v and sigmoid beta to row
    ``r`` of the k / g / v / beta caches.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one TP16 request, seven drafts, accepted counts cycling over rounds",
            spec=dict(requests=1, num_spec=7, rounds=4),
        ),
        Cell(
            why="four requests with different accepted counts, three drafts",
            spec=dict(requests=4, num_spec=3, rounds=4),
        ),
        Cell(
            why="eight requests at seven drafts (64 rows, the projection's multi-stream ceiling below 128)",
            spec=dict(requests=8, num_spec=7, rounds=3),
        ),
        Cell(
            why="32 requests at seven drafts: more (head, request) tiles than one wave (two-CTA kernel)",
            spec=dict(requests=32, num_spec=7, rounds=2),
        ),
        Cell(
            why="four requests whose replay slot differs from their state slot in every row, three drafts",
            spec=dict(requests=4, num_spec=3, rounds=3, distinct_replay_slots=True),
        ),
    )
    note = """
    Kimi K3's SA (or any speculation whose drafts kda_mtp_decode supports
    without the K3 per-draft records) verify. The call form is fixed:
    packed_token_layout=True, cu_seqlens=None, fuse_output_norm=True. x_q /
    x_k / x_v / onorm_g are strided [1, N T, H, 128] bf16 views of the fused
    [q | k | v | g] projection, g the bf16 f_b gate [1, N T, H, 128], beta a
    strided bf16 [1, N T, H] view of the [f_a | b] projection (raw logits).
    w_* fp32 [H 128, 4]; A_log [H] and dt_bias [H 128] the module's bf16
    parameters; onorm_weight fp32 [128]; lower_bound -5, scale 128 ** -0.5.

    Pools are the manager's. recurrent_state fp32 V-first [slots, H, 128, 128]
    (rows strided) is indexed by ssm_state_indices, int32 [N], 16-byte
    aligned. The replay caches are indexed by replay_state_indices, int32 [N]:
    cs_* bf16 [replay slots, H 128, 3 + num_spec] dim-contiguous, k / v
    caches bf16 and g cache fp32 [replay slots, num_spec, H 128], beta cache
    bf16 [replay slots, num_spec, H] in head-padded rows, and
    num_accepted_tokens, the int32 prev_num_accepted_tokens (clamped to
    [0, num_spec]). Without the drafts' records the manager keeps the replay
    caches in a pool of their own, so the replay slots differ from the state
    slots; replay_state_indices None indexes both by ssm_state_indices. The
    slots of each index must be unique.

    The kernel recurrence is 3xTF32; the selected kernel (one-wave, two-CTA)
    is bitwise the same. Output band 2e-2, state band 1e-3 against fp32.
    num_spec <= 7 (MAX_NUM_SPEC); above it the manager keeps the sequential
    verify's buffers instead. Requires CuTe DSL.
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
        cu_seqlens: Optional[torch.Tensor],
        num_spec: int,
        num_accepted_tokens: torch.Tensor,
        lower_bound: float,
        scale: float,
        packed_token_layout: bool,
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
        fuse_output_norm: bool,
        replay_state_indices: Optional[torch.Tensor] = None,
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
            cu_seqlens=cu_seqlens,
            num_spec=num_spec,
            num_accepted_tokens=num_accepted_tokens,
            lower_bound=lower_bound,
            scale=scale,
            packed_token_layout=packed_token_layout,
            onorm_g=onorm_g,
            onorm_weight=onorm_weight,
            onorm_eps=onorm_eps,
            fuse_output_norm=fuse_output_norm,
            replay_state_indices=replay_state_indices,
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
        cu_seqlens: Optional[torch.Tensor],
        num_spec: int,
        num_accepted_tokens: torch.Tensor,
        lower_bound: float,
        scale: float,
        packed_token_layout: bool,
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
        fuse_output_norm: bool,
        replay_state_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """fp32 per request, in the kernel's order; writes every pool the op writes. Only the packed, fused-norm
        call form (``cu_seqlens`` is None)."""
        heads, dim = x_q.shape[-2:]
        width = w_q.shape[1]
        steps = num_spec + 1
        wq, wk, wv = (w.float().view(heads, dim, width) for w in (w_q, w_k, w_v))
        rate = A_log.float().exp()  # [H]
        bias = dt_bias.float().view(heads, dim)
        out = torch.empty(x_v.shape, dtype=torch.float32, device=x_v.device)

        def conv_silu(hist, raw, w):  # hist [H, D, W - 1], raw [H, D]
            full = torch.cat([hist, raw.unsqueeze(-1)], dim=-1)
            return torch.nn.functional.silu((full * w).sum(-1)), full[..., 1:]

        def step(
            state, k, log_decay, v, b
        ):  # state [H, V, K]; k / log_decay [H, K]; v [H, V]; b [H]
            state = state * log_decay.exp().unsqueeze(1)
            err = (v - (state * k.unsqueeze(1)).sum(-1)) * b.unsqueeze(-1)
            return state + err.unsqueeze(-1) * k.unsqueeze(1)

        replay_slots = ssm_state_indices if replay_state_indices is None else replay_state_indices
        for n, (slot, rslot) in enumerate(zip(ssm_state_indices.tolist(), replay_slots.tolist())):
            pending = min(max(int(num_accepted_tokens[rslot]), 0), num_spec)
            caches = [c[rslot].float().view(heads, dim, -1) for c in (cs_q, cs_k, cs_v)]
            hist = [c[..., : width - 1] for c in caches]
            state = recurrent_state[slot].float()
            for i in range(pending):
                state = step(
                    state,
                    k_cache[rslot, i].float().view(heads, dim),
                    g_cache[rslot, i].float().view(heads, dim),
                    v_cache[rslot, i].float().view(heads, dim),
                    beta_cache[rslot, i, :heads].float(),
                )
                hist = [
                    torch.cat([h[..., 1:], c[..., width - 1 + i].unsqueeze(-1)], dim=-1)
                    for h, c in zip(hist, caches)
                ]
            drafts = []
            for t in range(steps):
                row = n * steps + t
                raws = [x[0, row].float() for x in (x_q, x_k, x_v)]
                (q, hq), (k, hk), (v, hv) = (
                    conv_silu(h, r, w) for h, r, w in zip(hist, raws, (wq, wk, wv))
                )
                hist = [hq, hk, hv]
                q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * scale
                k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
                log_decay = lower_bound * torch.sigmoid(
                    rate.unsqueeze(-1) * (g[0, row].float() + bias)
                )
                b = beta[0, row].float().sigmoid()
                state = step(state, k, log_decay, v, b)
                o = (state * q.unsqueeze(1)).sum(-1)
                rms = torch.rsqrt(o.square().mean(-1, keepdim=True) + onorm_eps)
                out[0, row] = o * rms * onorm_weight.float() * onorm_g[0, row].float().sigmoid()
                if t == 0:
                    recurrent_state[slot] = state.to(recurrent_state.dtype)
                    for c, h in zip((cs_q, cs_k, cs_v), hist):
                        c[rslot, :, : width - 1] = h.reshape(heads * dim, width - 1).to(c.dtype)
                else:
                    drafts.append((raws, k, log_decay, v, b))
            for i, (raws, k, log_decay, v, b) in enumerate(drafts):
                for c, r in zip((cs_q, cs_k, cs_v), raws):
                    c[rslot, :, width - 1 + i] = r.reshape(-1).to(c.dtype)
                k_cache[rslot, i] = k.reshape(-1).to(k_cache.dtype)
                g_cache[rslot, i] = log_decay.reshape(-1).to(g_cache.dtype)
                v_cache[rslot, i] = v.reshape(-1).to(v_cache.dtype)
                beta_cache[rslot, i, :heads] = b.to(beta_cache.dtype)
        return out.to(x_v.dtype)

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
        cu_seqlens: Optional[torch.Tensor],
        num_spec: int,
        num_accepted_tokens: torch.Tensor,
        lower_bound: float,
        scale: float,
        packed_token_layout: bool,
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
        fuse_output_norm: bool,
        replay_state_indices: Optional[torch.Tensor] = None,
    ) -> None:
        assert packed_token_layout and cu_seqlens is None and fuse_output_norm, (
            "only the packed, fused-norm call form is certified"
        )
        slots = ssm_state_indices.tolist()
        assert len(set(slots)) == len(slots), (
            "ssm_state_indices must name one unique slot per request"
        )
        if replay_state_indices is not None:
            replay_slots = replay_state_indices.tolist()
            assert len(replay_slots) == len(slots), (
                "replay_state_indices must name one replay slot per request"
            )
            assert len(set(replay_slots)) == len(replay_slots), (
                "replay_state_indices must name one unique slot per request"
            )
        assert x_q.shape[1] == len(slots) * (num_spec + 1), (
            "rows must be N (1 + num_spec) packed tokens"
        )
        assert cs_q.shape[-1] == w_q.shape[1] - 1 + num_spec, (
            "conv caches must hold W - 1 + num_spec columns"
        )


kda_mtp_decode = KdaMtpDecode()
