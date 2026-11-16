# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's KDA speculative verify of N requests of 1 + num_spec tokens, from the fused projection rows to the
gated-norm core output, committing the golden token's state and every draft's records."""

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_kda_verify import (  # noqa: F401 — registers the op
    op as _k3_kda_verify_op,
)
from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_kda_mtp_ops  # noqa: F401


class K3KdaVerify(OpWrapper):
    """Return the gated-norm core output bf16 [N (1 + num_spec), H, 128] of the verify tokens whose fused
    projection rows are ``proj``; updates the slots' conv caches, state and per-draft states in place."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (
        Cell(
            why="one TP16 request, seven drafts and six rounds of accepted-draft replay",
            spec=dict(requests=1, num_spec=7, layers=2, rounds=6),
        ),
        Cell(
            why="four requests with independently changing acceptance counts, seven drafts",
            spec=dict(requests=4, num_spec=7, layers=2, rounds=6),
        ),
        Cell(
            why="eight requests at the manager's maximum batch, seven drafts",
            spec=dict(requests=8, num_spec=7, layers=2, rounds=6),
        ),
        Cell(
            why="four requests with a three-draft manager and shorter replay windows",
            spec=dict(requests=4, num_spec=3, layers=2, rounds=6),
        ),
    )
    note = """
    K = V = 128, conv width 4; the Kimi K3 TP16 slice has H = 6. Projection
    rows are bf16 [N (1 + num_spec), cols], padded to a multiple of 8, ordered
    [q | k | v | og | f_a | b | pad]. The folded f_b gate is rounded to bf16;
    g_ext, contiguous bf16 [N (1 + num_spec), H K], supplies that gate instead.
    With g_ext the kernel is bit-exact with
    kda_mtp_decode on the same gate and accepted-draft records. The grid is
    (H, N, 8), with each CTA owning 16 V rows.

    slots and pending, the initial state and the conv window are read before
    the grid-dependency wait. Metadata must be written before a CUDA graph
    runs, never by a kernel of the same step ahead of this launch. Dependents
    are released before this wait, so successive launches on the same pools
    need an intervening kernel that waits, or a non-PDL kernel. Another verify
    launch on other pools is not enough. Round order cannot be checked from
    one call: the swapped-round negative control silently gives another
    history (sm_100 relative output/state differences 0.98/0.98).

    First use of each H, num_spec, lower_bound, scale, eps, g_ext mode and PDL
    configuration compiles outside graph capture; capture raises RuntimeError.
    slots and pending may start at any element offset (0..3 are certified).
    Pool offsets are 64-bit, including pools beyond 2 GiB, covered by
    test_k3_kda_pools_past_2g.py. Repeated schedules and graph replays are
    bit-identical; stateful tests drive changing acceptance through the next
    round, compare committed histories, and preserve other slots.
    k3_kda_attn fuses this verify with projection for one request of eight tokens.
    """

    def __call__(
        self,
        proj: torch.Tensor,
        w_fb: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        onorm_w: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        ssm: torch.Tensor,
        state_tok: torch.Tensor,
        slots: torch.Tensor,
        pending: torch.Tensor,
        num_spec: int,
        lower_bound: float,
        scale: float,
        eps: float,
        g_ext: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_kda_verify(
            proj,
            w_fb,
            w_q,
            w_k,
            w_v,
            a_log,
            dt_bias,
            onorm_w,
            cs_q,
            cs_k,
            cs_v,
            ssm,
            state_tok,
            slots,
            pending,
            num_spec,
            lower_bound,
            scale,
            eps,
            g_ext,
        )

    def reference(
        self,
        proj: torch.Tensor,
        w_fb: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        onorm_w: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        ssm: torch.Tensor,
        state_tok: torch.Tensor,
        slots: torch.Tensor,
        pending: torch.Tensor,
        num_spec: int,
        lower_bound: float,
        scale: float,
        eps: float,
        g_ext: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        heads, width = ssm.shape[1], ssm.shape[-1]
        hk, steps = heads * width, num_spec + 1
        rows = proj.float()
        with true_fp32_matmul():
            gate = (
                (rows[:, 4 * hk : 4 * hk + width] @ w_fb.float().t()).to(proj.dtype).float()
                if g_ext is None
                else g_ext.float()
            )
        weights = torch.stack((w_q, w_k, w_v)).float()
        out = torch.empty(rows.shape[0], heads, width, dtype=torch.float32, device=proj.device)
        for n, slot in enumerate(slots.tolist()):
            accepted = int(pending[slot].item())
            state = ssm[slot].float().clone()
            for draft in range(accepted):
                vn, beta_k, decay = state_tok[slot, :, draft].float().unbind(0)
                state = state * decay[:, None, :] + vn[:, :, None] * beta_k[:, None, :]
            window = torch.stack(
                [cache[slot, :, accepted : accepted + 3] for cache in (cs_q, cs_k, cs_v)]
            ).float()
            raw_rows = rows[n * steps : (n + 1) * steps, : 3 * hk].reshape(steps, 3, hk)
            history = torch.cat((window[:, :, 1:], raw_rows.permute(1, 2, 0)), dim=-1)
            for token in range(steps):
                row = n * steps + token
                u = torch.cat((window, raw_rows[token, :, :, None]), dim=-1)
                act = (u * weights).sum(-1)
                act = act * torch.sigmoid(act)
                q, k, v = act.reshape(3, heads, width).unbind(0)
                q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
                k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
                beta = torch.sigmoid(rows[row, 4 * hk + width : 4 * hk + width + heads])
                xg = torch.exp(a_log.float())[:, None] * (
                    gate[row].reshape(heads, width) + dt_bias.float().reshape(heads, width)
                )
                decay = torch.exp(lower_bound * torch.sigmoid(xg))
                sd = state * decay[:, None, :]
                vn = v - (sd * k[:, None, :]).sum(-1)
                beta_k = beta[:, None] * k
                state = sd + vn[:, :, None] * beta_k[:, None, :]
                o = (state * q[:, None, :]).sum(-1)
                rms = torch.rsqrt((o * o).mean(-1, keepdim=True) + eps)
                output_gate = torch.sigmoid(rows[row, 3 * hk : 4 * hk].reshape(heads, width))
                out[row] = o * rms * onorm_w.float() * output_gate
                if token == 0:
                    ssm[slot].copy_(state)
                else:
                    state_tok[slot, :, token - 1].copy_(torch.stack((vn, beta_k, decay)))
                window = u[:, :, 1:]
            for channel, cache in enumerate((cs_q, cs_k, cs_v)):
                cache[slot].copy_(history[channel])
        return out.to(proj.dtype)

    def is_valid(
        self,
        proj: torch.Tensor,
        w_fb: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        onorm_w: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        ssm: torch.Tensor,
        state_tok: torch.Tensor,
        slots: torch.Tensor,
        pending: torch.Tensor,
        num_spec: int,
        lower_bound: float,
        scale: float,
        eps: float,
        g_ext: Optional[torch.Tensor] = None,
    ) -> None:
        heads, width = ssm.shape[1], ssm.shape[-1]
        hk = heads * width
        for weight in (w_q, w_k, w_v):
            assert weight.shape == (hk, 4), "conv taps must be [H K, 4], oldest first"
            if weight.t().is_contiguous():
                assert weight.is_contiguous(), (
                    "conv taps must store each channel's four taps together"
                )
        assert a_log.shape == (heads,), "a_log must have one value per head"
        assert dt_bias.shape == (hk,), "dt_bias must have one value per key channel"
        for cache in (cs_q, cs_k, cs_v):
            assert cache.shape[1] == hk, "conv caches must cover every channel"
            if cache.is_contiguous():
                assert cache.stride(-2) == 1, "conv caches must be channel-contiguous"
        for cache in (cs_k, cs_v):
            assert cache.shape[-1] == 3 + num_spec, "conv caches must cover every replay column"
        assert slots.ndim == 1 and slots.is_contiguous(), "slots must be a contiguous vector"
        assert pending.ndim == 1 and pending.stride(0) > 0, "pending must be a strided vector"
        indices = slots.long()
        assert indices.unique().numel() == indices.numel(), "a call must name each slot once"
        pool_size = min(
            ssm.shape[0],
            state_tok.shape[0],
            cs_q.shape[0],
            cs_k.shape[0],
            cs_v.shape[0],
            pending.numel(),
        )
        assert bool(((indices >= 0) & (indices < pool_size)).all()), (
            "slots must be inside every pool and pending"
        )
        accepted = pending[indices]
        assert bool(((accepted >= 0) & (accepted <= num_spec)).all()), (
            "pending must be in 0..num_spec"
        )
        if g_ext is not None:
            assert g_ext.shape == (proj.shape[0], hk), (
                "g_ext must provide every token's gate channels"
            )


k3_kda_verify = K3KdaVerify()
