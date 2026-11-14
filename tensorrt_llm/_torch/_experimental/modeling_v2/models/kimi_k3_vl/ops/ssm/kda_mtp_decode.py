# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused KDA multi-token verify (CuTe DSL): replay of the accepted drafts, short convolution,
gates, delta rule over the new tokens, and the gated output norm, committing the state in place."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_kda import mtp_ops

HEAD_DIM = 128  # K = V of the kernel
CONV_WIDTH = 4
MXFP8_BLOCK = 128  # elements per UE8M0 scale of the MXFP8 output
FP8_MAX = 448.0

# The deployments' verify shapes: (target, KDA heads per rank, requests, drafts).
_SHAPES = (
    ("helix8", 12, 16, 5, "c240 generation: 16 requests with five drafts"),
    ("helix8", 12, 8, 7, "c75 / c174 generation: 8 requests with seven drafts"),
    ("helix8", 12, 2, 7, "c24 / c42 generation: 2 requests with seven drafts"),
    ("helix16_moetp4ep4", 6, 1, 7, "ULL c1: one request with seven drafts"),
    ("helix16_moetp4ep4", 6, 4, 7, "ULL c4: 4 requests with seven drafts"),
    ("dep16", 96, 16, 5, "c383 generation: all 96 heads, 16 requests with five drafts"),
)


def _variants(heads: int) -> tuple[tuple[str, str], ...]:
    """The verify variants a layer of ``heads`` heads runs (the f_a gate needs at most 16)."""
    found = [
        ("bf16", "BF16 core after the fab / F.linear gate, PDL launch"),
        ("mxfp8", "MXFP8 core for the prequantized o_proj"),
    ]
    if heads <= 16:
        found.append(("fused_fb", "forget gate computed from f_a in the kernel, PDL launch"))
    return tuple(found)


def _ceil_log2_exponent(x: torch.Tensor) -> torch.Tensor:
    """``ceil(log2(x))`` of positive fp32 ``x``, exactly (the UE8M0 round-up)."""
    mantissa, exponent = torch.frexp(x)
    return exponent - (mantissa == 0.5).to(exponent.dtype)


def _mxfp8_rows(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 ``[rows, heads * V]`` and the flat R128c4 UE8M0 scales of bf16 ``y`` ``[rows, heads,
    V]``: one power-of-two scale per (row, head), ``2 ** ceil(log2(amax / 448))``, replicated four
    times. Scale bytes of the 128-row block's padding rows are zero here (never written)."""
    rows, heads, width = y.shape
    yf = y.float()
    amax = yf.abs().amax(dim=-1).clamp_min(1e-10)
    exponent = _ceil_log2_exponent(amax * (1.0 / FP8_MAX))
    quantized = (yf * torch.exp2(-exponent.float())[..., None]).to(torch.float8_e4m3fn)
    padded_rows = -(-rows // MXFP8_BLOCK) * MXFP8_BLOCK
    scales = torch.zeros(padded_rows * heads * 4, dtype=torch.uint8, device=y.device)
    row = torch.arange(rows, device=y.device)[:, None]
    head = torch.arange(heads, device=y.device)[None, :]
    base = (
        (row // MXFP8_BLOCK) * heads * 512 + head * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4
    )
    code = (exponent + 127).to(torch.uint8)
    for replica in range(4):
        scales[(base + replica).flatten()] = code.flatten()
    return quantized.reshape(rows, heads * width), scales


class KdaMtpDecode(OpWrapper):
    """Verify ``1 + num_spec`` new tokens per request against its state slot; return the core.

    Request ``n`` owns slot ``ssm_state_indices[n]``. It first replays its ``a`` accepted drafts
    of the previous round from the replay caches (``k_cache`` / ``g_cache`` / ``v_cache`` /
    ``beta_cache``, the conv inputs in ``cs_*[..., W - 1:]``), then runs the new tokens: depthwise
    convolution over the conv window, SiLU, Q/K L2 norm (Q scaled by ``scale``), the gate
    ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` (``g`` the raw gate, or
    ``bf16(fa @ w_fb^T)`` when ``fa`` / ``w_fb`` are given), ``sigmoid(beta)`` and the delta rule.
    With ``packed_token_layout`` the new tokens are rows ``n * (1 + num_spec) + j`` and ``a`` is
    ``num_accepted_tokens[slot]``; otherwise they start at ``cu_seqlens[n] + a`` with
    ``a = num_accepted_tokens[n]``.

    Returns the core ``[1, T, H, V]`` (bf16, ``rmsnorm(bf16(o)) * onorm_weight *
    sigmoid(onorm_g)`` with ``fuse_output_norm``), or with ``quantize_output`` that normalized
    core as E4M3 ``[T, H * V]`` and its UE8M0 1x128 scales in the flat R128c4 layout.

    In place: after the first new token the slot's state goes to ``recurrent_state`` and its
    conv window to ``cs_*[..., :W - 1]``; the other new tokens are cached as the next round's
    drafts (``k`` / ``g`` / ``v`` / ``beta`` post-activation, raw conv inputs in ``cs_*``).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{target}, {heads} heads: {why}; {variant_why}",
            spec=dict(heads=heads, requests=requests, num_spec=num_spec, variant=variant),
        )
        for target, heads, requests, num_spec, why in _SHAPES
        for variant, variant_why in _variants(heads)
    )
    note = """
    Kimi K3 calls it from the KDA layer's verify (K3KDAKernelDispatch.mtp_verify /
    mtp_verify_fused_fb) with packed_token_layout, fuse_output_norm, K = V = 128, conv width 4,
    lower_bound -5, scale 128**-0.5, onorm_eps 1e-5 and the cache manager's pools: fp32 V-first
    recurrent_state, dim-contiguous bf16 conv caches [slots, H * 128, 3 + num_spec], replay caches
    k / v / beta bf16 and g fp32, int32 slot-indexed accepted counts. x_q / x_k / x_v / onorm_g
    are row-strided sections of the QKVG output, beta (and fa) of the [f_a | b] output. The
    variants: fa / w_fb (the forget gate computed in the kernel, BF16 core, use_pdl), BF16 core
    (use_pdl), quantize_output (MXFP8 core, no PDL). The legacy cu_seqlens layout and the
    zero_accepted / regular_metadata hints are accepted but only K3's packed form is certified.

    Kernel selection is per call: the v2 kernel for the packed fused-norm BF16 contract (unless
    TRTLLM_K3_KDA_MTP_V2=0), on SM107 the two-CTA kernel for the MXFP8 output, else the v1
    kernel (SM107: its compact layout on multi-wave grids with at most three drafts); all compute
    the same arithmetic. TRTLLM_KDA_MTP_BF16_MATH=1 switches the recurrence to bf16 operands.

    A process-wide cache holds the compiled kernels, keyed by layouts and constexpr flags; warm
    each key before graph capture. Outputs are new tensors. Launches go to the current stream;
    with use_pdl the kernel loads state, replay entries and weights before griddepcontrol.wait.
    Importing the entry needs CuTe DSL.
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
        g: Optional[torch.Tensor],
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
        scale: Optional[float] = None,
        zero_accepted_hint: bool = False,
        regular_metadata_hint: bool = False,
        packed_token_layout: bool = False,
        onorm_g: Optional[torch.Tensor] = None,
        onorm_weight: Optional[torch.Tensor] = None,
        onorm_eps: float = 1e-5,
        fuse_output_norm: bool = False,
        quantize_output: bool = False,
        fa: Optional[torch.Tensor] = None,
        w_fb: Optional[torch.Tensor] = None,
        use_pdl: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return mtp_ops.kda_mtp_decode_impl(
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
            zero_accepted_hint=zero_accepted_hint,
            regular_metadata_hint=regular_metadata_hint,
            packed_token_layout=packed_token_layout,
            onorm_g=onorm_g,
            onorm_weight=onorm_weight,
            onorm_eps=onorm_eps,
            fuse_output_norm=fuse_output_norm,
            quantize_output=quantize_output,
            fa=fa,
            w_fb=w_fb,
            use_pdl=use_pdl,
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
        g: Optional[torch.Tensor],
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
        scale: Optional[float] = None,
        zero_accepted_hint: bool = False,
        regular_metadata_hint: bool = False,
        packed_token_layout: bool = False,
        onorm_g: Optional[torch.Tensor] = None,
        onorm_weight: Optional[torch.Tensor] = None,
        onorm_eps: float = 1e-5,
        fuse_output_norm: bool = False,
        quantize_output: bool = False,
        fa: Optional[torch.Tensor] = None,
        w_fb: Optional[torch.Tensor] = None,
        use_pdl: bool = False,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        """fp64 token-by-token verify: ``(output, post)``, nothing written.

        ``output`` is what the op returns; ``post`` maps ``cs_q`` / ``cs_k`` / ``cs_v`` /
        ``recurrent_state`` / ``k_cache`` / ``g_cache`` / ``v_cache`` / ``beta_cache`` to their
        expected contents after the call. Rows the op does not write are zero (the legacy
        layout's replay holes; the scale bytes of the padding rows). The hints and ``use_pdl``
        do not enter the result.
        """
        f64 = torch.float64
        heads = A_log.shape[0]
        key_dim, value_dim = x_q.shape[-1], x_v.shape[-1]
        width = w_q.shape[1]
        rows = x_q.shape[1]
        scale = key_dim**-0.5 if scale is None else scale
        names = ("cs_q", "cs_k", "cs_v", "recurrent_state", "k_cache", "g_cache", "v_cache")
        sources = (cs_q, cs_k, cs_v, recurrent_state, k_cache, g_cache, v_cache)
        post = {name: t.clone() for name, t in zip(names, sources)}
        post["beta_cache"] = beta_cache.clone()
        convs = (cs_q, cs_k, cs_v)
        weights = tuple(w.double() for w in (w_q, w_k, w_v))
        inputs = tuple(x[0].reshape(rows, -1) for x in (x_q, x_k, x_v))
        if fa is not None:
            raw_gate = (fa[0].double() @ w_fb.double().t()).to(torch.bfloat16).double()
        else:
            raw_gate = g[0].reshape(rows, -1).double()
        raw_gate = raw_gate.view(rows, heads, key_dim) + dt_bias.double().view(heads, key_dim)
        rate = A_log.double().exp()[:, None]
        out = torch.zeros(rows, heads, value_dim, dtype=f64, device=x_q.device)

        def delta_rule(s, k_t, g_t, v_t, b_t):
            s = s * g_t.exp()[:, None, :]
            err = v_t - torch.einsum("hvk,hk->hv", s, k_t)
            return s + b_t[:, None, None] * err[:, :, None] * k_t[:, None, :]

        slots = ssm_state_indices.tolist()
        accepted = num_accepted_tokens.tolist()
        starts = None if cu_seqlens is None else cu_seqlens.tolist()
        for n, slot in enumerate(slots):
            if packed_token_layout:
                drafts, first = accepted[slot], n * (1 + num_spec)
            else:
                drafts = accepted[n]
                first = starts[n] + drafts
            window = [c[slot, :, : width - 1].double() for c in convs]
            s = recurrent_state[slot].double()
            for i in range(drafts):
                window = [
                    torch.cat([w[:, 1:], c[slot, :, width - 1 + i].double()[:, None]], dim=1)
                    for w, c in zip(window, convs)
                ]
                s = delta_rule(
                    s,
                    k_cache[slot, i].double().view(heads, key_dim),
                    g_cache[slot, i].double().view(heads, key_dim),
                    v_cache[slot, i].double().view(heads, value_dim),
                    beta_cache[slot, i].double().view(heads),
                )
            for j in range(1 + num_spec):
                row = first + j
                raw = [x[row].double() for x in inputs]
                window = [torch.cat([w, r[:, None]], dim=1) for w, r in zip(window, raw)]
                act = [(w * wt).sum(-1) for w, wt in zip(window, weights)]
                act = [a * torch.sigmoid(a) for a in act]
                window = [w[:, 1:] for w in window]
                q_t = torch.nn.functional.normalize(act[0].view(heads, key_dim), dim=-1) * scale
                k_t = torch.nn.functional.normalize(act[1].view(heads, key_dim), dim=-1)
                v_t = act[2].view(heads, value_dim)
                g_t = lower_bound * torch.sigmoid(rate * raw_gate[row])
                b_t = torch.sigmoid(beta[0, row].double())
                s = delta_rule(s, k_t, g_t, v_t, b_t)
                out[row] = torch.einsum("hvk,hk->hv", s, q_t)
                if j == 0:
                    post["recurrent_state"][slot] = s.to(recurrent_state.dtype)
                    for name, w in zip(("cs_q", "cs_k", "cs_v"), window):
                        post[name][slot, :, : width - 1] = w.to(cs_q.dtype)
                    continue
                draft = j - 1
                post["k_cache"][slot, draft] = k_t.reshape(k_cache[slot, draft].shape)
                post["g_cache"][slot, draft] = g_t.reshape(g_cache[slot, draft].shape)
                post["v_cache"][slot, draft] = v_t.reshape(v_cache[slot, draft].shape)
                post["beta_cache"][slot, draft] = b_t.reshape(beta_cache[slot, draft].shape)
                for name, r in zip(("cs_q", "cs_k", "cs_v"), raw):
                    post[name][slot, :, width - 1 + draft] = r.to(cs_q.dtype)

        if not (fuse_output_norm or quantize_output):
            return out.to(x_q.dtype).unsqueeze(0), post
        core = out.to(torch.bfloat16).double()
        rms = torch.rsqrt(core.square().mean(-1, keepdim=True) + onorm_eps)
        gate = torch.sigmoid(onorm_g[0].double().view(rows, heads, value_dim))
        normed = (core * rms * onorm_weight.double() * gate).to(torch.bfloat16)
        if quantize_output:
            return _mxfp8_rows(normed), post
        return normed.unsqueeze(0), post

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
        g: Optional[torch.Tensor],
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
        scale: Optional[float] = None,
        zero_accepted_hint: bool = False,
        regular_metadata_hint: bool = False,
        packed_token_layout: bool = False,
        onorm_g: Optional[torch.Tensor] = None,
        onorm_weight: Optional[torch.Tensor] = None,
        onorm_eps: float = 1e-5,
        fuse_output_norm: bool = False,
        quantize_output: bool = False,
        fa: Optional[torch.Tensor] = None,
        w_fb: Optional[torch.Tensor] = None,
        use_pdl: bool = False,
    ) -> None:
        slots = ssm_state_indices.tolist()
        accepted = num_accepted_tokens.tolist()
        counts = [accepted[s] for s in slots] if packed_token_layout else accepted[: len(slots)]
        # Two requests on one slot race on its state, conv window and replay caches.
        assert len(set(slots)) == len(slots), "ssm_state_indices must name one slot per request"
        # Only num_spec drafts are cached: a larger count replays entries no round wrote.
        assert all(0 <= a <= num_spec for a in counts), "accepted counts must be in [0, num_spec]"
        # The hints select specialized kernels that trust them.
        assert not zero_accepted_hint or not any(counts), "zero_accepted_hint with accepted drafts"
        if regular_metadata_hint:
            requests = len(slots)
            assert slots == list(range(requests)), "regular_metadata_hint needs identity slots"
            assert cu_seqlens is not None and cu_seqlens.tolist() == [
                n * (2 * num_spec + 1) for n in range(requests + 1)
            ], "regular_metadata_hint needs cu_seqlens = arange * (2 * num_spec + 1)"


kda_mtp_decode = KdaMtpDecode()
