# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA multi-token speculative verify of one layer on Rubin (CuTe DSL), packed rows, fused gated
RMSNorm, BF16 or MXFP8 output."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.ssm.rubin_kda_mtp_ops import kda_mtp_decode_impl as _verify
from ..gemm.rubin_mxfp8_gemm import mxfp8_scales_to_r128c4, quantize_mxfp8_reference


def _kda_step(state, k, gate, v, beta):
    """One delta-rule step on a K-first `[N, H, K, V]` fp32 state."""
    state = state * gate.exp().unsqueeze(-1)
    predicted = (state * k.unsqueeze(-1)).sum(dim=-2)
    v_new = (v - predicted) * beta.unsqueeze(-1)
    return state + k.unsqueeze(-1) * v_new.unsqueeze(-2)


def _catalog_mxfp8_rows(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """E4M3 codes `[T, H * 128]` and R128c4 UE8M0 bytes of bf16 `x` `[T, H, 128]` in the catalog
    `mxfp8_quantize`'s form: one scale per 32 values of a row (`rubin_mxfp8_linear`'s operand)."""
    rows, heads, width = x.shape
    codes, byte = quantize_mxfp8_reference(x.reshape(rows, heads * width))
    return codes, mxfp8_scales_to_r128c4(byte)


def _verify_point(why: str, **spec) -> Cell:
    return Cell(why=why, spec=dict(dict(state_dtype="float32", out="bf16"), **spec))


class RubinKdaMtpVerify(OpWrapper):
    """Verify `1 + num_spec` new rows per request against the KDA recurrence of its state slot.

    Per request `n` (slot `s = ssm_state_indices[n]`, `a = num_accepted_tokens[s]` drafts of the
    previous round accepted): replay the `a` cached drafts, then run rows `n * (1 + num_spec) + j`:
    causal conv (width 4) + SiLU of q / k / v, L2-normalized q (times `scale`) and k, gate
    `lower_bound * sigmoid((g + dt_bias) * exp(A_log))`, `beta = sigmoid(beta)`, the delta-rule
    step, `o = S^T q`. The forget-gate logits are `g` [1, T, H, 128]. After row `j = 0` the state
    and the conv windows are committed in place; rows `j >= 1` become the next round's pending
    drafts. The output is the gated RMSNorm of `bf16(o)`: bf16 `[1, T, H, V]`, or with
    `quantize_output` the catalog `mxfp8_quantize`'s MXFP8 form of that BF16 output
    `(e4m3 [T, H * V], R128c4 UE8M0 bytes)`, one scale per 32 values (the operand
    `rubin_mxfp8_linear` makes of it, bit for bit).

    Mutates `recurrent_state` (fp32 or bf16, V-first `[pool, H, V, K]`), `cs_q / cs_k / cs_v`,
    `k_cache / v_cache` (bf16), `g_cache` (fp32) and `beta_cache` (bf16), at the slots of
    `ssm_state_indices` only.
    """

    ARCHS: frozenset[Arch] = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        _verify_point(
            "DEP16 GEN b12k7: 96 KDA heads per rank (attention DP)", N=12, H=96, num_spec=7
        ),
        _verify_point("helix8 GEN b16k5: 12 heads (KDA TP8)", N=16, H=12, num_spec=5),
        _verify_point("TEP8 GEN b1k7: 12 heads, one request", N=1, H=12, num_spec=7),
        _verify_point("helix8 GEN b24k5 (mid M): 12 heads", N=24, H=12, num_spec=5),
        _verify_point(
            "DEP16 GEN b12k7: x_q/x_k/x_v, onorm_g and beta views of the stacked GEMM output "
            "(row pitch 49376)",
            N=12,
            H=96,
            num_spec=7,
            stacked=True,
        ),
        _verify_point(
            "TEP8 GEN b1k7: the same five views (row pitch 6288)",
            N=1,
            H=12,
            num_spec=7,
            stacked=True,
        ),
        _verify_point(
            "DEP16 GEN b12k7, o_proj's operand in the catalog quantizer's 1x32 MXFP8 form",
            N=12,
            H=96,
            num_spec=7,
            out="mxfp8",
        ),
        _verify_point(
            "BF16 recurrent pool (mamba_ssm_cache_dtype bfloat16)",
            N=8,
            H=12,
            num_spec=7,
            state_dtype="bfloat16",
        ),
    )
    note = """
    The runtime verify contract: packed rows (exactly 1 + num_spec per
    request), the fused gated RMSNorm, K = V = 128, conv width 4,
    1 <= num_spec <= 7. q / k / v / onorm_g may be column sections
    of one fused per-token projection; beta may be the tail of the padded
    [f_a | b] projection. The specialized targets pass all five as views of
    one [q | k | v | g | f_a | b] GEMM output (rows of 6288 or 49376). Conv
    caches are bf16 [pool, H * 128, >= 3 +
    num_spec], dim-contiguous; conv weights fp32 [H * 128, 4] contiguous;
    num_accepted_tokens int32, indexed by slot.

    Two kernels serve this contract with the same arithmetic in the same order
    (bitwise equal results): kda_decode_mtp_v2_kernel within one wave of tiles
    and kda_decode_mtp_2cta_kernel above one wave. quantize_output writes the
    MXFP8 output in the catalog mxfp8_quantize's 1x32 form, an epilogue of the
    kernel the BF16 output takes (both have it), so rubin_mxfp8_gemm gets
    exactly the operands of rubin_mxfp8_linear.

    3xTF32 tensor-core recurrence: the state within 3e-5 relative RMS of an
    fp32 step-by-step evaluation; the bf16 output within one ulp of it. The
    MXFP8 codes are the e4m3 rounding of that bf16 output.
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
        quantize_output: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return _verify(
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
            num_spec=num_spec,
            num_accepted_tokens=num_accepted_tokens,
            lower_bound=lower_bound,
            scale=scale,
            onorm_g=onorm_g,
            onorm_weight=onorm_weight,
            onorm_eps=onorm_eps,
            quantize_output=quantize_output,
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
        quantize_output: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sequential fp32 recurrence, one token step at a time, batched over requests and heads.

        Applies the op's in-place updates to its own arguments (pass clones).
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
            q_t = torch.nn.functional.normalize(torch.nn.functional.silu(conv["q"]), dim=-1) * scale
            k_t = torch.nn.functional.normalize(torch.nn.functional.silu(conv["k"]), dim=-1)
            v_t = torch.nn.functional.silu(conv["v"])
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
        normed = (normed * onorm_weight.to(f32) * torch.sigmoid(onorm_in)).to(torch.bfloat16)
        if quantize_output:
            return _catalog_mxfp8_rows(normed.reshape(N * T, H, V))
        return normed.reshape(1, N * T, H, V)

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
        quantize_output: bool,
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


rubin_kda_mtp_verify = RubinKdaMtpVerify()
