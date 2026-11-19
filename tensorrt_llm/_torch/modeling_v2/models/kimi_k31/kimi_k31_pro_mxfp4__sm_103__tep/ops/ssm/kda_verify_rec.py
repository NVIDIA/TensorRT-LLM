# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA multi-token verify as an FP32 recurrence (packed rows, fused gated RMSNorm), through the family's
``kda_verify_rec`` kernel."""

from __future__ import annotations

from typing import NamedTuple

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.ssm import kda_verify_rec as _kda_verify_rec


def supported(
    recurrent_state: torch.Tensor, num_spec: int, head_k_dim: int, head_v_dim: int, conv_width: int
) -> bool:
    """Whether :data:`k31_kda_verify_rec` takes this layer's verify (the kernel's own check: SM100 / SM103 / SM107, FP32
    V-first state with 16-byte rows, up to 7 drafts, K = V = 128, conv width 4). False without the CuTe DSL."""
    if not IS_CUTLASS_DSL_AVAILABLE:
        return False
    return _kda_verify_rec.supported(recurrent_state, num_spec, head_k_dim, head_v_dim, conv_width)


def _tep(heads: int, num_spec: int, requests: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(heads=heads, num_spec=num_spec, requests=requests))


class VerifyRecResult(NamedTuple):
    """What the op leaves in each buffer it writes, as FP64 values (the op stores them in the buffers' dtypes)."""

    out: torch.Tensor
    recurrent_state: torch.Tensor
    k_cache: torch.Tensor
    g_cache: torch.Tensor
    v_cache: torch.Tensor
    beta_cache: torch.Tensor
    cs_q: torch.Tensor
    cs_k: torch.Tensor
    cs_v: torch.Tensor


class K31KdaVerifyRec(OpWrapper):
    """One KDA layer's speculative verify of ``N`` requests with ``1 + num_spec`` packed rows each.

    Per request (slot ``s = ssm_state_indices[n]``, ``a = num_accepted_tokens[s]`` accepted drafts of the previous
    round): replay the ``a`` cached drafts (k / g / v / beta caches, raw conv columns) onto the slot's state, then
    run the new rows -- causal conv + SiLU of q / k / v, L2 norm of q (times ``scale``) and k, the lower-bound
    sigmoid gate ``lower_bound * sigmoid((g + dt_bias) * exp(A_log))``, ``sigmoid(beta)`` and the delta rule.
    The state and conv windows are committed in place after the first new (golden) row; the remaining new rows
    are cached for the next round. Each new row's output is ``RMSNorm(o) * onorm_weight * sigmoid(onorm_g)``,
    stored as BF16 into ``out`` ``[1, T, H, 128]``, which the op returns.

    Mutates ``recurrent_state`` (``[slots, H, V, K]`` FP32), the replay caches, the conv caches and ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Kimi K3.1 KDA: 96 heads, K = V = 128, conv width 4; attention TP 4 / 8 / 16 -> 24 / 12 / 6 heads per rank.
    # DFlash2 verifies 1 + 7 rows per request (num_spec 7); 4 covers a shorter draft.
    CELLS: tuple[Cell, ...] = (
        _tep(3, 7, 1, "TEP32 (ll c1), one request of DFlash2 width 8"),
        _tep(3, 4, 1, "TEP32 (ll c1), one request of DFlash2 width 5"),
        _tep(6, 4, 2, "TEP16 (mid c16), two requests of width 5"),
        _tep(6, 7, 12, "TEP16 (M2 c64), 12 requests of width 8: 72 tiles, 2-CTA clusters"),
        _tep(
            12,
            7,
            1,
            "TEP8, one request: one (head, request) tile row of 8-CTA clusters",
        ),
        _tep(
            12,
            7,
            16,
            "TEP8, 16 requests: the cluster narrows as the tiles fill the SMs",
        ),
        _tep(24, 7, 8, "TEP4, 24 heads per rank"),
        _tep(6, 4, 64, "TEP16, 64 requests of a 4-draft round: 384 tiles, 2-CTA clusters"),
        _tep(6, 7, 24, "TEP16, 24 requests"),
    )

    note = """
    The TEP target runs this kernel (instead of the 3xTF32 tensor-core verify) for every fused verify outside
    torch.compile, when `supported(...)` holds:
    SM100 / SM103 / SM107, an FP32 state with 16-byte rows, 1 <= num_spec <= 7, K = V = 128, conv width 4.
    Outputs and the committed state agree with the tensor-core kernels to FP32 rounding (not bitwise); the
    gate / value / beta caches and the conv windows are bitwise theirs, the k cache within one BF16 step.
    Rows are packed: T = N * (1 + num_spec). Launches compile per layout and constants; the kernel keeps no
    state between launches, so a captured CUDA graph replays it.

    The launch is a programmatic dependent one: it loads the state, replay entries, conv windows and weights
    before ``griddepcontrol.wait`` and releases its dependent right after the wait.
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
        scale: float,
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
        out: torch.Tensor,
    ) -> torch.Tensor:
        _kda_verify_rec.k31_kda_verify_rec(
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
            out,
        )
        return out

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
        scale: float,
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
        out: torch.Tensor,
    ) -> VerifyRecResult:
        """FP64 replay semantics, token by token; reads the inputs, writes nothing (``out`` only names the
        destination). The output is rounded to BF16 before the norm, as the kernel's epilogue reads it."""
        f = lambda t: t.double()  # noqa: E731
        _, T, H, K = x_q.shape
        V = x_v.shape[-1]
        W = w_q.shape[-1]
        M = num_spec
        NT = M + 1
        state = f(recurrent_state).clone()
        cs = [f(c).clone() for c in (cs_q, cs_k, cs_v)]
        kc, gc, vc, bc = (f(t).clone() for t in (k_cache, g_cache, v_cache, beta_cache))
        w = [f(x).view(H, K, W) for x in (w_q, w_k, w_v)]
        xq, xk, xv = f(x_q)[0], f(x_k)[0], f(x_v)[0]
        graw, braw, og = f(g)[0], f(beta)[0], f(onorm_g)[0]
        A = torch.exp(f(A_log)).view(H, 1)
        dtb = f(dt_bias).view(H, K)
        norm_w = f(onorm_weight)
        result = torch.zeros(T, H, V, dtype=torch.float64, device=x_q.device)
        slots = ssm_state_indices.tolist()
        accepted = num_accepted_tokens[ssm_state_indices.long()].tolist()
        for n, (s, a) in enumerate(zip(slots, accepted)):
            a = min(a, M)
            h = state[s].transpose(1, 2).clone()  # [H, K, V]
            hist = [c[s].view(H, K, -1)[:, :, : W - 1].clone() for c in cs]
            tail = [[] for _ in range(3)]
            newk, newg, newv, newb = [], [], [], []
            committed_hist = None
            for i in range(a + NT):
                if i < a:
                    kt, gt, vt, bt = (
                        kc[s, i].view(H, K),
                        gc[s, i].view(H, K),
                        vc[s, i].view(H, V),
                        bc[s, i],
                    )
                    raws = [c[s].view(H, K, -1)[:, :, W - 1 + i] for c in cs]
                    qt = None
                else:
                    r = n * NT + i - a
                    raws = [xq[r], xk[r], xv[r]]
                    conv = []
                    for j in range(3):
                        y = (torch.cat([hist[j], raws[j].unsqueeze(-1)], dim=-1) * w[j]).sum(-1)
                        conv.append(y * torch.sigmoid(y))
                    qt = conv[0] / torch.sqrt((conv[0] ** 2).sum(-1, keepdim=True) + 1e-6) * scale
                    kt = conv[1] / torch.sqrt((conv[1] ** 2).sum(-1, keepdim=True) + 1e-6)
                    vt = conv[2]
                    gt = lower_bound * torch.sigmoid((graw[r] + dtb) * A)
                    bt = torch.sigmoid(braw[r])
                hist = [
                    torch.cat([hist[j][:, :, 1:], raws[j].unsqueeze(-1)], dim=-1) for j in range(3)
                ]
                h = h * torch.exp(gt).unsqueeze(-1)
                vn = (vt - torch.einsum("hkv,hk->hv", h, kt)) * bt.unsqueeze(-1)
                h = h + kt.unsqueeze(-1) * vn.unsqueeze(1)
                if qt is not None:
                    o = torch.einsum("hkv,hk->hv", h, qt).to(torch.bfloat16).double()
                    rms = torch.rsqrt((o * o).mean(-1, keepdim=True) + onorm_eps)
                    row = n * NT + i - a
                    result[row] = o * norm_w * torch.sigmoid(og[row]) * rms
                if i == a:
                    state[s] = h.transpose(1, 2)
                    committed_hist = [x.clone() for x in hist]
                if i > a:
                    newk.append(kt.reshape(-1))
                    newg.append(gt.reshape(-1))
                    newv.append(vt.reshape(-1))
                    newb.append(bt.reshape(-1))
                    for j in range(3):
                        tail[j].append(raws[j].reshape(-1))
            for j in range(3):
                cs[j][s] = torch.cat(
                    [committed_hist[j].reshape(H * K, -1), torch.stack(tail[j], -1)], -1
                )
            kc[s], gc[s], vc[s], bc[s] = (
                torch.stack(newk),
                torch.stack(newg),
                torch.stack(newv),
                torch.stack(newb),
            )
        return VerifyRecResult(result.view(1, T, H, V), state, kc, gc, vc, bc, *cs)

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
        scale: float,
        onorm_g: torch.Tensor,
        onorm_weight: torch.Tensor,
        onorm_eps: float,
        out: torch.Tensor,
    ) -> None:
        # The kernel loads its operands in 16-byte vectors from DLPack views compiled with an assumed alignment
        # (16 B; 2 B for beta and its cache; 4 B for the int32 metadata) and reads the state as FP32 rows of 16
        # bytes. A view that breaks either is read at the wrong address rather than rejected.
        st = recurrent_state
        assert (
            st.dtype == torch.float32
            and st.dim() == 4
            and st.stride(-1) == 1
            and st.stride(-2) % 4 == 0
            and st.data_ptr() % 16 == 0
        ), "recurrent_state must be an FP32 [slots, H, V, K] view with 16-byte aligned rows"
        for name, t, align in (
            ("x_q", x_q, 16),
            ("x_k", x_k, 16),
            ("x_v", x_v, 16),
            ("g", g, 16),
            ("onorm_g", onorm_g, 16),
            ("out", out, 16),
            ("cs_q", cs_q, 16),
            ("cs_k", cs_k, 16),
            ("cs_v", cs_v, 16),
            ("k_cache", k_cache, 16),
            ("g_cache", g_cache, 16),
            ("v_cache", v_cache, 16),
            ("beta", beta, 2),
            ("beta_cache", beta_cache, 2),
            ("ssm_state_indices", ssm_state_indices, 4),
            ("num_accepted_tokens", num_accepted_tokens, 4),
        ):
            assert t.data_ptr() % align == 0 and all(
                (s * t.element_size()) % align == 0 for s in t.stride() if s != 1
            ), f"{name} must be {align}-byte aligned with {align}-byte aligned strides"
        assert ssm_state_indices.dtype == torch.int32, "ssm_state_indices must be int32"


k31_kda_verify_rec = K31KdaVerifyRec()
