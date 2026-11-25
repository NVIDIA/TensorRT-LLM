# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram gate: add a gated n-gram value into every residual copy (one Triton launch; plain torch for a token mask)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_TILE = 4096
_BLOCK = 1024


@triton.jit
def _gate_kernel(
    H, K, V, W, OUT, hc, d, k_tok, k_copy, v_tok, eps, inv_sqrt_d, BLOCK: tl.constexpr
):
    """One (token, copy): the three reductions over `d`, the gate, then `h + gate * value` (fp32 math)."""
    pid = tl.program_id(0).to(tl.int64)
    t, c = pid // hc, pid % hc
    row = pid * d
    krow = t * k_tok + c * k_copy
    hh = tl.zeros([BLOCK], dtype=tl.float32)
    kk = tl.zeros([BLOCK], dtype=tl.float32)
    hk = tl.zeros([BLOCK], dtype=tl.float32)
    for a in range(0, d, BLOCK):
        cols = a + tl.arange(0, BLOCK)
        m = cols < d
        h = tl.load(H + row + cols, mask=m, other=0.0).to(tl.float32)
        k = tl.load(K + krow + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + c * d + cols, mask=m, other=0.0)
        hh += h * h
        kk += k * k
        hk += h * w * k
    rstd = tl.rsqrt(tl.sum(hh, 0) / d + eps) * tl.rsqrt(tl.sum(kk, 0) / d + eps)
    dot = tl.sum(hk, 0) * rstd * inv_sqrt_d
    # copysign keeps the clamp floor at zero, negative zero included (stock's `engram_gate`)
    gate = 1.0 / (1.0 + tl.exp(-libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), 1e-6)), dot)))
    for a in range(0, d, BLOCK):
        cols = a + tl.arange(0, BLOCK)
        m = cols < d
        h = tl.load(H + row + cols, mask=m, other=0.0).to(tl.float32)
        v = tl.load(V + t * v_tok + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(OUT + row + cols, (h + gate * v).to(OUT.dtype.element_ty), mask=m)


class EngramGate(OpWrapper):
    """`h [n, hc, d]`; `key [n, hc, d]`; `value [n, d]` -> `h + gate * value`, `[n, hc, d]`.

    `gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))` with
    `dot = sum_d(h * q_weight * k_weight * key) * rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps) / sqrt(d)`
    per (token, copy). Rows outside `token_mask` pass through unchanged.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="decode token at the model's width, no mask",
            spec=dict(n=1, hc=4, d=5120, masked=False),
        ),
        Cell(
            why="verify-sized step with a mask that drops two rows",
            spec=dict(n=5, hc=4, d=5120, masked=True),
        ),
        Cell(why="prefill tile edge, 4097 rows", spec=dict(n=4097, hc=4, d=1024, masked=False)),
    )
    note = "The model passes `token_mask=None`; the masked path is certified but not called by the targets."

    def __call__(
        self,
        h: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dim = h.shape[-1]
        weight = q_weight.float() * k_weight.float()
        out = torch.empty_like(h)
        if (
            token_mask is None
            and h.is_contiguous()
            and key.stride(-1) == 1
            and value.stride(-1) == 1
        ):
            # key / value may be column slices of one projection (the engram's fused k | v output)
            if h.numel():
                with torch.cuda.device(h.device):
                    _gate_kernel[(h.shape[0] * h.shape[1],)](
                        h,
                        key,
                        value,
                        weight.contiguous(),
                        out,
                        h.shape[1],
                        dim,
                        key.stride(0),
                        key.stride(1),
                        value.stride(0),
                        float(eps),
                        dim**-0.5,
                        BLOCK=_BLOCK,
                    )
            return out
        for a in range(0, h.shape[0], _TILE):
            t = slice(a, min(a + _TILE, h.shape[0]))
            hf, kf = h[t].float(), key[t].float()
            rstd = torch.rsqrt(hf.square().mean(-1) + eps) * torch.rsqrt(kf.square().mean(-1) + eps)
            dot = (hf * weight * kf).sum(-1) * rstd * dim**-0.5
            gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
            update = hf + gate.unsqueeze(-1) * value[t].float().unsqueeze(-2)
            if token_mask is not None:
                # select, not a zero gate: 0 * a non-finite value would not leave the row unchanged
                update = torch.where(token_mask[t].view(-1, 1, 1), update, hf)
            out[t] = update.to(h.dtype)
        return out

    def reference(
        self,
        h: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per (token, copy), untiled, float64."""
        n, hc, d = h.shape
        out = torch.empty_like(h)
        hd, kd = h.double(), key.double()
        w = q_weight.double() * k_weight.double()
        for c in range(hc):
            hh, kk = hd[:, c], kd[:, c]
            rstd = (hh.square().mean(-1) + eps).rsqrt() * (kk.square().mean(-1) + eps).rsqrt()
            dot = (hh * w[c] * kk).sum(-1) * rstd * d**-0.5
            g = torch.sigmoid(torch.copysign(dot.abs().clamp(min=1e-6).sqrt(), dot))
            update = hh + g.unsqueeze(-1) * value.double()
            if token_mask is not None:
                update = torch.where(token_mask.unsqueeze(-1), update, hh)
            out[:, c] = update.to(h.dtype)
        return out

    def is_valid(
        self,
        h: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        token_mask: torch.Tensor | None = None,
    ) -> None:
        n, hc, d = h.shape
        assert key.shape == h.shape and value.shape == (n, d)
        assert q_weight.shape == (hc, d) and k_weight.shape == (hc, d)
        if token_mask is not None:
            assert token_mask.dtype == torch.bool and token_mask.shape == (n,), (
                "token_mask is a bool [n]; an integer mask is inverted by `~` bitwise"
            )


engram_gate = EngramGate()


class EngramGateKv(OpWrapper):
    """`engram_gate` over the WKV projection's fused output, on stock's kernel (`modules.engram.engram_gate` with
    `add_residual` and the cached fp32 `q_weight * k_weight`, as `DeepseekV41Engram` runs it): `h [n, hc, d]` bf16
    contiguous, `kv [n, (hc + 1) * d]` (keys, then the shared value), `weight` fp32 `[hc, d]` = `q_weight * k_weight`
    -> `h + gate * value`. One pass over each row (h is read once); stock's register cap."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="decode token at the model's width", spec=dict(n=1, hc=4, d=5120)),
        Cell(why="a 32300-row context step", spec=dict(n=32300, hc=4, d=5120)),
    )
    note = "Same math as `engram_gate`; gated against its float64 reference."

    def __call__(self, h, kv, q_weight, k_weight, weight, eps: float) -> torch.Tensor:
        from tensorrt_llm._torch.modules.engram.functional import engram_gate as stock_engram_gate

        return stock_engram_gate(
            h, kv, q_weight, k_weight, eps, add_residual=True, norm_weight_product=weight
        )

    def reference(self, h, kv, q_weight, k_weight, weight, eps: float) -> torch.Tensor:
        n, hc, d = h.shape
        return engram_gate.reference(
            h, kv[:, : hc * d].unflatten(-1, (hc, d)), kv[:, hc * d :], q_weight, k_weight, eps
        )

    def is_valid(self, h, kv, q_weight, k_weight, weight, eps: float) -> None:
        n, hc, d = h.shape
        assert h.dtype == kv.dtype == torch.bfloat16 and h.is_contiguous() and kv.is_contiguous()
        assert (
            kv.shape == (n, (hc + 1) * d)
            and weight.dtype == torch.float32
            and weight.shape == (hc, d)
        )


engram_gate_kv = EngramGateKv()
