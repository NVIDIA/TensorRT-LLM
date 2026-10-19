# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet single-token decode over an fp32 slot pool, gates fused in.

Wraps FlashInfer's public
``flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose`` (tier-3
sourcing, the installed FlashInfer). With an fp32 pool and ``T == 1`` it
runs the CuTe-DSL "pretranspose" kernel (``run_pretranspose_decode``): eight
CTAs per (sequence, v-head) state, each streaming 16 of its 128 V rows through
shared memory, the pool read and written in place through
``initial_state_indices``. A bf16 pool or ``T > 1`` route elsewhere inside the
same callable and are not certified here.

FlashInfer API: ``flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose``.
"""

from __future__ import annotations

from typing import Optional

import flashinfer.gdn_decode
import torch

from .._op import Arch, Cell, OpWrapper

#: qwen3.8-flash-next-nvfp4's Gated DeltaNet at tp1: (H, HV, K, V) -- 16 q/k
#: heads, 48 v heads, head dims 128.
QWEN38_GEOMETRY = (16, 48, 128, 128)
#: Largest slot count the cache manager was seen to allocate the recurrent pool
#: at for qwen3.8-flash-next-nvfp4 (10 .. 461 over twelve runtime-value dumps).
ENGINE_SLOTS = 461


def _scattered(n: int, slots: int = ENGINE_SLOTS) -> tuple[int, ...]:
    """`n` distinct slots of a `slots`-row pool, both ends first, then a
    stride-37 walk over the interior (37 is coprime to every pool size here)."""
    rest = tuple((i * 37 + 4) % (slots - 2) + 1 for i in range(max(n - 2, 0)))
    return ((slots - 1, 0) + rest)[:n]


class GatedDeltaRuleDecodePretranspose(OpWrapper):
    """One gated-delta-rule step per sequence, gates computed in-kernel.

    Row `n` of `q`/`k` `[B, 1, H, K]`, `v` `[B, 1, HV, V]`, `a`/`b` `[B, 1, HV]`;
    v-head `hv` reads q/k head `hv // (HV // H)`; state `h` `[V, K]` (K
    innermost) read from `initial_state[initial_state_indices[n]]`, in fp32:

        g    = -exp(A_log) * softplus(a + dt_bias)    (softplus beta 1, threshold 20)
        beta = sigmoid(b)
        q, k = x * rsqrt(sum(x^2) + 1e-6)               (only with use_qk_l2norm)
        h   *= exp(g);  h += k (beta * (v - h k))^T;   o = h (scale * q)

    `scale=None` is `K ** -0.5`. The final `h` goes back to the same slot, or
    to `output_state_indices[n]` when given. A negative index skips the row:
    no state read or write, `o` written as zero.

    Fusion boundary: both gates, the optional q/k norm, scale, recurrence,
    indexed gather and scatter. The conv, the q/k/v split and the gated RMS
    norm on `o` are the caller's.

    Returns `(output, initial_state)` as the callable does: `output`
    `[B, 1, HV, V]` (the `output` buffer itself when given, bf16; else a fresh
    tensor in `q.dtype`) and the pool, advanced in place.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: geometry (H, HV, K, V), m (rows), slots (pool rows), indices (None
    # -> _scattered(m, slots)), qkv ("views": column views of one
    # [m, 2*H*K + HV*V] conv output, as a decode forward splits it |
    # "contiguous"), output ("buffer" | None), pool ("contiguous" | "padded":
    # slot stride HV*V*K + 64 | "layer_view": pool[1] of a [3, slots, ...]
    # cache). Every cell but the last is qwen3.8-flash-next-nvfp4's geometry.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"qwen3.8 decode M={m}, 461-slot pool, scattered slots",
            spec=dict(geometry=QWEN38_GEOMETRY, m=m, slots=ENGINE_SLOTS),
        )
        for m in (1, 2, 8, 32, 64, 128)
    ) + (
        Cell(
            why="qwen3.8 output None: fresh q.dtype output, M=4",
            spec=dict(geometry=QWEN38_GEOMETRY, m=4, slots=ENGINE_SLOTS, output=None),
        ),
        Cell(
            why="qwen3.8 CUDA-graph pad rows (-1) between slot 0 and 460",
            spec=dict(
                geometry=QWEN38_GEOMETRY,
                m=4,
                slots=ENGINE_SLOTS,
                indices=(0, -1, ENGINE_SLOTS - 1, -1),
            ),
        ),
        Cell(
            why="qwen3.8 padded slot stride (HV*V*K + 64), M=8",
            spec=dict(geometry=QWEN38_GEOMETRY, m=8, slots=40, pool="padded"),
        ),
        Cell(
            why="qwen3.8 pool[1] of a [3, slots, HV, V, K] cache, M=4",
            spec=dict(geometry=QWEN38_GEOMETRY, m=4, slots=24, pool="layer_view"),
        ),
        Cell(
            why="qwen3.8 contiguous q/k/v instead of conv-output views, M=16",
            spec=dict(geometry=QWEN38_GEOMETRY, m=16, slots=ENGINE_SLOTS, qkv="contiguous"),
        ),
        Cell(
            why="HV/H = 2 (16 q/k heads, 32 v heads), M=8, 40-slot pool",
            spec=dict(geometry=(16, 32, 128, 128), m=8, slots=40),
        ),
    )

    note = """
    Metadata consumed: none.

    Call site as qwen3.8-flash-next-nvfp4's decode `_gdn_block` makes it (in
    place of `fused_sigmoid_gating_delta_rule_update`): `q=reshape(qv, [M, 1, 16, 128])`,
    `k=...`, `v=reshape(vv, [M, 1, 48, 128])` -- column views of the conv
    output, read through their strides -- `state=None`, `A_log`/`dt_bias`
    fp32 `[48]`, `a`/`b` the `[M, 48]` projections viewed `[M, 1, 48]`,
    `scale=None`, `output=` a bf16 `[M, 1, 48, 128]` buffer, `use_qk_l2norm=
    True`, `initial_state=` the layer's fp32 pool `[slots, 48, 128, 128]`,
    `initial_state_indices=` int32 slots. `output.view(1, M, 48, 128)` is what
    `_gdn_epilogue` reads.

    Preconditions not checked by anything:
    - fp32 pool, K-innermost `(V, K)` trailing axes, slot stride a multiple of
      4 elements (asserted) -- a transposed `[S, HV, K, V]` pool at K == V has
      the same shape and is accepted.
    - Indices distinct and in `[0, S)` or negative. An index >= S is an
      unchecked device access: measured, slot 460 of a 40-slot pool raised
      `cudaErrorIllegalAddress` on the next sync and poisoned the process.
    - `A_log` fp32 (asserted), `dt_bias` fp32 or bf16 (asserted).

    Loud: every tensor's base pointer must be 16-byte aligned (the kernel is
    compiled with `assumed_align=16`). Measured: indices taken as
    `idx[n_c:]` of an int32 vector at an offset not a multiple of 4 raise
    `RuntimeError: Tensor data pointer is not aligned to 16 bytes.` -- so a
    mixed step's decode half, `si_all` split after its context rows, needs
    a copy. The decode-only step's `si_all` starts at offset 0.

    JIT and capture: the first call per (`T, H, HV, K, V, q.dtype, scale,
    use_qk_l2norm`, pool inner strides) compiles a CuTe-DSL kernel in-process
    (seconds); the first call per batch size allocates two small int32 aux
    tensors, unused on the pool path. Both must happen in eager warmup, before
    graph capture. Capture-safe after that: no host sync, no allocation when
    `output` is given (one `torch.zeros` per call otherwise).

    Numerics (measured at qwen3.8's geometry, fp64 reference): o max abs 1.2e-4
    at M=128 (|o| <= 6.2e-2), equal to the in-tree Triton op's
    (`modules/fla/fused_sigmoid_gating_recurrent`); written state max abs
    2.6e-7 (Triton 2.0e-7). Not bitwise to the Triton op: rsqrt(sum + eps) vs
    1 / (sqrt(sum) + eps) for the q/k norm, and fastmath exp/log.

    Not certified: bf16/fp16 pool (another kernel), `T > 1` (routes to
    `gated_delta_rule_mtp`), `state=` direct path, `output_state_indices`,
    fp16 operands, K or V other than 128.

    Hard limits
      none probed.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: Optional[torch.Tensor],
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        b: torch.Tensor,
        scale: Optional[float] = None,
        output: Optional[torch.Tensor] = None,
        use_qk_l2norm: bool = True,
        initial_state: Optional[torch.Tensor] = None,
        initial_state_indices: Optional[torch.Tensor] = None,
        output_state_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose(
            q=q,
            k=k,
            v=v,
            state=state,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            scale=scale,
            output=output,
            use_qk_l2norm=use_qk_l2norm,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            output_state_indices=output_state_indices,
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: Optional[torch.Tensor],
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        b: torch.Tensor,
        scale: Optional[float] = None,
        output: Optional[torch.Tensor] = None,
        use_qk_l2norm: bool = True,
        initial_state: Optional[torch.Tensor] = None,
        initial_state_indices: Optional[torch.Tensor] = None,
        output_state_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(o, pool_after)`: fp64, rounded once -- `o` to `output`'s dtype (bf16
        when given, else `q.dtype`), the state to the pool's. `state=` (no pool)
        is treated as a pool of `B` rows read and written in order. Mutates
        nothing."""
        dt = torch.float64
        batch, _, num_k_heads, k_dim = q.shape
        hv = v.shape[2]
        rep = hv // num_k_heads
        scale = k_dim**-0.5 if scale is None else scale
        pool_in = initial_state if initial_state is not None else state
        if initial_state is not None:
            reads = initial_state_indices.tolist()
        else:
            reads = list(range(batch))
        writes = reads if output_state_indices is None else output_state_indices.tolist()
        pool = pool_in.clone()
        o = torch.zeros(batch, 1, hv, v.shape[3], dtype=dt, device=q.device)
        neg_a = -torch.exp(A_log.to(dt))
        for n in range(batch):
            if reads[n] < 0:
                continue
            q_n = q[n, 0].to(dt).repeat_interleave(rep, dim=0)
            k_n = k[n, 0].to(dt).repeat_interleave(rep, dim=0)
            if use_qk_l2norm:
                q_n = q_n * torch.rsqrt(q_n.pow(2).sum(-1, keepdim=True) + 1e-6)
                k_n = k_n * torch.rsqrt(k_n.pow(2).sum(-1, keepdim=True) + 1e-6)
            x = a[n, 0].to(dt) + dt_bias.to(dt)
            softplus = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
            beta = torch.sigmoid(b[n, 0].to(dt))
            h = pool_in[reads[n]].to(dt) * torch.exp(neg_a * softplus)[:, None, None]
            u = (v[n, 0].to(dt) - torch.einsum("hvk,hk->hv", h, k_n)) * beta[:, None]
            h = h + u[:, :, None] * k_n[:, None, :]
            o[n, 0] = torch.einsum("hvk,hk->hv", h, q_n * scale)
            if writes[n] >= 0:
                pool[writes[n]] = h.to(pool.dtype)
        out_dtype = output.dtype if output is not None else q.dtype
        return o.to(out_dtype), pool


gated_delta_rule_decode_pretranspose = GatedDeltaRuleDecodePretranspose()
