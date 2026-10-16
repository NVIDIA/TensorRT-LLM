# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet recurrent core with the gating fused in, over a slot pool:
the GDN decode step.

Wraps the in-tree module callable
``tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update``
(tier-2 sourcing: the gated delta rule has no ``torch.ops.trtllm`` surface).
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as _fused_sigmoid_gating_delta_rule_update,
)

#: The pool width the bulk of the cells run at, from the old test: a width the
#: engine never allocates, kept because its cells were certified there.
SLOTS = 24
#: Largest slot count the cache manager was seen to allocate the recurrent pool
#: at for this checkpoint (10 .. 461 over twelve runtime-value dumps, batch never
#: above 32). The target passes the layer's whole pool unsliced.
ENGINE_SLOTS = 461


def _scattered(n_seqs: int) -> tuple[int, ...]:
    """`n_seqs` distinct engine-pool slots, both ends included."""
    return ((0, ENGINE_SLOTS - 1) + tuple(i * 13 + 4 for i in range(n_seqs)))[:n_seqs]


class FusedSigmoidGatingDeltaRuleUpdate(OpWrapper):
    """Advance the gated delta rule over every sequence, gates computed in-kernel.

    Sequence `n` (tokens `cu_seqlens[n]:cu_seqlens[n+1]`, or row `n` without
    `cu_seqlens`), v-head `hv` reading k/q head `hv // (HV // H)`, state `h`
    `[K, V]` loaded from `initial_state_source[initial_state_indices[n]]`
    (stored `[V, K]`, K innermost), per token, all in fp32:

        g    = -exp(A_log) * softplus_b(a + dt_bias)    beta = sigmoid(b)
        q, k = q / (|q| + 1e-6), k / (|k| + 1e-6)       (only with use_qk_l2norm_in_kernel)
        h   *= exp(g);  u = (v - h^T k) * beta;  h += k u^T;  o = h^T (scale * q)

    `softplus_b(x) = log(1 + exp(beta_s x)) / beta_s` below `softplus_threshold`
    (in `beta_s x`), `x` above; `scale=None` is `K ** -0.5`, bitwise. The final
    `h` is written back to the same slot. A negative index means no slot: zero
    initial state, nothing written. `initial_state_source=None`: zero state,
    nothing written.

    Fusion boundary: both gates, the optional q/k norm, scale, recurrence,
    indexed gather and scatter are inside. The conv, the q/k/v split (plain
    views) and the gated RMS norm on `o` are the caller's.

    Returns `o` `[B, T, HV, V]` in `output`'s dtype (it *is* `output` when
    given), else `q`'s. The pool is advanced in place and not returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: seq_lens, slots (pool rows; None for no pool), indices (None ->
    # arange), pool_dtype (fp32), l2norm (True), normalize_qk (caller-normalized
    # q/k when l2norm is off), output ("none" | "prefix": a length-prefix slice
    # of a longer buffer, as the stock mixer passes), ab ("packed": column
    # halves of a [T, 96] projection | "separate": two contiguous [T, 48], as
    # the target's two matmuls make them), layout ("varlen" | "layer_view":
    # pool[1] of a [3, slots, ...] cache | "batched": cu_seqlens None).
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"target decode call, N={n}, 461-slot pool",
                spec=dict(
                    seq_lens=(1,) * n, slots=ENGINE_SLOTS, indices=_scattered(n), ab="separate"
                ),
            )
            for n in (1, 4, 16, 32)
        )
        + (
            Cell(
                why="varlen [3, 5, 1, 7] at the 461-slot pool",
                spec=dict(seq_lens=(3, 5, 1, 7), slots=ENGINE_SLOTS, indices=_scattered(4)),
            ),
            Cell(
                why="CUDA-graph pad rows (-1) between slot 0 and 460",
                spec=dict(
                    seq_lens=(1, 1, 1, 1),
                    slots=ENGINE_SLOTS,
                    indices=(0, -1, ENGINE_SLOTS - 1, -1),
                    ab="separate",
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"decode N={n}, one token per sequence, 24 slots",
                spec=dict(seq_lens=(1,) * n, slots=SLOTS),
            )
            for n in (1, 2, 4, 8, 16)
        )
        + tuple(
            Cell(
                why=f"output= a length-prefix slice of a longer buffer, N={n}",
                spec=dict(seq_lens=(1,) * n, slots=SLOTS, output="prefix"),
            )
            for n in (1, 4, 16)
        )
        + (
            Cell(
                why="scattered, unsorted slots [17, 3, 0, 11]",
                spec=dict(seq_lens=(1,) * 4, slots=SLOTS, indices=(17, 3, 0, 11)),
            ),
            Cell(
                why="scattered slots [23, 5]",
                spec=dict(seq_lens=(1, 1), slots=SLOTS, indices=(23, 5)),
            ),
            Cell(
                why="negative slot indices [2, -1, 5, -1]",
                spec=dict(seq_lens=(1,) * 4, slots=SLOTS, indices=(2, -1, 5, -1)),
            ),
            Cell(
                why="every slot index negative",
                spec=dict(seq_lens=(1, 1), slots=SLOTS, indices=(-1, -1)),
            ),
        )
        + tuple(
            Cell(
                why=f"varlen multi-token sequences {list(s)}",
                spec=dict(seq_lens=s, slots=SLOTS),
            )
            for s in ((3, 5), (2, 1, 7, 1), (64, 33))
        )
        + (
            Cell(
                why="l2norm off, raw single-token q/k (one step cannot amplify)",
                spec=dict(seq_lens=(1,) * 4, slots=SLOTS, l2norm=False),
            ),
        )
        + tuple(
            Cell(
                why=f"l2norm off, caller-normalized q/k, {list(s)}",
                spec=dict(seq_lens=s, slots=SLOTS, l2norm=False, normalize_qk=True),
            )
            for s in ((3, 5), (2, 1, 7, 1))
        )
        + tuple(
            Cell(
                why=f"bf16 pool on the Triton branch (T != N), {list(s)}",
                spec=dict(seq_lens=s, slots=SLOTS, pool_dtype=torch.bfloat16),
            )
            for s in ((3, 5), (2, 6))
        )
        + (
            Cell(
                why="initial_state_source None: zero state, no write-back",
                spec=dict(seq_lens=(1,) * 4, slots=None),
            ),
            Cell(
                why="pool[layer] of a [layers, slots, HV, V, K] cache",
                spec=dict(seq_lens=(1,) * 4, slots=SLOTS, layout="layer_view"),
            ),
            Cell(
                why="cu_seqlens None: batched [N, T, ...] layout, N=3, T=4",
                spec=dict(seq_lens=(4, 4, 4), slots=SLOTS, layout="batched"),
            ),
        )
    )

    note = """
    Metadata consumed: none.

    Call site (`_gdn_block` decode): `(a_log, a, dt_bias, 1.0, 20.0, q, k, v,
    b, ssm_states, si_all, None, True, qsl_long, None)` -- q/k/v reshaped
    column views of the conv output `[T, 10240]` (read through `stride(1)`, no
    repack), `a`/`b` two contiguous `[T, 48]` projections, the layer's whole
    fp32 pool `[slots, 48, 128, 128]`, int32 slots, int64 `cu_seqlens`, one
    token per sequence. Certified at 24 and 461 slots; at 461 output and state
    are bitwise identical to 24 for the same slot content.

    Branch: the callable takes a FlashInfer CuTe-DSL kernel when the pool is
    bf16, `cu_seqlens` is given and `T == N` (head dims 128, supported arch);
    everything else -- every fp32-pool call, i.e. this target's
    `--mamba_ssm_cache_dtype=float32` -- runs the vendored Triton kernel. Only
    the Triton branch is certified; the test proves the fp32 pool never
    reaches FlashInfer and that the witness is not blind.

    Preconditions not checked by anything:
    - `HV % H == 0`.
    - Pool trailing axes `(V, K)`, K innermost. The callable asserts the
      strides `(V*K, K, 1)`, but at K == V a transposed `[S, HV, K, V]` pool
      has the same strides: accepted, measured 2.15e-2 off (vs 3.8e-6
      correct). The slot stride is free.
    - Indices distinct (a shared slot races on write-back: outputs fine, the
      slot holds one writer's state) and in `[0, S)` or negative
      (out-of-range is an unchecked device access -- `tl.device_assert` is
      compiled out -- not probed).
    - q/k unit-norm conditioned: in-kernel norm on, or normalized by the
      caller. Raw multi-token q/k with the norm off drifted the fp32 state
      1.7e-2 relative from an fp32 sequential reference: domain, not defect.
    - `output` contiguous, `[B, T, HV, V]`: guarded by `is_valid`. Its dtype
      wins (an fp32 `output` with bf16 q returns fp32).

    Deterministic: repeated calls are bitwise identical. Grid
    `(1, ceil(V/32), min(N*HV, 65535))`, grid-striding past the cap (read,
    not measured: N > 1365 at this geometry).

    Not certified: the FlashInfer branch, K or V other than 128, fp16/fp32
    operands, `softplus_beta`/`softplus_threshold` other than 1.0/20.0,
    `B > 1` with `cu_seqlens`.

    Hard limits
      none probed.
    """

    def __call__(
        self,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        softplus_beta: float,
        softplus_threshold: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        initial_state_source: Optional[torch.Tensor],
        initial_state_indices: torch.Tensor,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            output=output,
        )

    def reference(
        self,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        softplus_beta: float,
        softplus_threshold: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        initial_state_source: Optional[torch.Tensor],
        initial_state_indices: torch.Tensor,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """`(o, pool_after)`: sequential over tokens, fp32 throughout like the
        kernel, `o` rounded once to the result dtype and the state once to the
        pool's. `output` contributes only its dtype; nothing is mutated."""
        batch, total, num_k_heads, k_dim = q.shape
        hv, v_dim = v.shape[2], v.shape[3]
        repeat = hv // num_k_heads
        scale = k_dim**-0.5 if scale is None else scale
        if cu_seqlens is None:
            spans = [(n, 0, total) for n in range(batch)]
        else:
            bounds = cu_seqlens.tolist()
            spans = [(0, bounds[n], bounds[n + 1]) for n in range(len(bounds) - 1)]
        a3, b3 = (a, b) if a.dim() == 3 else (a.unsqueeze(0), b.unsqueeze(0))
        out = torch.zeros(batch, total, hv, v_dim, device=q.device, dtype=torch.float32)
        pool = None if initial_state_source is None else initial_state_source.clone()
        neg_exp_a = -torch.exp(A_log.float())
        slots = initial_state_indices.tolist()
        for n, (row, bos, eos) in enumerate(spans):
            slot = slots[n]
            if pool is not None and slot >= 0:
                state = initial_state_source[slot].float().transpose(1, 2)
            else:
                state = torch.zeros(hv, k_dim, v_dim, device=q.device, dtype=torch.float32)
            for t in range(bos, eos):
                q_t = q[row, t].float().repeat_interleave(repeat, dim=0)
                k_t = k[row, t].float().repeat_interleave(repeat, dim=0)
                x = a3[row, t].float() + dt_bias.float()
                bx = softplus_beta * x
                softplus = torch.where(
                    bx <= softplus_threshold, torch.log(1.0 + torch.exp(bx)) / softplus_beta, x
                )
                beta = torch.sigmoid(b3[row, t].float())
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / (q_t.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
                    k_t = k_t / (k_t.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
                state = state * torch.exp(neg_exp_a * softplus)[:, None, None]
                u = (v[row, t].float() - (state * k_t[:, :, None]).sum(1)) * beta[:, None]
                state = state + k_t[:, :, None] * u[:, None, :]
                out[row, t] = (state * (q_t * scale)[:, :, None]).sum(1)
            if pool is not None and slot >= 0:
                pool[slot] = state.transpose(1, 2).to(pool.dtype)
        return out.to(q.dtype if output is None else output.dtype), pool

    def is_valid(
        self,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        softplus_beta: float,
        softplus_threshold: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        initial_state_source: Optional[torch.Tensor],
        initial_state_indices: torch.Tensor,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> None:
        # Measured on the Triton branch: a last-dim slice of a doubly wide
        # buffer and a token-strided view were both accepted, came back 5.3e-2
        # and 4.8e-2 off at an output absmax of 4.1e-2, and overwrote elements
        # outside the view.
        assert output is None or output.is_contiguous(), (
            f"output must be contiguous; got strides {output.stride()} for shape "
            f"{tuple(output.shape)} -- the kernel writes it with packed "
            "[NK, B, T, HV, V] arithmetic, misplacing values and clobbering the gaps"
        )


fused_sigmoid_gating_delta_rule_update = FusedSigmoidGatingDeltaRuleUpdate()
