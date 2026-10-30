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

#: A recurrent-pool slot count of a full-size engine run. The kernels address a slot by
#: its index alone, so a pool of another size (a bf16 pool holds more slots in the same
#: memory) takes the same path. The target passes the layer's whole pool unsliced.
ENGINE_SLOTS = 461


def _scattered(n_seqs: int) -> tuple[int, ...]:
    """`n_seqs` distinct engine-pool slots, both ends first, then a stride-13 walk over the
    interior (13 is coprime to its `ENGINE_SLOTS - 2` slots)."""
    rest = tuple((i * 13 + 4) % (ENGINE_SLOTS - 2) + 1 for i in range(max(n_seqs - 2, 0)))
    return ((0, ENGINE_SLOTS - 1) + rest)[:n_seqs]


class FusedSigmoidGatingDeltaRuleUpdate(OpWrapper):
    """Advance the gated delta rule over every sequence, gates computed in-kernel.

    Sequence `n` (tokens `cu_seqlens[n]:cu_seqlens[n+1]`, or row `n` without
    `cu_seqlens`), v-head `hv` reading k/q head `hv // (HV // H)`, state `h`
    `[K, V]` loaded from `initial_state_source[initial_state_indices[n]]`
    (stored `[V, K]`, K innermost), per token, all in fp32:

        g    = -exp(A_log) * softplus_b(a + dt_bias)    beta = sigmoid(b)
        q, k = q / sqrt(q.q + 1e-6), k / sqrt(k.k + 1e-6)  (only with use_qk_l2norm_in_kernel)
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

    # spec: seq_lens, slots (pool rows), indices (None -> arange), pool_dtype
    # (fp32), ab ("separate": two contiguous [T, 48] | "in_proj": q/k/v/a/b all
    # column views of one [T, 16480] projection, as the target's one input GEMM
    # makes them), layout ("varlen": [1, T, ...] with int64 cu_seqlens |
    # "rows": cu_seqlens None, one token per row, q/k/v [N, 1, heads, 128]
    # views of the packed [N, 10240] block whose token stride is the row
    # stride, and a/b [N, 1, 48] -- the target's bf16 call). In-kernel q/k
    # norm on, no output buffer.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="target bf16 decode call, rows form N=128, 461-slot pool",
            spec=dict(
                seq_lens=(1,) * 128,
                slots=ENGINE_SLOTS,
                indices=_scattered(128),
                ab="separate",
                layout="rows",
                pool_dtype=torch.bfloat16,
            ),
        ),
        Cell(
            why="bf16 rows form, pad rows (-1) between slot 0 and 460",
            spec=dict(
                seq_lens=(1, 1, 1, 1),
                slots=ENGINE_SLOTS,
                indices=(0, -1, ENGINE_SLOTS - 1, -1),
                ab="separate",
                layout="rows",
                pool_dtype=torch.bfloat16,
            ),
        ),
        Cell(
            why="fp32 pool, in_proj column views, N=128: the block test's reference",
            spec=dict(seq_lens=(1,) * 128, slots=128, ab="in_proj"),
        ),
    )

    note = """
    Metadata consumed: none.

    Call site (`_gdn_block` decode over a bf16 pool), one token per sequence,
    int32 slots, the layer's whole pool `[slots, 48, 128, 128]`: the rows form,
    `cu_seqlens` None -- q/k/v `[T, 1, heads, 128]` split from the whole
    `[T, 10240]` block reshaped to `[T, 1, 80, 128]`, so the token stride is the
    row stride (`is_valid` refuses other strides), `a`/`b` `[T, 1, 48]`. The
    cells drive it at 128 rows of a 461-slot pool, and with CUDA-graph pad rows
    (-1) at 4 rows. One more cell certifies the packed form the GDN block test
    uses as its fp32-pool reference: `(a_log, a, dt_bias, 1.0, 20.0, q, k, v,
    b, ssm_states, si_all, None, True, qsl_long, None)`, q/k/v `[1, T, heads,
    128]` and `a`/`b` column views of one `[T, 16480]` input-projection output
    (row stride 16480), int64 `cu_seqlens`, at 128 rows.

    Branch: the callable takes a FlashInfer CuTe-DSL kernel
    (`gdn_decode_bf16_state.gated_delta_rule`) when the pool is bf16,
    `cu_seqlens` is given and `T == N` (head dims 128, supported arch);
    everything else runs the vendored Triton kernel, and only the Triton
    branch is certified. The target's rows form (no `cu_seqlens`) and the reference's
    packed form over an fp32 pool both stay on it. The bf16 call takes
    the rows form because FlashInfer is not batch-invariant: at 48 value
    heads it runs one kernel up to N = 10 and another from N = 11, and one
    fixed row's single-step output and state differ between the two, so a
    row's tokens would depend on its batch. The Triton rows are one program
    per (row, head); the test shows a fixed row's output and state bitwise
    equal across batch sizes 1..32. FlashInfer also differs from `reference`
    in reading and writing a negative slot as slot 0 (Triton gives a zero
    state and no write) and in its L2-norm epsilon.

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
    operands, an fp16 pool, `softplus_beta`/`softplus_threshold` other than
    1.0/20.0, `B > 1` with `cu_seqlens`; the rows form over an fp32 pool or above
    128 rows; CUDA-graph replay against the eager call; the packed form over a
    bf16 pool, with more than one token per sequence, with an `output` buffer,
    with `l2norm` off, without a pool or on a layer view of one.

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
                    q_t = q_t / torch.sqrt(q_t.pow(2).sum(-1, keepdim=True) + 1e-6)
                    k_t = k_t / torch.sqrt(k_t.pow(2).sum(-1, keepdim=True) + 1e-6)
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
        if cu_seqlens is None:
            # q/k/v reshaped to [N, 1, heads, 128] from column slices of the packed
            # [N, 10240] block carry stride(1) = heads * 128, not the row stride: they are
            # accepted and read other rows' columns.
            for name, t in (("q", q), ("k", k), ("v", v), ("a", a), ("b", b)):
                assert t.dim() < 3 or t.stride(0) == t.shape[1] * t.stride(1), (
                    f"{name} {tuple(t.shape)} has strides {t.stride()}: without "
                    "cu_seqlens the kernel finds row n's tokens at n * T * stride(1), "
                    "so stride(0) must equal T * stride(1)"
                )


fused_sigmoid_gating_delta_rule_update = FusedSigmoidGatingDeltaRuleUpdate()
