# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked gated delta rule on FlashInfer's Blackwell kernel: the GDN prefill core.

Wraps the in-tree module callable
``tensorrt_llm._torch.modules.fla.flashinfer_chunk.chunk_gated_delta_rule``
(tier-2 sourcing), the adapter in front of the installed FlashInfer's
``flashinfer.chunk_gated_delta_rule`` (CuTe-DSL tcgen05 kernel on sm_100/103).
One call is: ``g -> exp(g)``, a Triton gather of the indexed pool slots into a
packed fp32 ``[N, HV, V, K]`` buffer, the FlashInfer kernel, and a Triton
scatter of the final states back into the same slots. Call-compatible with the
vendored Triton ``chunk_gated_delta_rule`` (``modules/fla/chunk``).

FlashInfer API: ``flashinfer.chunk_gated_delta_rule`` (``flashinfer.gdn_prefill``),
reached through the adapter above.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.fla.flashinfer_chunk import (
    chunk_gated_delta_rule as _fi_chunk_gated_delta_rule,
)

#: qwen3.8-flash-next-nvfp4's Gated DeltaNet at tp1: (H, HV, K, V) -- 16 q/k
#: heads, 48 v heads, head dims 128.
QWEN38_GEOMETRY = (16, 48, 128, 128)
#: Largest slot count the cache manager was seen to allocate the recurrent pool
#: at for qwen3.8-flash-next-nvfp4 (10 .. 461 over twelve runtime-value dumps).
ENGINE_SLOTS = 461


def _engine_slots(num_seqs: int) -> tuple[int, ...]:
    """`num_seqs` distinct slots of the engine pool, slot 0 and the last live."""
    return ((0, ENGINE_SLOTS - 1) + tuple(i * 13 + 4 for i in range(num_seqs)))[:num_seqs]


class FlashinferChunkGatedDeltaRule(OpWrapper):
    """Run the gated delta rule over whole sequences, returning `(o, final_state)`.

    Per token `t` and value head, `h` the `[V, K]` state, `q_h`/`k_h` the key
    heads broadcast (value head `i` reads key head `i // (HV // H)`), fp32
    accumulation:

        h *= exp(g[t]);  u = (v[t] - h k_h[t]) * beta[t];  h += u k_h[t]^T
        o[t] = h (scale * q_h[t])

    computed 64 tokens at a time on tensor cores (WY form, bf16 operands) --
    the same recurrence as the Triton op, a different sum. `g` is in log space
    (the adapter exponentiates it); `scale=None` is `K ** -0.5`;
    `use_qk_l2norm_in_kernel` normalizes q/k first (a Triton `l2norm_fwd`).

    One packed row (`q.shape[0] == 1`) split at `cu_seqlens` (required).
    `initial_state` is required: a pool `[S, HV, V, K]`; sequence `n` starts
    from `initial_state[initial_state_indices[n]]`. With
    `inplace_indexed_state_update=True` its final state is written back into
    that slot and `final_state` is `None`; otherwise `output_final_state`
    returns a fresh `[N, HV, V, K]` in the pool's dtype and the pool is not
    written.

    Fusion boundary: gate exponentiation, optional q/k norm, scale, slot
    gather, recurrence, output, slot scatter. The conv, q/k/v/g/beta
    preparation, slot allocation and the gated norm on `o` are the caller's.

    Returns `(o, final_state)`: `o` `[1, T, HV, V]` in `q.dtype` (`output`
    itself when given).
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: geometry (H, HV, K, V); seqlens; `slots`-row pool with `indices`
    # (or _engine_slots); output (None | "buffer"). Every cell is
    # qwen3.8-flash-next-nvfp4's geometry, in place over an fp32 pool.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"qwen3.8 prefill into output=, 461-slot pool, seqlens {list(s)}",
            spec=dict(
                geometry=QWEN38_GEOMETRY,
                seqlens=s,
                slots=ENGINE_SLOTS,
                indices=_engine_slots(len(s)),
                output="buffer",
            ),
        )
        for s in ((200, 37, 91), (8192,))
    ) + (
        Cell(
            why="qwen3.8 context chunk + 1-token rows [8, 1, 1, 1], 461-slot pool",
            spec=dict(
                geometry=QWEN38_GEOMETRY,
                seqlens=(8, 1, 1, 1),
                slots=ENGINE_SLOTS,
                indices=_engine_slots(4),
            ),
        ),
    )

    note = """
    Metadata consumed: none.

    Call site as qwen3.8-flash-next-nvfp4's `_gdn_block` makes it (in place of
    the Triton `chunk_gated_delta_rule`):
    the same positional arguments, `(q, k, v, g, beta, None, ssm_states,
    si, True, False, qsl_long, False, False, None)` with `head_first=False`
    in the 12th slot, after `fused_gdn_post_conv` (q/k already normalized,
    contiguous `[1, T, H, 128]`; `g` fp32 log-space; `beta` fp32). Give it
    context rows only: single-token rows are slow here. Measured per layer
    at 8065 + 127 x 1 tokens: 1078.8 us with every row in this op, 399.4 us
    with the 127 decode rows sent to FlashInfer's decode kernel
    (`gdn_decode.gated_delta_rule_decode_pretranspose`) instead (both halves
    plus the concatenation).

    Certified at the 461-slot fp32 pool: three ragged context rows and one
    8192-token row, each into an `output=` buffer, and an 8-token context
    chunk with three one-token rows.

    Memory per call: a packed fp32 copy of the named slots in and out (2 x
    3 MiB per sequence), `exp(g)`, and the output. q/k/v/beta are not copied
    when already contiguous / fp32.

    JIT and streams: the first call per process compiles the CuTe-DSL
    kernel in-process (seconds), and the installed FlashInfer keeps one TMA
    workspace per device, so two streams must not run this op concurrently.
    Needs CUDA >= 13 torch (FlashInfer raises otherwise).

    Numerics (measured vs an fp64 recurrence, 1 x 8192 tokens, qwen3.8
    geometry): o rel-L2 4.06e-3, max abs 1.26e-3; state rel-L2 3.31e-3 --
    the Triton op's are
    4.00e-3 / 1.13e-3 / 3.26e-3. Not bitwise to it. No cold-call defect:
    nothing in the call is autotuned.

    Loud, from the callable: `head_first=True`, `q.shape[0] != 1`, H_q != H_k,
    fp32 q, non-fp32 `g`, `cu_seqlens` or `initial_state` None, in-place
    without indices.

    Not checked: indices distinct and in `[0, S)`; q/k unit-norm (by the
    caller or the flag).

    Not certified: `state_workspace=`, bf16/fp16 pool, `scale` other than the
    default, fp16 operands, K or V other than 128, other head ratios, and what the
    target never passes: `inplace_indexed_state_update=False` (final states
    returned) and `use_qk_l2norm_in_kernel`.

    Hard limits
      none probed.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        initial_state_indices: Optional[torch.Tensor] = None,
        inplace_indexed_state_update: bool = False,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        head_first: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        output: Optional[torch.Tensor] = None,
        state_workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return _fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            inplace_indexed_state_update=inplace_indexed_state_update,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output=output,
            state_workspace=state_workspace,
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        initial_state_indices: Optional[torch.Tensor] = None,
        inplace_indexed_state_update: bool = False,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        head_first: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        output: Optional[torch.Tensor] = None,
        state_workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """`(o, final_state, initial_state as the call leaves it)`.

        fp64 token-at-a-time recurrence, rounded once: `o` to `q.dtype`, the
        states to the pool's dtype. `final_state` is `None` on the in-place
        path, as the callable returns it. Mutates nothing; `output`,
        `head_first` and `state_workspace` are ignored (they choose where
        results land)."""
        dt = torch.float64
        _, _, num_k_heads, head_k = q.shape
        num_v_heads = v.shape[2]
        repeats = num_v_heads // num_k_heads
        scale = head_k**-0.5 if scale is None else scale
        bounds = cu_seqlens.tolist()
        slots = initial_state_indices.tolist()
        out = torch.zeros(1, q.shape[1], num_v_heads, v.shape[3], dtype=dt, device=q.device)
        finals = []
        for n in range(len(bounds) - 1):
            state = initial_state[slots[n]].to(dt)
            for t in range(bounds[n], bounds[n + 1]):
                q_t, k_t = q[0, t].to(dt), k[0, t].to(dt)
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / torch.sqrt(q_t.pow(2).sum(-1, keepdim=True) + 1e-6)
                    k_t = k_t / torch.sqrt(k_t.pow(2).sum(-1, keepdim=True) + 1e-6)
                q_t = q_t.repeat_interleave(repeats, dim=0) * scale
                k_t = k_t.repeat_interleave(repeats, dim=0)
                state = state * g[0, t].to(dt).exp()[:, None, None]
                u = (v[0, t].to(dt) - (state * k_t[:, None, :]).sum(-1)) * beta[0, t].to(dt)[
                    :, None
                ]
                state = state + u[:, :, None] * k_t[:, None, :]
                out[0, t] = (state * q_t[:, None, :]).sum(-1)
            finals.append(state)
        pool = initial_state.clone()
        final_state = None
        if inplace_indexed_state_update:
            for slot, state in zip(slots, finals):
                pool[slot] = state.to(pool.dtype)
        elif output_final_state:
            final_state = torch.stack(finals).to(initial_state.dtype)
        return out.to(q.dtype), final_state, pool


flashinfer_chunk_gated_delta_rule = FlashinferChunkGatedDeltaRule()
