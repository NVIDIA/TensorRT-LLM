# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked gated delta rule: the Gated DeltaNet prefill core.

Wraps the in-tree module callable
``tensorrt_llm._torch.modules.fla.chunk.chunk_gated_delta_rule`` -- the
**vendored Triton** implementation (tier-2 sourcing: the gated delta rule has
no ``torch.ops.trtllm`` surface). Its interior is a fixed sequence of Triton
steps (gate cumsum, scaled K K^T, triangular solve, WY recompute, chunked state
recurrence, output epilogue) no caller invokes on its own: one fused op.

The GDN mixer resolves between this and a FlashInfer-backed sibling of the same
name (``tensorrt_llm._torch.modules.fla.flashinfer_chunk``) through
``gdn_mixer._resolve_chunk_gated_delta_rule``, which picks FlashInfer on sm_103
unless ``TLLM_USE_FLASHINFER_GDN_PREFILL=0``. This entry names the Triton one
directly, so neither the resolver nor the variable changes what it runs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as _chunk_gated_delta_rule

#: This checkpoint's Gated DeltaNet geometry at tp1.
NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM = 16, 48, 128, 128
#: Largest slot count the cache manager was seen to allocate the recurrent pool
#: at for this checkpoint (10, 35, 366, 368, 396, 399, 432, 461 over twelve
#: runtime-value dumps; the batch never above 32). The target passes the
#: layer's whole pool unsliced.
ENGINE_SLOTS = 461


def _engine_slots(num_seqs: int) -> tuple[int, ...]:
    """`num_seqs` distinct slots of the engine pool, slot 0 and the last live."""
    return ((0, ENGINE_SLOTS - 1) + tuple(i * 13 + 4 for i in range(num_seqs)))[:num_seqs]


class ChunkGatedDeltaRule(OpWrapper):
    """Run the gated delta rule over whole sequences, returning `(o, final_state)`.

    Per token `t` and value head, `h` the `[V, K]` state, `q_h`/`k_h` the key
    heads broadcast by `repeat_interleave` (value head `i` reads key head
    `i // (H // Hg)`), fp32 accumulation:

        h *= exp(g[t]);  u = (v[t] - h k_h[t]) * beta[t];  h += u k_h[t]^T
        o[t] = h (scale * q_h[t])

    computed 64 tokens at a time over a WY representation rounded to
    `q.dtype` -- the same recurrence, a different sum. `scale=None` is
    `K ** -0.5`; `g` is in log space. `use_qk_l2norm_in_kernel` normalizes q/k
    as `x / sqrt(sum(x^2) + 1e-6)` first.

    With `cu_seqlens` the batch is one packed row (`q.shape[0] == 1`) split at
    those offsets; without it each row is a sequence. Initial state: zeros
    (`initial_state=None`); `initial_state[n]`, read-only; or, with
    `initial_state_indices` and `inplace_indexed_state_update=True`,
    `initial_state[indices[n]]` with the final state **written back into that
    slot**. `output_final_state` only decides whether a fresh fp32
    `[N, H, V, K]` final state is also returned (on the indexed path, a copy
    gathered after the write-back).

    Fusion boundary: gate cumsum, optional q/k norm, scale, recurrence, state
    read and write, output epilogue. The conv, q/k/v/g/beta preparation, slot
    allocation and the gated norm on `o` are the caller's.

    Returns `(o, final_state)` unchanged: `o` `[B, T, H, V]` in `q.dtype`
    (`output` itself when given), `final_state` or `None`. The pool write is in
    place and not returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: seqlens (packed with cu_seqlens) or batch/num_tokens (no
    # cu_seqlens); initial: "indexed" (a `slots`-row pool, `indices` or arange)
    # | "positional" | None; output: None | "buffer" | "window" (wide[:, 50:146]
    # of a 200-token buffer); output_final_state; scale; normalize_qk (False
    # with l2norm feeds raw q/k).
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"target prefill call, 461-slot pool, seqlens {list(s)}",
                spec=dict(
                    seqlens=s, initial="indexed", slots=ENGINE_SLOTS, indices=_engine_slots(len(s))
                ),
            )
            for s in ((1,), (64,), (200, 37, 91), (1024,))
        )
        + (
            Cell(
                why="461-slot pool, output= buffer (the stock mixer's call)",
                spec=dict(
                    seqlens=(200, 37, 91),
                    initial="indexed",
                    slots=ENGINE_SLOTS,
                    indices=_engine_slots(3),
                    output="buffer",
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"indexed, seqlens {list(s)}, {len(s) + 3}-slot pool, output= buffer",
                spec=dict(seqlens=s, initial="indexed", slots=len(s) + 3, output="buffer"),
            )
            for s in ((1,), (8,), (64,), (137,), (128, 64), (200, 37, 91), (8, 8, 8), (1024,))
        )
        + tuple(
            Cell(
                why=f"positional initial state, final state out, {list(s)}",
                spec=dict(seqlens=s, initial="positional", output_final_state=True),
            )
            for s in ((96, 48), (333,))
        )
        + (
            Cell(
                why="initial_state None: zero start",
                spec=dict(seqlens=(77,), initial=None, output_final_state=True),
            ),
            Cell(
                why="use_qk_l2norm_in_kernel on raw q/k",
                spec=dict(
                    seqlens=(100,),
                    initial=None,
                    output_final_state=True,
                    l2norm=True,
                    normalize_qk=False,
                ),
            ),
            Cell(
                why="no cu_seqlens: two equal-length rows",
                spec=dict(batch=2, num_tokens=96, initial="positional", output_final_state=True),
            ),
            Cell(
                why="explicit scale 0.05 replaces the default",
                spec=dict(seqlens=(80,), initial=None, scale=0.05),
            ),
            Cell(
                why="output= a token-window slice of a wider buffer",
                spec=dict(seqlens=(96,), initial="positional", output="window"),
            ),
        )
    )

    note = """
    Metadata consumed: none.

    Call site (`_gdn_block`): `(q, k, v, g, beta, None, ssm_states, si_all,
    True, False, qsl_long, False, None)` after `fused_gdn_post_conv` (q/k
    already normalized) -- the layer's whole fp32 pool `[slots, 48, 128, 128]`,
    int32 slots, int64 `cu_seqlens`; the MTP-verify context half passes the
    `si_p` / `qsl_long_p` prefixes. Certified at small pools and at 461 slots;
    at 461 output and state are bitwise identical to a 6-slot pool with the
    same slot content.

    The first indexed in-place call for an autotune key in a process is as correct as a
    warm one: the state kernel's autotuner snapshots and restores the indexed state slots
    its benchmark launches write (`chunk_delta_h.py`, `_save_autotune_state`). Certified
    in a fresh interpreter at this geometry and at 5 / 20 heads. The positional path
    never aliased its state.

    Loud, from the callable: fp32 or mixed-dtype q/k/v, a 2-D `beta`,
    `q.shape[0] != 1` with `cu_seqlens`, a sequence-count mismatch of indices
    or initial states, indices without `inplace_indexed_state_update`.

    Not checked: slots must not alias (not probed); q/k must be unit-norm,
    by the caller or the in-kernel flag (raw q/k with the flag off return
    finite values ~1e25). `head_first` is pinned False.

    Sequence lengths are unconstrained: 1..1024, off and on the 64 boundary,
    mixed ragged, all correct. Memory: a `[B, NT, H, K, V]` per-chunk state
    scratch in `q.dtype`, 24 MiB per 1024 tokens here. Deterministic.

    Not certified: the FlashInfer sibling, fp16 (accepted), bf16 `g`/`beta`/
    pool, `K = 256` (accepted, unmeasured), empty `T` or zero-length spans.

    Hard limits
      K  state kernel asserts K <= 256 (read, not probed)
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
        use_qk_l2norm_in_kernel: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return _chunk_gated_delta_rule(
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
            head_first=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output=output,
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
        use_qk_l2norm_in_kernel: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """`(o, final_state, initial_state as the call leaves it)`.

        fp64 token-at-a-time recurrence, rounded once: `o` to `q.dtype`, the
        states to fp32 / the pool's dtype. Mutates nothing. `output` is ignored
        (it only chooses where `o` lands).
        """
        dt = torch.float64
        batch, num_tokens, num_k_heads, head_k = q.shape
        num_v_heads, head_v = v.shape[2], v.shape[3]
        repeats = num_v_heads // num_k_heads
        scale = head_k**-0.5 if scale is None else scale
        if cu_seqlens is None:
            spans = [(b, 0, num_tokens) for b in range(batch)]
        else:
            bounds = cu_seqlens.tolist()
            spans = [(0, bounds[n], bounds[n + 1]) for n in range(len(bounds) - 1)]
        slots = None if initial_state_indices is None else initial_state_indices.tolist()
        out = torch.zeros(batch, num_tokens, num_v_heads, head_v, dtype=dt, device=q.device)
        finals = []
        for n, (row, bos, eos) in enumerate(spans):
            if initial_state is None:
                state = torch.zeros(num_v_heads, head_v, head_k, dtype=dt, device=q.device)
            else:
                state = initial_state[n if slots is None else slots[n]].to(dt)
            for t in range(bos, eos):
                q_t, k_t = q[row, t].to(dt), k[row, t].to(dt)
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / torch.sqrt(q_t.pow(2).sum(-1, keepdim=True) + 1e-6)
                    k_t = k_t / torch.sqrt(k_t.pow(2).sum(-1, keepdim=True) + 1e-6)
                q_t = q_t.repeat_interleave(repeats, dim=0) * scale
                k_t = k_t.repeat_interleave(repeats, dim=0)
                state = state * g[row, t].to(dt).exp()[:, None, None]
                u = (v[row, t].to(dt) - (state * k_t[:, None, :]).sum(-1)) * beta[row, t].to(dt)[
                    :, None
                ]
                state = state + u[:, :, None] * k_t[:, None, :]
                out[row, t] = (state * q_t[:, None, :]).sum(-1)
            finals.append(state)
        final_state = torch.stack(finals).float() if output_final_state else None
        pool = None if initial_state is None else initial_state.clone()
        if slots is not None and inplace_indexed_state_update:
            for slot, state in zip(slots, finals):
                pool[slot] = state.to(pool.dtype)
        return out.to(q.dtype), final_state, pool

    def is_valid(
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
        use_qk_l2norm_in_kernel: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> None:
        if output is None:
            return
        # Measured: a head-slice view (strides (1179648, 12288, 128, 1)) was
        # accepted and came back 9.9e-2 off where correct is ~4e-4; an fp32
        # buffer was filled correctly but the returned bf16 tensor no longer
        # aliased it. A token-window slice that reports contiguous is fine.
        assert output.is_contiguous(), (
            f"output must be contiguous; got strides {tuple(output.stride())} -- the "
            "epilogue addresses it through strides derived from its shape"
        )
        assert output.dtype == q.dtype, (
            f"output must have q's dtype {q.dtype}; got {output.dtype} -- the result "
            "is cast to q.dtype and stops aliasing the buffer"
        )


chunk_gated_delta_rule = ChunkGatedDeltaRule()
