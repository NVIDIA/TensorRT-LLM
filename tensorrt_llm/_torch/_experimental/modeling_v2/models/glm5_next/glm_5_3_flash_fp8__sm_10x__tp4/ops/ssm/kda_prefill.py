# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill: the gated delta rule over packed raw q / k / v from each sequence's resident state row, the
final state written back to the row.

Copied from kimi_k3_vl's entry (``models/kimi_k3_vl/ops/ssm/kda_prefill.py``), which wraps the same
``trtllm::kda_prefill`` op. This entry adds the op's ``sm_reserve`` argument, and its reference returns the state
pool as the call leaves it (each final state rounded once to the pool dtype) where kimi_k3_vl's writes the pool in
place.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops  # noqa: F401 -- registers trtllm::kda_prefill
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: KDA's K = V.
_HEAD_DIM = 128
#: Tokens per BT16 chunk; ``cu_chunks`` counts these.
_CHUNK = 16


class KdaPrefill(OpWrapper):
    """Run the gated delta rule over each sequence from ``state_pool[state_indices[b]]``; return ``o``.

    Per token, with q and k L2-normalized per head (q then scaled), ``beta = sigmoid(beta_logit)`` and the
    per-key decay ``exp(lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias)))``:
    ``S <- S * decay``, ``S <- S + beta * (v - S k) k^T``, ``o = S q``, with ``S`` the V-first ``[H, V, K]``
    state. The final state of sequence b is written back to its row, rounded to the pool dtype.

    Args:
        q, k, v: contiguous bf16 ``[1, T, H, 128]`` raw (post-conv, unnormalized) inputs.
        g: contiguous bf16 ``[1, T, H, 128]`` raw forget-gate logits (before ``dt_bias``).
        beta: bf16 ``[1, T, H]`` beta logits; any token stride with a unit head stride.
        state_pool: bf16 or fp32 ``[slots, H, 128, 128]`` V-first pool, updated in place; the slot stride is free
            (16-byte multiple), rows dense.
        state_indices: int32 ``[B]`` unique slots, one per sequence; a fresh sequence's row must be zeroed first.
        cu_seqlens: int64 ``[B + 1]`` packed token offsets (``query_start_loc_long``).
        A_log: contiguous fp32 ``[H]``; dt_bias: contiguous fp32 ``[H * 128]``.
        scale: q scale (``128 ** -0.5``); lower_bound: the gate bound (finite, negative).
        mode: ``"auto"``, ``"bt16"``, ``"engine"``, ``"decomp"`` or ``"small-bh"`` (the kernel route).
        max_seqlen: zero, or a bound on every sequence length.
        seq_order: optional int32 ``[B]`` stable descending-length order of the sequences.
        cu_chunks: optional int32 ``[B + 1]`` prefix sums of ``ceil(length / 16)`` in request order; given
            together with ``seq_order`` or not at all.
        sm_reserve: SMs kept out of the Small-BH persistent grid.

    Returns:
        A new contiguous bf16 ``[1, T, H, 128]`` output.
    """

    ARCHS = frozenset({Arch.SM_100})
    # GLM-5.3-Flash's KDA at TP4 (16 of 64 heads). The op's own edges; the prefill steps the shipped target runs
    # (engine route, scheduler-planned metadata) are in this entry's test.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="auto route without planned metadata: on-device schedule, ragged lengths, empty sequence",
            spec=dict(
                heads=16,
                seq_lens=(100, 0, 37, 300),
                slots=(7, 2, 10, 4),
                fresh=(True, False, True, False),
                state_dtype=torch.float32,
                mode="auto",
                planned=False,
                pool=12,
            ),
        ),
        Cell(
            why="engine route with planned metadata on a bf16 pool: chunk edges 15 / 16 / 17 / 63 / 65",
            spec=dict(
                heads=16,
                seq_lens=(15, 16, 17, 63, 65),
                slots=(0, 3, 5, 8, 11),
                fresh=(True, False, True, False, True),
                state_dtype=torch.bfloat16,
                mode="engine",
                planned=True,
                pool=12,
            ),
        ),
    )
    note = """
    Q/K normalization (x / max(|x|, 1e-12)), gate and beta activation are fused;
    the convolution and the output norm are not. The kernel evaluates 16-token
    chunks with bf16 tensor-core operands and fp32 accumulation, the state held in
    fp32 across the call and rounded to the pool dtype once at the end.

    The caller zeroes fresh rows (reset_recurrent_state_rows) before the call:
    every selected row is read as the initial state. Only selected rows are
    written; other slots and slot padding are untouched. Each slot appears once.

    Routes: auto and bt16 pick the engine or the chunk decomposition by
    occupancy (decomposition iff 2 * B * H fits the SMs); engine and decomp force
    one. small-bh is opt-in only and needs a bf16 pool; with one it diverges on
    long sequences mixing near-unit and strong decay. max_seqlen and sm_reserve
    feed the Small-BH route only (and the argument checks). seq_order and
    cu_chunks, when given, replace the schedule the kernel otherwise builds on
    the device; they must be refreshed when the lengths change. None of the
    scheduling arguments change the recurrence.

    The first call per (pool dtype, bound, route) compiles; warm it outside
    CUDA-graph capture. Inputs must not alias the pool (the op checks).
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        lower_bound: float,
        mode: str,
        max_seqlen: int,
        seq_order: torch.Tensor | None,
        cu_chunks: torch.Tensor | None,
        sm_reserve: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            state_pool,
            state_indices,
            cu_seqlens,
            A_log,
            dt_bias,
            scale,
            lower_bound,
            mode,
            max_seqlen,
            seq_order,
            cu_chunks,
            sm_reserve,
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        lower_bound: float,
        mode: str,
        max_seqlen: int,
        seq_order: torch.Tensor | None,
        cu_chunks: torch.Tensor | None,
        sm_reserve: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(o, state_pool as the call leaves it)``; mutates nothing.

        The token-by-token recurrence in fp32 (elementwise products and sums, no matmul), each final state
        rounded once to the pool dtype and ``o`` once to bf16. ``mode``, ``max_seqlen``, ``seq_order``,
        ``cu_chunks`` and ``sm_reserve`` schedule the kernel and are ignored.
        """
        heads = q.shape[2]
        qf = torch.nn.functional.normalize(q[0].float(), dim=-1, eps=1e-12) * scale
        kf = torch.nn.functional.normalize(k[0].float(), dim=-1, eps=1e-12)
        vf = v[0].float()
        rate = A_log.float().exp().reshape(heads, 1)
        gate = lower_bound * torch.sigmoid(
            rate * (g[0].float() + dt_bias.float().reshape(heads, -1))
        )
        decay = gate.exp()  # [T, H, K]
        bf = beta[0].float().sigmoid()  # [T, H]
        out = torch.zeros_like(vf)
        pool = state_pool.clone()
        bounds = cu_seqlens.tolist()
        for seq, slot in enumerate(state_indices.tolist()):
            state = state_pool[slot].float()  # [H, V, K]
            for token in range(bounds[seq], bounds[seq + 1]):
                state = state * decay[token].unsqueeze(-2)
                key = kf[token].unsqueeze(-2)  # [H, 1, K]
                err = vf[token] - (state * key).sum(-1)  # [H, V]
                state = state + (bf[token].unsqueeze(-1) * err).unsqueeze(-1) * key
                out[token] = (state * qf[token].unsqueeze(-2)).sum(-1)
            pool[slot] = state.to(pool.dtype)
        return out.unsqueeze(0).to(v.dtype), pool

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        lower_bound: float,
        mode: str,
        max_seqlen: int,
        seq_order: torch.Tensor | None,
        cu_chunks: torch.Tensor | None,
        sm_reserve: int,
    ) -> None:
        # The op checks dtypes, shapes, strides, alignment and aliasing; it never reads the metadata values on the
        # host, and these are the ones it takes and gets wrong.
        bounds = cu_seqlens.tolist()
        lengths = [hi - lo for lo, hi in zip(bounds, bounds[1:])]
        assert bounds[0] == 0 and bounds[-1] == q.shape[1] and min(lengths) >= 0, (
            "cu_seqlens must monotonically partition the packed tokens"
        )
        slots = state_indices.tolist()
        assert len(slots) == len(set(slots)), (
            "state_indices must name one unique slot per sequence: two sequences on a slot race"
        )
        assert all(0 <= slot < state_pool.shape[0] for slot in slots), "state slot out of range"
        assert max_seqlen == 0 or max(lengths) <= max_seqlen, (
            "max_seqlen is below a sequence length"
        )
        if seq_order is not None and cu_chunks is not None:
            assert seq_order.tolist() == sorted(
                range(len(lengths)), key=lambda i: (-lengths[i], i)
            ), "seq_order must be the stable descending sequence-length order"
            chunks = [0]
            for length in lengths:
                chunks.append(chunks[-1] + (length + _CHUNK - 1) // _CHUNK)
            assert cu_chunks.tolist() == chunks, "cu_chunks must prefix-sum ceil(length / 16)"


kda_prefill = KdaPrefill()
