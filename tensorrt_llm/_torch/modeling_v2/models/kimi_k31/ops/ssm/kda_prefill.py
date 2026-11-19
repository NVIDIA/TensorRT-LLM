# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed raw-input KDA prefill (CuTe DSL BT16 chunked kernel) with indexed resident recurrent state."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops  # noqa: F401 — registers the op
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class KdaPrefill(OpWrapper):
    """Run the gated delta rule over each sequence from `state_pool[state_indices[b]]`; return `o`.

    The final state of sequence b is written back to `state_pool[state_indices[b]]` (FP32,
    V-first `[slots, H, V, K]`). Q/K L2 normalization, the lower-bound sigmoid gate and the beta
    sigmoid are fused. Returns a new bf16 `o` shaped like `v`.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{heads} KDA heads per rank: ragged packed prefill, fresh and continuation rows, fp32 pool",
                spec=dict(
                    heads=heads,
                    seq_lens=(100, 64, 300, 37),
                    slots=(7, 2, 10, 4),
                    fresh=(True, False, True, False),
                    state_dtype="float32",
                ),
            )
            for heads in (96, 48, 24, 12, 6, 3)
        ),
        Cell(
            why="one long chunked-prefill continuation (2048 tokens) at 96 heads",
            spec=dict(
                heads=96, seq_lens=(2048,), slots=(3,), fresh=(False,), state_dtype="float32"
            ),
        ),
        Cell(
            why="many short sequences (occupancy router's decomposition path) at 12 heads",
            spec=dict(
                heads=12,
                seq_lens=(5, 1, 16, 17, 33, 2, 64, 9),
                slots=(0, 1, 2, 3, 4, 5, 6, 7),
                fresh=(True, False, True, False, True, False, True, False),
                state_dtype="float32",
            ),
        ),
        Cell(
            why="sm_107 TEP8 context: a 4096-token chunk continuation at 12 heads",
            spec=dict(
                heads=12, seq_lens=(4096,), slots=(1,), fresh=(False,), state_dtype="float32"
            ),
        ),
    )
    note = """
    Raw contiguous bf16 q/k/v/g in packed [1, T, H, 128] storage (the fused
    convolution's outputs and the f_b projection); beta holds bf16 logits and
    may be the strided column view of the merged [f_a | b | pad] projection.
    int64 cu_seqlens partitions the tokens; int32 state_indices names unique
    resident pool rows (V-first, contiguous per slot; the slot stride may be
    padded but must be a multiple of 8 elements, which the compiled kernel
    checks and rejects otherwise). The caller zeroes fresh rows
    (reset_recurrent_state_rows) before the call; continuation rows carry
    the previous chunk's state. A_log [H] and dt_bias [H * 128] contiguous
    fp32. The target passes scale = 128**-0.5, lower_bound = -5, mode = "bt16"
    (pinned; "auto" selects the same engine for the fp32 pool), the
    metadata's kda_max_seqlen and its int32 seq_order (stable descending
    length order) and cu_chunks (prefix sums of ceil(L / 16)), shared by all
    layers of a step.

    Only selected rows are written; slot padding and other rows stay. The
    output is new. Chunked (16-token) evaluation with bf16 tensor-core
    products: within 2e-2 (output) / 1e-2 (state) of an fp32 recurrence,
    relative to the largest magnitude. Workspace and planned-metadata scratch
    are allocated per call from the torch allocator (graph pool under
    capture); compiled once per (state dtype, lower bound, mode), warm before
    capture. ARCHS lists only the architectures the cells were driven on:
    sm_103 (GB300) and sm_107 (VR200).
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
        mode: str = "auto",
        max_seqlen: int = 0,
        seq_order: Optional[torch.Tensor] = None,
        cu_chunks: Optional[torch.Tensor] = None,
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
        mode: str = "auto",
        max_seqlen: int = 0,
        seq_order: Optional[torch.Tensor] = None,
        cu_chunks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """fp32 token-by-token recurrence; `mode`, `max_seqlen`, `seq_order`, `cu_chunks` only schedule."""
        qf = torch.nn.functional.normalize(q.float(), dim=-1, eps=1e-12) * scale
        kf = torch.nn.functional.normalize(k.float(), dim=-1, eps=1e-12)
        vf = v.float()
        gate = g.float() + dt_bias.float().reshape(q.shape[-2:])
        rate = A_log.float().exp().reshape(-1, 1)
        decay = (lower_bound * torch.sigmoid(rate * gate)).exp()
        bf = beta.float().sigmoid()
        starts = cu_seqlens.tolist()
        out = torch.zeros_like(v, dtype=torch.float32)
        for slot, lo, hi in zip(state_indices.tolist(), starts, starts[1:]):
            state = state_pool[slot].float()
            for token in range(lo, hi):
                state = state * decay[0, token].unsqueeze(-2)
                key = kf[0, token]
                err = vf[0, token] - (state * key.unsqueeze(-2)).sum(-1)
                state = state + bf[0, token, :, None, None] * err.unsqueeze(-1) * key.unsqueeze(-2)
                out[0, token] = (state * qf[0, token].unsqueeze(-2)).sum(-1)
            state_pool[slot] = state.to(state_pool.dtype)
        return out.to(v.dtype)

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
        mode: str = "auto",
        max_seqlen: int = 0,
        seq_order: Optional[torch.Tensor] = None,
        cu_chunks: Optional[torch.Tensor] = None,
    ) -> None:
        starts = cu_seqlens.tolist()
        lengths = [hi - lo for lo, hi in zip(starts, starts[1:])]
        assert starts[0] == 0 and starts[-1] == q.shape[1] and min(lengths) >= 0, (
            "cu_seqlens must monotonically partition the packed tokens"
        )
        slots = state_indices.tolist()
        assert len(slots) == len(lengths) and len(set(slots)) == len(slots), (
            "state_indices must name one unique slot per sequence"
        )
        assert all(0 <= slot < state_pool.shape[0] for slot in slots), "state slot out of range"
        assert max_seqlen == 0 or max(lengths) <= max_seqlen, (
            "max_seqlen is below a sequence length"
        )
        if seq_order is not None:
            assert seq_order.tolist() == sorted(range(len(lengths)), key=lambda i: -lengths[i]), (
                "seq_order must be the stable descending sequence-length permutation"
            )
            chunks = [0]
            for length in lengths:
                chunks.append(chunks[-1] + (length + 15) // 16)
            assert cu_chunks.tolist() == chunks, "cu_chunks must prefix-sum ceil(length / 16)"


kda_prefill = KdaPrefill()
