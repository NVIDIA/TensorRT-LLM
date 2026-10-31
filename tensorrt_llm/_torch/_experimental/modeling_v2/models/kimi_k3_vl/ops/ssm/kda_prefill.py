# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed raw-input KDA prefill with indexed resident recurrent state."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops  # noqa: F401 — registers the op
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class KdaPrefill(OpWrapper):
    """Run the gated delta rule over each sequence from `state_pool[state_indices[b]]`; return `o`.

    The final state of sequence b is written back to `state_pool[state_indices[b]]` (BF16 or FP32,
    V-first `[slots, H, V, K]`). Q/K normalization and gate/beta activation are fused. Returns new `o`.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"TP{96 // heads} ragged packed prefill with {state_dtype} padded state rows",
                spec=dict(
                    heads=heads,
                    seq_lens=(100, 64, 300, 37),
                    slots=(7, 2, 10, 4),
                    fresh=(True, False, True, False),
                    state_dtype=state_dtype,
                ),
            )
            for heads, state_dtype in ((6, torch.float32), (24, torch.float32), (6, torch.bfloat16))
        ),
        Cell(
            why="equal-length packed sequences, including an empty continuation",
            spec=dict(
                heads=6,
                seq_lens=(256, 0, 256),
                slots=(1, 7, 9),
                fresh=(True, False, True),
                state_dtype=torch.float32,
            ),
        ),
    )
    note = """
    Raw contiguous BF16 q/k/v/g use packed [1,T,H,128] storage; beta contains
    BF16 logits. Mandatory int64 cu_seqlens partitions the packed tokens and
    int32 state_indices names unique resident BF16/FP32 V-first pool rows.
    The caller zeroes fresh rows; continuation calls reuse the same slot in
    order. A_log and dt_bias are contiguous FP32. Kimi K3 uses scale=128**-0.5
    and lower_bound=-5. Q/K normalization and gate/beta activation run inside
    the prefill kernel; projections, convolution and output norm do not.

    mode selects auto, bt16, small-bh, engine or decomp. max_seqlen is zero
    or an upper bound on each sequence length. Optional seq_order and cu_chunks
    are a stable descending length permutation and prefix sums of ceil(L/16)
    in request order; both must be refreshed when lengths change. These
    scheduling arguments do not change the reference recurrence.

    Only selected state rows are written; slot padding and unselected rows
    remain unchanged. Every output is newly allocated. The tests compare
    packed ragged and empty sequences, both resident state dtypes, ordered
    continuation and preplanned metadata against the FP32 recurrence, with
    output/state bands of 2e-2/1e-2. CuTe DSL and SM100 are required.
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
        qf = torch.nn.functional.normalize(q.float(), dim=-1, eps=1e-12) * scale
        kf = torch.nn.functional.normalize(k.float(), dim=-1, eps=1e-12)
        vf = v.float()
        gate = g.float() + dt_bias.float().reshape(q.shape[-2:])
        rate = A_log.float().exp().reshape(-1, 1)
        decay = (lower_bound * torch.sigmoid(rate * gate)).exp()
        bf = beta.float().sigmoid()
        starts = cu_seqlens.tolist()
        spans = [(0, lo, hi) for lo, hi in zip(starts, starts[1:])]
        out = torch.zeros_like(v, dtype=torch.float32)
        for slot, (row, lo, hi) in zip(state_indices.tolist(), spans):
            state = state_pool[slot].float()
            for token in range(lo, hi):
                state = state * decay[row, token].unsqueeze(-2)
                key = kf[row, token]
                err = vf[row, token] - (state * key.unsqueeze(-2)).sum(-1)
                state = state + (
                    bf[row, token, :, None, None] * err.unsqueeze(-1) * key.unsqueeze(-2)
                )
                out[row, token] = (state * qf[row, token].unsqueeze(-2)).sum(-1)
            state_pool[slot] = state
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
