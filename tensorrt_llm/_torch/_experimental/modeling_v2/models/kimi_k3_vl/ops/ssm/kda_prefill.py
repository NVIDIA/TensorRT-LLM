# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked KDA (Kimi Delta Attention) prefill, reading and writing each sequence's recurrent state in place."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops  # noqa: F401 — registers the op
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class KdaPrefill(OpWrapper):
    """Run the gated delta rule over each sequence from `state_pool[state_indices[b]]`; return `o`.

    The final state of sequence b is written back to `state_pool[state_indices[b]]` (fp32, V-first
    `[slots, H, V, K]`). Returns a new `o` shaped like `v`.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"TP{96 // heads} ragged varlen with {cu_dtype}, scattered padded pool slots",
                spec=dict(
                    heads=heads,
                    seq_lens=(100, 64, 300, 37),
                    slots=(7, 2, 10, 4),
                    fresh=(True, False, True, False),
                    cu_dtype=cu_dtype,
                    batch_rows=1,
                ),
            )
            for heads, cu_dtype in ((6, torch.int64), (24, torch.int64), (6, torch.int32))
        ),
        Cell(
            why="two equal-length 256-token sequences without cumulative sequence metadata",
            spec=dict(
                heads=6,
                seq_lens=(256, 256),
                slots=(1, 9),
                fresh=(True, True),
                cu_dtype=torch.int64,
                batch_rows=2,
            ),
        ),
    )
    note = """
    Caller-owned fp32 V-first recurrent pool, possibly padded between slots.
    Only state_indices rows are written; fresh rows must be zeroed by the
    caller. Continuation chunks must run in order on the same slot. The tests
    compare 256 + 256 against 512 tokens and a swapped-order negative control.
    Returned outputs are newly allocated and later calls leave them untouched.
    The runner caches compiled kernels and intermediate scratch process-wide,
    not in a caller-owned object.

    cu_seqlens, chunk_indices and state_indices plus varlen_is_aligned and
    single_sequence_length are caller-built batch metadata. A single varlen
    sequence is caller-padded to a 64-token multiple while cu_seqlens retains
    its real length. q/k are L2-normalized bf16; v/g are bf16, beta/A_log/dt_bias
    fp32, K=V=128. Kimi K3 uses scale=128**-0.5, safe_gate=True, lower_bound=-5,
    gate and beta activation enabled and int64 cumulative metadata.
    Projections, convolution, q/k normalization and gated output norm are not
    fused. The layer sends four or more chunks here and smaller batches to
    chunk_kda; bf16 pools are staged through dense fp32 rows here.
    CuTe DSL and FlashInfer are required for registration, and SM 10.x for
    execution. The bf16 chunked evaluation carries fp32 state; tests use bands
    of 2e-2 for outputs and 1e-2 for states against the recurrence.
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
        scale: float,
        cu_seqlens: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
        safe_gate: bool = False,
        lower_bound: Optional[float] = None,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = True,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        varlen_is_aligned: Optional[bool] = None,
        single_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            state_pool,
            state_indices,
            scale,
            cu_seqlens,
            chunk_indices,
            chunk_size,
            safe_gate,
            lower_bound,
            use_gate_in_kernel,
            use_beta_sigmoid_in_kernel,
            A_log,
            dt_bias,
            varlen_is_aligned,
            single_sequence_length,
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
        scale: float,
        cu_seqlens: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
        safe_gate: bool = False,
        lower_bound: Optional[float] = None,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = True,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        varlen_is_aligned: Optional[bool] = None,
        single_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        qf, kf, vf = q.float() * scale, k.float(), v.float()
        gate = g.float()
        if dt_bias is not None:
            gate = gate + dt_bias.float().reshape(q.shape[-2:])
        rate = A_log.float().exp().reshape(-1, 1)
        gate = (
            (-5.0 if lower_bound is None else lower_bound) * torch.sigmoid(rate * gate)
            if safe_gate
            else -rate * torch.nn.functional.softplus(gate)
        )
        decay = gate.exp()
        bf = beta.float().sigmoid() if use_beta_sigmoid_in_kernel else beta.float()
        if cu_seqlens is None:
            spans = [(row, 0, q.shape[1]) for row in range(q.shape[0])]
        else:
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
        scale: float,
        cu_seqlens: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
        safe_gate: bool = False,
        lower_bound: Optional[float] = None,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = True,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        varlen_is_aligned: Optional[bool] = None,
        single_sequence_length: Optional[int] = None,
    ) -> None:
        assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128, (
            "KDA prefill kernels require K == V == 128"
        )
        assert all(
            tensor is None or tensor.device == q.device
            for tensor in (k, g, beta, A_log, dt_bias, cu_seqlens, chunk_indices)
        ), "all prefill inputs must share q's device"
        if cu_seqlens is not None and cu_seqlens.numel() == 2:
            length = int(cu_seqlens[-1])
            if q.shape[1] > length:
                assert all(
                    not bool(tensor[:, length:].count_nonzero()) for tensor in (q, k, v, beta)
                ), "single-sequence padding of q/k/v/beta must be zero"


kda_prefill = KdaPrefill()
