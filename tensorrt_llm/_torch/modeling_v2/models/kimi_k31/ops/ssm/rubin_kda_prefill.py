# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA chunked prefill (CuTe DSL BT16) on Rubin: indexed resident state, optional state snapshots."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.ssm.rubin_kda_prefill_ops import KDA_PREFILL_SNAPSHOT_CHUNK
from ...kernels.ssm.rubin_kda_prefill_ops import kda_prefill as _kda_prefill

_RAGGED = dict(seq_lens=(100, 64, 300, 37), slots=(7, 2, 10, 4), fresh=(True, False, True, False))


class RubinKdaPrefill(OpWrapper):
    """Run the gated delta rule over each sequence from `state_pool[state_indices[b]]`; return `o`.

    Sequence b's final state goes back to `state_pool[state_indices[b]]` (BF16 or FP32, V-first
    `[slots, H, V, K]`). With `snapshot_rows` / `snapshot_chunks`, sequence b with
    `snapshot_chunks[b] = c > 0` also stores its state after its first
    `KDA_PREFILL_SNAPSHOT_CHUNK * c` tokens into `state_pool[snapshot_rows[b]]` (-1: none).
    Q/K L2 norm, the lower-bound sigmoid gate and the beta sigmoid are fused. Returns a new bf16
    `o` shaped like `q`.
    """

    ARCHS: frozenset[Arch] = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 context (attention DP, 96 KDA heads per rank): ragged batch, fp32 pool",
            spec=dict(heads=96, state_dtype="float32", **_RAGGED),
        ),
        Cell(
            why="TEP8 context (12 KDA heads per rank): ragged batch, fp32 pool",
            spec=dict(heads=12, state_dtype="float32", **_RAGGED),
        ),
        Cell(
            why="one long chunked-prefill continuation (2048 tokens) at 96 heads",
            spec=dict(
                heads=96, seq_lens=(2048,), slots=(3,), fresh=(False,), state_dtype="float32"
            ),
        ),
        Cell(
            why="many short sequences (the occupancy router's decomposition path) at 12 heads",
            spec=dict(
                heads=12,
                seq_lens=(5, 1, 16, 17, 33, 2, 64, 9),
                slots=(0, 1, 2, 3, 4, 5, 6, 7),
                fresh=(True, False, True, False, True, False, True, False),
                state_dtype="float32",
            ),
        ),
        Cell(
            why="mid-chunk state snapshots for KDA block reuse (rows past the batch's slots)",
            spec=dict(
                heads=12,
                state_dtype="float32",
                snapshot_rows=(20, 21, -1, 22),
                snapshot_chunks=(1, 1, 0, 0),
                **_RAGGED,
            ),
        ),
        Cell(
            why="snapshots at 96 heads, one taken past the first chunk boundary",
            spec=dict(
                heads=96,
                seq_lens=(300, 130),
                slots=(1, 2),
                fresh=(True, False),
                snapshot_rows=(5, 6),
                snapshot_chunks=(4, 2),
                state_dtype="float32",
            ),
        ),
        Cell(
            why="beta as the strided column view of the merged [f_a | b | pad] projection",
            spec=dict(heads=12, state_dtype="float32", beta_row_width=144, **_RAGGED),
        ),
        Cell(
            why="TEP8 context: beta the b columns of the stacked [q|k|v|g|f_a|b] GEMM output, row pitch 6288",
            spec=dict(heads=12, state_dtype="float32", stacked=True, **_RAGGED),
        ),
        Cell(
            why="DEP16 context, one sequence: beta from the stacked GEMM output, row pitch 49376",
            spec=dict(
                heads=96,
                seq_lens=(300,),
                slots=(3,),
                fresh=(False,),
                state_dtype="float32",
                stacked=True,
            ),
        ),
        *(
            Cell(
                why=f"BF16 recurrent pool (mamba_ssm_cache_dtype bfloat16) at {heads} heads",
                spec=dict(heads=heads, state_dtype="bfloat16", **_RAGGED),
            )
            for heads in (96, 12)
        ),
    )
    note = """
    Raw contiguous bf16 q / k / v / g in packed [1, T, H, 128] storage (the
    fused convolution's outputs and the f_b projection); beta holds bf16
    logits and may be a strided column view (unit head stride, 16-byte
    aligned row pitch). int64 cu_seqlens partitions the tokens; int32
    state_indices names unique resident rows (contiguous per slot, the slot
    stride may be padded). The caller zeroes fresh rows before the call.
    A_log [H] and dt_bias [H * 128] contiguous fp32. The targets pass
    scale = 128**-0.5 and lower_bound = -5; seq_order / cu_chunks are the
    step's shared int32 schedule (both or neither).
    Snapshot rows must differ from state_indices. The kernel stores them
    through its state checkpoints, which it compiles as a compact
    [n, H, 128, 128] tensor: with snapshots the slot stride must be a whole
    number of 128 x 128 blocks (the V2 pool's slots are whole state pages),
    and a part-block pad is refused.

    Chunked (16-token) evaluation with bf16 tensor-core products: within
    2e-2 (output) and 1e-2 (state) of an fp32 recurrence, relative to the
    largest magnitude. Workspace and planned metadata are allocated per
    call from the torch allocator; compiled once per (state dtype, lower
    bound), so warm it before graph capture.
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
        seq_order: Optional[torch.Tensor] = None,
        cu_chunks: Optional[torch.Tensor] = None,
        snapshot_rows: Optional[torch.Tensor] = None,
        snapshot_chunks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _kda_prefill(
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
            seq_order,
            cu_chunks,
            snapshot_rows,
            snapshot_chunks,
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
        seq_order: Optional[torch.Tensor] = None,
        cu_chunks: Optional[torch.Tensor] = None,
        snapshot_rows: Optional[torch.Tensor] = None,
        snapshot_chunks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """fp32 token-by-token recurrence; `seq_order` and `cu_chunks` only schedule."""
        qf = torch.nn.functional.normalize(q.float(), dim=-1, eps=1e-12) * scale
        kf = torch.nn.functional.normalize(k.float(), dim=-1, eps=1e-12)
        vf = v.float()
        gate = g.float() + dt_bias.float().reshape(q.shape[-2:])
        rate = A_log.float().exp().reshape(-1, 1)
        decay = (lower_bound * torch.sigmoid(rate * gate)).exp()
        bf = beta.float().sigmoid()
        starts = cu_seqlens.tolist()
        num_seqs = len(starts) - 1
        snap_rows = snapshot_rows.tolist() if snapshot_rows is not None else [-1] * num_seqs
        snap_chunks = snapshot_chunks.tolist() if snapshot_chunks is not None else [0] * num_seqs
        out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        for b, (slot, lo, hi) in enumerate(zip(state_indices.tolist(), starts, starts[1:])):
            snap_after = KDA_PREFILL_SNAPSHOT_CHUNK * snap_chunks[b]
            take_snapshot = snap_rows[b] >= 0 and 0 < snap_after <= hi - lo
            state = state_pool[slot].float()
            for token in range(lo, hi):
                state = state * decay[0, token].unsqueeze(-2)
                key = kf[0, token]
                err = vf[0, token] - (state * key.unsqueeze(-2)).sum(-1)
                state = state + bf[0, token, :, None, None] * err.unsqueeze(-1) * key.unsqueeze(-2)
                out[0, token] = (state * qf[0, token].unsqueeze(-2)).sum(-1)
                if take_snapshot and token - lo + 1 == snap_after:
                    state_pool[snap_rows[b]] = state.to(state_pool.dtype)
            state_pool[slot] = state.to(state_pool.dtype)
        return out.to(q.dtype)

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
        seq_order: Optional[torch.Tensor] = None,
        cu_chunks: Optional[torch.Tensor] = None,
        snapshot_rows: Optional[torch.Tensor] = None,
        snapshot_chunks: Optional[torch.Tensor] = None,
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
        if seq_order is not None:
            assert seq_order.tolist() == sorted(range(len(lengths)), key=lambda i: -lengths[i]), (
                "seq_order must be the stable descending sequence-length permutation"
            )
            chunks = [0]
            for length in lengths:
                chunks.append(chunks[-1] + (length + 15) // 16)
            assert cu_chunks.tolist() == chunks, "cu_chunks must prefix-sum ceil(length / 16)"
        if snapshot_rows is not None:
            for row, chunk, length in zip(
                snapshot_rows.tolist(), snapshot_chunks.tolist(), lengths
            ):
                if row < 0 or chunk <= 0:
                    continue
                # The kernel drops a snapshot past the sequence's end without a word.
                assert KDA_PREFILL_SNAPSHOT_CHUNK * chunk <= length, (
                    "a snapshot must fall inside its sequence"
                )
                assert 0 <= row < state_pool.shape[0] and row not in slots, (
                    "a snapshot row must be a pool row the batch does not update"
                )


rubin_kda_prefill = RubinKdaPrefill()
