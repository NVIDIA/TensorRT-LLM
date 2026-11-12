# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked KDA prefill (CuTe DSL: fused K1-K3, A_kk inverse, persistent K4) on L2-normalized Q/K,
updating indexed fp32 V-first state rows in place."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _kda_prefill as prefill_ops

#: Kimi K3's gate_lower_bound, the launcher's default under ``safe_gate``.
_DEFAULT_LOWER_BOUND = -5.0


def _sequences(q: torch.Tensor, cu_seqlens: Optional[torch.Tensor]) -> list[tuple[int, int, int]]:
    """``(row, start, end)`` of every sequence: equal-length rows, or the ``cu_seqlens`` spans of
    row 0."""
    if cu_seqlens is None:
        return [(row, 0, q.shape[1]) for row in range(q.shape[0])]
    starts = cu_seqlens.tolist()
    return [(0, lo, hi) for lo, hi in zip(starts, starts[1:])]


class KdaPrefill(OpWrapper):
    """Run the gated delta rule over each sequence from ``state_pool[state_indices[b]]``; return
    ``o`` ``[B, T, H, V]``.

    Sequences are the ``B`` rows of equal length, or (``cu_seqlens``, ``B == 1``) the packed
    spans. ``q`` / ``k`` arrive L2-normalized; ``q`` is scaled by ``scale``. With
    ``use_gate_in_kernel`` the log-decay is ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
    (``safe_gate``) or ``-exp(A_log) * softplus(g + dt_bias)``, else ``g`` itself; with
    ``use_beta_sigmoid_in_kernel`` beta is ``sigmoid(beta)``. The final state of sequence ``b`` is
    written to ``state_pool[state_indices[b]]``; with ``snapshot_rows`` / ``snapshot_chunks`` the
    state after its first ``64 * snapshot_chunks[b]`` tokens also goes to
    ``state_pool[snapshot_rows[b]]`` (``-1`` skips).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # The tp1 context server holds all 96 KDA heads; generation servers do not prefill.
    CELLS = (
        Cell(
            "tp1 context: one fresh 300-token prompt, Q/K/V/beta chunk-padded by the post-conv "
            "step and g not",
            dict(heads=96, seq_lens=(300,), fresh=(True,)),
        ),
        Cell(
            "tp1 context: a 1024-token chunk continuing a long prompt from its slot",
            dict(heads=96, seq_lens=(1024,), fresh=(False,)),
        ),
        Cell(
            "tp1 context: four packed prompts, fresh and continued, with partial chunks",
            dict(heads=96, seq_lens=(64, 2, 129, 63), fresh=(True, False, True, False)),
        ),
        Cell(
            "tp1 context: three packed prompts with in-forward state snapshots (one skipped)",
            dict(
                heads=96,
                seq_lens=(256, 200, 130),
                fresh=(False, True, False),
                snapshot_chunks=(2, -1, 1),
            ),
        ),
    )
    note = """
    Kimi K3 calls it from the KDA layer's prefill (K3KDAKernelDispatch.prefill_chunk_kda) with
    K = V = 128, chunk_size 64, safe_gate, lower_bound -5, use_gate_in_kernel and
    use_beta_sigmoid_in_kernel, bf16 Q/K/V/g, fp32 beta, fp32 A_log / dt_bias, and the
    metadata's int64 cu_seqlens, chunk_indices, varlen_is_aligned and single_sequence_length
    (so no layer reads cu_seqlens back from the GPU). A single unaligned sequence arrives
    zero-padded to a chunk multiple (Q/K/V/beta; g may stay unpadded): the op rebuilds g's
    sentinel tail itself, and output rows past cu_seqlens[-1] are unspecified (the caller crops
    them). Equal-length inputs with T not a multiple of 256 take a padded copy (B == 1 only).

    In place: the state_pool rows named by state_indices (and snapshot_rows). The caller zeroes
    fresh rows first; continuation chunks reuse the slot in order. The returned o is a view of a
    process-wide scratch buffer keyed by shape: the next call with the same shape overwrites it.
    Process-wide caches also hold the compiled kernels (keyed by layout and constexpr flags), the
    intermediate workspaces and padded-input buffers; warm each shape before graph capture.
    Launches go to the current stream. Importing the entry needs CuTe DSL and FlashInfer.
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
        snapshot_rows: Optional[torch.Tensor] = None,
        snapshot_chunks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return prefill_ops.kda_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            state_pool=state_pool,
            state_indices=state_indices,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            varlen_is_aligned=varlen_is_aligned,
            single_sequence_length=single_sequence_length,
            snapshot_rows=snapshot_rows,
            snapshot_chunks=snapshot_chunks,
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
        snapshot_rows: Optional[torch.Tensor] = None,
        snapshot_chunks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp64 token-by-token recurrence: ``(o, state_pool after the call)``; nothing is written.

        ``o`` rows past ``cu_seqlens[-1]`` are zero (the op leaves them unspecified).
        ``chunk_indices``, ``chunk_size``, ``varlen_is_aligned`` and ``single_sequence_length``
        are scheduling hints and do not enter the result.
        """
        heads, key_dim = q.shape[2], q.shape[3]
        out = torch.zeros(v.shape, dtype=torch.float64, device=v.device)
        pool = state_pool.clone()
        slots = state_indices.tolist()
        snaps = (
            [-1] * len(slots)
            if snapshot_chunks is None
            else list(zip(snapshot_rows.tolist(), snapshot_chunks.tolist()))
        )
        bound = _DEFAULT_LOWER_BOUND if lower_bound is None else lower_bound
        for b, (row, lo, hi) in enumerate(_sequences(q, cu_seqlens)):
            qd = q[row, lo:hi].double() * scale
            kd = k[row, lo:hi].double()
            vd = v[row, lo:hi].double()
            gd = g[row, lo:hi].double()
            bd = beta[row, lo:hi].double()
            if use_gate_in_kernel:
                if dt_bias is not None:
                    gd = gd + dt_bias.double().view(heads, key_dim)
                rate = A_log.double().exp()[:, None]
                if safe_gate:
                    gd = bound * torch.sigmoid(rate * gd)
                else:
                    gd = -rate * torch.nn.functional.softplus(gd)
            if use_beta_sigmoid_in_kernel:
                bd = torch.sigmoid(bd)
            decay = gd.exp()
            s = state_pool[slots[b]].double()
            snap = snaps[b]
            for t in range(hi - lo):
                if snap != -1 and snap[1] >= 0 and t == chunk_size * snap[1]:
                    pool[snap[0]] = s.to(pool.dtype)
                s = s * decay[t][:, None, :]
                err = vd[t] - torch.einsum("hvk,hk->hv", s, kd[t])
                s = s + bd[t][:, None, None] * err[:, :, None] * kd[t][:, None, :]
                out[row, lo + t] = torch.einsum("hvk,hk->hv", s, qd[t])
            pool[slots[b]] = s.to(pool.dtype)
        return out.to(v.dtype), pool

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
        snapshot_rows: Optional[torch.Tensor] = None,
        snapshot_chunks: Optional[torch.Tensor] = None,
    ) -> None:
        spans = _sequences(q, cu_seqlens)
        lengths = [hi - lo for _, lo, hi in spans]
        slots = state_indices.tolist()
        # Two sequences on one slot race on its row.
        assert len(set(slots)) == len(slots), "state_indices must name one slot per sequence"
        if cu_seqlens is not None:
            # The hints replace device reads of cu_seqlens: a wrong one runs the mask-free
            # kernel on a partial chunk, or pads at the wrong length, without an error.
            if single_sequence_length is not None:
                assert len(lengths) == 1 and lengths[0] == single_sequence_length, (
                    "single_sequence_length must be the one sequence's cu_seqlens length"
                )
            if varlen_is_aligned is not None:
                assert varlen_is_aligned == all(n % chunk_size == 0 for n in lengths), (
                    "varlen_is_aligned must say whether every sequence is a chunk multiple"
                )
            if chunk_indices is not None:
                expected = [
                    (s, c) for s, n in enumerate(lengths) for c in range(-(-n // chunk_size))
                ]
                assert chunk_indices.tolist() == [list(e) for e in expected], (
                    "chunk_indices must list (sequence, chunk) of every chunk of cu_seqlens"
                )
        if snapshot_chunks is not None:
            rows = snapshot_rows.tolist()
            for row, chunks, n in zip(rows, snapshot_chunks.tolist(), lengths):
                if chunks < 0:
                    continue
                assert chunks < -(-n // chunk_size), "a snapshot must fall inside its sequence"
                assert row not in slots, "a snapshot row must not be a live state row"


kda_prefill = KdaPrefill()
