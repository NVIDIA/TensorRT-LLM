# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer decode / verify logits over the paged FP4 index-key cache (CuTe DSL, Rubin).

The FP4 counterpart of ``rubin_fp8_paged_mqa_logits``: the same paged cache walk, schedule and
caller-owned output, with MXFP4 queries and keys (``rubin_idx_fp4_quantize``'s format) whose block
scales the block-scaled MMA applies. The kernel is the family copy
``kernels/attention/rubin_fp4_paged_mqa_logits.py``; this entry imports it on its first call, so
importing the entry (a CPU test, a GPU that never runs it) imports no CuTe DSL.

The Rubin default target calls it for every generation row of the decode indexer when the index-key
cache is FP4: plain decode and the DFlash2 verify window. With helix8's key-shard inputs (this
rank's pages, per-row local bounds, its schedule) and the key-shard merge workspace as ``out`` it is
also helix8's key-shard scoring.

``__call__`` compiles through the kernel copy's runner ``RubinFP4PagedMQALogitsRunner`` (one compile
cache) and launches the cached kernel into the caller's ``out``.
No torch op is registered and nothing is allocated.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .rubin_idx_fp4_quantize import mxfp4_values

#: The kernel's key tile: keys per math warpgroup. A logits split is two of them (256 keys).
COMPUTE_BLOCK_KV = 128
#: The kernel writes each row's last split whole: a row needs this alignment of room.
SPLIT_KV = 256
#: The widest MMA N tile (``atom * num_heads``) the kernel builds on Rubin: one atom covers a verify
#: window of ``next_n`` rows while ``next_n * num_heads <= 256``.
MAX_N_TILE = 256
#: Bytes of one index key in a page: 64 packed E2M1 bytes, then (after all P keys) its int32 scales.
DATA_BYTES = 64
TOKEN_BYTES = DATA_BYTES + 4


@functools.cache
def _kernel() -> tuple:
    """The kernel copy's runner and the atom-split helpers, imported on the first call."""
    from tensorrt_llm._torch.attention.kernels.blackwell.paged_mqa_logits.mqa_logits_util import (
        is_uneven_split,
        legal_atoms,
        validate_atom_split,
    )

    from ...kernels.attention.rubin_fp4_paged_mqa_logits import (
        RubinFP4PagedMQALogitsRunner,
        _get_num_sms,
    )

    return (
        RubinFP4PagedMQALogitsRunner,
        _get_num_sms,
        is_uneven_split,
        legal_atoms,
        validate_atom_split,
    )


def _spec(batch, next_n, context_lens, max_context_len, kv_evict_first, **overrides) -> dict:
    """``context_lens``: (low, high) of the per-request key counts, drawn per cell; every
    ``zero_every``-th request then has none (a helix8 rank without keys of it)."""
    spec = dict(
        batch=batch,
        next_n=next_n,
        num_next_n_atoms=1,
        atom=None,
        num_heads=32,
        head_dim=128,
        tokens_per_block=64,
        context_lens=tuple(context_lens),
        max_context_len=max_context_len,
        kv_evict_first=kv_evict_first,
        page_stride_pad=0,
        data="int",
        zero_every=None,
    )
    spec.update(overrides)
    return spec


class RubinFp4PagedMqaLogits(OpWrapper):
    """``out[b * n + t, j] = sum_h weights[b * n + t, h] * relu(q[b, t, h] . k[j])`` for the keys
    ``j <= context_lens[b] - n + t`` of request ``b`` (``n`` = ``q.shape[1]``).

    ``q[b, t, h]`` and ``k[j]`` are MXFP4 rows: 64 bytes of E2M1 pairs and four UE8M0 block
    exponents in an int32 (``rubin_idx_fp4_quantize``). ``q`` ``[B, n, 32, 64]`` (any 1-byte dtype)
    with ``sf_q`` int32 ``[B, n, 32]``. Key ``j`` of request ``b`` is token ``j % P`` of page
    ``block_table[b, j // P]`` of ``kv_cache`` (uint8 ``[pages, P, 1, 68]``: per page P rows of 64
    key bytes, then P int32 scales). ``weights`` fp32 ``[B * n, 32]`` (no query scale: the MMA
    applies ``sf_q``); ``context_lens`` int32 ``[B]``; ``schedule_meta`` the DeepGEMM schedule
    (``get_paged_mqa_logits_metadata`` with block_kv 64) over the ``B * num_next_n_atoms`` atom rows.

    ``out`` is an fp32 ``[>= B * n, max_context_len]`` view with unit column stride. The kernel
    writes each row's last 256-key split whole, so the row pitch must be at least
    ``max_context_len`` rounded up to 256 and the buffer must hold that much past the start of row
    ``B * n - 1``. It writes the first B * n rows and returns ``out``. Columns past a row's causal end
    hold unspecified values, the padding of the last split included. ``kv_evict_first`` only sets
    the L2 policy of the key loads.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="plain decode, one request at the top-k edge (2049 keys), default L2 policy",
            spec=_spec(1, 1, (2049, 2049), 4096, False),
        ),
        Cell(
            why="plain decode with the evict-first hint, fewer keys than one 256-key split",
            spec=_spec(1, 1, (300, 300), 4096, True),
        ),
        Cell(
            why="no speculative decoding at batch 64: one row per request",
            spec=_spec(64, 1, (1000, 30000), 32768, False),
        ),
        Cell(
            why="k1 verify window: 2 rows x 32 heads in one atom",
            spec=_spec(4, 2, (3000, 9000), 16384, True),
        ),
        Cell(
            why="k2 verify window: 3 rows x 32 heads in one atom",
            spec=_spec(4, 3, (3000, 9000), 16384, True),
        ),
        Cell(
            why="single-atom k7 verify window: 8 rows x 32 heads fill the 256-wide MMA tile",
            spec=_spec(4, 8, (3000, 9000), 16384, True),
        ),
        Cell(
            why="padded page stride: the KV cache manager V2's per-layer view of a wider page",
            spec=_spec(4, 8, (3000, 9000), 16384, True, page_stride_pad=512),
        ),
        Cell(
            why="real-valued MXFP4 keys, queries and weights: accumulation order visible",
            spec=_spec(4, 8, (3000, 9000), 16384, True, data="randn"),
        ),
        Cell(
            why="DEP16 GEN verify b12 k7: 12 x 8 = 96 rows x 32 heads",
            spec=_spec(12, 8, (24000, 64000), 65536, True),
        ),
        Cell(
            why="helix8 GEN key-shard scoring b16 k5: 16 x 6 = 96 rows on one key shard",
            spec=_spec(16, 6, (3000, 24000), 32768, True),
        ),
        Cell(
            why="helix8 key-shard scoring b16 k5 where this rank holds no key of some requests",
            spec=_spec(16, 6, (3000, 24000), 32768, True, zero_every=4),
        ),
        Cell(
            why="TEP8 GEN verify b1 k7: 8 rows over one long context",
            spec=_spec(1, 8, (200000, 260000), 262144, True),
        ),
        Cell(
            why="DEP16 GEN verify b12 k7 on real-valued data",
            spec=_spec(12, 8, (24000, 64000), 65536, True, data="randn"),
        ),
    )
    note = """
    The kernel is the engine's FP4 paged MQA logits kernel plus one compile option, the L2 hint
    (``kv_evict_first``); the hint changes no arithmetic, so this entry's logits are the engine
    kernel's bit for bit.

    The cells are the default target's FP4 shapes: one request's verify window is one atom
    (next_n * 32 <= 256 for every draft length the target takes, none to 7), so num_next_n_atoms
    is 1. The atom splits (num_next_n_atoms > 1, ``atom``) are kept from the engine runner but
    certified by no cell here.

    ``out`` is caller-owned and persistent: the decode-logits arena (rows of the aligned width,
    stable across CUDA-graph replays), or the key-shard merge workspace that peer ranks read (its
    rows are aligned the same way). Unlike the FP8 kernel, this one stores whole 256-key splits,
    so the slack columns up to the aligned width do change; the columns past it and the rows past
    B * n do not. Compiles one kernel per (page size, heads, n, atom split, SM count,
    kv_evict_first) on first use.
    """

    def __call__(
        self,
        q: torch.Tensor,
        sf_q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_next_n_atoms: int,
        atom: Optional[int],
        kv_evict_first: bool,
        out: torch.Tensor,
    ) -> torch.Tensor:
        runner, num_sms, is_uneven_split, legal_atoms, validate_atom_split = _kernel()
        batch, next_n, num_heads, half_dim = q.shape
        atom = validate_atom_split(
            next_n, num_heads, num_next_n_atoms, atom, legal_atoms(num_heads, MAX_N_TILE, 8)
        )
        atom_rows = batch * num_next_n_atoms
        q_bytes = q.view(torch.uint8).reshape(-1)
        if is_uneven_split(next_n, num_next_n_atoms, atom):
            rows = batch * next_n * num_heads
            q_3d = q_bytes.as_strided((rows, half_dim, 1), (half_dim, 1, rows * half_dim))
            sf_2d = sf_q.reshape(-1)
            w_2d = weights.reshape(-1)
        else:
            n_tile = atom * num_heads
            q_3d = q_bytes.as_strided(
                (n_tile, half_dim, atom_rows), (half_dim, 1, n_tile * half_dim)
            )
            sf_2d = sf_q.reshape(-1).as_strided((n_tile, atom_rows), (1, n_tile))
            w_2d = weights.reshape(-1).as_strided((n_tile, atom_rows), (1, n_tile))
        num_pages, page = kv_cache.shape[0], kv_cache.shape[1]
        f32 = torch.float32
        # The runner's compile key of a launch without emission: page, heads, head dim, next_n,
        # SM count, one epilogue subtile, fp32 epilogue and logits, no emission (cand_cap 0,
        # accept_cap 8192), then the atom split and the L2 hint.
        key = (COMPUTE_BLOCK_KV, page, num_heads, 2 * half_dim, next_n, num_sms(), 1, f32, f32)
        key = key + (False, False, False, False, False, False, 0, False, 8192)
        key = key + (num_next_n_atoms, atom, bool(kv_evict_first))
        kernel = runner.kernel_cache.get(key)
        if kernel is None:
            runner._compile(*key)
            kernel = runner.kernel_cache[key]
        kernel(
            kv_cache.reshape(num_pages, -1),
            q_3d,
            sf_2d,
            w_2d,
            out,
            block_table,
            context_lens,
            schedule_meta,
            num_pages,
            atom_rows,
            *(None,) * 9,
        )
        return out

    def reference(
        self,
        q: torch.Tensor,
        sf_q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_next_n_atoms: int,
        atom: Optional[int],
        kv_evict_first: bool,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """fp64 evaluation over each row's causal keys, -inf elsewhere, into a new tensor.

        ``schedule_meta``, ``num_next_n_atoms`` and ``atom`` only partition the work, and
        ``kv_evict_first`` only sets a cache policy: all ignored. ``out`` is left untouched.
        """
        batch, next_n = q.shape[:2]
        page = kv_cache.shape[1]
        queries = mxfp4_values(q, sf_q).double()
        logits = torch.full(
            (batch * next_n, max_context_len), float("-inf"), dtype=torch.float32, device=q.device
        )
        for b in range(batch):
            ctx = int(context_lens[b])
            if ctx <= 0:  # no keys of this request on this rank (helix8): every row stays -inf
                continue
            pages = block_table[b, : -(-ctx // page)].long()
            raw = kv_cache[pages].reshape(pages.shape[0], -1)
            keys = raw[:, : page * DATA_BYTES].reshape(-1, DATA_BYTES)
            scales = raw[:, page * DATA_BYTES : page * TOKEN_BYTES].reshape(-1, 4)
            k = mxfp4_values(keys, scales)[:ctx].double()
            for t in range(next_n):
                row = b * next_n + t
                end = ctx - next_n + t + 1
                if end <= 0:
                    continue
                score = (queries[b, t] @ k[:end].t()).relu()
                logits[row, :end] = (score * weights[row].double()[:, None]).sum(0).float()
        return logits

    def is_valid(
        self,
        q: torch.Tensor,
        sf_q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_next_n_atoms: int,
        atom: Optional[int],
        kv_evict_first: bool,
        out: torch.Tensor,
    ) -> None:
        # An fp32 per-head scale is an FP8 query's: its bits would be read as four exponents.
        assert sf_q.dtype == torch.int32 and sf_q.shape == q.shape[:3], (
            f"sf_q must be int32 [B, n, heads], got {sf_q.dtype} {tuple(sf_q.shape)}"
        )
        # The kernel takes the page bytes as one flat row per page; a page view whose 68-byte rows
        # are not packed is read at the wrong offsets.
        assert kv_cache.stride(1) == kv_cache.shape[3] and kv_cache.stride(3) == 1, (
            "kv_cache rows within a page must be packed"
        )
        # A page is P * 64 key bytes, then P int32 scales: an FP8 page (132-byte tokens) puts the
        # keys and scales elsewhere.
        assert kv_cache.shape[2:] == (1, q.shape[3] + 4), (
            f"kv_cache must be [pages, P, 1, 68], got {tuple(kv_cache.shape)}"
        )
        max_ctx = int(context_lens.max())
        assert max_ctx <= max_context_len, (
            "a context longer than max_context_len writes past the logits row"
        )
        # Key j of request b is on page block_table[b, j // P], and the column is never
        # bounded: a narrower table is read past its row.
        pages = -(-max_ctx // kv_cache.shape[1])
        assert block_table.shape[1] >= pages, (
            f"block_table must have >= ceil(max context / P) = {pages} columns, "
            f"got {block_table.shape[1]}"
        )
        # On the GPU the kernel runs one CTA per SM, and CTA i reads schedule rows i and i + 1: a
        # schedule built for another SM count skips work or reads past its end.
        if schedule_meta.is_cuda:
            num_sms = torch.cuda.get_device_properties(schedule_meta.device).multi_processor_count
            assert schedule_meta.shape == (num_sms + 1, 2), (
                f"schedule_meta must be [num_sms + 1, 2] with num_sms = {num_sms}, "
                f"got {tuple(schedule_meta.shape)}"
            )
        # q, sf_q and weights are re-strided from their base pointers: a non-packed view is read
        # as another request's rows.
        assert q.is_contiguous() and sf_q.is_contiguous() and weights.is_contiguous(), (
            "q, sf_q and weights must be packed"
        )
        assert weights.shape == (q.shape[0] * q.shape[1], q.shape[2]), (
            "weights must hold one row per query row"
        )
        # The kernel addresses rows through out's row stride, never checks the row count, and
        # stores each row's last split whole: a short buffer or a pitch below the aligned width is
        # written past its rows.
        rows = q.shape[0] * q.shape[1]
        aligned = -(-max_context_len // SPLIT_KV) * SPLIT_KV
        assert out.dtype == torch.float32 and out.stride(1) == 1, (
            "out must be fp32 with unit column stride"
        )
        assert out.shape[0] >= rows and out.shape[1] == max_context_len, (
            f"out must be [>= B * n, max_context_len], got {tuple(out.shape)}"
        )
        room = out.untyped_storage().nbytes() // out.element_size() - out.storage_offset()
        assert out.stride(0) >= aligned and (rows - 1) * out.stride(0) + aligned <= room, (
            f"out rows must have room for {aligned} columns (max_context_len aligned to "
            f"{SPLIT_KV}), got a row pitch of {out.stride(0)}"
        )


rubin_fp4_paged_mqa_logits = RubinFp4PagedMqaLogits()
