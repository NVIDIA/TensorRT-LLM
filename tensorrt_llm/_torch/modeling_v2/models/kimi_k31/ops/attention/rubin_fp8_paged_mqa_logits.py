# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer decode / verify logits over the paged FP8 index-key cache (CuTe DSL, Rubin).

The family copy of the engine kernel (``kernels/attention/rubin_fp8_paged_mqa_logits.py``) adds
``kv_evict_first``: the K-cache and scale TMA loads carry the L2 evict-first hint, so the logits
stay in L2 for the Top-K that reads them next. The logits are bit-identical either way.

The specialized Rubin targets call it for every generation row of the decode indexer: plain decode
and the DFlash2 verify window. With a key-shard workspace as ``out`` it is also the helix key-shard
scoring (the engine's ``cute_dsl_fp8_paged_mqa_logits_into``).

``__call__`` compiles through the kernel copy's runner ``RubinPagedMQALogitsRunner`` (one compile
cache, shared with the copy's launchers) and launches the cached kernel into the caller's ``out``.
No torch op is registered and nothing is allocated.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

if IS_CUTLASS_DSL_AVAILABLE:
    from tensorrt_llm._torch.attention.kernels.blackwell.paged_mqa_logits.mqa_logits_util import (
        is_uneven_split,
        legal_atoms,
        validate_atom_split,
    )

    from ...kernels.attention.rubin_fp8_paged_mqa_logits import (
        RubinPagedMQALogitsRunner,
        _get_num_sms,
    )

#: The kernel's key tile: keys per math warpgroup. A logits split is two of them (256 keys).
COMPUTE_BLOCK_KV = 128
#: The widest MMA N tile (``atom * num_heads``) the kernel builds: one atom covers a verify window
#: of ``next_n`` rows while ``next_n * num_heads <= 256``.
MAX_N_TILE = 256


def _spec(batch, next_n, context_lens, max_context_len, kv_evict_first, **overrides) -> dict:
    """``context_lens``: (low, high) of the per-request key counts, drawn per cell."""
    spec = dict(
        batch=batch,
        next_n=next_n,
        num_heads=32,
        head_dim=128,
        tokens_per_block=64,
        context_lens=tuple(context_lens),
        max_context_len=max_context_len,
        kv_evict_first=kv_evict_first,
        page_stride_pad=0,
        data="int",
    )
    spec.update(overrides)
    return spec


class RubinFp8PagedMqaLogits(OpWrapper):
    """``out[b * n + t, j] = k_scale[j] * sum_h weights[b * n + t, h] * relu(q[b, t, h] . k[j])``
    for the keys ``j <= context_lens[b] - n + t`` of request ``b`` (``n`` = ``q.shape[1]``).

    Key ``j`` of request ``b`` is token ``j % P`` of page ``block_table[b, j // P]`` of ``kv_cache``
    ([pages, P, 1, 132] uint8: per page P rows of 128 E4M3 bytes, then P fp32 scales). ``q`` E4M3
    [B, n, 32, 128]; ``weights`` fp32 [B * n, 32]; ``context_lens`` int32 [B]; ``schedule_meta`` the
    DeepGEMM schedule (``get_paged_mqa_logits_metadata`` with block_kv 64) over the B requests.
    ``out`` is an fp32 [>= B * n, max_context_len] view with unit column stride; the kernel writes
    its first B * n rows and returns it. Columns past a row's causal end hold unspecified values.
    ``kv_evict_first`` only sets the L2 policy of the K loads.
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
            why="single-atom k7 verify window: 8 rows x 32 heads fill the 256-wide MMA tile",
            spec=_spec(4, 8, (3000, 9000), 16384, True),
        ),
        Cell(
            why="padded page stride: the KV cache manager V2's per-layer view of a wider page",
            spec=_spec(4, 8, (3000, 9000), 16384, True, page_stride_pad=512),
        ),
        Cell(
            why="real-valued UE8M0-scaled keys and weights: accumulation order visible",
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
            why="TEP8 GEN verify b1 k7: 8 rows over one long context",
            spec=_spec(1, 8, (200000, 260000), 262144, True),
        ),
        Cell(
            why="DEP16 GEN verify b12 k7 on real-valued data",
            spec=_spec(12, 8, (24000, 64000), 65536, True, data="randn"),
        ),
    )
    note = """
    The kernel is the engine's with one compile option, so its other architectures are the
    engine's business; the family's sm_103 path is ``cute_dsl_fp8_paged_mqa_logits``.

    The cells are the specialized targets' shapes: one request's verify window is one atom
    (next_n * 32 <= 256 at every draft length the targets take, 3 to 7), so the op launches one
    atom per request. The kernel keeps the engine runner's atom splits (num_next_n_atoms > 1,
    ``atom``); no Rubin target runs a window wider than one atom, so the op does not expose them.

    ``out`` is caller-owned and persistent: one fp32 buffer per engine (stable across CUDA-graph
    replays), or the key-shard merge workspace that peer ranks read. Any row stride works: a
    stride of exactly max_context_len gives the same logits, and the slack columns of a wider
    buffer stay untouched (the GPU test checks both). Compiles one kernel per (page size, heads,
    n, atom split, SM count, kv_evict_first, pdl) on first use.

    ``pdl`` launches the kernel class's PDL build (programmatic stream serialization; the TMA warps
    wait on the predecessor before their first load, the math warps release the dependents after
    their last store): only the launch and the waits move, the logits are bit-identical. The K3.1
    dep16 target turns it on with its logits_pdl lever, the helix8 target with its
    dsa_idx_logits_pdl constant.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        kv_evict_first: bool,
        out: torch.Tensor,
        pdl: bool = False,
    ) -> torch.Tensor:
        runner = RubinPagedMQALogitsRunner
        batch, next_n, num_heads, head_dim = q.shape
        # Every target scores one request's verify window as one atom (next_n * 32 <= 256).
        num_next_n_atoms, atom = 1, None
        atom = validate_atom_split(
            next_n, num_heads, num_next_n_atoms, atom, legal_atoms(num_heads, MAX_N_TILE, 8)
        )
        atom_rows = batch * num_next_n_atoms
        if is_uneven_split(next_n, num_next_n_atoms, atom):
            rows = batch * next_n * num_heads
            q_3d = q.reshape(-1).as_strided((rows, head_dim, 1), (head_dim, 1, rows * head_dim))
            w_2d = weights.reshape(-1)
        else:
            n_tile = atom * num_heads
            q_3d = q.reshape(-1).as_strided(
                (n_tile, head_dim, atom_rows), (head_dim, 1, n_tile * head_dim)
            )
            w_2d = weights.reshape(-1).as_strided((n_tile, atom_rows), (1, n_tile))
        num_pages, page = kv_cache.shape[0], kv_cache.shape[1]
        f32 = torch.float32
        key = (
            COMPUTE_BLOCK_KV,
            page,
            num_heads,
            head_dim,
            next_n,
            _get_num_sms(),
            1,
            f32,
            f32,
            f32,
        )
        key = key + (num_next_n_atoms, atom, bool(kv_evict_first))
        if pdl:  # its own cache entry; without it the key and the kernel are the runner's own
            key = key + (True,)
        kernel = runner.kernel_cache.get(key)
        if kernel is None:
            runner._compile(*key)
            kernel = runner.kernel_cache[key]
        kernel(
            kv_cache.reshape(num_pages, -1),
            q_3d.view(torch.uint8),
            w_2d,
            out,
            block_table,
            context_lens,
            schedule_meta,
            num_pages,
            atom_rows,
        )
        return out

    def reference(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        kv_evict_first: bool,
        out: torch.Tensor,
        pdl: bool = False,
    ) -> torch.Tensor:
        """fp64 evaluation over each row's causal keys, -inf elsewhere, into a new tensor.

        ``schedule_meta`` only partitions the work, ``kv_evict_first`` only sets a cache policy and
        ``pdl`` the launch: all ignored. ``out`` is left untouched.
        """
        batch, next_n, num_heads, head_dim = q.shape
        page = kv_cache.shape[1]
        logits = torch.full(
            (batch * next_n, max_context_len), float("-inf"), dtype=torch.float32, device=q.device
        )
        for b in range(batch):
            ctx = int(context_lens[b])
            pages = block_table[b, : -(-ctx // page)].long()
            raw = kv_cache[pages].reshape(pages.shape[0], -1)
            keys = raw[:, : page * head_dim].contiguous().view(torch.float8_e4m3fn)
            scales = raw[:, page * head_dim : page * (head_dim + 4)].contiguous()
            scales = scales.view(torch.float32)
            k = keys.reshape(-1, head_dim)[:ctx].double() * scales.reshape(-1)[:ctx, None].double()
            for t in range(next_n):
                row = b * next_n + t
                end = ctx - next_n + t + 1
                if end <= 0:
                    continue
                score = (q[b, t].double() @ k[:end].t()).relu()
                logits[row, :end] = (score * weights[row].double()[:, None]).sum(0).float()
        return logits

    def is_valid(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        kv_evict_first: bool,
        out: torch.Tensor,
        pdl: bool = False,
    ) -> None:
        # The bytes of q are read as E4M3: any other dtype is reinterpreted, not converted.
        assert q.dtype == torch.float8_e4m3fn, f"q must be E4M3, got {q.dtype}"
        # The kernel takes the page bytes as one flat row per page; a page view whose 132-byte
        # rows are not packed is read at the wrong offsets.
        assert kv_cache.stride(1) == kv_cache.shape[3] and kv_cache.stride(3) == 1, (
            "kv_cache rows within a page must be packed"
        )
        # A page is P * head_dim key bytes, then P fp32 scales, with head_dim taken from q: any
        # other token width puts the keys and scales elsewhere.
        assert kv_cache.shape[2:] == (1, q.shape[3] + 4), (
            f"kv_cache must be [pages, P, 1, head_dim + 4], got {tuple(kv_cache.shape)}"
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
        # On the GPU the kernel runs _get_num_sms() CTAs, and CTA i reads schedule rows i and
        # i + 1: a schedule built for another SM count skips work or reads past its end.
        if schedule_meta.is_cuda:
            num_sms = _get_num_sms()
            assert schedule_meta.shape == (num_sms + 1, 2), (
                f"schedule_meta must be [num_sms + 1, 2] with num_sms = {num_sms}, "
                f"got {tuple(schedule_meta.shape)}"
            )
        # q and weights are re-strided from their base pointer: a non-packed view is read as
        # another request's rows.
        assert q.is_contiguous() and weights.is_contiguous(), "q and weights must be packed"
        assert weights.shape == (q.shape[0] * q.shape[1], q.shape[2]), (
            "weights must hold one row per query row"
        )
        # The kernel addresses rows through out's row stride and never checks the row count: a
        # buffer with fewer rows than query rows is written past its end.
        assert out.dtype == torch.float32 and out.stride(1) == 1, (
            "out must be fp32 with unit column stride"
        )
        assert out.shape[0] >= q.shape[0] * q.shape[1] and out.shape[1] == max_context_len, (
            f"out must be [>= B * n, max_context_len], got {tuple(out.shape)}"
        )


rubin_fp8_paged_mqa_logits = RubinFp8PagedMqaLogits()
