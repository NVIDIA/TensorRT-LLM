# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA: attention over an explicitly selected set of paged K/V tokens."""

from __future__ import annotations

from typing import Optional

import torch

# Approved exception: QSA's sparse path never reaches the thop attention op's
# sparse argument group, so there is no torch.ops surface for this computation.
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
    can_fuse_qsa_splitk_output_gate as _can_fuse_qsa_splitk_output_gate,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
    triton_qsa_paged_sparse_gqa as _triton_qsa_paged_sparse_gqa,
)

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

_CHECKPOINT = dict(
    q_heads=24,
    kv_heads=2,
    head_dim=256,
    tokens_per_block=32,
    compress_ratio=4,
    token_topk=2048,
    cache_dtype=torch.bfloat16,
    q_dtype=torch.bfloat16,
    # q is the first 24 heads of the packed [T, 24 + 2 + 2, 256] QKV buffer;
    # "fused_row" is the target's fused input projection row, 55 heads of 256.
    q_layout="qkv_slice",
    request_dtype=torch.int32,
    # The engine's table: 260 pages per slot, -1 past each sequence.
    pages=260,
)

_SPLIT_K = ("splitk", "merge")
_PREFILL = ("prefill",)
_FP8 = dict(cache_dtype=torch.float8_e4m3fn)


def _spec(**overrides) -> dict:
    """A cell spec: this checkpoint's geometry unless the cell says otherwise.

    `segments` is `(first position, rows)` per request, in row order;
    `prefill_pair` says whether `query_positions` / `compress_ratio` are
    passed, which the target does exactly when the step has a context request.
    """
    return {**_CHECKPOINT, **overrides}


def _decode(segments, launches=_SPLIT_K, **overrides) -> dict:
    base = dict(segments=segments, slots=32, prefill_pair=False, launches=launches)
    return _spec(**{**base, **overrides})


# Step-5-Preview's sparse layers: 64 q / 4 kv heads under attention DP, 16 / 1
# under TP4, head_dim 192 zero-padded to 256, one call per KV group on the
# head-sliced pool view. The selection holds the top-512 complete regions of 8
# tokens (score order, -1 padded), each expanded to its 8 logical positions,
# and the current region's 8 columns, -1 past the query. `layout` says where
# the current region sits: "regions_then_tail" (columns 4096..4103) or
# "tail_first" (columns 0..7, every resident token then precedes every -1
# region slot).
_STEP5 = dict(
    q_heads=16,
    kv_heads=1,
    head_dim=256,
    real_head_dim=192,
    tokens_per_block=32,
    region_size=8,
    region_topk=512,
    width=512 * 8 + 8,
    softmax_scale=192**-0.5,
    cache_dtype=torch.bfloat16,
    q_dtype=torch.bfloat16,
    request_dtype=torch.int32,
    selection="step5_regions",
    prefill_pair=False,
    # max_seq_len 9216 / tokens_per_block 32.
    pages=288,
    # max_batch_size 128, plus one row so slot and row never coincide.
    slots=129,
    # A pool larger than the 128 spread decode requests map (18.6k pages).
    pool_pages=20000,
)


def _spread(n: int) -> tuple:
    """`n` decode requests, one row each, spread over every position 1..9215."""
    return tuple((1 + (9214 * i) // (n - 1), 1) for i in range(n))


#: 128 decode requests, one row each, spread over every position 1..9215.
_SPREAD_128 = _spread(128)


def _step5(
    segments, pool_kv_heads, head_view, splits, layout="regions_then_tail", **overrides
) -> dict:
    """A Step-5 cell: the split-K kernel, `splits` ways, merged when more than one."""
    launches = _SPLIT_K if splits > 1 else ("splitk",)
    base = dict(
        segments=segments,
        pool_kv_heads=pool_kv_heads,
        head_view=head_view,
        splits=splits,
        launches=launches,
        layout=layout,
    )
    return {**_STEP5, **base, **overrides}


_TAIL_FIRST = dict(layout="tail_first")
_FP8_STEP5 = dict(cache_dtype=torch.float8_e4m3fn)

#: The engine's pool: `pool_pages` lifecycle slots of `slot_pages` pages each
#: (K then V of each of the 23 sparse layers), the call reading layer
#: `layer`'s K and V pages (2 * layer and 2 * layer + 1) of every slot, the
#: table mapping the pool's last slot. 18.95 GiB at each cell's slot count.
_SLOT_POOL = dict(slot_pages=46, include_last_page=True, large=True)


class QsaPagedSparseGqa(OpWrapper):
    """Sparse GQA over the selected tokens of an HND paged K/V pool.

    Wraps `tensorrt_llm._torch.attention.backends.sparse.qsa.kernels.
    triton_qsa_paged_sparse_gqa`. For row `r`, query head `h`, KV head
    `h // (Hq // Hkv)` (contiguous groups, not interleaved), over the columns
    `j` of `selected_tokens[r]` that name a resident token::

        t     = selected_tokens[r, j]                      # logical position
        page  = block_table[request_indices[r], t // tokens_per_block]
        s_j   = (q[r, h] . k_cache[page, kv, t % tpb]) * softmax_scale
        out[r, h] = sum_j softmax(s)_j * v_cache[page, kv, t % tpb]

    A negative index, a page past the table's width, a negative or
    out-of-pool table entry, and a request index outside the table's rows all
    contribute nothing; a row left with nothing returns zeros. Indices are not
    deduplicated: a token named twice is attended twice. Returns a fresh,
    contiguous `[rows, Hq, D]` tensor of `q.dtype`, even for a strided `q`.

    Two kernels, chosen inside the call: with `query_positions` and
    `compress_ratio` given and `rows * Hkv > 512`, a causal-prefix prefill
    kernel; otherwise split-K, plus a merge kernel when it used more than one
    split. Each cell names the launches it was measured on.

    **The prefill pair is a bound, not a mask.** The prefill kernel reads only
    `min(visible // cr, block_topk) * cr + visible % cr` leading columns of a
    row (`visible = query_positions[r] + 1`, rounded up to a 64-wide tile) and
    nothing after them. That is exactly what QSA's own top-k + expansion emit,
    packed at the front; a selection with valid entries past the bound is
    silently truncated (measured: a planted index moving the right answer by
    0.77 moved the op's by nothing). The split-K kernel ignores the pair.

    **The output gate rides the merge when `output_gate` is passed.** Each
    merged head row is rounded to `q.dtype`, multiplied in fp32 by
    `sigmoid(output_gate)` and rounded again: the arithmetic of the ungated
    call followed by `fused_sigmoid_mul_inplace`. `output_gate` is `[rows,
    Hq * D]` or `[rows, Hq, D]` in `q.dtype`, unit stride in its last dim, any
    row stride (the gate columns of the fused projection row). Only a split-K
    launch with more than one split has a merge to carry it; `fuses_output_gate`
    says whether a call does, and the op raises `ValueError` if handed a gate
    it cannot carry.

    Fusion boundary: the block-table gather, scale, online softmax, P.V, the
    split-K merge and, when passed, the output gate are inside. Deciding what
    to attend to, appending K/V to the pool, and q's projection, norm and RoPE
    are not.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # ---- generation: no prefill pair, split-K ---------------------------
        Cell(
            why="decode MTP7: 32 slots x 8 rows = 256 rows, 512 programs, 4-way split-K, "
            "q and the gate off the fused row, the gate fused into the merge",
            spec=_decode(
                tuple((100 + (499 * g) % 8000, 8) for g in range(32)),
                q_layout="fused_row",
                gate="fused_row",
            ),
        ),
        # ---- context: the prefill pair is passed ----------------------------
        Cell(
            why="fp8 context 272 rows from 2032: causal-prefix prefill kernel",
            spec=_decode(((2032, 272),), launches=_PREFILL, prefill_pair=True, **_FP8),
        ),
        Cell(
            why="q off the fused projection row (55 x 256), 8191-row prefill",
            spec=_decode(((0, 8191),), launches=_PREFILL, prefill_pair=True, q_layout="fused_row"),
        ),
        # ---- Step-5-Preview: split-K only, width 4104, single-KV-head views --
        # Regions then the current region: every resident token precedes the
        # first -1 of each split here (rows of 512 regions, or one split).
        Cell(
            why="Step-5 dp 1 row at pos 9215 (max_seq_len's last), head view 2 of 4: "
            "64-way split-K under PDL, merged",
            spec=_step5(((9215, 1),), pool_kv_heads=4, head_view=2, splits=64, pdl=True),
        ),
        Cell(
            why="Step-5 dp 512-row chunk at 4096 over a reused prefix + a decode "
            "row at region end 8703, head view 1: 513 rows, one split",
            spec=_step5(((4096, 512), (8703, 1)), pool_kv_heads=4, head_view=1, splits=1),
        ),
        Cell(
            why="Step-5 envelope: dp 8192 rows at pos 1024..9215 after a reused "
            "1024-token prefix, head view 2: every row's -1s after its tokens",
            spec=_step5(((1024, 8192),), pool_kv_heads=4, head_view=2, splits=1),
        ),
        # Regions then the current region at rows with fewer than 512 regions:
        # whole tiles of -1 region slots come before a split's first resident
        # token (8-way: rows whose regions end before column 3584; one split:
        # rows 0..7, which hold only the current region).
        Cell(
            why="Step-5 dp 128 decode rows at pos 1..9215, head view 0 of 4: 8-way split-K, "
            "merged, short rows' -1 region slots ahead of the current region",
            spec=_step5(_SPREAD_128, pool_kv_heads=4, head_view=0, splits=8),
        ),
        Cell(
            why="Step-5 tp4 1-head pool, 128 decode rows at pos 1..9215: 8-way split-K, "
            "merged, short rows' -1 region slots ahead of the current region",
            spec=_step5(_SPREAD_128, pool_kv_heads=1, head_view=None, splits=8),
        ),
        Cell(
            why="Step-5 dp fp8 pool, 128 decode rows at pos 1..9215, head view 2: 8-way "
            "split-K, merged, short rows' -1 region slots ahead of the current region",
            spec=_step5(_SPREAD_128, pool_kv_heads=4, head_view=2, splits=8, **_FP8_STEP5),
        ),
        Cell(
            why="Step-5 dp 8192-row prefill from 0, head view 3 of 4: one split, rows 0..7 "
            "reach their only resident tokens after 512 -1 region slots",
            spec=_step5(((0, 8192),), pool_kv_heads=4, head_view=3, splits=1),
        ),
        Cell(
            why="Step-5 tp4 fp8 1-head pool, 8192-row prefill from 0: one split, rows 0..7 "
            "reach their only resident tokens after 512 -1 region slots",
            spec=_step5(((0, 8192),), pool_kv_heads=1, head_view=None, splits=1, **_FP8_STEP5),
        ),
        # The other split-K launch classes the padded capture batches and the
        # chunks reach, regions then the current region.
        Cell(
            why="Step-5 dp 2 decode rows at pos 1 and 9215, head view 1 of 4: 64-way "
            "split-K under PDL, merged",
            spec=_step5(_spread(2), pool_kv_heads=4, head_view=1, splits=64, pdl=True),
        ),
        Cell(
            why="Step-5 tp4 fp8 1-head pool, 4 decode rows at pos 1..9215: 64-way split-K "
            "under PDL, merged",
            spec=_step5(
                _spread(4), pool_kv_heads=1, head_view=None, splits=64, pdl=True, **_FP8_STEP5
            ),
        ),
        Cell(
            why="Step-5 dp 8 decode rows at pos 1..9215, head view 0 of 4: 32-way split-K "
            "(BLOCK_N 16), merged",
            spec=_step5(_spread(8), pool_kv_heads=4, head_view=0, splits=32),
        ),
        Cell(
            why="Step-5 dp fp8 pool, 16 decode rows at pos 1..9215, head view 3: 32-way "
            "split-K, merged",
            spec=_step5(_spread(16), pool_kv_heads=4, head_view=3, splits=32, **_FP8_STEP5),
        ),
        Cell(
            why="Step-5 tp4 1-head pool, 24 decode rows at pos 1..9215: 32-way split-K, merged",
            spec=_step5(_spread(24), pool_kv_heads=1, head_view=None, splits=32),
        ),
        Cell(
            why="Step-5 dp 120 decode rows at pos 1..9215, head view 2 of 4: 8-way split-K, merged",
            spec=_step5(_spread(120), pool_kv_heads=4, head_view=2, splits=8),
        ),
        Cell(
            why="Step-5 tp4 1-head pool, 256-row chunk at pos 4096..4351: 8-way split-K, "
            "merged, the class's largest row count",
            spec=_step5(((4096, 256),), pool_kv_heads=1, head_view=None, splits=8),
        ),
        Cell(
            why="Step-5 dp 512-row chunk at pos 4096..4607, head view 0 of 4: 4-way split-K, "
            "merged, the class's largest row count",
            spec=_step5(((4096, 512),), pool_kv_heads=4, head_view=0, splits=4),
        ),
        Cell(
            why="Step-5 tp4 fp8 1-head pool, 300-row last chunk at pos 8704..9003: 4-way "
            "split-K, merged",
            spec=_step5(((8704, 300),), pool_kv_heads=1, head_view=None, splits=4, **_FP8_STEP5),
        ),
        # Current region first (columns 0..7): every resident token precedes
        # every -1 region slot, the other layout the target may emit.
        Cell(
            why="Step-5 tail-first dp 128 decode rows at pos 1..9215, head view 0 of 4: "
            "8-way split-K, merged",
            spec=_step5(_SPREAD_128, pool_kv_heads=4, head_view=0, splits=8, **_TAIL_FIRST),
        ),
        Cell(
            why="Step-5 tail-first dp 8192-row prefill from 0, head view 3 of 4: one split, "
            "all regions below pos 4104, top-512 above",
            spec=_step5(((0, 8192),), pool_kv_heads=4, head_view=3, splits=1, **_TAIL_FIRST),
        ),
        Cell(
            why="Step-5 tail-first tp4 1-head pool, 128 decode rows at pos 1..9215: "
            "8-way split-K, merged",
            spec=_step5(_SPREAD_128, pool_kv_heads=1, head_view=None, splits=8, **_TAIL_FIRST),
        ),
        Cell(
            why="Step-5 tail-first dp fp8 pool, 128 decode rows at pos 1..9215, head view 2: "
            "8-way split-K, merged",
            spec=_step5(
                _SPREAD_128, pool_kv_heads=4, head_view=2, splits=8, **_TAIL_FIRST, **_FP8_STEP5
            ),
        ),
        Cell(
            why="Step-5 tail-first tp4 fp8 1-head pool, 8192-row prefill from 0: one split",
            spec=_step5(
                ((0, 8192),),
                pool_kv_heads=1,
                head_view=None,
                splits=1,
                **_TAIL_FIRST,
                **_FP8_STEP5,
            ),
        ),
        # Current region first in the other split-K launch classes: the
        # padded capture batches (`capture`: also replayed from a CUDA graph)
        # and 257..512-row launches, including chunks below position 4096
        # whose rows hold fewer than 512 regions, so `-1` region slots follow
        # the resident tokens inside a split and whole splits hold none.
        Cell(
            why="Step-5 tail-first dp 2 decode rows at pos 1 and 9215, head view 1 of 4: "
            "64-way split-K under PDL, merged, captured",
            spec=_step5(
                _spread(2),
                pool_kv_heads=4,
                head_view=1,
                splits=64,
                pdl=True,
                capture=True,
                **_TAIL_FIRST,
            ),
        ),
        Cell(
            why="Step-5 tail-first tp4 fp8 1-head pool, 4 decode rows at pos 1..9215: 64-way "
            "split-K under PDL, merged, captured",
            spec=_step5(
                _spread(4),
                pool_kv_heads=1,
                head_view=None,
                splits=64,
                pdl=True,
                capture=True,
                **_TAIL_FIRST,
                **_FP8_STEP5,
            ),
        ),
        Cell(
            why="Step-5 tail-first dp 16 decode rows at pos 1..9215, head view 3 of 4: 32-way "
            "split-K (BLOCK_N 16), merged, captured",
            spec=_step5(
                _spread(16), pool_kv_heads=4, head_view=3, splits=32, capture=True, **_TAIL_FIRST
            ),
        ),
        Cell(
            why="Step-5 tail-first tp4 fp8 1-head pool, 24 decode rows at pos 1..9215: 32-way "
            "split-K, merged, captured",
            spec=_step5(
                _spread(24),
                pool_kv_heads=1,
                head_view=None,
                splits=32,
                capture=True,
                **_TAIL_FIRST,
                **_FP8_STEP5,
            ),
        ),
        Cell(
            why="Step-5 tail-first dp 32 decode rows at pos 1..9215, head view 0 of 4: 8-way "
            "split-K, merged, the class's smallest capture batch, captured",
            spec=_step5(
                _spread(32), pool_kv_heads=4, head_view=0, splits=8, capture=True, **_TAIL_FIRST
            ),
        ),
        Cell(
            why="Step-5 tail-first tp4 fp8 1-head pool, 64 decode rows at pos 1..9215: 8-way "
            "split-K, merged, captured",
            spec=_step5(
                _spread(64),
                pool_kv_heads=1,
                head_view=None,
                splits=8,
                capture=True,
                **_TAIL_FIRST,
                **_FP8_STEP5,
            ),
        ),
        Cell(
            why="Step-5 tail-first dp 512-row first chunk at pos 0..511, head view 0 of 4: "
            "4-way split-K, merged, rows of 0..63 regions",
            spec=_step5(((0, 512),), pool_kv_heads=4, head_view=0, splits=4, **_TAIL_FIRST),
        ),
        Cell(
            why="Step-5 tail-first tp4 fp8 1-head pool, 300-row first chunk at pos 0..299: "
            "4-way split-K, merged, rows of 0..37 regions",
            spec=_step5(
                ((0, 300),),
                pool_kv_heads=1,
                head_view=None,
                splits=4,
                **_TAIL_FIRST,
                **_FP8_STEP5,
            ),
        ),
        Cell(
            why="Step-5 tail-first dp fp8 pool, 512-row chunk at pos 3584..4095, head view 2: "
            "4-way split-K, merged, rows of 448..511 regions",
            spec=_step5(
                ((3584, 512),),
                pool_kv_heads=4,
                head_view=2,
                splits=4,
                **_TAIL_FIRST,
                **_FP8_STEP5,
            ),
        ),
        Cell(
            why="Step-5 tail-first tp4 1-head pool, 512-row chunk at pos 4096..4607: 4-way "
            "split-K, merged, every row 512 regions",
            spec=_step5(((4096, 512),), pool_kv_heads=1, head_view=None, splits=4, **_TAIL_FIRST),
        ),
        Cell(
            why="Step-5 tail-first dp 384-row context from 0, then 128 decode rows at pos "
            "1..9215, head view 1: 512 rows on 129 slots, 4-way split-K, merged",
            spec=_step5(
                ((0, 384),) + _SPREAD_128, pool_kv_heads=4, head_view=1, splits=4, **_TAIL_FIRST
            ),
        ),
        # The engine's slot layout (`Step5CacheManager.get_kv_slot_buffers`
        # HND views): a slot stride of 46 pages, layer 22's K/V at pages 44/45
        # (layer 0's at 0/1), the pool's last slot mapped, so the highest
        # element offset read passes 2^33 (2^31 from slot 5699 on one head,
        # 1425 on four).
        Cell(
            why="Step-5 slot pool tp4 bf16 128 decode rows: 27001 slots of 46 pages, layer 22, "
            "last slot read at 1.02e10 elements, 8-way split-K, merged",
            spec=_step5(
                _SPREAD_128,
                pool_kv_heads=1,
                head_view=None,
                splits=8,
                pool_pages=27001,
                layer=22,
                **_SLOT_POOL,
            ),
        ),
        Cell(
            why="Step-5 slot pool tp4 bf16 1 row at 9215: 27001 slots of 46 pages, layer 22, "
            "last slot read, 64-way split-K under PDL, merged",
            spec=_step5(
                ((9215, 1),),
                pool_kv_heads=1,
                head_view=None,
                splits=64,
                pdl=True,
                pool_pages=27001,
                layer=22,
                **_SLOT_POOL,
            ),
        ),
        Cell(
            why="Step-5 slot pool tp4 bf16 512-row chunk at 4096 + a decode row at 8703: 27001 "
            "slots of 46 pages, layer 22, last slot read, one split",
            spec=_step5(
                ((4096, 512), (8703, 1)),
                pool_kv_heads=1,
                head_view=None,
                splits=1,
                pool_pages=27001,
                layer=22,
                **_SLOT_POOL,
            ),
        ),
        Cell(
            why="Step-5 slot pool tp4 bf16 tail-first 128 decode rows, layer 0 (pages 0/1 of "
            "each slot): 27001 slots of 46 pages, last slot read, 8-way split-K, merged",
            spec=_step5(
                _SPREAD_128,
                pool_kv_heads=1,
                head_view=None,
                splits=8,
                pool_pages=27001,
                layer=0,
                **_TAIL_FIRST,
                **_SLOT_POOL,
            ),
        ),
        Cell(
            why="Step-5 slot pool tp4 fp8 128 decode rows: 54001 slots of 46 pages, layer 22, "
            "last slot read at 2.03e10 elements, 8-way split-K, merged",
            spec=_step5(
                _SPREAD_128,
                pool_kv_heads=1,
                head_view=None,
                splits=8,
                pool_pages=54001,
                layer=22,
                **_FP8_STEP5,
                **_SLOT_POOL,
            ),
        ),
        Cell(
            why="Step-5 slot pool tp4 fp8 8192-row prefill from 0: 54001 slots of 46 pages, "
            "layer 22, last slot read, one split",
            spec=_step5(
                ((0, 8192),),
                pool_kv_heads=1,
                head_view=None,
                splits=1,
                pool_pages=54001,
                layer=22,
                **_FP8_STEP5,
                **_SLOT_POOL,
            ),
        ),
        Cell(
            why="Step-5 slot pool dp bf16 32 decode rows, head view 3 of 4: 6751 slots of 46 "
            "pages, layer 22, last slot read at 1.02e10 elements, 8-way split-K, merged",
            spec=_step5(
                _spread(32),
                pool_kv_heads=4,
                head_view=3,
                splits=8,
                pool_pages=6751,
                layer=22,
                **_SLOT_POOL,
            ),
        ),
        Cell(
            why="Step-5 slot pool dp bf16 8192 rows at 1024..9215, head view 2 of 4: 6751 slots "
            "of 46 pages, layer 22, last slot read, one split",
            spec=_step5(
                ((1024, 8192),),
                pool_kv_heads=4,
                head_view=2,
                splits=1,
                pool_pages=6751,
                layer=22,
                **_SLOT_POOL,
            ),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. What the target
    passes: `block_table` is `qsa_block_table`, physical page ids into dim 0
    of the caches (the *unscaled* table; a reference assuming logical ==
    physical lands 306x outside tolerance); `selected_tokens` is
    `qsa_expand_block_indices`' output, logical positions, `-1`-padded;
    `k_cache` / `v_cache` are the layer pool's role planes.

    Preconditions:
      - Whenever the prefill kernel can run, each selection row is packed at
        the front with exactly the bound's count of valid entries (see the
        class docstring). QSA's producer -- `qsa_topk_select` with zero
        `row_starts` and `row_ends = (pos + 1) // cr`, then the expansion --
        was measured to satisfy it at 300, 8191 and 8192 rows, including with
        NaN/inf planted in the scores. Checking it needs the contents, so it
        is not an `is_valid` guard.
      - The prefill pair is a property of the launch: passed because some
        rows are context, it puts the generation rows on the prefill kernel
        too, and the bound applies to them (measured with a planted index; not
        pinned by a test).
      - There is no KV scale argument: an fp8 pool is read at face value (each code
        widens exactly to bf16 under a bf16 q), so it must hold K/V written at unit scale.
      - `D` must be a power of two (192 is refused); `Hq % Hkv == 0`.

    The op validates its own inputs loudly (`ValueError`s; the test pins the
    refusal of positions passed without a compress ratio; a selection width
    the compress ratio cannot produce is refused on the prefill kernel only),
    so `is_valid` adds nothing. `rows == 0` returns an empty `[0, Hq, D]`
    tensor (not certified).

    PDL: `TRTLLM_ENABLE_PDL` (default on) enables programmatic dependent
    launch for multi-split launches of 4 rows or fewer. The QSA target sends
    gated decode batches that small to `qsa_paged_sparse_gqa_decode`; the
    Step-5 1-, 2- and 4-row cells run one (64 splits, then the merge), each
    also captured. Calls are bitwise deterministic (pinned on a
    32-split launch and its merge, and by every graph replay matching an eager
    call).

    The QSA decode site carries up to 8 rows per request (max_draft_len 7):
    the decode cell runs 32 requests of 8 rows, 512 `rows * Hkv` programs and
    a 4-way split-K launch with the output gate in its merge.

    Step-5-Preview (the `Step-5` cells). One call per KV group: q is the
    group's 16 heads sliced from the `[T, 64 | 16, 256]` projection, the
    caches are the single-head views `pool[:, 0 | 1, g:g+1]` of a
    `[pages, 2, 4 | 1, 32, 256]` pool, head_dim 192 zero-padded to 256 in q
    and the pool (the padded output columns come back exactly zero), scale
    192^-0.5, no prefill pair, no gate. Selection width 4104: 512 region
    slots of 8 logical positions and the current region's 8, `-1` past the
    query. The split-K launch is chosen from `rows * Hkv` (= rows here):
    <= 4 rows BLOCK_N 16, 64 splits, PDL; 5..31 BLOCK_N 16, 32 splits;
    32..256 BLOCK_N 64, 8 splits; 257..512 BLOCK_N 64, 4 splits; > 512 one
    split and no merge. Cells drive 1, 2 and 4 rows (64 splits), 8, 16 and
    24 (32 splits), 32, 64, 120, 128 and 256 (8 splits), 300 and 512 (4
    splits; one 512-row cell is a 384-row context request followed by 128
    one-row decode requests on 129 slots), 513 and 8192 (one split). Pool page
    count, table height and width, selection width and every stride are
    compile-time constants: each new pool geometry or view compiles anew
    (first call only). Passing the prefill pair at compress ratio 8 is
    refused above 512 rows (token_topk 4097 is not a multiple of 8) and
    ignored at or below (bit-identical output); both are pinned by a test.

    The engine's pool (the `Step-5 slot pool` cells): the target's caches
    are `Step5CacheManager.get_kv_slot_buffers(layer, "HND")[:, 0 | 1]`,
    head-sliced, of a coalesced pool whose lifecycle slot is 46 pages (K
    then V of each of the 23 sparse layers), so `k_cache.stride(0)` is 46
    pages and layer 22's K/V sit at pages 44/45 of every slot. The cells
    hold that layout at 27001 slots (bf16, one head), 54001 (fp8, one head)
    and 6751 (bf16, four heads, head views 3 and 2), 18.95 GiB each, for
    layer 22 and, on one cell, layer 0; the table maps the pool's last slot
    under the last row's current region, so the highest element offset read
    is about 1.02e10 (both bf16 pools) and 2.03e10 (fp8), past 2^33.
    Every page but the call's K/V pages of the mapped slots is NaN, and the
    outputs carry none. A NaN K/V tile does not come back as NaN: by the
    kernel's code it poisons its split's running sum, `running_sum > 0` is
    false, and the split contributes zeros (measured: the neighbouring
    layer's view, all NaN, returned all-zero rows on every slot-pool cell,
    256 / 256 ulp off the reference), so the band, not a NaN check, catches
    a wrong offset. Measured per cell: at most 2.000 element / 0.636 RMS
    ulp, a peak of 20.68 GiB, at most 2.12 s in pytest; marked `large`
    (2^31 probes). They span 1 row (64 splits, PDL), 128 decode rows in
    both layouts and 32 rows (8 splits), and 513 and 8192 rows (one split).
    One-head offsets pass 2^31 elements from slot 5699 on, four-head
    ones from slot 1425.

    Selection layout: the current region may sit in the last 8 columns
    ("regions_then_tail") or the first 8 ("tail_first"); both are certified
    in every launch class. "regions_then_tail" cells drive 1, 2, 4, 8, 16,
    24, 120, 128, 256, 300, 512, 513 and 8192 rows; "tail_first" cells,
    the layout the target emits, drive 2, 4, 16, 24, 32, 64 and 128 spread
    decode rows, 300- and 512-row chunks from position 0 (rows of fewer
    than 512 regions on 4 splits), 512-row chunks at 3584 and 4096, the
    384-row context plus 128 decode rows, and the 8192-row prefill from 0.
    In "tail_first" every resident token precedes every `-1` region slot,
    so a short row's later splits hold no resident token at all and merge
    as empty. Resident columns need not be packed at the front on the
    split-K kernel. In "regions_then_tail" a row with fewer than 512 regions
    has `-1` region slots between its regions and its current region: the
    rows whose regions end before the last split's first column (4032 on 64
    splits, 3968 on 32, 3584 on 8, 3072 on 4) meet whole tiles of them before
    that split's first resident token, and on one split rows 0..7 hold only
    the current region, after 4096 `-1` columns. The spread decode cells hold
    such rows on 64, 32 and 8 splits; the 4-split "regions_then_tail"
    cells' rows all hold 512 regions. The online-softmax correction is held at 1 while a
    split's running maximum is still `-inf`, so such tiles leave the split's
    state at zero and every later token counts; the cells pin it.

    Not certified: capture in a CUDA graph other than Step-5's 1, 2, 8 and
    120 rows ("regions_then_tail") and 2, 4, 16, 24, 32, 64 and 128 rows
    ("tail_first") (the split-K path allocates its partials inside the
    call); the capture batches 40, 48, 56 and 72..112, neither captured nor
    driven eagerly; head geometries other than 24/2 x 256 and Step-5's 16/1
    x 256; `tokens_per_block` other than 32; more than 8192 rows; row
    counts inside a launch class other than the cells' (40..56 and 72..112
    decode rows, 257..511-row launches other than 300); on 4 splits,
    "regions_then_tail" rows with fewer than 512 regions; "tail_first" at
    one split with decode rows after a chunk (513 rows); the QSA split-K
    kernel with one split or without the gate; rows whose selection
    resolves no token (all padding, a foreign request, unmapped pages); an
    fp16 pool or q; standalone contiguous K/V buffers; int64 request indices;
    Step-5 pools other than the contiguous 20000 pages and the 46-page-slot
    pools of 27001, 54001 and 6751 slots above; in particular the engines'
    measured pools (runtime_facts kv_tokens / 32: 84979 slots at core@tep4
    bf16, 114336 at kv_fp8@tep4, 27465 at core@dep4 bf16 with four heads,
    59.7 to 77.1 GiB) are past the 40 GiB cell budget and are not cells:
    their highest offsets (about 3.2e10, 4.3e10 and 4.1e10 elements) are in
    the same int64 page-offset class the slot-pool cells drive; any sparse
    layer other than 0 and 22; tables other than 129 x 288 (chosen from
    max_batch_size 128 and max_seq_len 9216; the engine's own were not
    measured).
    Hard limits: the kernel's row offsets are int32 products of the program
    id and a stride (computed, not probed; neither envelope reaches one), so
    `row * selected_stride_row` crosses 2^31 above 523264 rows at width
    4104, `row * q_stride_row` above 131072 rows for a 16-head slice of a
    64-head projection, the output above 524288 rows. Page ids are widened
    to int64 before the pool stride: driven by the slot-pool cells, whose
    last slot's K page starts at element 10174824448 (one-head bf16),
    20349288448 (fp8) and 10175930368 (four heads, head view 3), all past
    2^33 (the test asserts it).
    Positions (<= 9215) never pass through fp32. The largest certified
    extent is 8192 rows x 4104 selected tokens through a [129, 288] table
    into a 54001-slot, 46-page-per-slot fp8 pool (18.95 GiB) at Step-5's
    geometry, and 8191 rows x 2051 selected tokens through a [32, 260] table
    at QSA's.
    """

    def __call__(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        query_positions: Optional[torch.Tensor] = None,
        compress_ratio: Optional[int] = None,
        output_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _triton_qsa_paged_sparse_gqa(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            selected_tokens=selected_tokens,
            request_indices=request_indices,
            tokens_per_block=tokens_per_block,
            softmax_scale=softmax_scale,
            query_positions=query_positions,
            compress_ratio=compress_ratio,
            output_gate=output_gate,
        )

    @staticmethod
    def fuses_output_gate(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        selected_tokens: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> bool:
        """Whether a call with these operands and no prefill pair carries `output_gate`."""
        return _can_fuse_qsa_splitk_output_gate(q, k_cache, selected_tokens, output_gate)

    def reference(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        query_positions: Optional[torch.Tensor] = None,
        compress_ratio: Optional[int] = None,
        output_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """fp32 sparse GQA over every resident selected token, rounded once;
        with `output_gate`, then gated in fp32 and rounded again.

        `query_positions` and `compress_ratio` are ignored: on a prefix-packed
        selection the prefill kernel's bound reads exactly the valid entries.
        Walked 32 rows at a time to bound the gathered K/V; both products are
        true fp32 (TF32 off).
        """
        rows, num_q_heads, head_dim = q.shape
        num_pages, num_kv_heads = k_cache.shape[0], k_cache.shape[1]
        groups = num_q_heads // num_kv_heads
        num_requests, table_width = block_table.shape
        requests = request_indices.long()
        out = torch.empty(rows, num_q_heads, head_dim, dtype=torch.float32, device=q.device)
        for start in range(0, rows, 32):
            stop = min(start + 32, rows)
            chunk = selected_tokens[start:stop].long()
            logical_page = chunk.clamp_min(0) // tokens_per_block
            in_page = chunk.clamp_min(0) % tokens_per_block
            request = requests[start:stop]
            owned = (request >= 0) & (request < num_requests)
            physical = block_table[
                request.clamp(0, num_requests - 1)[:, None].expand_as(logical_page),
                logical_page.clamp(max=table_width - 1),
            ].long()
            resident = (
                (chunk >= 0)
                & owned[:, None]
                & (logical_page < table_width)
                & (physical >= 0)
                & (physical < num_pages)
            )
            physical = physical.clamp(0, num_pages - 1)
            keys = k_cache[physical, :, in_page, :].float()
            values = v_cache[physical, :, in_page, :].float()
            grouped = q[start:stop].reshape(-1, num_kv_heads, groups, head_dim).float()
            with true_fp32_matmul():
                scores = torch.einsum("bhgd,bkhd->bhgk", grouped, keys) * softmax_scale
                scores = scores.masked_fill(~resident[:, None, None, :], float("-inf"))
                # A row with nothing resident softmaxes over all -inf: NaN, i.e. zeros.
                weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
                pooled = torch.einsum("bhgk,bkhd->bhgd", weights, values)
            out[start:stop] = pooled.reshape(stop - start, num_q_heads, head_dim)
        if output_gate is None:
            return out.to(q.dtype)
        gate = output_gate.reshape(rows, num_q_heads, head_dim).float()
        return (out.to(q.dtype).float() * torch.sigmoid(gate)).to(q.dtype)


qsa_paged_sparse_gqa = QsaPagedSparseGqa()
