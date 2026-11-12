# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA: attention over an explicitly selected set of paged K/V tokens.

This target's copy of the family entry `models/qwen4_exp/ops/attention/qsa_paged_sparse_gqa.py`,
launching this target's stable-JIT kernels."""

from __future__ import annotations

import math
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .qsa_triton_kernels__stable_jit import (
    can_fuse_splitk_output_gate as _can_fuse_qsa_splitk_output_gate,
)
from .qsa_triton_kernels__stable_jit import qsa_paged_sparse_gqa as _triton_qsa_paged_sparse_gqa

_CHECKPOINT = dict(
    q_heads=24,
    kv_heads=2,
    head_dim=256,
    tokens_per_block=32,
    compress_ratio=4,
    token_topk=2048,
    cache_dtype=torch.bfloat16,
    q_dtype=torch.bfloat16,
    # q is the first 24 heads of the packed [T, 24 + 2 + 2, 256] QKV buffer, or ("fused_row")
    # of the fused input projection row.
    q_layout="qkv_slice",
    request_dtype=torch.int32,
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


#: The engine's trunk layout (`Qwen4ExpHybridCacheManager.get_buffers`): a pool page holds the
#: K and V planes of the 12 attention layers (24 role pages); layer 11's planes are the last.
_TRUNK = dict(slot_pages=24, layer=11)

#: The element offset the trunk cells' pages straddle in the K and V planes.
_OFFSET_EDGE = 2**31

#: One tp4 / tep4 rank: 6 q heads and one KV head; a 128-row table of 292 columns
#: (max_seq_len 9216), 516 at MTP's 16384.
_TP4 = dict(q_heads=6, kv_heads=1, slots=128, pages=292, **_TRUNK)
_TP4_MTP = dict(_TP4, pages=516)
#: One dep4 rank: 24 / 2 heads and a 32-row table.
_DEP4 = dict(q_heads=24, kv_heads=2, slots=32, pages=292, **_TRUNK)
_DEP4_FP8 = dict(_DEP4, **_FP8)

#: q and the output gate off one rank's fused input projection row `[q | k | v | gate | index]`
#: in heads of 256: 17 heads at a tp4 / tep4 rank, 55 at dep4.
_TP4_ROW = dict(q_layout="fused_row", gate="fused_row", row_heads=17)
_DEP4_ROW = dict(q_layout="fused_row", gate="fused_row", row_heads=55)

#: One tp2 / tep2 rank: 12 q heads and one KV head, a 128-row table of 292 columns.
_TP2 = dict(q_heads=12, kv_heads=1, slots=128, pages=292, **_TRUNK)
#: A tp2 / tep2 rank's fused row: 29 heads.
_TP2_ROW = dict(q_layout="fused_row", gate="fused_row", row_heads=29)
#: One tp1 or dep2 rank: 24 / 2 heads and a 128-row table.
_TP1 = dict(q_heads=24, kv_heads=2, slots=128, pages=292, **_TRUNK)
_TP1_FP8 = dict(_TP1, **_FP8)


def _straddling(segments, kv_heads: int) -> int:
    """The pool page count that puts a cell's requests (the pool's last pages) on both sides of
    `_OFFSET_EDGE` elements of the K plane."""
    tpb, d, slot_pages, layer = 32, 256, _TRUNK["slot_pages"], _TRUNK["layer"]
    needed = sum(math.ceil((s + n) / tpb) for s, n in segments) + 7
    role_page = kv_heads * tpb * d
    edge_page = math.ceil((_OFFSET_EDGE - 2 * layer * role_page) / (slot_pages * role_page))
    return max(edge_page + needed // 2, needed)


def _default(segments, member: dict, launches, **overrides) -> dict:
    """A cell at `member`'s heads, table and trunk layout, its pool straddling `_OFFSET_EDGE`."""
    return _decode(
        segments,
        launches=launches,
        rows=sum(n for _, n in segments),
        pool_pages=_straddling(segments, member["kv_heads"]),
        offset_edge=_OFFSET_EDGE,
        **{**member, **overrides},
    )


def _spread(count: int, rows: int, last_start: int) -> tuple:
    """`count` requests of `rows` rows: one starting at `last_start` (its last row at the
    envelope's last position), the others spread from position 0 up."""
    span = last_start + 1
    return tuple(((5 + 1129 * g) % span, rows) for g in range(count - 1)) + ((last_start, rows),)


class QsaPagedSparseGqa(OpWrapper):
    """Sparse GQA over the selected tokens of an HND paged K/V pool.

    For row `r`, query head `h`, KV head `h // (Hq // Hkv)` (contiguous groups), over the
    columns `j` of `selected_tokens[r]` that name a resident token::

        t     = selected_tokens[r, j]                      # logical position
        page  = block_table[request_indices[r], t // tokens_per_block]
        s_j   = (q[r, h] . k_cache[page, kv, t % tpb]) * softmax_scale
        out[r, h] = sum_j softmax(s)_j * v_cache[page, kv, t % tpb]

    Non-resident columns (negative index, page past the table, negative or out-of-pool table
    entry, request outside the table) contribute nothing; a row left with nothing returns zeros.
    Indices are not deduplicated. Returns a fresh contiguous `[rows, Hq, D]` tensor of `q.dtype`.

    Two kernels, chosen inside the call: with `query_positions` and `compress_ratio` given and
    `rows * Hkv > 512`, a causal-prefix prefill kernel; otherwise split-K, plus a merge kernel
    when it used more than one split.

    **The prefill pair is a bound, not a mask.** The prefill kernel reads only the
    `min(visible // cr, block_topk) * cr + visible % cr` leading columns of a row
    (`visible = query_positions[r] + 1`); a selection with valid entries past the bound is
    silently truncated. QSA's top-k + expansion emit exactly that prefix.

    **The output gate rides the merge when `output_gate` is passed:** each merged head row is
    rounded to `q.dtype`, multiplied in fp32 by `sigmoid(output_gate)` and rounded again (the
    ungated call followed by `fused_sigmoid_mul_inplace`). `output_gate` is `[rows, Hq * D]` or
    `[rows, Hq, D]` in `q.dtype` with unit last stride. Only a multi-split launch has a merge to
    carry it; `fuses_output_gate` says whether a call does, and the op raises `ValueError` if
    handed a gate it cannot carry.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # ---- tp4 / tep4 / dep4 ranks: 6 / 1 and 24 / 2 heads, trunk-layout pools ----------
        Cell(
            why="tp4/tep4 decode, 128 requests x 1 row (the largest captured batch): "
            "6 q / 1 kv heads, 8-way split-K + merge, [128, 292] table",
            spec=_default(_spread(128, 1, 9215), _TP4, _SPLIT_K),
        ),
        Cell(
            why="tep4 MTP7 verify, 128 requests x 8 rows = 1024: 6 q / 1 kv heads, one split "
            "and no merge, [128, 516] table (max_seq_len 16384), rows to position 16383",
            spec=_default(_spread(128, 8, 16376), _TP4_MTP, ("splitk",), rows_per_request=8),
        ),
        # ---- the gated decode / verify site, one cell per 6 / 1 launch bucket -------------
        Cell(
            why="tep4 gated decode 1 row in a CUDA graph: q and the gate off the 17-head "
            "(4352-wide) fused row, 64-way split-K + gated merge with PDL, captured, refilled "
            "and replayed",
            spec=_default(
                _spread(1, 1, 9215), _TP4, _SPLIT_K, splits=64, pdl=True, capture=True, **_TP4_ROW
            ),
        ),
        Cell(
            why="tep4 gated decode 3 rows (PDL's 2..4-row branch): 64-way split-K + gated merge "
            "with PDL, 17-head fused row",
            spec=_default(_spread(3, 1, 9215), _TP4, _SPLIT_K, splits=64, pdl=True, **_TP4_ROW),
        ),
        Cell(
            why="tep4 gated decode 16 rows (the 5..31-row branch): 32-way split-K + gated merge, "
            "17-head fused row",
            spec=_default(_spread(16, 1, 9215), _TP4, _SPLIT_K, splits=32, **_TP4_ROW),
        ),
        Cell(
            why="tep4 gated decode 128 x 1 row in a CUDA graph (the largest captured batch): "
            "8-way split-K + gated merge, 17-head fused row, captured, refilled and replayed",
            spec=_default(
                _spread(128, 1, 9215), _TP4, _SPLIT_K, splits=8, capture=True, **_TP4_ROW
            ),
        ),
        Cell(
            why="tep4 MTP3 gated verify 128 x 4 rows = 512 (the 257..512-row branch): 4-way "
            "split-K + gated merge, [128, 516] table, rows to position 16383, 17-head fused row",
            spec=_default(
                _spread(128, 4, 16380),
                _TP4_MTP,
                _SPLIT_K,
                splits=4,
                rows_per_request=4,
                **_TP4_ROW,
            ),
        ),
        Cell(
            why="dep4 kv_fp8 context 8192 rows from position 0 with the prefill pair: 24 q / 2 "
            "kv heads over an e4m3 pool on the prefill kernel, [32, 292] table",
            spec=_default(((0, 8192),), _DEP4_FP8, _PREFILL, prefill_pair=True),
        ),
        Cell(
            why="tep4 8192-row context step (max_num_tokens) with the prefill pair, q off the "
            "17-head fused row: a context request 0..8064 and 127 generation rows to position "
            "9215, all on the prefill kernel",
            spec=_default(
                ((0, 8065),) + _spread(127, 1, 9215),
                _TP4,
                _PREFILL,
                prefill_pair=True,
                q_layout="fused_row",
                row_heads=17,
            ),
        ),
        # ---- this target's tp2 / tep2 ranks: 12 / 1 heads off the 29-head fused row --------
        Cell(
            why="tp2/tep2 gated decode 1 row at position 9215 in a CUDA graph: 12 q / 1 kv heads, "
            "q and the gate off the 29-head (7424-wide) fused row, 64-way split-K + gated merge "
            "with PDL, [128, 292] table, captured, refilled and replayed",
            spec=_default(
                _spread(1, 1, 9215), _TP2, _SPLIT_K, splits=64, pdl=True, capture=True, **_TP2_ROW
            ),
        ),
        Cell(
            why="tp2/tep2 8192-row step (max_num_tokens) with the prefill pair, q off the 29-head "
            "fused row: a context request 0..8064 and 127 generation rows to position 9215, all "
            "on the prefill kernel, 12 / 1 heads",
            spec=_default(
                ((0, 8065),) + _spread(127, 1, 9215),
                _TP2,
                _PREFILL,
                prefill_pair=True,
                q_layout="fused_row",
                row_heads=29,
            ),
        ),
        # ---- this target's tp1 / dep2 ranks: 24 / 2 heads off the 55-head fused row --------
        Cell(
            why="tp1/dep2 gated decode 128 x 1 row in a CUDA graph: 24 q / 2 kv heads, q and the "
            "gate off the 55-head (14080-wide) fused row, 8-way split-K + gated merge, [128, 292] "
            "table",
            spec=_default(
                _spread(128, 1, 9215), _TP1, _SPLIT_K, splits=8, capture=True, **_DEP4_ROW
            ),
        ),
        Cell(
            why="tp1/dep2 MTP3 verify 128 x 4 = 512 rows in a CUDA graph: at 24 / 2 heads that is "
            "1024 programs, so one split and no merge, and the gate off the 55-head row cannot "
            "ride it (fuses_output_gate False); q off that row, rows to position 9215",
            spec=_default(
                _spread(128, 4, 9212),
                _TP1,
                ("splitk",),
                splits=1,
                capture=True,
                rows_per_request=4,
                gate_fused=False,
                **_DEP4_ROW,
            ),
        ),
        Cell(
            why="kv_fp8 gated decode 128 x 1 row in a CUDA graph: 24 / 2 heads over an e4m3 pool, "
            "8-way split-K + gated merge, 55-head fused row, [128, 292] table",
            spec=_default(
                _spread(128, 1, 9215), _TP1_FP8, _SPLIT_K, splits=8, capture=True, **_DEP4_ROW
            ),
        ),
        Cell(
            why="tp1/dep2 8192-row step (max_num_tokens) with the prefill pair, q off the 55-head "
            "fused row: a context request 0..8064 and 127 generation rows to position 9215, all "
            "on the prefill kernel, 24 / 2 heads, [128, 292] table",
            spec=_default(
                ((0, 8065),) + _spread(127, 1, 9215),
                _TP1,
                _PREFILL,
                prefill_pair=True,
                q_layout="fused_row",
                row_heads=55,
            ),
        ),
    )

    note = """
    Source: `_qsa_prefill_pair_stable_kernel`, `_qsa_splitk_stable_kernel` and
    `_qsa_merge_stable_kernel` in `qsa_triton_kernels__stable_jit` (Triton JIT), copies of the
    shared QSA kernels with pool and table extents, row count and metadata alignment
    unspecialized: one compile per call site layout, pool dtype and launch bucket
    (`splitk_launch_config`), whatever the engine's pool or batch.

    The target passes `qsa_block_table` (physical page ids), `qsa_expand_block_indices`'
    output as `selected_tokens` (logical positions, `-1`-padded) and the layer's role planes.

    Preconditions:
      - whenever the prefill kernel can run, each selection row is packed at the front with
        exactly the bound's count of valid entries (`qsa_topk_select` + expansion satisfy it);
        passed because some rows are context, the pair puts the generation rows on the
        prefill kernel too.
      - an fp8 pool is read at unit scale (there is no KV scale argument).
      - `D` a power of two; `Hq % Hkv == 0`.
    Inputs are validated loudly (`ValueError`), so `is_valid` adds nothing. Page ids are
    widened to int64 before the stride product (the trunk cells straddle 2^31 elements).
    The 1-row and 128-row decodes and the one-split verify are captured in a CUDA graph and
    replayed over inputs refilled in place.

    Not certified: head geometries other than 24/2, 12/1 and 6/1 x 256; `tokens_per_block`
    other than 32; more than 8192 rows; `rows == 0`; capture of the prefill kernel and of the
    1024-row one-split verify at 6/1; gated decode at 24/2 below 128 rows; a trunk with the MTP
    draft layer appended (26 role pages); an fp8 pool at 6/1 or 12/1; positions past 16383;
    rows whose selection resolves no token; an fp16 pool or q; int64 request indices; the
    engines' own pool page counts (the cells' pools hold 5592 to 33167 pages).
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
        """fp32 sparse GQA over every resident selected token, rounded once; with
        `output_gate`, then gated in fp32 and rounded again. `query_positions` and
        `compress_ratio` are ignored (on a prefix-packed selection the bound reads exactly the
        valid entries)."""
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
            # What a non-resident column gathers (a clamped page) is not part of the answer:
            # zeroed, so a pool holding NaN outside the mapped pages cannot reach the sum.
            absent = ~resident[:, :, None, None]
            keys = k_cache[physical, :, in_page, :].float().masked_fill(absent, 0.0)
            values = v_cache[physical, :, in_page, :].float().masked_fill(absent, 0.0)
            grouped = q[start:stop].reshape(-1, num_kv_heads, groups, head_dim).float()
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
