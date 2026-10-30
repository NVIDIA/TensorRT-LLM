# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA: attention over an explicitly selected set of paged K/V tokens.

This target's local variant of the tp1 target's `ops/attention/qsa_paged_sparse_gqa`,
carrying the tp4/tep4 and dep4 cells until the promotion merges the two into the catalog.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Approved exception: QSA's sparse path never reaches the thop attention op's
# sparse argument group, so there is no torch.ops surface for this computation.
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
    can_fuse_qsa_splitk_output_gate as _can_fuse_qsa_splitk_output_gate,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
    triton_qsa_paged_sparse_gqa as _triton_qsa_paged_sparse_gqa,
)

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


#: The engine's trunk layout (`Qwen4ExpHybridCacheManager.get_buffers`): a lifecycle slot
#: holds the 24 role pages of the 12 attention layers, K then V each, and a layer's
#: `k_cache` / `v_cache` are its two planes, so one pool page advances 24 role pages.
#: Layer 11's planes are the slot's last.
_TRUNK = dict(slot_pages=24, layer=11)

#: The element offset the unified target's cells straddle: the requests' pages lie on both
#: sides of it in the layer's K and V planes. An engine's pool reaches far past it (tep4's
#: bf16 engine measured 245835 pages, 4.8e10 elements of trunk).
_OFFSET_EDGE = 2**31

#: One tp4 / tep4 rank (attention TP 4): 6 q heads and one of the 2 KV heads, q the first 6
#: heads of the packed [T, 6 + 1 + 1, 256] QKV buffer; a `max_batch_size` 128-row table,
#: `ceil((max_seq_len + 1) / 32)` padded to a multiple of 4 columns: 292 at max_seq_len 9216,
#: 516 at MTP's 16384.
_TP4 = dict(q_heads=6, kv_heads=1, slots=128, pages=292, **_TRUNK)
_TP4_MTP = dict(_TP4, pages=516)
#: One dep4 rank (attention DP): tp1's 24 / 2 heads, a 32-row table (128 requests over 4
#: ranks), the kv_fp8 variant's e4m3 pool.
_DEP4_FP8 = dict(
    q_heads=24, kv_heads=2, slots=32, pages=292, cache_dtype=torch.float8_e4m3fn, **_TRUNK
)


def _straddling(segments, kv_heads: int) -> int:
    """The pool page count that puts a cell's requests -- the pool's last pages, as the
    test lays them out -- on both sides of `_OFFSET_EDGE` elements of the K plane (half
    of them below it, or the whole pool when they need more than twice the pages below)."""
    tpb, d, slot_pages, layer = 32, 256, _TRUNK["slot_pages"], _TRUNK["layer"]
    needed = sum(math.ceil((s + n) / tpb) for s, n in segments) + 7
    role_page = kv_heads * tpb * d
    edge_page = math.ceil((_OFFSET_EDGE - 2 * layer * role_page) / (slot_pages * role_page))
    return max(edge_page + needed // 2, needed)


def _unified(segments, member: dict, launches, **overrides) -> dict:
    """A cell of this target's call: `member`'s heads, table and trunk layout, its pool
    straddling `_OFFSET_EDGE`."""
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
        # ---- the unified target: one rank's heads, table and trunk-layout pool ------
        Cell(
            why="tp4/tep4 decode, 1 row at position 9215 (max_seq_len, table column 287): "
            "6 q / 1 kv heads, 64-way split-K + merge with PDL, [128, 292] table",
            spec=_unified(_spread(1, 1, 9215), _TP4, _SPLIT_K, pdl=True),
        ),
        Cell(
            why="tp4/tep4 decode, 128 requests x 1 row (the largest captured batch): "
            "6 q / 1 kv heads, 8-way split-K + merge, [128, 292] table",
            spec=_unified(_spread(128, 1, 9215), _TP4, _SPLIT_K),
        ),
        Cell(
            why="tep4 MTP7 verify, 128 requests x 8 rows = 1024: 6 q / 1 kv heads, one split "
            "and no merge, [128, 516] table (max_seq_len 16384), rows to position 16383",
            spec=_unified(_spread(128, 8, 16376), _TP4_MTP, ("splitk",), rows_per_request=8),
        ),
        Cell(
            why="tp4/tep4 8192-row step (max_num_tokens) with the prefill pair: a context "
            "request 0..8064 and 127 generation rows to position 9215, all on the prefill "
            "kernel, 6 q / 1 kv heads",
            spec=_unified(((0, 8065),) + _spread(127, 1, 9215), _TP4, _PREFILL, prefill_pair=True),
        ),
        Cell(
            why="dep4 kv_fp8 decode, 32 requests x 1 row (a rank's batch): 24 q / 2 kv heads "
            "over an e4m3 pool, 8-way split-K + merge, [32, 292] table",
            spec=_unified(_spread(32, 1, 9215), _DEP4_FP8, _SPLIT_K),
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
        too, and the bound applies to them (measured with a planted index;
        the unified 8192-row cell drives 127 generation rows through it).
      - There is no KV scale argument: an fp8 pool is read at face value (each code
        widens exactly to bf16 under a bf16 q), so it must hold K/V written at unit scale.
      - `D` must be a power of two (192 is refused); `Hq % Hkv == 0`.

    The op validates its own inputs loudly (`ValueError`s; the test pins the
    refusal of positions passed without a compress ratio; a selection width
    the compress ratio cannot produce is refused on the prefill kernel only),
    so `is_valid` adds nothing. `rows == 0` returns an empty `[0, Hq, D]`
    tensor (not certified).

    PDL: `TRTLLM_ENABLE_PDL` (default on) enables programmatic dependent
    launch for multi-split launches of 4 rows or fewer; the unified 1-row
    cell runs one (split-K and merge both launched with PDL, asserted). The
    tp1 target sends gated decode batches that small to
    `qsa_paged_sparse_gqa_decode`. Calls are bitwise deterministic (pinned
    on a 32-split launch and its merge).

    The decode site carries up to 8 rows per request (max_draft_len 7): the
    decode cell runs 32 requests of 8 rows, 512 `rows * Hkv` programs and a
    4-way split-K launch with the output gate in its merge.

    The unified target (tp4 / tep4 / dep4). A tp4 or tep4 rank holds 6 q heads
    and one KV head (`Hkv` 1, a group of 6 padded to 16 MMA rows), so its
    launch follows `rows`: 64-way split-K up to 4 rows, 32-way below 32,
    8-way to 256, 4-way to 512, then one split and no merge (the 1024-row
    MTP7 verify step, where the output gate cannot ride a merge and
    `fuses_output_gate` is False); with the prefill pair, the prefill kernel
    past 512 rows. A dep4 rank holds tp1's 24 / 2 heads, 32 requests, and
    under kv_fp8 an e4m3 pool, read on the split-K kernel. Its cells drive
    the engine's trunk layout (a page advances 24 role pages; layer 11's
    planes) over `[128, 292]`, `[128, 516]` (MTP's max_seq_len 16384) and
    `[32, 292]` tables, with pools whose mapped pages straddle 2^31 elements of
    the K and V planes: both kernels widen the page id to int64 before its
    stride product. Measured against `reference` on GB300 over the five
    cells: at most 2.00 bf16 ulp of the row scale (p99.9 1.31), relative RMS
    at most 0.572 ulp, cosine at least 0.9999975.

    Not certified: capture in a CUDA graph (the split-K path allocates its
    partials inside the call); head geometries other than 24/2 and 6/1 x 256;
    `tokens_per_block` other than 32; more than 8192 rows; the output gate at
    6/1 heads (the gate is certified at 24/2 only, in the merge of a 4-way
    split); q off the fused projection row at tp4's 17-head (4352-wide) row;
    an fp8 pool on the prefill kernel at 24/2 beyond 272 rows, and at 6/1;
    rows whose selection resolves no token (all padding, a foreign request,
    unmapped pages); an fp16 pool or q; standalone contiguous K/V buffers;
    int64 request indices; a measured dep4 pool page count (its cell's pool
    is sized to straddle 2^31 elements, not to an engine's allocation).
    Hard limits: not probed; the largest certified extent is 8192 rows x 2051
    selected tokens, a [128, 516] table and a 33167-page trunk pool.
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
        Walked 32 rows at a time to bound the gathered K/V.
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
