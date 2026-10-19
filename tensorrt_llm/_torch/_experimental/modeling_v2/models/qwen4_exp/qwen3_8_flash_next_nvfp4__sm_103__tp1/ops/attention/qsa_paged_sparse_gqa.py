# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA: attention over an explicitly selected set of paged K/V tokens."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Approved exception: QSA's sparse path never reaches the thop attention op's
# sparse argument group, so there is no torch.ops surface for this computation.
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
    # K and V are the two role planes of one [pages, 2, kv, tpb, D] pool.
    cache_layout="pool_planes",
    # q is the first 24 heads of the packed [T, 24 + 2 + 2, 256] QKV buffer.
    q_layout="qkv_slice",
    request_dtype=torch.int32,
    # The engine's table: 260 pages per slot, -1 past each sequence.
    pages=260,
)

_SPLIT_K = ("splitk", "merge")
_PREFILL = ("prefill",)


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


def _mixed(requests, context_lens) -> dict:
    return _spec(mixed=(requests, context_lens), slots=32, prefill_pair=True, launches=_PREFILL)


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

    Fusion boundary: the block-table gather, scale, online softmax, P.V and
    the split-K merge are inside. Deciding what to attend to, appending K/V to
    the pool, and q's projection, norm, RoPE and output gate are not.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # ---- generation: no prefill pair, split-K ---------------------------
        Cell(
            why="decode: 1 row past the budget, split-K + merge, PDL",
            spec=_decode(((4000, 1),)),
        ),
        Cell(
            why="decode: 4 requests x 1 row, split-K + merge, PDL",
            spec=_decode(((4000, 1), (4037, 1), (4074, 1), (4111, 1))),
        ),
        Cell(
            why="decode MTP: 4 requests x 2 rows, split-K + merge",
            spec=_decode(tuple((4000 + 37 * i, 2) for i in range(4))),
        ),
        Cell(
            why="decode MTP: 8 requests x 2 rows, 8-way split-K",
            spec=_decode(tuple((4000 + 37 * i, 2) for i in range(8))),
        ),
        Cell(
            why="decode MTP: all 32 slots x 2 rows, positions 100..8099",
            spec=_decode(tuple((100 + (499 * g) % 8000, 2) for g in range(32))),
        ),
        Cell(
            why="300 rows, no prefill pair: split-K, one split, no merge",
            spec=_decode(((600, 1),) * 300, launches=("splitk",), slots=384),
        ),
        Cell(
            why="16-wide selection: one tile, the split-K kernel writes out",
            spec=_decode(((600, 1),) * 4, launches=("splitk",), width=16),
        ),
        # ---- context: the prefill pair is passed ----------------------------
        Cell(
            why="context 128 rows from 2032: pair passed, still split-K",
            spec=_decode(((2032, 128),), prefill_pair=True),
        ),
        Cell(
            why="context 272 rows from 2032: causal-prefix prefill kernel",
            spec=_decode(((2032, 272),), launches=_PREFILL, prefill_pair=True),
        ),
        Cell(
            why="context: two requests (150 @1900, 150 @900) in one launch",
            spec=_decode(((1900, 150), (900, 150)), launches=_PREFILL, prefill_pair=True),
        ),
        Cell(
            why="context 300 rows from 0: rows see 1..300 tokens",
            spec=_decode(((0, 300),), launches=_PREFILL, prefill_pair=True),
        ),
        Cell(
            why="warm-up capture: 8191 rows of one request from 0",
            spec=_decode(((0, 8191),), launches=_PREFILL, prefill_pair=True),
        ),
        Cell(
            why="warm-up capture: 2 requests x 4096 rows from 0",
            spec=_decode(((0, 4096), (0, 4096)), launches=_PREFILL, prefill_pair=True),
        ),
        # ---- mixed steps from the mmmu gate's runtime dump -------------------
        Cell(why="mixed 257 rows: 1 chunk + 31 MTP decodes", spec=_mixed(32, (195,))),
        Cell(why="mixed 318 rows: 2 chunks + 30 MTP decodes", spec=_mixed(32, (200, 58))),
        Cell(why="mixed 478 rows over 28 slots: 1 chunk", spec=_mixed(28, (424,))),
        Cell(
            why="mixed 869 rows: 5 chunks + 27 MTP decodes",
            spec=_mixed(32, (400, 200, 100, 80, 35)),
        ),
        Cell(
            why="mixed 1334 rows: 4 chunks + 28 MTP decodes",
            spec=_mixed(32, (900, 200, 150, 28)),
        ),
        Cell(
            why="mixed 3806 rows: 9 chunks + 23 MTP decodes",
            spec=_mixed(32, (1024, 900, 700, 500, 300, 200, 100, 30, 6)),
        ),
        Cell(
            why="mixed 5130 rows: 10 chunks + 22 MTP decodes",
            spec=_mixed(32, (2048, 1024, 700, 500, 400, 200, 120, 60, 30, 4)),
        ),
        Cell(
            why="mixed 8169 rows over 31 slots: 10 chunks",
            spec=_mixed(31, (4096, 2048, 1024, 500, 200, 120, 80, 40, 15, 4)),
        ),
        Cell(why="mixed 8190 rows: 4 chunks + 28 MTP decodes", spec=_mixed(32, (8000, 100, 30, 4))),
        # ---- formats and layouts --------------------------------------------
        Cell(
            why="fp16 pool and fp16 q",
            spec=_decode(((900, 1),) * 4, cache_dtype=torch.float16, q_dtype=torch.float16),
        ),
        Cell(
            why="fp8-e4m3 pool under a bf16 q, read unscaled",
            spec=_decode(((900, 1),) * 4, cache_dtype=torch.float8_e4m3fn),
        ),
        Cell(
            why="int64 request indices", spec=_decode(((1000, 2),) * 4, request_dtype=torch.int64)
        ),
        Cell(
            why="standalone contiguous K and V buffers, contiguous q",
            spec=_decode(((900, 1),) * 4, cache_layout="contiguous", q_layout="contiguous"),
        ),
        Cell(
            why="holes: all-padding row, foreign request, unmapped page",
            spec=_decode(((900, 1),) * 4, holes=True),
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
        too, and the bound applies to them (asserted with a planted index).
      - There is no KV scale argument: an fp8 pool is read at face value.
      - `D` must be a power of two (192 is refused); `Hq % Hkv == 0`.

    The op validates its own inputs loudly (11 `ValueError`s pinned by the
    test, plus a selection width the compress ratio cannot produce, refused
    on the prefill kernel only), so `is_valid` adds nothing. `rows == 0`
    returns an empty `[0, Hq, D]` tensor.

    PDL: `TRTLLM_ENABLE_PDL` (default on) enables programmatic dependent
    launch for multi-split launches of 4 rows or fewer; both launch modes are
    inside the cells. Calls are bitwise deterministic.

    Not certified: capture in a CUDA graph (the split-K path allocates its
    partials inside the call); head geometries other than 24/2 x 256;
    `tokens_per_block` other than 32; more than 8192 rows; the split-K kernel
    above 300 rows. Hard limits: not probed; the largest certified extent is
    8192 rows x 2051 selected tokens through a [32, 260] table.
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
        )

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
    ) -> torch.Tensor:
        """fp32 sparse GQA over every resident selected token, rounded once.

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
            keys = k_cache[physical, :, in_page, :].float()
            values = v_cache[physical, :, in_page, :].float()
            grouped = q[start:stop].reshape(-1, num_kv_heads, groups, head_dim).float()
            scores = torch.einsum("bhgd,bkhd->bhgk", grouped, keys) * softmax_scale
            scores = scores.masked_fill(~resident[:, None, None, :], float("-inf"))
            # A row with nothing resident softmaxes over all -inf: NaN, i.e. zeros.
            weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            pooled = torch.einsum("bhgk,bkhd->bhgd", weights, values)
            out[start:stop] = pooled.reshape(stop - start, num_q_heads, head_dim)
        return out.to(q.dtype)


qsa_paged_sparse_gqa = QsaPagedSparseGqa()
