# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA: attention over an explicitly selected set of paged K/V tokens.

This target's local variant of the tp1 target's `ops/attention/qsa_paged_sparse_gqa`,
carrying the tp4/tep4 and dep4 cells until the promotion merges the two into the catalog.
"""

from __future__ import annotations

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
    launch for multi-split launches of 4 rows or fewer; no cell runs one (the
    target sends gated decode batches that small to
    `qsa_paged_sparse_gqa_decode`). Calls are bitwise deterministic (pinned
    on a 32-split launch and its merge).

    The decode site carries up to 8 rows per request (max_draft_len 7): the
    decode cell runs 32 requests of 8 rows, 512 `rows * Hkv` programs and a
    4-way split-K launch with the output gate in its merge.

    Not certified: capture in a CUDA graph (the split-K path allocates its
    partials inside the call); PDL launches; head geometries other than 24/2
    x 256; `tokens_per_block` other than 32; more than 8191 rows; the split-K
    kernel above 256 rows, with one split (no merge) or without the gate; an
    fp8 pool on the split-K kernel; rows whose selection resolves no token
    (all padding, a foreign request, unmapped pages); an fp16 pool or q;
    standalone contiguous K/V buffers; int64 request indices.
    Hard limits: not probed; the largest certified extent is 8191 rows x 2051
    selected tokens through a [32, 260] table.
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
            keys = k_cache[physical, :, in_page, :].float()
            values = v_cache[physical, :, in_page, :].float()
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
