# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic per-row top-k over each row's own column range (FlashInfer ragged top-k).

The callable is the FlashInfer Python API ``flashinfer.top_k_ragged_transform``, with
``deterministic=True``, ``tie_break=TopKTieBreak.SMALL`` and ``dsa_graph_safe=True`` pinned: on
that path one call launches FlashInfer's one-block-per-row ``FilteredTopKUnifiedKernel`` (scalar
loads) and then its ``FinalizeTopKIndicesKernel``, which sorts each row's selection into
ascending order.
"""

from __future__ import annotations

from typing import Optional

import torch
from flashinfer import top_k_ragged_transform
from flashinfer.topk import TopKTieBreak

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_K = 512  # the checkpoint's index_topk


def _engine_widths() -> tuple[tuple[int, int], ...]:
    """(W, compress ratio) of every static generation width: the 8192-token envelope's eleven, then
    for each long max_seq_len L the DSpark eager `(L + 14) // r` and graph `(L + 19) // r` widths and
    the DSpark-off `L // r`."""
    widths = [
        (8192, 2),
        (16384, 1),
        (32768, 1),
        (8194, 2),
        (16386, 2),
        (16389, 1),
        (32773, 1),
        (8199, 2),
        (8201, 2),
        (16398, 1),
        (16403, 1),
    ]
    for length in _MAX_SEQ_LENS[1:]:
        for w, r in (
            (length + 14, 1),
            (length + 19, 1),
            ((length + 14) // 2, 2),
            ((length + 19) // 2, 2),
            (length, 1),
            (length // 2, 2),
        ):
            if (w, r) not in widths:
                widths.append((w, r))
    return tuple(widths)


#: max_seq_len values of the envelope; with DSpark the cache manager's length is L + 14.
_MAX_SEQ_LENS = (16384, 32768, 65536, 131072, 262144, 1048576)
_ENGINE_WIDTHS = _engine_widths()
#: The generation rows' scores: the paged-logits view at every width, and the re-index layers'
#: candidate-masked contiguous copy at the ratio-1 widths where candidate blocks can bind.
_GEN_FORMS = tuple((w, r, "paged_view") for w, r in _ENGINE_WIDTHS) + tuple(
    (w, r, "candidate_masked") for w, r in _ENGINE_WIDTHS if r == 1 and w > 16384
)
#: The last 16384-token chunk of one request at each manager length Lm, both ratios, in the
#: prefill-logits layout, and at ratio 1 above 16384 positions as the candidate-masked copy.
_FAR_END = tuple((n + d, r) for n in _MAX_SEQ_LENS for d in (14, 0) for r in (1, 2))
_FAR_END_FORMS = tuple((lm, r, "mqa_view") for lm, r in _FAR_END) + tuple(
    (lm, 1, "candidate_masked") for lm, r in _FAR_END if r == 1 and lm > 16384
)
#: Long-context steps the target runs in row tiles, as (cached, new) requests in batch order.
_TILE_STEPS = (
    ("far_Lm1048590", ((1048590 - 16384, 16384),)),
    ("far_Lm1048576", ((1048576 - 16384, 16384),)),
    ("s1048219", ((1048219 - 8192, 8192),)),
    ("s16384", ((0, 16384),)),
    ("mixed_long", ((262158 - 4000, 4000), (131086 - 8000, 8000), (0, 4000), (65550 - 384, 384))),
    ("two_1M", ((1048590 - 8192, 8192),) * 2),
    ("b256_1M", ((1048590 - 64, 64),) * 256),
)
_TILE_MASKED = ("far_Lm1048590", "s1048219", "mixed_long", "two_1M")
#: Where the K-th value is tied: (rows, W, ratio, layout).
_TIE_FORMS = (
    ("decode", 16384, 1, "paged_view"),
    ("decode", 16403, 1, "candidate_masked"),
    ("decode", 32768, 1, "paged_view"),
    ("decode", 32773, 1, "candidate_masked"),
    ("decode", 8201, 2, "paged_view"),
    ("prefill", 8192, 1, "mqa_view"),
    ("prefill", 4096, 2, "mqa_view"),
    ("decode", 1048595, 1, "paged_view"),
    ("decode", 1048595, 1, "candidate_masked"),
    ("decode", 524297, 2, "paged_view"),
    ("prefill", 131072, 1, "mqa_view"),
)
_EVERY_LEN_LM = 262158
_EVERY_LEN_CHUNKS = -(-_EVERY_LEN_LM // 16384)


class FlashinferTopKRaggedDeterministic(OpWrapper):
    """Return a fresh int32 ``[N, k]`` of each row's top-k columns, in ascending column order.

    For row ``i`` with ``s = row_starts[i]`` (0 when None), ``L = lengths[i]``, ``o = offsets[i]``:

    - ``L <= 0``: all -1;
    - ``0 < L <= k``: ``o, o + 1, ..., o + L - 1`` then -1 (the scores are not read);
    - ``L > k``: the ``k`` columns ``j < L`` with the largest ``scores[i, s + j]`` (fp32 bit-pattern
      order: -0.0 below +0.0; -inf entries tie), ties at the k-th value to the smaller ``j``,
      sorted ascending, plus ``o``. No -1.

    Fusion boundary: selection, the ascending sort and adding ``offsets`` are inside the call.
    Producing the scores (masking applied as -inf beforehand), the lengths and column starts, and
    turning window-local indices into pool slots are the caller's. Returns the new tensor; the
    caller copies it into its rows of the index buffer when a step has both kinds of rows.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        (
            Cell(
                why="row semantics: lengths -3..4096, offsets and row_starts, 96 rows",
                spec=dict(kind="semantics"),
            ),
        )
        + tuple(
            Cell(
                why=f"gen W={w} r={r} {layout}: 1536 verify rows, every prefix, next_n 1",
                spec=dict(kind="gen", width=w, ratio=r, layout=layout),
            )
            for w, r, layout in _GEN_FORMS
        )
        + tuple(
            Cell(
                why=f"ctx every length r={r} chunk={c}: 16384 rows of a 262158-token request",
                spec=dict(kind="every_len", ratio=r, chunk=c),
            )
            for r in (1, 2)
            for c in range(_EVERY_LEN_CHUNKS)
        )
        + tuple(
            Cell(
                why=f"ctx widest step r={r}: 32 requests x 256 rows, the row past 2^{33 - r} elements",
                spec=dict(kind="widest", ratio=r),
            )
            for r in (1, 2)
        )
        + tuple(
            Cell(
                why=f"ctx 24K passkey step {form}: 7674 rows x 24058, two requests",
                spec=dict(kind="long_ctx", layout=form),
            )
            for form in ("mqa_view", "candidate_masked")
        )
        + tuple(
            Cell(
                why=f"ctx far end Lm={lm} r={r} {form}: the last 16384-token chunk",
                spec=dict(kind="far_end", lm=lm, ratio=r, layout=form),
            )
            for lm, r, form in _FAR_END_FORMS
        )
        + tuple(
            Cell(
                why=f"ctx tiles {name} r={r} {form}: every tile as the target calls it",
                spec=dict(kind="tiles", name=name, reqs=reqs, ratio=r, layout=form),
            )
            for form, ratios, steps in (
                ("mqa_view", (1, 2), _TILE_STEPS),
                ("candidate_masked", (1,), tuple(s for s in _TILE_STEPS if s[0] in _TILE_MASKED)),
            )
            for name, reqs in steps
            for r in ratios
        )
        + tuple(
            Cell(
                why=f"ties {kind} {rows} W={w} r={r} {layout}: the K-th value straddles the cut",
                spec=dict(kind="ties", rows=rows, width=w, ratio=r, layout=layout, scores=kind),
            )
            for kind in ("bf16", "levels", "zero_plateau", "tie_at_kth")
            for rows, w, r, layout in _TIE_FORMS
        )
        + (
            Cell(
                why="signed zero: -0.0 ranks below +0.0, not a tie",
                spec=dict(kind="signed_zero"),
            ),
            Cell(
                why="fewer than 512 finite scores in a selecting row, the rest -inf",
                spec=dict(kind="few_finite"),
            ),
        )
    )

    note = """
    Call sites. Context rows: `row_starts = ks`, `lengths = ke - ks` (the metadata's
    `compressed_visible_lens_cuda[r]`, `(p + 1) // r`), `offsets` a `[:rows]` view of a zero
    buffer, `scores` the prefill-logits view `[rows, S]` (row stride `align(S + 256, 256)`), or at
    re-index layers when `S > 16384` the contiguous `torch.where(candidate_mask, logits, -inf)`
    copy, whole or per row tile (a tile of at most 2^28 padded logits, or one row) with `[a:b]`
    views. Generation rows: `row_starts = None`, one length per row (`next_n` 1 and 6 alike), the
    paged-logits view `[G * next_n, W]` (row stride `ceil256(W)`) or the masked copy. Cells cover
    the 40 static generation widths (8192-token envelope and L = 32768 .. 1048576, DSpark on and
    off, ratio 1 and 2) in both layouts at every row count 1..1536, and context steps of up to
    16384 rows and 256 requests with rows seeing up to 1,048,590 positions, ties straddling the cut
    in every tie cell (up to 357,542 columns tied with a row's K-th score).

    Deterministic by construction: the result is a function of the in-range scores, `lengths`,
    `row_starts`, `offsets` and `k` alone -- not of the call, the process, the GPU, the row count,
    the row stride or a CUDA-graph replay. Certified bitwise across 5 calls and 3 fresh processes on
    each of 2 GPUs, captured replays included. The knobs are pinned for this: a tie-break other than
    NONE sends every shape to the filtered kernel, `deterministic=True` adds the ascending finalize,
    and `dsa_graph_safe=True` pins scalar loads (with it off the kernel picks a vector width from
    the row stride alone and faulted with a misaligned address on a 4-byte-aligned view). One call
    launches `FilteredTopKUnifiedKernel` and `FinalizeTopKIndicesKernel`, once each, at every shape.

    Metadata consumed: none; the target passes metadata views as arguments.

    Preconditions: `scores` fp32 (fp64 raises; bf16 is accepted but not certified), contiguous or a
    row-padded view (unit column stride, row stride > W) whose last row's padding lies inside the
    storage (otherwise the view raises); a transposed view raises. `row_starts[i] + lengths[i] <= W`
    is not checked: a longer range reads the next row's columns, or the padding. `k <= 2048`
    (4096 raises). A short `row_starts` and a strided `lengths` raise. Columns outside each row's
    range are never read (NaN there is certified). The first call in a process loads FlashInfer's
    JIT-compiled top-k module; a first call inside a CUDA-graph capture is certified.

    Measured on sm_103: deterministic costs time. Generation rows take 0.99-2.55x the CuTe DSL decode
    top-k's time, context rows 1.68-4.33x the C++ prefill top-k's; neither alternative fixes the
    order or the tie choice.

    Not certified: `k` other than 512; scores other than fp32; more than 16384 context rows or 1536
    generation rows in one call; more than 256 requests or a request longer than 1,048,590 tokens;
    an untiled context call above 16384 x 1,049,088 fp32 logits (random steps hold at most 2^32);
    generation widths other than the 40; CUDA-graph capture at row counts other than G and 6G for
    G in 1..32, 40..128 step 8, 192, 256, or of context rows; NaN inside a row's range; `S = 0` or
    zero rows.
    """

    def raw_call(
        self,
        scores: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        row_starts: Optional[torch.Tensor],
        k: int,
    ) -> torch.Tensor:
        x = scores
        if (
            not scores.is_contiguous()
            and scores.stride(1) == 1
            and scores.stride(0) > scores.size(1)
        ):
            # A row-padded view (the logits kernels' outputs). The op takes only contiguous input,
            # so hand it the padded rows as they are in memory (a view: no kernel, no copy);
            # row_starts and lengths keep every read inside the view's own columns.
            x = scores.as_strided((scores.size(0), scores.stride(0)), (scores.stride(0), 1))
        return top_k_ragged_transform(
            x,
            offsets,
            lengths,
            k,
            deterministic=True,
            tie_break=TopKTieBreak.SMALL,
            dsa_graph_safe=True,
            row_starts=row_starts,
        )

    def is_valid(
        self,
        scores: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        row_starts: Optional[torch.Tensor],
        k: int,
    ) -> None:
        # The op reads these three through an int32 pointer and never compares the lengths of
        # offsets / lengths with the row count: an int64 tensor or a short one gives wrong rows
        # without an error. Measured on 8 rows: int64 lengths, offsets or row_starts, and a
        # 4-entry view of lengths or of offsets (the op read the next 4 entries), each gave 4 of 8
        # rows wrong with no error.
        assert offsets.dtype == torch.int32 and lengths.dtype == torch.int32, (
            "offsets and lengths must be int32"
        )
        assert row_starts is None or row_starts.dtype == torch.int32, "row_starts must be int32"
        assert offsets.numel() == scores.size(0) and lengths.numel() == scores.size(0), (
            "offsets and lengths must hold one entry per row of scores"
        )

    def reference(
        self,
        scores: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        row_starts: Optional[torch.Tensor],
        k: int,
    ) -> torch.Tensor:
        """Each row's top-k from torch, in the op's order, as int32 ``[N, k]``.

        Rows with ``L <= k`` are ``0 .. L - 1`` then -1. A selecting row gathers its own range
        ``[s, s + L)`` in fp64 with -0.0 moved to -1e-300 (so it ranks below +0.0 and above every
        negative fp32 value, the bit-pattern order the op ranks by), takes the first ``k`` of a
        stable descending sort (equal scores keep column order: ties to the smaller column) and
        sorts them ascending; ``offsets`` is then added to every valid entry. Selecting rows are
        ranked a chunk at a time (at most 2^26 gathered scores), so the reference runs at the
        widths the cells reach; columns outside a row's range are never read.
        """
        n, w = scores.shape
        dev = scores.device
        lens = lengths.long().clamp(min=0)
        pos = torch.arange(k, device=dev)
        idx = torch.where(pos[None, :] < lens[:, None], pos[None, :], -1)
        starts = torch.zeros_like(lens) if row_starts is None else row_starts.long()
        rows = (lens > k).nonzero().flatten().tolist()
        lens_host = lens.tolist()
        a = 0
        while a < len(rows):
            b, widest = a, 1
            while b < len(rows):
                cand = max(widest, lens_host[rows[b]])
                if b > a and (b + 1 - a) * cand > 1 << 26:
                    break
                widest, b = cand, b + 1
            rr = torch.tensor(rows[a:b], device=dev)
            col = torch.arange(widest, device=dev)[None, :]
            cols = starts[rr][:, None] + col
            live = (col < lens[rr][:, None]) & (cols >= 0) & (cols < w)
            x = scores[rr[:, None], cols.clamp(0, w - 1)].double()
            x = torch.where((x == 0) & torch.signbit(x), torch.full_like(x, -1e-300), x)
            x = x.masked_fill(~live, float("-inf"))
            order = torch.sort(x, dim=1, descending=True, stable=True).indices[:, :k]
            idx[rr] = order.sort(dim=1).values
            del x, order
            a = b
        return torch.where(idx >= 0, idx + offsets.long()[:, None], -1).to(torch.int32)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Exact, element for element: the selected set, its ascending order and the -1 padding.

        The op is deterministic by construction and the reference ranks by the same order with the
        same tie rule, so there is nothing to round: any differing entry is a wrong selection.
        Controls measured on sm_103: lengths one past each row's own read one out-of-range column
        and change 1534 of 1536 rows; ties to the larger column change exactly the rows whose K-th
        value straddles the cut; the non-deterministic mode changes hundreds to thousands of rows
        from call to call; one request's `ks` shifted by one column is seen in every step.
        """
        assert out.dtype == ref.dtype == torch.int32, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        if not torch.equal(out, ref):
            bad = (out != ref).any(dim=1).nonzero().flatten()
            raise AssertionError(
                f"{bad.numel()} of {out.size(0)} rows differ from the reference, first {int(bad[0])}"
            )


flashinfer_top_k_ragged_deterministic = FlashinferTopKRaggedDeterministic()
