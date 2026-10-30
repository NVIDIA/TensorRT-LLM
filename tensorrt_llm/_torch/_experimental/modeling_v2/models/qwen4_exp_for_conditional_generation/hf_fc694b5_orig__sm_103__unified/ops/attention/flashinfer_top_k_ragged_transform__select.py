# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA block selection in one FlashInfer call: each window's canonical top-k, ascending.

Wraps FlashInfer's public `flashinfer.top_k_ragged_transform` (`flashinfer/topk.py`; the
FilteredTopK ragged-transform kernel and its tie-break finalize pass in
`flashinfer/topk.cuh`). With `tie_break=TopKTieBreak.LARGE` the kernel keeps the
higher-index columns of the tie class at the k-th value, which makes the selected set the
canonical one; native torch glue over the `[rows, k]` result then orders it ascending with
`-1` last. One call replaces `qsa_topk_select` followed by `qsa_canonical_selection` at the
target's `_select`.
"""

from __future__ import annotations

from typing import Optional

import flashinfer
import torch
from flashinfer.topk import TopKTieBreak

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's block budget: `indexer_budget` 2048 tokens over
#: `indexer_compress_ratio` 4.
_BLOCK_TOPK = 512
#: Score-row widths: the block table's pages x 32 tokens per page, compressed 4:1.
#: The tp1 target's 260-page table; core's `max_seq_len` 9216 (288 pages); MTP's
#: `max_seq_len` 16384 (512 pages); the tune's `max_seq_len` 262144 (8192 pages).
_TP1_COLUMNS = 260 * 32 // 4
_CORE_COLUMNS = 9216 // 4
_MTP_COLUMNS = 16384 // 4
_ENVELOPE_COLUMNS = 262144 // 4
#: XOR with the sign bit maps int32 order onto unsigned order, which puts the `-1`
#: padding (all ones) after every selected index.
_SIGN_BIT = torch.iinfo(torch.int32).min
#: Rows of the reference's int64 key matrix held at once.
_REFERENCE_TILE_ELEMENTS = 1 << 25


def _cell(why: str, rows: int, cols: int, windows: tuple, scores: str) -> Cell:
    return Cell(
        why=why,
        spec=dict(
            rows=rows,
            cols=cols,
            windows=windows,
            scores=scores,
            # Every column outside a row's window is NaN, +inf or 3e38.
            tail="poison",
            dtype=torch.float32,
            k=_BLOCK_TOPK,
            offsets="zero",
            deterministic=False,
            tie_break=int(TopKTieBreak.LARGE),
            dsa_graph_safe=False,
        ),
    )


class FlashinferTopKRaggedTransform(OpWrapper):
    """The canonical top-k of each row's window, ascending, `-1`-padded.

    Row `r` selects from the window `[s, s + lengths[r])` of `input[r]`, `s = row_starts[r]`
    (0 when `row_starts` is None). With `tie_break=TopKTieBreak.LARGE` the selection is the
    `k` largest `(score, column)` pairs: every window column scoring above the k-th value
    `v`, then the highest-index columns scoring exactly `v` until there are `k`; or the
    whole window when `lengths[r] <= k`. Returns a new int32 `[rows, k]` tensor holding
    each selected window-local offset plus `offsets[r]`, in ascending order, `-1` in every
    slot past `min(k, lengths[r])` -- `qsa_canonical_selection`'s row when `offsets` is 0.

    Fusion boundary: selection and ordering. The scores come in computed; expanding the
    selected blocks to token indices is the caller's next call.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target calls this at `_select` over the `[rows, cols]` score matrix whose
    # columns past each window are allocator garbage, NaN or -inf (every cell poisons
    # them): the decode sparse fallback past 4096 columns, the MTP draft loop and the
    # eager prefill fallback. `windows` names how each row's window is placed: `edges`
    # puts the first rows at 0, 1, k - 1, k, k + 1 blocks and the full table, the rest at
    # random positions; `random` draws positions between the two bounds; `verify` is
    # `requests` MTP verify blocks of `width` consecutive positions; `chunk` is one
    # request's consecutive prefill positions from the one given; `ragged` draws non-zero
    # row starts. `scores` `relu` is half zeros; `relu_sparse` one positive in ten;
    # `relu_rare` one in 200, so the zero tie class straddles the budget in every window
    # wider than `k` up to the envelope's 65536 columns.
    CELLS: tuple[Cell, ...] = (
        _cell(
            "decode: one row over the tp1 target's 260-page table",
            1,
            _TP1_COLUMNS,
            ("random", 2051, 4 * _TP1_COLUMNS - 1),
            "relu",
        ),
        _cell(
            "decode batch 128 at 2080 columns, window edges first",
            128,
            _TP1_COLUMNS,
            ("edges",),
            "relu",
        ),
        _cell(
            "decode batch 128 at core's max_seq_len 9216 (2304 columns)",
            128,
            _CORE_COLUMNS,
            ("edges",),
            "relu",
        ),
        _cell(
            "decode batch 128 at MTP's max_seq_len 16384 (4096 columns)",
            128,
            _MTP_COLUMNS,
            ("edges",),
            "relu",
        ),
        _cell(
            "MTP7 verify: 128 requests x 8 = 1024 rows at 4096 columns",
            1024,
            _MTP_COLUMNS,
            ("verify", 128, 8, 0, 4 * _MTP_COLUMNS - 8),
            "relu_sparse",
        ),
        _cell(
            "decode: one row over the tune's 262144-token table (65536 columns)",
            1,
            _ENVELOPE_COLUMNS,
            ("random", 2051, 4 * _ENVELOPE_COLUMNS - 1),
            "relu",
        ),
        _cell(
            "decode batch 128 at 65536 columns, zero ties straddle the budget",
            128,
            _ENVELOPE_COLUMNS,
            ("random", 2051, 4 * _ENVELOPE_COLUMNS - 1),
            "relu_rare",
        ),
        _cell(
            "prefill: an 8192-row chunk from position 0 at 2304 columns",
            8192,
            _CORE_COLUMNS,
            ("chunk", 0),
            "relu",
        ),
        _cell(
            "prefill: an 8192-row chunk ending a 262144-token prompt, ties straddle",
            8192,
            _ENVELOPE_COLUMNS,
            ("chunk", 4 * _ENVELOPE_COLUMNS - 8192),
            "relu_rare",
        ),
        _cell(
            "non-zero row starts: 128 ragged windows in a 4096-column table",
            128,
            _MTP_COLUMNS,
            ("ragged",),
            "relu",
        ),
    )

    note = """
    Source: FlashInfer's public Python API (tier 3), JIT-compiled -- `gen_topk_module`
    builds `topk.cu` and its bindings on the first call into FlashInfer's per-architecture
    cache; this install carries no AOT artifact for it. The cells ran that JIT build for
    sm_103a (`compute_103a`) on GB300. The first call must come before CUDA-graph capture:
    the engine's warmup makes it.

    Path: any tie-break other than NONE forces FilteredTopK -- one 1024-thread CTA per row
    with 16K-index shared-memory candidate buffers -- followed by a finalize launch that
    applies the tie-break. `k` above 2048 (`FILTERED_TOPK_MAX_K`) returns
    cudaErrorNotSupported, raised by the binding. The multi-CTA radix path and its shared
    `radix_topk_row_states` buffer never run here. The relu_rare cells hold a zero tie
    class of tens of thousands of columns at the k-th value, past the candidate buffers.

    Glue: one XOR with the sign bit, a `torch.sort` over `[rows, k]`, one XOR back. No host
    read and no data-dependent shape: the test replays one CUDA graph per decode batch (1
    and 128 rows at 65536 columns, one shared pool) after refilling every input. The output
    is allocated per call (from the graph's pool under capture).

    Measured on GB300 (sm_103), eager, per call, every cell index-equal to `reference`:
      rows x cols      entry     the FlashInfer call alone   op peak
      1 x 2080         0.08 ms   0.02 ms                     <1 MiB
      128 x 4096       0.09 ms   0.02 ms                     1 MiB
      1024 x 4096      0.10 ms   0.07 ms                     8 MiB
      128 x 65536      0.13 ms   0.11 ms                     1 MiB
      8192 x 2304      0.45 ms   0.26 ms                     64 MiB
      8192 x 65536     6.5 ms    6.4 ms                      64 MiB
    The sort glue's launches dominate the decode calls; inside a captured step they are
    replayed rather than launched. The pair it replaces (`qsa_topk_select` then
    `qsa_canonical_selection`) was measured at 0.40 ms (128 x 65536), 3.24 ms
    (1024 x 65536) and 26.4 ms (8192 x 65536) for the rewrite alone.

    Silent, refused by `is_valid` (each measured in the test):
      - `tie_break` other than LARGE: NONE keeps an arbitrary part of the tie class at the
        k-th value, SMALL the lowest-index part; neither is the canonical row.
      - `offsets`, `lengths` or `row_starts` not int32: the binding reads every one of them
        as int32 data whatever its dtype.
      - `offsets` or `lengths` with other than one element per row: the binding checks only
        that they are 1-D and reads one element per row past a shorter tensor's end.

    Preconditions (not checked; values, not metadata):
      - `0 <= row_starts[r]` and `row_starts[r] + lengths[r] <= input.shape[1]`: the kernel
        reads the window wherever it says, past the row if it says so.
      - No NaN inside a window: the kernel ranks NaN above +inf.
      - Zero sign: the kernel ranks -0.0 below +0.0, while the canonical row (`reference`)
        holds them as one tie class. A row whose tie class at the k-th value mixes the two
        gets the kernel's order (the test pins it); this target's scores are relu sums
        times a positive scale, so its zeros are +0.0.

    Uncertified: `deterministic` True, `dsa_graph_safe` True, non-zero `offsets`, fp16 and
    bf16 `input`, `k` other than 512, table widths between the cells' (e.g. MTP's 4104 with
    draft pages), and row counts between them.

    Index limits: `input` is addressed with a 64-bit row offset; the output's `row * k`
    is 32-bit unsigned, so rows x k < 2^32 (8388608 rows at k 512; the envelope's 8192 is
    1024x below). Indices are int32 and bounded by the table width, far below 2^31.
    """

    def __call__(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        k: int,
        deterministic: bool = False,
        tie_break: int = TopKTieBreak.NONE,
        dsa_graph_safe: bool = False,
        row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        selected = flashinfer.top_k_ragged_transform(
            input,
            offsets,
            lengths,
            k,
            deterministic=deterministic,
            tie_break=tie_break,
            dsa_graph_safe=dsa_graph_safe,
            row_starts=row_starts,
        )
        ordered = torch.sort(selected.bitwise_xor_(_SIGN_BIT), dim=1).values
        return ordered.bitwise_xor_(_SIGN_BIT)

    def reference(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        k: int,
        deterministic: bool = False,
        tie_break: int = TopKTieBreak.NONE,
        dsa_graph_safe: bool = False,
        row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """`qsa_canonical_selection`'s reference, framed by `row_starts` and `lengths`
        and lifted by `offsets`: a top-k over one int64 key per column whose high word
        orders the scores as fp32 does (-0.0 folded into +0.0) and whose low word is the
        column. States the LARGE tie-break only; `deterministic` and `dsa_graph_safe`
        pick a kernel, not a result, and are ignored."""
        assert int(tie_break) == TopKTieBreak.LARGE, "the reference states the LARGE tie-break"
        num_rows, num_cols = input.shape
        device = input.device
        starts_all = (
            torch.zeros(num_rows, dtype=torch.int64, device=device)
            if row_starts is None
            else row_starts.long()
        )
        ends_all = starts_all + lengths.long()
        columns = torch.arange(num_cols, device=device)[None, :]
        width = min(k, num_cols)
        lowest, pad = torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max
        out = torch.full((num_rows, k), -1, dtype=torch.int32, device=device)
        tile = max(1, _REFERENCE_TILE_ELEMENTS // max(num_cols, 1))
        for lo in range(0, num_rows, tile):
            hi = min(lo + tile, num_rows)
            starts, ends = starts_all[lo:hi, None], ends_all[lo:hi, None]
            in_window = (columns >= starts) & (columns < ends)
            # `+ 0.0` folds -0.0 into +0.0, so the two are one tie class.
            bits = (input[lo:hi].float() + 0.0).view(torch.int32).long()
            order = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
            key = torch.where(in_window, (order << 32) | columns, lowest)
            top = torch.topk(key, width, dim=1).indices
            taken = torch.gather(in_window, 1, top)
            ordered = torch.sort(torch.where(taken, top, pad), dim=1).values
            local = ordered - starts + offsets[lo:hi, None].long()
            out[lo:hi, :width] = torch.where(ordered == pad, -1, local).to(torch.int32)
        return out

    def is_valid(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        k: int,
        deterministic: bool = False,
        tie_break: int = TopKTieBreak.NONE,
        dsa_graph_safe: bool = False,
        row_starts: Optional[torch.Tensor] = None,
    ) -> None:
        assert int(tie_break) == TopKTieBreak.LARGE, (
            f"tie_break must be TopKTieBreak.LARGE for the canonical row; got {tie_break!r}, "
            "which keeps another part of the tie class at the k-th value"
        )
        for name, tensor in (
            ("offsets", offsets),
            ("lengths", lengths),
            ("row_starts", row_starts),
        ):
            if tensor is not None:
                assert tensor.dtype == torch.int32, (
                    f"{name} must be int32; the binding reads it as int32 data whatever its "
                    f"dtype, so a {tensor.dtype} {name} is misread without an error"
                )
        num_rows = input.shape[0]
        for name, tensor in (("offsets", offsets), ("lengths", lengths)):
            assert tensor.numel() == num_rows, (
                f"{name} needs one element per input row ({num_rows}); got "
                f"{tensor.numel()}, and the binding reads past a shorter one's end"
            )


flashinfer_top_k_ragged_transform = FlashinferTopKRaggedTransform()
