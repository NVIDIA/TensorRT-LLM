# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Region top-k: each row's `k` best visible columns as int32 ids, ties to the lower id."""

from __future__ import annotations

from typing import Optional

import torch
from flashinfer.topk import TopKTieBreak, top_k_ragged_transform

from .._op import Arch, Cell, OpWrapper

#: Step-5's sparse-attention budget: 512 regions of 8 tokens per row and group.
_REGION_TOPK = 512
#: Regions at `max_seq_len` 9216: the decode site's fixed score width.
_DECODE_COLUMNS = 1152


def _spec(**overrides) -> dict:
    """A cell spec: the call the Step-5 sparse layers make unless the cell says otherwise."""
    return {
        "k": _REGION_TOPK,
        "region_size": 8,
        "deterministic": False,
        "dsa_graph_safe": False,
        "scores": "relu_heads",
        "tail": "nan",
        **overrides,
    }


class FlashinferTopKRaggedTransform(OpWrapper):
    """Top-`k` columns of each row's visible prefix, as int32 ids plus a per-row offset.

    FlashInfer API: ``flashinfer.topk.top_k_ragged_transform`` (tier 3; no trtllm
    op wraps it; it takes no `backend=`). For row `i` it selects the `k` largest
    of `input[i, :lengths[i]]` and returns `local_id + offsets[i]` for each, in
    a fresh `[rows, k]` int32 tensor: `min(k, lengths[i])` ids as a prefix and
    `-1` in every remaining slot. Columns at or past `lengths[i]` are never
    read.

    `tie_break=TopKTieBreak.SMALL` makes the selected *set* exact: values tied
    at the cut are taken in ascending column order (the Step-5 contract:
    `torch.topk` over `(-score, id)`). `deterministic=True` additionally sorts
    each row's ids ascending; with `deterministic=False` the order inside the
    prefix is unspecified and changes between identical launches.

    Step-5 calls it per sparse layer on the indexer scores `[G*T, C]` fp32
    (group-major rows) with `offsets` all zero, `lengths = position // 8` (the
    row's complete history regions), `k = 512`, `tie_break=SMALL` and
    `deterministic=True`; the output is each row's logical region ids.

    Fusion boundary: selection and the offset add. Scoring, the causal bound
    the caller encodes in `lengths`, and appending the current region are
    outside.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: groups, positions (("ragged", b) decode rows: the edge positions
    # then seeded draws over 0..9215; ("chunk", first, n): n consecutive
    # positions), columns (score width), k, region_size, deterministic,
    # dsa_graph_safe, scores ("relu_heads": sum over 4 heads of relu(x) * w with
    # signed w; "zero_cut": positive w and mostly-negative x, so the cut at k
    # falls inside the exact-zero group; "producer": the csa_index_scores entry's
    # own output), tail (what fills columns past each row's length: "nan" or
    # "-inf", the producer's fill).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode dp: 512 rows (4 groups x 128) x 1152, -inf tail",
            spec=_spec(groups=4, positions=("ragged", 128), columns=_DECODE_COLUMNS, tail="-inf"),
        ),
        Cell(
            why="decode tp4: 128 rows (1 group x 128) x 1152, NaN tail",
            spec=_spec(groups=1, positions=("ragged", 128), columns=_DECODE_COLUMNS),
        ),
        Cell(
            why="decode dp batch 1: 4 rows at position 9215, 1151 visible",
            spec=_spec(groups=4, positions=("chunk", 9215, 1), columns=_DECODE_COLUMNS),
        ),
        Cell(
            why="decode dp, dsa_graph_safe=True (scalar loads, graph-safe path)",
            spec=_spec(
                groups=4, positions=("ragged", 128), columns=_DECODE_COLUMNS, dsa_graph_safe=True
            ),
        ),
        Cell(
            why="decode dp, deterministic=True: ascending ids, compared exactly",
            spec=_spec(
                groups=4, positions=("ragged", 128), columns=_DECODE_COLUMNS, deterministic=True
            ),
        ),
        Cell(
            why="decode dp, cut inside the exact-zero group: ties to lower ids",
            spec=_spec(
                groups=4, positions=("ragged", 128), columns=_DECODE_COLUMNS, scores="zero_cut"
            ),
        ),
        Cell(
            why="prefill dp: 32768 rows (4 x 8192 at 1024..9215) x 1152",
            spec=_spec(groups=4, positions=("chunk", 1024, 8192), columns=_DECODE_COLUMNS),
        ),
        Cell(
            why="prefill dp deterministic: 32768 rows x 1152, exact order",
            spec=_spec(
                groups=4,
                positions=("chunk", 1024, 8192),
                columns=_DECODE_COLUMNS,
                deterministic=True,
            ),
        ),
        Cell(
            why="prefill tp4: 8192 rows at 0..8191 x 1023 (odd width)",
            spec=_spec(groups=1, positions=("chunk", 0, 8192), columns=1023),
        ),
        Cell(
            why="chunked dp: 2048 rows (4 x 512 at 8192..8703) x 1087",
            spec=_spec(groups=4, positions=("chunk", 8192, 512), columns=1087),
        ),
        Cell(
            why="first chunk dp: 2048 rows at 0..511 x 63, narrower than k",
            spec=_spec(groups=4, positions=("chunk", 0, 512), columns=63),
        ),
        Cell(
            why="chunked dp on csa_index_scores output: 2048 rows x 1087",
            spec=_spec(
                groups=4, positions=("chunk", 8192, 512), columns=1087, scores="producer", tail=None
            ),
        ),
        # The call Step-5's modeling makes: deterministic=True at every sparse
        # layer, 1 group per rank (tp4 / tep4) and 4 (dep4).
        Cell(
            why="det 1 group decode batch 1: 1 row at 9215, 1151 visible",
            spec=_spec(
                groups=1, positions=("chunk", 9215, 1), columns=_DECODE_COLUMNS, deterministic=True
            ),
        ),
        Cell(
            why="det 1 group decode batch 8: the 8 edge positions x 1152",
            spec=_spec(
                groups=1, positions=("ragged", 8), columns=_DECODE_COLUMNS, deterministic=True
            ),
        ),
        Cell(
            why="det 1 group decode batch 24: 24 rows x 1152",
            spec=_spec(
                groups=1, positions=("ragged", 24), columns=_DECODE_COLUMNS, deterministic=True
            ),
        ),
        Cell(
            why="det 1 group decode batch 128: 128 rows x 1152, NaN tail",
            spec=_spec(
                groups=1, positions=("ragged", 128), columns=_DECODE_COLUMNS, deterministic=True
            ),
        ),
        Cell(
            why="det 1 group decode 128 x 1152, cut inside the exact-zero group",
            spec=_spec(
                groups=1,
                positions=("ragged", 128),
                columns=_DECODE_COLUMNS,
                deterministic=True,
                scores="zero_cut",
            ),
        ),
        Cell(
            why="det 1 group prefill: 8192 rows at 0..8191 x 1023 (odd width)",
            spec=_spec(groups=1, positions=("chunk", 0, 8192), columns=1023, deterministic=True),
        ),
        Cell(
            why="det 1 group chunk: 512 rows at 8192..8703 x 1087",
            spec=_spec(groups=1, positions=("chunk", 8192, 512), columns=1087, deterministic=True),
        ),
        Cell(
            why="det 1 group first chunk: 512 rows at 0..511 x 63, narrower than k",
            spec=_spec(groups=1, positions=("chunk", 0, 512), columns=63, deterministic=True),
        ),
        Cell(
            why="det 1 group reused-prefix chunk: 4966 rows at 1135..6100 x 762",
            spec=_spec(groups=1, positions=("chunk", 1135, 4966), columns=762, deterministic=True),
        ),
        Cell(
            why="det 1 group chunk on csa_index_scores output: 512 rows x 1087",
            spec=_spec(
                groups=1,
                positions=("chunk", 8192, 512),
                columns=1087,
                deterministic=True,
                scores="producer",
                tail=None,
            ),
        ),
        Cell(
            why="det 4 groups decode batch 1: 4 rows at 9215, 1151 visible",
            spec=_spec(
                groups=4, positions=("chunk", 9215, 1), columns=_DECODE_COLUMNS, deterministic=True
            ),
        ),
        Cell(
            why="det 4 groups chunk: 2048 rows (4 x 512 at 8192..8703) x 1087",
            spec=_spec(groups=4, positions=("chunk", 8192, 512), columns=1087, deterministic=True),
        ),
        Cell(
            why="det 4 groups first chunk: 2048 rows at 0..511 x 63, narrower than k",
            spec=_spec(groups=4, positions=("chunk", 0, 512), columns=63, deterministic=True),
        ),
        Cell(
            why="det 4 groups decode 512 x 1152, cut inside the exact-zero group",
            spec=_spec(
                groups=4,
                positions=("ragged", 128),
                columns=_DECODE_COLUMNS,
                deterministic=True,
                scores="zero_cut",
            ),
        ),
    )

    note = """
    Metadata consumed: none. Stateless; every input is an argument. The call
    allocates its `[rows, k]` int32 output (`torch.empty`) and, on first use per
    device, FlashInfer's cached 1 MiB `radix_topk_row_states_<device>` buffer
    (zero-filled once; read only by the multi-CTA radix path, which a
    tie-broken call never takes).

    Kernel path (FlashInfer's `TopKRaggedTransformDispatch`): any `tie_break`
    other than NONE forces the FilteredTopK kernel -- one 1024-thread CTA per
    row, 128 KiB of dynamic shared memory -- followed by
    `FinalizeTopKIndicesKernel`, which adds `offsets` (and, under
    `deterministic=True`, sorts each row's ids ascending first). Two kernel
    launches per call, from one public callable (profiled at 512 x 1152 for
    every `deterministic` x `dsa_graph_safe` pair: `FilteredTopKUnifiedKernel`
    then `FinalizeTopKIndicesKernel`, nothing else). Input loads are
    vectorized by `gcd(columns, 4)` fp32 lanes (4 at 1152, 1 at 1023 / 1087 /
    63); `dsa_graph_safe=True` forces 1 (the profiled kernel's VEC_SIZE
    template argument: 4, and 1 with it). The cluster top-k path is never
    taken with a tie-break.

    Preconditions (the op checks none of these):
      - `0 <= lengths[i] <= input.shape[1]`. A length past the width reads the
        next row's scores and returns ids >= the width (measured: lengths 1200
        on a 1152-wide input returned ids up to 1199). A negative length
        returns an all `-1` row.
      - Scores inside the window are not NaN. Columns past the window are never
        read, so the producer's `-inf` fill (or anything) is fine there; the
        cells fill them with NaN to prove it.
      - No `-0.0` among the scores the cut can reach. The kernel orders fp32 by
        bit pattern, so `-0.0` ranks strictly below `+0.0`; Step-5's selector
        treats them as equal. Measured: 300 ones, then 400 `-0.0`, then
        `+0.0`, `k=512`: the op took ids 700..911 (`+0.0`) where the contract
        takes 300..511 (`test_negative_zero_ranks_below_positive_zero` pins
        it). The csa_index_scores entry accumulates onto `+0.0`, so its exact
        zeros are all `+0.0`; the `producer` cell checks that and selects on
        its output.
      - `tie_break=TopKTieBreak.SMALL` is the Step-5 contract and must be
        passed; the API's default, NONE, takes an arbitrary subset of a tied
        group at the cut (measured on the `zero_cut` scores:
        `test_tie_break_is_what_makes_the_set_exact`).

    Loud rejections (measured): non-contiguous `input` ("input must be
    contiguous"); `rows == 0` ("TopKRaggedTransform failed with error code
    invalid argument" -- a zero-row grid, so the caller must skip the call);
    `k > 2048` with a tie-break ("operation not supported"). Width 0 and widths
    below `k` are legal: every row takes its trivial path (ids `0..length-1`).

    Silent wrong answers `is_valid` refuses (measured): `lengths` or `offsets`
    not int32 (the binding casts the pointer without checking the dtype; int64
    lengths returned a wrong selection), and `lengths` / `offsets` with fewer
    than `rows` entries (read past their end: a 511-entry offsets view of a
    512-entry buffer returned the buffer's next value as the last row's
    offset).

    JIT: the module is compiled on the first call into FlashInfer's workspace
    (175 s cold on GB300; 0.28 s from the cache). A target's first call has to
    be an eager warm-up, never inside a capture.

    CUDA graphs: capturable at both `dsa_graph_safe` settings and both
    `deterministic` settings. `test_cuda_graph_capture_and_replay` captures 512
    decode rows at 1152 and replays against new scores and new `lengths`
    written in place; the output tensor the capture allocated is rewritten on
    every replay. `dsa_graph_safe` changes only the load width at these
    shapes; neither setting is needed for capture.
    `test_cuda_graph_deterministic_decode` captures Step-5's own decode call
    (`deterministic=True`, `dsa_graph_safe=False`) at 1 and 4 groups x batch 8
    and 128 at 1152, and requires every replay to equal an eager call (ids and
    order) and the reference, over in-place rewrites that alternate the plain
    and the zero-cut scores.

    Speed (GB300, eager, warm, 50 calls; `deterministic=False` /
    `dsa_graph_safe=True` / `deterministic=True`): 128 x 1152 13.9 / 14.0 /
    13.4 us; 512 x 1152 19.3 / 17.4 / 21.6 us; 8192 x 1023 155 / 156 / 169 us;
    32768 x 1152 815 / 717 / 870 us.

    Hard limits: the FilteredTopK kernel addresses its output row as
    `bid * top_k` in 32-bit unsigned arithmetic, so `rows * k` wraps at 2^32
    (8388608 rows at k = 512); inputs are addressed in 64-bit. The envelope's
    largest call is 32768 x 512 (8192 prefill rows x 4 groups). Lengths and
    ids are int32 (up to 2^31 columns); no index passes through fp32. Not
    probed.

    Not certified: fp16 / bf16 scores, `tie_break` other than SMALL, non-zero
    `offsets`, `row_starts`, `k` other than 512, widths other than 1152 / 1087
    / 1023 / 762 / 63, decode batches other than 1, 8, 24 and 128 (captured: 8
    and 128 only), context chunks at other widths or first positions, sm_100.
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
        return top_k_ragged_transform(
            input,
            offsets,
            lengths,
            k,
            deterministic=deterministic,
            tie_break=tie_break,
            dsa_graph_safe=dsa_graph_safe,
            row_starts=row_starts,
        )

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
        """Each row's selected ids, ascending, `-1`-padded, plus `offsets`.

        Ranks the visible prefix by `(-score, id)` with a stable descending
        sort -- numeric order, so `-0.0 == +0.0` -- and keeps the first
        `min(k, length)`, so a tied group at the cut is taken in ascending id
        order: `tie_break` SMALL, the only mode a cell passes (NONE leaves the
        set unspecified, LARGE is not stated). The ids come back ascending, which is
        exactly what `deterministic=True` returns; under `deterministic=False`
        the test compares sets. `deterministic` and `dsa_graph_safe` change no
        selection and are ignored; `row_starts` is narrowed to None (no cell
        passes it).
        """
        if TopKTieBreak(tie_break) != TopKTieBreak.SMALL:
            raise NotImplementedError(f"the reference states tie_break SMALL only, got {tie_break}")
        if row_starts is not None:
            raise NotImplementedError("no cell passes row_starts")
        rows, columns = input.shape
        visible = lengths.long().clamp(0, columns)
        column_ids = torch.arange(columns, device=input.device)
        ranked = input.float().masked_fill(column_ids[None, :] >= visible[:, None], float("-inf"))
        order = torch.sort(ranked, dim=1, descending=True, stable=True).indices
        if columns < k:
            order = torch.nn.functional.pad(order, (0, k - columns), value=columns)
        taken = torch.arange(k, device=input.device)[None, :] < visible.clamp(max=k)[:, None]
        chosen = order[:, :k].masked_fill(~taken, columns)
        chosen = torch.sort(chosen, dim=1).values
        ids = chosen + offsets.long()[:, None]
        return ids.masked_fill(chosen == columns, -1).to(torch.int32)

    def is_valid(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        k: int,
        *unused_args,
        **unused_kwargs,
    ) -> None:
        # The binding casts `offsets` and `lengths` to int32 pointers without a
        # dtype check and never compares their length to the row count.
        rows = input.shape[0]
        for name, tensor in (("lengths", lengths), ("offsets", offsets)):
            assert tensor.dtype == torch.int32, f"{name} must be int32, got {tensor.dtype}"
            assert tensor.shape == (rows,), (
                f"{name} must be [{rows}] (one entry per input row), got {tuple(tensor.shape)}"
            )


flashinfer_top_k_ragged_transform = FlashinferTopKRaggedTransform()
