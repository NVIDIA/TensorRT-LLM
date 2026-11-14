# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the bmm_out catalog entry.

The entry owns the contract. This file turns a cell into tensors and holds the
claims no cell carries: the guard, the refusals the op makes itself, the empty
call and the layout of `b`. The Mistral Large 4 cells add six spec keys:
`b_layout` ("nk_t": `b` is the transposed view of the V-B rows of a regrouped
`wkv_b`), `m_range` (every row count of the range, each its own call),
`m_window_edges` (`(lo, hi, width)`: every row count of the first `width`-row
window, then the first and last of each later one), `graph_m` (each row count
captured in a CUDA graph and replayed), `tail_from` (one call on the
trailing rows of full-size buffers) and `out_heads` (`out` is heads 0..batch-1 of a
`[M, out_heads, out_row]` buffer, the rest of which the op must not touch). A cell whose
spec says `large` runs in `test_large_certified_cells`, marked `large`, which pre-merge
deselects. The entry's own cells and deepseek's say `transposed_out` instead: `out` is a
transposed view of a token-major buffer, or a plain one.

Mistral Large 4's cells are built from `_ml4` (TEP4 and TP4, 16 heads per rank), `_ML4_64`
(DEP4, all 64 heads on every rank), `_ML4_16_FQ64` (16 heads per rank into a 64-head
buffer, on the fp8_ds_mla latent KV) and `_ML4_8` (TEP8 and TP8, 8 heads per rank; the
`_ML4_8_FQ64` cells are their K-B BMM into heads 0..7 of the 64-head fused query FlashMLA
reads on the fp8_ds_mla latent KV).
"""

import sys
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import (  # noqa: E501
    bmm_out as op,
)

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "bmm_out requires a CUDA device"

DEV = "cuda"
#: A bf16 NaN payload nothing here computes: what an unwritten element holds.
SENT = 0x7FC1
#: Four sentinels in one 64-bit word, for the sweep's word-wide reset.
SENT_WORD = 0x7FC1_7FC1_7FC1_7FC1


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference.

    Kernel and reference consume bit-identical bf16 operands and round once
    to bf16; they differ only in the order cuBLAS and torch sum over K, so an
    element can land on the other side of a bf16 rounding boundary. One such
    flip is one bf16 spacing, at most 2^-7 of its row's largest magnitude: 2
    of `assert_within_ulp`'s 2^-8 units at the row scale. Torch's default
    band cannot say this -- near-zero outputs read as relative differences of
    10 on absolute differences of 1.5e-4.

    So: 4 units element-wise against the row scale, 2 units of relative RMS.
    Measured at Mistral Large 4's three shapes over 9216 rows: kernel and
    reference each within 0.996 units of an fp64 product, at most 1.84 units
    apart, relative RMS at most 0.013 units. The nearest wrong products measured --
    a head's weights rolled by one, `a` shifted by one token, the rope half of q in
    place of the nope half, the V-B `b` reshaped instead of transposed -- land 641
    to 790 units away. Over 32768 rows: each within 0.996 units of fp64, at most
    1.97 units apart (V-B; K-B 1.87 and 1.93), relative RMS at most 0.0125
    units, and each of those wrong products, at the shapes it applies to, 682
    to 821 units away. At DEP4's 64 heads over 32768 rows: each within 0.9962
    units of fp64, at most 1.98 units apart (V-B; K-B 1.88 and 1.95),
    relative RMS at most 0.0124 units, and the same wrong products 716 to 900
    units away, 362 units of relative RMS. At TEP8's and TP8's 8 heads over
    32768 rows: each within 0.996 units of fp64, at most 1.97 units apart
    (V-B; K-B 1.57 and 1.87), relative RMS at most 0.0122 units, and the same
    wrong products 673 to 787 units away, 362 units of relative RMS. At TEP4's
    and TP4's 16 heads into the 64-head fused query (token stride 36864) over
    32768 rows: each within 0.9961 units of fp64, at most 1.87 units apart (K-B
    NoPE; RoPE 1.86), relative RMS at most 0.0062 units, and the same wrong
    products, with the neighbouring head's q in place of a head's own, 678.6 to
    857.0 units away (678.6 to 794.5 on TP4's draw), 361.7 to 362.1 units of
    relative RMS. At TEP8's 8 heads into the 64-head fused query (token stride
    36864) over 32768 rows: each within 0.9960 units of fp64, at most 1.91
    units apart (K-B NoPE; RoPE 1.86), relative RMS at most 0.0063 units, and
    the same wrong products, with the neighbouring head's q in place of a
    head's own, 672.0 to 804.6 units away, 361.9 to 362.1 units of relative
    RMS. Over the `transposed_out` cells the worst case measured is 1.5e-4
    absolute, roughly 3e-4 units.
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _gate(a: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    """`out` against the entry's reference, 16 heads at a time.

    The slices keep a 64-head, 32768-row gate's fp32 temporaries to a
    quarter: with the whole batch gated at once a K-B sweep peaked 36.8 GiB
    above base (46.8 GiB reserved), sliced 15.1 GiB (24.1 GiB). Each slice
    passing the RMS gate implies the whole batch passes it, so the slices are
    no looser; at 16 heads or fewer it is one `_compare`.
    """
    for h in range(0, a.shape[0], 16):
        _compare(out[h : h + 16], op.reference(a[h : h + 16], w[h : h + 16], out[h : h + 16]))


def _bits(t: torch.Tensor) -> torch.Tensor:
    """bf16 bit patterns, so `!=` is bitwise (the NaN sentinel included)."""
    return t.view(torch.int16)


def _words(t: torch.Tensor) -> torch.Tensor:
    """The same bit patterns four to a 64-bit word, for the sweep's per-call
    reset and compare over whole contiguous rows."""
    return t.view(torch.int64)


def _weights(spec, g):
    bsz, k, n, dt = spec["batch"], spec["k"], spec["n"], spec["dtype"]
    if spec.get("b_layout", "kn") == "kn":
        return torch.randn(bsz, k, n, generator=g, device=DEV).to(dt)
    # A regrouped wkv_b: 64 K-B rows per head ahead of n V-B rows per head,
    # each k wide; V-B reads its rows as [B, n, k], transposed.
    kvb = torch.randn(bsz * 64 + bsz * n, k, generator=g, device=DEV).to(dt)
    return torch.transpose(kvb[bsz * 64 :].reshape(bsz, n, k), 1, 2)


def _build(spec, seed, m=None):
    """Materialize one cell at `m` rows (the spec's own by default).

    With `a_row` / `out_row` set, `a` and `out` are transposed head-major views
    of token-major `[M, B, row]` buffers, as the targets pass them: `a` is the
    first `k` columns of each row, `out` the first `n`. With `out_heads`, `out`'s
    parent is `[M, out_heads, out_row]` and `out` its first B heads, as the tep4,
    tp4 and tep8 deployments' fused query is. Both parents come back too: `a`'s so a
    captured graph can be fed new rows, `out`'s because "did it land there, and
    only there" is the thing worth asserting. `out`'s parent starts as the
    sentinel, so an element the op did not write shows.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    bsz, k, n, dt = spec["batch"], spec["k"], spec["n"], spec["dtype"]
    m = spec["m"] if m is None else m
    w = _weights(spec, g)
    if spec["a_row"] is None:
        a = a_parent = torch.randn(bsz, m, k, generator=g, device=DEV).to(dt)
    else:
        a_parent = torch.randn(m, bsz, spec["a_row"], generator=g, device=DEV).to(dt)
        a = a_parent[..., :k].transpose(0, 1)
    if spec["out_row"] is None:
        out = parent = torch.empty(bsz, m, n, dtype=dt, device=DEV)
    else:
        heads = spec.get("out_heads", bsz)
        parent = torch.empty(m, heads, spec["out_row"], dtype=dt, device=DEV)
        out = parent[:, :bsz, :n].transpose(0, 1)
        # Only a decode-shaped cell escapes this: a dim of extent 1 leaves a
        # view torch may still call contiguous.
        assert m == 1 or not out.is_contiguous()
    _bits(parent).fill_(SENT)
    return a, w, out, a_parent, parent


def _other_heads_untouched(spec, parent) -> None:
    """With `out_heads`, the heads past the call's batch hold the sentinel."""
    if spec.get("out_heads", spec["batch"]) > spec["batch"]:
        assert bool((_bits(parent[:, spec["batch"] :]) == SENT).all()), (
            "the op wrote a head past the call's batch"
        )


def _single(spec, seed) -> None:
    a, w, out, _, parent = _build(spec, seed)
    parent_ptr = parent.data_ptr()
    # Identity, not equality: the op must have written this very buffer and
    # returned it, not produced a copy that happens to match.
    assert op(a, w, out) is out, "the op writes in place and returns what it wrote"
    # The product has to be in the caller's buffer, not in a replacement the op
    # allocated: that is the whole reason this op exists over torch.bmm.
    assert parent.data_ptr() == parent_ptr
    _compare(out, op.reference(a, w, out))
    # A strided `out` must not spill into the parent's other columns.
    if spec["out_row"] is not None:
        assert bool((_bits(parent[..., spec["n"] :]) == SENT).all()), (
            "the op wrote outside the out view"
        )
        _other_heads_untouched(spec, parent)


def _window_edges(lo: int, hi: int, width: int) -> list[int]:
    """Every m of the first `width`-row window from `lo`, then the first and
    last m of each later window, up to `hi`."""
    first = range(lo, min(hi, lo + width - 1) + 1)
    rest = (m for s in range(lo + width, hi + 1, width) for m in (s, min(hi, s + width - 1)))
    return sorted({*first, *rest})


def _sweep(spec, seed, counts) -> None:
    """Every m of `counts`, each its own call, against one run at the largest.

    The run at the largest m is gated against the reference; every shorter
    call must then write bitwise its leading rows -- a row's product reads
    only its own row of `a`, and cuBLAS's choice of kernel, which changes with
    m, was measured not to move a bit. Descending, so the check is complete at
    one comparison per call: before each call the rows it owns are reset to
    the sentinel, so a skipped or partial write shows; after the last call row
    r holds what the smallest call above r wrote and the parent's columns past
    n hold the sentinel, so one comparison of the whole parent catches a write
    past any call's rows or columns.

    Each call's rows are reset and compared whole, as 64-bit words, and the
    compare is reduced with `any` on the device, so the sweep synchronizes
    once. Measured at 32768 rows of the fused query: an element count of bf16
    patterns is far slower per compare than `any` over words, which is what
    fits a 16-head K-B sweep to 1..32768 in a cell's time. A failing m
    is re-run to count its elements.

    With `out_heads`, what a call owns of each row is the batch's heads, so
    only those are reset and compared; the heads past them are never reset,
    so a write there by any call outlasts the sweep and fails the whole-parent
    comparison, and the full run's own are checked before `full` is taken.
    Measured on TEP4's 16 heads of the 64-head fused query, every m 1..32768:
    9.23 s (K-B RoPE) and 9.41 s (NoPE) resetting the batch's heads, 21.52 s
    and 21.89 s resetting whole token rows (on TP4's: 9.26 s and 21.55 s). On
    TEP8's 8 heads of it: 5.49 s (K-B RoPE) and 5.76 s (NoPE) resetting the
    batch's heads, 20.77 s and 20.84 s resetting whole token rows.
    """
    counts = sorted(counts, reverse=True)
    hi = counts[0]
    a, w, out, _, parent = _build(spec, seed, m=hi)
    assert op(a, w, out) is out
    _gate(a, w, out)
    _other_heads_untouched(spec, parent)
    full = parent.clone()
    rows, full_rows = _words(parent), _words(full)
    if spec.get("out_heads", spec["batch"]) > spec["batch"]:
        rows, full_rows = rows[:, : spec["batch"]], full_rows[:, : spec["batch"]]
    differs = []
    for m in counts:
        rows[:m].fill_(SENT_WORD)
        op(a[:, :m], w, out[:, :m])
        differs.append(torch.ne(rows[:m], full_rows[:m]).any())
    failing = torch.nonzero(torch.stack(differs)).flatten().tolist()
    if failing:
        m = counts[failing[-1]]
        _bits(parent[:m]).fill_(SENT)
        op(a[:, :m], w, out[:, :m])
        raise AssertionError(
            f"{len(failing)} row counts differ from the {hi}-row run; first m={m}, "
            f"re-run: {int((_bits(parent[:m]) != _bits(full[:m])).sum())} elements"
        )
    stray = int((_bits(parent) != _bits(full)).sum())
    assert stray == 0, f"{stray} elements past a call's view were written"


def _replays(spec, seed) -> None:
    """Each row count captured in its own CUDA graph, replayed on new rows.

    The replay has to equal an eager call on the same rows bit for bit -- a
    graph that froze something of the capture would differ -- has to leave
    the parent's other columns (and, with `out_heads`, its other heads) alone,
    and is gated against the reference like any other output. Each capture is
    preceded by one call on the capture stream, as an engine's warm-up is.
    """
    stream = torch.cuda.Stream()
    for m in spec["graph_m"]:
        a, w, out, a_parent, parent = _build(spec, seed + m, m=m)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(a, w, out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            op(a, w, out)
        for replay in range(2):
            fresh, _, eager, fresh_parent, _ = _build(spec, seed + 1000 * (replay + 1) + m, m=m)
            a_parent.copy_(fresh_parent)
            _bits(parent).fill_(SENT)
            graph.replay()
            op(fresh, w, eager)
            assert torch.equal(_bits(out), _bits(eager)), f"m={m}: replay {replay} differs"
            assert bool((_bits(parent[..., spec["n"] :]) == SENT).all())
            _other_heads_untouched(spec, parent)
            _compare(out, op.reference(fresh, w, out))


def _tail(spec, seed) -> None:
    """The trailing rows of full-size buffers, as a mixed step's generation
    rows are: bitwise the same rows of a full-size call, gated against the
    reference, and nothing written ahead of them or past their columns."""
    a, w, out, _, parent = _build(spec, seed)
    op(a, w, out)
    full = parent.clone()
    _bits(parent).fill_(SENT)
    t, n = spec["tail_from"], spec["n"]
    a_tail, out_tail = a[:, t:], out[:, t:]
    assert out_tail.storage_offset() > 0
    assert op(a_tail, w, out_tail) is out_tail
    _compare(out_tail, op.reference(a_tail, w, out_tail))
    assert torch.equal(_bits(parent[t:, :, :n]), _bits(full[t:, :, :n]))
    assert bool((_bits(parent[:t]) == SENT).all()), "the op wrote rows ahead of the slice"
    assert bool((_bits(parent[t:, :, n:]) == SENT).all()), "the op wrote outside the out view"
    _other_heads_untouched(spec, parent)


def _build_transposed(spec, seed):
    """Materialize one `transposed_out` cell.

    `transposed_out` is the part a spec cannot express as a plain `empty`: the
    targets hand the op a transposed view of a buffer the forward already
    holds, so the product lands in place. The parent buffer comes back too,
    because "did it land there" is the thing worth asserting.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    b, m, k, n, dt = spec["batch"], spec["m"], spec["k"], spec["n"], spec["dtype"]
    a = torch.randn(b, m, k, generator=g, device="cuda").to(dt)
    w = torch.randn(b, k, n, generator=g, device="cuda").to(dt)
    if spec["transposed_out"]:
        parent = torch.zeros(m, b, n, dtype=dt, device="cuda")
        out = torch.transpose(parent, 0, 1)
        # Only a decode-shaped cell escapes this: transposing a dim of extent 1
        # leaves a tensor torch still calls contiguous, so the assertion would
        # be about the shape rather than about the view.
        assert m == 1 or not out.is_contiguous()
    else:
        parent = torch.zeros(b, m, n, dtype=dt, device="cuda")
        out = parent
    return a, w, out, parent


def _transposed_single(spec, seed) -> None:
    a, w, out, parent = _build_transposed(spec, seed)
    parent_ptr = parent.data_ptr()
    # Identity, not equality: the op must have written this very buffer and
    # returned it, not produced a copy that happens to match.
    assert op(a, w, out) is out, "the op writes in place and returns what it wrote"
    # The product has to be in the caller's buffer, not in a replacement the op
    # allocated: that is the whole reason this op exists over torch.bmm.
    assert parent.data_ptr() == parent_ptr
    _compare(out, op.reference(a, w, out))


# Mistral Large 4's two absorbed BMMs per layer, batch = the H heads a rank holds:
# this rank's 16 at TEP4 and TP4, all 64 at DEP4 (attention data parallel),
# this rank's 8 at TEP8 and TP8. `a_row` / `out_row` are the token-major
# buffers' per-head widths.
_ML4_CALLS = {
    # K-B on the 54 RoPE layers: the 64-wide nope half of q [T, H, 128]
    # (nope | rope) into the first 512 columns of the fused query [T, H, 576].
    "k_b_rope": dict(k=64, n=512, a_row=128, out_row=576, b_layout="kn"),
    # K-B on the 18 NoPE layers: all 128 columns of q are nope.
    "k_b_nope": dict(k=128, n=512, a_row=128, out_row=576, b_layout="kn"),
    # V-B: the latent attention output [T, H, 512] into the [T, H, 128] view
    # of the [T, H x 128] attention output; `b` is the transposed view of the
    # V-B rows of the regrouped `wkv_b`, as `weights.py` lays it out.
    "v_b": dict(k=512, n=128, a_row=512, out_row=128, b_layout="nk_t"),
}


#: max_num_tokens of the two Mistral Large 4 targets: the FP8 target serves 9216,
#: the NVFP4 target 32768. Neither chunks prefill, so one step carries up to
#: that many rows through both BMMs.
_FP8_ROWS = 9216
_NVFP4_ROWS = 32768


def _ml4(call: str, **overrides) -> dict:
    """One Mistral Large 4 call at the FP8 target's max_num_tokens 9216, unless `m`
    is overridden."""
    return dict(dict(batch=16, m=_FP8_ROWS, dtype=torch.bfloat16, **_ML4_CALLS[call]), **overrides)


#: The NVFP4 checkpoint at DEP4: batch = all 64 heads, over the rank's own
#: tokens, up to the same max_num_tokens 32768 per rank. Each cell spreads one
#: `_ML4_CALLS` entry over this.
_ML4_64 = dict(batch=64, m=_NVFP4_ROWS, dtype=torch.bfloat16)

#: The NVFP4 checkpoint at TEP8 and TP8 (the two shard attention alike): batch = this
#: rank's 8 of the 64 heads, over every token of the step (attention DP off), up to
#: max_num_tokens 32768.
#: Each cell spreads one `_ML4_CALLS` entry over this.
_ML4_8 = dict(batch=8, m=_NVFP4_ROWS, dtype=torch.bfloat16)

#: The same at TEP8 and TP8 on the fp8_ds_mla latent KV: FlashMLA's sparse kernels take 64
#: query heads, so the K-B BMM writes the rank's 8 heads into heads 0..7 of a `[T, 64, 576]`
#: fused query (`out_heads`; the token stride is 64 x 576, the head stride 576) whose heads
#: 8..63 stay zero.
_ML4_8_FQ64 = dict(_ML4_8, out_heads=64)

#: The NVFP4 checkpoint at 16 heads per rank (TEP4, TP4) on the fp8_ds_mla latent KV: FlashMLA's
#: sparse kernels take 64 query heads, so the K-B BMM writes the rank's 16 heads into heads 0..15
#: of a `[T, 64, 576]` fused query (`out_heads`; the token stride is 64 x 576 = 36864, the head
#: stride 576) whose heads 16..63 stay zero for the step. Chunked prefill keeps a step's rows at
#: max_num_tokens 32768 or fewer.
_ML4_16_FQ64 = dict(batch=16, m=_NVFP4_ROWS, dtype=torch.bfloat16, out_heads=64)


def _ml4_dspark(heads: int) -> tuple[Cell, ...]:
    """DSpark's absorbed K-B and V-B at `heads` heads per rank, graph rows 1..64.

    These row counts are below the eager envelope and the integer-width
    boundaries the entry's note records.
    """
    return tuple(
        Cell(
            why=f"DSpark absorbed {call}, {heads} heads, graph rows 1..64",
            spec=dict(
                batch=heads,
                m=64,
                dtype=torch.bfloat16,
                **_ML4_CALLS[call],
                graph_m=tuple(range(1, 65)),
            ),
        )
        for call in ("k_b_rope", "k_b_nope", "v_b")
    )


# A `b_layout` of "nk_t" is the transposed view the targets pass for V-B; the
# other cells' `b` is contiguous, which the targets pass for K-B.
#
# Mistral Large 4 NVFP4 at TEP4 and TP4: 16 heads per rank.
_ML4_TP4_CELLS: tuple[Cell, ...] = (
    *_ml4_dspark(16),
    Cell(
        why="Mistral Large 4 K-B RoPE (k 64), every m 1..9216: each m's rows bitwise "
        "the leading rows of the 9216-row run, which is gated (~3 s)",
        spec=_ml4("k_b_rope", m_range=(1, 9216)),
    ),
    Cell(
        why="Mistral Large 4 K-B NoPE (k 128), every m 1..9216: each m's rows bitwise "
        "the leading rows of the 9216-row run, which is gated (~3 s)",
        spec=_ml4("k_b_nope", m_range=(1, 9216)),
    ),
    Cell(
        why="Mistral Large 4 V-B (transposed-view b), every m 1..9216: each m's rows "
        "bitwise the leading rows of the 9216-row run, which is gated (~1.5 s)",
        spec=_ml4("v_b", m_range=(1, 9216)),
    ),
    Cell(
        why="Mistral Large 4 K-B RoPE (k 64), CUDA graphs at the captured, padded "
        "decode batches 1/2/4/8/16: replays bitwise equal to eager",
        spec=_ml4("k_b_rope", graph_m=(1, 2, 4, 8, 16)),
    ),
    Cell(
        why="Mistral Large 4 K-B NoPE (k 128), CUDA graphs at the captured, padded "
        "decode batches 1/2/4/8/16: replays bitwise equal to eager",
        spec=_ml4("k_b_nope", graph_m=(1, 2, 4, 8, 16)),
    ),
    Cell(
        why="Mistral Large 4 V-B (transposed-view b), CUDA graphs at the captured, "
        "padded decode batches 1/2/4/8/16: replays bitwise equal to eager",
        spec=_ml4("v_b", graph_m=(1, 2, 4, 8, 16)),
    ),
    Cell(
        why="Mistral Large 4 K-B RoPE (k 64), a mixed step's 15 generation rows: "
        "rows 9201..9215 of 9216-row buffers, a nonzero storage offset",
        spec=_ml4("k_b_rope", tail_from=9201),
    ),
    Cell(
        why="Mistral Large 4 K-B NoPE (k 128), a mixed step's 15 generation rows: "
        "rows 9201..9215 of 9216-row buffers, a nonzero storage offset",
        spec=_ml4("k_b_nope", tail_from=9201),
    ),
    Cell(
        why="Mistral Large 4 V-B (transposed-view b), a mixed step's 15 generation "
        "rows: rows 9201..9215 of 9216-row buffers, a nonzero storage offset",
        spec=_ml4("v_b", tail_from=9201),
    ),
    # The NVFP4 target's envelope: max_num_tokens 32768 without chunked
    # prefill, so one context step passes both BMMs up to 32768 rows (one
    # request, or up to 16 packed), and the generation rows of a mixed step
    # sit behind up to 32753 context rows (max_batch_size 16). The captured
    # decode batches are 1/2/4/8/16 again, which the graph cells above hold.
    Cell(
        why="Mistral Large 4 32768 K-B RoPE (k 64), every m 1..32768, the NVFP4 "
        "target's max_num_tokens: each m's rows bitwise the leading rows of the "
        "32768-row run, which is gated (~7 s)",
        spec=_ml4("k_b_rope", m=_NVFP4_ROWS, m_range=(1, _NVFP4_ROWS)),
    ),
    Cell(
        why="Mistral Large 4 32768 K-B NoPE (k 128), every m 1..32768, the NVFP4 "
        "target's max_num_tokens: each m's rows bitwise the leading rows of the "
        "32768-row run, which is gated (~7 s)",
        spec=_ml4("k_b_nope", m=_NVFP4_ROWS, m_range=(1, _NVFP4_ROWS)),
    ),
    Cell(
        why="Mistral Large 4 32768 V-B (transposed-view b), every m 1..32768, the "
        "NVFP4 target's max_num_tokens: each m's rows bitwise the leading rows "
        "of the 32768-row run, which is gated (~4 s)",
        spec=_ml4("v_b", m=_NVFP4_ROWS, m_range=(1, _NVFP4_ROWS)),
    ),
    Cell(
        why="Mistral Large 4 32768 K-B RoPE (k 64), a mixed step's 15 generation "
        "rows: rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
        spec=_ml4("k_b_rope", m=_NVFP4_ROWS, tail_from=_NVFP4_ROWS - 15),
    ),
    Cell(
        why="Mistral Large 4 32768 K-B NoPE (k 128), a mixed step's 15 generation "
        "rows: rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
        spec=_ml4("k_b_nope", m=_NVFP4_ROWS, tail_from=_NVFP4_ROWS - 15),
    ),
    Cell(
        why="Mistral Large 4 32768 V-B (transposed-view b), a mixed step's 15 "
        "generation rows: rows 32753..32767 of 32768-row buffers, a nonzero "
        "storage offset",
        spec=_ml4("v_b", m=_NVFP4_ROWS, tail_from=_NVFP4_ROWS - 15),
    ),
    # TEP4 and TP4 on the fp8_ds_mla latent KV (PrefillTarget._attention and
    # DecodeTarget._attention, one call over every row of the step): the two K-B
    # calls write the rank's 16 heads into heads 0..15 of the 64-head fused query,
    # so the out token stride is 36864 where the `_ml4` cells' is 9216. V-B reads
    # the compact `[T, 16, 512]` latent output, which the `_ml4` V-B cells above
    # hold. m is a step's rows: every count 1..32768 eager (chunked prefill keeps
    # a step at max_num_tokens), the padded decode batches 1/2/4/8/16 captured
    # (the LLM API's default list reaches every batch 1..16), and a mixed step's
    # generation rows at the tail.
    Cell(
        why="fused q64 K-B RoPE (k 64), 16 heads into heads 0..15 of [T, 64, 576], "
        "every m 1..32768: each m's rows bitwise the leading rows of the 32768-row "
        "run, which is gated; heads 16..63 never written (~9 s)",
        spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_rope"], m_range=(1, _NVFP4_ROWS)),
    ),
    Cell(
        why="fused q64 K-B NoPE (k 128), 16 heads into heads 0..15 of [T, 64, 576], "
        "every m 1..32768: each m's rows bitwise the leading rows of the 32768-row "
        "run, which is gated; heads 16..63 never written (~9 s)",
        spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_nope"], m_range=(1, _NVFP4_ROWS)),
    ),
    Cell(
        why="fused q64 K-B RoPE (k 64), 16 heads, CUDA graphs at every decode batch "
        "1..16: replays bitwise equal to eager, heads 16..63 never written",
        spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_rope"], graph_m=tuple(range(1, 17))),
    ),
    Cell(
        why="fused q64 K-B NoPE (k 128), 16 heads, CUDA graphs at every decode batch "
        "1..16: replays bitwise equal to eager, heads 16..63 never written",
        spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_nope"], graph_m=tuple(range(1, 17))),
    ),
    Cell(
        why="fused q64 K-B RoPE (k 64), 16 heads, a mixed step's 15 generation rows: "
        "rows 32753..32767 of 32768-row buffers, heads 16..63 never written",
        spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_rope"], tail_from=_NVFP4_ROWS - 15),
    ),
    Cell(
        why="fused q64 K-B NoPE (k 128), 16 heads, a mixed step's 15 generation rows: "
        "rows 32753..32767 of 32768-row buffers, heads 16..63 never written",
        spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_nope"], tail_from=_NVFP4_ROWS - 15),
    ),
)

# The same two targets over eight ranks, TEP8 and TP8: 8 heads per rank.
_ML4_TP8_CELLS: tuple[Cell, ...] = (
    *_ml4_dspark(8),
    *(
        Cell(
            why=f"DSpark verify {call}, 8 into 64 heads, graph rows 1..64",
            spec=dict(_ML4_8_FQ64, **_ML4_CALLS[call], graph_m=tuple(range(1, 65))),
        )
        for call in ("k_b_rope", "k_b_nope")
    ),
    # TEP8 and TP8 (the "tep8" cells are both targets' calls): the same three
    # calls at batch 8, every rank holding 8 of the 64 heads and running every
    # token of the step (attention DP off), so m is the step's token count: up
    # to 32768 in a context step (no chunked prefill), every batch 1..16 in a
    # captured decode step (the gates' padded 1/2/4/8/16 and the LLM API
    # default's unpadded batches), and a mixed step's generation rows behind
    # up to 32753 context rows. At half the 16-head bytes the every-m sweeps
    # fit a cell's time.
    Cell(
        why="tep8 8 heads K-B RoPE (k 64), every m 1..32768: each m's rows "
        "bitwise the leading rows of the 32768-row run, which is gated (~5 s)",
        spec=dict(_ML4_8, **_ML4_CALLS["k_b_rope"], m_range=(1, _NVFP4_ROWS)),
    ),
    Cell(
        why="tep8 8 heads K-B NoPE (k 128), every m 1..32768: each m's rows "
        "bitwise the leading rows of the 32768-row run, which is gated (~5 s)",
        spec=dict(_ML4_8, **_ML4_CALLS["k_b_nope"], m_range=(1, _NVFP4_ROWS)),
    ),
    Cell(
        why="tep8 8 heads V-B (transposed-view b), every m 1..32768: each m's "
        "rows bitwise the leading rows of the 32768-row run, which is gated (~4 s)",
        spec=dict(_ML4_8, **_ML4_CALLS["v_b"], m_range=(1, _NVFP4_ROWS)),
    ),
    Cell(
        why="tep8 8 heads K-B RoPE (k 64), CUDA graphs at every decode batch "
        "1..16, padded and unpadded: replays bitwise equal to eager",
        spec=dict(_ML4_8, **_ML4_CALLS["k_b_rope"], graph_m=tuple(range(1, 17))),
    ),
    Cell(
        why="tep8 8 heads K-B NoPE (k 128), CUDA graphs at every decode batch "
        "1..16, padded and unpadded: replays bitwise equal to eager",
        spec=dict(_ML4_8, **_ML4_CALLS["k_b_nope"], graph_m=tuple(range(1, 17))),
    ),
    Cell(
        why="tep8 8 heads V-B (transposed-view b), CUDA graphs at every decode "
        "batch 1..16, padded and unpadded: replays bitwise equal to eager",
        spec=dict(_ML4_8, **_ML4_CALLS["v_b"], graph_m=tuple(range(1, 17))),
    ),
    Cell(
        why="tep8 8 heads K-B RoPE (k 64), a mixed step's 15 generation rows: "
        "rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
        spec=dict(_ML4_8, **_ML4_CALLS["k_b_rope"], tail_from=_NVFP4_ROWS - 15),
    ),
    Cell(
        why="tep8 8 heads K-B NoPE (k 128), a mixed step's 15 generation rows: "
        "rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
        spec=dict(_ML4_8, **_ML4_CALLS["k_b_nope"], tail_from=_NVFP4_ROWS - 15),
    ),
    Cell(
        why="tep8 8 heads V-B (transposed-view b), a mixed step's 15 generation "
        "rows: rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
        spec=dict(_ML4_8, **_ML4_CALLS["v_b"], tail_from=_NVFP4_ROWS - 15),
    ),
    # TEP8 and TP8 on the fp8_ds_mla latent KV (PrefillTarget._attention and
    # DecodeTarget._attention, one call over every row of the step): the two K-B
    # calls into heads 0..7 of the 64-head fused query, so the out token stride
    # is 36864 where the `_ML4_8` cells' is 4608 (V-B reads the compact latent
    # output, as above). Row counts at both edges of every 16-row window (a
    # 64-head row is eight times the 8-head one, so a sweep resetting whole rows
    # at every m would not fit a cell's time), CUDA graphs at every decode batch
    # 1..16 and a mixed step's tail:
    Cell(
        why="tep8 K-B RoPE (k 64) into heads 0..7 of the 64-head fused query, m 1..16 "
        "and both edges of every 16-row window to 32768: each m's rows bitwise the "
        "leading rows of the 32768-row run, which is gated; heads 8..63 untouched",
        spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_rope"], m_window_edges=(1, _NVFP4_ROWS, 16)),
    ),
    Cell(
        why="tep8 K-B NoPE (k 128) into heads 0..7 of the 64-head fused query, m 1..16 "
        "and both edges of every 16-row window to 32768: each m's rows bitwise the "
        "leading rows of the 32768-row run, which is gated; heads 8..63 untouched",
        spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_nope"], m_window_edges=(1, _NVFP4_ROWS, 16)),
    ),
    Cell(
        why="tep8 K-B RoPE (k 64) into the 64-head fused query, CUDA graphs at every "
        "decode batch 1..16: replays bitwise equal to eager, heads 8..63 untouched",
        spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_rope"], graph_m=tuple(range(1, 17))),
    ),
    Cell(
        why="tep8 K-B NoPE (k 128) into the 64-head fused query, CUDA graphs at every "
        "decode batch 1..16: replays bitwise equal to eager, heads 8..63 untouched",
        spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_nope"], graph_m=tuple(range(1, 17))),
    ),
    Cell(
        why="tep8 K-B RoPE (k 64) into the 64-head fused query, a mixed step's 15 "
        "generation rows: rows 32753..32767 of 32768-row buffers",
        spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_rope"], tail_from=_NVFP4_ROWS - 15),
    ),
    Cell(
        why="tep8 K-B NoPE (k 128) into the 64-head fused query, a mixed step's 15 "
        "generation rows: rows 32753..32767 of 32768-row buffers",
        spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_nope"], tail_from=_NVFP4_ROWS - 15),
    ),
    # ... and every row count 1..32768: at max_seq_len 1048576 with chunked
    # prefill (max_num_tokens 32768) a step is any mix of chunk rows -- a
    # request's last chunk is its prompt length mod 32768 -- and up to 16 decode
    # rows, so every m to max_num_tokens occurs. The sweep resets and compares
    # only the call's 8 heads of each row, which keeps it within a cell's time.
    Cell(
        why="tep8 fused q64 K-B RoPE (k 64), heads 0..7 of [T, 64, 576], every m "
        "1..32768: each m's rows bitwise the leading rows of the 32768-row run, "
        "which is gated; heads 8..63 never written (~5 s)",
        spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_rope"], m_range=(1, _NVFP4_ROWS)),
    ),
    Cell(
        why="tep8 fused q64 K-B NoPE (k 128), heads 0..7 of [T, 64, 576], every m "
        "1..32768: each m's rows bitwise the leading rows of the 32768-row run, "
        "which is gated; heads 8..63 never written (~5 s)",
        spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_nope"], m_range=(1, _NVFP4_ROWS)),
    ),
)

# Mistral Large 4 NVFP4 at DEP4 and DEP8: all 64 heads on every rank.
_ML4_DP_CELLS: tuple[Cell, ...] = (
    *_ml4_dspark(64),
    # DEP4: the same three calls at batch 64, every rank holding all heads
    # and running only its own requests, so m is the rank's own token count:
    # up to 32768 in a context step (no chunked prefill), 1/2/4/8/16 in the
    # captured decode batches, and a mixed step's generation rows behind up
    # to 32753 context rows. Every m 1..32768 is four times the 16-head
    # sweep's bytes and past a cell's time, so those three cells are
    # `large`; pre-merge drives every m where cuBLAS's choice changes.
    Cell(
        why="dep4 64 heads K-B RoPE (k 64), every m 1..32768: each m's rows "
        "bitwise the leading rows of the 32768-row run, which is gated "
        "(large: ~27 s)",
        spec=dict(_ML4_64, **_ML4_CALLS["k_b_rope"], m_range=(1, _NVFP4_ROWS), large=True),
    ),
    Cell(
        why="dep4 64 heads K-B NoPE (k 128), every m 1..32768: each m's rows "
        "bitwise the leading rows of the 32768-row run, which is gated "
        "(large: ~27 s)",
        spec=dict(_ML4_64, **_ML4_CALLS["k_b_nope"], m_range=(1, _NVFP4_ROWS), large=True),
    ),
    Cell(
        why="dep4 64 heads V-B (transposed-view b), every m 1..32768: each m's "
        "rows bitwise the leading rows of the 32768-row run, which is gated "
        "(large: ~12 s)",
        spec=dict(_ML4_64, **_ML4_CALLS["v_b"], m_range=(1, _NVFP4_ROWS), large=True),
    ),
    Cell(
        why="dep4 64 heads K-B RoPE (k 64), edges of every 16-row window to "
        "32768 and every m 1..16, the only places its kernel and grid change: "
        "bitwise the leading rows of the gated 32768-row run (~4 s)",
        spec=dict(_ML4_64, **_ML4_CALLS["k_b_rope"], m_window_edges=(1, _NVFP4_ROWS, 16)),
    ),
    Cell(
        why="dep4 64 heads K-B NoPE (k 128), edges of every 16-row window to "
        "32768 and every m 1..16, the only places its kernel and grid change: "
        "bitwise the leading rows of the gated 32768-row run (~4 s)",
        spec=dict(_ML4_64, **_ML4_CALLS["k_b_nope"], m_window_edges=(1, _NVFP4_ROWS, 16)),
    ),
    Cell(
        why="dep4 64 heads V-B (transposed-view b), edges of every 16-row window "
        "to 32768 and every m 1..16, the only places its kernel and grid change: "
        "bitwise the leading rows of the gated 32768-row run (~2 s)",
        spec=dict(_ML4_64, **_ML4_CALLS["v_b"], m_window_edges=(1, _NVFP4_ROWS, 16)),
    ),
    Cell(
        why="dep4 64 heads K-B RoPE (k 64), CUDA graphs at the captured, padded "
        "decode batches 1/2/4/8/16: replays bitwise equal to eager",
        spec=dict(_ML4_64, **_ML4_CALLS["k_b_rope"], graph_m=(1, 2, 4, 8, 16)),
    ),
    Cell(
        why="dep4 64 heads K-B NoPE (k 128), CUDA graphs at the captured, padded "
        "decode batches 1/2/4/8/16: replays bitwise equal to eager",
        spec=dict(_ML4_64, **_ML4_CALLS["k_b_nope"], graph_m=(1, 2, 4, 8, 16)),
    ),
    Cell(
        why="dep4 64 heads V-B (transposed-view b), CUDA graphs at the captured, "
        "padded decode batches 1/2/4/8/16: replays bitwise equal to eager",
        spec=dict(_ML4_64, **_ML4_CALLS["v_b"], graph_m=(1, 2, 4, 8, 16)),
    ),
    Cell(
        why="dep4 64 heads K-B RoPE (k 64), a mixed step's 15 generation rows: "
        "rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
        spec=dict(_ML4_64, **_ML4_CALLS["k_b_rope"], tail_from=_NVFP4_ROWS - 15),
    ),
    Cell(
        why="dep4 64 heads K-B NoPE (k 128), a mixed step's 15 generation rows: "
        "rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
        spec=dict(_ML4_64, **_ML4_CALLS["k_b_nope"], tail_from=_NVFP4_ROWS - 15),
    ),
    Cell(
        why="dep4 64 heads V-B (transposed-view b), a mixed step's 15 generation "
        "rows: rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
        spec=dict(_ML4_64, **_ML4_CALLS["v_b"], tail_from=_NVFP4_ROWS - 15),
    ),
)

# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="deepseek default at attention TP8, decode absorb: 16 heads per rank",
            spec=dict(batch=16, m=1, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="deepseek default at attention TP4, decode expand: 32 heads per rank",
            spec=dict(batch=32, m=1, k=512, n=128, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="deepseek default at attention TP2, prefill-height absorb: 64 heads per rank",
            spec=dict(batch=64, m=2048, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
    ),
    "mistral_large4/mistral_large_4_nvfp4__sm_103__tep": (*_ML4_TP4_CELLS, *_ML4_TP8_CELLS),
    "mistral_large4/mistral_large_4_nvfp4__sm_103__tp": (*_ML4_TP4_CELLS, *_ML4_TP8_CELLS),
    "mistral_large4/mistral_large_4_nvfp4__sm_103__dep": _ML4_DP_CELLS,
}

CELLS = certified_cells(op, TARGET_CELLS)


def _drive(cell) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    if "transposed_out" in spec:
        _transposed_single(spec, seed)
    elif "m_range" in spec:
        lo, hi = spec["m_range"]
        _sweep(spec, seed, range(lo, hi + 1))
    elif "m_window_edges" in spec:
        _sweep(spec, seed, _window_edges(*spec["m_window_edges"]))
    elif "graph_m" in spec:
        _replays(spec, seed)
    elif "tail_from" in spec:
        _tail(spec, seed)
    else:
        _single(spec, seed)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — the fixture, by name
    if cell.spec.get("large"):
        pytest.skip("past the per-cell budget; test_large_certified_cells drives it")
    certifying(op)
    _drive(cell)


_LARGE_CELLS = tuple(c for c in CELLS if c.spec.get("large"))


@pytest.mark.large
@pytest.mark.parametrize("cell", _LARGE_CELLS, ids=[c.why[:44] for c in _LARGE_CELLS])
def test_large_certified_cells(cell, certifying) -> None:  # noqa: F811 — the fixture, by name
    certifying(op)
    _drive(cell)


def _ml4_spec(k: int, mode: str, batch: int = 16) -> dict:
    """The Mistral Large 4 cell of K width `k` and `batch` heads whose spec carries
    the key `mode`."""
    return next(
        dict(c.spec)
        for c in CELLS
        if c.spec["batch"] == batch and c.spec["k"] == k and mode in c.spec
    )


def _ok(dtype=torch.bfloat16):
    a = torch.randn(4, 8, 16, dtype=dtype, device=DEV)
    w = torch.randn(4, 16, 32, dtype=dtype, device=DEV)
    out = torch.zeros(4, 8, 32, dtype=dtype, device=DEV)
    return a, w, out


def test_a_wrong_shaped_out_is_resized_and_the_guard_refuses_it() -> None:
    """The op resizes a wrong-shaped `out` in place, with only a deprecation
    warning. On Mistral Large 4's K-B view one token short, the view is re-strided
    contiguous over the fused query's storage: the product is there, but not
    in the buffer's layout, and it overwrites the rope halves. A standalone
    `out` is re-allocated, detaching the caller's buffer."""
    spec = _ml4_spec(64, "tail_from")
    assert spec["out_row"] == 576
    a, w, out, _, parent = _build(spec, seed=21, m=16)
    ref = op.reference(a, w, out)
    short = out[:, :15]
    with pytest.warns(UserWarning, match="was resized"):
        op(a, w, short)
    assert short.shape == (16, 16, 512) and short.is_contiguous()
    _compare(short, ref)
    assert int((_bits(parent[..., 512:]) != SENT).sum()) > 0, "the rope halves were overwritten"
    in_layout = int((_bits(parent[..., :512].transpose(0, 1)) == _bits(ref)).sum())
    assert in_layout < ref.numel() // 10, f"{in_layout} of {ref.numel()} in the buffer's layout"

    standalone = torch.zeros(16, 16, 511, dtype=torch.bfloat16, device=DEV)
    ptr = standalone.data_ptr()
    with pytest.warns(UserWarning, match="was resized"):
        op(a, w, standalone)
    assert standalone.shape == (16, 16, 512) and standalone.data_ptr() != ptr

    a, w, out, _, parent = _build(spec, seed=22, m=16)
    for wrong, refusal in (
        (out[:, :15], r"out must be \[B, M, N\]"),
        (out[..., :511], r"out must be \[B, M, N\]"),
        (out[0], "a, b, out must be 3D"),
        (torch.zeros_like(out)[:15], r"out must be \[B, M, N\]"),
    ):
        with pytest.raises(AssertionError, match=refusal):
            with validating(op):
                op(a, w, wrong)
    assert bool((_bits(parent) == SENT).all())


def test_guard_refuses_a_wrong_shaped_out() -> None:
    """The op silently reallocates it, detaching the caller's buffer."""
    a, w, _ = _ok()
    with pytest.raises(AssertionError, match=r"out must be \[B, M, N\]"):
        with validating(op):
            op(a, w, torch.zeros(4, 8, 31, dtype=torch.bfloat16, device="cuda"))


def test_guard_refuses_a_2d_operand() -> None:
    a, w, out = _ok()
    with pytest.raises(AssertionError, match="must be 3D"):
        with validating(op):
            op(a[0], w, out)


def test_guard_refuses_mixed_dtypes() -> None:
    """Promoted rather than rejected, which changes the required out dtype."""
    a, w, out = _ok()
    with pytest.raises(AssertionError, match="share one dtype"):
        with validating(op):
            op(a, w.to(torch.float16), out)


@pytest.mark.parametrize(
    "case, match",
    [
        ("2d_a", "batch1 must be a 3D tensor"),
        ("2d_b", "batch2 must be a 3D tensor"),
        ("k_mismatch", r"Expected size for first two dimensions of batch2 tensor"),
        ("batch_mismatch", r"Expected size for first two dimensions of batch2 tensor"),
        ("bf16_fp16_into_bf16", r"Expected out tensor to have dtype c10::Half"),
        ("bf16_fp16_into_fp16", "expected scalar type BFloat16 but found Half"),
        ("bf16_into_fp32", r"Expected out tensor to have dtype c10::BFloat16, but got float"),
    ],
)
def test_the_op_refuses_the_rest_itself(case, match) -> None:
    """What the op refuses itself, unguarded: every one of these is refused
    before anything is written. The guard restates the 3-D and single-dtype
    checks (the `test_guard_refuses_*` cases)."""
    a, w, out = _ok()
    out.fill_(7.0)
    args = {
        "2d_a": (a[0], w, out),
        "2d_b": (a, w[0], out),
        "k_mismatch": (a, w[:, :15], out),
        "batch_mismatch": (a, w[:3], out),
        "bf16_fp16_into_bf16": (a, w.half(), out),
        "bf16_fp16_into_fp16": (a, w.half(), out.half()),
        "bf16_into_fp32": (a, w, out.float()),
    }[case]
    with pytest.raises(RuntimeError, match=match):
        op(*args)
    assert bool((args[2] == 7.0).all())


def test_zero_rows_write_nothing() -> None:
    spec = _ml4_spec(64, "tail_from")
    a, w, out, _, parent = _build(spec, seed=23, m=4)
    with validating(op):
        result = op(a[:, :0], w, out[:, :0])
    assert result.shape == (16, 0, 512)
    assert bool((_bits(parent) == SENT).all())


@pytest.mark.parametrize("batch", [8, 16, 64])
def test_a_transposed_b_view_matches_the_contiguous_b(batch) -> None:
    """The V-B `b` as the targets hold it -- a transposed view of a weight's
    row slice -- and the same values contiguous: bitwise the same product,
    although cuBLAS runs a different kernel for each. At TEP8's and TP8's 8
    heads, TEP4's 16 and DEP4's 64."""
    spec = _ml4_spec(512, "tail_from", batch)
    assert spec["b_layout"] == "nk_t"
    for m in (1, 16, 9216, 32768):
        a, w, out, _, _ = _build(spec, seed=24 + m, m=m)
        assert w.stride(1) == 1 and w.storage_offset() > 0
        contiguous = torch.empty_like(out)
        op(a, w, out)
        op(a, w.contiguous(), contiguous)
        assert torch.equal(_bits(out), _bits(contiguous)), f"m={m}"


@pytest.mark.parametrize("shift", [1, 48])
def test_fused_q64_harness_sees_a_write_past_the_batch_heads(shift) -> None:
    """The control for the `out_heads` assertion: the same product written
    `shift` heads over in the 64-head buffer (heads 1..16, and the last 16,
    48..63) must fail it, and the product at heads 0..15 must pass it. Without
    this, "heads 16..63 never written" could be a check that cannot fire."""
    spec = _ml4_spec(64, "out_heads")
    assert spec["out_heads"] == 64 and spec["batch"] == 16
    a, w, out, _, parent = _build(spec, seed=25, m=16)
    op(a, w, out)
    _other_heads_untouched(spec, parent)
    op(a, w, parent[:, shift : shift + 16, :512].transpose(0, 1))
    with pytest.raises(AssertionError, match="past the call's batch"):
        _other_heads_untouched(spec, parent)


def test_fused_q64_sweep_sees_a_shorter_call_writing_past_the_batch_heads(monkeypatch) -> None:
    """The control for the sweep's own-heads reset: heads 16..63 are never reset
    or compared per call, so a write there must still fail the sweep. The op
    is wrapped so that one shorter call (17 of 64 rows) also copies its head 0
    into head 16; every per-call compare of the batch's heads still passes,
    and the sweep must fail on the whole-parent comparison. The same sweep
    with the op itself passes."""
    spec = _ml4_spec(64, "out_heads")
    assert "m_range" in spec and spec["out_heads"] == 64 and spec["batch"] == 16
    real, at, bsz = op, 17, spec["batch"]

    class _StrayHead:
        def __call__(self, a, b, out):
            real(a, b, out)
            if out.shape[1] == at:
                head0 = out[:1]
                stray = torch.as_strided(
                    head0, head0.shape, head0.stride(), out.storage_offset() + bsz * out.stride(0)
                )
                stray.copy_(head0)
            return out

        def reference(self, a, b, out):
            return real.reference(a, b, out)

    monkeypatch.setattr(sys.modules[__name__], "op", _StrayHead())
    with pytest.raises(AssertionError, match="past a call's view were written"):
        _sweep(spec, seed=26, counts=range(1, 65))
    monkeypatch.undo()
    _sweep(spec, seed=26, counts=range(1, 65))


@pytest.mark.parametrize("shift", [1, 56])
def test_fused_q64_harness_sees_a_write_past_the_batch_heads_at_8_heads(shift) -> None:
    """The control for the `out_heads` assertion at TEP8's 8 heads: the same
    product written `shift` heads over in the 64-head buffer (heads 1..8, and
    the last 8, 56..63) must fail it, and the product at heads 0..7 must pass
    it. Without this, "heads 8..63 never written" could be a check that
    cannot fire."""
    spec = _ml4_spec(64, "out_heads", batch=8)
    assert spec["out_heads"] == 64 and spec["batch"] == 8
    a, w, out, _, parent = _build(spec, seed=25, m=16)
    op(a, w, out)
    _other_heads_untouched(spec, parent)
    op(a, w, parent[:, shift : shift + 8, :512].transpose(0, 1))
    with pytest.raises(AssertionError, match="past the call's batch"):
        _other_heads_untouched(spec, parent)


def test_fused_q64_sweep_sees_a_shorter_call_writing_past_the_batch_heads_at_8_heads(
    monkeypatch,
) -> None:
    """The control for the sweep's own-heads reset at TEP8's 8 heads: heads
    8..63 are never reset or compared per call, so a write there must still
    fail the sweep. The op is wrapped so that one shorter call (17 of 64 rows)
    also copies its head 0 into head 8; every per-call compare of the batch's
    heads still passes, and the sweep must fail on the whole-parent
    comparison. The same sweep with the op itself passes."""
    spec = _ml4_spec(64, "out_heads", batch=8)
    assert spec["out_heads"] == 64 and spec["batch"] == 8
    real, at, bsz = op, 17, spec["batch"]

    class _StrayHead:
        def __call__(self, a, b, out):
            real(a, b, out)
            if out.shape[1] == at:
                head0 = out[:1]
                stray = torch.as_strided(
                    head0, head0.shape, head0.stride(), out.storage_offset() + bsz * out.stride(0)
                )
                stray.copy_(head0)
            return out

        def reference(self, a, b, out):
            return real.reference(a, b, out)

    monkeypatch.setattr(sys.modules[__name__], "op", _StrayHead())
    with pytest.raises(AssertionError, match="past a call's view were written"):
        _sweep(spec, seed=26, counts=range(1, 65))
    monkeypatch.undo()
    _sweep(spec, seed=26, counts=range(1, 65))
