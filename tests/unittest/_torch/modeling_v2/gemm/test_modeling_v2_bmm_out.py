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
spec says `large` is marked `large`, which pre-merge deselects.
"""

import sys
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import (  # noqa: E501
    bmm_out as op,  # noqa: E501
)

__extra_import_path__ = [".."]
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
    RMS.
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
    tp4 and tep8 targets' fused query is. Both parents come back too: `a`'s so a
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


@pytest.mark.parametrize(
    "cell",
    [pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS],
    ids=[c.why[:44] for c in op.CELLS],
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — the fixture, by name
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    if "m_range" in spec:
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


def _ml4_spec(k: int, mode: str, batch: int = 16) -> dict:
    """The Mistral Large 4 cell of K width `k` and `batch` heads whose spec carries
    the key `mode`."""
    return next(
        dict(c.spec)
        for c in op.CELLS
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
    for wrong in (out[:, :15], out[..., :511], out[0], torch.zeros_like(out)[:15]):
        with pytest.raises(AssertionError, match=r"out must be \[B, M, N\]"):
            with validating(op):
                op(a, w, wrong)
    assert bool((_bits(parent) == SENT).all())


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
    """What the guard leaves alone, because the op raises on it first: every
    one of these is refused before anything is written, guard armed."""
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
        with validating(op):
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
