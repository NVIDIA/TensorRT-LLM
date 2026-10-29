# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flash_mla_sparse_prefill catalog entry.

bf16 inputs, bf16 P, fp32 accumulation against an fp32 reference: relative L2 per (row, head)
under 1e-2, whole call under 4e-3; the reference is a per-row loop and the kernel runs every
row. The workspace cells (`kv_rows`) are each driven by the harness of the run that certified
them, named by the why ("1M tp4"; "tep4 1M", "dep4 1M" and "tep8 1M" share one, dep4's
extension of tep4's, which tep8's is too).

A cell of more than 1024 rows outside the "tep4 1M", "dep4 1M" and "tep8 1M" ones is gated on
its last 128 rows, 896 drawn ones, the first and last 8 of each request and, when the query
reaches it, the 16 rows around its 2**31-byte offset. A "1M tp4" cell also checks every row:
finite (a read of the stale NaN rows behind the workspace view would show), each 4096-row
window re-run bitwise equal to the full call's rows, and the rows of `out`'s buffer past the
call untouched.
The 1M cells of 16 rows or more measured per-row 4.1e-3..4.8e-3 and whole-call
1.8e-3..2.1e-3 on sm_103, the fresh 2- and 7-row dense steps at most 2.5e-3 per row, the
1-row step (one key) exactly 0; the two tests after the cells show the kernel skips an index
past the view without reading its row, and that the gate sees the workspace defects next to
this call.

A "tep4 1M", "dep4 1M" or "tep8 1M" workspace cell is gated on 1024 sampled rows plus each
request's first and last rows and the rows where q's byte offset passes 2**31. It holds every
row finite (the buffer's stale rows and the rows after q and out are NaN) and bitwise equal to
the same rows called alone in windows of `_WINDOW`. A `row_sweep` cell calls every leading row
count of its call and a `row_counts` cell the counts it lists, each bitwise the full call's
leading rows and writing none of out's rows past its own; `large` cells are marked for the
stage that runs them.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flash_mla_sparse_prefill import (
    flash_mla_sparse_prefill as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "flash_mla_sparse_prefill requires a CUDA device"

SCALE = 128**-0.5
_GATED_ROWS = 1024
#: The bits (bf16 NaN) of every workspace row past the view and of `out`'s buffer tail.
_STALE = -1
_OUT_TAIL = 16
_WINDOW_TP4 = 4096
#: Query row whose bytes straddle 2**31 in a [rows, 64, 576] bf16 query (29127).
_Q_BYTE_EDGE = 2**31 // (64 * 576 * 2)
#: NaN rows after the q scratch's and the out buffer's rows, so a read or a write past the
#: call's rows shows.
_PAD = 16
#: Row-independence windows: 4093 rows, so no window starts or ends on a power of two.
_WINDOW = 4093
#: q is [rows, 64, 576] bf16, 73728 bytes a row: its byte offset passes 2**31 inside this row.
_Q_2GB_ROW = 2**31 // (64 * 576 * 2)


def _positions(spec):
    """Each query row's position in its request and its request's run offset in `kv`."""
    kv_lens = spec["kv_len"]
    new = spec.get("new", kv_lens)
    pos, base = [], []
    off = 0
    for n_kv, n_new in zip(kv_lens, new):
        pos.append(torch.arange(n_kv - n_new, n_kv, device="cuda"))
        base.append(torch.full((n_new,), off, device="cuda"))
        off += n_kv
    return torch.cat(pos), torch.cat(base)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv_lens = spec["kv_len"]
    kv = (torch.randn(sum(kv_lens), 576, device="cuda", generator=g) * 2).to(torch.bfloat16)
    # Row j of a request sees positions [0, p_j]: min(p_j + 1, topk) draws of them (with
    # repeats, which the kernel and the reference both count), offset to the request's run.
    pos, base = _positions(spec)
    col = torch.arange(spec["topk"], device="cuda")[None]
    draw = torch.rand(pos.numel(), spec["topk"], device="cuda", generator=g) * (pos[:, None] + 1)
    idx = torch.where(col <= pos[:, None], draw.long() + base[:, None], -1)
    if spec.get("holes"):
        idx[torch.rand(idx.shape, device="cuda", generator=g) < 0.2] = -1
    idx = idx.to(torch.int32)
    assert idx.shape[0] == spec["rows"]
    q = torch.zeros(spec["rows"], spec["h_q"], 576, device="cuda")
    q[:, : spec["h_out"]] = torch.randn(
        spec["rows"], spec["h_out"], 576, device="cuda", generator=g
    )
    out = torch.full((spec["rows"], spec["h_out"], 512), float("nan"), dtype=torch.bfloat16,
                     device="cuda")  # fmt: skip
    return q.to(torch.bfloat16), kv, idx, out


def _selection(pos: torch.Tensor, topk: int, g: torch.Generator, dense: bool) -> torch.Tensor:
    """Request-local selections, int64 [rows, topk], for rows at positions `pos`.

    `dense`: the engine's dense causal index for a skipped indexer, 0..p then -1 (DSA
    metadata `_get_dense_topk_indices`). Otherwise as the indexer's top-k returns them: a
    row with p + 1 <= topk candidates lists 0..p in order then -1; a longer one holds topk
    distinct positions, one per stratum of [0, p] (0 and p among them), in shuffled order.
    """
    col = torch.arange(topk, device=pos.device)[None]
    causal = torch.where(col <= pos[:, None], col, -1)
    if dense:
        return causal
    n = pos[:, None] + 1
    lo = col * n // topk
    hi = (col + 1) * n // topk
    draw = torch.rand(lo.shape, device=pos.device, generator=g)
    pick = torch.minimum(lo + (draw * (hi - lo)).long(), hi - 1)
    pick[:, 0] = 0
    pick[:, -1] = pos
    order = torch.rand(lo.shape, device=pos.device, generator=g).argsort(dim=1)
    return torch.where(n > topk, pick.gather(1, order), causal)


def _build_1m(spec, seed):
    """The call's arguments, and the buffer `out` leads.

    `kv` is the first N rows of a `kv_rows`-row workspace whose later rows are NaN, as the
    target's view of its 2**19-row buffer; `q` is zero past `h_out` heads; `out` leads a
    NaN buffer `_OUT_TAIL` rows longer, as the context rows lead the step's `lat_out`.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = sum(spec["kv_len"])
    buf = torch.empty(spec["kv_rows"], 576, dtype=torch.bfloat16, device="cuda")
    buf.view(torch.int16).fill_(_STALE)
    buf[:n] = (torch.randn(n, 576, device="cuda", generator=g) * 2).to(torch.bfloat16)
    pos, base = _positions(spec)
    sel = _selection(pos, spec["topk"], g, spec.get("dense", False))
    idx = torch.where(sel >= 0, sel + base[:, None], sel).to(torch.int32)
    rows = spec["rows"]
    assert idx.shape[0] == rows
    q = torch.zeros(rows, spec["h_q"], 576, dtype=torch.bfloat16, device="cuda")
    q[:, : spec["h_out"]] = torch.randn(rows, spec["h_out"], 576, device="cuda", generator=g)
    out_buf = torch.empty(rows + _OUT_TAIL, spec["h_out"], 512, dtype=torch.bfloat16,
                          device="cuda")  # fmt: skip
    out_buf.view(torch.int16).fill_(_STALE)
    return (q, buf[:n], idx, out_buf[:rows]), out_buf


def _gated_rows_tp4(spec) -> torch.Tensor:
    rows = torch.arange(spec["rows"], device="cuda")
    if spec["rows"] <= _GATED_ROWS:
        return rows
    g = torch.Generator(device="cuda").manual_seed(0)
    picked = torch.randperm(spec["rows"], device="cuda", generator=g)[: _GATED_ROWS - 128]
    edges = []
    start = 0
    for n_new in spec.get("new", spec["kv_len"]):
        edges += [*range(start, start + min(8, n_new)), *range(max(start, start + n_new - 8),
                                                              start + n_new)]  # fmt: skip
        start += n_new
    if spec["rows"] > _Q_BYTE_EDGE + 8 and spec["h_q"] == 64:
        edges += range(_Q_BYTE_EDGE - 8, _Q_BYTE_EDGE + 8)
    extra = torch.tensor(edges, device="cuda", dtype=rows.dtype)
    return torch.unique(torch.cat([rows[-128:], picked, extra]))


def _errors_tp4(o: torch.Tensor, r: torch.Tensor) -> tuple[float, float]:
    """Largest relative L2 per (row, head), and the whole call's."""
    o, r = o.float(), r.float()
    per = (o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
    return float(per.max()), float((o - r).norm() / r.norm())


def _assert_close(o: torch.Tensor, r: torch.Tensor) -> None:
    per, whole = _errors_tp4(o, r)
    assert per < 1e-2 and whole < 4e-3, (per, whole)


def _check_windows_tp4(q, kv, idx, out) -> int:
    """Re-run every `_WINDOW_TP4` rows at query rows 0.. of their own call: each bitwise the full
    call's rows, so no row's answer depends on where in the step it sits. Returns how many
    calls ran."""
    rows = q.shape[0]
    scratch = torch.empty(_WINDOW_TP4, out.shape[1], 512, dtype=out.dtype, device=out.device)
    bad = torch.zeros((), dtype=torch.bool, device=out.device)
    calls = 0
    for a in range(0, rows, _WINDOW_TP4):
        b = min(a + _WINDOW_TP4, rows)
        got = op(q[a:b], kv, idx[a:b], SCALE, scratch[: b - a])
        bad |= (got.view(torch.int16) != out[a:b].view(torch.int16)).any()
        calls += 1
    assert not bad.item(), "a window's rows differ from the full call's"
    return calls


def _runs(spec):
    """Each query row's request-local position, and its request's offset in the workspace (the
    run of a request starts where the previous request's `kv_len` ends)."""
    pos, base = [], []
    off = 0
    for n_kv, n_new in zip(spec["kv_len"], spec["new"]):
        pos.append(torch.arange(n_kv - n_new, n_kv, device="cuda"))
        base.append(torch.full((n_new,), off, device="cuda"))
        off += n_kv
    return torch.cat(pos), torch.cat(base)


def _topk_selection(pos, topk, g):
    """The indexer's top-`topk` output shape for rows at request-local positions `pos`. A row
    that sees more than `topk` keys selects `topk` distinct ones, one in each of `topk` equal
    strata of [0, p] (0 and p among them, so the run's first and last rows are read); a row that
    sees fewer selects all of them, then -1. The valid selections are in a shuffled order."""
    col = torch.arange(topk, device="cuda")[None]
    span = pos[:, None] + 1
    lo = col * span // topk
    width = (col + 1) * span // topk - lo
    pick = lo + (torch.rand(lo.shape, device="cuda", generator=g) * width).long()
    pick[:, 0] = 0
    pick[:, -1] = pos
    sel = torch.where(span > topk, pick, torch.where(col < span, col, -1))
    del lo, width, pick
    key = torch.rand(sel.shape, device="cuda", generator=g).masked_fill_(sel < 0, 2.0)
    return torch.gather(sel, 1, key.argsort(dim=1))


def _dense_selection(pos, topk):
    """DSA metadata's `_get_dense_topk_indices` for rows at request-local positions `pos` (the
    cached prefix counted): column j holds j where j <= p, else -1."""
    col = torch.arange(topk, device="cuda")[None]
    return torch.where(col <= pos[:, None], col, -1)


def _build_ws(spec, seed):
    """A workspace cell's arguments, and the buffers they lead.

    kv is the first N = sum(kv_len) rows of a `kv_rows`-row bf16 buffer whose later rows are
    NaN, as the bounded workspace's rows past the step's KV hold earlier steps' rows. The
    indices are request-local selections plus each request's offset in the workspace, -1 kept.
    q is the first `rows` rows of the 64-head scratch, `h_out` live heads (all 64 at dep4) and
    zeros after them, followed by `_PAD` NaN rows; out leads a NaN buffer `_PAD` rows longer.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, n = spec["rows"], sum(spec["kv_len"])
    assert n <= spec["kv_rows"]
    buf = torch.full((spec["kv_rows"], 576), float("nan"), dtype=torch.bfloat16, device="cuda")
    buf[:n] = (torch.randn(n, 576, device="cuda", generator=g) * 2).to(torch.bfloat16)
    pos, base = _runs(spec)
    assert pos.numel() == rows
    if spec.get("dense"):
        sel = _dense_selection(pos, spec["topk"])
    else:
        assert spec.get("selection", "topk") == "topk"
        sel = _topk_selection(pos, spec["topk"], g)
    idx = torch.where(sel >= 0, sel + base[:, None], -1).to(torch.int32)
    del sel
    h_q, h_out = spec["h_q"], spec["h_out"]
    q_buf = torch.full((rows + _PAD, h_q, 576), float("nan"), dtype=torch.bfloat16, device="cuda")
    q_buf[:rows] = 0
    q_buf[:rows, :h_out] = torch.randn(rows, h_out, 576, device="cuda", generator=g)
    out_buf = torch.full((rows + _PAD, h_out, 512), float("nan"), dtype=torch.bfloat16,
                         device="cuda")  # fmt: skip
    return (q_buf[:rows], buf[:n], idx, out_buf[:rows]), (q_buf, buf, out_buf)


def _gated_rows(spec, rows):
    """The rows the reference gates: every row up to `_GATED_ROWS`, else the last 128 and 896
    drawn; a workspace cell adds each request's first and last 4 rows and the 8 rows around
    `_Q_2GB_ROW`."""
    every = torch.arange(rows, device="cuda")
    if rows <= _GATED_ROWS:
        return every
    g = torch.Generator(device="cuda").manual_seed(0)
    picked = torch.randperm(rows, device="cuda", generator=g)[: _GATED_ROWS - 128]
    gated = [every[-128:], picked]
    if "kv_rows" not in spec:
        return torch.cat(gated)
    start = 0
    for n_new in spec["new"]:
        gated += [every[start : start + 4], every[start + n_new - 4 : start + n_new]]
        start += n_new
    if rows > _Q_2GB_ROW + 4:
        gated.append(every[_Q_2GB_ROW - 4 : _Q_2GB_ROW + 4])
    return torch.unique(torch.cat(gated))


def _errors(out, ref):
    """Largest per-(row, head) relative L2 error, and the whole call's."""
    o, r = out.float(), ref.float()
    per = (o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
    return float(per.max()), float((o - r).norm() / r.norm())


def _bits(t):
    return t.view(torch.int16)


def _check_windows(q, kv, idx, out):
    """Every `_WINDOW` rows called alone answer bitwise as in the full call: no row depends on
    the rows beside it or on where its call starts. One sync for all windows."""
    differ = torch.zeros((), dtype=torch.bool, device="cuda")
    for s in range(0, q.shape[0], _WINDOW):
        e = min(s + _WINDOW, q.shape[0])
        part = op(q[s:e], kv, idx[s:e], SCALE, torch.empty_like(out[s:e]))
        differ |= (_bits(part) != _bits(out[s:e])).any()
    assert not differ.item(), "a window of rows answered other than in the full call"


def _sweep(q, kv, idx, full, counts, call=op):
    """For each m in `counts`, the call on q[:m] and idx[:m] into the first m rows of a NaN
    buffer answers bitwise as the full call's first m rows, and no call writes a row of the
    buffer past its own: each m's rows go back to NaN after their compare, so every row is NaN
    at the end. Largest m first, so the kernel's own output reuses the first call's block. Rows
    compare as int64 words, one flag per m, one sync at the end. `call` stands in for the op in
    the harness control."""
    rows = q.shape[0]
    buf = torch.full((rows + _PAD, *full.shape[1:]), float("nan"), dtype=full.dtype,
                     device="cuda")  # fmt: skip
    words = buf.view(rows + _PAD, -1).view(torch.int64)
    want = full.reshape(rows, -1).view(torch.int64)
    nan = int(words[-1, 0])
    order = sorted(counts, reverse=True)
    bad = torch.zeros(len(order), dtype=torch.bool, device="cuda")
    for i, m in enumerate(order):
        call(q[:m], kv, idx[:m], SCALE, buf[:m])
        bad[i] = torch.ne(words[:m], want[:m]).any()
        words[:m].fill_(nan)
    failed = [m for m, b in zip(order, bad.tolist()) if b]
    assert not failed, f"{len(failed)} row counts answered other than the full call: {failed[:8]}"
    assert not bool(torch.ne(words, nan).any()), "a call wrote out's rows past its own"


def _run_ws(spec, seed) -> None:
    """A "tep4 1M", "dep4 1M" or "tep8 1M" workspace cell, as its run certified it."""
    (q, kv, idx, out), (_, _, out_buf) = _build_ws(spec, seed)
    got = op(q, kv, idx, SCALE, out)
    assert got.data_ptr() == out.data_ptr()
    rows = _gated_rows(spec, q.shape[0])
    ref = op.reference(q[rows], kv, idx[rows], SCALE, out[rows])
    per, whole = _errors(out[rows], ref)
    assert per < 1e-2 and whole < 4e-3, (per, whole)
    assert bool(torch.isfinite(out).all()), "a row read the buffer's stale rows or q's padding"
    assert bool(torch.isnan(out_buf[q.shape[0] :]).all()), "the call wrote past out's rows"
    if q.shape[0] > _WINDOW:
        _check_windows(q, kv, idx, out)
    if spec.get("row_sweep"):
        _sweep(q, kv, idx, out, range(1, q.shape[0] + 1))
    elif "row_counts" in spec:
        _sweep(q, kv, idx, out, spec["row_counts"])


@pytest.mark.parametrize(
    "cell",
    [
        pytest.param(c, id=c.why[:60], marks=[pytest.mark.large] if c.spec.get("large") else [])
        for c in op.CELLS
    ],
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    seed = zlib.crc32(cell.why.encode())
    if "kv_rows" in cell.spec and not cell.why.startswith("1M tp4 "):
        _run_ws(cell.spec, seed)
        return
    one_m = "kv_rows" in cell.spec
    if one_m:
        (q, kv, idx, out), out_buf = _build_1m(cell.spec, seed)
    else:
        q, kv, idx, out = _build(cell.spec, seed)
    got = op(q, kv, idx, SCALE, out)
    assert got.data_ptr() == out.data_ptr()
    rows = _gated_rows_tp4(cell.spec)
    ref = op.reference(q[rows], kv, idx[rows], SCALE, out[rows])
    _assert_close(out[rows], ref)
    if one_m:
        assert bool(torch.isfinite(out).all()), "a row came back non-finite"
        assert not (out_buf[q.shape[0] :].view(torch.int16) != _STALE).any(), "wrote past out"
        if q.shape[0] > _WINDOW_TP4:
            assert _check_windows_tp4(q, kv, idx, out) > 1


_SHORT = dict(rows=512, kv_len=[8192], new=[512], h_q=64, h_out=16, topk=2048, kv_rows=16384,
              selection="topk")  # fmt: skip


def test_indices_past_kv_are_skipped_not_read() -> None:
    """An index at or past kv.shape[0] is skipped as -1 is (FlashMLA's contract for its
    indices): bitwise the answer with -1 in its place, and the NaN rows it names behind the
    view are never read. Control: the same indices over the whole buffer read those rows,
    and the rows come back NaN, so the stale fill is visible to this harness."""
    (q, kv, idx, out), _ = _build_1m(_SHORT, 11)
    n = kv.shape[0]
    full = kv.as_strided((_SHORT["kv_rows"], 576), kv.stride())
    past = idx.clone()
    past[:, 1::97] = n
    past[:, 2::97] = _SHORT["kv_rows"] - 1
    past[:, 3::97] = 2**31 - 1
    dropped = torch.where(past >= n, -1, past)
    got = op(q, kv, past, SCALE, torch.empty_like(out))
    want = op(q, kv, dropped, SCALE, torch.empty_like(out))
    assert bool(torch.isfinite(got).all())
    assert not (got.view(torch.int16) != want.view(torch.int16)).any()
    stale = torch.where(past == 2**31 - 1, -1, past)
    seen = op(q, full, stale, SCALE, torch.empty_like(out))
    assert bool(torch.isnan(seen).all(dim=-1).all()), "the stale rows were not read"


def test_gate_sees_a_short_workspace_and_wrong_offsets() -> None:
    """The gate's sight of the workspace defects next to this call: a view of only the new
    tokens' rows (the step's KV counted as new tokens, not cached + new) drops every
    selection past it without an error, and runs offset by the new tokens instead of the
    whole KV read the wrong rows. Both must land above 0.3, 30x the per-row band and 75x
    the whole-call one (on the 1M cells' builds both measured 0.95-2.2 whole-call)."""
    spec = dict(rows=64, kv_len=[4096, 6144], new=[32, 32], h_q=64, h_out=16, topk=2048,
                kv_rows=16384, selection="topk")  # fmt: skip
    (q, kv, idx, out), _ = _build_1m(spec, 13)
    ref = op.reference(q, kv, idx, SCALE, out)
    _assert_close(op(q, kv, idx, SCALE, torch.empty_like(out)), ref)
    short = op(q, kv[:64], idx, SCALE, torch.empty_like(out))
    per, whole = _errors_tp4(short, ref)
    assert per > 0.3 and whole > 0.3, (per, whole)
    wrong = torch.where(idx[32:] >= 0, idx[32:] - 4096 + 32, idx[32:])
    moved = op(q[32:], kv, wrong.contiguous(), SCALE, torch.empty_like(out[32:]))
    per, whole = _errors_tp4(moved, ref[32:])
    assert per > 0.3 and whole > 0.3, (per, whole)


def test_indices_at_or_past_n_are_skipped() -> None:
    """An index at or past kv.shape[0] (N, the buffer's last row, 2**31 - 1) answers bitwise as
    -1 does: the row is skipped and never read, so a workspace view cut short drops selections
    without an error or a NaN. Control: the same indices over the whole buffer, whose rows past
    N are NaN, read them."""
    spec = dict(rows=64, kv_len=[3000], new=[64], h_q=64, h_out=16, topk=2048, kv_rows=4096)
    (q, kv, idx, out), (_, buf, _) = _build_ws(spec, 7)
    n = kv.shape[0]
    past = idx.clone()
    past[:, 5::64] = n
    past[:, 6::64] = buf.shape[0] - 1
    past[:, 7::64] = 2**31 - 1
    skipped = torch.where(past >= n, -1, past).to(torch.int32)
    got = op(q, kv, past, SCALE, torch.empty_like(out))
    want = op(q, kv, skipped, SCALE, torch.empty_like(out))
    assert bool(torch.isfinite(got).all())
    assert bool((_bits(got) == _bits(want)).all()), "an index past N changed the answer"
    seen = op(q, buf, torch.where(past == 2**31 - 1, -1, past), SCALE, torch.empty_like(out))
    assert bool(torch.isnan(seen).all()), "the control did not read the NaN rows past N"


def test_gate_refuses_request_local_indices() -> None:
    """The gate sees an offset error: the dense three-request cell's selections passed without
    their requests' workspace offsets answer far outside the band."""
    cell = next(c for c in op.CELLS if c.spec.get("dense") and len(c.spec["kv_len"]) == 3)
    (q, kv, idx, out), _ = _build_ws(cell.spec, zlib.crc32(cell.why.encode()))
    _, base = _runs(cell.spec)
    local = torch.where(idx >= 0, idx - base[:, None], -1).to(torch.int32)
    wrong = op(q, kv, local, SCALE, torch.empty_like(out))
    per, whole = _errors(wrong, op.reference(q, kv, idx, SCALE, out))
    assert whole > 25 * 4e-3, (per, whole)


def test_sweep_sees_a_wrong_row_and_a_stray_write() -> None:
    """The row sweep's harness sees what it certifies: against a full answer one bit off in row
    40 every m past 40 fails and none up to it, and a call that also writes the row after its
    own fails the stray-write check."""
    spec = dict(rows=64, kv_len=[3000], new=[64], h_q=64, h_out=16, topk=2048, kv_rows=4096)
    (q, kv, idx, out), _ = _build_ws(spec, 11)
    op(q, kv, idx, SCALE, out)
    _sweep(q, kv, idx, out, range(1, 65))
    off = out.clone()
    off.view(torch.int16)[40, 3, 7] ^= 1
    with pytest.raises(AssertionError, match=r"^24 row counts .*: \[64, 63, 62"):
        _sweep(q, kv, idx, off, range(1, 65))

    def spill(q, kv, idx, scale, o):
        op(q, kv, idx, scale, o)
        after = o.storage_offset() + o.shape[0] * o.stride(0)
        torch.as_strided(o, (1, *o.shape[1:]), o.stride(), after).zero_()
        return o

    with pytest.raises(AssertionError, match="past its own"):
        _sweep(q, kv, idx, out, range(1, 65), call=spill)


def test_indices_at_or_past_n_are_skipped_at_64_heads() -> None:
    """An index at or past kv.shape[0] (N, the buffer's last row, 2**31 - 1) answers bitwise as
    -1 does: the row is skipped and never read, so a workspace view cut short drops selections
    without an error or a NaN. Control: the same indices over the whole buffer, whose rows past
    N are NaN, read them."""
    spec = dict(rows=64, kv_len=[3000], new=[64], h_q=64, h_out=64, topk=2048, kv_rows=4096)
    (q, kv, idx, out), (_, buf, _) = _build_ws(spec, 7)
    n = kv.shape[0]
    past = idx.clone()
    past[:, 5::64] = n
    past[:, 6::64] = buf.shape[0] - 1
    past[:, 7::64] = 2**31 - 1
    skipped = torch.where(past >= n, -1, past).to(torch.int32)
    got = op(q, kv, past, SCALE, torch.empty_like(out))
    want = op(q, kv, skipped, SCALE, torch.empty_like(out))
    assert bool(torch.isfinite(got).all())
    assert bool((_bits(got) == _bits(want)).all()), "an index past N changed the answer"
    seen = op(q, buf, torch.where(past == 2**31 - 1, -1, past), SCALE, torch.empty_like(out))
    assert bool(torch.isnan(seen).all()), "the control did not read the NaN rows past N"


def test_sweep_sees_a_wrong_row_and_a_stray_write_at_64_heads() -> None:
    """The row sweep's harness sees what it certifies, at 64 live heads: against a full answer
    one bit off in row 40 every m past 40 fails and none up to it, and a call that also writes
    the row after its own fails the stray-write check."""
    spec = dict(rows=64, kv_len=[3000], new=[64], h_q=64, h_out=64, topk=2048, kv_rows=4096)
    (q, kv, idx, out), _ = _build_ws(spec, 11)
    op(q, kv, idx, SCALE, out)
    _sweep(q, kv, idx, out, range(1, 65))
    off = out.clone()
    off.view(torch.int16)[40, 63, 511] ^= 1
    with pytest.raises(AssertionError, match=r"^24 row counts .*: \[64, 63, 62"):
        _sweep(q, kv, idx, off, range(1, 65))

    def spill(q, kv, idx, scale, o):
        op(q, kv, idx, scale, o)
        after = o.storage_offset() + o.shape[0] * o.stride(0)
        torch.as_strided(o, (1, *o.shape[1:]), o.stride(), after).zero_()
        return o

    with pytest.raises(AssertionError, match="past its own"):
        _sweep(q, kv, idx, out, range(1, 65), call=spill)


def test_indices_at_or_past_n_are_skipped_at_8_heads() -> None:
    """An index at or past kv.shape[0] (N, the buffer's last row, 2**31 - 1) answers bitwise as
    -1 does at the tep8 call's 8 live heads: the row is skipped and never read, so a workspace
    view cut short drops selections without an error or a NaN. Control: the same indices over
    the whole buffer, whose rows past N are NaN, read them."""
    spec = dict(rows=64, kv_len=[3000], new=[64], h_q=64, h_out=8, topk=2048, kv_rows=4096)
    (q, kv, idx, out), (_, buf, _) = _build_ws(spec, 7)
    n = kv.shape[0]
    past = idx.clone()
    past[:, 5::64] = n
    past[:, 6::64] = buf.shape[0] - 1
    past[:, 7::64] = 2**31 - 1
    skipped = torch.where(past >= n, -1, past).to(torch.int32)
    got = op(q, kv, past, SCALE, torch.empty_like(out))
    want = op(q, kv, skipped, SCALE, torch.empty_like(out))
    assert bool(torch.isfinite(got).all())
    assert bool((_bits(got) == _bits(want)).all()), "an index past N changed the answer"
    seen = op(q, buf, torch.where(past == 2**31 - 1, -1, past), SCALE, torch.empty_like(out))
    assert bool(torch.isnan(seen).all()), "the control did not read the NaN rows past N"


def test_sweep_sees_a_wrong_row_and_a_stray_write_at_8_heads() -> None:
    """The row sweep's harness sees what it certifies, at the tep8 call's 8 live heads: against
    a full answer one bit off in row 40 (the last live head's last element) every m past 40
    fails and none up to it, and a call that also writes the row after its own fails the
    stray-write check."""
    spec = dict(rows=64, kv_len=[3000], new=[64], h_q=64, h_out=8, topk=2048, kv_rows=4096)
    (q, kv, idx, out), _ = _build_ws(spec, 11)
    op(q, kv, idx, SCALE, out)
    _sweep(q, kv, idx, out, range(1, 65))
    off = out.clone()
    off.view(torch.int16)[40, 7, 511] ^= 1
    with pytest.raises(AssertionError, match=r"^24 row counts .*: \[64, 63, 62"):
        _sweep(q, kv, idx, off, range(1, 65))

    def spill(q, kv, idx, scale, o):
        op(q, kv, idx, scale, o)
        after = o.storage_offset() + o.shape[0] * o.stride(0)
        torch.as_strided(o, (1, *o.shape[1:]), o.stride(), after).zero_()
        return o

    with pytest.raises(AssertionError, match="past its own"):
        _sweep(q, kv, idx, out, range(1, 65), call=spill)
