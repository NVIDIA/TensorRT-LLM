# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cute_dsl_radix_topk_decode catalog entry, and its equivalence to the
trtllm indexer_topk_decode op it replaces."""

import struct
import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the CuTe DSL kernel is certified on sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_radix_topk_decode import (  # noqa: E402, E501
    TOP_K,
    blocks_per_row,
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_radix_topk_decode import (  # noqa: E402, E501
    cute_dsl_radix_topk_decode as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.indexer_topk_decode import (  # noqa: E402, E501
    indexer_topk_decode as old_op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

#: Written past each row's causal end. A kernel that read past the end would select these.
POISON = 1.0e9
#: What `indices` holds before a call; a slot the op should not write keeps it.
UNWRITTEN = -7
#: The CUDA-graph padding dummy's KV length (one token, no speculation).
PAD_SEQ_LEN = 1
#: key(-FLT_MAX), the op's padding entries' key.
PAD_KEY = 0xFF7FFFFF
#: Blocks per row the DSA metadata's radix scratch reserves for the trtllm op.
AUX_BLOCKS = 10


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


#: fp32 patterns the scorer does not write but the op still orders: +-NaN (two payloads),
#: +-inf, -FLT_MAX, +FLT_MAX, -0.0, +0.0 and both signs of a denormal.
SPECIAL = [
    0x7FC00000, 0x7F800001, 0xFFC00000, 0xFF800001, 0x7F800000, 0xFF800000,
    0xFF7FFFFF, 0x7F7FFFFF, 0x80000000, 0x00000000, 0x00000001, 0x80000001,
]  # fmt: skip


def _compiled(next_n: int = 1):
    return build(torch.cuda.current_device(), next_n=next_n)


def _end(seq_lens, next_n: int, row: int, width: int) -> int:
    return max(0, min(width, seq_lens[row // next_n] - next_n + row % next_n + 1))


def _scores(kind: str, row: int, end: int, g: torch.Generator) -> torch.Tensor:
    """One row's causal scores (see the cells' `scores`)."""
    if kind == "perm":
        return torch.randperm(end, generator=g).float()
    if kind == "indexer":
        # Head-weighted relu sums: about half the columns have every head at relu's floor.
        s = (torch.randn(8, end, generator=g) - 2.0).clamp_min_(0)
        w = torch.randn(8, 1, generator=g)
        return (s * w).sum(0)
    if kind == "ties":
        if row % 3 == 0:
            return torch.randint(0, 16, (end,), generator=g).float()
        zeros = torch.zeros(end)
        if row % 3 == 2:
            zeros[torch.rand(end, generator=g) < 0.5] = -0.0
        return zeros
    if kind == "wide":
        exponent = torch.randperm(end, generator=g).double() / max(end, 1) * 200.0 - 100.0
        sign = torch.where(torch.rand(end, generator=g) < 0.5, -1.0, 1.0).double()
        return (sign * torch.exp2(exponent)).float()
    if kind == "special":
        # A few finite scores; the rest special patterns, so the op's padding decides.
        vals = torch.randperm(end, generator=g).float() - end / 2
        pick = torch.randint(0, len(SPECIAL), (end,), generator=g)
        bits = torch.tensor(SPECIAL, dtype=torch.int64)[pick]
        special = (bits - (bits >= 2**31).long() * 2**32).to(torch.int32).view(torch.float32)
        frac = 0.7 if row % 2 == 0 else 0.95
        return torch.where(torch.rand(end, generator=g) < frac, special, vals)
    if kind == "negnan":
        # Distinct scores (even rows) or a floor of +0.0 under 1000 scores (odd rows), with
        # negative NaNs (two payloads) and a few positive ones sprinkled through.
        if row % 2 == 0:
            vals = torch.randperm(end, generator=g).float() - end / 3
        else:
            vals = torch.zeros(end)
            vals[torch.randperm(end, generator=g)[:1000]] = torch.rand(1000, generator=g) + 1.0
        nan_bits = torch.tensor([0xFFC00000, 0xFF800001, 0x7FC00000], dtype=torch.int64)
        pick = nan_bits[torch.randint(0, 3, (end,), generator=g)]
        nans = (pick - (pick >= 2**31).long() * 2**32).to(torch.int32).view(torch.float32)
        return torch.where(torch.rand(end, generator=g) < 0.03, nans, vals)
    if kind == "crowded":
        if row % 2 == 0:
            # 1000 positive scores over a floor of +0.0: the cut lands in a group of equals.
            vals = torch.zeros(end)
            vals[torch.randperm(end, generator=g)[:1000]] = torch.rand(1000, generator=g) + 1.0
            return vals
        # Distinct scores in [1.0, 1.25): one 11-bit top digit for all of them.
        return 1.0 + torch.randperm(end, generator=g).float() / (4.0 * max(end, 1))
    raise ValueError(kind)


def _logits(spec, seq_lens, g: torch.Generator) -> torch.Tensor:
    next_n, width = spec["next_n"], spec["width"]
    rows = len(seq_lens) * next_n
    parent = POISON + torch.arange(width, dtype=torch.float32).repeat(rows, 1)
    for r in range(rows):
        end = _end(seq_lens, next_n, r, width)
        parent[r, :end] = _scores(spec.get("scores", "perm"), r, end, g)
    return parent.cuda()


def _keys(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(bits >= 0x80000000, bits, (~bits) & 0x7FFFFFFF)


def _compare(out, ref, logits, seq_lens, next_n) -> None:
    """Equal up to exact ties: per row, the same multiset of keys over all top_k slots.

    A -1 counts as a pad, keyed like -FLT_MAX. Columns must be distinct and causal. A
    selection that differs only in which of several bit-identical scores (or a pad and a
    -FLT_MAX column) it took at the cut passes; any other difference changes the multiset.
    """
    assert out.dtype == ref.dtype == torch.int32 and out.shape == ref.shape
    width = logits.shape[1]
    keys = _keys(logits)
    for r in range(out.shape[0]):
        end = _end(seq_lens, next_n, r, width)
        sets = []
        for name, sel in (("op", out[r]), ("reference", ref[r])):
            cols = sel[sel >= 0].long()
            assert bool(((sel >= 0) | (sel == -1)).all()), f"{name} row {r}: unwritten slots"
            assert cols.unique().numel() == cols.numel(), f"{name} row {r}: repeated columns"
            assert bool((cols < end).all()), f"{name} row {r}: a column past the end {end}"
            pads = torch.full((int((sel == -1).sum()),), PAD_KEY, device=keys.device)
            k = torch.cat([keys[r, cols], pads])
            sets.append(k.sort().values)
        assert torch.equal(sets[0], sets[1]), f"row {r} (end {end}): a different selection"


def _old(logits, lens, indices, next_n):
    """The trtllm op, as the targets called it, with the DSA metadata's radix scratch."""
    rows = logits.shape[0]
    shape = (max(16, rows), AUX_BLOCKS, TOP_K)
    aux_i = torch.empty(shape, dtype=torch.int32, device="cuda")
    aux_l = torch.empty(shape, dtype=torch.float32, device="cuda")
    return old_op(logits, lens, indices, next_n, TOP_K, 1, aux_i, aux_l)


def _drive(spec, seq_lens, logits, kw) -> None:
    rows = logits.shape[0]
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    tail_of = spec.get("tail_of")
    if tail_of is None:
        buf = torch.full((rows, TOP_K), UNWRITTEN, dtype=torch.int32, device="cuda")
        indices = buf
    else:
        # A mixed step: the context rows' selection sits ahead of these rows in one buffer.
        buf = torch.full((tail_of, TOP_K), UNWRITTEN, dtype=torch.int32, device="cuda")
        indices = buf[tail_of - rows :]
    out = op(logits, lens, indices, **kw)
    assert out is indices, "the op writes the caller's buffer and returns it"
    _compare(out, op.reference(logits, lens, indices, **kw), logits, seq_lens, kw["next_n"])
    if tail_of is not None:
        assert bool((buf[: tail_of - rows] == UNWRITTEN).all()), "a row ahead was written"


def _drive_graph(spec, g, kw) -> None:
    """Capture once per batch and replay on new scores and seq_lens."""
    next_n, width = spec["next_n"], spec["width"]
    for batch in spec["graph_batches"]:
        capture_lens = spec["seq_lens"][:batch]
        logits = _logits(spec, capture_lens, g)
        lens = torch.tensor(capture_lens, dtype=torch.int32, device="cuda")
        indices = torch.full((batch * next_n, TOP_K), UNWRITTEN, dtype=torch.int32, device="cuda")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(logits, lens, indices, **kw)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(logits, lens, indices, **kw)
        for _ in range(spec["replays"]):
            new_lens = torch.randint(1, width + 1, (batch,), generator=g).tolist()
            if batch > 2:
                new_lens[-1] = PAD_SEQ_LEN
            new = _logits(spec, new_lens, g)
            logits.copy_(new)
            lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
            indices.fill_(UNWRITTEN)
            graph.replay()
            torch.cuda.synchronize()
            _compare(indices, op.reference(new, lens, indices, **kw), new, new_lens, next_n)


def _cell_inputs(cell):
    spec = cell.spec
    g = torch.Generator().manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    return spec, g


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    spec, g = _cell_inputs(cell)
    kw = dict(next_n=spec["next_n"], compiled=_compiled(spec["next_n"]))
    if "graph_batches" in spec:
        _drive_graph(spec, g, kw)
        return
    logits = _logits(spec, spec["seq_lens"], g)
    for batch in spec.get("batches", (len(spec["seq_lens"]),)):
        _drive(spec, spec["seq_lens"][:batch], logits[: batch * spec["next_n"]], kw)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_selects_the_trtllm_ops_set(cell) -> None:
    """Old against new on identical scores: the trtllm indexer_topk_decode op (the catalog's
    indexer_topk_decode, as the targets called it) and this entry select the same entries up
    to exact ties, -1 padding included, at every cell's shapes and scores."""
    spec, g = _cell_inputs(cell)
    next_n = spec["next_n"]
    kw = dict(next_n=next_n, compiled=_compiled(next_n))
    batches = spec.get("graph_batches", spec.get("batches", (len(spec["seq_lens"]),)))
    for batch in batches:
        seq_lens = spec["seq_lens"][:batch]
        logits = _logits(spec, seq_lens, g)
        lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        rows = logits.shape[0]
        new = op(logits, lens, torch.full((rows, TOP_K), UNWRITTEN, dtype=torch.int32,
                                          device="cuda"), **kw)  # fmt: skip
        old = _old(logits, lens, torch.full_like(new, UNWRITTEN), next_n)
        _compare(new, old, logits, seq_lens, next_n)


def test_short_rows_list_their_columns_in_order() -> None:
    """A row no longer than top_k lists its columns in order, then -1."""
    spec = dict(next_n=1, width=32768)
    logits = _logits(spec, [5, 40], torch.Generator().manual_seed(11))
    lens = torch.tensor([5, 40], dtype=torch.int32, device="cuda")
    out = torch.full((2, TOP_K), UNWRITTEN, dtype=torch.int32, device="cuda")
    op(logits, lens, out, next_n=1, compiled=_compiled())
    assert out[0].tolist() == list(range(5)) + [-1] * (TOP_K - 5)
    assert out[1].tolist() == list(range(40)) + [-1] * (TOP_K - 40)


def test_padding_outranks_minus_inf_as_in_the_op() -> None:
    """Where the op's pads exist, a column scored -inf or -NaN loses to them: a short row
    drops it, and a long row of 16 rows (9 blocks of 2366 columns, no pads) keeps it."""
    width = 32768
    logits = torch.full((16, width), POISON, device="cuda")
    logits[0, :100] = torch.arange(100, dtype=torch.float32, device="cuda")
    logits[0, 50] = float("-inf")
    logits[0, 60] = _f32(0xFFC00000)
    logits[1, :21300] = float("-inf")
    logits[1, 21300:21310] = 1.0
    lens = torch.tensor([100, 21300 + 10] + [1] * 14, dtype=torch.int32, device="cuda")
    assert blocks_per_row(16, width, torch.cuda.get_device_properties(0).multi_processor_count) == 9
    out = torch.full((16, TOP_K), UNWRITTEN, dtype=torch.int32, device="cuda")
    op(logits, lens, out, next_n=1, compiled=_compiled())
    assert set(out[0].tolist()) == set(range(100)) - {50, 60} | {-1}
    assert int((out[1] >= 0).sum()) == TOP_K
    old = _old(logits, lens, torch.full_like(out, UNWRITTEN), 1)
    _compare(out, old, logits, lens.tolist(), 1)


def test_refuses_a_shape_the_op_runs_as_one_block() -> None:
    """Under 4096 columns the op takes its single-block path, which this does not reproduce."""
    logits = torch.zeros((1, 2048), device="cuda")
    lens = torch.tensor([2048], dtype=torch.int32, device="cuda")
    out = torch.empty((1, TOP_K), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="one block per row"):
        op(logits, lens, out, next_n=1, compiled=_compiled())


def test_next_n_and_compiled_are_keyword_only() -> None:
    logits = torch.zeros((1, 4096), device="cuda")
    lens = torch.tensor([4096], dtype=torch.int32, device="cuda")
    out = torch.empty((1, TOP_K), dtype=torch.int32, device="cuda")
    with pytest.raises(TypeError):
        op(logits, lens, out, 1, _compiled())


def test_guard_refuses_a_causal_end_past_the_row() -> None:
    """The kernel would clamp it to the row and answer for the truncated row."""
    logits = torch.zeros((1, 32768), device="cuda")
    lens = torch.tensor([32769], dtype=torch.int32, device="cuda")
    out = torch.empty((1, TOP_K), dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="exceeds the 32768-column logits"):
        with validating(op):
            op(logits, lens, out, next_n=1, compiled=_compiled())
