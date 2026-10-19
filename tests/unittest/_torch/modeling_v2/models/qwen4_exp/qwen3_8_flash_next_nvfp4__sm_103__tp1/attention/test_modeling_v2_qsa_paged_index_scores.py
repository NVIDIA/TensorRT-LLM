# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_index_scores catalog entry.

The paged state is real: a `[pages, tokens_per_block, 1, dim]` index cache
sized from the pages the table maps, and an int32 block table whose rows are
request *slots* -- more of them than the batch has requests, assigned by a
seeded permutation -- with shuffled physical pages and `-1` past each
sequence on the serving layouts, which is what the engine's table carries.

The call has two kernels behind it. `_watch_launches` swaps the two Triton
entry points for recording proxies that still launch the real kernel, so every
cell asserts the branch it names.
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_index_scores import (  # noqa: E501
    qsa_paged_index_scores as op,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_index_scores requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """torch's default fp32 band on every specified column; `-inf` exactly.

    bf16 and fp16 products are exact in the kernel's fp32 accumulator, so
    only summation order separates it from the reference: measured at most
    2.9e-6 over every cell (2.4e-6 at 8191 rows x 2080 columns, scores to
    11.6), inside the default rtol 1.3e-6 / atol 1e-5. Wrong variants (no
    relu, mean over heads, first group member as key, unscaled,
    1/head_dim scale, block table ignored) land more than 1000x outside
    (asserted at 64 and 2080 columns; measured 3e5x to 4e6x). Columns the
    reference marks NaN are unspecified and not read.
    """
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    specified = ~torch.isnan(ref)
    torch.testing.assert_close(out[specified], ref[specified])


DEV = torch.device("cuda")
#: The callable's own threshold: at or below it every call is row-wise.
DECODE_ROW_LIMIT = 256
#: Index heads must divide this into more than one row for the query tile.
DOT_TILE_WIDTH = 16
CAPTURE_PAGES = 260


class _LaunchRecorder:
    """Transparent proxy over a Triton entry point: records the grid, then launches."""

    def __init__(self, inner, name, log):
        self._inner, self._name, self._log = inner, name, log

    def __getitem__(self, grid):
        self._log.append((self._name, tuple(grid)))
        return self._inner[grid]

    def __getattr__(self, item):
        return getattr(self._inner, item)


class _watch_launches:
    """Record which of the entry's two Triton kernels a call launches."""

    _NAMES = {
        "_qsa_paged_index_scores_kernel": "row_wise",
        "_qsa_paged_index_scores_qtile_kernel": "query_tile",
    }

    def __enter__(self):
        self.log = []
        self._saved = {name: getattr(_qsa_kernels, name) for name in self._NAMES}
        for name, tag in self._NAMES.items():
            setattr(_qsa_kernels, name, _LaunchRecorder(self._saved[name], tag, self.log))
        return self

    def __exit__(self, *exc):
        for name, kernel in self._saved.items():
            setattr(_qsa_kernels, name, kernel)
        return False

    @property
    def branch(self):
        assert len(self.log) == 1, f"expected exactly one kernel launch, saw {self.log}"
        return self.log[0][0]

    @property
    def grid(self):
        assert len(self.log) == 1, f"expected exactly one kernel launch, saw {self.log}"
        return self.log[0][1]


def _case(rows, requests, **overrides):
    """A narrow-table spec for the non-cell tests, same shape as the entry's own."""
    spec = dict(
        layout="random",
        rows=rows,
        requests=requests,
        slots=requests + 4,
        pages=8,
        heads=4,
        head_dim=128,
        tokens_per_block=32,
        compress_ratio=4,
        dtype=torch.bfloat16,
        positions_dtype=torch.int32,
        request_dtype=torch.int32,
        q_layout="packed",
        only_visible_blocks=False,
        context_rows=False,
    )
    spec.update(overrides)
    return spec


def _rows_and_requests(spec, gen):
    """Per-row positions, per-row request ordinal, and each request's mapped pages."""
    layout, tpb, pages = spec["layout"], spec["tokens_per_block"], spec["pages"]

    def pages_for(seq_len):
        return math.ceil(seq_len / tpb)

    if layout in ("random", "shuffled", "past_table", "arange"):
        rows, requests = spec["rows"], spec["requests"]
        if layout == "shuffled":
            req = torch.randint(0, requests, (rows,), generator=gen, device=DEV)
        else:
            req = torch.arange(rows, device=DEV) * requests // max(rows, 1)
        if layout == "past_table":
            positions = torch.full((rows,), pages * tpb * 3, device=DEV)
        elif layout == "arange":
            positions = torch.arange(rows, device=DEV)
        else:
            positions = torch.randint(0, pages * tpb, (rows,), generator=gen, device=DEV)
        return positions, req, [pages] * requests
    if layout == "far":
        rows = spec["rows"]
        positions = torch.full((rows,), pages * tpb - 1, device=DEV)
        return positions, torch.zeros(rows, dtype=torch.long, device=DEV), [pages]

    segments = []  # (first position, row count) per request, in row order
    if layout == "capture":
        segments = [(0, n) for n in spec["lengths"]]
    elif layout == "decode":
        segments = [(p, spec["rows_per_request"]) for p in spec["gen_positions"]]
    elif layout == "mixed":
        max_position = pages * tpb - 1
        contexts = spec["context_lens"]
        segments = [
            (0 if i == 0 else (907 * i) % (max_position - n + 2), n) for i, n in enumerate(contexts)
        ]
        # MTP: each generation request carries two rows at consecutive positions.
        segments += [(100 + (499 * g) % 8000, 2) for g in range(spec["requests"] - len(contexts))]
    else:
        raise AssertionError(f"unknown layout {layout!r}")
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    mapped = [pages_for(s + n) for s, n in segments]
    assert max(mapped) <= pages
    return positions, req, mapped


def _build(spec, seed):
    """Real paged state for one spec, keyed by `__call__`'s parameter names."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    heads, head_dim, dtype = spec["heads"], spec["head_dim"], spec["dtype"]
    positions, req, mapped = _rows_and_requests(spec, gen)
    rows, slots = positions.numel(), spec["slots"]
    assert len(mapped) <= slots

    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(mapped)]
    pool_pages = sum(mapped) + 3
    page_ids = torch.randperm(pool_pages, generator=gen, device=DEV).to(torch.int32)
    block_table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    index_cache = torch.randn(
        pool_pages, spec["tokens_per_block"], 1, head_dim, generator=gen, device=DEV
    ).to(dtype)

    if spec["q_layout"] == "scratch_slice":
        # [T, index heads | 1 k | 1 v, D], q is the first `heads` slots.
        scratch = torch.randn(rows, heads + 2, head_dim, generator=gen, device=DEV).to(dtype)
        q = scratch[:, :heads]
    elif spec["q_layout"] == "dim_slice":
        wide = torch.zeros(rows, heads, 2 * head_dim, dtype=dtype, device=DEV)
        wide[:, :, :head_dim] = torch.randn(rows, heads, head_dim, generator=gen, device=DEV)
        q = wide[:, :, :head_dim]
    else:
        q = torch.randn(rows, heads, head_dim, generator=gen, device=DEV).to(dtype)
    if spec["q_layout"] != "packed" and rows > 1:  # one row is trivially contiguous
        assert not q.is_contiguous()

    return dict(
        q=q,
        index_cache=index_cache,
        block_table=block_table,
        query_positions=positions.to(spec["positions_dtype"]),
        request_indices=slot_of.to(DEV)[req].to(spec["request_dtype"]),
        tokens_per_block=spec["tokens_per_block"],
        compress_ratio=spec["compress_ratio"],
        only_visible_blocks=spec["only_visible_blocks"],
        context_rows=spec["context_rows"],
    )


def _branch_rule(rows, spec):
    heads = spec["heads"]
    packs = DOT_TILE_WIDTH % heads == 0 and DOT_TILE_WIDTH // heads > 1
    tiled = spec["context_rows"] and rows > DECODE_ROW_LIMIT and packs
    return "query_tile" if tiled else "row_wise"


def _columns(args):
    return args["block_table"].shape[1] * args["tokens_per_block"] // args["compress_ratio"]


def _visible(args):
    blocks = torch.arange(_columns(args), device=DEV)
    bound = (args["query_positions"].long() + 1) // args["compress_ratio"]
    return blocks[None, :] < bound[:, None]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=sum(map(ord, cell.why)))
    rows = args["q"].shape[0]
    assert cell.spec["branch"] == _branch_rule(rows, cell.spec), "the cell names the wrong kernel"
    with _watch_launches() as watcher:
        out = op(**args)
        torch.cuda.synchronize()
    assert watcher.branch == cell.spec["branch"]
    assert out.shape == (rows, _columns(args)) and out.dtype == torch.float32
    _compare(out, op.reference(**args))


@pytest.mark.parametrize("context_rows,branch", [(False, "row_wise"), (True, "query_tile")])
def test_capture_width_needs_more_than_one_column_program(context_rows, branch) -> None:
    """Why the 260-page table is an axis of its own, not a bigger number.

    Both kernels walk the score columns in a grid dimension of their own. At 8
    pages (64 columns) one program covers every column; at 260 pages (2080) it
    does not, so only the wide cells ever run a later column program.
    """
    grids = {}
    for tag, pages in (("narrow", 8), ("wide", CAPTURE_PAGES)):
        args = _build(_case(300, 4, pages=pages, context_rows=context_rows), seed=8500 + pages)
        with _watch_launches() as watcher:
            out = op(**args)
            torch.cuda.synchronize()
        assert watcher.branch == branch
        grids[tag] = watcher.grid
        _compare(out, op.reference(**args))
    assert grids["narrow"][1] == 1, grids
    assert grids["wide"][1] > 1, grids


def _capture_args(seed, only_visible_blocks=False):
    spec = _case(
        0,
        1,
        layout="capture",
        lengths=(8191,),
        slots=8,
        pages=CAPTURE_PAGES,
        only_visible_blocks=only_visible_blocks,
        context_rows=True,
    )
    return _build(spec, seed)


@pytest.mark.parametrize("shape", ["512rows_narrow", "8191rows_capture"])
def test_the_two_kernels_agree(shape) -> None:
    """One entry, two kernels: no further apart than fp32 reordering.

    Measured 9.5e-7 apart (one fp32 ulp at the score's magnitude) at 320 rows
    and again at 8191, so the packed kernel does not drift with batch size.
    """
    if shape == "512rows_narrow":
        args = _build(_case(512, 6), seed=404)
    else:
        args = _capture_args(seed=8700)
    visible = _visible(args)
    row_wise = op(**{**args, "context_rows": False})
    query_tile = op(**{**args, "context_rows": True})
    torch.testing.assert_close(row_wise[visible], query_tile[visible])


@pytest.mark.parametrize("shape", ["300rows_row_wise", "300rows_query_tile", "8191rows_capture"])
def test_results_are_bitwise_reproducible(shape) -> None:
    """Three identical calls, bitwise identical, in the padded mode so the
    whole tensor is written."""
    if shape == "8191rows_capture":
        args = _capture_args(seed=8800)
    else:
        args = _build(_case(300, 5, context_rows=shape.endswith("tile")), seed=505)
    runs = [op(**args).clone() for _ in range(3)]
    torch.cuda.synchronize()
    assert torch.equal(runs[0], runs[1]) and torch.equal(runs[1], runs[2])


@pytest.mark.parametrize(
    "shape", ["64rows_row_wise", "300rows_row_wise", "300rows_query_tile", "8191rows_capture"]
)
def test_only_visible_blocks_leaves_the_tail_unspecified(shape) -> None:
    """`only_visible_blocks=True` writes the visible columns and nothing else.

    The output is a fresh `torch.empty`, and the caching allocator hands back
    the block freed just before: poisoning one of the right size makes the
    unwritten tail observable. Asserted in the allocator-independent form: the
    tail is NOT the `-inf` the padded mode writes.
    """
    if shape == "8191rows_capture":
        args = _capture_args(seed=8900, only_visible_blocks=True)
    else:
        rows = int(shape.split("rows")[0])
        spec = _case(rows, 4, only_visible_blocks=True, context_rows=shape.endswith("tile"))
        args = _build(spec, seed=606 + rows)
    expected = op.reference(**args)
    poison = torch.full(expected.shape, 98765.0, dtype=torch.float32, device=DEV)
    del poison
    out = op(**args)
    torch.cuda.synchronize()
    _compare(out, expected)
    tail = torch.isnan(expected)
    assert bool(tail.any()), "this case must have unscored columns to say anything"
    assert not bool(torch.isneginf(out[tail]).all()), "the tail must not be the -inf padding"


def test_both_modes_agree_on_the_visible_columns() -> None:
    """The flag decides what lands outside the causal bound, nothing inside."""
    args = _build(_case(96, 4), seed=707)
    padded = op(**{**args, "only_visible_blocks": False})
    unspecified = op(**{**args, "only_visible_blocks": True})
    visible = _visible(args)
    assert torch.equal(padded[visible], unspecified[visible])


def test_capture_shape_survives_cuda_graph_capture_and_replay() -> None:
    """Capture is why 8191 rows exist, so capture it -- and replay it again
    after overwriting the operands in place, which a graph that baked in its
    inputs would fail."""
    args = _capture_args(seed=9100, only_visible_blocks=True)
    op(**args)  # warm the Triton launch outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = op(**args)
        graph.replay()
        torch.cuda.synchronize()
        _compare(captured, op.reference(**args))

        fresh = _capture_args(seed=9101, only_visible_blocks=True)
        for name in ("q", "index_cache"):
            args[name].copy_(fresh[name])
        # A new logical->physical map for the same slot: the pages it maps,
        # reshuffled in place.
        table = args["block_table"]
        mapped = table >= 0
        gen = torch.Generator(device=DEV).manual_seed(9102)
        pages = table[mapped]
        table[mapped] = pages[torch.randperm(pages.numel(), generator=gen, device=DEV)]
        graph.replay()
        torch.cuda.synchronize()
        _compare(captured, op.reference(**args))
    finally:
        del graph
        torch.cuda.synchronize()


def _wrong_variants(args):
    """Plausible mis-implementations in plain torch; none of them is the op."""
    q, index_cache, table = args["q"], args["index_cache"], args["block_table"].long()
    requests = args["request_indices"].long()
    tpb, ratio = args["tokens_per_block"], args["compress_ratio"]
    rows, _, head_dim = q.shape
    columns = _columns(args)

    def scores(*, offset=ratio - 1, relu=True, scale=head_dim**-0.5, mean=False, no_table=False):
        anchors = torch.arange(columns, device=DEV) * ratio + offset
        logical, in_page = anchors // tpb, anchors % tpb
        physical = logical.expand(rows, columns) if no_table else table[requests][:, logical]
        keys = index_cache[physical.clamp_min(0), in_page.expand(rows, columns)][:, :, 0, :]
        per_head = torch.einsum("rhd,rbd->rbh", q.float(), keys.float())
        per_head = torch.relu(per_head) if relu else per_head
        return (per_head.mean(-1) if mean else per_head.sum(-1)) * scale

    return {
        "no_relu": scores(relu=False),
        "mean_over_heads": scores(mean=True),
        "key_from_the_first_group_member": scores(offset=0),
        "unscaled": scores(scale=1.0),
        "scaled_by_one_over_head_dim": scores(scale=1.0 / head_dim),
        "block_table_ignored": scores(no_table=True),
    }


@pytest.mark.parametrize("width", ["8pages", "260pages"])
def test_wrong_variants_fail_the_gate(width) -> None:
    """The default fp32 band is only a gate if a wrong answer fails it.

    Each variant is rejected by `_compare` and lands more than 1000x outside the
    band the correct result clears; measured 3e5x to 4e6x.
    """
    if width == "8pages":
        args = _build(_case(64, 4), seed=1601)
    else:
        args = _build(_case(8, 8, pages=CAPTURE_PAGES), seed=9000)
        gen = torch.Generator(device=DEV).manual_seed(9001)
        args["query_positions"] = torch.randint(4096, 8192, (8,), generator=gen, device=DEV).to(
            torch.int32
        )
    out = op(**args)
    expected = op.reference(**args)
    _compare(out, expected)
    visible = _visible(args)
    gate = 1e-5 + 1.3e-6 * expected[visible].abs().max()
    assert (out[visible] - expected[visible]).abs().max() < gate
    for name, variant in _wrong_variants(args).items():
        wrong = expected.clone()
        wrong[visible] = variant[visible]
        assert (wrong[visible] - out[visible]).abs().max() > 1000 * gate, name
        with pytest.raises(AssertionError):
            _compare(out, wrong)


def test_rows_before_a_complete_group_see_nothing() -> None:
    """Positions 0..7 at compress_ratio 4: 0,0,0,1,1,1,1,2 visible columns.

    A row that has not closed its first group comes back entirely `-inf`; the
    caller's top-k has to know that row can happen.
    """
    args = _build(_case(8, 2, layout="arange"), seed=1251)
    out = op(**args)
    _compare(out, op.reference(**args))
    assert torch.isfinite(out).sum(dim=1).tolist() == [0, 0, 0, 1, 1, 1, 1, 2]


def test_empty_row_batch() -> None:
    """Zero rows: the correctly shaped empty fp32 tensor, and no launch."""
    args = _build(_case(0, 2), seed=1301)
    with _watch_launches() as watcher:
        out = op(**args)
    assert watcher.log == [], "an empty batch must not launch a kernel"
    assert out.shape == (0, _columns(args)) and out.dtype == torch.float32


def test_inputs_are_not_written() -> None:
    """Nothing is in place, and the result is a fresh allocation."""
    args = _build(_case(64, 4), seed=1401)
    snapshots = {k: v.clone() for k, v in args.items() if torch.is_tensor(v)}
    out = op(**args)
    torch.cuda.synchronize()
    for name, snapshot in snapshots.items():
        assert torch.equal(args[name], snapshot), f"{name} was modified"
    inputs = {args[k].untyped_storage().data_ptr() for k in snapshots}
    assert out.untyped_storage().data_ptr() not in inputs


def test_float32_operands_use_a_reduced_precision_dot() -> None:
    """fp32 q and cache are accepted and do NOT hold `_compare`'s band.

    `tl.dot` rounds fp32 operands to a tensor-core format. Measured: 8.1e-3
    max abs, 0.29 max relative, on scores to 7.8 -- where bf16 operands give
    9.5e-7. That is why no cell covers fp32. The loosened band below
    (rtol 2e-3, atol 2e-2) is ~2.5x the measured error, and every wrong
    variant still lands more than 25x outside it.
    """
    args = _build(_case(64, 4, dtype=torch.float32), seed=1501)
    out = op(**args)
    expected = op.reference(**args)
    visible = _visible(args)
    with pytest.raises(AssertionError):
        _compare(out, expected)
    torch.testing.assert_close(out[visible], expected[visible], rtol=2e-3, atol=2e-2)
    gate = (2e-2 + 2e-3 * expected[visible].abs()).max()
    for name, variant in _wrong_variants(args).items():
        assert (variant[visible] - out[visible]).abs().max() > 25 * gate, name


def _mutated(args, **overrides):
    return {**args, **overrides}


_REJECTIONS = {
    "two_dimensional_q": (
        lambda a: _mutated(a, q=a["q"][:, 0]),
        "QSA index queries must have shape",
    ),
    "three_dimensional_cache": (
        lambda a: _mutated(a, index_cache=a["index_cache"][:, :, 0]),
        "QSA index cache must have shape",
    ),
    "two_kv_head_cache": (
        lambda a: _mutated(a, index_cache=a["index_cache"].expand(-1, -1, 2, -1)),
        "QSA index cache must have shape",
    ),
    "int64_block_table": (
        lambda a: _mutated(a, block_table=a["block_table"].long()),
        "two-dimensional int32 block table",
    ),
    "one_dimensional_block_table": (
        lambda a: _mutated(a, block_table=a["block_table"][0]),
        "two-dimensional int32 block table",
    ),
    "empty_block_table": (
        lambda a: _mutated(a, block_table=a["block_table"][:, :0].contiguous()),
        "requires a non-empty block table",
    ),
    "mismatched_dtypes": (
        lambda a: _mutated(a, index_cache=a["index_cache"].float()),
        "dtypes must match",
    ),
    "mismatched_head_dims": (
        lambda a: _mutated(a, index_cache=a["index_cache"][..., :64].contiguous()),
        "dimensions do not match",
    ),
    "compress_ratio_1": (lambda a: _mutated(a, compress_ratio=1), "compression ratio >= 2"),
    "tokens_per_block_0": (lambda a: _mutated(a, tokens_per_block=0), "compression ratio >= 2"),
    "float_metadata": (
        lambda a: _mutated(a, query_positions=a["query_positions"].float()),
        "must use integer storage",
    ),
    "short_metadata": (
        lambda a: _mutated(a, query_positions=a["query_positions"][:4]),
        "one entry per row",
    ),
    "host_q": (lambda a: _mutated(a, q=a["q"].cpu()), "requires CUDA tensors"),
}


@pytest.mark.parametrize("case", list(_REJECTIONS), ids=list(_REJECTIONS))
def test_rejects(case) -> None:
    """The callable's own refusals, messages as observed -- the reason this
    entry has no `is_valid`."""
    mutate, message = _REJECTIONS[case]
    args = mutate(_build(_case(16, 2), seed=1701))
    with pytest.raises(ValueError, match=message):
        op(**args)
