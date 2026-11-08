# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp qsa_paged_index_scores entry.

The paged state is real: a `[pages, tokens_per_block, 1, dim]` index cache
(a view of a wider page under the NVFP4 cells) sized from the pages the table maps, and an
int32 block table whose rows are request *slots* -- more of them than the batch has requests, assigned by a
seeded permutation -- with shuffled physical pages and `-1` past each
sequence on the serving layouts, which is what the engine's table carries.

The four-rank layouts' cells add one rank's measured pool: every attention layer's
plane in one page (the last layer's plane taken), NaN wherever no request maps a
page, with the pool's first and last page among those the cell reads -- the last
one past 2^31 elements from the plane's base.

The call has four kernels behind it: the entry's own visible-split kernel for
ungrouped decode-sized visible-only calls and request-group kernel for verify-shaped
ones, and the engine's row-wise and query-tile kernels. `_watch_launches` swaps the four Triton entry points
for recording proxies that still launch the real kernel, so every cell asserts
the branch it names. Beyond the cells: capture and replay at every measured-pool decode
and verify cell, and the refusal of metadata shorter than the rows by both launchers.
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention import (
    qsa_paged_index_scores as _entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_index_scores import (  # noqa: E501
    qsa_paged_index_scores as op,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

__extra_import_path__ = ["../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_index_scores requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Torch's default fp32 band on every specified column; `-inf` exactly.

    bf16 and fp16 products are exact in the kernel's fp32 accumulator, so
    only summation order separates it from the reference, well inside the
    default rtol 1.3e-6 / atol 1e-5 on every cell, the measured-pool ones
    included. Wrong variants (no relu, mean over heads, first group member as
    key, unscaled, 1/head_dim scale, block table ignored) land orders of
    magnitude outside it. Columns the reference marks NaN are unspecified and
    not read.
    """
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    specified = ~torch.isnan(ref)
    torch.testing.assert_close(out[specified], ref[specified])


DEV = torch.device("cuda")
#: The largest batch the visible-split kernel serves.
DECODE_ROW_LIMIT = 256
#: Index heads must divide this into more than one row for the query tile.
DOT_TILE_WIDTH = 128
#: Index heads must divide this for the visible-split kernel.
SPLIT_DOT_WIDTH = 16
#: A table width with columns enough for several of the request-group kernel's tiles.
_WIDTH_PAGES = 260


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
    """Record which of the entry's four Triton kernels a call launches."""

    _NAMES = (
        (_qsa_kernels, "_qsa_paged_index_scores_kernel", "row_wise"),
        (_qsa_kernels, "_qsa_paged_index_scores_qtile_kernel", "query_tile"),
        (_entry, "_qsa_index_scores_group_kernel", "request_group"),
        (_entry, "_qsa_index_scores_visible_split_kernel", "visible_split"),
    )

    def __enter__(self):
        self.log = []
        self._saved = [(module, name, getattr(module, name)) for module, name, _ in self._NAMES]
        for (module, name, tag), (_, _, kernel) in zip(self._NAMES, self._saved):
            setattr(module, name, _LaunchRecorder(kernel, tag, self.log))
        return self

    def __exit__(self, *exc):
        for module, name, kernel in self._saved:
            setattr(module, name, kernel)
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
    """A narrow-table spec for the non-cell tests, same geometry as the entry's own."""
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

    if layout == "random":
        rows, requests = spec["rows"], spec["requests"]
        req = torch.arange(rows, device=DEV) * requests // max(rows, 1)
        positions = torch.randint(0, pages * tpb, (rows,), generator=gen, device=DEV)
        return positions, req, [pages] * requests

    # (first position, row count) per request, in row order
    if layout == "capture":
        segments = [(0, n) for n in spec["lengths"]]
    elif layout == "decode":
        segments = [(p, spec["rows_per_request"]) for p in spec["gen_positions"]]
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
    total = sum(mapped)
    pool_pages = spec.get("pool_pages", total + 3)
    page_ids = torch.randperm(pool_pages, generator=gen, device=DEV)
    if "pool_pages" in spec:
        # A measured pool: its first and last page go to the first request's first two
        # logical pages, so the cell reads both ends.
        ends = torch.tensor([pool_pages - 1, 0], device=DEV)
        page_ids = torch.cat([ends, page_ids[(page_ids != 0) & (page_ids != pool_pages - 1)]])
    page_ids = page_ids[:total].to(torch.int32)
    block_table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    tpb = spec["tokens_per_block"]
    if "page_pad" in spec:
        # The trunk pool's page: every attention layer's index buffer (the fp8 and NVFP4
        # pools then the position page), so one layer's plane is a view at page stride
        # buffers * tpb * dim + page_pad -- the last layer's here. Pages no request maps
        # hold NaN: a read of one shows up in the scores.
        buffers, buffer = spec["pool_buffers"], tpb * head_dim
        page = torch.full(
            (pool_pages, buffers * buffer + spec["page_pad"]), float("nan"), dtype=dtype, device=DEV
        )
        index_cache = page[:, (buffers - 1) * buffer : buffers * buffer].view(
            pool_pages, tpb, 1, head_dim
        )
        index_cache[page_ids.long()] = torch.randn(
            total, tpb, 1, head_dim, generator=gen, device=DEV
        ).to(dtype)
        assert index_cache.stride() == (page.stride(0), head_dim, head_dim, 1)
    else:
        index_cache = torch.randn(pool_pages, tpb, 1, head_dim, generator=gen, device=DEV).to(dtype)

    if spec["q_layout"] == "scratch_slice":
        # [T, index heads | 1 k | 1 v, D], q is the first `heads` slots.
        scratch = torch.randn(rows, heads + 2, head_dim, generator=gen, device=DEV).to(dtype)
        q = scratch[:, :heads]
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
        use_pdl=spec.get("use_pdl", False),
        prefetch=spec.get("prefetch", False),
        rows_per_request=spec.get("call_rows_per_request", 1),
    )


def test_visible_split_metadata_offsets_reuse_the_compiled_kernel() -> None:
    """Mixed-step metadata offsets preserve results without new alignment variants."""
    cell = next(c for c in op.CELLS if c.spec["branch"] == "visible_split")
    args = _build(cell.spec, seed=81)
    rows = args["q"].shape[0]
    position_storage = torch.zeros(rows + 4, dtype=args["query_positions"].dtype, device=DEV)
    request_storage = torch.zeros(rows + 4, dtype=args["request_indices"].dtype, device=DEV)
    compiled = []
    for position_offset, request_offset in ((0, 0), (0, 1), (1, 1), (1, 3)):
        positions = position_storage[position_offset : position_offset + rows]
        requests = request_storage[request_offset : request_offset + rows]
        positions.copy_(args["query_positions"])
        requests.copy_(args["request_indices"])
        shifted = {**args, "query_positions": positions, "request_indices": requests}
        with _watch_launches() as watcher:
            got = op(**shifted)
            torch.cuda.synchronize()
        assert watcher.branch == "visible_split"
        _compare(got, op.reference(**shifted))
        cache = _entry._qsa_index_scores_visible_split_kernel.device_caches[0][0]
        compiled.append(len(cache))
    assert len(set(compiled)) == 1, compiled


def _assert_reaches_the_pool_ends(args, spec) -> None:
    """A measured-pool cell scores keys on the pool's first and last page, and the last page
    lies past 2^31 elements from the plane's base, so a 32-bit offset would wrap.
    """
    cache, pool_pages = args["index_cache"], spec["pool_pages"]
    tpb, dim = spec["tokens_per_block"], spec["head_dim"]
    page_stride = spec["pool_buffers"] * tpb * dim + spec["page_pad"]
    assert cache.shape[0] == pool_pages and cache.stride() == (page_stride, dim, dim, 1)
    assert (pool_pages - 1) * page_stride >= 2**31
    cr = args["compress_ratio"]
    logical = (torch.arange(_columns(args), device=DEV) * cr + cr - 1) // tpb
    pages = args["block_table"][args["request_indices"].long()][:, logical]
    read = pages[_visible(args)]
    assert bool((read == 0).any()) and bool((read == pool_pages - 1).any()), "a pool end is unread"


def _branch_rule(rows, spec):
    heads = spec["heads"]
    if (
        spec.get("call_rows_per_request", 1) == 1
        and spec["only_visible_blocks"]
        and not spec.get("prefetch", False)
        and rows <= DECODE_ROW_LIMIT
        and SPLIT_DOT_WIDTH % heads == 0
    ):
        return "visible_split"
    packs = DOT_TILE_WIDTH % heads == 0 and DOT_TILE_WIDTH // heads > 1
    if spec["context_rows"] and rows > DECODE_ROW_LIMIT and packs:
        return "query_tile"
    grouped = (
        spec.get("call_rows_per_request", 1) > 1
        and spec["only_visible_blocks"]
        and not spec.get("prefetch", False)
        and heads in (2, 4, 8)
    )
    return "request_group" if grouped else "row_wise"


def _columns(args):
    return args["block_table"].shape[1] * args["tokens_per_block"] // args["compress_ratio"]


def _engine_scores(**args):
    """The engine's own launcher on the entry's arguments; `rows_per_request` is the entry's."""
    args.pop("rows_per_request", None)
    return _qsa_kernels.triton_qsa_paged_index_scores(**args)


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
    if "pool_pages" in cell.spec:
        _assert_reaches_the_pool_ends(args, cell.spec)
    with _watch_launches() as watcher:
        out = op(**args)
        torch.cuda.synchronize()
    assert watcher.branch == cell.spec["branch"]
    assert out.shape == (rows, _columns(args)) and out.dtype == torch.float32
    _compare(out, op.reference(**args))


#: The decode and verify cells over a measured pool: the sites the step's CUDA graph captures.
_CAPTURED = tuple(c for c in op.CELLS if "pool_pages" in c.spec and not c.spec["context_rows"])


@pytest.mark.parametrize("cell", _CAPTURED, ids=[c.why[:44] for c in _CAPTURED])
def test_captured_call_replays_like_eager(cell) -> None:
    """Capture the call at the cell's batch and flags over the measured pool, rewrite q and
    every mapped key in place, replay, and match an eager call bitwise on every specified
    column.
    """
    args = _build(cell.spec, seed=sum(map(ord, cell.why)) + 1)
    visible = _visible(args)
    table = args["block_table"]
    mapped = torch.unique(table[table >= 0]).long()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(**args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(**args)
    gen = torch.Generator(device=DEV).manual_seed(7)
    q, cache = args["q"], args["index_cache"]
    for trial in range(2):
        q.copy_(torch.randn(q.shape, generator=gen, device=DEV).to(q.dtype))
        cache[mapped] = torch.randn(
            (mapped.numel(), *cache.shape[1:]), generator=gen, device=DEV
        ).to(cache.dtype)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured[visible], op(**args)[visible]), f"replay {trial}"


#: The visible-split cells on the tune's table: their grid is fixed at capture, the windows not.
_SPLIT_CAPTURED = tuple(
    c for c in _CAPTURED if c.spec["branch"] == "visible_split" and c.spec["pages"] > 516
)


@pytest.mark.parametrize("cell", _SPLIT_CAPTURED, ids=[c.why[:44] for c in _SPLIT_CAPTURED])
def test_captured_visible_split_follows_the_positions(cell) -> None:
    """The visible-split grid is sized by the device, and each program reads its rows'
    positions when it runs: capture at the cell's positions, then replay with every row's
    window shrunk by a different amount (to empty for some) and grown back, and match the
    reference on every specified column each time.
    """
    args = _build(cell.spec, seed=sum(map(ord, cell.why)) + 2)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(**args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(**args)
    positions = args["query_positions"]
    original = positions.clone()
    gen = torch.Generator(device=DEV).manual_seed(5)
    for trial in range(3):
        if trial < 2:
            keep = torch.rand(original.shape, generator=gen, device=DEV) ** (trial + 1)
            positions.copy_((original.double() * keep).to(original.dtype))
        else:
            positions.copy_(original)
        graph.replay()
        torch.cuda.synchronize()
        _compare(captured, op.reference(**args))


def test_prefetch_reads_the_fresh_anchor_key_and_q_after_the_wait() -> None:
    """`prefetch` reads pages and keys before the PDL wait. Under graph replay a producer
    rewrites q and, at rows whose position ends a group, the last visible block's anchor
    key -- what the decode prologue writes -- right before the op; both must be seen.
    The other keys are static, as they are in the decode step. Without prefetch the
    entry takes its visible-split kernel, so the plain row-wise launch it is held to
    bitwise is the engine callable's.
    """
    args = _build(_case(4, 4, only_visible_blocks=True), seed=411)
    cr, tpb = args["compress_ratio"], args["tokens_per_block"]
    pos = args["query_positions"].clone()
    assert bool((pos >= cr).all())
    ends = torch.tensor([True, False, True, True], device=DEV)
    args["query_positions"] = torch.where(ends, pos - (pos + 1) % cr, pos).to(pos.dtype)
    pos = args["query_positions"].long()
    assert bool(((pos + 1) % cr == 0).eq(ends).all())
    pages = args["block_table"][args["request_indices"].long(), pos // tpb].long()
    anchors = (pages[ends], (pos % tpb)[ends])
    visible = _visible(args)
    plain = _engine_scores(**args)
    assert torch.equal(op(**{**args, "use_pdl": True, "prefetch": True})[visible], plain[visible])
    src_q = args["q"].clone()
    src_k = args["index_cache"][anchors].clone()

    def step():
        args["q"].copy_(src_q * 1.0)
        args["index_cache"][anchors] = src_k * 1.0
        return op(**{**args, "use_pdl": True, "prefetch": True})

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    for trial in range(3):
        fresh = _build(_case(4, 4, only_visible_blocks=True), seed=412 + trial)
        src_q.copy_(fresh["q"])
        src_k.copy_(torch.randn(src_k.shape, device=DEV).to(src_k.dtype))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_out[visible], _engine_scores(**args)[visible]), f"replay {trial}"
        _compare(graph_out, op.reference(**args))


@pytest.mark.parametrize(
    "group, rows, heads, mixed, use_pdl",
    [
        (2, 16, 4, False, False),
        (3, 13, 4, False, False),
        (4, 64, 4, False, True),
        (4, 62, 4, False, False),
        (4, 64, 4, True, False),
        (8, 40, 4, False, False),
        (8, 36, 4, True, True),
        (4, 16, 2, False, False),
        (4, 16, 8, False, False),
        (2, 16, 16, False, False),  # 16 heads stay row-wise
        (4, 256, 4, False, True),
    ],
)
def test_request_group_is_bitwise_the_row_wise_kernel(group, rows, heads, mixed, use_pdl) -> None:
    """`rows_per_request` picks the request-group kernel; its scores are the row-wise
    kernel's bit patterns on every visible column. Runs of `group` rows of one request
    (a partial last run when `group` does not divide the rows), or with `mixed` a
    random request a row, so runs that name several requests gather row by row.
    """
    spec = _case(rows, 8, heads=heads, pages=_WIDTH_PAGES, only_visible_blocks=True)
    spec.update(q_layout="scratch_slice", use_pdl=use_pdl)
    args = _build(spec, seed=97 * group + rows + heads)
    slots = torch.unique(args["request_indices"])
    gen = torch.Generator(device=DEV).manual_seed(rows)
    if mixed:
        pick = torch.randint(0, slots.numel(), (rows,), generator=gen, device=DEV)
    else:
        pick = (torch.arange(rows, device=DEV) // group) % slots.numel()
    args["request_indices"] = slots[pick].to(args["request_indices"].dtype)
    # Every row sees at least one block (every request maps the whole table).
    args["query_positions"] = args["query_positions"].clamp_min(args["compress_ratio"] - 1)
    visible = _visible(args)
    with _watch_launches() as watcher:
        row_wise = _engine_scores(**args)
    assert watcher.branch == "row_wise"
    with _watch_launches() as watcher:
        grouped = op(**{**args, "rows_per_request": group})
    assert watcher.branch == ("request_group" if heads in (2, 4, 8) else "row_wise")
    assert torch.equal(grouped[visible].view(torch.int32), row_wise[visible].view(torch.int32)), (
        "the request-group kernel's bits differ from the row-wise kernel's"
    )
    _compare(grouped, op.reference(**args))


@pytest.mark.parametrize("only_visible_blocks", [False, True], ids=["engine", "visible_split"])
def test_rejects_metadata_shorter_than_the_rows(only_visible_blocks) -> None:
    """The callable's own refusal, message as observed -- the reason this entry has no
    `is_valid`: positions for fewer rows than `q` carries. Both kernels' launchers refuse.
    """
    args = _build(_case(16, 2, only_visible_blocks=only_visible_blocks), seed=1701)
    args["query_positions"] = args["query_positions"][:4]
    with pytest.raises(ValueError, match="one entry per row"):
        op(**args)
