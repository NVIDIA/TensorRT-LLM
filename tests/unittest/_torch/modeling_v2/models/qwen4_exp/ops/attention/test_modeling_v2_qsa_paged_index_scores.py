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

The call has five kernels behind it: the entry's own visible-split kernel for
ungrouped decode-sized visible-only calls and request-group kernel (and its prefetching form)
for verify-shaped ones, and the engine's row-wise and query-tile kernels. `_watch_launches`
swaps the five Triton entry points
for recording proxies that still launch the real kernel, so every cell asserts
the branch it names. Beyond the cells: capture and replay at every measured-pool decode
and verify cell, and the refusal of metadata shorter than the rows by both launchers.
"""

import math

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention import (
    qsa_paged_index_scores as _entry,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_index_scores import (  # noqa: E501
    qsa_paged_index_scores as op,
)

__extra_import_path__ = ["../../../.."]
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
        (_entry, "_qsa_index_scores_group_prefetch_kernel", "request_group_prefetch"),
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
        spec.get("call_rows_per_request", 1) > 1
        and spec["only_visible_blocks"]
        and spec.get("prefetch", False)
        and spec.get("use_pdl", False)
        and heads in (2, 4, 8)
        and not (spec["context_rows"] and rows > DECODE_ROW_LIMIT)
    ):
        return "request_group_prefetch"
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


@pytest.mark.parametrize("requests, mixed", [(1, False), (2, False), (4, True)])
def test_request_group_prefetch_is_bitwise_and_reads_fresh_keys(requests, mixed) -> None:
    """`prefetch` with `rows_per_request` 4 and PDL: the request-group kernel with each run's
    pages and keys read before the wait. Under graph replay a producer rewrites q and, at every
    row's own position and the anchor of a block ending at one, the index key -- what the
    attention decode prologue writes -- right before the op; the scores must be the plain
    request-group kernel's bits on every visible column. With `mixed` the runs name several
    requests (gathered after the wait). The producer here is a plain launch, complete before
    the op starts: this pins the bits and the capture, not the re-read after the wait
    (`test_request_group_prefetch_rereads_what_a_late_producer_stores` does)."""
    rows = 4 * requests
    spec = _case(rows, 8, pages=_WIDTH_PAGES, only_visible_blocks=True)
    spec.update(q_layout="scratch_slice", use_pdl=True)
    args = _build(spec, seed=503 + rows + mixed)
    slots = torch.unique(args["request_indices"])
    gen = torch.Generator(device=DEV).manual_seed(rows)
    if mixed:
        pick = torch.randint(0, slots.numel(), (rows,), generator=gen, device=DEV)
    else:
        pick = (torch.arange(rows, device=DEV) // 4) % slots.numel()
    args["request_indices"] = slots[pick].to(args["request_indices"].dtype)
    cr, tpb = args["compress_ratio"], args["tokens_per_block"]
    pos = args["query_positions"].long().clamp_min(cr - 1)
    if not mixed:  # a verify run: four consecutive positions of one request
        pos = (pos[::4].repeat_interleave(4) // 4) * 4 + torch.arange(rows, device=DEV) % 4
        pos = pos.clamp_min(cr - 1)
    args["query_positions"] = pos.to(args["query_positions"].dtype)
    group = {**args, "rows_per_request": 4}
    pages = args["block_table"][args["request_indices"].long(), pos // tpb].long()
    written = (pages, pos % tpb)  # each row's own index slot (the store; anchors among them)
    visible = _visible(args)
    with _watch_launches() as watcher:
        plain = op(**group)
    assert watcher.branch == "request_group"
    with _watch_launches() as watcher:
        early = op(**{**group, "prefetch": True})
    assert watcher.branch == "request_group_prefetch"
    assert torch.equal(early[visible].view(torch.int32), plain[visible].view(torch.int32))
    src_q = args["q"].clone()
    src_k = args["index_cache"][written].clone()

    def step():
        args["q"].copy_(src_q * 1.0)
        args["index_cache"][written] = src_k * 1.0
        return op(**{**group, "prefetch": True})

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    for trial in range(3):
        src_q.copy_(torch.randn(src_q.shape, generator=gen, device=DEV).to(src_q.dtype))
        src_k.copy_(torch.randn(src_k.shape, generator=gen, device=DEV).to(src_k.dtype))
        graph.replay()
        torch.cuda.synchronize()
        want = op(**group)
        assert torch.equal(graph_out[visible].view(torch.int32), want[visible].view(torch.int32)), (
            f"replay {trial}"
        )
        _compare(graph_out, op.reference(**args))


@triton.jit
def _late_slot_store(
    src, index_cache, pages, slots, page_stride, slot_stride, delay_ns, ROW: tl.constexpr
):
    """Releases its PDL dependents at entry, then (~`delay_ns` later) stores row i of `src` into
    the index plane's slot (`pages[i]`, `slots[i]`): a dependent that reads that slot before its
    griddepcontrol.wait sees the old key."""
    tl.extra.cuda.gdc_launch_dependents()
    t0 = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )
    t = t0
    while t - t0 < delay_ns:
        t = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )
    i = tl.program_id(0)
    page = tl.load(pages + i)
    slot = tl.load(slots + i)
    offs = tl.arange(0, ROW)
    tl.store(
        index_cache + page * page_stride + slot * slot_stride + offs,
        tl.load(src + i * ROW + offs),
    )


def _late_store(index_cache, pages, slots, keys, delay_ns: int) -> None:
    """`index_cache[pages, slots] = keys` by a producer that releases a PDL-launched successor
    at entry and stores ~`delay_ns` later (what the attention decode prologue may still be
    doing when the scores start)."""
    row = index_cache.shape[2] * index_cache.shape[3]
    assert index_cache.stride(3) == 1 and index_cache.stride(2) == index_cache.shape[3]
    assert keys.dtype == index_cache.dtype and keys.numel() == pages.numel() * row
    _late_slot_store[(pages.numel(),)](
        keys.contiguous(),
        index_cache,
        pages.to(torch.int64).contiguous(),
        slots.to(torch.int64).contiguous(),
        index_cache.stride(0),
        index_cache.stride(1),
        delay_ns,
        ROW=row,
    )


def test_request_group_prefetch_rereads_what_a_late_producer_stores() -> None:
    """`prefetch` behind a producer that releases it at entry and stores, ~500 us later, the
    run's own index slots and block 5's anchor (position 23, below the re-read bound): the keys
    at or past the run's smallest position minus one compressed block are read again after the
    wait, so those scores are the plain request-group kernel's bits over the stored keys.

    Block 5 is the overlap signal of the same invocation: the op reads it before the wait and
    never again, so its score is the old key's exactly when the op's loads ran before the
    producer's store. Where they did not overlap (CUDA may serialize the pair) the re-read check
    could not fail, and the attempt is retried, then reported inconclusive (skipped)."""
    rows, p0 = 4, 4096  # one request's verify run at 4096..4099: row 3 ends a group
    spec = _case(rows, 8, pages=_WIDTH_PAGES, only_visible_blocks=True)
    spec.update(q_layout="scratch_slice", use_pdl=True)
    args = _build(spec, seed=611)
    cr, tpb = args["compress_ratio"], args["tokens_per_block"]
    args["request_indices"] = args["request_indices"][:1].repeat(rows).contiguous()
    pos = p0 + torch.arange(rows, device=DEV)
    assert int(pos[-1]) < _WIDTH_PAGES * tpb
    args["query_positions"] = pos.to(args["query_positions"].dtype)
    group = {**args, "rows_per_request": 4}
    req = args["request_indices"].long()
    visible = _visible(args)
    q, cache = args["q"], args["index_cache"]
    # Keys whose score is positive (a row's own index head 0): the ReLU cannot hide them.
    key_of = {r: q[r, 0].reshape(1, -1) for r in range(rows)}

    # The run's own slots (slot 4099 is the anchor row 3 reads at column 1024), then block 5's.
    stale = torch.tensor([5 * cr + cr - 1], device=DEV)
    positions = torch.cat([pos, stale])
    pages = args["block_table"][req[0], positions // tpb].long()
    slots = positions % tpb
    keys = torch.cat([key_of[3]] * rows + [key_of[0]])
    anchor = (p0 + rows) // cr - 1
    old = cache[pages, slots].clone()
    # Compile both launches outside the timed pair (a compile would let the store finish).
    _late_store(cache, pages, slots, old, delay_ns=0)
    op(**{**group, "prefetch": True})
    before = op(**group)
    torch.cuda.synchronize()
    fresh = visible.clone()
    fresh[:, 5] = False
    for attempt in range(5):
        _late_store(cache, pages, slots, keys, delay_ns=500_000)
        with _watch_launches() as watcher:
            early = op(**{**group, "prefetch": True})
        torch.cuda.synchronize()
        assert watcher.branch == "request_group_prefetch"
        plain = op(**group)  # over the stored keys, the wait first
        cache[pages, slots] = old
        torch.cuda.synchronize()
        assert float(plain[3, anchor]) > 0.0 and float(before[3, anchor]) != float(plain[3, anchor])
        assert float(plain[0, 5]) > 0.0 and float(before[0, 5]) != float(plain[0, 5])
        # Every column but block 5 is the post-store bits, overlapped or not (the run's own
        # slots are read again after the wait).
        assert torch.equal(early[fresh].view(torch.int32), plain[fresh].view(torch.int32))
        # Block 5 is read once, before the wait: the old key's score or the new one's.
        if torch.equal(early[:, 5], before[:, 5]):
            return  # overlapped: the check above ran on a schedule where it can fail
        assert torch.equal(early[:, 5], plain[:, 5])
    pytest.skip("the op and the late producer never overlapped here: the re-read is unobserved")


#: A page index far past any pool: a load through it faults.
_FAR_PAGE = 1 << 28


def _far_page_mixed_run(seed: int, **spec_overrides):
    """Two requests in one two-row run at positions 3 and 35 (ratio 4, 32-token pages): the
    first sees one block, the second nine. The first request's table entries past its one page
    name a page far past the pool, so a load through them faults."""
    spec = _case(
        2, 2, layout="decode", gen_positions=[3, 35], rows_per_request=1, only_visible_blocks=True,
        q_layout="scratch_slice", call_rows_per_request=2,
    )  # fmt: skip
    spec.update(spec_overrides)
    args = _build(spec, seed=seed)
    short = int(args["request_indices"][0])
    args["block_table"][short, 1:] = _FAR_PAGE
    return args


@pytest.mark.parametrize("prefetch", [True, False], ids=["prefetch", "request_group"])
def test_a_short_request_in_a_mixed_run_reads_only_its_own_pages(prefetch) -> None:
    """A run whose rows name two requests of different lengths: each row's pages and keys are
    read only up to its own visible blocks, on the prefetching request-group kernel (with PDL)
    and on the plain one. Both rows' visible scores match the reference."""
    args = _far_page_mixed_run(97, use_pdl=prefetch, prefetch=prefetch)
    with _watch_launches() as watcher:
        got = op(**args)
        torch.cuda.synchronize()
    assert watcher.branch == ("request_group_prefetch" if prefetch else "request_group")
    # The reference gathers every column of every row (then masks): it reads the table past
    # the short sequence as the engine's (-1), not the far page the kernels must not touch.
    table = args["block_table"]
    table = torch.where(table == _FAR_PAGE, torch.full_like(table, -1), table)
    _compare(got, op.reference(**{**args, "block_table": table}))


def test_split_shape_is_a_launch_option_of_call_and_reference() -> None:
    """`split_shape` is accepted by `__call__` and `reference` alike (catalog validation hands
    both the same keywords). Rows that come 4 a request, consecutive and tile-aligned, at the
    target's `(programs a SM, tile // 4)`: bitwise the default launch."""
    spec = _case(
        16, 4, layout="decode", gen_positions=[700, 3000, 64, 9000], rows_per_request=4,
        only_visible_blocks=True, q_layout="scratch_slice", pages=300,
    )  # fmt: skip
    args = _build(spec, seed=131)
    tile = min(16, SPLIT_DOT_WIDTH // args["q"].shape[1])
    shape = (_entry._SPLIT_PROGRAMS_PER_SM, tile // 4)
    with _watch_launches() as watcher:
        default = op(**args)
        torch.cuda.synchronize()
    assert watcher.branch == "visible_split"
    with _watch_launches() as watcher:
        shaped = op(**args, split_shape=shape)
        torch.cuda.synchronize()
    assert watcher.branch == "visible_split" and watcher.grid[1] == tile // 4
    visible = _visible(args)
    assert torch.equal(shaped[visible].view(torch.int32), default[visible].view(torch.int32))
    _compare(shaped, op.reference(**args, split_shape=shape))


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


@pytest.mark.parametrize(
    "bad, match",
    [
        (dict(tokens_per_block=0), "positive pages"),
        (dict(compress_ratio=1), "compression ratio"),
        ("float_positions", "integer storage"),
        ("cpu_table", "one CUDA device"),
        ("empty_table", "non-empty block table"),
    ],
    ids=["tokens_per_block-0", "compress_ratio-1", "float-positions", "cpu-table", "empty-table"],
)
def test_prefill_query_tile_rejects_what_the_engine_rejects(bad, match) -> None:
    """The query-tiled launch refuses the operands the engine's launcher refuses."""
    gen = torch.Generator(device=DEV).manual_seed(1703)
    rows, tpb, cr = _entry._DECODE_ROW_LIMIT + 44, 32, 4
    cache = (torch.randn(16, tpb, 1, 128, generator=gen, device=DEV) * 0.5).to(torch.bfloat16)
    q = (torch.randn(rows, 4, 128, generator=gen, device=DEV) * 0.5).to(torch.bfloat16)
    args = dict(
        q=q,
        index_cache=cache,
        block_table=torch.arange(16, dtype=torch.int32, device=DEV).view(1, 16),
        query_positions=torch.arange(rows, dtype=torch.int64, device=DEV),
        request_indices=torch.zeros(rows, dtype=torch.int32, device=DEV),
        tokens_per_block=tpb,
        compress_ratio=cr,
        only_visible_blocks=True,
        context_rows=True,
    )
    assert _entry._takes_prefill_query_tile(q, True, True, False)
    if bad == "float_positions":
        args["query_positions"] = args["query_positions"].float()
    elif bad == "cpu_table":
        args["block_table"] = args["block_table"].cpu()
    elif bad == "empty_table":
        args["block_table"] = args["block_table"][:, :0]
    else:
        args.update(bad)
    with pytest.raises(ValueError, match=match):
        _entry._qsa_index_scores_prefill_query_tile(
            **{k: v for k, v in args.items() if k not in ("only_visible_blocks", "context_rows")}
        )


@pytest.mark.parametrize(
    "chunks",
    [((67584, 8190),), ((5000, 300), (20000, 1700)), ((131072, 4093),)],
    ids=["one-request-8190-at-67k", "two-requests", "ragged-4093-at-128k"],
)
def test_prefill_query_tile_is_bitwise_the_engine_launch(chunks) -> None:
    """`prefill_query_tile` takes the deeper query-tiled launch, bitwise the engine's launch on
    every visible column. `chunks`: `(position of the first row, rows)` per request."""
    gen = torch.Generator(device=DEV).manual_seed(len(chunks) * 7919 + chunks[0][0])
    tpb, cr, width = 32, 4, 8196
    pages = [math.ceil((p + n) / tpb) for p, n in chunks]
    pool = sum(pages) + 8
    cache = (torch.randn(pool, tpb, 1, 128, generator=gen, device=DEV) * 0.5).to(torch.bfloat16)
    table = torch.full((16, width), -1, dtype=torch.int32, device=DEV)
    perm = 8 + torch.randperm(sum(pages), generator=gen, device=DEV).to(torch.int32)
    slots = [3 + 5 * r for r in range(len(chunks))]
    start = 0
    for slot, n_pages in zip(slots, pages):
        table[slot, :n_pages] = perm[start : start + n_pages]
        start += n_pages
    rows = sum(n for _, n in chunks)
    q = (torch.randn(rows, 6, 128, generator=gen, device=DEV) * 0.5).to(torch.bfloat16)[:, :4]
    positions = torch.cat(
        [torch.arange(p, p + n, dtype=torch.int32, device=DEV) for p, n in chunks]
    )
    requests = torch.cat(
        [torch.full((n,), s, dtype=torch.int32, device=DEV) for s, (_, n) in zip(slots, chunks)]
    )
    args = dict(
        q=q,
        index_cache=cache,
        block_table=table,
        query_positions=positions,
        request_indices=requests,
        tokens_per_block=tpb,
        compress_ratio=cr,
        only_visible_blocks=True,
        context_rows=True,
    )
    assert _entry._takes_prefill_query_tile(q, True, True, False)
    with _watch_launches() as watcher:
        out = op(**args, prefill_query_tile=True)
        torch.cuda.synchronize()
    assert watcher.branch == "query_tile"
    assert watcher.grid[0] == math.ceil(rows / _entry._PREFILL_QUERY_TILE)
    with _watch_launches() as watcher:
        default = op(**args)
        torch.cuda.synchronize()
    assert watcher.branch == "query_tile"
    assert watcher.grid[0] != math.ceil(rows / _entry._PREFILL_QUERY_TILE)
    ref = _engine_scores(**args)
    vis = _visible(args)
    assert torch.equal(torch.where(vis, out, 0.0), torch.where(vis, ref, 0.0))
    assert torch.equal(torch.where(vis, default, 0.0), torch.where(vis, ref, 0.0))


def test_visible_split_compiles_one_variant_up_to_a_row_tile() -> None:
    """Launches of up to one row tile (`16 // heads` rows) share one compiled variant."""
    dev = torch.cuda.current_device()
    kernel = _entry._qsa_index_scores_visible_split_kernel
    heads, dim, tpb, cr = 4, 128, 32, 4
    g = torch.Generator(device=DEV).manual_seed(7)
    cache = torch.randn(64, tpb, 1, dim, generator=g, device=DEV).to(torch.bfloat16)
    table = torch.randint(0, 64, (8, 40), generator=g, device=DEV, dtype=torch.int32)
    seen = []
    for rows in (4, 1, 2, 3, 5, 8):
        q = torch.randn(rows, heads, dim, generator=g, device=DEV).to(torch.bfloat16)
        positions = torch.randint(100, 1279, (rows,), generator=g, device=DEV, dtype=torch.int64)
        requests = torch.randint(0, 8, (rows,), generator=g, device=DEV, dtype=torch.int32)
        args = (q, cache, table, positions, requests, tpb, cr, True, False)
        _compare(op(*args), op.reference(*args))
        seen.append(len(kernel.device_caches[dev][0]))
    assert len(set(seen)) == 1, seen


@pytest.mark.parametrize("rows", [32, 64, 96])
def test_visible_split_wide_batch_programs_are_bitwise_the_narrow_split(rows: int) -> None:
    """`wide_split` scores every visible column bitwise as the narrow split, over ragged windows."""
    g = torch.Generator(device=DEV).manual_seed(rows)
    tpb, cr, heads, dim = 32, 4, 4, 128
    ctx = torch.randint(64, 200000, (rows,), generator=g, device=DEV)
    pages = (ctx + tpb - 1) // tpb
    max_pages = int(pages.max())
    total = int(pages.sum())
    cache = torch.randn(total, tpb, 1, dim, generator=g, device=DEV).to(torch.bfloat16)
    perm = torch.randperm(total, generator=g, device=DEV).to(torch.int32)
    bt = torch.zeros(rows, max_pages, dtype=torch.int32, device=DEV)
    off = 0
    for r in range(rows):
        bt[r, : int(pages[r])] = perm[off : off + int(pages[r])]
        off += int(pages[r])
    q = torch.randn(rows, heads, dim, generator=g, device=DEV).to(torch.bfloat16)
    pos = (ctx - 1).to(torch.int64)
    req = torch.arange(rows, dtype=torch.int64, device=DEV)
    wide = _entry._qsa_index_scores_visible_split(q, cache, bt, pos, req, tpb, cr, False, wide=True)
    narrow = _entry._qsa_index_scores_visible_split(q, cache, bt, pos, req, tpb, cr, False)
    torch.cuda.synchronize()
    visible = torch.clamp((pos + 1) // cr, max=wide.shape[1])
    mask = torch.arange(wide.shape[1], device=DEV)[None, :] < visible[:, None]
    assert torch.equal(torch.where(mask, wide, 0.0), torch.where(mask, narrow, 0.0))
