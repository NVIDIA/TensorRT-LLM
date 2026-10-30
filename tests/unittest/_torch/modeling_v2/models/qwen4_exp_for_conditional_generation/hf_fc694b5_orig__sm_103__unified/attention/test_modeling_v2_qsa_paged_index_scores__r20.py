# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_index_scores catalog entry.

The paged state is real: a `[pages, tokens_per_block, 1, dim]` index cache
(a view of a wider page under the NVFP4 cells) sized from the pages the table maps, and an
int32 block table whose rows are request *slots* -- more of them than the batch has requests, assigned by a
seeded permutation -- with shuffled physical pages and `-1` past each
sequence on the serving layouts, which is what the engine's table carries.

The call has two kernels behind it. `_watch_launches` swaps the two Triton
entry points for recording proxies that still launch the real kernel, so every
cell asserts the branch it names.
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.attention.qsa_paged_index_scores__r20 import (  # noqa: E501
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
    1/head_dim scale, block table ignored) were measured 3e5x to 4e6x
    outside it. Columns the reference marks NaN are unspecified and not read.
    """
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    specified = ~torch.isnan(ref)
    torch.testing.assert_close(out[specified], ref[specified])


DEV = torch.device("cuda")
#: The callable's own threshold: at or below it every call is row-wise.
DECODE_ROW_LIMIT = 256
#: Index heads must divide this into more than one row for the query tile.
DOT_TILE_WIDTH = 128


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
    pool_pages = sum(mapped) + 3
    page_ids = torch.randperm(pool_pages, generator=gen, device=DEV).to(torch.int32)
    block_table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    tpb = spec["tokens_per_block"]
    keys = torch.randn(pool_pages, tpb, 1, head_dim, generator=gen, device=DEV).to(dtype)
    if "page_pad" in spec:
        # The NVFP4 pool's page: the other index buffers, this layer's, then the position
        # page, so the plane is a view at page stride buffers * tpb * dim + page_pad.
        buffers, buffer = spec["pool_buffers"], tpb * head_dim
        page = torch.zeros(pool_pages, buffers * buffer + spec["page_pad"], dtype=dtype, device=DEV)
        index_cache = page[:, (buffers - 1) * buffer : buffers * buffer].view(
            pool_pages, tpb, 1, head_dim
        )
        index_cache.copy_(keys)
        assert index_cache.stride() == (page.stride(0), head_dim, head_dim, 1)
    else:
        index_cache = keys

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


def test_prefetch_reads_the_fresh_anchor_key_and_q_after_the_wait() -> None:
    """`prefetch` reads pages and keys before the PDL wait. Under graph replay a producer
    rewrites q and, at rows whose position ends a group, the last visible block's anchor
    key -- what the decode prologue writes -- right before the op; both must be seen.
    The other keys are static, as they are in the decode step."""
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
    plain = op(**args)
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
        assert torch.equal(graph_out[visible], op(**args)[visible]), f"replay {trial}"


def test_rejects_metadata_shorter_than_the_rows() -> None:
    """The callable's own refusal, message as observed -- the reason this entry has no
    `is_valid`: positions for fewer rows than `q` carries."""
    args = _build(_case(16, 2), seed=1701)
    args["query_positions"] = args["query_positions"][:4]
    with pytest.raises(ValueError, match="one entry per row"):
        op(**args)
