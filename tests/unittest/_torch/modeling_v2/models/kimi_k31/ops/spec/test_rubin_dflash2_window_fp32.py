# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 window decode with Triton-staged operands (``rubin_dflash2_window_fp32``).

The entry claims ``sparse_mla_rows_eager``'s output bit for bit, so every gate here is equality,
never a band. Each cell builds a paged history whose window pages are random pages of the pool
and whose other page-table columns point at page 0 (finite values, never read in a live column).

GPU (SM107): every cell against ``reference`` bit for bit, and the staged operands against the
torch staging; one CUDA graph captured on one input set and replayed on another equals an eager
run on the second; new batch sizes and lengths compile nothing after the warmup pass.

CPU (any host): ``is_valid`` admits every cell's inputs and refuses an index past the pool, a
strided query and an int64 index list.
"""

from __future__ import annotations

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec import (
    rubin_dflash2_window_fp32 as kw,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dflash2_window import (  # noqa: E501
    dflash2_window_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_window_fp32 import (  # noqa: E501
    rubin_dflash2_window_fp32 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

_ON_SM107 = torch.cuda.is_available() and get_sm_version() == 107
gpu = pytest.mark.skipif(not _ON_SM107, reason="the entry is certified on SM107 (Rubin)")


def _ids(cells):
    return [f"cell{i}" for i in range(len(cells))]


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode())


def build(spec: dict, seed: int, device, lengths=None):
    """``(query, pool_rows, rows)``: window pages drawn at random from the pool, the rest page 0."""
    g = torch.Generator().manual_seed(seed)
    lengths = list(spec["lengths"] if lengths is None else lengths)
    batch, blk, page = len(lengths), spec["block_size"], spec["page_size"]
    dim = spec["latent"] + spec["rope"]
    span = (spec["window"] + blk) // page + 2  # pages a window and its block can touch
    columns = (max(lengths) + blk) // page + 1
    num_pages = 1 + batch * span
    table = torch.zeros(batch, columns, dtype=torch.int64)
    perm = torch.randperm(num_pages - 1, generator=g) + 1
    for b, length in enumerate(lengths):
        lo = max(0, length - spec["window"]) // page
        hi = min(columns, (length + blk - 1) // page + 1)
        table[b, lo:hi] = perm[b * span : b * span + (hi - lo)]
    pool = 0.5 * torch.randn(num_pages * page, dim, generator=g)
    pool = pool.to(torch.float8_e4m3fn if spec["pool"] == "fp8" else torch.bfloat16)
    query = (0.5 * torch.randn(batch * blk, spec["heads"], dim, generator=g)).to(torch.bfloat16)
    ctx = torch.tensor(lengths, dtype=torch.int64)
    rows, _ = dflash2_window_rows(ctx, table, blk, page, spec["window"], spec["window"])
    return query.to(device), pool.to(device), rows.to(device)


def _args(spec, query, pool, rows):
    return (query, pool, rows, spec["scale"], spec["latent"], spec["kv_scale"])


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16) if x.element_size() == 2 else x.view(torch.int32)


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=_ids(op.CELLS))
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    args = _args(spec, *build(spec, _seed(cell), "cuda"))
    out = op(*args)
    ref = op.reference(*args)
    assert out.dtype == ref.dtype and out.shape == ref.shape
    assert torch.equal(_bits(out), _bits(ref)), "the staged decode differs from the torch decode"


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=_ids(op.CELLS))
def test_staged_operands_are_the_torch_staging(cell):
    spec = cell.spec
    query, pool, rows = build(spec, _seed(cell), "cuda")
    kv = torch.empty(*rows.shape, query.shape[-1], dtype=torch.float32, device="cuda")
    q32 = torch.empty(query.shape, dtype=torch.float32, device="cuda")
    kw.launch_window_operands(query, pool, rows, spec["kv_scale"], kv, q32)
    want = pool[rows.clamp_min(0).long()].float()
    if spec["kv_scale"] != 1.0:
        want = want * spec["kv_scale"]
    assert torch.equal(_bits(kv), _bits(want)) and torch.equal(_bits(q32), _bits(query.float()))
    scores = torch.einsum("rhc,rkc->rhk", q32, kv)
    masked = (scores * spec["scale"]).masked_fill(~(rows >= 0).unsqueeze(1), float("-inf"))
    kw.launch_scale_mask(scores, rows, spec["scale"])
    assert torch.equal(_bits(scores), _bits(masked))


@gpu
def test_capture_on_one_input_set_replay_on_another():
    """Capture on inputs A, copy inputs B into the same buffers, replay: equal to an eager run
    on B bit for bit."""
    spec = op.CELLS[2].spec
    sets = [
        build(spec, 1, "cuda", lengths=(1031, 9000, 0, 65)),
        build(spec, 2, "cuda", lengths=(190000, 1, 4097, 7)),
    ]
    pages = max(s[1].shape[0] for s in sets)
    query = torch.zeros_like(sets[0][0])
    pool = torch.zeros(pages, sets[0][1].shape[1], dtype=sets[0][1].dtype, device="cuda")
    rows = torch.zeros_like(sets[0][2])

    def load(s):
        query.copy_(s[0])
        pool.view(torch.uint8).zero_()
        pool[: s[1].shape[0]] = s[1]
        rows.copy_(s[2])

    def step():
        return op(query, pool, rows, spec["scale"], spec["latent"], spec["kv_scale"])

    load(sets[0])
    step()  # eager warmup: both kernels compiled before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    load(sets[1])
    graph.replay()
    replayed = graph_out.clone()
    eager = op.reference(query, pool, rows, spec["scale"], spec["latent"], spec["kv_scale"])
    assert torch.equal(_bits(replayed), _bits(eager))


def _compiled_variants() -> int:
    total = 0
    for fn in (kw._window_operands_kernel, kw._scale_mask_kernel):
        caches = getattr(fn, "device_caches", None)
        if caches is None:
            pytest.skip("this Triton keeps no per-device kernel cache to count")
        total += sum(len(entry[0]) for entry in caches.values())
    return total


@gpu
def test_no_compile_after_the_warmup_pass():
    """New batch sizes, lengths and pools never compile: only the row width, the tiles and the
    dequant flag select a variant."""
    spec = op.CELLS[0].spec
    rounds = [
        [(1031,), (9000, 1), (0, 63, 4097, 190000)],
        [(65,), (190000, 4097), (1, 1022, 65536, 5)],
    ]
    counts = []
    for lengths_per_batch in rounds:
        for lengths in lengths_per_batch:
            op(*_args(spec, *build(spec, len(lengths), "cuda", lengths=lengths)))
        torch.cuda.synchronize()
        counts.append(_compiled_variants())
    assert counts[1] == counts[0], f"compiled {counts[1] - counts[0]} variants after warmup"


_SMALL = dict(op.CELLS[1].spec, lengths=(0, 5, 300, 1030))


@pytest.mark.parametrize("cell", op.CELLS, ids=_ids(op.CELLS))
def test_is_valid_admits_every_cell(cell):
    spec = dict(cell.spec, lengths=tuple(min(x, 4097) for x in cell.spec["lengths"]))
    op.is_valid(*_args(spec, *build(spec, _seed(cell), "cpu")))


def test_is_valid_refuses_an_index_past_the_pool():
    query, pool, rows = build(_SMALL, 3, "cpu")
    rows = rows.clone()
    rows[0, 0] = pool.shape[0]
    with pytest.raises(AssertionError, match="past the pool"):
        op.is_valid(*_args(_SMALL, query, pool, rows))


def test_is_valid_refuses_a_strided_query_and_int64_rows():
    query, pool, rows = build(_SMALL, 4, "cpu")
    strided = torch.empty(query.shape[1], query.shape[0], query.shape[2], dtype=query.dtype)
    strided = strided.transpose(0, 1)
    strided.copy_(query)
    with pytest.raises(AssertionError, match="contiguous query"):
        op.is_valid(*_args(_SMALL, strided, pool, rows))
    with pytest.raises(AssertionError):
        op.is_valid(*_args(_SMALL, query, pool, rows.long()))
