# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 window-rows entry (CuTe DSL).

Every certified cell is gated element for element against the entry's reference (the drafter's
torch index lists): the same int32 rows, ``-1`` padding and lengths. Page tables hold random
distinct page ids, so a wrong page column, offset or window edge moves entries.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_rows import (  # noqa: E501
    dflash2_window_rows as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x",
)


def _ctx_lengths(kind: str, batch: int, g: torch.Generator) -> torch.Tensor:
    if kind == "short":
        return torch.full((batch,), 3)
    if kind == "long":
        return torch.full((batch,), 5000)
    if kind == "edge":
        return torch.tensor([1018, 1019, 1020, 1023, 1024, 0, 4096])[:batch]
    lengths = torch.randint(0, 20000, (batch,), generator=g)
    lengths[0] = 0
    return lengths


def _build(spec, seed):
    g = torch.Generator().manual_seed(seed)
    batch, width, page_size = spec["batch"], spec["width"], spec["page_size"]
    ctx_len = _ctx_lengths(spec["ctx"], batch, g)
    ring = spec["ring"]
    if ring:
        # A ring slot holds ring_rows rows whatever the context length.
        pages = ring // page_size + spec["spare_pages"]
    else:
        pages = max(
            1, (int(ctx_len.max()) + width + page_size - 1) // page_size + spec["spare_pages"]
        )
    tables = torch.randperm(1 << 20, generator=g)[: batch * pages].view(batch, pages)
    if spec["table"] == "strided":
        wide = torch.full((2 * batch, pages + 5), -7, dtype=torch.int64)
        wide[::2, :pages] = tables
        tables = wide.cuda()[::2, :pages]
    else:
        tables = tables.cuda()
    dtype = spec["dtype"]
    return (
        ctx_len.to(dtype).cuda(),
        tables.to(dtype) if tables.dtype != dtype else tables,
        width,
        page_size,
        spec["window"],
        spec["index_width"],
        ring,
    )


def _assert_equal(got, want) -> None:
    for name, a, b in zip(("rows", "lengths"), got, want):
        assert a.shape == b.shape and a.dtype == b.dtype, (name, a.shape, b.shape, a.dtype, b.dtype)
        mismatched = int((a != b).sum())
        assert mismatched == 0, f"{name}: {mismatched}/{a.numel()} entries differ"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    _assert_equal(op(*args), op.reference(*args))


def test_one_launch_per_call() -> None:
    """The op is one kernel; the reference is the launch count it replaces (printed)."""
    args = _build(op.CELLS[2].spec, seed=1)
    op(*args), op.reference(*args)
    torch.cuda.synchronize()
    counts = {}
    for name, fn in (("op", op), ("reference", op.reference)):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(*args)
            torch.cuda.synchronize()
        counts[name] = sum(
            1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        )
    print(f"device launches per call: {counts}")
    assert counts["op"] == 1, counts


def test_cuda_graph_replay() -> None:
    """Captured after one eager call, replayed on new lengths and tables: still equal."""
    spec = op.CELLS[3].spec
    ctx_len, tables, *rest = _build(spec, seed=2)
    op(ctx_len, tables, *rest)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(ctx_len, tables, *rest)
    g = torch.Generator().manual_seed(3)
    for _ in range(2):
        ctx_len.copy_(torch.randint(0, 1000 * tables.shape[1] // 64, ctx_len.shape, generator=g))
        tables.copy_(torch.randperm(1 << 20, generator=g)[: tables.numel()].view_as(tables))
        graph.replay()
        torch.cuda.synchronize()
        _assert_equal(out, op.reference(ctx_len, tables, *rest))


def test_guard_refuses_float_lengths() -> None:
    ctx_len, tables, *rest = _build(op.CELLS[0].spec, seed=4)
    with validating(op), pytest.raises(AssertionError, match="integer ctx_len"):
        op(ctx_len.float(), tables, *rest)


def test_narrow_index_width_raises() -> None:
    ctx_len, tables, width, page_size, window, *_ = _build(op.CELLS[0].spec, seed=5)
    with pytest.raises(ValueError, match="cannot hold"):
        op(ctx_len, tables, width, page_size, window, window - 4)


def test_small_ring_raises() -> None:
    ctx_len, tables, width, page_size, window, index_width, _ = _build(op.CELLS[0].spec, seed=6)
    with pytest.raises(ValueError, match="cannot hold"):
        op(ctx_len, tables, width, page_size, window, index_width, window + width - 2)


def test_ring_cuda_graph_replay() -> None:
    """The ring cell captured after one eager call, replayed on lengths past the ring: equal."""
    spec = op.CELLS[7].spec
    ctx_len, tables, *rest = _build(spec, seed=7)
    op(ctx_len, tables, *rest)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(ctx_len, tables, *rest)
    g = torch.Generator().manual_seed(8)
    for _ in range(2):
        ctx_len.copy_(torch.randint(0, 50000, ctx_len.shape, generator=g))
        tables.copy_(torch.randperm(1 << 20, generator=g)[: tables.numel()].view_as(tables))
        graph.replay()
        torch.cuda.synchronize()
        _assert_equal(out, op.reference(ctx_len, tables, *rest))
