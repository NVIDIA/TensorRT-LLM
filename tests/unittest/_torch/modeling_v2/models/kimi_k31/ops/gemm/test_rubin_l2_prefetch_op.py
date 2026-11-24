# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_l2_prefetch entry: an L2 prefetch changes no byte, eagerly or replayed.

The kernel's own tests (``kernels/gemm/test_rubin_l2_prefetch.py``) cover the byte ranges and the
side-stream use; these drive the entry's cells.
"""

import pytest
import torch
from _rubin_gemm_util import generator, requires_sm107

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_l2_prefetch import (  # noqa: E501
    MAX_CTAS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_l2_prefetch import (  # noqa: E501
    rubin_l2_prefetch as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm107

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _ranges(tensors):
    """The ``(address, bytes)`` range of each tensor, by the kernel module's ``tensor_range``."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm import (
        rubin_l2_prefetch as kernel,
    )

    return tuple(kernel.tensor_range(t) for t in tensors)


def _buffers(spec, gen):
    return [
        torch.randint(0, 256, (size,), generator=gen, device="cuda", dtype=torch.uint8)
        for size in spec["sizes"]
    ]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    buffers = _buffers(cell.spec, generator(cell.why))
    before = [b.clone() for b in buffers]
    ranges = _ranges(buffers)
    assert op(ranges, max_ctas=cell.spec["max_ctas"]) is None
    torch.cuda.synchronize()
    assert op.reference(ranges, max_ctas=cell.spec["max_ctas"]) is None
    assert all(torch.equal(b, a) for b, a in zip(buffers, before))


def test_a_replayed_graph_leaves_memory_untouched() -> None:
    buffers = _buffers(CELLS[0].spec, generator("graph"))
    before = [b.clone() for b in buffers]
    ranges = _ranges(buffers)
    op(ranges)  # compiles outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(ranges)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(b, a) for b, a in zip(buffers, before))


def test_ranges_and_the_cta_cap_are_the_kernels() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm import (
        rubin_l2_prefetch as kernel,
    )

    assert MAX_CTAS == kernel.DEFAULT_MAX_CTAS
    w = torch.empty(256, 384, dtype=torch.float8_e4m3fn, device="cuda")
    # torch gives a tensor with no elements a null data pointer, so an empty slice is (0, 0).
    assert _ranges([w, w[:0]]) == ((w.data_ptr(), 256 * 384), (0, 0))


def test_empty_ranges_never_reach_the_kernel(monkeypatch) -> None:
    """The launcher drops a 0-byte range, at address 0 or any other, before it counts, checks or
    launches: empty ranges alone launch nothing, and beside a full one only that one launches."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm import (
        rubin_l2_prefetch as kernel,
    )

    launches = []
    monkeypatch.setattr(
        kernel, "_compiled", lambda n, ctas: lambda *args: launches.append((n, ctas))
    )
    w = torch.empty(256, 384, dtype=torch.float8_e4m3fn, device="cuda")
    op(_ranges([w[:0]]))
    op([(w.data_ptr(), 0)] * (kernel.MAX_RANGES + 1))  # empty ranges do not count against the 8
    assert launches == []
    op(_ranges([w[:0], w, w[:0]]))
    assert launches == [(1, 1)]  # w alone: 12 chunks of 8 KiB, one CTA
