# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cu_seqlens_from_seq_lens entry.

Integer arithmetic, gated bitwise. A near-miss vector is monotonic and
plausible, so each cell also proves the vector actually splits a packed
buffer.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.cu_seqlens_from_seq_lens import (  # noqa: E501
    cu_seqlens_from_seq_lens as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "cu_seqlens_from_seq_lens requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise: integer arithmetic, no tolerance to loosen."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert out.is_contiguous()
    assert torch.equal(out, ref), f"{out.tolist()} != {ref.tolist()}"


DEV = "cuda"


def _seq_lens(lengths, dtype=torch.int32):
    return torch.tensor(list(lengths), dtype=dtype, device=DEV)


def _build(spec):
    lengths, dtype = spec["lengths"], spec["in_dtype"]
    layout = spec.get("layout", "contiguous")
    if layout == "prefix":
        # The metadata buffer is sized for max_num_requests; the call site
        # passes its first `batch` entries.
        buf = torch.full((len(lengths) + 9,), 99, dtype=dtype, device=DEV)
        buf[: len(lengths)] = _seq_lens(lengths, dtype)
        return buf[: len(lengths)]
    return _seq_lens(lengths, dtype)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    seq_lens = _build(spec)
    before = seq_lens.clone()
    out = op(seq_lens, spec["out_dtype"])
    assert torch.equal(seq_lens, before), "seq_lens is read-only"
    assert out.device.type == "cuda"
    _compare(out, op.reference(seq_lens, spec["out_dtype"]))

    # The entry's purpose: slicing a packed buffer with the vector recovers
    # exactly the sequences packed into it.
    lengths = spec["lengths"]
    g = torch.Generator(device=DEV).manual_seed(zlib.crc32(cell.why.encode()))
    pieces = [torch.randint(0, 10_000, (n,), generator=g, device=DEV) for n in lengths]
    packed = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int64, device=DEV)
    bounds = out.tolist()
    assert bounds[-1] == packed.shape[0]
    for s, piece in enumerate(pieces):
        assert torch.equal(packed[bounds[s] : bounds[s + 1]], piece), f"sequence {s} mis-sliced"


#: Graph-captured batches: the MTP-7 verify of 32 requests, and the largest captured
#: batch (128 requests) at the call site's int64 output.
_GRAPH_BATCHES = ((32, torch.int32), (128, torch.int64))


@pytest.mark.parametrize(
    "batch, dtype", _GRAPH_BATCHES, ids=[f"{b}req-{str(d)[6:]}" for b, d in _GRAPH_BATCHES]
)
def test_cuda_graph_capture_and_replay(batch, dtype) -> None:
    """A captured graph follows lengths rewritten in place: only the shape
    reaches Python, the values are read on device and never synchronized on.
    Replayed at one row a request, at MTP-7's 8, at one longer row, and at a
    mixed step whose rows fill `max_num_tokens` (8192)."""
    p, max_num_tokens = 8, 8192
    static_lens = _seq_lens([p] * batch)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            op(static_lens, dtype)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = op(static_lens, dtype)
    for lengths in (
        [1] * batch,
        [p] * batch,
        [p + 1] + [1] * (batch - 1),
        [max_num_tokens - p * (batch - 1)] + [p] * (batch - 1),
    ):
        static_lens.copy_(_seq_lens(lengths))
        graph.replay()
        torch.cuda.synchronize()
        assert static_out.dtype == dtype
        assert torch.equal(static_out, op.reference(static_lens, dtype))


def test_guard_refuses_non_integer_seq_lens() -> None:
    """Unguarded, a float32 seq_lens is truncated toward zero before the scan."""
    floats = torch.tensor([7.9, 1.2, 13.6], dtype=torch.float32, device=DEV)
    assert op(floats, torch.int32).tolist() == [0, 7, 8, 21], "hazard gone"
    with pytest.raises(AssertionError, match="seq_lens must have an integer dtype"):
        with validating(op):
            op(floats, torch.int32)
