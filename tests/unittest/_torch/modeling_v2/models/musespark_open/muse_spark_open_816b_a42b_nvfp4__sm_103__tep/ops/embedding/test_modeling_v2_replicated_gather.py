# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``replicated_embedding_gather`` (``replicated_embedding``).

The entry's claim is bit equality with the vocab-parallel embedding summed by
the one-shot all-reduce, so every cell is gated with ``torch.equal`` on the
16-bit patterns against the entry's ``reference`` (the eight masked shard
lookups, the payload sanitizing and the fp32 sum), eagerly and replayed from a
CUDA graph. The real table and the real all-reduce are the 8-rank test's
(``test_modeling_v2_replicated_embedding_tp8.py``).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.embedding.replicated_gather import (  # noqa: E501
    ReplicatedEmbeddingGather,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.embedding.replicated_gather import (  # noqa: E501
    replicated_embedding_gather as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "replicated_embedding_gather requires a CUDA device"

#: A table of the target's width, with a vocabulary small enough for the sweeps.
VOCAB, HIDDEN = 8 * 2048, 8192


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _table(kind: str, g: torch.Generator) -> torch.Tensor:
    table = (torch.randn(VOCAB, HIDDEN, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    if kind == "signed_zeros":
        holes = torch.rand(VOCAB, HIDDEN, generator=g, device="cuda") < 0.05
        signs = torch.rand(VOCAB, HIDDEN, generator=g, device="cuda") < 0.5
        zeros = torch.where(signs, -0.0, 0.0).to(torch.bfloat16)
        table = torch.where(holes, zeros, table)
    return table.contiguous()


def _ids(spec, g: torch.Generator) -> torch.Tensor:
    shape, kind = spec["shape"], spec["ids"]
    dtype = torch.int32 if spec["dtype"] == "int32" else torch.int64
    count = 1
    for n in shape:
        count *= n
    if kind == "invalid":
        flat = torch.tensor([-1, VOCAB, VOCAB + 7, 0, VOCAB - 1, -5], device="cuda")[:count]
    elif kind == "edges":
        width = VOCAB // 8
        edges = [r * width for r in range(8)] + [(r + 1) * width - 1 for r in range(8)]
        flat = torch.tensor(edges, device="cuda")[:count]
    elif kind == "mixed":
        flat = torch.randint(0, VOCAB, (count,), generator=g, device="cuda")
        holes = torch.rand(count, generator=g, device="cuda") < 0.1
        flat = torch.where(holes, torch.full_like(flat, -1), flat)
        flat[::17] = VOCAB + 3
    elif kind == "noise":
        bonus = torch.randint(0, VOCAB, (1,), generator=g, device="cuda")
        flat = torch.cat([bonus, torch.full((count - 1,), 7, device="cuda")])
    else:
        flat = torch.randint(0, VOCAB, (count,), generator=g, device="cuda")
    return flat.to(dtype).reshape(shape).contiguous()


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _check(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == ref.dtype == torch.bfloat16
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert torch.equal(_bits(out), _bits(ref)), "the gathered rows differ from the sharded sum"


@pytest.mark.parametrize(
    "cell",
    ReplicatedEmbeddingGather.CELLS,
    ids=[c.why[:48] for c in ReplicatedEmbeddingGather.CELLS],
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    table = _table(cell.spec["table"], g)
    for _ in range(5):
        ids = _ids(cell.spec, g)
        out = op(table, ids)
        _check(out, op.reference(table, ids))
        if cell.spec["table"] == "signed_zeros":
            assert not (_bits(out) == -32768).any(), "a -0.0 survived"
            assert (_bits(table) == -32768).any(), "the table should hold -0.0 values"


def test_every_row_of_a_table_is_its_sharded_sum():
    """All VOCAB ids, 8 at a time (the decode width), on a table with signed zeros."""
    g = torch.Generator(device="cuda").manual_seed(11)
    table = _table("signed_zeros", g)
    all_ids = torch.arange(VOCAB, device="cuda", dtype=torch.int32)
    out = op(table, all_ids)
    ref = op.reference(table, all_ids)
    _check(out, ref)
    for start in range(0, VOCAB, 8 * 257):
        ids = all_ids[start : start + 8].contiguous()
        _check(op(table, ids), ref[start : start + 8])


def test_cuda_graph_replay_follows_the_ids():
    g = torch.Generator(device="cuda").manual_seed(5)
    table = _table("signed_zeros", g)
    ids = _ids(dict(shape=(8,), dtype="int32", ids="valid"), g)
    op(table, ids)  # compile outside the capture
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        out = op(table, ids)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(50):
        ids.copy_(_ids(dict(shape=(8,), dtype="int32", ids="valid"), g))
        graph.replay()
        torch.cuda.synchronize()
        _check(out, op.reference(table, ids))


def test_guard_refuses_what_the_kernel_cannot_take():
    table = torch.zeros(16, 64, dtype=torch.bfloat16, device="cuda")
    with validating(op):
        with pytest.raises(AssertionError):
            op(table.float(), torch.zeros(2, dtype=torch.int64, device="cuda"))
        with pytest.raises(AssertionError):
            op(table, torch.zeros(2, device="cuda"))
        with pytest.raises(AssertionError):
            op(table.t(), torch.zeros(2, dtype=torch.int64, device="cuda"))
