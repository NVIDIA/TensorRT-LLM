# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 masked_embedding_gather entry.

A gather, so everything is gated bitwise: the cells against the entry's reference, and
the op against the Embedding module's own masked `F.embedding` chain at TP1, which it
replaces. Then CUDA-graph replay, id shapes, and the inputs the guard refuses.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.embedding.masked_embedding_gather import (  # noqa: E501
    masked_embedding_gather as op,
)
from tensorrt_llm._torch.modules.embedding import pre_comm_embedding_ops
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3.8-27b sm_120 tp1 entries are certified on sm_120"
)

#: The checkpoint's vocabulary and hidden size.
VOCAB, HIDDEN = 248320, 5120


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


@pytest.fixture(scope="module")
def table():
    """The [248320, 5120] bf16 table (2.5 GB), built once per module."""
    g = torch.Generator(device="cuda").manual_seed(_seed("table"))
    weight = torch.empty(VOCAB, HIDDEN, device="cuda", dtype=torch.bfloat16)
    weight.normal_(0.0, 0.02, generator=g)
    yield weight
    del weight
    torch.cuda.empty_cache()


def _ids(spec, seed):
    """`random`: uniform in-vocab int32 ids. `edges`: the first and last rows, the -1 pad,
    ids at and past the vocab and the int32 extremes first, then random."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(0, VOCAB, (spec["tokens"],), generator=g, device="cuda")
    if spec["ids"] == "edges":
        edges = torch.tensor([-1, 0, VOCAB - 1, VOCAB, VOCAB + 5, 2**31 - 1, -(2**31)])
        ids[: edges.numel()] = edges.to(ids.device)
    return ids.to(torch.int32)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise, as raw 16-bit patterns: rows are copies (and -0.0 / +0.0 rows differ)."""
    assert out.dtype == ref.dtype and out.shape == ref.shape, (out.shape, ref.shape)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), "rows differ"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, table) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    ids = _ids(cell.spec, seed=_seed(cell.why))
    before = ids.clone()
    out = op(ids, table)
    assert out.shape == (*ids.shape, HIDDEN) and out.dtype == table.dtype
    assert out.is_contiguous()
    assert torch.equal(ids, before), "the ids are read only"
    _compare(out, op.reference(ids, table))


def test_bitwise_equal_to_the_embedding_module_chain(table) -> None:
    """The masked `F.embedding` chain the Embedding module runs at TP1, with its
    out-of-range guard: the op computes exactly that, +0.0 rows included."""
    ids = _ids(dict(tokens=64, ids="edges"), seed=_seed("chain"))
    chain = pre_comm_embedding_ops(
        ids,
        table,
        tp_size=1,
        tp_rank=0,
        tp_mode=None,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        gather_output=False,
        padding_size=0,
    )
    _compare(op(ids, table), chain)


def test_cuda_graph_replay_reads_the_ids_at_replay(table) -> None:
    """The decode step's lookup runs inside its CUDA graph: replay reads the ids the
    buffer holds at replay time."""
    spec = op.CELLS[0].spec
    ids = _ids(dict(spec, tokens=8), seed=_seed("graph"))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(ids, table)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(ids, table)

    for kind in ("random", "edges"):
        ids.copy_(_ids(dict(tokens=8, ids=kind), seed=_seed(f"graph {kind}")))
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _compare(captured, op.reference(ids, table))


def test_the_output_follows_the_id_shape(table) -> None:
    """Any id shape gives `ids.shape + (hidden,)`; no ids give an empty result without a
    launch."""
    ids = _ids(dict(tokens=12, ids="edges"), seed=_seed("shape"))
    flat = op(ids, table)
    grid = op(ids.view(3, 4), table)
    assert grid.shape == (3, 4, HIDDEN)
    _compare(grid.view(12, HIDDEN), flat)
    assert op(ids[:0], table).shape == (0, HIDDEN)


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda t: (torch.arange(4, device="cuda").float(), t), "integer"),
        (lambda t: (torch.arange(8, device="cuda", dtype=torch.int32)[::2], t), "contiguous"),
        (lambda t: (torch.arange(4, device="cuda", dtype=torch.int32), t[:64].t()), "stride"),
    ],
    ids=["float ids", "strided ids", "strided table"],
)
def test_guard_refuses_what_the_kernel_reads_silently(make, match, table) -> None:
    """Float ids are truncated to an index, a strided 1-D id view is read as if dense, a
    column-strided table is read as if contiguous: each returns plausible rows."""
    input_ids, weight = make(table)
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(input_ids, weight)
