# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 tp16 fused_embedding entry (sm_107).

A gather, so everything is gated bitwise: the cells against the entry's
reference, and the op against the Embedding module's own masked F.embedding
chain, which it replaces. Then CUDA-graph replay, id shapes, and the inputs the
op refuses.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.embedding.fused_embedding import (  # noqa: E501
    HIDDEN,
    VOCAB,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.embedding.fused_embedding import (  # noqa: E501
    fused_embedding as op,
)
from tensorrt_llm._torch.modules.embedding import pre_comm_embedding_ops
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's entries are certified on sm_107"
)

assert torch.cuda.is_available(), "fused_embedding requires a CUDA device"

#: model.embed_tokens.weight's standard deviation in the checkpoint.
TABLE_STD = 0.0076


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


@pytest.fixture(scope="module")
def table():
    """The replicated [154880, 6144] bf16 table (1.9 GB), built once per module."""
    g = torch.Generator(device="cuda").manual_seed(_seed("table"))
    weight = torch.randn(VOCAB, HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    yield weight.mul_(TABLE_STD)
    del weight
    torch.cuda.empty_cache()


def _ids(spec, seed):
    """`random`: uniform in-vocab ids. `edges`: the first and last rows, the -1
    padding, ids at and past the vocab, the id type's extremes, then random."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(0, VOCAB, (spec["tokens"],), generator=g, device="cuda")
    if spec["ids"] == "edges":
        if spec["ids_dtype"] == torch.int64:
            extremes = [2**33, -(2**40)]
        else:
            extremes = [2**31 - 1, -(2**31)]
        edges = torch.tensor([-1, 0, VOCAB - 1, VOCAB, VOCAB + 1000, *extremes], device="cuda")
        ids[: edges.numel()] = edges
    return ids.to(spec["ids_dtype"])


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise, compared as raw 16-bit patterns: rows are copies, so there is
    no tolerance to give, and a raw compare also holds for -0.0 and NaN rows."""
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
    assert torch.equal(ids, before), "the ids are read-only"
    _compare(out, op.reference(ids, table))


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_bitwise_equal_to_the_embedding_module_chain(ids_dtype, table) -> None:
    """The masked F.embedding chain the Embedding module runs for an unsharded
    table, with its out-of-range guard: the op computes exactly that."""
    ids = _ids(dict(tokens=64, ids_dtype=ids_dtype, ids="edges"), seed=_seed("chain"))
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


def test_cuda_graph_capture_and_replay(table) -> None:
    """The verify forward's lookup runs inside the step's CUDA graph: replay
    reads the ids the buffer holds at replay time."""
    spec = op.CELLS[0].spec
    ids = _ids(spec, seed=_seed("graph"))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            op(ids, table)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(ids, table)

    for seed, kind in (("graph random", "random"), ("graph edges", "edges")):
        ids.copy_(_ids(dict(spec, ids=kind), seed=_seed(seed)))
        graph.replay()
        torch.cuda.synchronize()
        _compare(captured, op.reference(ids, table))


def test_the_output_follows_the_id_shape(table) -> None:
    """Any id shape gives `ids.shape + (hidden,)`; strided ids are made
    contiguous first; no ids give an empty result without a launch."""
    ids = _ids(dict(tokens=12, ids_dtype=torch.int64, ids="edges"), seed=_seed("shape"))
    flat = op(ids, table)
    grid = op(ids.view(3, 4), table)
    assert grid.shape == (3, 4, HIDDEN)
    _compare(grid.view(12, HIDDEN), flat)
    _compare(op(ids[::2], table), flat[::2].contiguous())
    assert op(ids[:0], table).shape == (0, HIDDEN)


def _small_table(dtype=torch.bfloat16, hidden=64):
    return torch.randn(32, hidden, device="cuda").to(dtype)


def _small_ids():
    return torch.arange(4, device="cuda", dtype=torch.int32)


@pytest.mark.parametrize(
    "make",
    [
        lambda: (_small_ids().float(), _small_table()),
        lambda: (_small_ids().cpu(), _small_table()),
        lambda: (_small_ids(), _small_table(torch.float16)),
        lambda: (_small_ids(), _small_table(hidden=128)[:, :64]),
        lambda: (_small_ids(), _small_table(hidden=60)),
    ],
    ids=["float ids", "cpu ids", "fp16 table", "strided table", "hidden 60"],
)
def test_inputs_outside_the_domain_raise(make) -> None:
    """Loud, never a silently wrong answer: these are why `is_valid` has
    nothing left to refuse."""
    input_ids, weight = make()
    with pytest.raises(RuntimeError):
        with validating(op):
            op(input_ids, weight)
