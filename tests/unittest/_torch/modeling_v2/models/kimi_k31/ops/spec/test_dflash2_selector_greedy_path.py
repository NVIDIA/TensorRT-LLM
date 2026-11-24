# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k31_dflash2_selector_greedy_path entry: every cell bitwise against its reference, the
selector's edge scores (``dflash2_score_edges``), the greedy walk (``dflash2_walk_candidate_paths``) and the
``-inf`` block with the realized scores at the candidates."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_selector_greedy_path import (  # noqa: E501
    k31_dflash2_selector_greedy_path as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: F401 — needs the path declared above

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (103, 107),
    reason="the K3.1 DFlash2 selector entry is certified on sm_103 and sm_107",
)

VOCAB = 163840
RANK = 256
HIDDEN = 7168
_CODEBOOKS = {}


def _codebooks():
    if not _CODEBOOKS:
        g = torch.Generator(device="cuda").manual_seed(7)
        for name in ("pred", "succ"):
            _CODEBOOKS[name] = (0.05 * torch.randn(VOCAB, RANK, generator=g, device="cuda")).to(
                torch.bfloat16
            )
    return _CODEBOOKS["pred"], _CODEBOOKS["succ"]


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, slots = spec["n"], spec["slots"]
    pred_cb, succ_cb = _codebooks()
    weight = (0.02 * torch.randn(RANK, HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    cand = torch.stack(
        [torch.randperm(VOCAB, generator=g, device="cuda")[:32] for _ in range(n * slots)]
    ).view(n, slots, 32)
    unary = (
        (10.0 + 2.0 * torch.randn(n, slots, 32, generator=g, device="cuda"))
        .sort(dim=-1, descending=True)
        .values
    )
    if spec["scores"] == "ties":
        unary = torch.zeros_like(unary)
        weight = torch.zeros_like(weight)
    hidden = torch.randn(n, slots, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (n,), generator=g, device="cuda", dtype=torch.int32)
    block = torch.randn(n, slots, VOCAB, generator=g, device="cuda").to(spec["block"])
    return [pred_cb, succ_cb, weight, cand, unary.to(spec["unary"]), hidden, anchor, block]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    ref_args = list(args)
    ref_args[7] = args[7].clone()
    got = op(*args)
    expected = op.reference(*ref_args)
    assert got.data_ptr() == args[7].data_ptr()
    assert torch.equal(got, expected)


def test_guard_refuses_a_repeated_candidate() -> None:
    args = _build(op.CELLS[0].spec, seed=1)
    args[3][0, 0, 1] = args[3][0, 0, 0]
    with pytest.raises(AssertionError, match="repeated candidate id"):
        with validating(op):
            op(*args)


def test_cuda_graph_replay() -> None:
    args = _build(op.CELLS[1].spec, seed=8)
    op(*args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            out = op(*args)
    torch.cuda.current_stream().wait_stream(stream)
    fresh = _build(op.CELLS[1].spec, seed=9)
    for i in (3, 4, 5, 6):
        args[i].copy_(fresh[i])
    graph.replay()
    ref_args = list(args)
    ref_args[7] = torch.empty_like(args[7])
    expected = op.reference(*ref_args)
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
