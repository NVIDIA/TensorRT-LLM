# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the uGPU target head (ops/gemm/m3_ugpu_head): the die-placed copy of the head shard is the weight bit
for bit with its halves in the two locality domains; the module's own weight (what every other reader of the head
uses) is never placed; the per-die split GEMM gives the in-tree head's logits bit for bit at the certified row counts,
eager and in a CUDA graph; an in-place change of the weight makes the copy stale."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.gemm import (  # noqa: E501
    m3_ugpu_head as op,
)
from tensorrt_llm._torch.modules.linear import Linear

assert torch.cuda.is_available(), "the uGPU head requires a CUDA device"


def _locality():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.m3_locality import (  # noqa: E501
        LocalityMap,
    )

    try:
        return LocalityMap.get(torch.cuda.current_device())
    except RuntimeError as err:  # one locality domain (not a two-die GPU)
        pytest.skip(f"no two locality domains: {err}")


@pytest.fixture(scope="module")
def placed():
    lm = _locality()
    head = Linear(op.HIDDEN, op.TARGET_SHARD, bias=False, dtype=torch.bfloat16)
    head.to("cuda")
    g = torch.Generator(device="cuda").manual_seed(20261015)
    with torch.no_grad():
        head.weight.copy_(
            (0.02 * torch.randn(op.TARGET_SHARD, op.HIDDEN, generator=g, device="cuda")).to(
                torch.bfloat16
            )
        )
    before = head.weight.detach().clone()
    x = torch.randn(16, op.HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    stock = {m: head.apply_linear(x[:m], None) for m in (4, 8, 12, 16)}
    ugpu = op.prepare_head(head)
    assert ugpu is not None and ugpu.source is head.weight and op.current(ugpu, head.weight)
    return lm, head, ugpu, before, x, stock


def test_no_reader_of_the_module_sees_placed_memory(placed) -> None:
    """A kernel reading die-placed memory from all SMs runs 1.5-1.7x slower: every tensor of the head module (what
    the in-tree GEMM, prefill logits, warmup and fallbacks read) must stay in ordinary memory; only the op's copy is
    placed."""
    lm, head, ugpu, before, _, _ = placed
    for name, t in list(head.named_parameters()) + list(head.named_buffers()):
        end = t.data_ptr() + t.numel() * t.element_size() - 1
        assert lm.domain_of(t.data_ptr()) == -1 and lm.domain_of(end) == -1, name
        assert t.data_ptr() != ugpu.weight.data_ptr(), name
    assert torch.equal(head.weight, before)


def test_placement_is_exact_and_per_die(placed) -> None:
    lm, head, ugpu, before, _, _ = placed
    w = ugpu.weight
    assert torch.equal(w, before)
    row = w.shape[1] * w.element_size()
    assert lm.domain_of(w.data_ptr()) == 0
    assert lm.domain_of(w.data_ptr() + ugpu.split * row - 1) == 0
    assert lm.domain_of(w.data_ptr() + ugpu.split * row) == 1
    assert lm.domain_of(w.data_ptr() + w.numel() * w.element_size() - 1) == 1
    assert (ugpu.split * row) % lm.granularity == 0 and abs(2 * ugpu.split - w.shape[0]) < 2 * 512


@pytest.mark.parametrize(
    "cell", op.m3_ugpu_head.CELLS, ids=[c.why[:44] for c in op.m3_ugpu_head.CELLS]
)
def test_split_head_is_the_in_tree_head_bit_for_bit(placed, cell) -> None:
    m = cell.spec["m"]
    _, head, ugpu, _, x, stock = placed
    got = op.m3_ugpu_head(x[:m], ugpu)
    torch.cuda.synchronize()
    assert torch.equal(got, stock[m])
    assert torch.equal(got, op.m3_ugpu_head.reference(x[:m], ugpu))
    assert torch.equal(
        head.apply_linear(x[:m], None), stock[m]
    )  # stock readers of the placed weight


@pytest.mark.parametrize(
    "cell", op.m3_ugpu_head.CELLS, ids=[c.why[:44] for c in op.m3_ugpu_head.CELLS]
)
def test_split_head_in_a_cuda_graph(placed, cell) -> None:
    m = cell.spec["m"]
    _, _, ugpu, _, x, stock = placed
    xs = x[:m].clone()
    box = {}
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        box["y"] = op.m3_ugpu_head(xs, ugpu)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        box["y"] = op.m3_ugpu_head(xs, ugpu)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(box["y"], stock[m])


def test_an_in_place_weight_change_makes_the_copy_stale(placed) -> None:
    _, head, ugpu, _, _, _ = placed
    assert op.current(ugpu, head.weight)
    with torch.no_grad():
        head.weight.add_(0)  # an in-place write bumps the version counter
    assert not op.current(ugpu, head.weight)
