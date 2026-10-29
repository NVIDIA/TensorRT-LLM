# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for m3_spec_offsets (RuntimeHooks.spec_offsets_op): the overlap scheduler's two speculative step corrections
in one plain launch, bit for bit the in-place ATen adds, eager and in a CUDA graph."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.spec.m3_spec_offsets import (  # noqa: E501
    m3_spec_offsets as op,
)

assert torch.cuda.is_available(), "m3_spec_offsets requires a CUDA device"

I32 = torch.iinfo(torch.int32)


def _inputs(tokens, seqs, seed, wrap=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    lo, hi = (I32.min, I32.max) if wrap else (0, 1 << 20)
    pos = torch.randint(lo, hi, (tokens,), generator=g, device="cuda", dtype=torch.int32)
    kv = torch.randint(lo, hi, (seqs,), generator=g, device="cuda", dtype=torch.int32)
    pos_off = torch.randint(
        -8 if not wrap else lo,
        8 if not wrap else hi,
        (tokens,),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    )
    kv_off = torch.randint(
        -8 if not wrap else lo,
        8 if not wrap else hi,
        (seqs,),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    )
    if wrap:
        pos[:2], pos_off[:2] = I32.max, 1  # overflow wraps, as the ATen add does
        kv[:1], kv_off[:1] = I32.min, -1
    return pos, pos_off, kv, kv_off


CELLS = [c.spec for c in op.CELLS] + [
    dict(tokens=1, seqs=1),
    dict(tokens=300, seqs=75),
    dict(tokens=4, seqs=0),
]


@pytest.mark.parametrize("spec", CELLS, ids=lambda s: "-".join(f"{k}{v}" for k, v in s.items()))
def test_bitwise_aten_adds(spec) -> None:
    for trial in range(3):
        pos, pos_off, kv, kv_off = _inputs(
            spec["tokens"], spec["seqs"], 97 * trial + spec["tokens"], spec.get("wrap", False)
        )
        ref_pos, ref_kv = pos.clone(), kv.clone()
        op.reference(ref_pos, pos_off, ref_kv, kv_off)
        assert op(pos, pos_off, kv, kv_off) is True
        torch.cuda.synchronize()
        assert torch.equal(pos, ref_pos) and torch.equal(kv, ref_kv), f"trial {trial}"


def test_graph_replays_read_the_live_offsets() -> None:
    pos, pos_off, kv, kv_off = _inputs(32, 8, 5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(pos, pos_off, kv, kv_off)
    for step in range(3):
        fresh = _inputs(32, 8, 11 + step)
        for dst, src in zip((pos, pos_off, kv, kv_off), fresh):
            dst.copy_(src)
        ref_pos, ref_kv = pos.clone(), kv.clone()
        op.reference(ref_pos, pos_off, ref_kv, kv_off)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(pos, ref_pos) and torch.equal(kv, ref_kv), f"step {step}"


def test_runner_views_are_accepted() -> None:
    """The runner passes position_ids[0, ctx:] and kv_lens_cuda[ctx_reqs:seqs]: 1-D slices of contiguous buffers."""
    position_ids = torch.arange(40, dtype=torch.int32, device="cuda").view(1, 40)
    kv_lens = torch.arange(12, dtype=torch.int32, device="cuda")
    pos_off = torch.full((40,), 2, dtype=torch.int32, device="cuda")
    kv_off = torch.full((12,), 3, dtype=torch.int32, device="cuda")
    assert op(position_ids[0, 8:], pos_off[:32], kv_lens[2:10], kv_off[:8]) is True
    torch.cuda.synchronize()
    assert torch.equal(
        position_ids[0], torch.cat([torch.arange(8), torch.arange(8, 40) + 2]).int().cuda()
    )
    assert torch.equal(
        kv_lens,
        torch.cat([torch.arange(2), torch.arange(2, 10) + 3, torch.arange(10, 12)]).int().cuda(),
    )


@pytest.mark.parametrize(
    "bad",
    ["int64", "strided", "length", "2d"],
)
def test_declines_other_layouts_without_writing(bad) -> None:
    pos, pos_off, kv, kv_off = _inputs(8, 2, 3)
    if bad == "int64":
        kv = kv.long()
    elif bad == "strided":
        pos = torch.zeros(16, dtype=torch.int32, device="cuda")[::2]
    elif bad == "length":
        pos_off = pos_off[:7]
    else:
        pos = pos.view(2, 4)
    before = (pos.clone(), kv.clone())
    assert op(pos, pos_off, kv, kv_off) is False
    torch.cuda.synchronize()
    assert torch.equal(pos, before[0]) and torch.equal(kv, before[1])


def test_launches_plainly() -> None:
    """It writes step inputs that pre-wait readers take before their grid wait (MSA contract, dispatch
    PRODUCER_PREWAIT_STEP_INPUTS): its kernel node carries no programmatic (PDL) serialization."""
    from cuda.bindings import runtime as rt

    pos, pos_off, kv, kv_off = _inputs(32, 8, 7)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        op(pos, pos_off, kv, kv_off)
    raw = rt.cudaGraph_t(init_value=graph.raw_cuda_graph())
    err, _, n = rt.cudaGraphGetNodes(raw, 0)
    err, nodes, n = rt.cudaGraphGetNodes(raw, n)
    flags = []
    for node in nodes[:n]:
        err, kind = rt.cudaGraphNodeGetType(node)
        if kind == rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            err, attr = rt.cudaGraphKernelNodeGetAttribute(
                node, rt.cudaLaunchAttributeID.cudaLaunchAttributeProgrammaticStreamSerialization
            )
            flags.append(0 if attr is None else int(attr.programmaticStreamSerializationAllowed))
    assert flags == [0], flags
