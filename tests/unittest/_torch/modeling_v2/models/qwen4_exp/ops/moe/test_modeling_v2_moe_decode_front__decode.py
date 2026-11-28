# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`moe_decode_front` (`ops/moe/moe_decode_front__decode`): router GEMM, routing, pre-routed
tables and NVFP4 activation of a small decode MoE block in one launch.

GPU, every cell, eagerly and replayed from a CUDA graph:

* the logits are within `logits_off_bound` of the exact GEMM rounded once: two bf16 steps plus
  an fp32 sum's worst-case reorder error (the class this op is in: another summation order than
  other router GEMMs);
* given those logits, the codes and block scales are bitwise `fp4_quantize`'s, the experts and
  weights bitwise `renorm_moe_routing_op`'s, and the tables hold their meaning
  (`moe_route_front__decode.check_tables`) for the cell's window and tile;
* the staging buffer is back at zero after every launch (the next launch relies on it).

The window output through the runner entered pre-routed is held to the route front's by the
target's build check (`modeling._tep_decode_front_check`).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import moe_decode_front__decode as op
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import (
    moe_route_front__decode as route,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the decode front is certified on SM103",
)


def _operands(spec, seed: int, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, hidden = spec["rows"], spec["hidden"]
    x = torch.randn([rows, hidden], generator=gen, device=device).to(torch.bfloat16)
    router = (torch.randn([op.NUM_EXPERTS, hidden], generator=gen, device=device) * 0.02).to(
        torch.bfloat16
    )
    global_scale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device=device)
    n = sum(route.table_sizes(rows, op.TOP_K, spec["num_local"], spec["tile"]))
    tables = (torch.arange(n, device=device) % rows).to(torch.int32)
    return x, router, global_scale, tables


def _assert_matches(spec, x, router, global_scale, tables, stage, got):
    logits, xq, xsf, ids, wts = got
    off = op.logits_off_bound(logits, x, router)
    assert not bool(off.any()), f"logits: {int(off.sum())} outside the bound"
    _, rxq, rxsf, _, _ = op.moe_decode_front.reference(
        x,
        router,
        global_scale,
        tables,
        stage,
        spec["tile"],
        spec["local_offset"],
        spec["num_local"],
    )
    rids, rwts = renorm_moe_routing_op(logits, op.TOP_K, torch.bfloat16)
    for name, a, b in (("codes", xq, rxq), ("scales", xsf, rxsf), ("experts", ids, rids)):
        assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)), name
    assert torch.equal(wts.view(torch.int16), rwts.view(torch.int16)), "weights"
    views = route.table_views(tables, x.shape[0], op.TOP_K, spec["num_local"], spec["tile"])
    why = route.check_tables(views, ids, spec["local_offset"], spec["num_local"], spec["tile"])
    assert not why, why
    assert not bool(stage.any()), "the staging buffer is not back at zero"


@_GPU
@pytest.mark.parametrize("cell", op.moe_decode_front.CELLS, ids=lambda c: c.why[:40])
def test_certified_cells(cell):
    spec, device = cell.spec, torch.device("cuda")
    seed = zlib.crc32(cell.why.encode()) % 2**31
    x, router, global_scale, tables = _operands(spec, seed, device)
    assert op.compile_front(
        spec["rows"], spec["hidden"], x.stride(0), spec["num_local"], spec["tile"]
    )
    stage = op.staging_buffer(device)
    args = (spec["tile"], spec["local_offset"], spec["num_local"])
    got = op.moe_decode_front(x, router, global_scale, tables, stage, *args)
    torch.cuda.synchronize()
    _assert_matches(spec, x, router, global_scale, tables, stage, got)

    # Captured once, replayed back to back over fresh inputs in the static buffers.
    s_x = x.clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        out = op.moe_decode_front(s_x, router, global_scale, tables, stage, *args)
    torch.cuda.current_stream().wait_stream(stream)
    for k in range(3):
        x, _, _, _ = _operands(spec, seed + 1 + k, device)
        s_x.copy_(x)
        graph.replay()
        # Stream-ordered snapshots between the two replays (no host synchronization): what the
        # first replay left, including its staging buffer, which the second must not be the one
        # to clear.
        first = [t.clone() for t in out]
        first_stage, first_tables = stage.clone(), tables.clone()
        graph.replay()
        torch.cuda.synchronize()
        _assert_matches(spec, s_x, router, global_scale, first_tables, first_stage, first)
        _assert_matches(spec, s_x, router, global_scale, tables, stage, out)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rows, hidden, stride, local",
    [
        (17, 2560, 2560, 128),  # past MAX_ROWS
        (4, 2048, 2048, 128),  # K not a multiple of 64 x K_WARPS
        (4, 2560, 2564, 128),  # rows not 16-byte aligned
        (4, 2560, 2560, 104),  # a window not a multiple of 32
    ],
)
def test_compile_declines_shapes_the_kernel_does_not_take(rows, hidden, stride, local):
    assert op.compile_front(rows, hidden, stride, local, 8) is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("tile", [0, 4, 12, 512])
def test_compile_declines_tiles_the_kernel_does_not_take(tile):
    """A tile outside the kernel's powers of two in [8, 256] is declined, not raised."""
    assert op.compile_front(4, 2560, 2560, 128, tile) is False


@pytest.mark.cpu_only
def test_the_reference_rounds_the_exact_sum_once():
    """`1 + 2^-8 + 2^-25` lies just above the midpoint of 1 and `1 + 2^-7`: rounded once it is
    `1 + 2^-7`; through fp32 it would come back as 1."""
    exact = torch.tensor([1 + 2.0**-8 + 2.0**-25], dtype=torch.float64)
    assert float(op.round_to_bf16(exact)) == 1 + 2.0**-7
    x = torch.tensor([[1.0, 2.0**-8, 2.0**-25]], dtype=torch.bfloat16)
    router = torch.ones(1, 3, dtype=torch.bfloat16)
    assert float(op.reference_logits(x, router)) == 1 + 2.0**-7
    # ties to even, negative values, zero and the largest finite bf16 round as torch does there
    values = torch.tensor(
        [1 + 2.0**-8, 1 + 3 * 2.0**-8, -(1 + 2.0**-8 + 2.0**-25), 0.0, 3.3895313892515355e38, -7.5],
        dtype=torch.float64,
    )
    want = torch.tensor([1.0, 1 + 2.0**-6, -(1 + 2.0**-7), 0.0, 3.3895313892515355e38, -7.5])
    assert torch.equal(op.round_to_bf16(values), want)


@pytest.mark.cpu_only
def test_a_nan_logit_is_off_the_bound():
    """A NaN or infinite logit where the exact one is finite is off the bound."""
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(2, 64, generator=gen).to(torch.bfloat16)
    router = (torch.randn(8, 64, generator=gen) * 0.02).to(torch.bfloat16)
    logits = op.reference_logits(x, router).to(torch.bfloat16)
    assert not bool(op.logits_off_bound(logits, x, router).any())
    logits[1, 3] = float("nan")
    logits[0, 5] = float("inf")
    off = op.logits_off_bound(logits, x, router)
    assert off.nonzero().tolist() == [[0, 5], [1, 3]]
