# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP8 CTXLEN_FUSE on the GPU: the one-launch resume update is bitwise the in-tree eager lines (random slots, stored
masks, clamps, int32 / int64 positions, strided positions), eagerly and replayed from a CUDA graph."""

import importlib.util
import random
from pathlib import Path

import pytest
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
KERNEL = (
    ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8"
    / "kernels/m3_ctxlen_resume.py"
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)


@pytest.fixture(scope="module")
def mod():
    pytest.importorskip("cuda.bindings")
    spec = importlib.util.spec_from_file_location("_tp8_ctxlen_resume_gpu", KERNEL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    reason = m.prepare(torch.cuda.current_device())
    assert reason is None, reason
    return m


def _case(rng, pos_dtype, pos_step):
    num_slots = rng.randint(1, 300)
    n = rng.randint(1, num_slots)
    stride = rng.randint(1, 17)
    first = rng.randint(0, 64)
    max_ctx = rng.choice([1, 8, 4096, 262144, 1 << 20, (1 << 31) - 1])
    npos = first + stride * n + rng.randint(0, 8)
    base = torch.randint(0, 1 << 22, (npos * pos_step,), dtype=torch.int64, device="cuda")
    pos = base.to(pos_dtype)[::pos_step]
    slots = torch.tensor(rng.sample(range(num_slots), n), dtype=torch.int64, device="cuda")
    stored = torch.rand(num_slots, device="cuda") < rng.random()
    ctx_len = torch.randint(0, 1 << 22, (num_slots,), dtype=torch.int64, device="cuda")
    return pos, slots, stored, ctx_len, first, stride, n, max_ctx


@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("pos_step", [1, 3])
def test_bitwise_vs_eager(mod, pos_dtype, pos_step):
    rng = random.Random(7 + pos_step)
    for _ in range(200):
        pos, slots, stored, ctx_len, first, stride, n, max_ctx = _case(rng, pos_dtype, pos_step)
        want = ctx_len.clone()
        got = ctx_len.clone()
        mod.ctxlen_resume_eager_(pos, slots, stored, want, first, stride, n, max_ctx)
        assert mod.outside_contract(pos, slots, stored, got) is None
        mod.ctxlen_resume_(pos, slots, stored, got, first, stride, n, max_ctx)
        torch.cuda.synchronize()
        assert torch.equal(got, want)


def test_slice_of_slots_and_graph_replay(mod):
    """Slots as a slice of a persistent batch-to-slot table (as the worker passes them), captured once and replayed
    with new positions / stored flags / lengths written into the same buffers."""
    rng = random.Random(99)
    num_slots, n, stride, first, max_ctx = 64, 8, 8, 0, 1 << 20
    table = torch.zeros(32, dtype=torch.int64, device="cuda")
    pos = torch.zeros(first + stride * n, dtype=torch.int32, device="cuda")
    stored = torch.zeros(num_slots, dtype=torch.bool, device="cuda")
    ctx_len = torch.zeros(num_slots, dtype=torch.int64, device="cuda")
    slots = table[4 : 4 + n]

    def fill():
        table.copy_(torch.tensor(rng.sample(range(num_slots), 32), device="cuda"))
        pos.copy_(torch.randint(0, 1 << 21, pos.shape, device="cuda", dtype=torch.int32))
        stored.copy_(torch.rand(num_slots, device="cuda") < 0.5)
        ctx_len.copy_(torch.randint(0, 1 << 21, (num_slots,), device="cuda"))

    fill()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mod.ctxlen_resume_(pos, slots, stored, ctx_len, first, stride, n, max_ctx)  # warm
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mod.ctxlen_resume_(pos, slots, stored, ctx_len, first, stride, n, max_ctx)
    for _ in range(50):
        fill()
        want = ctx_len.clone()
        mod.ctxlen_resume_eager_(pos, slots.clone(), stored, want, first, stride, n, max_ctx)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ctx_len, want)


def test_contract_rejects(mod):
    pos = torch.zeros(16, dtype=torch.int32, device="cuda")
    slots = torch.zeros(2, dtype=torch.int64, device="cuda")
    stored = torch.zeros(4, dtype=torch.bool, device="cuda")
    ctx_len = torch.zeros(4, dtype=torch.int64, device="cuda")
    assert mod.outside_contract(pos.view(1, 16), slots, stored, ctx_len) is not None
    assert mod.outside_contract(pos, slots, stored.to(torch.uint8), ctx_len) is not None
    assert mod.outside_contract(pos.float(), slots, stored, ctx_len) is not None
    assert mod.outside_contract(pos.cpu(), slots, stored, ctx_len) is not None
    assert (
        mod.outside_contract(pos, slots, stored, ctx_len.to(torch.int32)) is not None
    )  # not compiled
