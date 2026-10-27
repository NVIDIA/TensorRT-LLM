# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAIL16_G3: block draft probabilities stored with minimax_m31_scatter_rows.
CPU: the stock fallback and the kill switch. GPU (SM107, op built): bitwise the stock index_put store (random, NaN, -inf
rows; K 3 and 7; padded last dim), and under graph replay."""

import os
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    dspark_s4 as w1,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.spec import (  # noqa: E501
    m31_markov_epilogue as epi,
)
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase


def test_g3_is_on_by_default():
    assert w1.TAIL16_G3


def test_kill_switch():
    code = f"from {w1.__name__} import TAIL16_G3\nprint(TAIL16_G3)"
    env = {**os.environ, "TRTLLM_M31_DISABLE_TAIL16_G3": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-1] == "False"


def test_unsupported_dtype_takes_the_stock_store(monkeypatch):
    seen = []
    monkeypatch.setattr(
        SpecWorkerBase, "_store_block_draft_probs", staticmethod(lambda *a: seen.append(1))
    )
    monkeypatch.setattr(w1, "markov_epilogue_op", lambda: object())
    w1.S4WorkerMixin._store_block_draft_probs(torch.zeros(8, 3, 16, dtype=torch.float64), torch.tensor([1, 2]),
                                              torch.zeros(2, 3, 16, dtype=torch.float64))  # fmt: skip
    assert seen == [1]


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or epi.markov_epilogue_op() is None,
    reason="the TAIL16_G3 op runs on SM107 with the op built",
)


@_GPU
@pytest.mark.parametrize("k", [3, 7])
@pytest.mark.parametrize("pad", [0, 8])
def test_bitwise_the_stock_store(k, pad):
    v = 200064
    dst0 = torch.randn(64, 7, v + pad, device="cuda")
    probs = torch.rand(4, k, v, device="cuda")
    probs[0, 0, 5] = float("nan")
    probs[1, -1, :100] = float("-inf")
    slots = torch.tensor([3, 17, 40, 9], dtype=torch.int64, device="cuda")
    ref, got = dst0.clone(), dst0.clone()
    SpecWorkerBase._store_block_draft_probs(ref, slots, probs)
    w1.S4WorkerMixin._store_block_draft_probs(got, slots, probs)
    assert torch.equal(ref.view(torch.int32), got.view(torch.int32))


@_GPU
def test_graph_replay():
    v = 200064
    dst0 = torch.zeros(64, 3, v, device="cuda")
    probs = torch.rand(4, 3, v, device="cuda")
    slots = torch.tensor([3, 17, 40, 9], dtype=torch.int64, device="cuda")
    ref = dst0.clone()
    SpecWorkerBase._store_block_draft_probs(ref, slots, probs)
    got = dst0.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        w1.S4WorkerMixin._store_block_draft_probs(got, slots, probs)
    torch.cuda.current_stream().wait_stream(s)
    got.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(ref.view(torch.int32), got.view(torch.int32))


def test_cpu_fp32_uses_stock(monkeypatch):
    monkeypatch.setattr(w1, "markov_epilogue_op", lambda: object())
    dst = torch.zeros(4, 3, 16)
    src = torch.randn(2, 3, 16)
    slots = torch.tensor([3, 1])
    ref = dst.clone()
    SpecWorkerBase._store_block_draft_probs(ref, slots, src)
    w1.S4WorkerMixin._store_block_draft_probs(dst, slots, src)
    assert torch.equal(ref, dst)


def test_old_native_library_uses_stock(monkeypatch):
    monkeypatch.setattr(w1, "markov_epilogue_op", lambda: object())
    monkeypatch.setattr(torch.ops.trtllm, "minimax_m31_scatter_rows", None, raising=False)
    seen = []
    monkeypatch.setattr(
        SpecWorkerBase, "_store_block_draft_probs", staticmethod(lambda *a: seen.append(a))
    )
    w1.S4WorkerMixin._store_block_draft_probs(None, None, None)
    assert seen == [(None, None, None)]


@_GPU
@pytest.mark.parametrize(
    "n,k,v,pad,offset",
    [
        (56, 7, 200064, 8, 0),
        (112, 3, 200064, 0, 0),
        (9, 3, 16383, 1, 0),
        (9, 3, 16385, 3, 0),
        (9, 7, 16384, 0, 1),
        (9, 7, 16388, 1, 0),
    ],
)
def test_wide_and_scalar_layouts(n, k, v, pad, offset):
    assert getattr(torch.ops.trtllm, "minimax_m31_scatter_rows", None) is not None
    backing = torch.zeros((n + 4) * (k + 1) * (v + pad) + offset, device="cuda")
    dst = backing[offset:].view(n + 4, k + 1, v + pad)
    src_backing = torch.randn(n * k * v + offset, device="cuda")
    src = src_backing[offset:].view(n, k, v)
    src.view(-1)[0:4] = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), -0.0], device="cuda"
    )
    slots = torch.randperm(n + 4, device="cuda")[:n]
    ref = dst.clone()
    SpecWorkerBase._store_block_draft_probs(ref, slots, src)
    w1.S4WorkerMixin._store_block_draft_probs(dst, slots, src)
    assert torch.equal(ref.view(torch.int32), dst.view(torch.int32))


@_GPU
def test_graph_changed_inputs_and_duplicate_padding():
    assert getattr(torch.ops.trtllm, "minimax_m31_scatter_rows", None) is not None
    n, k, v = 56, 3, 16385
    dst = torch.zeros(n + 4, k + 1, v + 1, device="cuda")
    src = torch.randn(n, k, v, device="cuda")
    slots = torch.arange(n, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        w1.S4WorkerMixin._store_block_draft_probs(dst, slots, src)
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (17, 83, 137):
        torch.manual_seed(seed)
        src.normal_()
        slots.copy_(torch.randperm(n + 4, device="cuda")[:n])
        # Graph padding can share a dummy slot; identical source bits are required.
        slots[-1].copy_(slots[-2])
        src[-1].copy_(src[-2])
        dst.zero_()
        ref = dst.clone()
        SpecWorkerBase._store_block_draft_probs(ref, slots, src)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ref.view(torch.int32), dst.view(torch.int32))
