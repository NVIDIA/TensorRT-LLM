# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check selection, bias semantics, tie policy, fallback and graph replay."""

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenRoutingMethod,
    _routing_weights,
)


def _router(bias):
    return MuseSparkOpenRoutingMethod(
        top_k=8, num_experts=256, callable_e_score_correction_bias=lambda: bias, renorm_eps=1e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 2, 16, 128])
@pytest.mark.parametrize("tied", [False, True])
def test_full_routing_selection_and_weights(tokens, tied):
    torch.manual_seed(910)
    logits = torch.randn(tokens, 256, device="cuda", dtype=torch.float32)
    bias = torch.randn(256, device="cuda", dtype=torch.float32) * 0.03
    if tied:
        logits.zero_()
        bias.zero_()
    router = _router(bias)
    selected, weights = router.apply(logits)
    scores = logits.sigmoid()
    if tied:
        expected = torch.arange(8, device="cuda").expand(tokens, -1)
    else:
        expected = torch.topk(scores + bias, 8, sorted=True).indices
    torch.testing.assert_close(selected.long(), expected, rtol=0, atol=0)
    raw = scores.gather(1, expected)
    torch.testing.assert_close(
        weights, raw / (raw.sum(-1, keepdim=True) + 1e-6), rtol=1e-5, atol=1e-6
    )
    router.clear_routing_memo()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = router.apply(logits)
    router.clear_routing_memo()
    logits.normal_()
    want = router.apply(logits)
    graph.replay()
    torch.cuda.synchronize()
    for actual, reference in zip(got, want):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_full_routing_cpu_fallback():
    logits = torch.randn(2, 256)
    bias = torch.randn(256)
    router = _router(bias)
    ids, weights = router.apply(logits)
    scores = logits.sigmoid()
    expected = torch.topk(scores + bias, 8, sorted=False).indices
    torch.testing.assert_close(ids.long(), expected)
    raw = scores.gather(1, expected)
    torch.testing.assert_close(weights, raw / (raw.sum(-1, keepdim=True) + 1e-6))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [256, 512, 4096, 32768])
def test_large_batch_routing_mutable_graph(monkeypatch, tokens):
    monkeypatch.setenv("MUSESPARK_OPEN_FULL_ROUTING_ALL_BATCHES", "1")
    bias = torch.zeros(256, device="cuda")
    logits = torch.zeros(tokens, 256, device="cuda")
    router = _router(bias)
    # The setting is resolved at construction and cannot change captured work.
    monkeypatch.setenv("MUSESPARK_OPEN_FULL_ROUTING_ALL_BATCHES", "0")
    router.apply(logits)
    router.clear_routing_memo()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected, weights = router.apply(logits)
    for seed in range(4):
        torch.manual_seed(seed)
        # Quantized logits create exact ties with well-separated distinct scores.
        # Stable sorting independently expresses the lowest-ID tie policy.
        logits.copy_(torch.randint(-512, 513, logits.shape, device="cuda") / 256)
        if seed == 0:
            logits.zero_()
        graph.replay()
        scores = logits.sigmoid()
        expected = scores.argsort(dim=-1, descending=True, stable=True)[:, :8]
        torch.testing.assert_close(selected.long(), expected, rtol=0, atol=0)
        raw = scores.gather(1, expected)
        torch.testing.assert_close(
            weights, raw / (raw.sum(-1, keepdim=True) + 1e-6), rtol=1e-5, atol=1e-6
        )


@triton.jit
def _reference_argmax_routing_kernel(logits, bias, selected, weights, eps, EXPERTS: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.arange(0, EXPERTS)
    slots = tl.arange(0, 8)
    scores = tl.sigmoid(tl.load(logits + row * EXPERTS + expert))
    ranked = scores + tl.load(bias + expert)
    ids = tl.full((8,), 0, tl.int32)
    values = tl.full((8,), 0.0, tl.float32)
    for k in tl.static_range(8):
        best = tl.argmax(ranked, axis=0, tie_break_left=True)
        value = tl.sum(tl.where(expert == best, scores, 0.0), axis=0)
        ids = tl.where(slots == k, best, ids)
        values = tl.where(slots == k, value, values)
        ranked = tl.where(expert == best, -float("inf"), ranked)
    values = tl.div_rn(values, tl.sum(values, axis=0) + eps)
    tl.store(selected + row * 8 + slots, ids)
    tl.store(weights + row * 8 + slots, values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [256, 512, 4096, 32768])
@torch.inference_mode()
def test_large_routing_torch_order(monkeypatch, tokens):
    monkeypatch.setenv("MUSESPARK_OPEN_FULL_ROUTING_ALL_BATCHES", "1")
    monkeypatch.setenv("MUSESPARK_OPEN_FULL_ROUTING_TORCH_ORDER", "1")
    logits = torch.zeros(tokens, 256, device="cuda")
    bias = torch.zeros(256, device="cuda")
    router = _router(bias)
    router.apply(logits)
    router.clear_routing_memo()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = router.apply(logits)
    for seed in range(8):
        torch.manual_seed(seed)
        logits.normal_()
        bias.normal_(0, 0.03)
        if seed == 0:
            logits.zero_()
            bias.zero_()
        elif seed == 1:
            logits.copy_(torch.randint(-512, 513, logits.shape, device="cuda") / 256)
            bias.zero_()
        elif seed == 2:
            logits.mul_(1e-6)
            bias.zero_()
        graph.replay()
        scores, ranked = router.get_scores(logits, bias)
        ids = torch.topk(ranked, 8, sorted=False).indices
        expected = _routing_weights(scores, ids, 1e-6, torch.float32)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 2, 16, 128])
@torch.inference_mode()
def test_sorted_routing_matches_original_on_mutable_inputs(tokens):
    logits = torch.randn(tokens, 256, device="cuda")
    bias = torch.randn(256, device="cuda")
    router = _router(bias)
    router.apply(logits)
    router.clear_routing_memo()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = router.apply(logits)
    for seed in range(32):
        torch.manual_seed(seed)
        logits.normal_()
        bias.normal_(0, 0.01)
        if seed == 0:
            logits.zero_()
            bias.fill_(-0.5)
        if seed == 1:
            logits.mul_(1e-6)
            bias.zero_()
        if seed == 2:
            bias.normal_(-1, 0.01)
        if seed == 3:
            logits.fill_(100)
            bias.zero_()
        if seed == 4:
            logits.floor_()
            bias.zero_()
        graph.replay()
        expected = (torch.empty_like(actual[0]), torch.empty_like(actual[1]))
        _reference_argmax_routing_kernel[(tokens,)](
            logits, bias, *expected, 1e-6, EXPERTS=256, num_warps=4
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=1e-7)
