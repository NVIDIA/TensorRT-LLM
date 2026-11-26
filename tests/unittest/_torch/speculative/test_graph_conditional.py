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
"""Merged greedy/sampling CUDA graphs: conditional-node capture without model weights."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import sampling_variant
from tensorrt_llm._torch.speculative import graph_conditional as gc

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and gc.conditional_nodes_available()),
    reason="needs a GPU and a torch that captures CUDA graph conditional nodes",
)

BATCH, VOCAB = 8, 512


@pytest.fixture
def inputs():
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(BATCH, VOCAB, device="cuda", generator=generator)
    noise = torch.rand(BATCH, VOCAB, device="cuda", generator=generator).clamp_min_(1e-9)
    return logits, noise


def _greedy(logits):
    return logits.argmax(dim=-1), torch.ones(BATCH, dtype=torch.int32, device="cuda")


def _sampled(logits, noise):
    gumbel = -torch.log(-torch.log(noise))
    return (logits + gumbel).argmax(dim=-1), torch.full((BATCH,), 2, dtype=torch.int32, device="cuda")


def _capture(pred, if_fn, else_fn, host_pred=True):
    for _ in range(2):  # eager warmup outside capture
        with gc.warm_up_both_branches():
            gc.graph_if_else(pred, if_fn, else_fn, host_pred=host_pred)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = gc.graph_if_else(pred, if_fn, else_fn, host_pred=host_pred)
    return graph, result


def test_replay_follows_in_place_predicate_flip(inputs):
    logits, noise = inputs
    greedy_tokens, greedy_count = _greedy(logits)
    sampled_tokens, sampled_count = _sampled(logits, noise)
    assert not torch.equal(greedy_tokens, sampled_tokens), "branches must differ"

    pred = torch.ones((), dtype=torch.bool, device="cuda")
    graph, (tokens, count) = _capture(
        pred, lambda: _greedy(logits), lambda: _sampled(logits, noise)
    )
    for value in (True, False, False, True, False, True):
        pred.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        expected_tokens, expected_count = (
            (greedy_tokens, greedy_count) if value else (sampled_tokens, sampled_count)
        )
        assert torch.equal(tokens, expected_tokens), f"pred={value}"
        assert torch.equal(count, expected_count), f"pred={value}"


def test_outputs_are_not_clobbered_by_other_allocations(inputs):
    logits, noise = inputs
    pred = torch.zeros((), dtype=torch.bool, device="cuda")
    graph, (tokens, _) = _capture(pred, lambda: _greedy(logits), lambda: _sampled(logits, noise))
    expected, _ = _sampled(logits, noise)
    for _ in range(3):
        scratch = [torch.randn(BATCH, VOCAB, device="cuda") * 3 for _ in range(4)]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(tokens, expected)
        del scratch


def test_eager_runs_only_the_host_selected_branch(inputs):
    logits, noise = inputs
    calls = []

    def if_fn():
        calls.append("if")
        return _greedy(logits)

    def else_fn():
        calls.append("else")
        return _sampled(logits, noise)

    pred = torch.ones((), dtype=torch.bool, device="cuda")
    gc.graph_if_else(pred, if_fn, else_fn, host_pred=True)
    gc.graph_if_else(pred, if_fn, else_fn, host_pred=False)
    assert calls == ["if", "else"]


def test_warmup_runs_both_branches_and_returns_the_selected_one(inputs):
    logits, noise = inputs
    calls = []

    def if_fn():
        calls.append("if")
        return _greedy(logits)

    def else_fn():
        calls.append("else")
        return _sampled(logits, noise)

    pred = torch.ones((), dtype=torch.bool, device="cuda")
    with gc.warm_up_both_branches():
        tokens, _ = gc.graph_if_else(pred, if_fn, else_fn, host_pred=False)
    assert calls == ["if", "else"]
    assert torch.equal(tokens, _sampled(logits, noise)[0])


def test_mismatched_branches_are_rejected_at_capture(inputs):
    logits, _ = inputs
    pred = torch.ones((), dtype=torch.bool, device="cuda")
    with pytest.raises(RuntimeError, match="different tensors"):
        _capture(pred, lambda: logits.argmax(dim=-1), lambda: logits.argmax(dim=-1).to(torch.int32))


def test_variant_override_restores_the_flags():
    metadata = SimpleNamespace(
        is_all_greedy_sample=False, is_all_temperature_only=True, temperature_only=True
    )
    with gc.sampling_variant_override(metadata, all_greedy=True):
        assert metadata.is_all_greedy_sample is True
        assert metadata.is_all_temperature_only is False
        assert metadata.temperature_only is False
    assert (
        metadata.is_all_greedy_sample,
        metadata.is_all_temperature_only,
        metadata.temperature_only,
    ) == (False, True, True)


@pytest.mark.parametrize(
    "flags, merged, expected",
    [
        (dict(is_all_greedy_sample=True), False, "greedy"),
        (dict(is_all_greedy_sample=False, temperature_only=True), False, "temperature_only"),
        (dict(is_all_greedy_sample=False), False, "advanced"),
        # Merged graphs replay one graph for every batch, keyed as the advanced variant.
        (dict(is_all_greedy_sample=True), True, "advanced"),
        (dict(is_all_greedy_sample=False, temperature_only=True), True, "advanced"),
    ],
)
def test_sampling_variant_key_with_merged_graphs(flags, merged, expected):
    metadata = SimpleNamespace(
        is_all_temperature_only=False, merged_sampling_graph=merged, **flags
    )
    assert sampling_variant(metadata) == expected
