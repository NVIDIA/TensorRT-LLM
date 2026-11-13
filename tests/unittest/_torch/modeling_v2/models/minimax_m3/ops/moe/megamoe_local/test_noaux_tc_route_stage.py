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
"""Bitwise tests for the family's noaux_tc_route_stage_out against noaux_tc_fp4_quantize_stage_out's ids and weights."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not 100 <= get_sm_version() < 110,
    reason="needs an SM10x GPU",
)

IDX_SENTINEL = 7
WEIGHT_SENTINEL = 3.0
HIDDEN = 64


def _inputs(num_tokens, num_experts, seed, ties):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn(num_tokens, num_experts, generator=gen, device="cuda", dtype=torch.float32)
    bias = 0.1 * torch.randn(num_experts, generator=gen, device="cuda", dtype=torch.float32)
    if ties:
        scores[:, ::3] = scores[:, :1]
        bias.zero_()
    return scores, bias


def _ids_weights(cap_idx, cap_w, topk, extra_cols):
    return (
        torch.full((cap_idx, topk + extra_cols), IDX_SENTINEL, dtype=torch.int32, device="cuda"),
        torch.full((cap_w, topk + extra_cols), WEIGHT_SENTINEL, dtype=torch.float32, device="cuda"),
    )


def _fused(scores, bias, topk, scale, idx, w, pad_mask=None):
    num_tokens = scores.shape[0]
    hidden = torch.zeros(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    gs = torch.ones(1, device="cuda")
    fp4 = torch.empty(max(num_tokens, 1), HIDDEN // 2, dtype=torch.uint8, device="cuda")
    sf = torch.empty(max(num_tokens, 1), HIDDEN // 16, dtype=torch.uint8, device="cuda")
    e = (
        torch.empty(0, topk, dtype=torch.int32, device="cuda"),
        torch.empty(0, topk, dtype=torch.float32, device="cuda"),
        torch.empty(0, HIDDEN // 2, dtype=torch.uint8, device="cuda"),
        torch.empty(0, HIDDEN // 16, dtype=torch.uint8, device="cuda"),
    )
    family_ops().noaux_tc_fp4_quantize_stage_out(
        scores,
        bias,
        hidden,
        gs,
        topk,
        scale,
        num_tokens,
        idx,
        w[:, :topk],
        fp4,
        sf,
        *e,
        **({} if pad_mask is None else {"pad_mask": pad_mask}),
    )


def _route(scores, bias, topk, scale, idx, w, pad_mask=None):
    family_ops().noaux_tc_route_stage_out(
        scores,
        bias,
        topk,
        scale,
        idx,
        w[:, :topk],
        **({} if pad_mask is None else {"pad_mask": pad_mask}),
    )


def _assert_same(a, b):
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1].view(torch.int32), b[1].view(torch.int32))


@pytest.mark.parametrize("num_experts,topk", [(128, 4), (128, 8), (256, 8)])
@pytest.mark.parametrize("num_tokens", [0, 1, 3, 47, 192, 2048, 4100])
@pytest.mark.parametrize("cap_extra", [0, 5, 16384])
@pytest.mark.parametrize("ties", [False, True])
def test_route_stage_matches_the_fused_stage(num_experts, topk, num_tokens, cap_extra, ties):
    scores, bias = _inputs(num_tokens, num_experts, num_tokens + cap_extra, ties)
    cap = num_tokens + cap_extra
    fused, routed = _ids_weights(cap, cap + 3, topk, 2), _ids_weights(cap, cap + 3, topk, 2)
    _fused(scores, bias, topk, 2.0, *fused)
    _route(scores, bias, topk, 2.0, *routed)
    _assert_same(fused, routed)
    assert bool((routed[0][:, topk:] == IDX_SENTINEL).all())
    assert bool((routed[1][:, topk:] == WEIGHT_SENTINEL).all())
    assert bool((routed[0][num_tokens:, :topk] == -1).all())
    assert bool((routed[1][num_tokens:, :topk] == 0).all())
    assert not bool(torch.signbit(routed[1][num_tokens:, :topk]).any())


@pytest.mark.parametrize("num_tokens", [1, 47, 2048])
@pytest.mark.parametrize("pattern", ["every_third", "all", "tail"])
def test_pad_mask_rows_match_the_fused_stage(num_tokens, pattern):
    topk = 4
    scores, bias = _inputs(num_tokens, 128, 11, False)
    rows = torch.arange(num_tokens, device="cuda")
    pad_mask = {
        "every_third": rows % 3 == 1,
        "all": rows >= 0,
        "tail": rows >= num_tokens - 5,
    }[pattern].to(torch.uint8)
    fused, routed = (
        _ids_weights(num_tokens + 300, num_tokens + 300, topk, 1),
        _ids_weights(num_tokens + 300, num_tokens + 300, topk, 1),
    )
    _fused(scores, bias, topk, 2.0, *fused, pad_mask=pad_mask)
    _route(scores, bias, topk, 2.0, *routed, pad_mask=pad_mask)
    _assert_same(fused, routed)


def test_cuda_graph_replay_reads_the_pad_mask():
    num_tokens, topk = 64, 4
    scores, bias = _inputs(num_tokens, 128, 3, False)
    pad_mask = torch.zeros(num_tokens, dtype=torch.uint8, device="cuda")
    idx, w = _ids_weights(4096, 4096, topk, 0)
    _route(scores, bias, topk, 2.0, idx, w, pad_mask=pad_mask)
    torch.cuda.synchronize()
    eager = (idx.clone(), w.clone())
    idx.fill_(1)
    w.fill_(1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _route(scores, bias, topk, 2.0, idx, w, pad_mask=pad_mask)
    graph.replay()
    torch.cuda.synchronize()
    _assert_same(eager, (idx, w))
    pad_mask[[5, 40]] = 1
    graph.replay()
    torch.cuda.synchronize()
    assert bool((idx[[5, 40]] == -1).all()) and bool((idx[[4, 6, 39, 41]] >= 0).all())


def test_rejects_bad_arguments():
    topk = 4
    scores, bias = _inputs(8, 128, 0, False)
    idx, w = _ids_weights(8, 8, topk, 0)
    with pytest.raises(RuntimeError):
        _route(scores, bias, topk, 2.0, idx[:7], w)
    with pytest.raises(RuntimeError):
        _route(scores, bias, topk, 2.0, idx, w[:7])
    with pytest.raises(RuntimeError):
        _route(scores, bias, topk, 2.0, idx.float(), w)
    with pytest.raises(RuntimeError):
        _route(scores.double(), bias, topk, 2.0, idx, w)
    with pytest.raises(RuntimeError):
        _route(scores, bias[:-1], topk, 2.0, idx, w)
    with pytest.raises(RuntimeError):
        _route(scores, bias, 33, 2.0, *_ids_weights(8, 8, 33, 0))
    for bad in (
        torch.zeros(8, dtype=torch.bool, device="cuda"),
        torch.zeros(7, dtype=torch.uint8, device="cuda"),
        torch.zeros(8, dtype=torch.uint8),
    ):
        with pytest.raises(RuntimeError):
            _route(scores, bias, topk, 2.0, idx, w, pad_mask=bad)
