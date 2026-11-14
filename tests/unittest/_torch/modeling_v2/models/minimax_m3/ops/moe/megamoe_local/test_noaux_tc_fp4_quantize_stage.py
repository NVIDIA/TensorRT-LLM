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
"""Bitwise tests for the family's noaux_tc_fp4_quantize_stage_out against noaux_tc_op + fp4_quantize."""

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
BYTE_SENTINEL = 0xAB


def _reference(scores, bias, hidden, global_scale, topk, scale):
    values, indices = torch.ops.trtllm.noaux_tc_op(scores, bias, 1, 1, topk, scale)
    fp4, sf = torch.ops.trtllm.fp4_quantize(hidden, global_scale, 16, False, False)
    return indices, values, fp4.view(torch.uint8), sf.view(torch.uint8).view(hidden.shape[0], -1)


def _stage(cap, topk, hidden, extra_cols, device):
    return dict(
        topk_indices=torch.full(
            (cap, topk + extra_cols), IDX_SENTINEL, dtype=torch.int32, device=device
        ),
        topk_weights=torch.full(
            (cap, topk + extra_cols), WEIGHT_SENTINEL, dtype=torch.float32, device=device
        ),
        fp4_out=torch.full((cap, hidden // 2), BYTE_SENTINEL, dtype=torch.uint8, device=device),
        sf_out=torch.full((cap, hidden // 16), BYTE_SENTINEL, dtype=torch.uint8, device=device),
    )


def _run(scores, bias, hidden, global_scale, topk, scale, rows0, stages, pad_mask=None):
    s0, s1 = stages
    family_ops().noaux_tc_fp4_quantize_stage_out(
        scores,
        bias,
        hidden,
        global_scale,
        topk,
        scale,
        rows0,
        s0["topk_indices"],
        s0["topk_weights"],
        s0["fp4_out"],
        s0["sf_out"],
        s1["topk_indices"],
        s1["topk_weights"],
        s1["fp4_out"],
        s1["sf_out"],
        **({} if pad_mask is None else {"pad_mask": pad_mask}),
    )


def _check_stage(stage, ref, token_rows, topk):
    ref_idx, ref_w, ref_fp4, ref_sf = ref
    n = token_rows.stop - token_rows.start
    idx, w = stage["topk_indices"], stage["topk_weights"]
    assert torch.equal(idx[:n, :topk], ref_idx[token_rows])
    assert torch.equal(w[:n, :topk].view(torch.int32), ref_w[token_rows].view(torch.int32))
    assert torch.equal(stage["fp4_out"][:n], ref_fp4[token_rows])
    assert torch.equal(stage["sf_out"][:n], ref_sf[token_rows])
    assert bool((idx[n:, :topk] == -1).all())
    assert bool((w[n:, :topk] == 0).all()) and not bool(torch.signbit(w[n:, :topk]).any())
    assert bool((idx[:, topk:] == IDX_SENTINEL).all())
    assert bool((w[:, topk:] == WEIGHT_SENTINEL).all())
    assert bool((stage["fp4_out"][n:] == BYTE_SENTINEL).all())
    assert bool((stage["sf_out"][n:] == BYTE_SENTINEL).all())


def _inputs(num_tokens, num_experts, hidden_size, seed, ties):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn(num_tokens, num_experts, generator=gen, device="cuda", dtype=torch.float32)
    bias = 0.1 * torch.randn(num_experts, generator=gen, device="cuda", dtype=torch.float32)
    if ties:
        scores[:, ::3] = scores[:, :1]
        bias.zero_()
    hidden = (4 * torch.randn(num_tokens, hidden_size, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    if num_tokens > 0:
        hidden[0, :32] = 0
        hidden[-1, 16:48] = torch.finfo(torch.bfloat16).tiny
    global_scale = torch.tensor([448.0 * 6.0 / 12.5], device="cuda", dtype=torch.float32)
    return scores, bias, hidden, global_scale


@pytest.mark.parametrize("num_experts,topk", [(128, 4), (128, 8), (256, 8)])
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 47, 96, 148, 191, 192, 296, 384])
@pytest.mark.parametrize("split", ["half", "all_first", "all_second"])
@pytest.mark.parametrize("ties", [False, True])
def test_stage_out_matches_separate_ops(num_experts, topk, num_tokens, split, ties):
    hidden_size, scale = 6144, 2.0
    scores, bias, hidden, global_scale = _inputs(
        num_tokens, num_experts, hidden_size, num_tokens, ties
    )
    rows0 = {"half": (num_tokens + 1) // 2, "all_first": num_tokens, "all_second": 0}[split]
    stages = (
        _stage(rows0 + 5, topk, hidden_size, 0, "cuda"),
        _stage(num_tokens - rows0 + 3, topk, hidden_size, 2, "cuda"),
    )
    _run(scores, bias, hidden, global_scale, topk, scale, rows0, stages)
    ref = _reference(scores, bias, hidden, global_scale, topk, scale)
    _check_stage(stages[0], ref, range(0, rows0), topk)
    _check_stage(stages[1], ref, range(rows0, num_tokens), topk)


@pytest.mark.parametrize("hidden_size", [64, 3072])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_stage_out_other_hidden_sizes_and_scales(hidden_size, scale):
    num_tokens, topk = 37, 4
    scores, bias, hidden, global_scale = _inputs(num_tokens, 128, hidden_size, 11, False)
    rows0 = 19
    stages = (_stage(19, topk, hidden_size, 0, "cuda"), _stage(18, topk, hidden_size, 0, "cuda"))
    _run(scores, bias, hidden, global_scale, topk, scale, rows0, stages)
    ref = _reference(scores, bias, hidden, global_scale, topk, scale)
    _check_stage(stages[0], ref, range(0, rows0), topk)
    _check_stage(stages[1], ref, range(rows0, num_tokens), topk)


def test_zero_tokens_resets_both_tails():
    topk, hidden_size = 4, 6144
    scores, bias, hidden, global_scale = _inputs(0, 128, hidden_size, 0, False)
    stages = (_stage(32, topk, hidden_size, 0, "cuda"), _stage(32, topk, hidden_size, 0, "cuda"))
    _run(scores, bias, hidden, global_scale, topk, 2.0, 0, stages)
    for stage in stages:
        assert bool((stage["topk_indices"] == -1).all())
        assert bool((stage["topk_weights"] == 0).all())
        assert bool((stage["fp4_out"] == BYTE_SENTINEL).all())


def test_cuda_graph_replay_matches_eager():
    num_tokens, topk, hidden_size = 96, 4, 6144
    scores, bias, hidden, global_scale = _inputs(num_tokens, 128, hidden_size, 5, False)
    rows0 = 48
    stages = (_stage(64, topk, hidden_size, 0, "cuda"), _stage(64, topk, hidden_size, 0, "cuda"))
    _run(scores, bias, hidden, global_scale, topk, 2.0, rows0, stages)
    torch.cuda.synchronize()
    eager = [{k: v.clone() for k, v in stage.items()} for stage in stages]
    for stage in stages:
        for value in stage.values():
            value.fill_(1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(scores, bias, hidden, global_scale, topk, 2.0, rows0, stages)
    graph.replay()
    torch.cuda.synchronize()
    for stage, expected in zip(stages, eager):
        for key in ("fp4_out", "sf_out"):
            assert torch.equal(stage[key][:48], expected[key][:48])
        assert torch.equal(stage["topk_indices"], expected["topk_indices"])
        assert torch.equal(stage["topk_weights"], expected["topk_weights"])


def test_rejects_bad_arguments():
    topk, hidden_size = 4, 6144
    scores, bias, hidden, global_scale = _inputs(8, 128, hidden_size, 0, False)
    stages = (_stage(8, topk, hidden_size, 0, "cuda"), _stage(8, topk, hidden_size, 0, "cuda"))
    with pytest.raises(RuntimeError):
        _run(scores, bias, hidden, global_scale, topk, 2.0, 9, stages)
    small = (_stage(2, topk, hidden_size, 0, "cuda"), _stage(8, topk, hidden_size, 0, "cuda"))
    with pytest.raises(RuntimeError):
        _run(scores, bias, hidden, global_scale, topk, 2.0, 4, small)
    narrow = _inputs(8, 128, 96, 0, False)
    with pytest.raises(RuntimeError):
        _run(
            *narrow, topk, 2.0, 4, (_stage(8, topk, 96, 0, "cuda"), _stage(8, topk, 96, 0, "cuda"))
        )
    with pytest.raises(RuntimeError):
        _run(scores, bias, hidden.float(), global_scale, topk, 2.0, 4, stages)


def _pad_mask(num_tokens, pattern):
    rows = torch.arange(num_tokens, device="cuda")
    masked = {
        "none": rows < 0,
        "every_third": rows % 3 == 1,
        "all": rows >= 0,
        "tail": rows >= num_tokens - 5,
    }[pattern]
    return masked.to(torch.uint8)


@pytest.mark.parametrize("num_tokens", [1, 47, 96, 192])
@pytest.mark.parametrize("split", ["half", "all_first", "all_second"])
@pytest.mark.parametrize("pattern", ["none", "every_third", "all", "tail"])
def test_pad_mask_rows_take_the_tail_values(num_tokens, split, pattern):
    topk, hidden_size = 4, 6144
    scores, bias, hidden, global_scale = _inputs(num_tokens, 128, hidden_size, 11, False)
    rows0 = {"half": num_tokens // 2, "all_first": num_tokens, "all_second": 0}[split]
    pad_mask = _pad_mask(num_tokens, pattern)
    plain = (_stage(256, topk, hidden_size, 1, "cuda"), _stage(256, topk, hidden_size, 1, "cuda"))
    masked = (_stage(256, topk, hidden_size, 1, "cuda"), _stage(256, topk, hidden_size, 1, "cuda"))
    _run(scores, bias, hidden, global_scale, topk, 2.0, rows0, plain)
    _run(scores, bias, hidden, global_scale, topk, 2.0, rows0, masked, pad_mask=pad_mask)
    torch.cuda.synchronize()
    pad = pad_mask.bool()
    for p, m, tokens in zip(plain, masked, (slice(0, rows0), slice(rows0, num_tokens))):
        n, row_pad = tokens.stop - tokens.start, pad[tokens]
        for key in ("fp4_out", "sf_out"):
            assert torch.equal(m[key], p[key])
        idx, w = m["topk_indices"][:n], m["topk_weights"][:n]
        assert torch.equal(idx[~row_pad], p["topk_indices"][:n][~row_pad])
        assert torch.equal(
            w[~row_pad].view(torch.int32), p["topk_weights"][:n][~row_pad].view(torch.int32)
        )
        assert bool((idx[row_pad][:, :topk] == -1).all())
        assert bool((w[row_pad][:, :topk] == 0).all())
        assert not bool(torch.signbit(w[row_pad][:, :topk]).any())
        assert bool((idx[row_pad][:, topk:] == IDX_SENTINEL).all())
        assert bool((w[row_pad][:, topk:] == WEIGHT_SENTINEL).all())
        assert torch.equal(m["topk_indices"][n:], p["topk_indices"][n:])
        assert torch.equal(m["topk_weights"][n:], p["topk_weights"][n:])


def test_pad_mask_is_read_at_replay():
    num_tokens, topk, hidden_size = 64, 4, 6144
    scores, bias, hidden, global_scale = _inputs(num_tokens, 128, hidden_size, 3, False)
    pad_mask = torch.zeros(num_tokens, dtype=torch.uint8, device="cuda")
    stages = (_stage(64, topk, hidden_size, 0, "cuda"), _stage(64, topk, hidden_size, 0, "cuda"))
    _run(scores, bias, hidden, global_scale, topk, 2.0, num_tokens, stages, pad_mask=pad_mask)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(scores, bias, hidden, global_scale, topk, 2.0, num_tokens, stages, pad_mask=pad_mask)
    pad_mask[[5, 40]] = 1
    graph.replay()
    torch.cuda.synchronize()
    idx = stages[0]["topk_indices"]
    assert bool((idx[[5, 40]] == -1).all()) and bool((idx[[4, 6, 39, 41]] >= 0).all())
    pad_mask.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert bool((idx[:num_tokens] >= 0).all())


def test_rejects_bad_pad_masks():
    topk, hidden_size = 4, 6144
    scores, bias, hidden, global_scale = _inputs(8, 128, hidden_size, 0, False)
    stages = (_stage(8, topk, hidden_size, 0, "cuda"), _stage(8, topk, hidden_size, 0, "cuda"))
    bad_masks = (
        torch.zeros(8, dtype=torch.bool, device="cuda"),
        torch.zeros(8, dtype=torch.int32, device="cuda"),
        torch.zeros(7, dtype=torch.uint8, device="cuda"),
        torch.zeros(8, 1, dtype=torch.uint8, device="cuda"),
        torch.zeros(16, dtype=torch.uint8, device="cuda")[::2],
        torch.zeros(8, dtype=torch.uint8),
    )
    for bad in bad_masks:
        with pytest.raises(RuntimeError):
            _run(scores, bias, hidden, global_scale, topk, 2.0, 4, stages, pad_mask=bad)
