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
"""``trtllm::lm_head_argmax`` on one rank: the greedy tokens of ``input @ weight.T`` against
``torch.argmax`` over the logits the GEMM stores (ties to the lowest index)."""

import pytest
import torch
from utils.util import skip_pre_hopper

import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
from tensorrt_llm._torch.modules.bf16_draft_head import SCRATCH_WORDS, max_rows


def _run(x, w, vocab_offset=0, round_logits=True, scratch=None):
    if scratch is None:
        scratch = torch.zeros(SCRATCH_WORDS, dtype=torch.int64, device="cuda")
    tokens = torch.full((x.shape[0],), -1, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.lm_head_argmax(
        x, w, vocab_offset, round_logits, scratch, None, [], 0, 0, tokens
    )
    return tokens


def _exact_operands(m, n, k, seed, dtype=torch.bfloat16):
    """Small integers times powers of two: every dot product is exact in fp32 whatever the
    summation order, so the stored (bf16 / fp16) logits of any GEMM agree bit for bit and their
    ties are real."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randint(-3, 4, (m, k), device="cuda", generator=gen).to(dtype)
    w = (torch.randint(-2, 3, (n, k), device="cuda", generator=gen) / 16.0).to(dtype)
    return x, w


@skip_pre_hopper
@pytest.mark.parametrize("k", [256, 1024, 2560, 4096, 7168])
def test_max_rows(k):
    assert max_rows(k, torch.cuda.current_device()) == 8
    assert torch.ops.trtllm.lm_head_argmax_max_rows(k + 128) == 0


@skip_pre_hopper
@pytest.mark.parametrize("vocab_offset", [0, 38720])
@pytest.mark.parametrize("k", [256, 2560, 4096])
@pytest.mark.parametrize("n", [16, 1000, 38720])
@pytest.mark.parametrize("m", list(range(1, 9)))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_gemm_argmax(dtype, m, n, k, vocab_offset):
    """Exact operands: the stored-logit argmax of the cuBLAS GEMM, ties included."""
    x, w = _exact_operands(m, n, k, seed=m * n + k, dtype=dtype)
    logits = x @ w.T  # what the LM head stores
    want = torch.argmax(logits.float(), dim=-1) + vocab_offset
    scratch = torch.zeros(SCRATCH_WORDS, dtype=torch.int64, device="cuda")
    # Repeated calls reuse the scratch (the CTA counter resets itself).
    for _ in range(3):
        assert torch.equal(_run(x, w, vocab_offset, scratch=scratch).long(), want)
    # Without the rounding to the operand dtype the kernel compares the fp32 sums (an fp32-output head).
    want_fp32 = torch.argmax(x.float() @ w.float().T, dim=-1) + vocab_offset
    assert torch.equal(_run(x, w, vocab_offset, round_logits=False).long(), want_fp32)


@skip_pre_hopper
@pytest.mark.parametrize("m", [1, 4, 8])
def test_random_operands(m):
    """Random bf16 operands: rows whose top two logits are clearly apart agree with cuBLAS."""
    torch.manual_seed(m)
    n, k = 51200, 2048
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    logits = (x @ w.T).float()
    top2 = logits.topk(2, dim=-1).values
    clear = top2[:, 0] - top2[:, 1] > 1e-2
    tokens = _run(x, w).long()
    assert torch.equal(tokens[clear], logits.argmax(-1)[clear])
    assert bool(((tokens >= 0) & (tokens < n)).all())


@skip_pre_hopper
def test_ties_signed_zero_and_strides():
    k, n, m = 4096, 2048, 4
    x = torch.ones(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
    w[[1500, 9, 7]] = 0.01  # three equal maxima: the lowest index wins
    assert _run(x, w, 100).tolist() == [107] * m
    # All logits zero, some of them -0.0: equal, so the lowest index.
    w.zero_()
    w[:5, 0] = -1.0
    x[:, 0] = 0.0
    assert _run(x, w).tolist() == [0] * m
    # Row-strided input and weight (unit inner stride, 16-byte aligned rows).
    big_x = torch.randn(2 * m, k + 8, dtype=torch.bfloat16, device="cuda")
    big_w = torch.randn(n, k + 64, dtype=torch.bfloat16, device="cuda")
    xs, ws = big_x[::2, :k], big_w[:, 64:]
    want = torch.argmax(xs.contiguous() @ ws.contiguous().T, dim=-1).to(torch.int32)
    logits = (xs.float() @ ws.float().T).topk(2, dim=-1).values
    clear = logits[:, 0] - logits[:, 1] > 1e-2
    assert torch.equal(_run(xs, ws)[clear], want[clear])


@skip_pre_hopper
def test_cuda_graph_replay():
    x, w = _exact_operands(5, 30000, 2048, seed=5)
    scratch = torch.zeros(SCRATCH_WORDS, dtype=torch.int64, device="cuda")
    tokens = torch.empty(5, dtype=torch.int32, device="cuda")

    def call():
        torch.ops.trtllm.lm_head_argmax(x, w, 7, True, scratch, None, [], 0, 0, tokens)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for seed in range(3):
        x.copy_(_exact_operands(5, 1, 2048, seed=100 + seed)[0])
        graph.replay()
        torch.cuda.synchronize()
        want = torch.argmax((x @ w.T).float(), dim=-1).to(torch.int32) + 7
        assert torch.equal(tokens, want)


@skip_pre_hopper
def test_rejects_unsupported_shapes():
    scratch = torch.zeros(SCRATCH_WORDS, dtype=torch.int64, device="cuda")
    w = torch.zeros(64, 4096, dtype=torch.bfloat16, device="cuda")
    for x in (
        torch.zeros(9, 4096, dtype=torch.bfloat16, device="cuda"),  # more rows than a launch takes
        torch.zeros(
            2, 1000, dtype=torch.bfloat16, device="cuda"
        ),  # hidden size not a multiple of 256
    ):
        tokens = torch.empty(x.shape[0], dtype=torch.int32, device="cuda")
        with pytest.raises(RuntimeError):
            torch.ops.trtllm.lm_head_argmax(
                x, w[:, : x.shape[1]].contiguous(), 0, True, scratch, None, [], 0, 0, tokens
            )
    # Several ranks need the exchange channel.
    tokens = torch.empty(1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.lm_head_argmax(
            torch.zeros(1, 4096, dtype=torch.bfloat16, device="cuda"),
            w,
            0,
            True,
            scratch,
            None,
            [0, 0],
            0,
            0,
            tokens,
        )
