# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``trtllm::lm_head_argmax`` (one rank): the greedy tokens of ``input @ weight.T`` against torch's argmax."""

import pytest
import torch
from utils.util import skip_pre_hopper

import tensorrt_llm  # noqa: F401  (registers the trtllm ops)

SCRATCH_WORDS = 256 * 8 + 2


def _reference(x, w, vocab_offset):
    logits = x.float() @ w.float().T
    top2 = logits.topk(2, dim=-1)
    gap = top2.values[:, 0] - top2.values[:, 1]
    return logits.argmax(-1) + vocab_offset, gap


@skip_pre_hopper
@pytest.mark.parametrize("vocab_offset", [0, 38720])
@pytest.mark.parametrize("k", [1024, 4096])
@pytest.mark.parametrize("n", [16, 1000, 38720])
@pytest.mark.parametrize("m", [1, 3, 8])
def test_lm_head_argmax_matches_torch(m, n, k, vocab_offset):
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    scratch = torch.zeros(SCRATCH_WORDS, dtype=torch.int64, device="cuda")
    want, gap = _reference(x, w, vocab_offset)
    # Repeated calls reuse the scratch (the CTA counter resets itself).
    for _ in range(3):
        tokens = torch.full((m,), -1, dtype=torch.int32, device="cuda")
        torch.ops.trtllm.lm_head_argmax(x, w, vocab_offset, scratch, [], 0, tokens, 1)
        # bf16 MMAs with fp32 accumulation against an fp32 GEMM: only clear winners must agree.
        clear = gap > 1e-2
        assert torch.equal(tokens.long()[clear], want[clear])
        assert bool(((tokens >= vocab_offset) & (tokens < vocab_offset + n)).all())


@skip_pre_hopper
def test_lm_head_argmax_ties_and_stride():
    k, n, m = 4096, 2048, 4
    x = torch.ones(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
    w[[7, 9, 1500]] = 0.01  # three equal maxima: the lowest index wins
    scratch = torch.zeros(SCRATCH_WORDS, dtype=torch.int64, device="cuda")
    out = torch.full((m * 3,), -1, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.lm_head_argmax(x, w, 100, scratch, [], 0, out, 3)
    assert out[0::3].tolist() == [107] * m
    assert out[1::3].tolist() == [-1] * m


@skip_pre_hopper
def test_lm_head_argmax_rejects_unsupported_shapes():
    scratch = torch.zeros(SCRATCH_WORDS, dtype=torch.int64, device="cuda")
    tokens = torch.empty(16, dtype=torch.int32, device="cuda")
    w = torch.zeros(64, 4096, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError):  # more rows than one call takes
        torch.ops.trtllm.lm_head_argmax(
            torch.zeros(9, 4096, dtype=torch.bfloat16, device="cuda"),
            w,
            0,
            scratch,
            [],
            0,
            tokens,
            1,
        )
    with pytest.raises(RuntimeError):  # hidden size not a multiple of the k-chunk
        torch.ops.trtllm.lm_head_argmax(
            torch.zeros(2, 1000, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(64, 1000, dtype=torch.bfloat16, device="cuda"),
            0,
            scratch,
            [],
            0,
            tokens,
            1,
        )
