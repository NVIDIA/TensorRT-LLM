# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target logprobs of the tokens a speculative step emits."""

import pytest
import torch

from tensorrt_llm._torch.speculative.target_logprobs import target_logprobs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _reference(logits, tokens, lens, temperature, num_contexts, width, row_start):
    """fp64 log_softmax of the emitted rows; 0 where a request or position is skipped."""
    out = torch.zeros(tokens.shape[0], width, dtype=torch.float64)
    for b in range(tokens.shape[0]):
        if temperature[b] == 0:
            continue
        for j in range(int(lens[b])):
            if row_start is not None:
                row = int(row_start[b]) + j
            else:
                row = b if b < num_contexts else num_contexts + (b - num_contexts) * width + j
            scaled = logits[row].double() / float(temperature[b])
            out[b, j] = scaled[int(tokens[b, j])] - torch.logsumexp(scaled, dim=0)
    return out


@pytest.mark.parametrize(
    "dtype,num_contexts,num_gens,vocab,ragged",
    [
        # Many rows: one program per row, uniform layout.
        (torch.bfloat16, 2, 6, 32003, False),
        # Few rows over a large vocabulary: rows are split and combined, ragged layout.
        (torch.float32, 1, 1, 151936, True),
        (torch.bfloat16, 0, 1, 151936, False),
    ],
)
def test_target_logprobs_matches_reference(dtype, num_contexts, num_gens, vocab, ragged):
    torch.manual_seed(0)
    width = 6
    batch = num_contexts + num_gens
    lens = torch.randint(1, width + 1, (batch,))
    lens[:num_contexts] = 1
    if ragged:
        windows = torch.where(
            torch.arange(batch) < num_contexts, 1, torch.randint(1, width + 1, (batch,))
        )
        lens = torch.minimum(lens, windows)
        row_start = torch.cumsum(windows, 0) - windows
        rows = int(windows.sum())
    else:
        row_start = None
        rows = num_contexts + num_gens * width
    logits = (torch.randn(rows, vocab) * 4).to(dtype)
    tokens = torch.randint(0, vocab, (batch, width), dtype=torch.int32)
    # RAW (1), PROCESSED (0.7) and a request that did not ask (0).
    temperature = torch.tensor([[1.0, 0.7, 0.0][i % 3] for i in range(batch)])
    expected = _reference(logits, tokens, lens, temperature, num_contexts, width, row_start)

    out = target_logprobs(
        logits.cuda(),
        tokens.cuda(),
        lens.to(torch.int32).cuda(),
        temperature.cuda(),
        num_contexts=num_contexts,
        width=width,
        row_start=None if row_start is None else row_start.to(torch.int32).cuda(),
    )
    torch.testing.assert_close(out.cpu().double(), expected, atol=2e-4, rtol=1e-5)


def test_target_logprobs_masked_vocab_and_padding():
    """A guided-decoding mask (-inf) leaves the normaliser finite; padding tokens are never read."""
    vocab, width = 4096, 3
    logits = torch.randn(width, vocab, device="cuda")
    logits[:, vocab // 2 :] = float("-inf")
    # Position 2 is past new_tokens_lens, and its token is out of range: it must not be loaded.
    tokens = torch.tensor([[5, 7, vocab * 10]], dtype=torch.int32, device="cuda")
    lens = torch.tensor([2], dtype=torch.int32, device="cuda")
    out = target_logprobs(
        logits,
        tokens,
        lens,
        torch.ones(1, device="cuda"),
        num_contexts=0,
        width=width,
    )
    expected = torch.log_softmax(logits.double(), dim=-1)
    assert torch.isfinite(out[0, :2]).all()
    torch.testing.assert_close(
        out[0, :2].double(),
        expected[torch.arange(2), tokens[0, :2].long()],
        atol=1e-5,
        rtol=1e-6,
    )
    assert out[0, 2] == 0
