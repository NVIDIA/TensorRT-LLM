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
"""Correctness of the fused universal sampling op against the TorchSampler pipeline.

The oracle is the production one -- ``min_p_renorm_probs`` then top-k then top-p renorm,
the order ``sampler_strategy._compute_probs`` uses and ``docs/source/features/sampling.md``
documents -- not a reimplementation, so a semantic drift in either shows up here.

What is deliberately NOT asserted: equality of sampled token *ids* against the flashinfer
path. The two consume their RNG differently, so identical ids were never expected and a
test demanding them would be wrong. Token behavior is checked by support, determinism and
distribution instead.
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as fi
from tensorrt_llm._torch.pyexecutor.sampler.ops import universal as uni
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import min_p_renorm_probs

DISABLE_TOPK = torch.iinfo(torch.int32).max
DISABLE_TOPP = 1.0
DISABLE_MINP = 0.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not uni.is_available(),
    reason="requires CUDA and the universal sampling op (set TLLM_UNIVERSAL_SAMPLING_JIT=1 in a checkout)",
)


def _params(rows, *, device, temperature=0.8, top_k=None, top_p=None, min_p=None):
    """Per-row buffers in the disable-sentinel convention the op expects."""

    def col(value, default, dtype):
        filled = torch.full((rows,), default, device=device, dtype=dtype)
        if value is not None:
            filled[:] = value
        return filled

    return (
        col(temperature, 1.0, torch.float32),
        col(top_k, DISABLE_TOPK, torch.int32),
        col(top_p, DISABLE_TOPP, torch.float32),
        col(min_p, DISABLE_MINP, torch.float32),
    )


def _reference_probs(logits, temperatures, top_ks, top_ps, min_ps):
    """The TorchSampler pipeline: temperature+softmax, then min-p, top-k, top-p."""
    probs = torch.softmax(logits.float() / temperatures.unsqueeze(-1), dim=-1)
    if (min_ps > 0).any():
        probs = min_p_renorm_probs(probs, min_ps)
    if (top_ks < logits.shape[-1]).any():
        probs = fi.top_k_renorm_probs_op(probs, fi.sanitize_top_k(top_ks.clone(), logits.shape[-1]))
    if (top_ps < 1.0).any():
        probs = fi.top_p_renorm_probs_op(probs, top_ps)
    return probs


def _l1_per_row(a, b):
    return (a - b).abs().sum(-1)


def _rng(rows, device, seed=7, offset=0):
    return (
        torch.tensor([seed], dtype=torch.int64, device=device),
        torch.tensor([offset], dtype=torch.int64, device=device),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_neutral_row_is_plain_softmax(dtype):
    """Every filter neutral: the op must reduce to temperature + softmax and nothing else.

    This is the row the 1.15x perf gate is about, so it is also the row whose correctness
    has to be beyond doubt -- any filtering here is a silent distribution change.
    """
    dev = "cuda"
    torch.manual_seed(0)
    logits = (torch.randn(16, 4096, device=dev) * 2.0).to(dtype)
    temps, top_ks, top_ps, min_ps = _params(16, device=dev)

    probs = uni.universal_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    expected = torch.softmax(logits.float() / temps.unsqueeze(-1), dim=-1)
    assert _l1_per_row(probs, expected).max().item() < 1e-4


def test_min_p_zero_changes_nothing():
    """min_p == 0 is the disable sentinel and must be bit-identical to not asking."""
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(8, 8192, device=dev) * 2.0
    temps, top_ks, top_ps, min_ps = _params(8, device=dev)

    with_zero = uni.universal_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    neutral = torch.softmax(logits / temps.unsqueeze(-1), dim=-1)
    torch.testing.assert_close(with_zero, neutral, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("min_p", [0.01, 0.1, 0.5])
def test_min_p_matches_reference(min_p):
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(16, 8192, device=dev) * 2.0
    temps, top_ks, top_ps, min_ps = _params(16, device=dev, min_p=min_p)

    probs = uni.universal_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    expected = _reference_probs(logits, temps, top_ks, top_ps, min_ps)
    assert _l1_per_row(probs, expected).max().item() < 1e-3
    # min-p is a hard support cut, so the kept sets must agree exactly -- there is no
    # boundary-tie excuse here as there is for top-p.
    assert torch.equal(probs > 0, expected > 0)


def test_min_p_one_keeps_only_the_argmax():
    """min_p == 1.0 is documented explicit-greedy: only p == p_max survives."""
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(8, 4096, device=dev) * 3.0
    temps, top_ks, top_ps, min_ps = _params(8, device=dev, min_p=1.0)

    probs = uni.universal_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    assert torch.equal(probs.argmax(-1), logits.argmax(-1))
    torch.testing.assert_close(
        probs.max(-1).values, torch.ones(8, device=dev), atol=1e-5, rtol=1e-5
    )
    assert (probs > 0).sum(-1).max().item() == 1


@pytest.mark.parametrize(
    "top_k,top_p,min_p",
    [
        (50, None, None),
        (None, 0.9, None),
        (50, 0.9, None),
        (50, None, 0.05),
        (None, 0.9, 0.05),
        (50, 0.9, 0.05),
    ],
)
def test_filter_combinations_match_reference(top_k, top_p, min_p):
    """Every combination, against the production pipeline.

    Bounded on total probability mass rather than pointwise: top-p's cutoff is not
    run-to-run reproducible right at the nucleus boundary, so a single boundary token can
    flip. That costs at most that token's mass, while a genuinely wrong filter moves
    O(0.1) -- orders of magnitude apart.
    """
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(16, 16384, device=dev) * 2.0
    temps, top_ks, top_ps, min_ps = _params(16, device=dev, top_k=top_k, top_p=top_p, min_p=min_p)

    probs = uni.universal_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    expected = _reference_probs(logits, temps, top_ks, top_ps, min_ps)
    assert _l1_per_row(probs, expected).max().item() < 5e-3
    torch.testing.assert_close(probs.sum(-1), torch.ones(16, device=dev), atol=1e-4, rtol=1e-4)


def test_mixed_batch_rows_are_independent():
    """The per-row skip is a claim, so it gets tested: a row's answer in a mixed batch
    must equal its answer computed alone."""
    dev = "cuda"
    torch.manual_seed(0)
    rows, vocab = 8, 8192
    logits = torch.randn(rows, vocab, device=dev) * 2.0

    temps = torch.full((rows,), 0.8, device=dev, dtype=torch.float32)
    top_ks = torch.full((rows,), DISABLE_TOPK, device=dev, dtype=torch.int32)
    top_ps = torch.full((rows,), DISABLE_TOPP, device=dev, dtype=torch.float32)
    min_ps = torch.full((rows,), DISABLE_MINP, device=dev, dtype=torch.float32)
    # A batch that mixes all four states the op can be in.
    top_ks[1] = 50
    top_ps[2] = 0.9
    min_ps[3] = 0.05
    top_ks[4], top_ps[4], min_ps[4] = 20, 0.8, 0.02

    batched = uni.universal_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    for r in range(rows):
        alone = uni.universal_compute_probs_from_logits(
            logits[r : r + 1].contiguous(),
            temps[r : r + 1].contiguous(),
            top_ks[r : r + 1].contiguous(),
            top_ps[r : r + 1].contiguous(),
            min_ps[r : r + 1].contiguous(),
        )
        torch.testing.assert_close(batched[r], alone[0], atol=1e-6, rtol=1e-5)


def test_sampled_tokens_lie_in_the_filtered_support():
    """A token the filters removed must never be sampled -- the failure that silently
    degrades output instead of raising."""
    dev = "cuda"
    torch.manual_seed(0)
    rows, vocab = 32, 8192
    logits = torch.randn(rows, vocab, device=dev) * 2.0
    temps, top_ks, top_ps, min_ps = _params(rows, device=dev, top_k=40, top_p=0.9, min_p=0.02)

    for step in range(8):
        seed, offset = _rng(rows, dev, offset=step)
        tokens, probs = uni.universal_sample_from_logits_with_probs(
            logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
        )
        picked = probs.gather(1, tokens.long().unsqueeze(-1)).squeeze(-1)
        assert (picked > 0).all(), f"step {step} sampled a token outside the support"


def test_tokens_and_probs_agree_with_the_probs_only_op():
    """The rejection path takes tokens from one call and probs from another; if the two
    entry points filtered differently, acceptance would be computed against a
    distribution neither side sampled from."""
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(16, 8192, device=dev) * 2.0
    temps, top_ks, top_ps, min_ps = _params(16, device=dev, top_k=50, top_p=0.9, min_p=0.05)
    seed, offset = _rng(16, dev)

    _, probs_both = uni.universal_sample_from_logits_with_probs(
        logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
    )
    probs_only = uni.universal_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    torch.testing.assert_close(probs_both, probs_only, atol=1e-6, rtol=1e-5)


def test_same_seed_and_offset_reproduce_the_same_tokens():
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(16, 4096, device=dev) * 2.0
    temps, top_ks, top_ps, min_ps = _params(16, device=dev, top_p=0.95)
    seed, offset = _rng(16, dev, seed=1234, offset=5)

    first = uni.universal_sample_from_logits(
        logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
    )
    second = uni.universal_sample_from_logits(
        logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
    )
    assert torch.equal(first, second)


def test_sampling_follows_the_filtered_distribution():
    """Many draws over a small vocabulary should reproduce the probs the op reports.

    Chi-square would need a distributional table; the total-variation distance is enough
    to catch a sampler drawing from the wrong (e.g. unfiltered) distribution, which is
    the bug worth catching.
    """
    dev = "cuda"
    torch.manual_seed(0)
    vocab, draws = 64, 4096
    logits = (torch.randn(1, vocab, device=dev) * 1.5).expand(draws, vocab).contiguous()
    temps, top_ks, top_ps, min_ps = _params(draws, device=dev, temperature=1.0, min_p=0.05)

    # One row per draw, each with its own RNG subsequence, so a single call yields the
    # whole sample.
    seed = torch.tensor([99], dtype=torch.int64, device=dev)
    offset = torch.tensor([0], dtype=torch.int64, device=dev)
    tokens, probs = uni.universal_sample_from_logits_with_probs(
        logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
    )

    counts = torch.bincount(tokens.long(), minlength=vocab).float() / draws
    tv = 0.5 * (counts - probs[0]).abs().sum().item()
    assert tv < 0.05, f"total-variation distance {tv:.3f} between draws and the reported probs"
