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
"""Correctness of the fused sampling op against Torch references.

Semantic tests use the production pipeline -- ``min_p_renorm_probs`` then top-k then
top-p renorm, the order ``sampler_strategy._compute_probs`` uses and
``docs/source/features/sampling.md`` documents.  A separate numerical-error matrix uses
a mechanically independent pure-Torch implementation, so sharing a backend cannot make
the fused op and its oracle reproduce the same arithmetic mistake.

What is deliberately NOT asserted: equality of sampled token *ids* against the flashinfer
path. The two consume their RNG differently, so identical ids are not expected; token
behavior is checked by support, determinism and distribution instead.
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as fi
from tensorrt_llm._torch.pyexecutor.sampler.ops import fused
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import min_p_renorm_probs

DISABLE_TOPK = torch.iinfo(torch.int32).max
DISABLE_TOPP = 1.0
DISABLE_MINP = 0.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused.is_available(),
    reason=(
        "requires CUDA and the fused sampling op "
        "(set TLLM_FUSED_SAMPLING_JIT=1 in an editable checkout)"
    ),
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


def _pure_torch_reference_probs(logits, temperatures, top_ks, top_ps, min_ps):
    """Independent Torch implementation of the documented filter pipeline.

    Unlike ``_reference_probs``, this does not call FlashInfer or any fused-op
    helper.  Keeping the oracle mechanically independent is important for measuring
    numerical error: a shared backend could otherwise reproduce the same mistake.
    The row loop is intentional -- this is a correctness oracle for small batches,
    not an implementation whose performance matters.
    """
    rows, vocab = logits.shape
    scaled_logits = logits.float() / temperatures.unsqueeze(-1)
    reference = []

    for row_idx in range(rows):
        row = torch.softmax(scaled_logits[row_idx], dim=-1)

        min_p = min_ps[row_idx]
        if min_p > 0:
            row = torch.where(row >= min_p * row.max(), row, 0.0)
            row = row / row.sum()

        top_k = int(top_ks[row_idx].item())
        if 0 < top_k < vocab:
            indices = torch.topk(row, top_k, sorted=False).indices
            filtered = torch.zeros_like(row)
            filtered.scatter_(0, indices, row[indices])
            row = filtered / filtered.sum()

        top_p = top_ps[row_idx]
        if top_p < 1:
            sorted_probs, sorted_indices = torch.sort(row, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = torch.searchsorted(cumulative_probs, top_p, right=False)
            keep = torch.arange(vocab, device=logits.device) <= cutoff
            sorted_probs = torch.where(keep, sorted_probs, 0.0)
            filtered = torch.zeros_like(row)
            filtered.scatter_(0, sorted_indices, sorted_probs)
            row = filtered / filtered.sum()

        reference.append(row)

    return torch.stack(reference)


def _l1_per_row(a, b):
    return (a - b).abs().sum(-1)


def _print_probability_error(label, actual, expected):
    absolute_error = (actual - expected).abs()
    row_l1 = absolute_error.sum(-1)
    print(
        "REFERENCE_ERROR "
        f"case={label} max_abs={absolute_error.max().item():.9e} "
        f"max_row_l1={row_l1.max().item():.9e} "
        f"mean_row_l1={row_l1.mean().item():.9e} "
        f"support_mismatches="
        f"{(actual > 0).logical_xor(expected > 0).sum().item()}"
    )
    return row_l1.max().item()


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

    probs = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    expected = torch.softmax(logits.float() / temps.unsqueeze(-1), dim=-1)
    assert _l1_per_row(probs, expected).max().item() < 2e-5


def test_min_p_zero_changes_nothing():
    """min_p == 0 is the disable sentinel and must be bit-identical to not asking."""
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(8, 8192, device=dev) * 2.0
    temps, top_ks, top_ps, min_ps = _params(8, device=dev)

    with_zero = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    neutral = torch.softmax(logits / temps.unsqueeze(-1), dim=-1)
    torch.testing.assert_close(with_zero, neutral, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("min_p", [0.01, 0.1, 0.5])
def test_min_p_matches_reference(min_p):
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(16, 8192, device=dev) * 2.0
    temps, top_ks, top_ps, min_ps = _params(16, device=dev, min_p=min_p)

    probs = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    expected = _reference_probs(logits, temps, top_ks, top_ps, min_ps)
    max_row_l1 = _print_probability_error(f"min_p={min_p}", probs, expected)
    assert max_row_l1 < 2e-5
    # min-p is a hard support cut, so the kept sets must agree exactly -- there is no
    # boundary-tie excuse here as there is for top-p.
    assert torch.equal(probs > 0, expected > 0)


def test_min_p_one_keeps_only_the_argmax():
    """min_p == 1.0 is documented explicit-greedy: only p == p_max survives."""
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(8, 4096, device=dev) * 3.0
    temps, top_ks, top_ps, min_ps = _params(8, device=dev, min_p=1.0)

    probs = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
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

    probs = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    expected = _reference_probs(logits, temps, top_ks, top_ps, min_ps)
    max_row_l1 = _print_probability_error(
        f"top_k={top_k},top_p={top_p},min_p={min_p}", probs, expected
    )
    assert max_row_l1 < 2e-5
    torch.testing.assert_close(probs.sum(-1), torch.ones(16, device=dev), atol=1e-6, rtol=1e-6)


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

    batched = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    for r in range(rows):
        alone = fused.fused_compute_probs_from_logits(
            logits[r : r + 1].contiguous(),
            temps[r : r + 1].contiguous(),
            top_ks[r : r + 1].contiguous(),
            top_ps[r : r + 1].contiguous(),
            min_ps[r : r + 1].contiguous(),
        )
        torch.testing.assert_close(batched[r], alone[0], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("vocab", [65535, 65536, 65537])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_large_vocab_dispatch_boundary_matches_reference(dtype, vocab):
    """Exercise both sides of the multi-CTA vocabulary boundary."""
    dev = "cuda"
    torch.manual_seed(0)
    rows = 4
    logits = (torch.randn(rows, vocab, device=dev) * 2.0).to(dtype)
    temps, top_ks, top_ps, min_ps = _params(rows, device=dev)
    top_ps[1] = 0.9
    top_ks[2] = 50
    top_ks[3], top_ps[3], min_ps[3] = 50, 0.9, 0.05

    probs = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    expected = _reference_probs(logits, temps, top_ks, top_ps, min_ps)
    max_row_l1 = _print_probability_error(f"large_vocab,dtype={dtype}", probs, expected)
    assert max_row_l1 < 2e-5
    torch.testing.assert_close(probs.sum(-1), torch.ones(rows, device=dev), atol=1e-6, rtol=1e-6)

    seed, offset = _rng(rows, dev)
    tokens, probs_both = fused.fused_sample_from_logits_with_probs(
        logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
    )
    torch.testing.assert_close(probs_both, probs, atol=1e-6, rtol=1e-5)
    assert (probs_both.gather(1, tokens.long().unsqueeze(-1)) > 0).all()


@pytest.mark.parametrize("rows", [1, 8, 31, 32, 33, 64])
def test_fp32_probability_error_against_pure_torch_reference(rows):
    """Measure fused-op error against an independent Torch oracle.

    The 128256-token vocabulary is representative of Llama 3.1. Rows 31/32/33
    exercise both sides of the multi-CTA row boundary, while row 64 covers its fallback.
    Rows cycle through every combination of min-p, top-k and top-p; the single-row
    case uses all three filters.  FP32 is intentional: both target and Eagle3 draft
    logits pass through ``LogitsProcessor``, which casts them before the production
    fused-op call.

    Keep the metric printout: the test threshold prevents regressions, while the raw
    values make it possible to judge whether that threshold is defensible.
    """
    dev = "cuda"
    vocab = 128256
    torch.manual_seed(20260903)
    # Production enters this op in FP32, where random continuous logits make exact
    # top-k/min-p ties vanishingly unlikely.  Top-p may still differ by its single
    # boundary token because the two implementations reduce mass in different orders;
    # that case is checked explicitly below instead of hidden by a wide L1 tolerance.
    logits = torch.randn(rows, vocab, device=dev) * 2.0
    temperatures, top_ks, top_ps, min_ps = _params(rows, device=dev)

    temperatures.copy_(
        torch.tensor([0.7, 0.8, 1.0, 1.3], device=dev).repeat((rows + 3) // 4)[:rows]
    )
    filters = (
        ("neutral", None, None, None),
        ("min_p", None, None, 0.05),
        ("top_k", 50, None, None),
        ("top_p", None, 0.9, None),
        ("min_p_top_k", 50, None, 0.05),
        ("min_p_top_p", None, 0.9, 0.05),
        ("top_k_top_p", 50, 0.9, None),
        ("all_filters", 50, 0.9, 0.05),
    )
    if rows == 1:
        filters = (filters[-1],)
    row_filter_names = []
    for row_idx in range(rows):
        filter_name, top_k, top_p, min_p = filters[row_idx % len(filters)]
        row_filter_names.append(filter_name)
        if top_k is not None:
            top_ks[row_idx] = top_k
        if top_p is not None:
            top_ps[row_idx] = top_p
        if min_p is not None:
            min_ps[row_idx] = min_p

    actual = fused.fused_compute_probs_from_logits(logits, temperatures, top_ks, top_ps, min_ps)
    expected = _pure_torch_reference_probs(logits, temperatures, top_ks, top_ps, min_ps)
    absolute_error = (actual - expected).abs()
    row_l1 = absolute_error.sum(-1)
    support_difference = (actual > 0).logical_xor(expected > 0)
    support_mismatches = support_difference.sum().item()
    max_abs = absolute_error.max().item()
    mean_abs = absolute_error.mean().item()
    max_row_l1 = row_l1.max().item()
    mean_row_l1 = row_l1.mean().item()
    max_mass_error = (actual.sum(-1) - 1.0).abs().max().item()

    print(
        "NUMERIC_ERROR "
        f"dtype={logits.dtype} rows={rows} vocab={vocab} "
        f"max_abs={max_abs:.9e} mean_abs={mean_abs:.9e} "
        f"max_row_l1={max_row_l1:.9e} mean_row_l1={mean_row_l1:.9e} "
        f"max_tv={0.5 * max_row_l1:.9e} "
        f"support_mismatches={support_mismatches} "
        f"max_mass_error={max_mass_error:.9e}"
    )
    for filter_name in dict.fromkeys(row_filter_names):
        filter_rows = torch.tensor(
            [name == filter_name for name in row_filter_names], device=dev, dtype=torch.bool
        )
        filter_support_mismatches = (
            (actual[filter_rows] > 0).logical_xor(expected[filter_rows] > 0).sum().item()
        )
        print(
            "NUMERIC_FILTER "
            f"dtype={logits.dtype} rows={rows} filter={filter_name} "
            f"max_row_l1={row_l1[filter_rows].max().item():.9e} "
            f"mean_row_l1={row_l1[filter_rows].mean().item():.9e} "
            f"support_mismatches={filter_support_mismatches}"
        )

    # Measured on H100 against the independent oracle: max row-L1 was 8.01e-6.
    # Keep modest headroom for reduction-order variance, but reject errors hundreds
    # of times smaller than the old 5e-3 allowance.
    assert max_abs < 1e-5
    assert max_row_l1 < 2e-5
    assert max_mass_error < 1e-6

    # A top-p reduction may place the one cutoff token on the opposite side of 0.9.
    # No other filter is allowed a support mismatch, and top-p gets at most that one
    # boundary token per row -- not a blanket tolerance over the whole distribution.
    top_p_enabled = top_ps < 1.0
    assert not support_difference[~top_p_enabled].any()
    assert support_difference.sum(-1).max().item() <= 1


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
        tokens, probs = fused.fused_sample_from_logits_with_probs(
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

    _, probs_both = fused.fused_sample_from_logits_with_probs(
        logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
    )
    probs_only = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    torch.testing.assert_close(probs_both, probs_only, atol=1e-6, rtol=1e-5)


def test_same_seed_and_offset_reproduce_the_same_tokens():
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(16, 4096, device=dev) * 2.0
    temps, top_ks, top_ps, min_ps = _params(16, device=dev, top_p=0.95)
    seed, offset = _rng(16, dev, seed=1234, offset=5)

    first = fused.fused_sample_from_logits(
        logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
    )
    second = fused.fused_sample_from_logits(
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
    tokens, probs = fused.fused_sample_from_logits_with_probs(
        logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
    )

    counts = torch.bincount(tokens.long(), minlength=vocab).float() / draws
    tv = 0.5 * (counts - probs[0]).abs().sum().item()
    assert tv < 0.05, f"total-variation distance {tv:.3f} between draws and the reported probs"


# --- The tokens-only rejection path ------------------------------------------------
#
# ``fused_sample_from_logits`` (tokens, no probs) does not solve for the cutoff at
# all: it draws from a superset of the kept set and retries until the draw lands inside.
# The tests above reach it only through ``test_same_seed_and_offset_reproduce_the_same_
# tokens``, and determinism says nothing about *which* distribution is reproduced -- so
# the ones below drive it directly, against the descent path's own probs.

# Rows without top-k take rejection; pure top-k keeps the descent. A small top-k + top-p
# batch uses the hybrid covered separately below, while these deliberately wide batches
# exercise its deterministic fallback. All routes are listed because they must agree on
# the distribution -- a routing change must fail here rather than quietly sample from a
# different set.
_REJECT_FILTERS = [
    pytest.param({}, id="neutral"),
    pytest.param({"top_p": 0.8}, id="top_p"),
    pytest.param({"min_p": 0.05}, id="min_p"),
    pytest.param({"min_p": 0.02, "top_p": 0.8}, id="min_p_top_p"),
    pytest.param({"top_k": 8}, id="top_k_descent"),
    pytest.param({"min_p": 0.02, "top_k": 16}, id="min_p_top_k_descent"),
    pytest.param({"top_k": 16, "top_p": 0.8}, id="top_k_top_p_descent"),
]


@pytest.mark.parametrize("filters", _REJECT_FILTERS)
def test_rejection_tokens_lie_in_the_filtered_support(filters):
    """A rejection sampler that accepts too readily still returns a plausible token, so
    membership is the assertion that catches it."""
    dev = "cuda"
    torch.manual_seed(0)
    rows, vocab = 256, 4096
    logits = torch.randn(rows, vocab, device=dev) * 2.0
    temps, top_ks, top_ps, min_ps = _params(rows, device=dev, **filters)
    reference = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)

    for step in range(8):
        seed, offset = _rng(rows, dev, offset=step)
        tokens = fused.fused_sample_from_logits(
            logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
        )
        picked = reference.gather(1, tokens.long().unsqueeze(-1)).squeeze(-1)
        assert (picked > 0).all(), f"step {step} sampled a token the filters removed"


@pytest.mark.parametrize("filters", _REJECT_FILTERS)
def test_rejection_tokens_follow_the_filtered_distribution(filters):
    """The acceptance test decides *which* distribution the loop converges to; an
    off-by-one in it (``<`` vs ``<=`` against the count/mass target) keeps or drops the
    boundary token and shifts the distribution without ever leaving the support."""
    dev = "cuda"
    torch.manual_seed(0)
    vocab, draws = 64, 32768
    logits = (torch.randn(1, vocab, device=dev) * 1.5).expand(draws, vocab).contiguous()
    temps, top_ks, top_ps, min_ps = _params(draws, device=dev, temperature=1.0, **filters)

    seed = torch.tensor([99], dtype=torch.int64, device=dev)
    offset = torch.tensor([0], dtype=torch.int64, device=dev)
    tokens = fused.fused_sample_from_logits(
        logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
    )
    expected = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)[0]

    counts = torch.bincount(tokens.long(), minlength=vocab).float() / draws
    tv = 0.5 * (counts - expected).abs().sum().item()
    assert tv < 0.05, f"total-variation distance {tv:.3f} from the filtered distribution"


def test_small_batch_top_k_top_p_hybrid_follows_filtered_distribution():
    """Rows <= 8 solve top-k, then reject only for top-p instead of descending twice.

    The large one-row-per-draw distribution test above intentionally exceeds that routing
    cutoff, so collect independent launches here to exercise the actual small-batch path.
    Offsets are spaced by the rejection-round budget so one launch cannot reuse another
    launch's later Philox draw.
    """
    dev = "cuda"
    torch.manual_seed(0)
    rows, vocab, steps = 8, 64, 1024
    logits = (torch.randn(1, vocab, device=dev) * 1.5).expand(rows, vocab).contiguous()
    temps, top_ks, top_ps, min_ps = _params(rows, device=dev, temperature=1.0, top_k=16, top_p=0.8)
    expected = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)[0]

    draws = []
    for step in range(steps):
        seed, offset = _rng(rows, dev, seed=99, offset=step * 32)
        draws.append(
            fused.fused_sample_from_logits(
                logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
            )
        )

    tokens = torch.stack(draws).flatten().long()
    assert (expected[tokens] > 0).all(), "hybrid sampled a token outside the filtered support"
    counts = torch.bincount(tokens, minlength=vocab).float() / tokens.numel()
    tv = 0.5 * (counts - expected).abs().sum().item()
    assert tv < 0.05, f"total-variation distance {tv:.3f} from the filtered distribution"


@pytest.mark.parametrize(
    "filters",
    [pytest.param({"top_p": 1e-6}, id="top_p"), pytest.param({"top_k": 1}, id="top_k_descent")],
)
def test_rejection_converges_when_the_support_barely_shrinks(filters):
    """The round budget's worst case: a near-flat row keeping exactly one token.

    Every rejection here removes only the weights at or below the draw, and on a flat row
    a draw is uniform over the support -- so the support halves per round rather than
    collapsing, which is the slowest descent the loop can be given. Both filters admit the
    argmax alone, so a budget that ran out would show up as a wrong token, not a slow one.
    """
    dev = "cuda"
    rows, vocab = 512, 4096
    # Distinct but nearly equal, so no tie lets the loop finish early.
    logits = (torch.arange(vocab, device=dev, dtype=torch.float32) * 1e-4).expand(rows, vocab)
    temps, top_ks, top_ps, min_ps = _params(rows, device=dev, temperature=1.0, **filters)

    for step in range(4):
        seed, offset = _rng(rows, dev, offset=step)
        tokens = fused.fused_sample_from_logits(
            logits.contiguous(), temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
        )
        assert torch.equal(tokens.long(), torch.full_like(tokens.long(), vocab - 1))


def test_rejection_and_descent_keep_the_same_set():
    """Mixed batch: rows that take the rejection path and rows that fall through to the
    descent are dispatched per row, inside one launch. Both must sample from the set the
    probs-only op reports for that row."""
    dev = "cuda"
    torch.manual_seed(0)
    rows, vocab = 64, 4096
    logits = torch.randn(rows, vocab, device=dev) * 2.0
    temps, top_ks, top_ps, min_ps = _params(rows, device=dev)
    # Row r cycles through: neutral, top-k, top-p, min-p, and the top-k+top-p fallback.
    top_ks[1::5] = 32
    top_ps[2::5] = 0.7
    min_ps[3::5] = 0.05
    top_ks[4::5] = 32
    top_ps[4::5] = 0.7

    reference = fused.fused_compute_probs_from_logits(logits, temps, top_ks, top_ps, min_ps)
    for step in range(8):
        seed, offset = _rng(rows, dev, offset=step)
        tokens = fused.fused_sample_from_logits(
            logits, temps, top_ks, top_ps, min_ps, seed=seed, offset=offset
        )
        picked = reference.gather(1, tokens.long().unsqueeze(-1)).squeeze(-1)
        assert (picked > 0).all(), f"step {step} sampled outside its row's support"
