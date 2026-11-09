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
"""Block verification (Sun et al., 2024): vLLM parity, losslessness on prefix-dependent targets, degenerate rows."""

import math

import pytest
import torch

from tensorrt_llm._torch.speculative.block_verification import (
    block_acceptance_thresholds,
    block_verify,
    counter_uniforms,
    sample_from_weights,
)

# References and statistical helpers: toy prefix-conditioned models and a chi-square check.
# A prefix table holds one next-token distribution per prefix of length 0..block, rows laid
# out by prefix length and then base-``vocab``.


def random_prefix_table(
    generator: torch.Generator,
    vocab: int,
    block: int,
    sharpness: float = 1.0,
    zero_frac: float = 0.0,
) -> torch.Tensor:
    """Random next-token distributions for every prefix; ``zero_frac`` of entries get no mass."""
    num_prefixes = sum(vocab**length for length in range(block + 1))
    logits = sharpness * torch.randn(num_prefixes, vocab, generator=generator)
    zero = torch.rand(num_prefixes, vocab, generator=generator) < zero_frac
    zero[:, 0] = False  # keep every row nonempty
    return torch.softmax(logits.masked_fill(zero, float("-inf")).double(), -1)


def prefix_ids(tokens: torch.Tensor, vocab: int) -> torch.Tensor:
    """``[N, m]`` prefixes -> their prefix-table row ids."""
    length = tokens.shape[1]
    ids = torch.zeros(tokens.shape[0], dtype=torch.int64)
    for m in range(length):
        ids = ids * vocab + tokens[:, m]
    return ids + sum(vocab**m for m in range(length))


def exact_sequence_probs(table: torch.Tensor, length: int) -> torch.Tensor:
    """Exact probability of every length-``length`` sequence under ``table``, in base-vocab order."""
    vocab = table.shape[-1]
    grid = torch.cartesian_prod(*[torch.arange(vocab)] * length)
    prob = torch.ones(grid.shape[0], dtype=torch.float64)
    for m in range(length):
        rows = table[prefix_ids(grid[:, :m], vocab)]
        prob = prob * rows.gather(-1, grid[:, m : m + 1]).squeeze(-1)
    return prob


def complete_and_count(
    table: torch.Tensor,
    accepted_tokens: torch.Tensor,
    num_accepted: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Emitted tokens, completed by sampling ``table`` to the block length, counted per sequence."""
    vocab = table.shape[-1]
    num, length = accepted_tokens.shape
    sequences = torch.zeros(num, length, dtype=torch.int64)
    for m in range(length):
        cont = torch.multinomial(table[prefix_ids(sequences[:, :m], vocab)], 1, generator=generator)
        keep = m < num_accepted.long()
        sequences[:, m] = torch.where(keep, accepted_tokens[:, m].long(), cont.squeeze(-1))
    ids = torch.zeros(num, dtype=torch.int64)
    for m in range(length):
        ids = ids * vocab + sequences[:, m]
    return torch.bincount(ids, minlength=vocab**length)


def chi_square(counts: torch.Tensor, probs: torch.Tensor) -> tuple[float, float]:
    """Pearson statistic over cells with expected count >= 5 (the rest pooled) vs its 1e-4 critical value."""
    assert counts[probs == 0].sum() == 0, "observations in zero-probability cells"
    total = counts.sum().item()
    expected = probs * total
    big = expected >= 5
    observed = torch.cat([counts[big].double(), counts[~big].double().sum().view(1)])
    expected = torch.cat([expected[big], expected[~big].sum().view(1)])
    keep = expected > 0
    stat = (((observed - expected) ** 2)[keep] / expected[keep]).sum().item()
    dof = int(keep.sum()) - 1
    z = 3.719  # one-sided 1e-4 normal quantile; Wilson-Hilferty approximation below
    return stat, dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3


def _vllm_block_reference(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    temperature: float,
    draft_tokens: torch.Tensor,
    uniforms: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Independent float64 re-implementation of vLLM's block path for one request.

    No vLLM code is copied. It follows
    vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py at
    vllm-project/vllm@66875d3127534183b5ea723964fc7ec388e89722, mirroring
    ``_compute_cumulative_log_p_kernel``, ``_compute_local_residual_mass_kernel``,
    the ``USE_BLOCK_VERIFICATION`` branch of ``_rejection_kernel`` and the block
    branch of ``_resample_kernel`` with draft logits present.

    Args:
        target_logits: ``[K + 1, V]`` processed target logits (``-inf`` where filtered).
        draft_logits: ``[K, V]`` cached pre-temperature draft logits.
        temperature: the request's temperature.
        draft_tokens: ``[K]``.
        uniforms: ``[K]``.
    Returns:
        ``(accepted_length, thresholds [K], resample distribution [V])``.
    """
    target_logits = target_logits.double()
    draft_scaled = draft_logits.double() / temperature
    num_drafts = draft_tokens.shape[0]
    target_lse = torch.logsumexp(target_logits, -1)
    draft_lse = torch.logsumexp(draft_scaled, -1)

    cumulative_log_p = []
    log_p = 0.0
    for step in range(num_drafts):
        token = int(draft_tokens[step])
        target_logprob = float(target_logits[step, token] - target_lse[step])
        draft_logprob = float(draft_scaled[step, token] - draft_lse[step])
        log_p = min(log_p + (target_logprob - draft_logprob), 0.0)
        cumulative_log_p.append(log_p)

    residual_mass = {}
    for logit_idx in range(1, num_drafts):
        p = math.exp(cumulative_log_p[logit_idx - 1])
        m_b = torch.exp(target_logits[logit_idx] - target_lse[logit_idx])
        m_s = torch.exp(draft_scaled[logit_idx] - draft_lse[logit_idx])
        residual_mass[logit_idx] = float(torch.clamp(p * m_b - m_s, min=0.0).sum())

    accepted_length = 0
    thresholds = []
    for i in range(num_drafts):
        prefix_joint_ratio = math.exp(cumulative_log_p[i])
        if i < num_drafts - 1:
            mass = residual_mass[i + 1]
            denom = mass + 1.0 - prefix_joint_ratio
            h = mass / denom if denom > 0.0 else 1.0
        else:
            h = prefix_joint_ratio
        thresholds.append(h)
        if float(uniforms[i]) <= h:
            accepted_length = i + 1

    if accepted_length == num_drafts:
        residual_logits = target_logits[num_drafts]
    else:
        idx = accepted_length
        target_log_probs = target_logits[idx] - target_lse[idx]
        if idx > 0:
            target_log_probs = target_log_probs + cumulative_log_p[idx - 1]
        draft_log_probs = draft_scaled[idx] - draft_lse[idx]
        ratio = torch.exp(draft_log_probs - target_log_probs)
        residual_logits = torch.where(
            ratio < 1.0, target_log_probs + torch.log1p(-ratio), float("-inf")
        )
    return (
        accepted_length,
        torch.tensor(thresholds, dtype=torch.float64),
        torch.softmax(residual_logits, -1),
    )


def _random_case(
    generator: torch.Generator,
    batch: int,
    num_drafts: int,
    vocab: int,
    temperature: float = 1.0,
    mask_frac: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Processed target logits with a top-p-like mask, raw draft logits, and drafts drawn from them."""
    target_logits = 2.0 * torch.randn(batch, num_drafts + 1, vocab, generator=generator)
    masked = torch.rand(batch, num_drafts + 1, vocab, generator=generator) < mask_frac
    masked[..., 0] = False  # keep every row nonempty
    target_logits = target_logits.masked_fill(masked, float("-inf"))
    # Correlated with the target so blocks are neither always nor never accepted.
    draft_logits = target_logits[:, :num_drafts].nan_to_num(neginf=-4.0) + torch.randn(
        batch, num_drafts, vocab, generator=generator
    )
    draft_probs = torch.softmax(draft_logits.double() / temperature, -1).float()
    draft_tokens = torch.multinomial(draft_probs.reshape(-1, vocab), 1, generator=generator).view(
        batch, num_drafts
    )
    target_probs = torch.softmax(target_logits.double(), -1).float()
    return target_logits, draft_logits, draft_probs, draft_tokens, target_probs


# Equivalence with vLLM


@pytest.mark.parametrize("num_drafts", [1, 3, 7])
@pytest.mark.cpu_only
def test_block_verify_matches_vllm_given_identical_uniforms(num_drafts: int) -> None:
    generator = torch.Generator().manual_seed(1000 + num_drafts)
    batch, vocab, temperature = 256, 24, 0.6
    target_logits, draft_logits, draft_probs, draft_tokens, target_probs = _random_case(
        generator, batch, num_drafts, vocab, temperature
    )
    uniforms = torch.rand(batch, num_drafts, generator=generator, dtype=torch.float64)
    resample_uniforms = torch.rand(batch, generator=generator, dtype=torch.float64)

    _, thresholds = block_acceptance_thresholds(draft_tokens, draft_probs, target_probs)
    accepted_tokens, num_accepted = block_verify(
        draft_tokens, draft_probs, target_probs, uniforms, resample_uniforms
    )

    checked = 0
    lengths = set()
    for b in range(batch):
        ref_length, ref_thresholds, ref_resample = _vllm_block_reference(
            target_logits[b], draft_logits[b], temperature, draft_tokens[b], uniforms[b]
        )
        torch.testing.assert_close(thresholds[b].double(), ref_thresholds, atol=2e-5, rtol=0)
        if (uniforms[b] - ref_thresholds).abs().min() < 1e-4:
            continue  # a decision on a rounding boundary, not a semantic difference
        checked += 1
        lengths.add(ref_length)
        assert int(num_accepted[b]) == ref_length + 1
        assert accepted_tokens[b, :ref_length].tolist() == draft_tokens[b, :ref_length].tolist()
        assert (accepted_tokens[b, ref_length + 1 :] == -1).all()
        expected = sample_from_weights(ref_resample[None], resample_uniforms[b : b + 1])[0]
        assert int(accepted_tokens[b, ref_length]) == int(expected)
    assert checked > 0.95 * batch
    if num_drafts > 1:
        assert len(lengths) > 2, "the case generator should exercise several accept lengths"


# Losslessness on a toy model with prefix-dependent conditionals

_VOCAB = 4
_BLOCK = 3  # drafts per block; the verified sequences have _BLOCK + 1 tokens


def _verify_and_complete(
    target: torch.Tensor,
    draft: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
    stored_draft: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draft ancestrally from the ``draft`` table, verify, then complete from ``target``.

    ``stored_draft`` is the table handed to the verifier as the proposal (``draft`` itself
    unless a test deliberately stores the wrong one).
    """
    stored = draft if stored_draft is None else stored_draft
    drafts = torch.empty(num_samples, 0, dtype=torch.int64)
    draft_rows = []
    for _ in range(_BLOCK):
        rows = prefix_ids(drafts, _VOCAB)
        draft_rows.append(stored[rows])
        drafts = torch.cat([drafts, torch.multinomial(draft[rows], 1, generator=generator)], dim=1)
    target_rows = [target[prefix_ids(drafts[:, :j], _VOCAB)] for j in range(_BLOCK + 1)]
    draft_probs = torch.stack(draft_rows, dim=1).float()
    target_probs = torch.stack(target_rows, dim=1).float()
    uniforms = torch.rand(num_samples, _BLOCK, generator=generator, dtype=torch.float64)
    resample_uniforms = torch.rand(num_samples, generator=generator, dtype=torch.float64)
    tokens, num_accepted = block_verify(
        drafts, draft_probs, target_probs, uniforms, resample_uniforms
    )
    return complete_and_count(target, tokens, num_accepted, generator), num_accepted.long()


def _tables(
    generator: torch.Generator, zero_frac: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """A sharper target table (optionally with zero-mass entries) and a draft table."""
    target = random_prefix_table(generator, _VOCAB, _BLOCK, sharpness=1.5, zero_frac=zero_frac)
    return target, random_prefix_table(generator, _VOCAB, _BLOCK)


@pytest.mark.parametrize("zero_frac", [0.0, 0.3])
@pytest.mark.cpu_only
def test_verification_is_lossless(zero_frac: float) -> None:
    generator = torch.Generator().manual_seed(11 if zero_frac else 12)
    target, draft = _tables(generator, zero_frac)
    counts, _ = _verify_and_complete(target, draft, 200_000, generator)

    exact = exact_sequence_probs(target, _BLOCK + 1)
    assert counts[exact == 0].sum() == 0, "emitted a sequence the target cannot produce"
    stat, critical = chi_square(counts, exact)
    assert stat < critical, f"chi-square {stat:.1f} >= {critical:.1f}"


@pytest.mark.cpu_only
def test_regression_independent_draws_from_mismatched_rows_are_not_lossless() -> None:
    """Guard for the test itself: storing a q the drafts were not drawn from must fail it."""
    generator = torch.Generator().manual_seed(3)
    target, draft = _tables(generator)
    wrong = torch.softmax(torch.log(draft) * 0.3, -1)
    counts, _ = _verify_and_complete(target, draft, 200_000, generator, stored_draft=wrong)
    stat, critical = chi_square(counts, exact_sequence_probs(target, _BLOCK + 1))
    assert stat > critical


# Output layout, degenerate rows, uniforms


@pytest.mark.cpu_only
def test_output_layout_and_identical_draft_and_target() -> None:
    """q == p: everything is accepted and the bonus comes from the last row."""
    generator = torch.Generator().manual_seed(9)
    batch, num_drafts, vocab = 32, 5, 11
    probs = torch.softmax(torch.randn(batch, num_drafts + 1, vocab, generator=generator), -1)
    drafts = torch.multinomial(probs[:, :num_drafts].reshape(-1, vocab), 1, generator=generator)
    drafts = drafts.view(batch, num_drafts)
    uniforms = torch.rand(batch, num_drafts, generator=generator, dtype=torch.float64)
    resample_uniforms = torch.rand(batch, generator=generator, dtype=torch.float64)
    tokens, num_accepted = block_verify(
        drafts, probs[:, :num_drafts], probs, uniforms, resample_uniforms
    )
    assert tokens.dtype == num_accepted.dtype == torch.int32
    assert (num_accepted == num_drafts + 1).all()
    assert torch.equal(tokens[:, :num_drafts].long(), drafts)
    bonus = sample_from_weights(probs[:, num_drafts], resample_uniforms)
    assert torch.equal(tokens[:, num_drafts].long(), bonus)


@pytest.mark.cpu_only
def test_rejected_block_pads_with_minus_one_and_never_emits_a_filtered_token() -> None:
    batch, num_drafts, vocab = 4, 3, 6
    target = torch.zeros(batch, num_drafts + 1, vocab)
    target[..., 1] = 1.0  # the target only ever allows token 1
    draft = torch.zeros(batch, num_drafts, vocab)
    draft[..., 4] = 1.0  # the drafter always proposes token 4
    drafts = torch.full((batch, num_drafts), 4)
    uniforms = torch.full((batch, num_drafts), 0.5, dtype=torch.float64)
    tokens, num_accepted = block_verify(
        drafts, draft, target, uniforms, torch.full((batch,), 0.99, dtype=torch.float64)
    )
    assert (num_accepted == 1).all()
    assert (tokens[:, 0] == 1).all()
    assert (tokens[:, 1:] == -1).all()


@pytest.mark.cpu_only
def test_a_draft_its_proposal_gives_no_mass_is_rejected() -> None:
    """A row whose stored q misses the draft (stale/padding rows) must not accept it."""
    vocab = 5
    target = torch.full((1, 2, vocab), 1.0 / vocab)
    draft = torch.zeros(1, 1, vocab)
    draft[..., 2] = 1.0
    tokens, num_accepted = block_verify(
        torch.tensor([[3]]),
        draft,
        target,
        torch.tensor([[1e-9]], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
    )
    assert int(num_accepted) == 1


@pytest.mark.parametrize("corrupt", ["nan", "negative"])
@pytest.mark.cpu_only
def test_a_corrupt_look_ahead_row_never_accepts_a_draft_the_target_rules_out(corrupt: str) -> None:
    """An unwritten proposal row (``draft_probs`` is uninitialized) must not force an accept."""
    vocab = 8
    target = torch.zeros(1, 4, vocab)
    target[0, torch.arange(4), torch.tensor([5, 6, 7, 3])] = 1.0  # greedy one-hot rows
    drafts = torch.tensor([[1, 6, 7]])  # the target gives token 1 zero probability
    draft = torch.zeros(1, 3, vocab)
    draft[0, torch.arange(3), drafts[0]] = 1.0
    draft[0, 1, 2] = float("nan") if corrupt == "nan" else -3.0  # unrelated column, row 1
    draft[0, 2, 4] = float("nan") if corrupt == "nan" else -3.0  # and the emitted row
    tokens, num_accepted = block_verify(
        drafts,
        draft,
        target,
        torch.full((1, 3), 0.5, dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
    )
    assert int(num_accepted) == 1
    assert int(tokens[0, 0]) == 5


@pytest.mark.cpu_only
def test_nan_rows_stay_in_vocab() -> None:
    """CUDA-graph padding rows can carry NaN; their output must still be a valid id."""
    batch, num_drafts, vocab = 3, 4, 7
    target = torch.full((batch, num_drafts + 1, vocab), float("nan"))
    draft = torch.full((batch, num_drafts, vocab), float("nan"))
    tokens, num_accepted = block_verify(
        torch.zeros(batch, num_drafts, dtype=torch.int64),
        draft,
        target,
        torch.rand(batch, num_drafts, dtype=torch.float64),
        torch.rand(batch, dtype=torch.float64),
    )
    emitted = tokens.long().gather(1, (num_accepted.long() - 1).view(-1, 1))
    assert ((emitted >= 0) & (emitted < vocab)).all()


@pytest.mark.cpu_only
def test_counter_uniforms_are_keyed_and_uniform() -> None:
    rows = 4096
    seed = torch.full((rows,), 42, dtype=torch.int64)
    offset = (1 << 62) + 32 * torch.arange(rows, dtype=torch.int64)
    u = counter_uniforms(seed, offset, 8)
    assert u.dtype == torch.float64 and u.shape == (rows, 8)
    assert ((u > 0) & (u < 1)).all()
    assert torch.equal(u, counter_uniforms(seed, offset, 8))
    assert torch.equal(u[:, :3], counter_uniforms(seed, offset, 3))
    assert not torch.equal(u, counter_uniforms(seed + 1, offset, 8))
    assert not torch.equal(u, counter_uniforms(seed, offset + 1, 8))
    assert not torch.equal(u, counter_uniforms(-seed, offset, 8))
    flat = u.flatten()
    assert abs(flat.mean().item() - 0.5) < 0.01
    assert abs(flat.var().item() - 1 / 12) < 0.005
    histogram = torch.histc(flat.float(), bins=16, min=0, max=1)
    assert (histogram - flat.numel() / 16).abs().max() < 5 * math.sqrt(flat.numel() / 16)
    # Neighboring draws, offsets and seeds are uncorrelated.
    centered = u - 0.5
    for a, b in [(centered[:, 0], centered[:, 1]), (centered[:-1, 0], centered[1:, 0])]:
        assert abs((a * b).mean().item() / (1 / 12)) < 0.05
