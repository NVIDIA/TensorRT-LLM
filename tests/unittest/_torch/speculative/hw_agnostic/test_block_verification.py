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
"""Block verification (Sun et al., 2024) for one-model speculative decoding.

Pins four properties of ``block_verification.block_verify``:

* it is the rule vLLM serves as ``rejection_sample_method="block"``: a direct
  port of vLLM's kernels makes the same decisions from the same uniforms;
* with one draft it is token-level rejection sampling, decision for decision;
* it is lossless: drafted blocks verified by it, then completed by sampling
  the target, are distributed exactly as the target's own samples;
* it runs inside a captured CUDA graph.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.speculative.block_verification import (
    block_acceptance_thresholds,
    block_verify,
    counter_uniforms,
    sample_from_weights,
)

# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _vllm_block_reference(target_logits, draft_logits, temperature, draft_tokens, uniforms):
    """Port of vLLM's block path for one request (``rejection_sampler_utils.py``).

    Mirrors ``_compute_cumulative_log_p_kernel``, ``_compute_local_residual_mass_kernel``,
    the ``USE_BLOCK_VERIFICATION`` branch of ``_rejection_kernel`` and the block
    branch of ``_resample_kernel`` with draft logits present, in float64.

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


def _token_level_reference(draft_tokens, draft_probs, target_probs, uniforms, resample_uniforms):
    """Token-level rejection sampling (Leviathan et al., 2023), one request at a time."""
    batch, num_drafts = draft_tokens.shape
    accepted_tokens = torch.full((batch, num_drafts + 1), -1, dtype=torch.int64)
    num_accepted = torch.empty(batch, dtype=torch.int64)
    for b in range(batch):
        length = num_drafts
        for j in range(num_drafts):
            token = int(draft_tokens[b, j])
            p, q = float(target_probs[b, j, token]), float(draft_probs[b, j, token])
            if float(uniforms[b, j]) <= p / q:
                accepted_tokens[b, j] = token
                continue
            length = j
            break
        if length == num_drafts:
            weights = target_probs[b, num_drafts]
        else:
            weights = (target_probs[b, length] - draft_probs[b, length]).clamp_min(0)
        accepted_tokens[b, length] = sample_from_weights(
            weights[None].double(), resample_uniforms[b : b + 1]
        )[0]
        num_accepted[b] = length + 1
    return accepted_tokens, num_accepted


def _random_case(generator, batch, num_drafts, vocab, temperature=1.0, mask_frac=0.3):
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


# ---------------------------------------------------------------------------
# Equivalence with vLLM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_drafts", [1, 3, 7])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_block_verify_matches_vllm_given_identical_uniforms(num_drafts, temperature):
    generator = torch.Generator().manual_seed(1000 + num_drafts)
    batch, vocab = 256, 24
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


def test_residual_distribution_matches_vllm():
    """The emitted token's distribution, not just one draw from it, is vLLM's."""
    generator = torch.Generator().manual_seed(7)
    batch, num_drafts, vocab = 24, 4, 16
    target_logits, draft_logits, draft_probs, draft_tokens, target_probs = _random_case(
        generator, batch, num_drafts, vocab
    )
    uniforms = torch.rand(batch, num_drafts, generator=generator, dtype=torch.float64)
    resample_draws = 4096
    for b in range(batch):
        ref_length, _, ref_resample = _vllm_block_reference(
            target_logits[b], draft_logits[b], 1.0, draft_tokens[b], uniforms[b]
        )
        # Every draw of the emitted token with this request's acceptance uniforms.
        grid = (torch.arange(resample_draws, dtype=torch.float64) + 0.5) / resample_draws
        tokens, num_accepted = block_verify(
            draft_tokens[b : b + 1].expand(resample_draws, -1),
            draft_probs[b : b + 1].expand(resample_draws, -1, -1),
            target_probs[b : b + 1].expand(resample_draws, -1, -1),
            uniforms[b : b + 1].expand(resample_draws, -1),
            grid,
        )
        if (num_accepted != ref_length + 1).any():
            continue  # boundary case, covered by the test above
        emitted = tokens[:, ref_length].long()
        histogram = torch.bincount(emitted, minlength=vocab).double() / resample_draws
        torch.testing.assert_close(histogram, ref_resample, atol=2.0 / resample_draws, rtol=0)


def test_sample_emitted_hook_receives_the_residual():
    """A caller-supplied sampler sees exactly the residual vLLM resamples from."""
    generator = torch.Generator().manual_seed(8)
    batch, num_drafts, vocab = 64, 4, 16
    target_logits, draft_logits, draft_probs, draft_tokens, target_probs = _random_case(
        generator, batch, num_drafts, vocab
    )
    uniforms = torch.rand(batch, num_drafts, generator=generator, dtype=torch.float64)
    seen = {}

    def sampler(weights):
        seen["weights"] = weights
        return weights.argmax(-1)

    tokens, num_accepted = block_verify(
        draft_tokens, draft_probs, target_probs, uniforms, sample_emitted=sampler
    )
    weights = seen["weights"].double()
    for b in range(batch):
        ref_length, ref_thresholds, ref_resample = _vllm_block_reference(
            target_logits[b], draft_logits[b], 1.0, draft_tokens[b], uniforms[b]
        )
        if (uniforms[b] - ref_thresholds).abs().min() < 1e-4:
            continue
        assert int(num_accepted[b]) == ref_length + 1
        torch.testing.assert_close(weights[b] / weights[b].sum(), ref_resample, atol=1e-5, rtol=0)
        assert int(tokens[b, ref_length]) == int(weights[b].argmax())


# ---------------------------------------------------------------------------
# One draft: block verification is token-level verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_draft_block_equals_token_level(seed):
    generator = torch.Generator().manual_seed(seed)
    batch, vocab = 512, 32
    _, _, draft_probs, draft_tokens, target_probs = _random_case(generator, batch, 1, vocab)
    uniforms = torch.rand(batch, 1, generator=generator, dtype=torch.float64)
    resample_uniforms = torch.rand(batch, generator=generator, dtype=torch.float64)

    block_tokens, block_accepted = block_verify(
        draft_tokens, draft_probs, target_probs, uniforms, resample_uniforms
    )
    token_tokens, token_accepted = _token_level_reference(
        draft_tokens, draft_probs, target_probs, uniforms, resample_uniforms
    )
    p_x = target_probs[:, 0].gather(-1, draft_tokens).squeeze(-1).double()
    q_x = draft_probs[:, 0].gather(-1, draft_tokens).squeeze(-1).double()
    off_boundary = (uniforms[:, 0] - (p_x / q_x).clamp(max=1)).abs() > 1e-6
    assert off_boundary.float().mean() > 0.99
    assert torch.equal(block_accepted[off_boundary].long(), token_accepted[off_boundary])
    assert torch.equal(block_tokens[off_boundary].long(), token_tokens[off_boundary])
    # Both outcomes occur, so the comparison covers acceptance and resampling.
    assert set(block_accepted.tolist()) == {1, 2}


# ---------------------------------------------------------------------------
# Losslessness on a toy model with prefix-dependent conditionals
# ---------------------------------------------------------------------------

_VOCAB = 3
_BLOCK = 3  # drafts per block; the verified sequences have _BLOCK + 1 tokens


class _PrefixModel:
    """A distribution over the next token for every prefix of length 0.._BLOCK."""

    def __init__(self, generator, zero_frac=0.0, sharpness=1.0):
        num_prefixes = sum(_VOCAB**length for length in range(_BLOCK + 1))
        logits = sharpness * torch.randn(num_prefixes, _VOCAB, generator=generator)
        zero = torch.rand(num_prefixes, _VOCAB, generator=generator) < zero_frac
        zero[:, 0] = False
        self.table = torch.softmax(logits.masked_fill(zero, float("-inf")).double(), -1)

    @staticmethod
    def prefix_ids(tokens: torch.Tensor) -> torch.Tensor:
        """[N, m] prefixes -> row ids (prefixes of each length laid out base-_VOCAB)."""
        length = tokens.shape[1]
        base = sum(_VOCAB**m for m in range(length))
        ids = torch.zeros(tokens.shape[0], dtype=torch.int64)
        for m in range(length):
            ids = ids * _VOCAB + tokens[:, m]
        return ids + base

    def probs(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.table[self.prefix_ids(tokens)]

    def sequence_probs(self) -> torch.Tensor:
        """Exact probability of every length-(_BLOCK + 1) sequence, in base-_VOCAB order."""
        grid = torch.cartesian_prod(*[torch.arange(_VOCAB)] * (_BLOCK + 1))
        prob = torch.ones(grid.shape[0], dtype=torch.float64)
        for m in range(_BLOCK + 1):
            prob = prob * self.probs(grid[:, :m]).gather(-1, grid[:, m : m + 1]).squeeze(-1)
        return prob


def _verify_and_complete(verify, target, draft, num_samples, generator):
    """Draft ancestrally, verify, then sample the target up to _BLOCK + 1 tokens."""
    drafts = torch.empty(num_samples, 0, dtype=torch.int64)
    draft_rows = []
    for _ in range(_BLOCK):
        q = draft.probs(drafts)
        draft_rows.append(q)
        drafts = torch.cat([drafts, torch.multinomial(q, 1, generator=generator)], dim=1)
    target_rows = [target.probs(drafts[:, :j]) for j in range(_BLOCK + 1)]
    draft_probs = torch.stack(draft_rows, dim=1).float()
    target_probs = torch.stack(target_rows, dim=1).float()
    uniforms = torch.rand(num_samples, _BLOCK, generator=generator, dtype=torch.float64)
    resample_uniforms = torch.rand(num_samples, generator=generator, dtype=torch.float64)
    tokens, num_accepted = verify(drafts, draft_probs, target_probs, uniforms, resample_uniforms)

    sequences = torch.zeros(num_samples, _BLOCK + 1, dtype=torch.int64)
    emitted = num_accepted.long()
    for m in range(_BLOCK + 1):
        continued = torch.multinomial(target.probs(sequences[:, :m]), 1, generator=generator)
        sequences[:, m] = torch.where(m < emitted, tokens[:, m].long(), continued.squeeze(-1))
    ids = torch.zeros(num_samples, dtype=torch.int64)
    for m in range(_BLOCK + 1):
        ids = ids * _VOCAB + sequences[:, m]
    return torch.bincount(ids, minlength=_VOCAB ** (_BLOCK + 1)), emitted


def _chi_square_ok(counts: torch.Tensor, probs: torch.Tensor) -> tuple[float, float]:
    """Pearson statistic over cells with expected count >= 5 (the rest pooled) vs its 1e-4 critical value."""
    total = counts.sum().item()
    expected = probs * total
    big = expected >= 5
    observed = torch.cat([counts[big].double(), counts[~big].double().sum().view(1)])
    expected = torch.cat([expected[big], expected[~big].sum().view(1)])
    keep = expected > 0
    stat = (((observed - expected) ** 2)[keep] / expected[keep]).sum().item()
    dof = int(keep.sum()) - 1
    z = 3.719  # one-sided 1e-4 normal quantile; Wilson-Hilferty approximation below
    critical = dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3
    return stat, critical


def _block(drafts, draft_probs, target_probs, uniforms, resample_uniforms):
    return block_verify(drafts, draft_probs, target_probs, uniforms, resample_uniforms)


@pytest.mark.parametrize("verify_name", ["block", "token"])
@pytest.mark.parametrize("zero_frac", [0.0, 0.3])
def test_verification_is_lossless(verify_name, zero_frac):
    generator = torch.Generator().manual_seed(11 if zero_frac else 12)
    target = _PrefixModel(generator, zero_frac=zero_frac, sharpness=1.5)
    draft = _PrefixModel(generator, sharpness=1.0)
    verify = _block if verify_name == "block" else _token_level_reference
    num_samples = 200_000 if verify_name == "block" else 20_000
    counts, _ = _verify_and_complete(verify, target, draft, num_samples, generator)

    exact = target.sequence_probs()
    assert counts[exact == 0].sum() == 0, "emitted a sequence the target cannot produce"
    stat, critical = _chi_square_ok(counts, exact)
    assert stat < critical, f"chi-square {stat:.1f} >= {critical:.1f}"


def test_block_accepts_at_least_as_much_as_token_level():
    generator = torch.Generator().manual_seed(5)
    target = _PrefixModel(generator, sharpness=1.5)
    draft = _PrefixModel(generator, sharpness=1.0)
    num_samples = 20_000
    _, block_emitted = _verify_and_complete(_block, target, draft, num_samples, generator)
    _, token_emitted = _verify_and_complete(
        _token_level_reference, target, draft, num_samples, generator
    )
    block_mean = block_emitted.double().mean().item()
    token_mean = token_emitted.double().mean().item()
    # Three standard errors of the difference of two means of values in [1, 4].
    assert block_mean > token_mean - 3 * 1.5 * math.sqrt(2 / num_samples)


def test_regression_independent_draws_from_mismatched_rows_are_not_lossless():
    """Guard for the test itself: storing a q the drafts were not drawn from must fail it."""
    generator = torch.Generator().manual_seed(3)
    target = _PrefixModel(generator, sharpness=1.5)
    draft = _PrefixModel(generator, sharpness=1.0)

    def verify_with_wrong_q(drafts, draft_probs, target_probs, uniforms, resample_uniforms):
        wrong = torch.softmax(torch.log(draft_probs.double()) * 0.3, -1).float()
        return block_verify(drafts, wrong, target_probs, uniforms, resample_uniforms)

    counts, _ = _verify_and_complete(verify_with_wrong_q, target, draft, 200_000, generator)
    stat, critical = _chi_square_ok(counts, target.sequence_probs())
    assert stat > critical


# ---------------------------------------------------------------------------
# Output layout, degenerate rows, uniforms
# ---------------------------------------------------------------------------


def test_output_layout_and_identical_draft_and_target():
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


def test_rejected_block_pads_with_minus_one_and_never_emits_a_filtered_token():
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


def test_a_draft_its_proposal_gives_no_mass_is_rejected():
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


def test_nan_rows_stay_in_vocab():
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


def test_counter_uniforms_are_keyed_and_uniform():
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


# ---------------------------------------------------------------------------
# CUDA graph
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs CUDA")
def test_block_verify_captures_and_replays_in_a_cuda_graph():
    generator = torch.Generator().manual_seed(17)
    batch, num_drafts, vocab = 8, 7, 4096

    def make_inputs():
        _, _, draft_probs, draft_tokens, target_probs = _random_case(
            generator, batch, num_drafts, vocab
        )
        seed = torch.randint(0, 1 << 40, (batch,), generator=generator)
        offset = torch.randint(0, 1 << 40, (batch,), generator=generator)
        return [t.cuda() for t in (draft_tokens, draft_probs, target_probs, seed, offset)]

    def run(draft_tokens, draft_probs, target_probs, seed, offset):
        uniforms = counter_uniforms(seed, offset, num_drafts + 1)
        return block_verify(
            draft_tokens, draft_probs, target_probs, uniforms[:, :num_drafts], uniforms[:, -1]
        )

    static = make_inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(*static)  # warm up outside the graph
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_tokens, graph_accepted = run(*static)

    for _ in range(3):
        fresh = make_inputs()
        for dst, src in zip(static, fresh):
            dst.copy_(src)
        graph.replay()
        eager_tokens, eager_accepted = run(*fresh)
        torch.testing.assert_close(graph_accepted, eager_accepted, rtol=0, atol=0)
        torch.testing.assert_close(graph_tokens, eager_tokens, rtol=0, atol=0)
        cpu_tokens, cpu_accepted = run(*[t.cpu() for t in fresh])
        assert torch.equal(cpu_accepted, eager_accepted.cpu())
