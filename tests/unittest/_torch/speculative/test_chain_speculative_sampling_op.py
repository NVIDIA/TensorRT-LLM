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
"""Correctness of ``trtllm::chain_speculative_sampling``.

Distribution tests give every row its own seed, so one launch over many identical rows is
many independent trials of the same request.
"""

import pytest
import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.speculative.interface import rejection_sampling_one_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEV = "cuda"


def _probs(rows: int, vocab: int, *, scale: float = 1.5, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return torch.softmax(torch.randn(rows, vocab, device=DEV, generator=gen) * scale, dim=-1)


def _request(
    num_draft: int, vocab: int, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One request: draft probs, draft tokens sampled from them, and target probs."""
    draft_probs = _probs(num_draft, vocab, seed=seed)
    target_probs = _probs(num_draft + 1, vocab, seed=seed + 1)
    gen = torch.Generator(device=DEV).manual_seed(seed + 2)
    draft_tokens = torch.multinomial(draft_probs, 1, generator=gen).squeeze(-1).int()
    return draft_probs, draft_tokens, target_probs


def _repeat(rows: int, *tensors: torch.Tensor) -> list[torch.Tensor]:
    return [t.unsqueeze(0).expand(rows, *t.shape).contiguous() for t in tensors]


def _rng(rows: int, *, seed: int = 1234, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((rows,), seed, dtype=torch.int64, device=DEV),
        torch.full((rows,), offset, dtype=torch.int64, device=DEV),
    )


def _independent_trials(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row seeds, so the rows are independent trials."""
    return (
        torch.arange(rows, dtype=torch.int64, device=DEV),
        torch.zeros(rows, dtype=torch.int64, device=DEV),
    )


def _tv(samples: torch.Tensor, probs: torch.Tensor) -> float:
    counts = torch.bincount(samples.long(), minlength=probs.numel()).float()
    return 0.5 * (counts / samples.numel() - probs).abs().sum().item()


def test_output_layout() -> None:
    """Accepted draft tokens, one corrected or bonus token, then -1."""
    num_draft, vocab, rows = 4, 64, 256
    draft_probs, draft_tokens, target_probs = _repeat(rows, *_request(num_draft, vocab))
    tokens, num_tokens = rejection_sampling_one_model(
        draft_probs, draft_tokens, target_probs, *_independent_trials(rows)
    )

    assert tokens.shape == (rows, num_draft + 1) and tokens.dtype == torch.int32
    assert num_tokens.shape == (rows,) and num_tokens.dtype == torch.int32
    assert ((num_tokens >= 1) & (num_tokens <= num_draft + 1)).all()
    for row in range(rows):
        n = num_tokens[row].item()
        assert torch.equal(tokens[row, : n - 1], draft_tokens[row, : n - 1])
        assert 0 <= tokens[row, n - 1].item() < vocab
        assert (tokens[row, n:] == -1).all()


@pytest.mark.parametrize("rows", [2, 8, 33])
def test_rows_with_the_same_rng_state_agree_wherever_they_sit(rows: int) -> None:
    """A request's result depends on its own (seed, offset) alone, not on its row or on
    the other requests in the launch."""
    num_draft, vocab = 4, 1000
    request = _request(num_draft, vocab, seed=3)
    others = [_request(num_draft, vocab, seed=10 + i) for i in range(rows)]

    def run(position: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = list(others)
        batch[position] = request
        draft_probs, draft_tokens, target_probs = (torch.stack(t) for t in zip(*batch))
        seed = torch.arange(100, 100 + rows, dtype=torch.int64, device=DEV)
        offset = torch.arange(rows, dtype=torch.int64, device=DEV) * 64
        seed[position], offset[position] = 7, 32
        tokens, num_tokens = rejection_sampling_one_model(
            draft_probs, draft_tokens, target_probs, seed, offset
        )
        return tokens[position], num_tokens[position]

    first_tokens, first_num = run(0)
    for position in range(1, rows):
        tokens, num = run(position)
        assert torch.equal(tokens, first_tokens) and torch.equal(num, first_num)


def test_offset_moves_the_stream() -> None:
    num_draft, vocab, rows = 4, 1000, 64
    draft_probs, draft_tokens, target_probs = _repeat(rows, *_request(num_draft, vocab))
    seed = torch.full((rows,), 7, dtype=torch.int64, device=DEV)
    offset = torch.arange(rows, dtype=torch.int64, device=DEV) * (num_draft + 1)
    tokens, _ = rejection_sampling_one_model(draft_probs, draft_tokens, target_probs, seed, offset)
    assert tokens.unique(dim=0).shape[0] > 1


def test_same_rng_state_reproduces() -> None:
    num_draft, vocab, rows = 4, 1000, 16
    inputs = _repeat(rows, *_request(num_draft, vocab))
    first = rejection_sampling_one_model(*inputs, *_independent_trials(rows))
    second = rejection_sampling_one_model(*inputs, *_independent_trials(rows))
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


@pytest.mark.parametrize("vocab", [16, 17], ids=["vectorized", "scalar"])
def test_first_token_follows_the_target(vocab: int) -> None:
    """Speculative sampling is lossless: the first emitted token, accepted draft or
    correction, is distributed as the target at that position."""
    rows = 32768
    draft_probs, draft_tokens, target_probs = _request(1, vocab, seed=5)
    gen = torch.Generator(device=DEV).manual_seed(11)
    trial_tokens = torch.multinomial(draft_probs[0], rows, replacement=True, generator=gen)
    draft_probs, target_probs = _repeat(rows, draft_probs, target_probs)
    tokens, _ = rejection_sampling_one_model(
        draft_probs, trial_tokens.int().unsqueeze(-1), target_probs, *_independent_trials(rows)
    )
    tv = _tv(tokens[:, 0], target_probs[0, 0])
    assert tv < 0.02, f"total-variation distance {tv:.4f} from the target"


def test_acceptance_rate_is_the_overlap() -> None:
    """A draft token sampled from q is accepted with probability sum(min(p, q))."""
    rows, vocab = 32768, 64
    draft_probs, _, target_probs = _request(1, vocab, seed=6)
    gen = torch.Generator(device=DEV).manual_seed(12)
    trial_tokens = torch.multinomial(draft_probs[0], rows, replacement=True, generator=gen)
    expected = torch.minimum(draft_probs[0], target_probs[0]).sum().item()
    draft_probs, target_probs = _repeat(rows, draft_probs, target_probs)
    _, num_tokens = rejection_sampling_one_model(
        draft_probs, trial_tokens.int().unsqueeze(-1), target_probs, *_independent_trials(rows)
    )
    rate = (num_tokens == 2).float().mean().item()
    assert abs(rate - expected) < 0.015, f"acceptance rate {rate:.4f}, expected {expected:.4f}"


def test_bonus_token_follows_the_last_target() -> None:
    """When the draft matches the target every token is accepted, and the bonus token is
    drawn from the last target distribution."""
    rows, vocab, num_draft = 32768, 32, 3
    target_probs = _probs(num_draft + 1, vocab, seed=8)
    draft_probs = target_probs[:num_draft].clone()
    draft_tokens = torch.zeros(num_draft, dtype=torch.int32, device=DEV)
    draft_probs, draft_tokens, target_probs = _repeat(rows, draft_probs, draft_tokens, target_probs)
    tokens, num_tokens = rejection_sampling_one_model(
        draft_probs, draft_tokens, target_probs, *_independent_trials(rows)
    )
    assert (num_tokens == num_draft + 1).all()
    tv = _tv(tokens[:, num_draft], target_probs[0, num_draft])
    assert tv < 0.02, f"total-variation distance {tv:.4f} from the last target"


def test_empty_residual_falls_back_to_the_target() -> None:
    """Draft probs that dominate the target leave max(0, p - q) with no mass; the
    correction then comes from the target instead of an out-of-range id."""
    rows, vocab = 32768, 32
    target_probs = _probs(2, vocab, seed=9)
    draft_probs = target_probs[:1] * 2.0
    draft_tokens = torch.zeros(1, dtype=torch.int32, device=DEV)
    draft_probs, draft_tokens, target_probs = _repeat(rows, draft_probs, draft_tokens, target_probs)
    tokens, num_tokens = rejection_sampling_one_model(
        draft_probs, draft_tokens, target_probs, *_independent_trials(rows)
    )
    rejected = num_tokens == 1
    assert 0.4 < rejected.float().mean().item() < 0.6
    corrections = tokens[rejected, 0]
    assert ((corrections >= 0) & (corrections < vocab)).all()
    assert _tv(corrections, target_probs[0, 0]) < 0.03


def test_long_drafts_run() -> None:
    """Drafts longer than one RNG slot (32 draws) are supported."""
    num_draft, vocab, rows = 40, 128, 8
    target_probs = _probs(num_draft + 1, vocab, seed=10)
    draft_probs = target_probs[:num_draft].clone()
    draft_tokens = torch.multinomial(draft_probs, 1).squeeze(-1).int()
    inputs = _repeat(rows, draft_probs, draft_tokens, target_probs)
    tokens, num_tokens = rejection_sampling_one_model(*inputs, *_independent_trials(rows))
    assert (num_tokens == num_draft + 1).all()
    assert torch.equal(tokens[:, :num_draft], inputs[1])


@pytest.mark.skipif(not IS_FLASHINFER_AVAILABLE, reason="requires flashinfer")
def test_acceptance_length_matches_flashinfer() -> None:
    """Same inputs, independent trials: the accepted-length distribution matches the
    flashinfer kernel this op replaces."""
    import flashinfer

    rows, vocab, num_draft = 16384, 256, 4
    draft_probs, draft_tokens, target_probs = _repeat(rows, *_request(num_draft, vocab, seed=13))
    _, num_tokens = rejection_sampling_one_model(
        draft_probs, draft_tokens, target_probs, *_independent_trials(rows)
    )
    emitted = torch.zeros(rows, dtype=torch.int32, device=DEV)
    _, _, emitted = flashinfer.sampling.chain_speculative_sampling(
        draft_probs,
        draft_tokens,
        target_probs,
        maybe_output_accepted_token_num=torch.zeros(rows, dtype=torch.int32, device=DEV),
        maybe_output_emitted_draft_token_num=emitted,
        deterministic=True,
        generator=None,
        seed=99,
        offset=0,
    )
    ours = torch.bincount(num_tokens.long(), minlength=num_draft + 2).float() / rows
    theirs = torch.bincount(emitted.long() + 1, minlength=num_draft + 2).float() / rows
    tv = 0.5 * (ours - theirs).abs().sum().item()
    assert tv < 0.03, f"accepted-length distributions differ by {tv:.4f}"
