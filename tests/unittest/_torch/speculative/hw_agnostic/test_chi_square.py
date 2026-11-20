# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The chi-square helper the sampler distribution tests use (``_torch.helpers.chi_square``)."""

import pytest
import torch
from _torch.helpers import chi_square

pytestmark = pytest.mark.cpu_only


def test_chi_square_tests_tokens_below_five_expected_draws():
    """No token of a uniform distribution reaches 5 expected draws (one each), yet the low-expectation
    tokens are grouped into many bins: draws from the distribution pass, and all draws on one token
    are rejected."""
    vocab = 2000
    probs = torch.full((vocab,), 1.0 / vocab, dtype=torch.float64)
    draws = torch.multinomial(
        probs, vocab, replacement=True, generator=torch.Generator().manual_seed(0)
    )
    stat, crit = chi_square(torch.bincount(draws, minlength=vocab), probs)
    assert stat < crit, (stat, crit)
    one_token = torch.zeros(vocab, dtype=torch.int64)
    one_token[0] = vocab
    stat, crit = chi_square(one_token, probs)
    assert stat > crit, (stat, crit)
