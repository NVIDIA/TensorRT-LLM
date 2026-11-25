# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest
from utils.util import skip_ray

from .llm_logprobs_test_utils import llm_return_logprobs_test_harness


@skip_ray
@pytest.mark.parametrize(
    "prompt_logprobs, logprobs, return_context_logits, return_generation_logits, backend",
    [
        (2, None, True, False, "pytorch"),  # prompt_logprobs with context_logits
        (None, 1, False, False, "pytorch"),  # generation logprobs only (top-1, PyTorch limit)
        (2, None, False, False, "pytorch"),  # prompt_logprobs without context_logits
        (None, None, False, False, "pytorch"),  # no logprobs at all
    ],
)
def test_llm_return_logprobs(
    prompt_logprobs: Optional[int],
    logprobs: Optional[int],
    return_context_logits: bool,
    return_generation_logits: bool,
    backend: str,
):
    llm_return_logprobs_test_harness(
        prompt_logprobs, logprobs, return_context_logits, return_generation_logits, backend=backend
    )


@skip_ray
@pytest.mark.parametrize(
    "prompt_logprobs, logprobs, return_context_logits, return_generation_logits",
    [
        (None, 1, False, False),  # generation logprobs only (top-1, PyTorch limit)
        (2, None, True, False),  # prompt_logprobs with context_logits
        (2, None, False, False),  # prompt_logprobs only
        (2, 1, False, False),  # both prompt and generation logprobs
        (2, 3, False, False),  # both prompt and generation logprobs
    ],
)
def test_llm_return_logprobs_streaming(
    prompt_logprobs, logprobs, return_context_logits, return_generation_logits
):
    llm_return_logprobs_test_harness(
        prompt_logprobs,
        logprobs,
        return_context_logits,
        return_generation_logits,
        streaming=True,
        backend="pytorch",
    )
