# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the Kimi K3 target classifies a step for its decode kernels (host-side, no GPU): ``decode_step`` of
``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``."""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.modeling import (  # noqa: E501
    DecodeStep,
    decode_step,
)

pytestmark = pytest.mark.cpu_only


def _metadata(seq_lens, num_contexts=0):
    return types.SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=len(seq_lens) - num_contexts,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "requests,tokens",
    [(1, 1), (1, 8), (8, 1), (2, 4), (4, 2), (3, 1), (5, 1), (2, 8), (4, 8), (8, 8), (8, 2)],
)
def test_pure_decode_steps(requests, tokens):
    """R <= 8 generation requests of the same T <= 8 tokens: a decode step, small up to one token tile, wide above."""
    step = decode_step(_metadata([tokens] * requests), requests * tokens)
    assert step == DecodeStep(requests * tokens, requests, tokens)
    assert step.decode
    assert step.small == (requests * tokens <= 8)
    assert step.wide == (requests * tokens > 8)


@pytest.mark.parametrize(
    "seq_lens,num_contexts,rows",
    [
        ([5], 1, 5),  # a short prefill
        ([3, 1], 1, 4),  # a short mixed step
        ([4, 4], 0, 4),  # rows that are not the step's tokens
        ([3, 2], 0, 5),  # a ragged decode step
    ],
)
def test_short_steps_are_small_only(seq_lens, num_contexts, rows):
    """Other steps of at most 8 tokens take the token-count kernels only."""
    step = decode_step(_metadata(seq_lens, num_contexts), rows)
    assert step == DecodeStep(rows)
    assert step.small and not step.decode and not step.wide


@pytest.mark.parametrize(
    "seq_lens,num_contexts,rows",
    [
        ([1024], 1, 1024),  # prefill
        ([100, 8], 1, 108),  # mixed step
        ([1] * 9, 0, 9),  # more requests than the kernels take
        ([16], 0, 16),  # more tokens per request than the kernels take
        ([8, 4], 0, 12),  # ragged decode step
        ([4, 4], 0, 16),  # rows that are not the step's tokens
        ([], 0, 0),  # empty step
    ],
)
def test_other_steps_take_the_generic_path(seq_lens, num_contexts, rows):
    assert decode_step(_metadata(seq_lens, num_contexts), rows) is None
