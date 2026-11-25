# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the Kimi K3 family's ``step.py`` classifies a step for the decode kernels (host-side, no GPU): ``decode_step``
and which steps the MoE layers push on (``latent_push``).

The case tables cover both targets' steps: the tp16_moetp16ep1 target serves one token per generation request, the
tp16_moetp4ep4 target also DSpark verify steps of up to 8 tokens per request."""

import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl import step
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def step_metadata(seq_lens, num_contexts=0):
    return types.SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=len(seq_lens) - num_contexts,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
    )


#: (requests, tokens): R <= 8 generation requests of the same T <= 8 tokens, a pure decode step.
PURE_DECODE_STEPS = [
    (1, 1),
    (1, 8),
    (8, 1),
    (2, 4),
    (4, 2),
    (3, 1),
    (5, 1),
    (2, 8),
    (4, 8),
    (8, 8),
    (8, 2),
]


@pytest.mark.parametrize("requests,tokens", PURE_DECODE_STEPS)
def test_pure_decode_steps(requests, tokens):
    """R <= 8 generation requests of the same T <= 8 tokens: a decode step, small up to one token tile, wide above."""
    s = step.decode_step(step_metadata([tokens] * requests), requests * tokens)
    assert s == step.DecodeStep(requests * tokens, requests, tokens)
    assert s.decode
    assert s.small == (requests * tokens <= 8)
    assert s.wide == (requests * tokens > 8)


#: (seq_lens, num_contexts, rows): other steps of at most 8 tokens.
SHORT_STEPS = [
    ([5], 1, 5),  # a short prefill
    ([3, 1], 1, 4),  # a short mixed step
    ([4, 4], 0, 4),  # rows that are not the step's tokens
    ([3, 2], 0, 5),  # a ragged decode step
]


@pytest.mark.parametrize("seq_lens,num_contexts,rows", SHORT_STEPS)
def test_short_steps_are_small_only(seq_lens, num_contexts, rows):
    """Other steps of at most 8 tokens take the token-count kernels only."""
    s = step.decode_step(step_metadata(seq_lens, num_contexts), rows)
    assert s == step.DecodeStep(rows)
    assert s.small and not s.decode and not s.wide


#: (seq_lens, num_contexts, rows): steps no decode kernel takes.
GENERIC_STEPS = [
    ([1024], 1, 1024),  # prefill
    ([100, 8], 1, 108),  # mixed step
    ([1] * 9, 0, 9),  # more requests than the kernels take
    ([16], 0, 16),  # more tokens per request than the kernels take
    ([8, 4], 0, 12),  # ragged decode step
    ([4, 4], 0, 16),  # rows that are not the step's tokens
    ([], 0, 0),  # empty step
]


@pytest.mark.parametrize("seq_lens,num_contexts,rows", GENERIC_STEPS)
def test_other_steps_take_the_generic_path(seq_lens, num_contexts, rows):
    assert step.decode_step(step_metadata(seq_lens, num_contexts), rows) is None


#: (step, given, pushes): the flags ``latent_push`` is given beyond the defaults of ``test_latent_push``, and whether
#: the MoE layers push.
LATENT_PUSH_STEPS = [
    ((1, 1, 1), {}, True),  # one token per request on the KDA decode kernels
    ((8, 8, 1), {}, True),
    ((8, 1, 8), {"kda_draft_records": True}, True),  # a DSpark verify with per-token states
    ((4, 2, 2), {"kda_draft_records": True}, True),
    ((8, 1, 8), {}, False),  # the built-in KDA verify
    ((5,), {}, False),  # a context request, or rows that are not the step's tokens
    ((16, 2, 8), {"kda_draft_records": True}, False),  # wide: the routed experts' all-reduce
    ((1, 1, 1), {"capturing": False}, False),  # an eager step
    ((1, 1, 1), {"breakable": True}, False),
    ((1, 1, 1), {"kda_decode_kernels": False}, False),  # a KDA layer on the built-in kernels
    ((1, 1, 1), {"mla_decode_branch": False}, False),  # MLA on the built-in path
]


@pytest.mark.parametrize("fields,given,pushes", LATENT_PUSH_STEPS)
def test_latent_push(fields, given, pushes):
    """Which steps the MoE layers push on (``latent_push``): a pure decode step of at most 8 tokens, captured into a
    CUDA graph, whose attention layers all run the decode kernels; every other step keeps the routed experts'
    all-reduce."""
    flags = dict(
        capturing=True,
        breakable=False,
        kda_draft_records=False,
        kda_decode_kernels=True,
        mla_decode_branch=True,
    )
    flags.update(given)
    assert step.latent_push(step.DecodeStep(*fields), **flags) == pushes
