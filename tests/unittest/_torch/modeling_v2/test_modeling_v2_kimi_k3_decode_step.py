# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the Kimi K3 target classifies a step for its decode kernels (host-side, no GPU): ``decode_step`` of
``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``, the attention-residual epilogue ceiling it sets, and the one path each
route target's text model runs for a step."""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import (  # noqa: E501
    modeling as target,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.modeling import (  # noqa: E501
    DecodeStep,
    _attn_res_max_tokens,
    decode_step,
    latent_push,
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


@pytest.mark.parametrize(
    "step,ceiling",
    [
        (DecodeStep(5), 8),  # a short prefill or mixed step
        (DecodeStep(8, 8, 1), 8),  # a decode step of one token tile
        (DecodeStep(6, 1, 6), 8),
        (DecodeStep(16, 2, 8), 32),  # a wide decode step
        (DecodeStep(64, 8, 8), 32),
    ],
)
def test_attn_res_epilogue_ceiling(step, ceiling):
    """The fused attn_res kernels take a classified step up to one token tile and a wide decode step up to 32
    tokens."""
    assert _attn_res_max_tokens(step) == ceiling


@pytest.mark.parametrize(
    "step,given,pushes",
    [
        (DecodeStep(1, 1, 1), {}, True),  # one token per request on the KDA decode kernels
        (DecodeStep(8, 8, 1), {}, True),
        (
            DecodeStep(8, 1, 8),
            {"kda_draft_records": True},
            True,
        ),  # a DSpark verify with per-token states
        (DecodeStep(4, 2, 2), {"kda_draft_records": True}, True),
        (DecodeStep(8, 1, 8), {}, False),  # the built-in KDA verify
        (DecodeStep(5), {}, False),  # a context request, or rows that are not the step's tokens
        (
            DecodeStep(16, 2, 8),
            {"kda_draft_records": True},
            False,
        ),  # wide: the routed experts' all-reduce
        (DecodeStep(1, 1, 1), {"capturing": False}, False),  # an eager step
        (DecodeStep(1, 1, 1), {"breakable": True}, False),
        (
            DecodeStep(1, 1, 1),
            {"kda_decode_kernels": False},
            False,
        ),  # a KDA layer on the built-in kernels
        (DecodeStep(1, 1, 1), {"mla_decode_branch": False}, False),  # MLA on the built-in path
    ],
)
def test_latent_push(step, given, pushes):
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
    assert latent_push(step, **flags) == pushes


class _Layer:
    is_moe = False

    def __init__(self, calls):
        self.calls = calls

    def __call__(
        self, hidden_states, block_residual, num_snapshots, attn_metadata, *, step, **kwargs
    ):
        self.calls.append(step)
        return hidden_states, num_snapshots

    def forward_generic(
        self, hidden_states, block_residual, num_snapshots, attn_metadata, capture=None
    ):
        self.calls.append("generic")
        return hidden_states, num_snapshots


@pytest.mark.parametrize(
    "seq_lens,num_contexts,rows,path",
    [
        ([1], 0, 1, (1, 1, 1)),
        ([1] * 8, 0, 8, (8, 8, 1)),
        ([8], 0, 8, (8, 1, 8)),
        ([8, 8], 0, 16, (16, 2, 8)),
        ([5], 1, 5, (5,)),
        ([3, 1], 1, 4, (4,)),
        ([3, 2], 0, 5, (5,)),
        ([1024], 1, 1024, "generic"),
        ([100, 8], 1, 108, "generic"),
        ([1] * 9, 0, 9, "generic"),
        ([1] * 16, 0, 16, "generic"),
        ([8, 4], 0, 12, "generic"),
    ],
)
def test_each_step_takes_one_path_through_every_layer(seq_lens, num_contexts, rows, path):
    model = object.__new__(target.KimiLinearModel)
    torch.nn.Module.__init__(model)
    calls = []
    model.layers = [_Layer(calls), _Layer(calls), _Layer(calls)]
    model.embed_tokens = torch.nn.Embedding(4, 8)
    model.num_attn_res_snapshots = 2
    model.decode_gemvs = None
    model._latent_push = lambda attn_metadata, step: False
    model._final_norm = lambda hidden_states, *args: hidden_states
    model.forward(_metadata(seq_lens, num_contexts), input_ids=torch.zeros(rows, dtype=torch.long))
    assert calls == [path if path == "generic" else target.DecodeStep(*path)] * 3
