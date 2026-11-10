# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the tp16_moetp16ep1 target classifies a step for its decode kernels (host-side, no GPU): ``decode_step``, the
attention-residual epilogue ceiling it sets, which steps the MoE layers push on, the one path the text model runs for a
step through every layer, and the DSpark worker reading the current batch-to-slot buffer.

The target decodes without speculation, so its classification checks take the steps it serves: one token per
generation request, no verify or wide step, and no per-token KDA states (``kda_draft_records`` is False)."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

__extra_import_path__ = [".."]
from _kimi_k3_decode_step import (  # noqa: E402
    ATTN_RES_CEILINGS,
    GENERIC_STEPS,
    LATENT_PUSH_STEPS,
    PURE_DECODE_STEPS,
    SHORT_STEPS,
    SLOT_ORDERS,
    STEP_PATHS,
    check_attn_res_epilogue_ceiling,
    check_drafter_inputs_use_current_slot_buffer,
    check_each_step_takes_one_path_through_every_layer,
    check_latent_push,
    check_pure_decode_step,
    check_short_step_is_small_only,
    check_spec_accept_uses_current_slot_buffer,
    check_step_takes_the_generic_path,
)

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
        modeling as target,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
        spec_worker,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def _one_token_per_request(step) -> bool:
    """Whether a step (a ``DecodeStep``'s fields) has at most one token per generation request."""
    return len(step) == 1 or step[2] == 1


@pytest.mark.parametrize("requests,tokens", [(r, t) for r, t in PURE_DECODE_STEPS if t == 1])
def test_pure_decode_steps(requests, tokens):
    """R <= 8 generation requests of one token each: a decode step of at most one token tile."""
    check_pure_decode_step(target, requests, tokens)


@pytest.mark.parametrize("seq_lens,num_contexts,rows", SHORT_STEPS)
def test_short_steps_are_small_only(seq_lens, num_contexts, rows):
    check_short_step_is_small_only(target, seq_lens, num_contexts, rows)


@pytest.mark.parametrize("seq_lens,num_contexts,rows", GENERIC_STEPS)
def test_other_steps_take_the_generic_path(seq_lens, num_contexts, rows):
    check_step_takes_the_generic_path(target, seq_lens, num_contexts, rows)


@pytest.mark.parametrize(
    "step,ceiling", [(s, c) for s, c in ATTN_RES_CEILINGS if _one_token_per_request(s)]
)
def test_attn_res_epilogue_ceiling(step, ceiling):
    check_attn_res_epilogue_ceiling(target, step, ceiling)


def test_kda_draft_records_off():
    """The hybrid cache manager keeps no per-token KDA states, and the latent-push decision reads that."""
    assert object.__new__(target.KimiLinearModel).kda_draft_records is False


@pytest.mark.parametrize(
    "step,given,pushes",
    [
        (s, g, p)
        for s, g, p in LATENT_PUSH_STEPS
        if _one_token_per_request(s) and not g.get("kda_draft_records")
    ],
)
def test_latent_push(step, given, pushes):
    check_latent_push(target, step, given, pushes)


@pytest.mark.parametrize("seq_lens,num_contexts,rows,path", STEP_PATHS)
def test_each_step_takes_one_path_through_every_layer(seq_lens, num_contexts, rows, path):
    check_each_step_takes_one_path_through_every_layer(target, seq_lens, num_contexts, rows, path)


@pytest.mark.parametrize("request_ids,expected", SLOT_ORDERS)
def test_spec_accept_uses_current_slot_buffer(monkeypatch, request_ids, expected):
    check_spec_accept_uses_current_slot_buffer(monkeypatch, spec_worker, request_ids, expected)


@pytest.mark.parametrize("num_contexts", [0, 1])
@pytest.mark.parametrize("paged", [False, True])
def test_drafter_inputs_use_current_slot_buffer(monkeypatch, num_contexts, paged):
    check_drafter_inputs_use_current_slot_buffer(monkeypatch, spec_worker, num_contexts, paged)
