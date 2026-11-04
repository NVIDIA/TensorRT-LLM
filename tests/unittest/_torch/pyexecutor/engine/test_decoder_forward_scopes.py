# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DecoderRunner.forward scopes its own device copier to the step."""

from typing import Callable, Optional
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import device_copy
from tensorrt_llm._torch.pyexecutor.device_copy import CopyPlan, CopySite, DeviceCopier
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner as decoder_runner_module
from tensorrt_llm._torch.pyexecutor.sampler.sampler import (
    GREEDY_HIDDEN_STATES_KEY,
    GREEDY_TOKENS_KEY,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def _copiers_without_check_mode(monkeypatch):
    """These tests read merged destinations after a discarded step; check mode would leave its poison there (the
    unit-test conftest turns it on everywhere)."""
    monkeypatch.setattr(device_copy, "CHECK_MODE", False)


# A host-to-device site the plan merges; its buffer owner skips unchanged uploads (device_copy.DeviceCopyBuffer).
_SITE = CopySite.ATTN_PROMPT_LENS
_PLAN = CopyPlan(merge={_SITE})


class _ModelWithoutWeights(torch.nn.Module):
    """What DecoderRunner.forward reads from the model before it runs a step."""

    extra_attrs: dict = {}


class _StepFailed(RuntimeError):
    pass


def _copier() -> DeviceCopier:
    return DeviceCopier(_PLAN, torch.device("cuda", torch.cuda.current_device()))


def _runner(step: Callable[..., object], copier: Optional[DeviceCopier]) -> DecoderRunner:
    # forward reads only the model's extra attributes, the per-engine copier and the step.
    runner = object.__new__(DecoderRunner)
    runner._model = _ModelWithoutWeights()
    runner._copier = copier
    runner._forward_decoder = step
    return runner


def _host(values):
    return torch.tensor(values, dtype=torch.int32).pin_memory()


def test_forward_activates_the_runner_copier_for_the_step_only():
    copier = _copier()
    dst = torch.zeros(3, dtype=torch.int32, device="cuda")
    records = device_copy.DeviceCopyBuffer(dst, site=_SITE)
    values = _host([1, 2, 3])
    seen = {}

    def step(inputs, resource_manager, is_dummy=False):
        seen["copier"] = device_copy.current() is copier
        device_copy.begin()
        seen["pack_open"] = device_copy.is_open()
        seen["copies"] = [records.update(values) for _ in range(2)]
        seen["staged"] = len(copier._pack.staged)
        device_copy.flush()
        return "outputs"

    runner = _runner(step, copier)

    assert runner.forward(object(), resource_manager=None) == "outputs"

    assert seen == {"copier": True, "pack_open": True, "copies": [True, False], "staged": 1}
    # Outside the step it is not active any more.
    assert device_copy.current() is None
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3]
    # The owner's record outlives the step.
    assert records.update(values) is False


def test_forward_without_a_plan_activates_nothing():
    seen = {}

    def step(inputs, resource_manager, is_dummy=False):
        device_copy.begin()
        seen["copier"] = device_copy.current()
        seen["pack_open"] = device_copy.is_open()

    _runner(step, None).forward(object(), resource_manager=None)

    assert seen == {"copier": None, "pack_open": False}


def test_forward_hides_an_outer_context_copier():
    """Another engine's copier, active around this call, is not used by this runner's step."""
    outer = _copier()
    seen = {}

    def step(inputs, resource_manager, is_dummy=False):
        seen["copier"] = device_copy.current()
        seen["pack_open"] = device_copy.is_open()

    with device_copy.active(outer):
        outer.begin()
        _runner(step, None).forward(object(), resource_manager=None)
        assert device_copy.is_open()
        outer.discard()

    assert seen == {"copier": None, "pack_open": False}


def test_a_failed_step_discards_its_pack_and_the_owners_records_of_it():
    copier = _copier()
    dst = torch.zeros(3, dtype=torch.int32, device="cuda")
    records = device_copy.DeviceCopyBuffer(dst, site=_SITE)
    values = _host([7, 8, 9])

    def failing_step(inputs, resource_manager, is_dummy=False):
        device_copy.begin()
        assert records.update(values) is True
        assert copier._pack.staged
        raise _StepFailed("forward failed after staging its inputs")

    runner = _runner(failing_step, copier)

    with pytest.raises(_StepFailed):
        runner.forward(object(), resource_manager=None)

    assert not copier.is_open
    assert copier._pack.staged == [] and copier._pack.h2d == [] and copier._pack.d2d == []
    assert device_copy.current() is None
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 0, 0]

    # The merged copy's on_drop dropped the owner's record: the next step re-sends the values.
    def next_step(inputs, resource_manager, is_dummy=False):
        device_copy.begin()
        sent = records.update(values)
        device_copy.flush()
        return sent

    runner._forward_decoder = next_step
    assert runner.forward(object(), resource_manager=None) is True
    torch.cuda.synchronize()
    assert dst.tolist() == [7, 8, 9]


class _LogitsProcessor:
    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits):
        return lm_head(hidden_states) * 2


class _GreedyFallbackModel(torch.nn.Module):
    """The LM head and logits processor the greedy fallback runs eagerly."""

    def __init__(self) -> None:
        super().__init__()
        self.lm_head = torch.nn.Identity()
        self.logits_processor = _LogitsProcessor()


class _ScheduledWithContext:
    """A batch with a context request: greedy tokens cannot serve it."""

    num_context_requests = 1
    generation_requests = ()


def test_greedy_fallback_leaves_the_replayed_graph_outputs_intact():
    runner = object.__new__(DecoderRunner)
    runner._model = _GreedyFallbackModel()
    # max_beam_width is read from the runner config; the fallback reads nothing else from it.
    runner._config = Mock(spec_set=["max_beam_width"], max_beam_width=1)
    runner.sample_in_graph_callable = None
    hidden = torch.ones(2, 4, device="cuda")
    tokens = torch.zeros(2, dtype=torch.int32, device="cuda")
    # The dict a CUDA graph runner returns on every replay of its key.
    graph_outputs = {GREEDY_TOKENS_KEY: tokens, GREEDY_HIDDEN_STATES_KEY: hidden}

    outputs = runner._logits_for_greedy_tokens(_ScheduledWithContext(), graph_outputs)

    assert GREEDY_TOKENS_KEY not in outputs and GREEDY_HIDDEN_STATES_KEY not in outputs
    assert torch.equal(outputs["logits"], hidden * 2)
    # The next replay of the same graph still sees its greedy outputs and takes the fallback again.
    assert graph_outputs == {GREEDY_TOKENS_KEY: tokens, GREEDY_HIDDEN_STATES_KEY: hidden}


class _ScheduledGenerationOnly:
    """A generation-only batch every request of which greedy tokens could serve (greedy_tokens_usable patched)."""

    num_context_requests = 0
    generation_requests = (object(),)


class _ScheduledIdle:
    """An idle attention-DP rank's empty batch."""

    num_context_requests = 0
    generation_requests = ()


@pytest.mark.parametrize("keep", [True, False])
def test_greedy_only_outputs_follow_the_guided_decoder_switch(keep, monkeypatch):
    """With an executor-side guided decoder (keep_greedy_only_outputs False) the batch gets logits; otherwise the
    forward's greedy-only outputs are returned as they are."""
    runner = object.__new__(DecoderRunner)
    runner._model = _GreedyFallbackModel()
    runner._config = Mock(spec_set=["max_beam_width"], max_beam_width=1)
    runner.sample_in_graph_callable = None
    runner.keep_greedy_only_outputs = keep
    monkeypatch.setattr(decoder_runner_module, "greedy_tokens_usable", lambda request: True)
    hidden = torch.ones(2, 4, device="cuda")
    tokens = torch.zeros(2, dtype=torch.int32, device="cuda")
    graph_outputs = {GREEDY_TOKENS_KEY: tokens, GREEDY_HIDDEN_STATES_KEY: hidden}

    outputs = runner._logits_for_greedy_tokens(_ScheduledGenerationOnly(), graph_outputs)

    if keep:
        assert outputs is graph_outputs and "logits" not in outputs
    else:
        assert torch.equal(outputs["logits"], hidden * 2)


def test_an_idle_batch_gets_logits_for_its_dummy_row():
    """An empty batch (an idle attention-DP rank's dummy row) never keeps greedy-only outputs."""
    runner = object.__new__(DecoderRunner)
    runner._model = _GreedyFallbackModel()
    runner._config = Mock(spec_set=["max_beam_width"], max_beam_width=1)
    runner.sample_in_graph_callable = None
    hidden = torch.ones(1, 4, device="cuda")
    tokens = torch.zeros(1, dtype=torch.int32, device="cuda")
    outputs = runner._logits_for_greedy_tokens(
        _ScheduledIdle(), {GREEDY_TOKENS_KEY: tokens, GREEDY_HIDDEN_STATES_KEY: hidden}
    )
    assert torch.equal(outputs["logits"], hidden * 2)
