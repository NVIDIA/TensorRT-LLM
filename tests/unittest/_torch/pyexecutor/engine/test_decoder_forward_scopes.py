# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DecoderRunner.forward scopes its own step-input pack and H2D skip cache to the step."""

from typing import Callable, Optional
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import h2d_cache, step_inputs
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner as decoder_runner_module
from tensorrt_llm._torch.pyexecutor.h2d_cache import H2DSkipCache, copy_h2d_if_changed
from tensorrt_llm._torch.pyexecutor.sampler.sampler import (
    GREEDY_HIDDEN_STATES_KEY,
    GREEDY_TOKENS_KEY,
)
from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _ModelWithoutWeights(torch.nn.Module):
    """What DecoderRunner.forward reads from the model before it runs a step."""

    extra_attrs: dict = {}


class _StepFailed(RuntimeError):
    pass


def _runner(
    step: Callable[..., object],
    skip_cache: Optional[H2DSkipCache],
    pack: Optional[StepInputPack],
) -> DecoderRunner:
    # forward reads only the model's extra attributes, the two per-engine objects and the step.
    runner = object.__new__(DecoderRunner)
    runner._model = _ModelWithoutWeights()
    runner._h2d_cache = skip_cache
    runner._step_pack = pack
    runner._forward_decoder = step
    return runner


def _host(values):
    return torch.tensor(values, dtype=torch.int32).pin_memory()


def test_forward_activates_the_runner_cache_and_pack_for_the_step_only():
    skip_cache = H2DSkipCache()
    pack = StepInputPack(torch.device("cuda", torch.cuda.current_device()))
    dst = torch.zeros(3, dtype=torch.int32, device="cuda")
    values = _host([1, 2, 3])
    seen = {}

    def step(inputs, resource_manager, is_dummy=False):
        seen["skip_enabled"] = h2d_cache.enabled()
        step_inputs.begin()
        seen["pack_open"] = pack.open
        seen["copies"] = [copy_h2d_if_changed(dst, values) for _ in range(2)]
        step_inputs.flush()
        return "outputs"

    runner = _runner(step, skip_cache, pack)

    assert runner.forward(object(), resource_manager=None) == "outputs"

    assert seen == {"skip_enabled": True, "pack_open": True, "copies": [True, False]}
    # Outside the step neither is active any more.
    assert not h2d_cache.enabled()
    assert not step_inputs.is_open()
    # The shadow the step recorded belongs to the runner's own cache.
    with h2d_cache.active(skip_cache):
        assert copy_h2d_if_changed(dst, values) is False
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3]


def test_forward_without_armed_hooks_activates_nothing():
    seen = {}

    def step(inputs, resource_manager, is_dummy=False):
        step_inputs.begin()
        seen["skip_enabled"] = h2d_cache.enabled()
        seen["pack_open"] = step_inputs.is_open()

    _runner(step, None, None).forward(object(), resource_manager=None)

    assert seen == {"skip_enabled": False, "pack_open": False}


def test_forward_hides_an_outer_context_cache_and_pack():
    """Another engine's objects, active around this call, are not used by this runner's step."""
    outer_cache = H2DSkipCache()
    outer_pack = StepInputPack(torch.device("cuda", torch.cuda.current_device()))
    seen = {}

    def step(inputs, resource_manager, is_dummy=False):
        seen["skip_enabled"] = h2d_cache.enabled()
        seen["pack_open"] = step_inputs.is_open()

    with h2d_cache.active(outer_cache), step_inputs.active(outer_pack):
        outer_pack.begin()
        _runner(step, None, None).forward(object(), resource_manager=None)
        assert step_inputs.is_open()
        outer_pack.discard()

    assert seen == {"skip_enabled": False, "pack_open": False}


def test_a_failed_step_discards_its_pack_and_clears_its_cache():
    skip_cache = H2DSkipCache()
    pack = StepInputPack(torch.device("cuda", torch.cuda.current_device()))
    dst = torch.zeros(3, dtype=torch.int32, device="cuda")
    values = _host([7, 8, 9])

    def failing_step(inputs, resource_manager, is_dummy=False):
        step_inputs.begin()
        assert copy_h2d_if_changed(dst, values, pack=True) is True
        assert pack.staged
        raise _StepFailed("forward failed after staging its inputs")

    runner = _runner(failing_step, skip_cache, pack)

    with pytest.raises(_StepFailed):
        runner.forward(object(), resource_manager=None)

    assert not pack.open
    assert pack.staged == [] and pack.h2d == [] and pack.d2d == []
    assert not h2d_cache.enabled() and not step_inputs.is_open()
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 0, 0]

    # The next step re-sends the values the failed one never delivered.
    def next_step(inputs, resource_manager, is_dummy=False):
        step_inputs.begin()
        sent = copy_h2d_if_changed(dst, values, pack=True)
        step_inputs.flush()
        return sent

    runner._forward_decoder = next_step
    if hasattr(torch.ops.trtllm, "step_inputs_scatter"):
        assert runner.forward(object(), resource_manager=None) is True
        torch.cuda.synchronize()
        assert dst.tolist() == [7, 8, 9]
    else:
        # Without the scatter op the pack cannot be issued; the cache decision is what matters.
        with h2d_cache.active(skip_cache):
            assert copy_h2d_if_changed(dst, values) is True


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
