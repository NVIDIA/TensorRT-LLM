# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The phase of a model step of the GLM-5.2 tp8 target: the scope, the targets' phases and the
shell's forward, which runs every step under the phase of the target that takes it.

Host-only: the shell is created without its constructor around an adopted built-in text model
without layers, and the built-in causal LM's forward is replaced by a recorder that also calls
a stand-in speculative worker, so each test sees the phase every part of a step runs under.
"""

from __future__ import annotations

import inspect

import pytest
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    dispatch,
)
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Model
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM

pytestmark = pytest.mark.cpu_only

PREFILL, DECODE = dispatch.Phase.PREFILL, dispatch.Phase.DECODE


class _Metadata:
    def __init__(self, num_contexts: int) -> None:
        self.num_contexts = num_contexts


@pytest.fixture(autouse=True)
def _no_phase():
    assert dispatch.phase() is None
    yield
    assert dispatch.phase() is None


# --- the scope ----------------------------------------------------------------------------


def test_there_is_no_phase_outside_a_step():
    assert dispatch.phase() is None


@pytest.mark.parametrize("phase", [PREFILL, DECODE])
def test_a_step_runs_under_its_phase(phase):
    with dispatch.step(phase):
        assert dispatch.phase() is phase
    assert dispatch.phase() is None


def test_steps_nest_and_restore_the_enclosing_phase():
    with dispatch.step(PREFILL):
        with dispatch.step(DECODE):
            assert dispatch.phase() is DECODE
            with dispatch.step(PREFILL):
                assert dispatch.phase() is PREFILL
            assert dispatch.phase() is DECODE
        assert dispatch.phase() is PREFILL


def test_a_step_that_raises_restores_the_enclosing_phase():
    with dispatch.step(PREFILL):
        with pytest.raises(ValueError, match="inside"):
            with dispatch.step(DECODE):
                raise ValueError("inside")
        assert dispatch.phase() is PREFILL


def test_the_phases():
    assert [p.value for p in dispatch.Phase] == ["prefill", "decode"]


# --- the targets --------------------------------------------------------------------------


def test_each_target_names_its_phase():
    assert modeling.PrefillTarget.phase is PREFILL
    assert modeling.DecodeTarget.phase is DECODE


@pytest.mark.parametrize(
    "num_contexts, phase",
    [(0, DECODE), (1, PREFILL), (3, PREFILL)],
    ids=["decode", "prefill", "mixed"],
)
def test_the_selected_target_carries_the_phase_of_the_step(num_contexts, phase):
    core = modeling.GlmMoeDsaTp8ModelingV2.adopt(_builtin_text_model())
    assert core._select_target(_Metadata(num_contexts)).phase is phase


# --- the shell's forward ------------------------------------------------------------------


def _builtin_text_model() -> DeepseekV3Model:
    model = DeepseekV3Model.__new__(DeepseekV3Model)
    nn.Module.__init__(model)
    return model


@pytest.fixture
def shell_forward(monkeypatch):
    """A shell around an adopted core, and the phases its step's parts ran under."""
    seen = []

    def builtin_forward(
        self,
        attn_metadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        return_context_logits=False,
        spec_metadata=None,
        resource_manager=None,
        **kwargs,
    ):
        seen.append(("model", dispatch.phase()))
        seen.append(
            (
                "arguments",
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                return_context_logits,
                spec_metadata,
                resource_manager,
                kwargs,
            )
        )
        # The speculative worker and its draft loop run inside the built-in forward.
        seen.append(("spec worker", dispatch.phase()))
        if kwargs.get("fail"):
            raise RuntimeError("forward failed")
        return "logits"

    monkeypatch.setattr(SpecDecOneEngineForCausalLM, "forward", builtin_forward)
    shell = modeling.ModelingV2Glm52Nvfp4Sm103Tp8.__new__(modeling.ModelingV2Glm52Nvfp4Sm103Tp8)
    nn.Module.__init__(shell)
    shell.model = modeling.GlmMoeDsaTp8ModelingV2.adopt(_builtin_text_model())
    return shell, seen


@pytest.mark.parametrize(
    "num_contexts, phase",
    [(0, DECODE), (1, PREFILL), (2, PREFILL)],
    ids=["decode", "prefill", "mixed"],
)
def test_the_shell_runs_the_whole_step_under_the_phase_of_its_target(
    shell_forward, num_contexts, phase
):
    shell, seen = shell_forward
    md = _Metadata(num_contexts)

    out = shell.forward(
        attn_metadata=md,
        input_ids="ids",
        position_ids="positions",
        inputs_embeds=None,
        return_context_logits=True,
        spec_metadata="spec",
        resource_manager="resources",
        lora_params="lora",
    )

    assert out == "logits"
    assert seen == [
        ("model", phase),
        (
            "arguments",
            md,
            "ids",
            "positions",
            None,
            True,
            "spec",
            "resources",
            {"lora_params": "lora"},
        ),
        ("spec worker", phase),
    ]
    assert dispatch.phase() is None


def test_the_shell_takes_the_builtin_arguments_positionally(shell_forward):
    shell, seen = shell_forward
    md = _Metadata(0)

    assert shell.forward(md, "ids", "positions", None, False, "spec", "resources") == "logits"
    assert seen[1] == ("arguments", md, "ids", "positions", None, False, "spec", "resources", {})


def test_a_step_that_fails_leaves_no_phase_behind(shell_forward):
    shell, seen = shell_forward
    with pytest.raises(RuntimeError, match="forward failed"):
        shell.forward(attn_metadata=_Metadata(0), fail=True)
    assert seen[-1] == ("spec worker", DECODE)
    assert dispatch.phase() is None


def test_the_shell_forward_takes_the_builtin_parameters():
    builtin = inspect.signature(SpecDecOneEngineForCausalLM.forward)
    shell = inspect.signature(modeling.ModelingV2Glm52Nvfp4Sm103Tp8.forward)
    assert list(shell.parameters) == list(builtin.parameters)
    for name, parameter in builtin.parameters.items():
        assert shell.parameters[name].default == parameter.default, name
        assert shell.parameters[name].kind == parameter.kind, name
