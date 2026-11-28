# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-loop hooks of the DeepSeek-V4.1 targets, on CPU: every target declares the built-in model's hooks
(``DeepseekV4ForCausalLM.runtime_hooks``), the speculative set exactly when a speculative config is present."""

import importlib
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.step_hooks import (
    SPEC_STEP_HOOKS,
    STEP_HOOKS,
    step_hooks,
)
from tensorrt_llm._torch.models import modeling_deepseekv4 as builtin
from tensorrt_llm._torch.pyexecutor.runtime_hooks import runtime_hooks_of

pytestmark = pytest.mark.cpu_only

_PKG = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41"
_TARGETS = {
    "dep2": ("flash_nvfp4__sm_103__dep2", "ModelingV2DeepseekV41FlashNvfp4Sm103Dep2"),
    "dep": ("flash_nvfp4__sm_103__dep", "ModelingV2DeepseekV41FlashNvfp4Sm103Dep"),
    "tep8": (
        "flash_nvfp4__sm_103__tep8_moe_tp4_ep2",
        "ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2",
    ),
}


def test_hooks_are_the_builtin_models():
    assert STEP_HOOKS == builtin._DEEPSEEK_V4_RUNTIME_HOOKS
    assert SPEC_STEP_HOOKS == builtin._DEEPSEEK_V4_SPEC_RUNTIME_HOOKS
    assert step_hooks(None) is STEP_HOOKS
    assert step_hooks(object()) is SPEC_STEP_HOOKS


@pytest.mark.parametrize("name", sorted(_TARGETS))
@pytest.mark.parametrize("spec", [False, True])
def test_target_declares_the_builtin_hooks(name, spec):
    module, cls_name = _TARGETS[name]
    cls = getattr(importlib.import_module(f"{_PKG}.{module}.modeling"), cls_name)
    # The property reads only `spec_config`; a bare stand-in avoids building the model.
    stub = SimpleNamespace(spec_config=object() if spec else None)
    hooks = cls.runtime_hooks.fget(stub)
    expected = SPEC_STEP_HOOKS if spec and name != "dep2" else STEP_HOOKS
    assert hooks == expected
    assert runtime_hooks_of(SimpleNamespace(runtime_hooks=hooks)) is hooks
