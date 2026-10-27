# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VR_DRAFT_FMHA_SM100F: ``M3DrafterSlidingFiFmha`` is FlashInfer's trtllm-gen FMHA admitted on SM 10.7 only, and
the target installs it on the standalone DSpark drafter by default; ``TRTLLM_M31_DISABLE_VR_DRAFT_FMHA_SM100F`` turns
it off. The constant is evaluated from modeling.py's source against a private copy of m31_switches. CPU only.
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

TARGET = "minimax_m31_nvfp4__sm_107__tp8"
CONSTANT = "VR_DRAFT_FMHA_SM100F"
_DIR = Path(_family.__file__).resolve().parent / TARGET
_SWITCHES = _DIR / "m31_switches.py"
_MODELING = _DIR / "modeling.py"


def _value(monkeypatch, env) -> bool:
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(f"_m31_switches_draft_fmha_{id(env)}", _SWITCHES)
    switches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switches)
    tree = ast.parse(_MODELING.read_text(), filename=str(_MODELING))
    statements = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == CONSTANT
    ]
    assert len(statements) == 1
    namespace = {"m31_disabled": switches.m31_disabled, "m31_enabled": switches.m31_enabled}
    exec(compile(ast.Module(statements, []), str(_MODELING), "exec"), namespace)
    return namespace[CONSTANT]


@pytest.mark.cpu_only
def test_on_by_default(monkeypatch):
    assert _value(monkeypatch, {}) is True


@pytest.mark.cpu_only
def test_the_kill_switch_turns_it_off(monkeypatch):
    assert _value(monkeypatch, {"TRTLLM_M31_DISABLE_VR_DRAFT_FMHA_SM100F": "1"}) is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm,available", [(100, False), (103, False), (107, True)])
def test_the_library_is_sm107_only(monkeypatch, sm, available):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels import (  # noqa: E501
        m3_drafter_attn as mod,
    )

    monkeypatch.setattr(mod, "get_sm_version", lambda: sm)
    monkeypatch.setattr(mod, "IS_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(
        mod.M3DrafterSlidingFiFmha, "_missing_fused_nanobind_ops", staticmethod(lambda: [])
    )
    assert mod.M3DrafterSlidingFiFmha.is_environment_available() is available
