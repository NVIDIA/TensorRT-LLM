# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's kill-switch registry and wiring (m31_switches): every id the target's sources read is registered,
the pack flags follow their prerequisite chain, and the runtime hooks build on this engine. The value contract
(spellings) is tested once, on the sm_103 tp4 copy (the parser is identical).

Each case loads a private copy of the module from its file, so the in-tree module (read by every other module of the
target at import) is never reloaded. CPU only.
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

import tensorrt_llm._torch.modeling_v2.models.minimax_m3 as _family

TARGET = "minimax_m3_1_flash_nvfp4__sm_103__tp8"
_SWITCHES = Path(_family.__file__).resolve().parent / TARGET / "m31_switches.py"

pytestmark = pytest.mark.cpu_only

PREFIX = "TRTLLM_M31_DISABLE_"


def _load(monkeypatch, env):
    """A private copy of m31_switches, imported under ``env`` (every other TRTLLM_M31_DISABLE_* variable cleared)."""
    for name in list(os.environ):
        if name.startswith(PREFIX):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(f"_m31_switches_contract_{id(env)}", _SWITCHES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal_reads():
    """The ids the target's sources pass to ``m31_disabled`` as string literals, with where."""
    reads = []
    for path in sorted(_SWITCHES.parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None)) == "m31_disabled"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                reads.append(
                    (node.args[0].value, f"{path.relative_to(_SWITCHES.parent)}:{node.lineno}")
                )
    return reads


def test_every_id_the_target_reads_is_registered(monkeypatch):
    """A read of an unregistered id raises at runtime; catch it here instead."""
    items = _load(monkeypatch, {}).ITEMS
    reads = _literal_reads()
    assert reads
    missing = sorted(f"{item} ({where})" for item, where in reads if item not in items)
    assert not missing, missing


# The step-input pack items and the order each one needs: a killed prerequisite turns off
# every item that needs it.


def test_the_runtime_hooks_build_on_this_engine():
    """The target's modeling module imports, and its runtime hooks build on this engine: every hook it requests that
    the engine lacks is one runtime_hooks_compat classifies (and drops)."""
    import importlib

    from tensorrt_llm._torch.modeling_v2._native import NoTarget

    try:
        modeling = importlib.import_module(f"{_family.__name__}.{TARGET}.modeling")
    except NoTarget as no_target:
        pytest.skip(str(no_target))
    hooks = modeling._runtime_hooks(None)
    assert hooks.pack_step_inputs and hooks.pack_in_graph is modeling.PACK_IN_GRAPH
