# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The kill switches of this target's two host-tail items ported from the sm_103 HOSTTAIL bundle: ``_runtime_hooks``
turns on RuntimeHooks.skip_accepted_draft_upload (``SKIP_ACCEPTED_UPLOAD``) and
RuntimeHooks.spec_batch_map_changed_only (``BATCH_CHANGED_ONLY``) by default, and
``TRTLLM_M31_DISABLE_<ID>`` turns off only its own item (this target has no bundle switch).

The ``STEP_*`` expressions are evaluated from modeling.py's source against a private copy of m31_switches loaded
under each environment (as test_modeling_v2_m31_switches_sm107_tp4 does), so neither the in-tree switches nor the
heavy modeling module is reloaded. CPU only, except the last test, which imports the in-tree module.
"""

import ast
import importlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    BaseMambaCacheManager,
    MambaHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

TARGET = "minimax_m31_nvfp4__sm_107__tp4"
_DIR = Path(_family.__file__).resolve().parent / TARGET
_SWITCHES = _DIR / "m31_switches.py"
_MODELING = _DIR / "modeling.py"

# Kill-switch id -> (the module constant, the RuntimeHooks field it sets).
ITEMS = {
    "SKIP_ACCEPTED_UPLOAD": ("STEP_SKIP_ACCEPTED_UPLOAD", "skip_accepted_draft_upload"),
    "BATCH_CHANGED_ONLY": ("STEP_BATCH_CHANGED_ONLY", "spec_batch_map_changed_only"),
}


def _module_tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


def _item_statements(tree: ast.Module) -> list[ast.stmt]:
    """modeling.py's module-level assignments of the ITEMS constants, in source order."""
    constants = {constant for constant, _ in ITEMS.values()}
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in constants
    ]


def _switches(monkeypatch, env):
    """A private copy of m31_switches imported under ``env`` (every other TRTLLM_M31_* variable cleared)."""
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(f"_m31_switches_host_tail_{id(env)}", _SWITCHES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _values(monkeypatch, env) -> dict[str, bool]:
    """The ITEMS constants modeling.py computes under ``env``, by item id."""
    switches = _switches(monkeypatch, env)
    namespace = {"m31_disabled": switches.m31_disabled, "m31_enabled": switches.m31_enabled}
    exec(
        compile(ast.Module(_item_statements(_module_tree()), []), str(_MODELING), "exec"),
        namespace,
    )
    return {item: namespace[constant] for item, (constant, _) in ITEMS.items()}


@pytest.mark.cpu_only
def test_the_switches_are_the_two_items():
    names = [node.targets[0].id for node in _item_statements(_module_tree())]

    assert names == [constant for constant, _ in ITEMS.values()]
    for _, field in ITEMS.values():
        assert getattr(RuntimeHooks(), field) is False, field


@pytest.mark.cpu_only
def test_every_item_is_on_by_default(monkeypatch):
    assert _values(monkeypatch, {}) == {item: True for item in ITEMS}


@pytest.mark.cpu_only
@pytest.mark.parametrize("item", sorted(ITEMS))
@pytest.mark.parametrize("value", ["1", "true", "ON", " yes "])
def test_each_kill_switch_turns_off_only_its_own_item(monkeypatch, item, value):
    values = _values(monkeypatch, {"TRTLLM_M31_DISABLE_" + item: value})

    assert values == {other: other != item for other in ITEMS}


@pytest.mark.cpu_only
@pytest.mark.parametrize("item", sorted(ITEMS))
@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_an_unset_kill_switch_keeps_the_item(monkeypatch, item, value):
    assert _values(monkeypatch, {"TRTLLM_M31_DISABLE_" + item: value}) == {
        other: True for other in ITEMS
    }


@pytest.mark.cpu_only
@pytest.mark.parametrize("item", sorted(ITEMS))
def test_a_typo_fails_at_import_naming_the_variable(monkeypatch, item):
    with pytest.raises(ValueError, match="TRTLLM_M31_DISABLE_" + item):
        _values(monkeypatch, {"TRTLLM_M31_DISABLE_" + item: "2"})


@pytest.mark.cpu_only
def test_an_enable_variable_does_not_turn_an_item_off_or_on(monkeypatch):
    env = {"TRTLLM_M31_ENABLE_" + item: "1" for item in ITEMS}
    assert _values(monkeypatch, env) == {item: True for item in ITEMS}


@pytest.mark.cpu_only
def test_no_bundle_switch_on_this_target(monkeypatch):
    """HOSTTAIL is the sm_103 targets' bundle switch; this target has none, so it changes nothing here."""
    assert _values(monkeypatch, {"TRTLLM_M31_DISABLE_HOSTTAIL": "1"}) == {
        item: True for item in ITEMS
    }


@pytest.mark.cpu_only
def test_runtime_hooks_wires_each_field_to_its_switch():
    """``_runtime_hooks`` passes each constant to its RuntimeHooks field, and nothing else sets those."""
    fn = next(
        node
        for node in _module_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "_runtime_hooks"
    )
    calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RuntimeHooks"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    for constant, field in ITEMS.values():
        assert isinstance(keywords.get(field), ast.Name), field
        assert keywords[field].id == constant, field


@pytest.mark.cpu_only
def test_the_target_kv_cache_is_not_a_mamba_cache():
    """skip_accepted_draft_upload is on only together with a non-mamba cache: the mamba state update is the one
    consumer of accepted counts on the drafter path (it takes them as an argument, not from the spec metadata)."""
    assert not issubclass(MiniMaxM3KVCacheManagerV2, BaseMambaCacheManager)
    assert not issubclass(MiniMaxM3KVCacheManagerV2, MambaHybridCacheManager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="imports the target's modeling module")
def test_the_in_tree_hooks_follow_the_switches():
    modeling = importlib.import_module(
        f"tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.{TARGET}.modeling"
    )
    hooks = modeling._runtime_hooks(None)

    for constant, field in ITEMS.values():
        assert getattr(hooks, field) is getattr(modeling, constant), field

    def disabled(item):
        return os.environ.get("TRTLLM_M31_DISABLE_" + item, "").strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
        )

    for item, (_, field) in ITEMS.items():
        assert getattr(hooks, field) is not disabled(item), item
