# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HOSTTAIL's kill switches on this target: ``_runtime_hooks`` turns on all six host-tail RuntimeHooks fields by
default, ``TRTLLM_M31_DISABLE_<ID>`` turns off only its own field and ``TRTLLM_M31_DISABLE_HOSTTAIL`` all six.

The ``HOSTTAIL_*`` expressions are evaluated from modeling.py's source against a private copy of m31_switches loaded
under each environment (as test_modeling_v2_m31_switches does), so neither the in-tree switches nor the heavy
modeling module is reloaded. CPU only, except the last test, which imports the in-tree module.
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

TARGET = "minimax_m31_nvfp4__sm_103__tp4"
_DIR = Path(_family.__file__).resolve().parent / TARGET
_SWITCHES = _DIR / "m31_switches.py"
_MODELING = _DIR / "modeling.py"

# Kill-switch id -> (the module constant, the RuntimeHooks field it sets).
ITEMS = {
    "HOST_D2H_SIDE": ("HOSTTAIL_D2H_SIDE", "sampler_d2h_side_stream"),
    "SMALL_H2D": ("HOSTTAIL_SMALL_H2D", "spec_small_h2d"),
    "BATCH_CHANGED_ONLY": ("HOSTTAIL_BATCH_CHANGED_ONLY", "spec_batch_map_changed_only"),
    "SKIP_ACCEPTED_UPLOAD": ("HOSTTAIL_SKIP_ACCEPTED_UPLOAD", "skip_accepted_draft_upload"),
    "PHILOX_PACK": ("HOSTTAIL_PHILOX_PACK", "pack_spec_rng_uploads"),
    "KV_LENS_ONCE": ("HOSTTAIL_KV_LENS_ONCE", "kv_lens_update_once"),
}
_SWITCH_IDS = ("HOSTTAIL", *ITEMS)


def _module_tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


def _hosttail_statements(tree: ast.Module) -> list[ast.stmt]:
    """modeling.py's module-level ``_HOSTTAIL_OFF = ...`` and ``HOSTTAIL_* = ...`` assignments, in source order."""
    statements = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and (
                target.id == "_HOSTTAIL_OFF" or target.id.startswith("HOSTTAIL_")
            ):
                statements.append(node)
    return statements


def _switches(monkeypatch, env):
    """A private copy of m31_switches imported under ``env`` (the HOSTTAIL variables cleared first)."""
    for item in _SWITCH_IDS:
        for prefix in ("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_"):
            monkeypatch.delenv(prefix + item, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(f"_m31_switches_hosttail_{id(env)}", _SWITCHES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hosttail(monkeypatch, env) -> dict[str, bool]:
    """The six HOSTTAIL_* values modeling.py computes under ``env``, by item id."""
    switches = _switches(monkeypatch, env)
    namespace = {"m31_disabled": switches.m31_disabled, "m31_enabled": switches.m31_enabled}
    exec(
        compile(ast.Module(_hosttail_statements(_module_tree()), []), str(_MODELING), "exec"),
        namespace,
    )
    return {item: namespace[constant] for item, (constant, _) in ITEMS.items()}


def test_the_switches_are_the_six_items():
    names = [node.targets[0].id for node in _hosttail_statements(_module_tree())]

    assert names == ["_HOSTTAIL_OFF"] + [constant for constant, _ in ITEMS.values()]
    for _, field in ITEMS.values():
        assert getattr(RuntimeHooks(), field) is False, field


def test_every_item_is_on_by_default(monkeypatch):
    assert _hosttail(monkeypatch, {}) == {item: True for item in ITEMS}


@pytest.mark.parametrize("item", sorted(ITEMS))
def test_each_kill_switch_turns_off_only_its_own_item(monkeypatch, item):
    values = _hosttail(monkeypatch, {"TRTLLM_M31_DISABLE_" + item: "1"})

    assert values == {other: other != item for other in ITEMS}


@pytest.mark.parametrize("value", ["1", "true", "on", "yes"])
def test_the_bundle_switch_turns_off_all_six(monkeypatch, value):
    assert _hosttail(monkeypatch, {"TRTLLM_M31_DISABLE_HOSTTAIL": value}) == {
        item: False for item in ITEMS
    }


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_an_unset_bundle_switch_keeps_all_six(monkeypatch, value):
    assert _hosttail(monkeypatch, {"TRTLLM_M31_DISABLE_HOSTTAIL": value}) == {
        item: True for item in ITEMS
    }


def test_an_enable_variable_does_not_turn_an_item_off_or_on(monkeypatch):
    env = {"TRTLLM_M31_ENABLE_" + item: "1" for item in _SWITCH_IDS}
    assert _hosttail(monkeypatch, env) == {item: True for item in ITEMS}


def test_runtime_hooks_wires_each_field_to_its_switch():
    """``_runtime_hooks`` passes each HOSTTAIL_* constant to its RuntimeHooks field, and nothing else sets those."""
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
        assert getattr(hooks, field) is not (disabled("HOSTTAIL") or disabled(item)), item
