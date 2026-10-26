# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The kill switches of this target's host-side step-loop items: ``_runtime_hooks`` turns on all ten RuntimeHooks fields
in ITEMS by default; ``TRTLLM_M31_DISABLE_<ID>`` turns off its own field (and, for the step-input pack, the fields that
need it: PACK_IN_GRAPH needs PACK_PSCATTER, PACK_EARLY_COPY needs PACK_IN_GRAPH); ``TRTLLM_M31_DISABLE_HOSTTAIL``
turns off the six items of the sm_103 targets' HOSTTAIL bundle.

The ``STEP_*`` expressions are evaluated from modeling.py's source against a private copy of m31_switches loaded
under each environment (as test_modeling_v2_m31_switches_sm107_tp8 does), so neither the in-tree switches nor the
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

TARGET = "minimax_m31_nvfp4__sm_107__tp8"
_DIR = Path(_family.__file__).resolve().parent / TARGET
_SWITCHES = _DIR / "m31_switches.py"
_MODELING = _DIR / "modeling.py"

# Kill-switch id -> (the module constant, the RuntimeHooks field it sets).
ITEMS = {
    "SKIP_ACCEPTED_UPLOAD": ("STEP_SKIP_ACCEPTED_UPLOAD", "skip_accepted_draft_upload"),
    "BATCH_CHANGED_ONLY": ("STEP_BATCH_CHANGED_ONLY", "spec_batch_map_changed_only"),
    "HOST_D2H_SIDE": ("STEP_HOST_D2H_SIDE", "sampler_d2h_side_stream"),
    "SMALL_H2D": ("STEP_SMALL_H2D", "spec_small_h2d"),
    "PHILOX_PACK": ("STEP_PHILOX_PACK", "pack_spec_rng_uploads"),
    "KV_LENS_ONCE": ("STEP_KV_LENS_ONCE", "kv_lens_update_once"),
    "PACK_PSCATTER": ("STEP_PACK_PSCATTER", "pack_parallel_scatter"),
    "PACK_IN_GRAPH": ("STEP_PACK_IN_GRAPH", "pack_in_graph"),
    "PACK_EARLY_COPY": ("STEP_PACK_EARLY_COPY", "pack_early_copy"),
    "H2D_CHANGED_SPAN": ("STEP_H2D_CHANGED_SPAN", "h2d_changed_span"),
}
# The sm_103 HOSTTAIL bundle's six items, all turned off by TRTLLM_M31_DISABLE_HOSTTAIL.
HOSTTAIL = (
    "HOST_D2H_SIDE",
    "SMALL_H2D",
    "BATCH_CHANGED_ONLY",
    "SKIP_ACCEPTED_UPLOAD",
    "PHILOX_PACK",
    "KV_LENS_ONCE",
)
# Item -> the items it needs (off with it).
NEEDS = {"PACK_IN_GRAPH": ("PACK_PSCATTER",), "PACK_EARLY_COPY": ("PACK_IN_GRAPH", "PACK_PSCATTER")}


def _off_with(item: str) -> set:
    """The items a kill switch of ``item`` turns off: its own and every item that needs it."""
    return {item} | {other for other, needs in NEEDS.items() if item in needs}


def _module_tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


def _item_statements(tree: ast.Module) -> list[ast.stmt]:
    """modeling.py's module-level assignments of ``_HOSTTAIL_OFF`` and the ITEMS constants, in source order."""
    constants = {"_HOSTTAIL_OFF"} | {constant for constant, _ in ITEMS.values()}
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
def test_the_switches_are_the_ten_items():
    names = [node.targets[0].id for node in _item_statements(_module_tree())]

    assert names == ["_HOSTTAIL_OFF", *(constant for constant, _ in ITEMS.values())]
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

    assert values == {other: other not in _off_with(item) for other in ITEMS}


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
def test_the_bundle_switch_turns_off_the_six_hosttail_items(monkeypatch):
    """HOSTTAIL turns off the sm_103 bundle's six items, as on those targets; the pack items stay on."""
    assert _values(monkeypatch, {"TRTLLM_M31_DISABLE_HOSTTAIL": "1"}) == {
        item: item not in HOSTTAIL for item in ITEMS
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
def test_the_step_input_pack_and_its_epoch_stay_on():
    """The fields without a switch here: the pack itself, its step epoch and the one-launch prologue (the pack keeps
    the epoch only with pack_in_graph and calls the prologue op only with pack_early_copy, so the switches above
    turn those off with them)."""
    fn = next(
        node
        for node in _module_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "_runtime_hooks"
    )
    call = next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RuntimeHooks"
    )
    keywords = {kw.arg: kw.value for kw in call.keywords}
    for field in ("pack_step_inputs", "pack_step_epoch"):
        assert isinstance(keywords.get(field), ast.Constant), field
        assert keywords[field].value is True, field
    assert isinstance(keywords.get("pack_prologue_op"), ast.Name)


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
        off = disabled(item) or any(disabled(need) for need in NEEDS.get(item, ()))
        off = off or (item in HOSTTAIL and disabled("HOSTTAIL"))
        assert getattr(hooks, field) is not off, item
