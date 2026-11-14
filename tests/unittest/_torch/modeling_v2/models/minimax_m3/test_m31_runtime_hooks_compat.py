# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``runtime_hooks_compat.build``: the MiniMax-M3.1-Flash runtime hooks on the engine this process runs.

Each requested field the engine lacks is handled by its class: dropped (and logged once per target) when running
without it costs only speed or acceptance, ``NoTarget`` when the target is not correct without it, and ``TypeError``
when nobody classified it. Fields the engine has pass through unchanged. The engine's own dataclasses are used, so a
field the engine gains later passes through instead of being dropped.
"""

from __future__ import annotations

import dataclasses

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import runtime_hooks_compat
from tensorrt_llm._torch.pyexecutor.runtime_hooks import MsaRuntimePolicy, RuntimeHooks

pytestmark = pytest.mark.cpu_only

_HOOKS = frozenset(f.name for f in dataclasses.fields(RuntimeHooks))
_MSA = frozenset(f.name for f in dataclasses.fields(MsaRuntimePolicy))
_DROPPABLE = (
    runtime_hooks_compat.MAIN_DEFAULT
    | runtime_hooks_compat.PERF_ONLY
    | runtime_hooks_compat.DRAFT_ONLY
    | runtime_hooks_compat.CONDITIONAL
)


@pytest.fixture
def logged(monkeypatch):
    """The info lines ``build`` logs, with its once-per-target memory cleared and the KV pool default left on."""
    lines = []
    monkeypatch.setattr(runtime_hooks_compat, "_LOGGED", set())
    monkeypatch.setattr(runtime_hooks_compat.logger, "info", lines.append)
    monkeypatch.delenv("TRTLLM_M3_KV_POOL_ZERO_INIT", raising=False)
    return lines


def _missing(names) -> list:
    """The names of ``names`` this engine's ``RuntimeHooks`` lacks, in a stable order."""
    return sorted(set(names) - _HOOKS)


def test_the_classes_are_disjoint():
    classes = (
        runtime_hooks_compat.MAIN_DEFAULT,
        runtime_hooks_compat.PERF_ONLY,
        runtime_hooks_compat.DRAFT_ONLY,
        runtime_hooks_compat.CONDITIONAL,
        runtime_hooks_compat.REQUIRED,
    )
    assert sum(len(c) for c in classes) == len(frozenset().union(*classes))


def test_fields_the_engine_has_pass_through(logged):
    hooks = runtime_hooks_compat.build(
        dict(pack_step_inputs=True, kv_lens_update_once=True),
        dict(short_sequence_identity=True, reuse_pool_views=True),
        target="t",
    )
    assert hooks.pack_step_inputs and hooks.kv_lens_update_once
    assert hooks.msa == MsaRuntimePolicy(short_sequence_identity=True, reuse_pool_views=True)
    assert logged == []


@pytest.mark.parametrize(
    "kind",
    ["MAIN_DEFAULT", "PERF_ONLY", "DRAFT_ONLY", "CONDITIONAL"],
)
def test_each_droppable_class_is_dropped_and_logged_once(logged, kind):
    names = _missing(getattr(runtime_hooks_compat, kind))
    if not names:
        pytest.skip(f"this engine has every {kind} field")
    request = dict.fromkeys(names, True)
    hooks = runtime_hooks_compat.build(dict(request, pack_step_inputs=True), target="t")
    assert hooks == RuntimeHooks(pack_step_inputs=True)
    assert len(logged) == 1 and all(name in logged[0] for name in names)
    runtime_hooks_compat.build(request, target="t")
    assert len(logged) == 1, "a target logs its dropped hooks once"
    runtime_hooks_compat.build(request, target="other")
    assert len(logged) == 2, "each target logs its own"
    assert runtime_hooks_compat.requested("t") == (request, {})


def test_a_field_requested_off_is_dropped_silently(logged):
    names = _missing(runtime_hooks_compat.PERF_ONLY)
    if not names:
        pytest.skip("this engine has every PERF_ONLY field")
    runtime_hooks_compat.build(dict.fromkeys(names, False), target="t")
    assert logged == []


def test_msa_fields_the_engine_lacks_are_dropped(logged):
    names = sorted(runtime_hooks_compat.MSA_PERF_ONLY - _MSA)
    if not names:
        pytest.skip("this engine has every MSA_PERF_ONLY field")
    hooks = runtime_hooks_compat.build({}, dict.fromkeys(names, 1), target="t")
    assert hooks.msa == MsaRuntimePolicy()
    assert len(logged) == 1 and all(f"msa.{name}" in logged[0] for name in names)


def test_an_unclassified_field_is_a_programming_error(logged):
    with pytest.raises(TypeError, match="no_such_hook"):
        runtime_hooks_compat.build(dict(no_such_hook=True), target="t")
    with pytest.raises(TypeError, match="msa.no_such_policy"):
        runtime_hooks_compat.build({}, dict(no_such_policy=True), target="t")
    with pytest.raises(TypeError, match="msa="):
        runtime_hooks_compat.build(dict(msa=MsaRuntimePolicy()), target="t")


def test_a_required_field_the_engine_lacks_declines_the_target(logged):
    names = _missing(runtime_hooks_compat.REQUIRED)
    if not names:
        pytest.skip("this engine has every REQUIRED field")
    with pytest.raises(NoTarget, match=names[0]):
        runtime_hooks_compat.build(dict.fromkeys(names, True), target="t")
    # Requested off, the target does not need it.
    runtime_hooks_compat.build(dict.fromkeys(names, False), target="t")


def test_a_main_default_turned_off_declines_the_target(logged, monkeypatch):
    names = _missing(runtime_hooks_compat.MAIN_DEFAULT)
    if "msa_kv_cache_fixes" not in names:
        pytest.skip("this engine carries the MSA KV cache fixes as a hook")
    monkeypatch.setenv("TRTLLM_M3_KV_POOL_ZERO_INIT", "0")
    assert runtime_hooks_compat.missing_defaults()
    with pytest.raises(NoTarget, match="zero-initialized"):
        runtime_hooks_compat.build(dict(msa_kv_cache_fixes=True), target="t")
    monkeypatch.setenv("TRTLLM_M3_KV_POOL_ZERO_INIT", "1")
    assert runtime_hooks_compat.missing_defaults() == []
    runtime_hooks_compat.build(dict(msa_kv_cache_fixes=True), target="t")


def test_has_hook_reads_the_engine_dataclass():
    assert runtime_hooks_compat.has_hook("pack_step_inputs")
    assert not runtime_hooks_compat.has_hook("no_such_hook")
    assert {
        name for name in _DROPPABLE if runtime_hooks_compat.has_hook(name)
    } == _DROPPABLE & _HOOKS
