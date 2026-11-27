# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3.1 Rubin targets opt into the engine features they rely on.

The engine keeps these features off by default; each target turns them on
through its ``RuntimeHooks``. The targets need GPUs to import, so this test
reads the ``RuntimeHooks(...)`` call in each target's source instead. CPU only.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm._torch.modeling_v2.models.minimax_m3 as _family

pytestmark = pytest.mark.cpu_only

_MODELS = Path(_family.__file__).resolve().parent
_PREFIX = "minimax_m3_1_flash_nvfp4__sm_107__"

_EVERY_TARGET = {
    "async_stats_rpc": "True",
    "bounce_scatter_on_order_stream": "True",
    "context_onehot_index_fill": "True",
    "ctx_oop_retry_gate": "True",
    "ctx_srpt_aging": "True",
    "dflash_host_ctx_alloc": "True",
    "dspark_ancestral_override": "True",
    "gpu_timestamps": "True",
    "kv_cache_gil_fast_path": "True",
    "moe_flashinfer_fp4_quantize": "True",
    "response_guard": "True",
    "sparse_attention": "_msa_engine.sparse_attention_classes()",
}
_DECODE = {"disagg_gen_completion_reservation": "True"}
_TP_DECODE = {
    **_DECODE,
    "disagg_gen_prefetch_requests": "1",
    "draft_kv_reserve_guard": "True",
    "draft_pool_fixes": "True",
}
_CONTEXT = {
    "ctx_only_preemption": "True",
    "draft_kv_reserve_guard": "True",
    "draft_pool_fixes": "True",
}
# Off on every target: they need a KV connector that answers the prefix probe.
_UNSET = ("ctx_store_aware_ranking", "store_lease_touch")

EXPECTED = {
    "dep": {
        **_DECODE,
        "draft_pool_fixes": "True",
        "kv_fabric_write": "True",
        "executor_thread_pin_cores": "2",
        "postlaunch_admission": "True",
    },
    "dep2ctx": {**_CONTEXT, "kv_fabric_write": "True"},
    "tp2": _TP_DECODE,
    "tp2ctx": _CONTEXT,
    "tp4": _TP_DECODE,
    "tp8": _TP_DECODE,
}


def _runtime_hooks_kwargs(target: str) -> dict:
    source = (_MODELS / f"{_PREFIX}{target}" / "modeling.py").read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "RuntimeHooks"
    ]
    assert calls, f"{target}: no RuntimeHooks(...) call"
    # The target's full hooks; a kill-switch fallback may build a smaller one.
    call = max(calls, key=lambda node: len(node.keywords))
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}


@pytest.mark.parametrize("target", sorted(EXPECTED))
def test_target_opts_into_engine_features(target):
    kwargs = _runtime_hooks_kwargs(target)
    for name, value in {**_EVERY_TARGET, **EXPECTED[target]}.items():
        assert kwargs.get(name) == value, f"{target}: {name}={kwargs.get(name)!r}, want {value}"
    for name in _UNSET:
        assert name not in kwargs, f"{target}: {name}={kwargs[name]!r}, want unset"


@pytest.mark.parametrize("target", ["tp2", "tp4", "tp8"])
def test_tp_decode_targets_install_native_dflash_merge(target):
    source = (_MODELS / f"{_PREFIX}{target}" / "modeling.py").read_text()
    assert "_dflash_merge.install()" in source


@pytest.mark.parametrize("target", ["tp2", "tp4", "tp8"])
def test_per_instance_hook_targets_declare_draft_pool_fixes_on_the_class(target):
    """These targets build RuntimeHooks in __init__, so routing (before construction) reads the
    class-level ``draft_pool_fixes``; without it the windowed drafter is refused."""
    source = (_MODELS / f"{_PREFIX}{target}" / "modeling.py").read_text()
    name = f"ModelingV2MiniMaxM31FlashNvfp4Sm107{target.capitalize()}"
    cls = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    values = {
        stmt.targets[0].id: ast.unparse(stmt.value)
        for stmt in cls.body
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
    }
    assert values.get("draft_pool_fixes") == "True", f"{target}: {values.get('draft_pool_fixes')!r}"


def test_every_target_is_a_routed_target():
    """The native loader names a target's torch namespace from the plain ``return <class>`` statements in
    ``routing.py``; a target missing there registers its fakes under the family namespace and fails to load."""
    from tensorrt_llm._torch.modeling_v2 import _native

    _native._routed_targets.cache_clear()
    routed = _native._routed_targets("minimax_m3")
    missing = sorted(f"{_PREFIX}{t}" for t in EXPECTED if f"{_PREFIX}{t}" not in routed)
    assert not missing, f"not routed (fake_ops would use the family namespace): {missing}"
