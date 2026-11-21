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
    "bounce_scatter_on_order_stream": "True",
    "context_onehot_index_fill": "True",
    "dflash_host_ctx_alloc": "True",
    "gpu_timestamps": "True",
    "sparse_attention": "_msa_engine.sparse_attention_classes()",
}
_DECODE = {"disagg_gen_completion_reservation": "True"}
_TP_DECODE = {
    **_DECODE,
    "disagg_gen_prefetch_requests": "1",
    "draft_kv_reserve_guard": "True",
}
_CONTEXT = {
    "ctx_only_preemption": "True",
    "ctx_store_aware_ranking": "True",
    "draft_kv_reserve_guard": "True",
}

EXPECTED = {
    "dep": {
        **_DECODE,
        "kv_fabric_write": "True",
        "executor_thread_pin_cores": "2",
        "postlaunch_admission": "True",
    },
    "dep2ctx": {**_CONTEXT, "kv_fabric_write": "True", "store_lease_touch": "True"},
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
    assert len(calls) == 1, f"{target}: expected one RuntimeHooks(...) call, found {len(calls)}"
    return {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords if kw.arg}


@pytest.mark.parametrize("target", sorted(EXPECTED))
def test_target_opts_into_engine_features(target):
    kwargs = _runtime_hooks_kwargs(target)
    for name, value in {**_EVERY_TARGET, **EXPECTED[target]}.items():
        assert kwargs.get(name) == value, f"{target}: {name}={kwargs.get(name)!r}, want {value}"


@pytest.mark.parametrize("target", ["tp2", "tp4", "tp8"])
def test_tp_decode_targets_install_native_dflash_merge(target):
    source = (_MODELS / f"{_PREFIX}{target}" / "modeling.py").read_text()
    assert "_dflash_merge.install()" in source
