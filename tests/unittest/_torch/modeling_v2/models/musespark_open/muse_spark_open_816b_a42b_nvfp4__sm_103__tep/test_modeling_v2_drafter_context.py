# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The musespark_open TEP target over each layout of the DFlash drafter's context.

* ``compose`` plans every fast path over the dense arena and every one but
  ``DENSE_ARENA_PATHS`` and their dependents over a paged context, whatever the
  attention backend that keeps it (no GPU).
"""

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import ModelingV2Mode
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)

_KEYS = sorted(modeling.TARGET_PATHS)


# --------------------------------------------------------------------------
# compose: the plan per drafter context layout
# --------------------------------------------------------------------------
def _model_config(backend, tp):
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    spec = DFlashDecodingConfig(
        max_draft_len=7, speculative_model="/drafter", attention_backend=backend
    )
    mapping = types.SimpleNamespace(tp_size=tp, gpus_per_node=4, tp_group=list(range(tp)))
    return types.SimpleNamespace(
        mapping=mapping,
        spec_config=spec,
        quant_config=types.SimpleNamespace(quant_algo=QuantAlgo.NVFP4),
    )


def _plan(backend, monkeypatch, spec_worker=None, key=(103, 4)):
    """A composition's plan with no precondition read and no TP collective."""
    monkeypatch.setattr(dispatch, "missing_preconditions", lambda *args: [])
    monkeypatch.setattr(dispatch, "agreed_union", lambda mapping, local: list(dict.fromkeys(local)))
    sm, tp = key
    plan = dispatch.compose(
        None,
        _model_config(backend, tp),
        ModelingV2Mode.REQUIRE,
        modeling.TARGET_PATHS,
        divmod(sm, 10),
        spec_worker,
    )
    return set(modeling.TARGET_PATHS[key].served), set(plan.paths)


def _left_out(served, *names):
    return set(dispatch.dependents(names)) & served


@pytest.mark.parametrize("key", _KEYS)
def test_the_full_dense_arena_keeps_every_full_arena_fast_path(key, monkeypatch):
    served, planned = _plan("VANILLA", monkeypatch, key=key)
    assert served - planned == _left_out(served, *dispatch.WINDOWED_CONTEXT_ARENA_PATHS)


@pytest.mark.parametrize("key", _KEYS)
@pytest.mark.parametrize("backend", ["TRTLLM", "FA4"])
def test_a_paged_context_drops_only_the_dense_arena_paths(backend, key, monkeypatch):
    served, planned = _plan(backend, monkeypatch, key=key)
    dropped = dispatch.DENSE_ARENA_PATHS + dispatch.WINDOWED_CONTEXT_ARENA_PATHS
    assert served - planned == _left_out(served, *dropped)
    assert "context_kv_fork" not in planned


def test_the_window_rule_bounds_the_dense_arena_only(monkeypatch):
    windowed = types.SimpleNamespace(windowed_ctx_arena=True)
    served, dense = _plan("VANILLA", monkeypatch, windowed)
    # sm103 tep4 serves the window compaction, so a windowed arena keeps every path.
    assert dense == served
    assert _plan("TRTLLM", monkeypatch, windowed)[1] == _plan("TRTLLM", monkeypatch)[1]


def test_the_worker_backend_settles_an_open_auto(monkeypatch):
    worker = types.SimpleNamespace(_dflash_attention_backend="TRTLLM")
    assert _plan("AUTO", monkeypatch, worker)[1] == _plan("TRTLLM", monkeypatch)[1]
