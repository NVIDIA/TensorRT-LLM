# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public routing of configurable NVFP4 deployments and specialized SM107 deployments."""

import itertools

import pytest
import test_routing_sm107 as fixtures

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31 import routing

pytestmark = pytest.mark.cpu_only

_RECIPES = tuple(itertools.product(("fp8", "nvfp4"), ("fp8", "fp4"), ("float32", "bfloat16")))


@pytest.fixture(autouse=True)
def targets(monkeypatch):
    fixtures._stand_in_targets(monkeypatch)
    monkeypatch.setattr(routing, "disagg_role", lambda llm_args=None: "generation")


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("recipe", _RECIPES)
@pytest.mark.parametrize("history", ["auto", "fp8"])
@pytest.mark.parametrize("backend", ["TRTLLM", "CUTEDSL", "MEGAMOE_CUTEDSL"])
def test_independent_formats_on_each_architecture(
    monkeypatch, tmp_path, capability, recipe, history, backend
):
    fixtures._on_gpu(monkeypatch, capability)
    main, indexer, state = recipe
    config = fixtures._modelopt_config(
        tmp_path, fixtures._TEP8, fixtures._dflash2(tmp_path, 7, context_kv_dtype=history)
    )
    args = fixtures._llm_args(
        backend=backend,
        kv_dtype=main,
        indexer_k_dtype=indexer,
        ssm_dtype=state,
        disable_mm_encoder=False,
    )
    assert modeling_v2_resolve(config, args).__name__ == fixtures._DEFAULT[1]


@pytest.mark.parametrize(
    "mapping",
    [
        dict(world_size=4, tp_size=4, moe_tp_size=2, moe_ep_size=2),
        dict(world_size=8, tp_size=8, enable_attention_dp=True, moe_ep_size=8),
        dict(
            world_size=4,
            tp_size=1,
            cp_size=4,
            cp_config={"cp_type": "HELIX"},
            moe_tp_size=2,
            moe_ep_size=2,
        ),
    ],
)
@pytest.mark.parametrize("speculative", [False, True])
def test_rubin_configurable_parallelism(monkeypatch, tmp_path, mapping, speculative):
    fixtures._on_gpu(monkeypatch, (10, 7))
    spec = fixtures._dflash2(tmp_path, 4, context_kv_dtype="auto") if speculative else None
    config = fixtures._modelopt_config(tmp_path, mapping, spec)
    args = fixtures._llm_args(
        backend="CUTEDSL", ssm_dtype="bfloat16", kv_cache={"enable_block_reuse": False}
    )
    assert modeling_v2_resolve(config, args).__name__ == fixtures._DEFAULT[1]


@pytest.mark.parametrize(
    "layout,role,backend",
    [
        ("tep8", "context", "CUTEDSL"),
        ("tep8", "generation", "CUTEDSL_FC12"),
        ("dep16", "context", "CUTEDSL"),
        ("dep16", "generation", "MEGAMOE_CUTEDSL"),
        ("helix8", "generation", "CUTEDSL_FC12"),
    ],
)
def test_eligible_fixed_layout_selects_specialized(monkeypatch, tmp_path, layout, role, backend):
    fixtures._on_gpu(monkeypatch, (10, 7))
    monkeypatch.setattr(routing, "disagg_role", lambda llm_args=None: role)
    config = fixtures._modelopt_config(
        tmp_path, fixtures._LAYOUTS[layout], fixtures._dflash2(tmp_path, 7)
    )
    args = fixtures._llm_args(backend=backend, kv_cache={"enable_block_reuse": False})
    assert modeling_v2_resolve(config, args).__name__ == fixtures._SPECIALIZED[layout][1]


@pytest.mark.parametrize("vision,history", [(True, "fp8"), (False, "auto")])
def test_specialized_layout_falls_back_for_vision_or_bf16_history(
    monkeypatch, tmp_path, vision, history
):
    fixtures._on_gpu(monkeypatch, (10, 7))
    monkeypatch.setattr(routing, "disagg_role", lambda llm_args=None: "context")
    config = fixtures._modelopt_config(
        tmp_path, fixtures._TEP8, fixtures._dflash2(tmp_path, 7, context_kv_dtype=history)
    )
    args = fixtures._llm_args(backend="CUTEDSL", disable_mm_encoder=not vision)
    assert modeling_v2_resolve(config, args).__name__ == fixtures._DEFAULT[1]
