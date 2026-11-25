# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 GEMM backend tuple of the MiniMax-H3 scenario ``minimax_h3_nvfp4__sm_107__u1_u4``: its
default and validation, the Linear attribute it becomes, and the tuner key that carries it (CPU only)."""

import ast
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4 import (
    config as scenario_config,
)

pytestmark = pytest.mark.cpu_only

FAMILY_TUPLE = ("cutlass", "cublaslt", "cutedsl", "cuda_core")


@pytest.fixture(autouse=True)
def _no_selection_yet(monkeypatch):
    monkeypatch.setattr(scenario_config, "SCENARIO", scenario_config.ScenarioConfig())
    monkeypatch.setattr(scenario_config, "_SELECTED", None)


def test_default_is_cublaslt_only():
    cfg = scenario_config.ScenarioConfig()
    cfg.validate()
    assert cfg.nvfp4_gemm_backends == ("cublaslt",)
    assert set(FAMILY_TUPLE) == scenario_config.NVFP4_GEMM_BACKENDS


def test_family_tuple_is_selectable_from_a_list_in_order():
    cfg = scenario_config.apply({"nvfp4_gemm_backends": list(FAMILY_TUPLE)})
    assert cfg.nvfp4_gemm_backends == FAMILY_TUPLE
    assert scenario_config.SCENARIO is cfg


@pytest.mark.parametrize(
    "backends",
    [(), ("cublaslt", "cublaslt"), ("cublaslt", "marlin"), ("cuBLASLt",), ("cutedsl", "nope")],
)
def test_invalid_tuple_raises_and_changes_nothing(backends):
    before = (scenario_config.SCENARIO, scenario_config._SELECTED)
    with pytest.raises(ValueError, match="nvfp4_gemm_backends"):
        scenario_config.apply({"nvfp4_gemm_backends": list(backends)})
    assert (scenario_config.SCENARIO, scenario_config._SELECTED) == before


def test_linear_reports_the_tuple_and_the_tuner_key_carries_it():
    cfg = scenario_config.apply(None)
    linear = Linear(
        32,
        32,
        bias=False,
        dtype=torch.bfloat16,
        nvfp4_allowed_backends=list(cfg.nvfp4_gemm_backends),
    )
    assert linear.nvfp4_allowed_backends == ["cublaslt"]
    key = NVFP4GemmUnifiedRunner(0, torch.bfloat16, linear.nvfp4_allowed_backends).unique_id()
    assert key == (0, torch.bfloat16, ("cublaslt",))
    family = NVFP4GemmUnifiedRunner(0, torch.bfloat16, list(FAMILY_TUPLE)).unique_id()
    assert family != key and family[2] == FAMILY_TUPLE


def test_linear_default_is_the_family_tuple():
    linear = Linear(32, 32, bias=False, dtype=torch.bfloat16)
    assert tuple(linear.nvfp4_allowed_backends) == FAMILY_TUPLE


def test_modeling_sets_every_linear_from_the_setting():
    """Static (the scenario's modules build SM 107 kernels at import): ``ScenarioTransformer`` assigns
    ``nvfp4_allowed_backends`` from ``SCENARIO.nvfp4_gemm_backends`` in its constructor."""
    source = (Path(scenario_config.__file__).parent / "modeling.py").read_text()
    tree = ast.parse(source)
    transformer = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScenarioTransformer"
    )
    methods = {n.name: n for n in transformer.body if isinstance(n, ast.FunctionDef)}
    assert "_restrict_nvfp4_backends" in methods
    init_calls = {
        n.func.attr
        for n in ast.walk(methods["__init__"])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "_restrict_nvfp4_backends" in init_calls
    body = methods["_restrict_nvfp4_backends"]
    reads = {
        n.attr
        for n in ast.walk(body)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "SCENARIO"
    }
    assert reads == {"nvfp4_gemm_backends"}
    assigned = {
        t.attr
        for n in ast.walk(body)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute)
    }
    assert assigned == {"nvfp4_allowed_backends"}
