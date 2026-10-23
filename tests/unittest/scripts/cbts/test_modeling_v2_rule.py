#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for CBTS ModelingV2Rule: per-family selection over Python and native paths."""

from __future__ import annotations

from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, Stage, YAMLIndex
from cbts.rules.base import PRInputs
from cbts.rules.modeling_v2 import ModelingV2Rule, _is_mv2_claim, _snake

pytestmark = pytest.mark.cpu_only

_PY = "tensorrt_llm/_torch/_experimental/modeling_v2"
_CPP = "cpp/tensorrt_llm/modeling_v2"
_CATALOG_UNIT = (
    "unittest/_torch/modeling_v2 --ignore=unittest/_torch/modeling_v2/comm "
    "--ignore=unittest/_torch/modeling_v2/models TIMEOUT (30)"
)
_FIXTURE_UNIT = "unittest/_torch/modeling_v2/models/fixture"
_GPT_OSS_UNIT = "unittest/_torch/modeling_v2/models/gpt_oss"
_GPT_OSS_ACCURACY = "accuracy/modeling_v2/test_gpt_oss.py::TestGptOss::test_gsm8k[tp1]"
_CLAIMS = "unittest/_torch/modeling_v2/test_modeling_v2_claims.py"


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    for family in ("fixture", "gpt_oss", "minimax_m3"):
        package = tmp_path / _PY / "models" / family
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('"""Family."""\n')
    return tmp_path


@pytest.fixture
def rule(repo_root: Path) -> ModelingV2Rule:
    yaml_index = YAMLIndex()
    yaml_index.blocks = [
        Block(
            yaml_stem="l0_b300",
            block_index=0,
            condition={},
            tests=[
                "accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_nvfp4",
                _CATALOG_UNIT,
                _FIXTURE_UNIT,
                _GPT_OSS_UNIT,
                _GPT_OSS_ACCURACY,
            ],
        ),
        Block(yaml_stem="l0_cpu", block_index=0, condition={}, tests=["unittest/tools", _CLAIMS]),
    ]
    stages = {
        "B300-PyTorch-1": Stage("B300-PyTorch-1", "l0_b300", "aarch64", 1, 2),
        "CPU-Generic-x86-1": Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1),
    }
    return ModelingV2Rule(yaml_index, stages, repo_root=repo_root)


def _apply(rule: ModelingV2Rule, *files: str):
    return rule.apply(PRInputs(changed_files=list(files), diffs={}))


def _selected(result) -> set[str]:
    return {
        entry
        for prefixes in result.block_filters.values()
        for entries in prefixes.values()
        for entry in entries
    }


def test_claims_python_and_native_sources_but_not_docs_or_stock():
    assert _is_mv2_claim(f"{_PY}/models/fixture/scale_add__sm_80__default/__init__.py")
    assert _is_mv2_claim(f"{_CPP}/fixture/scaleAddSm80Default/kernels/scaleAdd.cu")
    assert _is_mv2_claim(f"{_CPP}/contractSurface.txt")
    assert _is_mv2_claim("cpp/cmake/modules/modelingV2.cmake")
    assert not _is_mv2_claim(f"{_CPP}/README.md")
    assert not _is_mv2_claim("cpp/tensorrt_llm/thop/modelingV2ContractOp.cpp")
    assert not _is_mv2_claim("tensorrt_llm/_torch/pyexecutor/model_loader.py")


def test_family_of_reads_both_trees(rule):
    assert rule.family_of(f"{_PY}/models/minimax_m3/routing.py") == "minimax_m3"
    assert rule.family_of(f"{_CPP}/minimaxM3/shared/attnMega.cu") == "minimax_m3"
    assert rule.family_of(f"{_PY}/catalog/_op.py") is None
    assert rule.family_of(f"{_PY}/models/__init__.py") is None
    assert rule.family_of(f"{_CPP}/common/torchLibrary.h") is None
    assert rule.family_of(f"{_CPP}/CMakeLists.txt") is None


def test_snake_inverts_camel_without_a_repo_root():
    assert _snake("glmMoeDsa") == "glm_moe_dsa"
    assert _snake("minimaxM3") == "minimax_m3"
    assert _snake("kimiK3Vl") == "kimi_k3_vl"


@pytest.mark.parametrize(
    "changed",
    [
        f"{_PY}/models/fixture/scale_add__sm_80__default/ops/elementwise/scale_add.py",
        f"{_CPP}/fixture/scaleAddSm80Default/kernels/scaleAdd.cu",
        f"{_CPP}/fixture/CMakeLists.txt",
    ],
)
def test_a_family_change_selects_that_family_and_the_checks(rule, changed):
    result = _apply(rule, changed)
    assert result.scope == "modelingv2only"
    assert _selected(result) == {_FIXTURE_UNIT, _CLAIMS}
    assert result.sanity_relevant is False


def test_two_families_union(rule):
    result = _apply(
        rule,
        f"{_PY}/models/fixture/scale_add__sm_80__default/__init__.py",
        f"{_PY}/models/gpt_oss/routing.py",
    )
    assert _selected(result) == {_FIXTURE_UNIT, _GPT_OSS_UNIT, _GPT_OSS_ACCURACY, _CLAIMS}


@pytest.mark.parametrize(
    "changed",
    [
        f"{_PY}/catalog/_op.py",
        f"{_PY}/_native.py",
        f"{_CPP}/common/torchLibrary.h",
        f"{_CPP}/contractSurface.txt",
        "cpp/cmake/modules/modelingV2.cmake",
    ],
)
def test_shared_modeling_v2_code_selects_everything_of_modeling_v2(rule, changed):
    result = _apply(rule, changed)
    assert result.scope == "modelingv2only"
    assert _selected(result) == {
        _CATALOG_UNIT,
        _FIXTURE_UNIT,
        _GPT_OSS_UNIT,
        _GPT_OSS_ACCURACY,
        _CLAIMS,
    }


def test_a_family_with_no_test_entry_falls_back(rule):
    result = _apply(rule, f"{_CPP}/minimaxM3/shared/attnMega.cu")
    assert result.scope is None
    assert "minimax_m3" in result.reason


def test_stock_changes_are_not_claimed(rule):
    assert _apply(rule, "tensorrt_llm/_torch/pyexecutor/model_loader.py") is None
