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
"""Tests for CBTS ModelingV2Rule: per-family selection, native code included."""

from __future__ import annotations

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, Stage, YAMLIndex, _entry_target
from cbts.rules.base import PRInputs
from cbts.rules.modeling_v2 import ModelingV2Rule, _is_mv2_claim

pytestmark = pytest.mark.cpu_only

_PY = "tensorrt_llm/_torch/_experimental/modeling_v2"
_CPP = "cpp/tensorrt_llm/modeling_v2"
_UNIT = "unittest/_torch/modeling_v2 --ignore=unittest/_torch/modeling_v2/comm TIMEOUT (30)"
_COMM = "unittest/_torch/modeling_v2/comm"
_KIMI_UNIT = "unittest/_torch/modeling_v2/models/kimi_k3_vl/test_modeling_v2_kimi_k3_native.py"
_DEEPSEEK_ACCURACY = "accuracy/modeling_v2/test_deepseek_v3.py::TestDeepseek::test_gsm8k[tp4]"
_GPT_OSS_ACCURACY = "accuracy/modeling_v2/test_gpt_oss.py::TestGptOss::test_gsm8k[tp1]"
_CLAIMS = "unittest/_torch/modeling_v2/test_modeling_v2_claims.py"


@pytest.fixture
def rule() -> ModelingV2Rule:
    yaml_index = YAMLIndex()
    yaml_index.blocks = [
        Block(
            yaml_stem="l0_b300",
            block_index=0,
            condition={},
            tests=[
                "accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_nvfp4",
                _UNIT,
                _GPT_OSS_ACCURACY,
            ],
        ),
        Block(
            yaml_stem="l0_gb300_multi_gpus",
            block_index=0,
            condition={},
            tests=[_COMM, _DEEPSEEK_ACCURACY],
        ),
        Block(yaml_stem="l0_b200", block_index=0, condition={}, tests=[_KIMI_UNIT]),
        Block(yaml_stem="l0_cpu", block_index=0, condition={}, tests=["unittest/tools", _CLAIMS]),
    ]
    stages = {
        "B300-PyTorch-1": Stage("B300-PyTorch-1", "l0_b300", "aarch64", 1, 2),
        "GB300-4_GPUs-PyTorch-1": Stage(
            "GB300-4_GPUs-PyTorch-1", "l0_gb300_multi_gpus", "aarch64", 1, 1
        ),
        "B200-PyTorch-1": Stage("B200-PyTorch-1", "l0_b200", "x86_64", 1, 1),
        "CPU-Generic-x86-1": Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1),
    }
    return ModelingV2Rule(yaml_index, stages)


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
    assert _is_mv2_claim(f"{_PY}/models/gpt_oss/gpt_oss_120b__sm_10x__default/__init__.py")
    assert _is_mv2_claim(
        f"{_PY}/models/gpt_oss/gpt_oss_120b__sm_10x__default/csrc/swizzleBlockScales.cu"
    )
    assert _is_mv2_claim(f"{_PY}/models/gpt_oss/csrc/CMakeLists.txt")
    assert _is_mv2_claim(f"{_CPP}/common/namespace.h")
    assert _is_mv2_claim("cpp/cmake/modules/modeling_v2.cmake")
    assert not _is_mv2_claim(f"{_CPP}/README.md")
    assert not _is_mv2_claim("cpp/tensorrt_llm/thop/attentionOp.cpp")
    assert not _is_mv2_claim("tensorrt_llm/_torch/pyexecutor/model_loader.py")


def test_family_of_reads_the_family_directory_native_code_included(rule):
    assert rule.family_of(f"{_PY}/models/minimax_m3/routing.py") == "minimax_m3"
    assert rule.family_of(f"{_PY}/models/minimax_m3/csrc/attnMega.cu") == "minimax_m3"
    assert (
        rule.family_of(f"{_PY}/models/minimax_m3/m3__sm_107__tp2/csrc/draftStep.cu") == "minimax_m3"
    )
    assert rule.family_of(f"{_PY}/catalog/_op.py") is None
    assert rule.family_of(f"{_PY}/models/__init__.py") is None
    assert rule.family_of(f"{_CPP}/common/torchLibrary.h") is None
    assert rule.family_of(f"{_CPP}/CMakeLists.txt") is None


@pytest.mark.parametrize(
    "changed",
    [
        f"{_PY}/models/gpt_oss/gpt_oss_120b__sm_10x__default/weights.py",
        f"{_PY}/models/gpt_oss/gpt_oss_120b__sm_10x__default/csrc/swizzleBlockScales.cu",
        f"{_PY}/models/gpt_oss/csrc/CMakeLists.txt",
    ],
)
def test_a_family_change_selects_what_runs_that_family_and_the_checks(rule, changed):
    """The directory entry collects every family's unit tests; comm/ is --ignored."""
    result = _apply(rule, changed)
    assert result.scope == "modelingv2only"
    assert _selected(result) == {_UNIT, _GPT_OSS_ACCURACY, _CLAIMS}
    assert result.sanity_relevant is False


def test_a_family_keeps_its_own_entries_on_other_gpus(rule):
    result = _apply(rule, f"{_PY}/models/kimi_k3_vl/csrc/k3AttnResDecode.cu")
    assert _selected(result) == {_UNIT, _KIMI_UNIT, _CLAIMS}


def test_two_families_union(rule):
    result = _apply(
        rule,
        f"{_PY}/models/deepseek_v3/routing.py",
        f"{_PY}/models/gpt_oss/csrc/CMakeLists.txt",
    )
    assert _selected(result) == {_UNIT, _DEEPSEEK_ACCURACY, _GPT_OSS_ACCURACY, _CLAIMS}


@pytest.mark.parametrize(
    ("entry", "runs"),
    [
        ("unittest/_torch/modeling_v2", True),
        ("unittest/_torch/modeling_v2/ --ignore=unittest/_torch/modeling_v2/comm", True),
        ("unittest/_torch/modeling_v2/models", True),
        ("unittest/_torch/modeling_v2/models/gpt_oss/test_x.py::test_y", True),
        ("unittest/_torch/modeling_v2 --ignore=unittest/_torch/modeling_v2/models", False),
        ("unittest/_torch/modeling_v2 --ignore=unittest/_torch/modeling_v2/models/gpt_oss/", False),
        ("unittest/_torch/modeling_v2/models/gpt_oss_mini", False),
        ("unittest/_torch/modeling_v2/comm", False),
        ("accuracy/modeling_v2/test_gpt_oss.py::TestGptOss", True),
        ("accuracy/modeling_v2/test_deepseek_v3.py", False),
    ],
)
def test_entry_runs_family(entry, runs):
    assert ModelingV2Rule._entry_runs_family(entry, _entry_target(entry), "gpt_oss") is runs


@pytest.mark.parametrize(
    "changed",
    [
        f"{_PY}/catalog/_op.py",
        f"{_PY}/_native.py",
        f"{_CPP}/common/torchLibrary.h",
        f"{_CPP}/CMakeLists.txt",
        "cpp/cmake/modules/modeling_v2.cmake",
    ],
)
def test_shared_modeling_v2_code_selects_everything_of_modeling_v2(rule, changed):
    result = _apply(rule, changed)
    assert result.scope == "modelingv2only"
    assert _selected(result) == {
        _UNIT,
        _COMM,
        _KIMI_UNIT,
        _DEEPSEEK_ACCURACY,
        _GPT_OSS_ACCURACY,
        _CLAIMS,
    }


def test_a_family_no_entry_runs_falls_back(rule):
    rule.yaml_index.blocks[0].tests.remove(_UNIT)
    result = _apply(rule, f"{_PY}/models/minimax_m3/csrc/attnMega.cu")
    assert result.scope is None
    assert "minimax_m3" in result.reason


def test_stock_changes_are_not_claimed(rule):
    assert _apply(rule, "tensorrt_llm/_torch/pyexecutor/model_loader.py") is None
