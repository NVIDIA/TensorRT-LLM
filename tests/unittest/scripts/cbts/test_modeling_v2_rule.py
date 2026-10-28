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
from cbts.blocks import Block, Stage, YAMLIndex
from cbts.rules.base import PRInputs
from cbts.rules.modeling_v2 import ModelingV2Rule, _is_mv2_claim, _kept_entries

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
def rule(tmp_path) -> ModelingV2Rule:
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
    for family, target, cls in (
        ("gpt_oss", "gpt_oss_120b__sm_10x__default", "GptOss"),
        ("deepseek_v3", "target", "Deepseek"),
        ("kimi_k3_vl", "target", "Kimi"),
    ):
        root = tmp_path / _PY / "models" / family
        (root / target).mkdir(parents=True)
        (root / target / "modeling.py").touch()
        (root / "routing.py").write_text(f"from .{target}.modeling import {cls}\n")
    return ModelingV2Rule(yaml_index, stages, repo_root=tmp_path)


def _apply(rule: ModelingV2Rule, *files: str):
    return rule.apply(PRInputs(changed_files=list(files), diffs={}))


def _selected(result) -> set[str]:
    return {
        entry
        for prefixes in result.block_filters.values()
        for entries in prefixes.values()
        for entry in entries
    }


def test_claims_and_families():
    """Python and native modeling_v2 sources are claimed; docs and stock are not.

    A path under models/<family>/ belongs to that family; anything else claimed is
    shared by every family.
    """
    for path in (
        f"{_PY}/models/gpt_oss/gpt_oss_120b__sm_10x__default/csrc/swizzleBlockScales.cu",
        f"{_PY}/models/gpt_oss/csrc/CMakeLists.txt",
        f"{_CPP}/common/namespace.h",
        "cpp/cmake/modules/modeling_v2.cmake",
    ):
        assert _is_mv2_claim(path), path
    for path in (
        f"{_CPP}/README.md",
        "cpp/tensorrt_llm/thop/attentionOp.cpp",
        "tensorrt_llm/_torch/pyexecutor/model_loader.py",
    ):
        assert not _is_mv2_claim(path), path


def test_entry_runs_family():
    for entry, runs in (
        ("unittest/_torch/modeling_v2 --ignore=unittest/_torch/modeling_v2/comm", True),
        ("unittest/_torch/modeling_v2/models/gpt_oss/test_x.py::test_y", True),
        ("accuracy/modeling_v2/test_gpt_oss.py::TestGptOss", True),
        ("unittest/_torch/modeling_v2 --ignore=unittest/_torch/modeling_v2/models", False),
        ("unittest/_torch/modeling_v2/models/gpt_oss_mini", False),
        ("accuracy/modeling_v2/test_deepseek_v3.py", False),
    ):
        prefixes = {
            "unittest/_torch/modeling_v2/models/gpt_oss",
            "accuracy/modeling_v2/test_gpt_oss.py",
        }
        assert bool(_kept_entries([entry], prefixes)) is runs, entry


def test_scope(rule):
    """A family change selects what runs that family's tests, plus the CPU checks.

    Families union; shared code selects every modeling_v2 entry; an unreadable family
    selects all modeling_v2 entries.
    """
    result = _apply(rule, f"{_PY}/models/gpt_oss/gpt_oss_120b__sm_10x__default/csrc/k.cu")
    assert result.scope == "modelingv2only" and result.sanity_relevant is False
    assert _selected(result) == {_UNIT, _GPT_OSS_ACCURACY, _CLAIMS}
    result = _apply(
        rule, f"{_PY}/models/deepseek_v3/routing.py", f"{_PY}/models/kimi_k3_vl/csrc/k.cu"
    )
    assert _selected(result) == {_UNIT, _DEEPSEEK_ACCURACY, _KIMI_UNIT, _CLAIMS}
    result = _apply(rule, f"{_CPP}/common/torchLibrary.h")
    assert _selected(result) == {
        _UNIT,
        _COMM,
        _KIMI_UNIT,
        _DEEPSEEK_ACCURACY,
        _GPT_OSS_ACCURACY,
        _CLAIMS,
    }
    assert _apply(rule, "tensorrt_llm/_torch/pyexecutor/model_loader.py") is None
    rule.yaml_index.blocks[0].tests.remove(_UNIT)
    result = _apply(rule, f"{_PY}/models/minimax_m3/csrc/attnMega.cu")
    assert result.scope == "modelingv2only"
    assert _COMM in _selected(result)  # unreadable family falls back to all modeling_v2
