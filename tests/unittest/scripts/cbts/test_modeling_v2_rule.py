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

from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, Stage, YAMLIndex, block_matches_stage, parse_stages_from_groovy
from cbts.rules.base import PRInputs
from cbts.rules.modeling_v2 import ModelingV2Rule, _is_mv2_claim, _kept_entries

pytestmark = pytest.mark.cpu_only

_PY = "tensorrt_llm/_torch/_experimental/modeling_v2"
_CPP = "cpp/tensorrt_llm/modeling_v2"
_UNIT = "unittest/_torch/modeling_v2 --ignore=unittest/_torch/modeling_v2/comm TIMEOUT (30)"
_COMM = "unittest/_torch/modeling_v2/comm"
_NATIVE_UNIT = "unittest/_torch/modeling_v2/models/glm_moe_dsa/test_modeling_v2_glm_native.py"
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
        Block(yaml_stem="l0_b200", block_index=0, condition={}, tests=[_NATIVE_UNIT]),
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
        ("glm_moe_dsa", "target", "Glm"),
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
        rule, f"{_PY}/models/deepseek_v3/routing.py", f"{_PY}/models/glm_moe_dsa/csrc/k.cu"
    )
    assert _selected(result) == {_UNIT, _DEEPSEEK_ACCURACY, _NATIVE_UNIT, _CLAIMS}
    result = _apply(rule, f"{_CPP}/common/torchLibrary.h")
    assert _selected(result) == {
        _UNIT,
        _COMM,
        _NATIVE_UNIT,
        _DEEPSEEK_ACCURACY,
        _GPT_OSS_ACCURACY,
        _CLAIMS,
    }
    assert _apply(rule, "tensorrt_llm/_torch/pyexecutor/model_loader.py") is None
    rule.yaml_index.blocks[0].tests.remove(_UNIT)
    result = _apply(rule, f"{_PY}/models/minimax_m3/csrc/attnMega.cu")
    assert result.scope == "modelingv2only"
    assert _COMM in _selected(result)  # unreadable family falls back to all modeling_v2


@pytest.mark.parametrize("relative", ["csrc/kernel.cu", "fake_ops.py", "routing.py"])
def test_k3_uses_modeling_v2_selection(rule, relative):
    path = f"{_PY}/models/kimi_k3_vl/{relative}"
    assert _is_mv2_claim(path)
    assert _apply(rule, path).scope == "modelingv2only"


@pytest.mark.parametrize("relative", ["csrc/k3MnnvlAllreduce.cu", "target/modeling.py"])
def test_k3_family_paths_select_kernel_collective_and_host_tests(rule, relative):
    family = rule._src_root / "models/kimi_k3_vl"
    (family / "target").mkdir(parents=True)
    (family / "target/modeling.py").touch()
    (family / "routing.py").write_text("from .target.modeling import KimiK3\n")
    prefix = "unittest/_torch/modeling_v2/models/kimi_k3_vl"
    kernel_tests = {
        f"{prefix}/test_k3_mnnvl_comm.py",
        f"{prefix}/test_k3_sandwich.py",
    }
    matrix = f"{prefix}/test_modeling_v2_mnnvl_allgather_split_op_matrix.py"
    norm = f"{prefix}/test_modeling_v2_attn_res_rmsnorm_fwd.py"
    native = f"{prefix}/test_modeling_v2_k3_native.py"
    host = f"{prefix}/test_modeling_v2_kimi_k3_decode_step.py"
    accuracy = "accuracy/modeling_v2/test_kimi_k3_vl.py::TestKimiK3::test_gsm8k"
    # The broad single-GPU entry excludes the flat family directory; its
    # single-GPU tests and collective tests have separate stage entries.
    rule.yaml_index.blocks[0].tests.remove(_UNIT)
    rule.yaml_index.blocks[0].tests.extend([f"{_UNIT} --ignore={prefix}", native, accuracy])
    rule.yaml_index.blocks[1].tests.extend([*sorted(kernel_tests), matrix])
    rule.yaml_index.blocks[2].tests.append(norm)
    rule.yaml_index.blocks[3].tests.append(host)
    result = _apply(rule, f"{_PY}/models/kimi_k3_vl/{relative}")
    selected = kernel_tests | {matrix, norm, native, host, accuracy, _CLAIMS}
    assert _selected(result) == selected
    assert result.affected_stages == {
        "B300-PyTorch-1",
        "B200-PyTorch-1",
        "GB300-4_GPUs-PyTorch-1",
        "CPU-Generic-x86-1",
    }
    assert result.scope == "modelingv2only"
    other = _selected(_apply(rule, f"{_PY}/models/gpt_oss/csrc/kernel.cu"))
    assert not (selected - {_CLAIMS}) & other


def test_k3_ci_runs_once_on_b300_or_gb300_by_topology():
    repo = Path(__file__).resolve().parents[4]
    prefix = "unittest/_torch/modeling_v2/models/kimi_k3_vl/"
    expected = {p.name for p in (repo / "tests" / prefix).glob("test_*.py")}
    collective_tests = {
        "test_k3_latent_reduce.py",
        "test_k3_markov.py",
        "test_k3_mnnvl_comm.py",
        "test_k3_moe_front.py",
        "test_k3_moe_push.py",
        "test_k3_sandwich.py",
        "test_k3_spec_accept_sharded.py",
        "test_modeling_v2_k3_latent_reduce_op_matrix.py",
        "test_modeling_v2_k3_moe_front_op_matrix.py",
        "test_modeling_v2_k3_sandwich_oproj_op_matrix.py",
        "test_modeling_v2_k3_sandwich_plain_op_matrix.py",
        "test_modeling_v2_k3_sandwich_tail_op_matrix.py",
        "test_modeling_v2_kimi_k3_decode_comm_op_matrix.py",
        "test_modeling_v2_kimi_k3_route_b_decode_moe_op_matrix.py",
        "test_modeling_v2_mnnvl_allgather_split_op_matrix.py",
        "test_modeling_v2_mnnvl_allreduce_attn_res_op_matrix.py",
        "test_modeling_v2_mnnvl_fusion_allreduce_op_matrix.py",
    }
    index = YAMLIndex.load(repo / "tests/integration/test_lists/test-db")
    stages = parse_stages_from_groovy(repo / "jenkins/L0_Test.groovy")
    seen = set()
    for block in index.blocks:
        for entry in block.tests:
            target = entry.split()[0].split("::")[0]
            if not target.startswith(prefix):
                # Broad directory entries must exclude this family outside its
                # dedicated stages, or the GPU coverage would be duplicated.
                if prefix.startswith(target.rstrip("/") + "/"):
                    assert any(
                        prefix.rstrip("/") == arg.removeprefix("--ignore=").rstrip("/")
                        for arg in entry.split()
                        if arg.startswith("--ignore=")
                    ), (block.yaml_stem, entry)
                continue
            name = target.removeprefix(prefix)
            assert name not in seen, (block.yaml_stem, name)
            seen.add(name)
            if block.yaml_stem == "l0_cpu":
                assert name not in collective_tests
                continue
            multi = name in collective_tests
            assert block.yaml_stem == ("l0_gb300" if multi else "l0_b300")
            count = 4 if multi else 1
            assert block.condition["ranges"]["system_gpu_count"] == {"gte": count, "lte": count}
            assert block.condition["terms"]["stage"] == "pre_merge"
            matched = {
                stage.name
                for stage in stages.values()
                if stage.yaml_stem == block.yaml_stem and block_matches_stage(block, stage)
            }
            assert matched == (
                {"GB300-4_GPUs-PyTorch-1"} if multi else {"B300-PyTorch-1", "B300-PyTorch-2"}
            )
    assert seen == expected
