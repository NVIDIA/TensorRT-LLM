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
"""Tests for CBTS ModelingV2Rule: per-target and per-family selection, native code included."""

from __future__ import annotations

from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, Stage, YAMLIndex, block_matches_stage, parse_stages_from_groovy
from cbts.rules.base import PRInputs
from cbts.rules.modeling_v2 import ModelingV2Rule, _is_mv2_claim, _kept_entries

pytestmark = pytest.mark.cpu_only

_PY = "tensorrt_llm/_torch/modeling_v2"
_CPP = "cpp/tensorrt_llm/modeling_v2"
_UNIT = "unittest/_torch/modeling_v2 --ignore=unittest/_torch/modeling_v2/comm TIMEOUT (30)"
_COMM = "unittest/_torch/modeling_v2/comm"
_NATIVE_UNIT = "unittest/_torch/modeling_v2/models/glm_moe_dsa/test_modeling_v2_glm_native.py"
_GPT_OSS_TARGET = "gpt_oss_120b__sm_10x__default"
_DEEPSEEK_ACCURACY = (
    "accuracy/modeling_v2/deepseek_v3/target/test_accuracy.py::TestDeepseek::test_gsm8k[tp4]"
)
_GPT_OSS_ACCURACY = (
    f"accuracy/modeling_v2/gpt_oss/{_GPT_OSS_TARGET}/test_accuracy.py::TestGptOss::test_gsm8k[tp1]"
)
_TREE = "unittest/_torch/modeling_v2/test_modeling_v2_routing.py"


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
        Block(yaml_stem="l0_cpu", block_index=0, condition={}, tests=["unittest/tools", _TREE]),
    ]
    stages = {
        "B300-PyTorch-1": Stage("B300-PyTorch-1", "l0_b300", "aarch64", 1, 2),
        "GB300-4_GPUs-PyTorch-1": Stage(
            "GB300-4_GPUs-PyTorch-1", "l0_gb300_multi_gpus", "aarch64", 1, 1
        ),
        "B200-PyTorch-1": Stage("B200-PyTorch-1", "l0_b200", "x86_64", 1, 1),
        "CPU-Generic-x86-1": Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1),
    }
    for family, target in (
        ("gpt_oss", _GPT_OSS_TARGET),
        ("deepseek_v3", "target"),
        ("glm_moe_dsa", "target"),
        ("kimi_k3_vl", "target"),
    ):
        (tmp_path / _PY / "models" / family / target).mkdir(parents=True)
        (tmp_path / _PY / "models" / family / target / "modeling.py").touch()
        (tmp_path / _PY / "models" / family / "routing.py").write_text(
            f'@routes("Arch")\ndef route(config):\n    from .{target}.modeling import Model\n\n'
            "    return Model\n"
        )
        (tmp_path / "tests/integration/defs/accuracy/modeling_v2" / family / target).mkdir(
            parents=True
        )
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
        ("accuracy/modeling_v2/gpt_oss/t/test_accuracy.py::TestGptOss", True),
        ("unittest/_torch/modeling_v2 --ignore=unittest/_torch/modeling_v2/models", False),
        ("unittest/_torch/modeling_v2/models/gpt_oss_mini", False),
        ("accuracy/modeling_v2/gpt_oss_mini/t/test_accuracy.py", False),
        ("accuracy/modeling_v2/deepseek_v3/t/test_accuracy.py", False),
    ):
        prefixes = {
            "unittest/_torch/modeling_v2/models/gpt_oss",
            "accuracy/modeling_v2/gpt_oss",
        }
        assert bool(_kept_entries([entry], prefixes)) is runs, entry


def test_scope(rule):
    """A target or family change selects what runs its tests, plus the CPU checks.

    Changes union; shared code selects every modeling_v2 entry; a family that no
    entry runs keeps only the CPU checks.
    """
    result = _apply(rule, f"{_PY}/models/gpt_oss/{_GPT_OSS_TARGET}/csrc/k.cu")
    assert result.scope == "modelingv2only" and result.sanity_relevant is False
    assert _selected(result) == {_UNIT, _GPT_OSS_ACCURACY, _TREE}
    result = _apply(
        rule, f"{_PY}/models/deepseek_v3/routing.py", f"{_PY}/models/glm_moe_dsa/csrc/k.cu"
    )
    assert _selected(result) == {_UNIT, _DEEPSEEK_ACCURACY, _NATIVE_UNIT, _TREE}
    result = _apply(rule, f"{_CPP}/common/torchLibrary.h")
    assert _selected(result) == {
        _UNIT,
        _COMM,
        _NATIVE_UNIT,
        _DEEPSEEK_ACCURACY,
        _GPT_OSS_ACCURACY,
        _TREE,
    }
    assert _apply(rule, "tensorrt_llm/_torch/pyexecutor/model_loader.py") is None
    rule.yaml_index.blocks[0].tests.remove(_UNIT)
    result = _apply(rule, f"{_PY}/models/minimax_m3/csrc/attnMega.cu")
    assert result.scope == "modelingv2only"
    assert _selected(result) == {_TREE}


@pytest.mark.parametrize("relative", ["csrc/kernel.cu", "fake_ops.py", "routing.py"])
def test_k3_uses_modeling_v2_selection(rule, relative):
    path = f"{_PY}/models/kimi_k3_vl/{relative}"
    assert _is_mv2_claim(path)
    assert _apply(rule, path).scope == "modelingv2only"


@pytest.mark.parametrize("relative", ["csrc/k3MnnvlAllreduce.cu", "routing.py"])
def test_k3_family_paths_select_kernel_collective_and_host_tests(rule, relative):
    prefix = "unittest/_torch/modeling_v2/models/kimi_k3_vl"
    kernel_tests = {
        f"{prefix}/ops/comm/test_k3_mnnvl_comm.py",
        f"{prefix}/kernels/k3_sandwich/test_k3_sandwich.py",
    }
    matrix = f"{prefix}/ops/comm/test_modeling_v2_mnnvl_allgather_split_op_matrix.py"
    norm = f"{prefix}/ops/norm/test_modeling_v2_attn_res_rmsnorm_fwd.py"
    native = f"{prefix}/test_modeling_v2_k3_native.py"
    host = f"{prefix}/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/test_tp16_moetp4ep4_decode_step.py"
    accuracy = "accuracy/modeling_v2/kimi_k3_vl/target/test_accuracy.py::TestKimiK3::test_gsm8k"
    # The broad single-GPU entry excludes the family directory; its
    # single-GPU tests and collective tests have separate stage entries.
    rule.yaml_index.blocks[0].tests.remove(_UNIT)
    rule.yaml_index.blocks[0].tests.extend([f"{_UNIT} --ignore={prefix}", native, accuracy])
    rule.yaml_index.blocks[1].tests.extend([*sorted(kernel_tests), matrix])
    rule.yaml_index.blocks[2].tests.append(norm)
    rule.yaml_index.blocks[3].tests.append(host)
    result = _apply(rule, f"{_PY}/models/kimi_k3_vl/{relative}")
    selected = kernel_tests | {matrix, norm, native, host, accuracy, _TREE}
    assert _selected(result) == selected
    assert result.affected_stages == {
        "B300-PyTorch-1",
        "B200-PyTorch-1",
        "GB300-4_GPUs-PyTorch-1",
        "CPU-Generic-x86-1",
    }
    assert result.scope == "modelingv2only"
    other = _selected(_apply(rule, f"{_PY}/models/gpt_oss/csrc/kernel.cu"))
    assert not (selected - {_TREE}) & other


def test_k3_ci_runs_once_on_b300_or_gb300_by_topology():
    repo = Path(__file__).resolve().parents[4]
    prefix = "unittest/_torch/modeling_v2/models/kimi_k3_vl/"
    # The unit tests mirror the family's source tree (and its targets), so they sit in subdirectories.
    expected = {
        p.relative_to(repo / "tests" / prefix).as_posix()
        for p in (repo / "tests" / prefix).rglob("test_*.py")
    }
    # The 4-GPU tests, by path under the family's unit directory: they run on GB300, every other GPU test on B300.
    collective_tests = {
        "kernels/k3_fused_moe/test_k3_fused_moe_push.py",
        "kernels/k3_fused_moe/test_k3_latent_reduce.py",
        "kernels/k3_fused_moe/test_k3_moe_front.py",
        "kernels/k3_markov/test_k3_markov.py",
        "kernels/k3_sandwich/test_k3_sandwich.py",
        "kernels/k3_spec_accept/test_k3_spec_accept_sharded.py",
        "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/test_k3_moe_push.py",
        "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/test_modeling_v2_kimi_k3_decode_comm_tp16_moetp16ep1_op_matrix.py",
        "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/test_modeling_v2_kimi_k3_route_b_decode_moe_op_matrix.py",
        "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/test_modeling_v2_kimi_k3_decode_comm_tp16_moetp4ep4_op_matrix.py",
        "ops/comm/test_k3_mnnvl_comm.py",
        "ops/comm/test_modeling_v2_k3_latent_reduce_op_matrix.py",
        "ops/comm/test_modeling_v2_k3_sandwich_oproj_op_matrix.py",
        "ops/comm/test_modeling_v2_k3_sandwich_plain_op_matrix.py",
        "ops/comm/test_modeling_v2_k3_sandwich_tail_op_matrix.py",
        "ops/comm/test_modeling_v2_mnnvl_allgather_split_op_matrix.py",
        "ops/comm/test_modeling_v2_mnnvl_allreduce_attn_res_op_matrix.py",
        "ops/comm/test_modeling_v2_mnnvl_fusion_allreduce_op_matrix.py",
        "ops/moe/test_modeling_v2_k3_moe_front_op_matrix.py",
    }
    assert collective_tests <= expected, collective_tests - expected
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
