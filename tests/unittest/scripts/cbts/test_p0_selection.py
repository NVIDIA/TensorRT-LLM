# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for verified formatting and VisualGen schema selection."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, Stage, YAMLIndex, parse_stages_from_groovy
from cbts.command.main import Selector, _load_pr_inputs, main
from cbts.rules.base import PRInputs
from cbts.rules.python_formatting import PythonFormattingRule, is_formatting_only_change
from cbts.rules.tests_def import TestsDefRule as CbtsTestsDefRule
from cbts.rules.visual_gen import VisualGenRule, _is_vg_claim

pytestmark = pytest.mark.cpu_only
REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "tensorrt_llm/schemas/trtllm-serve-visual-gen-config.schema.json"
SCHEMA_TEST = "unittest/llmapi/test_trtllm_serve_config_schemas.py"


def _diff(before: str, after: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("def f():\n    return (1, )\n", "def f():\n    return (1,)\n", True),
        (
            "def f(x):\n    return isinstance(\n        x,\n        int) and x\n",
            "def f(x):\n    return isinstance(x, int) and x\n",
            True,
        ),
        ("value = (\n    1 + 2\n)\n", "value = 1 + 2\n", True),
        ("value = 'a'\n", 'value = "a"\n', True),
        ("value = 1\n", "value = 2\n", False),
        ("def f():\n    return 1\n", "def f():\n    return 2\n", False),
        ("def f():\n    'old'\n    pass\n", "def f():\n    'new'\n    pass\n", False),
        ("value = 1 # noqa: F841\n", "value = 1 # noqa: E501\n", False),
        ("# coding: utf-8\nvalue = 1\n", "# coding: latin-1\nvalue = 1\n", False),
        ("value = 1 # type: int\n", "value = 1 # type: str\n", False),
        ("", "value = 1\n", False),
        ("value = 1\n", "", False),
        ("def f(:\n", "def f( :\n", False),
        ("value = '''a\nb'''\n", "value = '''a\n b'''\n", False),
    ],
)
def test_formatting_requires_equal_complete_syntax_and_comments(
    before: str, after: str, expected: bool
) -> None:
    assert is_formatting_only_change(after, _diff(before, after)) is expected


@pytest.mark.parametrize("patch", ["", "@@ -1,1 +1,1 @@\n-value=1\n+value= 2\n"])
def test_missing_or_mismatched_patch_is_not_formatting(patch: str) -> None:
    assert not is_formatting_only_change("value = 1\n", patch)


@pytest.mark.parametrize("prefix", ["rename from old.py\n", "old mode 100644\n"])
def test_path_and_mode_changes_are_not_formatting(prefix: str) -> None:
    assert not is_formatting_only_change("value = 1\n", prefix + _diff("value=1\n", "value = 1\n"))


def _formatting_pr(tmp_path: Path, path: str = "tensorrt_llm/example.py") -> PRInputs:
    before, after = "def f():\n    return (1, )\n", "def f():\n    return (1,)\n"
    source = tmp_path / path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(after)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"changed_files": [path], "diffs": {path: _diff(before, after)}})
    )
    return _load_pr_inputs(input_path, tmp_path)


@pytest.mark.parametrize(
    "path",
    ["tensorrt_llm/example.py", "tensorrt_llm/visual_gen/args.py", "tests/unittest/example.py"],
)
def test_formatting_bypasses_source_rules_without_adding_cpu_tests(
    tmp_path: Path, path: str
) -> None:
    pr = _formatting_pr(tmp_path, path)
    assert pr.formatting_only_files == {path}
    result = CliRunner().invoke(
        main,
        [
            str(tmp_path / "input.json"),
            "--repo-root",
            str(tmp_path),
            "--test-db",
            str(REPO_ROOT / "tests/integration/test_lists/test-db"),
            "--groovy-file",
            str(REPO_ROOT / "jenkins/L0_Test.groovy"),
        ],
    )
    assert result.exit_code == 0, result.output
    decision = json.loads(result.stdout)
    assert decision["scope"] == "noop"
    assert decision["affected_stages"] == []
    assert not decision["sanity_required"]
    assert not decision["perfsanity_required"]


def test_formatting_does_not_hide_an_unhandled_code_change(tmp_path: Path) -> None:
    pr = _formatting_pr(tmp_path)
    pr.changed_files.append("cpp/changed.cpp")
    result = Selector({}).run(pr, [PythonFormattingRule()])
    assert result.scope is None
    assert result.reasons[-1]["files"] == ["cpp/changed.cpp"]


def test_formatting_does_not_hide_a_renamed_files_old_path(tmp_path: Path) -> None:
    pr = _formatting_pr(tmp_path)
    pr.changed_files.append("tensorrt_llm/old.py")
    result = Selector({}).run(pr, [PythonFormattingRule()])
    assert result.scope is None
    assert result.reasons[-1]["files"] == ["tensorrt_llm/old.py"]


def test_rename_from_noop_path_is_not_classified_as_formatting(tmp_path: Path) -> None:
    _formatting_pr(tmp_path)
    input_path = tmp_path / "input.json"
    data = json.loads(input_path.read_text())
    old_path = "tests/microbenchmarks/old.py"
    data["changed_files"].append(old_path)
    data["diffs"][old_path] = data["diffs"]["tensorrt_llm/example.py"]
    input_path.write_text(json.dumps(data))
    pr = _load_pr_inputs(input_path, tmp_path)
    assert not pr.formatting_only_files


def _vg_rule(*, schema_tests: bool = True, cpu_stage: bool = True) -> VisualGenRule:
    index = YAMLIndex()
    index.blocks = [
        Block("l0_b200", 0, {}, ["unittest/_torch/visual_gen/test_wan.py"]),
        Block("l0_cpu", 0, {}, [SCHEMA_TEST] if schema_tests else ["unittest/tools"]),
    ]
    stages = {"B200-PyTorch-1": Stage("B200-PyTorch-1", "l0_b200", "x86_64", 1, 1)}
    if cpu_stage:
        stages["CPU-Generic-x86-1"] = Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1)
    return VisualGenRule(index, stages)


def test_schema_claim_is_exact_and_retains_validation() -> None:
    assert _is_vg_claim(SCHEMA)
    assert not _is_vg_claim(SCHEMA + ".bak")
    assert not _is_vg_claim("tensorrt_llm/schemas/trtllm-serve-config.schema.json")
    result = _vg_rule().apply(PRInputs([SCHEMA], {}))
    assert result is not None and result.scope == "visualgenonly"
    assert result.affected_stages == {"B200-PyTorch-1", "CPU-Generic-x86-1"}
    assert result.block_filters[("l0_cpu", 0)] == {SCHEMA_TEST: {SCHEMA_TEST}}


@pytest.mark.parametrize(("schema_tests", "cpu_stage"), [(False, True), (True, False)])
def test_schema_requires_runnable_validation(schema_tests: bool, cpu_stage: bool) -> None:
    result = _vg_rule(schema_tests=schema_tests, cpu_stage=cpu_stage).apply(PRInputs([SCHEMA], {}))
    assert result is not None and result.scope is None


@pytest.mark.parametrize("suffix", [" -k unrelated", " --ignore=anything", " SKIP (bug)"])
def test_schema_requires_unfiltered_validation(suffix: str) -> None:
    rule = _vg_rule()
    rule.yaml_index.blocks[1].tests[:] = [SCHEMA_TEST + suffix]
    result = rule.apply(PRInputs([SCHEMA], {}))
    assert result is not None and result.scope is None
    assert "schema validation stages unavailable" in result.reason


def test_public_api_falls_back_without_shared_validation() -> None:
    result = _vg_rule().apply(PRInputs([SCHEMA, "tensorrt_llm/visual_gen/args.py"], {}))
    assert result is not None and result.scope is None


def test_internal_visualgen_selection_is_unchanged() -> None:
    result = _vg_rule().apply(PRInputs(["tensorrt_llm/_torch/visual_gen/pipeline.py"], {}))
    assert result is not None and result.scope == "visualgenonly"
    assert result.affected_stages == {"B200-PyTorch-1"}


def _public_vg_rule() -> VisualGenRule:
    rule = _vg_rule()
    rule.yaml_index.blocks[1].tests.extend(
        [
            "unittest/api_stability",
            "unittest/llmapi/apps/test_serve_lazy_import.py",
            "unittest/others",
            "unittest/tools",
        ]
    )
    rule.yaml_index.blocks.append(
        Block(
            "l0_a10",
            0,
            {},
            [
                "test_e2e.py::test_trtllm_serve_example[pytorch] TIMEOUT (90)",
                "test_e2e.py::test_openai_chat_example[pytorch]",
                "test_e2e.py::test_openai_completions_example[pytorch]",
                "examples/serve/test_serve.py::test_config_override_precedence",
                "disaggregated/test_disaggregated.py::test_disaggregated_single_gpu",
                "llmapi/test_llm_examples.py::test_llmapi_server_example",
                "unittest/disaggregated/test_openai_server_info.py",
                "accuracy/test_llama.py",
            ],
        )
    )
    rule._stages_by_yaml["l0_a10"] = [
        ("A10-PyTorch-1", Stage("A10-PyTorch-1", "l0_a10", "x86_64", 1, 1))
    ]
    return rule


@pytest.mark.parametrize(
    "path",
    [
        "tensorrt_llm/visual_gen/args.py",
        "tensorrt_llm/visual_gen/sparse_attention.py",
        "tensorrt_llm/visual_gen/__init__.py",
        "tensorrt_llm/visual_gen/params.py",
        "tensorrt_llm/media/encoding.py",
    ],
)
def test_public_api_keeps_cpu_serving_and_sanity(path: str) -> None:
    rule = _public_vg_rule()
    result = rule.apply(PRInputs([path, SCHEMA], {}))
    assert result is not None and result.scope == "visualgenonly"
    assert result.affected_stages == {"B200-PyTorch-1", "CPU-Generic-x86-1", "A10-PyTorch-1"}
    assert result.sanity_relevant and result.perfsanity_relevant
    kept = {
        entry
        for prefixes in result.block_filters.values()
        for entries in prefixes.values()
        for entry in entries
    }
    assert set(rule.yaml_index.blocks[1].tests) <= kept
    assert set(rule.yaml_index.blocks[2].tests[:-1]) <= kept
    assert "accuracy/test_llama.py" not in kept


@pytest.mark.parametrize(
    "missing",
    [
        SCHEMA_TEST,
        "unittest/api_stability",
        "unittest/llmapi/apps/test_serve_lazy_import.py",
        "unittest/others",
        "test_e2e.py::test_openai_chat_example[pytorch]",
        "test_e2e.py::test_openai_completions_example[pytorch]",
        "examples/serve/test_serve.py::test_config_override_precedence",
    ],
)
def test_public_api_requires_every_shared_check(missing: str) -> None:
    rule = _public_vg_rule()
    for block in rule.yaml_index.blocks:
        block.tests[:] = [entry for entry in block.tests if entry != missing]
    result = rule.apply(PRInputs(["tensorrt_llm/visual_gen/args.py"], {}))
    assert result is not None and result.scope is None
    assert "required shared validation unavailable" in result.reason


@pytest.mark.parametrize("stem", ["l0_cpu", "l0_a10"])
def test_public_api_requires_runnable_validation_stages(stem: str) -> None:
    rule = _public_vg_rule()
    rule._stages_by_yaml.pop(stem)
    result = rule.apply(PRInputs(["tensorrt_llm/visual_gen/args.py"], {}))
    assert result is not None and result.scope is None


@pytest.mark.parametrize("suffix", [" -k unrelated", " --ignore=anything", " SKIP (bug)"])
def test_filtered_validation_does_not_satisfy_public_api_guard(suffix: str) -> None:
    rule = _public_vg_rule()
    block = rule.yaml_index.blocks[1]
    block.tests[:] = [entry + suffix if entry == SCHEMA_TEST else entry for entry in block.tests]
    result = rule.apply(PRInputs(["tensorrt_llm/visual_gen/args.py"], {}))
    assert result is not None and result.scope is None


def test_public_api_does_not_claim_unrelated_core_changes() -> None:
    rule = _public_vg_rule()
    result = Selector({}).run(
        PRInputs(["tensorrt_llm/visual_gen/args.py", "tensorrt_llm/llmapi/llm_args.py"], {}),
        [rule],
    )
    assert result.scope is None
    assert result.reasons[-1]["files"] == ["tensorrt_llm/llmapi/llm_args.py"]


@pytest.mark.parametrize(
    "name", ["A10-PyTorch-Post-Merge-1", "A10-OnDemand-PyTorch-1", "A10-2_GPUs-PyTorch-1"]
)
def test_public_api_requires_serving_check_in_baseline_trigger_mode(name: str) -> None:
    rule = _public_vg_rule()
    rule._stages_by_yaml["l0_a10"] = [(name, Stage(name, "l0_a10", "x86_64", 1, 1))]
    result = rule.apply(PRInputs(["tensorrt_llm/visual_gen/args.py"], {}))
    assert result is not None and result.scope is None


def test_public_api_post_merge_can_use_post_merge_validation() -> None:
    rule = _public_vg_rule()
    name = "A10-PyTorch-Post-Merge-1"
    rule._stages_by_yaml["l0_a10"] = [(name, Stage(name, "l0_a10", "x86_64", 1, 1))]
    result = rule.apply(PRInputs(["tensorrt_llm/visual_gen/args.py"], {}, post_merge=True))
    assert result is not None and result.scope == "visualgenonly"


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (
            "tensorrt_llm/_torch/modeling_v2/catalog/_op.py",
            {"default_pipeline", "fasth3", "ops", "second_consumer", "second_other"},
        ),
        (
            "tensorrt_llm/_torch/visual_gen/models/wan/ops/attention.py",
            {
                "default_pipeline",
                "fasth3",
                "ops",
                "legacy",
                "catalog",
                "unused",
                "second_consumer",
                "second_other",
            },
        ),
        (
            "tensorrt_llm/_torch/visual_gen/catalog/attention/_native.py",
            {"fasth3", "ops", "catalog", "contracts", "second_consumer"},
        ),
        (
            "tensorrt_llm/_torch/visual_gen/catalog/attention/unused.py",
            {"unused", "contracts"},
        ),
        ("tensorrt_llm/_torch/modeling_v2/catalog/attention/core.py", {"fasth3", "ops"}),
        (
            "tensorrt_llm/_torch/modeling_v2/catalog/gemm/cublas_mm.py",
            {"default_pipeline", "fasth3"},
        ),
        (
            "tensorrt_llm/_torch/visual_gen/models/minimax_h3_v2/ops/vsa.py",
            {"fasth3", "ops", "contracts"},
        ),
        (
            "tensorrt_llm/_torch/visual_gen/models/minimax_h3_v2/default_pipeline/pipeline.py",
            {"default_pipeline"},
        ),
        (
            "tensorrt_llm/_torch/visual_gen/models/minimax_h3_v2/pipeline_registry.py",
            {"default_pipeline", "fasth3", "ops"},
        ),
    ],
)
def test_visualgen_v2_selects_consumers(tmp_path, changed, expected):
    family = "minimax_h3_v2"
    unit = f"unittest/_torch/visual_gen/models/{family}"
    quality = f"visual_gen/{family}"
    files = {
        f"tensorrt_llm/_torch/visual_gen/models/{family}/pipeline_registry.py": (
            "from .default_pipeline.pipeline import Default\nfrom .fasth3.pipeline import Fast\n"
            "def route(config):\n    if config.fast:\n        return Fast\n    return Default\n"
        ),
        f"tensorrt_llm/_torch/visual_gen/models/{family}/default_pipeline/pipeline.py": (
            "from ..components.linear import project\nclass Default: pass"
        ),
        f"tensorrt_llm/_torch/visual_gen/models/{family}/fasth3/pipeline.py": (
            "from ..ops.vsa import sparse\nfrom ..components.linear import project\nclass Fast: pass\n"
        ),
        f"tensorrt_llm/_torch/visual_gen/models/{family}/components/linear.py": (
            "from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm as project"
        ),
        f"tensorrt_llm/_torch/visual_gen/models/{family}/ops/vsa.py": (
            "from tensorrt_llm._torch.modeling_v2.catalog.attention.adapter import sparse\n"
            "from tensorrt_llm._torch.visual_gen.catalog.attention.sparse import launch"
        ),
        "tensorrt_llm/_torch/visual_gen/models/second_v2/pipeline_registry.py": (
            "from .consumer.pipeline import Consumer\nfrom .other.pipeline import Other\n"
            "def route(config):\n    if config.fast:\n        return Consumer\n    return Other\n"
        ),
        "tensorrt_llm/_torch/visual_gen/models/second_v2/consumer/pipeline.py": (
            "from tensorrt_llm._torch.visual_gen.catalog.attention.sparse import launch\nclass Consumer: pass"
        ),
        "tensorrt_llm/_torch/visual_gen/catalog/attention/sparse.py": "from ._native import launch",
        "tests/unittest/_torch/visual_gen/catalog/attention/test_sparse.py": (
            "from tensorrt_llm._torch.visual_gen.catalog.attention.sparse import launch"
        ),
        "tensorrt_llm/_torch/modeling_v2/catalog/attention/adapter.py": "from .core import sparse",
        f"tests/{unit}/ops/test_vsa.py": (
            "from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.ops.vsa import sparse"
        ),
    }
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    entries = {
        "default_pipeline": {
            f"{unit}/default_pipeline/test_model.py",
            f"{quality}/default_pipeline/test_quality.py",
        },
        "fasth3": {f"{unit}/fasth3/test_pipeline.py", f"{quality}/fasth3/test_quality.py"},
        "ops": {f"{unit}/ops/test_vsa.py"},
        "legacy": {"visual_gen/test_unrelated.py", "examples/visual_gen/test_minimax_h3_e2e.py"},
        "catalog": {"unittest/_torch/visual_gen/catalog/attention/test_sparse.py"},
        "unused": {"unittest/_torch/visual_gen/catalog/attention/test_unused.py"},
        "second_consumer": {"visual_gen/second_v2/consumer/test_quality.py"},
        "second_other": {"visual_gen/second_v2/other/test_quality.py"},
        "contracts": {"unittest/_torch/modeling_v2/test_modeling_v2_contracts.py"},
    }
    index = YAMLIndex()
    index.blocks = [
        Block(
            "l0_b200",
            0,
            {},
            sorted(set.union(*(paths for key, paths in entries.items() if key != "contracts"))),
        ),
        Block("l0_cpu", 0, {}, sorted(entries["contracts"])),
    ]
    stages = {"CPU-Generic-x86-1": Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1)}
    rule = VisualGenRule(index, stages, tmp_path)
    result = rule.apply(PRInputs([changed], {}))
    assert result is not None
    selected = {
        e for block in result.block_filters.values() for tests in block.values() for e in tests
    }
    assert selected == set.union(*(entries[key] for key in expected))


def test_h3_v2_quality_changes_select_only_importing_l0_tests():
    index = YAMLIndex.load(REPO_ROOT / "tests/integration/test_lists/test-db")
    rule = CbtsTestsDefRule(index, {}, REPO_ROOT)
    family = "visual_gen/minimax_h3_v2"
    result = rule.apply(
        PRInputs([f"tests/integration/defs/{family}/minimax_h3_quality_cases.py"], {})
    )
    assert result is not None and result.scope == "testdefonly"
    selected = {path for filters in result.block_filters.values() for path in filters}
    assert selected == {
        f"{family}/default_pipeline/test_loading.py",
        f"{family}/default_pipeline/test_parallel_quality.py",
        f"{family}/default_pipeline/test_quality.py",
    }


def _visualgen_selection(changed):
    index = YAMLIndex.load(REPO_ROOT / "tests/integration/test_lists/test-db")
    stages = parse_stages_from_groovy(REPO_ROOT / "jenkins/L0_Test.groovy")
    rule = VisualGenRule(index, stages, REPO_ROOT)
    result = rule.apply(PRInputs([changed], {}))
    assert result is not None and result.scope == "visualgenonly"
    return {path for filters in result.block_filters.values() for path in filters}


@pytest.mark.parametrize(
    "op",
    [
        "block_sparse_attention",
        "cute_fmha",
        "flash_attention_4",
        "flashinfer_prefill",
        "ulysses_qkv",
    ],
)
def test_visualgen_ops_select_only_importing_families(op):
    scope = "models/minimax_h3_v2/ops" if op == "ulysses_qkv" else "catalog/attention"
    selected = _visualgen_selection(f"tensorrt_llm/_torch/visual_gen/{scope}/{op}.py")
    unit = "unittest/_torch/visual_gen/models/minimax_h3_v2"
    mirror = f"unittest/_torch/visual_gen/{scope}/test_{op}.py"
    contracts = "unittest/_torch/modeling_v2/test_modeling_v2_contracts.py"
    assert {
        mirror,
        contracts,
        f"{unit}/default_pipeline/test_minimax_h3_pipeline.py",
        f"{unit}/fasth3/test_fasth3_pipeline.py",
    } <= selected
    assert all(
        path.startswith((unit + "/", "visual_gen/minimax_h3_v2/"))
        for path in selected - {mirror, contracts}
    )
