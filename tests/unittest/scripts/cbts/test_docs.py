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
"""Tests for CBTS DocsRule: routing documentation changes to docs and CPU validation."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
import cbts.command.main as cbts_main
from cbts.blocks import Block, Stage, YAMLIndex
from cbts.command.main import Selector, _combine_scopes
from cbts.coverage.tier import coverage_preflight
from cbts.python_docstring_analysis import split_docstring_diff
from cbts.rules._helpers import (
    iter_diff_deleted_post_lines,
    iter_diff_post_line_numbers,
    reconstruct_diff_pre_image,
)
from cbts.rules.agent_flow import AGENT_FLOW_STAGE, AgentFlowRule, _is_agent_flow_claim
from cbts.rules.base import PRInputs
from cbts.rules.docs import CPU_TEST_YAML_STEM, DOCS_STAGE, DocsRule, is_docs_path
from cbts.rules.modeling_v2 import _entry_is_mv2, _is_mv2_claim
from cbts.rules.openengine import _is_openengine_claim
from cbts.rules.out_of_scope import OutOfScopeRule, is_out_of_scope
from cbts.rules.spec_dec import _is_spec_claim
from cbts.rules.tests_def import TestsDefRule as CbtsTestsDefRule
from cbts.rules.visual_gen import _is_vg_claim

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CPU_STAGE_NAMES = set(
    re.findall(
        r'^\s*"(CPU-Generic-[^"]+)":\s*\["cpu",',
        (_REPO_ROOT / "jenkins/L0_Test.groovy").read_text(),
        re.MULTILINE,
    )
)


def _docs_rule() -> DocsRule:
    yaml_index = YAMLIndex()
    yaml_index.blocks = [
        Block(
            yaml_stem=CPU_TEST_YAML_STEM,
            block_index=0,
            condition={},
            tests=["unittest/tools", "unittest/usage -k telemetry"],
        ),
        Block(
            yaml_stem="l0_h100",
            block_index=0,
            condition={},
            tests=["unittest/_torch"],
        ),
    ]
    stages = {
        name: Stage(
            name=name,
            yaml_stem=CPU_TEST_YAML_STEM,
            cpu_arch="x86_64" if "x86" in name else "aarch64",
            split_id=1,
            total_splits=1,
        )
        for name in _CPU_STAGE_NAMES
    }
    return DocsRule(yaml_index, stages)


@pytest.mark.parametrize(
    "path",
    (
        "docs/source/index.rst",
        "docs/source/conf.py",
        "docs/source/_static/diagram.png",
        "README.md",
        "examples/guide.rst",
    ),
)
def test_docs_rule_claims_documentation_paths(path: str) -> None:
    assert is_docs_path(path)


@pytest.mark.parametrize(
    "path",
    (
        ".github/CODEOWNERS",
        ".github/workflows/docs.yml",
        "examples/config.yaml",
        "security_scanning/metadata.json",
    ),
)
def test_docs_rule_rejects_non_documentation_paths(path: str) -> None:
    assert not is_docs_path(path)


def test_docs_rule_routes_changes_to_docs_and_complete_cpu_suite() -> None:
    rule = _docs_rule()
    result = rule.apply(
        PRInputs(
            changed_files=["docs/source/conf.py", "README.md", ".github/CODEOWNERS"],
            diffs={},
        )
    )

    assert result is not None
    assert result.handled_files == {"docs/source/conf.py", "README.md"}
    assert result.affected_stages == _CPU_STAGE_NAMES | {DOCS_STAGE}
    assert result.scope == "docsonly"
    assert result.block_filters == {
        (CPU_TEST_YAML_STEM, 0): {
            "unittest/tools": {"unittest/tools"},
            "unittest/usage": {"unittest/usage -k telemetry"},
        }
    }
    assert not result.sanity_relevant
    assert not result.perfsanity_relevant


def test_docs_rule_falls_back_when_cpu_suite_cannot_be_resolved() -> None:
    result = DocsRule(YAMLIndex(), {}).apply(
        PRInputs(changed_files=["docs/source/index.rst"], diffs={})
    )

    assert result is not None
    assert result.scope is None
    assert not result.affected_stages


def test_docs_rule_routes_non_docs_markdown_to_docs_build_only() -> None:
    result = DocsRule(YAMLIndex(), {}).apply(
        PRInputs(changed_files=["README.md", "examples/eagle/guide.rst"], diffs={})
    )

    assert result is not None
    assert result.handled_files == {"README.md", "examples/eagle/guide.rst"}
    assert result.affected_stages == {DOCS_STAGE}
    assert result.scope == "docsonly"
    assert not result.block_filters


def test_markdown_is_not_out_of_scope() -> None:
    assert not is_out_of_scope("README.md")
    assert not is_out_of_scope("docs/source/index.rst")


def test_codeowners_is_exact_path_noop() -> None:
    assert is_out_of_scope(".github/CODEOWNERS")
    assert not is_out_of_scope(".github/workflows/docs.yml")
    assert not is_out_of_scope("nested/.github/CODEOWNERS")


def test_documentation_does_not_trigger_backend_rules() -> None:
    assert not _is_agent_flow_claim("agent-flow/guide.rst")
    assert not _is_mv2_claim("tensorrt_llm/_torch/_experimental/modeling_v2/guide.rst")
    assert not _is_openengine_claim("tensorrt_llm/grpc/openengine/guide.rst")
    assert not _is_spec_claim("examples/eagle/guide.rst")
    assert not _is_vg_claim("examples/visual_gen/guide.rst")


def test_modeling_v2_entries_cover_unit_and_accuracy_tests() -> None:
    assert _entry_is_mv2("unittest/_torch/modeling_v2 --ignore=unittest/_torch/modeling_v2/comm")
    assert _entry_is_mv2("unittest/_torch/modeling_v2/comm")
    assert _entry_is_mv2(
        "accuracy/modeling_v2/test_qwen4_exp.py::TestModelingV2Qwen38FlashNextNvfp4Sm103Tp1::test_gsm8k"
    )
    assert _entry_is_mv2("accuracy/modeling_v2/test_gpt_oss.py::TestModelingV2GptOss120bSm103Tp1")
    assert not _entry_is_mv2("accuracy/test_llm_api_pytorch.py::TestQwen3_8B::test_bf16")


def test_documentation_under_tests_is_left_to_docs_rule() -> None:
    rule = CbtsTestsDefRule(YAMLIndex(), {}, repo_root=_REPO_ROOT)

    assert rule.apply(PRInputs(changed_files=["tests/unittest/README.md"], diffs={})) is None


def test_docs_rule_combines_with_other_targeted_rules() -> None:
    rules = [_docs_rule(), AgentFlowRule(YAMLIndex(), {})]
    result = Selector({}).run(
        PRInputs(changed_files=["README.md", "agent-flow/agent_flow/cli.py"], diffs={}),
        rules,
    )

    assert result.scope == "testsonly"
    assert result.affected_stages == {DOCS_STAGE, AGENT_FLOW_STAGE}
    assert _combine_scopes(["docsonly", "noop"]) == "docsonly"


def test_docs_stage_matches_jenkins_stage_key() -> None:
    groovy = (_REPO_ROOT / "jenkins/L0_Test.groovy").read_text()
    assert f'"{DOCS_STAGE}": [docBuildSpec, {{' in groovy


@pytest.mark.parametrize(
    "path",
    (
        "tests/integration/test_lists/qa/.gitignore",
        "tests/unittest/new/nested/.gitignore",
    ),
)
def test_gitignore_metadata_under_tests_adds_no_testdef_contribution(path: str) -> None:
    rule = CbtsTestsDefRule(YAMLIndex(), {}, repo_root=_REPO_ROOT)

    result = Selector({}).run(
        PRInputs(changed_files=[path], diffs={}), [rule, _out_of_scope_rule(_REPO_ROOT)]
    )

    assert result.scope == "noop"
    assert not result.affected_stages
    assert not result.block_filters


def _docstring_pr(tmp_path: Path, before: str, source: str, path: str) -> PRInputs:
    source_path = tmp_path / path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source)
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(True), source.splitlines(True), fromfile=path, tofile=path
        )
    )
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"changed_files": [path], "diffs": {path: diff}}))
    return cbts_main._load_pr_inputs(input_path, tmp_path)


@pytest.mark.parametrize(
    "path",
    (
        ".github/ISSUE_TEMPLATE/guide.md",
        ".codex/skills/new/README.rst",
        ".claude/skills/new/README.md",
    ),
)
def test_noop_metadata_documentation_adds_no_docs_contribution(path: str) -> None:
    pr = PRInputs(changed_files=[path], diffs={})

    result = Selector({}).run(pr, [_docs_rule(), _out_of_scope_rule(_REPO_ROOT)])

    assert not is_docs_path(path)
    assert _docs_rule().apply(pr) is None
    assert result.scope == "noop"
    assert not result.affected_stages
    assert not result.block_filters


@pytest.mark.parametrize(
    "path",
    (
        ".github/ISSUE_TEMPLATE/helper.py",
        ".codex/scripts/helper.py",
        ".claude/scripts/helper.py",
    ),
)
def test_noop_metadata_docstring_changes_stay_available_to_noop_rule(
    tmp_path: Path, path: str
) -> None:
    pr = _docstring_pr(
        tmp_path,
        'def helper():\n    "old docs"\n    return 1\n',
        'def helper():\n    "new docs"\n    return 1\n',
        path,
    )

    result = Selector({}).run(pr, [_docs_rule(), _out_of_scope_rule(tmp_path)])

    assert not pr.docstring_files
    assert not pr.docstring_only_files
    assert result.scope == "noop"
    assert not result.affected_stages
    assert not result.block_filters


@pytest.mark.parametrize(
    ("before", "source"),
    (
        ('"old docs"\nVALUE = 1\n', '"new docs"\nVALUE = 1\n'),
        (
            'class Example:\n    "old docs"\n    pass\n',
            'class Example:\n    "new docs"\n    pass\n',
        ),
        (
            'def helper():\n    "old docs"\n    return 1\n',
            'def helper():\n    "new docs"\n    return 1\n',
        ),
        (
            'async def helper():\n    "old docs"\n    return 1\n',
            'async def helper():\n    "new docs"\n    return 1\n',
        ),
        (
            'def outer():\n    def inner():\n        "old docs"\n        return 1\n    return inner\n',
            'def outer():\n    def inner():\n        "new docs"\n        return 1\n    return inner\n',
        ),
        ("def helper():\n    return 1\n", 'def helper():\n    "new docs"\n    return 1\n'),
        ('def helper():\n    "old docs"\n    return 1\n', "def helper():\n    return 1\n"),
        (
            '"""old docs\n# old example\n"""\nVALUE = 1\n',
            '"""new docs\n\n# new example\nmore docs\n"""\nVALUE = 1\n',
        ),
    ),
)
def test_docstring_changes_reuse_docs_and_cpu_rule(
    tmp_path: Path, before: str, source: str
) -> None:
    path = "tensorrt_llm/example.py"
    pr = _docstring_pr(tmp_path, before, source, path)

    result = Selector({}).run(pr, [_docs_rule()])

    assert pr.docstring_files == pr.docstring_only_files == {path}
    assert not iter_diff_post_line_numbers(pr.diffs[path])
    assert reconstruct_diff_pre_image(source, pr.raw_diffs[path]) == before
    assert result.scope == "docsonly"
    assert result.affected_stages == _CPU_STAGE_NAMES | {DOCS_STAGE}
    assert not result.sanity_required
    assert not result.perfsanity_required


@pytest.mark.parametrize("code_changed", (False, True))
def test_docstring_rule_combines_with_path_rule_in_same_file(
    tmp_path: Path, code_changed: bool
) -> None:
    path = "agent-flow/agent_flow/cli.py"
    before = 'def helper():\n    "old docs"\n    return 1\n'
    source = f'def helper():\n    "new docs"\n    return {2 if code_changed else 1}\n'
    pr = _docstring_pr(tmp_path, before, source, path)

    result = Selector({}).run(pr, [AgentFlowRule(YAMLIndex(), {}), _docs_rule()])

    assert result.scope == ("testsonly" if code_changed else "docsonly")
    assert result.affected_stages == _CPU_STAGE_NAMES | {DOCS_STAGE} | (
        {AGENT_FLOW_STAGE} if code_changed else set()
    )


def test_docstring_rule_combines_with_code_change_in_another_file(tmp_path: Path) -> None:
    pr = _docstring_pr(tmp_path, '"old docs"\n', '"new docs"\n', "tensorrt_llm/example.py")
    pr.changed_files.append("agent-flow/agent_flow/cli.py")

    result = Selector({}).run(pr, [_docs_rule(), AgentFlowRule(YAMLIndex(), {})])

    assert result.scope == "testsonly"
    assert result.affected_stages == _CPU_STAGE_NAMES | {DOCS_STAGE, AGENT_FLOW_STAGE}


@pytest.mark.parametrize(
    ("before", "source"),
    (
        ('VALUE = "old text"\n', 'VALUE = "new text"\n'),
        ('VALUE = 1\n"old text"\n', 'VALUE = 1\n"new text"\n'),
        ('f"{old()}"\n', 'f"{new()}"\n'),
        ('"old docs"; VALUE = 1\n', '"new docs"; VALUE = 2\n'),
        ('def helper(): "old docs"\n', 'def helper(): "new docs"\n'),
        ("VALUE = 1\n", 'VALUE = 1\ndef helper():\n    "new docs"\n    return 1\n'),
        ('"old docs"\n', '"new docs"\ndef helper():\n    return 1\n'),
        (
            'def helper():\n    "old docs"\ndef helper():\n    pass\n',
            'def helper():\n    "new docs"\ndef helper():\n    pass\n',
        ),
        ('"old docs"\n', '"new docs"\nif:\n'),
    ),
)
def test_docstring_analysis_preserves_executable_or_ambiguous_changes(
    tmp_path: Path, before: str, source: str
) -> None:
    path = "tensorrt_llm/example.py"
    pr = _docstring_pr(tmp_path, before, source, path)

    assert path not in pr.docstring_only_files
    assert iter_diff_post_line_numbers(pr.diffs[path])
    result = Selector({}).run(pr, [_docs_rule()])
    assert result.scope is None


@pytest.mark.parametrize("diff", (None, "", '@@ -1 +1 @@\n-"old docs"\n+"different docs"\n'))
def test_docstring_analysis_keeps_missing_or_mismatched_patch_unknown(
    tmp_path: Path, diff: str | None
) -> None:
    path = "tensorrt_llm/example.py"
    (tmp_path / "tensorrt_llm").mkdir()
    (tmp_path / path).write_text('"new docs"\n')
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"changed_files": [path], "diffs": {path: diff}}))

    pr = cbts_main._load_pr_inputs(input_path, tmp_path)

    assert not pr.docstring_files
    assert not pr.docstring_only_files
    assert Selector({}).run(pr, [_docs_rule()]).scope is None


def test_docstring_split_preserves_code_line_numbers_after_multiline_addition() -> None:
    before = "def helper():\n    return 1\n"
    source = 'def helper():\n    """new docs\n\nmore docs\n    """\n    return 2\n'
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))

    executable, changed = split_docstring_diff(source, diff)

    assert changed
    assert iter_diff_post_line_numbers(executable) == {6}
    assert iter_diff_deleted_post_lines(executable) == {6: ["    return 1"]}


@pytest.mark.parametrize(
    ("old_underline", "new_underline"), (("-----", "====="), ("=====", "+++++"))
)
@pytest.mark.parametrize("code_changed", (False, True))
def test_docstring_underlines_preserve_executable_changes(
    tmp_path: Path, old_underline: str, new_underline: str, code_changed: bool
) -> None:
    path = "tensorrt_llm/example.py"
    before = f'"""Title\n{old_underline}\n"""\nimport os\n\ndef f():\n    return 1\n'
    source = (
        f'"""Title\n{new_underline}\n"""\n'
        + ("" if code_changed else "import os\n")
        + f"\ndef f():\n    return {2 if code_changed else 1}\n"
    )

    pr = _docstring_pr(tmp_path, before, source, path)

    assert pr.docstring_files == {path}
    assert reconstruct_diff_pre_image(source, pr.raw_diffs[path]) == before
    assert pr.diffs[path].startswith(f"--- {path}\n+++ {path}\n@@")
    if code_changed:
        assert not pr.docstring_only_files
        assert "-import os\n" in pr.diffs[path]
        assert iter_diff_post_line_numbers(pr.diffs[path]) == {4, 6}
        assert iter_diff_deleted_post_lines(pr.diffs[path]) == {
            4: ["import os"],
            6: ["    return 1"],
        }
    else:
        assert pr.docstring_only_files == {path}
        assert not iter_diff_post_line_numbers(pr.diffs[path])


def test_docs_rule_falls_back_when_docstring_cpu_checks_are_unavailable(tmp_path: Path) -> None:
    pr = _docstring_pr(tmp_path, '"old docs"\n', '"new docs"\n', "tensorrt_llm/example.py")

    result = DocsRule(YAMLIndex(), {}).apply(pr)

    assert result is not None
    assert result.scope is None


def _out_of_scope_rule(repo_root: Path) -> OutOfScopeRule:
    return OutOfScopeRule(YAMLIndex(), {}, repo_root=repo_root)


def test_out_of_scope_rule_removes_ruff_config_preflight_blockers(tmp_path: Path) -> None:
    precommit = (
        "legacy-files: &legacy_files '^(legacy.py)$'\n"
        "repos:\n- repo: https://github.com/astral-sh/ruff-pre-commit\n  rev: v0.9.4\n"
        "  hooks:\n  - id: ruff\n"
        "    args: [--fix, --exit-non-zero-on-fix, --exclude=generated]\n"
        "    exclude: *legacy_files\n"
        "  - id: ruff-format\n    args: [--exclude=generated]\n    exclude: *legacy_files\n"
    )
    toml = (
        '[build-system]\nrequires = ["setuptools"]\n'
        '[tool.ruff]\nexclude = [\n    "vendor/**",\n]\n'
        '[tool.pytest.ini_options]\naddopts = "-q"\n'
    )
    changes = {
        ".pre-commit-config.yaml": (
            precommit,
            precommit.replace(", --exclude=generated", "").replace(
                "    args: [--exclude=generated]\n", ""
            ),
        ),
        "pyproject.toml": (
            toml,
            toml.replace('    "vendor/**",\n', '    "vendor/**",\n    "generated/**",\n'),
        ),
    }
    core_path = "tensorrt_llm/example.py"
    pr = PRInputs(changed_files=[*changes, core_path], diffs={})
    for path, (before, source) in changes.items():
        single = _docstring_pr(tmp_path, before, source, path)
        pr.diffs.update(single.diffs)
        pr.raw_diffs.update(single.raw_diffs)
    rule = _out_of_scope_rule(tmp_path)

    result = rule.apply(pr)

    assert result is not None
    assert result.handled_files == set(changes)
    assert result.scope == "noop"
    assert result.affected_stages == set()
    assert not result.block_filters
    residual, note = coverage_preflight(pr, [(rule, result)], result.handled_files)
    assert residual == [core_path]
    assert not note


@pytest.mark.parametrize(
    ("before", "source"),
    (
        ('[tool.ruff]\nexclude = ["old"]\n', '[tool.ruff]\nexclude = ["new"]\n'),
        ("[tool.ruff.format]\nline-length = 80\n", "[tool.ruff.format]\nline-length = 100\n"),
        ("[tool.isort]\nline_length = 80\n", "[tool.isort]\nline_length = 100\n"),
        ("[tool.yapf]\ncolumn_limit = 80\n", "[tool.yapf]\ncolumn_limit = 100\n"),
        ('[tool.codespell]\nskip = "old"\n', '[tool.codespell]\nskip = "new"\n'),
        (
            "[tool.autoflake]\nremove_unused_variables = false\n",
            "[tool.autoflake]\nremove_unused_variables = true\n",
        ),
        ('[tool.mypy]\nfiles = ["old"]\n', '[tool.mypy]\nfiles = ["new"]\n'),
        (
            '[[tool.mypy.overrides]]\nmodule = "old"\n',
            '[[tool.mypy.overrides]]\nmodule = "new"\n',
        ),
    ),
)
def test_out_of_scope_rule_claims_static_check_tables_as_noop(
    tmp_path: Path, before: str, source: str
) -> None:
    pr = _docstring_pr(tmp_path, before, source, "pyproject.toml")

    result = Selector({}).run(pr, [_out_of_scope_rule(tmp_path)])

    assert result.scope == "noop"
    assert result.affected_stages == set()
    assert not result.sanity_required
    assert not result.perfsanity_required


@pytest.mark.parametrize(
    ("before", "source"),
    (
        ('[build-system]\nrequires = ["old"]\n', '[build-system]\nrequires = ["new"]\n'),
        ('[project]\ndependencies = ["old"]\n', '[project]\ndependencies = ["new"]\n'),
        (
            '[tool.pytest.ini_options]\naddopts = "-q"\n',
            '[tool.pytest.ini_options]\naddopts = "-x"\n',
        ),
        ("[tool.ruff]\nexclude = []\n", "[tool.pytest.ini_options]\nexclude = []\n"),
        (
            "[build-system]\nrequires = []\n[tool.ruff]\nfix = false\n",
            '[build-system]\nrequires = ["new"]\n[tool.ruff]\nfix = true\n',
        ),
        ('description = """old\n[tool.ruff]\n"""\n', 'description = """new\n[tool.ruff]\n"""\n'),
        ('["tool".ruff]\nfix = false\n', '["tool".ruff]\nfix = true\n'),
        ("", "[tool.ruff]\nfix = true\n"),
    ),
)
def test_out_of_scope_rule_preserves_non_lint_and_ambiguous_toml_impact(
    tmp_path: Path, before: str, source: str
) -> None:
    pr = _docstring_pr(tmp_path, before, source, "pyproject.toml")

    assert _out_of_scope_rule(tmp_path).apply(pr) is None
    assert Selector({}).run(pr, [_out_of_scope_rule(tmp_path)]).scope is None


@pytest.mark.parametrize(
    "path",
    (
        ".gitignore",
        "agent-flow/.gitignore",
        "cpp/kernels/fmha_v2/.gitignore",
        "cpp/tests/resources/.gitignore",
        "scripts/attribution/sbom/.gitignore",
        "new/nested/.gitignore",
        ".github/ISSUE_TEMPLATE/01-installation.yml",
        ".github/ISSUE_TEMPLATE/02-new-model.yml",
        ".github/ISSUE_TEMPLATE/03-documentation.yml",
        ".github/ISSUE_TEMPLATE/04-questions.yml",
        ".github/ISSUE_TEMPLATE/05-feature-request.yml",
        ".github/ISSUE_TEMPLATE/06-bug-report.yml",
        ".github/ISSUE_TEMPLATE/07-performance-discussion.yml",
        ".github/ISSUE_TEMPLATE/08-RFC.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/ISSUE_TEMPLATE/new-form.yml",
        ".github/ISSUE_TEMPLATE/nested/helper.json",
        ".codex/config.toml",
        ".codex/agents/onboard_update_reviewer.toml",
        ".codex/new-config.toml",
        ".codex/scripts/new-helper.py",
        ".claude/agent-tests/perf-test-sync/promptfooconfig.yaml",
        ".claude/skills/perf-optimization-casebook/data/aliases.yaml",
        ".claude/skills/perf-optimization-casebook/data/patterns.yaml",
        ".claude/skills/perf-optimization-casebook/data/tags.yaml",
        ".claude/skills/trtllm-test-specialist/references/agg_config_template.yaml",
        ".claude/skills/trtllm-test-specialist/references/benchmark_config_template.yaml",
        ".claude/skills/trtllm-test-specialist/references/disagg_config_template.yaml",
        ".claude/skills/trtllm-test-specialist/references/smoke_test_config_template.yml",
        ".claude/skills/trtllm-test-specialist/references/test_config_template.yaml",
        ".claude/new-config.yaml",
        ".claude/scripts/new-helper.py",
        "LICENSE",
        ".pre-commit-config.yaml",
        "agent-flow/.pre-commit-config.yaml",
        ".clang-format",
        "cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/.clang-format",
        ".clang-tidy",
        ".clangd",
        ".cmake-format.json",
        ".editorconfig",
        ".cursorignore",
        ".coderabbit.yaml",
        ".nspect-allowlist.toml",
        "scripts/bandit.yaml",
        "jenkins/license_cpp.json",
        "legacy-files.txt",
        "ruff-legacy.toml",
        "ruff-legacy-baseline.json",
        "scripts/templates/precommit-legacy-files.j2",
        "scripts/templates/pyproject-format-exclude.j2",
        "scripts/templates/ruff-legacy.toml.j2",
    ),
)
def test_out_of_scope_rule_combines_dedicated_configs_with_core_changes(
    tmp_path: Path, path: str
) -> None:
    core_path = "tensorrt_llm/example.py"
    pr = PRInputs(changed_files=[path, core_path], diffs={})
    rule = _out_of_scope_rule(tmp_path)

    result = rule.apply(pr)

    assert result is not None
    assert result.handled_files == {path}
    assert result.scope == "noop"
    assert not result.block_filters
    residual, note = coverage_preflight(pr, [(rule, result)], result.handled_files)
    assert residual == [core_path]
    assert not note


def test_out_of_scope_rule_handles_precommit_revisions_and_global_settings(tmp_path: Path) -> None:
    before = (
        "exclude: old\nrepos:\n- repo: https://github.com/example/checks\n"
        "  rev: old\n  hooks: [{id: check}]\n"
    )
    pr = _docstring_pr(tmp_path, before, before.replace("old", "new"), ".pre-commit-config.yaml")

    result = Selector({}).run(pr, [_out_of_scope_rule(tmp_path)])

    assert result.scope == "noop"
    assert result.affected_stages == set()


def test_out_of_scope_rule_leaves_changed_local_hook_scripts_as_residual(tmp_path: Path) -> None:
    before = "repos:\n- repo: local\n  hooks:\n  - id: check\n    entry: python check.py\n"
    source = before.replace("check.py", "new_check.py")
    pr = _docstring_pr(tmp_path, before, source, ".pre-commit-config.yaml")
    pr.changed_files += ["scripts/new_check.py"]

    result = Selector({}).run(pr, [_out_of_scope_rule(tmp_path)])

    assert result.scope is None
    assert result.reasons[-1]["files"] == ["scripts/new_check.py"]


@pytest.mark.parametrize("present", (True, False))
def test_out_of_scope_rule_handles_added_and_deleted_dedicated_configs(
    tmp_path: Path, present: bool
) -> None:
    path = "cpp/new_project/.clang-format"
    if present:
        config = tmp_path / path
        config.parent.mkdir(parents=True)
        config.write_text("IndentWidth: 4\n")
    pr = PRInputs(changed_files=[path], diffs={})

    result = Selector({}).run(pr, [_out_of_scope_rule(tmp_path)])

    assert result.scope == "noop"
    assert result.affected_stages == set()


@pytest.mark.parametrize(
    "path",
    (
        ".dockerignore",
        ".gitattributes",
        ".devcontainer/devcontainer.env",
        ".devcontainer/devcontainer.json",
        ".devcontainer/docker-compose.yml",
        ".devcontainer/docker-compose.override-example.yml",
        "tests/unittest/pytest.ini",
        "tests/integration/defs/pytest.ini",
        "cpp/CMakeLists.txt",
        "requirements.txt",
        ".coveragerc",
        "cpp/templates/build.j2",
    ),
)
def test_out_of_scope_rule_leaves_build_runtime_and_test_execution_configs_unclaimed(
    tmp_path: Path, path: str
) -> None:
    pr = PRInputs(changed_files=[path], diffs={})

    assert _out_of_scope_rule(tmp_path).apply(pr) is None


@pytest.mark.parametrize(
    "path", ("agent-flow/pyproject.toml", "security_scanning/pyproject.toml", "new/pyproject.toml")
)
def test_out_of_scope_rule_checks_sections_of_nested_pyprojects(tmp_path: Path, path: str) -> None:
    before = '[project]\ndependencies = ["old"]\n[tool.ruff]\nline-length = 80\n'
    pr = _docstring_pr(tmp_path, before, before.replace("80", "100"), path)

    result = Selector({}).run(pr, [_out_of_scope_rule(tmp_path)])

    assert result.scope == "noop"
    pr = _docstring_pr(tmp_path, before, before.replace('"old"', '"new"'), path)
    assert _out_of_scope_rule(tmp_path).apply(pr) is None


@pytest.mark.parametrize("path", ("agent-flow/.pre-commit-config.yaml", "agent-flow/.gitignore"))
def test_out_of_scope_rule_does_not_trigger_agent_flow_for_tool_metadata(
    tmp_path: Path, path: str
) -> None:
    pr = PRInputs(changed_files=[path], diffs={})

    result = Selector({}).run(pr, [_out_of_scope_rule(tmp_path), AgentFlowRule(YAMLIndex(), {})])

    assert result.scope == "noop"
    assert not result.affected_stages


def test_out_of_scope_rule_combines_with_other_rules(tmp_path: Path) -> None:
    pr = _docstring_pr(
        tmp_path, "[tool.ruff]\nfix = false\n", "[tool.ruff]\nfix = true\n", "pyproject.toml"
    )
    pr.changed_files += ["README.md", "agent-flow/agent_flow/cli.py"]

    result = Selector({}).run(
        pr, [_out_of_scope_rule(tmp_path), _docs_rule(), AgentFlowRule(YAMLIndex(), {})]
    )

    assert result.scope == "testsonly"
    assert result.affected_stages == {DOCS_STAGE, AGENT_FLOW_STAGE}


def test_out_of_scope_rule_requires_verified_patch_for_shared_configs(tmp_path: Path) -> None:
    pr = _docstring_pr(
        tmp_path, "[tool.ruff]\nfix = false\n", "[tool.ruff]\nfix = true\n", "pyproject.toml"
    )
    result = _out_of_scope_rule(tmp_path).apply(pr)
    assert result is not None and result.scope == "noop"
    pr.raw_diffs = pr.diffs = {}
    assert _out_of_scope_rule(tmp_path).apply(pr) is None
    pr = PRInputs(changed_files=[".clang-format"], diffs={})
    result = OutOfScopeRule(YAMLIndex(), {}).apply(pr)
    assert result is not None and result.scope == "noop"


@pytest.mark.parametrize(
    "path",
    (
        "cpp/.clang-format",
        "cpp/.clang-tidy",
        "cpp/.clangd",
        "cpp/new/deep/directory/.clang-format",
        "cpp/new/deep/directory/.clang-tidy",
        "cpp/new/deep/directory/.clangd",
    ),
)
def test_noop_directory_filename_policy_handles_clang_configs(path: str) -> None:
    result = OutOfScopeRule(YAMLIndex(), {}).apply(PRInputs(changed_files=[path], diffs={}))

    assert result is not None
    assert result.handled_files == {path}
    assert result.scope == "noop"
    assert not result.affected_stages
    assert not result.block_filters


@pytest.mark.parametrize(
    "path",
    (
        "new-check.yaml",
        "new-check.yml",
        "new-check.json",
        "new-check.toml",
        "new-check.yaml.bak",
        "backup.pre-commit-config.yaml",
        "pyproject.toml",
        "scripts/new-check.py",
        "scripts/new-check.yaml",
        "scripts/new-check.yml",
        "scripts/nested/new-check.yaml",
        "scripts-extra/new-check.yaml",
        "scripts/templates/new-lint-config.j2",
        "scripts/templates/build.j2",
        "scripts/templates/ruff-legacy.toml.j2.backup",
        "jenkins/new-license-check.json",
        "jenkins/nested/new-check.json",
        "jenkins/new-check.groovy",
        ".github/workflows/new-check.yaml",
        "agent-flow/new-check.yaml",
        "agent-flow/agent_flow/new-config.yaml",
        "new/.pre-commit-config.yaml",
        "new/fixture.gitignore",
        "new/.gitignore.bak",
        "new/.gitignore-template",
        ".github/ISSUE_TEMPLATE-extra/new-form.yml",
        "nested/.github/ISSUE_TEMPLATE/new-form.yml",
        ".github/workflows/module-owners.json",
        ".claude-extra/new-config.yaml",
        ".codex-extra/new-config.toml",
        "nested/.claude/new-config.yaml",
        "nested/.codex/new-config.toml",
        ".devcontainer/new-config.json",
        ".devcontainer/make_env.py",
        "jenkins/current_image_tags.properties",
        "cpp/new/deep/directory/config.json",
        "cpp/new/deep/directory/runtime.clangd",
        "cpp/new/deep/directory/runtime.clang-format",
        "cpp/new/deep/directory/runtime.clang-tidy",
        "cpp/new/deep/directory/.clang-format.yaml",
        "cpp-extra/.clang-tidy",
        "tensorrt_llm/new-config.yaml",
        "examples/configs/new-config.yaml",
        "tests/new-fixture.json",
    ),
)
def test_noop_file_policy_preserves_unreviewed_product_and_execution_changes(path: str) -> None:
    assert not is_out_of_scope(path)
    assert OutOfScopeRule(YAMLIndex(), {}).apply(PRInputs(changed_files=[path], diffs={})) is None


def test_unreviewed_agent_flow_config_keeps_its_backend_tests(tmp_path: Path) -> None:
    pr = PRInputs(changed_files=["agent-flow/new-check.yaml"], diffs={})

    result = Selector({}).run(pr, [_out_of_scope_rule(tmp_path), AgentFlowRule(YAMLIndex(), {})])

    assert result.scope == "agentflowonly"
    assert result.affected_stages == {AGENT_FLOW_STAGE}


def test_noop_entrypoint_keeps_cpu_baseline(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"changed_files": [".pre-commit-config.yaml"], "diffs": {}}))

    result = CliRunner().invoke(cbts_main.main, [str(input_path), "--repo-root", str(_REPO_ROOT)])

    assert result.exit_code == 0, result.output
    decision = json.loads(result.stdout)
    assert decision["scope"] == "noop"
    assert set(decision["affected_stages"]) == _CPU_STAGE_NAMES
    assert decision["test_db_dir_override"] is None
    assert not decision["sanity_required"]
    assert not decision["perfsanity_required"]
