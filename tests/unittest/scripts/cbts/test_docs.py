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

from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, Stage, YAMLIndex
from cbts.command.main import Selector, _combine_scopes
from cbts.rules.agent_flow import AGENT_FLOW_STAGE, AgentFlowRule, _is_agent_flow_claim
from cbts.rules.base import PRInputs
from cbts.rules.docs import CPU_TEST_YAML_STEM, DOCS_STAGE, DocsRule, is_docs_path
from cbts.rules.modeling_v2 import _is_mv2_claim
from cbts.rules.openengine import _is_openengine_claim
from cbts.rules.out_of_scope import is_out_of_scope
from cbts.rules.spec_dec import _is_spec_claim
from cbts.rules.tests_def import TestsDefRule as CbtsTestsDefRule
from cbts.rules.visual_gen import _is_vg_claim

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CPU_STAGE_NAMES = {"CPU-Generic-x86-1", "CPU-Generic-arm-1"}


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
