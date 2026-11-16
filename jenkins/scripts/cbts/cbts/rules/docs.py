# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DocsRule — routes documentation changes to docs and CPU validation."""

from __future__ import annotations

from typing import Optional

from cbts.blocks import Stage, YAMLIndex, _entry_target, _target_in_filter_subtree

from ._helpers import resolve_affected_stages, stages_by_yaml_stem
from .base import PRInputs, Rule, RuleResult
from .out_of_scope import is_out_of_scope

# Stage key declared in jenkins/L0_Test.groovy::docBuildConfigs. This stage is
# not test-db-driven, so CBTS contributes its name literally.
DOCS_STAGE = "CPU-Build_Docs"
CPU_TEST_YAML_STEM = "l0_cpu"

_DOCS_PREFIX = "docs/"
_DOCS_SUFFIXES: tuple[str, ...] = (".md", ".rst")

# Documentation helpers/extensions, API documentation contracts, and tests that
# read documentation as data. Parent YAML entries retain their complete suite.
_DOCS_CPU_TEST_PATHS = (
    "unittest/docs",
    "unittest/api_stability",
    "unittest/tools",
    "unittest/usage",
    "unittest/llmapi/test_trtllm_serve_config_schemas.py",
    "unittest/_torch/pyexecutor/test_model_loader_mx.py",
)


def is_docs_path(path: str) -> bool:
    """Return whether ``path`` contributes to the repository documentation."""
    return not is_out_of_scope(path) and (
        path.startswith(_DOCS_PREFIX) or path.endswith(_DOCS_SUFFIXES)
    )


class DocsRule(Rule):
    name = "docs"
    needs_diff_for: tuple[str, ...] = ("**/*.py",)
    handles_docstrings = True

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)

    def _cpu_block_filters(self) -> dict[tuple[str, int], dict[str, set[str]]]:
        """Select registered documentation consumers, preserving YAML options."""
        filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        for block in self.yaml_index.blocks:
            if block.yaml_stem != CPU_TEST_YAML_STEM:
                continue
            entries: dict[str, set[str]] = {}
            for raw in block.tests:
                target = _entry_target(raw)
                if any(
                    _target_in_filter_subtree(target, path)
                    or _target_in_filter_subtree(path, target)
                    for path in _DOCS_CPU_TEST_PATHS
                ):
                    entries.setdefault(target, set()).add(raw)
            if entries:
                filters[(block.yaml_stem, block.block_index)] = entries
        return filters

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        docstring_files = {path for path in pr.docstring_files if not is_out_of_scope(path)}
        claimed = {path for path in pr.changed_files if is_docs_path(path)} | {
            path for path in pr.docstring_only_files if not is_out_of_scope(path)
        }
        if not claimed and not docstring_files:
            return None

        if not docstring_files and not any(path.startswith(_DOCS_PREFIX) for path in claimed):
            return RuleResult(
                handled_files=claimed,
                affected_stages={DOCS_STAGE},
                scope="docsonly",
                sanity_relevant=False,
                perfsanity_relevant=False,
                reason=(
                    f"docs: {len(claimed)} Markdown/reStructuredText file(s) outside "
                    f"{_DOCS_PREFIX} → {DOCS_STAGE}"
                ),
            )

        block_filters = self._cpu_block_filters()
        cpu_stages = resolve_affected_stages(block_filters, self.yaml_index, self._stages_by_yaml)
        if not block_filters or not cpu_stages:
            return RuleResult(
                handled_files=claimed,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"docs: {len(claimed | docstring_files)} documentation/docstring file(s); "
                    "no CPU test-db blocks or stages resolved — fallback"
                ),
            )

        affected_stages = cpu_stages | {DOCS_STAGE}
        return RuleResult(
            handled_files=claimed,
            affected_stages=affected_stages,
            scope="docsonly",
            block_filters=block_filters,
            sanity_relevant=False,
            perfsanity_relevant=False,
            reason=(
                f"docs: {len(claimed | docstring_files)} documentation/docstring file(s) → "
                f"{DOCS_STAGE} + "
                f"{len(cpu_stages)} CPU test stage(s)"
            ),
        )
