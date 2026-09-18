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

from blocks import Stage, YAMLIndex, _entry_target

from ._helpers import resolve_affected_stages, stages_by_yaml_stem
from .base import PRInputs, Rule, RuleResult

# Stage key declared in jenkins/L0_Test.groovy::docBuildConfigs. This stage is
# not test-db-driven, so CBTS contributes its name literally.
DOCS_STAGE = "CPU-Build_Docs"
CPU_TEST_YAML_STEM = "l0_cpu"

_DOCS_PREFIX = "docs/"
_DOCS_SUFFIXES: tuple[str, ...] = (".md", ".rst")


def is_docs_path(path: str) -> bool:
    """Return whether ``path`` contributes to the repository documentation."""
    return path.startswith(_DOCS_PREFIX) or path.endswith(_DOCS_SUFFIXES)


class DocsRule(Rule):
    name = "docs"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)

    def _cpu_block_filters(self) -> dict[tuple[str, int], dict[str, set[str]]]:
        """Keep every entry in every CPU test-db block.

        Docs changes run the complete CPU suite, not a consumer subset. Using
        each entry's own canonical target and raw text preserves `-k` variants
        through Layer 3's normal subtree and keyword filtering.
        """
        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        for block in self.yaml_index.blocks:
            if block.yaml_stem != CPU_TEST_YAML_STEM:
                continue
            prefix_to_entries: dict[str, set[str]] = {}
            for entry in block.tests:
                target = _entry_target(entry)
                if target:
                    prefix_to_entries.setdefault(target, set()).add(entry)
            block_filters[(block.yaml_stem, block.block_index)] = prefix_to_entries
        return block_filters

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {path for path in pr.changed_files if is_docs_path(path)}
        if not claimed:
            return None

        block_filters = self._cpu_block_filters()
        cpu_stages = resolve_affected_stages(block_filters, self.yaml_index, self._stages_by_yaml)
        if not block_filters or not cpu_stages:
            return RuleResult(
                handled_files=claimed,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"docs: {len(claimed)} documentation file(s); "
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
                f"docs: {len(claimed)} documentation file(s) → {DOCS_STAGE} + "
                f"{len(cpu_stages)} CPU test stage(s)"
            ),
        )
