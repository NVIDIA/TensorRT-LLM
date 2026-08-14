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
"""OpenEngineRule — narrows CI when OpenEngine adapter source changes."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from blocks import Stage, YAMLIndex, _entry_target

from ._helpers import iter_diff_changes, resolve_affected_stages, stages_by_yaml_stem
from .base import PRInputs, Rule, RuleResult

_OPENENGINE_SOURCE_PREFIX = "tensorrt_llm/grpc/openengine/"
_OPENENGINE_TEST_PREFIX = "unittest/grpc/openengine/"
_OPENENGINE_CI_ADDITIONS = {
    "jenkins/L0_MergeRequest.groovy": {
        "scopes: data.scopes ?: [],",
    },
    "jenkins/L0_Test.groovy": {
        "def cbtsScopes = testFilter[(CBTS_RESULT)]?.scopes ?: []",
        'if (cbtsScopes.contains("openengineonly")) {',
        'trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmSrc} && '
        'pip3 install -r requirements-openengine.txt")',
        "}",
    },
}


def _is_openengine_ci_setup_diff(path: str, diff: str) -> bool:
    """Recognize only the dedicated OpenEngine CBTS setup edits."""
    expected = _OPENENGINE_CI_ADDITIONS.get(path)
    if expected is None:
        return False
    changes = Counter(
        (sign, body.strip())
        for sign, body in iter_diff_changes(diff)
        if not body.lstrip().startswith("//")
    )
    return changes == Counter(("+", line) for line in expected)


def _is_openengine_claim(path: str) -> bool:
    """Return whether an OpenEngine source path needs its focused unit test."""
    return path.startswith(_OPENENGINE_SOURCE_PREFIX) and not path.endswith(".md")


class OpenEngineRule(Rule):
    name = "openengine"
    needs_diff_for = tuple(_OPENENGINE_CI_ADDITIONS)

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {path for path in pr.changed_files if _is_openengine_claim(path)}
        for path in _OPENENGINE_CI_ADDITIONS:
            if path in pr.changed_files and _is_openengine_ci_setup_diff(
                path, pr.diffs.get(path, "")
            ):
                claimed.add(path)
        if not claimed:
            return None

        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        for block in self.yaml_index.blocks:
            for entry in block.tests:
                target = _entry_target(entry)
                if not target.startswith(_OPENENGINE_TEST_PREFIX):
                    continue
                key = (block.yaml_stem, block.block_index)
                block_filters.setdefault(key, {}).setdefault(target, set()).add(entry)

        if not block_filters:
            return RuleResult(
                handled_files=claimed,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"openengine: {len(claimed)} claimed file(s); "
                    "no OpenEngine test entry found — fallback"
                ),
            )

        affected = resolve_affected_stages(block_filters, self.yaml_index, self._stages_by_yaml)
        return RuleResult(
            handled_files=claimed,
            affected_stages=affected,
            scope="openengineonly",
            block_filters=block_filters,
            sanity_relevant=False,
            perfsanity_relevant=False,
            reason=(
                f"openengine: {len(claimed)} claimed file(s) → "
                f"{len(block_filters)} test block(s), {len(affected)} stage(s)"
            ),
        )
