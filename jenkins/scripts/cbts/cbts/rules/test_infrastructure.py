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
"""Select registered consumers of explicitly reviewed test infrastructure.

The test-list checker validates lists in the independent Check-Test-List
pipeline branch, outside CBTS stage filtering. Its Python consumers (including
split_qa_by_gpu.py) are exercised by the scripts unit suite. Keep that entire
suite so indirect consumers are covered as well as the checker's own tests.
Other tools select their registered tools or usage tests. The committed LLM
args manifest also feeds the documentation build; product configuration
implementations are intentionally left to their normal selection policies.
"""

from collections.abc import Iterable

from cbts.blocks import Stage, YAMLIndex, _entry_target, _target_in_filter_subtree
from cbts.rules._helpers import resolve_affected_stages, stages_by_yaml_stem
from cbts.rules.base import PRInputs, Rule, RuleResult

_MANIFEST_PATH = "tensorrt_llm/usage/llm_args_golden_manifest.json"
_INFRASTRUCTURE_TEST_PATHS = {
    "scripts/check_test_list.py": ("unittest/scripts",),
    "scripts/split_qa_by_gpu.py": ("unittest/scripts",),
    "scripts/legacy_utils.py": ("unittest/scripts",),
    "scripts/templates/precommit-legacy-files.j2": ("unittest/scripts",),
    "scripts/templates/pyproject-format-exclude.j2": ("unittest/scripts",),
    "scripts/templates/ruff-legacy.toml.j2": ("unittest/scripts",),
    "scripts/test_to_stage_mapping.py": ("unittest/tools/test_test_to_stage_mapping.py",),
    "scripts/disagg_transfer_diagnostics.py": (
        "unittest/tools/test_disagg_transfer_diagnostics.py",
    ),
    "scripts/generate_llm_args_golden_manifest.py": (
        "unittest/usage/test_llmapi_config_telemetry_docs.py",
    ),
    _MANIFEST_PATH: ("unittest/usage/test_llmapi_config_telemetry_docs.py",),
}
_CBTS_PREFIX = "jenkins/scripts/cbts/"


def test_infrastructure_paths(paths: Iterable[str]) -> set[str]:
    """Return the reviewed paths owned by this rule, including CBTS's own files."""
    return {
        path
        for path in paths
        if path in _INFRASTRUCTURE_TEST_PATHS or path.startswith(_CBTS_PREFIX)
    }


class TestInfrastructureRule(Rule):
    name = "testinfra"

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        self.stages_by_yaml = stages_by_yaml_stem(stages)

    def apply(self, pr: PRInputs) -> RuleResult | None:
        claimed = test_infrastructure_paths(pr.changed_files)
        if not claimed:
            return None

        cbts_claimed = {path for path in claimed if path.startswith(_CBTS_PREFIX)}
        # Preserve the existing CBTS policy: its scripts suite is selected from
        # l0_cpu only. Other tools follow all registered serving stages.
        requirements = {
            (test_path, False)
            for path in claimed - cbts_claimed
            for test_path in _INFRASTRUCTURE_TEST_PATHS[path]
        }
        if cbts_claimed:
            requirements.add(("unittest/scripts", True))
        filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        matched: set[tuple[str, bool]] = set()
        stages: set[str] = set()
        for block in self.yaml_index.blocks:
            key = (block.yaml_stem, block.block_index)
            serving_stages = resolve_affected_stages(
                {key: {}}, self.yaml_index, self.stages_by_yaml
            )
            if not serving_stages:
                continue
            for entry in block.tests:
                target = _entry_target(entry)
                covered = {
                    (path, cpu_only)
                    for path, cpu_only in requirements
                    if (not cpu_only or block.yaml_stem == "l0_cpu")
                    and (
                        _target_in_filter_subtree(target, path)
                        or _target_in_filter_subtree(path, target)
                    )
                }
                if covered:
                    matched.update(covered)
                    stages.update(serving_stages)
                    filters.setdefault(key, {}).setdefault(target, set()).add(entry)
        if matched != requirements:
            return RuleResult(
                handled_files=claimed,
                affected_stages=set(),
                scope=None,
                reason="testinfra: mapped infrastructure tests have no test-db entry or serving stage",
            )
        if _MANIFEST_PATH in claimed:
            # DocsRule imports OutOfScopeRule, which consults the path predicate
            # above. Resolve the documentation stage after modules are loaded.
            from cbts.rules.docs import DOCS_STAGE

            stages.add(DOCS_STAGE)
        return RuleResult(
            handled_files=claimed,
            affected_stages=stages,
            scope="testsonly" if cbts_claimed else "testinfraonly",
            reason=f"testinfra: {len(claimed)} infrastructure file(s), {len(stages)} test stage(s)",
            block_filters=filters,
            sanity_relevant=False,
            perfsanity_relevant=False,
        )
