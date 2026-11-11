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
"""Force registered import checks without claiming runtime file ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from cbts.blocks import Stage, YAMLIndex, _entry_target, block_matches_stage
from cbts.coverage.selection.python_change_analysis import analyze_execution_phases
from cbts.coverage.selection.selector import CoverageSelector

from ._helpers import reconstruct_diff_pre_image
from .base import PRInputs, Rule, RuleResult


class ImportSanityRule(Rule):
    name = "import_sanity"
    needs_diff_for = ("**/*.py",)

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage], repo_root: Path):
        self.yaml_index = yaml_index
        self.stages = stages
        self.repo_root = repo_root

    def apply(self, pr: PRInputs) -> RuleResult | None:
        files: set[str] = set()
        for path in pr.changed_files:
            diff = (pr.raw_diffs or pr.diffs).get(path)
            if not path.endswith(".py") or not diff:
                continue
            try:
                source = (self.repo_root / path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                source = ""
            phases = analyze_execution_phases(source, reconstruct_diff_pre_image(source, diff))
            if phases.import_qualnames or phases.limitation:
                files.add(path)
        if not files:
            return None

        inventory_path = self.repo_root / "tests/integration/test_lists/import_sanity.yml"
        try:
            inventory = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            inventory = None
        filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        affected: set[str] = set()
        missing: list[str] = []
        if (
            not isinstance(inventory, list)
            or not inventory
            or not all(isinstance(entry, str) and entry.strip() for entry in inventory)
        ):
            missing.append("import sanity inventory unavailable")
            inventory = []
        for target in inventory:
            found = False
            test_path = target.partition("::")[0]
            if not _inventory_target_exists(self.repo_root / "tests", target):
                missing.append(target)
                continue
            for block in self.yaml_index.blocks:
                active = {
                    name
                    for name, stage in self.stages.items()
                    if stage.yaml_stem == block.yaml_stem
                    and block_matches_stage(block, stage)
                    and "-OnDemand-" not in name
                    and (pr.post_merge or "Post-Merge" not in name and "_GPUs" not in name)
                }
                if not active:
                    continue
                for entry in block.tests:
                    entry_target = _entry_target(entry)
                    if (
                        entry.strip() != entry_target
                        or not CoverageSelector._unittest_entry_covers(entry, test_path)
                        or "::" in entry_target
                        and target != entry_target
                        and not target.startswith(entry_target + "::")
                    ):
                        continue
                    filters.setdefault((block.yaml_stem, block.block_index), {}).setdefault(
                        entry_target, set()
                    ).add(entry)
                    affected.update(active)
                    found = True
            if not found:
                missing.append(target)
        return RuleResult(
            handled_files=set(),
            affected_stages=affected,
            scope=None if missing else "noop",
            reason="missing import sanity obligations: " + ", ".join(missing)
            if missing
            else "forced import sanity; runtime effects remain with rules and coverage",
            block_filters=filters,
            import_sanity_files=files,
        )


def _inventory_target_exists(tests_root: Path, target: str) -> bool:
    """Reject missing selectors instead of treating a registered parent as proof."""
    path, *selectors = target.split("::")
    if Path(path).is_absolute() or ".." in Path(path).parts or not path.endswith(".py"):
        return False
    try:
        source = (tests_root / path).read_text(encoding="utf-8")
        statements = ast.parse(source).body
    except (OSError, UnicodeDecodeError, SyntaxError):
        return False
    for selector in selectors:
        name = selector.partition("[")[0]
        matches = [
            node
            for node in statements
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ]
        if len(matches) != 1:
            return False
        statements = matches[0].body
    return True
