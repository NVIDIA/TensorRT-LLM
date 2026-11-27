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
"""Use the existing package sanity suite for import/definition edits."""

from __future__ import annotations

from pathlib import Path

from cbts.coverage.selection.python_change_analysis import analyze_execution_phases

from ._helpers import reconstruct_diff_pre_image
from .base import PRInputs, Rule, RuleResult


class ImportSanityRule(Rule):
    name = "import_sanity"
    needs_diff_for = ("**/*.py",)

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def apply(self, pr: PRInputs) -> RuleResult | None:
        files: set[str] = set()
        handled: set[str] = set()
        for path in pr.changed_files:
            diff = (pr.raw_diffs or pr.diffs).get(path)
            if not path.endswith(".py") or not diff:
                continue
            try:
                source = (self.repo_root / path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                source = ""
            phases = analyze_execution_phases(source, reconstruct_diff_pre_image(source, diff))
            if phases.limitation:
                return RuleResult(
                    set(), set(), None, f"import phase unknown: {path}: {phases.limitation}"
                )
            if phases.import_qualnames:
                files.add(path)
                if not phases.runtime_qualnames:
                    handled.add(path)
        if not files:
            return None

        # PackageSanityCheck already runs l0_sanity_check.yml. Its existing
        # sanity_required gate retains the full suite without per-test filters.
        return RuleResult(
            handled_files=handled,
            affected_stages=set(),
            scope="importsanityonly",
            reason="import/definition edits use l0_sanity_check; runtime edits retain coverage policy",
            import_sanity_files=files,
            sanity_relevant=True,
            perfsanity_relevant=False,
        )
