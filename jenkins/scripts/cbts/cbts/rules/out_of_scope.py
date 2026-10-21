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
"""OutOfScopeRule — claims changes outside product-test selection.

Emits scope="noop" with no test contribution and sanity / perfsanity off.
The main entry point still retains the always-run CPU stages; Build and
repository checks run upstream. When other rules also fire, the "noop"
scope gives way in `_combine_scopes` to the actionable scope.

`is_out_of_scope` is the single source of truth for these path patterns;
`TestsDefRule` imports it to skip the same paths, so out-of-scope claims
are not overridden by a same-file narrow contribution.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Optional

from cbts.blocks import Stage, YAMLIndex

from ._helpers import reconstruct_diff_pre_image
from .base import PRInputs, Rule, RuleResult

# Exact files that affect repository metadata or external gates, but not the
# pre-merge test stages CBTS selects. Do not widen this to `.github/`: workflow
# changes can alter CI execution and must keep falling back.
OUT_OF_SCOPE_EXACT_PATHS: tuple[str, ...] = (
    ".github/CODEOWNERS",
    "LICENSE",
    ".pre-commit-config.yaml",
    "agent-flow/.pre-commit-config.yaml",
    ".clang-format",
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
)

# Git uses this exact basename at every repository depth.
OUT_OF_SCOPE_FILENAMES: tuple[str, ...] = (".gitignore",)

# These conventional filenames identify clang tooling throughout the C++ tree.
# Match complete basenames so product files with a similar ending stay unclaimed.
OUT_OF_SCOPE_DIRECTORY_FILENAMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cpp", (".clang-format", ".clang-tidy", ".clangd")),
    ("cpp/**", (".clang-format", ".clang-tidy", ".clangd")),
)

_STATIC_CHECK_TABLES = frozenset(
    {"tool.ruff", "tool.isort", "tool.yapf", "tool.codespell", "tool.autoflake", "tool.mypy"}
)
_TABLE_RE = re.compile(r"^\s*(\[\[?)([\w.-]+)(\]\]?)\s*(?:#.*)?$")


# Path prefixes whose changes neither pre-merge nor post-merge L0 consumes.
# - .github/ISSUE_TEMPLATE/           : issue intake forms and guidance.
# - .codex/ and .claude/              : developer assistant configuration,
#   instructions, and local tooling.
# - tests/integration/test_lists/qa/   : QA test lists, run by separate
#   nightly QA workflows.
# - tests/integration/test_lists/dev/  : developer-side artifacts, not
#   consumed by any L0 pipeline.
# - tests/integration/defs/.test_durations : pytest-split timing cache;
#   used at runtime, doesn't affect test selection.
# - tests/integration/defs/.test_durations_* : per-cluster timing caches
#   (e.g. .test_durations_aws_dfw); same rationale as above.
# - tests/integration/defs/agg_unit_mem_df.csv : per-(gpu, case)
#   pytest-xdist parallel_factor table consumed by test_unittests.py;
#   tunes worker count only, no impact on which tests run or their
#   results.
# - tests/microbenchmarks/             : benchmarking scripts, not run
#   by any L0 stage.
# - jenkins/scripts/cbts/              : CBTS itself — rules, main, and
#   debug tools that run at decision time. Edits change narrowing logic
#   for future PRs but cannot affect the outcome of any test stage in
#   the current PR.
OUT_OF_SCOPE_PREFIXES: tuple[str, ...] = (
    ".github/ISSUE_TEMPLATE/",
    ".codex/",
    ".claude/",
    "tests/integration/test_lists/qa/",
    "tests/integration/test_lists/dev/",
    "tests/integration/defs/.test_durations",
    "tests/integration/defs/agg_unit_mem_df.csv",
    "tests/microbenchmarks/",
    "jenkins/scripts/cbts/",
)

# Excluded on purpose:
# - General `.md` / `.rst` matches: outside the exempt subtrees, DocsRule
#   routes documentation changes to CPU-Build_Docs.
# - General `.txt` matches: requirements.txt / constraints.txt affect runtime dependencies.
# - `.png` / `.jpg` / `.jpeg` / `.gif` / `.svg` / `.webp`: image files
#   under `examples/visual_gen/` are real test fixtures (e.g.
#   `cat_piano.png`, `woman_skyline_original_720p.jpeg` loaded by
#   `tests/unittest/_torch/visual_gen/`). Since location alone cannot
#   distinguish a fixture from a doc diagram, image edits fall back to
#   baseline rather than being claimed as noop here.


def _non_check_toml(source: str) -> list[str] | None:
    """Preserve text outside static-check tables; reject ambiguous TOML syntax."""
    if '"""' in source or "'''" in source:
        return None
    outside: list[str] = []
    check_table = False
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("["):
            match = _TABLE_RE.fullmatch(line)
            if match is None or len(match.group(1)) != len(match.group(3)):
                return None
            section = match.group(2)
            check_table = any(
                section == root or section.startswith(root + ".") for root in _STATIC_CHECK_TABLES
            )
        if not check_table:
            outside.append(line)
    return outside


def _check_tables_only_change(before: str, after: str) -> bool:
    if not before.strip():
        return False
    old = _non_check_toml(before)
    return old is not None and old == _non_check_toml(after)


def is_out_of_scope(path: str) -> bool:
    if path in OUT_OF_SCOPE_EXACT_PATHS:
        return True
    if any(path.startswith(p) for p in OUT_OF_SCOPE_PREFIXES):
        return True
    directory, _, name = path.rpartition("/")
    if name in OUT_OF_SCOPE_FILENAMES:
        return True
    # Outside the exempt subtrees, a pyproject can change runtime dependencies
    # or the wheel build and needs a verified diff.
    if name == "pyproject.toml":
        return False
    return any(
        fnmatchcase(directory, pattern) and name in filenames
        for pattern, filenames in OUT_OF_SCOPE_DIRECTORY_FILENAMES
    )


class OutOfScopeRule(Rule):
    name = "outofscope"
    needs_diff_for: tuple[str, ...] = ("pyproject.toml", "**/pyproject.toml")

    def __init__(
        self, yaml_index: YAMLIndex, stages: dict[str, Stage], repo_root: Path | None = None
    ) -> None:
        # Stored for parity with other rules' constructor shape; not used.
        self.yaml_index = yaml_index
        self._repo_root = repo_root

    def _is_check_only_pyproject(self, path: str, pr: PRInputs) -> bool:
        if self._repo_root is None or Path(path).name != "pyproject.toml":
            return False
        diff = pr.raw_diffs.get(path) or pr.diffs.get(path) or ""
        try:
            source = (self._repo_root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        before = reconstruct_diff_pre_image(source, diff)
        return before is not None and _check_tables_only_change(before, source)

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {
            path
            for path in pr.changed_files
            if is_out_of_scope(path) or self._is_check_only_pyproject(path, pr)
        }
        if not claimed:
            return None
        return RuleResult(
            handled_files=claimed,
            affected_stages=set(),
            scope="noop",
            sanity_relevant=False,
            perfsanity_relevant=False,
            reason=f"outofscope: {len(claimed)} file(s) have no product-test contribution",
        )
