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
"""Query historical test records for directly changed Python symbols."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cbts.coverage.selection.python_change_analysis import symbols_for_diff
from cbts.coverage.selection.touch_db import (
    _LAUNCH_MARKERS,
    _MIN_FUNCS,
    _SERVING_PATH_MARKERS,
    _UNTRUSTED_STAGE_MARKERS,
    _WORKER_SENTINEL,
    TouchDB,
    canon,
    split_stage,
    stage_family,
)


@dataclass
class CoverageResult:
    """Direct recorded impact; no inferred callers, caches or state consumers."""

    ok: bool
    reason: str
    impacted: dict[str, set[str]] = field(default_factory=dict)
    skippable: dict[str, set[str]] = field(default_factory=dict)
    n_untrusted: int = 0
    no_data_funcs: list[str] = field(default_factory=list)
    no_diff_files: list[str] = field(default_factory=list)
    changed_symbols: dict[str, set[str]] = field(default_factory=dict)
    import_files: list[str] = field(default_factory=list)
    # Compatibility fields: lookup-only selection never infers these bounds.
    caller_bounded_funcs: list[str] = field(default_factory=list)
    retained_external_tests: list[str] = field(default_factory=list)


NO_DATA_POLICIES = ("file", "importers", "ignore")
DEFAULT_NO_DATA_POLICY = "ignore"


class CoverageSelector:
    def __init__(
        self,
        db: TouchDB,
        repo_root: Path,
        *,
        worker_sentinel: str = _WORKER_SENTINEL,
        launch_markers: tuple[tuple[str, str], ...] = _LAUNCH_MARKERS,
        serving_path_markers: tuple[str, ...] = _SERVING_PATH_MARKERS,
        min_funcs: int = _MIN_FUNCS,
        untrusted_stage_markers: tuple[str, ...] = _UNTRUSTED_STAGE_MARKERS,
        no_data_policy: str = DEFAULT_NO_DATA_POLICY,
        read_source: Callable[[str], str | None] | None = None,
        external_references: Callable[[str, set[str]], set[str]] | None = None,
        external_reference_paths: Callable[[str, set[str]], dict[str, set[str]]] | None = None,
        test_is_retained: Callable[[str], bool] | None = None,
        import_sanity_files: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        self.db = db
        self.repo_root = Path(repo_root)
        self._worker_sentinel = worker_sentinel
        self._launch_markers = launch_markers
        self._serving_path_markers = serving_path_markers
        self._min_funcs = min_funcs
        self._untrusted_stage_markers = untrusted_stage_markers
        self._read_source = read_source or self._read_head
        self._import_sanity_files = import_sanity_files
        self._untrusted: set[str] | None = None

    def untrusted_tests(self) -> set[str]:
        """Stage-prefixed tests with incomplete-looking capture (cached, DB-wide)."""
        if self._untrusted is None:
            self._untrusted = self.db.untrusted_tests(
                self._worker_sentinel,
                self._launch_markers,
                self._serving_path_markers,
                self._min_funcs,
                self._untrusted_stage_markers,
            )
        return self._untrusted

    def untrusted_families(self) -> set[str]:
        """`untrusted_tests()` re-keyed to `<stage family>/<nodeid>`; any shard taints the family."""
        out: set[str] = set()
        for test in self.untrusted_tests():
            stage, nodeid = split_stage(test)
            if stage:
                out.add(f"{stage_family(stage)}/{nodeid}")
        return out

    @staticmethod
    def _unittest_entry_covers(entry: str, test_path: str) -> bool:
        """Return whether a unittest entry collects a path without excluding it."""
        try:
            tokens = shlex.split(entry)
        except ValueError:
            return False
        if not tokens or "-k" in tokens:
            return False
        entry_path = tokens[0].partition("::")[0]

        ignored: set[str] = set()
        for index, token in enumerate(tokens):
            if token.startswith("--ignore="):
                ignored.add(token.partition("=")[2])
            elif token == "--ignore" and index + 1 < len(tokens):
                ignored.add(tokens[index + 1])
        if any(test_path == path or test_path.startswith(f"{path}/") for path in ignored):
            return False

        if entry_path.endswith(".py"):
            return entry_path == test_path
        return test_path.startswith(f"{entry_path.rstrip('/')}/")

    def _read_head(self, path: str) -> str | None:
        try:
            return (self.repo_root / path).read_text()
        except (OSError, UnicodeDecodeError):
            return None

    def decide(
        self,
        residual_files: list[str],
        diffs: dict[str, str],
        *,
        selection_diffs: dict[str, str] | None = None,
    ) -> CoverageResult:
        """Look up both source images; absent symbols add no inferred tests."""
        impacted_tests: set[str] = set()
        no_data: set[str] = set()
        changed_symbols: dict[str, set[str]] = {}
        import_files: set[str] = set()
        for path in residual_files:
            if not path.endswith(".py"):
                return CoverageResult(ok=False, reason=f"non-Python residual file: {path}")
            diff = diffs.get(path) or ""
            if not diff.strip():
                return CoverageResult(
                    ok=False, reason=f"no usable diff: {path}", no_diff_files=[path]
                )
            source = self._read_source(path)
            if source is None:
                if "+++ /dev/null" in diff or "deleted file mode" in diff:
                    source = ""
                else:
                    return CoverageResult(ok=False, reason=f"no usable source: {path}")
            changes = symbols_for_diff(
                source,
                diff,
                selection_diff=selection_diffs.get(path, diff)
                if selection_diffs is not None
                else None,
            )
            if changes.limitation:
                return CoverageResult(ok=False, reason=f"{changes.limitation}: {path}")
            if changes.import_qualnames:
                if path not in self._import_sanity_files:
                    return CoverageResult(ok=False, reason=f"import sanity required: {path}")
                import_files.add(path)
            cf = canon(path)
            changed_symbols[cf] = changes.qualnames
            for qualname in sorted(changes.qualnames):
                tests = self.db.tests_touching_func(cf, qualname)
                impacted_tests.update(tests)
                if not tests:
                    no_data.add(f"{cf}::{qualname}")

        impacted: dict[str, set[str]] = {}
        for test in impacted_tests:
            stage, nodeid = split_stage(test)
            if stage:
                impacted.setdefault(stage_family(stage), set()).add(nodeid)
        untrusted = self.untrusted_families()
        skippable: dict[str, set[str]] = {}
        n_untrusted = 0
        for family, known in self.db.known_by_family().items():
            direct = impacted.get(family, set())
            forced = {nodeid for nodeid in known if f"{family}/{nodeid}" in untrusted}
            n_untrusted += len(forced - direct)
            skippable[family] = known - direct - forced
        return CoverageResult(
            ok=True,
            reason=(
                f"{len(residual_files)} file(s) -> {len(impacted_tests)} recorded test(s); "
                f"{n_untrusted} untrusted test(s) forced to run"
            ),
            impacted=impacted,
            skippable=skippable,
            n_untrusted=n_untrusted,
            no_data_funcs=sorted(no_data),
            changed_symbols=changed_symbols,
            import_files=sorted(import_files),
        )
