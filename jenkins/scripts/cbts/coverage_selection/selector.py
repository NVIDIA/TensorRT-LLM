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
"""Coverage-based selection: changed core-Python files -> per-stage impacted/skippable sets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from qualname_map import (
    PythonChangeAnalysis,
    analyze_python_changes,
    closure_attributed_qualnames,
    import_executed_qualnames,
    qualnames_for_lines,
)
from reference_map import RepositoryReferenceIndex
from rules._helpers import iter_diff_deleted_post_lines, iter_diff_post_line_numbers
from touch_db import (
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
    """Per-stage-family coverage decision over a set of residual core-Python files."""

    ok: bool
    reason: str
    # keyed by stage family (shard suffix stripped); see `touch_db.stage_family`
    impacted: dict[str, set[str]] = field(default_factory=dict)
    skippable: dict[str, set[str]] = field(default_factory=dict)
    n_untrusted: int = 0
    # functions with no DB rows (new/uninstrumented); bounded per `no_data_policy`
    no_data_funcs: list[str] = field(default_factory=list)
    # functions with no rows whose statically complete local callers provide the bound
    caller_bounded_funcs: list[str] = field(default_factory=list)
    # residual file the forge API returned no patch for (binary / rename / oversized); declines
    no_diff_files: list[str] = field(default_factory=list)


# Fallback for a changed function with no DB rows: whole file / its importers / nothing.
NO_DATA_POLICIES = ("file", "importers", "ignore")
DEFAULT_NO_DATA_POLICY = "file"


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
    ) -> None:
        self.db = db
        self.repo_root = Path(repo_root)
        self._worker_sentinel = worker_sentinel
        self._launch_markers = launch_markers
        self._serving_path_markers = serving_path_markers
        self._min_funcs = min_funcs
        self._untrusted_stage_markers = untrusted_stage_markers
        self._no_data_policy = no_data_policy
        # Defaults to the checkout; callers explaining a past commit inject their own.
        self._read_source = read_source or self._read_head
        self._external_references = (
            external_references or RepositoryReferenceIndex(self.repo_root).external_references
        )
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

    def _impacted_tests(
        self, residual_files: list[str], diffs: dict[str, str]
    ) -> tuple[set[str], list[str], list[str], list[str], str | None]:
        """Return impacted tests, diagnostics, and an optional decline reason."""
        impacted: set[str] = set()
        no_data: set[str] = set()
        caller_bounded: set[str] = set()
        no_diff: list[str] = []
        for path in residual_files:
            cf = canon(path)
            diff = diffs.get(path) or ""
            source = self._read_source(path)
            if not diff.strip() or source is None:
                # Patch omitted (binary / renamed / oversized): the changed scope is unknown.
                no_diff.append(path)
                return (
                    impacted,
                    sorted(no_data),
                    sorted(caller_bounded),
                    no_diff,
                    f"no usable diff: {path}",
                )
            lines = iter_diff_post_line_numbers(diff)
            qualnames, ok = qualnames_for_lines(source, lines)
            if not ok:
                return (
                    impacted,
                    sorted(no_data),
                    sorted(caller_bounded),
                    no_diff,
                    f"unparsable source: {path}",
                )
            if not lines:
                continue  # comment / blank only: nothing executable changed, so nothing runs
            import_executed = import_executed_qualnames(source)
            closures = closure_attributed_qualnames(source, lines)
            dependencies = analyze_python_changes(source, lines, iter_diff_deleted_post_lines(diff))
            import_resolved = False
            for qualname in sorted(qualnames):  # sorted -> deterministic no_data order
                if qualname in import_executed:
                    if dependencies.unresolved_reason:
                        why = (
                            f"import-executed change, no sound bound: {path}::{qualname} "
                            f"({dependencies.unresolved_reason})"
                        )
                        return impacted, sorted(no_data), sorted(caller_bounded), no_diff, why
                    if not import_resolved:
                        external = self._external_references(cf, dependencies.changed_bindings)
                        if external:
                            why = (
                                f"import-executed change has external binding reference(s): "
                                f"{path}::{', '.join(sorted(external))}"
                            )
                            return impacted, sorted(no_data), sorted(caller_bounded), no_diff, why
                        for consumer in sorted(dependencies.import_consumers):
                            if (
                                not self.db.tests_touching_func(cf, consumer)
                                and "." not in consumer
                                and self._external_references(cf, {consumer})
                            ):
                                why = (
                                    "import-derived consumer has external reference(s): "
                                    f"{path}::{consumer}"
                                )
                                return (
                                    impacted,
                                    sorted(no_data),
                                    sorted(caller_bounded),
                                    no_diff,
                                    why,
                                )
                            tests, bounded = self._qualname_impact(
                                cf, consumer, dependencies, allow_caller_bound=True
                            )
                            impacted |= tests
                            if not self.db.tests_touching_func(cf, consumer):
                                no_data.add(f"{cf}::{consumer}")
                                if bounded:
                                    caller_bounded.add(f"{cf}::{consumer}")
                        import_resolved = True
                    continue
                if qualname in closures:
                    tests = self._underrecorded_bound(cf, qualname)
                    if tests is None:
                        why = f"closure change, no wider row set: {path}::{qualname}"
                        return impacted, sorted(no_data), sorted(caller_bounded), no_diff, why
                    impacted |= tests
                    continue
                tests, bounded = self._qualname_impact(
                    cf,
                    qualname,
                    dependencies,
                    allow_caller_bound=qualname in dependencies.import_consumers,
                )
                impacted |= tests
                if not self.db.tests_touching_func(cf, qualname):
                    no_data.add(f"{cf}::{qualname}")
                    if bounded:
                        caller_bounded.add(f"{cf}::{qualname}")
        return impacted, sorted(no_data), sorted(caller_bounded), no_diff, None

    def _qualname_impact(
        self,
        cf: str,
        qualname: str,
        dependencies: PythonChangeAnalysis,
        *,
        allow_caller_bound: bool = False,
    ) -> tuple[set[str], bool]:
        """Return a qualname's tests and whether local callers bounded a no-data symbol."""
        tests = self.db.tests_touching_func(cf, qualname)
        if tests:
            return tests, False
        if allow_caller_bound:
            caller_tests = self._caller_bound(
                cf,
                qualname,
                dependencies.callers,
                dependencies.caller_escapes,
                set(),
            )
            if caller_tests is not None:
                return caller_tests, True
        return self._no_data_fallback(cf), False

    def _caller_bound(
        self,
        cf: str,
        qualname: str,
        callers: dict[str, set[str]],
        caller_escapes: set[str],
        visiting: set[str],
    ) -> set[str] | None:
        """Resolve every local direct-caller branch to a qualname with DB rows."""
        if qualname in visiting or qualname in caller_escapes or not callers.get(qualname):
            return None
        visiting = visiting | {qualname}
        bound: set[str] = set()
        for caller in callers[qualname]:
            tests = self.db.tests_touching_func(cf, caller)
            if tests:
                bound |= tests
                continue
            transitive = self._caller_bound(cf, caller, callers, caller_escapes, visiting)
            if transitive is None:
                return None
            bound |= transitive
        return bound or None

    def _underrecorded_bound(self, cf: str, qualname: str) -> set[str] | None:
        """File row set when it is wider than a closure's enclosing qualname's, else None."""
        file_tests = self.db.tests_touching_file(cf)
        if len(file_tests) > len(self.db.tests_touching_func(cf, qualname)):
            return file_tests
        return None

    def _no_data_fallback(self, cf: str) -> set[str]:
        """Tests to force-run for a changed function the DB never captured."""
        if self._no_data_policy == "file":
            return self.db.tests_touching_file(cf)
        if self._no_data_policy == "importers":
            return self.db.tests_touching_func(cf, "<module>")
        return set()

    def _read_head(self, path: str) -> str | None:
        try:
            return (self.repo_root / path).read_text()
        except (OSError, UnicodeDecodeError):
            return None

    def decide(self, residual_files: list[str], diffs: dict[str, str]) -> CoverageResult:
        """Decide over residual files; ok=False for non-core, not-in-DB, or import-executed changes."""
        for path in residual_files:
            # Gate on the repo path: canon() matches `tensorrt_llm/` anywhere.
            if not (path.endswith(".py") and path.startswith("tensorrt_llm/")):
                return CoverageResult(ok=False, reason=f"non-core-Python residual file: {path}")
            cf = canon(path)
            if not self.db.file_has_touch_rows(cf):
                return CoverageResult(
                    ok=False, reason=f"zero-touch residual file (new/uninstrumented): {path}"
                )

        impacted_tests, no_data_funcs, caller_bounded_funcs, no_diff_files, decline = (
            self._impacted_tests(residual_files, diffs)
        )
        if decline is not None:
            return CoverageResult(ok=False, reason=decline, no_diff_files=no_diff_files)

        impacted: dict[str, set[str]] = {}
        for test in impacted_tests:
            stage, nodeid = split_stage(test)
            if stage:
                impacted.setdefault(stage_family(stage), set()).add(nodeid)

        untrusted = self.untrusted_families()
        skippable: dict[str, set[str]] = {}
        n_untrusted = 0
        for family, known_nodeids in self.db.known_by_family().items():
            imp = impacted.get(family, set())
            keep_untrusted = {n for n in known_nodeids if f"{family}/{n}" in untrusted}
            n_untrusted += len(keep_untrusted - imp)
            skippable[family] = known_nodeids - imp - keep_untrusted

        return CoverageResult(
            ok=True,
            reason=(
                f"{len(residual_files)} file(s) -> {len(impacted_tests)} impacted test(s); "
                f"{n_untrusted} untrusted (incomplete-capture) test(s) forced to run"
            ),
            impacted=impacted,
            no_diff_files=no_diff_files,
            skippable=skippable,
            n_untrusted=n_untrusted,
            no_data_funcs=no_data_funcs,
            caller_bounded_funcs=caller_bounded_funcs,
        )
