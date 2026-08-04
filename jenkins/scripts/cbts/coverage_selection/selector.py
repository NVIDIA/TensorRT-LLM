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

from dataclasses import dataclass, field
from pathlib import Path

from qualname_map import qualnames_for_lines
from rules._helpers import iter_diff_post_line_numbers
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


# What to do with a changed function that has no rows in the DB — it was never
# captured, so "which tests exercise it" is unknown rather than empty.
#   file       fall back to every test that entered any function in the file
#   importers  fall back to the file's `<module>` touch set (its importers)
#   ignore     treat as impacting nothing
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
    ) -> None:
        self.db = db
        self.repo_root = Path(repo_root)
        self._worker_sentinel = worker_sentinel
        self._launch_markers = launch_markers
        self._serving_path_markers = serving_path_markers
        self._min_funcs = min_funcs
        self._untrusted_stage_markers = untrusted_stage_markers
        self._no_data_policy = no_data_policy
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
        """`untrusted_tests()` re-keyed to `<stage family>/<nodeid>`.

        Untrusted on any shard means untrusted for the family — the entry runs.
        """
        out: set[str] = set()
        for test in self.untrusted_tests():
            stage, nodeid = split_stage(test)
            if stage:
                out.add(f"{stage_family(stage)}/{nodeid}")
        return out

    def _impacted_tests(
        self, residual_files: list[str], diffs: dict[str, str]
    ) -> tuple[set[str], list[str]]:
        """Return (impacted stage-prefixed tests, file::qualname symbols with no DB rows)."""
        impacted: set[str] = set()
        no_data: list[str] = []
        for path in residual_files:
            cf = canon(path)
            lines = iter_diff_post_line_numbers(diffs.get(path, ""))
            source = self._read_head(path)
            if not lines or source is None:
                impacted |= self.db.tests_touching_file(cf)
                continue
            qualnames, ok = qualnames_for_lines(source, lines)
            if not ok:
                impacted |= self.db.tests_touching_file(cf)
                continue
            for qualname in sorted(qualnames):  # sorted -> deterministic no_data order
                tests = self.db.tests_touching_func(cf, qualname)
                impacted |= tests
                if not tests and qualname != "<module>":
                    no_data.append(f"{cf}::{qualname}")
                    impacted |= self._no_data_fallback(cf)
        return impacted, no_data

    def _no_data_fallback(self, cf: str) -> set[str]:
        """Tests to force-run for a changed function the DB never captured.

        No rows means the function was never observed, not that no test reaches
        it, so the file's own test set is the tightest sound bound available.
        """
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
        """Decide over residual files (repo-relative paths no rule claimed).

        Returns ok=False for any non-core-Python file or file absent from the DB.
        """
        for path in residual_files:
            cf = canon(path)
            if not (path.endswith(".py") and cf.startswith("tensorrt_llm/")):
                return CoverageResult(ok=False, reason=f"non-core-Python residual file: {path}")
            if not self.db.file_has_touch_rows(cf):
                return CoverageResult(
                    ok=False, reason=f"zero-touch residual file (new/uninstrumented): {path}"
                )

        impacted_tests, no_data_funcs = self._impacted_tests(residual_files, diffs)

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
            skippable=skippable,
            n_untrusted=n_untrusted,
            no_data_funcs=no_data_funcs,
        )
