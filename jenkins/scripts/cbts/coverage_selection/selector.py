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
    _WORKER_SENTINEL,
    TouchDB,
    canon,
    split_stage,
)


@dataclass
class CoverageResult:
    """Per-stage coverage decision over a set of residual core-Python files."""

    ok: bool
    reason: str
    impacted: dict[str, set[str]] = field(default_factory=dict)
    skippable: dict[str, set[str]] = field(default_factory=dict)
    n_untrusted: int = 0
    # functions with no DB rows (new/uninstrumented); bounded via importers
    no_data_funcs: list[str] = field(default_factory=list)


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
    ) -> None:
        self.db = db
        self.repo_root = Path(repo_root)
        self._worker_sentinel = worker_sentinel
        self._launch_markers = launch_markers
        self._serving_path_markers = serving_path_markers
        self._min_funcs = min_funcs
        self._untrusted: set[str] | None = None

    def untrusted_tests(self) -> set[str]:
        """Stage-prefixed tests with incomplete-looking capture (cached, DB-wide)."""
        if self._untrusted is None:
            self._untrusted = self.db.untrusted_tests(
                self._worker_sentinel,
                self._launch_markers,
                self._serving_path_markers,
                self._min_funcs,
            )
        return self._untrusted

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
        return impacted, no_data

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
                impacted.setdefault(stage, set()).add(nodeid)

        untrusted = self.untrusted_tests()
        skippable: dict[str, set[str]] = {}
        n_untrusted = 0
        for stage, known_nodeids in self.db.known_by_stage().items():
            imp = impacted.get(stage, set())
            keep_untrusted = {n for n in known_nodeids if f"{stage}/{n}" in untrusted}
            n_untrusted += len(keep_untrusted - imp)
            skippable[stage] = known_nodeids - imp - keep_untrusted

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
