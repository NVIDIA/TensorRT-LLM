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
"""Read-only accessor for the merged CBTS touch DB (`cbts_touchmap.sqlite`).

Implements the TOUCH_DB_CONTRACT.md queries, plus the two real-data details
the contract text predates:
  - the `test` column is `<stage>/<nodeid>` (stage-prefixed);
  - unit tests are recorded wrapped as
    `test_unittests.py::test_unittests_v2[<inner>]`, while test-db YAML lists
    the bare `<inner>` entry.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional

from blocks import normalize_test_id

# §3 producer canon(): keep the tail from the first `tensorrt_llm/` segment.
_CANON_RE = re.compile(r"(tensorrt_llm/.*)$")

# A DB test value is `<stage>/<nodeid>`; unit tests wrap the inner entry.
_UNITTEST_WRAP_RE = re.compile(r"::test_unittests_v2\[(?P<inner>.+)\]$")

# Coverage-completeness heuristic (interim, until the producer emits a per-test
# completeness/outcome signal). A test whose worker/server subprocess coverage
# was lost looks "clean but empty"; skipping it for an executor change would be
# an escape. `untrusted_tests()` uses these to force such tests to always run.
# They live here (not in the selector) so the audit tool and the selector share
# one source of truth for what "untrusted" means.
#
# A test that drove model execution but is missing the executor worker
# (`_WORKER_SENTINEL`) had its worker/server subprocess capture lost. Two ways
# to identify "drove execution", both footprint-independent:
#   - `_LAUNCH_MARKERS` `(file, qualname_substring)`: a call-based signal — the
#     LLM-API path enters `LLM.generate` / `GenerationExecutor.generate`.
#   - `_SERVING_PATH_MARKERS` nodeid substrings: for disagg, whose coordinator
#     runs no disagg-specific product code (only generic param validators — no
#     clean call marker exists) and whose ctx/gen servers run in uninstrumented
#     trtllm-serve subprocesses; identify it by nodeid path instead.
# `_MIN_FUNCS` is a last-resort catch-all for any other near-empty capture.
_WORKER_SENTINEL = "tensorrt_llm/_torch/pyexecutor/py_executor.py"
_LAUNCH_MARKERS: tuple[tuple[str, str], ...] = (
    ("tensorrt_llm/llmapi/llm.py", "generate"),
    ("tensorrt_llm/executor/executor.py", "GenerationExecutor.generate"),
)
_SERVING_PATH_MARKERS: tuple[str, ...] = ("disaggregated/",)
_MIN_FUNCS = 30


def canon(path: str) -> str:
    """Canonicalize a path to the DB's `file` form (`tensorrt_llm/...`)."""
    m = _CANON_RE.search(path)
    return m.group(1) if m else path


def split_stage(test: str) -> tuple[str, str]:
    """Split a DB `test` value `<stage>/<nodeid>` into `(stage, nodeid)`.

    Stage names carry no `/`; a nodeid always does (`dir/file.py::...`), so the
    first `/` is the boundary. Returns `("", test)` when there is no `/`.
    """
    stage, sep, nodeid = test.partition("/")
    return (stage, nodeid) if sep else ("", test)


def unwrap_unittest(nodeid: str) -> Optional[str]:
    """Return the inner `unittest/...` entry of a wrapped unittest nodeid, else None.

    `test_unittests.py::test_unittests_v2[unittest/x.py -m "part0"]`
        -> `unittest/x.py -m "part0"`
    """
    m = _UNITTEST_WRAP_RE.search(nodeid)
    return m.group("inner") if m else None


def db_key(entry: str) -> Optional[str]:
    """Map a test-db YAML `tests:` entry to the DB nodeid form, or None if not 1:1.

    The inverse of `unwrap_unittest`: unit tests wrap as
    `test_unittests.py::test_unittests_v2[<inner>]`, and a `-k` keyword entry
    expands to many nodeids at runtime (no single DB key) -> None.
    """
    e = normalize_test_id(entry)
    if e.startswith("unittest/"):
        return f"test_unittests.py::test_unittests_v2[{e}]"
    if " -k " in e:
        return None
    return e


class TouchDB:
    """Read-only view over a merged `cbts_touchmap.sqlite`."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def open(cls, sqlite_path: Path | str) -> "TouchDB":
        """Open the DB read-only (`mode=ro`) and verify the `touch` schema."""
        uri = f"file:{Path(sqlite_path).resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(touch)")}
        if not {"test", "file", "qualname"} <= cols:
            conn.close()
            raise ValueError(f"unexpected touch schema, columns={sorted(cols)}")
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TouchDB":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- meta (every key optional; read with a default) --

    def meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row is not None else default

    def schema_version(self) -> Optional[str]:
        return self.meta("schema_version")

    def collection_commit(self) -> Optional[str]:
        """Commit the DB was collected at, for staleness gating (absent today)."""
        return self.meta("commit") or self.meta("collection_commit")

    # -- reverse lookup (the core of selection); always `test != ''` --

    def tests_touching_file(self, file: str) -> set[str]:
        """Stage-prefixed tests that entered any function in `file` (file-level)."""
        return {
            row[0]
            for row in self._conn.execute(
                "SELECT DISTINCT test FROM touch WHERE file=? AND test!=''", (file,)
            )
        }

    def tests_touching_func(self, file: str, qualname: str) -> set[str]:
        """Stage-prefixed tests that entered `qualname` in `file` (function-level)."""
        return {
            row[0]
            for row in self._conn.execute(
                "SELECT DISTINCT test FROM touch WHERE file=? AND qualname=? AND test!=''",
                (file, qualname),
            )
        }

    def file_has_touch_rows(self, file: str) -> bool:
        """True iff any instrumented test entered a function in `file`."""
        row = self._conn.execute(
            "SELECT 1 FROM touch WHERE file=? AND test!='' LIMIT 1", (file,)
        ).fetchone()
        return row is not None

    # -- universe / per-stage --

    def known_tests(self) -> set[str]:
        """Every stage-prefixed test with coverage data."""
        return {
            row[0] for row in self._conn.execute("SELECT DISTINCT test FROM touch WHERE test!=''")
        }

    def instrumented_stages(self) -> set[str]:
        """Stage names the DB has data for — the stages coverage may narrow."""
        return {stage for stage, _ in map(split_stage, self.known_tests()) if stage}

    def known_by_stage(self) -> dict[str, set[str]]:
        """`{stage -> {bare nodeid, ...}}` over all known tests."""
        out: dict[str, set[str]] = {}
        for test in self.known_tests():
            stage, nodeid = split_stage(test)
            if stage:
                out.setdefault(stage, set()).add(nodeid)
        return out

    # -- forward lookup (debug / explain-why) --

    def files_touched_by(self, test: str) -> list[tuple[str, str]]:
        """`(file, qualname)` rows for a stage-prefixed `test`."""
        return [
            (row[0], row[1])
            for row in self._conn.execute("SELECT file, qualname FROM touch WHERE test=?", (test,))
        ]

    # -- coverage-completeness heuristic (interim, until the producer signals it) --

    def untrusted_tests(
        self,
        worker_file: str,
        launch_markers: tuple[tuple[str, str], ...],
        serving_path_markers: tuple[str, ...],
        min_funcs: int,
    ) -> set[str]:
        """Stage-prefixed tests whose per-test capture looks incomplete (must always run).

        Untrusted when the test drove model execution/serving but is missing the
        executor `worker_file` (its worker/server process was not captured) —
        identified either by entering a `launch_markers` `(file, qualname_substring)`
        (call-based) or by a `serving_path_markers` nodeid substring (path-based,
        for disagg) — OR when it entered fewer than `min_funcs` functions total
        (a near-empty capture, last resort).
        """
        drove_execution: set[str] = set()
        for file, qual_substr in launch_markers:
            drove_execution |= {
                row[0]
                for row in self._conn.execute(
                    "SELECT DISTINCT test FROM touch WHERE file=? AND qualname LIKE ? AND test!=''",
                    (file, f"%{qual_substr}%"),
                )
            }
        if serving_path_markers:
            drove_execution |= {
                test
                for test in self.known_tests()
                if any(marker in test for marker in serving_path_markers)
            }
        missing_worker = drove_execution - self.tests_touching_file(worker_file)
        tiny = {
            row[0]
            for row in self._conn.execute(
                "SELECT test FROM touch WHERE test!='' GROUP BY test HAVING COUNT(*) < ?",
                (min_funcs,),
            )
        }
        return missing_worker | tiny
