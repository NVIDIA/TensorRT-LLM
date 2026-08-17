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

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[3]
_CBTS = _ROOT / "jenkins/scripts/cbts"
_COVERAGE_UTILS = _CBTS / "coverage_utils"
_COVERAGE_SELECTION = _CBTS / "coverage_selection"
sys.path.insert(0, str(_CBTS))
sys.path.insert(0, str(_COVERAGE_UTILS))
sys.path.insert(0, str(_COVERAGE_SELECTION))

import pystart_report  # noqa: E402
from cbts_pystart import PyStartTracker  # noqa: E402
from compact_db import merge_databases, validate_database, write_leaf_database  # noqa: E402
from touch_db import TouchDB  # noqa: E402


def _rows(path: Path, query: str) -> list[tuple]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


class CompactCoverageTest(unittest.TestCase):
    def test_flat_database_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            flat = Path(temp_dir) / "flat.sqlite"
            connection = sqlite3.connect(flat)
            connection.execute(
                "CREATE TABLE touch(test TEXT, file TEXT, qualname TEXT, stage TEXT)"
            )
            connection.close()

            with self.assertRaises(ValueError):
                validate_database(flat)
            with self.assertRaises(ValueError):
                TouchDB.open(flat)

    def test_tracker_writes_compact_leaf_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = PyStartTracker([temp_dir], temp_dir, stage="stage-a")
            tracker._data = {
                "stage-a/test_a.py::test_one": {
                    (f"{temp_dir}/tensorrt_llm/a.py", "run"),
                    (f"{temp_dir}/tensorrt_llm/a.py", "<module>"),
                }
            }
            tracker._outcomes = {"stage-a/test_a.py::test_one": "passed"}
            tracker._expected = {"stage-a/test_a.py::test_one": 0}

            output = Path(tracker.save())
            validate_database(output)

            self.assertEqual(
                _rows(output, "SELECT test FROM case_stage"),
                [("test_a.py::test_one",)],
            )
            self.assertEqual(
                _rows(
                    output,
                    "SELECT test, file, qualname, stage FROM touch_rows "
                    "ORDER BY test, file, qualname, stage",
                ),
                [
                    (
                        "stage-a/test_a.py::test_one",
                        "tensorrt_llm/a.py",
                        "<module>",
                        "stage-a",
                    ),
                    (
                        "stage-a/test_a.py::test_one",
                        "tensorrt_llm/a.py",
                        "run",
                        "stage-a",
                    ),
                ],
            )
            self.assertEqual(
                _rows(
                    output,
                    "SELECT test, outcome, expected_workers, saved_procs FROM test_case_meta",
                ),
                [("stage-a/test_a.py::test_one", "passed", 0, 1)],
            )

    def test_hierarchical_merge_matches_direct_merge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            coordinator = root / "coordinator.sqlite"
            worker = root / "worker.sqlite"
            sbsa_leaf = root / "sbsa.sqlite"
            test_a = "stage-a/test_a.py::test_one"
            test_b = "stage-b/test_b.py::test_two"
            write_leaf_database(
                coordinator,
                stage="stage-a",
                process_uid="stage-a/coordinator",
                touches={
                    test_a: {
                        ("/workspace/tensorrt_llm/a.py", "run"),
                        ("/workspace/tensorrt_llm/a.py", "<module>"),
                    }
                },
                outcomes={test_a: "passed"},
                expected_workers={test_a: 1},
            )
            write_leaf_database(
                worker,
                stage="stage-a",
                process_uid="stage-a/worker",
                touches={test_a: {("/workspace/tensorrt_llm/worker.py", "execute")}},
                outcomes={},
                expected_workers={},
            )
            write_leaf_database(
                sbsa_leaf,
                stage="stage-b",
                process_uid="stage-b/coordinator",
                touches={test_b: {("/workspace/tensorrt_llm/b.py", "call")}},
                outcomes={test_b: "failed"},
                expected_workers={test_b: 0},
            )

            x86 = root / "x86.sqlite"
            sbsa = root / "sbsa-merged.sqlite"
            final = root / "final.sqlite"
            direct = root / "direct.sqlite"
            merge_databases([coordinator, worker], x86).close()
            merge_databases([sbsa_leaf], sbsa).close()
            merge_databases([x86, sbsa], final).close()
            merge_databases([coordinator, worker, sbsa_leaf], direct).close()

            touch_query = (
                "SELECT test, file, qualname, stage FROM touch_rows "
                "ORDER BY test, file, qualname, stage"
            )
            meta_query = (
                "SELECT test, stage, outcome, expected_workers, saved_procs "
                "FROM test_case_meta ORDER BY test, stage"
            )
            self.assertEqual(_rows(final, touch_query), _rows(direct, touch_query))
            self.assertEqual(_rows(final, meta_query), _rows(direct, meta_query))
            self.assertEqual(_rows(final, "PRAGMA foreign_key_check"), [])
            self.assertEqual(_rows(final, "PRAGMA integrity_check"), [("ok",)])
            self.assertEqual(
                _rows(final, meta_query),
                [
                    (test_a, "stage-a", "passed", 1, 2),
                    (test_b, "stage-b", "incomplete", 0, 1),
                ],
            )

            with TouchDB.open(final) as database:
                self.assertEqual(database.schema_version(), "3")
                self.assertEqual(database.known_tests(), {test_a, test_b})
                self.assertEqual(database.per_test_footprint(), {test_a: 3, test_b: 1})
                self.assertEqual(database.instrumented_stages(), {"stage-a", "stage-b"})
                self.assertEqual(database.tests_touching_file("tensorrt_llm/a.py"), {test_a})
                self.assertEqual(
                    database.tests_touching_func("tensorrt_llm/worker.py", "execute"),
                    {test_a},
                )
                self.assertTrue(database.file_has_touch_rows("tensorrt_llm/a.py"))
                self.assertFalse(database.file_has_touch_rows("tensorrt_llm/missing.py"))
                self.assertEqual(
                    set(database.files_touched_by(test_b)),
                    {("tensorrt_llm/b.py", "call")},
                )
                self.assertEqual(database.incomplete_capture_tests(), {test_b})
                self.assertEqual(
                    database.untrusted_tests(
                        "tensorrt_llm/worker.py",
                        (("tensorrt_llm/a.py", "run"),),
                        (),
                        2,
                    ),
                    {test_b},
                )

    def test_duplicate_intermediate_input_does_not_double_count_processes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            leaf = root / "leaf.sqlite"
            partial = root / "partial.sqlite"
            final = root / "final.sqlite"
            test = "stage-a/test_a.py::test_one"
            write_leaf_database(
                leaf,
                stage="stage-a",
                process_uid="stage-a/process-one",
                touches={test: {("tensorrt_llm/a.py", "run")}},
                outcomes={test: "passed"},
                expected_workers={test: 0},
            )
            merge_databases([leaf], partial).close()
            merge_databases([partial, partial], final).close()

            self.assertEqual(
                _rows(
                    final,
                    "SELECT expected_workers, saved_procs FROM test_case_meta",
                ),
                [(0, 1)],
            )
            self.assertEqual(_rows(final, "SELECT COUNT(*) FROM touch"), [(1,)])

    def test_report_outputs_read_compact_views(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            leaf = root / ".cbtscov.stage-a.process.sqlite"
            output = root / "cbts_touchmap.sqlite"
            report = root / "report"
            json_path = root / "touch.json"
            test = "stage-a/test_a.py::test_one"
            write_leaf_database(
                leaf,
                stage="stage-a",
                process_uid="stage-a/process-one",
                touches={test: {("tensorrt_llm/a.py", "run")}},
                outcomes={test: "passed"},
                expected_workers={test: 0},
            )

            with patch.object(
                sys,
                "argv",
                [
                    "pystart_report.py",
                    "--glob",
                    str(root / ".cbtscov.*.sqlite"),
                    "--out-sqlite",
                    str(output),
                    "--out-dir",
                    str(report),
                    "--out-json",
                    str(json_path),
                ],
            ):
                pystart_report.main()

            self.assertTrue((report / "index.html").is_file())
            self.assertIn(test, json_path.read_text())
            validate_database(output)
            self.assertEqual(
                dict(_rows(output, "SELECT key, value FROM meta"))["tests"],
                "1",
            )

    def test_compact_leaf_is_smaller_for_repeated_coverage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            compact = root / "compact.sqlite"
            flat = root / "flat.sqlite"
            symbols = {
                (f"/workspace/tensorrt_llm/module_{index // 10}.py", f"Class.method_{index}")
                for index in range(120)
            }
            touches = {
                f"stage-a/tests/test_long_name.py::TestSuite::test_case_{index}": symbols
                for index in range(40)
            }
            write_leaf_database(
                compact,
                stage="stage-a",
                process_uid="stage-a/large-process",
                touches=touches,
                outcomes={},
                expected_workers={},
            )

            connection = sqlite3.connect(flat)
            connection.execute("CREATE TABLE touch(test TEXT, file TEXT, qualname TEXT)")
            connection.executemany(
                "INSERT INTO touch VALUES (?, ?, ?)",
                (
                    (test, file, qualname)
                    for test, test_symbols in touches.items()
                    for file, qualname in test_symbols
                ),
            )
            connection.commit()
            connection.close()

            self.assertLess(compact.stat().st_size, flat.stat().st_size)


if __name__ == "__main__":
    unittest.main()
