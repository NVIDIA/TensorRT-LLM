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

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pytest
from click.testing import CliRunner

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.command.coverage.collection import pystart_report
from cbts.command.coverage.collection.pystart_report import merge_to_sqlite
from cbts.coverage.collection.compact_db import (
    merge_databases,
    validate_database,
    write_leaf_database,
)
from cbts.coverage.selection.touch_db import TouchDB

pytestmark = pytest.mark.cpu_only


def test_report_counts_tainted_complete_tests_as_unsafe(tmp_path: Path) -> None:
    """Taint and failed outcome each make a report row unsafe."""
    leaf = tmp_path / "leaf.sqlite"
    tests = ("test_safe", "test_tainted", "test_failed")
    write_leaf_database(
        leaf,
        stage="stage",
        process_uid="stage/coordinator",
        touches={test: {("tensorrt_llm/x.py", "run")} for test in tests},
        outcomes={test: "failed" if test == "test_failed" else "passed" for test in tests},
        taints=[("stage/unidentified_worker", "test_tainted", "attribution", "unconfirmed")],
    )
    output = tmp_path / "merged.sqlite"
    result = CliRunner().invoke(
        pystart_report.main, ["--glob", str(leaf), "--out-sqlite", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "3 (test, stage) rows, 2 not safe to skip" in result.output
    assert "or tainted" in result.output
    with sqlite3.connect(output) as connection:
        row = connection.execute(
            "SELECT outcome, tainted FROM test_case_meta WHERE test = 'stage/test_tainted'"
        ).fetchone()
    assert row == ("passed", 1)


def _rows(path: Path, query: str) -> list[tuple]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


def _merge(inputs: list[Path], output: Path) -> None:
    connection = merge_databases(inputs, output)
    connection.close()
    validate_database(output)


class CompactDatabaseTest(unittest.TestCase):
    def test_touch_db_rejects_missing_completeness_view(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            leaf = Path(temp_dir) / "leaf.sqlite"
            write_leaf_database(
                leaf,
                stage="stage",
                process_uid="stage/process",
                touches={
                    "stage/test_a.py::test_one": {
                        ("/workspace/tensorrt_llm/a.py", "run"),
                    }
                },
                outcomes={"stage/test_a.py::test_one": "passed"},
            )
            connection = sqlite3.connect(leaf)
            connection.execute("DROP VIEW test_case_meta")
            connection.commit()
            connection.close()

            with self.assertRaisesRegex(ValueError, "test_meta_columns=\\[\\]"):
                TouchDB.open(leaf)

    def test_report_glob_ignores_in_progress_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            leaf = root / ".cbtscov.stage.pid1.sqlite"
            write_leaf_database(
                leaf,
                stage="stage",
                process_uid="stage/process",
                touches={
                    "stage/test_a.py::test_one": {
                        ("/workspace/tensorrt_llm/a.py", "run"),
                    }
                },
                outcomes={"stage/test_a.py::test_one": "passed"},
            )
            in_progress = root / ".cbtscov.stage.pid2.sqlite.tmp"
            sqlite3.connect(in_progress).close()

            output = root / "merged.sqlite"
            connection, input_count = merge_to_sqlite(str(root / ".cbtscov.stage*"), output)
            connection.close()

            self.assertEqual(input_count, 1)
            validate_database(output)
            self.assertEqual(
                _rows(output, "SELECT test, file, qualname, stage FROM touch_rows"),
                [("stage/test_a.py::test_one", "tensorrt_llm/a.py", "run", "stage")],
            )

    def test_leaf_write_exposes_logical_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            leaf = Path(temp_dir) / "leaf.sqlite"
            write_leaf_database(
                leaf,
                stage="x86_64",
                process_uid="x86_64/coordinator",
                touches={
                    "x86_64/test_a.py::test_one": {
                        ("/workspace/tensorrt_llm/a.py", "run"),
                    }
                },
                outcomes={"x86_64/test_a.py::test_one": "passed"},
            )

            validate_database(leaf)
            self.assertEqual(
                _rows(
                    leaf,
                    "SELECT test, file, qualname, stage FROM touch_rows",
                ),
                [("x86_64/test_a.py::test_one", "tensorrt_llm/a.py", "run", "x86_64")],
            )
            self.assertEqual(
                _rows(
                    leaf,
                    "SELECT test, stage, outcome FROM test_case_meta",
                ),
                [("x86_64/test_a.py::test_one", "x86_64", "passed")],
            )

    def test_hierarchical_merge_is_equivalent_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x86_coordinator = root / "x86-coordinator.sqlite"
            x86_worker = root / "x86-worker.sqlite"
            sbsa_coordinator = root / "sbsa-coordinator.sqlite"

            x86_stage = "A10-PyTorch-1"
            sbsa_stage = "GH200-PyTorch-1"
            x86_test = f"{x86_stage}/test_a.py::test_one"
            sbsa_test = f"{sbsa_stage}/test_b.py::test_two"
            write_leaf_database(
                x86_coordinator,
                stage=x86_stage,
                process_uid=f"{x86_stage}/coordinator",
                touches={
                    x86_test: {
                        ("/workspace/tensorrt_llm/a.py", "run"),
                        ("/workspace/tensorrt_llm/shared.py", "shared"),
                    }
                },
                outcomes={x86_test: "passed"},
            )
            write_leaf_database(
                x86_worker,
                stage=x86_stage,
                process_uid=f"{x86_stage}/worker-0",
                touches={
                    x86_test: {
                        ("/workspace/tensorrt_llm/worker.py", "helper"),
                    }
                },
                outcomes={},
            )
            write_leaf_database(
                sbsa_coordinator,
                stage=sbsa_stage,
                process_uid=f"{sbsa_stage}/coordinator",
                touches={
                    sbsa_test: {
                        ("/workspace/tensorrt_llm/b.py", "run"),
                        ("/workspace/tensorrt_llm/shared.py", "shared"),
                    }
                },
                outcomes={sbsa_test: "failed"},
            )

            direct = root / "direct.sqlite"
            x86 = root / "x86.sqlite"
            sbsa = root / "sbsa.sqlite"
            hierarchical = root / "hierarchical.sqlite"
            duplicated = root / "duplicated.sqlite"
            _merge([x86_coordinator, x86_worker, sbsa_coordinator], direct)
            _merge([x86_coordinator, x86_worker], x86)
            _merge([sbsa_coordinator], sbsa)
            _merge([x86, sbsa], hierarchical)
            _merge([x86, x86, sbsa, sbsa], duplicated)

            touch_query = (
                "SELECT test, file, qualname, stage FROM touch_rows "
                "ORDER BY test, file, qualname, stage"
            )
            meta_query = "SELECT test, stage, outcome FROM test_case_meta ORDER BY test, stage"
            direct_touches = _rows(direct, touch_query)
            direct_metadata = _rows(direct, meta_query)
            for merged in (hierarchical, duplicated):
                with self.subTest(database=merged.name):
                    self.assertEqual(_rows(merged, touch_query), direct_touches)
                    self.assertEqual(_rows(merged, meta_query), direct_metadata)

            self.assertEqual(
                direct_metadata,
                [
                    (x86_test, x86_stage, "passed"),
                    (sbsa_test, sbsa_stage, "incomplete"),
                ],
            )
            self.assertEqual(_rows(duplicated, "SELECT COUNT(*) FROM process"), [(3,)])

            with TouchDB.open(duplicated) as database:
                self.assertEqual(database.schema_version(), "5")
                self.assertEqual(database.known_tests(), {x86_test, sbsa_test})
                self.assertEqual(
                    database.known_by_stage(),
                    {
                        x86_stage: {"test_a.py::test_one"},
                        sbsa_stage: {"test_b.py::test_two"},
                    },
                )
                self.assertEqual(
                    database.known_by_family(),
                    {
                        "A10-PyTorch": {"test_a.py::test_one"},
                        "GH200-PyTorch": {"test_b.py::test_two"},
                    },
                )
                self.assertEqual(
                    database.per_test_footprint(),
                    {x86_test: 3, sbsa_test: 2},
                )
                self.assertEqual(
                    database.tests_touching_file("tensorrt_llm/shared.py"),
                    {x86_test, sbsa_test},
                )
                self.assertEqual(
                    database.tests_touching_func("tensorrt_llm/worker.py", "helper"),
                    {x86_test},
                )
                self.assertTrue(database.file_has_touch_rows("tensorrt_llm/a.py"))
                self.assertFalse(database.file_has_touch_rows("tensorrt_llm/missing.py"))
                self.assertEqual(
                    set(database.files_touched_by(sbsa_test)),
                    {
                        ("tensorrt_llm/b.py", "run"),
                        ("tensorrt_llm/shared.py", "shared"),
                    },
                )
                self.assertEqual(database.incomplete_capture_tests(), {sbsa_test})
                self.assertEqual(
                    database.untrusted_tests(
                        "tensorrt_llm/worker.py",
                        (("tensorrt_llm/a.py", "run"),),
                        (),
                        2,
                    ),
                    {sbsa_test},
                )


if __name__ == "__main__":
    unittest.main()


class TaintScopeTest(unittest.TestCase):
    """A taint's test column is its scope: one test, or the whole stage."""

    _TOUCHES = {
        "alpha.py::test_one": {("/workspace/tensorrt_llm/a.py", "run")},
        "beta.py::test_two": {("/workspace/tensorrt_llm/b.py", "run")},
    }

    def _merged(self, temp_dir, taints):
        leaf = os.path.join(temp_dir, "leaf.sqlite")
        merged = os.path.join(temp_dir, "merged.sqlite")
        write_leaf_database(
            leaf,
            stage="x86_64",
            process_uid="x86_64/writer",
            touches=self._TOUCHES,
            outcomes={test: "passed" for test in self._TOUCHES},
            taints=taints,
        )
        merge_databases([leaf], merged)
        return _rows(merged, "SELECT test, tainted FROM test_case_meta ORDER BY test")

    def test_a_named_test_taints_only_that_test(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                self._merged(
                    temp_dir,
                    [
                        (
                            "x86_64/other",
                            "alpha.py::test_one",
                            "attribution",
                            "context_not_acknowledged",
                        )
                    ],
                ),
                [("x86_64/alpha.py::test_one", 1), ("x86_64/beta.py::test_two", 0)],
            )

    def test_the_empty_context_taints_every_test_in_the_stage(self):
        # The recorder could not say which tests it covers, so it covers all of them.
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                self._merged(
                    temp_dir, [("x86_64/other", "", "incomplete", "context_channel_unreachable")]
                ),
                [("", 1), ("x86_64/alpha.py::test_one", 1), ("x86_64/beta.py::test_two", 1)],
            )


class ProcessDiagnosticTest(unittest.TestCase):
    """Arbitrary {key: value} facts about a confirmed-dead subscriber, diagnostic only."""

    _TOUCHES = {
        "alpha.py::test_one": {("/workspace/tensorrt_llm/a.py", "run")},
    }

    def test_diagnostics_join_to_the_dead_subscriber_via_process_diagnostic_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            leaf = os.path.join(temp_dir, "leaf.sqlite")
            merged = os.path.join(temp_dir, "merged.sqlite")
            write_leaf_database(
                leaf,
                stage="x86_64",
                process_uid="x86_64/coordinator",
                touches=self._TOUCHES,
                outcomes={test: "passed" for test in self._TOUCHES},
                taints=[
                    (
                        "x86_64/dead_worker",
                        "alpha.py::test_one",
                        "incomplete",
                        "subscriber_confirmed_dead",
                    )
                ],
                diagnostics={
                    "x86_64/dead_worker": {
                        "cmdline": "/usr/bin/python3 -m mpi4py.futures worker",
                        "pytest_current_test": "unittest/_torch/modeling/test_foo.py::test_bar (call)",
                    }
                },
            )
            merge_databases([leaf], merged)
            self.assertEqual(
                _rows(
                    merged,
                    "SELECT key, value FROM process_diagnostic_rows "
                    "WHERE process = 'x86_64/dead_worker' ORDER BY key",
                ),
                [
                    ("cmdline", "/usr/bin/python3 -m mpi4py.futures worker"),
                    (
                        "pytest_current_test",
                        "unittest/_torch/modeling/test_foo.py::test_bar (call)",
                    ),
                ],
            )
            # A new key needs no schema or query change to show up here too.
            self.assertEqual(
                _rows(
                    merged,
                    "SELECT t.process, d.key, d.value FROM taint_rows t "
                    "JOIN process_diagnostic_rows d ON d.process = t.process "
                    "WHERE t.reason = 'subscriber_confirmed_dead' ORDER BY d.key",
                ),
                [
                    ("x86_64/dead_worker", "cmdline", "/usr/bin/python3 -m mpi4py.futures worker"),
                    (
                        "x86_64/dead_worker",
                        "pytest_current_test",
                        "unittest/_torch/modeling/test_foo.py::test_bar (call)",
                    ),
                ],
            )

    def test_a_taint_without_a_known_diagnostic_has_no_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            leaf = os.path.join(temp_dir, "leaf.sqlite")
            merged = os.path.join(temp_dir, "merged.sqlite")
            write_leaf_database(
                leaf,
                stage="x86_64",
                process_uid="x86_64/coordinator",
                touches=self._TOUCHES,
                outcomes={test: "passed" for test in self._TOUCHES},
                taints=[
                    (
                        "x86_64/unknown_worker",
                        "alpha.py::test_one",
                        "incomplete",
                        "subscriber_confirmed_dead",
                    )
                ],
            )
            merge_databases([leaf], merged)
            self.assertEqual(
                _rows(
                    merged,
                    "SELECT key, value FROM process_diagnostic_rows "
                    "WHERE process = 'x86_64/unknown_worker'",
                ),
                [],
            )

    def test_diagnostics_merge_across_leaf_databases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            leaf_a = os.path.join(temp_dir, "a.sqlite")
            leaf_b = os.path.join(temp_dir, "b.sqlite")
            merged = os.path.join(temp_dir, "merged.sqlite")
            # The outer pytest process (coordinator) is the one that learns of and records
            # the dead worker's diagnostic, matching how taints already record a process
            # that is not the writer.
            write_leaf_database(
                leaf_a,
                stage="x86_64",
                process_uid="x86_64/coordinator",
                touches={},
                outcomes={},
                taints=[
                    (
                        "x86_64/dead_worker",
                        "alpha.py::test_one",
                        "incomplete",
                        "subscriber_confirmed_dead",
                    )
                ],
                diagnostics={"x86_64/dead_worker": {"cmdline": "python3 worker.py"}},
            )
            write_leaf_database(
                leaf_b,
                stage="x86_64",
                process_uid="x86_64/other_coordinator",
                touches=self._TOUCHES,
                outcomes={test: "passed" for test in self._TOUCHES},
            )
            merge_databases([leaf_a, leaf_b], merged)
            self.assertEqual(
                _rows(
                    merged,
                    "SELECT key, value FROM process_diagnostic_rows "
                    "WHERE process = 'x86_64/dead_worker'",
                ),
                [("cmdline", "python3 worker.py")],
            )
