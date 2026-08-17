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

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "jenkins/scripts/cbts/coverage_utils"))

from compact_db import merge_databases, validate_database, write_leaf_database  # noqa: E402


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
                expected_workers={"x86_64/test_a.py::test_one": 0},
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
                    "SELECT test, stage, outcome, expected_workers, saved_procs "
                    "FROM test_case_meta",
                ),
                [("x86_64/test_a.py::test_one", "x86_64", "passed", 0, 1)],
            )

    def test_hierarchical_merge_is_equivalent_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x86_coordinator = root / "x86-coordinator.sqlite"
            x86_worker = root / "x86-worker.sqlite"
            sbsa_coordinator = root / "sbsa-coordinator.sqlite"

            x86_test = "x86_64/test_a.py::test_one"
            sbsa_test = "SBSA/test_b.py::test_two"
            write_leaf_database(
                x86_coordinator,
                stage="x86_64",
                process_uid="x86_64/coordinator",
                touches={
                    x86_test: {
                        ("/workspace/tensorrt_llm/a.py", "run"),
                        ("/workspace/tensorrt_llm/shared.py", "shared"),
                    }
                },
                outcomes={x86_test: "passed"},
                expected_workers={x86_test: 1},
            )
            write_leaf_database(
                x86_worker,
                stage="x86_64",
                process_uid="x86_64/worker-0",
                touches={
                    x86_test: {
                        ("/workspace/tensorrt_llm/worker.py", "helper"),
                    }
                },
                outcomes={},
                expected_workers={},
            )
            write_leaf_database(
                sbsa_coordinator,
                stage="SBSA",
                process_uid="SBSA/coordinator",
                touches={
                    sbsa_test: {
                        ("/workspace/tensorrt_llm/b.py", "run"),
                        ("/workspace/tensorrt_llm/shared.py", "shared"),
                    }
                },
                outcomes={sbsa_test: "failed"},
                expected_workers={sbsa_test: 0},
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
            meta_query = (
                "SELECT test, stage, outcome, expected_workers, saved_procs "
                "FROM test_case_meta ORDER BY test, stage"
            )
            direct_touches = _rows(direct, touch_query)
            direct_metadata = _rows(direct, meta_query)
            for merged in (hierarchical, duplicated):
                with self.subTest(database=merged.name):
                    self.assertEqual(_rows(merged, touch_query), direct_touches)
                    self.assertEqual(_rows(merged, meta_query), direct_metadata)

            self.assertEqual(
                direct_metadata,
                [
                    (sbsa_test, "SBSA", "incomplete", 0, 1),
                    (x86_test, "x86_64", "passed", 1, 2),
                ],
            )
            self.assertEqual(_rows(duplicated, "SELECT COUNT(*) FROM process"), [(3,)])


if __name__ == "__main__":
    unittest.main()
