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

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_TOOL_PATH = _ROOT / "jenkins/scripts/cbts/tools/compact_touch_db.py"
_SPEC = importlib.util.spec_from_file_location("compact_touch_db", _TOOL_PATH)
compact_touch_db = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compact_touch_db)


def _make_source(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE touch (
            test TEXT,
            file TEXT,
            qualname TEXT,
            stage TEXT,
            UNIQUE(test, file, qualname, stage)
        );
        CREATE TABLE test_meta (
            test TEXT,
            stage TEXT,
            outcome TEXT,
            expected_workers INTEGER,
            saved_procs INTEGER,
            PRIMARY KEY(test, stage)
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO touch VALUES
            ('stage-a/test_a.py::test_one', 'tensorrt_llm/a.py', 'run', 'stage-a'),
            ('stage-a/test_a.py::test_one', 'tensorrt_llm/a.py', '<module>', 'stage-a'),
            ('stage-a/test_b.py::test_two', 'tensorrt_llm/a.py', 'run', 'stage-a'),
            ('', 'tensorrt_llm/a.py', '<module>', 'stage-a'),
            ('stage-b/test_c.py::test_three', 'tensorrt_llm/b.py', 'C.call', 'stage-b');
        INSERT INTO test_meta VALUES
            ('stage-a/test_a.py::test_one', 'stage-a', 'passed', 1, 2),
            ('stage-a/test_b.py::test_two', 'stage-a', 'passed', 0, 1),
            ('stage-b/test_c.py::test_three', 'stage-b', 'failed', 2, 2);
        INSERT INTO meta VALUES ('schema_version', '2'), ('tests', '3');
        """
    )
    connection.close()


class CompactTouchDbTest(unittest.TestCase):
    def test_build_compact_database_preserves_relations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.sqlite"
            output = Path(temp_dir) / "compact.sqlite"
            _make_source(source)

            compact_touch_db.build_compact_database(source, output)
            result = compact_touch_db.evaluate(source, output, 0.0)

            self.assertTrue(result["verification"]["equivalent"])
            self.assertEqual(result["source"]["touch_rows"], 5)
            self.assertEqual(result["compact"]["touch_rows"], 5)

            connection = sqlite3.connect(output)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM stage").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM case_stage").fetchone()[0], 4)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM symbol").fetchone()[0], 3)
            self.assertEqual(
                dict(connection.execute("SELECT key, value FROM meta")),
                {"schema_version": "2", "tests": "3"},
            )
            connection.close()

    def test_build_compact_database_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.sqlite"
            output = Path(temp_dir) / "compact.sqlite"
            _make_source(source)
            output.touch()

            with self.assertRaises(FileExistsError):
                compact_touch_db.build_compact_database(source, output)


if __name__ == "__main__":
    unittest.main()
