#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for CBTS coverage artifact selection and architecture DB merging."""

from __future__ import annotations

import ast
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.command.coverage.selection import artifact
from cbts.coverage.collection.compact_db import write_leaf_database

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_CBTS_OUTER = _ROOT / "jenkins/scripts/cbts"
_MAIN_PATH = _CBTS_OUTER / "cbts/command/main.py"


class CoverageArtifactTest(unittest.TestCase):
    def test_selects_closest_complete_ancestor_pair(self) -> None:
        commits = {104: "newer", 102: "older-three", 101: "older-one"}

        def exists(url: str) -> bool:
            if "/103/" in url:
                return url.endswith("cbts_pystart_report_x86_64.tar.gz")
            return "/100/" not in url

        relations = {
            "newer": (1, "behind"),
            "older-three": (3, "ahead"),
            "older-one": (1, "ahead"),
        }
        with (
            mock.patch.object(artifact, "latest_build_number", return_value=104),
            mock.patch.object(artifact, "_exists", side_effect=exists),
            mock.patch.object(
                artifact, "build_commit", side_effect=lambda build, _base: commits[build]
            ),
            mock.patch.object(
                artifact, "drift", side_effect=lambda commit, _base: relations[commit]
            ),
            mock.patch.object(artifact, "compare_distance", return_value=7) as lag,
        ):
            selected = artifact.select_tarball(
                "pr-base", artifact_base="coverage", jenkins_base="jenkins", max_probe=5
            )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["build"], 101)
        self.assertEqual(selected["commit"], "older-one")
        self.assertEqual(selected["drift"], 1)
        self.assertEqual(selected["drift_status"], "ahead")
        self.assertEqual(
            [url.rsplit("/", 1)[-1] for url in selected["urls"]],
            list(artifact.ARCH_TARBALL_NAMES),
        )
        lag.assert_called_once_with("older-one")

    def test_accepts_artifact_collected_at_pr_base(self) -> None:
        with (
            mock.patch.object(artifact, "latest_build_number", return_value=7),
            mock.patch.object(artifact, "_exists", return_value=True),
            mock.patch.object(artifact, "build_commit", return_value="pr-base"),
            mock.patch.object(artifact, "drift", return_value=(0, "identical")),
            mock.patch.object(artifact, "compare_distance", return_value=4),
        ):
            selected = artifact.select_tarball("pr-base", max_probe=1)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["drift"], 0)
        self.assertEqual(selected["drift_status"], "identical")

    def test_prepare_merges_x86_and_sbsa_databases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x86_db = root / "x86.sqlite"
            sbsa_db = root / "sbsa.sqlite"
            output_dir = root / "prepared"
            write_leaf_database(
                x86_db,
                stage="A10-PyTorch-1",
                process_uid="A10-PyTorch-1/coordinator",
                touches={
                    "A10-PyTorch-1/test_x86.py::test_one": {
                        ("/workspace/tensorrt_llm/x86.py", "run")
                    }
                },
                outcomes={"A10-PyTorch-1/test_x86.py::test_one": "passed"},
                expected_workers={"A10-PyTorch-1/test_x86.py::test_one": 0},
            )
            write_leaf_database(
                sbsa_db,
                stage="GH200-PyTorch-1",
                process_uid="GH200-PyTorch-1/coordinator",
                touches={
                    "GH200-PyTorch-1/test_sbsa.py::test_two": {
                        ("/workspace/tensorrt_llm/sbsa.py", "run")
                    }
                },
                outcomes={"GH200-PyTorch-1/test_sbsa.py::test_two": "passed"},
                expected_workers={"GH200-PyTorch-1/test_sbsa.py::test_two": 0},
            )
            urls = artifact.tarball_urls(42)
            selection = {
                "url": urls[0],
                "urls": urls,
                "build": 42,
                "commit": "coverage-commit",
                "lag": 5,
                "base_commit": "pr-base",
                "drift": 2,
                "drift_status": "ahead",
            }

            def download(url: str, destination: Path) -> Path:
                return destination / url.rsplit("/", 1)[-1]

            def extract(tarball: Path, destination: Path) -> bool:
                source = x86_db if "x86_64" in tarball.name else sbsa_db
                shutil.copyfile(source, destination / artifact.DB_NAME)
                return True

            with (
                mock.patch.object(artifact, "merge_base", return_value="pr-base"),
                mock.patch.object(artifact, "select_tarball", return_value=selection) as select,
                mock.patch.object(artifact, "download", side_effect=download),
                mock.patch.object(artifact, "extract", side_effect=extract),
            ):
                ready = artifact.prepare(str(output_dir), "pr-head")

            self.assertIsNotNone(ready)
            assert ready is not None
            select.assert_called_once_with("pr-base")
            connection = sqlite3.connect(ready["path"])
            try:
                tests = {
                    row[0] for row in connection.execute("SELECT DISTINCT test FROM touch_rows")
                }
            finally:
                connection.close()
            self.assertEqual(
                tests,
                {
                    "A10-PyTorch-1/test_x86.py::test_one",
                    "GH200-PyTorch-1/test_sbsa.py::test_two",
                },
            )
            self.assertEqual(json.loads(Path(ready["meta"]).read_text()), selection)

    def test_freshness_default_is_thirty_commits(self) -> None:
        tree = ast.parse(_MAIN_PATH.read_text())
        defaults = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertEqual(defaults["DEFAULT_COVERAGE_MAX_DRIFT"], 30)


if __name__ == "__main__":
    unittest.main()
