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

import json
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.command.coverage.selection import artifact
from cbts.coverage.collection.compact_db import write_leaf_database

pytestmark = pytest.mark.cpu_only


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        timeout=120,
    ).stdout.strip()


class CoverageArtifactTest(unittest.TestCase):
    def test_prepare_rejects_print_selection(self) -> None:
        runner = CliRunner()
        result = runner.invoke(artifact.main, ["--prepare", "out", "--print-selection"])
        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIn("exactly one of", result.output)

    def test_patch_apply_status_detects_clean_and_conflicting_diffs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            _git(repo, "init")
            _git(repo, "config", "user.email", "cbts@example.com")
            _git(repo, "config", "user.name", "CBTS Test")
            source = repo / "source.py"
            waives = repo / "waives.txt"
            source.write_text("first\nbase\nlast\n")
            waives.write_text("base\n")
            _git(repo, "add", "source.py", "waives.txt")
            _git(repo, "commit", "-m", "base")
            base = _git(repo, "rev-parse", "HEAD")

            _git(repo, "checkout", "-b", "pr")
            source.write_text("first\npr\nlast\n")
            waives.write_text("pr\n")
            _git(repo, "commit", "-am", "pr")
            head = _git(repo, "rev-parse", "HEAD")

            _git(repo, "checkout", "-b", "db-clean", base)
            (repo / "other.py").write_text("coverage revision\n")
            _git(repo, "add", "other.py")
            _git(repo, "commit", "-m", "non-conflicting db")
            clean_db = _git(repo, "rev-parse", "HEAD")

            _git(repo, "checkout", "-b", "db-conflict", base)
            source.write_text("first\ndb\nlast\n")
            _git(repo, "commit", "-am", "conflicting db")
            conflicting_db = _git(repo, "rev-parse", "HEAD")

            _git(repo, "checkout", "-b", "db-irrelevant-conflict", base)
            waives.write_text("db\n")
            _git(repo, "commit", "-am", "conflicting non-residual file")
            irrelevant_conflict_db = _git(repo, "rev-parse", "HEAD")
            _git(repo, "checkout", "pr")

            self.assertEqual(
                artifact._patch_apply_status(base, head, clean_db, repo, str(repo)), "clean"
            )
            self.assertEqual(
                artifact._patch_apply_status(base, head, conflicting_db, repo, str(repo)),
                "conflict",
            )
            self.assertEqual(
                artifact._patch_apply_status(
                    base,
                    head,
                    irrelevant_conflict_db,
                    repo,
                    str(repo),
                    ["source.py"],
                ),
                "clean",
            )
            self.assertEqual(
                artifact._patch_apply_status(
                    base,
                    head,
                    irrelevant_conflict_db,
                    repo,
                    str(repo),
                    ["waives.txt"],
                ),
                "conflict",
            )

    def test_select_build_resolves_explicit_pinned_build(self) -> None:
        with (
            mock.patch.object(artifact, "_exists", return_value=True),
            mock.patch.object(artifact, "build_commit", return_value="coverage-commit"),
            mock.patch.object(artifact, "drift", return_value=(3, "behind")),
            mock.patch.object(artifact, "compare_distance", return_value=7),
        ):
            selected = artifact.select_build(42, "pr-base")

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["build"], 42)
        self.assertEqual(selected["commit"], "coverage-commit")
        self.assertEqual(selected["base_commit"], "pr-base")
        self.assertEqual(selected["drift"], 3)

    def test_select_build_rejects_changed_pinned_commit(self) -> None:
        with (
            mock.patch.object(artifact, "_exists", return_value=True),
            mock.patch.object(artifact, "build_commit", return_value="replacement-commit"),
            mock.patch.object(artifact, "drift") as drift,
        ):
            selected = artifact.select_build(
                42,
                "pr-base",
                expected_commit="pinned-commit",
            )

        self.assertIsNone(selected)
        drift.assert_not_called()

    def test_resolve_pin_reuses_matching_artifactory_pin(self) -> None:
        commit = "a" * 40
        pin = {
            "version": artifact.PIN_VERSION,
            "pr_number": "18802",
            "pr_head": "b" * 40,
            "coverage_db_build": 42,
            "coverage_db_commit": commit,
        }
        with (
            mock.patch.object(artifact, "_get", return_value=(200, json.dumps(pin).encode())),
            mock.patch.object(artifact, "select_tarball") as select_latest,
        ):
            plan = artifact.resolve_pin(
                "unused.json",
                "18802",
                "b" * 40,
                "pr-base",
            )

        self.assertEqual(
            plan,
            {
                "status": "ready",
                "build": 42,
                "commit": commit,
                "pin_upload_required": False,
            },
        )
        select_latest.assert_not_called()

    def test_resolve_pin_creates_file_for_jenkins_upload(self) -> None:
        commit = "a" * 40
        selection = {"build": 42, "commit": commit}
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(artifact, "_get", return_value=(404, None)),
            mock.patch.object(artifact, "select_tarball", return_value=selection),
        ):
            pin_path = Path(temp_dir) / artifact.PIN_NAME
            plan = artifact.resolve_pin(
                str(pin_path),
                "18802",
                "b" * 40,
                "pr-base",
            )
            pin = json.loads(pin_path.read_text())

        self.assertEqual(pin["coverage_db_build"], 42)
        self.assertEqual(pin["coverage_db_commit"], commit)
        self.assertTrue(plan["pin_upload_required"])
        self.assertEqual(plan["pin_path"], str(pin_path))
        self.assertEqual(
            plan["pin_target"],
            f"{artifact.PIN_BASE}/18802/{'b' * 40}/",
        )

    def test_resolve_pin_declines_invalid_or_unavailable_pin(self) -> None:
        with mock.patch.object(artifact, "_get", return_value=(200, b"{}")):
            invalid = artifact.resolve_pin("unused.json", "18802", "b" * 40, "pr-base")
        with mock.patch.object(artifact, "_get", return_value=(None, None)):
            unavailable = artifact.resolve_pin("unused.json", "18802", "b" * 40, "pr-base")

        self.assertEqual(invalid["status"], "declined")
        self.assertIn("invalid coverage DB pin", invalid["decline_reason"])
        self.assertEqual(unavailable["status"], "declined")
        self.assertIn("pin query failed", unavailable["decline_reason"])

    def test_resolve_pin_cli_prints_upload_plan(self) -> None:
        plan = {
            "status": "ready",
            "build": 42,
            "commit": "a" * 40,
            "pin_upload_required": True,
        }
        with (
            mock.patch.object(artifact, "merge_base", return_value="c" * 40),
            mock.patch.object(artifact, "resolve_pin", return_value=plan) as resolve_pin,
        ):
            result = CliRunner().invoke(
                artifact.main,
                [
                    "--resolve-pin",
                    "cbts_db_pin.json",
                    "--pr-number",
                    "18802",
                    "--pr-head",
                    "b" * 40,
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output), plan)
        resolve_pin.assert_called_once_with(
            "cbts_db_pin.json",
            "18802",
            "b" * 40,
            "c" * 40,
        )

    def test_resolve_build_prints_metadata_without_residual_paths(self) -> None:
        selection = {
            "build": 42,
            "commit": "coverage-commit",
            "base_commit": "pr-base",
        }
        with (
            mock.patch.object(artifact, "merge_base", return_value="pr-base"),
            mock.patch.object(artifact, "select_tarball", return_value=selection),
        ):
            result = CliRunner().invoke(artifact.main, ["--resolve-build", "--pr-head", "pr-head"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output), selection)

    def test_prepare_uses_explicit_pinned_build(self) -> None:
        selection = {
            "url": "x86-url",
            "urls": ["x86-url", "sbsa-url"],
            "build": 42,
            "commit": "coverage-commit",
            "base_commit": "pr-base",
            "drift": 2,
            "drift_status": "behind",
            "lag": 5,
        }
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(artifact, "merge_base", return_value="pr-base"),
            mock.patch.object(artifact, "select_build", return_value=selection) as select_build,
            mock.patch.object(artifact, "select_tarball") as select_latest,
            mock.patch.object(artifact, "_patch_apply_status", return_value="conflict"),
        ):
            ready = artifact.prepare(
                temp_dir,
                "pr-head",
                ["tensorrt_llm/source.py"],
                build=42,
                expected_commit="coverage-commit",
            )

        self.assertIsNotNone(ready)
        assert ready is not None
        self.assertIsNone(ready["path"])
        select_build.assert_called_once_with(
            42,
            "pr-base",
            expected_commit="coverage-commit",
        )
        select_latest.assert_not_called()

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
                mock.patch.object(artifact, "_patch_apply_status", return_value="clean") as apply,
                mock.patch.object(artifact, "download", side_effect=download),
                mock.patch.object(artifact, "extract", side_effect=extract),
            ):
                ready = artifact.prepare(str(output_dir), "pr-head", ["tensorrt_llm/source.py"])

            self.assertIsNotNone(ready)
            assert ready is not None
            select.assert_called_once_with("pr-base")
            apply.assert_called_once_with(
                "pr-base",
                "pr-head",
                "coverage-commit",
                relevant_paths=["tensorrt_llm/source.py"],
            )
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


if __name__ == "__main__":
    unittest.main()
