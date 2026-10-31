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

import hashlib
import json
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


# build_info.txt is read with jenkins/scripts/fetch_build_artifact.sh, which applies the read rule
# and is tested in tests/unittest/scripts/test_fetch_build_artifact.py. These tests check how
# artifact.py runs it and what it makes of each exit status.
def _build_info_path(build: int) -> str:
    return f"{artifact.ARTIFACT_BASE}/{build}/{artifact.BUILD_INFO_NAME}"


_BUILD = 88
_COMMIT = "a" * 40
_BUILD_INFO = f"commit={_COMMIT}\nbranch=main\ndate=2026-10-16 00:00:00 UTC\n"
_ARTIFACT_PATH = _build_info_path(_BUILD)
_LAYOUT_PATH = f"llm-artifacts/LLM/main/L0_PostMerge/{_BUILD}/{artifact.BUILD_INFO_NAME}"
_CREDENTIAL_ID = "aws-artifactory-credentials"
_HELPER_ENVIRONMENT = ("LLM_ARTIFACTS_PRIMARY", "ARTIFACTORY_NETRC", "ARTIFACTORY_CREDENTIAL_ID")

# Stands in for fetch_build_artifact.sh. It records its arguments and the environment it sees,
# writes to stdout and stderr, and exits with HELPER_STUB_STATUS; on 0 it first writes
# HELPER_STUB_CONTENT to <dest-file>.
_HELPER_STUB = r"""
import json
import os
import sys

with open(os.environ["HELPER_STUB_LOG"], "a", encoding="utf-8") as log:
    names = json.loads(os.environ["HELPER_STUB_ENV"])
    log.write(
        json.dumps({"args": sys.argv[1:], "env": {name: os.environ.get(name) for name in names}})
        + "\n"
    )
print("helper stdout")
print("helper stderr", file=sys.stderr)
status = int(os.environ["HELPER_STUB_STATUS"])
if status == 0:
    with open(sys.argv[2], "w", encoding="utf-8") as dest:
        dest.write(os.environ["HELPER_STUB_CONTENT"])
sys.exit(status)
"""


class _HelperStub:
    """Replaces the read helper that artifact.py runs, and records each call."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self._log = tmp_path / "helper_calls.jsonl"
        stub_script = tmp_path / "helper_stub.py"
        stub_script.write_text(_HELPER_STUB, encoding="utf-8")
        helper = tmp_path / "fetch_build_artifact.sh"
        helper.write_text(
            f'exec {shlex.quote(sys.executable)} -I -S {shlex.quote(str(stub_script))} "$@"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(artifact, "_FETCH_BUILD_ARTIFACT", helper)
        monkeypatch.setenv("HELPER_STUB_LOG", str(self._log))
        monkeypatch.setenv("HELPER_STUB_ENV", json.dumps(_HELPER_ENVIRONMENT))
        monkeypatch.setenv("HELPER_STUB_CONTENT", _BUILD_INFO)
        monkeypatch.setenv("LLM_ARTIFACTS_PRIMARY", "urm")
        monkeypatch.setenv("ARTIFACTORY_NETRC", str(tmp_path / "artifactory-netrc"))
        monkeypatch.setenv("ARTIFACTORY_CREDENTIAL_ID", _CREDENTIAL_ID)
        self._monkeypatch = monkeypatch
        self.exits_with(0)

    def exits_with(self, status: int) -> None:
        self._monkeypatch.setenv("HELPER_STUB_STATUS", str(status))

    @property
    def calls(self) -> list[dict]:
        if not self._log.exists():
            return []
        return [json.loads(line) for line in self._log.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def helper_stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _HelperStub:
    return _HelperStub(monkeypatch, tmp_path)


def test_build_commit_reads_build_info_through_the_helper(
    helper_stub: _HelperStub, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    assert artifact.build_commit(_BUILD) == _COMMIT

    [call] = helper_stub.calls
    artifact_path, dest = call["args"]
    assert artifact_path == _ARTIFACT_PATH
    assert Path(dest).name == artifact.BUILD_INFO_NAME
    # A temporary file, removed once read.
    assert not Path(dest).parent.exists()
    assert call["env"] == {
        "LLM_ARTIFACTS_PRIMARY": "urm",
        "ARTIFACTORY_NETRC": str(tmp_path / "artifactory-netrc"),
        "ARTIFACTORY_CREDENTIAL_ID": _CREDENTIAL_ID,
    }
    captured = capfd.readouterr()
    # stdout carries the command's JSON, so the helper's stdout is dropped; its stderr is the
    # job log's record of which location served the file, or why it could not.
    assert captured.out == ""
    assert "helper stderr" in captured.err


def test_build_commit_returns_none_when_neither_location_has_build_info(
    helper_stub: _HelperStub,
) -> None:
    helper_stub.exits_with(2)

    assert artifact.build_commit(_BUILD) is None
    assert len(helper_stub.calls) == 1


def test_build_commit_returns_none_when_the_transfer_fails(
    helper_stub: _HelperStub, capfd: pytest.CaptureFixture[str]
) -> None:
    helper_stub.exits_with(4)

    assert artifact.build_commit(_BUILD) is None
    # One call: the helper retries transient failures itself.
    assert len(helper_stub.calls) == 1
    errors = capfd.readouterr().err.splitlines()
    assert any(
        line.startswith("[artifact] ") and _ARTIFACT_PATH in line and "exit status 4" in line
        for line in errors
    )


@pytest.mark.parametrize(
    ("status", "error"),
    (
        (3, artifact.ArtifactAuthError),
        (64, artifact.ArtifactError),
        (127, artifact.ArtifactError),
    ),
    ids=("rejected", "usage-error", "other"),
)
def test_build_commit_raises_on_helper_failures(
    helper_stub: _HelperStub, status: int, error: type[artifact.ArtifactError]
) -> None:
    helper_stub.exits_with(status)

    with pytest.raises(artifact.ArtifactError) as raised:
        artifact.build_commit(_BUILD)

    assert type(raised.value) is error
    assert _ARTIFACT_PATH in str(raised.value)
    assert f"exit status {status}" in str(raised.value)
    # One call: the helper retries transient failures itself.
    assert len(helper_stub.calls) == 1


def test_build_commit_reports_a_helper_that_cannot_start(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_dir = tmp_path / "empty-path"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))

    with pytest.raises(artifact.ArtifactError) as raised:
        artifact.build_commit(_BUILD)

    assert type(raised.value) is artifact.ArtifactError
    assert _ARTIFACT_PATH in str(raised.value)
    assert helper_stub.calls == []


@pytest.mark.parametrize("status", (3, 4, 64), ids=("rejected", "transfer-failed", "usage-error"))
def test_main_reports_helper_failures_and_exits_1(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    helper_stub.exits_with(status)
    monkeypatch.setattr(artifact, "merge_base", lambda head: "c" * 40)
    monkeypatch.setattr(artifact, "_exists", lambda url: True)

    result = CliRunner().invoke(
        artifact.main,
        ["--resolve-build", "--pr-head", "b" * 40, "--build", str(_BUILD)],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    errors = [line for line in result.stderr.splitlines() if line.startswith("[artifact] ")]
    assert any(_ARTIFACT_PATH in line and f"exit status {status}" in line for line in errors)


def _make_builds_complete(monkeypatch: pytest.MonkeyPatch, newest: int) -> None:
    """Make the probe find builds `newest` down to 1, each with both coverage archives."""
    monkeypatch.setattr(artifact, "latest_build_number", lambda jenkins_base: newest)
    monkeypatch.setattr(artifact, "_exists", lambda url: True)


def test_probe_reads_each_build_once_and_declines_while_every_transfer_fails(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # How an outage of artifactory.nvidia.com while it is the primary location (the default) looks
    # to the probe: every read fails in transfer, whichever location holds the file.
    helper_stub.exits_with(4)
    _make_builds_complete(monkeypatch, 3)
    # Build 2 lacks its SBSA archive, so its build_info.txt is not read.
    sbsa_archive_of_2 = artifact.tarball_url(2, artifact.ARCH_TARBALL_NAMES[1])
    monkeypatch.setattr(artifact, "_exists", lambda url: url != sbsa_archive_of_2)

    assert artifact.select_tarball("c" * 40, max_probe=3) is None

    assert [call["args"][0] for call in helper_stub.calls] == [
        _build_info_path(build) for build in (3, 1)
    ]
    errors = capfd.readouterr().err
    for build in (3, 1):
        assert f"[artifact] build {build}: commit unknown; skipped" in errors
    assert (
        "[artifact] no complete x86/SBSA coverage pair with a known commit in the last 3 builds "
        "(2 skipped: commit unknown)"
    ) in errors


def test_probe_selects_an_older_build_when_one_transfer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fetch(artifact_path: str) -> bytes:
        if artifact_path == _build_info_path(3):
            raise artifact.ArtifactTransferError(f"could not read {artifact_path} (exit status 4)")
        return _BUILD_INFO.encode()

    _make_builds_complete(monkeypatch, 3)
    monkeypatch.setattr(artifact, "_fetch_build_artifact", fetch)
    monkeypatch.setattr(artifact, "drift", lambda commit, base: (0, "identical"))
    monkeypatch.setattr(artifact, "compare_distance", lambda commit: 0)

    selected = artifact.select_tarball("c" * 40, max_probe=3)

    assert selected is not None
    assert selected["build"] == 2
    assert selected["commit"] == _COMMIT


def test_probe_stops_when_the_primary_location_rejects_a_read(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper_stub.exits_with(3)
    _make_builds_complete(monkeypatch, 3)

    with pytest.raises(artifact.ArtifactAuthError):
        artifact.select_tarball("c" * 40, max_probe=3)

    # Every other build's read would be rejected the same way.
    assert len(helper_stub.calls) == 1


def test_resolve_pin_prints_a_declined_plan_when_every_transfer_fails(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper_stub.exits_with(4)
    _make_builds_complete(monkeypatch, 3)
    monkeypatch.setattr(artifact, "merge_base", lambda head: "c" * 40)
    monkeypatch.setattr(artifact, "_get", lambda url, headers=None: (404, None))
    pin_path = tmp_path / artifact.PIN_NAME

    result = CliRunner().invoke(
        artifact.main,
        ["--resolve-pin", str(pin_path), "--pr-number", "18802", "--pr-head", "b" * 40],
        catch_exceptions=False,
    )

    # L0_MergeRequest declines the coverage tier on this plan, as it does when the command exits 1.
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "status": "declined",
        "compatibility": "unknown",
        "decline_reason": "coverage tier declined: coverage DB could not be resolved",
        "pin_upload_required": False,
    }
    assert len(helper_stub.calls) == 3
    assert not pin_path.exists()


def test_prepare_of_the_pinned_build_exits_1_when_its_transfer_fails(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The call L0_MergeRequest makes once the PR head has a pin: only the pinned build is read.
    helper_stub.exits_with(4)
    monkeypatch.setattr(artifact, "merge_base", lambda head: "c" * 40)
    monkeypatch.setattr(artifact, "_exists", lambda url: True)
    paths_json = tmp_path / "cbts_coverage_residual.json"
    paths_json.write_text(json.dumps(["tensorrt_llm/source.py"]))
    prepare_dir = tmp_path / "cbts_cov"

    result = CliRunner().invoke(
        artifact.main,
        [
            "--prepare",
            str(prepare_dir),
            "--paths-json",
            str(paths_json),
            "--pr-head",
            "b" * 40,
            "--build",
            str(_BUILD),
            "--expected-commit",
            _COMMIT,
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert len(helper_stub.calls) == 1
    errors = result.stderr.splitlines()
    assert any(_ARTIFACT_PATH in line and "exit status 4" in line for line in errors)
    assert f"[artifact] pinned build {_BUILD}: commit unknown" in errors
    assert not (prepare_dir / artifact.META_NAME).exists()


class _ArtifactHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = self.server.routes.get(self.path)
        self.send_response(404 if body is None else 200)
        payload = b'{"errors": [{"status": 404}]}' if body is None else body
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Keep access logs out of the test output."""


class _ArtifactServer(ThreadingHTTPServer):
    """A loopback Artifactory stand-in that answers GETs from a path map, and 404 otherwise."""

    daemon_threads = True

    def __init__(self, host: str, routes: dict[str, bytes]) -> None:
        super().__init__((host, 0), _ArtifactHandler)
        self.routes = routes

    def base_url(self, repo: str) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/artifactory/{repo}"


def test_build_commit_with_the_real_helper_falls_back_to_urm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    content = _BUILD_INFO.encode()
    storage_info = {"checksums": {"sha1": hashlib.sha1(content, usedforsecurity=False).hexdigest()}}
    urm_routes = {
        f"/artifactory/api/storage/sw-tensorrt-generic/{_LAYOUT_PATH}": json.dumps(
            storage_info
        ).encode(),
        f"/artifactory/sw-tensorrt-generic/{_LAYOUT_PATH}": content,
    }
    # curl matches netrc entries by host name, so each location has its own loopback address.
    servers = []
    try:
        for host, routes in (("127.0.0.1", {}), ("127.0.0.2", urm_routes)):
            servers.append(_ArtifactServer(host, routes))
    except OSError as error:
        for server in servers:
            server.server_close()
        pytest.skip(f"cannot listen on the 127.0.0.0/8 loopback range: {error}")
    aws, urm = servers
    for server in servers:
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    netrc = tmp_path / "artifactory-netrc"
    netrc.write_text("machine 127.0.0.1 login ci-reader password reader-token-value\n")
    netrc.chmod(0o600)
    for name in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LLM_ARTIFACTS_PRIMARY", raising=False)
    monkeypatch.setenv("LLM_ARTIFACTS_AWS_URL", aws.base_url("sw-tensorrt-llm-generic-local"))
    monkeypatch.setenv("LLM_ARTIFACTS_URM_URL", urm.base_url("sw-tensorrt-generic"))
    monkeypatch.setenv("ARTIFACTORY_NETRC", str(netrc))
    monkeypatch.setenv("ARTIFACTORY_CREDENTIAL_ID", _CREDENTIAL_ID)

    try:
        commit = artifact.build_commit(_BUILD)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()

    assert commit == _COMMIT
    captured = capfd.readouterr()
    assert captured.out == ""
    assert f"fetch_build_artifact: served-by=urm path={_LAYOUT_PATH}" in captured.err
    assert "reader-token-value" not in captured.err


if __name__ == "__main__":
    unittest.main()
