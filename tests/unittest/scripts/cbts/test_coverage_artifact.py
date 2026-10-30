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
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
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


# build_info.txt and the coverage archives are read with jenkins/scripts/fetch_build_artifact.sh,
# which applies the read rule and is tested in tests/unittest/scripts/test_fetch_build_artifact.py.
# These tests check how artifact.py runs it and what it makes of each exit status.
def _build_info_path(build: int) -> str:
    return f"{artifact.ARTIFACT_BASE}/{build}/{artifact.BUILD_INFO_NAME}"


def _archive_path(build: int, name: str) -> str:
    """The legacy path of one of a build's coverage archives."""
    return f"{artifact.ARTIFACT_BASE}/{build}/cbts-coverage/{name}"


# A read is a helper call's arguments before <dest-file>, as _HelperStub.reads lists them.
def _exists_read(build: int, name: str) -> tuple[str, ...]:
    return ("--exists", _archive_path(build, name))


def _archive_read(build: int, name: str) -> tuple[str, ...]:
    return (_archive_path(build, name),)


def _build_info_read(build: int) -> tuple[str, ...]:
    return (_build_info_path(build),)


_BUILD = 88
_COMMIT = "a" * 40
_BUILD_INFO = f"commit={_COMMIT}\nbranch=main\ndate=2026-10-16 00:00:00 UTC\n"
_ARTIFACT_PATH = _build_info_path(_BUILD)
_LAYOUT_PATH = f"llm-artifacts/LLM/main/L0_PostMerge/{_BUILD}/{artifact.BUILD_INFO_NAME}"
_X86_ARCHIVE, _SBSA_ARCHIVE = artifact.ARCH_TARBALL_NAMES
_CREDENTIAL_ID = "aws-artifactory-credentials"
_HELPER_ENVIRONMENT = ("LLM_ARTIFACTS_PRIMARY", "ARTIFACTORY_NETRC", "ARTIFACTORY_CREDENTIAL_ID")
_STUB_URL_BASE = "https://artifacts.example/artifactory"

# Stands in for fetch_build_artifact.sh. It records its arguments and the environment it sees,
# writes to stderr, and exits with the status set for its read, or else HELPER_STUB_STATUS. On 0,
# --exists prints "<url> <sha1>" with the URL HELPER_STUB_URL_BASE/<artifact-path>; a download
# writes to stdout in any case, and on 0 writes <dest-file>: a copy of the file set for the
# artifact path in HELPER_STUB_FILES, or else HELPER_STUB_CONTENT.
_HELPER_STUB = r"""
import json
import os
import shutil
import sys

args = sys.argv[1:]
with open(os.environ["HELPER_STUB_LOG"], "a", encoding="utf-8") as log:
    names = json.loads(os.environ["HELPER_STUB_ENV"])
    log.write(
        json.dumps({"args": args, "env": {name: os.environ.get(name) for name in names}}) + "\n"
    )
exists = args[0] == "--exists"
read = " ".join(args if exists else args[:1])
status = json.loads(os.environ["HELPER_STUB_STATUSES"]).get(
    read, int(os.environ["HELPER_STUB_STATUS"])
)
if exists:
    if status == 0:
        print(f"{os.environ['HELPER_STUB_URL_BASE']}/{args[1]}", "0" * 40)
else:
    print("helper stdout")
    if status == 0:
        source = json.loads(os.environ["HELPER_STUB_FILES"]).get(args[0])
        if source:
            shutil.copyfile(source, args[1])
        else:
            with open(args[1], "w", encoding="utf-8") as dest:
                dest.write(os.environ["HELPER_STUB_CONTENT"])
print("helper stderr", file=sys.stderr)
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
        monkeypatch.setenv("HELPER_STUB_URL_BASE", _STUB_URL_BASE)
        monkeypatch.setenv("HELPER_STUB_STATUSES", "{}")
        monkeypatch.setenv("HELPER_STUB_FILES", "{}")
        monkeypatch.setenv("LLM_ARTIFACTS_PRIMARY", "urm")
        monkeypatch.setenv("ARTIFACTORY_NETRC", str(tmp_path / "artifactory-netrc"))
        monkeypatch.setenv("ARTIFACTORY_CREDENTIAL_ID", _CREDENTIAL_ID)
        self._monkeypatch = monkeypatch
        self._statuses: dict[str, int] = {}
        self._files: dict[str, str] = {}
        self.exits_with(0)

    def exits_with(self, status: int, *read: str) -> None:
        """Exit with `status` from `read`, or without `read` from every read not given one."""
        if read:
            self._statuses[" ".join(read)] = status
            self._monkeypatch.setenv("HELPER_STUB_STATUSES", json.dumps(self._statuses))
        else:
            self._monkeypatch.setenv("HELPER_STUB_STATUS", str(status))

    def serves(self, artifact_path: str, source: Path) -> None:
        """Download `artifact_path` as a copy of `source`."""
        self._files[artifact_path] = str(source)
        self._monkeypatch.setenv("HELPER_STUB_FILES", json.dumps(self._files))

    @staticmethod
    def url(artifact_path: str) -> str:
        """The URL that --exists prints for `artifact_path`."""
        return f"{_STUB_URL_BASE}/{artifact_path}"

    @property
    def calls(self) -> list[dict]:
        if not self._log.exists():
            return []
        return [json.loads(line) for line in self._log.read_text(encoding="utf-8").splitlines()]

    @property
    def reads(self) -> list[tuple[str, ...]]:
        """Each call's arguments before <dest-file>."""
        return [
            tuple(call["args"] if call["args"][0] == "--exists" else call["args"][:1])
            for call in self.calls
        ]


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


def test_coverage_archive_location_returns_the_url_that_exists_printed(
    helper_stub: _HelperStub, capfd: pytest.CaptureFixture[str]
) -> None:
    url = artifact._coverage_archive_location(_BUILD, _X86_ARCHIVE)

    assert url == helper_stub.url(_archive_path(_BUILD, _X86_ARCHIVE))
    assert helper_stub.reads == [_exists_read(_BUILD, _X86_ARCHIVE)]
    captured = capfd.readouterr()
    # The "<url> <sha1>" line is read, not passed on to the command's stdout.
    assert captured.out == ""
    assert "helper stderr" in captured.err


def test_download_coverage_archive_writes_it_into_dest_dir(
    helper_stub: _HelperStub, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    dest_dir = tmp_path / "downloads"
    dest_dir.mkdir()

    tarball = artifact._download_coverage_archive(_BUILD, _SBSA_ARCHIVE, dest_dir)

    assert tarball == dest_dir / _SBSA_ARCHIVE
    assert tarball.read_text(encoding="utf-8") == _BUILD_INFO
    [call] = helper_stub.calls
    assert call["args"] == [_archive_path(_BUILD, _SBSA_ARCHIVE), str(dest_dir / _SBSA_ARCHIVE)]
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "helper stderr" in captured.err


def _locate_x86_archive(dest_dir: Path) -> Optional[str]:
    return artifact._coverage_archive_location(_BUILD, _X86_ARCHIVE)


def _download_x86_archive(dest_dir: Path) -> Optional[Path]:
    return artifact._download_coverage_archive(_BUILD, _X86_ARCHIVE, dest_dir)


_COVERAGE_ARCHIVE_READS = pytest.mark.parametrize(
    "read", (_locate_x86_archive, _download_x86_archive), ids=("exists", "download")
)


@_COVERAGE_ARCHIVE_READS
def test_coverage_archive_reads_return_none_when_neither_location_has_the_archive(
    helper_stub: _HelperStub, tmp_path: Path, read: Callable[[Path], object]
) -> None:
    helper_stub.exits_with(2)

    assert read(tmp_path) is None
    assert len(helper_stub.calls) == 1


@pytest.mark.parametrize(
    ("status", "error"),
    (
        (3, artifact.ArtifactAuthError),
        (4, artifact.ArtifactTransferError),
        (64, artifact.ArtifactError),
        (127, artifact.ArtifactError),
    ),
    ids=("rejected", "transfer-failed", "usage-error", "other"),
)
@_COVERAGE_ARCHIVE_READS
def test_coverage_archive_reads_raise_on_helper_failures(
    helper_stub: _HelperStub,
    tmp_path: Path,
    read: Callable[[Path], object],
    status: int,
    error: type[artifact.ArtifactError],
) -> None:
    helper_stub.exits_with(status)

    with pytest.raises(artifact.ArtifactError) as raised:
        read(tmp_path)

    assert type(raised.value) is error
    assert _archive_path(_BUILD, _X86_ARCHIVE) in str(raised.value)
    assert f"exit status {status}" in str(raised.value)
    # One call: the helper retries transient failures itself.
    assert len(helper_stub.calls) == 1


@_COVERAGE_ARCHIVE_READS
def test_coverage_archive_reads_report_a_helper_that_cannot_start(
    helper_stub: _HelperStub,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    read: Callable[[Path], object],
) -> None:
    empty_dir = tmp_path / "empty-path"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))

    with pytest.raises(artifact.ArtifactError) as raised:
        read(tmp_path)

    assert type(raised.value) is artifact.ArtifactError
    assert _archive_path(_BUILD, _X86_ARCHIVE) in str(raised.value)
    assert helper_stub.calls == []


def test_coverage_archive_location_rejects_output_other_than_a_url_and_a_sha1(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The stub then prints "not a url/<artifact-path> <sha1>".
    monkeypatch.setenv("HELPER_STUB_URL_BASE", "not a url")

    with pytest.raises(artifact.ArtifactError) as raised:
        artifact._coverage_archive_location(_BUILD, _X86_ARCHIVE)

    assert type(raised.value) is artifact.ArtifactError
    assert _archive_path(_BUILD, _X86_ARCHIVE) in str(raised.value)


@pytest.mark.parametrize(
    "read",
    (_exists_read(_BUILD, _X86_ARCHIVE), _build_info_read(_BUILD)),
    ids=("coverage-archive", "build-info"),
)
@pytest.mark.parametrize("status", (3, 4, 64), ids=("rejected", "transfer-failed", "usage-error"))
def test_main_reports_helper_failures_and_exits_1(
    helper_stub: _HelperStub,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    read: tuple[str, ...],
) -> None:
    helper_stub.exits_with(status, *read)
    monkeypatch.setattr(artifact, "merge_base", lambda head: "c" * 40)

    result = CliRunner().invoke(
        artifact.main,
        ["--resolve-build", "--pr-head", "b" * 40, "--build", str(_BUILD)],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    errors = [line for line in result.stderr.splitlines() if line.startswith("[artifact] ")]
    assert any(read[-1] in line and f"exit status {status}" in line for line in errors)


def _set_newest_build(monkeypatch: pytest.MonkeyPatch, newest: int) -> None:
    """Make `newest` the latest build, where the probe starts."""
    monkeypatch.setattr(artifact, "latest_build_number", lambda jenkins_base: newest)


def _measure_topology_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the compare API without the network: every build is at the PR base."""
    monkeypatch.setattr(artifact, "drift", lambda commit, base: (0, "identical"))
    monkeypatch.setattr(artifact, "compare_distance", lambda commit: 0)


def _check_compatibility_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the PR base, and find every selection compatible, without the network or Git."""
    monkeypatch.setattr(artifact, "merge_base", lambda head: "c" * 40)
    monkeypatch.setattr(
        artifact,
        "_patch_apply_status",
        lambda base, head, db_commit, relevant_paths=None: "clean",
    )


def test_select_build_resolves_explicit_pinned_build(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(artifact, "drift", lambda commit, base: (3, "behind"))
    monkeypatch.setattr(artifact, "compare_distance", lambda commit: 7)

    selected = artifact.select_build(_BUILD, "pr-base")

    assert selected is not None
    assert selected["build"] == _BUILD
    assert selected["commit"] == _COMMIT
    assert selected["base_commit"] == "pr-base"
    assert selected["drift"] == 3
    paths = [_archive_path(_BUILD, name) for name in artifact.ARCH_TARBALL_NAMES]
    assert selected["paths"] == paths
    assert selected["urls"] == [helper_stub.url(path) for path in paths]
    assert selected["url"] == selected["urls"][0]
    assert helper_stub.reads == [
        _exists_read(_BUILD, _X86_ARCHIVE),
        _exists_read(_BUILD, _SBSA_ARCHIVE),
        _build_info_read(_BUILD),
    ]


def test_select_build_rejects_changed_pinned_commit(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    drift = mock.Mock()
    monkeypatch.setattr(artifact, "drift", drift)

    assert artifact.select_build(_BUILD, "pr-base", expected_commit="b" * 40) is None
    drift.assert_not_called()


@pytest.mark.parametrize(
    ("status", "message"),
    ((2, "coverage pair is incomplete"), (4, "coverage archive unreadable")),
    ids=("absent", "transfer-failed"),
)
def test_select_build_declines_when_an_archive_of_the_pinned_build_cannot_be_read(
    helper_stub: _HelperStub, capfd: pytest.CaptureFixture[str], status: int, message: str
) -> None:
    helper_stub.exits_with(status, *_exists_read(_BUILD, _SBSA_ARCHIVE))

    assert artifact.select_build(_BUILD, "pr-base") is None

    # The pinned build has no older candidate, and its build_info.txt is not read.
    assert helper_stub.reads == [
        _exists_read(_BUILD, _X86_ARCHIVE),
        _exists_read(_BUILD, _SBSA_ARCHIVE),
    ]
    assert f"[artifact] pinned build {_BUILD}: {message}" in capfd.readouterr().err.splitlines()


def test_select_build_raises_when_the_primary_location_rejects_an_archive_read(
    helper_stub: _HelperStub,
) -> None:
    helper_stub.exits_with(3)

    with pytest.raises(artifact.ArtifactAuthError):
        artifact.select_build(_BUILD, "pr-base")

    assert len(helper_stub.calls) == 1


def test_accepts_artifact_collected_at_pr_base(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_newest_build(monkeypatch, 7)
    monkeypatch.setattr(artifact, "drift", lambda commit, base: (0, "identical"))
    monkeypatch.setattr(artifact, "compare_distance", lambda commit: 4)

    selected = artifact.select_tarball(_COMMIT, max_probe=1)

    assert selected is not None
    assert selected["drift"] == 0
    assert selected["drift_status"] == "identical"


def test_probe_selection_keeps_the_urls_that_exists_printed(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    _set_newest_build(monkeypatch, _BUILD)
    _measure_topology_offline(monkeypatch)

    selected = artifact.select_tarball("c" * 40, max_probe=1)

    assert selected is not None
    paths = [_archive_path(_BUILD, name) for name in artifact.ARCH_TARBALL_NAMES]
    assert selected["paths"] == paths
    assert selected["urls"] == [helper_stub.url(path) for path in paths]
    assert selected["url"] == selected["urls"][0]
    assert helper_stub.reads == [
        _exists_read(_BUILD, _X86_ARCHIVE),
        _exists_read(_BUILD, _SBSA_ARCHIVE),
        _build_info_read(_BUILD),
    ]
    assert capfd.readouterr().out == ""


@pytest.mark.parametrize(
    ("absent", "reads_of_build_3"),
    (
        (_X86_ARCHIVE, [_exists_read(3, _X86_ARCHIVE)]),
        (_SBSA_ARCHIVE, [_exists_read(3, _X86_ARCHIVE), _exists_read(3, _SBSA_ARCHIVE)]),
    ),
    ids=("x86", "sbsa"),
)
def test_probe_moves_to_an_older_build_when_an_archive_is_absent(
    helper_stub: _HelperStub,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    absent: str,
    reads_of_build_3: list[tuple[str, ...]],
) -> None:
    helper_stub.exits_with(2, *_exists_read(3, absent))
    _set_newest_build(monkeypatch, 3)
    _measure_topology_offline(monkeypatch)

    selected = artifact.select_tarball("c" * 40, max_probe=3)

    assert selected is not None
    assert selected["build"] == 2
    # Build 3 is incomplete: no other archive of it and not its build_info.txt is read.
    assert helper_stub.reads == [
        *reads_of_build_3,
        _exists_read(2, _X86_ARCHIVE),
        _exists_read(2, _SBSA_ARCHIVE),
        _build_info_read(2),
    ]
    assert "skipped" not in capfd.readouterr().err


def test_probe_selects_an_older_build_when_an_archive_read_fails(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    helper_stub.exits_with(4, *_exists_read(3, _SBSA_ARCHIVE))
    _set_newest_build(monkeypatch, 3)
    _measure_topology_offline(monkeypatch)

    selected = artifact.select_tarball("c" * 40, max_probe=3)

    assert selected is not None
    assert selected["build"] == 2
    assert selected["commit"] == _COMMIT
    assert helper_stub.reads == [
        _exists_read(3, _X86_ARCHIVE),
        _exists_read(3, _SBSA_ARCHIVE),
        _exists_read(2, _X86_ARCHIVE),
        _exists_read(2, _SBSA_ARCHIVE),
        _build_info_read(2),
    ]
    errors = capfd.readouterr().err.splitlines()
    assert "[artifact] build 3: coverage archive unreadable; skipped" in errors
    assert any(
        line.startswith("[artifact] ")
        and _archive_path(3, _SBSA_ARCHIVE) in line
        and "exit status 4" in line
        for line in errors
    )


def test_probe_reads_each_build_once_and_declines_while_every_transfer_fails(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # How an outage of artifactory.nvidia.com while it is the primary location (the default) looks
    # to the probe: every read fails in transfer, whichever location holds the file.
    helper_stub.exits_with(4)
    _set_newest_build(monkeypatch, 3)

    assert artifact.select_tarball("c" * 40, max_probe=3) is None

    # A build is skipped once its first coverage archive cannot be read.
    assert helper_stub.reads == [_exists_read(build, _X86_ARCHIVE) for build in (3, 2, 1)]
    errors = capfd.readouterr().err
    for build in (3, 2, 1):
        assert f"[artifact] build {build}: coverage archive unreadable; skipped" in errors
    assert (
        "[artifact] no complete x86/SBSA coverage pair with a known commit in the last 3 builds "
        "(3 skipped: coverage archive unreadable)"
    ) in errors


def test_probe_reports_how_many_builds_it_skipped_for_each_reason(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    helper_stub.exits_with(4, *_exists_read(4, _X86_ARCHIVE))
    helper_stub.exits_with(4, *_build_info_read(3))
    helper_stub.exits_with(2, *_exists_read(2, _SBSA_ARCHIVE))
    helper_stub.exits_with(4, *_build_info_read(1))
    _set_newest_build(monkeypatch, 4)

    assert artifact.select_tarball("c" * 40, max_probe=4) is None

    assert helper_stub.reads == [
        _exists_read(4, _X86_ARCHIVE),
        _exists_read(3, _X86_ARCHIVE),
        _exists_read(3, _SBSA_ARCHIVE),
        _build_info_read(3),
        _exists_read(2, _X86_ARCHIVE),
        _exists_read(2, _SBSA_ARCHIVE),
        _exists_read(1, _X86_ARCHIVE),
        _exists_read(1, _SBSA_ARCHIVE),
        _build_info_read(1),
    ]
    errors = capfd.readouterr().err
    assert "[artifact] build 4: coverage archive unreadable; skipped" in errors
    assert "[artifact] build 3: commit unknown; skipped" in errors
    assert "[artifact] build 2:" not in errors
    assert "[artifact] build 1: commit unknown; skipped" in errors
    assert (
        "[artifact] no complete x86/SBSA coverage pair with a known commit in the last 4 builds "
        "(1 skipped: coverage archive unreadable; 2 skipped: commit unknown)"
    ) in errors


def test_probe_selects_an_older_build_when_one_transfer_fails(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper_stub.exits_with(4, *_build_info_read(3))
    _set_newest_build(monkeypatch, 3)
    _measure_topology_offline(monkeypatch)

    selected = artifact.select_tarball("c" * 40, max_probe=3)

    assert selected is not None
    assert selected["build"] == 2
    assert selected["commit"] == _COMMIT


def test_probe_from_a_start_build_stays_within_the_window(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The window is the last 3 builds, 5 down to 3; the probe starts at build 4.
    helper_stub.exits_with(2)
    _set_newest_build(monkeypatch, 5)

    assert artifact.select_tarball("c" * 40, max_probe=3, start_build=4) is None

    assert helper_stub.reads == [_exists_read(build, _X86_ARCHIVE) for build in (4, 3)]


@pytest.mark.parametrize(
    ("rejected", "reads"),
    (
        (_exists_read(3, _X86_ARCHIVE), [_exists_read(3, _X86_ARCHIVE)]),
        (
            _build_info_read(3),
            [_exists_read(3, _X86_ARCHIVE), _exists_read(3, _SBSA_ARCHIVE), _build_info_read(3)],
        ),
    ),
    ids=("coverage-archive", "build-info"),
)
def test_probe_stops_when_the_primary_location_rejects_a_read(
    helper_stub: _HelperStub,
    monkeypatch: pytest.MonkeyPatch,
    rejected: tuple[str, ...],
    reads: list[tuple[str, ...]],
) -> None:
    helper_stub.exits_with(3, *rejected)
    _set_newest_build(monkeypatch, 3)

    with pytest.raises(artifact.ArtifactAuthError):
        artifact.select_tarball("c" * 40, max_probe=3)

    # Every other build's read would be rejected the same way.
    assert helper_stub.reads == reads


def test_resolve_pin_exits_1_when_the_primary_location_rejects_a_read(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper_stub.exits_with(3)
    _set_newest_build(monkeypatch, 3)
    monkeypatch.setattr(artifact, "merge_base", lambda head: "c" * 40)
    monkeypatch.setattr(artifact, "_get", lambda url, headers=None: (404, None))
    pin_path = tmp_path / artifact.PIN_NAME

    result = CliRunner().invoke(
        artifact.main,
        ["--resolve-pin", str(pin_path), "--pr-number", "18802", "--pr-head", "b" * 40],
        catch_exceptions=False,
    )

    # L0_MergeRequest declines the coverage tier when the command prints no plan.
    assert result.exit_code == 1
    assert result.stdout == ""
    assert len(helper_stub.calls) == 1
    errors = [line for line in result.stderr.splitlines() if line.startswith("[artifact] ")]
    assert any(
        _archive_path(3, _X86_ARCHIVE) in line and "exit status 3" in line for line in errors
    )
    assert not pin_path.exists()


def test_resolve_pin_prints_a_declined_plan_when_every_transfer_fails(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper_stub.exits_with(4)
    _set_newest_build(monkeypatch, 3)
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


def _write_coverage_archives(directory: Path) -> dict[str, Path]:
    """Write each architecture's coverage archive, holding a touch DB of one test, by name."""
    directory.mkdir()
    archives = {}
    for name, stage, test, source in (
        (_X86_ARCHIVE, "A10-PyTorch-1", "A10-PyTorch-1/test_x86.py::test_one", "x86.py"),
        (_SBSA_ARCHIVE, "GH200-PyTorch-1", "GH200-PyTorch-1/test_sbsa.py::test_two", "sbsa.py"),
    ):
        db = directory / f"{stage}.sqlite"
        write_leaf_database(
            db,
            stage=stage,
            process_uid=f"{stage}/coordinator",
            touches={test: {(f"/workspace/tensorrt_llm/{source}", "run")}},
            outcomes={test: "passed"},
        )
        archives[name] = directory / name
        with tarfile.open(archives[name], "w:gz") as tar:
            tar.add(db, arcname=artifact.DB_NAME)
    return archives


# The tests in the archives that _write_coverage_archives writes.
_ARCHIVED_TESTS = {
    "A10-PyTorch-1/test_x86.py::test_one",
    "GH200-PyTorch-1/test_sbsa.py::test_two",
}


def _tests_in(db: str) -> set[str]:
    connection = sqlite3.connect(db)
    try:
        return {row[0] for row in connection.execute("SELECT DISTINCT test FROM touch_rows")}
    finally:
        connection.close()


def test_prepare_merges_x86_and_sbsa_databases(helper_stub: _HelperStub, tmp_path: Path) -> None:
    archives = _write_coverage_archives(tmp_path / "archives")
    paths = [_archive_path(42, name) for name in artifact.ARCH_TARBALL_NAMES]
    for path, name in zip(paths, artifact.ARCH_TARBALL_NAMES):
        helper_stub.serves(path, archives[name])
    selection = {
        "url": helper_stub.url(paths[0]),
        "urls": [helper_stub.url(path) for path in paths],
        "paths": paths,
        "build": 42,
        "commit": "coverage-commit",
        "lag": 5,
        "base_commit": "pr-base",
        "drift": 2,
        "drift_status": "ahead",
    }
    output_dir = tmp_path / "prepared"

    with (
        mock.patch.object(artifact, "merge_base", return_value="pr-base"),
        mock.patch.object(artifact, "select_tarball", return_value=selection) as select,
        mock.patch.object(artifact, "_patch_apply_status", return_value="clean") as apply,
    ):
        ready = artifact.prepare(str(output_dir), "pr-head", ["tensorrt_llm/source.py"])

    assert ready is not None
    select.assert_called_once_with("pr-base")
    apply.assert_called_once_with(
        "pr-base",
        "pr-head",
        "coverage-commit",
        relevant_paths=["tensorrt_llm/source.py"],
    )
    # Each archive is downloaded with the helper, by its legacy path, into a temporary directory
    # under the output directory.
    downloads = [call["args"] for call in helper_stub.calls]
    assert [path for path, _ in downloads] == paths
    for (_, dest), name in zip(downloads, artifact.ARCH_TARBALL_NAMES):
        assert Path(dest).name == name
        assert Path(dest).parent.parent == output_dir
        assert not Path(dest).parent.exists()
    assert _tests_in(ready["path"]) == _ARCHIVED_TESTS
    assert json.loads(Path(ready["meta"]).read_text()) == selection


# The commit of build 2 in the tests where a probed build's download fails.
_OLDER_COMMIT = "d" * 40


def _probe_past_a_failed_download(
    helper_stub: _HelperStub,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: int = 4,
    older_compatibility: str = "clean",
) -> list[str]:
    """Serve builds 3 and 2, build 2 at _OLDER_COMMIT, and end build 3's SBSA download with `status`.

    Returns the commits that the compatibility check is asked about, in order. It finds build 3
    clean and build 2 `older_compatibility`.
    """
    archives = _write_coverage_archives(tmp_path / "archives")
    for build in (3, 2):
        for name in artifact.ARCH_TARBALL_NAMES:
            helper_stub.serves(_archive_path(build, name), archives[name])
    older_build_info = tmp_path / "older_build_info.txt"
    older_build_info.write_text(_BUILD_INFO.replace(_COMMIT, _OLDER_COMMIT), encoding="utf-8")
    helper_stub.serves(_build_info_path(2), older_build_info)
    helper_stub.exits_with(status, *_archive_read(3, _SBSA_ARCHIVE))
    _set_newest_build(monkeypatch, 3)
    _measure_topology_offline(monkeypatch)
    monkeypatch.setattr(artifact, "merge_base", lambda head: "c" * 40)
    checked = []

    def patch_apply_status(base, head, db_commit, relevant_paths=None):
        checked.append(db_commit)
        return "clean" if db_commit == _COMMIT else older_compatibility

    monkeypatch.setattr(artifact, "_patch_apply_status", patch_apply_status)
    return checked


def test_prepare_moves_to_an_older_build_when_a_download_fails(
    helper_stub: _HelperStub,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    checked = _probe_past_a_failed_download(helper_stub, monkeypatch, tmp_path)

    ready = artifact.prepare(str(tmp_path / "cbts_cov"), "b" * 40, ["tensorrt_llm/source.py"])

    assert ready is not None
    assert _tests_in(ready["path"]) == _ARCHIVED_TESTS
    # The PR's diff is checked against each selected build's own commit.
    assert checked == [_COMMIT, _OLDER_COMMIT]
    meta = json.loads(Path(ready["meta"]).read_text())
    assert meta["build"] == 2
    assert meta["commit"] == _OLDER_COMMIT
    assert meta["patch_apply_status"] == "clean"
    assert meta["paths"] == [_archive_path(2, name) for name in artifact.ARCH_TARBALL_NAMES]
    # The probe resumes below build 3, and build 2 is checked and downloaded in turn.
    assert helper_stub.reads == [
        _exists_read(3, _X86_ARCHIVE),
        _exists_read(3, _SBSA_ARCHIVE),
        _build_info_read(3),
        _archive_read(3, _X86_ARCHIVE),
        _archive_read(3, _SBSA_ARCHIVE),
        _exists_read(2, _X86_ARCHIVE),
        _exists_read(2, _SBSA_ARCHIVE),
        _build_info_read(2),
        _archive_read(2, _X86_ARCHIVE),
        _archive_read(2, _SBSA_ARCHIVE),
    ]
    errors = capfd.readouterr().err.splitlines()
    assert "[artifact] build 3: coverage archive unreadable; skipped" in errors
    selected = [line for line in errors if line.startswith("[artifact] selected build ")]
    assert [line.split(",")[0] for line in selected] == [
        "[artifact] selected build 3",
        "[artifact] selected build 2",
    ]


def test_prepare_declines_the_older_build_when_the_pr_diff_conflicts_with_it(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checked = _probe_past_a_failed_download(
        helper_stub, monkeypatch, tmp_path, older_compatibility="conflict"
    )

    ready = artifact.prepare(str(tmp_path / "cbts_cov"), "b" * 40, ["tensorrt_llm/source.py"])

    assert ready is not None
    assert ready["path"] is None
    assert checked == [_COMMIT, _OLDER_COMMIT]
    meta = json.loads(Path(ready["meta"]).read_text())
    assert meta["build"] == 2
    assert meta["patch_apply_status"] == "conflict"
    assert _archive_read(2, _X86_ARCHIVE) not in helper_stub.reads


def test_prepare_stops_when_a_probed_download_is_rejected(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checked = _probe_past_a_failed_download(helper_stub, monkeypatch, tmp_path, status=3)

    with pytest.raises(artifact.ArtifactAuthError):
        artifact.prepare(str(tmp_path / "cbts_cov"), "b" * 40, ["tensorrt_llm/source.py"])

    # Only a transfer failure moves the probe on; a rejection would reject every older build too.
    assert checked == [_COMMIT]
    assert helper_stub.reads[-1] == _archive_read(3, _SBSA_ARCHIVE)


def _prepare_pinned_build(tmp_path: Path) -> tuple[Path, object]:
    """Run --prepare as L0_MergeRequest does once the PR head has a pin; (its DIR, the result)."""
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
    return prepare_dir, result


def test_prepare_of_the_pinned_build_exits_1_when_its_transfer_fails(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The call L0_MergeRequest makes once the PR head has a pin: only the pinned build is read.
    helper_stub.exits_with(4)
    monkeypatch.setattr(artifact, "merge_base", lambda head: "c" * 40)

    prepare_dir, result = _prepare_pinned_build(tmp_path)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert len(helper_stub.calls) == 1
    errors = result.stderr.splitlines()
    assert any(
        _archive_path(_BUILD, _X86_ARCHIVE) in line and "exit status 4" in line for line in errors
    )
    assert f"[artifact] pinned build {_BUILD}: coverage archive unreadable" in errors
    assert not (prepare_dir / artifact.META_NAME).exists()


def test_prepare_of_the_pinned_build_exits_1_when_its_download_fails(
    helper_stub: _HelperStub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper_stub.exits_with(4, *_archive_read(_BUILD, _X86_ARCHIVE))
    _measure_topology_offline(monkeypatch)
    _check_compatibility_offline(monkeypatch)

    def latest_build_number(jenkins_base: str) -> int:
        raise AssertionError("a pinned build has no older candidate to probe")

    monkeypatch.setattr(artifact, "latest_build_number", latest_build_number)

    _, result = _prepare_pinned_build(tmp_path)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert helper_stub.reads == [
        _exists_read(_BUILD, _X86_ARCHIVE),
        _exists_read(_BUILD, _SBSA_ARCHIVE),
        _build_info_read(_BUILD),
        _archive_read(_BUILD, _X86_ARCHIVE),
    ]
    errors = result.stderr.splitlines()
    assert any(
        _archive_path(_BUILD, _X86_ARCHIVE) in line and "exit status 4" in line for line in errors
    )
    assert f"[artifact] pinned build {_BUILD}: coverage archive unreadable" in errors


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


class _LoopbackArtifactory:
    """The aws and urm locations, each a loopback Artifactory stand-in, as the real helper sees them."""

    def __init__(self, aws: _ArtifactServer, urm: _ArtifactServer) -> None:
        self._locations = {
            "aws": (aws, "sw-tensorrt-llm-generic-local"),
            "urm": (urm, "sw-tensorrt-generic"),
        }

    def publish(self, location: str, layout_path: str, content: bytes) -> None:
        """Put a file on a location, with the SHA-1 that its storage API reports."""
        server, repo = self._locations[location]
        sha1 = hashlib.sha1(content, usedforsecurity=False).hexdigest()
        storage_info = json.dumps({"checksums": {"sha1": sha1}}).encode()
        server.routes[f"/artifactory/api/storage/{repo}/{layout_path}"] = storage_info
        server.routes[f"/artifactory/{repo}/{layout_path}"] = content

    def url(self, location: str, layout_path: str) -> str:
        """The URL of a file on a location, which the helper prints for --exists."""
        server, repo = self._locations[location]
        return f"{server.base_url(repo)}/{layout_path}"


@pytest.fixture
def loopback_artifactory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[_LoopbackArtifactory]:
    # curl matches netrc entries by host name, so each location has its own loopback address.
    servers = []
    try:
        for host in ("127.0.0.1", "127.0.0.2"):
            servers.append(_ArtifactServer(host, {}))
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
        yield _LoopbackArtifactory(aws, urm)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def test_build_commit_with_the_real_helper_falls_back_to_urm(
    loopback_artifactory: _LoopbackArtifactory, capfd: pytest.CaptureFixture[str]
) -> None:
    loopback_artifactory.publish("urm", _LAYOUT_PATH, _BUILD_INFO.encode())

    assert artifact.build_commit(_BUILD) == _COMMIT

    captured = capfd.readouterr()
    assert captured.out == ""
    assert f"fetch_build_artifact: served-by=urm path={_LAYOUT_PATH}" in captured.err
    assert "reader-token-value" not in captured.err


def test_prepare_with_the_real_helper_reads_an_archive_only_on_urm(
    loopback_artifactory: _LoopbackArtifactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    archives = _write_coverage_archives(tmp_path / "archives")
    x86_layout, sbsa_layout = (
        f"llm-artifacts/LLM/main/L0_PostMerge/{_BUILD}/cbts-coverage/{name}"
        for name in artifact.ARCH_TARBALL_NAMES
    )
    loopback_artifactory.publish("aws", _LAYOUT_PATH, _BUILD_INFO.encode())
    loopback_artifactory.publish("aws", x86_layout, archives[_X86_ARCHIVE].read_bytes())
    loopback_artifactory.publish("urm", sbsa_layout, archives[_SBSA_ARCHIVE].read_bytes())
    _set_newest_build(monkeypatch, _BUILD)
    _measure_topology_offline(monkeypatch)
    _check_compatibility_offline(monkeypatch)

    ready = artifact.prepare(str(tmp_path / "cbts_cov"), "b" * 40, ["tensorrt_llm/source.py"])

    assert ready is not None
    assert _tests_in(ready["path"]) == _ARCHIVED_TESTS
    meta = json.loads(Path(ready["meta"]).read_text())
    assert meta["urls"] == [
        loopback_artifactory.url("aws", x86_layout),
        loopback_artifactory.url("urm", sbsa_layout),
    ]
    assert meta["paths"] == [_archive_path(_BUILD, name) for name in artifact.ARCH_TARBALL_NAMES]
    captured = capfd.readouterr()
    assert captured.out == ""
    assert f"fetch_build_artifact: served-by=aws path={x86_layout}" in captured.err
    assert f"fetch_build_artifact: served-by=urm path={sbsa_layout}" in captured.err
    assert "reader-token-value" not in captured.err


if __name__ == "__main__":
    unittest.main()
