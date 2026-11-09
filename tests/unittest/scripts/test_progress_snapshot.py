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
"""Tests for the checkpoint scripts in ``jenkins/scripts/``.

``progress_snapshot.sh`` prepares a checkpoint tarball of a stage's results folder and the content
hash that goes with it. ``progress_collect_slurm.sh`` copies a Slurm job's latest results into that
folder when the remote ``results.xml`` changed. The pipeline uploads the tarball, so neither script
may start ``curl`` or read an Artifactory credential.

The scripts run with ``bash`` on temporary folders. The ``SLURM_*`` commands are shell stubs that
copy from a local folder standing in for the job's workspace and log each call. Stubs on ``PATH``
record ``sleep`` instead of waiting, and log and fail any ``curl`` call.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "jenkins" / "scripts"
PROGRESS_SNAPSHOT = SCRIPTS_DIR / "progress_snapshot.sh"
PROGRESS_COLLECT_SLURM = SCRIPTS_DIR / "progress_collect_slurm.sh"
TIMEOUT_XML_SCRIPT = SCRIPTS_DIR / "generate_timeout_xml.py"

STAGE = "A10-PyTorch-1"
POST_TAG = "-attempt-2"
PROGRESS_TAR = f"results-{STAGE}-progress.tar.gz"
UNFINISHED_TESTS = (
    "accuracy/test_llm_api.py::TestLlama::test_auto_dtype",
    "unittest/llmapi/test_llm.py",
)
REMOTE_MTIME = 1_700_000_100
PERF_FOLDERS = ("aggr-1", "disagg-2")

# Sentinel values for the credential variables, which neither script may read.
CREDENTIALS = {
    "ART_USER": "art-user-3f9a1c-must-not-be-read",
    "ART_PASS": "art-pass-77d2e0-must-not-be-read",
    "ARTIFACTORY_NETRC": "/netrc-5b8e4d-must-not-be-read",
}

# Variables of the scripts that CI may have set for itself, and the files bash reads when it starts.
_DROPPED_VARIABLES = {
    "WORKSPACE",
    "STAGE_NAME",
    "LABEL",
    "POST_TAG",
    "FINAL_SNAPSHOT",
    "TIMEOUT_XML_SCRIPT",
    "ART_USER",
    "ART_PASS",
    "BASH_ENV",
    "ENV",
}
_DROPPED_PREFIXES = ("PROGRESS_", "SLURM_", "ARTIFACTORY_")


@dataclass
class Result:
    returncode: int
    output: str
    sleeps: list[str]


def _write_stub(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


def _write_xml(path: Path, suite: str = "pytest", tests: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites>'
        f'<testsuite name="{suite}" errors="0" failures="0" skipped="0" tests="{tests}">'
        "</testsuite></testsuites>\n"
    )


def _write_unfinished(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "unfinished_test.txt").write_text("".join(f"{t}\n" for t in UNFINISHED_TESTS))


def _contents(directory: Path) -> dict[str, bytes]:
    """Return each regular file under ``directory`` by its relative path."""
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _read_tarball(path: Path) -> dict[str, bytes]:
    """Return each regular file in a gzipped tarball by its member name."""
    files = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                extracted = archive.extractfile(member)
                assert extracted is not None
                files[member.name] = extracted.read()
    return files


def _assert_no_credentials(files: dict[str, bytes]) -> None:
    for name, data in files.items():
        for value in CREDENTIALS.values():
            assert value not in name
            assert value.encode() not in data, f"{name} holds a credential sentinel"


class FakeJob:
    """A local folder standing in for a Slurm job's workspace, and ``SLURM_*`` stubs that read it.

    Each stub logs its name before it runs, so a test sees which commands ran and in which order.
    The copy stubs write to the stage's folder by a relative path, as the pipeline's commands do.
    The perf copy reads its stdin, as a command run through ssh can, so the tests show that it
    cannot consume the list of folders that the script is reading.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.remote = tmp_path / "remote"
        self.mtime_file = tmp_path / "progress.mtime"
        self._calls_log = tmp_path / "calls.log"
        _write_xml(self.remote / "results.xml", tests=3)
        _write_unfinished(self.remote)
        for folder in PERF_FOLDERS:
            (self.remote / folder).mkdir()
            (self.remote / folder / "perf_result.csv").write_text(f"{folder}\n")

    def calls(self) -> list[str]:
        return self._calls_log.read_text().split() if self._calls_log.exists() else []

    def stub(self, name: str, command: str) -> str:
        """Return ``command`` behind a prefix that logs ``name``."""
        return f"echo {name} >> {shlex.quote(str(self._calls_log))}; {command}"

    def env(self) -> dict[str, str]:
        remote = shlex.quote(str(self.remote))
        return {
            "SLURM_SSH_STAT_CMD": self.stub("stat", f"echo {REMOTE_MTIME}"),
            "SLURM_SSH_REFRESH_CACHE_CMD": self.stub("refresh", "true"),
            "SLURM_SCP_XML_CMD": self.stub("xml", f"cp {remote}/results*.xml {STAGE}/"),
            "SLURM_SCP_UNFINISHED_CMD": self.stub(
                "unfinished", f"cp {remote}/unfinished_test.txt {STAGE}/"
            ),
            "SLURM_SSH_LIST_PERF_CMD": self.stub(
                "list", f"find {remote} -mindepth 1 -maxdepth 1 -type d | sort"
            ),
            "SLURM_SCP_PERF_TEMPLATE": self.stub(
                "perf", f"cat > /dev/null; cp -r PERF_FOLDER_PLACEHOLDER {STAGE}/"
            ),
            "PROGRESS_MTIME_FILE": str(self.mtime_file),
        }


class Runner:
    """Runs a checkpoint script on a temporary workspace, with stub ``sleep`` and ``curl``."""

    def __init__(self, tmp_path: Path) -> None:
        self.workspace = tmp_path / "workspace"
        self.stage_dir = self.workspace / STAGE
        self.tarball = self.workspace / PROGRESS_TAR
        self.hash_file = self.workspace / f"{PROGRESS_TAR}.last_hash"
        self.pending = self.workspace / f"{PROGRESS_TAR}.last_hash.pending"
        self.cwd = tmp_path / "cwd"
        self.stub_dir = tmp_path / "bin"
        for directory in (self.workspace, self.cwd, self.stub_dir):
            directory.mkdir()

        self._sleep_log = tmp_path / "sleep.log"
        self._curl_log = tmp_path / "curl.log"
        _write_stub(self.stub_dir / "sleep", f'echo "$*" >> {shlex.quote(str(self._sleep_log))}')
        _write_stub(
            self.stub_dir / "curl", f'echo "$*" >> {shlex.quote(str(self._curl_log))}\nexit 7'
        )

        env = {
            name: value
            for name, value in os.environ.items()
            if name not in _DROPPED_VARIABLES and not name.startswith(_DROPPED_PREFIXES)
        }
        env.update(CREDENTIALS)
        env.update(
            {
                "PATH": f"{self.stub_dir}{os.pathsep}{os.environ['PATH']}",
                "WORKSPACE": str(self.workspace),
                "STAGE_NAME": STAGE,
                "PROGRESS_TAR": PROGRESS_TAR,
                "LABEL": "checkpoint",
            }
        )
        self.env = env

    def run(self, script: Path, **env_overrides: str | None) -> Result:
        """Run ``script``; an override of ``None`` removes that variable from its environment."""
        env = dict(self.env)
        for name, value in env_overrides.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        self._sleep_log.unlink(missing_ok=True)

        proc = subprocess.run(
            ["bash", str(script)],
            env=env,
            cwd=self.cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = proc.stdout + proc.stderr
        assert not self._curl_log.exists(), "the script started curl"
        for value in CREDENTIALS.values():
            assert value not in output
        assert list(self.cwd.iterdir()) == [], "the script wrote outside its workspace"
        sleeps = self._sleep_log.read_text().split() if self._sleep_log.exists() else []
        return Result(proc.returncode, output, sleeps)

    def snapshot(self, **env_overrides: str | None) -> Result:
        return self.run(PROGRESS_SNAPSHOT, **env_overrides)

    def collect(self, job: FakeJob, **env_overrides: str | None) -> Result:
        return self.run(PROGRESS_COLLECT_SLURM, **{**job.env(), **env_overrides})


@pytest.fixture
def runner(tmp_path: Path) -> Runner:
    return Runner(tmp_path)


@pytest.fixture
def job(tmp_path: Path) -> FakeJob:
    return FakeJob(tmp_path)


@pytest.mark.parametrize(
    "script", [PROGRESS_SNAPSHOT, PROGRESS_COLLECT_SLURM], ids=lambda p: p.name
)
def test_scripts_mention_neither_curl_nor_a_credential(script: Path) -> None:
    text = script.read_text()
    for word in ("curl", *CREDENTIALS):
        assert not re.search(rf"\b{word}\b", text), f"{script.name} mentions {word}"


# progress_snapshot.sh


@pytest.mark.parametrize("given_hash_file", [False, True], ids=["default-hash-file", "given"])
def test_snapshot_writes_the_tarball_and_the_pending_hash(
    runner: Runner, tmp_path: Path, given_hash_file: bool
) -> None:
    _write_xml(runner.stage_dir / "results.xml")
    _write_unfinished(runner.stage_dir)
    overrides = {}
    hash_file = runner.hash_file
    if given_hash_file:
        hash_file = tmp_path / "state" / "checkpoint.hash"
        hash_file.parent.mkdir()
        overrides["PROGRESS_HASH_FILE"] = str(hash_file)

    result = runner.snapshot(**overrides)

    assert result.returncode == 0, result.output
    assert f"[PROGRESS-UPLOAD] {STAGE}: checkpoint tarball ready" in result.output
    assert set(_read_tarball(runner.tarball)) == {
        f"{STAGE}/results.xml",
        f"{STAGE}/unfinished_test.txt",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", Path(f"{hash_file}.pending").read_text().strip())
    # The pipeline moves the pending hash over the hash file once every target has the tarball.
    assert not hash_file.exists()


def test_snapshot_of_content_every_target_received_exits_3(runner: Runner) -> None:
    _write_xml(runner.stage_dir / "results.xml")
    assert runner.snapshot().returncode == 0
    runner.pending.replace(runner.hash_file)
    runner.tarball.unlink()

    result = runner.snapshot()

    assert result.returncode == 3, result.output
    assert f"[PROGRESS-UPLOAD] {STAGE}: checkpoint content unchanged" in result.output
    assert not runner.tarball.exists()
    assert not runner.pending.exists()


def test_snapshot_of_changed_content_exits_0(runner: Runner) -> None:
    _write_xml(runner.stage_dir / "results.xml")
    assert runner.snapshot().returncode == 0
    runner.pending.replace(runner.hash_file)
    received = runner.hash_file.read_text()
    _write_xml(runner.stage_dir / "results.xml", tests=2)

    result = runner.snapshot()

    assert result.returncode == 0, result.output
    assert runner.tarball.exists()
    assert runner.pending.read_text() != received
    assert runner.hash_file.read_text() == received


def test_snapshot_exits_0_again_until_the_pending_hash_is_moved(runner: Runner) -> None:
    _write_xml(runner.stage_dir / "results.xml")
    assert runner.snapshot().returncode == 0
    pending = runner.pending.read_text()
    runner.tarball.unlink()

    result = runner.snapshot()

    assert result.returncode == 0, result.output
    assert runner.tarball.exists()
    assert runner.pending.read_text() == pending
    assert not runner.hash_file.exists()


def test_snapshot_with_an_empty_hash_file_name_does_not_deduplicate(runner: Runner) -> None:
    _write_xml(runner.stage_dir / "results.xml")

    for _ in range(2):
        result = runner.snapshot(PROGRESS_HASH_FILE="")
        assert result.returncode == 0, result.output

    assert sorted(path.name for path in runner.workspace.iterdir()) == [STAGE, PROGRESS_TAR]


def test_snapshot_post_tag_renames_results_in_the_tarball_only(runner: Runner) -> None:
    _write_xml(runner.stage_dir / "results.xml")
    _write_xml(runner.stage_dir / "results-timeout.xml")
    _write_xml(runner.stage_dir / "aggr-1" / "results.xml")
    _write_unfinished(runner.stage_dir)

    result = runner.snapshot(POST_TAG=POST_TAG)

    assert result.returncode == 0, result.output
    assert set(_read_tarball(runner.tarball)) == {
        f"{STAGE}/results{POST_TAG}.xml",
        f"{STAGE}/results-timeout{POST_TAG}.xml",
        f"{STAGE}/aggr-1/results.xml",
        f"{STAGE}/unfinished_test.txt",
    }
    assert sorted(_contents(runner.stage_dir)) == [
        "aggr-1/results.xml",
        "results-timeout.xml",
        "results.xml",
        "unfinished_test.txt",
    ]


def test_final_snapshot_writes_timeout_results_for_the_unfinished_tests(runner: Runner) -> None:
    _write_xml(runner.stage_dir / "results.xml")
    _write_unfinished(runner.stage_dir)

    result = runner.snapshot(
        FINAL_SNAPSHOT="1", TIMEOUT_XML_SCRIPT=str(TIMEOUT_XML_SCRIPT), LABEL="final snapshot"
    )

    assert result.returncode == 0, result.output
    timeout_xml = (runner.stage_dir / "results-timeout.xml").read_bytes()
    assert f'<testsuite name="{STAGE}" errors="2"'.encode() in timeout_xml
    assert b'name="test_auto_dtype"' in timeout_xml
    assert b'name="test_unittests_v2[unittest/llmapi/test_llm.py]"' in timeout_xml
    assert _read_tarball(runner.tarball)[f"{STAGE}/results-timeout.xml"] == timeout_xml


@pytest.mark.parametrize("missing", ["FINAL_SNAPSHOT", "TIMEOUT_XML_SCRIPT", "unfinished_test.txt"])
def test_snapshot_writes_no_timeout_results_without_all_of_its_inputs(
    runner: Runner, missing: str
) -> None:
    _write_xml(runner.stage_dir / "results.xml")
    if missing != "unfinished_test.txt":
        _write_unfinished(runner.stage_dir)
    overrides = {"FINAL_SNAPSHOT": "1", "TIMEOUT_XML_SCRIPT": str(TIMEOUT_XML_SCRIPT)}
    overrides.pop(missing, None)

    result = runner.snapshot(**overrides)

    assert result.returncode == 0, result.output
    assert not (runner.stage_dir / "results-timeout.xml").exists()


def test_snapshot_names_pytest_testsuites_after_the_stage(runner: Runner) -> None:
    _write_xml(runner.stage_dir / "results.xml")
    _write_xml(runner.stage_dir / "aggr-1" / "results.xml")
    _write_xml(runner.stage_dir / "results-custom.xml", suite="custom")

    result = runner.snapshot()

    assert result.returncode == 0, result.output
    tarball = _read_tarball(runner.tarball)
    for name in ("results.xml", "aggr-1/results.xml"):
        on_disk = (runner.stage_dir / name).read_bytes()
        assert f'<testsuite name="{STAGE}"'.encode() in on_disk
        assert b'name="pytest"' not in on_disk
        assert tarball[f"{STAGE}/{name}"] == on_disk
    assert b'<testsuite name="custom"' in tarball[f"{STAGE}/results-custom.xml"]


def test_snapshot_that_cannot_write_the_tarball_exits_1(runner: Runner) -> None:
    _write_xml(runner.stage_dir / "results.xml")
    # tar cannot open a directory as its archive.
    runner.tarball.mkdir()

    result = runner.snapshot()

    assert result.returncode == 1, result.output
    assert f"[PROGRESS-UPLOAD] {STAGE}: checkpoint tar failed" in result.output
    assert not runner.pending.exists()


@pytest.mark.parametrize("variable", ["WORKSPACE", "STAGE_NAME", "PROGRESS_TAR", "LABEL"])
def test_snapshot_without_a_required_variable_exits_1(runner: Runner, variable: str) -> None:
    _write_xml(runner.stage_dir / "results.xml")
    # Without the check, a missing STAGE_NAME would have tar archive the file system's root.
    _write_stub(runner.stub_dir / "tar", "exit 2")
    before = _contents(runner.workspace)

    result = runner.snapshot(**{variable: None})

    assert result.returncode == 1, result.output
    assert f"{variable} is not set" in result.output
    assert _contents(runner.workspace) == before


def test_snapshot_leaves_credentials_out_of_the_tarball(runner: Runner) -> None:
    _write_xml(runner.stage_dir / "results.xml")
    _write_unfinished(runner.stage_dir)

    result = runner.snapshot(
        FINAL_SNAPSHOT="1", TIMEOUT_XML_SCRIPT=str(TIMEOUT_XML_SCRIPT), POST_TAG=POST_TAG
    )

    assert result.returncode == 0, result.output
    _assert_no_credentials(_read_tarball(runner.tarball))
    _assert_no_credentials(_contents(runner.workspace))


# progress_collect_slurm.sh


@pytest.mark.parametrize("recorded", [None, REMOTE_MTIME - 100], ids=["first-pass", "older"])
def test_collect_copies_new_results_and_records_their_mtime(
    runner: Runner, job: FakeJob, recorded: int | None
) -> None:
    if recorded is not None:
        job.mtime_file.write_text(f"{recorded}\n")

    result = runner.collect(job)

    assert result.returncode == 0, result.output
    assert f"[PROGRESS-UPLOAD] {STAGE}: mtime={REMOTE_MTIME} last={recorded or 0}" in result.output
    assert job.calls() == ["stat", "refresh", "xml", "unfinished", "list", "perf", "perf"]
    assert _contents(runner.stage_dir) == {
        "results.xml": (job.remote / "results.xml").read_bytes(),
        "unfinished_test.txt": (job.remote / "unfinished_test.txt").read_bytes(),
        "aggr-1/perf_result.csv": b"aggr-1\n",
        "disagg-2/perf_result.csv": b"disagg-2\n",
    }
    assert job.mtime_file.read_text().strip() == str(REMOTE_MTIME)
    assert result.sleeps == []


@pytest.mark.parametrize("recorded", [REMOTE_MTIME, REMOTE_MTIME + 100], ids=["same", "newer"])
def test_collect_without_a_newer_mtime_exits_3(runner: Runner, job: FakeJob, recorded: int) -> None:
    job.mtime_file.write_text(f"{recorded}\n")

    result = runner.collect(job)

    assert result.returncode == 3, result.output
    assert job.calls() == ["stat"]
    assert not runner.stage_dir.exists()
    assert job.mtime_file.read_text() == f"{recorded}\n"


def test_collect_reads_the_mtime_among_other_stat_output(runner: Runner, job: FakeJob) -> None:
    stat = f"echo 'Warning: Permanently added the host key' >&2; echo {REMOTE_MTIME}"

    result = runner.collect(job, SLURM_SSH_STAT_CMD=stat)

    assert result.returncode == 0, result.output
    assert job.mtime_file.read_text().strip() == str(REMOTE_MTIME)


def test_collect_without_a_numeric_mtime_logs_the_stat_output_and_exits_3(
    runner: Runner, job: FakeJob
) -> None:
    stat = "echo 'srun: error: Unable to create step for job 42' >&2; echo 0x1f"

    result = runner.collect(job, SLURM_SSH_STAT_CMD=stat)

    assert result.returncode == 3, result.output
    assert (
        f"[PROGRESS-UPLOAD] {STAGE}: stat cmd returned no numeric mtime, raw output: "
        "srun: error: Unable to create step for job 42"
    ) in result.output
    assert not runner.stage_dir.exists()
    assert not job.mtime_file.exists()


def test_collect_retries_the_xml_copy(runner: Runner, job: FakeJob, tmp_path: Path) -> None:
    attempts = shlex.quote(str(tmp_path / "attempts"))
    copy = (
        f'echo x >> {attempts}; [ "$(wc -l < {attempts})" -ge 3 ] && '
        f"cp {shlex.quote(str(job.remote))}/results*.xml {STAGE}/"
    )

    result = runner.collect(job, SLURM_SCP_XML_CMD=job.stub("xml", copy))

    assert result.returncode == 0, result.output
    assert job.calls().count("xml") == 3
    assert result.sleeps == ["10", "10"]
    assert (runner.stage_dir / "results.xml").exists()
    assert job.mtime_file.read_text().strip() == str(REMOTE_MTIME)


def test_collect_exits_1_when_the_xml_copy_fails_three_times(runner: Runner, job: FakeJob) -> None:
    job.mtime_file.write_text(f"{REMOTE_MTIME - 100}\n")

    result = runner.collect(job, SLURM_SCP_XML_CMD=job.stub("xml", "false"))

    assert result.returncode == 1, result.output
    assert f"[PROGRESS-UPLOAD] {STAGE}: scp xml failed after 3 attempts" in result.output
    assert job.calls() == ["stat", "refresh", "xml", "xml", "xml"]
    assert result.sleeps == ["10", "10"]
    assert job.mtime_file.read_text() == f"{REMOTE_MTIME - 100}\n"


def test_collect_skips_the_optional_copies_when_unset(runner: Runner, job: FakeJob) -> None:
    result = runner.collect(
        job,
        SLURM_SCP_UNFINISHED_CMD=None,
        SLURM_SSH_LIST_PERF_CMD=None,
        SLURM_SCP_PERF_TEMPLATE=None,
    )

    assert result.returncode == 0, result.output
    assert job.calls() == ["stat", "refresh", "xml"]
    assert sorted(_contents(runner.stage_dir)) == ["results.xml"]
    assert job.mtime_file.read_text().strip() == str(REMOTE_MTIME)


@pytest.mark.parametrize("unset", ["SLURM_SSH_LIST_PERF_CMD", "SLURM_SCP_PERF_TEMPLATE"])
def test_collect_copies_perf_folders_only_with_both_perf_commands(
    runner: Runner, job: FakeJob, unset: str
) -> None:
    result = runner.collect(job, **{unset: None})

    assert result.returncode == 0, result.output
    assert job.calls() == ["stat", "refresh", "xml", "unfinished"]
    assert sorted(_contents(runner.stage_dir)) == ["results.xml", "unfinished_test.txt"]


def test_collect_only_logs_failed_optional_copies(runner: Runner, job: FakeJob) -> None:
    result = runner.collect(
        job,
        SLURM_SCP_UNFINISHED_CMD=job.stub("unfinished", "false"),
        SLURM_SCP_PERF_TEMPLATE=job.stub("perf", "false"),
    )

    assert result.returncode == 0, result.output
    assert job.calls() == ["stat", "refresh", "xml"] + ["unfinished"] * 3 + ["list"] + ["perf"] * 6
    assert result.sleeps == ["10"] * 6
    assert f"[PROGRESS-UPLOAD] {STAGE}: scp unfinished not available" in result.output
    for folder in PERF_FOLDERS:
        assert f"scp perf {job.remote / folder} failed after 3 attempts" in result.output
    assert sorted(_contents(runner.stage_dir)) == ["results.xml"]
    assert job.mtime_file.read_text().strip() == str(REMOTE_MTIME)


def test_collect_runs_without_the_cache_refresh_command(runner: Runner, job: FakeJob) -> None:
    result = runner.collect(job, SLURM_SSH_REFRESH_CACHE_CMD=None)

    assert result.returncode == 0, result.output
    assert job.calls() == ["stat", "xml", "unfinished", "list", "perf", "perf"]
    assert (runner.stage_dir / "results.xml").exists()


@pytest.mark.parametrize(
    "variable",
    ["WORKSPACE", "STAGE_NAME", "SLURM_SSH_STAT_CMD", "SLURM_SCP_XML_CMD", "PROGRESS_MTIME_FILE"],
)
def test_collect_without_a_required_variable_exits_1(
    runner: Runner, job: FakeJob, variable: str
) -> None:
    result = runner.collect(job, **{variable: None})

    assert result.returncode == 1, result.output
    assert f"{variable} is not set" in result.output
    assert job.calls() == []
    assert not runner.stage_dir.exists()
    assert not job.mtime_file.exists()


def test_collect_leaves_credentials_out_of_what_it_writes(runner: Runner, job: FakeJob) -> None:
    result = runner.collect(job)

    assert result.returncode == 0, result.output
    _assert_no_credentials(_contents(runner.workspace))
    _assert_no_credentials({job.mtime_file.name: job.mtime_file.read_bytes()})
