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
"""Tests for ``jenkins/scripts/progress_upload_watcher.sh``.

The watcher runs in the background of the step that runs a stage's tests. It reads its netrc into
memory and removes it, uploads a checkpoint of the stage's results to each target that lacks it on
every interval, and uploads the final snapshot once the done file appears.

The watcher runs with ``bash`` in a temporary workspace, with stubs on ``PATH``:

- ``sleep`` returns at once. A one-second sleep is the watcher's wait before a checkpoint, and
  before the n-th one the stub runs the shell snippet ``tick.<n>`` that a test wrote: it changes
  the results, or creates the done file. Other sleeps, the pauses between tries, are only recorded.
- ``curl`` records each request: its arguments and environment, the netrc it was given, the
  tarball it sent, and the state of the files around it. It answers with the status that the test
  queued for the request's location, 201 by default. A queued "000 <exit code>" stands for a
  request that got no response: the stub writes the message curl has for a timeout to stderr and
  exits with that code. No test makes a network request.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "jenkins" / "scripts"
WATCHER = SCRIPTS_DIR / "progress_upload_watcher.sh"
TIMEOUT_XML_SCRIPT = SCRIPTS_DIR / "generate_timeout_xml.py"

STAGE = "A10-PyTorch-1"
PROGRESS_TAR = f"results-{STAGE}-progress.tar.gz"
_RUN_PATH = f"llm-artifacts/LLM/main/L0_MergeRequest_PR/1/test-results/{PROGRESS_TAR}"
TARGETS = {
    "urm": f"https://urm.nvidia.com/artifactory/sw-tensorrt-generic/{_RUN_PATH}",
    "aws": f"https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local/{_RUN_PATH}",
}
PASSWORDS = {"urm": "urm-pass-4c1d9e-never-shown", "aws": "aws-pass-8b2f60-never-shown"}
NETRC_TEXT = (
    f"machine urm.nvidia.com login urm-user password {PASSWORDS['urm']}\n"
    f"machine artifactory.nvidia.com login aws-user password {PASSWORDS['aws']}\n"
)
REJECTED = {
    location: f"{location} rejected the credentials (HTTP HTTPSTATUS): renew the token."
    for location in TARGETS
}

# Variables of the watcher and its scripts that CI may have set for itself, and the files bash
# reads when it starts.
_DROPPED_VARIABLES = {
    "WORKSPACE",
    "STAGE_NAME",
    "LABEL",
    "LABEL_PREFIX",
    "POST_TAG",
    "XML_PATH",
    "TIMEOUT_XML_SCRIPT",
    "BASH_ENV",
    "ENV",
}
_DROPPED_PREFIXES = ("PROGRESS_", "SLURM_", "CHECKPOINT_", "FINAL_", "STUB_")

_CURL_STUB = r"""
# Records one request, and answers with the status queued for its location (201 by default). A
# queued "<status> <exit code>" also fails the way curl does when it gets no response.
state=$STUB_STATE
n=$(( $(cat "$state/curl.count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$state/curl.count"
request="$state/curl.$n"
mkdir "$request"
printf '%s\n' "$@" > "$request/args"
env > "$request/env"
netrc=""
upload=""
previous=""
url=""
for arg in "$@"; do
    case $previous in
        --netrc-file) netrc=$arg ;;
        -T) upload=$arg ;;
    esac
    previous=$arg
    url=$arg
done
printf '%s\n' "$netrc" > "$request/netrc_path"
cat "$netrc" > "$request/netrc" 2>/dev/null
cp "$upload" "$request/tarball"
if [ -e "$STUB_NETRC_DIR" ]; then echo present; else echo gone; fi > "$request/netrc_dir"
cat "$STUB_HASH_FILE" > "$request/last_hash" 2>/dev/null
if [ -n "$PROGRESS_STOP_FILE" ] && [ -e "$PROGRESS_STOP_FILE.busy" ]; then
    echo busy
else
    echo idle
fi > "$request/checkpoint_mark"
case $url in
    https://urm.nvidia.com/*) location=urm ;;
    https://artifactory.nvidia.com/*) location=aws ;;
    *) location=unknown ;;
esac
echo "$location" > "$request/location"
status=201
code=0
queue="$state/status.$location"
if [ -s "$queue" ]; then
    read -r status code < "$queue"
    tail -n +2 "$queue" > "$queue.rest"
    mv "$queue.rest" "$queue"
fi
printf '%s' "$status"
if [ "${code:-0}" != 0 ]; then
    echo "curl: ($code) Operation timed out after 300001 milliseconds with 0 bytes received" >&2
fi
exit "${code:-0}"
"""

_SLEEP_STUB = r"""
# Nothing sleeps. A one-second sleep is the watcher's wait before a checkpoint: before the n-th,
# run the test's tick.<n>. Other sleeps are only recorded.
state=$STUB_STATE
echo "$*" >> "$state/sleep.log"
[ "$*" = 1 ] || exit 0
n=$(( $(cat "$state/tick.count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$state/tick.count"
if [ -f "$state/tick.$n" ]; then
    . "$state/tick.$n"
fi
# A watcher that a test did not stop ends here.
if [ "$n" -ge 20 ]; then
    touch "$PROGRESS_DONE_FILE"
fi
exit 0
"""


@dataclass
class Request:
    location: str
    args: list[str]
    env: str
    netrc_path: str
    netrc: str
    netrc_dir: str
    last_hash: str
    checkpoint_mark: str
    tarball: dict[str, bytes]


@dataclass
class Result:
    returncode: int
    output: str
    requests: list[Request]
    sleeps: list[str]


def _write_stub(path: Path, body: str) -> None:
    """Write an executable ``sh`` stub with ``body`` at ``path``."""
    path.write_text(f"#!/bin/sh\n{body.lstrip()}")
    path.chmod(0o755)


def _write_xml(path: Path, tests: int = 1, mtime: int | None = None) -> None:
    """Write a results.xml whose test suite is named ``pytest``, optionally with the given mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites>'
        f'<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="{tests}">'
        "</testsuite></testsuites>\n"
    )
    if mtime is not None:
        os.utime(path, (mtime, mtime))


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


def _suite_tests(tarball: dict[str, bytes]) -> str:
    """Return the test count of the results.xml in a checkpoint."""
    match = re.search(rb'tests="(\d+)"', tarball[f"{STAGE}/results.xml"])
    assert match
    return match.group(1).decode()


def _set_mtime(path: Path, mtime: int) -> str:
    """Return the shell command that sets the mtime of ``path``."""
    return f"touch -d @{mtime} {shlex.quote(str(path))}"


def _dropped(name: str) -> bool:
    """Return whether a variable of CI's own environment is kept out of the watcher's."""
    return name in _DROPPED_VARIABLES or name.startswith(_DROPPED_PREFIXES)


class Watcher:
    """Runs the watcher with the stubs above in a temporary workspace (a pod stage by default)."""

    def __init__(self, tmp_path: Path) -> None:
        """Set up the workspace, the netrc, and the stubs.

        Args:
            tmp_path: The test's temporary directory, which holds everything the watcher touches.
        """
        self.tmp_path = tmp_path
        self.workspace = tmp_path / "workspace"
        self.stage_dir = self.workspace / STAGE
        self.xml = self.stage_dir / "results.xml"
        self.done = self.workspace / f".pytest-done-{STAGE}"
        self.tarball = self.workspace / PROGRESS_TAR
        self.hash_file = self.workspace / f"{PROGRESS_TAR}.last_hash"
        self.netrc_dir = tmp_path / "checkpoint-netrc"
        self.netrc = self.netrc_dir / "netrc"
        self.state = tmp_path / "state"
        self.cwd = tmp_path / "cwd"
        stub_dir = tmp_path / "bin"
        for directory in (self.workspace, self.state, self.cwd, stub_dir):
            directory.mkdir()
        self.netrc_dir.mkdir(mode=0o700)
        self.netrc.write_text(NETRC_TEXT)
        self.netrc.chmod(0o600)
        _write_stub(stub_dir / "curl", _CURL_STUB)
        _write_stub(stub_dir / "sleep", _SLEEP_STUB)

        env = {name: value for name, value in os.environ.items() if not _dropped(name)}
        env.update(
            {
                "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
                "STUB_STATE": str(self.state),
                "STUB_NETRC_DIR": str(self.netrc_dir),
                "STUB_HASH_FILE": str(self.hash_file),
                "WORKSPACE": str(self.workspace),
                "STAGE_NAME": STAGE,
                "PROGRESS_TAR": PROGRESS_TAR,
                "CHECKPOINT_NETRC_FILE": str(self.netrc),
                "CHECKPOINT_TARGETS": " ".join(f"{name}={url}" for name, url in TARGETS.items()),
                "PROGRESS_DONE_FILE": str(self.done),
                "PROGRESS_INTERVAL": "1",
                "LABEL_PREFIX": "checkpoint",
                "FINAL_LABEL": "run1 final snapshot",
                "XML_PATH": str(self.xml),
            }
        )
        env.update({f"CHECKPOINT_REJECTED_{name}": text for name, text in REJECTED.items()})
        self.env = env

    def tick(self, n: int, command: str) -> None:
        """Run ``command`` just before the watcher's n-th checkpoint."""
        with (self.state / f"tick.{n}").open("a") as tick:
            tick.write(f"{command}\n")

    def stop_at(self, n: int) -> None:
        """Create the done file in place of the watcher's n-th checkpoint."""
        self.tick(n, 'touch "$PROGRESS_DONE_FILE"')

    def answer(self, location: str, *statuses: str) -> None:
        """Queue the statuses of the next requests to ``location``.

        A status of the form "000 <exit code>" is a request that gets no response.
        """
        with (self.state / f"status.{location}").open("a") as queue:
            queue.writelines(f"{status}\n" for status in statuses)

    def staged_results(self, tests: int) -> Path:
        """Write a results.xml outside the workspace, for a tick to copy in."""
        path = self.tmp_path / f"staged-{tests}.xml"
        _write_xml(path, tests=tests)
        return path

    def run(self, **env_overrides: str | None) -> Result:
        """Run the watcher; an override of ``None`` removes that variable from its environment."""
        env = dict(self.env)
        for name, value in env_overrides.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value

        proc = subprocess.run(
            ["bash", str(WATCHER)],
            env=env,
            cwd=self.cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert list(self.cwd.iterdir()) == [], "the watcher wrote outside its workspace"
        sleep_log = self.state / "sleep.log"
        sleeps = sleep_log.read_text().split("\n")[:-1] if sleep_log.exists() else []
        return Result(proc.returncode, proc.stdout + proc.stderr, self._requests(), sleeps)

    def _requests(self) -> list[Request]:
        """Return the requests that the ``curl`` stub recorded, in order."""
        requests = []
        n = 1
        while (self.state / f"curl.{n}").is_dir():
            request = self.state / f"curl.{n}"

            def read(name: str, request: Path = request) -> str:
                path = request / name
                return path.read_text() if path.exists() else ""

            requests.append(
                Request(
                    location=read("location").strip(),
                    args=read("args").split("\n")[:-1],
                    env=read("env"),
                    netrc_path=read("netrc_path").strip(),
                    netrc=read("netrc"),
                    netrc_dir=read("netrc_dir").strip(),
                    last_hash=read("last_hash").strip(),
                    checkpoint_mark=read("checkpoint_mark").strip(),
                    tarball=_read_tarball(request / "tarball"),
                )
            )
            n += 1
        return requests

    def batch_env(self, **commands: str) -> dict[str, str | None]:
        """Return the overrides for a Slurm batch stage, whose SLURM_* commands are ``commands``."""
        return {
            "XML_PATH": None,
            "LABEL_PREFIX": "sbatch checkpoint",
            "FINAL_LABEL": "sbatch final snapshot",
            "SNAPSHOT_REQUIRES": str(self.xml),
            "FINAL_REQUIRES": str(self.xml),
            "PROGRESS_MTIME_FILE": str(self.tmp_path / "progress.mtime"),
            "SLURM_SSH_REFRESH_CACHE_CMD": "true",
            **commands,
        }


@pytest.fixture
def watcher(tmp_path: Path) -> Watcher:
    """Return a watcher runner for a pod stage on a temporary directory."""
    return Watcher(tmp_path)


def test_watcher_never_turns_on_xtrace() -> None:
    """The watcher's code never turns on xtrace, which would print its credentials."""
    text = WATCHER.read_text()
    assert text.startswith("#!/usr/bin/env bash\n")
    code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert "set +x" in code
    assert not any(re.search(r"\bset\s+-[A-Za-z]*x|\bbash\s+-[A-Za-z]*x", line) for line in code)


def test_netrc_is_read_and_removed_before_the_first_request(watcher: Watcher) -> None:
    """The netrc goes before the first request, and no argument, env, or file holds a password."""
    _write_xml(watcher.xml, mtime=1000)
    watcher.stop_at(2)

    result = watcher.run()

    assert result.returncode == 0, result.output
    assert f"[PROGRESS-UPLOAD] {STAGE}: credentials read; netrc removed" in result.output
    assert not watcher.netrc_dir.exists()
    assert [request.location for request in result.requests] == ["urm", "aws"]
    for request in result.requests:
        assert request.netrc_dir == "gone"
        # The credentials come as a stream, not as a file the tests could read.
        assert re.fullmatch(r"/(dev|proc/self)/fd/\d+", request.netrc_path), request.netrc_path
        assert request.netrc == NETRC_TEXT
        # -q first, so that no curlrc can turn on output that shows the Authorization header.
        assert request.args[0] == "-q"
        for password in PASSWORDS.values():
            assert password not in "\n".join(request.args)
            assert password not in request.env
    for password in PASSWORDS.values():
        assert password not in result.output
        for path in watcher.workspace.rglob("*"):
            if path.is_file():
                assert password.encode() not in path.read_bytes(), path


def test_each_changed_checkpoint_reaches_each_target_once(watcher: Watcher) -> None:
    """Changed results reach each target once, in order; unchanged results are not sent again."""
    _write_xml(watcher.xml, tests=1, mtime=1000)
    changed = watcher.staged_results(tests=2)
    # Before the second checkpoint the results are rewritten unchanged, before the third they
    # change, and before the fourth their mtime is the third's again.
    watcher.tick(2, _set_mtime(watcher.xml, 2000))
    watcher.tick(3, f"cp {shlex.quote(str(changed))} {shlex.quote(str(watcher.xml))}")
    watcher.tick(3, _set_mtime(watcher.xml, 3000))
    watcher.tick(4, _set_mtime(watcher.xml, 3000))
    watcher.stop_at(5)

    result = watcher.run()

    assert result.returncode == 0, result.output
    assert [request.location for request in result.requests] == ["urm", "aws", "urm", "aws"]
    assert [_suite_tests(request.tarball) for request in result.requests] == ["1", "1", "2", "2"]
    lines = result.output.splitlines()
    prefix = f"[PROGRESS-UPLOAD] {STAGE}: "
    assert f"{prefix}checkpoint (mtime=1000) uploaded to urm" in lines
    assert f"{prefix}checkpoint (mtime=2000) content unchanged, skipping upload" in lines
    assert [line for line in lines if "(mtime=3000)" in line] == [
        f"{prefix}checkpoint (mtime=3000) tarball ready",
        f"{prefix}checkpoint (mtime=3000) uploaded to urm",
        f"{prefix}checkpoint (mtime=3000) uploaded to aws",
    ]
    assert f"{prefix}run1 final snapshot content unchanged, skipping upload" in lines
    assert lines[-1] == f"{prefix}watcher exiting"


def test_a_failed_upload_is_sent_again_only_to_its_target(watcher: Watcher) -> None:
    """A target that missed a checkpoint gets it on the next interval; the other one does not."""
    _write_xml(watcher.xml, mtime=1000)
    watcher.answer("aws", "503", "503", "503")
    # The results do not change after the first checkpoint.
    watcher.tick(2, _set_mtime(watcher.xml, 1000))
    watcher.stop_at(3)

    result = watcher.run()

    assert result.returncode == 0, result.output
    # Three tries, ten seconds apart, in the first checkpoint, and one in the second, although the
    # results did not change in between.
    assert [request.location for request in result.requests] == ["urm", "aws", "aws", "aws", "aws"]
    assert result.sleeps.count("10") == 2
    lines = result.output.splitlines()
    prefix = f"[PROGRESS-UPLOAD] {STAGE}: "
    assert f"{prefix}checkpoint (mtime=1000) upload to aws failed (non-fatal): HTTP 503" in lines
    assert lines.count(f"{prefix}checkpoint (mtime=1000) uploaded to aws") == 1
    assert f"{prefix}run1 final snapshot content unchanged, skipping upload" in lines
    assert len({tuple(sorted(request.tarball.items())) for request in result.requests}) == 1
    # The content becomes the last checkpoint's only once both targets have it.
    assert [request.last_hash for request in result.requests] == [""] * 5
    received = {
        location: (watcher.workspace / f"{PROGRESS_TAR}.last_hash.{location}").read_text()
        for location in TARGETS
    }
    assert received["urm"] == received["aws"] == watcher.hash_file.read_text()


def test_an_upload_without_a_response_logs_only_the_exit_code_of_curl(watcher: Watcher) -> None:
    """An upload that gets no response is logged with curl's exit code, not with curl's message."""
    _write_xml(watcher.xml, mtime=1000)
    # curl gives up on a stalled transfer, on each of the three tries.
    watcher.answer("aws", "000 28", "000 28", "000 28")
    watcher.stop_at(2)

    result = watcher.run()

    assert result.returncode == 0, result.output
    # The final snapshot sends aws the content it lacks.
    assert [request.location for request in result.requests] == ["urm", "aws", "aws", "aws", "aws"]
    assert result.sleeps.count("10") == 2
    lines = result.output.splitlines()
    prefix = f"[PROGRESS-UPLOAD] {STAGE}: "
    failed = f"{prefix}checkpoint (mtime=1000) upload to aws failed (non-fatal): HTTP 000"
    assert lines.count(f"{failed} (curl exit 28)") == 1
    assert f"{prefix}run1 final snapshot uploaded to aws" in lines
    # The pipeline takes curl's message in the log of a failed step for an infrastructure failure.
    assert "curl: (" not in result.output
    assert "timed out" not in result.output.lower()


@pytest.mark.parametrize("status", ["401", "403"])
def test_a_rejected_upload_logs_the_credential_message(watcher: Watcher, status: str) -> None:
    """A 401 or 403 logs the location's credential message with its status; the watcher goes on."""
    _write_xml(watcher.xml, mtime=1000)
    watcher.answer("urm", status)
    watcher.stop_at(2)

    result = watcher.run()

    assert result.returncode == 0, result.output
    message = REJECTED["urm"].replace("HTTPSTATUS", status)
    assert (
        f"checkpoint (mtime=1000) upload to urm failed (non-fatal): HTTP {status}. {message}"
        in result.output
    )
    # A rejection is not tried again at once. aws still receives the checkpoint, and the final
    # snapshot sends urm the content it lacks.
    assert [request.location for request in result.requests] == ["urm", "aws", "urm"]
    assert "run1 final snapshot uploaded to urm" in result.output


def test_the_done_file_brings_the_final_snapshot(watcher: Watcher) -> None:
    """The done file brings the final snapshot, with its timeout results, to both targets."""
    _write_xml(watcher.xml, mtime=1000)
    (watcher.stage_dir / "unfinished_test.txt").write_text("unittest/llmapi/test_llm.py\n")
    watcher.stop_at(1)

    result = watcher.run(
        TIMEOUT_XML_SCRIPT=str(TIMEOUT_XML_SCRIPT), FINAL_REQUIRES=str(watcher.xml)
    )

    assert result.returncode == 0, result.output
    assert [request.location for request in result.requests] == ["urm", "aws"]
    for request in result.requests:
        assert f"{STAGE}/results-timeout.xml" in request.tarball
    assert "run1 final snapshot uploaded to aws" in result.output
    assert result.output.splitlines()[-1] == f"[PROGRESS-UPLOAD] {STAGE}: watcher exiting"
    assert watcher.hash_file.exists()


def test_no_final_snapshot_without_the_file_it_requires(watcher: Watcher) -> None:
    """No final snapshot is taken while the file it requires is missing."""
    _write_xml(watcher.xml, mtime=1000)
    watcher.stop_at(1)

    result = watcher.run(FINAL_REQUIRES=str(watcher.stage_dir / "missing.xml"))

    assert result.returncode == 0, result.output
    assert result.requests == []
    assert not watcher.tarball.exists()
    assert "final snapshot" not in result.output


@pytest.mark.parametrize("netrc_file", ["missing", "unset"])
def test_without_credentials_no_checkpoint_is_uploaded(watcher: Watcher, netrc_file: str) -> None:
    """Without a readable netrc nothing is uploaded, yet snapshots still rename the test suite."""
    _write_xml(watcher.xml, mtime=1000)
    watcher.stop_at(2)
    missing = str(watcher.tmp_path / "no-netrc" / "netrc")

    result = watcher.run(CHECKPOINT_NETRC_FILE=missing if netrc_file == "missing" else None)

    assert result.returncode == 0, result.output
    assert (
        f"[PROGRESS-UPLOAD] {STAGE}: no credentials, no checkpoint uploads in this step"
        in result.output
    )
    assert result.requests == []
    # No target received the content, so it is not recorded as the last checkpoint's: a later step
    # that has credentials still uploads it.
    assert not watcher.hash_file.exists()
    # The snapshots still rename the test suite, which a Slurm batch stage's final archive keeps.
    assert f'testsuite name="{STAGE}"' in watcher.xml.read_text()
    assert result.output.splitlines()[-1] == f"[PROGRESS-UPLOAD] {STAGE}: watcher exiting"


def test_a_failed_copy_of_a_batch_jobs_results_takes_no_checkpoint(watcher: Watcher) -> None:
    """A failed copy of a Slurm batch job's results takes no checkpoint."""
    watcher.stop_at(2)

    result = watcher.run(
        **watcher.batch_env(SLURM_SSH_STAT_CMD="echo 5000", SLURM_SCP_XML_CMD="false")
    )

    assert result.returncode == 0, result.output
    assert (
        "sbatch checkpoint skipped: copying the results back failed "
        "(progress_collect_slurm.sh exited 1)" in result.output
    )
    assert result.requests == []
    assert not watcher.tarball.exists()


@pytest.mark.parametrize("stage_folder", ["missing", "without results"])
def test_a_batch_stage_takes_no_checkpoint_before_its_job_writes_results(
    watcher: Watcher, stage_folder: str
) -> None:
    """No snapshot is tried before a batch job writes results.xml; its first results get one."""
    if stage_folder == "without results":
        # The lookup of tests to reuse leaves the stage's folder with only this list in it.
        watcher.stage_dir.mkdir()
        (watcher.stage_dir / "passed_test_list.txt").write_text("")
    remote = watcher.tmp_path / "remote"
    remote.mkdir()
    remote_mtime = watcher.tmp_path / "remote.mtime"
    # The stat command prints 0 while the job has not written results.xml.
    remote_mtime.write_text("0\n")
    results = watcher.staged_results(tests=4)
    early = watcher.state / "early-tarball"
    # Before the second and the third checkpoint, record whether an earlier one left a tarball.
    # Then, still before the third, the job writes its results.
    for n in (2, 3):
        watcher.tick(
            n, f"[ -e {shlex.quote(str(watcher.tarball))} ] && touch {shlex.quote(str(early))}"
        )
    watcher.tick(3, f"cp {shlex.quote(str(results))} {shlex.quote(str(remote / 'results.xml'))}")
    watcher.tick(3, f"echo 5000 > {shlex.quote(str(remote_mtime))}")
    watcher.stop_at(4)

    result = watcher.run(
        **watcher.batch_env(
            SLURM_SSH_STAT_CMD=f"cat {shlex.quote(str(remote_mtime))}",
            SLURM_SCP_XML_CMD=f"cp {shlex.quote(str(remote))}/results*.xml {STAGE}/",
        )
    )

    assert result.returncode == 0, result.output
    assert not early.exists(), "a tarball was written before the job's results existed"
    for text in ("tar failed", "not prepared", "Cannot stat", "No such file"):
        assert text not in result.output
    # The only checkpoint is the one with the job's first results.
    assert [request.location for request in result.requests] == ["urm", "aws"]
    assert [_suite_tests(request.tarball) for request in result.requests] == ["4", "4"]


def test_a_batch_stage_takes_no_checkpoint_once_its_step_asks_it_to_stop(
    watcher: Watcher,
) -> None:
    """Once the stop file exists no checkpoint starts, and one under way carries the busy mark."""
    remote = watcher.tmp_path / "remote"
    _write_xml(remote / "results.xml", tests=3)
    stats = watcher.tmp_path / "stat.log"
    stop = watcher.workspace / f".checkpoint-stop-{STAGE}"
    # The step creates the stop file before the second checkpoint, and the done file before the
    # third.
    watcher.tick(2, f"touch {shlex.quote(str(stop))}")
    watcher.stop_at(3)

    result = watcher.run(
        PROGRESS_STOP_FILE=str(stop),
        **watcher.batch_env(
            SLURM_SSH_STAT_CMD=f"echo stat >> {shlex.quote(str(stats))}; echo 5000",
            SLURM_SCP_XML_CMD=f"cp {shlex.quote(str(remote))}/results*.xml {STAGE}/",
        ),
    )

    assert result.returncode == 0, result.output
    # Only the first checkpoint copied the results back, and it was marked while under way.
    assert stats.read_text().split() == ["stat"]
    assert [request.location for request in result.requests] == ["urm", "aws"]
    assert [request.checkpoint_mark for request in result.requests] == ["busy", "busy"]
    assert [_suite_tests(request.tarball) for request in result.requests] == ["3", "3"]
    assert not Path(f"{stop}.busy").exists()
    # The final snapshot still comes with the done file.
    assert "sbatch final snapshot content unchanged, skipping upload" in result.output


def _ticks(watcher: Watcher) -> int:
    """Return how many times the watcher waited for its next checkpoint."""
    return int((watcher.state / "tick.count").read_text())


def test_a_signal_brings_the_final_snapshot_as_the_done_file_does(watcher: Watcher) -> None:
    """USR1, from a step that cannot create the done file, brings the final snapshot and the exit."""
    _write_xml(watcher.xml, mtime=1000)
    # The step signals in place of the watcher's first checkpoint. The done file never appears.
    watcher.tick(1, "kill -s USR1 $PPID")

    result = watcher.run(FINAL_REQUIRES=str(watcher.xml))

    assert result.returncode == 0, result.output
    assert not watcher.done.exists()
    assert _ticks(watcher) == 1
    assert [request.location for request in result.requests] == ["urm", "aws"]
    lines = result.output.splitlines()
    signalled = lines.index(
        f"[PROGRESS-UPLOAD] {STAGE}: the step signalled the end of the tests without the done file"
    )
    uploaded = f"[PROGRESS-UPLOAD] {STAGE}: run1 final snapshot uploaded to aws"
    assert uploaded in lines[signalled + 1 :]
    assert lines[-1] == f"[PROGRESS-UPLOAD] {STAGE}: watcher exiting"


def test_a_signal_during_a_checkpoint_lets_it_finish(watcher: Watcher) -> None:
    """A signal that arrives while a checkpoint is under way does not cut the checkpoint short."""
    remote = watcher.tmp_path / "remote"
    _write_xml(remote / "results.xml", tests=3)
    pid_file = shlex.quote(str(watcher.state / "watcher.pid"))
    stop = watcher.workspace / f".checkpoint-stop-{STAGE}"
    watcher.tick(1, f"echo $PPID > {pid_file}")

    # The signal arrives while the first checkpoint asks the job for its results.
    result = watcher.run(
        PROGRESS_STOP_FILE=str(stop),
        **watcher.batch_env(
            SLURM_SSH_STAT_CMD=f"kill -s USR1 $(cat {pid_file}); echo 5000",
            SLURM_SCP_XML_CMD=f"cp {shlex.quote(str(remote))}/results*.xml {STAGE}/",
        ),
    )

    assert result.returncode == 0, result.output
    assert _ticks(watcher) == 1
    # That checkpoint still copied the results back and reached both targets, under its mark.
    assert [request.location for request in result.requests] == ["urm", "aws"]
    assert [request.checkpoint_mark for request in result.requests] == ["busy", "busy"]
    assert [_suite_tests(request.tarball) for request in result.requests] == ["3", "3"]
    assert not Path(f"{stop}.busy").exists()
    # The final snapshot follows it, with neither the stop file nor the done file.
    assert not stop.exists()
    assert not watcher.done.exists()
    assert "sbatch final snapshot content unchanged, skipping upload" in result.output
    assert result.output.splitlines()[-1] == f"[PROGRESS-UPLOAD] {STAGE}: watcher exiting"


L0_TEST = REPO_ROOT / "jenkins" / "L0_Test.groovy"


def _step_lines(first: str, last: str) -> list[str]:
    r"""Return each run of lines from ``first`` to ``last`` in a step of L0_Test.groovy.

    The lines come back as the shell gets them, without the ``\`` that keeps Groovy from expanding
    a ``$``.
    """
    pattern = rf"^[ \t]*{re.escape(first)}.*?^[ \t]*{re.escape(last)}$"
    found = re.findall(pattern, L0_TEST.read_text(encoding="utf-8"), re.M | re.S)
    return [lines.replace("\\$", "$") for lines in found]


def _run_beside_the_watcher(watcher: Watcher, lines: str, **env: str) -> tuple[int, str]:
    """Run ``lines`` in a step that has the watcher in its background, as ``sh`` does.

    Returns the step's exit code and output. A step that does not return is killed, and so is a
    watcher that outlives its step.
    """
    script = "\n".join(
        [
            "set +e",
            f"bash {shlex.quote(str(WATCHER))} &",
            "WATCHER_PID=$!",
            # As in the steps, the tests start only once the watcher has removed its netrc.
            'while [ -e "$CHECKPOINT_NETRC_FILE" ]; do :; done',
            "rc=7",
            lines,
        ]
    )
    proc = subprocess.Popen(
        ["sh", "-c", script],
        env={**watcher.env, **env},
        cwd=watcher.cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        pytest.fail(f"the step did not return:\n{output}")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return proc.returncode, output


@pytest.mark.parametrize("step", range(3), ids=["batch", "rerun", "first run"])
def test_a_step_that_cannot_create_the_done_file_still_returns(watcher: Watcher, step: int) -> None:
    """Each step returns the tests' exit code although its done file cannot be created.

    The watcher exits only once it learns that the tests have ended, and the step waits for it, so
    a step that could only signal through the done file would never return.
    """
    endings = _step_lines('touch "\\$PROGRESS_DONE_FILE"', "exit \\$rc")
    assert len(endings) == 3
    _write_xml(watcher.xml, mtime=1000)

    returncode, output = _run_beside_the_watcher(
        watcher,
        endings[step],
        PROGRESS_DONE_FILE=str(watcher.workspace / "missing" / "done"),
        FINAL_REQUIRES=str(watcher.xml),
    )

    assert returncode == 7, output
    assert "the step signalled the end of the tests without the done file" in output
    assert "run1 final snapshot" in output
    assert output.splitlines()[-1] == f"[PROGRESS-UPLOAD] {STAGE}: watcher exiting"


def test_a_batch_step_that_cannot_create_the_stop_file_ends_the_watcher_first(
    watcher: Watcher,
) -> None:
    """Without the stop file, the batch step ends the watcher before it copies the last results.

    The watcher cannot be asked to pause then, and a mark of a checkpoint under way that could not
    be removed would otherwise keep the step waiting.
    """
    (stop_block,) = _step_lines('if ! touch "\\$PROGRESS_STOP_FILE"; then', "done")
    _write_xml(watcher.xml, mtime=1000)

    returncode, output = _run_beside_the_watcher(
        watcher,
        f'{stop_block}\nkill -0 $WATCHER_PID 2>/dev/null && echo "the watcher still runs"\nexit $rc',
        PROGRESS_STOP_FILE=str(watcher.workspace / "missing" / "stop"),
        FINAL_REQUIRES=str(watcher.xml),
    )

    assert returncode == 7, output
    assert "the watcher still runs" not in output
    assert f"[PROGRESS-UPLOAD] {STAGE}: watcher exiting" in output
