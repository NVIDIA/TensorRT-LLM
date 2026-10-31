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

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASH_UTILS = REPO_ROOT / "jenkins" / "scripts" / "bash_utils.sh"
SLURM_INSTALL = REPO_ROOT / "jenkins" / "scripts" / "slurm_install.sh"
L0_TEST = REPO_ROOT / "jenkins" / "L0_Test.groovy"
BUILD_GROOVY = REPO_ROOT / "jenkins" / "Build.groovy"
# The pipelines that run the read helper against the aws location with a netrc they write.
BUILD_ARTIFACT_READERS = (
    L0_TEST,
    REPO_ROOT / "jenkins" / "BuildDockerImage.groovy",
    REPO_ROOT / "jenkins" / "BoltProfileGen.groovy",
    REPO_ROOT / "jenkins" / "L0_MergeRequest.groovy",
)
AWS_HOST = "artifactory.nvidia.com"

# The launcher copies the scripts into the job workspace under this per-job prefix.
JOB_PREFIX = "cluster-multi_node_test-0a1b2c-"
ARTIFACT_PATH = (
    "sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/1234/TensorRT-LLM.tar.gz"
)
CREDENTIAL_ID = "test-reader-credential"
NETRC_PASSWORD = "netrc-password-5d1e9c0a"
PAYLOAD = "packaged build contents\n"
# What the job script exports for the read helper.
HELPER_ENVIRONMENT = ("LLM_ARTIFACTS_PRIMARY", "ARTIFACTORY_NETRC", "ARTIFACTORY_CREDENTIAL_ID")

# Stands in for fetch_build_artifact.sh. It records its arguments, the environment it sees, and
# whether the netrc exists, and plays back the planned exit statuses, one per call; the last one
# repeats. A successful call writes the packaged build to <dest-file>; a failed one leaves a partial
# file there, which must never be moved into place.
_FETCH_STUB = r"""
import json
import os
import sys

with open(os.environ["FETCH_STUB_LOG"], "a", encoding="utf-8") as log:
    log.write(
        json.dumps(
            {
                "args": sys.argv[1:],
                "env": {name: os.environ.get(name) for name in json.loads(os.environ["FETCH_STUB_ENV"])},
                "netrc_present": os.path.isfile(os.environ.get("ARTIFACTORY_NETRC", "")),
            }
        )
        + "\n"
    )
with open(os.environ["FETCH_STUB_LOG"], encoding="utf-8") as log:
    call_index = len(log.readlines()) - 1
statuses = json.loads(os.environ["FETCH_STUB_STATUSES"])
status = statuses[min(call_index, len(statuses) - 1)]
with open(sys.argv[2], "w", encoding="utf-8") as dest:
    dest.write(os.environ["FETCH_STUB_PAYLOAD"] if status == 0 else "partial")
sys.exit(status)
"""


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize("with_timeout", (False, True))
def test_retry_command_returns_terminal_status(with_timeout: bool) -> None:
    env = os.environ.copy()
    env["BASH_UTILS"] = str(BASH_UTILS)
    env["WITH_TIMEOUT"] = "1" if with_timeout else "0"

    command = r"""
set -e
source "$BASH_UTILS"
attempt=0

sleep() {
    :
}

timeout() {
    [[ "${1-}" == "1800" ]] || exit 90
    shift
    "$@"
}

fail() {
    attempt=$((attempt + 1))
    return 23
}

retry_args=(3 0)
if [[ "$WITH_TIMEOUT" == "1" ]]; then
    retry_args+=(--timeout 1800)
fi

if retry_command "${retry_args[@]}" fail; then
    exit 99
else
    rc=$?
fi
[[ "$rc" -eq 23 ]]
[[ "$attempt" -eq 3 ]]
"""
    result = subprocess.run(
        ["bash", "-c", command], env=env, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize(
    ("retry_on", "attempts"), (("23", 3), ("7,124", 1)), ids=("listed", "not-listed")
)
def test_retry_command_retries_only_the_listed_statuses(retry_on: str, attempts: int) -> None:
    env = os.environ.copy()
    env["BASH_UTILS"] = str(BASH_UTILS)
    env["RETRY_ON"] = retry_on
    env["ATTEMPTS"] = str(attempts)

    command = r"""
set -e
source "$BASH_UTILS"
attempt=0

sleep() {
    :
}

fail() {
    attempt=$((attempt + 1))
    return 23
}

if retry_command 3 0 --retry-on "$RETRY_ON" fail; then
    exit 99
else
    rc=$?
fi
[[ "$rc" -eq 23 ]]
[[ "$attempt" -eq "$ATTEMPTS" ]]
"""
    result = subprocess.run(
        ["bash", "-c", command], env=env, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize(
    "missing", ("tarName", "llmTarArtifactPath", "resourcePathNode", "stageName", "pytestCommand")
)
def test_slurm_install_requires_artifact_inputs(missing: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "tarName": "/tmp/TensorRT-LLM.tar.gz",
            "llmTarArtifactPath": ARTIFACT_PATH,
            "resourcePathNode": "/tmp",
            "stageName": "test-stage",
            "pytestCommand": "pytest",
        }
    )
    env.pop(missing)

    result = subprocess.run(
        ["bash", str(SLURM_INSTALL)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert f"{missing} is required" in result.stderr


@dataclass
class _Run:
    """Outcome of running slurm_install.sh from a job workspace."""

    returncode: int
    stdout: str
    stderr: str
    fetch_calls: list[dict[str, object]]
    archive: Path
    netrc: Path
    netrc_at_extraction: bool

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def _run_slurm_install(tmp_path: Path, statuses: list[int], step_num_nodes: int = 1) -> _Run:
    """Run slurm_install.sh as a node of a Slurm job does, up to extracting the downloaded package.

    The script, bash_utils.sh, and a stub read helper sit in a job workspace under one per-job
    prefix, as the launcher copies them. A stub ``tar`` records its arguments and whether the
    test's netrc still exists, and exits 42, which ends the setup right after the download; a stub
    ``sleep`` makes the retry interval instant, and a stub ``timeout`` records each attempt's limit
    before running the attempt. The node is one of the ``step_num_nodes`` nodes of the step;
    SLURM_NODEID only names its temporary file.
    """
    workspace = tmp_path / "job-workspace"
    workspace.mkdir()
    install = workspace / f"{JOB_PREFIX}slurm_install.sh"
    shutil.copy(SLURM_INSTALL, install)
    shutil.copy(BASH_UTILS, workspace / f"{JOB_PREFIX}bash_utils.sh")
    stub_script = tmp_path / "fetch_stub.py"
    stub_script.write_text(_FETCH_STUB, encoding="utf-8")
    _write_executable(
        workspace / f"{JOB_PREFIX}fetch_build_artifact.sh",
        f'exec {shlex.quote(sys.executable)} -I -S {shlex.quote(str(stub_script))} "$@"\n',
    )

    bin_dir = tmp_path / "stub_bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "tar",
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$TAR_STUB_LOG"\n'
        'if [ -e "$TAR_STUB_NETRC" ]; then touch "$TAR_STUB_NETRC_SEEN"; fi\nexit 42\n',
    )
    _write_executable(bin_dir / "sleep", '#!/bin/bash\necho "$@" >> "$SLEEP_STUB_LOG"\n')
    _write_executable(
        bin_dir / "timeout", '#!/bin/bash\necho "$1" >> "$TIMEOUT_STUB_LOG"\nshift\nexec "$@"\n'
    )

    netrc = workspace / "artifactory-netrc"
    netrc.write_text(
        f"machine artifactory.nvidia.com login ci-reader password {NETRC_PASSWORD}\n",
        encoding="utf-8",
    )
    netrc.chmod(0o600)

    resource_dir = tmp_path / "resource"
    resource_dir.mkdir()
    fetch_log = tmp_path / "fetch_calls.jsonl"
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in (*HELPER_ENVIRONMENT, "llmTarfile", "GITHUB_CLONE_TOKEN", "BASH_ENV")
        and not name.startswith("SLURM_")
    }
    env.update(
        {
            "PATH": os.pathsep.join((str(bin_dir), env.get("PATH", os.defpath))),
            "HOME": str(tmp_path),
            "tarName": "TensorRT-LLM.tar.gz",
            "llmTarArtifactPath": ARTIFACT_PATH,
            "resourcePathNode": str(resource_dir),
            "stageName": "test-stage",
            "pytestCommand": "pytest",
            "SLURM_LOCALID": "0",
            "SLURM_JOB_ID": "4321",
            "SLURM_STEP_ID": "0",
            "SLURM_STEP_NUM_NODES": str(step_num_nodes),
            "SLURM_NODEID": "2",
            "LLM_ARTIFACTS_PRIMARY": "aws",
            "ARTIFACTORY_NETRC": str(netrc),
            "ARTIFACTORY_CREDENTIAL_ID": CREDENTIAL_ID,
            "FETCH_STUB_LOG": str(fetch_log),
            "FETCH_STUB_ENV": json.dumps(HELPER_ENVIRONMENT),
            "FETCH_STUB_STATUSES": json.dumps(statuses),
            "FETCH_STUB_PAYLOAD": PAYLOAD,
            "TAR_STUB_LOG": str(tmp_path / "tar_args.txt"),
            "TAR_STUB_NETRC": str(netrc),
            "TAR_STUB_NETRC_SEEN": str(tmp_path / "tar_saw_netrc"),
            "SLEEP_STUB_LOG": str(tmp_path / "sleep_args.txt"),
            "TIMEOUT_STUB_LOG": str(tmp_path / "timeout_args.txt"),
        }
    )
    result = subprocess.run(
        ["bash", str(install)], env=env, capture_output=True, text=True, timeout=60
    )
    fetch_calls = (
        [json.loads(line) for line in fetch_log.read_text(encoding="utf-8").splitlines()]
        if fetch_log.exists()
        else []
    )
    run = _Run(
        result.returncode,
        result.stdout,
        result.stderr,
        fetch_calls,
        resource_dir / "TensorRT-LLM.tar.gz",
        netrc,
        (tmp_path / "tar_saw_netrc").exists(),
    )

    # The job script traces its commands, and the netrc reaches the helper only by path.
    assert any(line.startswith("+") for line in run.stderr.splitlines())
    assert NETRC_PASSWORD not in run.output
    for call in run.fetch_calls:
        assert call["args"] == [ARTIFACT_PATH, f"{run.archive}.tmp.4321.2"]
        assert call["env"] == {
            "LLM_ARTIFACTS_PRIMARY": "aws",
            "ARTIFACTORY_NETRC": str(netrc),
            "ARTIFACTORY_CREDENTIAL_ID": CREDENTIAL_ID,
        }
        # Every attempt, retries included, still finds the netrc.
        assert call["netrc_present"]
    assert list(resource_dir.glob("*.tmp.*")) == []
    return run


def test_slurm_install_setup_downloads_package_through_retries(tmp_path: Path) -> None:
    run = _run_slurm_install(tmp_path, [4, 0])

    assert run.returncode == 42, run.output
    assert (tmp_path / "tar_args.txt").read_text(encoding="utf-8").split() == [
        "-zxf",
        str(run.archive),
    ]
    assert run.archive.read_text(encoding="utf-8") == PAYLOAD
    assert len(run.fetch_calls) == 2
    assert (tmp_path / "timeout_args.txt").read_text(encoding="utf-8").split() == ["1800", "1800"]
    assert len((tmp_path / "sleep_args.txt").read_text(encoding="utf-8").splitlines()) == 1


def test_slurm_install_setup_deletes_the_netrc_before_extracting_the_package(
    tmp_path: Path,
) -> None:
    run = _run_slurm_install(tmp_path, [4, 0])

    assert run.returncode == 42, run.output
    # On one node, the netrc is gone before the package is extracted.
    assert not run.netrc_at_extraction
    assert not run.netrc.exists()


def test_slurm_install_setup_fails_after_retrying_a_failed_transfer(tmp_path: Path) -> None:
    run = _run_slurm_install(tmp_path, [4])

    assert run.returncode == 1, run.output
    assert len(run.fetch_calls) == 3
    assert f"Artifact download failed: {ARTIFACT_PATH}" in run.stdout
    assert not run.archive.exists()
    assert not (tmp_path / "tar_args.txt").exists()
    assert not run.netrc.exists()


@pytest.mark.parametrize("status", (2, 3, 64), ids=("absent", "rejected", "usage-error"))
def test_slurm_install_setup_does_not_retry_a_final_failure(tmp_path: Path, status: int) -> None:
    run = _run_slurm_install(tmp_path, [status])

    assert run.returncode == 1, run.output
    # The helper's other failures are final: one attempt, and no wait.
    assert len(run.fetch_calls) == 1
    assert not (tmp_path / "sleep_args.txt").exists()
    assert f"Artifact download failed: {ARTIFACT_PATH}" in run.stdout
    assert not run.archive.exists()
    assert not (tmp_path / "tar_args.txt").exists()
    assert not run.netrc.exists()


def test_slurm_install_setup_retries_an_attempt_that_timed_out(tmp_path: Path) -> None:
    run = _run_slurm_install(tmp_path, [124, 0])

    assert run.returncode == 42, run.output
    assert len(run.fetch_calls) == 2
    assert "Command timed out after 1800s. Attempt 1/3." in run.stdout
    assert run.archive.read_text(encoding="utf-8") == PAYLOAD


def test_slurm_install_setup_leaves_a_shared_netrc_to_the_step_after_a_failure(
    tmp_path: Path,
) -> None:
    run = _run_slurm_install(tmp_path, [2], step_num_nodes=2)

    assert run.returncode == 1, run.output
    # The step's other node may still be downloading with it.
    assert run.netrc.exists()


def _write_netrc(path: Path) -> Path:
    path.write_text(
        f"machine {AWS_HOST} login ci-reader password {NETRC_PASSWORD}\n", encoding="utf-8"
    )
    path.chmod(0o600)
    return path


def _remove_netrc_on_node(netrc: Path, **slurm: str) -> subprocess.CompletedProcess[str]:
    """Run slurm_install.sh's netrc removal as a node of a Slurm step does after its download.

    ``slurm`` sets Slurm variables over node 0 of step 0 of job 4321, the job's first run.
    """
    env = {
        name: value
        for name, value in os.environ.items()
        if name != "BASH_ENV" and not name.startswith("SLURM_")
    }
    env.update(
        {
            "SLURM_INSTALL": str(SLURM_INSTALL),
            "ARTIFACTORY_NETRC": str(netrc),
            "SLURM_JOB_ID": "4321",
            "SLURM_STEP_ID": "0",
            "SLURM_NODEID": "0",
            **slurm,
        }
    )
    return subprocess.run(
        ["bash", "-c", 'source "$SLURM_INSTALL" && remove_artifactory_netrc'],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_slurm_install_deletes_a_shared_netrc_once_every_node_has_downloaded(
    tmp_path: Path,
) -> None:
    netrc = _write_netrc(tmp_path / "artifactory-netrc")

    # The step runs on 2 of the job's 4 nodes.
    first = _remove_netrc_on_node(
        netrc, SLURM_NODEID="1", SLURM_STEP_NUM_NODES="2", SLURM_JOB_NUM_NODES="4"
    )
    assert first.returncode == 0, first.stderr
    # The step's other node may still be downloading with it.
    assert netrc.exists()

    last = _remove_netrc_on_node(
        netrc, SLURM_NODEID="0", SLURM_STEP_NUM_NODES="2", SLURM_JOB_NUM_NODES="4"
    )
    assert last.returncode == 0, last.stderr
    assert not netrc.exists()


@pytest.mark.parametrize(
    ("earlier", "this"),
    (
        ({"SLURM_JOB_ID": "1234"}, {}),
        ({"SLURM_STEP_ID": "1"}, {}),
        ({}, {"SLURM_RESTART_COUNT": "1"}),
    ),
    ids=("other-job", "other-step", "requeued-run"),
)
def test_slurm_install_counts_only_the_nodes_of_its_own_run_of_the_step(
    tmp_path: Path, earlier: dict[str, str], this: dict[str, str]
) -> None:
    netrc = _write_netrc(tmp_path / "artifactory-netrc")

    other = _remove_netrc_on_node(netrc, SLURM_NODEID="1", SLURM_STEP_NUM_NODES="2", **earlier)
    assert other.returncode == 0, other.stderr

    this_node = _remove_netrc_on_node(netrc, SLURM_NODEID="0", SLURM_STEP_NUM_NODES="2", **this)
    assert this_node.returncode == 0, this_node.stderr
    # Node 1 of this run of the step has not finished its download yet.
    assert netrc.exists()


@pytest.mark.parametrize("step_num_nodes", ("two", "0"))
def test_slurm_install_leaves_the_netrc_on_an_unexpected_node_count(
    tmp_path: Path, step_num_nodes: str
) -> None:
    netrc = _write_netrc(tmp_path / "artifactory-netrc")

    run = _remove_netrc_on_node(netrc, SLURM_STEP_NUM_NODES=step_num_nodes)

    assert run.returncode == 0, run.stderr
    assert netrc.exists()


def _l0_test_code_lines() -> list[str]:
    """Return the code lines of jenkins/L0_Test.groovy, as _groovy_code_lines does."""
    return _groovy_code_lines(L0_TEST)


def _groovy_code(path: Path) -> str:
    """Return the code lines of a Groovy file, as _groovy_code_lines does, joined by newlines."""
    return "\n".join(_groovy_code_lines(path))


def _groovy_code_lines(path: Path) -> list[str]:
    """Return the code lines of a Groovy file.

    Whole-line and block comments are dropped, and backslash-continued lines are joined; a
    statement that spans lines without a backslash stays split across them.
    """
    code = []
    in_block_comment = False
    continued = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if in_block_comment:
            in_block_comment = "*/" not in stripped
            continue
        if stripped.startswith("/*"):
            in_block_comment = "*/" not in stripped
            continue
        if stripped.startswith("//"):
            continue
        if line.rstrip().endswith("\\"):
            continued += line.rstrip().rstrip("\\") + " "
            continue
        code.append(continued + line)
        continued = ""
    return code


def _groovy_function_body(lines: list[str], name: str) -> str:
    """Return the top-level Groovy function ``name``, up to the next top-level definition."""
    starts = [index for index, line in enumerate(lines) if re.match(rf"(def|void) {name}\(", line)]
    assert len(starts) == 1, f"expected one top-level definition of {name}"
    start = starts[0]
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith(("def ", "void "))
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _enclosing_blocks(lines: list[str], index: int) -> list[str]:
    """Return the openers of the blocks around ``lines[index]``, innermost first, by indentation."""
    openers = []
    indent = len(lines[index]) - len(lines[index].lstrip())
    for line in reversed(lines[:index]):
        stripped = line.strip()
        line_indent = len(line) - len(stripped) if stripped else indent
        if line_indent < indent and stripped.endswith("{"):
            openers.append(stripped)
            indent = line_indent
    return openers


def test_l0_reads_packaged_builds_through_fetch_helper() -> None:
    lines = _l0_test_code_lines()
    definitions = [line for line in lines if re.match(r"\s*def fetchBuildArtifact\(", line)]
    assert len(definitions) == 1
    body = _groovy_function_body(lines, "fetchBuildArtifact")
    assert "llmExecStepWithRetry" in body
    assert "readHelperPath(pipeline)" in body
    helper_body = _groovy_function_body(lines, "readHelperPath")
    assert "fetch_build_artifact.sh" in helper_body
    assert "unstash(FETCH_BUILD_ARTIFACT_STASH)" in helper_body


def _fetch_script(command: str, status_path: Path) -> str:
    r"""Build the script that fetchBuildArtifact runs around the read helper, from the Groovy source.

    The string is put together as Groovy does: its literals are joined, ``${name}`` takes the value
    of the local variable or top-level constant, and ``\$`` and ``\"`` become ``$`` and ``"``.
    """
    body = _groovy_function_body(_l0_test_code_lines(), "fetchBuildArtifact")
    start = body.index("def script = ")
    literals = []
    for line in body[start:].splitlines():
        literals += re.findall(r'"((?:[^"\\]|\\.)*)"', line)
        if not line.rstrip().endswith("+"):
            break
    values = {
        "command": command,
        "statusPath": str(status_path),
        **dict(
            re.findall(
                r"^(FETCH_[A-Z_]+) = (\d+)$", L0_TEST.read_text(encoding="utf-8"), re.MULTILINE
            )
        ),
    }
    script = re.sub(r"(?<!\\)\$\{(\w+)\}", lambda match: values[match[1]], "".join(literals))
    return script.replace('\\"', '"').replace("\\$", "$")


@pytest.mark.parametrize("status", (0, 2, 3, 4, 64))
def test_l0_fetch_repeats_only_a_failed_transfer(tmp_path: Path, status: int) -> None:
    status_path = tmp_path / "fetch-status"
    script = _fetch_script(f"sh -c 'exit {status}'", status_path)

    # Jenkins runs the script of an sh step with -xe.
    result = subprocess.run(["sh", "-xe", "-c", script], capture_output=True, text=True, timeout=60)

    if status == 4:
        # The script fails, so llmExecStepWithRetry repeats the step.
        assert result.returncode == 4, result.stderr
        assert not status_path.exists()
    else:
        # The script succeeds, and fetchBuildArtifact fails the step on the recorded status.
        assert result.returncode == 0, result.stderr
        assert status_path.read_text(encoding="utf-8").strip() == str(status)


@pytest.mark.parametrize(
    "reader",
    [
        "runLLMTestlistWithSbatch",
        "runLLMDocBuild",
        "runLLMTestlistOnPlatformImpl",
    ],
)
def test_l0_packaged_build_reader_downloads_through_fetch_helper(reader: str) -> None:
    body = _groovy_function_body(_l0_test_code_lines(), reader)

    # A download whose destination names the packaged build; an --exists probe has none.
    assert re.search(r"\bfetchBuildArtifact\(pipeline, [^,]+, [^,)]*tarName[^,)]*\)", body)


def test_l0_has_no_direct_download_of_packaged_builds() -> None:
    package_names = ("llmTarfile", "llmTarArtifactPath", "TARNAME", "tarName")
    package_downloads = [
        line.strip()
        for line in _l0_test_code_lines()
        if re.search(r"\b(wget|curl)\b", line) and any(name in line for name in package_names)
    ]

    assert package_downloads == []


def test_l0_slurm_job_gets_the_read_helper_and_its_netrc() -> None:
    lines = _l0_test_code_lines()
    body = _groovy_function_body(lines, "runLLMTestlistWithSbatch")
    body_lines = body.splitlines()

    # The helper from the run's commit goes beside slurm_install.sh, which finds it by its own
    # file name.
    assert (
        'def scriptFetchHelperPathNode = "${jobWorkspace}/${jobUID}-fetch_build_artifact.sh"'
        in body
    )
    assert "def scriptFetchHelperLocalPath = readHelperPath(pipeline)" in body
    assert re.search(
        r"Utils\.copyFileToRemoteHost\(\s*pipeline,\s*remote,\s*scriptFetchHelperLocalPath,"
        r"\s*scriptFetchHelperPathNode,\s*true\s*\)",
        body,
    )

    # Every job gets the helper's netrc, since the helper on the node contacts
    # artifactory.nvidia.com whenever aws is the primary, wherever the package is.
    netrc_writes = [
        index
        for index, line in enumerate(body_lines)
        if "writeArtifactoryNetrc(pipeline, artifactoryNetrcPathLocal)" in line
    ]
    assert len(netrc_writes) == 1
    openers = _enclosing_blocks(body_lines, netrc_writes[0])
    assert 'stage("Initialize Test") {' in openers
    within_stage = openers[: openers.index('stage("Initialize Test") {')]
    assert [opener for opener in within_stage if opener.startswith("if")] == []
    assert re.search(
        r"Utils\.copyFileToRemoteHost\(\s*pipeline,\s*remote,\s*artifactoryNetrcPathLocal,"
        r"\s*artifactoryNetrcPathNode\s*\)",
        body,
    )

    # Before the netrc reaches the job workspace, the attempt's job is registered for the off-pod
    # cleanup under an entry of its own, whatever an earlier attempt left registered.
    registrations = [
        index
        for index, line in enumerate(body_lines)
        if "registerSlurmResource(stageName, [clusterName: partition.clusterName, jobUID: jobUID,"
        " usedSbatch: true])" in line
    ]
    assert len(registrations) == 1
    assert registrations[0] < netrc_writes[0]
    openers = _enclosing_blocks(body_lines, registrations[0])
    within_stage = openers[: openers.index('stage("Initialize Test") {')]
    assert [opener for opener in within_stage if opener.startswith("if")] == []
    register = _groovy_function_body(lines, "registerSlurmResource")
    assert 'id: "${stageName}/slurm/${fields.jobUID ?: fields.nodeName}"' in register

    # The job script exports what the helper and slurm_install.sh read, with no condition.
    # llmTarfile is the URL that --exists resolved on the agent, for the older slurm_install.sh
    # inside a package built before the helper existed.
    exports = [line.strip() for line in body_lines if re.search(r"\bexport \w+=", line)]
    for name, value in (
        ("llmTarArtifactPath", "$llmTarArtifactPath"),
        ("LLM_ARTIFACTS_PRIMARY", "${llmArtifactsPrimary()}"),
        ("ARTIFACTORY_NETRC", "$artifactoryNetrcPathNode"),
        ("ARTIFACTORY_CREDENTIAL_ID", "$LLM_ARTIFACTS_CREDENTIALS_ID"),
        ("llmTarfile", "$llmTarfile"),
    ):
        assert [line for line in exports if re.search(rf"\bexport {name}=", line)] == [
            f"export {name}={value}"
        ]
    assert "llmTarfileSha1" not in body
    assert "llmTarfileOnAws" not in body

    # A retry that reuses the job workspace keeps both files.
    keep_list = re.search(r"def filesToKeepWhenRetry = \[(?P<items>[^\]]*)\]", body)
    assert keep_list
    kept = {item.strip() for item in keep_list.group("items").split(",")}
    assert {"scriptFetchHelperPathNode", "artifactoryNetrcPathNode"} <= kept

    # A netrc "default" entry would send the token to every host, including the CDN.
    netrc_writer = _groovy_function_body(lines, "writeArtifactoryNetrc")
    assert "printf 'machine artifactory.nvidia.com login %s password %s" in netrc_writer
    assert not re.search(r"\bdefault\b", netrc_writer)


def test_l0_reads_build_artifacts_and_pulls_images_with_separate_credential_settings() -> None:
    lines = _l0_test_code_lines()

    # The helper's netrc and the credential name that the helper prints come from one setting.
    netrc_writer = _groovy_function_body(lines, "writeArtifactoryNetrc")
    assert "credentialsId: LLM_ARTIFACTS_CREDENTIALS_ID," in netrc_writer
    fetch = _groovy_function_body(lines, "fetchBuildArtifact")
    assert '"ARTIFACTORY_CREDENTIAL_ID=${LLM_ARTIFACTS_CREDENTIALS_ID}",' in fetch
    # fetchBuildArtifact gives the helper the netrc that writeArtifactoryNetrc writes, as the Slurm
    # launcher does.
    assert "writeArtifactoryNetrc(pipeline, netrcPath)" in fetch
    assert '"ARTIFACTORY_NETRC=${netrcPath}",' in fetch

    # The enroot credentials of Slurm jobs and the docker login before an image pull keep the
    # image-pull setting.
    logins = [
        index
        for index, line in enumerate(lines)
        if "machine ${ARTIFACTORY_DOCKER_HOST} login" in line
        or "docker login ${ARTIFACTORY_DOCKER_HOST}" in line
    ]
    assert logins
    for index in logins:
        credential = next(line for line in reversed(lines[:index]) if "credentialsId:" in line)
        assert credential.strip() == "credentialsId: ARTIFACTORY_CREDENTIALS_ID,", lines[index]

    # That setting holds the image-pull identity, not the credential that publishes build artifacts.
    publishing = _artifact_location_field(_groovy_code(BUILD_GROOVY), "aws", "credential")
    assert _groovy_string("\n".join(lines), "ARTIFACTORY_CREDENTIALS_ID") != publishing


# A netrc entry that a pipeline writes with printf, which keeps the secret off the command line.
# The two arguments after the format are the login and password variables.
_NETRC_ENTRY = re.compile(r"printf 'machine (?P<host>\S+) login %s password %s")
_USERNAME_PASSWORD_BINDING = re.compile(
    r"usernamePassword\(\s*credentialsId:\s*(?P<credential>[^,]+?),\s*"
    r"usernameVariable:\s*'\w+',\s*passwordVariable:\s*'(?P<password>\w+)'\s*\)"
)


def _assignments(code: str, name: str) -> list[str]:
    """Return the values that a pipeline assigns to a variable, with or without def."""
    return re.findall(rf"^[ \t]*(?:def[ \t]+)?{name}[ \t]*=[ \t]*(.+?)[ \t]*$", code, re.M)


def _artifact_location_field(code: str, location: str, field: str) -> str:
    """Return a field of one location in a pipeline's LLM_ARTIFACTS_LOCATIONS map."""
    locations = re.search(r"^LLM_ARTIFACTS_LOCATIONS = \[$(?P<entries>.*?)^\]$", code, re.M | re.S)
    assert locations, "no LLM_ARTIFACTS_LOCATIONS map"
    entry = re.search(rf"^[ \t]*{location}: \[(?P<fields>[^\]]*)\]", locations["entries"], re.M)
    assert entry, f"no {location} entry in LLM_ARTIFACTS_LOCATIONS"
    value = re.search(rf'\b{field}: "(?P<value>[^"]*)"', entry["fields"])
    assert value, f"no {field} in the {location} entry of LLM_ARTIFACTS_LOCATIONS"
    return value["value"]


def _groovy_string(code: str, expression: str) -> str:
    """Return the string that a Groovy expression in a pipeline holds.

    The expression is a string literal, a field of LLM_ARTIFACTS_LOCATIONS, or a variable that the
    pipeline assigns once.
    """
    expression = expression.strip()
    literal = re.fullmatch(r"""(["'])(?P<value>[^"'$]*)\1""", expression)
    if literal:
        return literal["value"]
    field = re.fullmatch(r"LLM_ARTIFACTS_LOCATIONS\.(?P<location>\w+)\.(?P<field>\w+)", expression)
    if field:
        return _artifact_location_field(code, field["location"], field["field"])
    assert re.fullmatch(r"\w+", expression), f"cannot evaluate {expression!r}"
    values = _assignments(code, expression)
    assert len(values) == 1, f"expected one assignment of {expression}, found {values}"
    return _groovy_string(code, values[0])


def _names_aws_host(code: str, host: str) -> bool:
    """Tell whether a netrc entry's host is the aws location's.

    That is the literal artifactory.nvidia.com, or a variable whose every assignment derives from
    LLM_ARTIFACTS_LOCATIONS.aws.root. ARTIFACTORY_DOCKER_HOST names the container registry instead.
    """
    variable = re.fullmatch(r"\$\{?(?P<name>\w+)\}?", host)
    if variable is None:
        return host == AWS_HOST
    values = _assignments(code, variable["name"])
    return bool(values) and all(
        value.startswith("LLM_ARTIFACTS_LOCATIONS.aws.root") for value in values
    )


def _aws_netrc_credential(code: str) -> str:
    """Return the credentialsId expression behind the one netrc entry for the aws host.

    The entry prints its password from a variable, which the nearest usernamePassword binding of
    that variable before the entry fills.
    """
    entries = [
        entry for entry in _NETRC_ENTRY.finditer(code) if _names_aws_host(code, entry["host"])
    ]
    assert len(entries) == 1, f"expected one netrc entry for {AWS_HOST}, found {len(entries)}"
    (entry,) = entries
    arguments = code[entry.end() :].splitlines()[0]
    _, password = re.findall(r"\$\{?(\w+)", arguments)[:2]
    bindings = [
        binding
        for binding in _USERNAME_PASSWORD_BINDING.finditer(code, 0, entry.start())
        if binding["password"] == password
    ]
    assert bindings, f"no usernamePassword binding of {password} before the netrc entry"
    return bindings[-1]["credential"]


def _helper_credential_ids(code: str) -> list[str]:
    """Return the ARTIFACTORY_CREDENTIAL_ID values that a pipeline gives the read helper."""
    names = []
    for value in re.findall(r"ARTIFACTORY_CREDENTIAL_ID=([^\s\"]+)", code):
        value = value.strip("'")
        interpolation = re.fullmatch(r"\$\{?(?P<expression>[\w.]+)\}?", value)
        names.append(_groovy_string(code, interpolation["expression"]) if interpolation else value)
    return names


def test_build_uploads_to_aws_with_the_credential_of_its_location_map() -> None:
    code = _groovy_code(BUILD_GROOVY)

    # Build.groovy uploads packaged builds through its ArtifactoryAws server, so the location map's
    # aws credential, which the readers are compared with, is the one those uploads use.
    servers = re.findall(
        r"rtServer\s*\(\s*id:\s*'ArtifactoryAws',[^)]*?credentialsId:\s*'(?P<credential>[^']*)'",
        code,
    )
    assert servers
    assert set(servers) == {_artifact_location_field(code, "aws", "credential")}


@pytest.mark.parametrize("pipeline", BUILD_ARTIFACT_READERS, ids=lambda path: path.name)
def test_build_artifacts_are_read_with_the_publishing_credential(pipeline: Path) -> None:
    publishing = _artifact_location_field(_groovy_code(BUILD_GROOVY), "aws", "credential")
    code = _groovy_code(pipeline)

    # The netrc that the read helper gets holds the credential that Build.groovy publishes with.
    assert _groovy_string(code, _aws_netrc_credential(code)) == publishing

    # When aws rejects it, the read helper names that credential.
    named = _helper_credential_ids(code)
    assert named
    assert set(named) == {publishing}
