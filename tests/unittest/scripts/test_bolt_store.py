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
"""Tests for the reads of the BOLT profile store.

``scripts/bolt/internal/artifactory.sh`` reads a branch's promoted profile bundle
(``pull-latest``) and the ref that ``latest.tar.gz`` names (``resolve-latest``) through
``jenkins/scripts/fetch_build_artifact.sh``, and ``scripts/bolt/internal/apply_latest.sh`` applies
the bundle that ``pull-latest`` unpacked. The tests copy each script into a temporary tree beside
stubs of the programs it runs, so no request leaves the host: a stub read helper records its
arguments and the environment it sees and plays back planned exit statuses, and ``curl`` and
``wget`` stubs on ``PATH`` record any attempt to start them.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import ANY

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BOLT_INTERNAL = REPO_ROOT / "scripts" / "bolt" / "internal"
ARTIFACTORY_SH = BOLT_INTERNAL / "artifactory.sh"
APPLY_LATEST_SH = BOLT_INTERNAL / "apply_latest.sh"

BRANCH = "main"
TRIPLE = "aarch64-linux-gnu"
REF = "0a1b2c3d4e5f"
# The legacy path that the reads give the helper, which maps it to each location.
STORE_DIR = f"sw-tensorrt-generic/llm-artifacts/bolt-profiles/{BRANCH}/{TRIPLE}"
LATEST_PATH = f"{STORE_DIR}/latest.tar.gz"
PINNED_PATH = f"{STORE_DIR}/bolt-profile-{REF}-{TRIPLE}.tar.gz"
# The hosts of the two locations, and what the helper's --exists prints when aws holds
# latest.tar.gz for readers.
AWS_HOST = "artifactory.test"
URM_HOST = "urm.test"
HELD_ON_AWS = (
    f"https://{AWS_HOST}/artifactory/sw-tensorrt-llm-generic-local/"
    f"{LATEST_PATH.split('/', 1)[1]} {'0' * 40}"
)

# The read helper's exit statuses other than success, and those of the store's readers.
FETCH_ABSENT = 2
FETCH_REJECTED = 3
FETCH_TRANSFER_FAILED = 4
FETCH_USAGE = 64
BOLT_NO_BUNDLE = 3
BOLT_STORE_UNREADABLE = 4
# The helper's failures that say the store could not be read, by name.
UNREADABLE = {
    "rejected": FETCH_REJECTED,
    "transfer-failed": FETCH_TRANSFER_FAILED,
    "usage": FETCH_USAGE,
}

# What the helper reads from its caller's environment.
HELPER_ENVIRONMENT = ("LLM_ARTIFACTS_PRIMARY", "ARTIFACTORY_NETRC", "ARTIFACTORY_CREDENTIAL_ID")
CREDENTIAL_ID = "test-publishing-credential"
# Settings that would change what the scripts under test ask for.
_SCRIPT_SETTINGS = ("BOLT_PROFILE_REF", "BOLT_ARTIFACTORY_REPO", "BOLT_PROFILE_PREFIX", "BASH_ENV")

PROFILE = "libtensorrt_llm.so.yaml"
PROFILE_TEXT = "functions: []\n"

# Stands in for fetch_build_artifact.sh. It records its arguments and the environment it sees, and
# plays back the planned answers, one per call; the last one repeats. An answer is an exit status
# and the file that a successful call copies to <dest-file>, the helper's last argument. For
# --exists, which has no <dest-file>, the answer is the line that a successful call prints.
_HELPER_STUB = r"""
import json
import os
import shutil
import sys

log_path = os.environ["HELPER_STUB_LOG"]
with open(log_path, "a", encoding="utf-8") as log:
    names = json.loads(os.environ["HELPER_STUB_ENV"])
    record = {"args": sys.argv[1:], "env": {name: os.environ.get(name) for name in names}}
    log.write(json.dumps(record) + "\n")
with open(log_path, encoding="utf-8") as log:
    call_index = len(log.readlines()) - 1
answers = json.loads(os.environ["HELPER_STUB_ANSWERS"])
status, payload = answers[min(call_index, len(answers) - 1)]
if status == 0:
    if "--exists" in sys.argv:
        print(payload)
    else:
        shutil.copyfile(payload, sys.argv[-1])
sys.exit(status)
"""

# Stands in for curl and wget: the scripts under test must start neither.
_DOWNLOADER_STUB = '#!/bin/bash\necho "$0 $*" >> "$DOWNLOADER_STUB_LOG"\nexit 97\n'

# Stands in for artifactory.sh beside apply_latest.sh. A successful pull-latest unpacks a profile
# and a manifest into <dest_dir>, its last argument.
_ARTIFACTORY_STUB = r"""#!/bin/bash
printf '%s\n' "$@" > "$ARTIFACTORY_STUB_LOG"
if [ "$ARTIFACTORY_STUB_STATUS" -eq 0 ]; then
    echo '{}' > "${!#}/manifest.json"
    echo 'functions: []' > "${!#}/profile.yaml"
fi
exit "$ARTIFACTORY_STUB_STATUS"
"""

# Stands in for apply_bolt.py. A successful apply writes the file named after --output.
_APPLY_BOLT_STUB = r"""
import json
import os
import sys

with open(os.environ["APPLY_STUB_LOG"], "w", encoding="utf-8") as log:
    json.dump(sys.argv[1:], log)
status = int(os.environ["APPLY_STUB_STATUS"])
if status == 0:
    with open(sys.argv[sys.argv.index("--output") + 1], "w", encoding="utf-8") as output:
        output.write("bolted\n")
sys.exit(status)
"""


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _bundle(path: Path, manifest: dict[str, object] | None) -> Path:
    """Write a profile bundle: one profile and, unless ``manifest`` is None, manifest.json."""
    members = {PROFILE: PROFILE_TEXT}
    if manifest is not None:
        members["manifest.json"] = json.dumps(manifest, indent=2)
    with tarfile.open(path, "w:gz") as archive:
        for name, text in members.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def _properties(path: Path, properties: dict[str, str], host: str = AWS_HOST) -> Path:
    """Write a location's storage API answer for a file's properties, as Artifactory sends it."""
    answer = {
        "properties": {name: [value] for name, value in properties.items()},
        "uri": f"https://{host}/artifactory/api/storage/{LATEST_PATH}",
    }
    path.write_text(json.dumps(answer, indent=2, separators=(",", " : ")) + "\n", encoding="utf-8")
    return path


@dataclass
class HelperCall:
    """One call of the stub read helper: its arguments and the environment it saw."""

    args: list[str]
    env: dict[str, str | None]


@dataclass
class Result:
    """The outcome of one run of a script under test."""

    returncode: int
    stdout: str
    stderr: str
    helper_calls: list[HelperCall]

    @property
    def helper_args(self) -> list[list[str]]:
        return [call.args for call in self.helper_calls]


class Tree:
    """A temporary repository tree that holds the scripts under test beside their stubs.

    The stub read helper is at ``jenkins/scripts/fetch_build_artifact.sh``, where artifactory.sh
    looks for it from ``scripts/bolt/internal/``.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "repo"
        self.internal = self.root / "scripts" / "bolt" / "internal"
        self.internal.mkdir(parents=True)
        self.helper = self.root / "jenkins" / "scripts" / "fetch_build_artifact.sh"
        stub = tmp_path / "helper_stub.py"
        stub.write_text(_HELPER_STUB, encoding="utf-8")
        _write_executable(
            self.helper, f'exec {shlex.quote(sys.executable)} -I -S {shlex.quote(str(stub))} "$@"\n'
        )

        bin_dir = tmp_path / "stub_bin"
        for downloader in ("curl", "wget"):
            _write_executable(bin_dir / downloader, _DOWNLOADER_STUB)
        self._downloader_log = tmp_path / "downloader_calls.txt"
        self._helper_log = tmp_path / "helper_calls.jsonl"
        self.tmpdir = tmp_path / "tmp"
        self.tmpdir.mkdir()
        self.netrc = tmp_path / "artifactory-netrc"
        self.netrc.write_text("machine artifactory.test login ci password secret\n")
        self.netrc.chmod(0o600)
        self.helper_env = {
            "LLM_ARTIFACTS_PRIMARY": "aws",
            "ARTIFACTORY_NETRC": str(self.netrc),
            "ARTIFACTORY_CREDENTIAL_ID": CREDENTIAL_ID,
        }

        env = {
            name: value
            for name, value in os.environ.items()
            if name not in (*HELPER_ENVIRONMENT, *_SCRIPT_SETTINGS)
        }
        env.update(
            {
                "PATH": os.pathsep.join((str(bin_dir), env.get("PATH", os.defpath))),
                "TMPDIR": str(self.tmpdir),
                "HELPER_STUB_LOG": str(self._helper_log),
                "HELPER_STUB_ENV": json.dumps(HELPER_ENVIRONMENT),
                "DOWNLOADER_STUB_LOG": str(self._downloader_log),
            }
        )
        self._env = env

    def install(self, script: Path) -> Path:
        """Copy a script of ``scripts/bolt/internal/`` to the same place in the tree."""
        return Path(shutil.copy(script, self.internal / script.name))

    def run(
        self,
        script: Path,
        *args: str,
        answers: list[tuple[int, Path | str | None]] | None = None,
        **variables: str,
    ) -> Result:
        """Run a script of the tree, from outside it, with the helper's environment of a CI caller.

        Every run checks that neither ``curl`` nor ``wget`` was started, that the helper saw the
        caller's environment, and that no temporary file is left.

        Args:
            script: The script to run with bash.
            *args: Its arguments.
            answers: The stub helper's answers, one per call: an exit status and, for a
                successful call, the file to copy to its destination, or the line that --exists
                prints.
            **variables: Further environment variables; one of the helper's replaces its value.
        """
        helper_env = {name: variables.pop(name, value) for name, value in self.helper_env.items()}
        planned = [[status, str(payload) if payload else None] for status, payload in answers or []]
        env = dict(self._env, **helper_env, **variables, HELPER_STUB_ANSWERS=json.dumps(planned))
        self._helper_log.unlink(missing_ok=True)

        proc = subprocess.run(
            ["bash", str(script), *args],
            env=env,
            cwd=self.root.parent,
            capture_output=True,
            text=True,
            timeout=60,
        )

        calls = (
            [HelperCall(**json.loads(line)) for line in self._helper_log.read_text().splitlines()]
            if self._helper_log.exists()
            else []
        )
        assert not self._downloader_log.exists(), self._downloader_log.read_text()
        for call in calls:
            assert call.env == helper_env
        assert list(self.tmpdir.iterdir()) == [], "temporary files were left behind"
        return Result(proc.returncode, proc.stdout, proc.stderr, calls)


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    return Tree(tmp_path)


@pytest.fixture
def artifactory(tree: Tree) -> Path:
    """The path of artifactory.sh in the tree."""
    return tree.install(ARTIFACTORY_SH)


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """A promoted bundle whose manifest names ``REF``."""
    return _bundle(tmp_path / "bundle.tar.gz", {"ref": REF, "workloads": []})


# ---------------------------------------------------------------------------
# artifactory.sh pull-latest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variables", "path"),
    [({}, LATEST_PATH), ({"BOLT_PROFILE_REF": REF}, PINNED_PATH)],
    ids=["latest", "pinned"],
)
def test_pull_latest_downloads_through_the_helper_and_unpacks(
    tree: Tree,
    artifactory: Path,
    bundle: Path,
    tmp_path: Path,
    variables: dict[str, str],
    path: str,
) -> None:
    dest = tmp_path / "profiles"

    result = tree.run(
        artifactory, "pull-latest", BRANCH, TRIPLE, str(dest), answers=[(0, bundle)], **variables
    )

    assert result.returncode == 0, result.stderr
    # One download: the legacy path, then a destination file inside <dest_dir>.
    assert len(result.helper_args) == 1
    asked, downloaded = result.helper_args[0]
    assert asked == path
    assert Path(downloaded).parent == dest
    # The bundle is unpacked, and the archive is not kept beside its files.
    assert sorted(entry.name for entry in dest.iterdir()) == sorted([PROFILE, "manifest.json"])
    assert (dest / PROFILE).read_text() == PROFILE_TEXT
    assert json.loads((dest / "manifest.json").read_text())["ref"] == REF


@pytest.mark.parametrize(
    ("helper_status", "status"),
    [
        (FETCH_ABSENT, BOLT_NO_BUNDLE),
        *((code, BOLT_STORE_UNREADABLE) for code in UNREADABLE.values()),
        (1, BOLT_STORE_UNREADABLE),
    ],
    ids=["absent", *UNREADABLE, "other-failure"],
)
def test_pull_latest_tells_no_bundle_from_an_unreadable_store(
    tree: Tree, artifactory: Path, tmp_path: Path, helper_status: int, status: int
) -> None:
    dest = tmp_path / "profiles"

    result = tree.run(
        artifactory, "pull-latest", BRANCH, TRIPLE, str(dest), answers=[(helper_status, None)]
    )

    assert result.returncode == status, result.stderr
    # A failed read is not repeated here: the helper has retried what can be retried.
    assert [args[0] for args in result.helper_args] == [LATEST_PATH]
    assert not dest.exists() or list(dest.iterdir()) == []
    assert "[artifactory][ERROR]" in result.stderr


def test_pull_latest_without_the_helper_reports_an_unreadable_store(
    tree: Tree, artifactory: Path, tmp_path: Path
) -> None:
    tree.helper.unlink()

    result = tree.run(artifactory, "pull-latest", BRANCH, TRIPLE, str(tmp_path / "profiles"))

    assert result.returncode == BOLT_STORE_UNREADABLE, result.stderr
    assert tree.helper.name in result.stderr


# ---------------------------------------------------------------------------
# artifactory.sh resolve-latest
# ---------------------------------------------------------------------------


def test_resolve_latest_prints_the_ref_property(
    tree: Tree, artifactory: Path, tmp_path: Path
) -> None:
    properties = _properties(
        tmp_path / "properties.json",
        {"bolt.branch": BRANCH, "bolt.ref": REF, "bolt.triple": TRIPLE},
    )

    result = tree.run(
        artifactory, "resolve-latest", BRANCH, TRIPLE, answers=[(0, HELD_ON_AWS), (0, properties)]
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{REF}\n"
    # The properties of the copy that readers get answer: the bundle is not downloaded.
    assert result.helper_args == [["--exists", LATEST_PATH], ["--properties", LATEST_PATH, ANY]]


def test_resolve_latest_ignores_the_properties_of_the_other_location(
    tree: Tree, artifactory: Path, bundle: Path, tmp_path: Path
) -> None:
    # aws holds an unlabelled latest.tar.gz, so the properties lookup went on to urm's copy.
    properties = _properties(tmp_path / "properties.json", {"bolt.ref": "f" * 12}, host=URM_HOST)

    result = tree.run(
        artifactory,
        "resolve-latest",
        BRANCH,
        TRIPLE,
        answers=[(0, HELD_ON_AWS), (0, properties), (0, bundle)],
    )

    assert result.returncode == 0, result.stderr
    # The ref is the one in the manifest of the bundle that readers get.
    assert result.stdout == f"{REF}\n"
    assert result.helper_args == [
        ["--exists", LATEST_PATH],
        ["--properties", LATEST_PATH, ANY],
        [LATEST_PATH, ANY],
    ]


@pytest.mark.parametrize("has_properties", [True, False], ids=["other-properties", "no-properties"])
def test_resolve_latest_without_the_property_reads_the_manifest(
    tree: Tree, artifactory: Path, bundle: Path, tmp_path: Path, has_properties: bool
) -> None:
    # The helper reports a file without any property as absent from the properties lookup.
    properties: tuple[int, Path | None] = (FETCH_ABSENT, None)
    if has_properties:
        properties = (0, _properties(tmp_path / "properties.json", {"build.name": "L0_PostMerge"}))

    result = tree.run(
        artifactory,
        "resolve-latest",
        BRANCH,
        TRIPLE,
        answers=[(0, HELD_ON_AWS), properties, (0, bundle)],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{REF}\n"
    assert result.helper_args == [
        ["--exists", LATEST_PATH],
        ["--properties", LATEST_PATH, ANY],
        [LATEST_PATH, ANY],
    ]


def test_resolve_latest_prints_nothing_when_latest_is_absent(tree: Tree, artifactory: Path) -> None:
    result = tree.run(artifactory, "resolve-latest", BRANCH, TRIPLE, answers=[(FETCH_ABSENT, None)])

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    # One lookup tells that neither location holds the file.
    assert result.helper_args == [["--exists", LATEST_PATH]]


@pytest.mark.parametrize(
    "manifest", [None, {"workloads": []}], ids=["no-manifest", "manifest-without-ref"]
)
def test_resolve_latest_prints_nothing_when_the_bundle_names_no_ref(
    tree: Tree, artifactory: Path, tmp_path: Path, manifest: dict[str, object] | None
) -> None:
    unlabelled = _bundle(tmp_path / "unlabelled.tar.gz", manifest)

    result = tree.run(
        artifactory,
        "resolve-latest",
        BRANCH,
        TRIPLE,
        answers=[(0, HELD_ON_AWS), (FETCH_ABSENT, None), (0, unlabelled)],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("helper_status", UNREADABLE.values(), ids=UNREADABLE)
@pytest.mark.parametrize("failed_read", ["lookup", "properties", "bundle"])
def test_resolve_latest_reports_an_unreadable_store(
    tree: Tree, artifactory: Path, failed_read: str, helper_status: int
) -> None:
    earlier: dict[str, list[tuple[int, Path | str | None]]] = {
        "lookup": [],
        "properties": [(0, HELD_ON_AWS)],
        "bundle": [(0, HELD_ON_AWS), (FETCH_ABSENT, None)],
    }
    answers = [*earlier[failed_read], (helper_status, None)]

    result = tree.run(artifactory, "resolve-latest", BRANCH, TRIPLE, answers=answers)

    assert result.returncode == BOLT_STORE_UNREADABLE, result.stderr
    assert result.stdout == ""
    assert len(result.helper_calls) == len(answers)
    assert "[artifactory][ERROR]" in result.stderr


@pytest.mark.parametrize("primary", ["aws", "urm"])
def test_reads_inherit_the_callers_environment(
    tree: Tree, artifactory: Path, bundle: Path, tmp_path: Path, primary: str
) -> None:
    netrc = tmp_path / "another-netrc"
    caller = {
        "LLM_ARTIFACTS_PRIMARY": primary,
        "ARTIFACTORY_NETRC": str(netrc),
        "ARTIFACTORY_CREDENTIAL_ID": "another-credential",
    }

    pulled = tree.run(
        artifactory,
        "pull-latest",
        BRANCH,
        TRIPLE,
        str(tmp_path / "profiles"),
        answers=[(0, bundle)],
        **caller,
    )
    resolved = tree.run(
        artifactory,
        "resolve-latest",
        BRANCH,
        TRIPLE,
        answers=[(0, HELD_ON_AWS), (FETCH_ABSENT, None), (0, bundle)],
        **caller,
    )

    assert pulled.returncode == 0, pulled.stderr
    assert resolved.returncode == 0, resolved.stderr
    assert [call.env for call in pulled.helper_calls] == [caller]
    assert [call.env for call in resolved.helper_calls] == [caller] * 3


# ---------------------------------------------------------------------------
# apply_latest.sh
# ---------------------------------------------------------------------------


@dataclass
class Applied:
    """The outcome of one run of apply_latest.sh."""

    result: Result
    # The arguments that the artifactory.sh stub got, when it stands in for the real script.
    pull_args: list[str]
    # The arguments that apply_bolt.py got, or None when it was not started.
    apply_args: list[str] | None
    output: Path


def _apply_latest(
    tree: Tree,
    tmp_path: Path,
    *,
    pull_status: int | None = 0,
    apply_status: int = 0,
    artifact: str = "TensorRT-LLM.tar.gz",
    answers: list[tuple[int, Path | None]] | None = None,
    **variables: str,
) -> Applied:
    """Run apply_latest.sh beside stubs of its siblings and of apply_bolt.py one level up.

    Args:
        tree: The tree to install the scripts in.
        tmp_path: The test's temporary directory.
        pull_status: The exit status of the artifactory.sh stub; None installs the real
            artifactory.sh instead, which reads through the stub helper.
        apply_status: The exit status of the apply_bolt.py stub.
        artifact: The name of the input artifact.
        answers: The stub helper's answers, for the real artifactory.sh.
        **variables: Further environment variables.
    """
    script = tree.install(APPLY_LATEST_SH)
    # The script sources this to put llvm-bolt on PATH.
    (tree.internal / "stage_llvm_bolt.sh").write_text(":\n", encoding="utf-8")
    if pull_status is None:
        tree.install(ARTIFACTORY_SH)
    else:
        _write_executable(tree.internal / "artifactory.sh", _ARTIFACTORY_STUB)
    (tree.internal.parent / "apply_bolt.py").write_text(_APPLY_BOLT_STUB, encoding="utf-8")

    source = tmp_path / artifact
    source.write_text("built\n", encoding="utf-8")
    output = tmp_path / f"bolted-{artifact}"
    pull_log = tmp_path / "pull_args.txt"
    apply_log = tmp_path / "apply_args.json"
    result = tree.run(
        script,
        BRANCH,
        TRIPLE,
        str(source),
        str(output),
        answers=answers,
        ARTIFACTORY_STUB_LOG=str(pull_log),
        ARTIFACTORY_STUB_STATUS=str(pull_status),
        APPLY_STUB_LOG=str(apply_log),
        APPLY_STUB_STATUS=str(apply_status),
        **variables,
    )
    return Applied(
        result,
        pull_log.read_text().splitlines() if pull_log.exists() else [],
        json.loads(apply_log.read_text()) if apply_log.exists() else None,
        output,
    )


@pytest.mark.parametrize(
    ("artifact", "mode"),
    [("TensorRT-LLM.tar.gz", "--tarball"), ("tensorrt_llm-1.0-cp312-linux_aarch64.whl", "--wheel")],
    ids=["tarball", "wheel"],
)
def test_apply_latest_applies_the_pulled_bundle(
    tree: Tree, tmp_path: Path, artifact: str, mode: str
) -> None:
    applied = _apply_latest(tree, tmp_path, artifact=artifact)

    assert applied.result.returncode == 0, applied.result.stderr
    assert applied.pull_args[:3] == ["pull-latest", BRANCH, TRIPLE]
    profiles = applied.pull_args[3]
    assert applied.apply_args is not None
    assert applied.apply_args[:2] == [mode, str(tmp_path / artifact)]
    assert applied.apply_args[-4:] == ["--profiles", profiles, "--output", str(applied.output)]
    assert applied.output.read_text() == "bolted\n"
    # The pulled profiles are removed when the script ends.
    assert not Path(profiles).exists()


@pytest.mark.parametrize(
    ("pull_status", "status", "message"),
    [
        (BOLT_NO_BUNDLE, 3, "no promoted bundle"),
        (BOLT_STORE_UNREADABLE, 4, "could not be read"),
        (1, 3, "no promoted bundle"),
    ],
    ids=["no-bundle", "store-unreadable", "other-failure"],
)
def test_apply_latest_keeps_the_status_of_pull_latest(
    tree: Tree, tmp_path: Path, pull_status: int, status: int, message: str
) -> None:
    applied = _apply_latest(tree, tmp_path, pull_status=pull_status)

    assert applied.result.returncode == status, applied.result.stderr
    assert message in applied.result.stderr
    assert applied.apply_args is None
    assert not applied.output.exists()


def test_apply_latest_exits_2_when_the_apply_fails(tree: Tree, tmp_path: Path) -> None:
    applied = _apply_latest(tree, tmp_path, apply_status=1)

    assert applied.result.returncode == 2, applied.result.stderr
    assert applied.apply_args is not None
    assert "apply_bolt failed" in applied.result.stderr
    assert not applied.output.exists()


@pytest.mark.parametrize(
    ("helper_status", "status"),
    [(0, 0), (FETCH_ABSENT, 3), (FETCH_REJECTED, 4), (FETCH_TRANSFER_FAILED, 4)],
    ids=["read", "absent", "rejected", "transfer-failed"],
)
def test_apply_latest_reads_the_pinned_bundle_through_the_helper(
    tree: Tree, tmp_path: Path, bundle: Path, helper_status: int, status: int
) -> None:
    applied = _apply_latest(
        tree,
        tmp_path,
        pull_status=None,
        answers=[(helper_status, bundle if helper_status == 0 else None)],
        BOLT_PROFILE_REF=REF,
    )

    assert applied.result.returncode == status, applied.result.stderr
    assert [args[0] for args in applied.result.helper_args] == [PINNED_PATH]
    assert (applied.apply_args is not None) == (status == 0)
