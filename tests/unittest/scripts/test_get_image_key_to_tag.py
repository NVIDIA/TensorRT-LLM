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
"""Tests for jenkins/scripts/get_image_key_to_tag.py.

The script prints a build's image-tag map, imageKeyToTag.json, which it reads with the
fetch_build_artifact.sh beside it. Each test runs a copy of the script that has a stub helper
beside it. The latest build number, which the script asks Jenkins for, is set by the test, and
urllib's urlopen is replaced, so no request leaves the test.
"""

import importlib.util
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "jenkins" / "scripts" / "get_image_key_to_tag.py"
# The pipeline whose builds branch mode walks, for the branch "main".
POST_MERGE = "LLM/main/L0_PostMerge"
CREDENTIAL_ID = "aws-artifactory-credentials"
HELPER_ENVIRONMENT = ("LLM_ARTIFACTS_PRIMARY", "ARTIFACTORY_NETRC", "ARTIFACTORY_CREDENTIAL_ID")

# Stands in for fetch_build_artifact.sh. It records its arguments and the environment it sees,
# writes to stdout and stderr, and exits with the status set for the artifact path: by default 0
# for a path that has a map, which it writes to <dest-file>, and FETCH_ABSENT for any other.
_HELPER_STUB = r"""
import json
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
config = json.loads((here / "helper_stub.json").read_text(encoding="utf-8"))
artifact_path, dest = sys.argv[1:]
with open(here / "helper_calls.jsonl", "a", encoding="utf-8") as log:
    env = {name: os.environ.get(name) for name in config["env"]}
    log.write(json.dumps({"args": sys.argv[1:], "env": env}) + "\n")
print("helper stdout")
print("helper stderr", file=sys.stderr)
image_map = config["maps"].get(artifact_path)
status = config["statuses"].get(artifact_path, 2 if image_map is None else 0)
if status == 0:
    Path(dest).write_text(image_map, encoding="utf-8")
sys.exit(status)
"""


def _map_path(pipeline: str, build: int) -> str:
    """The legacy path of a build's image-tag map."""
    return f"sw-tensorrt-generic/llm-artifacts/{pipeline}/{build}/imageKeyToTag.json"


def _map_text(build: int) -> str:
    """An image-tag map that differs from every other build's."""
    return json.dumps(
        {
            "NGC Release Image amd64": f"registry.example/release:x86_64-{build}",
            "NGC Release Image arm64": f"registry.example/release:sbsa-{build}",
        },
        indent=2,
    )


@dataclass
class _Result:
    exit_code: object
    stdout: str
    stderr: str


class _Lookup:
    """A copy of the script with a stub helper beside it, and the latest build that Jenkins has."""

    def __init__(
        self,
        directory: Path,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        self._directory = directory
        self._monkeypatch = monkeypatch
        self._capfd = capfd
        self._script = directory / SCRIPT_PATH.name
        self._script.write_text(SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        stub = directory / "helper_stub.py"
        stub.write_text(_HELPER_STUB, encoding="utf-8")
        (directory / "fetch_build_artifact.sh").write_text(
            f'exec {shlex.quote(sys.executable)} -I -S {shlex.quote(str(stub))} "$@"\n',
            encoding="utf-8",
        )
        self._maps: dict[str, str] = {}
        self._statuses: dict[str, int] = {}
        self._configure()

        spec = importlib.util.spec_from_file_location("get_image_key_to_tag", self._script)
        assert spec is not None and spec.loader is not None
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        # The build number that Jenkins reports as the latest, or None when it cannot tell.
        self.latest_build: int | None = None
        self.latest_build_requests: list[str] = []
        self.requested_urls: list[str] = []
        monkeypatch.setattr(self._module, "get_latest_build_number", self._latest_build_number)
        monkeypatch.setattr(self._module.urllib.request, "urlopen", self._urlopen)

    def _configure(self) -> None:
        config = {"env": HELPER_ENVIRONMENT, "maps": self._maps, "statuses": self._statuses}
        (self._directory / "helper_stub.json").write_text(json.dumps(config), encoding="utf-8")

    def _latest_build_number(self, jenkins_base: str) -> int | None:
        self.latest_build_requests.append(jenkins_base)
        return self.latest_build

    def _urlopen(self, url: object, *args: object, **kwargs: object) -> None:
        self.requested_urls.append(str(getattr(url, "full_url", url)))
        raise AssertionError(f"the script requested {self.requested_urls[-1]} itself")

    def holds(self, pipeline: str, build: int) -> str:
        """Publish a map for a build; returns the map's text."""
        text = _map_text(build)
        self._maps[_map_path(pipeline, build)] = text
        self._configure()
        return text

    def exits_with(self, status: int, pipeline: str, build: int) -> None:
        """End the helper's read of a build's map with `status`."""
        self._statuses[_map_path(pipeline, build)] = status
        self._configure()

    @property
    def calls(self) -> list[dict]:
        log = self._directory / "helper_calls.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    @property
    def reads(self) -> list[str]:
        """The artifact path of each helper call."""
        return [call["args"][0] for call in self.calls]

    def run(self, *args: str) -> _Result:
        """Run the script's command line; whatever it does, it requests no URL itself."""
        self._monkeypatch.setattr(sys, "argv", [str(self._script), *args])
        with pytest.raises(SystemExit) as exited:
            self._module.main()
        captured = self._capfd.readouterr()
        assert self.requested_urls == []
        return _Result(exited.value.code, captured.out, captured.err)


@pytest.fixture
def lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> _Lookup:
    monkeypatch.setenv("LLM_ARTIFACTS_PRIMARY", "urm")
    monkeypatch.setenv("ARTIFACTORY_NETRC", str(tmp_path / "artifactory-netrc"))
    monkeypatch.setenv("ARTIFACTORY_CREDENTIAL_ID", CREDENTIAL_ID)
    directory = tmp_path / "script-copy"
    directory.mkdir()
    return _Lookup(directory, monkeypatch, capfd)


def test_branch_mode_prints_the_map_of_the_latest_build(lookup: _Lookup) -> None:
    lookup.latest_build = 90
    image_map = lookup.holds(POST_MERGE, 90)

    result = lookup.run("main")

    assert result.exit_code == 0
    # Only the map: the helper's stdout and the script's progress messages stay off stdout.
    assert result.stdout == image_map
    assert "helper stderr" in result.stderr
    [jenkins_base] = lookup.latest_build_requests
    assert jenkins_base.endswith("/job/LLM/job/main/job/L0_PostMerge")
    [call] = lookup.calls
    artifact_path, dest = call["args"]
    assert artifact_path == (
        "sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/90/imageKeyToTag.json"
    )
    # A temporary file, removed once read.
    assert not Path(dest).parent.exists()


def test_branch_mode_walks_down_to_the_first_build_that_has_a_map(lookup: _Lookup) -> None:
    lookup.latest_build = 90
    image_map = lookup.holds(POST_MERGE, 88)
    lookup.holds(POST_MERGE, 87)

    result = lookup.run("main")

    assert result.exit_code == 0
    assert result.stdout == image_map
    assert lookup.reads == [_map_path(POST_MERGE, build) for build in (90, 89, 88)]


def test_branch_mode_exits_1_when_no_build_has_a_map(lookup: _Lookup) -> None:
    lookup.latest_build = 3

    result = lookup.run("main")

    assert result.exit_code == 1
    assert result.stdout == ""
    assert lookup.reads == [_map_path(POST_MERGE, build) for build in (3, 2, 1)]


@pytest.mark.parametrize(
    "status", (3, 4, 64, 127), ids=("rejected", "transfer-failed", "usage-error", "other")
)
def test_branch_mode_stops_the_walk_when_a_read_fails(lookup: _Lookup, status: int) -> None:
    lookup.latest_build = 90
    lookup.exits_with(status, POST_MERGE, 89)
    lookup.holds(POST_MERGE, 88)

    result = lookup.run("main")

    assert result.exit_code == 1
    assert result.stdout == ""
    # Build 88 has a map, and is not read: an older build's read would fail the same way.
    assert lookup.reads == [_map_path(POST_MERGE, build) for build in (90, 89)]
    # The helper's own message says why; the script names the read and the status.
    assert "helper stderr" in result.stderr
    assert any(
        _map_path(POST_MERGE, 89) in line and f"exit status {status}" in line
        for line in result.stderr.splitlines()
    )


def test_branch_mode_exits_1_when_the_latest_build_is_unknown(lookup: _Lookup) -> None:
    lookup.latest_build = None

    result = lookup.run("main")

    assert result.exit_code == 1
    assert result.stdout == ""
    assert lookup.reads == []


def test_pipeline_mode_prints_the_map_of_that_build(lookup: _Lookup) -> None:
    pipeline = "LLM/main/L0_MergeRequest_PR"
    image_map = lookup.holds(pipeline, 1234)
    lookup.holds(pipeline, 1233)

    result = lookup.run(pipeline, "1234")

    assert result.exit_code == 0
    assert result.stdout == image_map
    assert lookup.reads == [
        "sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/1234/imageKeyToTag.json"
    ]
    # The build is given, so Jenkins is not asked for one.
    assert lookup.latest_build_requests == []


@pytest.mark.parametrize(
    "status",
    (2, 3, 4, 64, 127),
    ids=("absent", "rejected", "transfer-failed", "usage-error", "other"),
)
def test_pipeline_mode_makes_one_read_and_exits_1_when_it_fails(
    lookup: _Lookup, status: int
) -> None:
    pipeline = "LLM/main/L0_MergeRequest_PR"
    lookup.exits_with(status, pipeline, 1234)
    lookup.holds(pipeline, 1233)

    result = lookup.run(pipeline, "1234")

    assert result.exit_code == 1
    assert result.stdout == ""
    assert lookup.reads == [_map_path(pipeline, 1234)]
    assert "helper stderr" in result.stderr
    errors = [line for line in result.stderr.splitlines() if line.startswith("Error: ")]
    if status == 2:
        assert any(
            "Could not fetch imageKeyToTag.json" in line and pipeline in line and "1234" in line
            for line in errors
        )
    else:
        assert any(
            _map_path(pipeline, 1234) in line and f"exit status {status}" in line for line in errors
        )


@pytest.mark.parametrize("args", (("main",), (POST_MERGE, "90")), ids=("branch", "pipeline"))
def test_a_helper_that_cannot_start_exits_1(
    lookup: _Lookup, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: tuple[str, ...]
) -> None:
    lookup.latest_build = 90
    lookup.holds(POST_MERGE, 90)
    empty_dir = tmp_path / "empty-path"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))

    result = lookup.run(*args)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert lookup.calls == []
    assert any(
        line.startswith("Error: ") and _map_path(POST_MERGE, 90) in line
        for line in result.stderr.splitlines()
    )


@pytest.mark.parametrize("primary", ("aws", "urm", None), ids=("aws", "urm", "unset"))
@pytest.mark.parametrize("args", (("main",), (POST_MERGE, "90")), ids=("branch", "pipeline"))
def test_the_helper_sees_the_callers_environment(
    lookup: _Lookup,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    args: tuple[str, ...],
    primary: str | None,
) -> None:
    if primary is None:
        monkeypatch.delenv("LLM_ARTIFACTS_PRIMARY")
    else:
        monkeypatch.setenv("LLM_ARTIFACTS_PRIMARY", primary)
    lookup.latest_build = 90
    lookup.holds(POST_MERGE, 90)

    assert lookup.run(*args).exit_code == 0

    [call] = lookup.calls
    # The script chooses no primary: an unset one is left for the helper's default.
    assert call["env"] == {
        "LLM_ARTIFACTS_PRIMARY": primary,
        "ARTIFACTORY_NETRC": str(tmp_path / "artifactory-netrc"),
        "ARTIFACTORY_CREDENTIAL_ID": CREDENTIAL_ID,
    }


def test_the_script_holds_no_urm_address() -> None:
    # The helper knows where each location is; the script asks it by legacy path alone.
    assert "urm.nvidia.com" not in SCRIPT_PATH.read_text(encoding="utf-8")
