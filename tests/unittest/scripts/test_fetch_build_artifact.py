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
"""Tests for ``jenkins/scripts/fetch_build_artifact.sh``.

The helper reads one build artifact from artifactory.nvidia.com, the location that CI names
``aws``. Two loopback HTTP servers stand in for that location and for the CDN that it redirects
downloads to. Each listens on its own host name, because curl matches netrc entries by host name.
A stub ``sleep`` on ``PATH`` records the back-off instead of waiting, and no request leaves the
loopback interface.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import socket
import subprocess
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FETCH_BUILD_ARTIFACT = REPO_ROOT / "jenkins" / "scripts" / "fetch_build_artifact.sh"

# The host of each fake server, by the server's name, and the repository of the one that stands
# in for the location.
HOSTS = {"aws": "127.0.0.1", "cdn": "127.0.0.2"}
REPOS = {"aws": "sw-tensorrt-llm-generic-local"}

BUILD_DIR = "llm-artifacts/LLM/main/L0_PostMerge/88"
LAYOUT_PATH = f"{BUILD_DIR}/TensorRT-LLM.tar.gz"
# Not one of the three in-scope kinds: the helper reads any file under llm-artifacts/.
OTHER_FILE = "tensorrt_llm-1.2.0-cp312-cp312-linux_x86_64.whl"
LEGACY_PATH = f"sw-tensorrt-generic/{LAYOUT_PATH}"
SYNOPSIS = "fetch_build_artifact.sh [--properties | --exists] <artifact-path> [<dest-file>]"
LIST_SYNOPSIS = "fetch_build_artifact.sh --list <directory-path>"

RESULTS_DIR = "llm-artifacts/LLM/main/L0_MergeRequest_PR/1234/test-results"
# The children of a run's test-results/, in the storage API's order; a name that ends in "/" is a
# folder.
RESULTS = [
    "results-A10-PyTorch-1.tar.gz",
    "results-A10-PyTorch-2-progress.tar.gz",
    "logs/",
    "doc-html-preview.tar.gz",
    "rerun_report.html",
]

NETRC_LOGIN = "ci-reader"
NETRC_PASSWORD = "Pw-6c1f0e93-must-never-be-printed"
CREDENTIAL_ID = "test-reader-credential"
BASIC_TOKEN = base64.b64encode(f"{NETRC_LOGIN}:{NETRC_PASSWORD}".encode()).decode()
AUTHORIZATION = f"Basic {BASIC_TOKEN}"

# Asks Artifactory to send a file itself instead of redirecting the download to its CDN.
REDIRECT_OPT_OUT = "x-jfrog-download-redirect-to"
# A signed CDN URL: the signature is a credential of its own and must not reach the log.
CDN_PATH = f"/cdn/{LAYOUT_PATH}?Expires=1798761600&Signature=c2lnbmVkLWNkbi11cmw&Key-Pair-Id=KTEST"

PROPERTIES = json.dumps(
    {"properties": {"bolt.ref": ["0123456789abcdef"]}, "uri": "storage-uri"}, indent=2
).encode()

# Log phrases that make the shared Jenkins library retry a step as a network fault. The message
# for a rejected credential must contain none of them.
RETRY_SIGNATURES = (
    "403 Forbidden",
    "URL returned error",
    "HTTP response code said error",
    "404 Not Found",
    "ERROR 404",
    "HTTP 50",
    "ERROR 50",
    "Server Error",
    "Error downloading",
    "Connection",
    "Read timed out",
    "Too Many Requests",
)

_PROXY_VARIABLES = {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy"}


@dataclass
class Reply:
    """An HTTP response for a fake server to send."""

    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Request:
    """A request that a fake server received; header names are lower case."""

    server: str
    path: str
    headers: dict[str, str]

    @property
    def authorization(self) -> str | None:
        return self.headers.get("authorization")


Route = Reply | Callable[[Request], Reply]


def _sha1(content: bytes) -> str:
    return hashlib.sha1(content).hexdigest()


def _storage_info(
    location: str, content: bytes, sha1: str, layout_path: str = LAYOUT_PATH
) -> bytes:
    """Build a storage API answer for the artifact at ``layout_path``.

    ``originalChecksums`` comes first and holds a different SHA-1, so only a parser that reads the
    ``checksums`` object gets the right value.
    """
    info = {
        "repo": REPOS[location],
        "path": f"/{layout_path}",
        "created": "2026-10-16T08:00:00.000Z",
        "mimeType": "application/x-gzip",
        "size": str(len(content)),
        "originalChecksums": {"sha1": "f" * 40, "sha256": "e" * 64},
        "checksums": {"sha1": sha1, "sha256": hashlib.sha256(content).hexdigest()},
        "uri": f"storage-uri/{layout_path}",
    }
    return json.dumps(info, indent=2).encode()


def _folder_info(
    location: str, children: list[str], layout_path: str = RESULTS_DIR
) -> dict[str, object]:
    """Build a storage API answer for the folder at ``layout_path``.

    A child whose name ends in ``/`` is a folder. As in Artifactory's answer, the folder's own
    ``uri`` follows its children.
    """
    return {
        "repo": REPOS[location],
        "path": f"/{layout_path}",
        "created": "2026-10-27T08:00:00.000Z",
        "lastModified": "2026-10-27T09:30:00.000Z",
        "children": [
            {"uri": f"/{name.removesuffix('/')}", "folder": name.endswith("/")} for name in children
        ],
        "uri": f"storage-uri/{layout_path}",
    }


def _artifactory_json(value: object, level: int = 0) -> str:
    """Lay out ``value`` as Artifactory's storage API does.

    An object has one member per line, indented by two spaces for each object around it, with
    ``" : "`` between key and value. An array stays on its line: ``[ {`` ... ``}, {`` ... ``} ]``.
    """
    if isinstance(value, dict):
        indent = "  " * (level + 1)
        members = [
            f"{indent}{json.dumps(key)} : {_artifactory_json(item, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(members) + "\n" + "  " * level + "}"
    if isinstance(value, list):
        items = ", ".join(_artifactory_json(item, level) for item in value)
        return f"[ {items} ]" if items else "[ ]"
    return json.dumps(value)


def _in_turn(*replies: Reply) -> Callable[[Request], Reply]:
    """Build a route that answers with ``replies`` in order, then repeats the last one."""
    remaining = list(replies)

    def answer(request: Request) -> Reply:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return answer


def _unused_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = cast(_Server, self.server)
        headers = {name.lower(): value for name, value in self.headers.items()}
        reply = server.artifactory.answer(Request(server.name, self.path, headers))
        self.send_response(reply.status)
        for name, value in reply.headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(reply.body)))
        self.end_headers()
        self.wfile.write(reply.body)

    def log_message(self, format: str, *args: object) -> None:
        """Keep access logs out of the test output."""


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, name: str, artifactory: FakeArtifactory) -> None:
        super().__init__((HOSTS[name], 0), _Handler)
        self.name = name
        self.artifactory = artifactory


class FakeArtifactory:
    """The location and the CDN, each a loopback HTTP server on its own host.

    Routes map ``(server, path with query)`` to a reply. A request without a route gets 404, which
    is how Artifactory answers for a missing file.
    """

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Route] = {}
        self.requests: list[Request] = []
        self._lock = threading.Lock()
        self._servers: dict[str, _Server] = {}
        try:
            for name in HOSTS:
                self._servers[name] = _Server(name, self)
        except OSError as error:
            self.close()
            pytest.skip(f"cannot listen on the 127.0.0.0/8 loopback range: {error}")
        for server in self._servers.values():
            # A short poll interval keeps shutdown(), which waits for the next poll, fast.
            threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()

    def close(self) -> None:
        for server in self._servers.values():
            server.shutdown()
            server.server_close()

    def origin(self, name: str) -> str:
        host, port = self._servers[name].server_address[:2]
        return f"http://{host}:{port}"

    def host(self, name: str) -> str:
        return self.origin(name).removeprefix("http://")

    def base_url(self, location: str) -> str:
        """Return the location's base URL, ``<root>/<repository>``."""
        return f"{self.origin(location)}/artifactory/{REPOS[location]}"

    def file_url(self, location: str) -> str:
        return f"{self.base_url(location)}/{LAYOUT_PATH}"

    @staticmethod
    def file_path(location: str, layout_path: str = LAYOUT_PATH) -> str:
        return f"/artifactory/{REPOS[location]}/{layout_path}"

    @staticmethod
    def storage_path(location: str, layout_path: str = LAYOUT_PATH) -> str:
        return f"/artifactory/api/storage/{REPOS[location]}/{layout_path}"

    @classmethod
    def properties_path(cls, location: str) -> str:
        return f"{cls.storage_path(location)}?properties"

    def publish(self, location: str, content: bytes, layout_path: str = LAYOUT_PATH) -> None:
        """Serve ``content`` as the artifact: its storage API entry and its download."""
        info = _storage_info(location, content, _sha1(content), layout_path)
        self.routes[(location, self.storage_path(location, layout_path))] = Reply(200, info)
        self.routes[(location, self.file_path(location, layout_path))] = Reply(200, content)

    def publish_folder(
        self, location: str, children: list[str], layout_path: str = RESULTS_DIR
    ) -> None:
        """Serve the storage API answer for a folder; a child ending in ``/`` is a folder."""
        info = _artifactory_json(_folder_info(location, children, layout_path)).encode()
        self.routes[(location, self.storage_path(location, layout_path))] = Reply(200, info)

    def answer(self, request: Request) -> Reply:
        with self._lock:
            self.requests.append(request)
            route = self.routes.get((request.server, request.path))
            if route is None:
                return Reply(404, b'{"errors": [{"status": 404, "message": "File not found."}]}')
            return route if isinstance(route, Reply) else route(request)

    def received(self, name: str) -> list[Request]:
        with self._lock:
            return [request for request in self.requests if request.server == name]

    def paths(self, name: str) -> list[str]:
        return [request.path for request in self.received(name)]


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str
    sleeps: list[str]


class Helper:
    """Runs the helper against the fake servers, with a stub ``sleep`` and a test netrc."""

    def __init__(self, tmp_path: Path, artifactory: FakeArtifactory) -> None:
        self.artifactory = artifactory
        self.dest = tmp_path / "out" / "TensorRT-LLM.tar.gz"
        self.part = self.dest.with_name(f"{self.dest.name}.part")
        self.cwd = tmp_path / "cwd"
        self.tmpdir = tmp_path / "tmp"
        stub_dir = tmp_path / "bin"
        home = tmp_path / "home"
        for directory in (self.dest.parent, self.cwd, self.tmpdir, stub_dir, home):
            directory.mkdir()

        self._sleep_log = tmp_path / "sleep.log"
        sleep = stub_dir / "sleep"
        sleep.write_text(f'#!/bin/sh\necho "$*" >> "{self._sleep_log}"\n')
        sleep.chmod(0o755)

        # curl must not read the caller's curlrc: a "verbose" line there would print the
        # Authorization header that the netrc produces.
        (home / ".curlrc").write_text("verbose\n")

        netrc = tmp_path / "netrc"
        netrc.write_text(f"machine {HOSTS['aws']} login {NETRC_LOGIN} password {NETRC_PASSWORD}\n")
        netrc.chmod(0o600)

        env = {
            name: value
            for name, value in os.environ.items()
            if name.lower() not in _PROXY_VARIABLES
            and not name.startswith(("LLM_ARTIFACTS_", "ARTIFACTORY_"))
            and name not in ("CURL_HOME", "XDG_CONFIG_HOME")
        }
        env.update(
            {
                "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
                "HOME": str(home),
                "TMPDIR": str(self.tmpdir),
                "LLM_ARTIFACTS_AWS_URL": artifactory.base_url("aws"),
                "ARTIFACTORY_NETRC": str(netrc),
                "ARTIFACTORY_CREDENTIAL_ID": CREDENTIAL_ID,
            }
        )
        self.env = env

    def run(self, *args: str, **env_overrides: str | None) -> Result:
        """Run the helper; an override of ``None`` removes that variable from its environment."""
        env = dict(self.env)
        for name, value in env_overrides.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        self._sleep_log.unlink(missing_ok=True)

        proc = subprocess.run(
            ["bash", str(FETCH_BUILD_ARTIFACT), *args],
            env=env,
            cwd=self.cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = proc.stdout + proc.stderr
        assert NETRC_PASSWORD not in output
        assert BASIC_TOKEN not in output
        assert "Signature=" not in output
        assert list(self.tmpdir.iterdir()) == [], "temporary files were left behind"
        assert list(self.cwd.iterdir()) == [], "files were written outside <dest-file>"
        sleeps = self._sleep_log.read_text().split() if self._sleep_log.exists() else []
        return Result(proc.returncode, proc.stdout, proc.stderr, sleeps)


@pytest.fixture
def artifactory() -> Iterator[FakeArtifactory]:
    fake = FakeArtifactory()
    yield fake
    fake.close()


@pytest.fixture
def helper(tmp_path: Path, artifactory: FakeArtifactory) -> Helper:
    return Helper(tmp_path, artifactory)


def _served_by(layout_path: str = LAYOUT_PATH) -> str:
    return f"fetch_build_artifact: served-by=aws path={layout_path}"


def _absent(artifactory: FakeArtifactory, layout_path: str = LAYOUT_PATH) -> str:
    """Return the line that reports a file or a folder that the location does not have."""
    return f"fetch_build_artifact: {artifactory.host('aws')} does not have {layout_path}."


def _rejection_message(credential_id: str) -> str:
    """Return the line naming a credential that aws rejected, the same for HTTP 401 and 403."""
    return (
        f"fetch_build_artifact: the Artifactory access token in credential '{credential_id}' "
        "needs renewal or lacks read permission on sw-tensorrt-llm-generic-local; an Artifactory "
        "access token for artifactory.nvidia.com is required."
    )


def _rejection_lines(
    artifactory: FakeArtifactory, status: int, layout_path: str = LAYOUT_PATH
) -> list[str]:
    """Return the lines that report the rejection of the credentials for ``layout_path``."""
    host = artifactory.host("aws")
    return [
        f"fetch_build_artifact: {host} rejected the credentials for {layout_path} (HTTP {status}).",
        _rejection_message(CREDENTIAL_ID),
    ]


def _listing(children: list[str] | None = None) -> list[str]:
    """Return the lines that list the files among ``children``, by default ``RESULTS``."""
    names = RESULTS if children is None else children
    return [name for name in names if not name.endswith("/")]


def _listed(n: int, layout_path: str = RESULTS_DIR) -> str:
    return f"fetch_build_artifact: listed {n} path={layout_path}"


def _assert_no_retry_signature(text: str) -> None:
    for signature in RETRY_SIGNATURES:
        assert signature not in text


def _assert_no_credential_message(text: str) -> None:
    assert "needs renewal" not in text
    assert CREDENTIAL_ID not in text


# ---------------------------------------------------------------------------
# Path mapping and the base URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("artifact_path", "layout_path"),
    [
        (LEGACY_PATH, LAYOUT_PATH),
        (f"sw-tensorrt-generic-local/{LAYOUT_PATH}", LAYOUT_PATH),
        (LAYOUT_PATH, LAYOUT_PATH),
        (f"sw-tensorrt-generic/{BUILD_DIR}/build_info.txt", f"{BUILD_DIR}/build_info.txt"),
        (
            f"sw-tensorrt-generic/{BUILD_DIR}/waive_list/waives.txt",
            f"{BUILD_DIR}/waive_list/waives.txt",
        ),
        (f"sw-tensorrt-generic/{BUILD_DIR}/{OTHER_FILE}", f"{BUILD_DIR}/{OTHER_FILE}"),
    ],
    ids=["legacy", "legacy-local", "layout", "build-info", "waive-list", "other-file"],
)
def test_reads_the_mapped_path(helper: Helper, artifact_path: str, layout_path: str) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package", layout_path)

    result = helper.run(artifact_path, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert not helper.part.exists()
    assert result.stderr.splitlines() == [_served_by(layout_path)]
    assert result.stdout == ""
    assert artifactory.paths("aws") == [
        artifactory.storage_path("aws", layout_path),
        artifactory.file_path("aws", layout_path),
    ]
    assert [request.authorization for request in artifactory.received("aws")] == [AUTHORIZATION] * 2
    assert result.sleeps == []


@pytest.mark.parametrize(
    "artifact_path",
    [
        f"sw-tensorrt-llm-generic-local/{LAYOUT_PATH}",
        "sw-tensorrt-generic/other-artifacts/TensorRT-LLM.tar.gz",
        f"/{LAYOUT_PATH}",
        "llm-artifacts-old/88/TensorRT-LLM.tar.gz",
        f"https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local/{LAYOUT_PATH}",
        "llm-artifacts/LLM/../../TensorRT-LLM.tar.gz",
        "llm-artifacts/LLM/main/L0_PostMerge/88/",
        "llm-artifacts/LLM/main//88/TensorRT-LLM.tar.gz",
        "llm-artifacts/LLM/main/./88/TensorRT-LLM.tar.gz",
        "llm-artifacts/LLM/main/L0 PostMerge/88/TensorRT-LLM.tar.gz",
        "llm-artifacts/LLM/main/L0_PostMerge/88/TensorRT-LLM.tar.gz\t",
        f"{LAYOUT_PATH};bolt.ref=0123456789abcdef",
        f"{LAYOUT_PATH}?properties",
        f"{LAYOUT_PATH}#fragment",
    ],
)
def test_unmappable_path_is_a_usage_error(helper: Helper, artifact_path: str) -> None:
    helper.artifactory.publish("aws", b"aws package")

    result = helper.run(artifact_path, str(helper.dest))

    assert result.returncode == 64
    assert helper.artifactory.requests == []
    assert not helper.dest.exists()


def test_base_url_with_credentials_is_a_usage_error(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    base_url = artifactory.base_url("aws").replace("http://", "http://someone:url-secret@")

    result = helper.run(LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_AWS_URL=base_url)

    assert result.returncode == 64
    assert "LLM_ARTIFACTS_AWS_URL must not contain credentials" in result.stderr
    assert "url-secret" not in result.stdout + result.stderr
    assert artifactory.requests == []


def test_default_base_url_is_the_repository_on_artifactory_nvidia_com(
    helper: Helper, tmp_path: Path
) -> None:
    # A stub curl records the URLs it is given and fails as it does for a host that cannot be
    # reached, so no request leaves this host.
    urls = tmp_path / "curl-urls.log"
    curl = tmp_path / "bin" / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do\n'
        '    case "$arg" in\n'
        f'        http://* | https://*) echo "$arg" >> "{urls}" ;;\n'
        "    esac\n"
        "done\n"
        "exit 7\n"
    )
    curl.chmod(0o755)

    result = helper.run(LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_AWS_URL=None)

    assert result.returncode == 4
    assert set(urls.read_text().split()) == {
        "https://artifactory.nvidia.com/artifactory/api/storage/"
        f"sw-tensorrt-llm-generic-local/{LAYOUT_PATH}"
    }
    assert helper.artifactory.requests == []


def test_names_of_another_location_in_the_environment_are_ignored(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")

    result = helper.run(
        LEGACY_PATH,
        str(helper.dest),
        LLM_ARTIFACTS_PRIMARY="urm",
        LLM_ARTIFACTS_URM_URL=f"{artifactory.origin('cdn')}/artifactory/sw-tensorrt-generic",
    )

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert result.stderr.splitlines() == [_served_by()]
    assert artifactory.received("cdn") == []


def test_verified_download_replaces_the_destination(helper: Helper) -> None:
    helper.artifactory.publish("aws", b"aws package")
    helper.dest.write_bytes(b"older download")
    helper.part.write_bytes(b"partial download left by an interrupted run")

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert not helper.part.exists()


# ---------------------------------------------------------------------------
# A file that the location does not have
# ---------------------------------------------------------------------------


def test_absent_file_exits_2(helper: Helper) -> None:
    artifactory = helper.artifactory

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.splitlines() == [_absent(artifactory)]
    assert not helper.dest.exists()
    assert not helper.part.exists()
    assert artifactory.paths("aws") == [artifactory.storage_path("aws")]
    assert result.sleeps == []
    _assert_no_retry_signature(result.stderr)


# ---------------------------------------------------------------------------
# Rejected credentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("step", ["lookup", "download"])
def test_rejected_credentials_exit_3(helper: Helper, step: str, status: int) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    rejected = artifactory.storage_path("aws") if step == "lookup" else artifactory.file_path("aws")
    artifactory.routes[("aws", rejected)] = Reply(status, b'{"errors": [{"status": %d}]}' % status)

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 3
    assert result.stderr.splitlines() == _rejection_lines(artifactory, status)
    _assert_no_retry_signature(result.stdout + result.stderr)
    assert artifactory.paths("aws").count(rejected) == 1
    assert result.sleeps == []
    assert not helper.dest.exists()
    assert not helper.part.exists()


def test_rejection_message_marks_an_unset_credential_id(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.routes[("aws", artifactory.storage_path("aws"))] = Reply(401)

    result = helper.run(LEGACY_PATH, str(helper.dest), ARTIFACTORY_CREDENTIAL_ID=None)

    assert result.returncode == 3
    assert _rejection_message("<unset: ARTIFACTORY_CREDENTIAL_ID>") in result.stderr.splitlines()


# ---------------------------------------------------------------------------
# Downloads redirected to the CDN
# ---------------------------------------------------------------------------


def _redirect_downloads_to_cdn(
    artifactory: FakeArtifactory, *, direct: Reply, cdn_origin: str | None = None
) -> None:
    """Make aws redirect downloads to the CDN, and answer ``direct`` to requests that opt out.

    ``cdn_origin`` replaces the CDN's origin in the redirect, for example with a port that nothing
    listens on.
    """
    origin = cdn_origin or artifactory.origin("cdn")
    redirect = Reply(302, headers={"Location": origin + CDN_PATH})

    def download(request: Request) -> Reply:
        return direct if request.headers.get(REDIRECT_OPT_OUT) == "None" else redirect

    artifactory.routes[("aws", artifactory.file_path("aws"))] = download


def test_download_through_the_cdn_carries_no_credential(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    _redirect_downloads_to_cdn(artifactory, direct=Reply(500))
    artifactory.routes[("cdn", CDN_PATH)] = Reply(200, b"aws package")

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert _served_by() in result.stderr.splitlines()
    assert artifactory.paths("aws").count(artifactory.file_path("aws")) == 1
    assert [request.authorization for request in artifactory.received("cdn")] == [None]


def test_inherited_xtrace_is_turned_off(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    _redirect_downloads_to_cdn(artifactory, direct=Reply(500))
    artifactory.routes[("cdn", CDN_PATH)] = Reply(200, b"aws package")

    # A caller running under `set -x` that exports SHELLOPTS passes xtrace on to the helper. A
    # BASH_ENV startup file would run, traced, before the helper's first command.
    result = helper.run(LEGACY_PATH, str(helper.dest), SHELLOPTS="xtrace", BASH_ENV=None)

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    # Only the command that turns xtrace off is traced. Helper.run checks that neither the netrc
    # password nor the CDN URL's signature reached the output.
    assert [line for line in result.stderr.splitlines() if not line.endswith("set +x")] == [
        _served_by()
    ]


def test_cdn_rejection_is_retried_without_the_redirect(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    _redirect_downloads_to_cdn(artifactory, direct=Reply(200, b"aws package"))
    artifactory.routes[("cdn", CDN_PATH)] = Reply(403, b"<Error><Code>AccessDenied</Code></Error>")

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert _served_by() in result.stderr.splitlines()
    downloads = [
        request
        for request in artifactory.received("aws")
        if request.path == artifactory.file_path("aws")
    ]
    assert [request.headers.get(REDIRECT_OPT_OUT) for request in downloads] == [None, "None"]
    assert [request.authorization for request in downloads] == [AUTHORIZATION] * 2
    assert [request.authorization for request in artifactory.received("cdn")] == [None]
    assert result.sleeps == []
    _assert_no_credential_message(result.stderr)


@pytest.mark.parametrize("direct", ["rejected", "redirected-again"])
def test_persistent_cdn_rejection_exits_4(helper: Helper, direct: str) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    redirect = Reply(302, headers={"Location": artifactory.origin("cdn") + CDN_PATH})
    _redirect_downloads_to_cdn(artifactory, direct=Reply(403) if direct == "rejected" else redirect)
    artifactory.routes[("cdn", CDN_PATH)] = Reply(403)

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    _assert_no_credential_message(result.stderr)
    # Three attempts, each retried once without the redirect.
    assert artifactory.paths("aws").count(artifactory.file_path("aws")) == 6
    assert result.sleeps == ["2", "4"]
    assert not helper.dest.exists()
    assert not helper.part.exists()


@pytest.mark.parametrize("failure", ["http-500", "connection-refused"])
def test_cdn_failure_other_than_403_is_retried_through_the_redirect(
    helper: Helper, failure: str
) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    if failure == "http-500":
        cdn_origin = artifactory.origin("cdn")
        artifactory.routes[("cdn", CDN_PATH)] = Reply(500)
    else:
        cdn_origin = f"http://{HOSTS['cdn']}:{_unused_port(HOSTS['cdn'])}"
    # A request that opts out of the redirect would get the file, so only staying on the redirect
    # can make the download fail.
    _redirect_downloads_to_cdn(
        artifactory, direct=Reply(200, b"aws package"), cdn_origin=cdn_origin
    )

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    _assert_no_credential_message(result.stderr)
    downloads = [
        request
        for request in artifactory.received("aws")
        if request.path == artifactory.file_path("aws")
    ]
    assert [request.headers.get(REDIRECT_OPT_OUT) for request in downloads] == [None] * 3
    if failure == "http-500":
        assert artifactory.paths("cdn") == [CDN_PATH] * 3
    else:
        assert artifactory.received("cdn") == []
    assert result.sleeps == ["2", "4"]
    assert not helper.dest.exists()
    assert not helper.part.exists()


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def test_checksum_mismatch_is_retried(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    artifactory.routes[("aws", artifactory.file_path("aws"))] = _in_turn(
        Reply(200, b"corrupted package"), Reply(200, b"aws package")
    )

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert not helper.part.exists()
    assert artifactory.paths("aws").count(artifactory.file_path("aws")) == 2
    assert result.sleeps == ["2"]


def test_persistent_checksum_mismatch_exits_4(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    artifactory.routes[("aws", artifactory.file_path("aws"))] = Reply(200, b"corrupted package")
    helper.dest.write_bytes(b"older download")

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    assert "checksum" in result.stderr
    assert _sha1(b"corrupted package") in result.stderr
    assert helper.dest.read_bytes() == b"older download"
    assert not helper.part.exists()
    assert artifactory.paths("aws").count(artifactory.file_path("aws")) == 3
    assert result.sleeps == ["2", "4"]


def test_file_deleted_after_its_lookup_is_absent(helper: Helper) -> None:
    """The lookup finds the file, and the file is deleted before its download."""
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    storage = ("aws", artifactory.storage_path("aws"))
    stored = artifactory.routes[storage]
    assert isinstance(stored, Reply)
    artifactory.routes[storage] = _in_turn(stored, Reply(404))
    del artifactory.routes[("aws", artifactory.file_path("aws"))]

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 2, result.stderr
    assert result.stderr.splitlines()[-1] == _absent(artifactory)
    assert artifactory.paths("aws") == [
        artifactory.storage_path("aws"),
        artifactory.file_path("aws"),
        artifactory.storage_path("aws"),
    ]
    assert result.sleeps == ["2"]
    assert not helper.part.exists()
    assert not helper.dest.exists()


def test_file_replaced_after_its_lookup_is_checked_against_its_new_sha1(helper: Helper) -> None:
    """The lookup reports the SHA-1 of a file that is replaced before its download."""
    artifactory = helper.artifactory
    artifactory.publish("aws", b"new package")
    storage = ("aws", artifactory.storage_path("aws"))
    new_info = artifactory.routes[storage]
    assert isinstance(new_info, Reply)
    old_info = Reply(200, _storage_info("aws", b"old package", _sha1(b"old package")))
    artifactory.routes[storage] = _in_turn(old_info, new_info)

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"new package"
    assert _served_by() in result.stderr.splitlines()
    assert artifactory.paths("aws") == [
        artifactory.storage_path("aws"),
        artifactory.file_path("aws"),
        artifactory.storage_path("aws"),
        artifactory.file_path("aws"),
    ]
    assert result.sleeps == ["2"]


def test_lookup_without_checksum_is_retried_then_exit_4(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    info = json.dumps({"repo": REPOS["aws"], "path": f"/{LAYOUT_PATH}", "children": []})
    artifactory.routes[("aws", artifactory.storage_path("aws"))] = Reply(200, info.encode())

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    assert artifactory.paths("aws") == [artifactory.storage_path("aws")] * 3
    assert result.sleeps == ["2", "4"]


# ---------------------------------------------------------------------------
# Transient failures are retried
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("step", "status"),
    [("lookup", 500), ("lookup", 429), ("download", 502), ("download", 404)],
)
def test_transient_errors_are_retried_then_exit_4(helper: Helper, step: str, status: int) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    failing = artifactory.storage_path("aws") if step == "lookup" else artifactory.file_path("aws")
    artifactory.routes[("aws", failing)] = Reply(status)

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    assert artifactory.paths("aws").count(failing) == 3
    assert result.sleeps == ["2", "4"]
    _assert_no_credential_message(result.stderr)
    assert not helper.dest.exists()
    assert not helper.part.exists()


def test_transient_error_then_success(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    storage = ("aws", artifactory.storage_path("aws"))
    stored = artifactory.routes[storage]
    assert isinstance(stored, Reply)
    artifactory.routes[storage] = _in_turn(Reply(503), stored)

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert _served_by() in result.stderr.splitlines()
    assert artifactory.paths("aws").count(artifactory.storage_path("aws")) == 2
    assert result.sleeps == ["2"]


def test_unreachable_host_is_retried_then_exit_4(helper: Helper) -> None:
    closed = f"http://{HOSTS['aws']}:{_unused_port(HOSTS['aws'])}/artifactory/{REPOS['aws']}"

    result = helper.run(LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_AWS_URL=closed)

    assert result.returncode == 4
    assert helper.artifactory.requests == []
    assert result.sleeps == ["2", "4"]
    assert not helper.dest.exists()


# ---------------------------------------------------------------------------
# --exists and --properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_dest", [False, True], ids=["no-dest", "dest-ignored"])
def test_exists_prints_url_and_sha1(helper: Helper, with_dest: bool) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"package")
    dest_args = [str(helper.dest)] if with_dest else []

    result = helper.run("--exists", LEGACY_PATH, *dest_args)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{artifactory.file_url('aws')} {_sha1(b'package')}\n"
    assert result.stderr.splitlines() == [_served_by()]
    assert list(helper.dest.parent.iterdir()) == []
    assert artifactory.paths("aws") == [artifactory.storage_path("aws")]


@pytest.mark.parametrize(("status", "returncode"), [(404, 2), (403, 3)])
def test_exists_reports_absent_and_rejected(helper: Helper, status: int, returncode: int) -> None:
    artifactory = helper.artifactory
    artifactory.routes[("aws", artifactory.storage_path("aws"))] = Reply(status)

    result = helper.run("--exists", LEGACY_PATH)

    assert result.returncode == returncode
    assert result.stdout == ""
    assert artifactory.paths("aws") == [artifactory.storage_path("aws")]


def test_properties_are_written_to_the_destination(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.routes[("aws", artifactory.properties_path("aws"))] = Reply(200, PROPERTIES)

    result = helper.run("--properties", LEGACY_PATH, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == PROPERTIES
    assert not helper.part.exists()
    assert result.stderr.splitlines() == [_served_by()]
    assert artifactory.paths("aws") == [artifactory.properties_path("aws")]
    assert [request.authorization for request in artifactory.received("aws")] == [AUTHORIZATION]


def test_absent_properties_exit_2(helper: Helper) -> None:
    artifactory = helper.artifactory

    result = helper.run("--properties", LEGACY_PATH, str(helper.dest))

    assert result.returncode == 2
    assert result.stderr.splitlines() == [
        f"fetch_build_artifact: {artifactory.host('aws')} has no properties for {LAYOUT_PATH}."
    ]
    assert not helper.dest.exists()
    assert not helper.part.exists()
    assert artifactory.paths("aws") == [artifactory.properties_path("aws")]


def test_properties_rejected_exit_3(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.routes[("aws", artifactory.properties_path("aws"))] = Reply(403)

    result = helper.run("--properties", LEGACY_PATH, str(helper.dest))

    assert result.returncode == 3
    assert _rejection_message(CREDENTIAL_ID) in result.stderr.splitlines()
    _assert_no_retry_signature(result.stderr)
    assert artifactory.paths("aws") == [artifactory.properties_path("aws")]
    assert not helper.dest.exists()


def test_properties_server_error_is_retried_then_exit_4(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.routes[("aws", artifactory.properties_path("aws"))] = Reply(500)

    result = helper.run("--properties", LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    assert artifactory.paths("aws") == [artifactory.properties_path("aws")] * 3
    assert result.sleeps == ["2", "4"]


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("has_folder", "returncode"), [(True, 0), (False, 2)], ids=["folder", "no-folder"]
)
def test_list_prints_the_files_in_the_folder(
    helper: Helper, has_folder: bool, returncode: int
) -> None:
    artifactory = helper.artifactory
    if has_folder:
        artifactory.publish_folder("aws", RESULTS)
    lines = _listing() if has_folder else []

    result = helper.run("--list", f"sw-tensorrt-generic/{RESULTS_DIR}")

    assert result.returncode == returncode, result.stderr
    # The names alone, in the storage API's order, and no folders.
    assert result.stdout.splitlines() == lines
    assert result.stderr.splitlines() == [
        *([] if has_folder else [_absent(artifactory, RESULTS_DIR)]),
        _listed(len(lines)),
    ]
    _assert_no_retry_signature(result.stderr)
    assert artifactory.paths("aws") == [artifactory.storage_path("aws", RESULTS_DIR)]
    assert [request.authorization for request in artifactory.received("aws")] == [AUTHORIZATION]
    assert result.sleeps == []


@pytest.mark.parametrize("children", [[], ["logs/"]], ids=["empty", "folders-only"])
def test_list_folder_without_files_exits_0(helper: Helper, children: list[str]) -> None:
    helper.artifactory.publish_folder("aws", children)

    result = helper.run("--list", RESULTS_DIR)

    # The location has the folder, so the listing is complete although it holds no file.
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr.splitlines() == [_listed(0)]


@pytest.mark.parametrize(
    "directory_path",
    [
        f"sw-tensorrt-generic/{RESULTS_DIR}",
        f"sw-tensorrt-generic-local/{RESULTS_DIR}",
        RESULTS_DIR,
        f"sw-tensorrt-generic/{RESULTS_DIR}/",
        f"{RESULTS_DIR}/",
    ],
    ids=["legacy", "legacy-local", "layout", "legacy-trailing-slash", "layout-trailing-slash"],
)
def test_list_maps_the_directory_path(helper: Helper, directory_path: str) -> None:
    artifactory = helper.artifactory
    artifactory.publish_folder("aws", RESULTS)

    result = helper.run("--list", directory_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == _listing()
    assert result.stderr.splitlines() == [_listed(4)]
    assert artifactory.paths("aws") == [artifactory.storage_path("aws", RESULTS_DIR)]


# Layouts of a folder's storage API answer that the helper must read alike.
JSON_LAYOUTS: dict[str, Callable[[dict[str, object]], str]] = {
    "artifactory": _artifactory_json,
    "compact": lambda info: json.dumps(info, separators=(",", ":")),
    "tabs-and-crlf": lambda info: json.dumps(info, indent="\t").replace("\n", "\r\n"),
    # Puts each child's "folder" before its "uri".
    "sorted-keys": lambda info: json.dumps(info, indent=2, sort_keys=True),
}


@pytest.mark.parametrize("layout", list(JSON_LAYOUTS))
def test_list_reads_any_json_layout(helper: Helper, layout: str) -> None:
    artifactory = helper.artifactory
    # JSON gives these characters a meaning, yet a name holds them unescaped.
    children = ["results-A10 [1], {2}: retry.tar.gz", "logs/", "rerun_report.html"]
    body = JSON_LAYOUTS[layout](_folder_info("aws", children))
    artifactory.routes[("aws", artifactory.storage_path("aws", RESULTS_DIR))] = Reply(
        200, body.encode()
    )

    result = helper.run("--list", RESULTS_DIR)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == _listing(children)
    assert result.stderr.splitlines() == [_listed(2)]


@pytest.mark.parametrize(
    "directory_path",
    [
        "llm-artifacts",
        "llm-artifacts/",
        f"{RESULTS_DIR}//",
        f"/{RESULTS_DIR}",
        f"sw-tensorrt-llm-generic-local/{RESULTS_DIR}",
        f"https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local/{RESULTS_DIR}",
        "llm-artifacts/LLM/main/L0_MergeRequest_PR/1234/../1235/test-results",
        f"{RESULTS_DIR}?list&deep=1",
    ],
)
def test_list_unmappable_path_is_a_usage_error(helper: Helper, directory_path: str) -> None:
    helper.artifactory.publish_folder("aws", RESULTS)

    result = helper.run("--list", directory_path)

    assert result.returncode == 64
    assert f"not a directory path under llm-artifacts/: '{directory_path}'" in result.stderr
    assert result.stdout == ""
    assert helper.artifactory.requests == []


@pytest.mark.parametrize(
    "args",
    [
        ("--list",),
        ("--list", RESULTS_DIR, "out"),
        ("--list", "--exists", RESULTS_DIR),
        ("--properties", "--list", RESULTS_DIR, "out"),
    ],
    ids=["no-path", "dest-file", "with-exists", "with-properties"],
)
def test_list_usage_errors_exit_64(helper: Helper, args: tuple[str, ...]) -> None:
    helper.artifactory.publish_folder("aws", RESULTS)

    result = helper.run(*args)

    assert result.returncode == 64
    assert LIST_SYNOPSIS in result.stderr
    assert result.stdout == ""
    assert helper.artifactory.requests == []


def test_list_needs_a_netrc(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish_folder("aws", RESULTS)

    result = helper.run("--list", RESULTS_DIR, ARTIFACTORY_NETRC=None)

    assert result.returncode == 64
    assert "ARTIFACTORY_NETRC is not set" in result.stderr
    assert result.stdout == ""
    assert artifactory.requests == []


@pytest.mark.parametrize("status", [401, 403])
def test_list_rejected_credentials_exit_3(helper: Helper, status: int) -> None:
    artifactory = helper.artifactory
    folder = artifactory.storage_path("aws", RESULTS_DIR)
    artifactory.routes[("aws", folder)] = Reply(status, b'{"errors": [{"status": %d}]}' % status)

    result = helper.run("--list", RESULTS_DIR)

    assert result.returncode == 3
    assert result.stdout == ""
    assert result.stderr.splitlines() == _rejection_lines(artifactory, status, RESULTS_DIR)
    _assert_no_retry_signature(result.stderr)
    assert artifactory.paths("aws") == [folder]
    assert result.sleeps == []


def test_list_transient_error_then_success(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish_folder("aws", RESULTS)
    folder = artifactory.storage_path("aws", RESULTS_DIR)
    listed = artifactory.routes[("aws", folder)]
    assert isinstance(listed, Reply)
    artifactory.routes[("aws", folder)] = _in_turn(Reply(503), listed)

    result = helper.run("--list", RESULTS_DIR)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == _listing()
    assert result.stderr.splitlines()[-1] == _listed(4)
    assert artifactory.paths("aws") == [folder] * 2
    assert result.sleeps == ["2"]


@pytest.mark.parametrize(
    "failure", ["http-500", "connection-refused", "answer-for-a-file", "truncated-answer"]
)
def test_list_failure_is_retried_then_exit_4(helper: Helper, failure: str) -> None:
    artifactory = helper.artifactory
    folder = artifactory.storage_path("aws", RESULTS_DIR)
    complete = _artifactory_json(_folder_info("aws", RESULTS))
    replies = {
        "http-500": Reply(500),
        # A file's answer has no "children".
        "answer-for-a-file": Reply(200, _storage_info("aws", b"pkg", _sha1(b"pkg"), RESULTS_DIR)),
        # Cut off after the last child, before the children array closes.
        "truncated-answer": Reply(200, complete[: complete.index("} ]") + 1].encode()),
    }
    overrides = {}
    if failure == "connection-refused":
        closed = f"http://{HOSTS['aws']}:{_unused_port(HOSTS['aws'])}/artifactory/{REPOS['aws']}"
        overrides["LLM_ARTIFACTS_AWS_URL"] = closed
    else:
        artifactory.routes[("aws", folder)] = replies[failure]

    result = helper.run("--list", RESULTS_DIR, **overrides)

    assert result.returncode == 4
    assert result.stdout == ""
    assert result.stderr.splitlines()[-1] == _listed(0)
    _assert_no_credential_message(result.stderr)
    assert artifactory.paths("aws") == ([] if failure == "connection-refused" else [folder] * 3)
    assert result.sleeps == ["2", "4"]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("netrc", ["unset", "empty", "missing-file"])
def test_a_read_needs_a_netrc(helper: Helper, tmp_path: Path, netrc: str) -> None:
    helper.artifactory.publish("aws", b"aws package")
    value = {"unset": None, "empty": "", "missing-file": str(tmp_path / "no-such-netrc")}[netrc]

    result = helper.run(LEGACY_PATH, str(helper.dest), ARTIFACTORY_NETRC=value)

    assert result.returncode == 64
    assert "ARTIFACTORY_NETRC" in result.stderr
    assert helper.artifactory.requests == []


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help(helper: Helper, flag: str) -> None:
    result = helper.run(flag)

    assert result.returncode == 0
    assert SYNOPSIS in result.stdout
    assert LIST_SYNOPSIS in result.stdout
    assert helper.artifactory.requests == []


@pytest.mark.parametrize(
    "args",
    [
        (),
        (LEGACY_PATH,),
        ("--properties", LEGACY_PATH),
        ("--exists", "--properties", LEGACY_PATH),
        ("--verbose", LEGACY_PATH, "out"),
        ("--secondary-optional", LEGACY_PATH, "out"),
        (LEGACY_PATH, "out", "extra"),
    ],
    ids=[
        "no-args",
        "no-dest",
        "properties-no-dest",
        "two-modes",
        "unknown-option",
        "second-location-option",
        "extra-arg",
    ],
)
def test_usage_errors_exit_64(helper: Helper, args: tuple[str, ...]) -> None:
    result = helper.run(*args)

    assert result.returncode == 64
    assert SYNOPSIS in result.stderr
    assert helper.artifactory.requests == []


@pytest.mark.parametrize("dest", ["missing-dir/TensorRT-LLM.tar.gz", "out"])
def test_destination_must_be_a_file_in_an_existing_directory(
    helper: Helper, tmp_path: Path, dest: str
) -> None:
    helper.artifactory.publish("aws", b"aws package")

    result = helper.run(LEGACY_PATH, str(tmp_path / dest))

    assert result.returncode == 64
    assert helper.artifactory.requests == []


# ---------------------------------------------------------------------------
# Exit-code names shared with the readers
# ---------------------------------------------------------------------------

# The helper's exit codes other than success, by the names that the helper defines for them.
EXIT_CODES = {
    "FETCH_ABSENT": 2,
    "FETCH_REJECTED": 3,
    "FETCH_TRANSFER_FAILED": 4,
    "FETCH_USAGE": 64,
}


@dataclass(frozen=True)
class ExitCodeReader:
    """A reader that acts on particular exit codes of the helper."""

    path: str
    # Matches the reader's definitions of the helper's names, as (name, value).
    definition: str
    # Picks the code that checks the helper's exit status out of the reader's source.
    checks: Callable[[str], str]


def _defined_exit_codes(text: str, definition: str) -> dict[str, int]:
    return {name: int(value) for name, value in re.findall(definition, text, re.MULTILINE)}


def _helper_exit_codes() -> dict[str, int]:
    return _defined_exit_codes(
        FETCH_BUILD_ARTIFACT.read_text(), r"^readonly (FETCH_[A-Z_]+)=(\d+)$"
    )


def _braced_block(text: str, start: str) -> str:
    """The code from ``start`` through the brace that closes the first block after it."""
    begin = text.index(start)
    depth = 0
    for i in range(text.index("{", begin), len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[begin : i + 1]
    raise AssertionError(f"no block closes after {start!r}")


def _between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin : text.index(end, begin)]


def _python_definitions(text: str, names: set[str]) -> str:
    """The source of the module-level functions and assignments called ``names``."""
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name in names:
            found[name] = ast.get_source_segment(text, node) or ""
    assert set(found) == names
    return "\n".join(found.values())


def _bare_exit_codes(code: str, exit_codes: set[int]) -> list[int]:
    """The numbers in ``code`` that are among ``exit_codes``.

    ``code`` is the part of a reader that checks the helper's exit status, so each of these
    numbers is treated as an exit code checked by number, whatever the form of the check: a
    comparison either way round, a shell test in ``[ ]`` or ``[[ ]]`` with or without quotes, a
    ``case`` arm, or an item or key of a list, range, set, dict, or map. A file descriptor in a
    shell redirection (``>&2``, ``2>``), a positional parameter (``$2``), and digits within a
    word, version, or decimal are not counted.
    """
    numbers = re.findall(r"(?<![\w$])(?<!\d\.)(?<![<>]&)\d+(?![\w<>])(?!\.\d)", code)
    return [int(number) for number in numbers if int(number) in exit_codes]


_GROOVY_DEFINITION = r"^(FETCH_[A-Z_]+) = (\d+)$"
_PYTHON_DEFINITION = r"^_(FETCH_[A-Z_]+) = (\d+)$"
_SHELL_DEFINITION = r"^(FETCH_[A-Z_]+)=(\d+)$"

EXIT_CODE_READERS = [
    ExitCodeReader(
        "jenkins/L0_Test.groovy",
        _GROOVY_DEFINITION,
        lambda text: _braced_block(text, "def boltedTarArtifactPath(")
        + _braced_block(text, "def fetchBuildArtifact(")
        + _braced_block(text, "def fetchOrUsePrefetched(")
        + _braced_block(text, "def prefetchBuildArtifact("),
    ),
    # A retried stage attempt's read of the earlier attempts' test-result archives.
    ExitCodeReader(
        "jenkins/L0_Test.groovy",
        _GROOVY_DEFINITION,
        lambda text: _between(
            text, "priorTags.each { priorTag ->", 'sh "cd ${priorDir} && tar -xzf ${tarName}"'
        ),
    ),
    # Collect Test Result and the coverage step list and read a run's test results.
    ExitCodeReader(
        "jenkins/scripts/read_test_results.sh",
        _SHELL_DEFINITION,
        lambda text: "".join(
            _braced_block(text, f"{function}() {{")
            for function in ("list_results", "read_archive", "read_finals", "read_checkpoints")
        ),
    ),
    # The Slurm job's download of the packaged build.
    ExitCodeReader(
        "jenkins/scripts/slurm_install.sh",
        _SHELL_DEFINITION,
        lambda text: _between(text, "if ! retry_command", "tar -zxf"),
    ),
    # The fat-image builder's download of the packaged build.
    ExitCodeReader(
        "jenkins/scripts/fat_build_inline.sh",
        # FETCH_MAX_ATTEMPTS is the builder's own retry count, not one of the helper's names.
        r"^(FETCH_(?!MAX_ATTEMPTS)[A-Z_]+)=(\d+)$",
        lambda text: _between(text, "attempt=1", "sleep 60"),
    ),
    ExitCodeReader(
        "jenkins/BuildDockerImage.groovy",
        _GROOVY_DEFINITION,
        lambda text: _braced_block(text, "for (file in requiredFiles)"),
    ),
    # The cluster script that resolves the input package.
    ExitCodeReader(
        "jenkins/BoltProfileGen.groovy",
        _GROOVY_DEFINITION,
        lambda text: _between(text, "def tarStage =", "'''.stripIndent()"),
    ),
    ExitCodeReader(
        "jenkins/scripts/cbts/cbts/command/coverage/selection/artifact.py",
        _PYTHON_DEFINITION,
        lambda text: _python_definitions(
            text,
            {
                "_HELPER_FAILURES",
                "_helper_failure",
                "_fetch_build_artifact",
                "_coverage_archive_location",
                "_download_coverage_archive",
            },
        ),
    ),
    ExitCodeReader(
        "scripts/get_wheel_from_package.py",
        _PYTHON_DEFINITION,
        lambda text: _python_definitions(text, {"download_package"}),
    ),
    # The BOLT profile store's reads: the bundle and the ref that latest.tar.gz points at.
    ExitCodeReader(
        "scripts/bolt/internal/artifactory.sh",
        _SHELL_DEFINITION,
        lambda text: _between(text, "store_read() {", 'case "${1:-}" in'),
    ),
    # The PLC job's lookup of an image build's image-tag map.
    ExitCodeReader(
        "jenkins/scripts/get_image_key_to_tag.py",
        _PYTHON_DEFINITION,
        lambda text: _python_definitions(
            text, {"_HELPER_FAILURES", "_helper_failure", "_read_image_map"}
        ),
    ),
]


def test_helper_names_its_exit_codes() -> None:
    text = FETCH_BUILD_ARTIFACT.read_text()

    assert _helper_exit_codes() == EXIT_CODES
    bare = re.findall(r"\b(?:exit|fail|return)\s+(\d+)\b", text)
    assert [int(code) for code in bare if int(code) in EXIT_CODES.values()] == []


@pytest.mark.parametrize("reader", EXIT_CODE_READERS, ids=lambda reader: Path(reader.path).name)
def test_readers_use_the_helpers_exit_code_names(reader: ExitCodeReader) -> None:
    helper_codes = _helper_exit_codes()
    text = (REPO_ROOT / reader.path).read_text()

    defined = _defined_exit_codes(text, reader.definition)
    assert defined, f"{reader.path} defines none of {sorted(helper_codes)}"
    assert defined == {name: helper_codes.get(name) for name in defined}
    checks = reader.checks(text)
    assert any(name in checks for name in defined)
    bare = _bare_exit_codes(checks, set(helper_codes.values()))
    assert bare == [], f"{reader.path} uses {bare} as numbers where it checks the exit status"


@pytest.mark.parametrize(
    ("code", "numbers"),
    [
        ("if rc == 2:", [2]),
        ("if (3 == exitCode) {", [3]),
        ('elif [ "$rc" -eq 2 ]; then', [2]),
        ("if [ $rc -eq 2 ]; then", [2]),
        ('if [[ "$rc" == "3" ]]; then', [3]),
        ('elif [ 2 -eq "$rc" ]; then', [2]),
        ("if (!(propsStatus in [0, 2, 4])) {", [2, 4]),
        ("if status in (0, 2):", [2]),
        ("if status not in {0, 2, 4}:", [2, 4]),
        ("if (exitCode in 2..4 || [64].contains(exitCode)) {", [2, 4, 64]),
        ('case "$rc" in 2) echo absent ;; 3|4) exit "$rc" ;; esac', [2, 3, 4]),
        ("switch (exitCode) {\n    case 64:", [64]),
        ("match status:\n    case 2 | 4:", [2, 4]),
        ("_HELPER_FAILURES = {\n    3: (AuthError, None),\n}", [3]),
        ("_HELPER_FAILURES = {3: (AuthError, None), 4: (TransferError, None)}", [3, 4]),
        ('def reasons = [3: "rejected", 4: "failed"]', [3, 4]),
    ],
    ids=[
        "comparison",
        "reversed-comparison",
        "quoted-test",
        "unquoted-test",
        "double-bracket-test",
        "reversed-test",
        "list",
        "tuple",
        "set",
        "range-and-contains",
        "shell-case-arms",
        "groovy-case-arm",
        "python-case-arm",
        "dict-key-per-line",
        "dict-keys-on-one-line",
        "map-keys-on-one-line",
    ],
)
def test_bare_number_check_finds_codes_in_any_form(code: str, numbers: list[int]) -> None:
    assert _bare_exit_codes(code, set(EXIT_CODES.values())) == numbers


@pytest.mark.parametrize(
    "code",
    [
        "if (!(propsStatus in [0, FETCH_ABSENT, FETCH_TRANSFER_FAILED])) {",
        "if (exitCode == FETCH_REJECTED) {",
        "FETCH_ABSENT=${FETCH_ABSENT}",
        'elif [ "$rc" -eq "$FETCH_ABSENT" ]; then',
        '[ "$rc" -eq 0 ] || exit "$rc"',
        "_HELPER_FAILURES = {\n    _FETCH_REJECTED: (ArtifactAuthError, None),\n}",
        "if status in (0, _FETCH_ABSENT):",
        "if rc not in (_FETCH_ABSENT, _FETCH_TRANSFER_FAILED):",
    ],
    ids=[
        "groovy-list",
        "groovy-comparison",
        "shell-definition",
        "shell-test",
        "shell-success",
        "python-dict",
        "python-tuple",
        "python-not-in",
    ],
)
def test_bare_number_check_passes_the_readers_named_forms(code: str) -> None:
    assert _bare_exit_codes(code, set(EXIT_CODES.values())) == []


@pytest.mark.parametrize(
    "code",
    [
        'echo "[ERROR] input tarball missing" >&2',
        'gzip -t "$DEST" 2>/dev/null',
        'cp "$1" "$2"',
        "if (config != LINUX_AARCH64_CONFIG || arch == 'x86_64' || version == '2.3.4') {",
    ],
    ids=["redirect-to-stderr", "redirect-stderr", "positional-parameter", "words-and-versions"],
)
def test_bare_number_check_ignores_digits_that_are_not_numbers(code: str) -> None:
    assert _bare_exit_codes(code, set(EXIT_CODES.values())) == []


# ---------------------------------------------------------------------------
# The location of what the pipelines publish under llm-artifacts
# ---------------------------------------------------------------------------

AWS_ALONE = (
    "The pipelines publish everything under llm-artifacts/ to artifactory.nvidia.com alone, the "
    "location that the shared library names aws, and read it from there alone. Every upload "
    "names aws, because the shared library's uploadArtifacts goes to urm when a call names no "
    'location, and maps its target with trtllm_utils.llmArtifactsRepoPath("aws", <legacy path>) '
    "or testResultsDirTarget, because uploadArtifacts sends a target as it is written. No "
    "pipeline file uses the library's llmArtifactsTargets or uploadToTargets, which add urm. No "
    "pipeline file holds a name of a primary or of an optional secondary location, in code or in "
    "a comment, and none names the urm location for llm-artifacts in its code: as a string, as "
    "an entry of LLM_ARTIFACTS_LOCATIONS, by its credential, or by an address under urm's "
    "Artifactory root other than its PyPI and GitHub mirrors."
)

PIPELINES = sorted(
    str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "jenkins").glob("*.groovy")
)

UPLOAD_TEST_RESULT = "def uploadTestResult(String filePattern, String directory)"

# The names of a primary location, as a variable in each spelling, and of an optional secondary
# one, as an argument, an option of the read helper, and an exit code.
LOCATION_CHOICE_NAMES = (
    "LLM_ARTIFACTS_PRIMARY",
    "llmArtifactsPrimary",
    "llm_artifacts_primary",
    "secondaryOptional",
    "--secondary-optional",
    "FETCH_SECONDARY_FAILED",
)

# How code names the urm location for llm-artifacts: as a string, as an entry of the shared
# library's locations, by its credential, and by an address under urm's Artifactory root, whole
# or completed by a constant or an interpolated path. The PyPI and GitHub mirrors under that
# root are other services, and so is the container registry, which is not under it.
URM_LOCATION_NAMES = re.compile(
    r"""["']urm["']"""
    r"|LLM_ARTIFACTS_LOCATIONS\.urm\b"
    r"|urm-artifactory-creds"
    r"|urm\.nvidia\.com/artifactory(?!/(?:api/pypi|github-go-remote)\b)"
)

# An upload target in the repository on artifactory.nvidia.com: this run's folder of outputs, or
# a legacy path that the shared library maps.
MAPPED_UPLOAD_TARGET = re.compile(
    r"""\btestResultsDirTarget\(|\btrtllm_utils\.llmArtifactsRepoPath\(\s*["']aws["']\s*,"""
)


def _has_line(text: str, line: str) -> bool:
    return re.search(rf"^{re.escape(line)}$", text, re.M) is not None


def _code_lines(block: str) -> list[str]:
    """The lines of Groovy code in ``block``, stripped, without blank and comment lines.

    A line is a comment only when its first non-blank characters are ``//``.
    """
    lines = (line.strip() for line in block.splitlines())
    return [line for line in lines if line and not line.startswith("//")]


def _urm_location_names(text: str) -> list[str]:
    """The lines of Groovy code in ``text`` that name the urm location for llm-artifacts."""
    return [line for line in _code_lines(text) if URM_LOCATION_NAMES.search(line)]


def _call_arguments(text: str, call: str) -> list[str]:
    """The text between the parentheses of each call in ``text`` that starts with ``call``.

    ``call`` ends with the opening parenthesis, which a call may write after blanks. Parentheses
    inside the arguments must balance, as they do in the pipelines' calls.
    """
    found = []
    for match in re.finditer(re.escape(call[:-1]) + r"[ \t]*\(", text):
        depth = 1
        for index in range(match.end(), len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    found.append(text[match.end() : index])
                    break
        else:
            raise AssertionError(f"no parenthesis closes {call!r}")
    return found


def _uploads_to_aws(arguments: str) -> bool:
    """Whether the arguments of an uploadArtifacts call send a mapped target to aws."""
    names_aws = re.search(r""",\s*["']aws["']\s*$""", arguments) is not None
    return names_aws and MAPPED_UPLOAD_TARGET.search(arguments) is not None


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('filePattern, testResultsDirTarget(directory), "aws"', True),
        ('"scan.log", trtllm_utils.llmArtifactsRepoPath("aws", scanLogDir), "aws"', True),
        ("\n    wheel,\n    trtllm_utils.llmArtifactsRepoPath('aws', dir),\n    'aws'", True),
        ('"scan.log", scanLogDir', False),
        ('"scan.log", trtllm_utils.llmArtifactsRepoPath("aws", scanLogDir)', False),
        ('"scan.log", trtllm_utils.llmArtifactsRepoPath("aws", scanLogDir), destination', False),
        ('file, "${UPLOAD_PATH}/dir/", "aws"', False),
        ('file, trtllm_utils.llmArtifactsRepoPath(location, dir), "aws"', False),
    ],
    ids=[
        "run-folder",
        "mapped-path",
        "over-several-lines",
        "no-destination",
        "mapped-but-no-destination",
        "destination-not-written-out",
        "legacy-path-as-written",
        "location-not-written-out",
    ],
)
def test_upload_check_requires_aws_and_a_mapped_target(arguments: str, expected: bool) -> None:
    assert _uploads_to_aws(arguments) is expected


def test_call_arguments_finds_each_calls_arguments() -> None:
    text = 'a.put("${dir(\'x\')}y", "aws")\nother(1)\na.put(\n    first,\n    second\n)\na.put (x, y)\n'

    assert _call_arguments(text, "a.put(") == [
        '"${dir(\'x\')}y", "aws"',
        "\n    first,\n    second\n",
        "x, y",
    ]


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_pipelines_publish_to_aws_alone(pipeline: str) -> None:
    text = (REPO_ROOT / pipeline).read_text()
    # The file uses none of the library helpers that add urm as a second place to publish to.
    for name in ("llmArtifactsTargets", "uploadToTargets", "testResultTargets"):
        assert name not in text, f"{pipeline}: {AWS_ALONE}"
    assert not re.search(r"\breplica:", text), f"{pipeline}: {AWS_ALONE}"
    # Every upload through the shared library names aws as its destination and maps its target
    # to the repository there.
    uploads = _call_arguments(text, "trtllm_utils.uploadArtifacts(")
    for arguments in uploads:
        assert _uploads_to_aws(arguments), f"{pipeline}: uploadArtifacts({arguments}): {AWS_ALONE}"
    # Each use of the library's upload is one of the calls read above.
    assert len(re.findall(r"\btrtllm_utils\s*\.\s*uploadArtifacts\b", text)) == len(uploads), (
        f"{pipeline}: {AWS_ALONE}"
    )
    # No pipeline defines or uses the plugin server named 'Artifactory', which is urm's.
    assert not re.search(r"""\bid:\s*["']Artifactory["']""", text), f"{pipeline}: {AWS_ALONE}"
    assert not re.search(r"""\bserverId:\s*["']Artifactory["']""", text), f"{pipeline}: {AWS_ALONE}"


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_pipelines_choose_no_location(pipeline: str) -> None:
    text = (REPO_ROOT / pipeline).read_text()
    found = [name for name in LOCATION_CHOICE_NAMES if name in text]
    assert found == [], f"{pipeline}: {found}: {AWS_ALONE}"


def test_read_helper_names_no_other_location() -> None:
    text = FETCH_BUILD_ARTIFACT.read_text()
    found = [name for name in (*LOCATION_CHOICE_NAMES, "LLM_ARTIFACTS_URM_URL") if name in text]
    assert found == [], f"fetch_build_artifact.sh: {found}"
    assert re.findall(r"\burm\b", text, re.IGNORECASE) == []


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_pipelines_do_not_name_the_urm_location(pipeline: str) -> None:
    text = (REPO_ROOT / pipeline).read_text()
    found = _urm_location_names(text)
    assert found == [], f"{pipeline}: {found}: {AWS_ALONE}"


@pytest.mark.parametrize(
    "line",
    [
        'trtllm_utils.uploadArtifacts(pattern, target, "urm")',
        "def host = trtllm_utils.llmArtifactsHost('urm')",
        "def repo = LLM_ARTIFACTS_LOCATIONS.urm.repo",
        "credentialsId: 'urm-artifactory-creds',",
        'def url = "https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/${path}"',
        'def url = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/${wheel}"',
        'def url = "https://urm.nvidia.com/artifactory/" + legacyPath',
        'def urmRoot = "https://urm.nvidia.com/artifactory"',
        "url: 'https://urm.nvidia.com/artifactory',",
        'upload(pattern, target) // goes to "urm"',
        '* @param location "aws" or "urm"',
    ],
    ids=[
        "double-quoted",
        "single-quoted",
        "locations-entry",
        "credential",
        "repository-url",
        "interpolated-path",
        "joined-path",
        "root-constant",
        "plugin-server-url",
        "comment-after-code",
        "block-comment",
    ],
)
def test_urm_location_check_finds_each_name(line: str) -> None:
    assert _urm_location_names(f"def before = 1\n    {line}\n") == [line]


@pytest.mark.parametrize(
    "line",
    [
        'DLFW_IMAGE = "urm.nvidia.com/docker/nvidia/pytorch:26.09-py3"',
        'def image = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:latest"',
        'sh "pip3 install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/simple"',
        'GITHUB_MIRROR="https://urm.nvidia.com/artifactory/github-go-remote"',
        'def enroot = image.replace("urm.nvidia.com/", "urm.nvidia.com#")',
        'UPLOAD_PATH = "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"',
        '// uploadArtifacts goes to "urm" when a call names no location.',
    ],
    ids=[
        "container-registry",
        "team-registry",
        "pypi-mirror",
        "github-mirror",
        "registry-host",
        "legacy-path",
        "whole-line-comment",
    ],
)
def test_urm_location_check_passes_other_services_and_comments(line: str) -> None:
    assert _urm_location_names(f"def before = 1\n    {line}\n") == []


@pytest.mark.parametrize("pipeline", ["jenkins/L0_Test.groovy", "jenkins/L0_MergeRequest.groovy"])
def test_pipelines_publish_test_results_to_aws(pipeline: str) -> None:
    text = (REPO_ROOT / pipeline).read_text()
    assert _has_line(text, UPLOAD_TEST_RESULT), f"{pipeline}: {AWS_ALONE}"
    # The upload fails the step when it fails: nothing in the function catches its failure.
    upload = _braced_block(text, "def uploadTestResult(")
    assert (
        'trtllm_utils.uploadArtifacts(filePattern, testResultsDirTarget(directory), "aws")'
        in upload
    ), f"{pipeline}: {AWS_ALONE}"
    assert "catch" not in upload, f"{pipeline}: {AWS_ALONE}"
    # The folder is this run's, in the repository on artifactory.nvidia.com.
    target = _braced_block(text, "def testResultsDirTarget(String directory)")
    assert (
        "return \"${trtllm_utils.llmArtifactsRepoPath('aws', UPLOAD_PATH)}/${directory}/\""
        in target
    ), f"{pipeline}: {AWS_ALONE}"


# ---------------------------------------------------------------------------
# The time limits of the pipelines' reads
# ---------------------------------------------------------------------------

OWN_TIME_LIMIT = (
    "A read that runs out of its own time limit takes its fallback. The expiry of a timeout step "
    "that encloses the read carries the same cause and must propagate, so the pipelines tell the "
    "two apart by the time since the read entered its own timeout step, and a read that entered "
    "none never counts as timed out."
)

# The reads that take a fallback when they run out of their own time limit, as the pipeline, the
# start of the function, the variable that holds the time at which the read entered its timeout
# step, and the variable that holds the limit in minutes.
TIMED_READS = [
    pytest.param(
        "jenkins/L0_MergeRequest.groovy",
        "def resolveBoltProfileRef(",
        "lookupStartMs",
        "lookupMinutes",
        id="bolt-profile-pin",
    ),
    pytest.param(
        "jenkins/L0_Test.groovy",
        "def prefetchBuildArtifact(",
        "startMs",
        "timeoutMinutes",
        id="prefetched-read",
    ),
    pytest.param(
        "jenkins/L0_Test.groovy",
        "def renderTestDB(",
        "readStartMs",
        "readMinutes",
        id="cbts-test-db",
    ),
]


@pytest.mark.parametrize("pipeline", ["jenkins/L0_Test.groovy", "jenkins/L0_MergeRequest.groovy"])
def test_pipelines_tell_their_own_time_limit_by_the_time_passed(pipeline: str) -> None:
    text = (REPO_ROOT / pipeline).read_text()
    check = "\n".join(
        _code_lines(_braced_block(text, "def ownTimeoutExpired(e, startMs, int limitMinutes)"))
    )

    # Without a timeout step of the caller's own, and before that step's limit has passed, the
    # answer is no, whatever interrupted the caller.
    too_early = check.index(
        "if (startMs == null || System.currentTimeMillis() - startMs < limitMinutes * 60000L) {"
    )
    no = check.index("return false", too_early)
    # Only after that does the cause decide, which keeps an abort from counting as the time limit.
    check.index("return e.causes.any { it.class.simpleName == 'ExceededTimeout' }", no)
    assert check.count("return ") == 2, f"{pipeline}: {OWN_TIME_LIMIT}"


@pytest.mark.parametrize(("pipeline", "function", "start", "limit"), TIMED_READS)
def test_reads_take_their_fallback_only_for_their_own_time_limit(
    pipeline: str, function: str, start: str, limit: str
) -> None:
    text = (REPO_ROOT / pipeline).read_text()
    lines = _code_lines(_braced_block(text, function))
    code = "\n".join(lines)
    message = f"{function}: {OWN_TIME_LIMIT}"

    # The read has one timeout step, and that step's limit has one value.
    steps = [index for index, line in enumerate(lines) if re.search(r"\btimeout\(time: ", line)]
    assert len(steps) == 1, message
    (step,) = steps
    entered = rf"(pipeline\.)?timeout\(time: {limit}, unit: 'MINUTES'\) \{{"
    assert re.fullmatch(entered, lines[step]), message
    assert len(re.findall(rf"\b{limit} = ", code)) == 1, message

    # The time is null, which stands for no timeout step, from before the try block around the
    # step until the line before the step.
    blocks = [index for index in range(step) if lines[index] == "try {"]
    assert blocks, message
    assert lines.index(f"def {start} = null") < blocks[-1], message
    assert lines[step - 1] == f"{start} = System.currentTimeMillis()", message
    assert len(re.findall(rf"\b{start} = ", code)) == 2, message

    # What interrupts the read counts as its time limit only by that time and that limit: the
    # read does not look at the cause itself.
    interrupted = code.index("FlowInterruptedException e) {", code.index(lines[step]))
    assert code.count("ownTimeoutExpired(") == 1, message
    assert "ExceededTimeout" not in code, message
    assert ".causes" not in code, message

    # The catch block starts with the decision, so the fallback is for the answer yes alone. On
    # the answer no the read throws what interrupted it again: in the branch for no, or after a
    # fallback that returns. The fallback opens no block of its own.
    own = re.escape(f"ownTimeoutExpired(e, {start}, {limit})")
    fallback = r"(?:(?!throw\b|\})[^\n]*(?<!\{)\n)"
    thrown_again = (
        rf"if \(!{own}\) \{{\nthrow e\n\}}\n"
        rf"|if \({own}\) \{{\n{fallback}+\}} else \{{\nthrow e\n\}}\n"
        rf"|if \({own}\) \{{\n{fallback}*return [^\n]*\n\}}\nthrow e\n"
    )
    decided_first = rf"FlowInterruptedException e\) \{{\n(?:{thrown_again})"
    assert re.match(decided_first, code[interrupted:]), message


def test_a_prefetched_read_without_a_time_limit_enters_no_timeout_step() -> None:
    text = (REPO_ROOT / "jenkins/L0_Test.groovy").read_text()
    lines = _code_lines(_braced_block(text, "def prefetchBuildArtifact("))

    # Such a read notes no time, so nothing that interrupts it counts as its time limit.
    step = lines.index("pipeline.timeout(time: timeoutMinutes, unit: 'MINUTES') {")
    assert lines[step - 2 : step] == [
        "if (timeoutMinutes > 0) {",
        "startMs = System.currentTimeMillis()",
    ], OWN_TIME_LIMIT


def test_the_plc_lookup_of_the_image_tags_has_a_time_limit() -> None:
    text = (REPO_ROOT / "jenkins/TensorRT_LLM_PLC.groovy").read_text()

    # The lookup runs in a stage without a time limit, before the scanning stages start, and
    # neither its query of Jenkins nor the read helper has an overall limit. The timeout command
    # gives it one.
    (minutes,) = re.findall(r"^IMAGE_KEY_TO_TAG_LOOKUP_MINUTES = (\d+)$", text, re.M)
    assert int(minutes) > 0
    read = _code_lines(_braced_block(text, "def readImageKeyToTag("))
    (command,) = [index for index, line in enumerate(read) if "get_image_key_to_tag.py" in line]
    assert read[command - 1 : command + 1] == [
        "timeout --kill-after=30 ${IMAGE_KEY_TO_TAG_LOOKUP_MINUTES}m \\\\",
        "python3 /tmp/get_image_key_to_tag.py ${imageScriptArgs} || {",
    ]
    # The step fails with the status that the timeout command ends with.
    assert read[command + 1] == "status=\\$?"
    assert 'exit "\\$status"' in read[command + 2 :]

    # Running out of that limit is a failed step, which the lookup records for the scanning stages
    # to report. An abort is not a failed lookup, and keeps propagating. A timeout step is absent:
    # its expiry would be an interruption as well.
    lookup = _code_lines(_braced_block(text, "def lookUpImageKeyToTag("))
    interrupted = lookup.index("} catch (InterruptedException e) {")
    assert lookup[interrupted - 1] == (
        "imageKeyToTagLookup = [output: readImageKeyToTag(getImageScriptArgs())]"
    )
    assert lookup[interrupted + 1 : interrupted + 4] == [
        "throw e",
        "} catch (Exception e) {",
        "imageKeyToTagLookup = [failure: e.toString()]",
    ]
    assert not re.search(r"\btimeout\(", "\n".join(read + lookup))
