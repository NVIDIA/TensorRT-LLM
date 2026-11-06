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

The helper reads one build artifact from the primary Artifactory location and falls back to the
secondary location only when the primary does not have it. Three loopback HTTP servers stand in
for the ``aws`` and ``urm`` locations and for the CDN that ``aws`` redirects downloads to. Each
listens on its own host name, because curl matches netrc entries by host name. A stub ``sleep`` on
``PATH`` records the back-off instead of waiting, and no request leaves the loopback interface.
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

HOSTS = {"aws": "127.0.0.1", "urm": "127.0.0.2", "cdn": "127.0.0.3"}
REPOS = {"aws": "sw-tensorrt-llm-generic-local", "urm": "sw-tensorrt-generic"}

BUILD_DIR = "llm-artifacts/LLM/main/L0_PostMerge/88"
LAYOUT_PATH = f"{BUILD_DIR}/TensorRT-LLM.tar.gz"
# Not one of the three in-scope kinds: the helper reads any file under llm-artifacts/.
OTHER_FILE = "tensorrt_llm-1.2.0-cp312-cp312-linux_x86_64.whl"
LEGACY_PATH = f"sw-tensorrt-generic/{LAYOUT_PATH}"
SYNOPSIS = "fetch_build_artifact.sh [--properties | --exists] <artifact-path> [<dest-file>]"

NETRC_LOGIN = "ci-reader"
NETRC_PASSWORD = "Pw-6c1f0e93-must-never-be-printed"
URM_NETRC_PASSWORD = "Pw-urm-41d8b2-must-never-be-sent"
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
    """The two locations and the CDN, each a loopback HTTP server on its own host.

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
        """Serve ``content`` as the artifact on a location: its storage API entry and download."""
        info = _storage_info(location, content, _sha1(content), layout_path)
        self.routes[(location, self.storage_path(location, layout_path))] = Reply(200, info)
        self.routes[(location, self.file_path(location, layout_path))] = Reply(200, content)

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

        # The urm entry, which a developer's netrc may hold, must never be used: urm is read
        # anonymously, so the helper may pass the netrc to curl only for aws requests.
        netrc = tmp_path / "netrc"
        netrc.write_text(
            f"machine {HOSTS['aws']} login {NETRC_LOGIN} password {NETRC_PASSWORD}\n"
            f"machine {HOSTS['urm']} login urm-reader password {URM_NETRC_PASSWORD}\n"
        )
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
                "LLM_ARTIFACTS_URM_URL": artifactory.base_url("urm"),
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
        assert URM_NETRC_PASSWORD not in output
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


def _served_by(location: str, layout_path: str = LAYOUT_PATH) -> str:
    return f"fetch_build_artifact: served-by={location} path={layout_path}"


def _rejection_message(credential_id: str) -> str:
    """Return the line naming a credential that aws rejected, the same for HTTP 401 and 403."""
    return (
        f"fetch_build_artifact: the Artifactory access token in credential '{credential_id}' "
        "needs renewal or lacks read permission on sw-tensorrt-llm-generic-local; an Artifactory "
        "access token for artifactory.nvidia.com is required."
    )


def _assert_no_retry_signature(text: str) -> None:
    for signature in RETRY_SIGNATURES:
        assert signature not in text


def _assert_no_credential_message(text: str) -> None:
    assert "needs renewal" not in text
    assert CREDENTIAL_ID not in text


# ---------------------------------------------------------------------------
# Path mapping and the primary location
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
def test_reads_the_primary_location(helper: Helper, artifact_path: str, layout_path: str) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package", layout_path)

    result = helper.run(artifact_path, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert not helper.part.exists()
    assert result.stderr.splitlines() == [_served_by("aws", layout_path)]
    assert result.stdout == ""
    assert artifactory.paths("aws") == [
        artifactory.storage_path("aws", layout_path),
        artifactory.file_path("aws", layout_path),
    ]
    assert [request.authorization for request in artifactory.received("aws")] == [AUTHORIZATION] * 2
    assert artifactory.paths("urm") == []
    assert result.sleeps == []


@pytest.mark.parametrize(
    "artifact_path",
    [
        f"sw-tensorrt-llm-generic-local/{LAYOUT_PATH}",
        "sw-tensorrt-generic/other-artifacts/TensorRT-LLM.tar.gz",
        f"/{LAYOUT_PATH}",
        "llm-artifacts-old/88/TensorRT-LLM.tar.gz",
        f"https://urm.nvidia.com/artifactory/{LEGACY_PATH}",
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


@pytest.mark.parametrize("location", ["aws", "urm"])
def test_base_url_with_credentials_is_a_usage_error(helper: Helper, location: str) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    variable = f"LLM_ARTIFACTS_{location.upper()}_URL"
    base_url = artifactory.base_url(location).replace("http://", "http://someone:url-secret@")

    result = helper.run(LEGACY_PATH, str(helper.dest), **{variable: base_url})

    assert result.returncode == 64
    assert f"{variable} must not contain credentials" in result.stderr
    assert "url-secret" not in result.stdout + result.stderr
    assert artifactory.requests == []


@pytest.mark.parametrize("primary", [None, "", "aws"], ids=["unset", "empty", "aws"])
def test_aws_is_the_default_primary(helper: Helper, primary: str | None) -> None:
    helper.artifactory.publish("aws", b"aws package")
    helper.artifactory.publish("urm", b"urm package")

    result = helper.run(LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_PRIMARY=primary)

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert _served_by("aws") in result.stderr.splitlines()
    assert helper.artifactory.paths("urm") == []


def test_verified_download_replaces_the_destination(helper: Helper) -> None:
    helper.artifactory.publish("aws", b"aws package")
    helper.dest.write_bytes(b"older download")
    helper.part.write_bytes(b"partial download left by an interrupted run")

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert not helper.part.exists()


# ---------------------------------------------------------------------------
# Fallback to the secondary location
# ---------------------------------------------------------------------------


def test_falls_back_to_urm_when_aws_lacks_the_file(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("urm", b"urm package")

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"urm package"
    assert result.stderr.splitlines() == [_served_by("urm")]
    assert artifactory.paths("aws") == [artifactory.storage_path("aws")]
    assert artifactory.paths("urm") == [
        artifactory.storage_path("urm"),
        artifactory.file_path("urm"),
    ]
    assert [request.authorization for request in artifactory.received("aws")] == [AUTHORIZATION]
    assert [request.authorization for request in artifactory.received("urm")] == [None, None]


def test_absent_from_both_locations_exits_2(helper: Helper) -> None:
    artifactory = helper.artifactory

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 2
    assert not helper.dest.exists()
    assert not helper.part.exists()
    assert artifactory.paths("aws") == [artifactory.storage_path("aws")]
    assert artifactory.paths("urm") == [artifactory.storage_path("urm")]
    assert result.sleeps == []
    _assert_no_retry_signature(result.stderr)


def test_rejection_by_the_secondary_exits_4(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.routes[("aws", artifactory.storage_path("aws"))] = Reply(401)

    result = helper.run(LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_PRIMARY="urm")

    assert result.returncode == 4
    assert _rejection_message(CREDENTIAL_ID) in result.stderr.splitlines()
    assert artifactory.paths("aws") == [artifactory.storage_path("aws")]
    assert result.sleeps == []


def test_errors_from_the_secondary_are_retried_then_exit_4(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.routes[("urm", artifactory.storage_path("urm"))] = Reply(500)

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    assert artifactory.paths("urm") == [artifactory.storage_path("urm")] * 3
    assert result.sleeps == ["2", "4"]
    assert not helper.dest.exists()


# ---------------------------------------------------------------------------
# Rejected requests never fall back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("step", ["lookup", "download"])
def test_rejected_credentials_exit_3_without_fallback(
    helper: Helper, step: str, status: int
) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    artifactory.publish("urm", b"urm package")
    rejected = artifactory.storage_path("aws") if step == "lookup" else artifactory.file_path("aws")
    artifactory.routes[("aws", rejected)] = Reply(status, b'{"errors": [{"status": %d}]}' % status)

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 3
    assert result.stderr.splitlines() == [
        f"fetch_build_artifact: {artifactory.host('aws')} rejected the credentials for "
        f"{LAYOUT_PATH} (HTTP {status}).",
        _rejection_message(CREDENTIAL_ID),
    ]
    _assert_no_retry_signature(result.stdout + result.stderr)
    assert artifactory.paths("aws").count(rejected) == 1
    assert artifactory.paths("urm") == []
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
    assert _served_by("aws") in result.stderr.splitlines()
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
        _served_by("aws")
    ]


def test_cdn_rejection_is_retried_without_the_redirect(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    artifactory.publish("urm", b"urm package")
    _redirect_downloads_to_cdn(artifactory, direct=Reply(200, b"aws package"))
    artifactory.routes[("cdn", CDN_PATH)] = Reply(403, b"<Error><Code>AccessDenied</Code></Error>")

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert _served_by("aws") in result.stderr.splitlines()
    downloads = [
        request
        for request in artifactory.received("aws")
        if request.path == artifactory.file_path("aws")
    ]
    assert [request.headers.get(REDIRECT_OPT_OUT) for request in downloads] == [None, "None"]
    assert [request.authorization for request in downloads] == [AUTHORIZATION] * 2
    assert [request.authorization for request in artifactory.received("cdn")] == [None]
    assert artifactory.paths("urm") == []
    assert result.sleeps == []
    _assert_no_credential_message(result.stderr)


@pytest.mark.parametrize("direct", ["rejected", "redirected-again"])
def test_persistent_cdn_rejection_exits_4(helper: Helper, direct: str) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    artifactory.publish("urm", b"urm package")
    redirect = Reply(302, headers={"Location": artifactory.origin("cdn") + CDN_PATH})
    _redirect_downloads_to_cdn(artifactory, direct=Reply(403) if direct == "rejected" else redirect)
    artifactory.routes[("cdn", CDN_PATH)] = Reply(403)

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    _assert_no_credential_message(result.stderr)
    # Three attempts, each retried once without the redirect.
    assert artifactory.paths("aws").count(artifactory.file_path("aws")) == 6
    assert artifactory.paths("urm") == []
    assert result.sleeps == ["2", "4"]
    assert not helper.dest.exists()
    assert not helper.part.exists()


@pytest.mark.parametrize("failure", ["http-500", "connection-refused"])
def test_cdn_failure_other_than_403_is_retried_through_the_redirect(
    helper: Helper, failure: str
) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    artifactory.publish("urm", b"urm package")
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
    assert artifactory.paths("urm") == []
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


def test_persistent_checksum_mismatch_exits_4_without_fallback(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    artifactory.publish("urm", b"urm package")
    artifactory.routes[("aws", artifactory.file_path("aws"))] = Reply(200, b"corrupted package")
    helper.dest.write_bytes(b"older download")

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    assert "checksum" in result.stderr
    assert _sha1(b"corrupted package") in result.stderr
    assert helper.dest.read_bytes() == b"older download"
    assert not helper.part.exists()
    assert artifactory.paths("aws").count(artifactory.file_path("aws")) == 3
    assert artifactory.paths("urm") == []
    assert result.sleeps == ["2", "4"]


def test_lookup_without_checksum_is_retried_then_exit_4(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    artifactory.publish("urm", b"urm package")
    info = json.dumps({"repo": REPOS["aws"], "path": f"/{LAYOUT_PATH}", "children": []})
    artifactory.routes[("aws", artifactory.storage_path("aws"))] = Reply(200, info.encode())

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    assert artifactory.paths("aws") == [artifactory.storage_path("aws")] * 3
    assert artifactory.paths("urm") == []
    assert result.sleeps == ["2", "4"]


# ---------------------------------------------------------------------------
# Transient failures are retried and never fall back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("step", "status"),
    [("lookup", 500), ("lookup", 429), ("download", 502), ("download", 404)],
)
def test_transient_errors_are_retried_then_exit_4(helper: Helper, step: str, status: int) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    artifactory.publish("urm", b"urm package")
    failing = artifactory.storage_path("aws") if step == "lookup" else artifactory.file_path("aws")
    artifactory.routes[("aws", failing)] = Reply(status)

    result = helper.run(LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    assert artifactory.paths("aws").count(failing) == 3
    assert artifactory.paths("urm") == []
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
    assert _served_by("aws") in result.stderr.splitlines()
    assert artifactory.paths("aws").count(artifactory.storage_path("aws")) == 2
    assert result.sleeps == ["2"]


def test_unreachable_primary_is_retried_then_exit_4(helper: Helper) -> None:
    helper.artifactory.publish("urm", b"urm package")
    closed = f"http://{HOSTS['aws']}:{_unused_port(HOSTS['aws'])}/artifactory/{REPOS['aws']}"

    result = helper.run(LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_AWS_URL=closed)

    assert result.returncode == 4
    assert helper.artifactory.paths("urm") == []
    assert result.sleeps == ["2", "4"]
    assert not helper.dest.exists()


# ---------------------------------------------------------------------------
# --exists and --properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("location", ["aws", "urm"])
@pytest.mark.parametrize("with_dest", [False, True], ids=["no-dest", "dest-ignored"])
def test_exists_prints_url_and_sha1(helper: Helper, location: str, with_dest: bool) -> None:
    artifactory = helper.artifactory
    artifactory.publish(location, b"package")
    dest_args = [str(helper.dest)] if with_dest else []

    result = helper.run("--exists", LEGACY_PATH, *dest_args)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{artifactory.file_url(location)} {_sha1(b'package')}\n"
    assert result.stderr.splitlines() == [_served_by(location)]
    assert list(helper.dest.parent.iterdir()) == []
    requested = artifactory.paths("aws") + artifactory.paths("urm")
    assert requested and all(path.startswith("/artifactory/api/storage/") for path in requested)


@pytest.mark.parametrize(("status", "returncode"), [(404, 2), (403, 3)])
def test_exists_reports_absent_and_rejected(helper: Helper, status: int, returncode: int) -> None:
    artifactory = helper.artifactory
    artifactory.routes[("aws", artifactory.storage_path("aws"))] = Reply(status)

    result = helper.run("--exists", LEGACY_PATH)

    assert result.returncode == returncode
    assert result.stdout == ""
    expected_urm = [artifactory.storage_path("urm")] if returncode == 2 else []
    assert artifactory.paths("urm") == expected_urm


@pytest.mark.parametrize("location", ["aws", "urm"])
def test_properties_follow_the_read_rule(helper: Helper, location: str) -> None:
    artifactory = helper.artifactory
    artifactory.routes[(location, artifactory.properties_path(location))] = Reply(200, PROPERTIES)

    result = helper.run("--properties", LEGACY_PATH, str(helper.dest))

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == PROPERTIES
    assert not helper.part.exists()
    assert result.stderr.splitlines() == [_served_by(location)]
    assert artifactory.paths("aws") == [artifactory.properties_path("aws")]
    expected_urm = [artifactory.properties_path("urm")] if location == "urm" else []
    assert artifactory.paths("urm") == expected_urm
    assert [request.authorization for request in artifactory.received("aws")] == [AUTHORIZATION]
    assert all(request.authorization is None for request in artifactory.received("urm"))


def test_properties_absent_from_both_exit_2(helper: Helper) -> None:
    artifactory = helper.artifactory

    result = helper.run("--properties", LEGACY_PATH, str(helper.dest))

    assert result.returncode == 2
    assert not helper.dest.exists()
    assert not helper.part.exists()
    assert artifactory.paths("aws") == [artifactory.properties_path("aws")]
    assert artifactory.paths("urm") == [artifactory.properties_path("urm")]


def test_properties_rejected_exit_3_without_fallback(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.routes[("aws", artifactory.properties_path("aws"))] = Reply(403)
    artifactory.routes[("urm", artifactory.properties_path("urm"))] = Reply(200, PROPERTIES)

    result = helper.run("--properties", LEGACY_PATH, str(helper.dest))

    assert result.returncode == 3
    assert _rejection_message(CREDENTIAL_ID) in result.stderr.splitlines()
    _assert_no_retry_signature(result.stderr)
    assert artifactory.paths("urm") == []
    assert not helper.dest.exists()


def test_properties_server_error_is_retried_then_exit_4(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.routes[("aws", artifactory.properties_path("aws"))] = Reply(500)
    artifactory.routes[("urm", artifactory.properties_path("urm"))] = Reply(200, PROPERTIES)

    result = helper.run("--properties", LEGACY_PATH, str(helper.dest))

    assert result.returncode == 4
    assert artifactory.paths("aws") == [artifactory.properties_path("aws")] * 3
    assert artifactory.paths("urm") == []
    assert result.sleeps == ["2", "4"]


# ---------------------------------------------------------------------------
# LLM_ARTIFACTS_PRIMARY=urm (rollback)
# ---------------------------------------------------------------------------


def test_primary_urm_reads_urm_first(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    artifactory.publish("urm", b"urm package")

    result = helper.run(LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_PRIMARY="urm")

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"urm package"
    assert result.stderr.splitlines() == [_served_by("urm")]
    assert artifactory.paths("aws") == []


def test_primary_urm_falls_back_to_aws(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")

    result = helper.run(LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_PRIMARY="urm")

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"aws package"
    assert result.stderr.splitlines() == [_served_by("aws")]
    assert artifactory.paths("urm") == [artifactory.storage_path("urm")]
    assert [request.authorization for request in artifactory.received("urm")] == [None]
    assert [request.authorization for request in artifactory.received("aws")] == [AUTHORIZATION] * 2


def test_primary_urm_rejection_exits_3_without_fallback(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")
    artifactory.routes[("urm", artifactory.storage_path("urm"))] = Reply(403)

    result = helper.run(LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_PRIMARY="urm")

    assert result.returncode == 3
    assert result.stderr.splitlines() == [
        f"fetch_build_artifact: {artifactory.host('urm')} rejected the anonymous request for "
        f"{LAYOUT_PATH} (HTTP 403)."
    ]
    assert artifactory.paths("aws") == []


@pytest.mark.parametrize("primary", ["bogus", "AWS", "urm "])
def test_unknown_primary_is_a_usage_error(helper: Helper, primary: str) -> None:
    helper.artifactory.publish("aws", b"aws package")

    result = helper.run(LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_PRIMARY=primary)

    assert result.returncode == 64
    assert "LLM_ARTIFACTS_PRIMARY" in result.stderr
    assert helper.artifactory.requests == []


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("netrc", ["unset", "empty", "missing-file"])
def test_aws_needs_a_netrc(helper: Helper, tmp_path: Path, netrc: str) -> None:
    helper.artifactory.publish("aws", b"aws package")
    value = {"unset": None, "empty": "", "missing-file": str(tmp_path / "no-such-netrc")}[netrc]

    result = helper.run(LEGACY_PATH, str(helper.dest), ARTIFACTORY_NETRC=value)

    assert result.returncode == 64
    assert "ARTIFACTORY_NETRC" in result.stderr
    assert helper.artifactory.requests == []


def test_primary_urm_needs_no_netrc_for_a_file_on_urm(helper: Helper) -> None:
    helper.artifactory.publish("urm", b"urm package")

    result = helper.run(
        LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_PRIMARY="urm", ARTIFACTORY_NETRC=None
    )

    assert result.returncode == 0, result.stderr
    assert helper.dest.read_bytes() == b"urm package"


def test_primary_urm_needs_a_netrc_to_fall_back_to_aws(helper: Helper) -> None:
    artifactory = helper.artifactory
    artifactory.publish("aws", b"aws package")

    result = helper.run(
        LEGACY_PATH, str(helper.dest), LLM_ARTIFACTS_PRIMARY="urm", ARTIFACTORY_NETRC=None
    )

    assert result.returncode == 64
    assert "ARTIFACTORY_NETRC" in result.stderr
    assert artifactory.paths("urm") == [artifactory.storage_path("urm")]
    assert artifactory.paths("aws") == []


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help(helper: Helper, flag: str) -> None:
    result = helper.run(flag)

    assert result.returncode == 0
    assert SYNOPSIS in result.stdout
    assert helper.artifactory.requests == []


@pytest.mark.parametrize(
    "args",
    [
        (),
        (LEGACY_PATH,),
        ("--properties", LEGACY_PATH),
        ("--exists", "--properties", LEGACY_PATH),
        ("--verbose", LEGACY_PATH, "out"),
        (LEGACY_PATH, "out", "extra"),
    ],
    ids=["no-args", "no-dest", "properties-no-dest", "two-modes", "unknown-option", "extra-arg"],
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
EXIT_CODES = {"FETCH_ABSENT": 2, "FETCH_REJECTED": 3, "FETCH_TRANSFER_FAILED": 4, "FETCH_USAGE": 64}


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
        + _braced_block(text, "def fetchBuildArtifact("),
    ),
    # The Slurm job's download of the packaged build.
    ExitCodeReader(
        "jenkins/scripts/slurm_install.sh",
        _SHELL_DEFINITION,
        lambda text: _between(text, "if ! retry_command", "tar -zxf"),
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
        lambda text: _python_definitions(text, {"_HELPER_FAILURES", "_fetch_build_artifact"}),
    ),
    ExitCodeReader(
        "scripts/get_wheel_from_package.py",
        _PYTHON_DEFINITION,
        lambda text: _python_definitions(text, {"download_package"}),
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
