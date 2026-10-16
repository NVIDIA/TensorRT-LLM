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
"""Tests for jenkins/scripts/check_artifactory_access.sh.

The script runs against in-memory stand-ins for the aws location (127.0.0.1), the urm location
(127.0.0.2), and the CDN that the aws location redirects downloads to (127.0.0.3), so the tests
need no network access and no real credentials.
"""

from __future__ import annotations

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
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECK_SCRIPT = REPO_ROOT / "jenkins" / "scripts" / "check_artifactory_access.sh"

AWS_REPO = "sw-tensorrt-llm-generic-local"
URM_REPO = "sw-tensorrt-generic"
PROBE_DIR = "llm-artifacts/access-check/20261016-test-agent"
ARTIFACT = "llm-artifacts/LLM/main/L0_PostMerge/88/TensorRT-LLM.tar.gz"

PUBLISHER = ("svc-publisher", "publisher-token-3f9a1c")
READER = ("svc-reader", "reader-token-7d2e4b")
URM_LOGIN = ("urm-user", "urm-token-5c8b0e")
CDN_SIGNATURE = "cdn-signature-91ab"

PUBLISHER_CHECKS = (
    "create probe.txt",
    "overwrite probe.txt with property access.check",
    "read properties of probe.txt",
    "copy probe.txt to probe-copy.txt",
    "list the probe directory",
    "delete probe-copy.txt",
)
READER_CHECKS = (
    "read probe.txt and match checksums.sha1",
    "read properties of probe.txt",
    "anonymous read of probe.txt is rejected",
    "write with reader credentials is rejected",
)

# Nothing secret may reach the script's output: passwords, the Basic credentials curl derives
# from them, or the signature of a CDN URL.
_SECRETS = (
    *(password for _, password in (PUBLISHER, READER, URM_LOGIN)),
    *(
        base64.b64encode(f"{login}:{password}".encode()).decode()
        for login, password in (PUBLISHER, READER, URM_LOGIN)
    ),
    CDN_SIGNATURE,
)
_SETTINGS = (
    "ARTIFACTORY_NETRC",
    "ARTIFACTORY_CREDENTIAL_ID",
    "LLM_ARTIFACTS_PRIMARY",
    "LLM_ARTIFACTS_AWS_URL",
    "LLM_ARTIFACTS_URM_URL",
)
_PROXY_SETTINGS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
_JSON = {"Content-Type": "application/json"}

Response = tuple[int, dict[str, str], bytes]


def _json(document: object) -> bytes:
    """Encode a document the way Artifactory does, with spaces around colons."""
    return json.dumps(document, indent=2, separators=(",", " : ")).encode()


def _error(status: int) -> bytes:
    return _json({"errors": [{"status": status, "message": f"status {status}"}]})


@dataclass
class StoredFile:
    """A file in a fake repository, with its Artifactory properties."""

    content: bytes
    properties: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class RecordedRequest:
    """One request a fake host received, and the status it answered with."""

    method: str
    path: str
    query: str
    identity: str
    headers: dict[str, str]
    status: int = 0


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def do_GET(self) -> None:
        self.server.fake.handle(self, "GET")

    def do_HEAD(self) -> None:
        self.server.fake.handle(self, "HEAD")

    def do_PUT(self) -> None:
        self.server.fake.handle(self, "PUT")

    def do_POST(self) -> None:
        self.server.fake.handle(self, "POST")

    def do_DELETE(self) -> None:
        self.server.fake.handle(self, "DELETE")

    def log_message(self, format: str, *args: object) -> None:
        """Keep request logs out of the test output."""


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], fake: _FakeHost) -> None:
        super().__init__(address, _Handler)
        self.fake = fake


class _FakeHost:
    """Serves HTTP on an ephemeral port of one loopback address and records every request."""

    def __init__(self, host: str) -> None:
        self.requests: list[RecordedRequest] = []
        self._lock = threading.Lock()
        self._server = _Server((host, 0), self)
        # A short poll interval keeps shutdown(), which waits for the next poll, fast.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.netloc = f"{host}:{self._server.server_port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def handle(self, handler: _Handler, method: str) -> None:
        parts = urlsplit(handler.path)
        length = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(length) if length else b""
        headers = {name.lower(): value for name, value in handler.headers.items()}
        request = RecordedRequest(
            method=method,
            path=unquote(parts.path),
            query=parts.query,
            identity=self.identify(headers.get("authorization")),
            headers=headers,
        )
        with self._lock:
            status, response_headers, payload = self.respond(request, body)
            request.status = status
            self.requests.append(request)
        handler.send_response(status)
        for name, value in response_headers.items():
            handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        if method != "HEAD":
            handler.wfile.write(payload)

    def identify(self, authorization: str | None) -> str:
        return "anonymous" if authorization is None else "invalid"

    def respond(self, request: RecordedRequest, body: bytes) -> Response:
        raise NotImplementedError


class FakeArtifactory(_FakeHost):
    """An in-memory Artifactory instance holding one repository.

    It answers the requests the access check makes: deploy with ``;key=value`` properties,
    downloads (redirected to ``cdn`` unless the request sends
    ``X-JFrog-Download-Redirect-To: None``), HEAD, the storage API for files, folders and
    properties, the copy API, and delete. ``users`` maps a (login, password) pair to
    "publisher", who may do everything, or "reader", who may only read. Requests without
    credentials may read only when ``anonymous_read`` is set.
    """

    def __init__(
        self, host: str, repo: str, users: dict[tuple[str, str], str], anonymous_read: bool
    ) -> None:
        super().__init__(host)
        self.repo = repo
        self.users = users
        self.anonymous_read = anonymous_read
        self.reader_can_write = False
        self.cdn: FakeCdn | None = None
        self.false_sha1: str | None = None
        # Artifactory lists "checksums" before "originalChecksums"; JSON readers must not rely
        # on that order.
        self.checksums_first = True
        self.files: dict[str, StoredFile] = {}
        # (method, regular expression on the path, status, body): the first match answers.
        self.overrides: list[tuple[str, str, int, bytes]] = []

    @property
    def base_url(self) -> str:
        return f"http://{self.netloc}/artifactory/{self.repo}"

    def identify(self, authorization: str | None) -> str:
        if authorization is None:
            return "anonymous"
        scheme, _, encoded = authorization.partition(" ")
        if scheme != "Basic":
            return "invalid"
        login, _, password = base64.b64decode(encoded).decode().partition(":")
        return self.users.get((login, password), "invalid")

    def respond(self, request: RecordedRequest, body: bytes) -> Response:
        for method, pattern, status, payload in self.overrides:
            if request.method == method and re.search(pattern, request.path):
                return status, _JSON, payload or _error(status)
        writes = request.method in ("PUT", "POST", "DELETE")
        if request.identity == "invalid" or (
            request.identity == "anonymous" and (writes or not self.anonymous_read)
        ):
            return 401, {**_JSON, "WWW-Authenticate": 'Basic realm="Artifactory"'}, _error(401)
        if request.identity == "reader" and writes and not self.reader_can_write:
            return 403, _JSON, _error(403)
        path = request.path.removeprefix("/artifactory/")
        if path.startswith("api/storage/"):
            return self._storage(path.removeprefix("api/storage/"), request.query)
        if path.startswith("api/copy/") and request.method == "POST":
            return self._copy(path.removeprefix("api/copy/"), request.query)
        repo, _, item = path.partition("/")
        if repo != self.repo or not item:
            return 404, _JSON, _error(404)
        if request.method == "PUT":
            return self._deploy(item, body)
        if request.method == "DELETE":
            return self._delete(item)
        return self._download(item, request.headers)

    def _file_info(self, item: str, content: bytes) -> dict[str, object]:
        sha1 = hashlib.sha1(content, usedforsecurity=False).hexdigest()
        original = {"sha1": sha1, "sha256": hashlib.sha256(content).hexdigest()}
        checksums = [
            ("checksums", {**original, "sha1": self.false_sha1 or sha1}),
            ("originalChecksums", original),
        ]
        if not self.checksums_first:
            checksums.reverse()
        return {
            "repo": self.repo,
            "path": f"/{item}",
            "size": str(len(content)),
            **dict(checksums),
            "uri": f"http://{self.netloc}/artifactory/api/storage/{self.repo}/{item}",
        }

    def _storage(self, path: str, query: str) -> Response:
        repo, _, item = path.partition("/")
        item = item.rstrip("/")
        if repo != self.repo:
            return 404, _JSON, _error(404)
        stored = self.files.get(item)
        if query == "properties":
            if stored is None or not stored.properties:
                return 404, _JSON, _error(404)
            return 200, _JSON, _json({"properties": stored.properties, "uri": f"/{item}"})
        if stored is not None:
            return 200, _JSON, _json(self._file_info(item, stored.content))
        prefix = f"{item}/" if item else ""
        children: dict[str, bool] = {}
        for key in self.files:
            if key.startswith(prefix):
                name, separator, _ = key.removeprefix(prefix).partition("/")
                children[name] = children.get(name, False) or bool(separator)
        if not children:
            return 404, _JSON, _error(404)
        listing = [
            {"uri": f"/{name}", "folder": folder} for name, folder in sorted(children.items())
        ]
        return 200, _JSON, _json({"repo": repo, "path": f"/{item}", "children": listing})

    def _copy(self, path: str, query: str) -> Response:
        repo, _, item = path.partition("/")
        target_repo, _, target = parse_qs(query).get("to", [""])[0].lstrip("/").partition("/")
        stored = self.files.get(item)
        if repo != self.repo or target_repo != self.repo or stored is None or not target:
            return 404, _JSON, _error(404)
        properties = {key: list(values) for key, values in stored.properties.items()}
        self.files[target] = StoredFile(stored.content, properties)
        message = f"copying {repo}:{item} to {target_repo}:{target} completed successfully"
        return 200, _JSON, _json({"messages": [{"level": "INFO", "message": message}]})

    def _deploy(self, item: str, body: bytes) -> Response:
        name, *parameters = item.split(";")
        properties: dict[str, list[str]] = {}
        for parameter in parameters:
            key, _, value = parameter.partition("=")
            properties.setdefault(key, []).append(value)
        self.files[name] = StoredFile(body, properties)
        return 201, _JSON, _json(self._file_info(name, body))

    def _delete(self, item: str) -> Response:
        if self.files.pop(item, None) is None:
            return 404, _JSON, _error(404)
        return 204, {}, b""

    def _download(self, item: str, headers: dict[str, str]) -> Response:
        stored = self.files.get(item)
        if stored is None:
            return 404, _JSON, _error(404)
        if self.cdn is not None and headers.get("x-jfrog-download-redirect-to") != "None":
            location = f"http://{self.cdn.netloc}/{item}?Expires=1&Signature={CDN_SIGNATURE}"
            return 302, {"Location": location}, b""
        return 200, {"Content-Type": "application/octet-stream"}, stored.content


class FakeCdn(_FakeHost):
    """Serves the files of an Artifactory stand-in at signed URLs, or rejects every request."""

    def __init__(self, host: str, source: FakeArtifactory) -> None:
        super().__init__(host)
        self.source = source
        self.reject = False

    def respond(self, request: RecordedRequest, body: bytes) -> Response:
        stored = self.source.files.get(request.path.removeprefix("/"))
        if self.reject or stored is None or request.identity != "anonymous":
            return 403, {"Content-Type": "text/xml"}, b"<Error><Code>AccessDenied</Code></Error>"
        return 200, {"Content-Type": "application/octet-stream"}, stored.content


@dataclass
class Servers:
    aws: FakeArtifactory
    urm: FakeArtifactory
    cdn: FakeCdn


@pytest.fixture
def servers() -> Iterator[Servers]:
    aws = FakeArtifactory(
        "127.0.0.1", AWS_REPO, {PUBLISHER: "publisher", READER: "reader"}, anonymous_read=False
    )
    urm = FakeArtifactory("127.0.0.2", URM_REPO, {}, anonymous_read=True)
    cdn = FakeCdn("127.0.0.3", aws)
    aws.cdn = cdn
    hosts = (aws, urm, cdn)
    for host in hosts:
        host.start()
    try:
        yield Servers(aws, urm, cdn)
    finally:
        for host in hosts:
            host.stop()


@pytest.fixture
def netrc_files(tmp_path: Path) -> dict[str, Path]:
    """Netrc files for both roles; each also names the urm host, which must never get them."""
    files = {}
    for role, (login, password) in (("publisher", PUBLISHER), ("reader", READER)):
        path = tmp_path / f"{role}.netrc"
        path.write_text(
            f"machine 127.0.0.1 login {login} password {password}\n"
            f"machine 127.0.0.2 login {URM_LOGIN[0]} password {URM_LOGIN[1]}\n"
        )
        path.chmod(0o600)
        files[role] = path
    return files


def run_check(
    servers: Servers,
    netrc: Path | None,
    *args: str,
    credential_id: str | None = None,
    aws_url: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the access check against the fake hosts and assert that it printed no secret."""
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in _SETTINGS and name.lower() not in _PROXY_SETTINGS
    }
    env.update(
        NO_PROXY="*",
        no_proxy="*",
        LLM_ARTIFACTS_AWS_URL=aws_url or servers.aws.base_url,
        LLM_ARTIFACTS_URM_URL=servers.urm.base_url,
    )
    if netrc is not None:
        env["ARTIFACTORY_NETRC"] = str(netrc)
    if credential_id is not None:
        env["ARTIFACTORY_CREDENTIAL_ID"] = credential_id
    result = subprocess.run(
        ["bash", str(CHECK_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    for secret in _SECRETS:
        assert secret not in result.stdout
        assert secret not in result.stderr
    return result


def publish_probe(servers: Servers, netrc_files: dict[str, Path]) -> None:
    result = run_check(
        servers, netrc_files["publisher"], "--role", "publisher", "--probe-dir", PROBE_DIR
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    for host in (servers.aws, servers.urm, servers.cdn):
        host.requests.clear()


def test_publisher_role_passes_every_check_in_order(
    servers: Servers, netrc_files: dict[str, Path]
) -> None:
    result = run_check(
        servers, netrc_files["publisher"], "--role", "publisher", "--probe-dir", PROBE_DIR
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert result.stdout.splitlines() == [f"PASS {check}" for check in PUBLISHER_CHECKS]
    assert list(servers.aws.files) == [f"{PROBE_DIR}/probe.txt"]
    (stamp,) = servers.aws.files[f"{PROBE_DIR}/probe.txt"].properties["access.check"]
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp)
    files = f"/artifactory/{AWS_REPO}/{PROBE_DIR}"
    storage = f"/artifactory/api/storage/{AWS_REPO}/{PROBE_DIR}"
    copy_target = f"to=/{AWS_REPO}/{PROBE_DIR}/probe-copy.txt"
    assert [(r.method, r.path, r.query, r.identity) for r in servers.aws.requests] == [
        ("PUT", f"{files}/probe.txt", "", "publisher"),
        ("PUT", f"{files}/probe.txt;access.check={stamp}", "", "publisher"),
        ("GET", f"{storage}/probe.txt", "properties", "publisher"),
        (
            "POST",
            f"/artifactory/api/copy/{AWS_REPO}/{PROBE_DIR}/probe.txt",
            copy_target,
            "publisher",
        ),
        ("GET", storage, "", "publisher"),
        ("DELETE", f"{files}/probe-copy.txt", "", "publisher"),
    ]
    assert not servers.urm.requests


def test_reader_role_passes_against_the_publisher_probe(
    servers: Servers, netrc_files: dict[str, Path]
) -> None:
    publish_probe(servers, netrc_files)

    result = run_check(servers, netrc_files["reader"], "--role", "reader", "--probe-dir", PROBE_DIR)

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert result.stdout.splitlines() == [f"PASS {check}" for check in READER_CHECKS]
    requests = servers.aws.requests
    downloads = [r for r in requests if r.path == f"/artifactory/{AWS_REPO}/{PROBE_DIR}/probe.txt"]
    assert [(r.method, r.identity, r.status) for r in downloads] == [
        ("GET", "reader", 200),
        ("GET", "anonymous", 401),
    ]
    assert downloads[0].headers.get("x-jfrog-download-redirect-to") == "None"
    storage = f"/artifactory/api/storage/{AWS_REPO}/{PROBE_DIR}/probe.txt"
    assert sorted((r.query, r.identity, r.status) for r in requests if r.path == storage) == [
        ("", "reader", 200),
        ("properties", "reader", 200),
    ]
    assert [(r.path, r.identity, r.status) for r in requests if r.method != "GET"] == [
        (f"/artifactory/{AWS_REPO}/{PROBE_DIR}/reader-write-probe.txt", "reader", 403)
    ]
    assert list(servers.aws.files) == [f"{PROBE_DIR}/probe.txt"]
    assert not servers.cdn.requests


def test_cleanup_deletes_the_probe_and_runs_no_other_check(
    servers: Servers, netrc_files: dict[str, Path]
) -> None:
    publish_probe(servers, netrc_files)

    result = run_check(
        servers,
        netrc_files["publisher"],
        "--role",
        "publisher",
        "--cleanup",
        "--probe-dir",
        f"sw-tensorrt-generic-local/{PROBE_DIR}/",
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert result.stdout.splitlines() == ["PASS delete probe.txt"]
    assert [(r.method, r.path) for r in servers.aws.requests] == [
        ("DELETE", f"/artifactory/{AWS_REPO}/{PROBE_DIR}/probe.txt")
    ]
    assert not servers.aws.files


def test_cleanup_without_a_probe_fails(servers: Servers, netrc_files: dict[str, Path]) -> None:
    result = run_check(
        servers,
        netrc_files["publisher"],
        "--role",
        "publisher",
        "--cleanup",
        "--probe-dir",
        PROBE_DIR,
    )

    assert result.returncode == 1
    assert result.stdout.splitlines() == ["FAIL delete probe.txt: got HTTP 404, expected 204"]


@pytest.mark.parametrize("checksums_first", (True, False))
def test_checksum_mismatch_fails_the_read_check(
    servers: Servers, netrc_files: dict[str, Path], checksums_first: bool
) -> None:
    publish_probe(servers, netrc_files)
    servers.aws.false_sha1 = "0" * 40
    servers.aws.checksums_first = checksums_first

    result = run_check(servers, netrc_files["reader"], "--role", "reader", "--probe-dir", PROBE_DIR)

    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert re.fullmatch(
        rf"FAIL {re.escape(READER_CHECKS[0])}: SHA-1 [0-9a-f]{{40}} of the download does not "
        rf"match checksums.sha1 0{{40}}",
        lines[0],
    )
    assert lines[1:] == [f"PASS {check}" for check in READER_CHECKS[1:]]


def test_publisher_needs_the_property_value_of_this_run(
    servers: Servers, netrc_files: dict[str, Path]
) -> None:
    stale = _json({"properties": {"access.check": ["20200101T000000Z"]}})
    servers.aws.overrides.append(("GET", r"/api/storage/.*/probe\.txt$", 200, stale))

    result = run_check(
        servers, netrc_files["publisher"], "--role", "publisher", "--probe-dir", PROBE_DIR
    )

    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert re.fullmatch(
        rf"FAIL {re.escape(PUBLISHER_CHECKS[2])}: the response has no access\.check=\d{{8}}T\d{{6}}Z "
        "property",
        lines[2],
    )
    assert "20200101T000000Z" not in lines[2]
    assert [line for index, line in enumerate(lines) if index != 2] == [
        f"PASS {check}" for index, check in enumerate(PUBLISHER_CHECKS) if index != 2
    ]


def _override(
    method: str, pattern: str, status: int, body: bytes = b""
) -> Callable[[Servers], None]:
    return lambda servers: servers.aws.overrides.append((method, pattern, status, body))


def _allow(attribute: str) -> Callable[[Servers], None]:
    return lambda servers: setattr(servers.aws, attribute, True)


@pytest.mark.parametrize(
    ("role", "arrange", "failed_check", "reason"),
    [
        pytest.param(
            "publisher",
            _override("PUT", r"/probe\.txt$", 403),
            0,
            "got HTTP 403, expected 201; the identity lacks this permission",
            id="create-forbidden",
        ),
        pytest.param(
            "publisher",
            _override("POST", r"/api/copy/", 500),
            3,
            "got HTTP 500, expected 200",
            id="copy-server-error",
        ),
        pytest.param(
            "publisher",
            _override(
                "POST",
                r"/api/copy/",
                200,
                _json({"messages": [{"level": "ERROR", "message": "copy failed"}]}),
            ),
            3,
            "the copy response reports an error",
            id="copy-error-message",
        ),
        pytest.param(
            "publisher",
            _override("GET", r"/api/storage/.*/20261016-test-agent$", 200, _json({"children": []})),
            4,
            "probe.txt is not among the children of the probe directory",
            id="listing-without-probe",
        ),
        pytest.param(
            "reader",
            _override("GET", r"/api/storage/.*/probe\.txt$", 404),
            0,
            "checksum lookup got HTTP 404, expected 200",
            id="checksum-lookup-missing",
        ),
        pytest.param(
            "reader",
            _override("GET", rf"^/artifactory/{AWS_REPO}/.*/probe\.txt$", 403),
            0,
            "download got HTTP 403, expected 200; the identity lacks this permission",
            id="download-forbidden",
        ),
        pytest.param(
            "reader",
            _override("GET", r"/api/storage/.*/probe\.txt$", 200, _json({"properties": {}})),
            1,
            "the response has no access.check property",
            id="properties-missing",
        ),
        pytest.param(
            "reader",
            _allow("anonymous_read"),
            2,
            "got HTTP 200, expected 401; anonymous reads are allowed",
            id="anonymous-read-allowed",
        ),
        pytest.param(
            "reader",
            _allow("reader_can_write"),
            3,
            "got HTTP 201, expected 403; the reader identity can write "
            "(it created reader-write-probe.txt)",
            id="reader-can-write",
        ),
    ],
)
def test_unexpected_response_fails_that_check(
    servers: Servers,
    netrc_files: dict[str, Path],
    role: str,
    arrange: Callable[[Servers], None],
    failed_check: int,
    reason: str,
) -> None:
    if role == "reader":
        publish_probe(servers, netrc_files)
    arrange(servers)

    result = run_check(servers, netrc_files[role], "--role", role, "--probe-dir", PROBE_DIR)

    checks = PUBLISHER_CHECKS if role == "publisher" else READER_CHECKS
    lines = result.stdout.splitlines()
    assert result.returncode == 1
    assert len(lines) == len(checks)
    assert lines[failed_check] == f"FAIL {checks[failed_check]}: {reason}"


def test_rejected_credentials_name_the_credential(servers: Servers, tmp_path: Path) -> None:
    netrc = tmp_path / "expired.netrc"
    netrc.write_text(f"machine 127.0.0.1 login {PUBLISHER[0]} password expired-token-0b1c\n")

    result = run_check(
        servers,
        netrc,
        "--role",
        "publisher",
        "--probe-dir",
        PROBE_DIR,
        credential_id="aws-artifactory-credentials",
    )

    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert lines[0] == (
        f"FAIL create probe.txt: got HTTP 401, expected 201; {servers.aws.netloc} did not accept "
        "the credentials in ARTIFACTORY_NETRC (credential 'aws-artifactory-credentials')"
    )
    assert len(lines) == len(PUBLISHER_CHECKS)
    assert all(line.startswith("FAIL ") for line in lines)
    assert "expired-token-0b1c" not in result.stdout + result.stderr


def test_unreachable_location_fails_every_check(
    servers: Servers, netrc_files: dict[str, Path]
) -> None:
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]

    result = run_check(
        servers,
        netrc_files["reader"],
        "--role",
        "reader",
        "--probe-dir",
        PROBE_DIR,
        aws_url=f"http://127.0.0.1:{port}/artifactory/{AWS_REPO}",
    )

    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert len(lines) == len(READER_CHECKS)
    for check, line in zip(READER_CHECKS, lines):
        assert re.fullmatch(
            rf"FAIL {re.escape(check)}: (checksum lookup )?got no HTTP response \(curl: \(7\).*\)",
            line,
        )


def _store_artifact(*locations: FakeArtifactory) -> None:
    content = bytes(range(256)) * 4096
    for location in locations:
        location.files[ARTIFACT] = StoredFile(content)


def _time_line(location: str, base_url: str, suffix: str = "") -> str:
    return (
        rf"TIME {location} \d+\.\d\d s \d+\.\d MB/s {re.escape(f'{base_url}/{ARTIFACT}')}"
        + re.escape(suffix)
    )


def test_time_reports_seconds_and_throughput_for_both_locations(
    servers: Servers, netrc_files: dict[str, Path]
) -> None:
    _store_artifact(servers.aws, servers.urm)
    publish_probe(servers, netrc_files)

    result = run_check(
        servers,
        netrc_files["reader"],
        "--role",
        "reader",
        "--probe-dir",
        PROBE_DIR,
        "--time",
        f"sw-tensorrt-generic/{ARTIFACT}",
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    lines = result.stdout.splitlines()
    assert lines[:4] == [f"PASS {check}" for check in READER_CHECKS]
    assert len(lines) == 6
    assert re.fullmatch(_time_line("aws", servers.aws.base_url), lines[4])
    assert re.fullmatch(_time_line("urm", servers.urm.base_url), lines[5])
    downloads = [r for r in servers.aws.requests if r.path.endswith(ARTIFACT)]
    assert [(r.identity, r.status) for r in downloads] == [("reader", 302)]
    assert [(r.identity, r.status) for r in servers.cdn.requests] == [("anonymous", 200)]
    assert [(r.path, r.identity, r.status) for r in servers.urm.requests] == [
        (f"/artifactory/{URM_REPO}/{ARTIFACT}", "anonymous", 200)
    ]


def test_time_retries_a_cdn_rejection_without_the_redirect(
    servers: Servers, netrc_files: dict[str, Path]
) -> None:
    _store_artifact(servers.aws, servers.urm)
    servers.cdn.reject = True

    result = run_check(
        servers,
        netrc_files["publisher"],
        "--role",
        "publisher",
        "--probe-dir",
        PROBE_DIR,
        "--time",
        ARTIFACT,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    lines = result.stdout.splitlines()
    assert lines[:6] == [f"PASS {check}" for check in PUBLISHER_CHECKS]
    assert re.fullmatch(
        _time_line("aws", servers.aws.base_url, " (after a CDN rejection, without the redirect)"),
        lines[6],
    )
    assert re.fullmatch(_time_line("urm", servers.urm.base_url), lines[7])
    downloads = [r for r in servers.aws.requests if r.path.endswith(ARTIFACT)]
    assert [(r.status, r.headers.get("x-jfrog-download-redirect-to")) for r in downloads] == [
        (302, None),
        (200, "None"),
    ]


def test_failed_timing_download_is_a_fail_line(
    servers: Servers, netrc_files: dict[str, Path]
) -> None:
    _store_artifact(servers.urm)

    result = run_check(
        servers,
        netrc_files["publisher"],
        "--role",
        "publisher",
        "--probe-dir",
        PROBE_DIR,
        "--time",
        ARTIFACT,
    )

    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert lines[:6] == [f"PASS {check}" for check in PUBLISHER_CHECKS]
    assert (
        lines[6] == f"FAIL time aws ({servers.aws.base_url}/{ARTIFACT}): got HTTP 404, expected 200"
    )
    assert re.fullmatch(_time_line("urm", servers.urm.base_url), lines[7])


@pytest.mark.parametrize(
    ("netrc_role", "args"),
    [
        pytest.param(None, ("--role", "reader", "--probe-dir", PROBE_DIR), id="no-netrc"),
        pytest.param("missing", ("--role", "reader", "--probe-dir", PROBE_DIR), id="missing-netrc"),
        pytest.param("reader", ("--role", "admin", "--probe-dir", PROBE_DIR), id="bad-role"),
        pytest.param("reader", ("--probe-dir", PROBE_DIR), id="no-role"),
        pytest.param("reader", ("--role",), id="role-without-value"),
        pytest.param("reader", ("--role", "reader"), id="no-probe-dir"),
        pytest.param(
            "reader",
            ("--role", "reader", "--probe-dir", "other-repo/llm-artifacts/x"),
            id="unmappable-probe-dir",
        ),
        pytest.param(
            "reader", ("--role", "reader", "--probe-dir", "llm-artifacts/"), id="probe-dir-is-root"
        ),
        pytest.param(
            "publisher",
            ("--role", "publisher", "--probe-dir", "llm-artifacts/x/../../builds"),
            id="dot-segments",
        ),
        pytest.param(
            "reader",
            ("--role", "reader", "--cleanup", "--probe-dir", PROBE_DIR),
            id="reader-cleanup",
        ),
        pytest.param(
            "publisher",
            ("--role", "publisher", "--cleanup", "--probe-dir", PROBE_DIR, "--time", ARTIFACT),
            id="cleanup-with-time",
        ),
        pytest.param(
            "reader",
            ("--role", "reader", "--probe-dir", PROBE_DIR, "--time", "builds/TensorRT-LLM.tar.gz"),
            id="unmappable-time-path",
        ),
        pytest.param(
            "reader",
            ("--role", "reader", "--probe-dir", PROBE_DIR, "--verbose"),
            id="unknown-option",
        ),
    ],
)
def test_usage_errors_exit_64(
    servers: Servers,
    netrc_files: dict[str, Path],
    tmp_path: Path,
    netrc_role: str | None,
    args: tuple[str, ...],
) -> None:
    netrc = tmp_path / "missing.netrc" if netrc_role == "missing" else None
    if netrc_role in netrc_files:
        netrc = netrc_files[netrc_role]

    result = run_check(servers, netrc, *args)

    assert result.returncode == 64, f"{result.stdout}\n{result.stderr}"
    assert result.stdout == ""
    assert "Usage:" in result.stderr
    assert not servers.aws.requests
    assert not servers.urm.requests


def test_base_url_with_credentials_is_refused_unprinted(
    servers: Servers, netrc_files: dict[str, Path]
) -> None:
    login, password = READER
    aws_url = f"http://{login}:{password}@{servers.aws.netloc}/artifactory/{AWS_REPO}"

    # run_check also asserts that the password is not printed.
    result = run_check(
        servers,
        netrc_files["reader"],
        "--role",
        "reader",
        "--probe-dir",
        PROBE_DIR,
        aws_url=aws_url,
    )

    assert result.returncode == 64
    assert "LLM_ARTIFACTS_AWS_URL must not contain credentials" in result.stderr
    assert not servers.aws.requests


def test_help_prints_usage(servers: Servers) -> None:
    result = run_check(servers, None, "--help")

    assert result.returncode == 0
    assert result.stdout.startswith("Usage:")
