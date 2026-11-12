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
"""Tests for ``fetch_verified`` in ``jenkins/BoltProfileGen.groovy``.

BOLT profile generation downloads its input build and llvm-bolt on a Slurm cluster with shell
functions that the pipeline keeps in the Groovy string ``boltFetchLib``. The tests build that
string from the Groovy source the way Groovy does, source it in bash, and call ``fetch_verified``
the way the stages do: as a plain command in a ``set -e`` script. Two loopback HTTP servers stand
in for the ``aws`` Artifactory location and for the CDN that it redirects downloads to. Each
listens on its own host, because curl matches netrc entries by host name, and no request leaves
the loopback interface. The tests cut the retry counts and delays in the extracted text, so that
each test takes a few seconds at most.
"""

from __future__ import annotations

import base64
import hashlib
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
BOLT_PROFILE_GEN = REPO_ROOT / "jenkins" / "BoltProfileGen.groovy"
# The statement that defines the library, through its opening quotes, and the text after its
# closing quotes.
LIBRARY_START = "def boltFetchLib = '''"
LIBRARY_END = "'''.stripIndent()"

HOSTS = {"aws": "127.0.0.1", "cdn": "127.0.0.3"}
FILE_PATH = (
    "/artifactory/sw-tensorrt-llm-generic-local/llm-artifacts/LLM/main/L0_PostMerge/88/"
    "TensorRT-LLM.tar.gz"
)
# A signed CDN URL: the signature is a credential of its own and must not reach the log.
CDN_PATH = f"/cdn{FILE_PATH}?Expires=1798761600&Signature=c2lnbmVkLWNkbi11cmw&Key-Pair-Id=KTEST"
CONTENT = b"packaged build " * 64

NETRC_LOGIN = "ci-reader"
NETRC_PASSWORD = "Pw-5e0b7a14-must-never-be-printed"
BASIC_TOKEN = base64.b64encode(f"{NETRC_LOGIN}:{NETRC_PASSWORD}".encode()).decode()
AUTHORIZATION = f"Basic {BASIC_TOKEN}"

# Asks Artifactory to send a file itself instead of redirecting the download to its CDN.
REDIRECT_OPT_OUT = "x-jfrog-download-redirect-to"

# The retry counts and delays in the library, and the values the tests use instead: curl's options
# in the parallel download, and fetch_verified's own retries of a failed single-stream attempt.
# curl waits --retry-delay seconds before each retry; without that option it waits 1 s, then twice
# as long.
FAST_RETRIES = (
    (r"--retry \d+", "--retry 1"),
    (r"--retry-delay \d+", "--retry-delay 1"),
    (r"\bretries=\d+", "retries=1"),
    (r"\bretry_delay=\d+", "retry_delay=1"),
)
# The single-stream attempts that fetch_verified makes with FAST_RETRIES before it gives up.
ATTEMPTS = 2

# The escapes of a single-quoted Groovy string. A backslash at the end of a line joins the next
# line to it.
_GROOVY_ESCAPES = {
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "$": "$",
    "\n": "",
}
_GROOVY_ESCAPE = re.compile(r"\\(u[0-9A-Fa-f]{4}|[0-3][0-7]{0,2}|[4-7][0-7]?|.)", re.DOTALL)

_PROXY_VARIABLES = {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "no_proxy"}

# Calls fetch_verified the way the stages do: as a plain command in a `set -e` script. The EXIT
# trap records the shell functions that were running when the script ended. There are none when
# fetch_verified returned, and there are some when `set -e` cut it short.
_SCRIPT = """\
set -e
trap 'printf "%s" "${FUNCNAME[*]:-}" > "$EXITED_IN"' EXIT
. "$BOLT_FETCH_LIB"
_verify_test() { [ "$(sha1sum < "$1" | cut -d' ' -f1)" = "$EXPECTED_SHA1" ]; }
fetch_verified "$URL" "$DEST" _verify_test
echo "fetch_verified returned 0"
"""


# ---------------------------------------------------------------------------
# The library as Groovy builds it
# ---------------------------------------------------------------------------


def _groovy_unescape(literal: str) -> str:
    """Return the value of the text between the quotes of a single-quoted Groovy string."""

    def replace(match: re.Match[str]) -> str:
        escape = match.group(1)
        if len(escape) == 5:
            return chr(int(escape[1:], 16))
        if escape[0] in "01234567":
            return chr(int(escape, 8))
        if escape not in _GROOVY_ESCAPES:
            pytest.fail(f"boltFetchLib contains \\{escape}, which is not a Groovy escape")
        return _GROOVY_ESCAPES[escape]

    return _GROOVY_ESCAPE.sub(replace, literal)


def _strip_indent(text: str) -> str:
    """Return ``text`` as Groovy's ``String.stripIndent()`` does.

    The smallest indent among the lines that are not blank is removed from every line, and blank
    lines become empty.
    """
    lines = text.split("\n")
    indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
    return "\n".join(line[indent:] if line.strip() else "" for line in lines)


def _bolt_fetch_lib() -> str:
    """Return the shell text of ``boltFetchLib`` as the pipeline builds it."""
    source = BOLT_PROFILE_GEN.read_text()
    start = source.find(LIBRARY_START)
    end = source.find(LIBRARY_END, start) if start != -1 else -1
    if end == -1 or source.count(LIBRARY_START) != 1:
        pytest.fail(
            f"cannot find the boltFetchLib string in {BOLT_PROFILE_GEN}: expected one "
            f"{LIBRARY_START!r} and then {LIBRARY_END!r}"
        )
    literal = source[start + len(LIBRARY_START) : end]
    # Groovy ends the string at the first ''' in it.
    assert "'''" not in literal, "boltFetchLib contains ''', which ends the Groovy string early"
    return _strip_indent(_groovy_unescape(literal))


def _with_fast_retries(library: str) -> str:
    """Return ``library`` with every curl retry count and delay replaced as ``FAST_RETRIES`` says.

    Each pattern must match, and no other retry value may remain, so that a change to how the
    library passes these options fails here instead of slowing the tests down.
    """
    for pattern, replacement in FAST_RETRIES:
        library, count = re.subn(rf"{pattern}\b", replacement, library)
        assert count, f"boltFetchLib has no {pattern!r} to replace"
    values = re.findall(r"(?:--retry(?:-delay)?(?:\s+|=)|\bretries=|\bretry_delay=)(\S+)", library)
    assert set(values) == {"1"}, f"retry values that FAST_RETRIES misses: {values}"
    return library


# ---------------------------------------------------------------------------
# Fake servers
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    """An HTTP response for a fake server to send; a HEAD request gets it without the body."""

    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    # A Content-Length above the body's length drops the connection partway through the body.
    length: int | None = None
    # After the body, keep the connection open without sending more until the test releases it.
    stall: bool = False


@dataclass
class Request:
    """A request that a fake server received; header names are lower case."""

    server: str
    method: str
    path: str
    headers: dict[str, str]

    @property
    def opted_out(self) -> bool:
        return self.headers.get(REDIRECT_OPT_OUT) == "None"


Route = Reply | Callable[[Request], Reply]


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._answer(with_body=True)

    def do_HEAD(self) -> None:
        self._answer(with_body=False)

    def _answer(self, *, with_body: bool) -> None:
        server = cast(_Server, self.server)
        headers = {name.lower(): value for name, value in self.headers.items()}
        reply = server.fake.answer(Request(server.name, self.command, self.path, headers))
        self.send_response(reply.status)
        for name, value in reply.headers.items():
            self.send_header(name, value)
        length = len(reply.body) if reply.length is None else reply.length
        self.send_header("Content-Length", str(length))
        self.end_headers()
        if with_body:
            self.wfile.write(reply.body)
            if reply.stall:
                self.wfile.flush()
                server.fake.hold()

    def log_message(self, format: str, *args: object) -> None:
        """Keep access logs out of the test output."""


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, name: str, fake: FakeServers) -> None:
        super().__init__((HOSTS[name], 0), _Handler)
        self.name = name
        self.fake = fake


class FakeServers:
    """The aws location and its CDN, each a loopback HTTP server on its own host.

    Routes map ``(server, path with query)`` to a reply. A request without a route gets 404.
    """

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Route] = {}
        self.requests: list[Request] = []
        self._lock = threading.Lock()
        self._released = threading.Event()
        self._held: list[threading.Thread] = []
        self._servers: dict[str, _Server] = {}
        try:
            for name in HOSTS:
                self._servers[name] = _Server(name, self)
        except OSError as error:
            for server in self._servers.values():
                server.server_close()
            pytest.skip(f"cannot listen on the 127.0.0.0/8 loopback range: {error}")
        # A short poll interval keeps shutdown(), which waits for the next poll, fast.
        self._threads = [
            threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
            for server in self._servers.values()
        ]
        for thread in self._threads:
            thread.start()

    def close(self) -> None:
        self.release()
        for server in self._servers.values():
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join()

    def hold(self) -> None:
        """Keep the calling handler's connection open until ``release()``."""
        with self._lock:
            self._held.append(threading.current_thread())
        self._released.wait(timeout=60)

    def release(self) -> None:
        """Let the handlers in ``hold()`` finish, and wait until they have.

        A test that stalls a transfer calls this before it returns, because the thread leak check
        runs before fixtures are torn down.
        """
        self._released.set()
        with self._lock:
            held = list(self._held)
        for thread in held:
            thread.join(timeout=10)

    def host(self, name: str) -> str:
        """Return the server's ``host:port``."""
        host, port = self._servers[name].server_address[:2]
        return f"{host}:{port}"

    def url(self, name: str, path: str) -> str:
        return f"http://{self.host(name)}{path}"

    def answer(self, request: Request) -> Reply:
        with self._lock:
            self.requests.append(request)
            route = self.routes.get((request.server, request.path))
            if route is None:
                return Reply(404)
            return route if isinstance(route, Reply) else route(request)

    def received(self, name: str) -> list[Request]:
        with self._lock:
            return [request for request in self.requests if request.server == name]

    def gets(self) -> list[tuple[str, bool]]:
        """Return the GET requests in the order received, as ``(server, opted out)``."""
        with self._lock:
            return [(r.server, r.opted_out) for r in self.requests if r.method == "GET"]


def _redirect_to(fake: FakeServers, location: str, *, direct: Reply) -> None:
    """Make aws redirect downloads to ``location``, and answer ``direct`` to those that opt out."""
    redirect = Reply(302, headers={"Location": location})
    fake.routes[("aws", FILE_PATH)] = lambda request: direct if request.opted_out else redirect


def _redirect_to_cdn(fake: FakeServers, *, cdn: Route, direct: Reply) -> None:
    """Make aws redirect downloads to the CDN, which answers with ``cdn``."""
    _redirect_to(fake, fake.url("cdn", CDN_PATH), direct=direct)
    fake.routes[("cdn", CDN_PATH)] = cdn


def _ranged(content: bytes) -> Callable[[Request], Reply]:
    """Build a route that serves ``content`` and answers byte-range requests."""

    def answer(request: Request) -> Reply:
        headers = {"Accept-Ranges": "bytes"}
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers.get("range", ""))
        if match is None:
            return Reply(200, content, headers)
        first, last = int(match[1]), int(match[2])
        headers["Content-Range"] = f"bytes {first}-{last}/{len(content)}"
        return Reply(206, content[first : last + 1], headers)

    return answer


def _in_turn(*replies: Reply) -> Callable[[Request], Reply]:
    """Build a route that answers successive GET requests with ``replies``, then repeats the last.

    A HEAD request gets the reply that the next GET request will get.
    """
    gets = 0

    def answer(request: Request) -> Reply:
        nonlocal gets
        reply = replies[min(gets, len(replies) - 1)]
        if request.method == "GET":
            gets += 1
        return reply

    return answer


def _unused_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


# ---------------------------------------------------------------------------
# Running fetch_verified
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """The outcome of a script that called ``fetch_verified``."""

    returncode: int
    stdout: str
    stderr: str
    # The shell functions that were running when the script exited: none when fetch_verified
    # returned, and fetch_verified among them when `set -e` cut it short.
    exited_in: str


class FetchVerified:
    """Runs ``fetch_verified`` from the extracted library against the fake servers."""

    def __init__(self, tmp_path: Path, fake: FakeServers, library: str) -> None:
        self.fake = fake
        self.dest = tmp_path / "out" / "TensorRT-LLM.tar.gz"
        self._cwd = tmp_path / "cwd"
        self._tmpdir = tmp_path / "tmp"
        self._exited_in = tmp_path / "exited_in"
        home = tmp_path / "home"
        for directory in (self.dest.parent, self._cwd, self._tmpdir, home):
            directory.mkdir()
        lib = tmp_path / "bolt_fetch_lib.sh"
        lib.write_text(library)
        self._script = tmp_path / "fetch.sh"
        self._script.write_text(_SCRIPT)

        entry = f"machine {HOSTS['aws']} login {NETRC_LOGIN} password {NETRC_PASSWORD}\n"
        netrc = tmp_path / "netrc"
        # curl reads ~/.netrc only when told to, so a download without FETCH_NETRC sends nothing.
        for path in (netrc, home / ".netrc"):
            path.write_text(entry)
            path.chmod(0o600)

        env = {
            name: value
            for name, value in os.environ.items()
            if name.lower() not in _PROXY_VARIABLES
            and name not in ("CURL_HOME", "XDG_CONFIG_HOME", "FETCH_NETRC", "PARTS")
        }
        env.update(
            {
                "HOME": str(home),
                "TMPDIR": str(self._tmpdir),
                "BOLT_FETCH_LIB": str(lib),
                "EXITED_IN": str(self._exited_in),
                "URL": fake.url("aws", FILE_PATH),
                "DEST": str(self.dest),
                "FETCH_NETRC": str(netrc),
            }
        )
        self._env = env

    def run(self, expected: bytes = CONTENT, **variables: str | None) -> Result:
        """Download the aws file to ``dest``, with a check that it holds ``expected``.

        Every run checks that the output has neither a signed CDN URL nor the netrc password, and
        that no file is left anywhere but at ``dest``.

        Args:
            expected: The content that the check accepts.
            **variables: Variables to set for the script, such as FETCH_NETRC; ``None`` unsets one.
        """
        env = dict(self._env, EXPECTED_SHA1=hashlib.sha1(expected).hexdigest())
        for name, value in variables.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        self._exited_in.unlink(missing_ok=True)

        proc = subprocess.run(
            ["bash", str(self._script)],
            env=env,
            cwd=self._cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = proc.stdout + proc.stderr
        assert "Signature=" not in output, "a signed CDN URL reached the output"
        assert NETRC_PASSWORD not in output
        assert BASIC_TOKEN not in output
        assert list(self._tmpdir.iterdir()) == [], "temporary files were left behind"
        assert list(self._cwd.iterdir()) == [], "files were written outside DEST"
        leftovers = [path.name for path in self.dest.parent.iterdir() if path != self.dest]
        assert leftovers == [], "partial files were left next to DEST"
        assert self._exited_in.exists(), "the script's EXIT trap did not run"
        return Result(proc.returncode, proc.stdout, proc.stderr, self._exited_in.read_text())


@pytest.fixture(scope="module")
def library() -> str:
    return _with_fast_retries(_bolt_fetch_lib())


@pytest.fixture
def fake() -> Iterator[FakeServers]:
    servers = FakeServers()
    yield servers
    servers.close()


@pytest.fixture
def fetch(tmp_path: Path, fake: FakeServers, library: str) -> FetchVerified:
    return FetchVerified(tmp_path, fake, library)


# ---------------------------------------------------------------------------
# The extracted library
# ---------------------------------------------------------------------------


def test_groovy_string_processing() -> None:
    literal = "\n    a \\\\\n      b\\\n c \\$x \\'q\\' \\u0041\\t\n\n    "

    assert _groovy_unescape(literal) == "\n    a \\\n      b c $x 'q' A\t\n\n    "
    assert _strip_indent(_groovy_unescape(literal)) == "\na \\\n  b c $x 'q' A\t\n\n"


def test_library_is_valid_bash(library: str, tmp_path: Path) -> None:
    lib = tmp_path / "bolt_fetch_lib.sh"
    lib.write_text(library)

    proc = subprocess.run(["bash", "-n", str(lib)], capture_output=True, text=True, timeout=30)

    assert proc.returncode == 0, proc.stderr
    # Groovy turned each doubled backslash into one.
    assert "printf '%s\\n'" in library
    assert "fetch_verified() {" in library


def test_long_scripts_are_not_sent_on_the_ssh_command_line() -> None:
    """Long stage scripts run from a file copied to the cluster.

    ``b64BashRemoteCmd`` and ``b64BashRemoteCmdStdin`` put the whole script, base64-encoded, into
    one word of the remote command, and a csh/tcsh login shell rejects a word that long with
    "Word too long". Every script that carries the library is long, and so is ``publish``, the
    script that publishes the BOLTed build.
    """
    source = BOLT_PROFILE_GEN.read_text()
    carriers = re.findall(
        r'def (\w+) = """(?:(?!""").)*"""\.stripIndent\(\) \+ boltFetchLib \+', source, re.S
    )

    assert sorted(carriers) == ["llvmStage", "tarStage"]
    for name in [*carriers, "publish"]:
        assert not re.search(rf"b64BashRemoteCmd(?:Stdin)?\(\s*{name}\b", source), name
        assert re.search(rf"bashRemoteScriptFile\(pipeline, remote, {name},", source), name


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("https://artifactory.nvidia.com/artifactory/repo/a.tar.gz", "artifactory.nvidia.com"),
        ("http://127.0.0.3:8080/cdn/a.tar.gz?Signature=c2ln", "127.0.0.3:8080"),
        ("https://reader@GitHub.com/llvm/a.tar.xz#part", "github.com"),
        ("https://d1.cloudfront.net?Signature=c2ln", "d1.cloudfront.net"),
    ],
)
def test_url_host(library: str, tmp_path: Path, url: str, host: str) -> None:
    lib = tmp_path / "bolt_fetch_lib.sh"
    lib.write_text(library)

    proc = subprocess.run(
        ["bash", "-c", '. "$1" && _bolt_url_host "$2"', "bash", str(lib), url],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"{host}\n"


# ---------------------------------------------------------------------------
# The single-stream download
# ---------------------------------------------------------------------------


def _retrying(reason: str) -> str:
    """Return the line that fetch_verified logs when its first single-stream attempt fails."""
    return f"[INFO] single-stream attempt 1 of {ATTEMPTS}: {reason}; retrying in 1s."


def _gave_up(reason: str) -> str:
    """Return the start of the line that fetch_verified logs when its last attempt fails."""
    return f"[ERROR] single-stream attempt {ATTEMPTS} of {ATTEMPTS}: {reason}: "


def test_cdn_rejection_is_retried_once_without_the_redirect(fetch: FetchVerified) -> None:
    fake = fetch.fake
    rejected = Reply(403, b"<Error><Code>AccessDenied</Code></Error>")
    _redirect_to_cdn(fake, cdn=rejected, direct=Reply(200, CONTENT))

    result = fetch.run()

    assert result.returncode == 0, result.stderr
    assert result.exited_in == ""
    assert fetch.dest.read_bytes() == CONTENT
    assert fake.gets() == [("aws", False), ("cdn", False), ("aws", True)]
    # Exactly the requests after the CDN rejected the download opt out of the redirect.
    requests = fake.requests
    cdn_get = next(i for i, r in enumerate(requests) if r.server == "cdn" and r.method == "GET")
    assert [r.opted_out for r in requests] == [i > cdn_get for i in range(len(requests))]
    assert (
        f"[INFO] {fake.host('cdn')} rejected the redirected download (HTTP 403); "
        f"downloading from {fake.host('aws')} without the redirect."
    ) in result.stdout.splitlines()
    assert "single-stream attempt" not in result.stdout + result.stderr


@pytest.mark.parametrize("status", [401, 404, 500, 520])
def test_cdn_failure_other_than_403_is_retried_through_the_redirect_then_fails(
    fetch: FetchVerified, status: int
) -> None:
    fake = fetch.fake
    _redirect_to_cdn(fake, cdn=Reply(status), direct=Reply(200, CONTENT))

    result = fetch.run()

    assert result.returncode == 1
    assert result.exited_in == ""
    assert not fetch.dest.exists()
    assert not any(request.opted_out for request in fake.requests)
    # Each attempt starts again from the aws URL, which redirects to the CDN again.
    assert fake.gets() == [("aws", False), ("cdn", False)] * ATTEMPTS
    reason = f"download answered HTTP {status} from {fake.host('cdn')}"
    assert _retrying(reason) in result.stdout.splitlines()
    assert _gave_up(reason) in result.stderr


def test_attempt_after_a_failed_one_can_succeed(fetch: FetchVerified) -> None:
    fake = fetch.fake
    cdn = _in_turn(Reply(520), Reply(200, CONTENT))
    _redirect_to_cdn(fake, cdn=cdn, direct=Reply(500))

    result = fetch.run()

    assert result.returncode == 0, result.stderr
    assert result.exited_in == ""
    assert fetch.dest.read_bytes() == CONTENT
    assert fake.gets() == [("aws", False), ("cdn", False)] * 2
    reason = f"download answered HTTP 520 from {fake.host('cdn')}"
    assert _retrying(reason) in result.stdout.splitlines()


def test_cdn_stall_is_retried_through_the_redirect_then_fails(
    tmp_path: Path, fake: FakeServers, library: str
) -> None:
    # curl ends a transfer that moves fewer than --speed-limit bytes a second for --speed-time
    # seconds.
    stalling_library, count = re.subn(r"--speed-time \d+\b", "--speed-time 1", library)
    assert count, "boltFetchLib has no --speed-time to shorten"
    fetch = FetchVerified(tmp_path, fake, stalling_library)
    stalled = Reply(200, CONTENT[:100], length=len(CONTENT), stall=True)
    _redirect_to_cdn(fake, cdn=stalled, direct=Reply(200, CONTENT))

    try:
        result = fetch.run()
    finally:
        fake.release()

    assert result.returncode == 1
    assert result.exited_in == ""
    assert not fetch.dest.exists()
    assert not any(request.opted_out for request in fake.requests)
    assert fake.gets() == [("aws", False), ("cdn", False)] * ATTEMPTS
    assert _gave_up(f"download from {fake.host('cdn')} failed (curl exit 28)") in result.stderr


def test_rejection_by_the_location_itself_is_not_retried_without_the_redirect(
    fetch: FetchVerified,
) -> None:
    fake = fetch.fake
    fake.routes[("aws", FILE_PATH)] = lambda request: (
        Reply(200, CONTENT) if request.opted_out else Reply(403)
    )

    result = fetch.run()

    assert result.returncode == 1
    assert result.exited_in == ""
    assert not fetch.dest.exists()
    assert not any(request.opted_out for request in fake.requests)
    # It fails the attempt like any other status, and the next attempt asks the same way.
    assert fake.gets() == [("aws", False)] * ATTEMPTS
    assert fake.received("cdn") == []
    assert _gave_up(f"download answered HTTP 403 from {fake.host('aws')}") in result.stderr


def test_download_without_a_redirect_does_not_opt_out(fetch: FetchVerified) -> None:
    fake = fetch.fake
    fake.routes[("aws", FILE_PATH)] = Reply(200, CONTENT)

    result = fetch.run()

    assert result.returncode == 0, result.stderr
    assert result.exited_in == ""
    assert fetch.dest.read_bytes() == CONTENT
    assert "single-stream curl" in result.stdout
    assert fake.gets() == [("aws", False)]
    assert not any(request.opted_out for request in fake.requests)
    assert fake.received("cdn") == []


def test_credentials_go_to_the_location_only(fetch: FetchVerified) -> None:
    fake = fetch.fake
    _redirect_to_cdn(fake, cdn=Reply(403), direct=Reply(200, CONTENT))

    result = fetch.run()

    assert result.returncode == 0, result.stderr
    assert {request.headers.get("authorization") for request in fake.received("aws")} == {
        AUTHORIZATION
    }
    assert {request.headers.get("authorization") for request in fake.received("cdn")} == {None}


@pytest.mark.parametrize("fetch_netrc", ["", None], ids=["empty", "unset"])
def test_download_without_fetch_netrc_is_anonymous(
    fetch: FetchVerified, fetch_netrc: str | None
) -> None:
    fake = fetch.fake
    _redirect_to_cdn(fake, cdn=Reply(200, CONTENT), direct=Reply(500))

    result = fetch.run(FETCH_NETRC=fetch_netrc)

    assert result.returncode == 0, result.stderr
    assert fetch.dest.read_bytes() == CONTENT
    assert fake.gets() == [("aws", False), ("cdn", False)]
    assert {request.headers.get("authorization") for request in fake.requests} == {None}


def test_parallel_download_follows_the_redirect(fetch: FetchVerified) -> None:
    fake = fetch.fake
    _redirect_to_cdn(fake, cdn=_ranged(CONTENT), direct=Reply(500))

    result = fetch.run()

    assert result.returncode == 0, result.stderr
    assert fetch.dest.read_bytes() == CONTENT
    assert "[INFO] fetched via 16-way parallel: " in result.stdout
    assert not any(request.opted_out for request in fake.requests)
    assert {request.headers.get("authorization") for request in fake.received("aws")} == {
        AUTHORIZATION
    }
    assert {request.headers.get("authorization") for request in fake.received("cdn")} == {None}


# ---------------------------------------------------------------------------
# Failures under set -e
# ---------------------------------------------------------------------------


@dataclass
class Failure:
    """How a download fails: the start of the line it logs, and its GET requests in order."""

    message: str
    gets: list[tuple[str, bool]]


def _cdn_server_error(fake: FakeServers) -> Failure:
    _redirect_to_cdn(fake, cdn=Reply(503), direct=Reply(200, CONTENT))
    return Failure(
        _gave_up(f"download answered HTTP 503 from {fake.host('cdn')}"),
        [("aws", False), ("cdn", False)] * ATTEMPTS,
    )


def _cdn_refuses_connections(fake: FakeServers) -> Failure:
    closed = f"{HOSTS['cdn']}:{_unused_port(HOSTS['cdn'])}"
    _redirect_to(fake, f"http://{closed}{CDN_PATH}", direct=Reply(200, CONTENT))
    return Failure(
        _gave_up(f"download from {closed} failed (curl exit 7)"), [("aws", False)] * ATTEMPTS
    )


def _cdn_drops_the_connection(fake: FakeServers) -> Failure:
    dropped = Reply(200, CONTENT[:100], length=len(CONTENT))
    _redirect_to_cdn(fake, cdn=dropped, direct=Reply(200, CONTENT))
    return Failure(
        _gave_up(f"download from {fake.host('cdn')} failed (curl exit 18)"),
        [("aws", False), ("cdn", False)] * ATTEMPTS,
    )


def _location_rejects(fake: FakeServers) -> Failure:
    fake.routes[("aws", FILE_PATH)] = Reply(403)
    return Failure(
        _gave_up(f"download answered HTTP 403 from {fake.host('aws')}"),
        [("aws", False)] * ATTEMPTS,
    )


def _download_without_the_redirect_fails(fake: FakeServers) -> Failure:
    _redirect_to_cdn(fake, cdn=Reply(403), direct=Reply(500))
    aws = fake.host("aws")
    return Failure(
        _gave_up(f"download from {aws} without the redirect answered HTTP 500 from {aws}"),
        [("aws", False), ("cdn", False), ("aws", True)] * ATTEMPTS,
    )


def _download_without_the_redirect_is_redirected(fake: FakeServers) -> Failure:
    redirect = Reply(302, headers={"Location": fake.url("cdn", CDN_PATH)})
    _redirect_to_cdn(fake, cdn=Reply(403), direct=redirect)
    return Failure(
        _gave_up(
            f"download from {fake.host('aws')} without the redirect answered HTTP 403 "
            f"from {fake.host('cdn')}"
        ),
        [("aws", False), ("cdn", False), ("aws", True), ("cdn", True)] * ATTEMPTS,
    )


def _download_fails_the_check(fake: FakeServers) -> Failure:
    _redirect_to_cdn(fake, cdn=Reply(200, b"corrupted package"), direct=Reply(200, CONTENT))
    return Failure("[ERROR] verification failed: ", [("aws", False), ("cdn", False)])


def _download_without_the_redirect_fails_the_check(fake: FakeServers) -> Failure:
    _redirect_to_cdn(fake, cdn=Reply(403), direct=Reply(200, b"corrupted package"))
    return Failure("[ERROR] verification failed: ", [("aws", False), ("cdn", False), ("aws", True)])


@pytest.mark.parametrize(
    "failure",
    [
        _cdn_server_error,
        _cdn_refuses_connections,
        _cdn_drops_the_connection,
        _location_rejects,
        _download_without_the_redirect_fails,
        _download_without_the_redirect_is_redirected,
        _download_fails_the_check,
        _download_without_the_redirect_fails_the_check,
    ],
    ids=lambda failure: failure.__name__.strip("_").replace("_", "-"),
)
def test_failure_returns_1_under_set_e_and_removes_dest(
    fetch: FetchVerified, failure: Callable[[FakeServers], Failure]
) -> None:
    expected = failure(fetch.fake)
    fetch.dest.write_bytes(b"earlier download")

    result = fetch.run()

    assert result.returncode == 1, result.stderr
    assert result.exited_in == "", f"set -e ended the script inside {result.exited_in}"
    assert "fetch_verified returned 0" not in result.stdout
    assert expected.message in result.stderr
    assert fetch.fake.gets() == expected.gets
    assert not fetch.dest.exists()
