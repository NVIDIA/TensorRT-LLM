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
"""Tests for the promotion of a BOLT profile bundle in ``jenkins/BoltProfileGen.groovy``.

``promoteBundle`` builds a shell script that runs on a Slurm cluster's frontend and uploads the
bundle to the profile store: to the primary location, which holds the required copy, and, while
``aws`` is the primary, to ``urm`` as a replica. The script reads its credentials as a netrc on
stdin, which the pipeline's agent writes with a line of shell in front of the ssh command. The
tests take the text of both from the Groovy source, evaluate the Groovy statements around them in
Python, and run the result the way the pipeline does: the agent's line with ``sh -xe``, and the
script with ``bash`` from a file beside the bundle, with the netrc on stdin. The loopback servers
of ``test_bolt_fetch_verified`` stand in for the two Artifactory locations, each on its own host,
because curl matches netrc entries by host name. No request leaves the loopback interface. The
tests cut curl's retry count in the script's text, so that each test takes a few seconds at most.
One test also reads ``publishBoltedCanonical``, whose upload script runs curl the same way.
"""

from __future__ import annotations

import base64
import os
import re
import stat
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from .test_bolt_fetch_verified import (
    _PROXY_VARIABLES,
    BOLT_PROFILE_GEN,
    FakeServers,
    Reply,
    Request,
    _groovy_unescape,
    _strip_indent,
    _unused_port,
)

pytestmark = pytest.mark.cpu_only

BRANCH = "main"
TRIPLE = "aarch64-linux-gnu"
REF = "0a1b2c3d4e5f"
BUNDLE_NAME = f"bolt-profile-{REF}-{TRIPLE}.tar.gz"
# Short enough that no curl version asks the server to confirm the upload before sending it.
BUNDLE = b"profile bundle " * 32

HOSTS = {"aws": "127.0.0.1", "urm": "127.0.0.2"}
# Where the store is on each location: the location's repository, then the same path.
REPOS = {"aws": "sw-tensorrt-llm-generic-local", "urm": "sw-tensorrt-generic"}
LAYOUT_DIR = f"llm-artifacts/bolt-profiles/{BRANCH}/{TRIPLE}"
STORE = {location: f"/artifactory/{repo}/{LAYOUT_DIR}" for location, repo in REPOS.items()}
# latest.tar.gz as its upload names it: the matrix parameters set the three properties in the same
# request as the bytes.
LATEST = f"latest.tar.gz;bolt.ref={REF};bolt.branch={BRANCH};bolt.triple={TRIPLE}"
# The permissions that the upload of each file needs, as a rejection names them.
PERMISSION = {BUNDLE_NAME: "deploy or delete", LATEST: "deploy, annotate, or delete"}

LOGINS = {"aws": "ci-publisher-aws", "urm": "ci-publisher-urm"}
# Each holds a printf conversion and a shell expansion, which must reach the netrc as written.
PASSWORDS = {
    "aws": "Pw-aws-7f3a91-%d-$HOME-must-never-be-printed",
    "urm": "Pw-urm-c08e55-%d-$HOME-must-never-be-printed",
}
BASIC_TOKENS = {
    location: base64.b64encode(f"{LOGINS[location]}:{PASSWORDS[location]}".encode()).decode()
    for location in HOSTS
}

# The statements of promoteBundle that the tests evaluate in Python instead of taking their
# text. Each must be in the function as written here: a change to one of them needs the same
# change wherever the Python stands for it: _promotion_locations, _layout_dir,
# _location_values, _promote_script, _netrc_feed, and Promotion.write_netrc. The binding of each
# location's credential and the check that a replica's credential binds are among them, because
# the tests take every credential to bind.
_MIRRORED_STATEMENTS = (
    "def bundleName = bundle.tokenize('/').last()",
    "def layoutDir = llmArtifactsLayoutPath(promoteDir)",
    'def locations = primary == "aws" ? ["aws", "urm"] : ["urm"]',
    "for (location in locations) {",
    "def base = llmArtifactsUrl(location, promoteDir)",
    r"""def host = base.replaceFirst(/^https?:\/\//, "").tokenize('/').first()""",
    "def repo = LLM_ARTIFACTS_LOCATIONS[location].repo",
    "def credentialId = LLM_ARTIFACTS_LOCATIONS[location].credential",
    'def userVariable = "ART_USER_" + location.toUpperCase()',
    'def passwordVariable = "ART_PASS_" + location.toUpperCase()',
    "def binding = pipeline.usernamePassword(credentialsId: credentialId,",
    "usernameVariable: userVariable, passwordVariable: passwordVariable)",
    "if (location != primary) {",
    "pipeline.withCredentials([binding]) {}",
    "continue",
    "bindings.add(binding)",
    "def runPromote = bashRemoteScriptFile(pipeline, remote, promote,",
    "pipeline.withCredentials(bindings) {",
    "script: feed + Utils.sshUserCmd(remote, runPromote))",
)
# How many lines of promoteBundle name each variable that the tests rebuild in Python, comment
# lines aside. The tests take each one's value from the statements above and from the strings
# they read, so one more line that names a variable could change it where the tests do not look:
# another location or another order, text added to the script, or a second reader of the netrc.
_NAMED_ON_LINES = {
    "locations": 3,
    "uploads": 3,
    "promote": 3,
    "bindings": 3,
    "netrcEntries": 3,
    "runPromote": 2,
    "feed": 2,
}
# The same, for the helpers that map a legacy path to a location's address.
_MIRRORED_HELPER_STATEMENTS = (
    r'def layoutPath = legacyPath.replaceFirst(/^sw-tensorrt-generic(-local)?\//, "")',
    'return "${LLM_ARTIFACTS_LOCATIONS[location].repo}/${llmArtifactsLayoutPath(legacyPath)}"',
    'return "${LLM_ARTIFACTS_LOCATIONS[location].root}/'
    '${llmArtifactsRepoPath(location, legacyPath)}"',
)

# What ends a location's part of the script when its upload fails: a replica's failure by
# default, and the required copy's when the location is the primary.
_ON_FAILURE = re.compile(
    r'^\s*def onFailure = "(?P<replica>(?:[^"\\\n]|\\.)*)"\n'
    r"\s*if \(location == primary\) \{\n"
    r'\s*onFailure = "(?P<required>(?:[^"\\\n]|\\.)*)"\n'
    r"\s*\}$",
    re.M,
)

# The line of shell that writes a location's netrc entry on the agent.
_NETRC_ENTRY = re.compile(r'^\s*netrcEntries\.add\("(?P<entry>(?:[^"\\\n]|\\.)*)"\)$', re.M)
# Where the agent's text holds the entries of all locations.
_JOINED_ENTRIES = "${netrcEntries.join('; ')}"

# What Artifactory sends besides the status when it has stored a file.
_STORED = b'{"repo": "store", "created": "2026-01-01T00:00:00.000Z"}\n'

# The parts of a double-quoted Groovy string that are not plain text: an escape, a `${name}`, and
# any other `$`.
_GSTRING_PART = re.compile(
    r"(?P<escape>\\(?:u[0-9A-Fa-f]{4}|[0-3][0-7]{0,2}|[4-7][0-7]?|.))|\$(?:\{(?P<name>\w+)\})?",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# The script as Groovy builds it
# ---------------------------------------------------------------------------


def _gstring(literal: str, values: Mapping[str, str]) -> str:
    """Return the value of the text between the quotes of a double-quoted Groovy string.

    Escapes are read as in a single-quoted string, and each ``${name}`` that is not escaped
    becomes ``values[name]``. Groovy also reads ``$name`` and ``${expression}``; the promotion's
    strings use neither, and one that did fails the test.
    """

    def replace(match: re.Match[str]) -> str:
        if match["escape"] is not None:
            return _groovy_unescape(match["escape"])
        if match["name"] is None or match["name"] not in values:
            rest = literal[match.start() : match.start() + 40]
            pytest.fail(f"cannot evaluate the Groovy string at {rest!r}")
        return values[match["name"]]

    return _GSTRING_PART.sub(replace, literal)


def _has_statement(source: str, statement: str) -> bool:
    return re.search(rf"^[ \t]*{re.escape(statement)}$", source, re.M) is not None


def _between(source: str, start: str, end: str) -> str:
    """Return the text of the Groovy string that follows the one ``start`` and ends at ``end``."""
    begin = source.find(start)
    stop = source.find(end, begin + len(start)) if begin != -1 else -1
    if stop == -1 or source.count(start) != 1:
        pytest.fail(f"promoteBundle: expected one {start!r} and then {end!r}")
    literal = source[begin + len(start) : stop]
    # Groovy ends the string at the first """ in it.
    assert '"""' not in literal, f'the string after {start!r} contains """'
    return literal


def _definition(source: str, name: str) -> str:
    """Return the text between the quotes of ``def <name> = "..."``."""
    match = re.search(rf'^\s*def {name} = "((?:[^"\\\n]|\\.)*)"$', source, re.M)
    if match is None:
        pytest.fail(f'promoteBundle: no `def {name} = "..."`')
    return match[1]


def _promote_bundle() -> str:
    """Return the source of ``promoteBundle``, with the statements that the tests mirror checked."""
    source = BOLT_PROFILE_GEN.read_text()
    function = re.search(r"^def promoteBundle\(.*?^\}$", source, re.M | re.S)
    if function is None:
        pytest.fail(f"cannot find promoteBundle in {BOLT_PROFILE_GEN}")
    for statement in _MIRRORED_STATEMENTS:
        assert _has_statement(function[0], statement), f"promoteBundle no longer has: {statement}"
    for statement in _MIRRORED_HELPER_STATEMENTS:
        assert _has_statement(source, statement), f"the pipeline no longer has: {statement}"
    code = [line.strip() for line in function[0].splitlines() if not line.lstrip().startswith("//")]
    for name, count in _NAMED_ON_LINES.items():
        named = [line for line in code if re.search(rf"\b{name}\b", line)]
        assert len(named) == count, f"promoteBundle names {name} on {len(named)} lines: {named}"
    return function[0]


def _locations() -> dict[str, dict[str, str]]:
    """Return the pipeline's ``LLM_ARTIFACTS_LOCATIONS``: each location's settings by name."""
    source = BOLT_PROFILE_GEN.read_text()
    block = re.search(r"^LLM_ARTIFACTS_LOCATIONS = \[\n(.*?)^\]$", source, re.M | re.S)
    if block is None:
        pytest.fail(f"cannot find LLM_ARTIFACTS_LOCATIONS in {BOLT_PROFILE_GEN}")
    locations = {
        name: {"root": root, "repo": repo, "credential": credential}
        for name, root, repo, credential in re.findall(
            r'(\w+): \[root: "([^"]+)", repo: "([^"]+)",\s*credential: "([^"]+)"\]', block[1]
        )
    }
    assert sorted(locations) == sorted(HOSTS), f"locations read from the source: {locations}"
    return locations


def _promotion_locations(primary: str) -> list[str]:
    """Return the locations that a promotion uploads to, the required copy first."""
    return ["aws", "urm"] if primary == "aws" else ["urm"]


def _layout_dir(function: str) -> str:
    """Return the store's directory below a location's repository, from ``promoteBundle``."""
    promote_dir = _gstring(
        _definition(function, "promoteDir"), {"BRANCH": BRANCH, "TRIPLE": TRIPLE}
    )
    return re.sub(r"^sw-tensorrt-generic(-local)?/", "", promote_dir)


def _location_values(location: str, root: str, layout_dir: str) -> dict[str, str]:
    """Return the values of the variables that ``promoteBundle`` sets for ``location``.

    ``root`` is the location's root, in place of the one in ``LLM_ARTIFACTS_LOCATIONS``.
    """
    settings = _locations()[location]
    base = f"{root}/{settings['repo']}/{layout_dir}"
    return {
        "base": base,
        "host": base.split("://", 1)[1].split("/", 1)[0],
        "repo": settings["repo"],
        "credentialId": settings["credential"],
        "userVariable": "ART_USER_" + location.upper(),
        "passwordVariable": "ART_PASS_" + location.upper(),
    }


def _promote_script(primary: str, roots: Mapping[str, str], bundle: Path) -> str:
    """Return the script that ``promoteBundle`` builds to promote ``bundle``.

    The script's text, the store's directory, and each location's repository and credential come
    from the Groovy source. ``roots`` gives each location's root, in place of the one in
    ``LLM_ARTIFACTS_LOCATIONS``. Every location's credential is taken to bind.
    """
    function = _promote_bundle()
    values = {
        "bundle": str(bundle),
        "bundleName": bundle.name,
        "BRANCH": BRANCH,
        "TRIPLE": TRIPLE,
        "BOLT_REF": REF,
    }
    for name in ("netrc", "response"):
        values[name] = _gstring(_definition(function, name), values)
    values["layoutDir"] = _layout_dir(function)

    on_failure = _ON_FAILURE.search(function)
    if on_failure is None:
        pytest.fail("promoteBundle: cannot find what it does when an upload fails")
    # Each string must end where its statement does: text that Groovy appended after it would
    # be missing from the script that the tests run.
    upload = _between(function, 'uploads += """', '""".stripIndent()\n')
    uploads = ""
    for location in _promotion_locations(primary):
        location_values = {
            **values,
            **_location_values(location, roots[location], values["layoutDir"]),
        }
        outcome = "required" if location == primary else "replica"
        location_values["onFailure"] = _gstring(on_failure[outcome], location_values)
        uploads += _strip_indent(_gstring(upload, location_values))

    promote = _between(function, 'def promote = """', '""".stripIndent() + uploads\n')
    return _strip_indent(_gstring(promote, values)) + uploads


def _netrc_feed(
    primary: str, roots: Mapping[str, str], hosts: Mapping[str, str] | None = None
) -> str:
    """Return the shell text that the agent runs in front of the ssh command.

    It writes the promotion's netrc to the pipe: an entry for each location, with the login and
    the password from the variables that the location's credential is bound to. The text comes
    from the Groovy source.

    Args:
        primary: The primary location, ``aws`` or ``urm``.
        roots: Each location's root, in place of the one in ``LLM_ARTIFACTS_LOCATIONS``.
        hosts: A host for each location's entry to name, in place of the one in its root.
    """
    function = _promote_bundle()
    entry = _NETRC_ENTRY.search(function)
    if entry is None:
        pytest.fail("promoteBundle: cannot find the netrc entry that it adds for a location")
    layout_dir = _layout_dir(function)
    entries = []
    for location in _promotion_locations(primary):
        values = _location_values(location, roots[location], layout_dir)
        if hosts is not None:
            values["host"] = hosts[location]
        entries.append(_gstring(entry["entry"], values))
    feed = _definition(function, "feed")
    assert feed.count(_JOINED_ENTRIES) == 1, f"the agent's text does not join the entries: {feed}"
    return _gstring(feed.replace(_JOINED_ENTRIES, "${entries}"), {"entries": "; ".join(entries)})


def _with_one_retry(script: str) -> str:
    """Return ``script`` with curl's retry count cut to one.

    curl waits 1 s before its first retry. The script must pass the count once and set no delay,
    so that a change to how it retries fails here instead of slowing the tests down.
    """
    script, count = re.subn(r"--retry \d+\b", "--retry 1", script)
    assert count == 1, f"the script passes --retry {count} times"
    assert "--retry-delay" not in script and "--retry-max-time" not in script
    return script


# ---------------------------------------------------------------------------
# Running the script
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """The outcome of a promotion script."""

    returncode: int
    stdout: str
    stderr: str

    def lines(self, level: str) -> list[str]:
        """Return the lines of either stream that start with ``[<level>]``."""
        output = self.stdout.splitlines() + self.stderr.splitlines()
        return [line for line in output if line.startswith(f"[{level}]")]


class Promotion:
    """Runs the script that ``promoteBundle`` builds against the fake locations."""

    def __init__(self, tmp_path: Path, fake: FakeServers) -> None:
        self.fake = fake
        self._store = tmp_path / "_bundle"
        self.bundle = self._store / BUNDLE_NAME
        self.netrc = self._store / f"{BUNDLE_NAME}.netrc"
        self._script = self._store / f"{BUNDLE_NAME}.promote.sh"
        self._cwd = tmp_path / "cwd"
        self._tmpdir = tmp_path / "tmp"
        # Where the stand-in for the ssh command keeps what it reads on stdin.
        self._ssh_stdin = tmp_path / "ssh.stdin"
        home = tmp_path / "home"
        for directory in (self._store, self._cwd, self._tmpdir, home):
            directory.mkdir()
        self.bundle.write_bytes(BUNDLE)
        # A curlrc that would show the Authorization header in the output, were curl to read it.
        (home / ".curlrc").write_text("verbose\n")
        # Each location's root on its fake server, with the path of the pipeline's root.
        self.roots = {
            location: fake.url(location, urlsplit(settings["root"]).path)
            for location, settings in _locations().items()
        }
        # The runner's proxy settings, curl configuration folders, and shell start-up files are
        # left out. Where `sh` is a bash that reads BASH_ENV, as it can be in the CI image, that
        # file's commands would be traced in front of the agent's.
        dropped = ("CURL_HOME", "XDG_CONFIG_HOME", "BASH_ENV", "ENV")
        env = {
            name: value
            for name, value in os.environ.items()
            if name.lower() not in _PROXY_VARIABLES and name not in dropped
        }
        # The C locale is always there, so bash has no warning about the locale to print.
        env.update({"HOME": str(home), "TMPDIR": str(self._tmpdir), "LC_ALL": "C"})
        self._env = env

    def write_netrc(
        self, primary: str, roots: Mapping[str, str], hosts: Mapping[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run the agent's part of a promotion, with a stand-in for the ssh command.

        The agent's shell step traces its commands, and it has the credential of each of the
        promotion's locations bound to that location's variables. The stand-in keeps what it
        reads on stdin in a file, as ssh passes its stdin to the cluster. The step's own output
        goes to the build's log, so it must be empty. In the result, the output is what the
        stand-in read, which is the netrc, and the error output is the trace.
        """
        env = dict(self._env, PS4="+ ", SSH_STDIN=str(self._ssh_stdin))
        for location in _promotion_locations(primary):
            env["ART_USER_" + location.upper()] = LOGINS[location]
            env["ART_PASS_" + location.upper()] = PASSWORDS[location]
        self._ssh_stdin.unlink(missing_ok=True)
        step = subprocess.run(
            ["sh", "-xe", "-c", _netrc_feed(primary, roots, hosts) + 'cat > "$SSH_STDIN"'],
            env=env,
            cwd=self._cwd,
            # The step has no input of its own for the stand-in to read.
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert step.stdout == "", "the agent's text wrote to the step's log"
        netrc = self._ssh_stdin.read_text() if self._ssh_stdin.exists() else ""
        return subprocess.CompletedProcess(step.args, step.returncode, netrc, step.stderr)

    def run(
        self, primary: str, *, before_netrc: Callable[[], None] | None = None, **roots: str
    ) -> Result:
        """Promote the bundle with ``primary`` as the primary location.

        The netrc on stdin is the one that the agent's part writes, except that each entry names
        its fake server's address without the port: curl looks an entry up by host name alone.
        Every run checks that no password reaches the script, its output, or the agent's trace,
        and that the script leaves nothing beside the bundle but itself.

        Args:
            primary: The primary location, ``aws`` or ``urm``.
            before_netrc: Called when the script has started and its stdin is still empty.
            **roots: A root to use for a location in place of its fake server's.
        """
        roots = {**self.roots, **roots}
        script = _with_one_retry(_promote_script(primary, roots, self.bundle))
        self._script.write_text(script)
        agent = self.write_netrc(primary, roots, HOSTS)
        assert agent.returncode == 0, agent.stderr

        with subprocess.Popen(
            ["bash", str(self._script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            cwd=self._cwd,
            text=True,
            # Whatever the test runner's umask is, the script starts with one that lets every
            # user read the files it creates.
            umask=0o022,
        ) as proc:
            try:
                if before_netrc is not None:
                    before_netrc()
            finally:
                try:
                    stdout, stderr = proc.communicate(agent.stdout, timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    raise

        for secret in (*PASSWORDS.values(), *BASIC_TOKENS.values()):
            assert secret not in script
            assert secret not in agent.stderr + stdout + stderr
        left = sorted(path.name for path in self._store.iterdir())
        assert left == sorted(p.name for p in (self.bundle, self._script) if p.exists()), (
            "the script left files beside the bundle"
        )
        assert list(self._cwd.iterdir()) == [], "files were written to the working directory"
        assert list(self._tmpdir.iterdir()) == [], "temporary files were left behind"
        return Result(proc.returncode, stdout, stderr)


@pytest.fixture
def fake() -> Iterator[FakeServers]:
    servers = FakeServers(HOSTS)
    yield servers
    servers.close()


@pytest.fixture
def promotion(tmp_path: Path, fake: FakeServers) -> Promotion:
    return Promotion(tmp_path, fake)


def _path(location: str, name: str) -> str:
    """Return the request path of the store's file ``name`` on ``location``."""
    return f"{STORE[location]}/{name}"


def _accept(fake: FakeServers, location: str) -> None:
    """Make ``location`` accept the upload of the bundle and of ``latest.tar.gz``."""
    for name in (BUNDLE_NAME, LATEST):
        fake.routes[(location, _path(location, name))] = Reply(201, _STORED)


def _failure(status: int) -> Reply:
    """Return a reply with ``status`` and a body, as Artifactory answers a request it fails."""
    return Reply(status, b'{"errors": [{"status": %d, "message": "failed"}]}\n' % status)


def _uploads(fake: FakeServers) -> list[tuple[str, str, str]]:
    """Return the requests in the order received, as ``(server, method, path)``."""
    return [(request.server, request.method, request.path) for request in fake.requests]


def _promoted(fake: FakeServers, location: str) -> str:
    """Return the line that the script logs when it has promoted the bundle on ``location``."""
    return f"[INFO] promoted {BUNDLE_NAME} and latest.tar.gz on {fake.host(location)}"


def _puts(location: str) -> list[tuple[str, str, str]]:
    """Return a location's two uploads in order: the versioned bundle, then ``latest.tar.gz``."""
    return [(location, "PUT", _path(location, name)) for name in (BUNDLE_NAME, LATEST)]


# ---------------------------------------------------------------------------
# The script's text
# ---------------------------------------------------------------------------


def test_groovy_gstring_processing() -> None:
    values = {"host": "store.test"}

    assert _gstring('echo \\"\\$A ${host}\\" \\\\\n', values) == 'echo "$A store.test" \\\n'
    for literal in ("$host", "${other}", "${host.size()}"):
        with pytest.raises(pytest.fail.Exception):
            _gstring(literal, values)


@pytest.mark.parametrize("primary", ["aws", "urm"])
def test_script_is_valid_bash_and_holds_no_credential(
    promotion: Promotion, tmp_path: Path, primary: str
) -> None:
    script = _promote_script(primary, promotion.roots, promotion.bundle)
    path = tmp_path / "promote.sh"
    path.write_text(script)

    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, timeout=30)

    assert proc.returncode == 0, proc.stderr
    # Groovy turned each escaped dollar into a shell expansion.
    assert 'status=$(http_status --speed-time 300 --speed-limit 1024 -T "' in script
    # The credentials are bound on the agent and reach the script only as the netrc on stdin.
    assert "ART_USER" not in script and "ART_PASS" not in script
    assert f'cat > "{promotion.netrc}"' in script
    assert script.count("if promote_to ") == len(_promotion_locations(primary))


@pytest.mark.parametrize("primary", ["aws", "urm"])
def test_agent_writes_a_netrc_entry_for_each_location(promotion: Promotion, primary: str) -> None:
    roots = {location: settings["root"] for location, settings in _locations().items()}

    agent = promotion.write_netrc(primary, roots)

    assert agent.returncode == 0, agent.stderr
    # With the pipeline's roots, each entry names its location's host.
    assert agent.stdout == "".join(
        f"machine {urlsplit(roots[location]).netloc} login {LOGINS[location]} "
        f"password {PASSWORDS[location]}\n"
        for location in _promotion_locations(primary)
    )
    # The step's trace was on, and it ends before the commands that hold the credentials.
    assert agent.stderr.splitlines() == ["+ set +x"]


@pytest.mark.parametrize("name", ["promoteBundle", "publishBoltedCanonical"])
def test_upload_scripts_keep_curl_from_reading_a_curlrc(name: str) -> None:
    function = re.search(rf"^def {name}\(.*?^\}}$", BOLT_PROFILE_GEN.read_text(), re.M | re.S)
    assert function is not None, f"cannot find {name} in {BOLT_PROFILE_GEN}"
    # Each curl command that is given the netrc. curl obeys -q only as its first argument.
    commands = re.findall(r"^[ \t]*(curl [^\n]*--netrc-file)", function[0], re.M)
    assert len(commands) == 1, f"{name} runs curl with the netrc {len(commands)} times"
    assert commands[0].startswith("curl -q "), f"{name}: {commands[0]}"


# ---------------------------------------------------------------------------
# aws as the primary
# ---------------------------------------------------------------------------


def test_aws_primary_promotes_to_aws_and_then_to_urm(promotion: Promotion) -> None:
    fake = promotion.fake
    _accept(fake, "aws")
    _accept(fake, "urm")

    result = promotion.run("aws")

    assert result.returncode == 0, result.stderr
    assert _uploads(fake) == _puts("aws") + _puts("urm")
    assert {request.body for request in fake.requests} == {BUNDLE}
    # Each location gets its own credential and no other.
    for location in HOSTS:
        assert {request.headers.get("authorization") for request in fake.received(location)} == {
            f"Basic {BASIC_TOKENS[location]}"
        }
    assert result.stdout.splitlines() == [_promoted(fake, "aws"), _promoted(fake, "urm")]
    assert result.stderr == ""


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("name", [BUNDLE_NAME, LATEST], ids=["bundle", "latest"])
def test_aws_rejection_fails_the_promotion_and_names_the_credential(
    promotion: Promotion, name: str, status: int
) -> None:
    fake = promotion.fake
    _accept(fake, "aws")
    _accept(fake, "urm")
    fake.routes[("aws", _path("aws", name))] = _failure(status)

    result = promotion.run("aws")

    assert result.returncode == 1
    # The rejected request is not repeated, the uploads stop at it, and urm is not asked.
    rejected = (BUNDLE_NAME, LATEST).index(name)
    assert _uploads(fake) == _puts("aws")[: rejected + 1]
    [error] = result.lines("ERROR")
    assert error in result.stderr
    file_name = name.split(";")[0]
    assert f"{fake.host('aws')} rejected the credentials for {LAYOUT_DIR}/{file_name} " in error
    assert f"(HTTP {status})" in error
    assert f"credential '{_locations()['aws']['credential']}'" in error
    assert f"lacks {PERMISSION[name]} permission on {REPOS['aws']}." in error
    assert result.lines("INFO") == [] and result.lines("WARNING") == []


@pytest.mark.parametrize(
    ("status", "requests"),
    [(302, 1), (404, 1), (409, 1), (500, 2), (503, 2)],
    ids=["redirect", "not-found", "conflict", "server-error", "unavailable"],
)
def test_aws_failure_fails_the_promotion_with_its_status(
    promotion: Promotion, status: int, requests: int
) -> None:
    fake = promotion.fake
    _accept(fake, "aws")
    _accept(fake, "urm")
    fake.routes[("aws", _path("aws", BUNDLE_NAME))] = _failure(status)

    result = promotion.run("aws")

    assert result.returncode == 1
    # curl repeats a request that a server error answered, once here; urm is not asked.
    assert _uploads(fake) == _puts("aws")[:1] * requests
    assert result.lines("ERROR") == [
        f"[ERROR] The request for {LAYOUT_DIR}/{BUNDLE_NAME} on {fake.host('aws')} failed "
        f"(HTTP {status})."
    ]
    assert result.lines("INFO") == []


def test_aws_out_of_reach_fails_the_promotion(promotion: Promotion) -> None:
    fake = promotion.fake
    _accept(fake, "urm")
    closed = f"{HOSTS['aws']}:{_unused_port(HOSTS['aws'])}"

    result = promotion.run("aws", aws=f"http://{closed}/artifactory")

    assert result.returncode == 1
    assert fake.requests == []
    [error] = result.lines("ERROR")
    assert f"{LAYOUT_DIR}/{BUNDLE_NAME} on {closed} failed (HTTP 000)" in error


# ---------------------------------------------------------------------------
# urm as the replica
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403, 404, 500])
@pytest.mark.parametrize("name", [BUNDLE_NAME, LATEST], ids=["bundle", "latest"])
def test_urm_replica_failure_is_logged_and_the_promotion_stands(
    promotion: Promotion, name: str, status: int
) -> None:
    fake = promotion.fake
    _accept(fake, "aws")
    _accept(fake, "urm")
    fake.routes[("urm", _path("urm", name))] = _failure(status)

    result = promotion.run("aws")

    assert result.returncode == 0, result.stderr
    # aws holds both files before urm is asked, and urm's uploads stop at the one that failed.
    failed = (BUNDLE_NAME, LATEST).index(name)
    assert _uploads(fake)[:2] == _puts("aws")
    assert set(_uploads(fake)[2:]) == set(_puts("urm")[: failed + 1])
    assert result.lines("INFO") == [_promoted(fake, "aws")]
    [warning] = result.lines("WARNING")
    assert warning in result.stderr
    assert f"{LAYOUT_DIR}/{name.split(';')[0]} " in warning
    assert f"(HTTP {status})" in warning
    assert f"{fake.host('urm')} holds a replica, so the promotion stands." in warning
    if status in (401, 403):
        assert f"credential '{_locations()['urm']['credential']}'" in warning
        assert f"lacks {PERMISSION[name]} permission on {REPOS['urm']}." in warning
    assert result.lines("ERROR") == []


def test_urm_replica_out_of_reach_is_logged_and_the_promotion_stands(
    promotion: Promotion,
) -> None:
    fake = promotion.fake
    _accept(fake, "aws")
    closed = f"{HOSTS['urm']}:{_unused_port(HOSTS['urm'])}"

    result = promotion.run("aws", urm=f"http://{closed}/artifactory")

    assert result.returncode == 0, result.stderr
    assert _uploads(fake) == _puts("aws")
    assert result.lines("INFO") == [_promoted(fake, "aws")]
    [warning] = result.lines("WARNING")
    assert f"{LAYOUT_DIR}/{BUNDLE_NAME} on {closed} failed (HTTP 000)" in warning
    assert result.lines("ERROR") == []


# ---------------------------------------------------------------------------
# urm as the primary
# ---------------------------------------------------------------------------


def test_urm_primary_promotes_to_urm_alone(promotion: Promotion) -> None:
    fake = promotion.fake
    _accept(fake, "aws")
    _accept(fake, "urm")

    result = promotion.run("urm")

    assert result.returncode == 0, result.stderr
    assert _uploads(fake) == _puts("urm")
    assert {request.body for request in fake.requests} == {BUNDLE}
    assert {request.headers.get("authorization") for request in fake.requests} == {
        f"Basic {BASIC_TOKENS['urm']}"
    }
    assert result.stdout.splitlines() == [_promoted(fake, "urm")]
    assert result.stderr == ""


@pytest.mark.parametrize("status", [403, 500])
def test_urm_primary_failure_fails_the_promotion(promotion: Promotion, status: int) -> None:
    fake = promotion.fake
    _accept(fake, "aws")
    fake.routes[("urm", _path("urm", BUNDLE_NAME))] = _failure(status)

    result = promotion.run("urm")

    assert result.returncode == 1
    assert fake.received("aws") == []
    assert {request.path for request in fake.received("urm")} == {_path("urm", BUNDLE_NAME)}
    [error] = result.lines("ERROR")
    assert f"{LAYOUT_DIR}/{BUNDLE_NAME} " in error and f"(HTTP {status})" in error
    assert fake.host("urm") in error
    assert result.lines("INFO") == [] and result.lines("WARNING") == []


# ---------------------------------------------------------------------------
# The netrc and the response file
# ---------------------------------------------------------------------------


def test_netrc_is_private_from_its_creation_to_the_last_upload(promotion: Promotion) -> None:
    fake = promotion.fake
    netrc = promotion.netrc
    created: list[int] = []
    uploading: list[int | None] = []

    def wait_for_netrc() -> None:
        # The script creates the file and then waits for its content, which run() holds back
        # until this returns.
        deadline = time.monotonic() + 30
        while not netrc.exists():
            assert time.monotonic() < deadline, "the script did not create the netrc"
            time.sleep(0.01)
        created.append(stat.S_IMODE(netrc.stat().st_mode))

    def accept(request: Request) -> Reply:
        uploading.append(stat.S_IMODE(netrc.stat().st_mode) if netrc.exists() else None)
        return Reply(201, _STORED)

    for location in HOSTS:
        for name in (BUNDLE_NAME, LATEST):
            fake.routes[(location, _path(location, name))] = accept

    result = promotion.run("aws", before_netrc=wait_for_netrc)

    assert result.returncode == 0, result.stderr
    # Only its owner can read the netrc, from before it holds a credential until the uploads
    # are over; run() checks that it is gone afterwards, with the response file.
    assert created == [0o600]
    assert uploading == [0o600] * 4


def test_missing_bundle_fails_before_any_request(promotion: Promotion) -> None:
    promotion.bundle.unlink()

    result = promotion.run("aws")

    assert result.returncode == 1
    assert promotion.fake.requests == []
    assert result.lines("ERROR") == [f"[ERROR] bundle missing: {promotion.bundle}"]
