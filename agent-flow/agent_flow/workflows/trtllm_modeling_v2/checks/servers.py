# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The servers of a deployment: started behind its launch prefix, owned, asked, and stopped.

A server is ``trtllm-serve <checkpoint> --config <options> --port 0 --report_addr <file>
--num_serve_frontends=1`` behind the deployment's ``launch`` prefix: the kernel picks the
port, the server writes ``<host>:<port>`` once it is bound (its host is rank 0's), and
the flow reads it back. A disaggregated deployment is a context server, a generation
server and ``trtllm-serve disaggregated`` with a proxy config written from the two
addresses. The flow knows no scheduler: more than one node is the prefix's business.

The flow owns what it starts. Each server runs in its own process group and under a name
(``MV2FLOW_SERVER``) the site's ``launch`` and ``stop`` strings may use for their step;
both are recorded in the workspace before the server is waited for. Stopping is a
SIGTERM to the group, a wait, the site's ``stop`` when it gives one, and then a look
through the same prefix at what still holds the devices: a holder that was not there
before the start means ranks were left behind, and the check has no verdict. A new start
first ends whatever an earlier process recorded and did not end. Nothing is ever killed
by a pattern.

One load at a time: :func:`engine_lock` is one lock per workspace, taken around every
check, a worker's own gate command included.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

from ..infra.files import write_text_atomic
from ..spec.deployments import ROLES, Deployment, Server, write_options

#: What this run started and has not ended, in the workspace.
OWNED_FILENAME = "servers.json"
#: The name a server runs under, exported to its launch and stop commands.
SERVER_NAME_ENV = "MV2FLOW_SERVER"
#: How long a server gets to exit after SIGTERM before its group is killed.
STOP_GRACE_S = 90
#: How long the devices get to empty after a stop.
RELEASE_TIMEOUT_S = 120
#: A deployment's start timeout when it states none: a load of a large checkpoint.
START_TIMEOUT_S = 1800
#: What a launch must not inherit from the orchestrator: the MPI and scheduler identity
#: of the process that happens to start it. The prefix brings its own.
SCRUBBED_PREFIXES: tuple[str, ...] = ("PMI_", "PMIX_", "OMPI_", "MPI_", "SLURM_", "MV2_")
#: The role names ``trtllm-serve --server_role`` takes for the two sides.
SERVER_ROLES: dict[str, str] = {"ctx": "context", "gen": "generation"}
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_HOLDERS = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"


class ServerError(RuntimeError):
    """A server that did not start, did not answer, or did not leave its devices."""


@dataclass
class Owned:
    """One server this run started: enough to end it from another process."""

    name: str
    pid: int
    pgid: int
    launch: str = ""
    stop: str = ""
    address_file: str = ""
    log: str = ""
    #: The device holders seen through the prefix before the start.
    before: list[str] = field(default_factory=list)


@dataclass
class Running:
    """A started deployment: the address to ask, and what to stop."""

    deployment: Deployment
    address: str
    owned: list[Owned]
    workspace: Path

    @property
    def url(self) -> str:
        return f"http://{self.address}"


# ------------------------------------------------------------------------------ the lock


@contextlib.contextmanager
def engine_lock(workspace: str | Path | None) -> Iterator[None]:
    """Hold the workspace's one engine lock: the run's checks and a worker's never overlap.

    The lock file is node-local (the temporary directory), named by the workspace: a
    shared filesystem does not always honour ``flock``. Without a workspace nothing is held.
    """
    if workspace is None:
        yield
        return
    digest = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"mv2flow-engine-{digest}.lock"
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# --------------------------------------------------------------------------- what is owned


def _owned_path(workspace: Path) -> Path:
    return Path(workspace) / OWNED_FILENAME


def load_owned(workspace: str | Path) -> list[Owned]:
    try:
        rows = json.loads(_owned_path(Path(workspace)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    known = set(Owned.__dataclass_fields__)
    return [Owned(**{k: v for k, v in row.items() if k in known}) for row in rows or ()]


def _save_owned(workspace: Path, owned: Sequence[Owned]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    write_text_atomic(_owned_path(workspace), json.dumps([asdict(o) for o in owned], indent=1))


def _alive(pgid: int) -> bool:
    """Whether any process of the group still runs; a leader that exited is reaped first."""
    with contextlib.suppress(ChildProcessError, OSError):
        os.waitpid(pgid, os.WNOHANG)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _end(owned: Owned, env: Mapping[str, str], grace_s: float) -> None:
    """End one server: its own group, then the site's stop. Its group only, never a pattern."""
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
        if not _alive(owned.pgid):
            break
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(owned.pgid, sig)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and _alive(owned.pgid):
            time.sleep(0.2)
    if owned.stop:
        subprocess.run(
            ["bash", "-c", owned.stop],
            env={**env, SERVER_NAME_ENV: owned.name},
            capture_output=True,
            timeout=600,
            check=False,
        )


def clean_owned(workspace: str | Path, env: Mapping[str, str] | None = None) -> list[str]:
    """End every server an earlier process of this run recorded; the names it ended."""
    workspace = Path(workspace)
    ended = []
    for owned in load_owned(workspace):
        _end(owned, launch_env(owned.launch, env or {}), STOP_GRACE_S)
        ended.append(owned.name)
    _save_owned(workspace, [])
    return ended


# ------------------------------------------------------------------------------ a launch


def launch_env(launch: str, declared: Mapping[str, str]) -> dict[str, str]:
    """The environment of a server's command: the process's, the spec's, and no borrowed identity.

    Behind a prefix nothing of the orchestrator's MPI or scheduler identity is passed on
    (:data:`SCRUBBED_PREFIXES`); a variable the spec declares is, whatever its name.
    """
    env = dict(os.environ)
    if launch.strip():
        env = {k: v for k, v in env.items() if not k.startswith(SCRUBBED_PREFIXES)}
    env.update({str(k): str(v) for k, v in declared.items()})
    return env


def serve_argv(
    model: str, options: str | Path, address_file: str | Path, *, role: str | None = None
) -> list[str]:
    """``trtllm-serve`` for one server: the kernel picks the port and the server reports it."""
    argv = ["trtllm-serve", str(model), "--config", str(options)]
    # Bound to every interface, so the address it reports names its host, not localhost.
    argv += ["--host", "0.0.0.0", "--port", "0", "--report_addr", str(address_file)]  # nosec B104
    argv.append("--num_serve_frontends=1")
    if role is not None:
        argv += ["--server_role", SERVER_ROLES[role]]
    return argv


def proxy_config(model: str, ctx_address: str, gen_address: str) -> dict[str, Any]:
    """The disaggregated server's config: one context and one generation server, by address."""
    return {
        "hostname": "0.0.0.0",  # nosec B104
        "port": 0,
        "model": str(model),
        "backend": "pytorch",
        "context_servers": {"num_instances": 1, "urls": [ctx_address]},
        "generation_servers": {"num_instances": 1, "urls": [gen_address]},
    }


def _holders(launch: str, env: Mapping[str, str]) -> list[str] | None:
    """The processes holding a device where ``launch`` lands; ``None`` when it cannot be asked."""
    line = f"{launch} {_HOLDERS}".strip()
    try:
        done = subprocess.run(
            ["bash", "-c", line], env=dict(env), capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode:
        return None
    return sorted({word for word in done.stdout.split() if word.isdigit()})


def _start_one(
    name: str,
    argv: Sequence[str],
    server: Server,
    env: Mapping[str, str],
    directory: Path,
    address_file: Path,
    workspace: Path,
    owned: list[Owned],
) -> Owned:
    """Start one command behind ``server``'s prefix in its own group, and record it."""
    address_file.unlink(missing_ok=True)
    log = directory / f"{name}.log"
    line = f"{server.launch} {' '.join(shlex.quote(a) for a in argv)}".strip()
    process_env = {**env, SERVER_NAME_ENV: name}
    before = _holders(server.launch, process_env)
    with log.open("wb") as sink:
        process = subprocess.Popen(
            ["bash", "-c", line],
            stdout=sink,
            stderr=subprocess.STDOUT,
            env=process_env,
            start_new_session=True,
        )
    record = Owned(
        name=name,
        pid=process.pid,
        pgid=process.pid,
        launch=server.launch,
        stop=server.stop,
        address_file=str(address_file),
        log=str(log),
        before=before or [],
    )
    owned.append(record)
    _save_owned(workspace, [*load_owned(workspace), record])
    return record


def _tail(path: str, lines: int = 20) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _await_address(record: Owned, deadline: float) -> str:
    """The address a server reported, once it is bound; a ``ServerError`` if it ends first."""
    path = Path(record.address_file)
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text().strip():
            return path.read_text().strip()
        if not _alive(record.pgid):
            raise ServerError(f"{record.name} exited before it was bound:\n{_tail(record.log)}")
        time.sleep(0.2)
    raise ServerError(f"{record.name} reported no address in time:\n{_tail(record.log)}")


def _await_health(record: Owned, address: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            with _DIRECT.open(f"http://{address}/health", timeout=10) as reply:
                if reply.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        if not _alive(record.pgid):
            raise ServerError(f"{record.name} exited before it was healthy:\n{_tail(record.log)}")
        time.sleep(0.5)
    raise ServerError(f"{record.name} was not healthy in time:\n{_tail(record.log)}")


def start(
    deployment: Deployment,
    *,
    model: str,
    workspace: str | Path,
    env: Mapping[str, str] | None = None,
    timeout: int | None = None,
) -> Running:
    """Start ``deployment``'s servers and wait until the one to ask is healthy.

    Ends first whatever an earlier process of the run left recorded. ``env`` is the
    spec's environment; the deployment's own is laid over it. On any failure everything
    started here is stopped before the error is raised.
    """
    workspace = Path(workspace)
    declared = {**(env or {}), **deployment.env}
    clean_owned(workspace, declared)
    directory = workspace / "deployments" / deployment.name
    directory.mkdir(parents=True, exist_ok=True)
    options = write_options(deployment, directory)
    deadline = time.monotonic() + (timeout or deployment.timeout or START_TIMEOUT_S)
    owned: list[Owned] = []
    running = Running(deployment, "", owned, workspace)
    try:
        if deployment.server is not None:
            server = deployment.server
            record = _start_one(
                deployment.name,
                serve_argv(model, options["server"], directory / "address"),
                server,
                launch_env(server.launch, declared),
                directory,
                directory / "address",
                workspace,
                owned,
            )
            running.address = _await_address(record, deadline)
            _await_health(record, running.address, deadline)
            return running
        addresses = {}
        for role in ROLES:
            server = getattr(deployment, role)
            record = _start_one(
                f"{deployment.name}.{role}",
                serve_argv(model, options[role], directory / f"address.{role}", role=role),
                server,
                launch_env(server.launch, declared),
                directory,
                directory / f"address.{role}",
                workspace,
                owned,
            )
            addresses[role] = record
        for role, record in addresses.items():
            addresses[role] = _await_address(record, deadline)
        config = directory / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(proxy_config(model, addresses["ctx"], addresses["gen"])), "utf-8"
        )
        remaining = max(60, int(deadline - time.monotonic()))
        proxy = _start_one(
            f"{deployment.name}.proxy",
            ["trtllm-serve", "disaggregated", "-c", str(config)]
            + ["--server_start_timeout", str(remaining), "-r", "360000"]
            + ["--report_addr", str(directory / "address.proxy")],
            Server(options={}),
            launch_env("", declared),
            directory,
            directory / "address.proxy",
            workspace,
            owned,
        )
        running.address = _await_address(proxy, deadline)
        _await_health(proxy, running.address, deadline)
        return running
    except BaseException:
        with contextlib.suppress(ServerError):
            stop(running, env=env)
        raise


def stop(running: Running, *, env: Mapping[str, str] | None = None) -> None:
    """Stop every server of ``running``, last started first, and see the devices emptied.

    Raises :class:`ServerError` naming what still holds a device where a prefix lands
    and was not there before the start: ranks were left behind, so whatever was measured
    is not a verdict, and the next load would share its devices.
    """
    declared = {**(env or {}), **running.deployment.env}
    left: list[str] = []
    for owned in reversed(running.owned):
        process_env = launch_env(owned.launch, declared)
        _end(owned, process_env, STOP_GRACE_S)
        deadline = time.monotonic() + RELEASE_TIMEOUT_S
        while True:
            now = _holders(owned.launch, {**process_env, SERVER_NAME_ENV: owned.name})
            stray = sorted(set(now or ()) - set(owned.before))
            if not stray or time.monotonic() >= deadline:
                break
            time.sleep(1)
        if stray:
            left.append(f"{owned.name}: {', '.join(stray)}")
    names = {o.name for o in running.owned}
    _save_owned(
        running.workspace, [o for o in load_owned(running.workspace) if o.name not in names]
    )
    running.owned.clear()
    if left:
        raise ServerError(
            "ranks were left holding a device after the stop (" + "; ".join(left) + "): the "
            "check has no verdict, and the site's `stop` must end them before the next load"
        )


def ask(running: Running, prompt: str, *, max_tokens: int, timeout: float = 600) -> str:
    """One greedy completion from the deployment's endpoint; its text."""
    body = {
        "model": "deployment",
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"{running.url}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with _DIRECT.open(request, timeout=timeout) as reply:
            payload = json.loads(reply.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ServerError(f"{running.deployment.name} did not answer a completion: {exc}") from exc
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        raise ServerError(f"{running.deployment.name} answered without a completion: {payload}")
    return str(choices[0].get("text") or "")


__all__ = [
    "OWNED_FILENAME",
    "SERVER_NAME_ENV",
    "Owned",
    "Running",
    "ServerError",
    "ask",
    "clean_owned",
    "engine_lock",
    "launch_env",
    "load_owned",
    "proxy_config",
    "serve_argv",
    "start",
    "stop",
]
