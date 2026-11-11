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

from ..infra.compute.environment import LLMAPI_LAUNCH, MPI_SETTINGS_KEPT, SINGLE_PROCESS_WORKER_ENV
from ..infra.compute.server_runtime import rank_launch
from ..infra.files import write_text_atomic
from ..spec.deployments import ROLES, Deployment, Server, write_options
from .check_runtime import MODE_REQUIRE, engine_env

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

    def log(self, role: str) -> str:
        """What the server of ``role`` printed so far: where its ranks say what they run."""
        name = self.deployment.name + ("" if role == "server" else f".{role}")
        path = next((record.log for record in self.owned if record.name == name), "")
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace") if path else ""
        except OSError:
            return ""


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


def local_launch(
    server: Server, devices: Sequence[int], env: Mapping[str, str], declared: Sequence[str]
) -> tuple[list[str], dict[str, str]]:
    """How a server with no launch prefix starts on this host: ``(launcher argv, its environment)``.

    Behind a prefix the site places the ranks and the devices. Here the server gets
    ``devices`` for itself; one rank runs its worker in its own process, with the MPI
    bootstrap variables this process may have inherited from an allocation taken out
    (all but those the spec ``declared``); more ranks start under ``mpirun`` and the LLM
    API's launcher, as every in-process check starts them.
    """
    if server.launch.strip():
        return [], dict(env)
    env = dict(env)
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
    if server.ranks > 1:
        return rank_launch(server.ranks, LLMAPI_LAUNCH), env
    kept = {"SLURM_PROCID", "SLURM_LOCALID", *MPI_SETTINGS_KEPT, *declared}
    bootstrap = ("PMI", "MPI", "OMPI", "SLURM")
    env = {k: v for k, v in env.items() if k in kept or not k.startswith(bootstrap)}
    env.setdefault(SINGLE_PROCESS_WORKER_ENV, "1")
    return [], env


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
    env: Mapping[str, str],
    server: Server,
    directory: Path,
    address_file: Path,
    workspace: Path,
    owned: list[Owned],
    cwd: str | None = None,
) -> Owned:
    """Start one command behind ``server``'s prefix in its own group, and record it."""
    address_file.unlink(missing_ok=True)
    log = directory / f"{name}.log"
    line = f"{server.launch} {' '.join(shlex.quote(a) for a in argv)}".strip()
    # The line as it is started, beside the log, for whoever reads a start that failed.
    (directory / f"{name}.launch").write_text(line + "\n", encoding="utf-8")
    process_env = {**env, SERVER_NAME_ENV: name}
    before = _holders(server.launch, process_env)
    with log.open("wb") as sink:
        process = subprocess.Popen(
            ["bash", "-c", line],
            stdout=sink,
            stderr=subprocess.STDOUT,
            env=process_env,
            cwd=cwd,
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
    tree: str | Path | None = None,
    python: str | None = None,
    devices: Sequence[int] = (),
) -> Running:
    """Start ``deployment``'s servers and wait until the one to ask is healthy.

    Ends first whatever an earlier process of the run left recorded. ``env`` is the
    spec's environment; the deployment's own is laid over it. Every server runs from
    ``tree`` with the Modeling V2 switch on and the rank hook loaded
    (``check_runtime.engine_env``), so its ranks say what they run. A server with no
    launch prefix takes its ranks' share of ``devices``, in the order the servers start
    (:func:`local_launch`). On any failure everything started here is stopped before the
    error is raised.
    """
    workspace = Path(workspace)
    declared = {**(env or {}), **deployment.env}
    clean_owned(workspace, declared)
    declared = engine_env(declared, tree=tree, python=python)
    directory = workspace / "deployments" / deployment.name
    directory.mkdir(parents=True, exist_ok=True)
    options = write_options(deployment, directory, {"modeling_v2": MODE_REQUIRE})
    cwd = str(tree) if tree is not None else None
    left = list(devices)

    def launched(server: Server, argv: list[str]) -> tuple[list[str], dict[str, str]]:
        """``argv`` and the environment ``server`` starts with, here or behind its prefix."""
        local = not server.launch.strip()
        own = [left.pop(0) for _ in range(min(server.ranks, len(left)) if local else 0)]
        launcher, env = local_launch(
            server, own, launch_env(server.launch, declared), list(declared)
        )
        return [*launcher, *argv], env

    deadline = time.monotonic() + (timeout or deployment.timeout or START_TIMEOUT_S)
    owned: list[Owned] = []
    running = Running(deployment, "", owned, workspace)
    try:
        if deployment.server is not None:
            server = deployment.server
            record = _start_one(
                deployment.name,
                *launched(server, serve_argv(model, options["server"], directory / "address")),
                server,
                directory,
                directory / "address",
                workspace,
                owned,
                cwd,
            )
            running.address = _await_address(record, deadline)
            _await_health(record, running.address, deadline)
            return running
        addresses = {}
        for role in ROLES:
            server = getattr(deployment, role)
            argv = serve_argv(model, options[role], directory / f"address.{role}", role=role)
            record = _start_one(
                f"{deployment.name}.{role}",
                *launched(server, argv),
                server,
                directory,
                directory / f"address.{role}",
                workspace,
                owned,
                cwd,
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
            launch_env("", declared),
            Server(options={}),
            directory,
            directory / "address.proxy",
            workspace,
            owned,
            cwd,
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


def complete(
    running: Running, prompt: str, *, max_tokens: int, timeout: float = 600
) -> Mapping[str, Any]:
    """One greedy completion from the deployment's endpoint: the response, as the server sent it."""
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
    if not isinstance(payload, dict) or not payload.get("choices"):
        raise ServerError(f"{running.deployment.name} answered without a completion: {payload}")
    return payload


def ask(running: Running, prompt: str, *, max_tokens: int, timeout: float = 600) -> str:
    """One greedy completion from the deployment's endpoint; its text."""
    payload = complete(running, prompt, max_tokens=max_tokens, timeout=timeout)
    return str(payload["choices"][0].get("text") or "")


def cached_tokens(payload: Mapping[str, Any]) -> int | None:
    """The prompt tokens a completion resumed from the cache, where the server reports them."""
    details = (payload.get("usage") or {}).get("prompt_tokens_details") or {}
    count = details.get("cached_tokens")
    return count if isinstance(count, int) and not isinstance(count, bool) else None


__all__ = [
    "OWNED_FILENAME",
    "SERVER_NAME_ENV",
    "Owned",
    "Running",
    "ServerError",
    "ask",
    "cached_tokens",
    "clean_owned",
    "complete",
    "engine_lock",
    "launch_env",
    "local_launch",
    "load_owned",
    "proxy_config",
    "serve_argv",
    "start",
    "stop",
]
