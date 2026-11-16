#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One session of one deployment: its servers started once, every check a client of them.

Usage (one process inside the job that holds the deployment's devices)::

    <tree's python> session_driver.py --spec session.json

The spec is a JSON file ``checks/session.py`` writes:

* ``deployment``, ``model`` (the checkpoint), ``directory`` (logs, addresses, the proxy
  config; on the shared filesystem), ``timeout`` (seconds for the whole session);
* ``env``: the deployment's own environment, laid over this process's for its servers;
* ``instances``: one per server, in the order they start: ``name`` (``server``, ``ctx0``,
  ``gen1``), ``role``, ``index``, ``ranks`` and ``options`` (the file ``--config`` takes);
* ``steps``: the checks, in the order they run. Each has an ``id`` and a ``kind``:
  ``requests`` (``requests``: ``{"id", "endpoint", "body"}`` each, and optionally a
  ``dataset`` to draw more prompts from, each sent as ``body`` is),
  ``stream`` (one streamed request), ``hop`` (``requests`` sent as the measured tree
  defines a request's metrics are asked for) and ``command`` (``argv`` with ``{python}``,
  ``{url}`` and ``{model}``, ``log``, ``env``). A step may state ``needs``: a command of
  its own that says, with no server, whether the step has what it needs on this node (exit
  0, or its last line says what is missing). A step of kind ``needs`` is only that.

What it does, in this order, each said on stdout as one JSON line behind a marker:

1. **Devices, before any load.** The job's devices are split among the instances, each
   instance its own, disjoint. ``MV2FLOW_DEVICES``: the plan, this host, and who held a
   device already. Too few devices, or a set that overlaps another, ends the session here.
   Then every step's ``needs`` is asked: a step that lacks something is not run, and its
   ``MV2FLOW_STEP`` says ``blocked`` with why. The other steps go on.
2. **Start.** One ``trtllm-serve`` per instance on its own devices (``mpirun`` under
   ``trtllm-llmapi-launch`` above one rank), each reporting the address it bound; with
   ``ctx`` and ``gen`` roles, ``trtllm-serve disaggregated`` over them. ``MV2FLOW_SERVER``
   per server once it is healthy, or with why it is not.
3. **Ranks.** Every ``MV2FLOW_RANK_ENGINE`` record the servers' ranks printed, passed on
   with the instance it came from.
4. **Steps**, one after the other on the one endpoint (the proxy's, for ``ctx`` and
   ``gen``): ``MV2FLOW_RECORD`` per request, ``MV2FLOW_STEP`` when the step ends.
5. **Stop.** Every server's group, then whatever of theirs still holds a device.
   ``MV2FLOW_SESSION`` last: ``loads`` (how many engines were started: one per instance),
   what was reaped and what was left.

Every wait ends at the session's ``timeout``. The servers and their ranks are stopped
whatever happened, a SIGTERM to this process included. Exits 0 when it ran to the end.

Runs by path under the measured tree's interpreter: standard library plus PyYAML, and no
``agent_flow``. The launch and reap helpers are ``infra/compute/server_runtime.py``'s.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import importlib.util
import json
import mimetypes
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent

DEVICES_MARKER = "MV2FLOW_DEVICES"
SERVER_MARKER = "MV2FLOW_SERVER"
STEP_MARKER = "MV2FLOW_STEP"
SESSION_MARKER = "MV2FLOW_SESSION"
#: How long a step's ``needs`` command may take to answer, before any server is started.
NEEDS_TIMEOUT_S = 300.0
#: Where a job of several steps keeps what its steps lack (asked by ``--placed``) for ``--attach``.
LACKING_FILE = "lacking.json"
#: Seconds a session waits, once its servers answer, for a line from every rank.
RANK_RECORDS_WAIT_S = 5.0
#: What each instance's ranks read to say which server they are.
#: The role instance a process belongs to (`ctx0`, `gen1`, `server`): the name the
#: pool's origin judge counts a job's stated ranks by.
ROLE_ENV = "MV2FLOW_ROLE"
#: Its role alone (`ctx`, `gen`, `server`), for the rank's own record.
SERVER_ROLE_ENV = "MV2FLOW_SERVER_ROLE"
INSTANCE_ENV = "MV2FLOW_INSTANCE"
#: The role names ``trtllm-serve --server_role`` takes.
SERVER_ROLES = {"ctx": "context", "gen": "generation"}
#: The engine's own variable of a server's role, whose values are those names. The engine's
#: serve command publishes it from ``--server_role`` in its own process, which is rank 0
#: alone: the other ranks of the server were started before it ran. Its models read it in
#: every rank, so every rank is given it here (the engine: ``DisaggLauncherEnvs`` and
#: ``_DISAGG_SERVER_ROLES`` of its serve command).
ENGINE_ROLE_ENV = "TRTLLM_DISAGG_ROLE"
DEFAULT_TIMEOUT_S = 1800
STOP_GRACE_S = 90
#: The finish reasons of an answer that completed. The executor also ends a request with
#: ``timeout`` or ``cancelled``: text may come with those, and it is no answer.
COMPLETED = ("stop", "length")
#: Variables of an enclosing MPI world or scheduler step that a server here must not read.
_BORROWED = ("PMI", "PMIX", "OMPI", "MPI", "SLURM")
_KEPT = ("OMPI_ALLOW_RUN_AS_ROOT", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "OMPI_MCA_", "PMIX_MCA_")

#: Requests to servers on this host, never through a proxy the environment names.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _load(name: str, path: Path) -> Any:
    """Import the file at ``path`` as module ``name``, without putting its directory on the path."""
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


runtime = _load(
    "mv2flow_server_runtime", _HERE.parent.parent / "infra" / "compute" / "server_runtime.py"
)
wio = _load("mv2flow_workload_io", _HERE / "workload_io.py")


def entry_point(name: str) -> str:
    """The console script ``name`` installed beside this interpreter, else the one on ``PATH``."""
    sibling = Path(sys.executable).with_name(name)
    if sibling.is_file():
        return str(sibling)
    return shutil.which(name) or name


# ------------------------------------------------------------------------------ devices


def job_devices(environ: dict[str, str], count: int) -> list[str]:
    """The devices this job was given: ``CUDA_VISIBLE_DEVICES``, else the first ``count``."""
    visible = [d for d in (environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if d.strip()]
    return visible or [str(index) for index in range(count)]


def plan_devices(instances: list[dict[str, Any]], devices: list[str]) -> dict[str, list[str]]:
    """Each instance's own devices, in the order the instances start. ``ValueError`` when short.

    Every rank sees all devices of its node unless told otherwise, so the split is the
    flow's: instance by instance, ``ranks`` devices each, none given twice.
    """
    need = sum(int(instance["ranks"]) for instance in instances)
    if len(devices) < need:
        raise ValueError(
            f"the job holds {len(devices)} device(s) ({','.join(devices) or 'none'}) and its "
            f"{len(instances)} server(s) need {need}: one per rank of each"
        )
    plan, left = {}, list(devices)
    for instance in instances:
        plan[instance["name"]] = [left.pop(0) for _ in range(int(instance["ranks"]))]
    return plan


def overlap(plan: dict[str, list[str]]) -> str:
    """Which device two instances were both given; ``""`` when the sets are disjoint."""
    seen: dict[str, str] = {}
    for name, devices in plan.items():
        for device in devices:
            if device in seen:
                return f"device {device} is given to both {seen[device]} and {name}"
            seen[device] = name
    return ""


def holders(devices: list[str]) -> list[int] | None:
    """The pids holding memory on ``devices``; ``None`` when they cannot be read."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "-i", ",".join(devices), "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode:
        return None
    return sorted({int(line) for line in done.stdout.split() if line.isdigit()})


# ------------------------------------------------------------------------------ servers


def server_env(base: dict[str, str], instance: dict[str, Any], devices: list[str]) -> dict:
    """The environment of one instance's server: its own devices, its label, no borrowed world.

    The MPI and scheduler identity this process may carry is taken out (the MPI settings
    stay): several ranks get theirs from the ``mpirun`` started here, and one rank runs
    its worker in its own process.
    """
    env = {
        key: value
        for key, value in base.items()
        if key.startswith(_KEPT) or not key.startswith(_BORROWED)
    }
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env[ROLE_ENV] = str(instance["name"])
    env[SERVER_ROLE_ENV] = str(instance["role"])
    env[INSTANCE_ENV] = str(instance["index"])
    # One engine has no role, and borrows none from the job that started it.
    env.pop(ENGINE_ROLE_ENV, None)
    if instance["role"] in SERVER_ROLES:
        env[ENGINE_ROLE_ENV] = SERVER_ROLES[instance["role"]]
    if int(instance["ranks"]) <= 1:
        env.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    return env


def serve_argv(model: str, instance: dict[str, Any], address_file: Path) -> list[str]:
    """``trtllm-serve`` for one instance: the kernel picks the port and the server reports it."""
    argv = [
        *runtime.rank_launch(int(instance["ranks"]), entry_point("trtllm-llmapi-launch")),
        entry_point("trtllm-serve"),
        str(model),
        "--config",
        str(instance["options"]),
        # Bound to every interface, so the address it reports names its host.
        "--host",
        "0.0.0.0",  # nosec B104
        "--port",
        "0",
        "--report_addr",
        str(address_file),
        "--num_serve_frontends=1",
    ]
    if instance["role"] in SERVER_ROLES:
        argv += ["--server_role", SERVER_ROLES[instance["role"]]]
    return argv


def proxy_config(model: str, ctx: list[str], gen: list[str]) -> dict[str, Any]:
    """The disaggregated server's config: every context and generation server, by address."""
    return {
        "hostname": "0.0.0.0",  # nosec B104
        "port": 0,
        "model": str(model),
        "backend": "pytorch",
        "context_servers": {"num_instances": len(ctx), "urls": list(ctx)},
        "generation_servers": {"num_instances": len(gen), "urls": list(gen)},
        # What each side did for each request: the proof of the hop is read from it.
        "return_perf_metrics": True,
        "perf_metrics_max_requests": 256,
    }


def read_placed(directory: Path, name: str, rank: int) -> dict[str, Any] | None:
    """What rank ``rank`` of ``name`` wrote before any load: its host, its devices, who held them."""
    try:
        words = (directory / f"placed.{name}.{rank}").read_text(encoding="utf-8").split()
    except OSError:
        return None
    if len(words) < 3:
        return None
    return {"host": words[0], "rank": words[1], "devices": words[2].split(","), "held": words[3:]}


def first_host(directory: Path, name: str) -> str:
    """The host of ``name``'s first rank, where its server answers; ``""`` when it never said."""
    return str((read_placed(directory, name, 0) or {}).get("host") or "")


def routable(address: str, host: str) -> str:
    """``address`` as another node reaches it: a server bound to every interface is on ``host``."""
    bound, _, port = address.rpartition(":")
    return f"{host}:{port}" if host and bound in ("0.0.0.0", "", "[::]") else address  # nosec B104


def placed(directory: Path, instances: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """``(records, problem)`` of what every rank of every instance said of its place.

    One ``MV2FLOW_DEVICES`` record per instance and host: the devices its ranks there
    were given. The problem, when there is one: a rank that said nothing, more ranks on
    a host than devices, or a device of one host given to two instances.
    """
    records: list[dict[str, Any]] = []
    taken: dict[tuple[str, str], str] = {}
    problem = ""
    for instance in instances:
        name = instance["name"]
        by_host: dict[str, dict[str, Any]] = {}
        for rank in range(int(instance["ranks"])):
            said = read_placed(directory, name, rank)
            if said is None:
                problem = problem or f"{name} rank {rank} said nothing of its place before the load"
                continue
            entry = by_host.setdefault(
                said["host"], {"devices": said["devices"], "ranks": 0, "held": []}
            )
            entry["ranks"] += 1
            entry["held"] += [pid for pid in said["held"] if pid not in entry["held"]]
        for host, entry in by_host.items():
            if entry["ranks"] > len(entry["devices"]):
                problem = problem or (
                    f"{name} has {entry['ranks']} rank(s) on {host} and "
                    f"{len(entry['devices'])} device(s) there"
                )
            for device in entry["devices"]:
                other = taken.setdefault((host, device), name)
                if other != name:
                    problem = problem or (
                        f"{name} and {other} were both given device {device} of {host}"
                    )
            records.append(
                {
                    "instance": name,
                    "role": instance["role"],
                    "index": instance["index"],
                    "host": host,
                    "devices": entry["devices"],
                    "ranks": entry["ranks"],
                    "held_before": [int(p) for p in entry["held"] if p.isdigit()] or None,
                }
            )
    return records, problem


def last_error(log_tail: str) -> str:
    """The last line of a server log that names an error, else its last line."""
    lines = [line.strip() for line in log_tail.splitlines() if line.strip()]
    for line in reversed(lines):
        if "Error" in line or "Exception" in line or "Invalid" in line:
            return line[:300]
    return lines[-1][:300] if lines else ""


class Session:
    """The servers of one deployment, from their start to their stop."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.directory = Path(spec["directory"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + float(spec.get("timeout") or DEFAULT_TIMEOUT_S)
        self.instances: list[dict[str, Any]] = list(spec["instances"])
        #: ``(name, process, log, its devices, who held them before its start)`` of every
        #: server started.
        self.started: list[tuple[str, subprocess.Popen, Path, list[str], set[int]]] = []
        self.addresses: dict[str, str] = {}
        self.url = ""
        self.loads = 0
        #: The servers were started by the session's own job, as steps on their hosts
        #: (a session wider than one node): this process starts the proxy and asks.
        self.attached = bool(spec.get("attach"))

    def left(self) -> float:
        """What remains of the session's time; a second at least, so a late wait fails as one."""
        return max(1.0, self.deadline - time.monotonic())

    def _await(self, name: str, address_file: Path | None, address: str = "") -> str:
        """Wait for ``name``'s address (or, given one, its health); ``SystemExit`` saying why not."""
        what = "healthy" if address else "bound to an address"
        while time.monotonic() < self.deadline:
            for other, process, log, _, _ in self.started:
                if process.poll() is not None:
                    raise SystemExit(
                        f"{other} exited with {process.returncode} before {name} was {what}; "
                        f"its log ({log}) says: {last_error(runtime.tail(log, 200))}"
                    )
            for instance in self.instances if self.attached else ():
                # A step that ended says so in a file: its launcher is not this process's.
                ended = self.directory / f"exit.{instance['name']}"
                if ended.is_file():
                    log = self.directory / f"{instance['name']}.log"
                    raise SystemExit(
                        f"{instance['name']}'s step ended with "
                        f"{ended.read_text(encoding='utf-8').strip() or '?'} before {name} was "
                        f"{what}; its log ({log}) says: {last_error(runtime.tail(log, 200))}"
                    )
            if not address and address_file is not None and address_file.is_file():
                text = address_file.read_text(encoding="utf-8").strip()
                if text:
                    return text
            if address:
                try:
                    with _DIRECT.open(f"http://{address}/health", timeout=5) as reply:
                        if reply.status == 200:
                            return address
                except (urllib.error.URLError, OSError):
                    pass
            time.sleep(0.2)
        raise SystemExit(f"{name} was not {what} within the session's timeout")

    def _begin(self, name: str, argv: list[str], env: dict[str, str]) -> Path:
        """Start ``argv`` in its own process group, its log kept and its launch line beside it."""
        log = self.directory / f"{name}.log"
        (self.directory / f"{name}.launch").write_text(" ".join(argv) + "\n", encoding="utf-8")
        devices = [d for d in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
        before = set(holders(devices) or ()) if devices else set()
        process = runtime.start_server(argv, log, env)
        self.started.append((name, process, log, devices, before))
        return log

    def start(self, plan: dict[str, list[str]]) -> None:
        """Start every instance on its devices, then the proxy over them; wait for health."""
        base = {**os.environ, **{str(k): str(v) for k, v in (self.spec.get("env") or {}).items()}}
        files: dict[str, Path] = {}
        for instance in self.instances:
            name = instance["name"]
            files[name] = self.directory / f"address.{name}"
            if self.attached:
                continue  # its step is already running, on the hosts the job gave it
            files[name].unlink(missing_ok=True)
            for stale in self.directory.glob(f"rank-engine.{name}.*.jsonl"):
                stale.unlink(missing_ok=True)
            argv = serve_argv(self.spec["model"], instance, files[name])
            env = server_env(base, instance, plan[name])
            # Each rank keeps its record in a file of its own here: see `rank_records`.
            env[wio.RANK_RECORDS_ENV] = str(self.directory)
            self._begin(name, argv, env)
            self.loads += 1
        for instance in self.instances:
            name = instance["name"]
            began = time.monotonic()
            address = self._await(name, files[name])
            if self.attached:
                address = routable(address, first_host(self.directory, name))
            self.addresses[name] = self._await(name, None, address)
            wio.emit(
                SERVER_MARKER,
                {
                    "name": name,
                    "role": instance["role"],
                    "index": instance["index"],
                    "healthy": True,
                    "address": self.addresses[name],
                    "waited_s": round(time.monotonic() - began, 1),
                    "log": str(self.directory / f"{name}.log"),
                },
            )
        roles = {instance["role"] for instance in self.instances}
        if not roles & set(SERVER_ROLES):
            self.url = f"http://{self.addresses[self.instances[0]['name']]}"
            return
        by_role = {
            role: [self.addresses[i["name"]] for i in self.instances if i["role"] == role]
            for role in SERVER_ROLES
        }
        config = self.directory / "proxy.yaml"
        import yaml

        config.write_text(
            yaml.safe_dump(proxy_config(self.spec["model"], by_role["ctx"], by_role["gen"])),
            encoding="utf-8",
        )
        address_file = self.directory / "address.proxy"
        address_file.unlink(missing_ok=True)
        argv = [entry_point("trtllm-serve"), "disaggregated", "-c", str(config)]
        argv += ["--server_start_timeout", str(int(self.left())), "-r", "360000"]
        argv += ["--report_addr", str(address_file)]
        proxy_env = {k: v for k, v in base.items() if not k.startswith(_BORROWED)}
        proxy_env["CUDA_VISIBLE_DEVICES"] = ""
        self._begin("proxy", argv, proxy_env)
        address = self._await("proxy", None, self._await("proxy", address_file))
        self.addresses["proxy"] = address
        self.url = f"http://{address}"
        wio.emit(
            SERVER_MARKER,
            {"name": "proxy", "role": "proxy", "healthy": True, "address": address},
        )

    def rank_records(self, instance: dict[str, Any]) -> list[dict[str, Any]]:
        """What the ranks of ``instance`` said of their engines.

        Each rank's own file first: it has one writer, so its record is whole. The
        server's log, which every rank of the world prints to, is read only for a rank
        that kept no file: two records printed there at once can be one broken line.
        """
        name = instance["name"]
        records = wio.kept_records(self.directory, name)
        kept = {str(record.get("rank")) for record in records}
        try:
            text = (self.directory / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for record in wio.parse_marked(text, wio.RANK_ENGINE_MARKER):
            if str(record.get("rank")) not in kept:
                records.append(record)
        return records

    def ranks(self, wait_s: float = 0.0) -> None:
        """Pass on what every server's ranks printed, each with the instance it came from.

        A server that answers has every rank of its world up, but a rank's line may still
        be on its way to the server's log (another process, another node): ``wait_s`` is
        how long to wait for one line from each rank before passing on what there is.
        """
        until = min(self.deadline, time.monotonic() + wait_s)
        while True:
            found = {i["name"]: self.rank_records(i) for i in self.instances}
            heard = all(
                len({record.get("rank") for record in found[i["name"]]}) >= int(i["ranks"])
                for i in self.instances
            )
            if heard or time.monotonic() >= until:
                break
            time.sleep(0.2)
        for instance in self.instances:
            for record in found[instance["name"]]:
                record.update(instance=instance["name"], role=instance["role"])
                wio.emit(wio.RANK_ENGINE_MARKER, record)

    def stop(self) -> dict[str, Any]:
        """Stop every server, last started first; what was reaped and what still holds a device."""
        reaped: list[int] = []
        left: list[int] = []
        codes: dict[str, Any] = {}
        for name, process, _, devices, before in reversed(self.started):
            # Only this server's own devices are looked at: another instance, still up on
            # its own, is not a rank left behind.
            def own(_count: int, devices: list[str] = devices) -> set[int]:
                return set(holders(devices) or ()) if devices else set()

            try:
                ended = runtime.stop_server(
                    process, len(devices) or 1, before, grace_s=STOP_GRACE_S, holders=own
                )
            except OSError as exc:
                ended = {"returncode": None, "reaped": [], "left": [], "error": str(exc)}
            codes[name] = ended.get("returncode")
            reaped += ended.get("reaped") or []
            left += ended.get("left") or []
        self.started.clear()
        return {"returncodes": codes, "reaped": sorted(set(reaped)), "left": sorted(set(left))}


# ------------------------------------------------------------------------------- clients


def lfs_object(pointer: Path) -> Path | None:
    """The content a git-lfs pointer names, where the LFS store of its repository holds it.

    A tree holds LFS files as pointers and its making fetches nothing. A checkout made with
    git-lfs keeps their content in its own git directory, which a tree's ``commondir`` names.
    """
    named = re.search(r"^oid sha256:([0-9a-f]{64})\s*$", pointer.read_text(errors="replace"), re.M)
    for above in pointer.resolve().parents if named else ():
        git = above / ".git"
        if git.is_file():  # a worktree: `gitdir: <its own directory>`
            git = above / git.read_text(encoding="utf-8").partition(":")[2].strip()
        if git.is_dir():
            common = git / "commondir"
            store = git / common.read_text(encoding="utf-8").strip() if common.is_file() else git
            oid = named.group(1)
            held = store / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
            return held if held.is_file() else None
    return None


def data_url(location: str) -> str:
    """The image file at ``location`` (a path or ``file://`` URI) as a base64 ``data:`` URL."""
    source = Path(location.removeprefix("file://"))
    if not source.is_file():
        raise FileNotFoundError(f"image not found: {source}")
    held = source
    if wio.is_lfs_pointer(source):
        held = lfs_object(source)
        if held is None:
            raise ValueError(
                f"{source} is a git-lfs pointer, and the image it names is not in the LFS "
                f"store of the checkout (a tree holds such files as pointers and never "
                f"fetches): run `git lfs pull --include {source.name}` in the checkout "
                f"itself, or state an image by its path (`images` of a prompt in "
                f"`accuracy.boot.prompts`)"
            )
    mime = mimetypes.guess_type(source.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(held.read_bytes()).decode('ascii')}"


def inline_images(body: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``body`` with every local image its messages name sent inline."""
    body = copy.deepcopy(body)
    for message in body.get("messages") or ():
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = str((part.get("image_url") or {}).get("url", ""))
            if not url.startswith(("http://", "https://", "data:")):
                part["image_url"] = {**part["image_url"], "url": data_url(url)}
    return body


def answer_of(payload: Any) -> dict[str, Any]:
    """What a completion or chat response says: its text, how it ended, what it counted."""
    if not isinstance(payload, dict):
        return {}
    choices = payload.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    text = choice.get("text")
    if text is None and isinstance(message, dict):
        text = message.get("content")
    if text is not None and not isinstance(text, str):
        text = json.dumps(text)
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    found: dict[str, Any] = {
        "text": text or "",
        "finish_reason": choice.get("finish_reason"),
        "tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens") if isinstance(details, dict) else None,
    }
    drafted = choice.get("speculative_decoding") or payload.get("speculative_decoding")
    if isinstance(drafted, dict):
        found["speculative_decoding"] = {
            key: drafted.get(key)
            for key in ("total_accepted_draft_tokens", "total_draft_tokens", "num_spec_steps")
        }
    return found


#: The engine's name for the fields that bracket a state transfer: what the file that
#: defines how a request's own metrics are asked for and returned is found by.
TRANSFER_FIELD = "kv_cache_transfer"


def hop_means(tree: str | Path | None) -> dict[str, Any]:
    """How this engine lets a client observe one request's state transfer, by its own definitions.

    Found in the source of the measured tree, never assumed: an engine moves its files
    and renames its headers. The file is the one of the tree's packages that names a
    transfer's fields (:data:`TRANSFER_FIELD`) and defines a request header that asks for
    a request's metrics, a response header of timestamps, and the names under which that
    header carries the start and the end of a transfer. ``means`` is ``headers`` with
    those, or ``none`` with ``why``.
    """
    if not tree or not Path(tree).is_dir():
        return {
            "means": "none",
            "why": "the session measures no tree whose definitions it can read",
        }
    root = Path(tree)
    naming = 0
    packages = sorted(p for p in root.iterdir() if (p / "__init__.py").is_file())
    for source in (path for package in packages for path in sorted(package.rglob("*.py"))):
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
            module = ast.parse(text) if TRANSFER_FIELD in text else None
        except (OSError, SyntaxError):
            module = None
        if module is None:
            continue
        naming += 1
        constants: dict[str, str] = {}
        names: dict[str, str] = {}
        for node in ast.walk(module):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                constants[node.targets[0].id] = node.value.value
            elif (
                isinstance(node, ast.Tuple)
                and len(node.elts) == 2
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts)
                and TRANSFER_FIELD in node.elts[1].value
                # A header's name for a field, not two fields listed side by side.
                and TRANSFER_FIELD not in node.elts[0].value
            ):
                names.setdefault(node.elts[1].value, node.elts[0].value)
        asks = [v for n, v in constants.items() if n.endswith("_HEADER") and "RETURN" in n]
        stamps = [v for n, v in constants.items() if n.endswith("_HEADER") and "START_END" in n]
        begin = next((item for field, item in names.items() if field.endswith("start")), None)
        end = next((item for field, item in names.items() if field.endswith("end")), None)
        if asks and stamps and begin and end:
            return {
                "means": "headers",
                "request_header": asks[0],
                "header": stamps[0],
                "begin": begin,
                "end": end,
                "source": source.relative_to(root).as_posix(),
            }
    return {
        "means": "none",
        "why": f"of the measured tree's packages, {naming} file(s) name a state transfer "
        f"(`{TRANSFER_FIELD}`) and none defines a request header for a request's metrics "
        f"with the names of a transfer's start and end in a response header",
    }


def ask(
    url: str,
    step: str,
    index: int,
    request: dict[str, Any],
    timeout: float,
    *,
    headers: dict[str, str] | None = None,
    keep: tuple[str, ...] = (),
) -> dict:
    """Send one request; its ``MV2FLOW_RECORD`` payload, whatever the server did.

    ``headers`` are sent with it; the response headers named in ``keep`` are recorded.
    """
    record: dict[str, Any] = {
        "step": step,
        "index": index,
        "id": str(request.get("id", index)),
        "endpoint": str(request["endpoint"]),
        "status": None,
        "text": "",
        "finish_reason": None,
        "error": "",
    }
    try:
        data = json.dumps(inline_images(dict(request["body"]))).encode("utf-8")
    except (OSError, ValueError) as exc:
        record["error"] = f"the request could not be built: {exc}"
        return record
    http = urllib.request.Request(
        f"{url}{request['endpoint']}",
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with _DIRECT.open(http, timeout=timeout) as response:
            record["status"] = response.status
            kept = {name: response.headers.get(name) for name in keep}
            if keep:
                record["headers"] = {name: value for name, value in kept.items() if value}
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        record["status"] = exc.code
        # The body of an error response is the server's own word on what is wrong.
        record["error"] = exc.read().decode("utf-8", errors="replace")[:2000]
        return record
    except (urllib.error.URLError, OSError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record.update(answer_of(payload))
    if record["finish_reason"] not in COMPLETED:
        record["error"] = (
            f"no completed answer (finish reason {record['finish_reason']!r}, not "
            f"{' or '.join(COMPLETED)})"
        )
    return record


def stream(url: str, step: str, request: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Send one streamed request; how many events came, their text, and how it ended."""
    record: dict[str, Any] = {
        "step": step,
        "index": 0,
        "id": str(request.get("id", "stream")),
        "endpoint": str(request["endpoint"]),
        "status": None,
        "text": "",
        "finish_reason": None,
        "events": 0,
        "done": False,
        "error": "",
    }
    http = urllib.request.Request(
        f"{url}{request['endpoint']}",
        data=json.dumps({**request["body"], "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _DIRECT.open(http, timeout=timeout) as response:
            record["status"] = response.status
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    record["done"] = True
                    break
                event = json.loads(data)
                record["events"] += 1
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                record["text"] += str(choice.get("text") or delta.get("content") or "")
                record["finish_reason"] = choice.get("finish_reason") or record["finish_reason"]
    except urllib.error.HTTPError as exc:
        record["status"] = exc.code
        record["error"] = exc.read().decode("utf-8", errors="replace")[:2000]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _words(argv: list[Any], url: str, model: str) -> list[str]:
    return [
        str(word)
        .replace("{python}", sys.executable)
        .replace("{url}", url)
        .replace("{model}", model)
        for word in argv
    ]


def lacking(spec: dict[str, Any]) -> dict[str, str]:
    """What each step that states ``needs`` lacks on this node, by step id; asked before any load.

    ``needs`` is the step's own command (an evaluator's loader of its dataset, say): its
    last line is why it cannot run. A command that gives no answer (it cannot be started, or
    its time is up) blocks nothing: the step then says for itself what it finds.
    """
    found: dict[str, str] = {}
    for step in spec.get("steps") or ():
        if not step.get("needs"):
            continue
        env = {**os.environ, **{str(k): str(v) for k, v in (step.get("env") or {}).items()}}
        env["CUDA_VISIBLE_DEVICES"] = ""
        try:
            done = subprocess.run(
                _words(step["needs"], "", str(spec["model"])),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=NEEDS_TIMEOUT_S,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if done.returncode:
            said = [line for line in done.stdout.splitlines() if line.strip()]
            found[str(step["id"])] = (
                said[-1] if said else f"its `needs` command exited {done.returncode}"
            )
    return found


def run_step(session: Session, step: dict[str, Any]) -> dict[str, Any]:
    """Run one step against the session's endpoint; its ``MV2FLOW_STEP`` payload."""
    began = time.monotonic()
    ended: dict[str, Any] = {"step": step["id"], "kind": step["kind"], "error": ""}
    kind = step["kind"]
    if time.monotonic() >= session.deadline:
        ended["error"] = "the session's timeout was up before this step began"
    elif kind == "requests":
        requests = list(step.get("requests") or ())
        if step.get("dataset"):
            # Prompts drawn from a dataset on this host, offline, each sent as `body` is.
            try:
                drawn = wio.normalize_requests(
                    {"dataset": step["dataset"], "max_tokens": step["body"].get("max_tokens", 64)}
                )
            except wio.DatasetError as exc:
                # This step has nothing to send; the steps after it are not its to end.
                drawn = []
                ended["error"] = f"the dataset of this step could not be read: {exc}"
            requests += [
                {
                    "id": row["id"],
                    "endpoint": "/v1/completions",
                    "body": {
                        **step["body"],
                        "prompt": row["prompt"],
                        "max_tokens": row["max_tokens"],
                    },
                }
                for row in drawn
            ]
        for index, request in enumerate(requests):
            wio.emit(
                wio.RECORD_MARKER, ask(session.url, step["id"], index, request, session.left())
            )
        ended["requests"] = len(requests)
    elif kind == "stream":
        wio.emit(
            wio.RECORD_MARKER, stream(session.url, step["id"], step["request"], session.left())
        )
    elif kind == "hop":
        # How the transfer can be observed is what the measured tree defines, found now.
        # The tree this process runs in: on a node the job's pin put it (`MV2FLOW_TREE`), else
        # the one the session's spec names.
        means = hop_means(os.environ.get("MV2FLOW_TREE") or session.spec.get("tree"))
        ended["observe"] = means
        by_headers = means["means"] == "headers"
        requests = list(step.get("requests") or ())
        for index, request in enumerate(requests):
            record = ask(
                session.url,
                step["id"],
                index,
                request,
                session.left(),
                headers={means["request_header"]: "1"} if by_headers else None,
                keep=(means["header"],) if by_headers else (),
            )
            wio.emit(wio.RECORD_MARKER, record)
        ended["requests"] = len(requests)
    elif kind == "command":
        argv = _words(step["argv"], session.url, str(session.spec["model"]))
        log = Path(step.get("log") or session.directory / f"{step['id']}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **{str(k): str(v) for k, v in (step.get("env") or {}).items()}}
        # A client holds no device: the servers do.
        env["CUDA_VISIBLE_DEVICES"] = ""
        try:
            with log.open("wb") as sink:
                done = subprocess.run(
                    argv, stdout=sink, stderr=subprocess.STDOUT, env=env, timeout=session.left()
                )
            ended["returncode"] = done.returncode
        except subprocess.TimeoutExpired:
            ended["error"] = "the session's timeout was up before the command ended"
        except OSError as exc:
            ended["error"] = f"could not start {argv[0]}: {exc}"
        ended["command"] = " ".join(argv)
        ended["log"] = str(log)
    else:
        ended["error"] = f"unknown step kind {kind!r}"
    ended["seconds"] = round(time.monotonic() - began, 2)
    return ended


def run(spec: dict[str, Any]) -> dict[str, Any]:
    """The whole session; returns its ``MV2FLOW_SESSION`` payload."""
    session = Session(spec)
    report: dict[str, Any] = {
        "deployment": spec.get("deployment"),
        "host": socket.gethostname(),
        "instances": len(session.instances),
        "loads": 0,
        "started": False,
        "error": "",
    }
    needed = sum(int(instance["ranks"]) for instance in session.instances)
    if session.attached:
        # The job placed every rank and said so before any load (`--placed`).
        devices, plan, clash = [], {}, ""
    else:
        devices = job_devices(dict(os.environ), needed)
        try:
            plan = plan_devices(session.instances, devices)
            clash = overlap(plan)
        except ValueError as exc:
            plan, clash = {}, str(exc)
    for instance in () if session.attached else session.instances:
        own = plan.get(instance["name"], [])
        wio.emit(
            DEVICES_MARKER,
            {
                "instance": instance["name"],
                "role": instance["role"],
                "index": instance["index"],
                "host": report["host"],
                "devices": own,
                "ranks": int(instance["ranks"]),
                "held_before": holders(own) if own else None,
            },
        )
    if clash:
        report["error"] = f"nothing was loaded: {clash}"
        wio.emit(SESSION_MARKER, report)
        return report
    if session.attached:
        # Asked by the job before it started its servers (`--placed`).
        kept = session.directory / LACKING_FILE
        unmet = json.loads(kept.read_text(encoding="utf-8")) if kept.is_file() else {}
    else:
        unmet = lacking(spec)
    for name, why in unmet.items():
        wio.emit(STEP_MARKER, {"step": name, "kind": "needs", "error": "", "blocked": why})
    try:
        try:
            session.start(plan)
        except SystemExit as exc:
            if not isinstance(exc.code, str):
                raise  # a signal, not a server: see runtime._terminate
            report["error"] = f"the deployment did not start: {exc.code}"
        else:
            report["started"] = True
            report["url"] = session.url
            session.ranks(float(spec.get("rank_wait_s", RANK_RECORDS_WAIT_S)))
            for step in spec.get("steps") or ():
                if step["kind"] != "needs" and step["id"] not in unmet:
                    wio.emit(STEP_MARKER, run_step(session, step))
    finally:
        if session.attached:
            # One load per instance whose ranks went past their placement to the engine.
            session.loads = sum(
                1
                for instance in session.instances
                if any(session.directory.glob(f"load.{instance['name']}.*"))
            )
        report["loads"] = session.loads
        report["addresses"] = dict(session.addresses)
        if not report["started"]:
            # What the ranks of a start that failed printed is still what they ran.
            session.ranks()
        report.update(session.stop())
        wio.emit(SESSION_MARKER, report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Read the spec, run the session, print records; 0 once the session record is out."""
    signal.signal(signal.SIGTERM, runtime._terminate)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path, help="the session spec, a JSON file")
    parser.add_argument(
        "--placed",
        action="store_true",
        help="only say where every rank was placed (MV2FLOW_DEVICES), exit 3 on a clash, and "
        "keep what the steps lack on this node for --attach",
    )
    parser.add_argument(
        "--attach",
        action="store_true",
        help="the servers are already started, by the session's own job: start the proxy and ask",
    )
    options = parser.parse_args(argv)
    spec = json.loads(options.spec.read_text(encoding="utf-8"))
    if options.placed:
        records, problem = placed(Path(spec["directory"]), list(spec["instances"]))
        for record in records:
            wio.emit(DEVICES_MARKER, record)
        if problem:
            wio.emit(SESSION_MARKER, {"error": f"nothing was loaded: {problem}", "loads": 0})
            return 3
        kept = Path(spec["directory"]) / LACKING_FILE
        kept.write_text(json.dumps(lacking(spec)), encoding="utf-8")
        return 0
    run({**spec, "attach": True} if options.attach else spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
