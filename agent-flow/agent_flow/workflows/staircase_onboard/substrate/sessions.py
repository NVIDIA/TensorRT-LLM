# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute sessions declared in ``task.yaml``: validation, entry scripts and hand-over.

A run stands on one or more **sessions**: allocations whose main process is
the spool scheduler. They are declared once, in ``environment.sessions``::

    environment:
      kind: session_spool
      spool: /shared/run/spool-a        # where gates go
      sessions:
        - name: a
          spool: /shared/run/spool-a
          role: gates                   # gates | work | orchestrator | any
          gpus_per_node: 4
          time_limit: "24:00:00"
          handover_lead: "01:00:00"
        - name: b
          spool: /shared/run/spool-b
          role: work                    # catalog tests and microbenchmarks
        - name: orch
          spool: /shared/run/spool-orch
          role: orchestrator            # GPU-free, long-lived
          gpus_per_node: 0
      allocator:
        kind: request_file              # or: command, with `command:`

Everything a session's entry script needs is rendered from that block by
:func:`render_session_entry` (``staircase-onboard session render``), so there
is no hand-edited entry script with a path in it to go stale (F001).

HAND-OVER. An allocation has a time limit, and a session that reaches it takes
its running jobs with it (F035). The scheduler therefore knows when its
allocation ends; ``handover_lead`` before that it writes
``<spool>/handover/request.json`` and stops admitting jobs that would not finish
in time, so they wait in the spool for the successor, which watches the same
spool. Who acquires the successor is pluggable (:func:`allocate`): the default
``request_file`` allocator leaves the request for the driver session, which
holds the cluster's broker tools (``staircase-onboard driver wait`` returns on
it), and a ``command`` allocator runs a shell command where one works. Either
way the orchestrator never talks to the scheduler of allocations.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import SCHEDULER_PATH, SCRIPTS_DIR
from .scheduler import parse_duration, read_json, write_json_atomic

#: What a session is for. The orchestrator routes nothing by role; roles are
#: what the prompts tell agents and what an operator reads.
SESSION_ROLES: tuple[str, ...] = ("gates", "work", "orchestrator", "any")

SESSION_KEYS: tuple[str, ...] = (
    "name",
    "spool",
    "role",
    "nodes",
    "gpus_per_node",
    "time_limit",
    "handover_lead",
    "env",
    "setup",
    "scheduler",
    "partition",
    "account",
    "qos",
    "container_image",
    "container_name",
    "mounts",
)

#: Scheduler options a session may set; each maps to a ``scheduler.py watch`` flag.
SCHEDULER_OPTIONS: dict[str, str] = {
    "poll": "--poll",
    "default_wall": "--default-wall",
    "aging": "--aging",
    "kill_grace": "--kill-grace",
    "orphan_after": "--orphan-after",
    "pressure_floor_gb": "--pressure-floor-gb",
    "stall": "--stall",
    "failure_stall": "--failure-stall",
    "engine_host_mem_gb": "--engine-host-mem-gb",
    "host_mem_wait": "--host-mem-wait",
}

ALLOCATOR_KINDS: tuple[str, ...] = ("request_file", "command")
ALLOCATOR_KEYS: tuple[str, ...] = ("kind", "command")
ORCHESTRATOR_KEYS: tuple[str, ...] = ("session",)

#: When a session declares a time limit but no lead, the successor is requested this early.
DEFAULT_HANDOVER_LEAD_S = 3600.0

#: Where, under a spool, the hand-over files live.
HANDOVER_DIRNAME = "handover"

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")


@dataclass(frozen=True)
class SessionSpec:
    """One declared compute session."""

    name: str
    spool: str
    role: str = "any"
    nodes: int = 1
    #: ``None`` lets the scheduler discover the devices; 0 makes a GPU-free session.
    gpus_per_node: int | None = None
    time_limit_s: float | None = None
    time_limit: str | None = None
    handover_lead_s: float = DEFAULT_HANDOVER_LEAD_S
    env: Mapping[str, str] = field(default_factory=dict)
    setup: tuple[str, ...] = ()
    scheduler: Mapping[str, Any] = field(default_factory=dict)
    #: Submission parameters for this session's allocation (and its successor).
    submit: Mapping[str, Any] = field(default_factory=dict)

    def node_spool(self, node: int) -> str:
        """The spool the watcher on ``node`` drains: the session's own on node 0."""
        return self.spool if node == 0 else f"{self.spool}-n{node}"

    @property
    def entry_path(self) -> Path:
        """Where :func:`render_session_entry` output is installed by default."""
        return Path(self.spool) / "session_entry.sh"


@dataclass(frozen=True)
class AllocatorSpec:
    """How a successor allocation is acquired."""

    kind: str = "request_file"
    command: str | None = None


def _environment_block(task_data: Mapping[str, Any]) -> Mapping[str, Any]:
    return task_data.get("environment") or {}


def sessions_from_task(task_data: Mapping[str, Any]) -> tuple[SessionSpec, ...]:
    """The run's sessions; a bare ``environment.spool`` is one implicit session ``default``."""
    block = _environment_block(task_data)
    declared = block.get("sessions") or []
    defaults = {
        k: block.get(k)
        for k in ("partition", "account", "container_image", "container_name", "mounts")
        if block.get(k)
    }
    out = []
    for raw in declared:
        lead = raw.get("handover_lead")
        limit = raw.get("time_limit")
        submit = {
            **defaults,
            **{
                k: raw[k]
                for k in (
                    "partition",
                    "account",
                    "qos",
                    "container_image",
                    "container_name",
                    "mounts",
                )
                if raw.get(k) is not None
            },
        }
        gpus = raw.get("gpus_per_node", block.get("gpus_per_node"))
        out.append(
            SessionSpec(
                name=str(raw["name"]),
                spool=str(raw["spool"]),
                role=str(raw.get("role", "any")),
                nodes=int(raw.get("nodes", 1)),
                gpus_per_node=None if gpus is None else int(gpus),
                time_limit_s=parse_duration(str(limit)) if limit else None,
                time_limit=str(limit) if limit else None,
                handover_lead_s=parse_duration(str(lead)) if lead else DEFAULT_HANDOVER_LEAD_S,
                env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
                setup=tuple(str(line) for line in raw.get("setup") or ()),
                scheduler=dict(raw.get("scheduler") or {}),
                submit=submit,
            )
        )
    if not out and block.get("kind") == "session_spool" and block.get("spool"):
        out.append(
            SessionSpec(
                name="default",
                spool=str(block["spool"]),
                gpus_per_node=block.get("gpus_per_node"),
                time_limit_s=parse_duration(str(block["time_limit"]))
                if block.get("time_limit")
                else None,
                time_limit=block.get("time_limit"),
                submit=defaults,
            )
        )
    return tuple(out)


def session_by_name(task_data: Mapping[str, Any], name: str) -> SessionSpec:
    for session in sessions_from_task(task_data):
        if session.name == name:
            return session
    known = ", ".join(s.name for s in sessions_from_task(task_data)) or "none"
    raise KeyError(f"no session named {name!r} in environment.sessions (declared: {known})")


def allocator_from_task(task_data: Mapping[str, Any]) -> AllocatorSpec:
    raw = _environment_block(task_data).get("allocator") or {}
    return AllocatorSpec(kind=str(raw.get("kind", "request_file")), command=raw.get("command"))


# ------------------------------------------------------------------ validation


def _check_duration(value: Any, where: str, errors: list[str]) -> None:
    try:
        parse_duration(str(value))
    except ValueError as exc:
        errors.append(f"'{where}' is not a duration (seconds, 90m, 2h or HH:MM:SS): {exc}")


def validate_sessions(
    block: Mapping[str, Any], errors: list[str], *, where: str = "environment"
) -> None:
    """Validate ``sessions``, ``orchestrator`` and ``allocator`` in an environment block.

    Appends to ``errors``; every block is closed, like the rest of the schema.
    """
    from ..task_schema import reject_unknown_keys

    sessions = block.get("sessions")
    names: list[str] = []
    spools: list[str] = []
    if sessions is not None:
        if not isinstance(sessions, list) or not sessions:
            errors.append(f"'{where}.sessions' must be a non-empty list of session blocks")
            sessions = []
        for index, raw in enumerate(sessions):
            at = f"{where}.sessions[{index}]"
            if not isinstance(raw, dict):
                errors.append(f"'{at}' must be a mapping")
                continue
            reject_unknown_keys(raw, SESSION_KEYS, errors, where=at)
            name = raw.get("name")
            if not isinstance(name, str) or not _NAME.match(name):
                errors.append(f"'{at}.name' must match {_NAME.pattern}")
            elif name in names:
                errors.append(f"'{at}.name' {name!r} is declared twice")
            else:
                names.append(name)
            spool = raw.get("spool")
            if not isinstance(spool, str) or not spool.startswith("/"):
                errors.append(f"'{at}.spool' must be an absolute path on the shared filesystem")
            elif spool in spools:
                errors.append(
                    f"'{at}.spool' {spool} is used by two sessions; one spool per session, "
                    f"or their watchers hand out the same devices twice"
                )
            else:
                spools.append(spool)
            role = raw.get("role", "any")
            if role not in SESSION_ROLES:
                errors.append(f"'{at}.role' must be one of {SESSION_ROLES}; got {role!r}")
            for key in ("nodes", "gpus_per_node"):
                value = raw.get(key)
                minimum = 1 if key == "nodes" else 0
                if value is not None and (
                    not isinstance(value, int) or isinstance(value, bool) or value < minimum
                ):
                    errors.append(f"'{at}.{key}' must be an integer >= {minimum}")
            for key in ("time_limit", "handover_lead"):
                if raw.get(key) is not None:
                    _check_duration(raw[key], f"{at}.{key}", errors)
            env = raw.get("env")
            if env is not None and (
                not isinstance(env, dict)
                or not all(isinstance(v, (str, int, float)) for v in env.values())
            ):
                errors.append(f"'{at}.env' must be a mapping of scalars")
            setup = raw.get("setup")
            if setup is not None and (
                not isinstance(setup, list) or not all(isinstance(s, str) for s in setup)
            ):
                errors.append(f"'{at}.setup' must be a list of shell lines")
            options = raw.get("scheduler")
            if options is not None:
                if not isinstance(options, dict):
                    errors.append(f"'{at}.scheduler' must be a mapping")
                else:
                    reject_unknown_keys(
                        options, tuple(SCHEDULER_OPTIONS), errors, where=f"{at}.scheduler"
                    )
                    for key in (
                        "default_wall",
                        "aging",
                        "kill_grace",
                        "orphan_after",
                        "stall",
                        "failure_stall",
                        "host_mem_wait",
                    ):
                        if key in options:
                            _check_duration(options[key], f"{at}.scheduler.{key}", errors)
    orchestrator = block.get("orchestrator")
    if orchestrator is not None:
        if not isinstance(orchestrator, dict):
            errors.append(f"'{where}.orchestrator' must be a mapping")
        else:
            reject_unknown_keys(
                orchestrator, ORCHESTRATOR_KEYS, errors, where=f"{where}.orchestrator"
            )
            session = orchestrator.get("session")
            if session not in names:
                errors.append(
                    f"'{where}.orchestrator.session' must name a declared session; got {session!r}"
                )
    allocator = block.get("allocator")
    if allocator is not None:
        if not isinstance(allocator, dict):
            errors.append(f"'{where}.allocator' must be a mapping")
        else:
            reject_unknown_keys(allocator, ALLOCATOR_KEYS, errors, where=f"{where}.allocator")
            kind = allocator.get("kind", "request_file")
            if kind not in ALLOCATOR_KINDS:
                errors.append(f"'{where}.allocator.kind' must be one of {ALLOCATOR_KINDS}")
            if kind == "command" and not isinstance(allocator.get("command"), str):
                errors.append(f"'{where}.allocator.command' is required when kind is 'command'")


# ------------------------------------------------------------------- rendering


def _export_lines(env: Mapping[str, str]) -> list[str]:
    from ..environment import MERGED_ENV_VARS, _prepend_to_inherited

    lines = []
    for key, value in sorted(env.items()):
        if key in MERGED_ENV_VARS:
            lines.append(f"export {key}={_prepend_to_inherited(key, str(value))}")
        else:
            lines.append(f"export {key}={shlex.quote(str(value))}")
    return lines


def scheduler_arguments(session: SessionSpec) -> list[str]:
    """``scheduler.py watch`` options for a session, besides the spool."""
    args: list[str] = []
    if session.gpus_per_node is not None:
        args += ["--gpus", str(session.gpus_per_node)]
    if session.time_limit_s:
        args += ["--time-limit", str(int(session.time_limit_s))]
        args += ["--handover-lead", str(int(session.handover_lead_s))]
    for key, flag in SCHEDULER_OPTIONS.items():
        if key in session.scheduler:
            args += [flag, str(session.scheduler[key])]
    return args


def render_session_entry(
    task_data: Mapping[str, Any], name: str, *, source: str = "task.yaml"
) -> str:
    """The entry script of session ``name``: set the run's environment, become its scheduler.

    The script is the allocation's command, so the scheduler it execs is the
    allocation's main process. On a multi-node allocation it runs once per
    node: node 0 drains the session's spool, node ``k`` drains ``<spool>-n<k>``,
    and each registers ``<spool>/hosts/<hostname>`` so the multi-node launcher
    can reach every node through its spool.
    """
    session = session_by_name(task_data, name)
    block = _environment_block(task_data)
    env = {str(k): str(v) for k, v in (block.get("env") or {}).items()}
    env.update(session.env)
    env.setdefault("PYTHONNOUSERSITE", "1")
    all_spools = ":".join(s.spool for s in sessions_from_task(task_data))
    env["STAIRCASE_SESSION"] = session.name
    env["STAIRCASE_SESSION_SPOOLS"] = all_spools
    if task_data.get("trtllm_repo_path"):
        env["STAIRCASE_REPO"] = str(task_data["trtllm_repo_path"])
    python = str(block.get("python") or "python3")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    watch = " ".join(shlex.quote(a) for a in scheduler_arguments(session))
    lines = [
        "#!/bin/bash",
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        f"# Entry script of compute session {session.name!r} ({session.role}), rendered at {stamp}",
        f"# from {source} by `staircase-onboard session render`. Do not edit it: change",
        "# environment.sessions and render it again.",
        "#",
        "# Run it as the allocation's command. The scheduler it execs is the",
        "# allocation's main process; work arrives as files in the spool.",
        "set -u",
        f"BASE_SPOOL={shlex.quote(session.spool)}",
        'NODEID="${SLURM_NODEID:-0}"',
        'if [ "$NODEID" = 0 ]; then SPOOL=$BASE_SPOOL; else SPOOL="$BASE_SPOOL-n$NODEID"; fi',
        *_export_lines(env),
        'export STAIRCASE_SPOOL="$SPOOL"',
        "ulimit -c 0",
        'mkdir -p "$SPOOL"',
    ]
    if session.nodes > 1:
        lines += [
            'mkdir -p "$BASE_SPOOL/hosts"',
            'echo "$SPOOL" > "$BASE_SPOOL/hosts/$(hostname)"',
        ]
    lines += list(session.setup)
    lines += [
        "{",
        '  echo "=== session entry $(date -Is) on $(hostname) node=$NODEID ==="',
        '  echo "session=$STAIRCASE_SESSION spool=$SPOOL job=${SLURM_JOB_ID:-?} nodes=${SLURM_JOB_NODELIST:-?}"',
        f"  {shlex.quote(python)} -c \"import sys; print('python', sys.version.split()[0], sys.executable)\" || true",
        f"  PYTHONNOUSERSITE=1 {shlex.quote(python)} -c "
        "\"import tensorrt_llm; print('trtllm', tensorrt_llm.__version__, tensorrt_llm.__file__)\" 2>&1 | tail -3",
        "  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv 2>&1",
        "  free -g 2>&1 | head -2; df -h /dev/shm 2>&1 | tail -1",
        "  cat /sys/fs/cgroup/memory.max 2>/dev/null",
        '} > "$SPOOL/session_entry.log" 2>&1',
        "",
        f'SESSION_TAG={shlex.quote(session.name)}; [ "$NODEID" = 0 ] || SESSION_TAG="$SESSION_TAG-n$NODEID"',
        f'exec "${{STAIRCASE_SCHEDULER_PYTHON:-python3}}" {shlex.quote(str(SCHEDULER_PATH))} watch "$SPOOL" '
        f'{watch} --session "$SESSION_TAG" --entry "$(readlink -f "${{BASH_SOURCE[0]}}")"',
    ]
    return "\n".join(lines) + "\n"


def install_session_entry(
    task_data: Mapping[str, Any], name: str, *, out: Path | None = None, source: str = "task.yaml"
) -> Path:
    """Render session ``name``'s entry and write it (executable) where its successor finds it."""
    session = session_by_name(task_data, name)
    target = Path(out) if out else session.entry_path
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.tmp")
    staging.write_text(render_session_entry(task_data, name, source=source), encoding="utf-8")
    staging.chmod(0o755)
    os.replace(staging, target)
    return target


# -------------------------------------------------------------------- hand-over


def handover_dir(spool: str | Path) -> Path:
    return Path(spool) / HANDOVER_DIRNAME


def pending_requests(task_data: Mapping[str, Any]) -> list[dict]:
    """Successor requests a session wrote that nobody has acknowledged yet."""
    out = []
    for session in sessions_from_task(task_data):
        directory = handover_dir(session.spool)
        request = read_json(directory / "request.json")
        if not request:
            continue
        ack = read_json(directory / "ack.json")
        if ack and ack.get("request_id") == request.get("request_id"):
            continue
        request["session"] = session.name
        request["request_file"] = str(directory / "request.json")
        out.append(request)
    return out


def acknowledge(spool: str | Path, *, job_id: str, by: str, request_id: str | None = None) -> Path:
    """Record that a successor allocation was requested for the session watching ``spool``."""
    directory = handover_dir(spool)
    request = read_json(directory / "request.json") or {}
    target = directory / "ack.json"
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        target,
        {
            "request_id": request_id or request.get("request_id"),
            "job_id": str(job_id),
            "by": by,
            "at": time.time(),
        },
    )
    return target


def broker_submission(
    task_data: Mapping[str, Any], name: str, *, after_job: str | None = None
) -> str:
    """The broker call that starts session ``name`` (after ``after_job``, for a successor).

    Rendered for the driver, which holds the broker tools; nothing here calls it.
    """
    session = session_by_name(task_data, name)
    submit = dict(session.submit)
    fields = [
        f"    image={json.dumps(submit.get('container_image') or '<container image>')}",
        f"    partition={json.dumps(submit.get('partition'))}" if submit.get("partition") else None,
        f"    account={json.dumps(submit.get('account'))}" if submit.get("account") else None,
        f"    qos={json.dumps(submit.get('qos'))}" if submit.get("qos") else None,
        f"    nodes={session.nodes}" if session.nodes > 1 else None,
        f"    segment={session.nodes}" if session.nodes > 1 else None,
        f"    gpus_per_node={session.gpus_per_node}" if session.gpus_per_node else None,
        f"    time_limit={json.dumps(session.time_limit)}" if session.time_limit else None,
        f"    mounts={json.dumps(list(submit.get('mounts') or []))}"
        if submit.get("mounts")
        else None,
        f"    dependency={json.dumps(f'afterany:{after_job}')}" if after_job else None,
        f"    job_name={json.dumps(f'staircase-{session.name}')}",
        f"    command={json.dumps(['bash', str(session.entry_path)])}",
    ]
    return "mcp__slurm-broker__slurm_submit(\n" + ",\n".join(f for f in fields if f) + ")"


def allocate(task_data: Mapping[str, Any], name: str, *, runner=subprocess.run) -> dict:
    """Act on session ``name``'s pending successor request with the configured allocator.

    ``request_file`` returns the broker call for the driver to make, and
    acknowledges nothing. ``command`` runs ``allocator.command`` with
    ``SESSION_NAME``, ``SESSION_SPOOL``, ``SESSION_ENTRY`` and ``AFTER_JOB`` in
    its environment, takes the first number it prints as the new job id and
    acknowledges the request.
    """
    session = session_by_name(task_data, name)
    request = read_json(handover_dir(session.spool) / "request.json") or {}
    after = str(request.get("slurm_job") or "") or None
    if after == "unknown":
        after = None
    install_session_entry(task_data, name)
    allocator = allocator_from_task(task_data)
    if allocator.kind != "command":
        return {
            "kind": "request_file",
            "submission": broker_submission(task_data, name, after_job=after),
        }
    env = {
        **os.environ,
        "SESSION_NAME": session.name,
        "SESSION_SPOOL": session.spool,
        "SESSION_ENTRY": str(session.entry_path),
        "AFTER_JOB": after or "",
    }
    done = runner(
        allocator.command,
        shell=True,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    match = re.search(r"\b(\d+)\b", done.stdout or "")
    if done.returncode != 0 or not match:
        return {"kind": "command", "ok": False, "stdout": done.stdout, "stderr": done.stderr}
    acknowledge(
        session.spool,
        job_id=match.group(1),
        by="allocator:command",
        request_id=request.get("request_id"),
    )
    return {"kind": "command", "ok": True, "job_id": match.group(1)}


def session_status(task_data: Mapping[str, Any], now: float | None = None) -> list[dict]:
    """One line of state per declared session, read from its spool."""
    now = time.time() if now is None else now
    out = []
    for session in sessions_from_task(task_data):
        snapshot = read_json(Path(session.spool) / "status.json") or {}
        age = now - float(snapshot.get("alive_epoch", 0)) if snapshot else None
        out.append(
            {
                "name": session.name,
                "role": session.role,
                "spool": session.spool,
                "heartbeat_age_s": None if age is None else round(age, 1),
                "slurm_job": snapshot.get("slurm_job"),
                "ends_at": snapshot.get("ends_at"),
                "remaining_s": snapshot.get("remaining_s"),
                "handover": snapshot.get("handover"),
                "running": len(snapshot.get("running") or []),
                "pending": len(snapshot.get("pending") or []),
            }
        )
    return out


def describe_sessions(sessions: Sequence[SessionSpec]) -> str:
    """The prompt lines naming each session and what it is for."""
    if len(sessions) < 2:
        return ""
    purpose = {
        "gates": "gates and anything that loads the full model",
        "work": "catalog tests, microbenchmarks and other single-device work",
        "orchestrator": "the orchestrator only; it has no GPUs",
        "any": "any work",
    }
    lines = ["This run has several compute sessions. Submit to the one meant for your work:", ""]
    for session in sessions:
        lines.append(
            f"- `{session.name}` — {purpose.get(session.role, session.role)}: `SPOOL={session.spool}`"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ALLOCATOR_KINDS",
    "AllocatorSpec",
    "DEFAULT_HANDOVER_LEAD_S",
    "HANDOVER_DIRNAME",
    "SCHEDULER_OPTIONS",
    "SCRIPTS_DIR",
    "SESSION_KEYS",
    "SESSION_ROLES",
    "SessionSpec",
    "acknowledge",
    "allocate",
    "allocator_from_task",
    "broker_submission",
    "describe_sessions",
    "handover_dir",
    "install_session_entry",
    "pending_requests",
    "render_session_entry",
    "scheduler_arguments",
    "session_by_name",
    "session_status",
    "sessions_from_task",
    "validate_sessions",
]
