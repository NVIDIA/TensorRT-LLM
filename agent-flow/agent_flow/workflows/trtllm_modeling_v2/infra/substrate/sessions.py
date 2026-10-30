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
          role: gates                   # gates | work | any
          gpus_per_node: 4
          time_limit: "24:00:00"
          handover_lead: "01:00:00"
        - name: b
          spool: /shared/run/spool-b
          role: work                    # catalog tests and microbenchmarks

Everything a session's entry script needs is rendered from that block by
:func:`render_session_entry` (``trtllm-modeling-v2 session render``), so there
is no hand-edited entry script with a path in it to go stale.

HAND-OVER. An allocation has a time limit, and a session that reaches it takes
its running jobs with it. The scheduler therefore knows when its
allocation ends; ``handover_lead`` before that it writes
``<spool>/handover/request.json`` and stops admitting jobs that would not finish
in time, so they wait in the spool for the successor, which watches the same
spool. The request is left for the driver session, which holds the cluster's
broker tools (``trtllm-modeling-v2 driver wait`` returns on it, and
:func:`allocate` renders the call that starts the successor after the live
watcher's allocation). The driver records the job it submitted with
:func:`acknowledge` in ``handover/acks/<allocation>.json``, keyed by the
allocation that job follows, so an ack written before the request answers it
too and a later ack for the next allocation leaves it in place. A request
stands only while the watcher that wrote it heart-beats
(:func:`review_requests`); the successor sets its predecessor's request and
acks aside when it starts. The orchestrator never talks to the scheduler of
allocations.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...spec.schema_keys import reject_unknown_keys
from . import SCHEDULER_PATH, SCRIPTS_DIR
from .scheduler import (
    HANDOVER_DIRNAME,
    SpoolPaths,
    answering_ack,
    following_ack,
    parse_duration,
    read_json,
    stale_request,
    utc_iso,
    write_json_atomic,
)

#: What a session is for. The orchestrator routes nothing by role; roles are
#: what the prompts tell agents and what an operator reads.
SESSION_ROLES: tuple[str, ...] = ("gates", "work", "any")

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
    "device_max_used_mib": "--device-max-used-mib",
    "device_release_wait": "--device-release-wait",
}

#: When a session declares a time limit but no lead, the successor is requested this early.
DEFAULT_HANDOVER_LEAD_S = 3600.0

#: How long the orchestrator waits for a lost gate session to heart-beat again
#: before it closes the run resumably (``environment.session_wait``). Long
#: enough for a requeued allocation or a successor to start in the common
#: case; a session that takes longer needs an operator more than a wait.
DEFAULT_SESSION_WAIT_S = 3600.0

#: A watcher that heart-beat within this long holds a live allocation.
LIVE_WATCHER_S = 180.0

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

    @property
    def entry_path(self) -> Path:
        """Where :func:`render_session_entry` output is installed by default."""
        return Path(self.spool) / "session_entry.sh"


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


def session_wait_from_task(task_data: Mapping[str, Any]) -> float:
    """Seconds to wait for a lost gate session to come back (``environment.session_wait``)."""
    raw = _environment_block(task_data).get("session_wait")
    return DEFAULT_SESSION_WAIT_S if raw is None else parse_duration(str(raw))


# ------------------------------------------------------------------ validation


def _check_duration(value: Any, where: str, errors: list[str]) -> None:
    try:
        parse_duration(str(value))
    except ValueError as exc:
        errors.append(f"'{where}' is not a duration (seconds, 90m, 2h or HH:MM:SS): {exc}")


# ---------------------------------------------------------------------------
# Variables whose declared value EXTENDS the inherited one instead of
# replacing it.
#
# Only `PATH` is on this list, and it earns the exception by being the
# variable that decides *which program runs at all*. A spec that wants its
# own interpreter ahead of everything else writes the obvious line:
#
#     PATH: <checkout>/.venv-3.12/bin:/usr/local/bin:/usr/bin:/bin
#
# which reads as "my venv first" and, as a plain assignment, means "and the
# container's MPI, CUDA and toolchain directories are gone". That line drops
# `/usr/local/mpi/bin`, and the next launcher line dies with
# `mpirun: command not found`, rc=127 --- several layers away from the spec
# that caused it, and presenting as a broken probe rather than a mis-declared
# path.
#
# Prepending keeps the property the author was after --- every entry they
# declared still wins the lookup, in the order they declared it --- and drops
# only the part they did not ask for. Rejecting a `PATH` not ending in
# `:$PATH` instead would teach every spec author a rule to make the obvious
# line legal, where this makes the obvious line correct.
#
# `PYTHONPATH` is merged the same way: the gates put their rank hook first on
# it (`checks/gates.RANK_HOOK_DIR`), and the spool scheduler has already put
# its resolve hook and the job's tree there. An unset `PYTHONPATH` must not
# leave a trailing `:`, which Python reads as the working directory.
# ---------------------------------------------------------------------------
MERGED_ENV_VARS: tuple[str, ...] = ("PATH", "PYTHONPATH")
#: Merged variables whose inherited value is appended only when it is set.
_SEPARATOR_WHEN_SET: tuple[str, ...] = ("PYTHONPATH",)


def prepend_to_inherited(name: str, value: str) -> str:
    """``/a:/b`` -> ``'/a:/b':"$PATH"``, idempotent against an explicit suffix.

    The expansion is deliberately outside the quoting: it has to reach the
    shell as an expansion, and ``shlex.quote`` over the whole value would
    deliver ``$PATH`` as five literal characters — which is the same deletion
    with a stranger spelling. A spec that already wrote the suffix by hand is
    accepted and not doubled. ``PYTHONPATH`` becomes
    ``'/a:/b'"${PYTHONPATH:+:$PYTHONPATH}"``: nothing is appended when the
    inherited one is unset or empty.
    """
    for suffix in (f":${name}", f":${{{name}}}"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    if not value or value in (f"${name}", f"${{{name}}}"):
        return f'"${name}"'
    if name in _SEPARATOR_WHEN_SET:
        return f'{shlex.quote(value)}"${{{name}:+:${name}}}"'
    return f'{shlex.quote(value)}:"${name}"'


def validate_sessions(
    block: Mapping[str, Any], errors: list[str], *, where: str = "environment"
) -> None:
    """Validate ``sessions`` and ``session_wait``.

    Appends to ``errors``; every block is closed, like the rest of the schema.
    """
    if block.get("session_wait") is not None:
        _check_duration(block["session_wait"], f"{where}.session_wait", errors)

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
                        "device_release_wait",
                    ):
                        if key in options:
                            _check_duration(options[key], f"{at}.scheduler.{key}", errors)
                    mib = options.get("device_max_used_mib")
                    if mib is not None and (
                        not isinstance(mib, (int, float)) or isinstance(mib, bool) or mib < 0
                    ):
                        errors.append(
                            f"'{at}.scheduler.device_max_used_mib' must be a number of MiB >= 0 "
                            f"(0 turns the device-memory check off)"
                        )


# ------------------------------------------------------------------- rendering


def _export_lines(env: Mapping[str, str]) -> list[str]:
    lines = []
    for key, value in sorted(env.items()):
        if key in MERGED_ENV_VARS:
            lines.append(f"export {key}={prepend_to_inherited(key, str(value))}")
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
    env["MV2FLOW_SESSION"] = session.name
    env["MV2FLOW_SESSION_SPOOLS"] = all_spools
    if task_data.get("trtllm_repo_path"):
        env["MV2FLOW_REPO"] = str(task_data["trtllm_repo_path"])
    python = str(block.get("python") or "python3")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    watch = " ".join(shlex.quote(a) for a in scheduler_arguments(session))
    lines = [
        "#!/bin/bash",
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        f"# Entry script of compute session {session.name!r} ({session.role}), rendered at {stamp}",
        f"# from {source} by `trtllm-modeling-v2 session render`. Do not edit it: change",
        "# environment.sessions and render it again.",
        "#",
        "# Run it as the allocation's command. The scheduler it execs is the",
        "# allocation's main process; work arrives as files in the spool.",
        "set -u",
        f"BASE_SPOOL={shlex.quote(session.spool)}",
        'NODEID="${SLURM_NODEID:-0}"',
        'if [ "$NODEID" = 0 ]; then SPOOL=$BASE_SPOOL; else SPOOL="$BASE_SPOOL-n$NODEID"; fi',
        *_export_lines(env),
        'export MV2FLOW_SPOOL="$SPOOL"',
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
        '  echo "session=$MV2FLOW_SESSION spool=$SPOOL job=${SLURM_JOB_ID:-?} nodes=${SLURM_JOB_NODELIST:-?}"',
        f"  {shlex.quote(python)} -c \"import sys; print('python', sys.version.split()[0], sys.executable)\" || true",
        f"  PYTHONNOUSERSITE=1 {shlex.quote(python)} -c "
        "\"import tensorrt_llm; print('trtllm', tensorrt_llm.__version__, tensorrt_llm.__file__)\" 2>&1 | tail -3",
        "  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv 2>&1",
        "  free -g 2>&1 | head -2; df -h /dev/shm 2>&1 | tail -1",
        "  cat /sys/fs/cgroup/memory.max 2>/dev/null",
        '} > "$SPOOL/session_entry.log" 2>&1',
        "",
        f'SESSION_TAG={shlex.quote(session.name)}; [ "$NODEID" = 0 ] || SESSION_TAG="$SESSION_TAG-n$NODEID"',
        f'exec "${{MV2FLOW_SCHEDULER_PYTHON:-python3}}" {shlex.quote(str(SCHEDULER_PATH))} watch "$SPOOL" '
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


def review_requests(
    task_data: Mapping[str, Any],
    *,
    archive: bool = False,
    by: str = "driver",
    now: float | None = None,
) -> tuple[list[dict], list[str]]:
    """Successor requests that need an answer, and a note on each one set aside.

    A request needs one while the watcher that wrote it heart-beats, no
    successor to it is up (see ``stale_request``) and no ack answers it
    (``ack_answers``). Any other request on disk is an ended allocation's:
    it is ignored, and with ``archive`` moved under ``handover/archive/``.
    """
    pending: list[dict] = []
    notes: list[str] = []
    for session in sessions_from_task(task_data):
        paths = SpoolPaths(session.spool)
        request = read_json(paths.handover_request)
        if not request:
            continue
        name = str(request.get("request_id") or "?")
        why = stale_request(request, paths.live_watchers(LIVE_WATCHER_S, now))
        if why:
            label = f"request-{request.get('slurm_job') or 'unknown'}"
            try:
                moved = archive and paths.archive_handover(
                    paths.handover_request, label, why=why, by=by
                )
            except OSError as exc:
                moved, why = None, f"{why} (not archived: {exc})"
            notes.append(
                f"session {session.name}: {'archived' if moved else 'ignored'} request {name} "
                f"(allocation {request.get('slurm_job')}): {why}"
            )
            continue
        if answering_ack(paths.read_acks(), request):
            continue
        request["session"] = session.name
        request["request_file"] = str(paths.handover_request)
        pending.append(request)
    return pending, notes


def pending_requests(task_data: Mapping[str, Any], now: float | None = None) -> list[dict]:
    """Successor requests that need an answer (see :func:`review_requests`); nothing is moved."""
    return review_requests(task_data, now=now)[0]


def successor_target(spool: str | Path, *, exclude: str = "", now: float | None = None) -> dict:
    """The live watcher whose allocation the session's next successor follows, or ``{}``.

    Of the live watchers with a known allocation (other than ``exclude``'s),
    the one that ends last: a successor queued after it never starts while
    an allocation of the session still has time, and one queued after an
    allocation already succeeded would start beside its successor.
    """
    live = [
        watcher
        for watcher in SpoolPaths(spool).live_watchers(LIVE_WATCHER_S, now)
        if str(watcher.get("slurm_job") or "") not in ("", "unknown", exclude)
    ]
    return max(live, key=lambda w: float(w.get("ends_at_epoch") or "inf"), default={})


def acknowledge(
    spool: str | Path,
    *,
    job_id: str,
    by: str,
    request_id: str | None = None,
    session: str = "",
    after: str | None = None,
) -> Path:
    """Record that ``job_id`` was submitted to succeed the session watching ``spool``.

    The ack names the allocation it follows (``succeeds``): ``after`` (the
    job the submission depends on), else the one :func:`successor_target`
    picks, else the one a request on disk is for. It is written to
    ``handover/acks/<allocation>.json``, so it answers that allocation's
    request whenever the scheduler writes it, and an ack for another
    allocation is never overwritten by it. An earlier ack for the same
    allocation, of another job, is archived.
    """
    paths = SpoolPaths(spool)
    request = read_json(paths.handover_request) or {}
    target: dict = {}
    allocation = str(after or "")
    if not allocation:
        target = successor_target(spool, exclude=str(job_id))
        allocation = str(target.get("slurm_job") or "")
    if not allocation and str(request.get("slurm_job") or "") not in ("", "unknown", str(job_id)):
        allocation = str(request["slurm_job"])
    if allocation and str(request.get("slurm_job") or allocation) != allocation:
        request = {}
    path = paths.ack_path(allocation, str(job_id))
    earlier = read_json(path)
    if earlier and str(earlier.get("job_id") or "") != str(job_id):
        paths.archive_handover(
            path,
            f"ack-{allocation or 'job-' + str(earlier.get('job_id'))}",
            why=f"replaced by the ack of job {job_id} at {utc_iso()}",
            by=by,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        path,
        {
            "request_id": request_id or request.get("request_id"),
            "job_id": str(job_id),
            "succeeds": allocation or None,
            "session": session or request.get("session") or target.get("session"),
            "by": by,
            "at": time.time(),
        },
    )
    return path


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
        f"    job_name={json.dumps(f'mv2flow-{session.name}')}",
        f"    command={json.dumps(['bash', str(session.entry_path)])}",
    ]
    return "mcp__slurm-broker__slurm_submit(\n" + ",\n".join(f for f in fields if f) + ")"


@dataclass(frozen=True)
class SuccessorPlan:
    """What :func:`plan_successor` decided, and why."""

    session: str
    #: The job the successor starts after, or ``None`` to start at once.
    after: str | None
    #: The broker call.
    call: str
    #: Why it starts when it does, in a sentence.
    reason: str
    #: Anything else the driver must know before submitting it.
    notes: tuple[str, ...] = ()


def plan_successor(
    task_data: Mapping[str, Any], name: str, *, now: float | None = None
) -> SuccessorPlan:
    """The submission that starts session ``name``'s successor, its entry installed first.

    It starts after the live watcher's allocation (:func:`successor_target`),
    never after the one a request on disk names: that one may have ended a
    day ago, and a dependency on an ended job starts the successor at once,
    beside the allocation that holds the spool now. With no live watcher the
    submission has no dependency, and says why.
    """
    session = session_by_name(task_data, name)
    paths = SpoolPaths(session.spool)
    target = successor_target(session.spool, now=now)
    after = str(target.get("slurm_job") or "") or None
    notes: list[str] = []
    if after:
        ends = target.get("ends_at") or "at an unknown time"
        reason = (
            f"It starts when allocation {after} ends ({ends}): the live watcher "
            f"{target.get('watcher')} on {session.spool}."
        )
        acked = following_ack(paths.read_acks(), after, session.name)
        if acked:
            notes.append(
                f"Allocation {after} already has a successor acknowledged: job "
                f"{acked.get('job_id')} (by {acked.get('by')}). Submit this only if that job is "
                f"gone (cancelled or failed): both would start when {after} ends."
            )
    else:
        live = paths.live_watchers(LIVE_WATCHER_S, now)
        reason = (
            f"No dependency: the live watcher on {session.spool} runs outside Slurm (allocation "
            f"unknown), so there is no job to start after."
            if live
            else f"No dependency: no watcher on {session.spool} has heart-beat within "
            f"{LIVE_WATCHER_S:.0f}s, so no allocation holds the spool and the session starts "
            f"as soon as Slurm schedules it."
        )
    request = read_json(paths.handover_request) or {}
    named = str(request.get("slurm_job") or "")
    if named and named != after:
        why = stale_request(request, paths.live_watchers(LIVE_WATCHER_S, now))
        notes.append(
            f"The request on disk names allocation {named}, which this does not follow"
            + (f": {why}." if why else ".")
        )
    install_session_entry(task_data, name)
    return SuccessorPlan(
        session=session.name,
        after=after,
        call=broker_submission(task_data, name, after_job=after),
        reason=reason,
        notes=tuple(notes),
    )


def allocate(task_data: Mapping[str, Any], name: str) -> str:
    """The broker call that starts session ``name``'s successor (see :func:`plan_successor`)."""
    return plan_successor(task_data, name).call


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
    "DEFAULT_HANDOVER_LEAD_S",
    "DEFAULT_SESSION_WAIT_S",
    "HANDOVER_DIRNAME",
    "LIVE_WATCHER_S",
    "SCHEDULER_OPTIONS",
    "SCRIPTS_DIR",
    "SESSION_KEYS",
    "SESSION_ROLES",
    "SessionSpec",
    "SuccessorPlan",
    "acknowledge",
    "allocate",
    "broker_submission",
    "describe_sessions",
    "handover_dir",
    "install_session_entry",
    "pending_requests",
    "plan_successor",
    "render_session_entry",
    "review_requests",
    "scheduler_arguments",
    "session_by_name",
    "session_status",
    "session_wait_from_task",
    "sessions_from_task",
    "successor_target",
    "validate_sessions",
]
