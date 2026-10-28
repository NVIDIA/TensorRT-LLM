# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtllm-modeling-v2 session ...``: render and hand over compute sessions.

A session is an allocation whose main process is the spool scheduler, declared in
``environment.sessions``; :func:`render_session_entry` (``trtllm-modeling-v2 session render``)
renders its entry script from that block. While a session is in hand-over (:mod:`.scheduler`)
the operator loop raises ``SESSION_ENDING``, until the driver session, which holds the cluster's
broker tools, submits the successor (:func:`plan_successor`) and records it (:func:`acknowledge`).

The CLI reads only the task file's ``environment`` block, so a session can be rendered
before its spool exists. ``trtllm-modeling-v2 monitor`` reports session state and alerts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ...spec.schema_keys import reject_unknown_keys
from ..files import write_json_atomic
from . import SCHEDULER_PATH
from .scheduler import (
    WATCHER_STALE_S,
    SpoolPaths,
    following_ack,
    latest_watcher,
    parse_duration,
    utc_iso,
)

#: What a session is for: gate jobs go to ``environment.spool`` or a ``gates`` session
#: (``select_command_executor``); otherwise the role only informs the prompts and the operator.
SESSION_ROLES: tuple[str, ...] = ("gates", "work", "any")

SESSION_KEYS: tuple[str, ...] = (
    "name",
    "spool",
    "role",
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
SCHEDULER_OPTIONS: dict[str, str] = {"default_wall": "--default-wall"}

#: When a session declares a time limit but no lead, it hands over this long before its end.
DEFAULT_HANDOVER_LEAD_S = 3600.0

#: How long a gate lost to its session waits for one to heart-beat again before the check is
#: left without a verdict, to run again later (``environment.session_wait``).
DEFAULT_SESSION_WAIT_S = 3600.0

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")


@dataclass(frozen=True)
class SessionSpec:
    """One declared compute session."""

    name: str
    spool: str
    role: str = "any"
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


def sessions_from_task(task_data: Mapping[str, Any]) -> tuple[SessionSpec, ...]:
    """The run's sessions; a bare ``environment.spool`` is one implicit session ``default``."""
    block = task_data.get("environment") or {}
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
    sessions = sessions_from_task(task_data)
    for session in sessions:
        if session.name == name:
            return session
    known = ", ".join(s.name for s in sessions) or "none"
    raise KeyError(f"no session named {name!r} in environment.sessions (declared: {known})")


def session_wait_from_task(task_data: Mapping[str, Any]) -> float:
    """Seconds to wait for a lost gate session to come back (``environment.session_wait``)."""
    raw = (task_data.get("environment") or {}).get("session_wait")
    return DEFAULT_SESSION_WAIT_S if raw is None else parse_duration(str(raw))


# ------------------------------------------------------------------ validation


def _check_duration(value: Any, where: str, errors: list[str]) -> None:
    try:
        parse_duration(str(value))
    except ValueError as exc:
        errors.append(f"'{where}' is not a duration (seconds, 90m, 2h or HH:MM:SS): {exc}")


# Variables whose declared value EXTENDS the inherited one. A plain `PATH: <venv>/bin:/usr/bin`
# drops `/usr/local/mpi/bin`, and the next launcher line dies `mpirun: command not found` (rc
# 127). `PYTHONPATH` already holds the gates' rank hook (`checks/check_runtime.RANK_HOOK_DIR`), the
# scheduler's resolve hook and the job's tree; unset, it must not leave a trailing `:`, which
# Python reads as the working directory.
MERGED_ENV_VARS: tuple[str, ...] = ("PATH", "PYTHONPATH")
#: Merged variables whose inherited value is appended only when it is set.
_SEPARATOR_WHEN_SET: tuple[str, ...] = ("PYTHONPATH",)


def prepend_to_inherited(name: str, value: str) -> str:
    """``/a:/b`` -> ``/a:/b:"$PATH"`` (the value shell-quoted), idempotent against a ``:$PATH``.

    The expansion stays outside the quoting so that the shell expands it. ``PYTHONPATH`` becomes
    ``/a:/b"${PYTHONPATH:+:$PYTHONPATH}"``: nothing is appended when it is unset or empty.
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
    """Validate ``sessions`` and ``session_wait`` into ``errors``; every block is closed."""
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
            gpus = raw.get("gpus_per_node")
            if gpus is not None and (
                not isinstance(gpus, int) or isinstance(gpus, bool) or gpus < 0
            ):
                errors.append(f"'{at}.gpus_per_node' must be an integer >= 0")
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
                    if "default_wall" in options:
                        _check_duration(
                            options["default_wall"], f"{at}.scheduler.default_wall", errors
                        )


# ------------------------------------------------------------------- rendering


def env_assignments(env: Mapping[str, str]) -> list[str]:
    """Stable shell assignments; path variables prepend rather than erase inherited values."""
    return [
        f"{key}={prepend_to_inherited(key, str(value)) if key in MERGED_ENV_VARS else shlex.quote(str(value))}"
        for key, value in sorted(env.items())
    ]


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
    """The entry script of session ``name``: set the run's environment, become its scheduler."""
    session = session_by_name(task_data, name)
    block = task_data.get("environment") or {}
    env = {str(k): str(v) for k, v in (block.get("env") or {}).items()}
    env.update(session.env)
    env.setdefault("PYTHONNOUSERSITE", "1")
    all_spools = ":".join(s.spool for s in sessions_from_task(task_data))
    env["MV2FLOW_SESSION"] = session.name
    env["MV2FLOW_SESSION_SPOOLS"] = all_spools
    if task_data.get("trtllm_repo_path"):
        env["MV2FLOW_REPO"] = str(task_data["trtllm_repo_path"])
    python = str(block.get("python") or "python3")
    stamp = utc_iso()
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
        f"SPOOL={shlex.quote(session.spool)}",
        *(f"export {assignment}" for assignment in env_assignments(env)),
        'export MV2FLOW_SPOOL="$SPOOL"',
        "ulimit -c 0",
        'mkdir -p "$SPOOL"',
    ]
    lines += list(session.setup)
    lines += [
        "{",
        '  echo "=== session entry $(date -Is) on $(hostname) ==="',
        '  echo "session=$MV2FLOW_SESSION spool=$SPOOL job=${SLURM_JOB_ID:-?} nodes=${SLURM_JOB_NODELIST:-?}"',
        f"  {shlex.quote(python)} -c \"import sys; print('python', sys.version.split()[0], sys.executable)\" || true",
        f"  PYTHONNOUSERSITE=1 {shlex.quote(python)} -c "
        "\"import tensorrt_llm; print('trtllm', tensorrt_llm.__version__, tensorrt_llm.__file__)\" 2>&1 | tail -3",
        "  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv 2>&1",
        "  free -g 2>&1 | head -2; df -h /dev/shm 2>&1 | tail -1",
        "  cat /sys/fs/cgroup/memory.max 2>/dev/null",
        '} > "$SPOOL/session_entry.log" 2>&1',
        "",
        f'exec "${{MV2FLOW_SCHEDULER_PYTHON:-python3}}" {shlex.quote(str(SCHEDULER_PATH))} watch "$SPOOL" '
        f"{watch} --session {shlex.quote(session.name)}",
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


def successor_target(spool: str | Path, *, exclude: str = "") -> dict:
    """The live watcher whose allocation the session's next successor follows, or ``{}``.

    Of the live watchers with a known allocation (but ``exclude``'s), the one that ends last: a
    successor queued after it never starts while the session still has time, and one queued after
    an allocation already succeeded would start beside its successor.
    """
    live = [
        watcher
        for watcher in SpoolPaths(spool).live_watchers(WATCHER_STALE_S)
        if str(watcher.get("slurm_job") or "") not in ("", "unknown", exclude)
    ]
    return max(live, key=lambda w: float(w.get("ends_at_epoch") or "inf"), default={})


def acknowledge(
    spool: str | Path,
    *,
    job_id: str,
    by: str,
    session: str = "",
    after: str | None = None,
) -> Path:
    """Record that ``job_id`` was submitted to succeed the session watching ``spool``.

    The ack names the allocation it follows (``succeeds``): ``after`` (the job the submission
    depends on), else :func:`successor_target`'s. A later ack for the same allocation replaces it.
    """
    target: dict = {}
    allocation = str(after or "")
    if not allocation:
        target = successor_target(spool, exclude=str(job_id))
        allocation = str(target.get("slurm_job") or "")
    path = SpoolPaths(spool).ack_path(allocation, str(job_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        path,
        {
            "job_id": str(job_id),
            "succeeds": allocation or None,
            "session": session or target.get("session"),
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


def plan_successor(task_data: Mapping[str, Any], name: str) -> SuccessorPlan:
    """The submission that starts session ``name``'s successor, its entry installed first.

    It depends on the live watcher's allocation (:func:`successor_target`): a dependency on one
    that already ended would start the successor at once, beside the allocation holding the spool.
    With no live watcher it has no dependency, and says why.
    """
    session = session_by_name(task_data, name)
    paths = SpoolPaths(session.spool)
    target = successor_target(session.spool)
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
        live = paths.live_watchers(WATCHER_STALE_S)
        reason = (
            f"No dependency: the live watcher on {session.spool} runs outside Slurm (allocation "
            f"unknown), so there is no job to start after."
            if live
            else f"No dependency: no watcher on {session.spool} has heart-beat within "
            f"{WATCHER_STALE_S:.0f}s, so no allocation holds the spool and the session starts "
            f"as soon as Slurm schedules it."
        )
    install_session_entry(task_data, name)
    return SuccessorPlan(
        session=session.name,
        after=after,
        call=broker_submission(task_data, name, after_job=after),
        reason=reason,
        notes=tuple(notes),
    )


def session_status(task_data: Mapping[str, Any], now: float | None = None) -> list[dict]:
    """One line of state per declared session, read from its spool."""
    now = time.time() if now is None else now
    out = []
    for session in sessions_from_task(task_data):
        snapshot = latest_watcher(session.spool)
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


class SessionConfigError(ValueError):
    """The task file's session configuration cannot be used."""


def load_session_config(path: str | Path) -> dict[str, Any]:
    """Read ``path`` and validate only what sessions need."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SessionConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("environment"), dict):
        raise SessionConfigError(f"{path} has no environment block")
    errors: list[str] = []
    validate_sessions(data["environment"], errors)
    if errors:
        raise SessionConfigError("invalid session configuration:\n  - " + "\n  - ".join(errors))
    if not sessions_from_task(data):
        raise SessionConfigError(f"{path} declares no session (environment.sessions)")
    return data


def check_session_tmpdir(task: dict[str, Any], name: str) -> None:
    """Hold the environment session ``name`` exports to preflight's ``TMPDIR`` rule."""
    from .environment import EnvironmentError, check_tmpdir

    session = session_by_name(task, name)
    env = {str(k): str(v) for k, v in (task["environment"].get("env") or {}).items()}
    env.update(session.env)
    try:
        check_tmpdir(env, require_set=False)
    except EnvironmentError as exc:
        raise SessionConfigError(f"session {name!r}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 session", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="action", required=True)

    def add(name: str, *, help_text: str) -> argparse.ArgumentParser:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--task", required=True, type=Path, help="the run's task.yaml")
        command.add_argument("--name", required=True, help="session name")
        return command

    render = add("render", help_text="write the session's entry script")
    render.add_argument(
        "--out", type=Path, help="where to write it (default <spool>/session_entry.sh)"
    )
    render.add_argument("--stdout", action="store_true", help="print it instead of writing it")
    submission = add("submission", help_text="the broker call that starts the session")
    submission.add_argument("--after", help="job id the new allocation starts after")
    add("allocate", help_text="the broker call that starts a session's successor")
    ack = add("ack", help_text="record the job id of a submitted successor")
    ack.add_argument("--job", required=True)
    ack.add_argument(
        "--after",
        help="the allocation the job was submitted to follow (default: the live watcher's)",
    )
    ack.add_argument("--by", default="driver")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        task = load_session_config(args.task)
        if args.action in ("render", "submission", "allocate"):
            check_session_tmpdir(task, args.name)
        if args.action == "render":
            if args.stdout:
                sys.stdout.write(render_session_entry(task, args.name, source=str(args.task)))
            else:
                path = install_session_entry(task, args.name, out=args.out, source=str(args.task))
                print(path)
            return 0
        if args.action == "submission":
            print(broker_submission(task, args.name, after_job=args.after))
            return 0
        if args.action == "allocate":
            plan = plan_successor(task, args.name)
            ack = (
                f"trtllm-modeling-v2 session ack --task {args.task} --name {plan.session} "
                f"--job <id>" + (f" --after {plan.after}" if plan.after else "")
            )
            print(f"Submit this through the broker, then record it with `{ack}`:\n")
            print(plan.call)
            print("\n" + "\n".join([plan.reason, *plan.notes]))
            return 0
        if args.action == "ack":
            session = session_by_name(task, args.name)
            print(
                acknowledge(
                    session.spool,
                    job_id=args.job,
                    by=args.by,
                    session=session.name,
                    after=args.after,
                )
            )
            return 0
    except (SessionConfigError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


__all__ = [
    "DEFAULT_HANDOVER_LEAD_S",
    "DEFAULT_SESSION_WAIT_S",
    "SCHEDULER_OPTIONS",
    "SESSION_KEYS",
    "SESSION_ROLES",
    "SessionConfigError",
    "SessionSpec",
    "SuccessorPlan",
    "acknowledge",
    "broker_submission",
    "build_parser",
    "check_session_tmpdir",
    "describe_sessions",
    "install_session_entry",
    "load_session_config",
    "main",
    "plan_successor",
    "render_session_entry",
    "scheduler_arguments",
    "session_by_name",
    "session_status",
    "session_wait_from_task",
    "sessions_from_task",
    "successor_target",
    "validate_sessions",
]
