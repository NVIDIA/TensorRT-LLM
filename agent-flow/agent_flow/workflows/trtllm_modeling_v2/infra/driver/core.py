# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver's half of the external agent transport, and the operator's loop of any run.

With ``--agent-transport external`` the orchestrator writes each role turn to
``<workspace>/dispatch/`` and blocks; an authenticated session (the *driver*)
hands each turn to a subagent and the subagent writes the answer back. This
module is everything the driver needs, so a run needs no hand-written shell
glue. A run whose turns run in process has no turns to hand out, and its
operator runs the same loop with ``sessions_only`` for the rest: the
orchestrator's exit, reloads and the compute sessions.

- :func:`wait` blocks until there is something to do: a new unclaimed turn
  (a worker's within the in-flight cap; the planner's or the critic's at once,
  ahead of any worker's), the orchestrator exiting, a reload completing,
  failing or running past its deadline, a session asking for a successor
  allocation, or a session needing attention. A run that closes writes no
  further request, so a waiter that watched only for requests would wait past
  the end of the run. A successor request no live watcher stands behind is
  archived, and the event it returns says so.
- :func:`request_reload` asks the orchestrator to drain and re-execute itself
  on the code now at its checkout (``reload.py``).
- :func:`claim` / :func:`release` record which driver took a turn, so two
  drivers never run one turn twice.
- :func:`render_prompt` produces the exact instruction to hand a subagent:
  every path resolved, the role's backend and model, its spool job prefix,
  its prior turns.
- :func:`render_instructions` renders the run's standing ``TURN_INSTRUCTIONS``
  from ``task.yaml``, with nothing machine-specific written into it.
- :func:`status` shows every turn, its claim, its age against its role's
  timeout, whether a relaunched orchestrator adopted it, the orchestrator's
  liveness, what it holds in flight, and a reload under way.

Nothing here runs a turn. It formats hand-offs and reads files; the verdicts
stay with the orchestrator.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import string
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..substrate import SCRIPTS_DIR, SPOOLCTL_PATH, watchdog
from ..substrate import sessions as sessions_mod
from ..substrate.scheduler import read_json
from ..transport import (
    ADOPTED_SUFFIX,
    CONSUMED_SUFFIX,
    DISPATCH_DIRNAME,
    ORCHESTRATOR_EXIT,
    ORCHESTRATOR_HEARTBEAT,
    READ_SUFFIX,
    job_prefix,
)
from . import reload as reload_mod

TEMPLATE_DIR = Path(__file__).resolve().parent
TURN_INSTRUCTIONS_TEMPLATE = TEMPLATE_DIR / "TURN_INSTRUCTIONS.md"
TURN_PROMPT_TEMPLATE = TEMPLATE_DIR / "turn_prompt.md"

#: Where claims live, under the dispatch directory.
CLAIMS_DIRNAME = "claims"

#: Worker turns a driver runs at once: the orchestrator's default task concurrency,
#: so a full set of tasks is never held back by the driver. Capped: many more
#: concurrent turns on one account stall the API stream.
DEFAULT_MAX_INFLIGHT = 8

#: The planner's turn and the critic's read, as ``(role, what)``; ``None`` is any
#: ``what``. Every other turn waits on what they decide, so they never queue behind
#: workers: the in-flight cap neither holds them back nor counts them. The
#: orchestrator runs at most one of each at a time.
PLANNER_TURNS: tuple[tuple[str, str | None], ...] = (
    ("planner", None),
    ("plan_reviewer", "critic"),
)

#: An orchestrator heartbeat older than this means the orchestrator is gone.
ORCHESTRATOR_STALE_S = 120.0

#: The file the orchestrator holds an exclusive ``flock`` on for its whole life,
#: under every transport (``cli._claim_workspace``).
WORKSPACE_LOCK = ".orchestrator.lock"

#: Appended to the rendered instructions when present in the workspace.
LOCAL_INSTRUCTIONS = "TURN_INSTRUCTIONS.local.md"


class DriverError(RuntimeError):
    """A driver action that cannot be carried out."""


@dataclass
class DriverConfig:
    """What a driver works against, resolved once from the workspace."""

    workspace: Path
    root: Path
    task: Mapping[str, Any] = field(default_factory=dict)
    max_inflight: int = DEFAULT_MAX_INFLIGHT
    #: ``(spool, job name)`` of the orchestrator when it runs as a spool job.
    orchestrator_job: tuple[Path, str] | None = None

    @property
    def dispatch(self) -> Path:
        return self.workspace / DISPATCH_DIRNAME

    @property
    def claims(self) -> Path:
        return self.dispatch / CLAIMS_DIRNAME

    @property
    def skills(self) -> Path:
        return self.workspace / ".claude" / "skills"

    @property
    def sessions(self) -> tuple[sessions_mod.SessionSpec, ...]:
        return sessions_mod.sessions_from_task(self.task)

    @property
    def command(self) -> str:
        return f"trtllm-modeling-v2 driver --workspace {self.workspace}"


def load_config(
    workspace: str | Path,
    *,
    root: str | Path | None = None,
    max_inflight: int = DEFAULT_MAX_INFLIGHT,
    orchestrator_job: str | None = None,
) -> DriverConfig:
    """Resolve a driver's configuration from ``<workspace>/task.yaml``.

    ``orchestrator_job`` is ``<spool>:<job name>`` when the orchestrator runs
    as a spool job; its receipt then also counts as the orchestrator exiting.
    """
    workspace = Path(workspace).resolve()
    if not (workspace / DISPATCH_DIRNAME).is_dir() and not (workspace / "task.yaml").is_file():
        raise DriverError(
            f"{workspace} is not a trtllm_modeling_v2 workspace (no task.yaml and no {DISPATCH_DIRNAME}/)"
        )
    task: Mapping[str, Any] = {}
    try:
        loaded = yaml.safe_load((workspace / "task.yaml").read_text(encoding="utf-8"))
        task = loaded if isinstance(loaded, dict) else {}
    except (OSError, yaml.YAMLError):
        task = {}
    resolved_root = Path(root or task.get("trtllm_repo_path") or Path.cwd()).resolve()
    job = None
    if orchestrator_job:
        spool, sep, name = orchestrator_job.rpartition(":")
        if not sep or not spool or not name:
            raise DriverError("--orchestrator-job must be <spool>:<job name>")
        job = (Path(spool), name)
    return DriverConfig(
        workspace=workspace,
        root=resolved_root,
        task=task,
        max_inflight=max_inflight,
        orchestrator_job=job,
    )


def _resolve(cfg: DriverConfig, value: str | None, cwd: str | None = None) -> str:
    """Request files carry paths as the orchestrator saw them; a driver elsewhere needs them absolute.

    A relative path is relative to the orchestrator's working directory, which
    the request records; a request that records none is resolved by looking
    for the file.
    """
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if cwd:
        return str(Path(cwd) / path)
    for base in (cfg.workspace.parent, cfg.root, Path.cwd()):
        if (base / path).exists():
            return str(base / path)
    return str(cfg.root / path)


# ------------------------------------------------------------------- requests


def _claim_record(cfg: DriverConfig, request_id: str) -> dict | None:
    return read_json(cfg.claims / f"{request_id}.json") or None


def requests(cfg: DriverConfig, now: float | None = None) -> list[dict]:
    """Every request in the dispatch directory, oldest first, with its state.

    Besides the driver's own state (answered, claimed), each says what the
    orchestrator side did with it: ``read`` once an answer was handed back,
    ``adopted`` (the marker's record) once a relaunched process took the turn
    over instead of asking again, ``consumed`` once the run is done with it.
    """
    now = time.time() if now is None else now
    out = []
    for path in sorted(cfg.dispatch.glob("*.request.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        request_id = str(data.get("id") or path.name[: -len(".request.json")])
        data["id"] = request_id
        data["answered"] = (cfg.dispatch / f"{request_id}.done").exists()
        data["read"] = (cfg.dispatch / f"{request_id}{READ_SUFFIX}").exists()
        data["adopted"] = read_json(cfg.dispatch / f"{request_id}{ADOPTED_SUFFIX}") or None
        data["consumed"] = (cfg.dispatch / f"{request_id}{CONSUMED_SUFFIX}").exists()
        data["claim"] = _claim_record(cfg, request_id)
        data["age_s"] = now - float(data.get("requested_at", now))
        data.setdefault("job_prefix", job_prefix(request_id))
        out.append(data)
    return out


def pending(cfg: DriverConfig) -> list[dict]:
    """Unanswered requests nobody has claimed."""
    return [r for r in requests(cfg) if not r["answered"] and not r["claim"]]


def in_flight(cfg: DriverConfig) -> list[dict]:
    """Claimed requests still waiting for their answer."""
    return [r for r in requests(cfg) if not r["answered"] and r["claim"]]


def planner_turn(request: Mapping[str, Any]) -> bool:
    """Whether ``request`` is the planner's turn or the critic's read (:data:`PLANNER_TURNS`)."""
    role, what = request.get("role"), request.get("what")
    return any(role == r and (w is None or what == w) for r, w in PLANNER_TURNS)


def _next_turn(cfg: DriverConfig) -> dict | None:
    """The unclaimed turn to hand out now, or ``None``.

    The planner's or the critic's first, whatever the in-flight cap; else the
    oldest worker's, while fewer than ``max_inflight`` worker turns are in flight.
    """
    open_requests = pending(cfg)
    first = next((r for r in open_requests if planner_turn(r)), None)
    if first is not None:
        return first
    workers = [r for r in open_requests if not planner_turn(r)]
    busy = sum(1 for r in in_flight(cfg) if not planner_turn(r))
    return workers[0] if workers and busy < cfg.max_inflight else None


def _find(cfg: DriverConfig, request_id: str) -> dict:
    for request in requests(cfg):
        if request["id"] == request_id:
            return request
    raise DriverError(f"no request with id {request_id!r} in {cfg.dispatch}")


def claim(cfg: DriverConfig, request_id: str, *, by: str | None = None) -> dict:
    """Take a turn for this driver; refuse one that is answered or already taken."""
    request = _find(cfg, request_id)
    if request["answered"]:
        raise DriverError(f"{request_id} is already answered")
    cfg.claims.mkdir(parents=True, exist_ok=True)
    record = {
        "id": request_id,
        "by": by or f"{socket.gethostname()}:{os.getpid()}",
        "at": time.time(),
        "role": request.get("role"),
    }
    try:
        fd = os.open(cfg.claims / f"{request_id}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = _claim_record(cfg, request_id) or {}
        raise DriverError(
            f"{request_id} is already claimed by {existing.get('by')} at "
            f"{time.strftime('%H:%M:%S', time.localtime(float(existing.get('at', 0))))}"
        ) from None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    return record


def release(cfg: DriverConfig, request_id: str) -> None:
    """Give a claimed turn back, so another driver (or a retry) can take it."""
    (cfg.claims / f"{request_id}.json").unlink(missing_ok=True)


def answer(cfg: DriverConfig, request_id: str, result_text: str) -> None:
    """Hand a turn back for a subagent that could not: the result first, ``.done`` last."""
    request = _find(cfg, request_id)
    result = Path(
        _resolve(cfg, request.get("result_file"), request.get("cwd"))
        or cfg.dispatch / f"{request_id}.result.md"
    )
    done = Path(
        _resolve(cfg, request.get("done_file"), request.get("cwd"))
        or cfg.dispatch / f"{request_id}.done"
    )
    for path, text in ((result, result_text), (done, "1\n")):
        staging = path.with_name(f".{path.name}.tmp")
        staging.write_text(text, encoding="utf-8")
        os.replace(staging, path)


# ------------------------------------------------------------- orchestrator


def orchestrator_state(cfg: DriverConfig, now: float | None = None) -> tuple[str, str]:
    """``("running"|"exited"|"unknown", detail)`` for the orchestrator on this workspace.

    The external transport's heartbeat says it while it speaks for the
    process the run's ledger (``inflight.json``) names. A process with no
    heartbeat (its turns run in process) is read from that ledger and the
    workspace lock instead (:func:`_ledger_state`).
    """
    now = time.time() if now is None else now
    if cfg.orchestrator_job:
        spool, name = cfg.orchestrator_job
        receipt = spool / "done" / f"{name}.rc"
        if receipt.exists():
            record = read_json(spool / "done" / f"{name}.json") or {}
            return "exited", (
                f"spool job {name} finished rc={receipt.read_text(encoding='utf-8').strip()} "
                f"reason={record.get('reason', 'exit')}; log {spool / 'logs' / f'{name}.log'}"
            )
    beat = read_json(cfg.dispatch / ORCHESTRATOR_HEARTBEAT)
    gone = read_json(cfg.dispatch / ORCHESTRATOR_EXIT)
    ledger = reload_mod.read_inflight(cfg.workspace)
    if ledger and not _speaks_for(beat or gone, ledger):
        return _ledger_state(cfg, ledger, now)
    if gone and (not beat or float(gone.get("at", 0)) >= float(beat.get("at", 0))):
        return (
            "exited",
            f"pid {gone.get('pid')} on {gone.get('host')} exited at {_clock(gone.get('at'))}",
        )
    if beat:
        age = now - float(beat.get("at", 0))
        if age > ORCHESTRATOR_STALE_S:
            return "exited", (
                f"pid {beat.get('pid')} on {beat.get('host')} stopped heart-beating {age:.0f}s ago "
                f"(killed, or its allocation ended)"
            )
        return "running", f"pid {beat.get('pid')} on {beat.get('host')}, heartbeat {age:.0f}s ago"
    return (
        "unknown",
        "no orchestrator heartbeat",
    )


def _speaks_for(beat: Mapping[str, Any] | None, ledger: Mapping[str, Any]) -> bool:
    """Whether a heartbeat record is the ledger's process's, or a later one's.

    A process's transport starts after the process does, so its heartbeat's
    ``started`` is never before the ledger's ``started_at``; a heartbeat from
    before it is an earlier process's, left in ``dispatch/``. Either side
    without a start time is taken at its word.
    """
    if not beat:
        return False
    started, since = beat.get("started"), ledger.get("started_at")
    if started is None or since is None:
        return True
    try:
        return float(started) >= float(since)
    except (TypeError, ValueError):
        return True


def _workspace_lock(cfg: DriverConfig) -> str | None:
    """``"held"`` or ``"free"``: whether a process holds the workspace lock; ``None`` if unknown.

    Probed with a shared, non-blocking ``flock`` on a read-only descriptor,
    released at once: it never truncates the file and never waits.
    """
    try:
        fd = os.open(cfg.workspace / WORKSPACE_LOCK, os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return "held"
    except OSError:
        return None
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return "free"
    finally:
        os.close(fd)


def _ledger_state(cfg: DriverConfig, ledger: Mapping[str, Any], now: float) -> tuple[str, str]:
    """The orchestrator's state from its ledger and the workspace lock, without a heartbeat.

    A held lock is a live orchestrator from any host. A free one means none
    only on the ledger's own host: a lock is not always visible across
    hosts. A run that closed says so in the ledger; a re-exec for a reload
    lets the lock go for the moment it takes.
    """
    host = str(ledger.get("host") or "?")
    who = f"pid {ledger.get('pid')} on {host}"
    stopped = ledger.get("stopped") or {}
    how = str(stopped.get("how") or "")
    lock = _workspace_lock(cfg)
    if lock == "held":
        return "running", f"{who} holds the workspace lock (no heartbeat: turns run in process)"
    if how.startswith("closed"):
        verdict = how.partition(":")[2] or "no verdict"
        return "exited", f"{who} closed the run ({verdict}) at {_clock(stopped.get('at'))}"
    if how == "reload" and now - float(stopped.get("at") or 0) < ORCHESTRATOR_STALE_S:
        return "running", f"{who} is re-executing itself for a reload"
    if lock == "free" and host == socket.gethostname():
        return "exited", (
            f"{who} no longer holds the workspace lock: it was stopped, or it crashed, "
            f"or its host went away; its log says which"
        )
    return "unknown", (
        f"{who} writes no heartbeat (turns run in process) and has not closed the run; "
        f"a driver on {host} also sees it stop"
    )


def _clock(epoch: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(epoch)))
    except (TypeError, ValueError):
        return "?"


# --------------------------------------------------------------------- reload


def request_reload(
    cfg: DriverConfig,
    *,
    force: bool = False,
    timeout_s: float = reload_mod.DEFAULT_DRAIN_TIMEOUT_S,
    by: str | None = None,
) -> dict:
    """Ask the orchestrator to drain and re-execute itself; refused when none is running.

    An orchestrator that exited loads the current code when it is launched
    again, and a request left for it would be answered by that start anyway.
    """
    state, detail = orchestrator_state(cfg)
    if state == "exited":
        raise DriverError(
            f"the orchestrator is not running ({detail}); launching it again loads the current code"
        )
    return reload_mod.request(cfg.workspace, force=force, timeout_s=timeout_s, by=by)


def cancel_reload(cfg: DriverConfig) -> dict | None:
    """Withdraw a reload request; a drain under way ends and the run goes on."""
    return reload_mod.cancel(cfg.workspace)


def reload_state(cfg: DriverConfig, now: float | None = None) -> dict | None:
    """The reload under way or the latest one, ``{"phase", "detail", ...}``, or ``None``.

    ``requested`` is a request the orchestrator has not taken up yet; it does
    in its planned loop, within a few seconds.
    """
    now = time.time() if now is None else now
    request = reload_mod.read_request(cfg.workspace)
    record = reload_mod.read_record(cfg.workspace) or {}
    if request is not None and record.get("id") != request.get("id"):
        age = now - float(request.get("requested_at") or now)
        force = ", forced at its deadline" if request.get("force") else ""
        return {
            "phase": "requested",
            "id": request.get("id"),
            "detail": f"requested by {request.get('by')} {age:.0f}s ago{force}; not taken up yet",
        }
    if not record:
        return None
    return {**record, "detail": reload_mod.describe(record, now)}


def _reload_event(cfg: DriverConfig) -> Event | None:
    """A reload outcome ``wait`` has not returned yet: it completed, failed, or is overdue.

    Each is returned once, so the next ``wait`` moves on.
    """
    record = reload_mod.read_record(cfg.workspace)
    if not record:
        return None
    phase = record.get("phase")
    if phase == reload_mod.PHASE_RELOADED:
        kind = "RELOADED"
    elif phase == reload_mod.PHASE_FAILED:
        kind = "RELOAD_FAILED"
    elif phase == reload_mod.PHASE_DRAINING and record.get("overdue"):
        kind = "RELOAD_OVERDUE"
    else:
        return None
    key = f"{record.get('id')}:{kind}"
    if key in reload_mod.seen(cfg.workspace):
        return None
    reload_mod.mark_seen(cfg.workspace, key)
    detail = reload_mod.describe(record)
    if kind == "RELOAD_OVERDUE":
        detail += (
            f"; `{cfg.command} reload --force` re-executes without them, "
            f"`{cfg.command} reload --cancel` withdraws the request"
        )
    return Event(kind, detail)


# -------------------------------------------------------------------- waiting


@dataclass(frozen=True)
class Event:
    """Why :func:`wait` returned."""

    kind: str
    detail: str = ""
    request_id: str | None = None
    #: What the wait did on the way, each said on a line of its own after the event.
    notes: tuple[str, ...] = ()

    def render(self) -> str:
        head = self.kind if self.request_id is None else f"{self.kind} {self.request_id}"
        return "\n".join([f"{head}  {self.detail}".rstrip(), *(f"  note: {n}" for n in self.notes)])


def poll_once(
    cfg: DriverConfig,
    *,
    watch_sessions: bool = True,
    sessions_only: bool = False,
    now: float | None = None,
    notes: list[str] | None = None,
) -> Event | None:
    """The first thing that needs the driver right now, or ``None``.

    ``sessions_only`` is a run whose turns run in process: there is no turn
    to hand out, so it never returns ``NEW``. ``notes`` collects what the
    poll did on the way (a stale successor request archived); the event
    returned carries them.
    """
    notes = [] if notes is None else notes
    event = _first_event(
        cfg, watch_sessions=watch_sessions, sessions_only=sessions_only, now=now, notes=notes
    )
    if event is not None and notes:
        event = replace(event, notes=(*event.notes, *notes))
    return event


def _first_event(
    cfg: DriverConfig, *, watch_sessions: bool, sessions_only: bool, now: float | None, notes: list
) -> Event | None:
    state, detail = orchestrator_state(cfg, now)
    if state == "exited":
        # Answering its turns would wait on nobody. Relaunch it: the new
        # process adopts every unanswered turn rather than asking again, and
        # every answered task turn whose task had no outcome yet.
        if sessions_only:
            return Event("ORCHESTRATOR_EXITED", detail)
        unanswered = sum(1 for r in requests(cfg) if not r["answered"])
        return Event("ORCHESTRATOR_EXITED", f"{detail}; {unanswered} turn(s) unanswered")
    reload_event = _reload_event(cfg)
    if reload_event is not None:
        return reload_event
    if not sessions_only:
        first = _next_turn(cfg)
        if first is not None:
            return Event("NEW", f"{first.get('role')}: {first.get('what')}", first["id"])
    if watch_sessions and cfg.sessions:
        waiting, archived = sessions_mod.review_requests(
            cfg.task, archive=True, by=f"driver {socket.gethostname()}:{os.getpid()}", now=now
        )
        notes.extend(archived)
        for request in waiting:
            return Event(
                "SESSION_REQUEST",
                f"session {request['session']} (allocation {request.get('slurm_job')}) ends "
                f"{request.get('ends_at')}; run `trtllm-modeling-v2 session allocate --task "
                f"{cfg.workspace / 'task.yaml'} --name {request['session']}`",
            )
        alerts = watchdog.check([s.spool for s in cfg.sessions])
        if alerts:
            return Event(f"SESSION_{alerts[0].kind}", f"{alerts[0].spool} {alerts[0].detail}")
    return None


def wait(
    cfg: DriverConfig,
    *,
    timeout_s: float = 540.0,
    poll_s: float = 10.0,
    watch_sessions: bool = True,
    sessions_only: bool = False,
    sleep=time.sleep,
) -> Event:
    """Block until :func:`poll_once` has something, or ``timeout_s`` passes.

    The default timeout keeps one call inside a tool-call limit of ten minutes.
    """
    end = time.monotonic() + timeout_s
    notes: list[str] = []
    while True:
        event = poll_once(
            cfg, watch_sessions=watch_sessions, sessions_only=sessions_only, notes=notes
        )
        if event is not None:
            return event
        remaining = end - time.monotonic()
        if remaining <= 0:
            if sessions_only:
                return Event("TIMEOUT", "nothing new", notes=tuple(notes))
            busy = sum(1 for r in in_flight(cfg) if not planner_turn(r))
            capped = f" (at the in-flight cap, {busy}/{cfg.max_inflight})" if pending(cfg) else ""
            return Event("TIMEOUT", f"nothing new{capped}", notes=tuple(notes))
        sleep(min(poll_s, remaining))


# -------------------------------------------------------------------- render


def _sessions_block(cfg: DriverConfig) -> str:
    sessions = cfg.sessions
    if not sessions:
        spool = (cfg.task.get("environment") or {}).get("spool") or "<spool>"
        return f"| session | for | spool |\n|---|---|---|\n| default | any work | `{spool}` |"
    purpose = {
        "gates": "gates, full-model loads, native rebuilds",
        "work": "catalog tests, microbenchmarks",
        "any": "any work",
    }
    lines = ["| session | for | spool |", "|---|---|---|"]
    lines += [f"| `{s.name}` | {purpose.get(s.role, s.role)} | `{s.spool}` |" for s in sessions]
    return "\n".join(lines)


def _embargo_block(cfg: DriverConfig) -> str:
    entries = [str(e) for e in cfg.task.get("source_embargo") or ()]
    if not entries:
        return "This run embargoes no source. Your role prompt still names what you may consult."
    listing = "\n".join(f"- `{e}`" for e in entries)
    return (
        "`task.yaml` lists files no role may read. The run exists to prove the pipeline "
        "*derives* the target; if you are about to open one of these, stop:\n\n"
        f"{listing}\n\nRunning them is allowed; reading them is not. Your system prompt repeats "
        "this list with the reason."
    )


def _commits_block(cfg: DriverConfig) -> str:
    helper = cfg.workspace / "bin" / "commit.sh"
    return (
        f"Commit only through `{helper}`: it supplies the git identity and the sign-off, stages "
        "only the paths you name, and refuses untracked symlinks and sandbox placeholder files. "
        "Do not write git config. A gate measures the committed tree, so do not leave the "
        "target's own path dirty when you hand back."
    )


def render_instructions(cfg: DriverConfig) -> str:
    """The run's ``TURN_INSTRUCTIONS``, rendered from ``task.yaml``."""
    env = (cfg.task.get("environment") or {}).get("env") or {}
    local = cfg.workspace / LOCAL_INSTRUCTIONS
    extra = (
        f"\n## 6 · This run\n\n{local.read_text(encoding='utf-8').strip()}\n"
        if local.is_file()
        else ""
    )
    template = string.Template(TURN_INSTRUCTIONS_TEMPLATE.read_text(encoding="utf-8"))
    return template.substitute(
        root=cfg.root,
        driver=cfg.command,
        skills=cfg.skills,
        sessions=_sessions_block(cfg),
        scripts=SCRIPTS_DIR,
        spoolctl=f"python3 {SPOOLCTL_PATH}",
        tmpdir=env.get("TMPDIR", "$TMPDIR (keep it under ~60 characters)"),
        embargo=_embargo_block(cfg),
        commits=_commits_block(cfg),
        local=extra,
    )


def install_instructions(cfg: DriverConfig, out: Path | None = None) -> Path:
    target = out or cfg.workspace / "TURN_INSTRUCTIONS.md"
    staging = target.with_name(f".{target.name}.tmp")
    staging.write_text(render_instructions(cfg), encoding="utf-8")
    os.replace(staging, target)
    return target


def render_prompt(cfg: DriverConfig, request_id: str) -> str:
    """The instruction a subagent is handed for one turn."""
    request = _find(cfg, request_id)
    decision = _resolve(cfg, request.get("decision_file"), request.get("cwd"))
    decision_clause = (
        f", including the decision file at\n`{decision}`.\n\n"
        "Two things the orchestrator checks mechanically, so get them exactly right: the decision "
        "file must parse and its `decision:` value must be one the role prompt lists; and any "
        "`vocabulary.yaml` you write has a schema the orchestrator walks and will reject if malformed."
        if decision
        else "."
    )
    prior = request.get("prior_turns") or []
    history = ""
    if prior:
        listing = "\n".join(
            f"   - asked: `{_resolve(cfg, t.get('prompt_file'), request.get('cwd'))}`\n"
            f"     said:  `{_resolve(cfg, t.get('result_file'), request.get('cwd'))}`"
            for t in prior
        )
        history = (
            "\n**You have done this role before in this run, and this transport starts a fresh "
            "session per turn.** Read these and treat them as your own prior reasoning before "
            "changing anything:\n\n" + listing + "\n"
        )
    agent = request.get("agent") or {}
    agent_clause = ""
    if agent.get("backend") or agent.get("model") or agent.get("reasoning_effort"):
        effort = agent.get("reasoning_effort")
        agent_clause = (
            f"\nThe run routes this role to backend `{agent.get('backend') or '?'}`, model "
            f"`{agent.get('model') or '?'}`"
            + (f", reasoning effort `{effort}`" if effort else "")
            + ". Run the turn on that agent if you can; if you cannot, say which you used in "
            "your result.\n"
        )
    timeout = request.get("timeout_s")
    template = string.Template(TURN_PROMPT_TEMPLATE.read_text(encoding="utf-8"))
    return template.substitute(
        root=cfg.root,
        role=request.get("role"),
        what=request.get("what"),
        request_id=request_id,
        agent_clause=agent_clause,
        system_prompt_file=_resolve(cfg, request.get("system_prompt_file"), request.get("cwd")),
        prompt_file=_resolve(cfg, request.get("prompt_file"), request.get("cwd")),
        history_clause=history,
        skills=cfg.skills,
        instructions=cfg.workspace / "TURN_INSTRUCTIONS.md",
        decision_clause=decision_clause,
        job_prefix=request.get("job_prefix") or job_prefix(request_id),
        result_file=_resolve(cfg, request.get("result_file"), request.get("cwd")),
        done_file=_resolve(cfg, request.get("done_file"), request.get("cwd")),
        timeout=f"{float(timeout) / 3600:.1f} h" if timeout else "the role's turn timeout",
    )


# --------------------------------------------------------------------- status


def status(cfg: DriverConfig, now: float | None = None) -> dict:
    """Every turn, the orchestrator and the sessions, as data."""
    now = time.time() if now is None else now
    state, detail = orchestrator_state(cfg, now)
    rows = []
    for request in requests(cfg, now):
        timeout = float(request.get("timeout_s") or 0)
        overdue = bool(timeout) and not request["answered"] and request["age_s"] > timeout
        rows.append(
            {
                "id": request["id"],
                "role": request.get("role"),
                "what": request.get("what"),
                "answered": request["answered"],
                "read": request["read"],
                "adopted": request["adopted"],
                "consumed": request["consumed"],
                "claimed_by": (request["claim"] or {}).get("by"),
                "planner": planner_turn(request),
                "age_s": round(request["age_s"], 1),
                "timeout_s": timeout or None,
                "overdue": overdue,
                "job_prefix": request["job_prefix"],
            }
        )
    ledger = reload_mod.read_inflight(cfg.workspace) or {}
    holds = ledger.get("held") if state != "exited" and not ledger.get("stopped") else []
    return {
        "workspace": str(cfg.workspace),
        "orchestrator": {"state": state, "detail": detail},
        "holds": [
            {"kind": h.get("kind"), "what": h.get("what"), "age_s": round(now - h["since"], 1)}
            for h in holds or ()
            if isinstance(h, dict) and "since" in h
        ],
        "reload": reload_state(cfg, now),
        "correctness": _correctness(cfg.workspace),
        "in_flight": sum(
            1 for r in rows if r["claimed_by"] and not r["answered"] and not r["planner"]
        ),
        "planner_in_flight": sum(
            1 for r in rows if r["claimed_by"] and not r["answered"] and r["planner"]
        ),
        "max_inflight": cfg.max_inflight,
        "requests": rows,
        "sessions": sessions_mod.session_status(cfg.task, now) if cfg.sessions else [],
    }


def _correctness(workspace: Path) -> str | None:
    """Where the run last delivered correctness, or ``None`` before it has."""
    from ...run import contract as contract_mod  # imported here: run imports infra

    milestone = contract_mod.latest_milestone(workspace)
    return milestone.describe() if milestone else None


def format_status(data: Mapping[str, Any], *, turns: bool = True) -> str:
    """:func:`status` as lines; without ``turns``, the dispatched turns are left out.

    In-process turns are what the orchestrator holds (``turn <role> <what>``).
    """
    lines = [
        f"orchestrator: {data['orchestrator']['state']} ({data['orchestrator']['detail']})",
    ]
    holds = data.get("holds") or []
    if holds:
        lines.append(
            "orchestrator holds: "
            + ", ".join(f"{h['kind']} {h['what']} {h['age_s'] / 60:.0f}m" for h in holds)
        )
    reload = data.get("reload")
    if reload:
        lines.append(f"reload: {reload.get('phase')} ({reload.get('detail')})")
    if data.get("correctness"):
        lines.append(f"correctness: {data['correctness']}")
    if turns:
        beside = data.get("planner_in_flight") or 0
        lines.append(
            f"in flight: {data['in_flight']}/{data['max_inflight']}"
            + (f", and {beside} planner or critic turn(s) beside the cap" if beside else "")
        )
        if not data["requests"]:
            lines.append("(no requests)")
    for row in data["requests"] if turns else ():
        if row["answered"]:
            mark = "done"
        elif row["claimed_by"]:
            mark = f"CLAIMED {row['age_s'] / 60:.0f}m"
        else:
            mark = f"WAITING {row['age_s'] / 60:.0f}m"
        if row["overdue"]:
            mark += " OVERDUE"
        # A relaunched process took this turn over instead of asking again.
        if row.get("adopted"):
            mark += " ADOPTED"
        lines.append(f"{row['id']:<20} {mark:<22} {row['role'] or '-':<14} {row['what'] or ''}")
    for session in data["sessions"]:
        beat = (
            "never" if session["heartbeat_age_s"] is None else f"{session['heartbeat_age_s']:.0f}s"
        )
        lines.append(
            f"session {session['name']:<8} {session['role']:<12} heartbeat {beat} "
            f"running={session['running']} pending={session['pending']}"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MAX_INFLIGHT",
    "DriverConfig",
    "DriverError",
    "Event",
    "PLANNER_TURNS",
    "answer",
    "cancel_reload",
    "claim",
    "format_status",
    "in_flight",
    "install_instructions",
    "load_config",
    "orchestrator_state",
    "pending",
    "planner_turn",
    "poll_once",
    "release",
    "reload_state",
    "render_instructions",
    "render_prompt",
    "request_reload",
    "requests",
    "status",
    "wait",
]
