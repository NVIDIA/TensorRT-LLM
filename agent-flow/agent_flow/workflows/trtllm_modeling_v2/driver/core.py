# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver's half of the external agent transport.

With ``--agent-transport external`` the orchestrator writes each role turn to
``<workspace>/dispatch/`` and blocks; an authenticated session (the *driver*)
hands each turn to a subagent and the subagent writes the answer back. This
module is everything the driver needs, so a run needs no hand-written shell
glue:

- :func:`wait` blocks until there is something to do: a new unclaimed turn
  (within the in-flight cap), the orchestrator exiting, a session asking for a
  successor allocation, or a session needing attention. A run that closes
  writes no further request, and a waiter that only watched for requests sat
  two hours past the end of one (F041).
- :func:`claim` / :func:`release` record which driver took a turn, so two
  drivers never run one turn twice.
- :func:`render_prompt` produces the exact instruction to hand a subagent:
  every path resolved, the role's backend and model, its spool job prefix,
  its prior turns.
- :func:`render_instructions` renders the run's standing ``TURN_INSTRUCTIONS``
  from ``task.yaml``, with nothing machine-specific written into it.
- :func:`status` shows every turn, its claim, its age against its role's
  timeout, and the orchestrator's liveness.

Nothing here runs a turn. It formats hand-offs and reads files; the verdicts
stay with the orchestrator.
"""

from __future__ import annotations

import json
import os
import socket
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from agent_flow.runtime.turns import (
    DISPATCH_DIRNAME,
    ORCHESTRATOR_EXIT,
    ORCHESTRATOR_HEARTBEAT,
    job_prefix,
)

from ..substrate import SCRIPTS_DIR, watchdog
from ..substrate import sessions as sessions_mod
from ..substrate.scheduler import read_json

TEMPLATE_DIR = Path(__file__).resolve().parent
TURN_INSTRUCTIONS_TEMPLATE = TEMPLATE_DIR / "TURN_INSTRUCTIONS.md"
TURN_PROMPT_TEMPLATE = TEMPLATE_DIR / "turn_prompt.md"

#: Where claims live, under the dispatch directory.
CLAIMS_DIRNAME = "claims"

#: The pre-packaged driver kept its claims beside the dispatch directory.
LEGACY_CLAIMS_DIRNAME = ".driver-claims"

#: Turns a driver runs at once. Twelve concurrent turns lost seven to API
#: stream stalls within twelve minutes on one account (F010).
DEFAULT_MAX_INFLIGHT = 8

#: An orchestrator heartbeat older than this means the orchestrator is gone.
ORCHESTRATOR_STALE_S = 120.0

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
    the request records; only a request written before it did is resolved by
    looking for the file.
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
    record = read_json(cfg.claims / f"{request_id}.json")
    if record:
        return record
    legacy = cfg.workspace / LEGACY_CLAIMS_DIRNAME / request_id
    if legacy.exists():
        return {"by": "legacy driver", "at": legacy.stat().st_mtime}
    return None


def requests(cfg: DriverConfig, now: float | None = None) -> list[dict]:
    """Every request in the dispatch directory, oldest first, with its state."""
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
    """``("running"|"exited"|"unknown", detail)`` for the orchestrator on this workspace."""
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
        "no orchestrator heartbeat (an orchestrator older than this driver writes none)",
    )


def _clock(epoch: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(epoch)))
    except (TypeError, ValueError):
        return "?"


# -------------------------------------------------------------------- waiting


@dataclass(frozen=True)
class Event:
    """Why :func:`wait` returned."""

    kind: str
    detail: str = ""
    request_id: str | None = None

    def render(self) -> str:
        head = self.kind if self.request_id is None else f"{self.kind} {self.request_id}"
        return f"{head}  {self.detail}".rstrip()


def poll_once(
    cfg: DriverConfig, *, watch_sessions: bool = True, now: float | None = None
) -> Event | None:
    """The first thing that needs the driver right now, or ``None``."""
    state, detail = orchestrator_state(cfg, now)
    open_requests = pending(cfg)
    if state == "exited":
        # Answering its turns would wait on nobody. Relaunch it: the new
        # process adopts every unanswered turn rather than asking again.
        unanswered = sum(1 for r in requests(cfg) if not r["answered"])
        return Event("ORCHESTRATOR_EXITED", f"{detail}; {unanswered} turn(s) unanswered")
    busy = len(in_flight(cfg))
    if open_requests and busy < cfg.max_inflight:
        first = open_requests[0]
        return Event("NEW", f"{first.get('role')}: {first.get('what')}", first["id"])
    if watch_sessions and cfg.sessions:
        for request in sessions_mod.pending_requests(cfg.task):
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
    sleep=time.sleep,
) -> Event:
    """Block until :func:`poll_once` has something, or ``timeout_s`` passes.

    The default timeout keeps one call inside a tool-call limit of ten minutes.
    """
    end = time.monotonic() + timeout_s
    while True:
        event = poll_once(cfg, watch_sessions=watch_sessions)
        if event is not None:
            return event
        remaining = end - time.monotonic()
        if remaining <= 0:
            busy = len(in_flight(cfg))
            capped = f" (at the in-flight cap, {busy}/{cfg.max_inflight})" if pending(cfg) else ""
            return Event("TIMEOUT", f"nothing new{capped}")
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
        "orchestrator": "the orchestrator only (no GPUs)",
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
            "session per turn.** Your role prompt may claim you are in the same session as your "
            "earlier turn; you are not. Read these and treat them as your own prior reasoning "
            "before changing anything:\n\n" + listing + "\n"
        )
    agent = request.get("agent") or {}
    agent_clause = ""
    if agent.get("backend") or agent.get("model"):
        agent_clause = (
            f"\nThe run configured this role for backend `{agent.get('backend') or '?'}`, model "
            f"`{agent.get('model') or '?'}`. Run the turn on that model if you can; if you cannot, "
            "say which model you used in your result, because a cross-model review run on the "
            "author's model reviews nothing.\n"
        )
    embargo = cfg.task.get("source_embargo") or ()
    embargo_clause = (
        "\n**Honour the source embargo** your system prompt states ("
        + ", ".join(f"`{e}`" for e in embargo)
        + "). Reading one of those files is the failure this run exists to detect.\n"
        if embargo
        else ""
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
        embargo_clause=embargo_clause,
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
                "claimed_by": (request["claim"] or {}).get("by"),
                "age_s": round(request["age_s"], 1),
                "timeout_s": timeout or None,
                "overdue": overdue,
                "job_prefix": request["job_prefix"],
            }
        )
    return {
        "workspace": str(cfg.workspace),
        "orchestrator": {"state": state, "detail": detail},
        "in_flight": sum(1 for r in rows if r["claimed_by"] and not r["answered"]),
        "max_inflight": cfg.max_inflight,
        "requests": rows,
        "sessions": sessions_mod.session_status(cfg.task, now) if cfg.sessions else [],
    }


def format_status(data: Mapping[str, Any]) -> str:
    lines = [
        f"orchestrator: {data['orchestrator']['state']} ({data['orchestrator']['detail']})",
        f"in flight: {data['in_flight']}/{data['max_inflight']}",
    ]
    if not data["requests"]:
        lines.append("(no requests)")
    for row in data["requests"]:
        if row["answered"]:
            mark = "done"
        elif row["claimed_by"]:
            mark = f"CLAIMED {row['age_s'] / 60:.0f}m"
        else:
            mark = f"WAITING {row['age_s'] / 60:.0f}m"
        if row["overdue"]:
            mark += " OVERDUE"
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
    "answer",
    "claim",
    "format_status",
    "in_flight",
    "install_instructions",
    "load_config",
    "orchestrator_state",
    "pending",
    "poll_once",
    "release",
    "render_instructions",
    "render_prompt",
    "requests",
    "status",
    "wait",
]
