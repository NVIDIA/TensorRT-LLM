# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver's half of the external transport: handing turns out and taking answers back.

With ``--agent-transport external`` the orchestrator writes each role turn to
``<workspace>/dispatch/`` and blocks (:mod:`.transport`); an authenticated
session (the *driver*) hands each turn to a subagent, and the subagent writes
the answer back. This module is everything the driver needs for that, so a run
needs no hand-written shell glue. The rest of the driver's loop (the
orchestrator's exit, reloads, the compute sessions) is the operator loop every
run has (``infra/driver/core.py``), which asks :func:`next_turn` for a ``NEW``
event when its turns are external.

Nothing here runs a turn. It formats hand-offs and reads files; the verdicts
stay with the orchestrator.
"""

from __future__ import annotations

import json
import os
import socket
import string
import time
from pathlib import Path
from typing import Any, Mapping

from ..dispatch import ADOPTED_SUFFIX, CONSUMED_SUFFIX, READ_SUFFIX
from ..driver.config import DriverConfig, DriverError
from ..substrate import SCRIPTS_DIR, SPOOLCTL_PATH
from ..substrate.scheduler import read_json, write_text_atomic
from .transport import job_prefix

TEMPLATE_DIR = Path(__file__).resolve().parent
TURN_INSTRUCTIONS_TEMPLATE = TEMPLATE_DIR / "TURN_INSTRUCTIONS.md"
TURN_PROMPT_TEMPLATE = TEMPLATE_DIR / "turn_prompt.md"

#: Appended to the rendered instructions when present in the workspace.
LOCAL_INSTRUCTIONS = "TURN_INSTRUCTIONS.local.md"


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
    """Whether ``request`` is the planner's turn.

    Every other turn waits on what it decides, so it never queues behind workers:
    the in-flight cap neither holds it back nor counts it. The orchestrator runs
    one at a time.
    """
    return request.get("role") == "planner"


def next_turn(cfg: DriverConfig) -> dict | None:
    """The unclaimed turn to hand out now, or ``None``.

    The planner's first, whatever the in-flight cap; else the
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
    write_text_atomic(result, result_text)
    write_text_atomic(done, "1\n")


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
    write_text_atomic(target, render_instructions(cfg))
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


def turn_rows(cfg: DriverConfig, now: float | None = None) -> list[dict]:
    """Every dispatched turn as a status row: its claim, its age against its timeout, its fate."""
    now = time.time() if now is None else now
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
    return rows


__all__ = [
    "LOCAL_INSTRUCTIONS",
    "TURN_INSTRUCTIONS_TEMPLATE",
    "TURN_PROMPT_TEMPLATE",
    "answer",
    "claim",
    "in_flight",
    "install_instructions",
    "next_turn",
    "pending",
    "planner_turn",
    "release",
    "render_instructions",
    "render_prompt",
    "requests",
    "turn_rows",
]
