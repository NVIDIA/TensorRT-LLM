"""Append-only, role-addressed notice queue.

This replaces a single notice file, where a second notice overwrote a first
one nobody had read yet. One JSON object per line, four record types::

    {"type": "notice", "id": "n7", "ts": ..., "blocking": true, "to": ["coder"], "message": "..."}
    {"type": "ack", "id": "n7", "ts": ..., "role": "coder", "text": "what I did about it"}
    {"type": "followup", "id": "n7", "ts": ..., "text": "the promised result"}
    {"type": "report", "id": "r8", "ts": ..., "text": "agent-originated status"}

Append-only means no read-modify-write, so a human writing a notice and an
agent acknowledging one cannot clobber each other. "Pending" is derived: a
notice is pending for a role until that role has acknowledged it. ``to`` is the
explicit list of addressee roles; a record without one is read as addressed to
every role, and an ack without a role settles the notice for everybody (that
was the behaviour before addressing existed, and old records must keep
meaning what they meant).

Ids are unique across every record type, minted under the lock. Counting only
notices was a real bug: an agent-minted ack id posted before the matching
notice existed pre-acknowledged the notice that later received that id, and
the real ack was dropped as "nothing pending".
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from agent_flow.ops.config import OpsConfig

QUEUE_NAME = "AGENT-NOTICES.jsonl"

_ROLES: tuple[str, ...] = ("coder", "reviewer")
_QUEUE: Path | None = None
_ROLE_CHECKOUTS: dict[str, Path] = {}
_ACK_CMD = "python -m agent_flow.ops.ack_notice"


def configure(cfg: OpsConfig, queue: Path | None = None) -> None:
    """Point the module at a run's queue and role set."""
    global _QUEUE, _ROLES, _ROLE_CHECKOUTS, _ACK_CMD
    _QUEUE = Path(queue) if queue else cfg.run_root / QUEUE_NAME
    _ROLES = tuple(cfg.roles)
    _ROLE_CHECKOUTS = {r: Path(p).resolve() for r, p in cfg.role_checkouts.items()}
    _ACK_CMD = str(cfg.get("notices", "ack_command", default=_ACK_CMD))


def set_roles(names) -> None:
    """Widen the addressable name set (see :mod:`agent_flow.ops.mailbox`).

    Roles are the built-in mailboxes; a run that registers extra mailboxes
    must have them recognised here too, or ``addressees()`` would silently
    drop a name and the notice would read as addressed to everyone.
    """
    global _ROLES
    _ROLES = tuple(dict.fromkeys(str(n) for n in names)) or _ROLES


def set_queue(path) -> None:
    """Repoint the queue only (archive replay, a second run directory)."""
    global _QUEUE
    _QUEUE = Path(path)


def queue_path() -> Path:
    if _QUEUE is None:
        raise RuntimeError("notices.configure(cfg) has not been called")
    return _QUEUE


def roles() -> tuple[str, ...]:
    return _ROLES


def ack_command() -> str:
    return _ACK_CMD


def _read() -> list[dict]:
    path = queue_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@contextmanager
def _locked():
    """Exclusive advisory lock over the queue, held for a read+append only.

    Append-only already rules out clobbering; the lock exists because id
    assignment is read-then-append, and two writers arriving together would
    both count N records and both mint the same id.
    """
    path = queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _append(rec: dict) -> None:
    path = queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _num(rec: dict) -> int:
    m = re.match(r"^[a-z](\d+)$", str(rec.get("id", "")))
    return int(m.group(1)) if m else 0


def _next_num(recs: list[dict]) -> int:
    return max([_num(r) for r in recs] + [0]) + 1


def addressees(rec: dict) -> tuple[str, ...]:
    """Roles a notice is addressed to, whatever era the record is from."""
    to = rec.get("to")
    if not to or to == "all":
        return _ROLES
    if isinstance(to, str):
        return (to,)
    return tuple(r for r in to if r in _ROLES) or _ROLES


def parse_to(to) -> list[str]:
    """``'coder'`` / ``'all'`` / ``'coder,reviewer'`` / a list -> role list."""
    if to is None or to == "all" or to == ["all"]:
        return list(_ROLES)
    parts = [p.strip() for p in (to.split(",") if isinstance(to, str) else to)]
    bad = [p for p in parts if p not in _ROLES]
    if bad or not parts:
        raise ValueError(f"addressees must be from {_ROLES} (or 'all'), got {to!r}")
    return [r for r in _ROLES if r in parts]


def owed(rec: dict, recs: list[dict] | None = None) -> list[str]:
    """Addressee roles that have not yet acknowledged ``rec``."""
    recs = _read() if recs is None else recs
    acks_for = {
        r.get("role") or "unknown"
        for r in recs
        if r.get("type") == "ack"
        and r.get("id") == rec.get("id")
        and r.get("ts", 0) >= rec.get("ts", 0)
    }
    if "unknown" in acks_for or ("to" not in rec and acks_for):
        return []
    return [a for a in addressees(rec) if a not in acks_for]


def _covered(recs: list[dict], role: str | None = None) -> set[str]:
    """Notice ids acknowledged by an ack that POSTDATES the notice.

    For ``role``, or by every addressee when ``role`` is None.
    """
    notices_ = {r["id"]: r for r in recs if r.get("type") == "notice"}
    acks_by: dict[str, set[str]] = {}
    for r in recs:
        if r.get("type") == "ack" and r.get("ts", 0) >= notices_.get(r.get("id"), {}).get(
            "ts", float("inf")
        ):
            acks_by.setdefault(r["id"], set()).add(r.get("role") or "unknown")
    out = set()
    for nid, n in notices_.items():
        got = acks_by.get(nid, set())
        if "to" not in n and got:
            out.add(nid)
            continue
        need = addressees(n) if role is None else ((role,) if role in addressees(n) else ())
        if all(("unknown" in got) or (a in got) for a in need):
            out.add(nid)
    return out


def local(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def add(message: str, blocking: bool = False, to="all") -> dict:
    with _locked():
        recs = _read()
        rec = {
            "type": "notice",
            "id": f"n{_next_num(recs)}",
            "ts": time.time(),
            "blocking": bool(blocking),
            "to": parse_to(to),
            "message": message.strip(),
        }
        _append(rec)
    return rec


def infer_role() -> str:
    """Which role is running this process.

    ``AGENT_NOTICE_ROLE`` wins. Otherwise the cwd is matched against the
    configured ``[roles].checkouts`` mapping (longest path first, so a
    worktree nested under a checkout still resolves to its own role). No
    fragment of any path is hard-coded here: a run that names its roles
    differently only edits the config.
    """
    r = os.environ.get("AGENT_NOTICE_ROLE", "").strip().lower()
    if r in _ROLES:
        return r
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return "unknown"
    best, best_len = "unknown", -1
    for role, path in _ROLE_CHECKOUTS.items():
        s = str(path)
        if (cwd == path or str(cwd).startswith(s + os.sep)) and len(s) > best_len:
            best, best_len = role, len(s)
    return best


def ack(
    text: str,
    notice_id: str | None = None,
    followup: bool = False,
    role: str | None = None,
) -> list[str]:
    """Acknowledge one notice, or every one pending for this role.

    ``followup=True`` means "received and acting, but the real answer depends
    on work still running". Ack FIRST so the sender knows the message landed,
    then post the result with :func:`followup` when it exists.
    """
    with _locked():
        recs = _read()
        known = {r["id"] for r in recs if r.get("type") == "notice"}
        if notice_id and notice_id not in known:
            # Agent-originated message answering no notice: record it as a
            # report with its own id so it can never pre-ack a future notice.
            return [_report(recs, text, claimed_id=notice_id)]
        role = role or infer_role()
        ids = (
            [notice_id]
            if notice_id
            else [r["id"] for r in pending(role if role in _ROLES else None)]
        )
        if not ids:
            return [_report(recs, text, role=role)]
        for i in ids:
            rec = {"type": "ack", "id": i, "ts": time.time(), "role": role, "text": text.strip()}
            if followup:
                rec["followup"] = True
            _append(rec)
    return ids


def _report(recs: list[dict], text: str, **extra) -> str:
    rid = f"r{_next_num(recs)}"
    _append({"type": "report", "id": rid, "ts": time.time(), "text": text.strip(), **extra})
    return rid


def report(text: str) -> str:
    """Agent-originated status message that answers no notice."""
    with _locked():
        return _report(_read(), text)


def reports() -> list[dict]:
    return [r for r in _read() if r.get("type") == "report"]


def followup(text: str, notice_id: str) -> str:
    """Post the promised follow-up for a notice acked with ``followup=True``."""
    with _locked():
        _append({"type": "followup", "id": notice_id, "ts": time.time(), "text": text.strip()})
    return notice_id


def awaiting_followup() -> list[dict]:
    """Notices whose latest ack promised a follow-up that has not arrived."""
    recs = _read()
    by_id = {r["id"]: r for r in recs if r.get("type") == "notice"}
    done = {
        r.get("id")
        for r in recs
        if r.get("type") == "followup"
        and r.get("ts", 0) >= by_id.get(r.get("id"), {}).get("ts", float("inf"))
    }
    promised = {
        r["id"]: r
        for r in recs
        if r.get("type") == "ack"
        and r.get("followup")
        and r.get("ts", 0) >= by_id.get(r["id"], {}).get("ts", float("inf"))
    }
    return [
        dict(by_id[i], ack_ts=a["ts"], ack_text=a.get("text", ""))
        for i, a in promised.items()
        if i not in done and i in by_id
    ]


def pending(role: str | None = None) -> list[dict]:
    """Notices still owed an ack, by any addressee (``role=None``) or by ``role``.

    Each record carries ``owed``: the roles that still have to ack.
    """
    recs = _read()
    covered = _covered(recs, role)
    return [
        dict(r, owed=owed(r, recs))
        for r in recs
        if r.get("type") == "notice"
        and r.get("id") not in covered
        and (role is None or role in addressees(r))
    ]


def acks() -> list[dict]:
    return [r for r in _read() if r.get("type") == "ack"]


def blocking_pending() -> list[dict]:
    return [r for r in pending() if r.get("blocking")]


def render(recs: list[dict]) -> str:
    """Human/agent readable block for a set of notices."""
    out = []
    for r in recs:
        tag = "BLOCKING " if r.get("blocking") else ""
        out.append(f"--- {tag}notice {r['id']} ({local(r['ts'])}) ---")
        out.append(r.get("message", ""))
    if out:
        out.append(
            f'Acknowledge NOW with: {_ACK_CMD} "<what you did or will do>" '
            f"[--id <notice id>]; add --later if the real answer depends on running "
            f'work, then post it with {_ACK_CMD} --followup "<result>" --id <notice id> '
            f"when it lands."
        )
    return "\n".join(out)
