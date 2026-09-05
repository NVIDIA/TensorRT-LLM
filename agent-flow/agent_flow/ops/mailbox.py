"""Named mailboxes over the append-only notice queue.

A notice used to be addressed to a *role*, and the role list was fixed by the
config. That is too narrow once more than two participants exist: a watchdog, a
human on call, a second project's coordinator all need an address, and none of
them is a role in the workflow. A **mailbox** is just a name any participant
can register; roles are mailboxes that happen to be declared in ``[roles]``.

    python -m agent_flow.ops.mailbox register oncall --kind human
    python -m agent_flow.ops.mailbox send --to coder,reviewer "switch to X"
    python -m agent_flow.ops.mailbox recv --as coder
    python -m agent_flow.ops.mailbox ack --as coder --id n7 "switched"
    python -m agent_flow.ops.mailbox nag        # re-deliver what is overdue
    python -m agent_flow.ops.mailbox fsck       # audit the queue

The queue is the same append-only JSONL file :mod:`agent_flow.ops.notices`
writes, with the same record types and the same id space, so old records keep
meaning what they meant and the two views can be used side by side.

Reliability, and why each rule exists:

* **Delivery is best-effort; the queue is the record.** Per-mailbox delivery
  hooks (prepend to the command cache, append to the live notes, write a file a
  harness hook injects into the next tool result) push a message at a running
  agent. Any of them can fail — a file the agent never opens, a harness without
  the hook — so a failing hook is reported, never raised, and never changes
  whether the message is pending.
* **Sends are idempotent under a client key.** A retrying caller that cannot
  tell whether its first send landed passes ``--key``; a second send with a key
  already in the queue returns the original record instead of a duplicate.
* **Ids are minted under the lock,** across every record type, so an ack can
  never be assigned an id a later message will take.
* **Acks are recorded by mailbox name.** Inferring the sender from the cwd is
  kept only as a fallback, and the record says which it was (``ack_source``),
  because a wrong inference silently settles someone else's message.
* **``fsck`` audits what cannot be fixed silently:** acks for a message that
  does not exist, acks stamped before the message they answer, and messages
  addressed to nobody.
* **``nag`` re-delivers what is overdue** — a message past its due time with an
  addressee that has not acked, or a promised follow-up that never arrived —
  and escalates: re-deliver, then mark it OVERDUE for the dashboard, then post
  to the human mailbox.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agent_flow.ops import notices
from agent_flow.ops.config import OpsConfig, add_config_argument, config_from_args

REGISTRY_NAME = "MAILBOXES.json"
DEFAULT_KIND = "agent"
KINDS = ("agent", "human", "service")
#: Escalation ladder used by :func:`nag`, one step per overdue round.
ESCALATION = ("redeliver", "mark-overdue", "notify-human")


@dataclass(frozen=True)
class Mailbox:
    """One addressable participant."""

    name: str
    kind: str = DEFAULT_KIND
    description: str = ""
    delivery: tuple[str, ...] = ()
    checkout: Path | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "delivery": list(self.delivery),
            "checkout": str(self.checkout) if self.checkout else "",
        }


@dataclass
class _State:
    cfg: OpsConfig | None = None
    boxes: dict[str, Mailbox] = field(default_factory=dict)


_state = _State()


# --------------------------------------------------------------------------
# delivery hooks
# --------------------------------------------------------------------------
#: hook name -> ``fn(mailbox, record, cfg) -> str`` describing what it did.
HOOKS: dict[str, Callable[[Mailbox, dict, OpsConfig], str]] = {}


def register_hook(name: str, fn: Callable[[Mailbox, dict, OpsConfig], str]) -> None:
    """Add a delivery hook a mailbox can name in its ``delivery`` list."""
    HOOKS[name] = fn


def _banner(rec: dict) -> str:
    from agent_flow.ops.notify_agent import BEGIN, END

    when = notices.local(rec["ts"])
    to = ", ".join(rec.get("to") or ["all"])
    tag = "BLOCKING " if rec.get("blocking") else ""
    title = rec.get("title") or "MESSAGE"
    body = "\n".join(f"> {ln}" for ln in str(rec.get("message", "")).splitlines())
    return (
        f"{BEGIN}\n> **{tag}{title} {rec['id']} ({when}) — for {to}** — newer than your "
        f"prompt and overrides it where they conflict. Acknowledge with "
        f'`{notices.ack_command()} --id {rec["id"]} "<what you did>"`.\n>\n{body}\n{END}\n'
    )


def _hook_command_cache(box: Mailbox, rec: dict, cfg: OpsConfig) -> str:
    from agent_flow.ops.notify_agent import cache_path, strip_block

    path = cache_path(cfg)
    if not path.exists():
        return f"skipped: no {path}"
    path.write_text(_banner(rec) + "\n" + strip_block(path.read_text()))
    return str(path)


#: Hooks that write one project-wide file. Running them once per addressee
#: would mirror the same message two or three times into the same file, so
#: :func:`deliver` runs each of these once and attributes it to every
#: addressee.
PROJECT_SCOPED_HOOKS = {"command_cache", "live_notes", "tool_result"}


def _hook_live_notes(box: Mailbox, rec: dict, cfg: OpsConfig) -> str:
    from agent_flow.ops.notify_agent import live_notes_path

    path = live_notes_path(cfg)
    if not path.exists():
        return f"skipped: no {path}"
    with path.open("a") as fh:
        to = ", ".join(rec.get("to") or [box.name])
        fh.write(f"\n## {notices.local(rec['ts'])} — message {rec['id']} for {to}\n")
        fh.write(str(rec.get("message", "")).strip() + "\n")
    return str(path)


def _hook_tool_result(box: Mailbox, rec: dict, cfg: OpsConfig) -> str:
    """Drop a file a harness PostToolUse hook injects into the next tool result.

    Optional by design: without such a hook installed the file is simply never
    read, which is why nothing here treats its absence as an error.
    """
    name = str(cfg.get("mailboxes", "tool_result_file", default=".pending-messages.md"))
    path = cfg.workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(_banner(rec))
    return str(path)


register_hook("command_cache", _hook_command_cache)
register_hook("live_notes", _hook_live_notes)
register_hook("tool_result", _hook_tool_result)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def registry_path(cfg: OpsConfig | None = None) -> Path:
    cfg = cfg or _cfg()
    return cfg.project_root / REGISTRY_NAME


def _read_registry(cfg: OpsConfig) -> dict:
    try:
        data = json.loads(registry_path(cfg).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _box_from(name: str, body: dict, default_delivery: tuple[str, ...]) -> Mailbox:
    checkout = body.get("checkout") or ""
    return Mailbox(
        name=name,
        kind=str(body.get("kind", DEFAULT_KIND)),
        description=str(body.get("description", "")),
        delivery=tuple(body.get("delivery", default_delivery) or ()),
        checkout=Path(checkout).expanduser() if checkout else None,
    )


def configure(cfg: OpsConfig) -> dict[str, Mailbox]:
    """Bind to a project and build its mailbox set.

    Sources, later winning: the workflow roles, ``[mailboxes.<name>]`` in the
    config, then the runtime registry file (so a participant can register
    itself without editing anyone's config).
    """
    notices.configure(cfg)
    default_delivery = tuple(cfg.get("mailboxes", "default_delivery", default=[]) or ())
    boxes: dict[str, Mailbox] = {}
    checkouts = cfg.role_checkouts
    for role in cfg.roles:
        boxes[role] = Mailbox(
            name=role, kind=DEFAULT_KIND, delivery=default_delivery, checkout=checkouts.get(role)
        )
    for name, body in (cfg.raw.get("mailboxes") or {}).items():
        if isinstance(body, dict):
            boxes[name] = _box_from(name, body, default_delivery)
    for name, body in _read_registry(cfg).items():
        if isinstance(body, dict):
            boxes[name] = _box_from(name, body, default_delivery)
    _state.cfg = cfg
    _state.boxes = boxes
    notices.set_roles(boxes)
    return boxes


def _cfg() -> OpsConfig:
    if _state.cfg is None:
        raise RuntimeError("mailbox.configure(cfg) has not been called")
    return _state.cfg


def mailboxes() -> dict[str, Mailbox]:
    return dict(_state.boxes)


def register(
    name: str,
    kind: str = DEFAULT_KIND,
    delivery=(),
    description: str = "",
    checkout=None,
) -> Mailbox:
    """Add or update a mailbox in the runtime registry."""
    if not name or any(c in name for c in " ,"):
        raise ValueError(f"mailbox name must be a single word, got {name!r}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    cfg = _cfg()
    box = Mailbox(
        name=name,
        kind=kind,
        delivery=tuple(delivery),
        description=description,
        checkout=Path(checkout).expanduser() if checkout else None,
    )
    path = registry_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_registry(cfg)
    data[name] = box.as_dict()
    path.write_text(json.dumps(data, indent=1, sort_keys=True))
    _state.boxes[name] = box
    notices.set_roles(_state.boxes)
    return box


def resolve_to(to) -> list[str]:
    """``'a'`` / ``'all'`` / ``'a,b'`` / a list -> mailbox names, ordered."""
    known = list(_state.boxes)
    if to is None or to == "all" or to == ["all"]:
        return known
    parts = [p.strip() for p in (to.split(",") if isinstance(to, str) else to) if str(p).strip()]
    unknown = [p for p in parts if p not in _state.boxes]
    if unknown or not parts:
        raise ValueError(
            f"unknown mailbox(es) {unknown or to!r}; registered: {', '.join(known) or 'none'}"
        )
    return [n for n in known if n in parts]


# --------------------------------------------------------------------------
# sending and receiving
# --------------------------------------------------------------------------
def _default_due_minutes() -> float:
    return float(_cfg().get("mailboxes", "default_due_minutes", default=0) or 0)


def deliver(rec: dict, hooks=None) -> list[dict]:
    """Run each addressee's delivery hooks. Never raises; reports per hook.

    ``hooks`` overrides the per-mailbox lists, for a sender that knows the
    message must be mirrored whatever the addressee normally asked for.
    """
    cfg = _cfg()
    out = []
    done: set[str] = set()
    for name in rec.get("to") or []:
        box = _state.boxes.get(name)
        if box is None:
            continue
        for hook in box.delivery if hooks is None else tuple(hooks):
            if hook in done:
                out.append({"to": name, "hook": hook, "ok": True, "detail": "already delivered"})
                continue
            if hook in PROJECT_SCOPED_HOOKS:
                done.add(hook)
            fn = HOOKS.get(hook)
            if fn is None:
                out.append({"to": name, "hook": hook, "ok": False, "detail": "no such hook"})
                continue
            try:
                detail = fn(box, rec, cfg)
                out.append({"to": name, "hook": hook, "ok": True, "detail": detail})
            except OSError as exc:  # best effort: the queue is the record
                out.append({"to": name, "hook": hook, "ok": False, "detail": str(exc)})
    return out


def find_by_key(key: str) -> dict | None:
    """The message a previous send with this client dedupe key produced."""
    if not key:
        return None
    return next(
        (r for r in notices._read() if r.get("type") == "notice" and r.get("key") == key),
        None,
    )


def send(
    message: str,
    to="all",
    blocking: bool = False,
    sender: str = "",
    key: str = "",
    due_minutes: float | None = None,
    title: str = "",
    hooks=None,
) -> tuple[dict, bool, list[dict]]:
    """Post a message. Returns ``(record, was_duplicate, delivery report)``.

    ``key`` makes the send idempotent: a retry with a key already in the queue
    returns the original record and delivers nothing again.
    """
    to_list = resolve_to(to)
    minutes = _default_due_minutes() if due_minutes is None else float(due_minutes)
    with notices._locked():
        recs = notices._read()
        if key:
            dup = next((r for r in recs if r.get("type") == "notice" and r.get("key") == key), None)
            if dup is not None:
                return dup, True, []
        now = time.time()
        rec = {
            "type": "notice",
            "id": f"n{notices._next_num(recs)}",
            "ts": now,
            "blocking": bool(blocking),
            "to": to_list,
            "message": message.strip(),
        }
        if sender:
            rec["from"] = sender
        if title:
            rec["title"] = title
        if key:
            rec["key"] = key
        if minutes > 0:
            rec["due"] = now + minutes * 60
        notices._append(rec)
    return rec, False, deliver(rec, hooks)


def whoami(explicit: str | None = None) -> tuple[str, str]:
    """``(mailbox name, how it was decided)``.

    Explicit beats the environment beats the cwd. The source travels with the
    ack because a cwd guess that settles the wrong mailbox's message is the
    failure mode worth being able to see afterwards.
    """
    if explicit:
        return explicit, "explicit"
    env = os.environ.get("AGENT_NOTICE_ROLE", "").strip().lower()
    if env in _state.boxes:
        return env, "env"
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return "unknown", "unknown"
    best, best_len = "unknown", -1
    for box in _state.boxes.values():
        if box.checkout is None:
            continue
        s = str(Path(box.checkout).resolve())
        if (str(cwd) == s or str(cwd).startswith(s + os.sep)) and len(s) > best_len:
            best, best_len = box.name, len(s)
    return best, ("cwd" if best != "unknown" else "unknown")


def pending(name: str | None = None) -> list[dict]:
    """Messages still owed an ack, for one mailbox or for any."""
    return notices.pending(name)


def recv(name: str, mark: bool = False) -> list[dict]:
    """Messages pending for ``name``; ``mark`` re-runs its delivery hooks."""
    if name not in _state.boxes:
        raise ValueError(f"unknown mailbox {name!r}")
    out = pending(name)
    if mark:
        for rec in out:
            deliver(rec)
    return out


def ack(
    text: str,
    message_id: str | None = None,
    as_: str | None = None,
    later: bool = False,
) -> tuple[list[str], str, str]:
    """Acknowledge as a mailbox. Returns ``(ids, mailbox, how it was decided)``."""
    name, source = whoami(as_)
    with notices._locked():
        recs = notices._read()
        known = {r["id"] for r in recs if r.get("type") == "notice"}
        if message_id and message_id not in known:
            # An ack for an id that does not exist must never pre-settle the
            # message that will later be given that id.
            return [notices._report(recs, text, claimed_id=message_id)], name, source
        ids = (
            [message_id]
            if message_id
            else [r["id"] for r in notices.pending(name if name in _state.boxes else None)]
        )
        if not ids:
            return [notices._report(recs, text, role=name)], name, source
        for i in ids:
            rec = {
                "type": "ack",
                "id": i,
                "ts": time.time(),
                "role": name,
                "mailbox": name,
                "ack_source": source,
                "text": text.strip(),
            }
            if later:
                rec["followup"] = True
            notices._append(rec)
    return ids, name, source


# --------------------------------------------------------------------------
# audit, overdue, nagging
# --------------------------------------------------------------------------
def fsck() -> dict:
    """Queue problems no writer can fix on its own.

    Each list is a self-describing row so a caller can print it verbatim.
    """
    recs = notices._read()
    msgs = {r["id"]: r for r in recs if r.get("type") == "notice"}
    orphan_acks, early_acks, unaddressed, unknown_addressees = [], [], [], []
    for r in recs:
        if r.get("type") not in ("ack", "followup"):
            continue
        msg = msgs.get(r.get("id"))
        if msg is None:
            orphan_acks.append(
                {"id": r.get("id"), "type": r["type"], "ts": r.get("ts"), "role": r.get("role")}
            )
        elif r.get("ts", 0) < msg.get("ts", 0):
            early_acks.append(
                {
                    "id": r.get("id"),
                    "ack_ts": r.get("ts"),
                    "message_ts": msg.get("ts"),
                    "role": r.get("role"),
                }
            )
    for mid, m in msgs.items():
        to = m.get("to")
        if to is None:
            continue  # pre-addressing record: means "everyone", not "nobody"
        missing = [t for t in to if t not in _state.boxes]
        if not to:
            unaddressed.append({"id": mid, "ts": m.get("ts")})
        elif missing:
            unknown_addressees.append({"id": mid, "unknown": missing})
    return {
        "orphan_acks": orphan_acks,
        "early_acks": early_acks,
        "unaddressed": unaddressed,
        "unknown_addressees": unknown_addressees,
        "ok": not (orphan_acks or early_acks or unaddressed or unknown_addressees),
        "messages": len(msgs),
        "records": len(recs),
    }


def _nag_levels(recs: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in recs:
        if r.get("type") == "nag":
            out[r.get("id")] = max(out.get(r.get("id"), 0), int(r.get("level", 0)))
    return out


def overdue(now: float | None = None) -> dict:
    """Messages past due and unacked, plus follow-ups promised and not posted."""
    now = time.time() if now is None else now
    recs = notices._read()
    levels = _nag_levels(recs)
    late_messages = []
    for rec in notices.pending():
        due = rec.get("due")
        if due and due <= now:
            late_messages.append(
                dict(rec, late_seconds=round(now - due), nag_level=levels.get(rec["id"], 0))
            )
    grace = _default_due_minutes() * 60
    late_followups = []
    for rec in notices.awaiting_followup():
        due = rec.get("ack_ts", 0) + grace
        if grace > 0 and due <= now:
            late_followups.append(
                dict(rec, late_seconds=round(now - due), nag_level=levels.get(rec["id"], 0))
            )
    return {"messages": late_messages, "followups": late_followups}


def status(now: float | None = None) -> dict:
    """Everything a dashboard needs about the queue, in one call."""
    late = overdue(now)
    pend = pending()
    late_ids = {r["id"] for r in late["messages"]} | {r["id"] for r in late["followups"]}
    return {
        "mailboxes": [b.as_dict() for b in _state.boxes.values()],
        "pending": pend,
        "blocking": [r for r in pend if r.get("blocking")],
        "overdue": late["messages"],
        "overdue_followups": late["followups"],
        "overdue_ids": sorted(late_ids),
        "awaiting_followup": notices.awaiting_followup(),
        "counts": {
            "pending": len(pend),
            "blocking": sum(1 for r in pend if r.get("blocking")),
            "overdue": len(late_ids),
        },
    }


def human_mailbox() -> str | None:
    name = _cfg().get("mailboxes", "human", default=None)
    if name and name in _state.boxes:
        return str(name)
    return next((b.name for b in _state.boxes.values() if b.kind == "human"), None)


def nag(now: float | None = None) -> list[dict]:
    """Escalate everything overdue by one step, and say what was done.

    Step 1 re-delivers, step 2 marks the message OVERDUE (``status()`` reports
    it, so the dashboard shows it), step 3 posts to the human mailbox. Each
    step is recorded, so a level is never repeated.
    """
    now = time.time() if now is None else now
    late = overdue(now)
    actions = []
    human = human_mailbox()
    for kind, rows in (("message", late["messages"]), ("followup", late["followups"])):
        for rec in rows:
            level = min(int(rec.get("nag_level", 0)) + 1, len(ESCALATION))
            step = ESCALATION[level - 1]
            detail = ""
            if step == "redeliver":
                report = deliver(rec)
                detail = f"{sum(1 for d in report if d['ok'])}/{len(report)} hooks"
            elif step == "notify-human" and human:
                text = (
                    f"{rec['id']} is still {kind == 'followup' and 'awaiting its follow-up' or 'unacknowledged'} "
                    f"{round(rec['late_seconds'] / 60)} min past due; owed by "
                    f"{', '.join(rec.get('owed') or rec.get('to') or [])}."
                )
                send(text, to=human, key=f"nag:{rec['id']}:{level}")
                detail = f"posted to {human}"
            elif step == "notify-human":
                step, detail = "mark-overdue", "no human mailbox registered"
            with notices._locked():
                notices._append(
                    {
                        "type": "nag",
                        "id": rec["id"],
                        "ts": now,
                        "level": level,
                        "step": step,
                        "kind": kind,
                    }
                )
            actions.append(
                {
                    "id": rec["id"],
                    "kind": kind,
                    "level": level,
                    "step": step,
                    "detail": detail,
                    "late_seconds": rec["late_seconds"],
                }
            )
    return actions


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _bind(a) -> OpsConfig:
    cfg = config_from_args(a)
    configure(cfg)
    return cfg


def cmd_list(a) -> int:
    _bind(a)
    for box in mailboxes().values():
        hooks = ",".join(box.delivery) or "-"
        print(f"{box.name:16s} {box.kind:8s} delivery={hooks:24s} {box.description}")
    return 0


def cmd_register(a) -> int:
    _bind(a)
    box = register(
        a.name,
        kind=a.kind,
        delivery=[d for d in (a.delivery or "").split(",") if d],
        description=a.description,
        checkout=a.checkout,
    )
    print(f"registered {box.name} ({box.kind}) in {registry_path()}")
    return 0


def cmd_send(a) -> int:
    _bind(a)
    rec, dup, report = send(
        " ".join(a.message),
        to=a.to,
        blocking=a.block,
        sender=a.sender or "",
        key=a.key or "",
        due_minutes=a.due,
    )
    if dup:
        print(f"already sent as {rec['id']} (dedupe key {a.key}); nothing delivered again")
        return 0
    print(f"{rec['id']} -> {', '.join(rec['to'])}")
    for d in report:
        print(f"  {d['to']}/{d['hook']}: {'ok' if d['ok'] else 'FAILED'} {d['detail']}")
    return 0


def cmd_recv(a) -> int:
    _bind(a)
    name, _ = whoami(a.as_)
    rows = recv(name, mark=a.deliver)
    if not rows:
        print(f"nothing pending for {name}")
        return 0
    print(notices.render(rows))
    return 0


def cmd_ack(a) -> int:
    _bind(a)
    ids, name, source = ack(" ".join(a.text) or "(no detail given)", a.mid, a.as_, later=a.later)
    print(f"acknowledged {', '.join(ids)} as {name} (from {source})")
    return 0


def cmd_fsck(a) -> int:
    _bind(a)
    out = fsck()
    if a.json:
        print(json.dumps(out, indent=1))
        return 0 if out["ok"] else 1
    print(f"{out['records']} records, {out['messages']} messages")
    for label in ("orphan_acks", "early_acks", "unaddressed", "unknown_addressees"):
        for row in out[label]:
            print(f"  {label}: {row}")
    print("ok" if out["ok"] else "PROBLEMS FOUND")
    return 0 if out["ok"] else 1


def cmd_nag(a) -> int:
    _bind(a)
    actions = nag()
    if not actions:
        print("nothing overdue")
        return 0
    for act in actions:
        print(
            f"{act['id']} ({act['kind']}) {round(act['late_seconds'] / 60)} min late: "
            f"level {act['level']} {act['step']} {act['detail']}"
        )
    return 0


def cmd_status(a) -> int:
    _bind(a)
    print(json.dumps(status(), indent=1, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_flow.ops.mailbox",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(p)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("register")
    r.add_argument("name")
    r.add_argument("--kind", default=DEFAULT_KIND, choices=KINDS)
    r.add_argument("--delivery", default="", help="comma list of hook names")
    r.add_argument("--description", default="")
    r.add_argument("--checkout", default=None, help="path whose cwd implies this mailbox")
    s = sub.add_parser("send")
    s.add_argument("message", nargs="+")
    s.add_argument("--to", default="all")
    s.add_argument("--block", action="store_true")
    s.add_argument("--sender", default=None)
    s.add_argument("--key", default=None, help="client dedupe key; a retry sends once")
    s.add_argument("--due", type=float, default=None, help="minutes until it counts as overdue")
    v = sub.add_parser("recv")
    v.add_argument("--as", dest="as_", default=None)
    v.add_argument("--deliver", action="store_true", help="re-run the delivery hooks")
    k = sub.add_parser("ack")
    k.add_argument("text", nargs="*")
    k.add_argument("--as", dest="as_", default=None)
    k.add_argument("--id", dest="mid", default=None)
    k.add_argument("--later", action="store_true")
    f = sub.add_parser("fsck")
    f.add_argument("--json", action="store_true")
    sub.add_parser("nag")
    sub.add_parser("status")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cmds = {
        "list": cmd_list,
        "register": cmd_register,
        "send": cmd_send,
        "recv": cmd_recv,
        "ack": cmd_ack,
        "fsck": cmd_fsck,
        "nag": cmd_nag,
        "status": cmd_status,
    }
    try:
        return cmds[a.cmd](a)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
