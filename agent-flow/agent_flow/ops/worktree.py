r"""Reservation table for shared git worktree slots.

Creating a worktree on a network filesystem takes minutes, so slots are
pre-created once and reused: an agent claims a free slot, checks out its
branch there, and releases the slot when done. The table is the same locked
JSON object the allocation table uses (``agent_flow.ops.table``), which
replaces the hand-edited markdown file this started as — two agents editing
markdown in the same minute silently lose one edit, and a lost worktree
reservation means two agents writing to the same checkout.

    python -m agent_flow.ops.worktree claim wt-1 --holder <name> \\
        --purpose "<what>" [--branch <branch>]
    python -m agent_flow.ops.worktree status [--history 10]
    python -m agent_flow.ops.worktree release wt-1 --holder <name> [--note "..."]

Exit codes: 0 done, 3 busy / not the holder.

Slots come from ``[worktrees].slots`` in the ops config, and the table lives
in ``[worktrees].dir`` (default: the run root).
"""

from __future__ import annotations

import argparse
import sys

from agent_flow.ops.config import OpsConfig, OpsConfigError, add_config_argument, config_from_args
from agent_flow.ops.table import LockedTable, now, reconcile

JSON_NAME = "WORKTREE-RESERVATIONS.json"
MD_NAME = "WORKTREE-RESERVATIONS.md"


def render(data: dict) -> str:
    out = [
        "# Worktree reservations",
        "",
        "Shared, reusable worktree slots. Claim before use, release when done.",
        "Managed by `python -m agent_flow.ops.worktree`; do not edit by hand",
        f"(regenerated from {JSON_NAME}).",
        "",
        "| slot | status | holder | branch | since | purpose |",
        "|---|---|---|---|---|---|",
    ]
    for key, s in data["slots"].items():
        status = "HELD" if s.get("holder") else "free"
        out.append(
            f"| {key} | {status} | {s.get('holder') or '-'} | {s.get('branch') or '-'} "
            f"| {s.get('since') or '-'} | {s.get('purpose') or '-'} |"
        )
    out += ["", "## History", ""]
    out += [f"- {h}" for h in data["history"]]
    return "\n".join(out) + "\n"


def table_dir(cfg: OpsConfig):
    return cfg.worktree_dir or cfg.run_root


def open_table(cfg: OpsConfig) -> LockedTable:
    d = table_dir(cfg)
    return LockedTable(d / JSON_NAME, d / MD_NAME, {"slots": {}, "history": []}, render)


def declared_slots(cfg: OpsConfig) -> dict[str, dict]:
    slots = cfg.worktree_slots
    if not slots:
        raise OpsConfigError(
            f"{cfg.path}: [worktrees].slots is empty; declare the pre-created "
            f"worktree slot names before claiming one."
        )
    return {name: {} for name in slots}


def _slot(t: LockedTable, key: str) -> dict:
    if key not in t.data["slots"]:
        raise SystemExit(
            f"error: unknown worktree slot {key!r}; declared: "
            f"{', '.join(t.data['slots']) or '(none)'}"
        )
    return t.data["slots"][key]


def cmd_claim(a, cfg: OpsConfig) -> int:
    with open_table(cfg) as t:
        reconcile(t.data, declared_slots(cfg), {})
        slot = _slot(t, a.slot)
        if slot.get("holder") and slot["holder"] != a.holder:
            print(
                f"BUSY {a.slot}: held by {slot['holder']} since {slot.get('since')} "
                f"for: {slot.get('purpose')}"
            )
            return 3
        slot.update(holder=a.holder, purpose=a.purpose, branch=a.branch or None, since=now())
        t.log(f"{a.slot} claimed by {a.holder} ({a.branch or 'no branch'}): {a.purpose}")
        print(f"HELD {a.slot} by {a.holder}")
        return 0


def cmd_release(a, cfg: OpsConfig) -> int:
    with open_table(cfg) as t:
        reconcile(t.data, declared_slots(cfg), {})
        slot = _slot(t, a.slot)
        if not slot.get("holder"):
            print(f"{a.slot} already free")
            return 0
        if slot["holder"] != a.holder and not a.force:
            print(f"REFUSED: {a.slot} is held by {slot['holder']}, not {a.holder}")
            return 3
        forced = " (forced)" if slot["holder"] != a.holder else ""
        t.log(f"{a.slot} released by {a.holder}{forced}" + (f": {a.note}" if a.note else ""))
        slot.update(holder=None, purpose=None, since=None)
        print(f"FREE {a.slot}")
        return 0


def cmd_status(a, cfg: OpsConfig) -> int:
    with open_table(cfg) as t:
        reconcile(t.data, declared_slots(cfg), {})
        data = t.data
    for key, slot in data["slots"].items():
        held = (
            f"HELD by {slot['holder']} since {slot.get('since')} "
            f"({slot.get('branch') or 'no branch'}): {slot.get('purpose')}"
            if slot.get("holder")
            else "free"
        )
        print(f"{key:10s} {held}")
    if a.history:
        for h in data["history"][-a.history :]:
            print("  " + h)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_flow.ops.worktree",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(p)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("claim")
    s.add_argument("slot")
    s.add_argument("--holder", required=True)
    s.add_argument("--purpose", required=True)
    s.add_argument("--branch", default="")
    s = sub.add_parser("release")
    s.add_argument("slot")
    s.add_argument("--holder", required=True)
    s.add_argument("--note", default="")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--history", type=int, default=5)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    try:
        return {"claim": cmd_claim, "release": cmd_release, "status": cmd_status}[a.cmd](a, cfg)
    except OpsConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
