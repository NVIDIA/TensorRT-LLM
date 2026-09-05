"""Reservation table for the shared machine allocations ("trays").

A few long-lived allocations are shared by several agents, and one engine or
server run takes essentially every GPU on an allocation, so exactly ONE holder
at a time. This tool is both the mutual exclusion and the human-readable
record:

    python -m agent_flow.ops.tray claim dev1 --holder <name> --purpose "<what>"
    python -m agent_flow.ops.tray wait dev1 --holder <name> --purpose "<what>"
    python -m agent_flow.ops.tray release dev1 --holder <name>
    python -m agent_flow.ops.tray status
    python -m agent_flow.ops.tray set-job dev1 <job id>

Exit codes: 0 done, 3 busy / not the holder, 124 wait timed out.

The allocations come from ``[allocations]`` in the ops config; renaming one
there migrates a live table through the ``aliases`` list rather than orphaning
a reservation. State is ``TRAY-RESERVATIONS.json`` in the run root (locked)
plus a rendered ``TRAY-RESERVATIONS.md`` next to it. Read the ``.md``; never
edit either by hand.

Etiquette for holders: claim BEFORE launching anything, release the moment the
job ends, keep holds short unless the purpose says otherwise, and confirm the
GPUs are actually idle before launching — the table says who INTENDS to use an
allocation, the machine says who is using it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from agent_flow.ops.config import OpsConfig, add_config_argument, config_from_args
from agent_flow.ops.table import LockedTable, now, reconcile

JSON_NAME = "TRAY-RESERVATIONS.json"
MD_NAME = "TRAY-RESERVATIONS.md"


def render(data: dict) -> str:
    out = [
        "# Allocation reservations",
        "",
        "Shared machine allocations. One holder each. Managed by",
        "`python -m agent_flow.ops.tray`; do not edit by hand (regenerated from",
        f"{JSON_NAME}).",
        "",
        "| tray | job | status | holder | purpose | since |",
        "|---|---|---|---|---|---|",
    ]
    for key, t in data["slots"].items():
        status = "HELD" if t.get("holder") else "free"
        desc = f" ({t['description']})" if t.get("description") else ""
        out.append(
            f"| {key}{desc} | {t.get('job_id') or '-'} | {status} | {t.get('holder') or '-'} "
            f"| {t.get('purpose') or '-'} | {t.get('since') or '-'} |"
        )
    out += ["", "## History", ""]
    out += [f"- {h}" for h in data["history"]]
    return "\n".join(out) + "\n"


def declared_slots(cfg: OpsConfig) -> dict[str, dict]:
    return {
        key: {"job_id": alloc.job_id, "description": alloc.description}
        for key, alloc in cfg.allocations.items()
    }


def table_path(cfg: OpsConfig):
    """Canonical JSON table other tools (the idle watcher, the dashboard) read."""
    return cfg.run_root / JSON_NAME


def open_table(cfg: OpsConfig) -> LockedTable:
    root = cfg.run_root
    return LockedTable(table_path(cfg), root / MD_NAME, {"slots": {}, "history": []}, render)


def canonical(cfg: OpsConfig, key: str) -> str:
    return cfg.alloc_aliases.get(key, key)


def job_state(job_id: str) -> str:
    """Scheduler state of an allocation, or a reason it is unknown."""
    if not job_id:
        return "-"
    try:
        r = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T %N %L"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return r.stdout.strip() or "not in queue"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"scheduler unavailable ({type(exc).__name__})"


def _slot(t: LockedTable, cfg: OpsConfig, key: str) -> dict:
    key = canonical(cfg, key)
    if key not in t.data["slots"]:
        raise SystemExit(
            f"error: unknown allocation {key!r}; declared: {', '.join(t.data['slots']) or '(none)'}"
        )
    return t.data["slots"][key]


def cmd_claim(a, cfg: OpsConfig) -> int:
    with open_table(cfg) as t:
        reconcile(t.data, declared_slots(cfg), cfg.alloc_aliases)
        key = canonical(cfg, a.tray)
        slot = _slot(t, cfg, key)
        if slot.get("holder") and slot["holder"] != a.holder:
            print(
                f"BUSY {key}: held by {slot['holder']} since {slot.get('since')} "
                f"for: {slot.get('purpose')}"
            )
            return 3
        slot.update(holder=a.holder, purpose=a.purpose, since=now())
        t.log(f"{key} claimed by {a.holder}: {a.purpose}")
        print(f"HELD {key} (job {slot.get('job_id') or '-'}) by {a.holder}")
        return 0


def cmd_wait(a, cfg: OpsConfig) -> int:
    deadline = time.time() + a.timeout
    while True:
        rc = cmd_claim(a, cfg)
        if rc == 0:
            return 0
        if time.time() >= deadline:
            print(f"TIMEOUT waiting for {canonical(cfg, a.tray)}")
            return 124
        time.sleep(min(a.poll, max(1, deadline - time.time())))


def cmd_release(a, cfg: OpsConfig) -> int:
    with open_table(cfg) as t:
        reconcile(t.data, declared_slots(cfg), cfg.alloc_aliases)
        key = canonical(cfg, a.tray)
        slot = _slot(t, cfg, key)
        if not slot.get("holder"):
            print(f"{key} already free")
            return 0
        if slot["holder"] != a.holder and not a.force:
            print(
                f"REFUSED: {key} is held by {slot['holder']}, not {a.holder} "
                f"(--force is the human's escape hatch)"
            )
            return 3
        forced = " (forced)" if slot["holder"] != a.holder else ""
        t.log(f"{key} released by {a.holder}{forced}" + (f": {a.note}" if a.note else ""))
        slot.update(holder=None, purpose=None, since=None)
        print(f"FREE {key}")
        return 0


def cmd_status(a, cfg: OpsConfig) -> int:
    with open_table(cfg) as t:
        reconcile(t.data, declared_slots(cfg), cfg.alloc_aliases)
        data = t.data
    for key, slot in data["slots"].items():
        state = "-" if a.no_scheduler else job_state(slot.get("job_id", ""))
        holder = (
            f"HELD by {slot['holder']} since {slot.get('since')}: {slot.get('purpose')}"
            if slot.get("holder")
            else "free"
        )
        print(f"{key:10s} job {slot.get('job_id') or '-':10s} [{state}]  {holder}")
    if a.history:
        for h in data["history"][-a.history :]:
            print("  " + h)
    return 0


def cmd_set_job(a, cfg: OpsConfig) -> int:
    with open_table(cfg) as t:
        reconcile(t.data, declared_slots(cfg), cfg.alloc_aliases)
        key = canonical(cfg, a.tray)
        slot = _slot(t, cfg, key)
        old = slot.get("job_id", "")
        slot["job_id"] = a.job_id
        t.log(f"{key} job id {old or '-'} -> {a.job_id}")
    print(f"{key}: {old or '-'} -> {a.job_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_flow.ops.tray",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(p)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("claim", "wait"):
        s = sub.add_parser(name)
        s.add_argument("tray")
        s.add_argument("--holder", required=True)
        s.add_argument("--purpose", required=True)
        if name == "wait":
            s.add_argument("--timeout", type=int, default=540)
            s.add_argument("--poll", type=int, default=20)
    s = sub.add_parser("release")
    s.add_argument("tray")
    s.add_argument("--holder", required=True)
    s.add_argument("--note", default="")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--history", type=int, default=5)
    s.add_argument("--no-scheduler", action="store_true", help="skip the scheduler query")
    s = sub.add_parser("set-job")
    s.add_argument("tray")
    s.add_argument("job_id")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    return {
        "claim": cmd_claim,
        "wait": cmd_wait,
        "release": cmd_release,
        "status": cmd_status,
        "set-job": cmd_set_job,
    }[a.cmd](a, cfg)


if __name__ == "__main__":
    sys.exit(main())
