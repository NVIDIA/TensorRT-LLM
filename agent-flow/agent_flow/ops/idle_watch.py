"""Alert a role when a shared allocation sits unreserved.

For the agents, not humans; humans see the same thing on the dashboard.

Polls the allocation table every POLL seconds. An allocation is "idle" when its
job is RUNNING, nobody holds it, and it has been free for at least IDLE_MIN
minutes (release -> claim handoffs seconds apart never fire). One notice per
idle interval, repeated every REMIND_MIN minutes while it stays idle. State is
in memory: a restart means a fresh debounce.

  python -m agent_flow.ops.bg start idle-watch -- \
      python -m agent_flow.ops.idle_watch
"""

from __future__ import annotations

import argparse  # noqa: E402
import json
import subprocess
import time
from datetime import datetime

from agent_flow.ops import notices, tray  # noqa: E402
from agent_flow.ops.config import add_config_argument, config_from_args  # noqa: E402

POLL, IDLE_MIN, REMIND_MIN = 60, 5, 30


def squeue_state(jobid: str) -> str:
    try:
        r = subprocess.run(
            ["squeue", "-h", "-j", jobid, "-o", "%T"], capture_output=True, text=True, timeout=25
        )
        return r.stdout.strip() or "GONE"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def notify(cfg, role: str, text: str) -> None:
    """Post the alert through the notice queue (same channels a human uses)."""
    notices.configure(cfg)
    notices.add(text, to=role)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="agent_flow.ops.idle_watch",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(ap)
    ap.add_argument("--to", default="", help="role to alert (default: the first configured role)")
    ap.add_argument("--once", action="store_true", help="one pass, then exit (for tests)")
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    role = a.to or (cfg.roles[0] if cfg.roles else "coder")
    table = tray.table_path(cfg)
    free_since: dict[str, float] = {}
    last_sent: dict[str, float] = {}
    log("started")
    while True:
        try:
            trays = json.loads(table.read_text())["slots"]
        except Exception as exc:  # noqa: BLE001
            log(f"table unreadable: {exc}")
            if a.once:
                return
            time.sleep(POLL)
            continue
        now = time.time()
        for key, t in trays.items():
            if t.get("holder"):
                if key in free_since:
                    log(
                        f"{key} claimed by {t['holder']} after {int((now - free_since[key]) / 60)} min free"
                    )
                free_since.pop(key, None)
                last_sent.pop(key, None)
                continue
            free_since.setdefault(key, now)
            idle_min = (now - free_since[key]) / 60
            if idle_min < IDLE_MIN:
                continue
            if key in last_sent and now - last_sent[key] < REMIND_MIN * 60:
                continue
            if squeue_state(t.get("job_id", "")) != "RUNNING":
                continue
            notify(
                cfg,
                role,
                f"Allocation watch, {datetime.now():%H:%M}: '{key}' (job {t.get('job_id')}) is "
                f"RUNNING and UNRESERVED, free for {int(idle_min)} min. If you have queued work, "
                f"claim it (python -m agent_flow.ops.tray claim {key} --holder <worker> "
                f"--purpose ...). If it is idle on purpose, say so in a ledger row so this is not "
                f"re-flagged. Reminder every {REMIND_MIN} min while idle.",
            )
            last_sent[key] = now
            log(f"notified: {key} idle {int(idle_min)} min")
        if a.once:
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
