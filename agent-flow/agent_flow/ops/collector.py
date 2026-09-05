"""Assemble one status dict for a project: ledger + tables + mailboxes.

The dashboard used to be one program that parsed, judged and drew. Splitting
the parsing out (:mod:`agent_flow.ops.ledger`) left this: a function that
gathers everything a viewer needs into a plain dict, and nothing that draws.
That ordering matters — a renderer can then be a page of formatting, a second
renderer costs nothing, and the whole thing is testable without a terminal.

Sources, all read-only and all file-based except the optional narrator:

* the verdict ledger and the gate reasons under the workspace,
* the allocation and worktree reservation tables (read without taking their
  locks: a viewer must never block a writer),
* the mailbox/notice queue, through a single :func:`agent_flow.ops.mailbox.status`
  call,
* whether a workflow process is alive for this project.

The narration is the one part that can call out to a model, and it is optional
in the strong sense: with no narrator configured, no binary on PATH, a
non-zero exit or a timeout, the field reads ``(narration unavailable)`` and
every other field is unaffected. A dashboard that cannot render without a model
call is a dashboard that stops working the day the model does.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from agent_flow.ops import mailbox, tray, worktree
from agent_flow.ops.config import OpsConfig
from agent_flow.ops.ledger import gate_reasons, ledger_rows, scoreboard

NARRATION_UNAVAILABLE = "(narration unavailable)"
NARRATOR_TIMEOUT = 60


def _read_table(path: Path) -> dict:
    """A reservation table, read without the lock (viewers never block writers)."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {"slots": {}, "history": []}
    return data if isinstance(data, dict) else {"slots": {}, "history": []}


def allocation_rows(cfg: OpsConfig) -> list[dict]:
    """Every declared allocation slot with its holder, config order first."""
    data = _read_table(tray.table_path(cfg))
    slots = data.get("slots") or {}
    order = [k for k in cfg.allocations if k in slots] + [
        k for k in slots if k not in cfg.allocations
    ]
    return [
        {
            "key": key,
            "job_id": slots[key].get("job_id") or "",
            "description": slots[key].get("description") or "",
            "holder": slots[key].get("holder"),
            "purpose": slots[key].get("purpose"),
            "since": slots[key].get("since"),
        }
        for key in order
    ]


def worktree_rows(cfg: OpsConfig) -> list[dict]:
    data = _read_table(worktree.table_dir(cfg) / worktree.JSON_NAME)
    slots = data.get("slots") or {}
    return [
        {
            "key": key,
            "holder": body.get("holder"),
            "purpose": body.get("purpose"),
            "since": body.get("since"),
        }
        for key, body in slots.items()
    ]


def gate_rows(workspace: Path) -> tuple[list[dict], dict]:
    """``([{id, state, runs, last_*, reason}], {green, total})`` from the ledger."""
    rows = ledger_rows(workspace)
    board = scoreboard(rows)
    reasons = gate_reasons(workspace)
    gates = []
    for gate in sorted(board):
        last = rows[gate][-1]
        gates.append(
            {
                "id": gate,
                "state": board[gate],
                "runs": len(rows[gate]),
                "last_time": last.get("time"),
                "last_run": last.get("run"),
                "last_commit": last.get("commit"),
                "reason": (reasons.get(gate) or {}).get("text"),
            }
        )
    green = sum(1 for v in board.values() if v == "pass")
    return gates, {"green": green, "total": len(board)}


def narrate(cfg: OpsConfig, status: dict) -> str:
    """One paragraph from the configured narrator, or why there is none.

    Every failure path returns the same string rather than raising: the
    narration is a nicety, and the status it describes is already complete.
    """
    command = cfg.get("dashboard", "narrator_command", default=None)
    if not command:
        return NARRATION_UNAVAILABLE
    model = cfg.get("dashboard", "narrator_model", default=None)
    argv = [str(command)] + (["--model", str(model)] if model else [])
    board = status["scoreboard"]
    prompt = (
        f"In three sentences, summarise this project's state for someone "
        f"returning to it. Gates green: {board['green']}/{board['total']}. "
        f"Red gates: "
        + (
            "; ".join(
                f"{g['id']} ({g['reason'] or 'no reason recorded'})"
                for g in status["gates"]
                if g["state"] != "pass"
            )
            or "none"
        )
        + f". Pending messages: {status['mailboxes']['counts']['pending']}, "
        f"overdue: {status['mailboxes']['counts']['overdue']}."
    )
    try:
        r = subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=NARRATOR_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return NARRATION_UNAVAILABLE
    text = (r.stdout or "").strip()
    return text if r.returncode == 0 and text else NARRATION_UNAVAILABLE


def collect(cfg: OpsConfig, now: float | None = None, narration: bool = False) -> dict:
    """Everything a viewer needs about one project, as a plain dict."""
    from agent_flow.ops.project import flow_pid

    now = time.time() if now is None else now
    workspace = cfg.workspace
    gates, board = gate_rows(workspace)
    mailbox.configure(cfg)
    rows = ledger_rows(workspace)
    newest = max((r["epoch"] for grows in rows.values() for r in grows), default=0.0)
    pid = flow_pid(cfg.project_root)
    status = {
        "project": cfg.project_name,
        "root": str(cfg.project_root),
        "workspace": str(workspace),
        "generated_at": now,
        "flow": {"pid": pid, "state": "running" if pid else "idle"},
        "gates": gates,
        "scoreboard": board,
        "last_ledger_epoch": newest,
        "allocations": allocation_rows(cfg),
        "worktrees": worktree_rows(cfg),
        "mailboxes": mailbox.status(now),
    }
    status["narration"] = narrate(cfg, status) if narration else NARRATION_UNAVAILABLE
    return status
