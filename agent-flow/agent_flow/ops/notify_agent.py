"""Get a RUNNING agent turn's attention without restarting the workflow.

The notice goes into the append-only queue (``agent_flow.ops.notices``) and is
also mirrored into the channels the roles are required to read mid-turn:

* the command cache the build-phase prompt makes every role read before
  running any test or eval command, and
* the live-notes file a PostToolUse hook injects into the agent's next tool
  result, when the harness supports it.

Both are best-effort delivery paths; the queue is the record. An earlier
version also echoed the notice around every wrapped cluster command. That was
REMOVED: the wrapper's output is redirected into the run's evidence logs, so
the banner ended up inside files acceptance criteria treat as evidence. Never
inject into the output of a wrapped command.

    python -m agent_flow.ops.notify_agent "Stop after this check and switch to X."
    python -m agent_flow.ops.notify_agent --to reviewer --block "<message>"
    python -m agent_flow.ops.notify_agent --clear
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys

from agent_flow.ops import notices
from agent_flow.ops.config import OpsConfig, add_config_argument, config_from_args

BEGIN, END = "<!-- human-notice -->", "<!-- /human-notice -->"


def strip_block(text: str) -> str:
    return re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n*", "", text, flags=re.S)


def cache_path(cfg: OpsConfig):
    return cfg.workspace / str(cfg.get("notices", "command_cache", default="test_command.md"))


def live_notes_path(cfg: OpsConfig):
    return cfg.workspace / str(cfg.get("notices", "live_notes", default="LIVE-NOTES.md"))


def addressee_banner(to_list: list[str], all_roles: tuple[str, ...]) -> str:
    if len(to_list) == 1:
        others = ", ".join(r for r in all_roles if r != to_list[0]) or "everyone else"
        return f"FOR THE {to_list[0].upper()} ONLY ({others}: read, do not ack)"
    return "for " + " and ".join(to_list).upper() + " (each role acks separately)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="agent_flow.ops.notify_agent",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(ap)
    ap.add_argument("message", nargs="?")
    ap.add_argument("--clear", action="store_true", help="remove the mirrored block")
    ap.add_argument("--to", default="all", help="addressees: a role, a comma list, or 'all'")
    ap.add_argument(
        "--block",
        action="store_true",
        help="hard gate: the container and background wrappers refuse to run "
        "anything (exit 4) until the notice is acknowledged",
    )
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    notices.configure(cfg)
    cache = cache_path(cfg)
    live = live_notes_path(cfg)

    if a.clear or not a.message:
        if cache.exists():
            cache.write_text(strip_block(cache.read_text()))
        print("cleared the mirrored block (queue history is append-only)")
        return 0

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
    try:
        to_list = notices.parse_to(a.to)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    who = addressee_banner(to_list, notices.roles())
    ack_cmd = notices.ack_command()
    body = (
        f"{BEGIN}\n> **HUMAN NOTICE ({stamp}) — {who}** — newer than your prompt and "
        f"overrides it where they conflict. Act on it now rather than finishing your "
        f"current approach first. When done, acknowledge with:\n>\n"
        f'> `{ack_cmd} --role <role> "<one line: what you did about it>"`\n>\n'
        + "\n".join(f"> {ln}" for ln in a.message.splitlines())
        + f"\n{END}\n"
    )
    rec = notices.add(a.message, blocking=a.block, to=to_list)
    delivered = [f"{notices.queue_path()} ({rec['id']}{', BLOCKING' if a.block else ''})"]
    if cache.exists():
        cache.write_text(body + "\n" + strip_block(cache.read_text()))
        delivered.append(str(cache))
    # Live notes are append-only history, so an ack does not remove the entry.
    if live.exists():
        with live.open("a") as fh:
            fh.write(
                f"\n## {stamp} — human notice ({who})"
                + (
                    " (BLOCKING: wrapped commands refuse to run until acknowledged)"
                    if a.block
                    else ""
                )
                + "\n"
                + a.message.strip()
                + f'\n\nAcknowledge with: `{ack_cmd} "<what you did>"`\n'
            )
        delivered.append(str(live))
    print("notice delivered to:\n  " + "\n  ".join(delivered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
