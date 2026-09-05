"""Acknowledge pending notices (agent-facing side of the notice queue).

    python -m agent_flow.ops.ack_notice "what I did about it"
    python -m agent_flow.ops.ack_notice --id n7 --later "acked, result after the run"
    python -m agent_flow.ops.ack_notice --id n7 --followup "the run finished: ..."

``--role`` names who is acking; without it the role is inferred from the cwd
via the config's role-to-checkout mapping.
"""

from __future__ import annotations

import argparse

from agent_flow.ops import notices
from agent_flow.ops.config import add_config_argument, config_from_args
from agent_flow.ops.notify_agent import cache_path, strip_block


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="agent_flow.ops.ack_notice",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(ap)
    ap.add_argument("text", nargs="*", help="one line: what you did about it")
    ap.add_argument("--id", dest="nid", help="acknowledge a single notice id")
    ap.add_argument(
        "--later",
        action="store_true",
        help="ack now, result later: the notice reads 'follow-up due' until --followup",
    )
    ap.add_argument(
        "--followup",
        action="store_true",
        help="post the promised follow-up for --id (that notice was acked with --later)",
    )
    ap.add_argument("--role", help="who is acking (default: inferred from the cwd)")
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    notices.configure(cfg)

    role = a.role or notices.infer_role()
    text = " ".join(a.text) or "(no detail given)"
    if a.followup:
        if not a.nid:
            ap.error("--followup needs --id <notice id>")
        notices.followup(text, a.nid)
        print(f"follow-up posted for {a.nid}: {text}")
        return 0
    ids = notices.ack(text, a.nid, followup=a.later, role=role)
    if ids and ids[0].startswith("r"):
        print(f"no pending notice matched; recorded as report {ids[0]}: {text}")
        return 0
    cache = cache_path(cfg)
    if not notices.pending() and cache.exists():
        cache.write_text(strip_block(cache.read_text()))
    suffix = " (follow-up due)" if a.later else ""
    print(f"acknowledged {', '.join(ids)} as {role}{suffix}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
