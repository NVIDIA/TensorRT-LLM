"""Run a long command detached, with separated stdout/stderr and an exit code.

Python port of the shell bg.sh. Same idea, three differences that matter:

  * stdout and stderr land in SEPARATE files, so wrapper-level alerts (which
    in_container.py writes to stderr) never mix into the command's own output.
  * `--status` reports rc, wall time, and the last line of each stream, so one
    call tells you whether to look further.
  * the command is recorded verbatim in <name>.cmd, so a run is reproducible
    without digging through a transcript.

Detachment is via a tmux session, which also works where ``nohup`` is denied.

  bg start job1 -- <cmd...>   # detached; writes logs/job1.{out,err,rc,cmd}
  bg status job1              # RUNNING, or DONE with rc + last lines
  bg tail job1 --lines 40     # last N lines of both streams
  bg wait job1                # block <=540 s; returns early on a new notice (125)
                              # or a 180 s output stall (124); 0 done, 1 failed
  bg list                     # every recorded run, newest first
  bg sleep 240 [--job job1 ...]  # interruptible sleep: returns early on a new notice
                              # (125) or when any --job run finishes (0/1); 124 = slept
                              # the full time. Use INSTEAD of `sleep N`.

Run as ``python -m agent_flow.ops.bg <subcommand>``.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from agent_flow.ops import notices  # noqa: E402
from agent_flow.ops.config import add_config_argument, config_from_args  # noqa: E402

ROOT = Path(".")  # set by configure()
LOGDIR = ROOT / "logs"
EXIT_BLOCKED = 4


def configure(cfg) -> None:
    """Bind the module to a run: log directory and notice queue."""
    global ROOT, LOGDIR
    ROOT = cfg.run_root
    LOGDIR = cfg.log_dir
    notices.configure(cfg)


def check_notice() -> None:
    """Same hard gate as the container wrapper.

    Backgrounding is an entry point too: without this, a blocking notice could
    be sidestepped by launching work that never goes through the wrapper.
    """
    blocking = notices.blocking_pending()
    if not blocking:
        return
    print(
        f"bg: REFUSING TO START - {len(blocking)} unacknowledged blocking "
        f"notice(s) pending.\nNothing was started; this is not a failure of your "
        f"command.\n" + "-" * 70 + f"\n{notices.render(blocking)}\n" + "-" * 70,
        file=sys.stderr,
    )
    raise SystemExit(EXIT_BLOCKED)


def paths(name: str) -> dict[str, Path]:
    return {k: LOGDIR / f"{name}.{k}" for k in ("out", "err", "rc", "cmd", "start")}


def last_line(p: Path, limit: int = 160) -> str:
    if not p.exists():
        return ""
    for line in reversed(p.read_text(errors="ignore").splitlines()):
        if line.strip():
            return " ".join(line.split())[:limit]
    return ""


def running(name: str) -> bool:
    return (
        subprocess.run(["tmux", "has-session", "-t", f"bg-{name}"], capture_output=True).returncode
        == 0
    )


def cmd_start(a) -> int:
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("no command given (put it after --)", file=sys.stderr)
        return 2
    check_notice()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    p = paths(a.name)
    for k in ("rc", "out", "err"):
        p[k].unlink(missing_ok=True)
    line = " ".join(shlex.quote(c) for c in cmd)
    p["cmd"].write_text(line + "\n")
    p["start"].write_text(str(time.time()))
    subprocess.run(["tmux", "kill-session", "-t", f"bg-{a.name}"], capture_output=True)
    inner = (
        f"{line} > {shlex.quote(str(p['out']))} 2> {shlex.quote(str(p['err']))}; "
        f"echo $? > {shlex.quote(str(p['rc']))}"
    )
    r = subprocess.run(
        ["tmux", "new-session", "-d", "-s", f"bg-{a.name}", inner], capture_output=True, text=True
    )
    if r.returncode:
        print(f"failed to start: {r.stderr.strip()}", file=sys.stderr)
        return 1
    print(f"started bg-{a.name}\n  stdout {p['out']}\n  stderr {p['err']}")
    return 0


def describe(name: str) -> str:
    p = paths(name)
    if not p["cmd"].exists():
        return f"{name}: no such run"
    started = float(p["start"].read_text()) if p["start"].exists() else None
    if running(name):
        age = f"{time.time() - started:.0f}s" if started else "?"
        return (
            f"{name}: RUNNING ({age})\n  $ {p['cmd'].read_text().strip()[:150]}\n"
            f"  out: {last_line(p['out']) or '(nothing yet)'}\n"
            f"  err: {last_line(p['err']) or '(nothing yet)'}"
        )
    rc = p["rc"].read_text().strip() if p["rc"].exists() else "?"
    verdict = "DONE" if rc == "0" else f"FAILED rc={rc}"
    return (
        f"{name}: {verdict}\n  $ {p['cmd'].read_text().strip()[:150]}\n"
        f"  out: {last_line(p['out']) or '(empty)'}\n"
        f"  err: {last_line(p['err']) or '(empty)'}"
    )


def cmd_status(a) -> int:
    print(describe(a.name))
    return 0


def cmd_tail(a) -> int:
    p = paths(a.name)
    for k in ("out", "err"):
        if p[k].exists():
            lines = p[k].read_text(errors="ignore").splitlines()[-a.lines :]
            print(f"--- {k} ({len(lines)} lines) ---")
            print("\n".join(lines))
    return 0


def cmd_wait(a) -> int:
    """Block until the run finishes or --timeout seconds pass, then print status.

    Exit 0 = finished with rc 0, 1 = finished non-zero, 124 = still running
    when the timeout expired, 125 = a human notice arrived (or a blocking one
    is pending) so the wait stopped early; the job keeps running. This is the
    sanctioned way for an agent to wait on a long job: one bounded call at a
    time (default 540 s, under the 10-minute tool-call cap), never a bare
    `sleep N` inside a tool call. The wait
    polls the notice queue every 5 s and returns as soon as a NEW notice lands
    (one posted after the wait began; notices already pending when it started
    are not counted, so a notice the agent is deliberately holding until a
    result arrives does not make every wait return instantly). Between calls
    the agent's hooks fire too, so steering latency is bounded by the poll,
    not the job. The --stall flag reports when neither stream has grown for
    that many seconds (a hung `srun` looks exactly like that).
    """
    p = paths(a.name)
    if not p["cmd"].exists():
        print(f"{a.name}: no such run", file=sys.stderr)
        return 2

    def quiet_for() -> float:
        newest = max(
            (f.stat().st_mtime for f in (p["out"], p["err"]) if f.exists()),
            default=float(p["start"].read_text()) if p["start"].exists() else t0,
        )
        return time.time() - newest

    t0 = time.time()
    seen = {r["id"] for r in notices.pending()}
    arrived: list[dict] = []
    stalled = False
    while running(a.name) and time.time() - t0 < a.timeout:
        time.sleep(min(5.0, max(0.0, a.timeout - (time.time() - t0))))
        pend = notices.pending()
        arrived = [r for r in pend if r["id"] not in seen or r.get("blocking")]
        if arrived:
            break
        if a.stall and quiet_for() > a.stall:
            stalled = True
            break
    print(describe(a.name))
    if arrived and running(a.name):
        print(
            f"  NOTICE: wait stopped early, {len(arrived)} human notice(s) need attention "
            f"(job still running):\n{notices.render(arrived)}"
        )
        return 125
    if running(a.name):
        if stalled:
            print(
                f"  STALL: no output for {quiet_for():.0f}s (> {a.stall:.0f}s). If `squeue -s -j <jobid>` "
                f"shows no step for it, the srun is hung: kill it and retry once."
            )
        return 124
    rc = p["rc"].read_text().strip() if p["rc"].exists() else "?"
    return 0 if rc == "0" else 1


def cmd_sleep(a) -> int:
    """Interruptible replacement for a bare `sleep N` inside a tool call.

    Sleeps up to --seconds, polling every 5 s, and returns early when a NEW
    human notice lands (exit 125, notice printed) or when any run named with
    --job stops running (exit 0 if every named run finished rc 0, else 1).
    Exit 124 = the full time elapsed. A bare `sleep` is blind for its whole
    duration: a notice posted one second in waits the full N seconds, and a
    job that finishes early wastes the rest. This one wakes on either.
    """
    seconds = min(a.seconds, 540.0)
    for name in a.job:
        if not paths(name)["cmd"].exists():
            print(f"{name}: no such run", file=sys.stderr)
            return 2
    t0 = time.time()
    seen = {r["id"] for r in notices.pending()}
    arrived: list[dict] = []
    finished: list[str] = []
    while True:
        # check first, sleep second: a job that already finished returns at once
        pend = notices.pending()
        arrived = [r for r in pend if r["id"] not in seen or r.get("blocking")]
        if arrived:
            break
        finished = [n for n in a.job if not running(n)]
        if finished:
            break
        left = seconds - (time.time() - t0)
        if left <= 0:
            break
        time.sleep(min(5.0, left))
    el = time.time() - t0
    if arrived:
        print(
            f"slept {el:.0f}s of {seconds:.0f}s; stopped early, {len(arrived)} human notice(s) "
            f"need attention:\n{notices.render(arrived)}"
        )
        return 125
    if finished:
        for n in finished:
            print(describe(n))
        ok = all(
            (paths(n)["rc"].read_text().strip() == "0") if paths(n)["rc"].exists() else False
            for n in finished
        )
        return 0 if ok else 1
    print(
        f"slept {el:.0f}s; no notice, "
        + (f"{', '.join(a.job)} still running" if a.job else "nothing watched")
    )
    return 124


def cmd_list(_a) -> int:
    runs = sorted(LOGDIR.glob("*.cmd"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not runs:
        print("no runs recorded")
    for f in runs:
        print(describe(f.stem).splitlines()[0])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_flow.ops.bg")
    add_config_argument(ap)
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("start")
    s.add_argument("name")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_start)
    s = sub.add_parser("status")
    s.add_argument("name")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("tail")
    s.add_argument("name")
    s.add_argument("--lines", type=int, default=40)
    s.set_defaults(fn=cmd_tail)
    s = sub.add_parser("wait")
    s.add_argument("name")
    s.add_argument(
        "--timeout",
        type=float,
        default=540,
        help="max seconds to block (default 540, under the 10-min tool-call cap)",
    )
    s.add_argument(
        "--stall",
        type=float,
        default=180,
        help="return early with a STALL warning if no output for this many seconds (0 = off)",
    )
    s.set_defaults(fn=cmd_wait)
    s = sub.add_parser("list")
    s.set_defaults(fn=cmd_list)
    s = sub.add_parser("sleep")
    s.add_argument("seconds", type=float)
    s.add_argument(
        "--job", action="append", default=[], help="bg run name to also wake on (repeatable)"
    )
    s.set_defaults(fn=cmd_sleep)
    a = ap.parse_args(argv)
    configure(config_from_args(a))
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
