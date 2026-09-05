"""Run one command inside a held allocation's container -- via the DISPATCHER.

Drop-in sibling of ``in_container`` (same ``--out/--err/--tail`` options, same
exit codes 3 = allocation not RUNNING, 4 = blocking notice), but it does NOT
spend a scheduler step per command. It drops a request file in the spool dir
watched by ``container_dispatch`` and tails the result back, so a whole day of
GPU probes costs zero steps instead of one each.

    python -m agent_flow.ops.dispatch_client -- hostname
    NTASKS=4 python -m agent_flow.ops.dispatch_client -- <launcher> python x.py
    python -m agent_flow.ops.dispatch_client --status
    python -m agent_flow.ops.dispatch_client --start   # the ONE step this costs
    python -m agent_flow.ops.dispatch_client --stop

There is deliberately NO per-command fallback to a fresh step: falling back is
exactly the behaviour that burns the allocation's step budget. If no live
dispatcher is found the client gives up quickly (exit 5) and prints the
--start command.

Exit codes: the command's own (max over ranks), or 3/4 as above, 5 = no live
dispatcher, 130 = interrupted (the request was cancelled in the container).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from agent_flow.ops import alloc, notices
from agent_flow.ops import in_container as ic
from agent_flow.ops.config import OpsConfig, add_config_argument, config_from_args

DAEMON_MODULE = "agent_flow.ops.container_dispatch"

STALE_S = float(os.environ.get("DISPATCH_STALE_S", "30"))  # heartbeat older than this = dead
START_WAIT_S = float(os.environ.get("DISPATCH_START_WAIT_S", "180"))
GIVEUP_S = float(os.environ.get("DISPATCH_GIVEUP_S", "20"))  # never hang on a dead daemon
DAEMON_TIME_MIN = int(os.environ.get("DISPATCH_TIME_MIN", "480"))
EXIT_NO_DAEMON = 5


def warn(*lines: str) -> None:
    ic.warn(*lines)


def spool_dir(root: Path, jobid: str, ntasks: int) -> Path:
    return Path(root) / str(jobid) / f"n{ntasks}"


def heartbeats(spool: Path, ntasks: int) -> list[float]:
    """Age in seconds of each rank's heartbeat; inf when the file is missing."""
    ages = []
    now = time.time()
    for k in range(ntasks):
        p = spool / f"alive.rank{k}"
        try:
            ages.append(now - p.stat().st_mtime)
        except OSError:
            ages.append(float("inf"))
    return ages


def is_live(spool: Path, ntasks: int) -> bool:
    return all(a < STALE_S for a in heartbeats(spool, ntasks))


def spool_root(cfg: OpsConfig) -> Path:
    """Where request/response files live: one subdir per (job id, ntasks)."""
    override = os.environ.get("DISPATCH_SPOOL_ROOT")
    if override:
        return Path(override)
    return cfg.run_root / str(cfg.get("dispatch", "spool_dir", default="dispatch"))


def start_cmd(cfg: OpsConfig, jobid: str, ntasks: int, spool: Path, time_min: int) -> list[str]:
    """The single scheduler step that hosts the daemon (one per (jobid, ntasks))."""
    argv = [
        "python3",
        "-m",
        DAEMON_MODULE,
        "--config",
        str(cfg.path),
        "--spool",
        str(spool),
        "--ntasks",
        str(ntasks),
    ]
    return ic.srun_command(cfg, jobid, ntasks, time_min, ic.remote_script(cfg, argv))


def do_start(cfg: OpsConfig, jobid: str, ntasks: int, spool: Path, time_min: int) -> int:
    if is_live(spool, ntasks):
        warn(f"dispatcher already live for job {jobid} ntasks={ntasks} ({spool})")
        return 0
    spool.mkdir(parents=True, exist_ok=True)
    for stale in ("stop",):
        (spool / stale).unlink(missing_ok=True)
    srun_log = spool / "srun.log"
    cmd = start_cmd(cfg, jobid, ntasks, spool, time_min)
    warn(
        f"starting dispatcher (1 scheduler step) for job {jobid} ntasks={ntasks}",
        f"  srun log: {srun_log}",
    )
    with open(srun_log, "ab", buffering=0) as f:
        f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd[:12])} ===\n".encode())
        subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    deadline = time.time() + START_WAIT_S
    while time.time() < deadline:
        if is_live(spool, ntasks):
            warn(f"dispatcher up after {START_WAIT_S - (deadline - time.time()):.0f}s")
            return 0
        time.sleep(1.0)
    warn(f"dispatcher did NOT come up within {START_WAIT_S:.0f}s; see {srun_log}")
    return EXIT_NO_DAEMON


def do_stop(spool: Path, ntasks: int) -> int:
    if not spool.exists():
        warn(f"no spool dir {spool}")
        return 0
    (spool / "stop").write_text(f"{time.time()}\n")
    deadline = time.time() + 60
    while time.time() < deadline:
        if not any(a < STALE_S for a in heartbeats(spool, ntasks)):
            warn("dispatcher stopped")
            return 0
        time.sleep(1.0)
    warn("dispatcher still heartbeating after 60s; the step may need scancel by a human")
    return 1


def do_status(jobid: str, ntasks: int, spool: Path) -> int:
    ages = heartbeats(spool, ntasks)
    live = all(a < STALE_S for a in ages)
    print(f"job {jobid} ntasks={ntasks} spool={spool}")
    print(f"  live: {live}")
    for k, a in enumerate(ages):
        print(f"  rank{k} heartbeat age: {'missing' if a == float('inf') else f'{a:.1f}s'}")
    pend = sorted(spool.glob("req-*.json")) if spool.exists() else []
    runs = sorted(spool.glob("run-*.json")) if spool.exists() else []
    print(f"  queued requests: {len(pend)}   executed (run-*.json): {len(runs)}")
    if not live:
        print("  start with: python -m agent_flow.ops.dispatch_client --start")
    return 0 if live else EXIT_NO_DAEMON


class Tailer:
    """Stream a growing log file to a stream, optionally rank-prefixed."""

    def __init__(self, path: Path, sink, prefix: str = ""):
        self.path, self.sink, self.prefix = path, sink, prefix
        self.fh = None
        self.buf = b""

    def pump(self) -> None:
        if self.fh is None:
            if not self.path.exists():
                return
            self.fh = open(self.path, "rb")
        chunk = self.fh.read()
        if not chunk:
            return
        if not self.prefix:
            self.sink.buffer.write(chunk)
            self.sink.flush()
            return
        self.buf += chunk
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            self.sink.buffer.write(self.prefix.encode() + line + b"\n")
        self.sink.flush()

    def close(self) -> None:
        self.pump()
        if self.prefix and self.buf:
            self.sink.buffer.write(self.prefix.encode() + self.buf + b"\n")
            self.sink.flush()
        if self.fh:
            self.fh.close()


def dispatch(
    spool: Path,
    ntasks: int,
    cmd: list[str],
    cwd: Path,
    env: dict,
    timeout_s: float,
    out: Path | None,
    err: Path | None,
    repo: str | None = None,
) -> int:
    rid = f"{time.time():.6f}-{uuid.uuid4().hex[:8]}"
    req = {
        "id": rid,
        "argv": cmd,
        "cwd": str(cwd),
        "env": env,
        "timeout_s": timeout_s,
        "client_pid": os.getpid(),
        "created": time.time(),
    }
    if repo:
        req["repo"] = str(repo)
    spool.mkdir(parents=True, exist_ok=True)
    tmp = spool / f".req-{rid}.tmp"
    tmp.write_text(json.dumps(req))
    tmp.rename(spool / f"req-{rid}.json")  # atomic publish

    # The command's stdout+stderr are merged by the dispatcher (one child, one
    # log). --out/--err therefore get the same bytes; that is the one behaviour
    # difference from the one-shot wrapper.
    sinks = []
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        sinks.append(open(out, "wb", buffering=0))
    if err and err != out:
        err.parent.mkdir(parents=True, exist_ok=True)
        sinks.append(open(err, "wb", buffering=0))

    class Fan:
        """Write to the client's stdout and any --out/--err files at once."""

        def __init__(self, files, echo):
            self.buffer = self
            self.files = files
            self.echo = echo

        def write(self, b):
            if self.echo:
                sys.stdout.buffer.write(b)
            for f in self.files:
                f.write(b)

        def flush(self):
            if self.echo:
                sys.stdout.flush()

    # Like in_container.py: with --out/--err the command's bytes go to the
    # files only, keeping this wrapper's stderr free as an alert channel.
    sink = Fan(sinks, echo=not sinks)
    tailers = [
        Tailer(spool / f"out-{rid}.rank{k}.log", sink, "" if ntasks == 1 else f"[r{k}] ")
        for k in range(ntasks)
    ]
    rc_paths = [spool / f"rc-{rid}.rank{k}" for k in range(ntasks)]
    cancelled = False

    def on_int(_s, _f):
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            (spool / f"cancel-{rid}").write_text("1\n")
            warn(
                "\ndispatch_client: cancel sent to the dispatcher; waiting for the child to die..."
            )
        else:
            raise KeyboardInterrupt

    prev = signal.signal(signal.SIGINT, on_int)
    last_seen_alive = time.time()
    try:
        while True:
            for t in tailers:
                t.pump()
            if all(p.exists() for p in rc_paths):
                break
            if is_live(spool, ntasks):
                last_seen_alive = time.time()
            elif time.time() - last_seen_alive > GIVEUP_S:
                for t in tailers:
                    t.close()
                warn(
                    f"dispatch_client: dispatcher for {spool} stopped heartbeating "
                    f"(>{GIVEUP_S:.0f}s) while the request was in flight.",
                    "NOT falling back to a per-command srun (that is what burns steps).",
                    "Restart it with: python -m agent_flow.ops.dispatch_client --start",
                )
                return EXIT_NO_DAEMON
            time.sleep(0.1)
        for t in tailers:
            t.close()
    finally:
        signal.signal(signal.SIGINT, prev)
        for f in sinks:
            f.close()

    rcs = []
    for p in rc_paths:
        try:
            rcs.append(int(p.read_text().strip()))
        except Exception:  # noqa: BLE001
            rcs.append(125)
    rc = max(rcs)
    if cancelled:
        return 130
    if rc != 0:
        warn(
            f"dispatch_client: command exited {rc}" + (f" (per rank: {rcs})" if ntasks > 1 else "")
        )
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_flow.ops.dispatch_client", add_help=True)
    add_config_argument(ap)
    ap.add_argument("--allocation", default="", help="allocation key (default: config's)")
    ap.add_argument("--out", type=Path, help="file for the COMMAND's output")
    ap.add_argument("--err", type=Path, help="file for the COMMAND's output (merged with --out)")
    ap.add_argument("--tail", type=int, default=0)
    ap.add_argument("--jobid", default=None)
    ap.add_argument("--ntasks", type=int, default=int(os.environ.get("NTASKS", "1")))
    ap.add_argument("--time-min", type=int, default=int(os.environ.get("TIMEOUT_MIN", "240")))
    ap.add_argument("--cwd", type=Path, default=None, help="cwd for the command (default: $REPO)")
    ap.add_argument("--repo", default=None, help="run this request against another checkout")
    ap.add_argument(
        "--daemon-time-min",
        type=int,
        default=DAEMON_TIME_MIN,
        help="--time of the daemon's step when starting it",
    )
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument(
        "--no-auto-start",
        action="store_true",
        default=bool(os.environ.get("DISPATCH_NO_AUTOSTART")),
    )
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    notices.configure(cfg)

    jobid = a.jobid or alloc.resolve(cfg, a.allocation)
    spool = spool_dir(spool_root(cfg), jobid, a.ntasks)

    if a.status:
        return do_status(jobid, a.ntasks, spool)
    if a.stop:
        return do_stop(spool, a.ntasks)

    ic.check_notice()
    ic.check_job(jobid)

    if a.start:
        return do_start(cfg, jobid, a.ntasks, spool, a.daemon_time_min)

    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given (put it after --)")

    if not is_live(spool, a.ntasks):
        if a.no_auto_start:
            warn(
                f"dispatch_client: no live dispatcher for job {jobid} ntasks={a.ntasks}.",
                f"Start one (costs ONE scheduler step): python -m "
                f"agent_flow.ops.dispatch_client --ntasks {a.ntasks} --start",
                "Refusing a per-command fallback step on purpose.",
            )
            return EXIT_NO_DAEMON
        if do_start(cfg, jobid, a.ntasks, spool, a.daemon_time_min) != 0:
            return EXIT_NO_DAEMON

    # --repo is a per-request override: the dispatcher re-renders the
    # in_container prefix against it, so the interpreter, import path and cwd
    # all point at that checkout.
    env = {}
    for kv in (os.environ.get("DISPATCH_ENV") or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k.strip()] = v
    repo = a.repo or os.environ.get("OPS_REPO")
    cwd = a.cwd or Path(repo or cfg.repo)

    rc = dispatch(spool, a.ntasks, cmd, cwd, env, a.time_min * 60, a.out, a.err, repo=repo)
    if a.tail and a.out:
        ic.tail(a.out, a.tail, "output")
    return rc


if __name__ == "__main__":
    sys.exit(main())
