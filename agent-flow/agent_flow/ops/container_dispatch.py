"""Persistent in-container command dispatcher (daemon side).

WHY: every ``srun`` into a held allocation is a scheduler *step*, and an
allocation dies after a few hundred of them. A one-shot wrapper spends one step
per command, including one-second GPU probes, so a day of polling kills the
allocation. This
daemon spends ONE step per (jobid, ntasks) pair and then executes an unbounded
number of commands inside it, talking to clients through a spool directory on
the shared filesystem.

One process per rank (the step is launched with ``--ntasks=<n>``), all ranks
running the same loop over the same spool dir:

    <spool>/req-<ts>-<uuid>.json   client -> dispatcher, written atomically
    <spool>/run-<seq>.json         rank 0 claims a request by renaming it
    <spool>/out-<id>.rank<k>.log   merged stdout+stderr of the child
    <spool>/rc-<id>.rank<k>        exit code, written LAST (completion marker)
    <spool>/cancel-<id>            client -> dispatcher, SIGTERM then SIGKILL
    <spool>/alive.rank<k>          heartbeat, touched every ~5 s
    <spool>/stop                   clean shutdown of every rank

The rename to ``run-<seq>.json`` is both the claim and the multi-rank barrier:
rank 0 is the only writer, and rank k>0 simply waits for the file named with
its own next sequence number. Because every rank executes requests strictly in
sequence order, ranks cannot diverge, and no lock is needed on the shared
filesystem (where cross-node flock is not something to bet a run on).

Requests run sequentially per daemon. A GPU engine run therefore blocks the
queue for its duration; that is intentional and cheap to work around (use a
different (jobid, ntasks) pair for a second lane), not something to fix with
concurrency here.

The child's environment is built by ``in_container.remote_script`` -- the same
setup the one-shot wrapper uses -- so the repo path, caches and per-rank
scratch dirs are identical. A per-request repo override is honoured by
re-rendering that prefix against the other checkout, so the interpreter and
import path really do change per request even though the container, image and
mounts are fixed at daemon start.

Run inside the container, e.g.:

    python3 -m agent_flow.ops.container_dispatch --spool <dir> --ntasks 4
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from agent_flow.ops import in_container
from agent_flow.ops.config import add_config_argument, config_from_args

HEARTBEAT_S = 5.0
POLL_S = 0.25
DEFAULT_TIMEOUT_S = 240 * 60
GPU_POLL_S = float(os.environ.get("DISPATCH_GPU_POLL_S", "30"))
GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def log(rank: int, msg: str) -> None:
    print(f"[dispatch r{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def request_name(spool: Path) -> Path:
    """Client-side helper: a fresh request path (FIFO by name -> by time)."""
    return spool / f"req-{time.time():.6f}-{uuid.uuid4().hex[:8]}.json"


CFG = None  # set in main(); the daemon runs with one fixed config
REPO_ENV = "OPS_REPO"


def child_script(req: dict) -> str:
    """Bash script for one request: in_container's prefix, then cwd/env/argv.

    ``in_container.remote_script`` bakes the repo path into the text, so a
    per-request repo override is applied by rendering the prefix against that
    checkout. Everything after the prefix (cd, env exports, exec) overrides it,
    which is why cwd and extra env win over the defaults.
    """
    inner = []
    cwd = req.get("cwd")
    if cwd:
        inner.append(f"cd {shlex.quote(str(cwd))}")
    for k, v in (req.get("env") or {}).items():
        inner.append(f"export {k}={shlex.quote(str(v))}")
    inner.append("exec " + " ".join(shlex.quote(c) for c in req["argv"]))
    body = "\n".join(inner)

    repo = req.get("repo") or (req.get("env") or {}).get(REPO_ENV)
    return in_container.remote_script(CFG, ["bash", "-c", body], Path(repo) if repo else None)


def run_request(spool: Path, rank: int, req: dict) -> int:
    rid = req["id"]
    out_path = spool / f"out-{rid}.rank{rank}.log"
    cancel = spool / f"cancel-{rid}"
    timeout = float(req.get("timeout_s") or DEFAULT_TIMEOUT_S)
    argv_s = " ".join(req["argv"])
    log(rank, f"run {rid}: {argv_s[:160]}")

    rc = 127
    started = time.time()
    with open(out_path, "wb", buffering=0) as fout:
        try:
            proc = subprocess.Popen(
                ["bash", "-c", child_script(req)],
                stdout=fout,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # own process group -> kill the tree
            )
        except Exception as exc:  # noqa: BLE001
            fout.write(f"dispatcher: failed to start child: {exc}\n".encode())
            _write_rc(spool, rid, rank, 126)
            return 126
        killed = False
        last_beat = 0.0
        last_gpu = 0.0
        while True:
            try:
                rc = proc.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.time()
            if now - last_beat > HEARTBEAT_S:  # keep the heartbeat fresh while busy
                _beat(spool, rank)
                last_beat = now
            # Keep gpu.json fresh DURING a long GPU run -- that is exactly when
            # a dashboard wants it, and the idle-loop sampler is blocked here.
            if rank == 0 and now - last_gpu > GPU_POLL_S:
                write_gpu(spool, gpu_sample())
                last_gpu = time.time()
            if not killed and (cancel.exists() or now - started > timeout):
                why = "cancelled" if cancel.exists() else f"timeout after {timeout:.0f}s"
                fout.write(f"\ndispatcher: {why}; SIGTERM to process group\n".encode())
                _signal_group(proc, signal.SIGTERM)
                killed = True
                deadline = now + 10
            elif killed and time.time() > deadline:
                fout.write(b"dispatcher: SIGKILL to process group\n")
                _signal_group(proc, signal.SIGKILL)
                deadline = time.time() + 3600
    if rc < 0:  # Popen.wait() reports a signal death as -N; shell convention is 128+N
        rc = 128 - rc
    if killed and rc == 0:
        rc = 143
    _write_rc(spool, rid, rank, rc)
    log(rank, f"done {rid}: rc={rc} in {time.time() - started:.1f}s")
    return rc


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except Exception:  # noqa: BLE001
        try:
            proc.send_signal(sig)
        except OSError:
            pass


def _write_rc(spool: Path, rid: str, rank: int, rc: int) -> None:
    """Written last and atomically: the client treats it as 'request done'."""
    tmp = spool / f".rc-{rid}.rank{rank}.tmp"
    tmp.write_text(f"{rc}\n")
    tmp.rename(spool / f"rc-{rid}.rank{rank}")


def _beat(spool: Path, rank: int) -> None:
    (spool / f"alive.rank{rank}").write_text(f"{time.time():.3f} pid={os.getpid()}\n")


def gpu_sample() -> dict:
    """One nvidia-smi query, parsed. Never raises: errors become the payload.

    Read-only query mode: no CUDA context, so this cannot disturb whatever is
    using the GPUs. It lives here rather than in the client because a probe
    from outside would cost a SLURM step every time -- the whole point.
    """
    try:
        r = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return {
                "at": time.time(),
                "error": f"nvidia-smi rc={r.returncode}: {r.stderr.strip()[:300]}",
            }
        gpus = []
        for line in r.stdout.splitlines():
            parts = [c.strip() for c in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "mem_used_mib": int(parts[1]),
                        "mem_total_mib": int(parts[2]),
                        "util_pct": int(parts[3]),
                    }
                )
            except ValueError:
                return {"at": time.time(), "error": f"unparsable nvidia-smi row: {line[:200]}"}
        if not gpus:
            return {"at": time.time(), "error": "nvidia-smi returned no GPU rows"}
        return {"at": time.time(), "gpus": gpus}
    except Exception as exc:  # noqa: BLE001
        return {"at": time.time(), "error": f"{type(exc).__name__}: {exc}"}


def write_gpu(spool: Path, payload: dict) -> None:
    """Atomic publish to <spool>/gpu.json plus a job-level copy one dir up.

    The job-level copy means a dashboard needs only the job id, not the ntasks
    of whichever daemon happens to be up. A copy, not a symlink: a symlink into
    n<ntasks>/ would dangle as soon as that daemon's spool is cleaned.
    """
    blob = json.dumps(payload)
    for target in (spool / "gpu.json", spool.parent / "gpu.json"):
        try:
            tmp = target.with_name(f".{target.name}.tmp{os.getpid()}")
            tmp.write_text(blob)
            tmp.rename(target)
        except OSError:
            pass


def claim_next(spool: Path, seq: int) -> Path | None:
    """Rank 0 only: oldest pending request -> run-<seq>.json (claim+barrier)."""
    reqs = sorted(spool.glob("req-*.json"), key=lambda p: (p.stat().st_mtime, p.name))
    for p in reqs:
        target = spool / f"run-{seq}.json"
        try:
            p.rename(target)
        except OSError:
            continue
        return target
    return None


def loop(spool: Path, rank: int, ntasks: int) -> int:
    spool.mkdir(parents=True, exist_ok=True)
    _beat(spool, rank)
    log(rank, f"up: spool={spool} ntasks={ntasks} host={os.uname().nodename} pid={os.getpid()}")
    seq = 0
    # Resume after a daemon restart in an existing spool dir.
    while (spool / f"run-{seq}.json").exists():
        seq += 1
    stop = spool / "stop"
    last_beat = 0.0
    last_gpu = 0.0
    while True:
        if time.time() - last_beat > HEARTBEAT_S:
            _beat(spool, rank)
            last_beat = time.time()
        # GPU heartbeat: rank 0 only, so no dashboard ever needs a probe step.
        if rank == 0 and time.time() - last_gpu > GPU_POLL_S:
            write_gpu(spool, gpu_sample())
            last_gpu = time.time()
        if stop.exists():
            log(rank, "stop file seen; exiting")
            return 0
        run_file = spool / f"run-{seq}.json"
        if rank == 0 and not run_file.exists():
            claim_next(spool, seq)
        if not run_file.exists():
            time.sleep(POLL_S)
            continue
        try:
            req = json.loads(run_file.read_text())
        except Exception as exc:  # noqa: BLE001
            log(rank, f"bad request {run_file.name}: {exc}; skipping")
            seq += 1
            continue
        try:
            run_request(spool, rank, req)
        except Exception as exc:  # noqa: BLE001
            log(rank, f"request {req.get('id')} blew up in the dispatcher: {exc}")
            try:
                _write_rc(spool, req.get("id", f"seq{seq}"), rank, 125)
            except Exception:  # noqa: BLE001
                pass
        seq += 1
        last_beat = 0.0


def main(argv: list[str] | None = None) -> int:
    global CFG
    ap = argparse.ArgumentParser(
        prog="agent_flow.ops.container_dispatch", description=__doc__.splitlines()[0]
    )
    add_config_argument(ap)
    ap.add_argument("--spool", type=Path, required=True)
    ap.add_argument("--ntasks", type=int, default=int(os.environ.get("SLURM_NTASKS", "1")))
    a = ap.parse_args(argv)
    CFG = config_from_args(a)
    rank = int(os.environ.get("SLURM_PROCID", "0"))

    stopping = {"v": False}

    def on_term(signum, _frame):
        stopping["v"] = True
        log(rank, f"signal {signum}; exiting")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    try:
        return loop(a.spool, rank, a.ntasks)
    finally:
        try:
            (a.spool / f"alive.rank{rank}").unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
