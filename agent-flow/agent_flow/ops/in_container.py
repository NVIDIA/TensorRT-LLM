r"""Run one command inside a held allocation's container.

Wrapper messages and command output are separated by construction:

* ``--out`` / ``--err`` send the WRAPPED COMMAND's stdout/stderr to files,
* this wrapper's own messages (job state, pending notices) always go to OUR
  stderr and are never written into those files.

That separation is not cosmetic: the wrapper's output is redirected into the
run's evidence logs, and a notice banner echoed on the command's stream once
ended up inside files the run cites as acceptance evidence. Output is
unbuffered, so a tail on the file is live.

    python -m agent_flow.ops.in_container -- python -c 'import mypkg'
    python -m agent_flow.ops.in_container --out logs/a.out --err logs/a.err \\
        --tail 20 -- pytest tests/... -q

Exit codes: the command's own, or 3 = allocation not RUNNING (wait; never
allocate your own), 4 = an unacknowledged blocking notice (read it, ack it,
retry).

Everything site-specific — image, container name, mounts, repo path, cache
directories, offline flags — comes from the ops config.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from agent_flow.ops import alloc, notices
from agent_flow.ops.config import OpsConfig, add_config_argument, config_from_args

EXIT_NOT_RUNNING = 3
EXIT_BLOCKED = 4


def warn(*lines: str) -> None:
    """Wrapper-level message. Always our stderr, never the command's streams."""
    for line in lines:
        print(line, file=sys.stderr, flush=True)


def check_notice() -> None:
    blocking = notices.blocking_pending()
    if not blocking:
        return
    warn(
        f"in_container: REFUSING TO RUN - {len(blocking)} unacknowledged blocking "
        f"notice(s) pending.",
        "Nothing was executed; this is not a failure of your command.",
        "-" * 70,
        notices.render(blocking),
        "-" * 70,
        f"Cluster commands keep failing with exit {EXIT_BLOCKED} until you acknowledge.",
    )
    raise SystemExit(EXIT_BLOCKED)


def check_job(job_id: str) -> None:
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        warn(f"in_container: could not query the scheduler ({exc}); refusing to run.")
        raise SystemExit(EXIT_NOT_RUNNING) from None
    if out != "RUNNING":
        warn(
            f"in_container: job {job_id} is {out or 'gone'!r}, not RUNNING. Wait for it "
            f"to come back; do NOT submit a new allocation, and do not record this as a "
            f"test failure."
        )
        raise SystemExit(EXIT_NOT_RUNNING)


def remote_script(cfg: OpsConfig, cmd: list[str], repo: Path | None = None) -> str:
    """Environment setup that must happen inside the container, then exec.

    Order is fixed: ``REPO``, then ``[container].env`` as plain exports, then
    ``[container].env_prologue`` verbatim (so a prologue line can reference or
    override anything above it), then ``cd $REPO`` and ``exec``.
    """
    repo = Path(repo) if repo else cfg.repo
    lines = ["set -eo pipefail", f"export REPO={shlex.quote(str(repo))}"]
    lines += [f"export {k}={v}" for k, v in cfg.container_env.items()]
    lines += list(cfg.env_prologue)
    lines += ["cd $REPO", "exec " + " ".join(shlex.quote(c) for c in cmd)]
    return "\n".join(lines) + "\n"


def srun_command(cfg: OpsConfig, job_id: str, ntasks: int, time_min: int, script: str) -> list[str]:
    """The scheduler command line for one step in the held allocation.

    The image is passed alongside the container NAME on purpose: a preempted
    allocation comes back on a different node with no containers on it, and
    attaching by name alone then fails. The container runtime attaches when
    the name exists and creates it otherwise, so passing both is safe.
    """
    cmd = [
        "srun",
        "--overlap",
        f"--jobid={job_id}",
        f"--container-name={cfg.container_name}",
    ]
    if cfg.container_image:
        cmd.append(f"--container-image={cfg.container_image}")
    if cfg.container_mounts:
        cmd.append("--container-mounts=" + ",".join(cfg.container_mounts))
    cmd += [f"--time={time_min}", f"--ntasks={ntasks}", "bash", "-c", script]
    return cmd


def tail(path: Path, n: int, label: str) -> None:
    if not path.exists():
        return
    lines = path.read_text(errors="ignore").splitlines()[-n:]
    if lines:
        warn(f"--- last {len(lines)} lines of {label} ({path}) ---", *lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="agent_flow.ops.in_container",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(ap)
    ap.add_argument("--out", type=Path, help="file for the COMMAND's stdout")
    ap.add_argument("--err", type=Path, help="file for the COMMAND's stderr")
    ap.add_argument(
        "--tail",
        type=int,
        default=0,
        help="after the run, echo the last N lines of --out/--err to this wrapper's "
        "stderr (handy inside an agent tool call)",
    )
    ap.add_argument("--allocation", default="", help="allocation key (default: config's)")
    ap.add_argument("--jobid", default="")
    ap.add_argument("--repo", type=Path, default=None, help="override the checkout to run from")
    ap.add_argument("--ntasks", type=int, default=int(os.environ.get("NTASKS", "1")))
    ap.add_argument("--time-min", type=int, default=int(os.environ.get("TIMEOUT_MIN", "240")))
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run (put it after --)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    notices.configure(cfg)

    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given (put it after --)")

    check_notice()
    job_id = a.jobid or alloc.resolve(cfg, a.allocation)
    if not job_id:
        warn("in_container: no job id (config has no allocation and none was given)")
        return EXIT_NOT_RUNNING
    check_job(job_id)

    srun = srun_command(cfg, job_id, a.ntasks, a.time_min, remote_script(cfg, cmd, a.repo))

    fout = ferr = None
    try:
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            fout = open(a.out, "wb", buffering=0)
        if a.err:
            a.err.parent.mkdir(parents=True, exist_ok=True)
            ferr = open(a.err, "wb", buffering=0)
        rc = subprocess.call(srun, stdout=fout or None, stderr=ferr or None)
    finally:
        for f in (fout, ferr):
            if f:
                f.close()

    if a.tail:
        if a.out:
            tail(a.out, a.tail, "stdout")
        if a.err:
            tail(a.err, a.tail, "stderr")
    if rc != 0:
        warn(f"in_container: command exited {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
