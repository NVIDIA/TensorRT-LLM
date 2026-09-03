"""The one place this workflow calls ``bench-disagg`` from Python.

A SOL-track campaign does not drive ``bench-trtllm-disagg``'s scripts. It
drives the CLI that repo ships, and that CLI's own README is explicit
about why: "`bench-disagg` is the only supported agent-facing interface.
The existing Python scripts are internal execution backends, not
compatibility commands."

**The agent drives it, not this module.** ``sweep submit``, ``sweep
status``, ``frontier build`` and ``frontier compare`` are Bash commands
in the SOL-track prompt sections, run by the role that needs them — the
same way an aggregate campaign runs ``trtllm-serve`` and
``benchmark_serving``. Wrapping them in Python would add a layer with no
caller and one more place for the command surface to drift.

Python needs the CLI for the three questions an agent cannot be trusted
with, all read-only and none of them queueing anything:

- ``sweep plan``, at schema validation, before any agent exists. That
  call answers what the sweep expands to — the operating points and the
  sequence lengths — so ``task.yaml`` can be reconciled against the run
  that will actually happen.
- ``frontier show``, after a measurement, so the score lands on disk in
  the shape the rest of perf-optimize already reads. Both ends of that
  translation are vocabulary — ``tps_per_user`` here,
  ``throughput_per_user`` there, a listed concurrency here, a total in
  flight there — and vocabulary that only exists in a prompt is
  vocabulary that gets it right most of the time.
- ``sweep status``, for the ctx track only, because nothing else reports
  its metric: a frontier snapshot is the rate-matched *generation* curve
  and takes the ctx side as its anchor, so it holds no ctx point at all.

The contract, verified against the installed 0.2.0 CLI rather than read
off its source:

- every leaf command prints exactly one JSON object on stdout, because
  the CLI redirects all subordinate output to stderr to keep it that way;
- that object has ``ok``, and on failure an ``error.code`` from a fixed
  taxonomy ("Agents branch on these; messages are for humans");
- the exit code agrees with ``ok`` (0 / 2).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The console script the wheel installs.
BENCH_DISAGG = "bench-disagg"


class BenchCliError(RuntimeError):
    """A `bench-disagg` command failed, or did not answer in the protocol.

    ``code`` is the CLI's own taxonomy value when there was a well-formed
    envelope and ``None`` when there was not — the distinction matters,
    because the second case means the contract itself is broken and no
    amount of retrying will fix it.
    """

    def __init__(self, message: str, code: str | None = None, details: Any = None):
        super().__init__(message)
        self.code = code
        self.details = details


def executable() -> str:
    """Where to find the CLI: PATH first, then beside this interpreter.

    The fallback matters because the two are installed together — the
    console script lands in the same ``bin/`` as the ``python`` running
    this code — but only an *activated* venv puts that directory on PATH.
    Failing on that difference would be failing on an environment
    variable rather than on a missing dependency, which is what it did the
    first time this ran for real.
    """
    found = shutil.which(BENCH_DISAGG)
    if found:
        return found
    sibling = Path(sys.executable).with_name(BENCH_DISAGG)
    return str(sibling) if sibling.is_file() else BENCH_DISAGG


def run(argv: Sequence[str], *, timeout: float | None = None) -> dict[str, Any]:
    """Run one `bench-disagg` command and return its ``data`` block.

    Raises :class:`BenchCliError` on any non-``ok`` envelope, carrying the
    CLI's own error code so a caller can branch on it.
    """
    try:
        completed = subprocess.run(
            [executable(), *argv], capture_output=True, text=True, check=False, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise BenchCliError(
            f"{BENCH_DISAGG} is not on PATH. A SOL-track campaign drives it rather "
            f"than the benchmark repo's scripts: `pip install trtllm-disagg-bench` "
            f"into the environment this workflow runs in."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise BenchCliError(f"{BENCH_DISAGG} {' '.join(argv)} timed out") from exc

    text = completed.stdout.strip()
    if not text:
        raise BenchCliError(
            f"{BENCH_DISAGG} {' '.join(argv)} printed nothing on stdout; the CLI "
            f"emits one JSON envelope per command, so an empty one means it did "
            f"not run (wrong PATH / venv) rather than that it found nothing"
        )
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchCliError(
            f"{BENCH_DISAGG} {' '.join(argv)} did not print a JSON envelope "
            f"(starts: {text[:200]!r}). Something bypassed the CLI's stdout protection."
        ) from exc

    if not envelope.get("ok"):
        error = envelope.get("error") or {}
        raise BenchCliError(
            error.get("message") or f"{BENCH_DISAGG} {' '.join(argv)} failed",
            code=error.get("code"),
            details=error.get("details"),
        )
    return dict(envelope.get("data") or {})


def plan(sweep: str | Path, workspace: str) -> dict[str, Any]:
    """Survey a sweep: the workload, the code identity, and every case."""
    return run(["sweep", "plan", "--workspace", workspace, "-c", str(sweep), "--cases"])


def frontier_show(workspace: str, snapshot: str = "latest") -> dict[str, Any]:
    """Read a built snapshot back: its points, and what each one scored.

    The second and last thing Python asks the CLI for, and for the same
    reason as :func:`plan` — read-only, queues nothing, and the answer has
    to come from the tool that computed it rather than from a file this
    workflow parses itself. Resolving ``latest`` in particular is not a
    directory listing: the CLI skips snapshots whose build did not
    complete, which is exactly the case where reading the newest directory
    would silently score a campaign on a failed postprocess.

    ``snapshot`` takes a snapshot id, ``"latest"``, or ``"baseline"``.
    """
    return run(["frontier", "show", "--workspace", workspace, "--snapshot", snapshot])


def status(workspace: str) -> dict[str, Any]:
    """Every case in a workspace, with where its artifacts landed.

    The third and last read-only question Python asks. It exists for the
    ctx track, which no other command scores: ``frontier build`` selects
    ``stage == GEN`` and refuses a workspace holding none, so a ctx
    campaign has no snapshot to read.

    This is not reaching around the CLI. Each case carries ``result`` —
    the artifact the backend *validated the case on* — and for a ctx case
    that is the ``run_*.json`` whose ``performance.request_throughput_req_s``
    is the very key ``_inspect_ctx`` required before calling the case
    successful. The CLI hands over the path and certifies the key is in
    it; reading it is following that contract, not going behind it.
    """
    return run(["sweep", "status", "--workspace", workspace, "--cases"])


def operating_point(config: Mapping[str, Any]) -> int | None:
    """Total requests in flight at one case, or ``None`` if unstated.

    Not ``config.concurrency`` verbatim. That is the sweep row's listed
    value, which is **per generation server**: the client is driven at
    ``concurrency * gen_num`` and the harness names its result directory
    after that product. ``task.yaml``'s ``concurrency`` means what an
    aggregate campaign means by it — total requests in flight — so the
    product is what belongs there, and the case name (which uses the
    listed value) stays the address.

    This is the one trap driving the CLI does not remove: it belongs to
    the harness, not to the file the numbers were read from. It applies
    identically to a planned case and to a measured frontier point, since
    both carry the sweep row's ``config`` unchanged — which is why the
    rule lives here rather than at either call site.
    """
    listed = config.get("concurrency")
    if not isinstance(listed, int) or isinstance(listed, bool):
        # A ctx case has no `concurrency`: its config is
        # `{isl, osl, max_batch, tp_size, ratio, mtp}`. `max_batch` is the
        # in-flight request count for a prefill-only run -- the server admits
        # that many and no more -- which is what `concurrency` means
        # everywhere else in this workflow. Reading it as the operating point
        # is not a convention invented here; it is what the script-driven
        # implementation reconciled against, and its ctx campaign ran to a
        # finished report on that basis.
        batch = config.get("max_batch")
        if isinstance(batch, int) and not isinstance(batch, bool):
            return batch
        return None
    gen_num = config.get("gen_num", 1)
    if not isinstance(gen_num, int) or isinstance(gen_num, bool) or gen_num < 1:
        gen_num = 1
    return listed * gen_num


def operating_points(plan_data: Mapping[str, Any]) -> list[int]:
    """The concurrency axis a `task.yaml` should carry, from a plan."""
    points = (operating_point(case.get("config") or {}) for case in plan_data.get("cases") or [])
    return sorted({point for point in points if point is not None})
