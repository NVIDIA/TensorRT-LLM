# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The allocation probe: is a holder of this shape usable, before any model is loaded on it.

It starts ``checks/harness/alloc_probe.py`` exactly as an engine of that shape is started and
judges what the ranks wrote: the launch made the world it was asked for, the build imports on
every rank, a collective crossed all of them, the cache directories take a file, the dataset
is there. A world that fits one node is a job of that many devices under that node's watcher
(``mpirun``); a wider one is a multi-node job whose script the holder's lead runs once, with
the environment's step line across the reserved hosts.

It loads no model and costs seconds, so it can run for each new holder and each shape. It
also holds every watcher to one thing the Clean rule stands on: a watcher that never read its devices can never
call one free, so that is said here, in one line, before anything waits on it.
"""

from __future__ import annotations

import json
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import pintree, spoolctl
from .commands import CommandSpec
from .environment import RunEnvironment
from .executor import select_command_executor
from .scheduler import WATCHER_STALE_S, SpoolPaths
from .server_runtime import rank_launch
from .slurm import inner_script

#: The probe a rank runs, by path: the interpreter that runs it is the engine's.
PROBE_PATH = Path(__file__).resolve().parents[2] / "checks" / "harness" / "alloc_probe.py"
#: How long the probe's job may run once it started; queue time is not counted.
PROBE_WALL_S = 600
ROLE = "probe"


@dataclass
class ProbeResult:
    """What the probe found for one shape."""

    ranks: int
    nodes: int
    #: One line per thing that is wrong; none when the shape is usable.
    problems: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    log: str = ""

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self) -> str:
        shape = f"{self.ranks} rank(s) on {self.nodes} node(s)"
        if self.ok:
            # What it stands for: a launch, the imports, one MPI collective. Said with it.
            return (
                f"allocation probe: {shape} ok ({', '.join(self.hosts) or 'no host named'}); "
                "not verified: the devices, the transport between them, that these are the "
                "hosts reserved"
            )
        return f"allocation probe: {shape} failed: " + "; ".join(self.problems)


def probe_arguments(
    out: Path,
    ranks: int,
    *,
    modules: Sequence[str],
    cache_dirs: Sequence[str],
    datasets: Sequence[str],
) -> list[str]:
    argv = [str(PROBE_PATH), "--out", str(out), "--expect-ranks", str(ranks), "--role", ROLE]
    for flag, values in (
        ("--module", modules),
        ("--cache-dir", cache_dirs),
        ("--dataset", datasets),
    ):
        for value in values:
            argv += [flag, str(value)]
    return argv


def judge(out: Path, ranks: int) -> tuple[list[str], list[str]]:
    """``(problems, hosts)`` of the records the ranks left in ``out``.

    The world passes only when every world rank reported **exactly once**, each says it is in
    a world of exactly ``ranks``, and none has a problem of its own. A rank that wrote nothing,
    one that was reported twice, and a record of another world are each named.
    """
    seen: dict[int, list[dict]] = {}
    problems = []
    for path in sorted(out.glob(f"{ROLE}.rank*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            problems.append(f"{path.name} is no record")
            continue
        rank = record.get("rank") if isinstance(record, dict) else None
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < ranks:
            problems.append(f"{path.name} claims world rank {rank!r}, outside a world of {ranks}")
            continue
        seen.setdefault(rank, []).append(record)
    missing = [rank for rank in range(ranks) if rank not in seen]
    if missing:
        problems.append(
            f"no record from rank(s) {', '.join(map(str, missing))} of {ranks}: the launch did "
            f"not start one process per rank"
        )
    for rank, records in sorted(seen.items()):
        if len(records) > 1:
            where = ", ".join(str(r.get("host")) for r in records)
            problems.append(
                f"world rank {rank} was reported {len(records)} times (on {where}): the launch "
                f"made more than one world"
            )
        for record in records:
            if record.get("size") != ranks:
                problems.append(
                    f"rank {rank} on {record.get('host')} is in a world of {record.get('size')}, "
                    f"{ranks} were asked for"
                )
            for problem in record.get("problems") or ():
                problems.append(f"rank {rank} on {record.get('host')}: {problem}")
    hosts = sorted({str(r.get("host")) for records in seen.values() for r in records})
    return list(dict.fromkeys(problems)), hosts


def watchers_without_reading(spool: str | Path, now: float | None = None) -> list[str]:
    """One line per live watcher that has devices and never read them: none of them is ever free."""
    now = time.time() if now is None else now
    lines = []
    for watcher in SpoolPaths(spool).live_watchers(WATCHER_STALE_S, now):
        if watcher.get("lead") or not watcher.get("devices"):
            continue
        if watcher.get("device_reading_epoch") is None:
            lines.append(
                f"watcher {watcher.get('watcher')} on {watcher.get('host')} has no device "
                f"reading (nvidia-smi fails or names no device there): a device that cannot be "
                f"read is never free, so no GPU job would start on it"
            )
    return lines


def run_allocation_probe(
    environment: RunEnvironment,
    *,
    ranks: int,
    nodes: int = 1,
    out: Path,
    modules: Sequence[str] = ("tensorrt_llm",),
    cache_dirs: Sequence[str] = (),
    datasets: Sequence[str] = (),
    tree: str | Path | None = None,
    pin: str | Path | None = None,
    wall_s: int = PROBE_WALL_S,
    note: Callable[[str], None] | None = None,
) -> ProbeResult:
    """Run the probe as ``ranks`` ranks on ``nodes`` nodes through the run's spool, and judge it."""
    out = Path(out) / f"{ranks}x{nodes}-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    argv = probe_arguments(out, ranks, modules=modules, cache_dirs=cache_dirs, datasets=datasets)
    result = ProbeResult(ranks=ranks, nodes=nodes)
    if nodes <= 1:
        launch = rank_launch(ranks, launcher=environment.python) or [environment.python]
        command = f"{environment.prefix} {shlex.join([*launch, *argv])}".strip()
        executor = select_command_executor(
            environment, gpus=ranks, name_prefix="alloc-probe", tree=tree, pin=pin, note=note
        )
        done = executor.execute(CommandSpec(argv=command, timeout_s=wall_s))
        result.log = done.stdout
        if done.infrastructure:
            result.problems.append(f"the probe's job has no result: {done.infrastructure}")
        elif done.returncode != 0:
            # Records a failed launch left behind prove nothing: the launch must end clean.
            result.problems.append(f"the probe's job exited {done.returncode}")
    else:
        result.log, lost = _run_across_nodes(
            environment, argv, ranks, nodes, tree, wall_s, note, pin=pin
        )
        if lost:
            result.problems.append(lost)
    problems, result.hosts = judge(out, ranks)
    result.problems += problems
    if nodes > 1 and not result.problems and len(result.hosts) != nodes:
        result.problems.append(
            f"the ranks ran on {len(result.hosts)} host(s) ({', '.join(result.hosts)}), not on "
            f"{nodes}: the step did not spread the world over the reserved nodes"
        )
    if environment.spool:
        result.problems += watchers_without_reading(environment.spool)
    return result


def _run_across_nodes(
    environment: RunEnvironment,
    argv: Sequence[str],
    ranks: int,
    nodes: int,
    tree: str | Path | None,
    wall_s: int,
    note: Callable[[str], None] | None,
    pin: str | Path | None = None,
) -> tuple[str, str]:
    """Queue the probe as a multi-node job the lead runs once; ``(log, why it has no result)``.

    With ``pin`` the job names it, and each rank's step makes (or finds) the pin's tree on its
    own node through the pin's prelude, instead of entering a tree by its path.
    """
    paths = SpoolPaths(environment.spool)
    paths.ensure()
    name = f"alloc-probe-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
    inner = paths.jobs / name / "probe.sh"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text(
        inner_script(
            shlex.join([environment.python, *argv]),
            python=environment.python,
            tree=str(tree) if tree and not pin else None,
            env=environment.launch_env if ranks == 1 else environment.env,
            role=ROLE,
        ),
        encoding="utf-8",
    )
    command = f"bash {shlex.quote(str(inner))}"
    if pin:
        prelude = pintree.step_argv(pin, name, paths.tree_log(name), environment.python)
        command = f"{shlex.join(prelude)} {command}"
    step = environment.step(
        ranks=ranks,
        nodes=nodes,
        hosts='"$MV2FLOW_JOB_HOSTS"',
        command=command,
    )
    token = uuid.uuid4().hex
    header: Mapping[str, object] = {
        "nodes": nodes,
        "wall": wall_s,
        "owner": "alloc-probe",
        "token": token,
        **({"pin": str(pin)} if pin else {}),
    }
    spoolctl.queue_job(
        paths.root,
        name,
        spoolctl.render_job(step, header=header),
        owner="alloc-probe",
        require_watcher=False,
    )
    try:
        record = spoolctl.wait_for_job(
            paths.root, name, token=token, timeout_s=wall_s, unwatched_s=wall_s, note=note
        )
    except spoolctl.SpoolError as exc:
        spoolctl.cancel_job(paths.root, name)
        return "", f"the probe's job has no result: {exc}"
    log = paths.log(name)
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    rc = int(record.get("rc", 1))
    if rc != 0 and record.get("reason") != "exit":
        return text, f"the probe's job ended {record.get('reason')}: {record.get('detail') or ''}"
    return text, f"the probe's job exited {rc}" if rc != 0 else ""


__all__ = [
    "PROBE_PATH",
    "PROBE_WALL_S",
    "ProbeResult",
    "judge",
    "probe_arguments",
    "run_allocation_probe",
    "watchers_without_reading",
]
