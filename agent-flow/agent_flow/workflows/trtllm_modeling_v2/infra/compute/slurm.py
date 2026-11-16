# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slurm's filling of the pool's slots: a holder, a step across its nodes, its end.

:mod:`.pool` decides *when* a holder is needed; this module is *how* on Slurm, and it holds only
what is public about Slurm: how a holder's batch script, its watcher step and a PMIx step are
built, which flags they carry, and how a submission's answer and a job's state are read. The
values (account, partition, QoS, image, mounts) are site facts: they arrive in
``environment.slurm`` from the user's spec or a site file, and nothing here names one.

A holder is one batch job::

    #SBATCH --job-name=<name> --account=... --partition=... --time=... --nodes=N --gpus-per-node=G
    srun --mpi=none --nodes=N --ntasks=N --ntasks-per-node=1 <container flags> bash watcher_entry.sh &
    exec bash lead_entry.sh

The watcher step runs one watcher per node inside the container and **carries every mount**: a
named container keeps the mounts of the step that created it, and a later step that attaches to
it cannot add one. The lead runs on the batch host, outside the container, where a step across
the holder's nodes can be started::

    srun --overlap --mpi=pmix --kill-on-bad-exit=1 --nodes=M --ntasks=R --ntasks-per-node=R/M
         --nodelist=<hosts> --container-image=... --container-name=... bash <inner script>

``--overlap`` lets the step share nodes and devices with the watcher step that already holds
them; a world of one rank is started with ``--mpi=none``. Every rank sees all devices of its
node, so the inner script (:func:`inner_script`) sets the devices of each role instance, and
``PATH`` and ``PYTHONPATH``, inside the launched shell: ``PATH`` set on the batch host does not
reach a container step.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ...spec.schema_keys import reject_unknown_keys
from .pool import DEFAULT_MAX_NODE_HOURS, ENDED, ENDING, PENDING, RUNNING, Reply, Row
from .scheduler import parse_duration

#: Keys every Slurm run states, each with the one line that says how to find its value.
REQUIRED_KEYS: dict[str, str] = {
    "account": (
        "the account allocations are charged to (`sacctmgr show assoc user=$USER "
        "format=account,partition,qos` lists yours)"
    ),
    "partition": ('the GPU partition (`sinfo -o "%P %G %l"` lists partitions, devices and limits)'),
    "time": "one allocation's limit; the longest job's wall plus 10 min must fit in it",
    "gpus_per_node": 'devices on one node of the partition (`sinfo -p <partition> -o "%G"`)',
}
OPTIONAL_KEYS: tuple[str, ...] = (
    "qos",
    "container_image",
    "container_mounts",
    "container_name",
    "host_python",
    "sbatch_args",
    "srun_args",
    "lead",
    # The node-hours the run's holders may ask the scheduler for.
    "max_node_hours",
)

#: The expected queue wait: how long before the last holder stops fitting a successor is asked for.
DEFAULT_LEAD_S = 1800.0

#: Flags the flow sets itself (from its own keys of ``environment.slurm``, the shape of the
#: work, the renewal rule, the spool): one in ``sbatch_args`` or ``srun_args`` is refused.
OWNED_SBATCH = frozenset(
    "--job-name -J --nodes -N --ntasks -n --ntasks-per-node --nodelist -w --time -t --begin -b "
    "--dependency -d --output -o --error -e --chdir -D --account -A --partition -p --qos -q "
    "--gpus-per-node --requeue --no-requeue --wrap --parsable".split()
)
OWNED_SRUN = frozenset(
    "--mpi --overlap --container-image --container-mounts --container-name --nodes -N --ntasks "
    "-n --ntasks-per-node --nodelist -w --output -o --chdir -D --job-name -J --jobid --time "
    "-t".split()
)

#: How long one Slurm command may take before its answer is counted as not received.
COMMAND_TIMEOUT_S = 120.0

#: What closes a submission: Slurm's own refusal of the request, or an error that says nothing
#: left this host. Only these are a refusal. Any other failure (a reply that timed out, an
#: answer nobody expected) leaves the submission unknown: the job may exist.
REFUSAL = re.compile(
    r"Invalid (account|qos|partition|generic resource|gres|feature|node|reservation|time|wckey|"
    r"license|job credential|argument|directive)|"
    r"violates accounting|QOS policy|QOS\w*Limit|Assoc\w*Limit|"
    r"Requested (node configuration|partition configuration|time limit|reservation)|"
    r"not permitted|permission denied|Access denied|"
    r"Node count specification invalid|Job dependency problem|"
    r"More processors requested than permitted|Memory specification can not be satisfied|"
    r"Unable to open file|No such file or directory|unrecognized option|invalid option|"
    r"Script arguments not permitted|"
    r"Unable to contact slurm controller \(connect failure\)|Connection refused",
    re.IGNORECASE,
)

#: Variables of the caller's environment that would set what the flow sets itself. Slurm reads
#: them over a script's directives, so they are dropped from a submission's environment, and the
#: owned values go on the command line, which wins over both.
OWNED_ENV = frozenset(
    "SBATCH_JOB_NAME SBATCH_NODES SBATCH_NTASKS SBATCH_NTASKS_PER_NODE SBATCH_NODELIST "
    "SBATCH_TIMELIMIT SBATCH_TIME SBATCH_BEGIN SBATCH_DEPENDENCY SBATCH_OUTPUT SBATCH_ERROR "
    "SBATCH_CHDIR SBATCH_ACCOUNT SBATCH_PARTITION SBATCH_QOS SBATCH_GPUS_PER_NODE "
    "SBATCH_ARRAY_INX SBATCH_WAIT SBATCH_WAIT_ALL_NODES SBATCH_NO_REQUEUE SBATCH_REQUEUE "
    "SBATCH_WRAP".split()
)
#: The same for a question: how a listing is selected and formatted is the flow's.
_OWNED_ENV_PREFIXES = ("SQUEUE_", "SACCT_")
#: What a Slurm client reads of ``SLURM_*`` wherever it runs; the rest describes one job.
_CLIENT_ENV = ("SLURM_CONF", "SLURM_CLUSTERS")

_STATES: dict[str, str] = {
    "PENDING": PENDING,
    "REQUEUED": PENDING,
    "REQUEUE_HOLD": PENDING,
    "REQUEUE_FED": PENDING,
    "CONFIGURING": RUNNING,
    "RUNNING": RUNNING,
    "SUSPENDED": RUNNING,
    "RESIZING": RUNNING,
    "SIGNALING": RUNNING,
    "STAGE_OUT": RUNNING,
    "COMPLETING": ENDING,
}

_JOB_ID = re.compile(r"\d+(;[\w.-]+)?")

Run = Callable[..., "subprocess.CompletedProcess[str]"]


def slurm_time_seconds(text: Any) -> float:
    """Seconds of a Slurm time limit: minutes, ``MM:SS``, ``HH:MM:SS`` or ``D-HH[:MM[:SS]]``."""
    value = str(text).strip()
    days, dash, rest = value.partition("-")
    if dash:
        parts = [int(p) for p in rest.split(":")]
        if not days.isdigit() or not 1 <= len(parts) <= 3:
            raise ValueError(f"not a Slurm time: {text!r}")
        hours, minutes, seconds = (parts + [0, 0])[:3]
        return int(days) * 86400.0 + hours * 3600 + minutes * 60 + seconds
    parts = [int(p) for p in value.split(":")]
    if len(parts) == 1:
        return parts[0] * 60.0
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60 + parts[2]
    raise ValueError(f"not a Slurm time: {text!r}")


def _flag(arg: str) -> str:
    """``--time=1:00`` -> ``--time``; ``-t1:00`` -> ``-t``."""
    if arg.startswith("--"):
        return arg.split("=", 1)[0]
    return arg[:2] if arg.startswith("-") else ""


def owned_flag_errors(block: Mapping[str, Any], where: str) -> list[str]:
    """One line per flag in ``sbatch_args`` or ``srun_args`` that the flow sets itself."""
    errors = []
    for key, owned in (("sbatch_args", OWNED_SBATCH), ("srun_args", OWNED_SRUN)):
        for arg in block.get(key) or ():
            flag = _flag(str(arg))
            if flag in owned:
                errors.append(f"{where}.{key} sets {flag}, which the flow sets itself")
    return errors


def validate(block: Any, errors: list[str], *, where: str = "environment.slurm") -> None:
    """Validate ``environment.slurm`` into ``errors``: one line per missing or wrong key."""
    if not isinstance(block, Mapping):
        errors.append(f"'{where}' must be a mapping")
        return
    if "max_holders" in block:
        errors.append(
            f"'{where}.max_holders' is no key any more: what limits a run's holders is the "
            f"node-hours it may ask for (`{where}.max_node_hours`)"
        )
    known = (*REQUIRED_KEYS, *OPTIONAL_KEYS, "max_holders")
    reject_unknown_keys(block, known, errors, where=where)
    for key, how in REQUIRED_KEYS.items():
        if block.get(key) in (None, ""):
            errors.append(f"{where}.{key} is missing: {how}")
    for key in ("account", "partition", "qos", "container_image", "container_name", "host_python"):
        value = block.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"'{where}.{key}' must be a non-empty string")
    if block.get("time") not in (None, ""):
        try:
            slurm_time_seconds(block["time"])
        except ValueError:
            errors.append(
                f"'{where}.time' is not a Slurm time (minutes, HH:MM:SS or D-HH:MM:SS): "
                f"{block['time']!r}"
            )
    per_node = block.get("gpus_per_node")
    if per_node is not None and (
        not isinstance(per_node, int) or isinstance(per_node, bool) or per_node < 1
    ):
        errors.append(f"'{where}.gpus_per_node' must be an integer >= 1")
    bound = block.get("max_node_hours")
    if bound is not None and (
        isinstance(bound, bool) or not isinstance(bound, (int, float)) or bound <= 0
    ):
        errors.append(f"'{where}.max_node_hours' must be a number of node-hours above 0")
    if block.get("lead") is not None:
        try:
            parse_duration(str(block["lead"]))
        except ValueError:
            errors.append(f"'{where}.lead' is not a duration (seconds, 30m, 1h or HH:MM:SS)")
    for key in ("sbatch_args", "srun_args", "container_mounts"):
        value = block.get(key)
        if value is not None and (
            not isinstance(value, list) or not all(isinstance(v, str) and v for v in value)
        ):
            errors.append(f"'{where}.{key}' must be a list of non-empty strings")
    if block.get("container_image") and not block.get("container_mounts"):
        errors.append(
            f"{where}.container_mounts is missing: with a container_image, every path the run "
            f"uses (the spool, the workspace, the checkpoint, the checkout) must be under a mount"
        )
    if block.get("container_mounts") and not block.get("container_image"):
        errors.append(f"'{where}.container_mounts' needs '{where}.container_image'")
    errors.extend(owned_flag_errors(block, where))


def mount_errors(block: Mapping[str, Any], paths: Mapping[str, str | None]) -> list[str]:
    """One line per path of the run that no container mount covers (``what`` -> path)."""
    if not block.get("container_image"):
        return []
    inside = []
    for mount in block.get("container_mounts") or ():
        parts = str(mount).split(":")
        inside.append(os.path.normpath(parts[1] if len(parts) > 1 else parts[0]))
    errors = []
    for what, path in paths.items():
        if not path:
            continue
        real = os.path.normpath(str(path))
        if not any(real == root or real.startswith(root.rstrip("/") + "/") for root in inside):
            errors.append(f"{path} (the {what}) is under no container mount")
    return errors


def lead_seconds(block: Mapping[str, Any]) -> float:
    return parse_duration(str(block["lead"])) if block.get("lead") is not None else DEFAULT_LEAD_S


def run_id(workspace: str | Path) -> str:
    """A short digest of the workspace path: two runs of one user never share a job name."""
    return hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:8]


def container_name(block: Mapping[str, Any], run: str) -> str:
    return str(block.get("container_name") or f"mv2flow-{run}")


def _container_flags(block: Mapping[str, Any], run: str, *, mounts: bool) -> list[str]:
    image = block.get("container_image")
    if not image:
        return []
    flags = [f"--container-image={image}"]
    if mounts and block.get("container_mounts"):
        flags.append("--container-mounts=" + ",".join(block["container_mounts"]))
    flags.append(f"--container-name={container_name(block, run)}")
    return flags


def watcher_step_line(block: Mapping[str, Any], *, run: str, nodes: int, entry: str | Path) -> str:
    """The step that runs one watcher per node, in the container, carrying every mount."""
    argv = [
        "srun",
        "--mpi=none",
        f"--nodes={nodes}",
        f"--ntasks={nodes}",
        "--ntasks-per-node=1",
        *_container_flags(block, run, mounts=True),
        *(str(a) for a in block.get("srun_args") or ()),
        "bash",
        str(entry),
    ]
    return shlex.join(argv)


def step_line(
    block: Mapping[str, Any],
    *,
    run: str,
    ranks: int,
    nodes: int,
    hosts: str,
    command: str,
    output: str | None = None,
) -> str:
    """The line that runs ``command`` as ``ranks`` tasks of one world on ``nodes`` nodes ``hosts``.

    ``hosts`` is the node list as the shell will read it: a literal list, or an expansion of
    ``$MV2FLOW_JOB_HOSTS`` (quoted by the caller). No mounts: the named container keeps those of
    the watcher step. The ranks fill the nodes evenly.
    """
    if ranks < 1 or nodes < 1 or ranks < nodes:
        raise ValueError(f"{ranks} rank(s) cannot be laid out on {nodes} node(s)")
    if ranks % nodes:
        raise ValueError(
            f"{ranks} ranks do not fill {nodes} nodes evenly: this step line lays out equal "
            f"nodes only"
        )
    flags = [
        "--overlap",
        "--mpi=none" if ranks == 1 else "--mpi=pmix",
        "--kill-on-bad-exit=1",
        f"--nodes={nodes}",
        f"--ntasks={ranks}",
        f"--ntasks-per-node={ranks // nodes}",
    ]
    tail = [
        *_container_flags(block, run, mounts=False),
        *(str(a) for a in block.get("srun_args") or ()),
        *([f"--output={output}"] if output else []),
    ]
    parts = ["srun", shlex.join(flags), f"--nodelist={hosts}", shlex.join(tail), command]
    return " ".join(part for part in parts if part)


def inner_script(
    command: str,
    *,
    python: str,
    tree: str | None = None,
    pythonpath: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    role: str | None = None,
    devices: Sequence[str | int] | None = None,
) -> str:
    """The script a step runs on each rank: everything is set inside the launched shell.

    ``PATH`` gets the interpreter's directory (a batch host's ``PATH`` does not reach a
    container step), ``PYTHONPATH`` the pinned tree first, ``MV2FLOW_ROLE`` names the role
    instance in each origin record, and ``devices`` (when two instances share a node) is the
    subset this one may use: every rank sees all devices of its node otherwise.
    """
    lines = ["#!/bin/bash", "set -u"]
    if os.path.isabs(python):
        lines.append(f'export PATH={shlex.quote(os.path.dirname(python))}:"$PATH"')
    entries = [*([tree] if tree else []), *pythonpath]
    if entries:
        joined = ":".join(shlex.quote(str(e)) for e in entries)
        lines.append(f'export PYTHONPATH={joined}"${{PYTHONPATH:+:$PYTHONPATH}}"')
    for key, value in sorted((env or {}).items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    if role:
        lines.append(f"export MV2FLOW_ROLE={shlex.quote(role)}")
    if devices is not None:
        lines.append("export CUDA_VISIBLE_DEVICES=" + ",".join(str(d) for d in devices))
    if tree:
        lines.append(f"cd {shlex.quote(tree)} || exit 1")
    lines.append(f"exec {command}")
    return "\n".join(lines) + "\n"


def holder_options(
    block: Mapping[str, Any], *, name: str, nodes: int, spool: str | Path, avoid: Sequence[str] = ()
) -> list[str]:
    """What the flow owns of a holder's request: its name, its shape, its limit, where it writes.

    The same options are the script's directives (for whoever reads it) and the submission's
    command line (where nothing in the caller's environment can override them). ``avoid`` are
    hosts the pool gave back; they join what the site's own ``--exclude=`` names.

    A holder may not be requeued: a requeued job is the same id on other nodes with its
    limit counted from zero again, so one record would hold more than its nodes times its
    time. One that loses its nodes ends, and the pool asks for the next under its bound.
    """
    sites = [
        a.split("=", 1)[1] for a in block.get("sbatch_args") or () if a.startswith("--exclude=")
    ]
    return [
        f"--job-name={name}",
        f"--account={block['account']}",
        f"--partition={block['partition']}",
        *([f"--qos={block['qos']}"] if block.get("qos") else []),
        f"--time={block['time']}",
        f"--nodes={nodes}",
        f"--gpus-per-node={block['gpus_per_node']}",
        "--no-requeue",
        f"--chdir={spool}",
        f"--output={spool}/slurm-%j.out",
        *([f"--exclude={','.join([*sites, *avoid])}"] if avoid else []),
    ]


def render_holder(
    block: Mapping[str, Any],
    *,
    run: str,
    name: str,
    nodes: int,
    spool: str | Path,
    watcher_entry: str | Path,
    lead_entry: str | Path,
    avoid: Sequence[str] = (),
) -> str:
    """The batch script of one holder of ``nodes`` nodes, named ``name``."""
    directives = [
        *holder_options(block, name=name, nodes=nodes, spool=spool, avoid=avoid),
        *(str(a) for a in block.get("sbatch_args") or ()),
    ]
    lines = [
        "#!/bin/bash",
        *(f"#SBATCH {directive}" for directive in directives),
        "#",
        f"# Holder {name}: one watcher per node in the container, the lead on this host.",
        "# Rendered by the flow; the record beside it was written before it was submitted.",
        "set -u",
        f"export MV2FLOW_HOLDER={shlex.quote(name)}",
        watcher_step_line(block, run=run, nodes=nodes, entry=watcher_entry) + " &",
        f"exec bash {shlex.quote(str(lead_entry))}",
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ commands


def command_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a Slurm command of the flow runs in: the caller's, less what the flow owns.

    A caller that is itself inside a job (the controller's own job, a session in an allocation)
    carries that job's ``SLURM_*`` variables. ``sbatch`` exports the environment into the job
    it makes, where the steps would read another job's memory, task count and end time as
    their own; so inside a job none of them is passed on.
    """
    source = os.environ if environ is None else environ
    in_job = bool(source.get("SLURM_JOB_ID"))
    kept = {
        key: value
        for key, value in source.items()
        if key not in OWNED_ENV
        and not key.startswith(_OWNED_ENV_PREFIXES)
        and not (in_job and key.startswith("SLURM_") and key not in _CLIENT_ENV)
    }
    kept["SLURM_TIME_FORMAT"] = "%s"
    return kept


def _run(argv: Sequence[str], **kwargs: Any) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_S,
        check=False,
        env=command_environment(),
        **kwargs,
    )


def have_sbatch() -> bool:
    return shutil.which("sbatch") is not None


def _text(done: "subprocess.CompletedProcess[str]") -> str:
    """Slurm's own words for an answer: its last error line, else its last line."""
    lines = [
        line.strip() for line in (done.stderr + "\n" + done.stdout).splitlines() if line.strip()
    ]
    errors = [line for line in lines if "error" in line.lower()]
    return (errors or lines or [f"exit code {done.returncode}"])[-1]


def sbatch(script: str | Path, *, options: Sequence[str] = (), run: Run = _run) -> Reply:
    """Submit ``script``: accepted with a job id, refused with Slurm's line, or unknown.

    ``options`` are what the flow owns of the request, given on the command line, where they
    win over the caller's environment and the script's own directives.

    A submission is closed only by an affirmative answer: Slurm's refusal of the request, or an
    error that says nothing left this host (:data:`REFUSAL`). Every other answer leaves it
    unknown, because the job may exist: the command timed out, it exited 0 with no job id, or
    it failed with words that do not say.
    """
    try:
        done = run(["sbatch", "--parsable", *options, str(script)])
    except subprocess.TimeoutExpired:
        return Reply("unknown", text=f"sbatch did not answer within {COMMAND_TIMEOUT_S:.0f}s")
    except OSError as exc:
        return Reply("refused", text=f"sbatch could not be run: {exc}")
    answer = done.stdout.strip()
    if done.returncode == 0:
        if _JOB_ID.fullmatch(answer):
            return Reply("accepted", job_id=answer.split(";")[0], text=answer)
        return Reply("unknown", text=f"sbatch exited 0 and printed no job id: {answer[:200]!r}")
    said = _text(done)
    if REFUSAL.search(done.stderr + done.stdout):
        return Reply("refused", text=said)
    return Reply("unknown", text=said)


def _epoch(value: str) -> float | None:
    value = value.strip()
    if not value or value in ("N/A", "Unknown", "None", "NONE"):
        return None
    if value.isdigit():
        return float(value) or None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _row(job_id: str, state: str, *, reason: str = "", start: str = "", end: str = "") -> Row:
    word = state.split()[0].rstrip("+") if state.split() else ""
    return Row(
        state=_STATES.get(word, ENDED),
        job_id=job_id.split("_")[0].split(".")[0],
        end=_epoch(end),
        start=_epoch(start),
        reason=reason.strip(),
        detail=state.strip(),
    )


def squeue(names: Sequence[str], *, run: Run = _run) -> dict[str, list[Row]] | None:
    """Every job of this user under one of ``names``, in every state Slurm still lists.

    ``None`` when the query failed: a failed query is never an empty queue.
    """
    if not names:
        return {}
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or str(os.getuid())
    try:
        done = run(
            [
                "squeue",
                "-h",
                "-u",
                user,
                "-t",
                "all",
                f"--name={','.join(names)}",
                "-o",
                "%i|%j|%T|%r|%S|%e",
            ]
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if done.returncode != 0:
        return None
    found: dict[str, list[Row]] = {}
    for line in done.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 6 or fields[1] not in names:
            continue
        job_id, name, state, reason, start, end = fields
        found.setdefault(name, []).append(_row(job_id, state, reason=reason, start=start, end=end))
    return found


def sacct(
    *, name: str = "", job_id: str = "", since: float | None = None, run: Run = _run
) -> list[Row] | None:
    """What accounting holds of a job by ``name`` or ``job_id``; ``None`` when it does not answer.

    An empty list proves nothing either: accounting can lag behind the controller.
    """
    argv = ["sacct", "-n", "-X", "-P", "-o", "JobID,JobName,State,End,Start"]
    argv += ["-j", job_id] if job_id else [f"--name={name}"]
    if since is not None:
        argv += ["-S", datetime.fromtimestamp(since).strftime("%Y-%m-%dT%H:%M:%S")]
    try:
        done = run(argv)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if done.returncode != 0:
        return None
    rows = []
    for line in done.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) not in (4, 5) or (name and fields[1] != name):
            continue
        # When it started and ended, for what an ended job held (the run's bound).
        start = fields[4] if len(fields) == 5 else ""
        rows.append(_row(fields[0], fields[2], start=start, end=fields[3]))
    return rows


def scancel(job_ids: Sequence[str], *, pending_only: bool = False, run: Run = _run) -> str:
    """Cancel exactly ``job_ids``; with ``pending_only`` only those still pending. ``""`` if it ran."""
    if not job_ids:
        return ""
    argv = ["scancel", *(["--state=PENDING"] if pending_only else []), *job_ids]
    try:
        done = run(argv)
    except subprocess.TimeoutExpired:
        return f"scancel did not answer within {COMMAND_TIMEOUT_S:.0f}s"
    except OSError as exc:
        return f"scancel could not be run: {exc}"
    return "" if done.returncode == 0 else _text(done)


def _controller_forgot(job_id: str, run: Run) -> bool:
    """Whether the controller itself says ``job_id`` is no job of its any more."""
    try:
        done = run(["squeue", "-h", "-j", job_id, "-t", "all", "-o", "%i"])
    except (subprocess.TimeoutExpired, OSError):
        return False
    return done.returncode != 0 and "invalid job id" in (done.stderr + done.stdout).lower()


class SlurmEnvironment:
    """The pool's slots on Slurm (:class:`.pool.Environment`)."""

    kind = "slurm"
    renews = True  # read by `checks/gate_cli.py` only: it goes with that line

    def __init__(
        self,
        block: Mapping[str, Any],
        *,
        spool: str | Path,
        run: str,
        watcher_entry: str | Path,
        lead_entry: str | Path,
        runner: Run = _run,
    ) -> None:
        self.block = dict(block)
        self.spool = Path(spool)
        self.run = run
        self.watcher_entry = Path(watcher_entry)
        self.lead_entry = Path(lead_entry)
        self.runner = runner
        self.lead_s = lead_seconds(block)
        self.time_s = slurm_time_seconds(block["time"])
        #: Stated, it fixes one shape for the whole run.
        #: The node-hours the run may ask for; never "none" on Slurm.
        self.max_node_hours = float(block.get("max_node_hours") or DEFAULT_MAX_NODE_HOURS)

    def holder_name(self, seq: int) -> str:
        return f"mv2flow-{self.run}-pool-{seq}"

    def submit(self, name: str, nodes: int, avoid: Sequence[str] = ()) -> Reply:
        """Render the holder's script beside its record and submit it."""
        script = self.spool / "holders" / f"{name}.sbatch"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            render_holder(
                self.block,
                run=self.run,
                name=name,
                nodes=nodes,
                spool=self.spool,
                watcher_entry=self.watcher_entry,
                lead_entry=self.lead_entry,
                avoid=avoid,
            ),
            encoding="utf-8",
        )
        options = holder_options(self.block, name=name, nodes=nodes, spool=self.spool, avoid=avoid)
        return sbatch(script, options=options, run=self.runner)

    def listing(self, records: Sequence[Mapping[str, Any]]) -> dict[str, list[Row]] | None:
        """What Slurm says of each record's name; a name it says nothing of is absent.

        The queue first (every state it still lists); for a name it does not list,
        accounting, by the recorded job id when one is known, else by name since the record
        was written.
        """
        names = [str(record["name"]) for record in records]
        listed = squeue(names, run=self.runner)
        if listed is None:
            return None
        for record in records:
            name = str(record["name"])
            if listed.get(name):
                continue
            job_id = str(record.get("job_id") or "")
            written = float(record.get("written_at") or 0)
            rows = sacct(
                name="" if job_id else name,
                job_id=job_id,
                since=written - 60 if written else None,
                run=self.runner,
            )
            if not rows and job_id and _controller_forgot(job_id, self.runner):
                rows = [
                    Row(
                        state=ENDED,
                        job_id=job_id,
                        detail="GONE (the controller no longer knows the job)",
                    )
                ]
            if rows:
                listed[name] = rows
        return listed

    def end(self, record: Mapping[str, Any], *, pending_only: bool = False) -> str:
        """Cancel the job this record names, by its recorded id only."""
        job_id = str(record.get("job_id") or "")
        if not job_id:
            return "no job id is recorded for it"
        return scancel([job_id], pending_only=pending_only, run=self.runner)


__all__ = [
    "DEFAULT_LEAD_S",
    "OPTIONAL_KEYS",
    "OWNED_SBATCH",
    "OWNED_SRUN",
    "REQUIRED_KEYS",
    "SlurmEnvironment",
    "OWNED_ENV",
    "REFUSAL",
    "command_environment",
    "holder_options",
    "container_name",
    "have_sbatch",
    "inner_script",
    "lead_seconds",
    "mount_errors",
    "owned_flag_errors",
    "render_holder",
    "run_id",
    "sacct",
    "sbatch",
    "scancel",
    "slurm_time_seconds",
    "squeue",
    "step_line",
    "validate",
    "watcher_step_line",
]
