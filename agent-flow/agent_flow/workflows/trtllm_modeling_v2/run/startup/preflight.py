# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight — everything that must hold before an agent is worth starting.

Deliberately **pure Python with no agent turn**. The one turn preflight does
run comes after these asserts: the agent probe (``agent_probe.py``), which
proves that the turns themselves can work here. The asserts run **in order**,
and the first failure ends the run before the branch exists.

- **Judge a device by ``memory.free`` and ``utilization.gpu``, never by the
  process table.** Orphaned processes holding a dead job's allocations do not
  appear in it at all.
- **A card at 100% utilization is *broken* for our purposes, not busy.**
- A card that fails a single-process, one-tensor allocation with no peers is
  unusable rather than contended (the lone-process probe).
"""

from __future__ import annotations

import ast
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ...checks import anchors as anchors_mod
from ...checks import session as session_mod
from ...infra.compute import environment as environment_mod
from ...infra.compute import pool as pool_mod
from ...infra.compute.commands import CommandResult, CommandSpec, LocalCommandExecutor
from ...infra.compute.environment import RunEnvironment, resolve_environment
from ...operators.coverage import MODELING_V2_PACKAGE
from ...spec import deployments as deployments_mod
from ...spec.task_schema import (
    derive_checkpoint_segment,
    derive_family,
    derive_slug,
    target_parallel,
    widest_world_size,
)

# A device is claimable when it is this close to empty. Not 100%: a healthy
# idle card still shows a little in use (ECC tables, the driver's own context).
MIN_FREE_MEMORY_FRACTION = 0.90


# Above this, treat the card as broken rather than busy.
MAX_UTILIZATION_PCT = 10

# The lone-process probe and the multi-rank substrate probe both run under
# a hard deadline, because the failures they look for do not raise — they
# hang.
PROBE_TIMEOUT_S = 120
MULTIRANK_PROBE_TIMEOUT_S = 300

CommandRunner = Callable[[Sequence[str], Mapping[str, str] | None, int], CommandResult]


@dataclass(frozen=True)
class DeviceReport:
    """One card as ``nvidia-smi`` describes it, plus our verdict on it."""

    index: int
    memory_free_mib: int
    memory_total_mib: int
    utilization_pct: int
    compute_cap: str

    @property
    def arch(self) -> str:
        """``10.3`` -> ``sm_103``. The form catalog receipts are keyed by."""
        major, _, minor = self.compute_cap.partition(".")
        return f"sm_{major}{minor or '0'}"

    @property
    def free_fraction(self) -> float:
        if self.memory_total_mib <= 0:
            return 0.0
        return self.memory_free_mib / self.memory_total_mib

    @property
    def looks_idle(self) -> bool:
        return (
            self.free_fraction >= MIN_FREE_MEMORY_FRACTION
            and self.utilization_pct <= MAX_UTILIZATION_PCT
        )

    def why_not_idle(self) -> str:
        reasons = []
        if self.free_fraction < MIN_FREE_MEMORY_FRACTION:
            reasons.append(
                f"{self.memory_free_mib} MiB free of {self.memory_total_mib} "
                f"({self.free_fraction:.0%}, need {MIN_FREE_MEMORY_FRACTION:.0%})"
            )
        if self.utilization_pct > MAX_UTILIZATION_PCT:
            reasons.append(
                f"{self.utilization_pct}% utilization (a card spinning this hard is "
                f"broken for our purposes, not merely busy)"
            )
        return "; ".join(reasons) or "idle"


@dataclass(frozen=True)
class PreflightResult:
    """The run's resolved identity and launch environment."""

    slug: str
    branch: str
    devices: tuple[int, ...]
    world_size: int
    environment: RunEnvironment
    family: str = ""
    target_rel: str = ""
    checkpoint_config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class PreflightError(RuntimeError):
    """A precondition failed. The run ends here, before the branch exists."""


class SpecInconsistencyError(PreflightError):
    """The spec cannot run as written, or asks the tree for something it does not have.

    Direct the operator to the spec or tree; infrastructure diagnostics do not apply.
    """


#: The one process-wide local executor for probes. Stateless.
_LOCAL_EXECUTOR = LocalCommandExecutor()


def _default_runner(
    argv: Sequence[str],
    env: Mapping[str, str] | None = None,
    timeout: int = PROBE_TIMEOUT_S,
) -> CommandResult:
    """Run one probe here, as an argv, with no shell in between."""
    return _LOCAL_EXECUTOR.execute(
        CommandSpec(argv=tuple(str(part) for part in argv), env=env, timeout_s=timeout)
    )


def query_devices(runner: CommandRunner = _default_runner) -> list[DeviceReport]:
    """Read every visible card's memory, utilization and compute capability."""
    result = runner(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total,utilization.gpu,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        None,
        PROBE_TIMEOUT_S,
    )
    if not result.ok:
        raise PreflightError(
            "nvidia-smi did not answer, so no device claim can be made on this "
            f"host (exit {result.returncode}). stderr: {result.stderr.strip()!r}"
        )
    devices: list[DeviceReport] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 5:
            raise PreflightError(f"unparsable nvidia-smi row: {line!r}")
        try:
            devices.append(
                DeviceReport(
                    index=int(fields[0]),
                    memory_free_mib=int(fields[1]),
                    memory_total_mib=int(fields[2]),
                    utilization_pct=int(fields[3]),
                    compute_cap=fields[4],
                )
            )
        except ValueError as exc:
            raise PreflightError(f"unparsable nvidia-smi row: {line!r} ({exc})") from exc
    return devices


def probe_device_alone(
    index: int,
    python: str,
    runner: CommandRunner = _default_runner,
) -> CommandResult:
    """Allocate one tensor on one card in a lone process: *broken* rather than *contended*."""
    script = "import torch; torch.cuda.set_device(0); torch.empty(1, device='cuda'); print('ok')"
    return runner(
        [python, "-c", script],
        {"CUDA_VISIBLE_DEVICES": str(index)},
        PROBE_TIMEOUT_S,
    )


def claim_devices(
    world_size: int,
    arch: str,
    python: str,
    *,
    runner: CommandRunner = _default_runner,
    notes: list[str] | None = None,
) -> tuple[int, ...]:
    """Claim ``world_size`` healthy cards of ``arch``, or raise."""
    notes = notes if notes is not None else []
    reports = query_devices(runner)
    if not reports:
        raise PreflightError("nvidia-smi reported no devices on this host.")

    wrong_arch = [r for r in reports if r.arch != arch]
    if len(wrong_arch) == len(reports):
        seen = sorted({r.arch for r in reports})
        raise PreflightError(
            f"task.yaml asks for {arch} but this host has {seen}. Catalog receipts "
            f"are per-architecture, so running here would gate the target against "
            f"another machine's certification. Move hosts or fix `arch`."
        )

    candidates = [r for r in reports if r.arch == arch]
    idle = [r for r in candidates if r.looks_idle]
    busy = [r for r in candidates if not r.looks_idle]
    for report in busy:
        notes.append(f"device {report.index} not claimable: {report.why_not_idle()}")

    claimed: list[int] = []
    for report in idle:
        if len(claimed) == world_size:
            break
        probe = probe_device_alone(report.index, python, runner)
        if probe.ok:
            claimed.append(report.index)
            continue
        detail = "timed out" if probe.timed_out else (probe.stderr.strip() or "non-zero exit")
        notes.append(
            f"device {report.index} looked idle but failed a lone-process probe "
            f"({detail}) — unusable rather than contended; no amount of waiting "
            f"will free it"
        )

    if len(claimed) < world_size:
        raise PreflightError(
            f"need {world_size} healthy {arch} device(s), claimed {len(claimed)}: "
            f"{claimed}. Details:\n  - " + "\n  - ".join(notes or ["(no devices examined)"])
        )
    return tuple(claimed)


def check_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Assert the checkpoint parses, and return its ``config.json``."""
    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.is_file():
        raise PreflightError(f"{config_path} does not exist; this is not a HF checkpoint dir.")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreflightError(f"{config_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PreflightError(f"{config_path} must hold a JSON object, got {type(data).__name__}")
    return data


def check_accuracy_anchors(
    task_data: Mapping[str, Any],
    units: Sequence[str] | None = None,
) -> list[str]:
    """What each benchmark of each deployment is held to, as notes; nothing here stops a run.

    The lookup is the session's own (``checks/anchors.py``). A benchmark with nothing to
    hold it to is blocked before its own load, with the reason, by its session; every
    other check of the deployment, and every other deployment, goes on.
    """
    notes: list[str] = []
    for deployment in deployments_mod.resolve(task_data):
        if units is not None and deployment.name not in units:
            continue
        for benchmark in deployment.benchmarks:
            anchor = anchors_mod.resolve(task_data, deployment, benchmark)
            where = f"accuracy anchor [{deployment.name}]: {benchmark.name}"
            if anchor.kind == anchors_mod.NONE:
                notes.append(
                    f"{where} has {anchor.source}: its check will be blocked before the "
                    f"load, and everything else goes on"
                )
            elif anchor.kind == anchors_mod.RELATIVE:
                notes.append(f"{where} is held to {anchor.source}, measured as it is")
            else:
                notes.append(f"{where} = {anchor.value:g} ({anchor.kind}: {anchor.source})")
    return notes


def check_gate_budgets(task_data: Mapping[str, Any]) -> list[str]:
    """Whether every deployment's session fits in one allocation of the sessions that run the gates.

    The spool admits a job only where it can finish before its allocation ends, so a
    session longer than a whole allocation is never admitted. A session's time is its
    deployment's ``timeout``, else what its checks add up to (``checks/session.py``).
    A session that cannot fit in one holder is refused here, as the pool refuses it; one
    wider than one node where no step can cross nodes is said here and again by its own check.
    """
    limits = gate_limits(task_data)
    unplaced = _unplaced(task_data)
    notes = []
    for deployment in deployments_mod.resolve(task_data):
        said = unplaced.get(deployment.name) or (
            f"{deployment.ranks} device(s), {limits[deployment.name]}s"
        )
        notes.append(f"session [{deployment.name}]: {said}")
    limit = pool_mod.holder_time_s(task_data.get("environment") or {})
    if not limit or not limits:
        return notes
    budget = int(limit)
    # Fit, by the pool's own rule: a holder's start takes its time, and a job starts only
    # where its wall and the margin end before the holder does.
    longest = int(pool_mod.fit_wall_s(limit))
    over = {key: wall for key, wall in limits.items() if wall > longest}
    if over:
        room = longest - session_mod.TEARDOWN_S
        # What a holder must last to admit the longest of them as the spec states it.
        enough = max(over.values()) + budget - longest
        raise PreflightError(
            "; ".join(f"the session of `{key}` is {wall}s" for key, wall in over.items())
            + f", longer than one holder ({budget}s) admits ({longest}s): the spool never "
            f"admits a job that cannot finish in its allocation, so it would wait for ever. "
            + (
                f"Set the deployment's `timeout` to at most {room}s (with room for loading "
                f"the model), or give the holders a longer time ({enough}s admits it as stated)."
                if room >= 1
                # A holder its own start, the margin and a session's end use up: no
                # `timeout` the spec could state fits, so none is advised.
                else f"No `timeout` fits a holder that short: give the holders a longer time "
                f"({enough}s admits it as stated)."
            )
        )
    return [
        *notes,
        "session time limits within one holder ("
        + ", ".join(f"{key} {wall}s" for key, wall in limits.items())
        + f" <= {budget}s)",
    ]


def _unplaced(task_data: Mapping[str, Any]) -> dict[str, str]:
    """Why a deployment's session cannot be placed as the spec stands, by deployment."""
    found = {}
    for deployment in deployments_mod.resolve(task_data):
        try:
            session_mod.place(task_data, deployment)
        except session_mod.WiderThanOneNode as exc:
            found[deployment.name] = str(exc)
    return found


def gate_limits(task_data: Mapping[str, Any]) -> dict[str, int]:
    """Each session's wall time as the spec states it, by the deployment that states it.

    A session's wall is its deployment's ``timeout``, else what its checks add up to,
    plus the time to stop its servers (``checks/session.py``).
    """
    limits = {}
    for deployment in deployments_mod.resolve(task_data):
        try:
            limits[deployment.name] = int(session_mod.place(task_data, deployment).wall)
        except session_mod.WiderThanOneNode:
            continue
    return limits


#: Prefix of the line the processor probe prints its verdict on.
PROCESSOR_PROBE_MARKER = "[mv2-processor]"
PROCESSOR_PROBE_TIMEOUT_S = 600
#: Loads the checkpoint's processor from local files and reports a failed
#: import by module name.
PROCESSOR_PROBE_SOURCE = f"""
import json, sys
try:
    from transformers import AutoProcessor
    AutoProcessor.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
    out = {{"ok": True}}
except ImportError as exc:
    name = exc.name if isinstance(exc, ModuleNotFoundError) else None
    out = {{"import_error": True, "module": name or "", "error": f"{{type(exc).__name__}}: {{exc}}"}}
except Exception as exc:
    out = {{"error": f"{{type(exc).__name__}}: {{exc}}"}}
print({PROCESSOR_PROBE_MARKER!r} + json.dumps(out), flush=True)
"""


def check_processor_dependencies(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    shell_runner: Any,
    timeout: int = PROCESSOR_PROBE_TIMEOUT_S,
) -> list[str]:
    """Load the checkpoint's processor with the run's interpreter when a unit is multimodal.

    A module the processor imports and the environment lacks is an
    infrastructure error, raised here with the module's name. Any other
    failure is returned as a note: the target may supply what the stock
    processor cannot.
    """
    if not any("image" in d.inputs for d in deployments_mod.resolve(task_data)):
        return []
    checkpoint = str(task_data["checkpoint_path"])
    command = environment.python_command(
        ["-c", PROCESSOR_PROBE_SOURCE, checkpoint], single_process=True
    )
    result = shell_runner(command, None, timeout)
    output = f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}"
    verdict: dict[str, Any] | None = None
    for line in output.splitlines():
        if line.startswith(PROCESSOR_PROBE_MARKER):
            try:
                verdict = json.loads(line[len(PROCESSOR_PROBE_MARKER) :])
            except json.JSONDecodeError:
                verdict = None
    if verdict is None:
        why = (
            f"timed out after {timeout}s"
            if getattr(result, "timed_out", False)
            else f"exited {getattr(result, 'returncode', '?')} without a verdict"
        )
        return [f"processor probe {why}; the multimodal unit's processor imports are unverified"]
    if verdict.get("ok"):
        return ["processor probe: the checkpoint's processor loads under the run's interpreter"]
    if verdict.get("import_error"):
        module = verdict.get("module") or ""
        what = f"module `{module}` is missing" if module else "an import failed"
        raise PreflightError(
            f"infrastructure: the checkpoint's processor cannot load under the run's "
            f"interpreter ({environment.python}): {what}. The multimodal unit needs it "
            f"for every image prompt. Install it in the run's environment and start "
            f"again; no modeling change fixes this.\n\n  {verdict.get('error', '')}"
        )
    return [
        f"processor probe: the checkpoint's processor did not load "
        f"({verdict.get('error', '')}); not an import failure, so the multimodal unit "
        f"may have to supply it"
    ]


def check_multirank_substrate(
    world_size: int,
    environment: RunEnvironment,
    runner: CommandRunner = _default_runner,
) -> None:
    """Exercise an NCCL collective and an MPI dynamic spawn under a deadline: both hang rather than raise."""
    if world_size <= 1:
        return
    script = (
        "import torch, torch.distributed as dist;"
        "dist.init_process_group('nccl');"
        "import os;torch.cuda.set_device(int(os.environ['LOCAL_RANK']));"
        "t = torch.ones(1, device='cuda');"
        "dist.all_reduce(t);"
        "dist.barrier();"
        "print('[preflight] OK' if t.item() == dist.get_world_size() else '[preflight] BAD')"
    )
    argv = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        "--nnodes=1",
        "--no-python",
        environment.python,
        "-c",
        script,
    ]
    env = dict(environment.env)
    if environment.devices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in environment.devices)
    result = runner(argv, env, MULTIRANK_PROBE_TIMEOUT_S)
    if result.timed_out:
        raise PreflightError(
            f"the {world_size}-rank NCCL/MPI probe did not finish within "
            f"{MULTIRANK_PROBE_TIMEOUT_S}s. It hangs rather than fails when the "
            f"launch substrate is wrong — typically an MPI connect-back landing on "
            f"an interface that does not carry it. Without this check the same "
            f"problem surfaces as an engine that never boots, hours into the run."
        )
    if not result.ok or "[preflight] OK" not in result.stdout:
        raise PreflightError(
            f"the {world_size}-rank NCCL/MPI probe failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout.strip()!r}\nstderr: {result.stderr.strip()!r}"
        )


def run_preflight(
    task_data: Mapping[str, Any],
    *,
    runner: CommandRunner = _default_runner,
) -> PreflightResult:
    """Assert every precondition in order and resolve the run's identity."""
    notes: list[str] = []

    checkpoint_config = check_checkpoint(task_data["checkpoint_path"])

    arch = str(task_data["arch"])
    # The most devices one session holds: the widest deployment's.
    world_size = widest_world_size(task_data, 1)
    notes.append(
        f"deployments: {', '.join(f'{d.name} ({d.shape})' for d in deployments_mod.resolve(task_data))}"
        f"; the widest session holds {world_size} device(s)"
    )

    # The checks that only read files run before anything that touches the host. They
    # are notes: only a spec that does not load stops a run before it starts.
    notes.extend(check_accuracy_anchors(task_data))
    notes.extend(check_gate_budgets(task_data))

    env_block = dict(task_data.get("environment") or {})
    declared_env = {str(k): str(v) for k, v in (env_block.get("env") or {}).items()}
    try:
        environment_mod.check_tmpdir(declared_env)
    except environment_mod.EnvironmentError as exc:
        raise PreflightError(str(exc)) from exc

    python = str(env_block.get("python", "python3"))
    # A device claim only means anything when the orchestrator and the GPUs
    # are the *same machine* (`local`); elsewhere the holder is the claim. On one host the
    # claim proves the devices; which job gets which is the local watcher's to say.
    kind = pool_mod.environment_kind(env_block)
    scheduled = kind != pool_mod.KIND_LOCAL
    devices: tuple[int, ...] = ()
    if not scheduled:
        devices = claim_devices(world_size, arch, python, runner=runner, notes=notes)
    if scheduled:
        notes.append(
            f"{kind}: devices are the pool's to allocate, so "
            f"preflight claims none and asserts no architecture locally. `arch: {arch}` "
            f"is taken on trust here and is what the receipts will be keyed by."
        )

    environment = resolve_environment(task_data, devices=devices, world_size=world_size)

    if not scheduled:
        check_multirank_substrate(world_size, environment, runner)

    slug = derive_slug(task_data)
    family, target_rel = resolve_target_identity(task_data, checkpoint_config)
    return PreflightResult(
        slug=slug,
        branch=f"trtllm-modeling-v2/{slug}",
        devices=devices,
        world_size=world_size,
        environment=environment,
        family=family,
        target_rel=target_rel,
        checkpoint_config=checkpoint_config,
        notes=notes,
    )


def routed_family(repo: Path, architecture: str) -> str | None:
    """The family directory whose ``routing.py`` registers ``architecture`` with ``@routes``."""
    for routing in sorted((repo / MODELING_V2_PACKAGE / "models").glob("*/routing.py")):
        try:
            tree = ast.parse(routing.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                func = decorator.func if isinstance(decorator, ast.Call) else None
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name == "routes" and any(
                    isinstance(arg, ast.Constant) and arg.value == architecture
                    for arg in decorator.args
                ):
                    return routing.parent.name
    return None


def resolve_target_identity(
    task_data: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
) -> tuple[str, str]:
    """Return ``(family, target_rel)`` for this run.

    The family and the checkpoint segment are derived and overridable through
    ``task.yaml``'s ``target`` block. The family comes from the checkpoint's
    own ``architectures[0]``, which is the same string routing keys on: the
    family whose ``routing.py`` registers it, else a new family's name.
    """
    override = dict(task_data.get("target") or {})

    architectures = checkpoint_config.get("architectures") or []
    if not architectures and not override.get("family"):
        raise PreflightError(
            "the checkpoint's config.json declares no `architectures`, so the "
            "modeling_v2 family cannot be derived. Set `target.family` in task.yaml."
        )
    family = override.get("family")
    if not family:
        architecture = str(architectures[0])
        repo = Path(str(task_data.get("trtllm_repo_path") or "."))
        family = routed_family(repo, architecture) or derive_family(architecture)
    segment = override.get("checkpoint_segment") or derive_checkpoint_segment(
        task_data["checkpoint_path"]
    )

    target_rel = (
        f"{MODELING_V2_PACKAGE}/models/{family}/"
        f"{segment}__{task_data['arch']}__{target_parallel(task_data)}"
    )
    return family, target_rel


def format_preflight_report(result: PreflightResult) -> str:
    """A human-readable summary for the journal and the console."""
    lines = [
        f"slug:        {result.slug}",
        f"branch:      {result.branch}",
        f"target:      {result.target_rel}",
        f"world size:  {result.world_size}",
        f"devices:     {', '.join(str(d) for d in result.devices) or '(assigned by compute sessions)'}",
        f"launch:      {result.environment.kind}",
        f"prefix:      {result.environment.prefix or '(none)'}",
        f"interpreter: {shlex.quote(result.environment.python)}",
    ]
    if result.notes:
        lines.append("notes:")
        lines.extend(f"  - {note}" for note in result.notes)
    return "\n".join(lines)


__all__ = [
    "CommandRunner",
    "DeviceReport",
    "PROCESSOR_PROBE_MARKER",
    "MAX_UTILIZATION_PCT",
    "MIN_FREE_MEMORY_FRACTION",
    "PreflightError",
    "PreflightResult",
    "SpecInconsistencyError",
    "check_gate_budgets",
    "gate_limits",
    "check_accuracy_anchors",
    "check_checkpoint",
    "check_multirank_substrate",
    "check_processor_dependencies",
    "claim_devices",
    "format_preflight_report",
    "probe_device_alone",
    "query_devices",
    "run_preflight",
]
