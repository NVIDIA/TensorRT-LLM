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

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ...infra.compute import environment as environment_mod
from ...infra.compute.commands import CommandResult, CommandSpec, LocalCommandExecutor
from ...infra.compute.environment import RunEnvironment, resolve_environment
from ...operators.coverage import MODELING_V2_PACKAGE
from ...spec import deployments as deployments_mod
from ...spec.derive import draft_layers, hybrid_marker, read_config
from ...spec.task_schema import (
    DEFAULT_PARALLEL,
    derive_checkpoint_segment,
    derive_family,
    derive_slug,
    routed_family,
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


def check_checkpoint_fit(task_data: Mapping[str, Any]) -> None:
    """Refuse a spec the checkpoint cannot meet, every reason at once, from ``config.json`` alone.

    Each server's ranks split the checkpoint's attention heads, KV heads and experts; the
    architecture runs its weight format; a deployment that drafts has the draft layers it
    needs, spelled as the checkpoint ships them; one that takes images has a tower.
    """
    config = read_config(task_data["checkpoint_path"]) or {}
    heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads") or heads
    experts = (
        config.get("num_experts")
        or config.get("n_routed_experts")
        or config.get("num_local_experts")
    )
    dspark = any(str(key).startswith("dspark") for key in config)
    problems = []
    for deployment in deployments_mod.resolve(task_data):
        for role, server in deployment.servers.items():
            who = deployment.name if role == "server" else f"{deployment.name}.{role}"
            options = server.options
            ranks = server.ranks
            split = int(options.get("moe_expert_parallel_size") or 1)
            # Attention DP replicates attention; without it the ranks split its heads.
            if ranks > 1 and not options.get("enable_attention_dp"):
                if heads and heads % ranks:
                    problems.append(f"{who}: {heads} attention heads do not split {ranks} ways")
                if kv_heads and kv_heads % ranks and ranks % kv_heads:
                    problems.append(f"{who}: {kv_heads} KV heads do not split {ranks} ways")
            if split > 1 and experts and experts % split:
                problems.append(f"{who}: {experts} experts do not split {split} ways")
            drafter = str((options.get("speculative_config") or {}).get("decoding_type") or "")
            if drafter.lower() == "mtp" and dspark:
                problems.append(
                    f"{who}: the checkpoint's draft stages are a DSpark drafter (dspark_* "
                    f"keys): its decoding type is DSpark, not MTP"
                )
            if drafter.lower() in ("mtp", "dspark") and not draft_layers(config):
                problems.append(
                    f"{who}: it drafts with {drafter}, and config.json states no draft layers"
                )
        if "image" in deployment.inputs and not {"vision_config", "audio_config"} & set(config):
            problems.append(f"{deployment.name}: it takes images, and the checkpoint has no tower")
    arch = re.match(r"^sm_(\d+)$", str(task_data.get("arch", "")))
    algo = str((config.get("quantization_config") or {}).get("quant_algo") or "").lower()
    if arch and "fp4" in algo and int(arch.group(1)) < 100:
        problems.append(f"the checkpoint's {algo} weights need sm_100 or later")
    if problems:
        raise SpecInconsistencyError(
            "the checkpoint cannot meet the spec:\n  - " + "\n  - ".join(dict.fromkeys(problems))
        )


#: Where a process's memory limit is, under cgroup v2 and v1.
#: Where the cgroup hierarchies are mounted, and each version's limit file.
CGROUP_ROOT = Path("/sys/fs/cgroup")
_V2_LIMIT, _V1_LIMIT = "memory.max", "memory.limit_in_bytes"


def _limit_at(path: Path) -> int | None:
    """The memory limit a cgroup file states, in bytes; ``None`` for none or unreadable."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    # v2 writes "max", v1 a page-aligned huge number, for no limit.
    return int(text) if text.isdigit() and int(text) < 1 << 60 else None


def cgroup_memory_limit(proc_cgroup: str | None = None, root: Path = CGROUP_ROOT) -> int | None:
    """The tightest memory limit on this process's own cgroup and its ancestors, in bytes.

    Reads the process's cgroup paths from ``/proc/self/cgroup`` (v2 ``0::<path>``, v1
    ``<n>:<controllers with memory>:<path>``) and walks each up to the root of its mount: a
    container or a Slurm step sets the limit on a cgroup of its own, not on the mount's root.
    """
    if proc_cgroup is None:
        try:
            proc_cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8")
        except OSError:
            proc_cgroup = ""
    candidates: list[Path] = []
    for line in proc_cgroup.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, rel = parts
        rel = rel.strip().lstrip("/")
        if controllers == "":
            base, name = root, _V2_LIMIT
        elif "memory" in controllers.split(","):
            base, name = root / "memory", _V1_LIMIT
        else:
            continue
        here = base / rel if rel else base
        while True:
            candidates.append(here / name)
            if here == base:
                break
            here = here.parent
    # A cgroup namespace shows its own cgroup at the mount's root.
    candidates += [root / _V2_LIMIT, root / "memory" / _V1_LIMIT]
    limits = [limit for limit in map(_limit_at, candidates) if limit is not None]
    return min(limits) if limits else None


def memory_limit_and_free() -> tuple[int | None, int | None]:
    """This process's cgroup memory limit and the node's free memory, in bytes; ``None`` if unknown."""
    try:
        free = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError):
        free = None
    return cgroup_memory_limit(), free


def check_kv_cache_configs(
    task_data: Mapping[str, Any], checkpoint_config: Mapping[str, Any]
) -> list[str]:
    """What each deployment's ``kv_cache_config`` will cost on this host, as notes.

    A server that bounds no KV host tier, on a host whose cgroup allows less memory than
    the node has free, pins that much host memory in every engine. A server that asks for
    block reuse with no Mamba state snapshot policy, on a checkpoint :func:`hybrid_marker`
    marks hybrid, has reuse turned off by the engine. Notes rather than refusals: the
    engine's own test decides, and its boot shows it.
    """
    unbounded: list[str] = []
    exposed: list[str] = []
    for deployment in deployments_mod.resolve(task_data):
        for role, server in deployment.servers.items():
            who = deployment.name if role == "server" else f"{deployment.name}.{role}"
            kv = server.options.get("kv_cache_config") or {}
            if "host_cache_size" not in kv:
                unbounded.append(who)
            state = kv.get("mamba_state_config") or {}
            snapshots = kv.get("mamba_state_cache_interval") or any(state.values())
            if kv.get("enable_block_reuse") and not snapshots:
                exposed.append(who)
    notes: list[str] = []
    limit, free = memory_limit_and_free()
    if unbounded and limit is not None and free is not None and limit < free:
        notes.append(
            f"host memory: {', '.join(unbounded)}: set no `kv_cache_config.host_cache_size`. "
            f"The engine then sizes a KV host tier of up to its device quota from the "
            f"node's free memory ({free / 2**30:.0f} GiB), while this process may use "
            f"{limit / 2**30:.0f} GiB (its cgroup's limit): every engine a check or a "
            f"worker starts pins that much host memory, and a few at once are OOM-killed. "
            f"Set `host_cache_size` (bytes, e.g. 8589934592) in the options."
        )
    marker = hybrid_marker(checkpoint_config)
    if marker and exposed:
        notes.append(
            f"block reuse: {', '.join(exposed)}: ask for KV-cache block reuse, and their "
            f"options set no Mamba state snapshot policy, on a hybrid checkpoint "
            f"({marker} in config.json). The engine turns block reuse off without one, and "
            f"their boot then fails on a repeated prompt that resumes nothing. Set "
            f"`kv_cache_config.mamba_state_config.periodic_snapshot_interval` (or snapshot "
            f"offsets) in their options, or turn block reuse off for them"
        )
    return notes


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
    """Load the checkpoint's processor with the run's interpreter when a deployment takes images.

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
    claim: bool = True,
) -> PreflightResult:
    """Assert every precondition in order and resolve the run's identity.

    Without ``claim`` (a dry run) nothing touches a GPU: no device is queried or
    claimed and the multi-rank probe does not run; the devices are taken to be
    the first ``world_size`` ones, for the commands the report shows.
    """
    notes: list[str] = []

    checkpoint_config = check_checkpoint(task_data["checkpoint_path"])

    arch = str(task_data["arch"])
    found = deployments_mod.resolve(task_data)
    # What runs on this host: a deployment behind a launch prefix places its own ranks.
    here = [d for d in found if not d.prefixed]
    world_size = max((s.ranks for d in here for s in d.servers.values()), default=1)
    need = max((d.ranks for d in here), default=0)

    # The checks that only read files run before anything that touches the host.
    check_checkpoint_fit(task_data)
    notes.extend(check_kv_cache_configs(task_data, checkpoint_config))

    env_block = dict(task_data.get("environment") or {})
    declared_env = {str(k): str(v) for k, v in (env_block.get("env") or {}).items()}
    try:
        environment_mod.check_tmpdir(declared_env)
    except environment_mod.EnvironmentError as exc:
        raise PreflightError(str(exc)) from exc

    python = str(env_block.get("python", "python3"))
    # The orchestrator and the GPUs are the same machine for what has no launch prefix: the
    # devices are claimed here, as many as the widest such deployment takes.
    if need > world_size:
        notes.append(
            f"devices: {need} claimed, not {world_size}: a disaggregated deployment starts "
            "two engines"
        )
    if not need:
        devices = ()
        notes.append("devices: none claimed here: every deployment starts behind its launch prefix")
    elif claim:
        devices = claim_devices(need, arch, python, runner=runner, notes=notes)
    else:
        devices = tuple(range(need))
        notes.append(
            "dry run: no device was queried or claimed, and the multi-rank probe did not run; "
            "a real start does both"
        )
    environment = resolve_environment(task_data, devices=devices, world_size=world_size)
    if claim and need:
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


def resolve_target_identity(
    task_data: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
) -> tuple[str, str]:
    """Return ``(family, target_rel)`` for this run.

    The family and the checkpoint segment are derived and overridable through
    ``task.yaml``'s ``target`` block. The family comes from the checkpoint's
    own ``architectures[0]``, which is the same string routing keys on: the
    family whose ``routing.py`` in the checkout declares it, else its name.
    """
    override = dict(task_data.get("target") or {})

    architectures = checkpoint_config.get("architectures") or []
    if not architectures and not override.get("family"):
        raise PreflightError(
            "the checkpoint's config.json declares no `architectures`, so the "
            "modeling_v2 family cannot be derived. Set `target.family` in task.yaml."
        )
    family = (
        override.get("family")
        or routed_family(task_data["trtllm_repo_path"], str(architectures[0]))
        or derive_family(str(architectures[0]))
    )
    segment = override.get("checkpoint_segment") or derive_checkpoint_segment(
        task_data["checkpoint_path"]
    )

    target_rel = (
        f"{MODELING_V2_PACKAGE}/models/{family}/"
        f"{segment}__{task_data['arch']}__{task_data.get('parallel') or DEFAULT_PARALLEL}"
    )
    return family, target_rel


def format_preflight_report(result: PreflightResult) -> str:
    """A human-readable summary for the journal and the console."""
    lines = [
        f"slug:        {result.slug}",
        f"branch:      {result.branch}",
        f"target:      {result.target_rel}",
        f"world size:  {result.world_size}",
        f"devices:     {', '.join(str(d) for d in result.devices) or '(none)'}",
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
    "check_kv_cache_configs",
    "check_checkpoint",
    "check_checkpoint_fit",
    "check_multirank_substrate",
    "check_processor_dependencies",
    "claim_devices",
    "format_preflight_report",
    "probe_device_alone",
    "query_devices",
    "run_preflight",
]
