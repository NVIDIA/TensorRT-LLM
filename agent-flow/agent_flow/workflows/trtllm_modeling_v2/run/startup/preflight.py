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
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from ...checks import accuracy as accuracy_checks
from ...checks import boot as boot_checks
from ...checks import check_config, check_runtime
from ...checks import disagg as disagg_checks
from ...checks.check_config import resolve_llm_api_options, speculation_flags
from ...checks.runtime_evidence import reports_unsupported_architecture
from ...infra.compute import environment as environment_mod
from ...infra.compute.commands import CommandResult, CommandSpec, LocalCommandExecutor
from ...infra.compute.environment import RunEnvironment, resolve_environment, scheduler_owns_devices
from ...infra.git import git_output
from ...operators.coverage import MODELING_V2_PACKAGE
from ...spec.derive import hybrid_marker
from ...spec.task_schema import (
    assembly_units,
    derive_checkpoint_segment,
    derive_family,
    derive_slug,
    features_through,
    members_for,
    parse_parallel_segment,
    resolve_per_unit,
    widest_world_size,
)
from ...spec.variants import ANCHOR_RELATIVE, anchor_state, base_unit, variant_gates
from ...spec.variants import variants as variants_of
from ..validation import contract as contract_mod

# A device is claimable when it is this close to empty. Not 100%: a healthy
# idle card still shows a little in use (ECC tables, the driver's own context).
MIN_FREE_MEMORY_FRACTION = 0.90

# What a `unified` segment claims when neither it nor its members say: the
# world size of the baseline TEP/DEP pair.
DEFAULT_UNIFIED_WORLD_SIZE = 4

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


class UnsupportedSpeculationError(SpecInconsistencyError):
    """A gate would ask a launch path for a speculation algorithm it has no branch for."""


class CaptureListMismatchError(SpecInconsistencyError):
    """A unit's launch paths would capture different decode CUDA-graph batch lists."""


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


def check_checkpoint_fit(task_data: Mapping[str, Any], world_size: int) -> None:
    """Refuse a spec the checkpoint cannot meet, every reason at once, from ``config.json`` alone.

    The members split its attention heads, KV heads and experts; the
    architecture runs its weight format; a speculative unit has the draft
    layers it needs, spelled as the checkpoint ships them; a multimodal unit
    has a tower.
    """
    from ...spec.derive import draft_layers, read_config

    config = read_config(task_data["checkpoint_path"]) or {}
    heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads") or heads
    experts = (
        config.get("num_experts")
        or config.get("n_routed_experts")
        or config.get("num_local_experts")
    )
    problems = []
    members = {
        m for unit in assembly_units(task_data) for m in members_for(task_data, unit, world_size)
    }
    for member in sorted(members):
        kind, world = parse_parallel_segment(member)
        if not world:
            continue
        # Attention DP replicates attention; tp and tep split its heads.
        if kind in ("tp", "tep") and heads and heads % world:
            problems.append(f"{member}: {heads} attention heads do not split {world} ways")
        if kind in ("tp", "tep") and kv_heads and kv_heads % world and world % kv_heads:
            problems.append(f"{member}: {kv_heads} KV heads do not split {world} ways")
        if kind in ("tep", "dep") and experts and experts % world:
            problems.append(f"{member}: {experts} experts do not split {world} ways")
    arch = re.match(r"^sm_(\d+)$", str(task_data.get("arch", "")))
    algo = str((config.get("quantization_config") or {}).get("quant_algo") or "").lower()
    if arch and "fp4" in algo and int(arch.group(1)) < 100:
        problems.append(f"the checkpoint's {algo} weights need sm_100 or later")
    features = [str(f) for f in task_data.get("features") or ()]
    dspark = any(str(key).startswith("dspark") for key in config)
    if "mtp" in features and dspark:
        problems.append(
            "the checkpoint's draft stages are a DSpark drafter (dspark_* keys): build dspark, not mtp"
        )
    if {"mtp", "dspark"} & set(features) and not draft_layers(config):
        problems.append("a speculative unit asks for draft layers, and config.json states none")
    if "multimodal" in features and not {"vision_config", "audio_config"} & set(config):
        problems.append("multimodal is built, and the checkpoint has no vision or audio tower")
    if problems:
        raise SpecInconsistencyError(
            "the checkpoint cannot meet the spec:\n  - " + "\n  - ".join(problems)
        )


def check_launch_settings(task_data: Mapping[str, Any]) -> list[str]:
    """Each unit's runtime flags and options file agree on every setting both state.

    Two values for one setting would measure two engines under one verdict, so
    they are refused here, every disagreement at once. Returns notes.
    """
    problems: list[str] = []
    notes: list[str] = []
    for unit in assembly_units(task_data):
        try:
            problems.extend(
                f"[{unit}] {line}" for line in check_config.launch_conflicts(task_data, unit)
            )
        except ValueError as exc:
            raise PreflightError(f"[{unit}] {exc}") from exc
    if check_config.checkpoint_remote_code(task_data.get("checkpoint_path")):
        notes.append(
            "launch: the checkpoint ships its own code (`auto_map`), so every launcher "
            "trusts it (`--trust_remote_code`, `trust_remote_code: true`)"
        )
    if problems:
        raise PreflightError(
            "the runtime flags and the options file state different values for one setting, "
            "so the shared check configuration is ambiguous:\n  "
            + "\n  ".join(problems)
            + "\nState each setting once, in either place: it reaches every launcher."
        )
    return notes


#: An anchor at or under this is a fraction or a placeholder: anchors are in percent.
PLACEHOLDER_ANCHOR_MAX = 1.0


def check_accuracy_anchors(
    task_data: Mapping[str, Any],
    units: Sequence[str] | None = None,
) -> list[str]:
    """Assert every unit's anchor ROW resolves, before any unit is built.

    The lookup is :func:`~.accuracy.resolve_reference`, the accuracy gate's own,
    so this check fails exactly when the gate would. Skipped rather than
    failed: a run with no ``accuracy`` block, a unit whose block names no
    reference file, and a reference file that is not in this checkout (a
    note records the weaker claim). Returns notes: every resolved anchor with
    the number it found.
    """
    accuracy = dict(task_data.get("accuracy") or {})
    if not accuracy:
        return []

    notes: list[str] = []
    for unit in units if units is not None else assembly_units(task_data):
        declared = str(resolve_per_unit(accuracy, "reference_yaml", unit) or "").strip()
        if not declared:
            notes.append(
                f"accuracy anchor [{unit}]: no reference_yaml for this unit, so there "
                f"is no row to check"
            )
            continue
        benchmark = str(resolve_per_unit(accuracy, "benchmark", unit) or "").strip()
        model_key = accuracy_checks.resolve_model_key(task_data, unit)
        # Resolved once and passed in, so the message and the lookup cannot
        # disagree about which row was searched for.
        wanted = accuracy_checks.resolve_discriminators(accuracy, unit)
        try:
            score = accuracy_checks.resolve_reference(
                task_data, model_key=model_key, discriminators=wanted, unit=unit
            )
        except FileNotFoundError as exc:
            notes.append(
                f"accuracy anchor [{unit}]: {exc}; the row is unchecked here, and if "
                f"it is still absent at gate time the unit dies there"
            )
            continue
        except (KeyError, TypeError, ValueError) as exc:
            # `KeyError.__str__` is `repr` of its argument, which folds the
            # multi-clause message into one escaped line: surface it as written.
            detail = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
            raise PreflightError(
                f"accuracy anchor for unit `{unit}` does not resolve, so the gate that "
                f"ENDS this unit cannot run.\n\n"
                f"  unit:           {unit}\n"
                f"  benchmark:      {benchmark or '(unset)'}\n"
                f"  reference_yaml: {declared}\n"
                f"  model_key:      {model_key}\n"
                f"  discriminators: {dict(wanted) if wanted else '{}'}\n\n"
                f"  {detail}\n\n"
                f"Nothing here is about the target — no target exists yet. Add the row "
                f"to the reference file, citing the written source the number came from "
                f"(an anchor is never invented), or fix `accuracy.model_key` / "
                f"`accuracy.discriminators` for this unit. Left alone, this unit would "
                f"assemble, boot on every member and die at its accuracy gate hours "
                f"from now, on this."
            ) from exc
        if score <= PLACEHOLDER_ANCHOR_MAX:
            raise PreflightError(
                f"accuracy anchor for unit `{unit}` is {score:g} ({declared}: {model_key}). "
                f"Anchors are recorded in percent, so this reads as a fraction. Record the "
                f"score in percent from a written source, or a low placeholder in percent "
                f"(e.g. 10 for GSM8K: it catches an engine that answers nothing) until a "
                f"measured one replaces it."
            )
        uncommitted = _uncommitted(task_data, declared)
        if uncommitted:
            raise PreflightError(
                f"accuracy anchor for unit `{unit}`: the checkout holds {uncommitted} "
                f"uncommitted. Each check measures a snapshot of a commit and reads the anchor "
                f"table from it, so this row exists for preflight and not for any gate: commit "
                f"it to the run's branch."
            )
        notes.append(
            f"accuracy anchor [{unit}]: {benchmark or 'eval'} {model_key} "
            f"{dict(wanted) if wanted else '{}'} = {score:g} ({declared})"
        )
    return notes


def _uncommitted(task_data: Mapping[str, Any], declared: str) -> str:
    """``declared`` (an anchor table) as the checkout's git holds it uncommitted; "" otherwise."""
    repo = Path(str(task_data.get("trtllm_repo_path") or ""))
    path = Path(declared) if Path(declared).is_absolute() else repo / declared
    try:
        rel = path.resolve().relative_to(repo.resolve())
    except (OSError, ValueError):
        return ""  # outside the checkout: a snapshot does not move it
    dirt = git_output(repo, "status", "--porcelain", "--untracked-files=all", "--", str(rel))
    return str(rel) if dirt and dirt.strip() else ""


def check_variant_anchors(task_data: Mapping[str, Any], world_size: int) -> list[str]:
    """Every configuration variant with an accuracy gate has an anchor its gate can read.

    A variant that names a row of its own looks it up under the variant's name,
    with its own ``accuracy`` overlay (``checks/check_config.variant_spec``). A
    relative one (``spec.variants.anchor_state``) is judged against its unit's
    benchmark at the same member, which the contract must owe.
    """
    notes: list[str] = []
    contract = None
    for name in variants_of(task_data):
        if "accuracy" not in variant_gates(task_data, name):
            continue
        if anchor_state(task_data, name) == ANCHOR_RELATIVE:
            contract = contract or contract_mod.compile_contract(task_data, world_size)
            own = [c for c in contract.checks if c.unit == name and c.gate.startswith("accuracy[")]
            alone = [c.member for c in own if not c.after]
            if alone:
                raise PreflightError(
                    f"configuration variant `{name}` is judged against its unit's own score "
                    f"(its anchor is relative), but the contract measures "
                    f"`{base_unit(task_data, name)}`'s benchmark at none of {', '.join(alone)}: "
                    f"run the variant's benchmark where its unit's runs (its `members`), or give "
                    f"it a reference row of its own (`accuracy: {{anchor: recorded, "
                    f"discriminators: ...}}`)."
                )
            notes.append(
                f"accuracy anchor [{name}]: relative, {base_unit(task_data, name)}'s own score "
                f"at {', '.join(c.member for c in own) or 'its member'}"
            )
            continue
        spec = check_config.variant_spec(task_data, name, world_size=world_size)
        notes.extend(check_accuracy_anchors(spec, [name]))
    return notes


def check_disagg_placement(task_data: Mapping[str, Any]) -> list[str]:
    """The disaggregated-serving test asks for no more devices than one node holds.

    The tree's launcher (``launch_disaggregated_llm``) starts every worker on one
    host; a job asking for more devices than a node has is refused by the spool.
    """
    spec = disagg_checks.DisaggSpec.from_task(task_data)
    if spec is None or not spec.devices or not spec.node_gpus:
        return []
    if spec.devices > spec.node_gpus:
        raise PreflightError(
            f"accuracy.disagg.devices is {spec.devices}, more than the {spec.node_gpus} devices "
            f"one node holds: the tree's disaggregated launcher starts every worker on one "
            f"host, so the test runs on one node. Write it for {spec.node_gpus} devices or fewer "
            f"(its context and generation workers between them) and set `devices` to match."
        )
    return [f"disagg: the test {spec.test} runs on {spec.devices} devices of one node"]


#: Public LLM configuration schema, read without importing the GPU runtime.
LLM_ARGS_REL = "tensorrt_llm/llmapi/llm_args.py"
_SPECULATIVE_UNION = "SpeculativeConfig"


def llm_api_decoding_types(path: Path | str) -> set[str] | None:
    """The ``speculative_config.decoding_type`` values the LLM API accepts.

    The ``decoding_type`` literals of the classes in ``SpeculativeConfig``'s
    union; every decoding config's literal when the union cannot be found.
    None when the file is absent or does not parse.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    literals: dict[str, str] = {}
    members: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                    and item.target.id == "decoding_type"
                    and isinstance(item.annotation, ast.Subscript)
                ):
                    value = item.annotation.slice
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        literals[node.name] = value.value
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            first = node.targets[0]
            target, value = (first.id if isinstance(first, ast.Name) else None), node.value
        if target == _SPECULATIVE_UNION and value is not None:
            members = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
    if members:
        return {literal for name, literal in literals.items() if name in members}
    return set(literals.values())


def check_speculation_support(task_data: Mapping[str, Any]) -> list[str]:
    """Every requested speculation type is accepted by the public LLM config union."""
    repo = Path(str(task_data.get("trtllm_repo_path") or "."))
    asked: set[str] = set()
    for unit in assembly_units(task_data):
        for flag in speculation_flags(features_through(task_data, unit)):
            if flag.startswith("--spec_decode_algo="):
                asked.add(flag.split("=", 1)[1])
        path = resolve_llm_api_options(task_data.get("accuracy") or {}, unit)
        loaded = check_config.load_llm_api_options(path)
        config = loaded.get("speculative_config") or {}
        if config.get("decoding_type"):
            asked.add(str(config["decoding_type"]))
    if not asked:
        return []
    accepted = llm_api_decoding_types(repo / LLM_ARGS_REL)
    if accepted is None:
        return [
            f"speculation: public LLM schema {repo / LLM_ARGS_REL} unavailable; support UNKNOWN until construction"
        ]
    missing = sorted(a for a in asked if a.lower() not in {v.lower() for v in accepted})
    if missing:
        raise UnsupportedSpeculationError(
            f"public LLM speculative_config does not support {', '.join(missing)}; "
            f"{repo / LLM_ARGS_REL} accepts {', '.join(sorted(accepted))}"
        )
    return [f"speculation: {', '.join(sorted(asked))} supported by public LLM configuration"]


_LLM_API_GATE = "accuracy (trtllm-eval)"


def check_capture_lists(task_data: Mapping[str, Any]) -> list[str]:
    """Validate one shared graph configuration and report its planning preview."""
    try:
        captures = check_config.launch_captures(task_data)
    except ValueError as exc:
        raise CaptureListMismatchError(
            f"CUDA-graph configuration cannot be resolved: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        return [f"cuda graphs: an options file does not parse ({exc}); capture lists unchecked"]
    return [
        f"cuda graphs: {c.name}: {c.resolved.describe() if c.resolved else 'eager'}, "
        "at boot, acceptance and accuracy alike (preview, not actual capture evidence)"
        for c in captures
    ]


def check_kv_cache_configs(
    task_data: Mapping[str, Any], checkpoint_config: Mapping[str, Any]
) -> list[str]:
    """The shared requested KV-cache configuration and possible hybrid reuse loss.

    A note states the resolved ``kv_cache_config`` keys every check takes.
    A unit (or configuration variant) that asks for block reuse
    with options that set no Mamba state snapshot policy, on a checkpoint
    :func:`hybrid_marker` marks hybrid, gets a note: the engine turns block
    reuse off in every gate it runs. A note rather than a refusal, because the
    marker is read from ``config.json`` while the engine's own test decides.
    """
    try:
        launches = check_config.launch_kv_caches(task_data)
    except yaml.YAMLError as exc:
        return [
            f"kv cache: an options file does not parse ({exc}); KV-cache configuration unchecked"
        ]

    def label(launch: check_config.LaunchKvCache) -> str:
        return launch.name if launch.name == launch.unit else f"{launch.name} ({launch.unit})"

    notes: list[str] = []
    handed: dict[str, list[str]] = {}
    for launch in launches:
        if launch.config:
            described = json.dumps(dict(launch.config), sort_keys=True)
            handed.setdefault(described, []).append(label(launch))
    for described, labels in handed.items():
        notes.append(
            f"kv cache: {', '.join(labels)}: boot, acceptance and {_LLM_API_GATE} use "
            f"`kv_cache_config` {described}"
        )
    marker = hybrid_marker(checkpoint_config)
    exposed = [label(launch) for launch in launches if launch.reuse and not launch.snapshots]
    if marker and exposed:
        notes.append(
            f"block reuse: {', '.join(exposed)}: ask for KV-cache block reuse, and their "
            f"options set no Mamba state snapshot policy, on a hybrid checkpoint "
            f"({marker} in config.json). The engine turns block reuse off without one, so it "
            f"will be off in every gate they run, and the boot, acceptance and accuracy "
            f"gates fail on the engine's warning. Set "
            f"`kv_cache_config.mamba_state_config.periodic_snapshot_interval` (or snapshot "
            f"offsets) in their options file, or turn `kv_cache_reuse` off for them"
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
    """Load the checkpoint's processor with the run's interpreter when a unit is multimodal.

    A module the processor imports and the environment lacks is an
    infrastructure error, raised here with the module's name. Any other
    failure is returned as a note: the target may supply what the stock
    processor cannot.
    """
    if "multimodal" not in assembly_units(task_data):
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
        "t = torch.ones(1, device='cuda');"
        "dist.all_reduce(t);"
        "dist.barrier();"
        "print('[preflight] OK' if t.item() == dist.get_world_size() else '[preflight] BAD')"
    )
    argv = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        "--nnodes=1",
        "-c",
        script,
    ]
    result = runner(argv, dict(environment.env), MULTIRANK_PROBE_TIMEOUT_S)
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


#: How long the substrate gate may take; it loads real weights.
SUBSTRATE_TIMEOUT_S = 1800


def _reports_unsupported_architecture(result: Any) -> bool:
    """Whether this failure is *"the stock path does not know this model"*.

    Narrow on purpose: a fallback keyed on any failure would answer a real
    kvCache sizing fault by silently measuring a different model. Every other
    failure the boot gate can produce is still a hard stop.
    """
    return reports_unsupported_architecture(result.detail, result.stdout_tail)


def _substrate_boot(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    shell_runner: Any,
    workdir: str | Path,
):
    """One substrate boot attempt, under the stock path and no features."""
    return boot_checks.run_boot_gate(
        environment,
        task_data=task_data,
        spec=boot_checks.BootSpec(prompts=("The capital of France is",), max_tokens=4),
        runner=shell_runner,
        timeout=SUBSTRATE_TIMEOUT_S,
        mode=check_runtime.MODE_OFF,
        features=[],
        workdir=workdir,
    )


def substrate_gate(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    shell_runner: Any,
    workdir: str | Path,
    notes: list[str] | None = None,
) -> None:
    """Prove this host can *generate a token* under the run's exact config.

    It catches failures that otherwise look like model defects
    (``MPI_ERR_SPAWN``, a kvCache that cannot fit ``max_seq_len``, NVRTC's
    missing ``cuda.h``, which appears only on the first generated token). It
    runs the real checkpoint under the real ``runtime`` block. When the stock
    path answers "I do not know this architecture" (and *only* then, see
    :func:`_reports_unsupported_architecture`), the gate re-runs itself
    against ``environment.substrate_checkpoint``, a proxy the stock path does
    support; the kvCache fault is then unproven, and the note says so.

    ``workdir`` is where the boot's driver spec is written. The proxy attempt
    writes under ``workdir/proxy``.
    """
    result = _substrate_boot(task_data, environment, shell_runner, workdir)
    if not result.passed and not _reports_unsupported_architecture(result):
        raise PreflightError(
            "substrate gate failed: this host cannot generate a token under the "
            "run's own environment and runtime block, using the STOCK modeling "
            "path. Nothing here is about the target — no target exists yet — so "
            "fix the host or the spec before dispatching an agent.\n\n"
            f"  {result.detail}\n\n"
            f"command:\n  {result.command}\n\n"
            f"last lines:\n{result.stdout_tail}"
        )

    if not result.passed:
        proxy = str((task_data.get("environment") or {}).get("substrate_checkpoint") or "").strip()
        if not proxy:
            # Deliberately NOT the raw architecture error, which reads as a
            # target defect when there is no target yet.
            raise PreflightError(
                "substrate gate: the stock modeling path does not know this "
                "checkpoint's architecture, which is the normal state of affairs "
                "for a model this workflow is onboarding — it is not a fault in "
                "the checkpoint or the host. But it means the substrate cannot be "
                "proven on this checkpoint at all.\n\n"
                "Declare `environment.substrate_checkpoint`: a checkpoint the stock "
                "path DOES support, which the gate will run under this same "
                "environment and runtime block instead. Do not reach for "
                "`--skip-substrate-gate` — that also gives up the MPI spawn and "
                "NVRTC checks, which do not depend on the checkpoint at all.\n\n"
                f"  {result.detail}\n\n"
                f"last lines:\n{result.stdout_tail}"
            )
        proxy_result = _substrate_boot(
            {**dict(task_data), "checkpoint_path": proxy},
            environment,
            shell_runner,
            Path(workdir) / "proxy",
        )
        if not proxy_result.passed:
            raise PreflightError(
                "substrate gate failed on the declared proxy checkpoint "
                f"{proxy}: this host cannot generate a token under the run's own "
                "environment and runtime block, using the STOCK modeling path and a "
                "checkpoint that path supports. Nothing here is about the target — "
                "no target exists yet — so fix the host or the spec before "
                "dispatching an agent.\n\n"
                f"  {proxy_result.detail}\n\n"
                f"command:\n  {proxy_result.command}\n\n"
                f"last lines:\n{proxy_result.stdout_tail}"
            )
        if notes is not None:
            # The weaker claim, stated in full.
            notes.append(
                f"substrate proven on `{proxy}`; this checkpoint's own geometry is "
                f"unverified, so a kvCache sizing fault will surface at the first "
                f"boot gate rather than here."
            )
        return

    if notes is not None:
        notes.append(
            f"substrate gate: stock path generated under the resolved environment "
            f"in {result.duration_s:.0f}s"
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
    segment = str(task_data["parallel"])
    kind, declared_world_size = parse_parallel_segment(segment)
    world_size = declared_world_size
    if world_size is None:
        # `unified` takes its world size from the MEMBER SET: the widest member,
        # the most devices one gate asks for.
        world_size = widest_world_size(task_data, DEFAULT_UNIFIED_WORLD_SIZE)
        declared_members = members_for(task_data, None, world_size)
        notes.append(
            f"unified segment: world size {world_size}, the widest of the member set "
            f"({', '.join(declared_members)})"
        )

    # The checks that only read files run before anything that touches the host.
    units = assembly_units(task_data)
    check_checkpoint_fit(task_data, world_size)
    notes.extend(check_accuracy_anchors(task_data, units))
    notes.extend(check_variant_anchors(task_data, world_size))
    notes.extend(check_launch_settings(task_data))
    notes.extend(check_disagg_placement(task_data))
    notes.extend(check_speculation_support(task_data))
    notes.extend(check_capture_lists(task_data))
    notes.extend(check_kv_cache_configs(task_data, checkpoint_config))

    env_block = dict(task_data.get("environment") or {})
    declared_env = {str(k): str(v) for k, v in (env_block.get("env") or {}).items()}
    try:
        environment_mod.check_tmpdir(declared_env)
    except environment_mod.EnvironmentError as exc:
        raise PreflightError(str(exc)) from exc

    python = str(env_block.get("python", "python3"))
    # A device claim only means anything when the orchestrator and the GPUs
    # are the *same machine* (`local`); under a scheduler the allocation is the claim.
    scheduled = scheduler_owns_devices(str(env_block.get("kind", "local")))
    devices: tuple[int, ...] = ()
    if not scheduled:
        devices = claim_devices(world_size, arch, python, runner=runner, notes=notes)
    if scheduled:
        notes.append(
            f"{env_block.get('kind')}: devices are the scheduler's to allocate, so "
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


def resolve_target_identity(
    task_data: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
) -> tuple[str, str]:
    """Return ``(family, target_rel)`` for this run.

    The family and the checkpoint segment are derived and overridable through
    ``task.yaml``'s ``target`` block. The family comes from the checkpoint's
    own ``architectures[0]``, which is the same string routing keys on.
    """
    override = dict(task_data.get("target") or {})

    architectures = checkpoint_config.get("architectures") or []
    if not architectures and not override.get("family"):
        raise PreflightError(
            "the checkpoint's config.json declares no `architectures`, so the "
            "modeling_v2 family cannot be derived. Set `target.family` in task.yaml."
        )
    family = override.get("family") or derive_family(str(architectures[0]))
    segment = override.get("checkpoint_segment") or derive_checkpoint_segment(
        task_data["checkpoint_path"]
    )

    target_rel = (
        f"{MODELING_V2_PACKAGE}/models/{family}/"
        f"{segment}__{task_data['arch']}__{task_data['parallel']}"
    )
    return family, target_rel


def format_preflight_report(result: PreflightResult) -> str:
    """A human-readable summary for the journal and the console."""
    lines = [
        f"slug:        {result.slug}",
        f"branch:      {result.branch}",
        f"target:      {result.target_rel}",
        f"world size:  {result.world_size}",
        f"devices:     {', '.join(str(d) for d in result.devices) or '(allocated by Slurm)'}",
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
    "CaptureListMismatchError",
    "PreflightError",
    "PreflightResult",
    "SpecInconsistencyError",
    "UnsupportedSpeculationError",
    "check_capture_lists",
    "check_disagg_placement",
    "check_variant_anchors",
    "check_kv_cache_configs",
    "check_speculation_support",
    "llm_api_decoding_types",
    "check_accuracy_anchors",
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
