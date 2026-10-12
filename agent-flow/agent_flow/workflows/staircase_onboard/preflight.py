# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase A — everything that must hold before an agent is worth starting.

This is the module that decides whether the run happens at all, and it is
deliberately **pure Python with no agent turn**. The failure mode it exists
to prevent is a run that spends its plan phase and its first assembler turn
discovering that a card is wedged, ``TMPDIR`` points into a full ``/tmp``,
MPI cannot spawn on this host, or the anchor row its release criterion
names is not in the reference file. Those are seconds of checking here and
hours of confusion there.

The asserts run **in order**, and the first failure ends the run before the
branch exists. Ordering is not stylistic: there is no point probing devices
on a host whose architecture does not match the one the receipts describe.

Two of these checks encode measurements that are easy to get wrong:

- **Judge a device by ``memory.free`` and ``utilization.gpu``, never by the
  process table.** ``nvidia-smi --query-compute-apps`` under-attributes
  badly — staircase measured 168 GiB in use against 2.7 GiB attributed —
  and orphaned processes holding a dead job's allocations do not appear in
  it at all.
- **A card at 100% utilization is *broken* for our purposes, not busy.** It
  makes engines stall mid-gate with no error and invalidates every number
  taken there.

And one that separates two causes of a single symptom:
``cudaErrorDevicesUnavailable`` from a multi-rank preflight is *equally*
what a broken card returns and what a concurrent job returns. The
lone-process probe tells them apart: a card that fails a single-process,
one-tensor allocation with no peers and nothing else running is unusable
rather than contended, and no amount of waiting will free it.
"""

from __future__ import annotations

import ast
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from agent_flow.runtime import CommandSpec, LocalCommandExecutor

from .delivery import MODELING_V2_PACKAGE
from .envelope import CertificationPlan, derive_certification
from .environment import RunEnvironment, resolve_environment
from .task_schema import (
    assembly_units,
    derive_checkpoint_segment,
    derive_family,
    derive_slug,
    members_for,
    parse_parallel_segment,
)

# A device is claimable when it is this close to empty. Not 100%: a healthy
# idle card still shows a little in use (ECC tables, the driver's own
# context), and demanding exactly zero would reject every real machine.
MIN_FREE_MEMORY_FRACTION = 0.90

# What a `unified` segment claims when neither it nor its members say. The
# segment string carries no world size by design — the member set decides —
# and this is the number `staircase-modeling-parallel` documents for the
# baseline pair.
DEFAULT_UNIFIED_WORLD_SIZE = 4

# Above this, treat the card as broken rather than busy. See the module
# docstring: a spinning neighbour does not slow a gate down, it makes the
# gate's number meaningless.
MAX_UTILIZATION_PCT = 10

# The lone-process probe and the multi-rank substrate probe both run under
# a hard deadline, because the failures they look for do not raise — they
# hang. An MPI spawn whose connect-back lands on an interface that does not
# carry it blocks forever, and "the engine never booted" three hours later
# is the same observation with none of the information.
PROBE_TIMEOUT_S = 120
MULTIRANK_PROBE_TIMEOUT_S = 300

# ---------------------------------------------------------------------------
# Why `TMPDIR` has a maximum length, and where each number comes from.
#
# The LLM API's RPC binds an IPC socket under `TMPDIR` named
# `rpc_test_<uuid4>` --- `get_unique_ipc_addr` in
# `tensorrt_llm/executor/rpc/rpc_common.py`, which is `os.path.join`ed onto
# `tempfile.gettempdir()`. A unix domain socket path is capped by the kernel
# at `sizeof(sockaddr_un.sun_path)`, and ZeroMQ reports the overflow as
# `ipc path "..." is longer than 107 characters`.
#
# So the budget is fixed on both sides and `TMPDIR` gets what is left. The
# numbers are named rather than reduced to a literal 61 because the reasoning
# is the part worth keeping: if the basename or the kernel limit ever moves,
# the derived bound moves with it and stays explicable.
#
# Measured directly rather than trusted: an `AF_UNIX` bind against synthetic
# paths succeeds at 107 characters and fails at 108 with `AF_UNIX path too
# long`, and `len("rpc_test_") + len(str(uuid.uuid4()))` is 9 + 36. The
# separator `os.path.join` inserts is the third character of the budget, and
# it is easy to drop --- doing so yields 62 and admits one `TMPDIR` that does
# not work.
# ---------------------------------------------------------------------------
UNIX_SOCKET_PATH_MAX = 107
RPC_SOCKET_BASENAME_LEN = 45
MAX_TMPDIR_LEN = UNIX_SOCKET_PATH_MAX - RPC_SOCKET_BASENAME_LEN - len("/")

CommandRunner = Callable[[Sequence[str], Mapping[str, str] | None, int], "CommandResult"]


@dataclass(frozen=True)
class CommandResult:
    """What a probe observed. ``timed_out`` is distinct from a non-zero exit."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


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
    checkpoint_segment: str = ""
    target_rel: str = ""
    checkpoint_config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: What the run certifies: the serving envelope, the gate tuples inside
    #: it, the envelope probes and the feature-matrix cells. See `envelope.py`.
    certification: CertificationPlan | None = None


class PreflightError(RuntimeError):
    """A precondition failed. The run ends here, before the branch exists."""


class SpecInconsistencyError(PreflightError):
    """The spec cannot run as written; the host is not at fault.

    Raised where preflight finds the spec's own blocks disagreeing -- a gate
    launched outside the declared serving envelope -- or asking the tree for
    something it does not have, so the operator is pointed at the spec or the
    tree rather than at the infrastructure troubleshooting guide every other
    preflight failure belongs in.
    """


class UnsupportedSpeculationError(SpecInconsistencyError):
    """A gate would ask a launch path for a speculation algorithm it has no branch for."""


#: The one process-wide local executor for probes. Stateless.
_LOCAL_EXECUTOR = LocalCommandExecutor()


def _default_runner(
    argv: Sequence[str],
    env: Mapping[str, str] | None = None,
    timeout: int = PROBE_TIMEOUT_S,
) -> CommandResult:
    """Run one probe here, as an argv, with no shell in between.

    A probe is a fixed command with fixed arguments, so there is nothing for
    a shell to do except find new ways to split them. The launch prefix —
    which *is* shell syntax — belongs to the gates, not here.
    """
    record = _LOCAL_EXECUTOR.execute(
        CommandSpec(
            argv=tuple(str(part) for part in argv), env=env, timeout_s=timeout, label="probe"
        )
    )
    return CommandResult(
        returncode=record.returncode,
        stdout=record.stdout,
        stderr=record.stderr,
        timed_out=record.timed_out,
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
    """Allocate one tensor on one card in a lone process.

    This is the check that separates *broken* from *contended*. Staircase
    measured a card reporting 0 MiB used, 0% utilization and a healthy
    fabric that still refused ``torch.cuda.set_device`` to a single
    process with no peers.
    """
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


def resolve_model_key(task_data: Mapping[str, Any], unit: str | None = None) -> str:
    """The HuggingFace repo name this unit's anchor row is keyed by.

    The same rule ``StaircaseOnboardWorkflow._model_key`` applies at gate
    time: an explicit ``accuracy.model_key`` when the spec states one,
    otherwise the last two components of the checkpoint path, which is the
    shape a repo name has (``openai/gpt-oss-120b``). It is stated here as
    well because preflight has to ask the same question hours earlier, and
    an anchor check that resolved a *different* key than the gate will is
    worse than no check at all — it passes on a row nothing ever reads.
    Once ``workflow`` is free to change, that method should delegate here so
    there is one implementation rather than two that agree today.

    ``unit`` honours the per-unit mapping form via
    :func:`~.gates.resolve_per_unit`, for the same reason ``reference_yaml``
    does: a feature increment whose anchor lives under another repo name has
    to be able to say so.
    """
    from . import gates as gates_mod

    accuracy = dict(task_data.get("accuracy") or {})
    explicit = gates_mod.resolve_per_unit(accuracy, "model_key", unit)
    if explicit:
        return str(explicit)
    parts = Path(str(task_data["checkpoint_path"]).rstrip("/")).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def check_accuracy_anchors(
    task_data: Mapping[str, Any],
    units: Sequence[str] | None = None,
) -> list[str]:
    """Assert every unit's anchor ROW resolves, before any unit is built.

    The accuracy gate that *ends* a unit looks its reference score up by
    ``model_key`` plus a discriminator set, and the match is exact — see
    :func:`~.gates.load_reference_score`. So a spec can name a benchmark, a
    reference file and a row that is not in it, and nothing says so until
    the unit has been assembled, reviewed, booted on every member and handed
    to ``trtllm-eval``: hours later, on a spec error knowable in
    milliseconds at startup, wearing the costume of a target defect.

    Not hypothetical. It is how the ``multimodal`` unit of the
    MiMo-V2.6-Flash run nearly ended — ``mmmu.yaml`` had no row for the
    checkpoint, and it was caught only because somebody happened to look
    while an earlier unit was still gating. Same class as
    :func:`check_checkpoint` and :func:`check_tmpdir`: seconds here, hours
    there, and it reads as the target's fault when it surfaces late.

    **Every unit, not just core.** ``benchmark``, ``reference_yaml``,
    ``model_key`` and ``discriminators`` each take the per-unit mapping
    form, and the unit that was broken was broken *precisely because* it is
    the one that differs from the rest.

    The lookup is :func:`~.gates.resolve_reference` — the accuracy gate's
    own, reached through the same ``resolve_per_unit`` /
    ``resolve_discriminators`` helpers — rather than a second implementation
    of the matching rule. Reuse is what makes the claim worth anything: this
    check fails exactly when the gate would, and the two cannot drift apart
    later. The import is deferred because ``gates`` already reaches back
    into this module for :func:`reports_unsupported_architecture`;
    :func:`_substrate_boot` defers for the same reason.

    Two situations are **skipped rather than failed**, because failing them
    would reject specs that work:

    - a run with no ``accuracy`` block, or a unit whose block names no
      reference file — there is no anchor to check;
    - a reference file that is not in this checkout. The path may be
      absolute and point outside the repo, or the tree may be sparse. What
      this check owns is the *row*; a file it cannot read says nothing about
      the row, and the note records the weaker claim verbatim.

    Returns notes rather than printing: every resolved anchor is recorded
    with the number it found, so the preflight report states what each unit
    will be measured against before a single turn is spent on it.
    """
    from . import gates as gates_mod

    accuracy = dict(task_data.get("accuracy") or {})
    if not accuracy:
        return []

    notes: list[str] = []
    for unit in units if units is not None else assembly_units(task_data):
        declared = str(gates_mod.resolve_per_unit(accuracy, "reference_yaml", unit) or "").strip()
        if not declared:
            notes.append(
                f"accuracy anchor [{unit}]: no reference_yaml for this unit, so there "
                f"is no row to check"
            )
            continue
        benchmark = str(gates_mod.resolve_per_unit(accuracy, "benchmark", unit) or "").strip()
        model_key = resolve_model_key(task_data, unit)
        # Resolved once and passed in, so the message and the lookup cannot
        # disagree about which row was searched for.
        wanted = gates_mod.resolve_discriminators(accuracy, unit)
        try:
            score = gates_mod.resolve_reference(
                task_data, model_key=model_key, discriminators=wanted, unit=unit
            )
        except FileNotFoundError as exc:
            notes.append(
                f"accuracy anchor [{unit}]: {exc}; the row is unchecked here, and if "
                f"it is still absent at gate time the unit dies there"
            )
            continue
        except (KeyError, TypeError, ValueError) as exc:
            # `KeyError.__str__` is `repr` of its argument, which folds a
            # carefully written multi-clause message into one escaped line.
            # That message is the valuable part -- it already names the file,
            # the model key and the discriminator sets that ARE available --
            # so surface it rather than flattening it to "not found".
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
        notes.append(
            f"accuracy anchor [{unit}]: {benchmark or 'eval'} {model_key} "
            f"{dict(wanted) if wanted else '{}'} = {score:g} ({declared})"
        )
    return notes


#: The launcher the boot, acceptance, witness and envelope-probe gates run
#: (see ``gates.run_boot_gate``), and the module whose ``SpeculativeConfig``
#: decides what ``trtllm-eval`` and the LLM API accept, both repo-relative.
QUICKSTART_REL = "examples/llm-api/quickstart_advanced.py"
LLM_ARGS_REL = "tensorrt_llm/llmapi/llm_args.py"

#: The variable ``quickstart_advanced.py`` dispatches its algorithms on.
_QUICKSTART_ALGO_NAME = "spec_decode_algo"
_SPECULATIVE_UNION = "SpeculativeConfig"


def quickstart_spec_algorithms(path: Path | str) -> set[str] | None:
    """The ``--spec_decode_algo`` values the quickstart script has a branch for.

    Read from the script's own ``spec_decode_algo == "<ALGO>"`` comparisons,
    upper-cased the way the script upper-cases its argument. None when the
    file is absent or does not parse, which means "cannot tell", not "none".
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if not any(isinstance(o, ast.Name) and o.id == _QUICKSTART_ALGO_NAME for o in operands):
            continue
        for operand in operands:
            constants = (
                operand.elts if isinstance(operand, (ast.Tuple, ast.List, ast.Set)) else [operand]
            )
            for constant in constants:
                if isinstance(constant, ast.Constant) and isinstance(constant.value, str):
                    found.add(constant.value.upper())
    return found


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
    """Every speculation algorithm the gates will emit has a branch in the launcher.

    Two launch paths, each asked differently:

    - the quickstart gates (boot, acceptance, witness, envelope probe) pass
      ``--spec_decode_algo=<ALGO>`` from ``gates.speculation_flags``. A value
      the script has no branch for falls through to ``spec_config = None``:
      the gate runs the unit WITHOUT speculation, its acceptance statistic is
      missing or 1.0, and the failure reads as a broken draft path;
    - the accuracy gate's ``trtllm-eval`` reads the unit's options YAML, whose
      ``speculative_config.decoding_type`` must be one the LLM API's
      ``SpeculativeConfig`` union accepts.

    Raises :class:`UnsupportedSpeculationError` naming the algorithm, the
    unit, the file and the branch it lacks. Returns notes, including one for a
    launcher this preflight could not read.
    """
    from .envelope import features_through
    from .gates import resolve_llm_api_options, speculation_flags

    repo = Path(str(task_data.get("trtllm_repo_path") or "."))
    accuracy = task_data.get("accuracy") or {}
    emitted: dict[str, list[str]] = {}
    asked: dict[str, list[tuple[str, str]]] = {}
    for unit in assembly_units(task_data):
        for flag in speculation_flags(features_through(task_data, unit)):
            if flag.startswith("--spec_decode_algo="):
                emitted.setdefault(flag.split("=", 1)[1].upper(), []).append(unit)
        options = resolve_llm_api_options(accuracy, unit) if isinstance(accuracy, dict) else None
        if options and Path(options).is_file():
            try:
                loaded = yaml.safe_load(Path(options).read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            config = loaded.get("speculative_config") if isinstance(loaded, dict) else None
            decoding = config.get("decoding_type") if isinstance(config, dict) else None
            if decoding:
                asked.setdefault(str(decoding), []).append((unit, str(options)))
    if not emitted and not asked:
        return []

    notes: list[str] = []
    errors: list[str] = []
    verified: set[str] = set()
    if emitted:
        quickstart = repo / QUICKSTART_REL
        supported = quickstart_spec_algorithms(quickstart)
        if supported is None:
            notes.append(
                f"speculation: cannot read {quickstart}, so whether it handles "
                f"{', '.join(sorted(emitted))} is unchecked; the boot gate will say"
            )
        else:
            for algo, units in sorted(emitted.items()):
                if algo in supported:
                    verified.add(algo)
                    continue
                errors.append(
                    f"unit(s) {', '.join(units)} make the boot/acceptance/witness/probe gates "
                    f"pass `--spec_decode_algo={algo}`, but {quickstart} has no "
                    f"`{_QUICKSTART_ALGO_NAME} == '{algo}'` branch (it handles "
                    f"{', '.join(sorted(supported)) or 'none'}); the value falls through to "
                    f"`spec_config = None` and every quickstart gate would run the unit "
                    f"without speculation. Add the {algo} branch to {QUICKSTART_REL} on this "
                    f"tree before the run"
                )
    if asked:
        llm_args = repo / LLM_ARGS_REL
        accepted = llm_api_decoding_types(llm_args)
        if accepted is None:
            notes.append(
                f"speculation: cannot read {llm_args}, so the decoding types "
                f"{', '.join(sorted(asked))} are unchecked"
            )
        else:
            lowered = {a.lower() for a in accepted}
            for decoding, uses in sorted(asked.items()):
                if decoding.lower() in lowered:
                    verified.add(decoding)
                    continue
                where = "; ".join(f"{unit} ({options})" for unit, options in uses)
                errors.append(
                    f"the accuracy gate's options ask trtllm-eval for "
                    f"`speculative_config.decoding_type: {decoding}` ({where}), but "
                    f"`{_SPECULATIVE_UNION}` in {llm_args} accepts only "
                    f"{', '.join(sorted(accepted)) or 'none'}. Add the {decoding} config to "
                    f"that union in {LLM_ARGS_REL}, or fix the options file"
                )
    if errors:
        raise UnsupportedSpeculationError(
            "a speculation algorithm the gates will ask for has no branch on this tree:\n  - "
            + "\n  - ".join(errors)
        )
    if verified:
        notes.append(
            f"speculation: {', '.join(sorted(verified))} supported by the gates' launch paths"
        )
    return notes


def check_tmpdir(env: Mapping[str, str]) -> None:
    """Assert ``TMPDIR`` and ``CLAUDE_CODE_TMPDIR`` are set, short, and not ``/tmp``.

    Engine builds, NVRTC scratch and the harness's own output all default
    there. On a shared host another tenant can fill ``/tmp`` to zero
    mid-run, and everything that was running stops at once with errors that
    point anywhere but here.

    **Length is the property that breaks first on a cluster whose storage
    paths are deep**, and it is the one an author is least likely to suspect.
    ``TMPDIR`` also has to hold a unix domain socket — see
    :data:`MAX_TMPDIR_LEN` — and a directory that is on the right filesystem,
    is writable, and has plenty of space still fails if its *name* is long.
    The observed failure is a ZeroMQ traceback naming neither this workflow
    nor the spec line that caused it, arriving after four ranks have started
    and loaded weights. Here it costs a second.

    The length bound applies to ``TMPDIR`` alone. ``CLAUDE_CODE_TMPDIR`` is
    the harness's scratch and binds no socket, so holding it to a socket's
    budget would reject a value that works.

    ``env`` is the environment **declared in task.yaml**, and this check
    deliberately does *not* fall back to the orchestrator's own
    ``os.environ``. Agents and gates run under the declared prefix, so a
    value that happens to be set in the launching shell is not a value the
    run can rely on — and a run whose scratch location is implicit is one
    nobody can reproduce later.
    """
    for name in ("TMPDIR", "CLAUDE_CODE_TMPDIR"):
        value = env.get(name)
        if not value:
            raise PreflightError(
                f"{name} is not set. Engine builds and NVRTC scratch default there, and "
                f"on a shared host the default fills to zero mid-run. Set it in "
                f"task.yaml's `environment.env`."
            )
        resolved = Path(value).resolve()
        if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
            raise PreflightError(
                f"{name}={value} points into /tmp. Another tenant can fill it to zero "
                f"mid-run; point it at scratch or lustre instead."
            )
        if name == "TMPDIR" and len(value) > MAX_TMPDIR_LEN:
            raise PreflightError(
                f"{name}={value} is {len(value)} characters. The LLM API binds an RPC "
                f"socket at `$TMPDIR/rpc_test_<uuid4>`, and a unix socket path is "
                f"capped at {UNIX_SOCKET_PATH_MAX} "
                f"(`sizeof(sockaddr_un.sun_path)`), so {name} must be at most "
                f"{MAX_TMPDIR_LEN}. Point it at a shorter path on the same storage — "
                f"this is about the length of the name, not about the filesystem, and "
                f"a short path on the same mount works."
            )


def check_multirank_substrate(
    world_size: int,
    environment: RunEnvironment,
    runner: CommandRunner = _default_runner,
) -> None:
    """Exercise an NCCL collective and an MPI dynamic spawn under a deadline.

    These are the two launch-substrate failures that otherwise present as
    an engine that simply never boots, hours in. Both hang rather than
    raise, which is why the deadline is the point of the check.
    """
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


#: How long the substrate gate may take. It loads real weights, so it is
#: minutes; but a substrate gate that needs longer than this is measuring
#: something other than "can this host run the engine at all".
SUBSTRATE_TIMEOUT_S = 1800

#: How the stock path says *"I have never heard of this architecture"*, quoted
#: from the two places that raise it. ``AutoModelForCausalLM.from_config`` and
#: the multimodal-encoder branch beside it, in
#: ``tensorrt_llm/_torch/models/modeling_auto.py``, raise ``ValueError`` after
#: ``get_registered_model_class`` returns ``None``;
#: ``AutoConfig.from_hugging_face`` and
#: ``AutoModelForCausalLM.get_trtllm_model_class`` in
#: ``tensorrt_llm/models/automodel.py`` raise ``NotImplementedError`` after
#: ``MODEL_MAP.get`` returns ``None``. Both registries are keyed on the
#: checkpoint's own ``architectures[0]``, which is the string this workflow
#: exists to introduce.
#:
#: Matched on the message text because that is all the gate has: it runs the
#: engine as a subprocess and reads a log, so there is no exception object to
#: catch. The markers are deliberately the invariant half of each message —
#: the architecture name is interpolated into it and varies per run.
UNSUPPORTED_ARCHITECTURE_MARKERS: tuple[str, ...] = (
    "Unknown architecture for AutoModelForCausalLM",
    "Unknown architecture for AutoModelForMultimodalEncoder",
    "is not supported in TRT-LLM yet",
    # `get_model_architecture`'s own miss, from `_torch/models/modeling_utils.py`:
    #   raise RuntimeError(f"Unknown model architecture: {arch}")
    # A different message from the two `AutoModelFor*` ones above and reached by
    # a different path -- the registry lookup rather than the config loader --
    # so a checkpoint can fail here without matching any of them. Found by the
    # reviewer of a unit whose acceptance gate depends on this list: if the
    # `=off` leg dies in the registry instead, the gate hard-fails rather than
    # falling back to its declared floor.
    "Unknown model architecture:",
    # `transformers`' own refusal, from `resolve_trust_remote_code`. It is the
    # same condition wearing a different coat: a checkpoint that ships an
    # `auto_map` and demands `trust_remote_code=True` is, by construction, one
    # whose architecture neither `transformers` nor TensorRT-LLM has built in —
    # the implementation travels with the weights. The stock path therefore
    # cannot load it, which is exactly what this fallback is for.
    #
    # Measured on MiMo-V2.6-Flash-RL: the substrate gate raised this rather
    # than any registry message, so a discriminator keyed only on the registry
    # missed it and the run stopped on a checkpoint nobody expected the stock
    # path to load.
    "contains custom code which must be executed to correctly load the model",
    # `transformers`' refusal of an unknown `model_type` with NO remote code to
    # fall back on, from `AutoConfig.from_pretrained`. The checkpoint is newer
    # than the installed `transformers` and ships no configuration module, so
    # neither the stock path nor `trust_remote_code` can load it -- the same
    # condition again, one step earlier than the registry.
    #
    # Measured on DeepSeek-V4.1-Flash-NVFP4 (`model_type: deepseek_v41`,
    # transformers 5.5.4): the substrate gate raised this and the run stopped,
    # instead of proving the substrate on the declared fallback checkpoint.
    "but Transformers does not recognize this architecture",
    # A registration shim: the architecture name is registered only so the
    # runtime can find something keyed on it (a multimodal input processor),
    # and building it without a modeling_v2 target raises. Once the run's own
    # config classes are registered, this is where the `=off` leg of the
    # acceptance gate dies -- after the config loader and the registry have
    # both succeeded -- so without this marker the gate hard-fails instead of
    # falling back to its declared floor. Measured on DeepSeek-V4.1-Flash-NVFP4
    # (`tensorrt_llm/_torch/models/modeling_deepseekv41.py`).
    "has no built-in implementation",
)


def reports_unsupported_architecture(detail: str, output: str) -> bool:
    """Whether this text says *"the stock path does not know this model"*.

    Text-based so that callers other than the substrate gate can use it. The
    acceptance gate needs exactly this question: its `=off` leg compares
    against built-in modeling, and a genuinely new architecture has none.
    """
    haystack = f"{detail}\n{output}"
    return any(marker in haystack for marker in UNSUPPORTED_ARCHITECTURE_MARKERS)


def _reports_unsupported_architecture(result: Any) -> bool:
    """Whether this failure is *"the stock path does not know this model"*.

    Narrow on purpose, and the narrowness is the whole design. A fallback
    keyed on any failure would answer a real kvCache sizing fault on the
    target checkpoint by silently measuring a different model and reporting
    green — the same class of mistake as gating under
    ``TRTLLM_MODELING_V2=auto``. So the discriminator is the architecture
    registry's own message and nothing else: every other failure the boot
    gate can produce is still a hard stop.
    """
    return reports_unsupported_architecture(result.detail, result.stdout_tail)


def _substrate_boot(task_data: Mapping[str, Any], environment: RunEnvironment, shell_runner: Any):
    """One substrate boot attempt, under the stock path and no features."""
    from . import gates as gates_mod

    return gates_mod.run_boot_gate(
        environment,
        task_data=task_data,
        spec=gates_mod.BootSpec(prompts=("The capital of France is",), max_tokens=4),
        runner=shell_runner,
        timeout=SUBSTRATE_TIMEOUT_S,
        mode=gates_mod.MODE_OFF,
        features=[],
    )


def substrate_gate(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    shell_runner: Any,
    notes: list[str] | None = None,
) -> None:
    """Prove this host can *generate a token* under the run's exact config.

    The single highest-value check in this module, and it was added after a
    run spent roughly half its wall clock on failures that all looked like
    model defects:

    ====================================  ==================================
    ``MPI_ERR_SPAWN: could not spawn``    the launcher choice
    ``Impossible to fit ... in kvCache``  ``max_seq_len`` defaulting to the
                                          checkpoint's 262144
    ``NVRTC ... cannot open "cuda.h"``    the venv's bin missing from PATH
    ====================================  ==================================

    Each surfaced *mid-run*, inside an agent's turn, reading as though the
    target were wrong. Each is a property of the host and the resolved
    ``runtime`` block, knowable in minutes before any agent exists.

    **It generates rather than constructs, and that is the whole point.**
    The NVRTC fault appeared on the *first generated token*, after the engine
    had loaded 22 GB of weights, allocated 150 GiB of KV cache and completed
    warm-up. A gate that stopped at "the engine constructed" would have
    passed it straight through.

    It runs the real checkpoint under the real ``runtime`` block rather than
    a toy model, because every one of the three failures above depended on
    this checkpoint's geometry or this run's flags.

    **And that is exactly why it needs a fallback.** Reaching for the run's
    own checkpoint only works for a checkpoint the stock path can load, and
    this workflow's entire purpose is checkpoints it cannot — so for a
    genuinely new architecture the highest-value check in the module is
    unavailable precisely when the run is doing the thing it exists for. When
    the stock path answers "I do not know this architecture" (and *only*
    then — see :func:`_reports_unsupported_architecture`), the gate re-runs
    itself against ``environment.substrate_checkpoint``, a proxy the stock
    path does support, under the same environment and the same ``runtime``
    block. The two host failures that do not depend on the checkpoint —
    ``MPI_ERR_SPAWN`` and NVRTC's missing ``cuda.h`` — are still proven; the
    kvCache one is not, and the note says so rather than letting the run
    believe otherwise.
    """
    result = _substrate_boot(task_data, environment, shell_runner)
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
            # Deliberately NOT the raw architecture error. That message names
            # the target's architecture and reads as a target defect, which is
            # the exact failure mode this gate was written to prevent — and
            # here there is no target yet to be defective.
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
            {**dict(task_data), "checkpoint_path": proxy}, environment, shell_runner
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
            # The weaker claim, stated in full rather than summarized. A note
            # that said only "substrate gate passed" would be read later as the
            # claim the unqualified gate makes, and the difference is precisely
            # which failure this run has been protected from.
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


def agent_plane_gate(probe: Any, notes: list[str] | None = None) -> None:
    """Prove one agent turn can happen at all, before a phase depends on it.

    Every unit test in this package replaces the agent layer with a fake, so
    a fully green suite says nothing about whether an agent can start. The
    first real run discovered that twice — once after a 24-minute planning
    turn, and once three hours later mid-batch — because a nested process
    could not write its credential lock.

    ``probe`` is a zero-argument callable returning the agent's reply; it is
    injected so this stays testable without spawning anything.
    """
    try:
        reply = probe()
    except Exception as exc:  # noqa: BLE001 - the cause is the message
        raise PreflightError(
            f"agent plane gate failed: a single agent turn could not complete, so "
            f"no phase that dispatches one can run. This is not a model or target "
            f"problem.\n\n  {type(exc).__name__}: {exc}"
        ) from exc
    if not str(reply or "").strip():
        raise PreflightError(
            "agent plane gate failed: the agent returned an empty reply. A role "
            "that cannot answer a trivial prompt will not produce a decision file."
        )
    if notes is not None:
        notes.append("agent plane gate: one agent turn completed")


def run_preflight(
    task_data: Mapping[str, Any],
    *,
    runner: CommandRunner = _default_runner,
    skip_device_claim: bool = False,
    unit_kinds: Callable[[str], Any] | None = None,
) -> PreflightResult:
    """Assert every precondition in order and resolve the run's identity.

    ``skip_device_claim`` exists for the CPU-only tests and for a
    ``--dry-run`` on a machine with no GPUs; it is never set by a real run.

    ``unit_kinds`` answers which gate kinds (boot, accuracy, witness,
    acceptance) the orchestrator runs for a unit; without it they are read
    from what the spec declares. See :func:`~.envelope.derive_certification`.
    """
    notes: list[str] = []

    checkpoint_config = check_checkpoint(task_data["checkpoint_path"])

    arch = str(task_data["arch"])
    segment = str(task_data["parallel"])
    kind, declared_world_size = parse_parallel_segment(segment)
    world_size = declared_world_size
    if world_size is None:
        # `unified` takes its world size from the MEMBER SET rather than from
        # the segment string, which is what `parse_parallel_segment` has
        # always said by returning None here. Every member declares the same
        # one -- the schema rejects a mixture, because one allocation serves
        # them all -- so the first answers for the set. 4 is the default the
        # parallel skill documents, and the size the default member set is
        # built at.
        declared_members = members_for(task_data, None, DEFAULT_UNIFIED_WORLD_SIZE)
        _, member_world_size = parse_parallel_segment(declared_members[0])
        world_size = member_world_size or DEFAULT_UNIFIED_WORLD_SIZE
        notes.append(
            f"unified segment: world size {world_size}, taken from the member set "
            f"({', '.join(declared_members)})"
        )

    # The member set and what it costs, stated where the run is CHOSEN rather
    # than discovered three gates in. Three members over four units is twelve
    # accuracy gates at roughly an hour each, and `members:` is the knob that
    # buys a cheaper answer honestly — one member, named — instead of a cheap
    # answer labelled as an expensive one. `--dry-run` prints this block, so
    # the bill is readable before anything is spent.
    units = assembly_units(task_data)
    per_unit = {unit: members_for(task_data, unit, world_size) for unit in units}
    distinct = {tuple(chosen) for chosen in per_unit.values()}
    total = sum(len(chosen) for chosen in per_unit.values())
    if len(distinct) == 1:
        shared = next(iter(distinct))
        listed = f"{', '.join(shared)} -> {len(shared)} x (boot + accuracy) per unit"
    else:
        # The per-unit form, so a run that bought a cheaper answer for one
        # unit says which unit and how cheap.
        listed = "; ".join(f"{unit}: {', '.join(chosen)}" for unit, chosen in per_unit.items())
    notes.append(
        f"members: {listed}, {total} accuracy gate(s) over {len(units)} unit(s): {', '.join(units)}"
    )

    # Immediately after the unit list is resolved, and before anything that
    # touches the host. This check is pure file reading -- it costs
    # milliseconds and needs no device, no allocation and no agent -- so
    # every expensive check below it is work that a failure here would have
    # wasted. It cannot come any earlier because it is asked PER UNIT, and
    # the unit list is decided here.
    notes.extend(check_accuracy_anchors(task_data, units))

    # The certified set, derived before anything touches the host: it is file
    # reading, and a gate that launches outside the declared envelope -- or a
    # gated member the envelope does not claim -- is a spec inconsistency the
    # run must not start with. Everything it derives (probes, external matrix
    # cells) is printed here so the bill is readable on a dry run.
    certification = derive_certification(
        task_data,
        world_size=world_size,
        gpus_per_node=_int_or_none((task_data.get("environment") or {}).get("gpus_per_node")),
        checkpoint_config=checkpoint_config,
        unit_kinds=unit_kinds,
    )
    if certification.errors:
        raise SpecInconsistencyError(
            "the serving envelope and the gates disagree:\n  - "
            + "\n  - ".join(certification.errors)
        )
    notes.extend(certification.notes())
    # File reading too, and a spec that asks a launcher for a speculation
    # algorithm it cannot build would otherwise surface at the first boot
    # gate as a unit that "drafts nothing".
    notes.extend(check_speculation_support(task_data))

    env_block = dict(task_data.get("environment") or {})
    declared_env = {str(k): str(v) for k, v in (env_block.get("env") or {}).items()}
    check_tmpdir(declared_env)

    python = str(env_block.get("python", "python3"))
    # A device claim only means anything when the orchestrator and the GPUs
    # are the *same machine*, which is to say `local`. Everywhere else the
    # allocation is the claim: the scheduler has already decided, and
    # probing this host's cards describes the wrong machine — or, on a
    # login node, no machine at all, since `nvidia-smi` cannot reach a
    # driver there and the claim fails for a run that is perfectly fine.
    #
    # Phrased as "not local" rather than as a list of remote kinds on
    # purpose: a kind added later is remote until someone says otherwise,
    # and that is the safe default. `session_spool` was exactly this case.
    scheduler_owns_devices = str(env_block.get("kind", "local")) != "local"
    devices: tuple[int, ...] = ()
    if not skip_device_claim and not scheduler_owns_devices:
        devices = claim_devices(world_size, arch, python, runner=runner, notes=notes)
    if scheduler_owns_devices:
        notes.append(
            f"{env_block.get('kind')}: devices are the scheduler's to allocate, so "
            f"preflight claims none and asserts no architecture locally. `arch: {arch}` "
            f"is taken on trust here and is what the receipts will be keyed by."
        )

    environment = resolve_environment(task_data, devices=devices, world_size=world_size)

    if not skip_device_claim and not scheduler_owns_devices:
        check_multirank_substrate(world_size, environment, runner)

    slug = derive_slug(task_data)
    family, segment, target_rel = resolve_target_identity(task_data, checkpoint_config)
    return PreflightResult(
        slug=slug,
        branch=f"staircase/{slug}",
        devices=devices,
        world_size=world_size,
        environment=environment,
        family=family,
        checkpoint_segment=segment,
        target_rel=target_rel,
        checkpoint_config=checkpoint_config,
        notes=notes,
        certification=certification,
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def resolve_target_identity(
    task_data: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Return ``(family, checkpoint_segment, target_rel)`` for this run.

    Both components are derived and both are overridable through
    ``task.yaml``'s ``target`` block. The family comes from the checkpoint's
    own ``architectures[0]``, which is the same string routing keys on — a
    hand-typed family that disagrees with it is a target nothing can reach.
    The checkpoint segment is a guess (see
    :func:`~.task_schema.derive_checkpoint_segment`), and it is the one
    identity component worth setting by hand.
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
    return family, segment, target_rel


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
    "CommandResult",
    "CommandRunner",
    "DeviceReport",
    "MAX_TMPDIR_LEN",
    "MAX_UTILIZATION_PCT",
    "MIN_FREE_MEMORY_FRACTION",
    "RPC_SOCKET_BASENAME_LEN",
    "UNIX_SOCKET_PATH_MAX",
    "PreflightError",
    "PreflightResult",
    "SpecInconsistencyError",
    "UnsupportedSpeculationError",
    "check_speculation_support",
    "llm_api_decoding_types",
    "quickstart_spec_algorithms",
    "UNSUPPORTED_ARCHITECTURE_MARKERS",
    "check_accuracy_anchors",
    "check_checkpoint",
    "check_multirank_substrate",
    "check_tmpdir",
    "claim_devices",
    "format_preflight_report",
    "probe_device_alone",
    "query_devices",
    "resolve_model_key",
    "run_preflight",
]
