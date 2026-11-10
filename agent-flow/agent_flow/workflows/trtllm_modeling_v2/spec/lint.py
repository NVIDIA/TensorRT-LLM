# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intake lint: what a spec will cost and where it is likely to stall, before any GPU is spent.

The schema says whether a spec is well formed. This says whether it is a good
one for the request: settings that make every gate slow (eager graphs, a batch
below the concurrency it claims), scope that multiplies the contract (variant
cross products, a feature proven at several members or again under another
axis, a member too narrow for one engine, a disaggregated member wider than a
session), accuracy that
cannot judge anything (a placeholder anchor, a chat-template protocol cut off
early), boot probes that fail on near ties, a launch the environment cannot
start, a compute session no gate reaches, and notes that tell the roles what
not to read but nothing about the model. Each finding names its rule, why it
costs time and the usual fix.

Every rule reads the spec and the checkpoint's ``config.json``, never a model
name: a :data:`RULES` entry is a function of a :class:`LintContext` yielding
:class:`Finding` s, and a new feature or engine limit adds one there (engine
limits on the checkpoint's geometry go in ``prior_art.ENGINE_RULES``).

A finding the spec keeps on purpose is accepted in ``notes`` with a line
``intake-accept: <rule>[, <rule>] -- <why>``: every role reads the reason.
``trtllm-modeling-v2 intake lint`` exits 1 while an unaccepted ``warn`` or any
``block`` stands. Preflight journals the same report and never stops on it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import yaml

from ..checks import accuracy_policy, check_config
from ..checks import disagg as disagg_checks
from ..operators import prior_art, tree_anchors
from .task_schema import (
    assembly_units,
    gpus_per_node,
    members_for,
    parse_parallel_segment,
    profile_of,
    widest_world_size,
)

BLOCK = "block"
WARN = "warn"
INFO = "info"

#: Blocks a request may still keep on purpose, accepted in notes like a warn: an accuracy
#: check run eager on purpose (a short subset, a capture still being built).
ACCEPTABLE_BLOCKS: frozenset[str] = frozenset({"accuracy-eager"})
#: A contract past this many required checks is worth a second look before it is paid for:
#: every check loads the model, and each certification round runs every one of them.
CONTRACT_REVIEW_SIZE = 30
#: Per benchmark: an anchor below this cannot tell a working target from a broken one.
PLAUSIBLE_ANCHOR: dict[str, float] = {"gsm8k": 60.0}
#: Output tokens a chat-template protocol needs for a model that reasons first.
CHAT_TEMPLATE_MIN_OUTPUT = 1024
#: The share of a device's memory one engine's weights may take: the rest holds the KV cache,
#: activations and CUDA graphs. A member below the width this implies cannot serve the model.
#: (A 344-GiB checkpoint has served disaggregated on two 277-GiB devices per engine: 62%.)
ENGINE_WEIGHT_FRACTION = 0.75
#: Notes past this many characters stop being a brief: every role reads them in every turn.
NOTES_BRIEF_CHARS = 6000

_ACCEPT = re.compile(r"intake-accept:\s*([a-z0-9-]+(?:\s*,\s*[a-z0-9-]+)*)", re.I)
# Words about the intake itself. At launch every role reads them as standing orders.
_INTAKE_PROCESS = re.compile(
    r"\bintake[- ]only\b"
    r"|\bthis intake\b"
    r"|\bnot (?:an? )?authori[sz]ation to (?:launch|run|start)"
    r"|\b(?:do not|don't|never)\s+(?:start|launch|run)\s+(?:the\s+)?(?:workflow|run)\b"
    r"|\bno\s+(?:workflow|agent turns?|builds?|slurm jobs?)\b[^.]{0,120}\b(?:allowed|permitted)\b",
    re.I,
)
_BRIEF_WORDS = ("architecture", "prior art")
_PLACEHOLDER_WORDS = ("placeholder", "uncalibrated", "unmeasured", "pending")


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    message: str
    fix: str = ""

    def render(self) -> str:
        fix = f"\n  - fix: {self.fix}" if self.fix else ""
        return f"- **{self.severity}** `{self.rule}`: {self.message}{fix}"


@dataclass
class LintReport:
    findings: list[Finding] = field(default_factory=list)
    accepted: set[str] = field(default_factory=set)
    contract_lines: list[str] = field(default_factory=list)
    prior_art: prior_art.PriorArt | None = None
    constraints: list[str] = field(default_factory=list)

    def open(self) -> list[Finding]:
        """Findings that still stand: every block and every warn the notes do not accept.

        An :data:`ACCEPTABLE_BLOCKS` block the notes accept no longer stands.
        """
        return [
            f
            for f in self.findings
            if (
                f.severity == BLOCK
                and not (f.rule in ACCEPTABLE_BLOCKS and f.rule in self.accepted)
            )
            or (f.severity == WARN and f.rule not in self.accepted)
        ]

    def summary(self) -> str:
        counts = {s: sum(f.severity == s for f in self.findings) for s in (BLOCK, WARN, INFO)}
        return (
            f"{counts[BLOCK]} block, {counts[WARN]} warn "
            f"({len([f for f in self.findings if f.severity == WARN and f.rule in self.accepted])} "
            f"accepted), {counts[INFO]} info"
        )

    def render(self, *, prior_art_limit: int = 12) -> str:
        lines = ["# Intake lint", "", self.summary(), "", "## Findings", ""]
        if not self.findings:
            lines.append("None.")
        for finding in self.findings:
            mark = " (accepted in notes)" if finding.rule in self.accepted else ""
            lines.append(finding.render() + mark)
        if self.contract_lines:
            lines += ["", "## Contract", "", *self.contract_lines]
        if self.constraints:
            lines += ["", "## Engine constraints for this geometry", ""]
            lines += [f"- {c}" for c in self.constraints]
        if self.prior_art is not None:
            lines += [
                "",
                "## Prior art in this tree",
                "",
                "Ranked by the checkpoint config's vocabulary. Read the ones that fit and name them "
                'in `notes` ("Prior art in this tree worth surveying: ..."); every role reads it.',
                "",
                self.prior_art.render(limit=prior_art_limit),
            ]
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class LintContext:
    """What every rule reads: the spec, and what intake found around it."""

    task: Mapping[str, Any]
    world: int
    #: The compiled contract, or ``None`` when it does not compile (``contract_error`` says why).
    contract: Any = None
    contract_error: str = ""
    #: The checkpoint's ``config.json``; ``None`` when it cannot be read.
    config: Mapping[str, Any] | None = None
    repo: str | Path | None = None
    #: The spec's directory, for the relative paths of a spec the schema did not resolve.
    base: Path | None = None
    constraints: list[str] = field(default_factory=list)
    prior_art: prior_art.PriorArt | None = None
    #: The checkpoint's weights in bytes (its safetensors index, else its files); ``None`` unread.
    weight_bytes: int | None = None

    @property
    def notes(self) -> str:
        return str(self.task.get("notes") or "")

    @property
    def accuracy(self) -> Mapping[str, Any]:
        return self.task.get("accuracy") or {}

    @property
    def environment(self) -> Mapping[str, Any]:
        return self.task.get("environment") or {}


Rule = Callable[[LintContext], Iterable[Finding]]


_gpus_per_node = gpus_per_node


def _benchmark(accuracy: Mapping[str, Any]) -> str:
    return str(accuracy.get("benchmark") or "gsm8k")


def _benchmark_args(accuracy: Mapping[str, Any]) -> list[str]:
    args = accuracy.get("benchmark_args") or []
    if isinstance(args, Mapping):
        args = args.get("default") or next(iter(args.values()), [])
    return [str(a) for a in args]


def checkpoint_weight_bytes(checkpoint: str | Path | None) -> int | None:
    """The checkpoint's weights in bytes: its safetensors index's total, else its files' sizes."""
    if not checkpoint:
        return None
    root = Path(str(checkpoint))
    try:
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        total = int((index.get("metadata") or {}).get("total_size") or 0)
        if total > 0:
            return total
    except (OSError, ValueError, TypeError):
        pass
    try:
        total = sum(f.stat().st_size for f in root.glob("*.safetensors"))
    except OSError:
        return None
    return total or None


def engine_width(weight_bytes: int | None, gpu_memory_gib: Any) -> int | None:
    """The fewest devices (a power of two) one engine's weights fit on; ``None`` unknown."""
    memory = gpu_memory_gib
    if not weight_bytes or not isinstance(memory, (int, float)) or memory <= 0:
        return None
    room = ENGINE_WEIGHT_FRACTION * float(memory) * 2**30
    width = 1
    while weight_bytes / width > room:
        width *= 2
    return width


def _engine_width(ctx: LintContext) -> int | None:
    return engine_width(ctx.weight_bytes, ctx.environment.get("gpu_memory_gib"))


def _gib(nbytes: float) -> str:
    return f"{nbytes / 2**30:.0f} GiB"


def _host_path(environment: Mapping[str, Any], path: str) -> Path | None:
    """Where ``path``, as the run's commands see it, lives on this host; ``None``: in the image only."""
    if not environment.get("container_image"):
        return Path(path)
    for mount in environment.get("mounts") or ():
        source, _, rest = str(mount).partition(":")
        target = (rest.split(":", 1)[0] or source).rstrip("/")
        if path == target or path.startswith(target + "/"):
            return Path(source.rstrip("/") + path[len(target) :])
    return None


# ---------------------------------------------------------------- the rules


def cuda_graph_rules(ctx: LintContext) -> Iterator[Finding]:
    """Every accuracy check runs captured: each unit's and variant's graphs as its gates resolve them."""
    try:
        captures = check_config.launch_captures(ctx.task)
    except Exception:  # an options file that does not resolve is the schema's to report
        return
    eager = [c.name for c in captures if c.resolved is None]
    if ctx.contract is None:
        checked = scored = set(eager)
    else:
        # The disaggregated-serving test configures its own engines, whatever `runtime` says.
        required = [c for c in ctx.contract.required() if c.gate != disagg_checks.GATE_NAME]
        checked = {c.unit for c in required}
        scored = {c.unit for c in required if str(c.gate).startswith("accuracy")}
    blocked = [name for name in eager if name in scored]
    fix = (
        "set `runtime.cuda_graph: true` and drop `cuda_graph: false` or `cuda_graph_config: null` "
        "from the units' options and the variants; a target that cannot capture yet is usually a "
        "task for the graph. To run eager on purpose (a short benchmark subset, say), keep it with "
        "`intake-accept: accuracy-eager -- <why>` in notes"
    )
    if blocked:
        yield Finding(
            "accuracy-eager",
            BLOCK,
            f"the accuracy benchmark runs without CUDA graphs for {', '.join(blocked)}: a full "
            "benchmark takes ~10-50x longer, each fix-and-recheck pays it again, and what is "
            "scored is not the captured path the target serves",
            fix,
        )
    rest = [name for name in eager if name in checked and name not in scored]
    if rest:
        yield Finding(
            "runtime-cuda-graph-off",
            WARN,
            f"the checks of {', '.join(rest)} run without CUDA graphs: none exercises the "
            "captured path the target serves there",
            fix,
        )


def kv_cache_rules(ctx: LintContext) -> Iterator[Finding]:
    """Every ``kv_cache_config.dtype`` a launch asks for is one the engine loads a model with."""
    accepted = prior_art.engine_kv_dtypes(ctx.repo)
    if accepted is None:
        return
    values, source = accepted
    asked: dict[str, list[str]] = {}
    try:
        launches = check_config.launch_kv_caches(ctx.task)
    except Exception:  # an options file that does not resolve is the schema's to report
        launches = []
    for launch in launches:
        dtype = (launch.config or {}).get("dtype")
        if dtype is not None:
            asked.setdefault(str(dtype), []).append(launch.name)
    for dtype, where in sorted(asked.items()):
        if dtype not in values:
            yield Finding(
                "kv-cache-dtype",
                BLOCK,
                f"kv_cache_config.dtype {dtype!r} ({', '.join(where)}) is not one the engine "
                f"loads a model with ({source}: {', '.join(values)}): every boot fails",
                f"use one of {', '.join(values)}; `auto` keeps the checkpoint's own KV dtype "
                "(the model dtype, e.g. BF16, when the checkpoint names no KV quantization)",
            )


def placement_rules(ctx: LintContext) -> Iterator[Finding]:
    """Every member holds one engine of the checkpoint; disaggregation's two engines fit one node."""
    task = ctx.task
    width = _engine_width(ctx)
    memory = ctx.environment.get("gpu_memory_gib")
    if ctx.contract is not None and width is not None and width > 1:
        members = {c.member for c in ctx.contract.required()}
        narrow = sorted((m for m in members if _width(m) < width), key=lambda m: (_width(m), m))
        if narrow:
            yield Finding(
                "member-too-narrow",
                WARN,
                f"{', '.join(narrow)} hold one engine on fewer than {width} devices: the "
                f"checkpoint's {_gib(ctx.weight_bytes or 0)} of weights take more than "
                f"{ENGINE_WEIGHT_FRACTION:.0%} of each {memory:g}-GiB device there, which leaves "
                "the KV cache and CUDA graphs no room, if it loads at all",
                f"gate members of {width} devices or more; an estimate from the weights alone, "
                "so accept it in notes where a member is known to serve",
            )
    per_node = _gpus_per_node(task)
    if not per_node:
        return
    if "disagg" not in assembly_units(task):
        return
    spec = disagg_checks.DisaggSpec.from_task(task) or disagg_checks.DisaggSpec(
        test="", node_gpus=per_node
    )
    required = ctx.contract.required() if ctx.contract is not None else ()
    gated = [c.member for c in required if c.gate == disagg_checks.GATE_NAME]
    member = gated[0] if gated else (members_for(task, "disagg", ctx.world) or [None])[0]
    if spec.devices and spec.devices > per_node:
        yield Finding(
            "disagg-placement",
            BLOCK,
            f"accuracy.disagg.devices is {spec.devices}, more than one {per_node}-GPU node: the "
            "tree's disaggregated launcher (`launch_disaggregated_llm`) starts every worker on "
            "one host, so the test runs on one node, and a job asking for more is refused",
            f"write the test for {per_node} devices or fewer (its context and generation "
            "workers between them) and set `devices` to match",
        )
        return
    devices = spec.device_count(member)
    if width is not None and devices and 2 * width > devices * spec.nodes:
        yield Finding(
            "disagg-placement",
            WARN,
            f"the disaggregated-serving test at {member} runs on {devices} devices of one node, "
            f"and one engine of the checkpoint takes {width} ({_gib(ctx.weight_bytes or 0)} of "
            f"weights at most {ENGINE_WEIGHT_FRACTION:.0%} of each {memory:g}-GiB device): its "
            "context and generation workers do not both fit",
            "an estimate from the weights alone: accept it in notes where both are known to "
            "fit; otherwise the tree's single-node launcher cannot serve this checkpoint "
            "disaggregated here, so accept `disagg@*` in coverage.accept and say why in notes",
        )


def _width(member: str) -> int:
    try:
        return parse_parallel_segment(member)[1] or 1
    except Exception:
        return 1


def contract_rules(ctx: LintContext) -> Iterator[Finding]:
    if ctx.contract is None:
        return
    required = ctx.contract.required()
    if len(required) > CONTRACT_REVIEW_SIZE:
        yield Finding(
            "contract-size",
            WARN,
            f"the contract owes {len(required)} checks: each loads the model, each accuracy "
            "check is a full benchmark run, and the certification runs all of them once, and "
            "again after each repair",
            "prove each requested feature once, at the member the request validates fully; "
            "members the request scopes to text-only take core alone; accept the rest "
            "explicitly (the report's per-unit lines show where the checks go)",
        )


def coverage_rules(ctx: LintContext) -> Iterator[Finding]:
    """Preflight's coverage check, run early: a requested cell no check covers stops the launch."""
    if ctx.contract is None:
        return
    try:
        from ..run import requirements as requirements_mod

        matrix = requirements_mod.requirements_matrix(ctx.task, ctx.contract, ctx.world)
    except Exception:  # the schema check owns malformed specs
        return
    if matrix.gaps():
        yield Finding("coverage-gap", BLOCK, requirements_mod.gap_message(matrix))


def _floors(accuracy: Mapping[str, Any], values: Iterable[float]) -> str:
    """The floor each anchor value puts under the gates, e.g. ``95 -> floor 93.80``."""
    samples = accuracy.get("samples") or accuracy_policy.KNOWN_BENCHMARK_SAMPLES.get(
        _benchmark(accuracy)
    )
    try:
        policy = accuracy_policy.resolve_floor_policy(accuracy)
        floors = {
            v: accuracy_policy.floor_for(policy, v, int(samples) if samples else None)
            for v in dict.fromkeys(values)
        }
    except (TypeError, ValueError):
        return "floors unknown"
    shown = [f"{v:g} -> floor {f.value:.2f}" for v, f in floors.items() if f is not None]
    return "; ".join(shown) if shown else "floors unknown"


def accuracy_rules(ctx: LintContext) -> Iterator[Finding]:
    accuracy = ctx.accuracy
    benchmark = _benchmark(accuracy)
    args = _benchmark_args(accuracy)
    if "--apply_chat_template" in args:
        output = next(
            (int(a.split("=", 1)[1]) for a in args if a.startswith("--max_output_length=")), 256
        )
        if output < CHAT_TEMPLATE_MIN_OUTPUT:
            yield Finding(
                "accuracy-chat-template-short-output",
                WARN,
                f"{benchmark} under the chat template with {output} output tokens: a model that "
                "reasons first is cut off before its answer, and the score measures truncation",
                "score few-shot completion without --apply_chat_template, or raise "
                f"--max_output_length to >= {CHAT_TEMPLATE_MIN_OUTPUT} with the model's stop "
                "strings; measure the anchor under the same protocol",
            )
    reference = accuracy.get("reference_yaml")
    if not reference:
        return
    path = Path(str(reference))
    if not path.is_absolute() and ctx.task.get("trtllm_repo_path"):
        path = Path(str(ctx.task["trtllm_repo_path"])) / path
    try:
        text = path.read_text(encoding="utf-8")
        rows = yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError):
        return
    key = accuracy.get("model_key")
    entries = rows.get(key) if isinstance(rows, Mapping) and key in rows else None
    if entries is None and isinstance(rows, Mapping) and len(rows) == 1:
        entries = next(iter(rows.values()))
    values = [
        float(e["accuracy"]) for e in entries or () if isinstance(e, Mapping) and "accuracy" in e
    ]
    plausible = PLAUSIBLE_ANCHOR.get(benchmark)
    low = [v for v in values if plausible is not None and v < plausible]
    marked = any(word in text.lower() for word in _PLACEHOLDER_WORDS)
    replace = (
        "write the measured anchor into the row, committed, before launch (the user's, or the "
        "reference implementation's on this hardware): the certification holds the benchmark "
        "to the row as written"
    )
    if low:
        yield Finding(
            "accuracy-placeholder-anchor",
            WARN,
            f"the {benchmark} anchor {', '.join(f'{v:g}' for v in low)} is under "
            f"{plausible:g}, where most served models score: if it is a placeholder, its floor "
            f"catches an engine that answers nothing and passes one that answers "
            f"({_floors(accuracy, values)}), so a run delivered on it certifies no accuracy; a "
            "configuration variant is judged against its unit's own score, not this row",
            f"{replace}; a measured score of a small model is right as it is: keep it with "
            "`intake-accept: accuracy-placeholder-anchor -- <how it was measured>` in notes",
        )
    elif marked:
        yield Finding(
            "accuracy-placeholder-anchor",
            WARN,
            f"the {benchmark} anchor is marked a placeholder but sits at a score a working "
            f"target may miss ({_floors(accuracy, values)}): a guess too high fails every "
            "accuracy check and sends the workers after a defect that is not there",
            f"set the placeholder well under the score you expect (e.g. 10); {replace}",
        )


def boot_rules(ctx: LintContext) -> Iterator[Finding]:
    envelope = ctx.accuracy.get("envelope") or {}
    on = envelope.get("enabled") or (envelope == {} and profile_of(ctx.task) == "certified")
    if on and not envelope.get("probes"):
        yield Finding(
            "envelope-open-ended-probes",
            WARN,
            "boot boundaries compare each probe's first tokens alone and under load with the "
            "default open-ended prompts; their near-tied tokens flip with batch shape (MoE "
            "routing, low-precision GEMM tactics) and fail every boot without a defect",
            "set accuracy.envelope.probes to continuations with one right answer, e.g. "
            '["1, 2, 3, 4, 5, 6, 7,", "Monday, Tuesday, Wednesday,"]',
        )
    if envelope.get("enabled") and profile_of(ctx.task) != "certified":
        from ..checks.boot import BootBoundarySpec

        spec = BootBoundarySpec.from_task(ctx.task)
        boots = ctx.contract.required() if ctx.contract is not None else ()
        at = [
            f"{c.unit}@{c.member}"
            for c in boots
            if c.gate == "boot" and spec.runs_at(ctx.task, c.unit, c.member, ctx.world)
        ]
        yield Finding(
            "envelope-on",
            INFO,
            f"accuracy.envelope is on: the boot at {', '.join(at) or 'one member of each unit'} "
            "also runs the boundary probes (alone, then under load, at the longest prompt the "
            "engine takes); a probe whose legs part past a near-tie fails that boot. Leave it off "
            "unless the request asks for boot boundaries; the certified profile turns it on",
        )


def disagg_rules(ctx: LintContext) -> Iterator[Finding]:
    """The disaggregated-serving test the run writes is named, and is there or planned."""
    spec = disagg_checks.DisaggSpec.from_task(ctx.task)
    if spec is None or "disagg" not in assembly_units(ctx.task) or not ctx.repo:
        return
    test_file = Path(ctx.repo) / spec.test.partition("::")[0]
    if not test_file.is_file():
        yield Finding(
            "disagg-test-pending",
            INFO,
            f"disagg's test {spec.test} is not in the tree yet: the run writes it beside the "
            "tree's tests that serve through `trtllm-serve disaggregated`; say in notes which task "
            "does",
        )


def environment_rules(ctx: LintContext) -> Iterator[Finding]:
    env = ctx.environment
    python = str(env.get("python") or "")
    host = _host_path(env, python) if python.startswith("/") else None
    if host is not None and not host.exists():
        yield Finding(
            "environment-python-missing",
            WARN,
            f"environment.python {python} does not exist yet: every gate's command runs under "
            "it, so preflight fails at its first step",
            "create it before launch (the editable install of the checkout the run measures), "
            "or name an interpreter the image already has",
        )


def agents_rules(ctx: LintContext) -> Iterator[Finding]:
    """Every role's model is stated: the run inherits none and stops before its first worker."""
    from ..agents.roles import AgentConfigError, resolve_role_agents

    try:
        resolve_role_agents(ctx.task)
    except AgentConfigError as exc:
        yield Finding(
            "agents-model-missing",
            BLOCK,
            str(exc),
            "write your own backend, model and reasoning_effort into agents.defaults "
            "(spec/INTAKE.md, section 7)",
        )


def notes_rules(ctx: LintContext) -> Iterator[Finding]:
    notes = ctx.notes
    lowered = notes.lower()
    missing = [w for w in _BRIEF_WORDS if w not in lowered]
    if missing:
        yield Finding(
            "notes-missing-brief",
            WARN,
            f"notes have no {' or '.join(repr(w) for w in missing)} section: every role reads "
            "them, and rules alone say nothing about the model or what the tree already has",
            "add an architecture brief (attention kinds and head dims, sparse/indexer, MoE, "
            "norms, towers) and 'Prior art in this tree worth surveying: ...' from the lint "
            "report's prior-art leads you checked",
        )
    if len(notes) > NOTES_BRIEF_CHARS:
        yield Finding(
            "notes-long",
            WARN,
            f"notes are {len(notes)} characters: every role reads them in every turn, and a "
            "brief that long buries the few facts that change what a role does",
            f"keep notes under ~{NOTES_BRIEF_CHARS} characters (reference, architecture, prior "
            "art, engine constraints, scope); keep evidence, provenance and launch "
            "caveats in intake files a role can open when it needs them",
        )
    # `intake-accept:` paragraphs explain the intake's choices; they are not orders.
    orders = "\n".join(p for p in re.split(r"\n\s*\n", notes) if "intake-accept:" not in p.lower())
    process = _INTAKE_PROCESS.search(orders)
    if process:
        yield Finding(
            "notes-intake-process",
            WARN,
            f'notes speak about the intake itself ("{process.group(0)}"): at launch every role '
            "reads them as standing orders, and a line such as 'no build is allowed' stops the "
            "run's own work",
            "keep notes to what the roles need while they work; put what the intake may or may "
            "not do, launch blockers and questions for the user in the intake summary",
        )


def engine_rules(ctx: LintContext) -> Iterator[Finding]:
    for constraint in ctx.constraints:
        yield Finding(
            "engine-constraint",
            WARN,
            constraint,
            "decide the design in notes before the survey (the way out it names, or a planned "
            "native kernel)",
        )


def tree_rules(ctx: LintContext) -> Iterator[Finding]:
    """What the lint reads off the tree and this checkout has nowhere: the check it then skips."""
    if not ctx.repo:
        return
    for gone in tree_anchors.missing(ctx.repo, prior_art.PRIOR_ART_ANCHORS):
        anchor = gone.anchor
        roots = ", ".join(f"`{r}/`" for r in anchor.roots)
        yield Finding(
            "tree-anchor-missing",
            INFO,
            f"{anchor.concept}: not at `{anchor.hint}`, and no {anchor.search} under {roots}, "
            f"so {anchor.consequence}",
            "if the engine renamed what identifies it, update its anchor "
            "(`operators/tree_anchors.py`)",
        )


#: Every rule, in report order. A new feature, scope or launch check adds its function here.
RULES: tuple[Rule, ...] = (
    cuda_graph_rules,
    kv_cache_rules,
    placement_rules,
    contract_rules,
    coverage_rules,
    accuracy_rules,
    boot_rules,
    disagg_rules,
    environment_rules,
    agents_rules,
    notes_rules,
    engine_rules,
    tree_rules,
)


def _compile(task: Mapping[str, Any], world: int) -> tuple[Any, str]:
    try:
        from ..run.contract import compile_contract

        return compile_contract(task, world), ""
    except Exception as exc:  # the schema check owns malformed specs
        return None, f"{type(exc).__name__}: {exc}"


def _contract_lines(contract: Any, error: str) -> list[str]:
    if contract is None:
        return [f"Not compiled: {error}"] if error else []
    by_gate: dict[str, int] = {}
    # Unit or variant -> gate -> its members, in the contract's order.
    by_unit: dict[str, dict[str, list[str]]] = {}
    required = contract.required()
    for check in required:
        gate = str(check.gate).split("[")[0]
        by_gate[gate] = by_gate.get(gate, 0) + 1
        by_unit.setdefault(check.unit, {}).setdefault(gate, []).append(check.member)
    lines = [
        f"{len(required)} required checks: "
        + ", ".join(f"{gate} {n}" for gate, n in sorted(by_gate.items()))
    ]
    if by_unit:
        lines.append("")
        lines += [
            f"- `{unit}`: "
            + "; ".join(f"{gate} {len(at)} ({', '.join(at)})" for gate, at in gates.items())
            for unit, gates in by_unit.items()
        ]
    return lines


def lint_task(
    task: Mapping[str, Any],
    *,
    checkpoint: str | Path | None = None,
    repo: str | Path | None = None,
    contract: Any = None,
    world_size: int | None = None,
    base: Path | None = None,
    rules: Iterable[Rule] = RULES,
) -> LintReport:
    """Lint ``task`` (a loaded spec); ``checkpoint`` and ``repo`` default to the spec's own.

    ``base`` is the spec's directory, for the relative paths of a spec the schema did not resolve.
    """
    world = world_size or widest_world_size(task, 4)
    error = ""
    if contract is None:
        contract, error = _compile(task, world)
    checkpoint = checkpoint or task.get("checkpoint_path")
    repo = repo or task.get("trtllm_repo_path")
    config = None
    if checkpoint:
        try:
            config = json.loads((Path(str(checkpoint)) / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = None
    ctx = LintContext(
        task=task,
        world=world,
        contract=contract,
        contract_error=error,
        config=config,
        repo=repo,
        base=base,
        weight_bytes=checkpoint_weight_bytes(checkpoint),
    )
    if config is not None:
        ctx.constraints = prior_art.engine_constraints(config, repo)
        if repo:
            ctx.prior_art = prior_art.find_prior_art(repo, config)
    report = LintReport(
        contract_lines=_contract_lines(contract, error),
        prior_art=ctx.prior_art,
        constraints=ctx.constraints,
    )
    for match in _ACCEPT.finditer(ctx.notes):
        report.accepted.update(r.strip().lower() for r in match.group(1).split(","))
    for rule in rules:
        report.findings.extend(rule(ctx))
    return report


def main(argv: list[str] | None = None) -> int:
    """``trtllm-modeling-v2 intake lint``: print the report; exit 1 while a finding stands."""
    import argparse

    from . import task_schema
    from .task_schema import TaskSchemaError, load_and_validate_task_yaml

    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 intake lint",
        description="Lint a spec before launch: cost, scope, accuracy, boot probes, environment, "
        "notes, engine constraints and prior art. CPU only; reads the checkpoint config and the tree.",
    )
    parser.add_argument("--task", type=Path, required=True, help="The spec to lint.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Default: checkpoint_path.")
    parser.add_argument("--repo", type=Path, default=None, help="Default: trtllm_repo_path.")
    parser.add_argument("--report", type=Path, default=None, help="Also write the report here.")
    args = parser.parse_args(argv)
    try:
        task = load_and_validate_task_yaml(args.task)
        schema_error = ""
    except TaskSchemaError as exc:
        task = yaml.safe_load(args.task.read_text(encoding="utf-8")) or {}
        schema_error = str(exc)
        # The rest of the lint reads the spec as a launch would: its paths and derived gates.
        try:
            task_schema._resolve_spec_relative_paths(task, args.task.resolve().parent)
            task_schema.derive_gate_defaults(task, args.task.resolve().parent)
        except Exception:  # a spec that cannot be resolved is already a schema block
            pass
    report = lint_task(
        task, checkpoint=args.checkpoint, repo=args.repo, base=args.task.resolve().parent
    )
    if schema_error:
        report.findings.insert(0, Finding("schema", BLOCK, schema_error))
    text = report.render()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
    return 1 if report.open() else 0


__all__ = [
    "BLOCK",
    "INFO",
    "RULES",
    "WARN",
    "Finding",
    "LintContext",
    "LintReport",
    "Rule",
    "lint_task",
    "main",
]
