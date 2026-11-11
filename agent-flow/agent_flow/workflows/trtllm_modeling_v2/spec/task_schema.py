# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema validation for trtllm-modeling-v2's ``task.yaml`` input.

A modeling_v2 target's identity is the ``(checkpoint, GPU arch, parallel
config)`` triple, and the run's slug is derived from it. ``arch`` is
**required and never inferred**: catalog receipts are per-architecture.
:func:`load_and_validate_task_yaml` batches every detected problem into one
message and fills in the workflow's defaults, so the resolved spec copied
into the workspace is fully explicit.
"""

from __future__ import annotations

import ast
import difflib
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from agent_flow.agent_runtime import AGENTS_FIELD, validate_agents

from ..agents.roles import ROLES
from ..checks import accuracy_policy
from ..infra.compute.environment import EnvironmentError, check_tmpdir
from ..infra.git import COMMIT_POLICY_FIELD, validate_commit_policy_block
from ..operators.coverage import MODELING_V2_PACKAGE
from .variants import ANCHOR_RECORDED, ANCHOR_RELATIVE, VARIANTS_FIELD, validate_variants

REQUIRED_PATH_FIELDS: tuple[str, ...] = (
    "checkpoint_path",
    "trtllm_repo_path",
)

ARCH_FIELD = "arch"
PARALLEL_FIELD = "parallel"
FEATURES_FIELD = "features"
ENVIRONMENT_FIELD = "environment"
ACCURACY_FIELD = "accuracy"

# ``sm_100`` (B200), ``sm_103`` (GB300), ... Matched rather than
# enumerated: preflight asserts the value against the host.
ARCH_PATTERN = re.compile(r"^sm_\d+$")

# ``tp4`` / ``tep4`` / ``dep4`` carry their world size in the trailing
# number; ``unified`` means one modeling codebase over a family of
# parallel configurations and takes its world size from the member set.
PARALLEL_PATTERN = re.compile(r"^(tp|tep|dep)(\d+)$")
PARALLEL_UNIFIED = "unified"
DEFAULT_PARALLEL = PARALLEL_UNIFIED

# Which concrete configurations a ``unified`` segment is actually gated over.
MEMBERS_FIELD = "members"

# TEP and DEP at the world size are the baseline pair a unified codebase is
# gated on; TP is the degenerate path, and the only member that folds every
# branch off.
DEFAULT_MEMBER_KINDS: tuple[str, ...] = ("tp", "tep", "dep")

# Feature name -> its family. The set is closed because each family's checks
# are wired by name.
FEATURE_FAMILIES: dict[str, str] = {
    "mtp": "speculator",
    "eagle3": "speculator",
    "dflash": "speculator",
    # A block drafter: one 3-stage pass proposes a 5-token block, where MTP
    # chains one-token heads.
    "dspark": "speculator",
    # "SWA Bounded Replay", decoder and encoder halves.
    "decoder_replay": "bounded_replay",
    "encoder_replay": "bounded_replay",
    # KV pools switch from bf16 containers holding the reference's
    # fake-quantized values (exact, and what core ships) to the checkpoint's
    # native storage formats.
    "kv_quant": "kv_quant",
    "multimodal": "multimodal",
    # A weight tensor too large to keep resident lives in pinned host memory
    # and the forward gathers what it needs.
    "host_offload": "host_offload",
    # Prefill and decode on separate engines.
    "disagg": "disagg",
}
#: The features that draft tokens a target model verifies.
SPECULATORS: frozenset[str] = frozenset(
    feature for feature, family in FEATURE_FAMILIES.items() if family == "speculator"
)

# Features that read the checkpoint's `mtp.*` namespace, which holds MTP
# modules OR DSpark stages and never both.
MTP_NAMESPACE_FEATURES: tuple[str, ...] = ("mtp", "dspark")

# `accuracy.disagg`: the tree's own disaggregated-serving accuracy test for this target
# (checks/disagg.py), the one check the `disagg` feature owes. The run writes it beside the
# tree's tests that serve through `trtllm-serve disaggregated`.
#
#   test     REQUIRED with the disagg feature: its repository-relative pytest node id, e.g.
#            `tests/integration/defs/accuracy/test_disaggregated_serving.py::TestX::test_y`
#   tests    optional refinements of test, each run as a separate native job
#   devices  the devices it starts its engines on (default two engines at the member's width)
#   timeout  seconds (default 7200)
#   env      the environment the test runs under (`LLM_MODELS_ROOT`, ...)
DISAGG_FEATURE = "disagg"
DISAGG_KEYS: tuple[str, ...] = ("test", "tests", "devices", "nodes", "timeout", "env")

TARGET_FIELD = "target"
RUNTIME_FIELD = "runtime"
SCOPE_FIELD = "scope"
NOTES_FIELD = "notes"

# Runtime switches the gates must exercise, and which therefore have to be
# declared rather than left to whatever the engine defaults to.
#
# The values are what the *user* wants true of the run, not what the flag
# says: two of these are spelled as negations on the command line
# (``--disable_overlap_scheduler``, ``--disable_kv_cache_reuse``).
RUNTIME_SWITCHES: dict[str, bool] = {
    "cuda_graph": True,
    "chunked_prefill": False,
    "overlap_scheduler": True,
    "kv_cache_reuse": True,
}

# Scopes a target may be built at. ``text_only`` is the useful one: a
# multimodal checkpoint whose vision tower is out of scope still has to
# load, route and gate as a language model.
SCOPES: tuple[str, ...] = ("full", "text_only")
DEFAULT_SCOPE = "full"

TOP_LEVEL_KEYS: tuple[str, ...] = (
    *REQUIRED_PATH_FIELDS,
    ARCH_FIELD,
    PARALLEL_FIELD,
    MEMBERS_FIELD,
    FEATURES_FIELD,
    ENVIRONMENT_FIELD,
    ACCURACY_FIELD,
    TARGET_FIELD,
    RUNTIME_FIELD,
    SCOPE_FIELD,
    NOTES_FIELD,
    AGENTS_FIELD,
    COMMIT_POLICY_FIELD,
    VARIANTS_FIELD,
    "coverage",
    "profile",
    # What the run brings up, stated as native options (spec/deployments.py). The keys
    # above it that say the same (`members`, `features`, `variants`, `runtime`, ...) are
    # shorthand, expanded into deployments when the spec is read.
    "deployments",
)
#: ``coverage.accept``: the requested feature@member cells the contract may leave uncovered.
COVERAGE_FIELD = "coverage"
COVERAGE_KEYS: tuple[str, ...] = ("accept",)
#: ``profile``: how much the contract owes, for the run or per unit (a mapping
#: from unit or variant name, and ``default``, to a profile). ``smoke`` owes a
#: unit's boot only; ``bringup`` gates each unit at its cheapest member, and no
#: configuration variant and no delivery check. A run whose every unit is one of
#: the two owes no CI files either (``completion.CI_DELIVERY_CHECKS``);
#: ``standard`` is the contract as the spec states it; ``certified`` also turns on
#: the serve gate. A switch the spec sets itself keeps its value.
PROFILE_FIELD = "profile"
PROFILE_SMOKE = "smoke"
PROFILE_BRINGUP = "bringup"
PROFILE_STANDARD = "standard"
PROFILE_CERTIFIED = "certified"
PROFILES: tuple[str, ...] = (PROFILE_SMOKE, PROFILE_BRINGUP, PROFILE_STANDARD, PROFILE_CERTIFIED)


def profile_of(task_data: Mapping[str, Any], unit: str | None = None) -> str:
    """``unit``'s contract profile (its own entry, else ``default``), or the run's; ``standard`` unsaid."""
    profile = task_data.get(PROFILE_FIELD)
    if isinstance(profile, Mapping):
        own = profile.get(unit) if unit is not None else None
        return str(own or profile.get("default") or PROFILE_STANDARD)
    return str(profile or PROFILE_STANDARD)


TARGET_KEYS: tuple[str, ...] = ("family", "checkpoint_segment")
#: The ``target`` keys that name a Python package directory.
TARGET_NAME_KEYS: tuple[str, ...] = ("family", "checkpoint_segment")

#: ``accuracy`` keys an earlier version of the spec read and this one does not, each with what
#: went: dropped at load and named once in a notice, so a spec written for it still loads.
IGNORED_ACCURACY_KEYS: dict[str, str] = {
    "envelope": "a boot runs its workload only; there are no boundary phases",
    "reference_logits": "a boot keeps no divergence record against a reference's logits",
    "acceptance": "a drafter's acceptance is judged by a native test, named in a deployment's "
    "`tests`; the flow measures no acceptance length of its own",
}

ACCURACY_KEYS: tuple[str, ...] = (
    "benchmark",
    "reference_yaml",
    "model_key",
    "tolerance",
    "timeout",
    "dataset_path",
    "llm_api_options",
    # Which ROW of reference_yaml this run anchors to. The references key a
    # model to a list discriminated by quant_algo / kv_cache_quant_algo /
    # spec_dec_algo, and the match is exact.
    "discriminators",
    # The disaggregated-serving test the `disagg` feature owes (checks/disagg.py).
    "disagg",
    # `trtllm-eval` takes these two as its OWN flags rather than through the
    # options YAML, so they have to be declarable here.
    "max_seq_len",
    "max_batch_size",
    "benchmark_args",
    "floor",
    "boot",
    "serve",
    # The benchmark's sample count, scalar or per unit, for the standard
    # error behind the statistical floor.
    "samples",
    "benchmark_members",
    "anchor",
    # How the gate reads the score from the benchmark's output, scalar or per unit: a
    # regular expression whose first group is the score, on the anchor's scale. Unset, the
    # gate reads lm-eval's `accuracy:` line and MMLU's weighted average.
    "score_pattern",
)

# `accuracy.anchor`: what a unit's or a variant's accuracy floor stands on.
#
#   recorded  a row of reference_yaml: a written source, a measurement, or a deliberately
#             low placeholder until one of those is known.
#   relative  a configuration variant only: its unit's own score at the same member, on
#             the same state of the target (a KV dtype against `auto`, say). The default
#             for a variant that names no reference row of its own.
ANCHOR_STATES: tuple[str, ...] = (ANCHOR_RECORDED, ANCHOR_RELATIVE)

# `accuracy.benchmark_members`: where each unit's accuracy benchmark runs.
#
#   every    (default) at every member.
#   <member> at that member (`dep4`, ...), for each unit that has it.
BENCHMARK_MEMBERS_EVERY = "every"
BENCHMARK_MEMBERS: tuple[str, ...] = (BENCHMARK_MEMBERS_EVERY,)

# Boot workloads allow run-wide keys and per-unit overlays. Keywords match
# generated text, never logs; absent multimodal images use the repository asset.
BOOT_KEYS: tuple[str, ...] = ("prompts", "prompts_file", "keywords", "max_tokens")
# How long one boot gate may run: `timeout` at either level of
# `accuracy.boot` (default 1800 s). Not a workload key: it
# changes no verdict, only how long a slow load is waited for.
GATE_TIMEOUT_KEY = "timeout"
BOOT_PROMPT_KEYS: tuple[str, ...] = ("prompt", "keywords", "keywords_any", "images", "render")
BOOT_RENDERS: tuple[str, ...] = ("loader", "raw")

# Serve defaults are derived from the declared sequence bound in derive_gate_defaults.
SERVE_KEYS: tuple[str, ...] = ("enabled", "prompt", "max_tokens", "timeout")

# Floor computation and systematic-shift findings are owned by checks/stats.py.
FLOOR_KEYS: tuple[str, ...] = ("rule", "k", "shift_min_members", "acknowledge")

ENVIRONMENT_KEYS: tuple[str, ...] = (
    "kind",
    "python",
    # The site's prefix for a one-GPU command (a kernel entry's test); a deployment
    # carries its own. Empty: the orchestrator's host.
    "launch",
    "env",
    # The image the GPU jobs run in, if any, and how host paths appear inside it
    # (`host:inside`).
    "container_image",
    "mounts",
    "gpus_per_node",
    # Each device's memory, GiB: the spec's warnings and derive size engines by it.
    "gpu_memory_gib",
)

#: Where the run's commands land: on the orchestrator's own host. Running them anywhere else
#: (a queue of compute sessions, a batch scheduler) is not this workflow's: start the
#: orchestrator where the GPUs are.
ENVIRONMENT_KINDS: tuple[str, ...] = ("local",)
#: Keys of the compute-session executor and the substrate gate the workflow no longer ships:
#: refused with why, so a spec written for them is not half-read.
REMOVED_ENVIRONMENT_KEYS: tuple[str, ...] = (
    "spool",
    "sessions",
    "session_wait",
    "substrate_checkpoint",
)
ENVIRONMENT_REQUIRED_FIELDS: tuple[str, ...] = ("python",)

# Accuracy tolerance, in benchmark points, under `accuracy.floor.rule:
# tolerance` only.
DEFAULT_ACCURACY_TOLERANCE = 5.0


class TaskSchemaError(ValueError):
    """Raised when ``task.yaml`` fails trtllm-modeling-v2 schema validation."""


def reject_unknown_keys(
    block: Mapping[str, Any],
    known: Sequence[str],
    errors: list[str],
    *,
    where: str = "",
) -> None:
    """Append an error for any key in ``block`` that ``known`` does not list.

    Every block in this schema is closed: a misspelled key that is silently
    ignored produces a run that gates a configuration nobody asked for. The
    nearest known key is suggested.
    """
    prefix = f"{where}." if where else ""
    for key in sorted(set(block) - set(known)):
        close = difflib.get_close_matches(str(key), [str(k) for k in known], n=1, cutoff=0.6)
        hint = f" — did you mean '{prefix}{close[0]}'?" if close else ""
        errors.append(
            f"unknown key '{prefix}{key}'{hint} Known keys here: {sorted(str(k) for k in known)}"
        )


def parse_parallel_segment(segment: str) -> tuple[str, int | None]:
    """Return ``(kind, world_size)`` for a parallel segment.

    ``world_size`` is ``None`` for ``unified``, whose member set — not the
    segment string — decides how many devices the run claims.
    """
    if segment == PARALLEL_UNIFIED:
        return PARALLEL_UNIFIED, None
    match = PARALLEL_PATTERN.match(segment)
    if match is None:
        raise TaskSchemaError(
            f"'{PARALLEL_FIELD}' must be 'tp<N>', 'tep<N>', 'dep<N>' or "
            f"'{PARALLEL_UNIFIED}'; got {segment!r}"
        )
    world_size = int(match.group(2))
    if world_size < 1:
        raise TaskSchemaError(f"'{PARALLEL_FIELD}' world size must be >= 1; got {segment!r}")
    return match.group(1), world_size


def members_for(data: Mapping[str, Any], unit: str | None, world_size: int) -> list[str]:
    """The concrete parallel configurations ``unit`` is gated over.

    A concrete segment is the **degenerate** member set — itself, once.
    ``members`` takes the same two shapes ``runtime`` and
    ``accuracy.llm_api_options`` take: a list applies to every unit, and a
    mapping from assembly unit to a list is layered over ``default``. This
    resolves an already-validated spec and falls back to the default set.
    """
    segment = str(data.get(PARALLEL_FIELD) or DEFAULT_PARALLEL)
    if segment != PARALLEL_UNIFIED:
        return [segment]
    block = data.get(MEMBERS_FIELD)
    if isinstance(block, dict):
        chosen = block.get(unit)
        block = block.get("default") if chosen is None else chosen
    if isinstance(block, list) and block:
        return [str(member) for member in block]
    return [f"{kind}{world_size}" for kind in DEFAULT_MEMBER_KINDS]


def widest_world_size(data: Mapping[str, Any], default: int) -> int:
    """The largest world size any member of a validated spec declares.

    A concrete segment answers for itself. A ``unified`` target takes the
    widest of its members, per-unit lists and configuration variants
    included; with no members declared it is ``default``.
    """
    _, size = parse_parallel_segment(str(data.get(PARALLEL_FIELD) or DEFAULT_PARALLEL))
    if size is not None:
        return size
    block = data.get(MEMBERS_FIELD)
    declared: list[Any] = []
    if isinstance(block, dict):
        for entries in block.values():
            declared.extend(entries or ())
    elif isinstance(block, list):
        declared.extend(block)
    for entry in (data.get(VARIANTS_FIELD) or {}).values():
        declared.extend((entry or {}).get("members") or ())
    sizes = []
    for member in declared:
        try:
            sizes.append(parse_parallel_segment(str(member))[1] or 0)
        except TaskSchemaError:
            continue
    return max(sizes, default=0) or default


def derive_slug(data: Mapping[str, Any]) -> str:
    """Return ``<checkpoint>-<arch>-<segment>`` for a validated spec.

    The checkpoint segment is the checkpoint directory's own name,
    lowercased with non-alphanumerics folded to hyphens.
    """
    checkpoint = Path(str(data["checkpoint_path"]).rstrip("/")).name
    checkpoint = re.sub(r"[^a-zA-Z0-9]+", "-", checkpoint).strip("-").lower()
    return f"{checkpoint}-{data[ARCH_FIELD]}-{data[PARALLEL_FIELD]}"


def routed_family(repo: str | Path, architecture: str) -> str | None:
    """The family whose ``routing.py`` declares ``@routes(architecture)``; ``None`` when none does.

    The tree's own registry of which family serves an ``architectures[0]``, read from the
    source without importing it: the orchestrator's interpreter need not have TensorRT-LLM.
    """
    models = Path(repo) / MODELING_V2_PACKAGE / "models"
    for routing in sorted(models.glob("*/routing.py")):
        try:
            tree = ast.parse(routing.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                called = decorator.func
                if getattr(called, "id", getattr(called, "attr", "")) != "routes":
                    continue
                if any(
                    isinstance(a, ast.Constant) and a.value == architecture for a in decorator.args
                ):
                    return routing.parent.name
    return None


def derive_family(architecture: str) -> str:
    """``GptOssForCausalLM`` -> ``gpt_oss``: the family directory of an architecture nothing routes yet.

    The class name without its head suffix (``ForCausalLM``, ``ForConditionalGeneration``, ...),
    in snake case. A family the tree already routes is read from it (:func:`routed_family`).
    """
    name = re.sub(r"For[A-Z][A-Za-z0-9]*$", "", architecture)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def derive_checkpoint_segment(checkpoint_path: str | Path) -> str:
    """``DeepSeek-R1-0528-NVFP4`` -> ``deep_seek_r1_0528_nvfp4``.

    **A guess, and the one identity component worth overriding**: set
    ``target.checkpoint_segment`` in task.yaml when the default reads badly.
    """
    name = Path(str(checkpoint_path).rstrip("/")).name
    # Split camelCase only at a genuine word boundary — a lowercase letter
    # followed by a capital. Allowing a digit on the left would break the
    # size suffixes these names are full of: `Qwen3-30B-A3B` would come out
    # as `qwen3_30_b_a3_b`.
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name)
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _validate_coverage(block: Any, errors: list[str]) -> None:
    """``coverage``: a mapping whose ``accept`` lists ``<feature>@<member>`` or ``<feature>@*``."""
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append(f"'{COVERAGE_FIELD}' must be a mapping, got {type(block).__name__}")
        return
    reject_unknown_keys(block, COVERAGE_KEYS, errors, where=COVERAGE_FIELD)
    raw = block.get("accept")
    if raw is not None and (
        not isinstance(raw, list) or not all(isinstance(x, str) and "@" in x for x in raw)
    ):
        errors.append(
            f"'{COVERAGE_FIELD}.accept' must be a list of '<feature>@<member>' or '<feature>@*'"
        )


def _validate_target(block: Any, errors: list[str]) -> None:
    """Validate the optional ``target`` block, which overrides the derivation."""
    if not isinstance(block, dict):
        raise TaskSchemaError(f"'{TARGET_FIELD}' must be a mapping, got {type(block).__name__}")
    reject_unknown_keys(block, TARGET_KEYS, errors, where=TARGET_FIELD)
    for name in TARGET_NAME_KEYS:
        value = block.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{TARGET_FIELD}.{name}' must be a non-empty string")
        elif not re.match(r"^[a-z0-9_]+$", value):
            errors.append(
                f"'{TARGET_FIELD}.{name}' is a Python package directory name, so it must "
                f"be lowercase with underscores; got {value!r}"
            )


def _validate_runtime_switches(block: dict, errors: list[str], where: str, *, fill: bool) -> None:
    """Validate one runtime block. ``fill`` writes the defaults in."""
    unknown = sorted(set(block) - set(RUNTIME_SWITCHES) - {"extra_flags"})
    if unknown:
        errors.append(
            f"'{where}' has unknown switch(es) {unknown}; known switches are "
            f"{sorted(RUNTIME_SWITCHES)} plus 'extra_flags'"
        )

    for name, default in RUNTIME_SWITCHES.items():
        value = block.get(name)
        if value is None:
            if fill:
                block[name] = default
        elif not isinstance(value, bool):
            errors.append(f"'{where}.{name}' must be a boolean, got {type(value).__name__}")

    extra = block.get("extra_flags")
    if extra is None:
        if fill:
            block["extra_flags"] = []
    elif not isinstance(extra, list) or not all(isinstance(f, str) for f in extra):
        errors.append(f"'{where}.extra_flags' must be a list of strings")


def _validate_runtime(block: Any, errors: list[str]) -> None:
    """Validate the optional ``runtime`` block in place, filling defaults."""
    if not isinstance(block, dict):
        raise TaskSchemaError(f"'{RUNTIME_FIELD}' must be a mapping, got {type(block).__name__}")

    # Per-unit form: a mapping whose values are themselves mappings. A per-unit
    # block is layered over `default`, so a unit states only what differs.
    if any(isinstance(v, dict) for v in block.values()):
        nested = {k: v for k, v in block.items() if isinstance(v, dict)}
        if len(nested) != len(block):
            errors.append(
                f"'{RUNTIME_FIELD}' mixes the flat form (cuda_graph: true) with the "
                f"per-unit form (core: {{...}}); pick one"
            )
        if "default" not in block:
            errors.append(
                f"'{RUNTIME_FIELD}' is a per-unit mapping, so it needs a 'default' key "
                f"-- a unit nobody listed would otherwise be gated under the engine's "
                f"defaults rather than this run's"
            )
        for unit, sub in nested.items():
            # Only `default` is filled in: a per-unit block is an OVERLAY, so
            # filling its absent switches with the schema defaults would
            # silently override `default` with them.
            _validate_runtime_switches(
                sub, errors, f"{RUNTIME_FIELD}.{unit}", fill=(unit == "default")
            )
        return

    _validate_runtime_switches(block, errors, RUNTIME_FIELD, fill=True)


def _validate_members(
    block: Any, parallel: Any, errors: list[str], *, environment_kind: str = "local"
) -> None:
    """Validate the optional ``members`` block: what a unified run is gated over."""
    if parallel != PARALLEL_UNIFIED:
        errors.append(
            f"'{MEMBERS_FIELD}' is only legal with '{PARALLEL_FIELD}: {PARALLEL_UNIFIED}'; "
            f"this spec says '{PARALLEL_FIELD}: {parallel}', which already names the one "
            f"configuration the run is gated at"
        )
        return

    if isinstance(block, dict):
        if "default" not in block:
            errors.append(
                f"'{MEMBERS_FIELD}' is a per-unit mapping, so it needs a 'default' key "
                f"-- a unit nobody listed would otherwise fall back to the built-in "
                f"member set and be gated over configurations this spec did not choose"
            )
        lists = {f"{MEMBERS_FIELD}.{unit}": value for unit, value in block.items()}
    elif isinstance(block, list):
        lists = {MEMBERS_FIELD: block}
    else:
        errors.append(
            f"'{MEMBERS_FIELD}' must be a list of parallel segments, or a mapping from "
            f"assembly unit to one; got {type(block).__name__}"
        )
        return

    world_sizes: dict[str, int] = {}
    for where, entries in lists.items():
        if not isinstance(entries, list) or not entries:
            errors.append(
                f"'{where}' must be a non-empty list of parallel segments; an empty one "
                f"means a unit that is built and never gated"
            )
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, str):
                errors.append(f"'{where}[{index}]' must be a string, got {type(entry).__name__}")
                continue
            try:
                kind, size = parse_parallel_segment(entry)
            except TaskSchemaError as exc:
                errors.append(f"'{where}[{index}]' is not a parallel segment: {exc}")
                continue
            if kind == PARALLEL_UNIFIED or size is None:
                errors.append(
                    f"'{where}[{index}]' is '{PARALLEL_UNIFIED}', which is the family "
                    f"rather than a configuration to serve at. A member is concrete: "
                    f"'tp4', 'tep4', 'dep4'."
                )
                continue
            world_sizes[entry] = size


def _validate_environment(block: Any, errors: list[str]) -> None:
    """Validate the ``environment`` block in place, appending to ``errors``."""
    if not isinstance(block, dict):
        raise TaskSchemaError(
            f"'{ENVIRONMENT_FIELD}' must be a mapping, got {type(block).__name__}"
        )

    removed = [key for key in REMOVED_ENVIRONMENT_KEYS if key in block]
    if removed or block.get("kind") == "session_spool":
        named = (
            ", ".join(f"'{ENVIRONMENT_FIELD}.{key}'" for key in removed) or "kind: session_spool"
        )
        errors.append(
            f"{named}: the compute-session executor and the substrate gate are no longer part "
            f"of this workflow. Use `kind: local`, start the orchestrator on the host that "
            f"holds the GPUs (inside the allocation, on a cluster), and drop these keys."
        )
    reject_unknown_keys(
        {k: v for k, v in block.items() if k not in REMOVED_ENVIRONMENT_KEYS},
        ENVIRONMENT_KEYS,
        errors,
        where=ENVIRONMENT_FIELD,
    )

    for name in ENVIRONMENT_REQUIRED_FIELDS:
        value = block.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{ENVIRONMENT_FIELD}.{name}' must be a non-empty string")

    kind = block.setdefault("kind", ENVIRONMENT_KINDS[0])
    if kind not in (*ENVIRONMENT_KINDS, "session_spool"):
        errors.append(
            f"'{ENVIRONMENT_FIELD}.kind' must be one of {ENVIRONMENT_KINDS}; got {kind!r}"
        )
    launch = block.get("launch")
    if launch is not None and not isinstance(launch, str):
        errors.append(f"'{ENVIRONMENT_FIELD}.launch' must be a string: the site's launch prefix")

    env = block.get("env")
    if env is None:
        block["env"] = {}
    elif not isinstance(env, dict):
        errors.append(f"'{ENVIRONMENT_FIELD}.env' must be a mapping, got {type(env).__name__}")
    else:
        for key, value in env.items():
            if not isinstance(value, (str, int, float)):
                errors.append(
                    f"'{ENVIRONMENT_FIELD}.env.{key}' must be a scalar the shell can "
                    f"carry, got {type(value).__name__}"
                )
        try:
            check_tmpdir({str(k): str(v) for k, v in env.items()}, require_set=False)
        except EnvironmentError as exc:
            errors.append(f"'{ENVIRONMENT_FIELD}.env': {exc}")

    mounts = block.get("mounts")
    if mounts is not None and (
        not isinstance(mounts, list) or not all(isinstance(m, str) for m in mounts)
    ):
        errors.append(f"'{ENVIRONMENT_FIELD}.mounts' must be a list of strings")

    memory = block.get("gpu_memory_gib")
    if memory is not None and not (_number(memory) and memory > 0):
        errors.append(f"'{ENVIRONMENT_FIELD}.gpu_memory_gib' must be a positive number of GiB")


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _nonempty_strings(value: Any) -> bool:
    return isinstance(value, list) and all(_text(v) for v in value)


_FLAG = (lambda v: isinstance(v, bool), "must be true or false")
_TEXT = (_text, "must be a non-empty string")
_POS_INT = (_positive_int, "must be an integer >= 1")
_STRINGS = (_strings, "must be a list of strings")
_NONEMPTY_STRINGS = (_nonempty_strings, "must be a list of non-empty strings")


def _validate_fields(
    block: Mapping[str, Any],
    rules: Mapping[str, tuple[Callable[[Any], bool], str]],
    errors: list[str],
    where: str,
) -> None:
    """Check optional scalar/list fields; absent and null values are left alone."""
    for key, (valid, message) in rules.items():
        value = block.get(key)
        if value is not None and not valid(value):
            errors.append(f"'{where}.{key}' {message}")


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _validate_disagg(block: Any, errors: list[str]) -> None:
    """Validate ``accuracy.disagg``: the disaggregated-serving test the run writes and names."""
    where = f"{ACCURACY_FIELD}.disagg"
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping")
        return
    reject_unknown_keys(block, DISAGG_KEYS, errors, where=where)
    test = block.get("test")
    if test is not None and (
        not isinstance(test, str) or "::" not in test or Path(test).is_absolute()
    ):
        errors.append(
            f"'{where}.test' must be a repository-relative pytest node id "
            f"(`tests/.../test_x.py::TestY::test_z`), got {test!r}"
        )
    tests = block.get("tests")
    if tests is not None:
        if not isinstance(tests, list) or not tests:
            errors.append(f"'{where}.tests' must be a nonempty list of native node ids")
        else:
            seen = set()
            for selected in tests:
                if (
                    not isinstance(selected, str)
                    or not isinstance(test, str)
                    or not (
                        selected == test
                        or selected.startswith(test + "[")
                        or selected.startswith(test + "::")
                    )
                ):
                    errors.append(f"'{where}.tests' must refine '{where}.test', got {selected!r}")
                elif selected in seen:
                    errors.append(f"'{where}.tests' repeats {selected!r}")
                else:
                    seen.add(selected)
    _validate_fields(
        block, {"devices": _POS_INT, "nodes": _POS_INT, "timeout": _POS_INT}, errors, where
    )
    env = block.get("env")
    if env is not None:
        if not isinstance(env, dict):
            errors.append(f"'{where}.env' must be a mapping from variable to value")
        else:
            for name, value in env.items():
                if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
                    errors.append(
                        f"'{where}.env' names {name!r}, which is not a shell variable name"
                    )
                if not isinstance(value, (str, int, float, bool)):
                    errors.append(f"'{where}.env.{name}' must be a scalar the shell can carry")


def load_prompts_file(path: str | Path) -> tuple[str, ...]:
    """Read a prompt list for the boot gate's declared workload.

    A YAML or JSON **list** of strings, or a **mapping** with a ``prompts:``
    list. Raises :class:`TaskSchemaError` naming the file for anything else,
    including an empty list.
    """
    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TaskSchemaError(f"cannot read the prompts file {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TaskSchemaError(f"the prompts file {source} is not valid YAML/JSON: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("prompts")
    if not isinstance(data, list) or not data:
        raise TaskSchemaError(
            f"the prompts file {source} must hold a non-empty list of strings, or a "
            f"mapping with one under 'prompts'"
        )
    if not all(isinstance(p, str) and p.strip() for p in data):
        raise TaskSchemaError(f"every prompt in {source} must be a non-empty string")
    return tuple(data)


def _validate_boot_level(level: Mapping[str, Any], errors: list[str], where: str) -> None:
    """Validate one level of ``accuracy.boot``."""
    prompts = level.get("prompts")
    if prompts is not None:
        if not isinstance(prompts, list) or not prompts:
            errors.append(f"'{where}.prompts' must be a non-empty list")
        else:
            for n, entry in enumerate(prompts):
                at = f"{where}.prompts[{n}]"
                if isinstance(entry, str):
                    if not entry.strip():
                        errors.append(f"'{at}' must be a non-empty string")
                    continue
                if not isinstance(entry, dict):
                    errors.append(f"'{at}' must be a string or a mapping")
                    continue
                reject_unknown_keys(entry, BOOT_PROMPT_KEYS, errors, where=at)
                if not isinstance(entry.get("prompt"), str) or not entry["prompt"].strip():
                    errors.append(f"'{at}.prompt' is required and must be a non-empty string")
                _validate_fields(
                    entry,
                    dict.fromkeys(("keywords", "keywords_any", "images"), _NONEMPTY_STRINGS),
                    errors,
                    at,
                )
                if entry.get("render") is not None and entry["render"] not in BOOT_RENDERS:
                    errors.append(f"'{at}.render' must be one of {BOOT_RENDERS}")
    prompts_file = level.get("prompts_file")
    if prompts_file is not None:
        if not isinstance(prompts_file, str) or not Path(prompts_file).is_file():
            errors.append(f"'{where}.prompts_file' does not exist: {prompts_file}")
    _validate_fields(
        level,
        {"keywords": _STRINGS, "max_tokens": _POS_INT, GATE_TIMEOUT_KEY: _POS_INT},
        errors,
        where,
    )


def _validate_boot(block: Any, errors: list[str]) -> None:
    """Validate ``accuracy.boot``: the run-wide keys, then each unit's entry."""
    where = f"{ACCURACY_FIELD}.boot"
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping")
        return
    _validate_boot_level(block, errors, where)
    for unit, entry in block.items():
        if unit in BOOT_KEYS or unit == GATE_TIMEOUT_KEY:
            continue
        if not isinstance(entry, dict):
            errors.append(
                f"'{where}.{unit}' is neither a boot key ({list(BOOT_KEYS)}) nor a unit's "
                f"mapping of them"
            )
            continue
        reject_unknown_keys(entry, (*BOOT_KEYS, GATE_TIMEOUT_KEY), errors, where=f"{where}.{unit}")
        _validate_boot_level(entry, errors, f"{where}.{unit}")


# The sample count a run actually scored, where the eval prints one: the MMLU
# line carries it in parentheses, and lm-eval's partial-score line (printed
# under `TLLM_EVAL_PARTIAL_SCORES_EVERY`) ends at `<total>/<total>`.
_MMLU_SAMPLES = re.compile(r"MMLU weighted average accuracy:\s*[0-9.]+\s*\((\d+)\)")
_PARTIAL_TOTAL = re.compile(r"Partial scores after\s+(\d+)/(\d+)\s+responses")
# `--num_samples=N` or `--num_samples N` among the benchmark arguments.
_NUM_SAMPLES_ARG = re.compile(r"^--num_samples(?:=(\d+))?$")


def resolve_per_unit(accuracy: Mapping[str, Any], key: str, unit: str | None) -> Any:
    """Read ``accuracy[key]``, honouring the per-unit mapping form.

    A mapping answers for ``unit``, else its ``default``. The scalar form means
    "the whole run".
    """
    value = accuracy.get(key)
    if isinstance(value, dict):
        chosen = value.get(unit)
        if chosen is None:
            chosen = value.get("default")
        return chosen
    return value


def resolve_gate_timeout(accuracy: Mapping[str, Any], block: str, unit: str | None) -> int | None:
    """``accuracy.<block>``'s ``timeout`` for ``unit``: its own entry's, else the block's; None unsaid."""
    level = accuracy.get(block)
    if not isinstance(level, Mapping):
        return None
    own = level.get(unit) if unit is not None else None
    for source in (own, level):
        if isinstance(source, Mapping) and source.get(GATE_TIMEOUT_KEY) is not None:
            return int(source[GATE_TIMEOUT_KEY])
    return None


def resolve_samples(
    accuracy: Mapping[str, Any],
    unit: str | None,
    benchmark: str,
    output: str = "",
) -> tuple[int | None, str]:
    """How many samples the accuracy score was (or will be) measured on, and from where.

    Most specific first: the count the eval itself printed (what was actually
    scored), then ``--num_samples`` among the unit's benchmark arguments, then
    ``accuracy.samples`` (scalar or per unit), then
    :data:`.accuracy_policy.KNOWN_BENCHMARK_SAMPLES`. ``(None, reason)`` when none of
    them answers.
    """
    if output:
        mmlu = _MMLU_SAMPLES.findall(output)
        if mmlu:
            return int(mmlu[-1]), "the eval's own output"
        partial = _PARTIAL_TOTAL.findall(output)
        if partial:
            return int(partial[-1][1]), "the eval's own output"
    args = [str(a) for a in resolve_per_unit(accuracy, "benchmark_args", unit) or ()]
    for i, arg in enumerate(args):
        matched = _NUM_SAMPLES_ARG.match(arg)
        if not matched:
            continue
        value = matched.group(1) or (args[i + 1] if i + 1 < len(args) else None)
        try:
            return int(str(value)), "--num_samples in benchmark_args"
        except (TypeError, ValueError):
            break
    declared = resolve_per_unit(accuracy, "samples", unit)
    if declared is not None:
        try:
            return int(declared), "accuracy.samples"
        except (TypeError, ValueError):
            pass
    known = accuracy_policy.KNOWN_BENCHMARK_SAMPLES.get(benchmark)
    if known is not None:
        return known, f"the {benchmark} split size"
    return None, f"no sample count for `{benchmark}`: declare `accuracy.samples`"


def _validate_floor(block: Mapping[str, Any], errors: list[str]) -> None:
    """Validate ``accuracy.floor`` and that its rule can be computed for every unit.

    The statistical rule needs a sample count per benchmark.
    """
    where = f"{ACCURACY_FIELD}.floor"
    floor = block.get("floor")
    if floor is not None:
        if not isinstance(floor, dict):
            errors.append(f"'{where}' must be a mapping with keys {list(FLOOR_KEYS)}")
            floor = {}
        reject_unknown_keys(floor, FLOOR_KEYS, errors, where=where)
        rule = floor.get("rule")
        if rule is not None and rule not in accuracy_policy.FLOOR_RULES:
            errors.append(
                f"'{where}.rule' must be one of {accuracy_policy.FLOOR_RULES}, got {rule!r}"
            )
        k = floor.get("k")
        if k is not None and (not _number(k) or k <= 0):
            errors.append(f"'{where}.k' must be a number > 0, got {k!r}")
        members = floor.get("shift_min_members")
        if members is not None and not _positive_int(members):
            errors.append(f"'{where}.shift_min_members' must be an integer >= 1, got {members!r}")
        acknowledge = floor.get("acknowledge")
        if acknowledge is not None and (
            not isinstance(acknowledge, dict)
            or not all(isinstance(v, str) and v.strip() for v in acknowledge.values())
        ):
            errors.append(
                f"'{where}.acknowledge' must map a unit to the reason its systematic shift "
                f"is accepted, as a non-empty string"
            )
    else:
        floor = {}

    samples = block.get("samples")
    if samples is not None:
        values = samples.values() if isinstance(samples, dict) else [samples]
        if not all(_positive_int(v) for v in values):
            errors.append(
                f"'{ACCURACY_FIELD}.samples' must be an integer >= 1, or a mapping from unit "
                f"to one; got {samples!r}"
            )
            return

    if (
        floor.get("rule") or accuracy_policy.FLOOR_STATISTICAL
    ) != accuracy_policy.FLOOR_STATISTICAL:
        return

    benchmarks = block.get("benchmark")
    pairs = (
        list(benchmarks.items())
        if isinstance(benchmarks, dict)
        else [(None, benchmarks or "gsm8k")]
    )
    for unit, benchmark in pairs:
        if not isinstance(benchmark, str):
            continue
        count, source = resolve_samples(block, None if unit == "default" else unit, benchmark)
        if count is None:
            errors.append(
                f"'{ACCURACY_FIELD}' uses the statistical floor, and there is {source} "
                f"(unit {unit or 'every unit'}); the floor is anchor - k * SE and the SE "
                f"needs it"
            )


def _validate_serve(block: Any, errors: list[str]) -> None:
    """Validate ``accuracy.serve``."""
    where = f"{ACCURACY_FIELD}.serve"
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping")
        return
    reject_unknown_keys(block, SERVE_KEYS, errors, where=where)
    _validate_fields(
        block,
        {"enabled": _FLAG, "max_tokens": _POS_INT, "timeout": _POS_INT, "prompt": _TEXT},
        errors,
        where,
    )


def _bounds_server_length(accuracy: Mapping[str, Any]) -> bool:
    """Whether the spec says what sequence length ``trtllm-serve`` sizes its cache for."""
    if accuracy.get("max_seq_len") is not None:
        return True
    options = accuracy.get("llm_api_options")
    if isinstance(options, dict):
        options = options.get("default")
    if not isinstance(options, str) or not Path(options).is_file():
        return False
    try:
        loaded = yaml.safe_load(Path(options).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(loaded, dict) and loaded.get("max_seq_len") is not None


def _profile_defaults(profile: Any, accuracy: dict[str, Any]) -> None:
    """The serve switch the run's ``profile`` sets where ``accuracy`` leaves it unsaid, in place.

    ``certified`` is ``standard`` with the serve gate on; ``bringup`` and ``smoke``
    turn it off. ``standard`` sets nothing.
    """
    if profile not in (PROFILE_SMOKE, PROFILE_BRINGUP, PROFILE_CERTIFIED):
        return
    on = profile == PROFILE_CERTIFIED
    block = accuracy.get("serve")
    if block is None:
        accuracy["serve"] = {"enabled": on}
    elif isinstance(block, dict):
        block.setdefault("enabled", on)


def derive_gate_defaults(data: dict[str, Any], spec_dir: Path) -> None:
    """Turn on, in place, the gates the spec leaves unsaid and the run can derive.

    ``accuracy.serve``: a block without ``enabled`` gets it, true wherever the
    spec bounds the server's sequence length (:func:`_bounds_server_length`). A
    spec that says ``enabled: false`` keeps it off. A ``profile`` sets its
    switches first (:func:`_profile_defaults`). ``spec_dir`` is the spec's
    directory.
    """
    accuracy = data.get(ACCURACY_FIELD)
    if not isinstance(accuracy, dict):
        return
    _profile_defaults(profile_of(data), accuracy)
    serve = accuracy.get("serve")
    if serve is None or (isinstance(serve, dict) and "enabled" not in serve):
        accuracy["serve"] = {**(serve or {}), "enabled": _bounds_server_length(accuracy)}


def _validate_score_pattern(value: Any, errors: list[str]) -> None:
    """``accuracy.score_pattern``: a regular expression with a group, or a per-unit mapping of them."""
    if value is None:
        return
    where = f"{ACCURACY_FIELD}.score_pattern"
    if isinstance(value, dict):
        if "default" not in value:
            errors.append(f"'{where}' is a per-unit mapping, so it needs a 'default' key")
        entries = [(f"{where}[{unit}]", entry) for unit, entry in value.items()]
    else:
        entries = [(where, value)]
    for name, entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            errors.append(f"'{name}' must be a non-empty regular expression")
            continue
        try:
            groups = re.compile(entry).groups
        except re.error as exc:
            errors.append(f"'{name}' is not a regular expression: {exc}")
            continue
        if groups < 1:
            errors.append(f"'{name}' needs a group around the score, e.g. `accuracy: ([0-9.]+)`")


def _validate_accuracy(block: Any, errors: list[str]) -> None:
    """Validate the ``accuracy`` block in place, appending to ``errors``."""
    if not isinstance(block, dict):
        raise TaskSchemaError(f"'{ACCURACY_FIELD}' must be a mapping, got {type(block).__name__}")

    reject_unknown_keys(block, ACCURACY_KEYS, errors, where=ACCURACY_FIELD)

    for name in ("benchmark", "reference_yaml"):
        value = block.get(name)
        if isinstance(value, str):
            if not value.strip():
                errors.append(f"'{ACCURACY_FIELD}.{name}' must be a non-empty string")
            continue
        if isinstance(value, dict):
            if "default" not in value:
                errors.append(
                    f"'{ACCURACY_FIELD}.{name}' is a per-unit mapping, so it needs a "
                    f"'default' key -- otherwise a unit nobody listed silently has no "
                    f"{name} and the gate falls back to a hard-coded one"
                )
            for unit, entry in value.items():
                if not isinstance(entry, str) or not entry.strip():
                    errors.append(f"'{ACCURACY_FIELD}.{name}[{unit}]' must be a non-empty string")
            continue
        errors.append(
            f"'{ACCURACY_FIELD}.{name}' must be a non-empty string, or a mapping "
            f"from assembly unit to one"
        )

    _validate_score_pattern(block.get("score_pattern"), errors)

    tolerance = block.get("tolerance")
    if tolerance is None:
        block["tolerance"] = DEFAULT_ACCURACY_TOLERANCE
    elif not _number(tolerance):
        errors.append(f"'{ACCURACY_FIELD}.tolerance' must be a number")
    elif tolerance < 0:
        errors.append(f"'{ACCURACY_FIELD}.tolerance' must be >= 0; got {tolerance}")

    # ``dataset_path`` is optional but load-bearing for MMLU: trtllm-eval
    # fetches that one dataset itself from a URL that now 404s, writes the
    # error page to data.tar, and dies inside tarfile.open — so the failure
    # reads as a corrupt archive rather than a dead link. Every other
    # benchmark comes through lm-eval and is unaffected.
    dataset_path = block.get("dataset_path")
    if dataset_path is not None and (not isinstance(dataset_path, str) or not dataset_path.strip()):
        errors.append(f"'{ACCURACY_FIELD}.dataset_path' must be a non-empty string when present")

    # Shared public LLM options are read from this YAML before each check launch.
    options = block.get("llm_api_options")
    if isinstance(options, dict):
        for unit, path in options.items():
            if not isinstance(path, str) or not Path(path).is_file():
                errors.append(f"'{ACCURACY_FIELD}.llm_api_options[{unit}]' does not exist: {path}")
        options = None
    if options is not None:
        if not isinstance(options, str) or not options.strip():
            errors.append(f"'{ACCURACY_FIELD}.llm_api_options' must be a non-empty path")
        elif not Path(options).is_file():
            errors.append(
                f"'{ACCURACY_FIELD}.llm_api_options' does not exist: {options}. Without it "
                f"the gate measures the engine's defaults, not the configuration the "
                f"anchor was measured under."
            )

    _validate_fields(
        block,
        dict.fromkeys(("max_seq_len", "max_batch_size", "timeout"), _POS_INT),
        errors,
        ACCURACY_FIELD,
    )

    _validate_floor(block, errors)

    if block.get("boot") is not None:
        _validate_boot(block["boot"], errors)

    if block.get("disagg") is not None:
        _validate_disagg(block["disagg"], errors)

    if block.get("serve") is not None:
        _validate_serve(block["serve"], errors)

    anchor = block.get("anchor")
    for where, state in (
        anchor.items() if isinstance(anchor, dict) else [("", anchor)] if anchor is not None else []
    ):
        if state not in ANCHOR_STATES:
            errors.append(
                f"'{ACCURACY_FIELD}.anchor{'.' + str(where) if where else ''}' must be one of "
                f"{list(ANCHOR_STATES)}; got {state!r}. Without a written anchor, record a "
                f"low placeholder in reference_yaml and replace it when a measured one is known"
            )

    members = block.get("benchmark_members")
    named = isinstance(members, str) and PARALLEL_PATTERN.match(members)
    if members is not None and members not in BENCHMARK_MEMBERS and not named:
        errors.append(
            f"'{ACCURACY_FIELD}.benchmark_members' must be one of {list(BENCHMARK_MEMBERS)} "
            f"or a member (tp4, dep4, ...); got {members!r}"
        )

    # A mapping of scalars (the whole run) or of unit -> mapping of scalars.
    discriminators = block.get("discriminators")
    if discriminators is not None:
        if not isinstance(discriminators, dict):
            errors.append(f"'{ACCURACY_FIELD}.discriminators' must be a mapping")
        else:
            nested = [v for v in discriminators.values() if isinstance(v, dict)]
            if nested and len(nested) != len(discriminators):
                errors.append(
                    f"'{ACCURACY_FIELD}.discriminators' mixes the flat form "
                    f"(quant_algo: ...) with the per-unit form (core: {{...}}); pick one"
                )
            for group in nested or [discriminators]:
                for key, value in group.items():
                    if not isinstance(value, (str, int, float, bool)):
                        errors.append(
                            f"'{ACCURACY_FIELD}.discriminators.{key}' must be a scalar "
                            f"matching a reference-file discriminator, got "
                            f"{type(value).__name__}"
                        )

    # The evaluation *protocol* -- chat template, few-shot handling, seed --
    # is expressed as options of the benchmark subcommand, so it cannot live
    # in the llm_api_options YAML.
    benchmark_args = block.get("benchmark_args")

    def _is_arg_list(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(a, str) for a in value)

    if benchmark_args is None:
        block["benchmark_args"] = []
    elif isinstance(benchmark_args, dict):
        if "default" not in benchmark_args:
            errors.append(
                f"'{ACCURACY_FIELD}.benchmark_args' is a per-unit mapping, so it needs "
                f"a 'default' key"
            )
        for unit, value in benchmark_args.items():
            if not _is_arg_list(value):
                errors.append(
                    f"'{ACCURACY_FIELD}.benchmark_args[{unit}]' must be a list of strings"
                )
    elif not _is_arg_list(benchmark_args):
        errors.append(
            f"'{ACCURACY_FIELD}.benchmark_args' must be a list of strings, or a "
            f"mapping from assembly unit to one"
        )


def _validate_profile(data: Mapping[str, Any], errors: list[str]) -> None:
    """``profile``: one of :data:`PROFILES`, or a mapping from unit, variant or ``default`` to one."""
    profile = data.get(PROFILE_FIELD)
    if profile is None:
        return
    if not isinstance(profile, Mapping):
        if profile not in PROFILES:
            errors.append(f"'{PROFILE_FIELD}' must be one of {list(PROFILES)}, or a mapping")
        return
    named = {"default", "core"}
    if isinstance(data.get(FEATURES_FIELD), list):
        named.update(str(f) for f in data[FEATURES_FIELD])
    if isinstance(data.get(VARIANTS_FIELD), Mapping):
        named.update(str(v) for v in data[VARIANTS_FIELD])
    for unit, value in profile.items():
        if unit not in named:
            errors.append(
                f"'{PROFILE_FIELD}.{unit}' names no unit or variant of this spec ({sorted(named)})"
            )
        if value not in PROFILES:
            errors.append(f"'{PROFILE_FIELD}.{unit}' must be one of {list(PROFILES)}")


def _require_relative_on_variants(data: Mapping[str, Any], errors: list[str]) -> None:
    """``anchor: relative`` names a configuration variant: a unit has no unit under it."""
    accuracy = data.get(ACCURACY_FIELD)
    anchor = accuracy.get("anchor") if isinstance(accuracy, dict) else None
    variants = data.get("variants") if isinstance(data.get("variants"), dict) else {}
    for where, state in (
        anchor.items() if isinstance(anchor, dict) else [("", anchor)] if anchor is not None else []
    ):
        if state == ANCHOR_RELATIVE and where not in variants:
            errors.append(
                f"'{ACCURACY_FIELD}.anchor{'.' + str(where) if where else ''}' is "
                f"{ANCHOR_RELATIVE!r}, which only a configuration variant can be: it is judged "
                f"against its unit's own score. A unit's anchor is a row of reference_yaml"
            )


def gpus_per_node(data: Mapping[str, Any]) -> int | None:
    """The devices the run's host holds (``environment.gpus_per_node``), when the spec says."""
    size = (data.get(ENVIRONMENT_FIELD) or {}).get("gpus_per_node")
    return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else None


def _require_disagg_test(data: Mapping[str, Any], errors: list[str]) -> None:
    """The ``disagg`` feature names the test that proves it: nothing else crosses the hop."""
    features = data.get(FEATURES_FIELD)
    if not isinstance(features, list) or DISAGG_FEATURE not in features:
        return
    accuracy = data.get(ACCURACY_FIELD)
    block = accuracy.get("disagg") if isinstance(accuracy, dict) else None
    if not isinstance(block, dict) or not block.get("test"):
        errors.append(
            f"feature {DISAGG_FEATURE!r} needs 'accuracy.disagg.test': the node id of the "
            f"disaggregated-serving accuracy test the run writes beside the tree's own (they "
            f"serve through `trtllm-serve disaggregated`); boot and accuracy run one aggregated "
            f"engine and never cross the hop."
        )


def _resolve_spec_relative_paths(data: dict[str, Any], spec_dir: Path) -> None:
    """Make paths the user writes NEXT TO the spec portable, in place.

    ``accuracy.llm_api_options``, the boot prompt files and boot images
    are resolved against the **spec's own directory** and written back as
    absolute. Absolute paths are untouched, so a workspace copy of an
    already-resolved spec re-resolves to itself.

    Deliberately NOT applied to ``checkpoint_path``, ``trtllm_repo_path`` or
    ``environment.spool``. Those genuinely differ per machine and are read by
    processes that never see the spec's directory.
    """

    def absolute(value: Any) -> Any:
        if not isinstance(value, str) or not value.strip():
            return value
        candidate = Path(value)
        return value if candidate.is_absolute() else str((spec_dir / candidate).resolve())

    accuracy = data.get(ACCURACY_FIELD)
    if not isinstance(accuracy, dict):
        return

    options = accuracy.get("llm_api_options")
    if isinstance(options, dict):
        accuracy["llm_api_options"] = {k: absolute(v) for k, v in options.items()}
    elif options is not None:
        accuracy["llm_api_options"] = absolute(options)

    # Boot prompt files, and boot images that sit beside the spec. An image
    # that does not is left relative: the gate resolves it against the tree
    # under test, where the repository's own test assets are.
    boot = accuracy.get("boot")
    if isinstance(boot, dict):
        units = (v for k, v in boot.items() if k not in BOOT_KEYS and isinstance(v, dict))
        for level in (boot, *units):
            if level.get("prompts_file") is not None:
                level["prompts_file"] = absolute(level["prompts_file"])
            for entry in level.get("prompts") or ():
                if isinstance(entry, dict) and isinstance(entry.get("images"), list):
                    entry["images"] = [
                        absolute(i)
                        if isinstance(i, str)
                        and not Path(i).is_absolute()
                        and (spec_dir / i).exists()
                        else i
                        for i in entry["images"]
                    ]


def _validate_shorthand(data: dict[str, Any], errors: list[str]) -> None:
    """Validate the keys that state a run as units, members and variants, filling their defaults.

    They are shorthand for deployments (``spec/deployments.py``), and the checks and
    the run still read them as written.
    """
    # After `parallel` has been defaulted, since legality depends on it.
    if MEMBERS_FIELD in data:
        environment = data.get(ENVIRONMENT_FIELD)
        kind = environment.get("kind", "local") if isinstance(environment, dict) else "local"
        _validate_members(
            data[MEMBERS_FIELD], data.get(PARALLEL_FIELD), errors, environment_kind=str(kind)
        )

    features = data.get(FEATURES_FIELD)
    if features is None:
        data[FEATURES_FIELD] = []
    elif not isinstance(features, list):
        errors.append(
            f"'{FEATURES_FIELD}' must be a list of strings, got {type(features).__name__}"
        )
    else:
        for index, feature in enumerate(features):
            if not isinstance(feature, str):
                errors.append(
                    f"'{FEATURES_FIELD}[{index}]' must be a string, got {type(feature).__name__}"
                )
            elif feature not in FEATURE_FAMILIES:
                errors.append(
                    f"'{FEATURES_FIELD}[{index}]' is {feature!r}, which is not a known "
                    f"feature. Known features: {sorted(FEATURE_FAMILIES)}"
                )
        readers = [f for f in MTP_NAMESPACE_FEATURES if f in features]
        if len(readers) > 1:
            errors.append(
                f"'{FEATURES_FIELD}' lists {readers}, and all of them read the checkpoint's "
                f"`mtp.*` weights, which are MTP modules or DSpark stages, never both. "
                f"Keep the one the checkpoint ships. An `mtp.*` that holds DSpark stages "
                f"(`dspark_*` keys in its config) is spelled `dspark`, even when the "
                f"request says 'MTP'."
            )

    if isinstance(data.get(FEATURES_FIELD), list):
        validate_variants(
            data.get(VARIANTS_FIELD),
            units=assembly_units(data),
            speculators=set(SPECULATORS),
            unified=data.get(PARALLEL_FIELD) == PARALLEL_UNIFIED,
            errors=errors,
        )

    _validate_coverage(data.get(COVERAGE_FIELD), errors)
    _validate_profile(data, errors)

    if RUNTIME_FIELD not in data:
        data[RUNTIME_FIELD] = dict(RUNTIME_SWITCHES) | {"extra_flags": []}
    else:
        try:
            _validate_runtime(data[RUNTIME_FIELD], errors)
        except TaskSchemaError as exc:
            errors.append(str(exc))

    scope = data.get(SCOPE_FIELD)
    if scope is None:
        data[SCOPE_FIELD] = DEFAULT_SCOPE
    elif scope not in SCOPES:
        errors.append(f"'{SCOPE_FIELD}' must be one of {SCOPES}; got {scope!r}")

    if ACCURACY_FIELD not in data:
        errors.append(
            f"missing required field '{ACCURACY_FIELD}'. The accuracy gate is the release "
            f"criterion, and its anchor comes from a written source — never from the run."
        )
    else:
        try:
            _validate_accuracy(data[ACCURACY_FIELD], errors)
        except TaskSchemaError as exc:
            errors.append(str(exc))
        _require_disagg_test(data, errors)
        _require_relative_on_variants(data, errors)


def drop_ignored_keys(data: dict[str, Any], notices: list[str] | None) -> None:
    """Take the keys this version no longer reads out of ``data``, one notice for each."""
    accuracy = data.get(ACCURACY_FIELD)
    if not isinstance(accuracy, dict):
        return
    blocks = [(ACCURACY_FIELD, accuracy)]
    declared = data.get(VARIANTS_FIELD)
    for name, entry in declared.items() if isinstance(declared, dict) else ():
        if isinstance(entry, dict) and isinstance(entry.get(ACCURACY_FIELD), dict):
            blocks.append((f"{VARIANTS_FIELD}.{name}.{ACCURACY_FIELD}", entry[ACCURACY_FIELD]))
    for where, block in blocks:
        for key, why in IGNORED_ACCURACY_KEYS.items():
            if key in block:
                del block[key]
                if notices is not None:
                    notices.append(f"'{where}.{key}' is ignored: {why}; drop the key")


def load_and_validate_task_yaml(
    path: str | Path, notices: list[str] | None = None
) -> dict[str, Any]:
    """Parse ``path`` as YAML, validate it, and fill in the workflow's defaults.

    Returns the resolved mapping. Raises :class:`TaskSchemaError` with
    **every** detected problem batched into a single message. A spec states what
    the run brings up either as ``deployments`` or through the shorthand keys;
    ``spec.deployments.resolve`` reads both from the mapping returned here. A key this
    version no longer reads (:data:`IGNORED_ACCURACY_KEYS`) is dropped, and
    ``notices``, when given, gains one line saying so.
    """
    task_path = Path(path)
    if not task_path.is_file():
        raise TaskSchemaError(f"task file not found: {task_path}")

    text = task_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TaskSchemaError(f"{task_path} is not valid YAML: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TaskSchemaError(
            f"{task_path} must be a YAML mapping at the top level, got {type(data).__name__}"
        )

    # Imported here: that module reads this one's shorthand helpers.
    from . import deployments as deployments_mod

    native = deployments_mod.is_native(data)
    spec_dir = task_path.resolve().parent
    drop_ignored_keys(data, notices)
    # BEFORE validation, deliberately. `accuracy.llm_api_options` and a deployment's
    # `options` have an existence check, and that check is a *consumer* of the resolved path.
    _resolve_spec_relative_paths(data, spec_dir)
    if native:
        deployments_mod.resolve_paths(data, spec_dir)
    else:
        # After the paths, before validation: what it derives is validated like
        # what the spec says.
        derive_gate_defaults(data, spec_dir)

    errors: list[str] = []

    validate_commit_policy_block(data.get(COMMIT_POLICY_FIELD), errors)

    if "perf" in data:
        errors.append(
            "'perf' is not read: the flow has no tuning stage, and performance work runs "
            "outside it on the delivered target; drop the block"
        )
    reject_unknown_keys({k: v for k, v in data.items() if k != "perf"}, TOP_LEVEL_KEYS, errors)
    errors.extend(validate_agents(data, ROLES))

    for name in REQUIRED_PATH_FIELDS:
        if name not in data:
            errors.append(f"missing required field '{name}'")
            continue
        value = data[name]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{name}' must be a non-empty string, got {type(value).__name__}")
            continue
        if not Path(value).exists():
            errors.append(f"'{name}' points to a non-existent path: {value}")

    arch = data.get(ARCH_FIELD)
    if arch is None:
        errors.append(
            f"missing required field '{ARCH_FIELD}'. It is never inferred: catalog "
            f"receipts are per-architecture, so the wrong value gates this target "
            f"against another machine's certification."
        )
    elif not isinstance(arch, str) or not ARCH_PATTERN.match(arch):
        errors.append(f"'{ARCH_FIELD}' must look like 'sm_100' / 'sm_103'; got {arch!r}")

    parallel = data.get(PARALLEL_FIELD)
    if parallel is None:
        data[PARALLEL_FIELD] = DEFAULT_PARALLEL
    elif not isinstance(parallel, str):
        errors.append(f"'{PARALLEL_FIELD}' must be a string, got {type(parallel).__name__}")
    else:
        try:
            parse_parallel_segment(parallel)
        except TaskSchemaError as exc:
            errors.append(str(exc))

    if native:
        deployments_mod.validate_native_spec(data, errors)
        accuracy = data.get(ACCURACY_FIELD)
        if isinstance(accuracy, dict) and accuracy.get("boot") is not None:
            _validate_boot(accuracy["boot"], errors)
    else:
        _validate_shorthand(data, errors)

    if ENVIRONMENT_FIELD not in data:
        errors.append(
            f"missing required field '{ENVIRONMENT_FIELD}'. It declares how GPU work is "
            f"launched; without it the agents would have to discover that themselves, "
            f"which is exactly what this workflow refuses to do."
        )
    else:
        try:
            _validate_environment(data[ENVIRONMENT_FIELD], errors)
        except TaskSchemaError as exc:
            errors.append(str(exc))

    if TARGET_FIELD not in data:
        data[TARGET_FIELD] = {}
    else:
        try:
            _validate_target(data[TARGET_FIELD], errors)
        except TaskSchemaError as exc:
            errors.append(str(exc))

    notes = data.get(NOTES_FIELD)
    if notes is not None and (not isinstance(notes, str) or not notes.strip()):
        errors.append(f"'{NOTES_FIELD}' must be a non-empty string")

    if errors:
        bullet = "\n  - "
        raise TaskSchemaError(
            f"{task_path} failed trtllm-modeling-v2 schema validation:{bullet}{bullet.join(errors)}"
        )

    return data


def features_through(task_data: Mapping[str, Any], unit: str) -> tuple[str, ...]:
    """The features a unit's target implements: every feature up to it."""
    features = [str(f) for f in task_data.get(FEATURES_FIELD) or ()]
    if unit not in features:
        return ()
    return tuple(features[: features.index(unit) + 1])


def assembly_units(data: Mapping[str, Any]) -> list[str]:
    """Return the assembly units in dispatch order: ``core`` then each feature."""
    return ["core", *[str(f) for f in data.get(FEATURES_FIELD, [])]]


__all__ = [
    "ANCHOR_RECORDED",
    "ANCHOR_RELATIVE",
    "ANCHOR_STATES",
    "gpus_per_node",
    "COVERAGE_FIELD",
    "COVERAGE_KEYS",
    "resolve_gate_timeout",
    "resolve_per_unit",
    "resolve_samples",
    "BENCHMARK_MEMBERS",
    "BENCHMARK_MEMBERS_EVERY",
    "derive_gate_defaults",
    "drop_ignored_keys",
    "DISAGG_FEATURE",
    "DISAGG_KEYS",
    "ACCURACY_FIELD",
    "ACCURACY_KEYS",
    "ARCH_FIELD",
    "ARCH_PATTERN",
    "DEFAULT_ACCURACY_TOLERANCE",
    "DEFAULT_MEMBER_KINDS",
    "DEFAULT_PARALLEL",
    "DEFAULT_SCOPE",
    "ENVIRONMENT_FIELD",
    "ENVIRONMENT_KEYS",
    "ENVIRONMENT_KINDS",
    "FEATURES_FIELD",
    "FEATURE_FAMILIES",
    "IGNORED_ACCURACY_KEYS",
    "SPECULATORS",
    "PROFILES",
    "PROFILE_BRINGUP",
    "PROFILE_CERTIFIED",
    "PROFILE_FIELD",
    "PROFILE_SMOKE",
    "PROFILE_STANDARD",
    "profile_of",
    "MEMBERS_FIELD",
    "MTP_NAMESPACE_FEATURES",
    "PARALLEL_FIELD",
    "PARALLEL_UNIFIED",
    "REQUIRED_PATH_FIELDS",
    "RUNTIME_FIELD",
    "RUNTIME_SWITCHES",
    "SCOPES",
    "SCOPE_FIELD",
    "SERVE_KEYS",
    "reject_unknown_keys",
    "TARGET_FIELD",
    "TARGET_KEYS",
    "TOP_LEVEL_KEYS",
    "TaskSchemaError",
    "assembly_units",
    "features_through",
    "derive_checkpoint_segment",
    "derive_family",
    "derive_slug",
    "load_and_validate_task_yaml",
    "load_prompts_file",
    "members_for",
    "parse_parallel_segment",
    "routed_family",
    "widest_world_size",
]
