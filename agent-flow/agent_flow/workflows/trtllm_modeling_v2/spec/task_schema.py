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

import re
import shlex
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from agent_flow.agent_runtime import AGENTS_FIELD, validate_agents

from ..agents.roles import ROLES
from ..checks import accuracy_policy
from ..infra.compute.environment import EnvironmentError, check_tmpdir
from ..infra.compute.sessions import validate_sessions
from ..infra.git import COMMIT_POLICY_FIELD, validate_commit_policy_block
from ..tune.schema import validate_perf_block
from .schema_keys import reject_unknown_keys
from .variants import VARIANTS_FIELD, validate_variants

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

# The engine batch limit BOTH legs of a token-comparing witness (residency,
# kv_quant, equivalence) run at. It replaces the runtime block's
# `max_batch_size` in leg A's public LLM configuration, and a `contrast_command` leg B
# is handed it as MV2FLOW_WITNESS_MAX_BATCH_SIZE. `1` is the value that
# gives both legs the same schedule. Unset keeps the runtime block's limit.
WITNESS_BATCH_SIZE_KEY = "max_batch_size"

# The keys every witness block may carry. `expected_mib` is the residency
# witness's prediction, `contrast_command` the equivalence witness's second
# launcher. `timeout` is seconds PER LEG, where the one-engine boot budget does
# not fit.
# The workload a token-comparing witness (residency, kv_quant, equivalence)
# declares, as the acceptance gate does: `prompts` or `prompts_file`,
# `max_tokens`, and `min_generated_tokens` (default 32) -- a witness whose
# continuations are shorter fails as too short to compare. With none, the
# declared acceptance prompts, else the boot prompts.
WITNESS_WORKLOAD_KEYS: tuple[str, ...] = (
    "prompts",
    "prompts_file",
    "max_tokens",
    "min_generated_tokens",
)
WITNESS_KEYS: tuple[str, ...] = (
    "contrast_env",
    "expected_mib",
    "contrast_command",
    "timeout",
    WITNESS_BATCH_SIZE_KEY,
    *WITNESS_WORKLOAD_KEYS,
)

# Features gated by the EQUIVALENCE witness (token identity across a hop).
#
#   contrast_command  REQUIRED, with `contrast_env`. Leg B's launcher: a context
#                     engine and a generation engine with the state handed
#                     between them. A word of it naming a file beside the spec
#                     is resolved against the spec's directory
#   prompt_tokens     optional list: one generated prompt of at least that many
#                     tokens each, added to the boot prompts on BOTH legs
#   legs_env          optional mapping, see WITNESS_LEGS_ENV_KEY
EQUIVALENCE_WITNESS_FEATURES: tuple[str, ...] = ("disagg",)
EQUIVALENCE_WITNESS_REQUIRED: tuple[str, ...] = ("contrast_command", "contrast_env")

# The environment BOTH legs of an equivalence witness run under: leg A's
# process, and leg B's launcher, which is handed it as
# MV2FLOW_WITNESS_LEGS_ENV to apply to every engine it starts. A variable may
# not be in both this and `contrast_env`: there it is the contrast.
WITNESS_LEGS_ENV_KEY = "legs_env"

# The cache transceiver both engines of leg B run, as the LLM API's
# `CacheTransceiverConfig` takes it (`backend`, `transceiver_runtime`, ...).
# The `contrast_command` is handed it as MV2FLOW_WITNESS_TRANSCEIVER.
WITNESS_TRANSCEIVER_KEY = "cache_transceiver_config"

EQUIVALENCE_WITNESS_KEYS: tuple[str, ...] = (
    "contrast_env",
    "contrast_command",
    "prompt_tokens",
    "timeout",
    WITNESS_BATCH_SIZE_KEY,
    WITNESS_LEGS_ENV_KEY,
    WITNESS_TRANSCEIVER_KEY,
    *WITNESS_WORKLOAD_KEYS,
)

# The bounded-replay witness parameters, per feature, each REQUIRED.
#
#   window         the checkpoint's sliding window (e.g. 128)
#   prompt_tokens  decoder_replay: the long prompt's length, at least. Must
#                  exceed `window`
#   prefix_tokens  encoder_replay: the shared prefix's length. Must exceed
#                  `window`
REPLAY_WITNESS_REQUIRED: dict[str, tuple[str, ...]] = {
    "decoder_replay": ("contrast_env", "window", "prompt_tokens"),
    "encoder_replay": ("contrast_env", "window", "prefix_tokens"),
}
# `reuse_alignment`: the reuse boundary must be a multiple of this. A model that
# compresses global KV in pairs (compress ratio 2) needs 2. Default 1, i.e. no
# constraint.
REPLAY_WITNESS_OPTIONAL: dict[str, tuple[str, ...]] = {
    "decoder_replay": ("timeout",),
    "encoder_replay": ("reuse_alignment", "timeout"),
}

# The witness builds its long prompt as a request-entry `--prompt` argument, one
# word per token at least. Linux caps ONE argv string at 128 KiB
# (MAX_ARG_STRLEN) and fails the exec with E2BIG, which surfaces as a leg that
# "did not complete". 16384 words of the witness's prose
# (`checks/witness_inputs.py`) are ~89 KB.
MAX_WITNESS_PROMPT_WORDS = 16384

# Native KV storage can be lossy (exact: false); only exact storage owes token identity.
KV_QUANT_WITNESS_KEYS: tuple[str, ...] = (
    "contrast_env",
    "expected_bytes_per_token",
    "exact",
    "bytes_tolerance",
    "timeout",
    WITNESS_BATCH_SIZE_KEY,
    *WITNESS_WORKLOAD_KEYS,
)
DEFAULT_BYTES_TOLERANCE = 0.02

# Every feature whose witness block is a DECLARATION -- a workload or a figure
# the verdict is judged against, or the launcher of the leg it compares -- and
# is therefore required at spec time, once witnesses are on.
DECLARED_WITNESS_FEATURES: tuple[str, ...] = (
    *REPLAY_WITNESS_REQUIRED,
    "kv_quant",
    *EQUIVALENCE_WITNESS_FEATURES,
)

# `accuracy.witness.enabled`: whether the contract owes the witnesses at all.
# Off by default.
WITNESS_ENABLED_KEY = "enabled"

# Workload and field documentation lives in task.example.yaml. These key sets
# close the accepted blocks; cross-field constraints are validated below.
ACCEPTANCE_WORKLOAD_KEYS: tuple[str, ...] = (
    "prompts",
    "prompts_file",
    "dataset",
    "num_prompts",
    "text_field",
    "prompt_template",
    "split",
    "dataset_config",
    "max_tokens",
    "min_draft_iterations",
    "min_prompts",
)
# The paired t-test against the previous tree's verdict, at the block's top
# level only: `alpha` (one-sided, default 0.05), `min_effect` (accepted tokens
# per iteration, default 0.1). A drop must be both significant and at least
# `min_effect` to fail.
ACCEPTANCE_REGRESSION_KEYS: tuple[str, ...] = ("alpha", "min_effect")
ACCEPTANCE_BLOCK_KEYS: tuple[str, ...] = (*ACCEPTANCE_WORKLOAD_KEYS, "regression")
# What one unit's entry in `accuracy.acceptance` may carry:
#   min_absolute        the absolute floor used when the `=off` leg cannot run
#   measured_reference  what a reference drafter measured on this workload; the
#                       floor is this x 0.95 when `min_absolute` is not given
#   floor_source        where that measurement was taken. A floor with neither
#                       a measured reference nor a source is used, and every
#                       verdict says it is not tied to a measurement
ACCEPTANCE_UNIT_KEYS: tuple[str, ...] = (
    "min_absolute",
    "measured_reference",
    "floor_source",
    *ACCEPTANCE_WORKLOAD_KEYS,
)

TARGET_FIELD = "target"
RUNTIME_FIELD = "runtime"
SCOPE_FIELD = "scope"
NOTES_FIELD = "notes"
SOURCE_EMBARGO_FIELD = "source_embargo"

# Runtime switches the gates must exercise, and which therefore have to be
# declared rather than left to whatever the engine defaults to.
#
# The values are what the *user* wants true of the run, not what the flag
# says: two of these are spelled as negations on the command line
# (``--disable_overlap_scheduler``, ``--disable_kv_cache_reuse``).
RUNTIME_SWITCHES: dict[str, bool] = {
    "cuda_graph": False,
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
    SOURCE_EMBARGO_FIELD,
    AGENTS_FIELD,
    COMMIT_POLICY_FIELD,
    "perf",
    VARIANTS_FIELD,
    "coverage",
    "profile",
)
#: ``coverage.accept``: the requested feature@member cells the contract may leave uncovered.
COVERAGE_FIELD = "coverage"
COVERAGE_KEYS: tuple[str, ...] = ("accept",)
#: ``profile``: how much the contract owes, for the run or per unit (a mapping
#: from unit or variant name, and ``default``, to a profile). ``smoke`` owes a
#: unit's boot only; ``bringup`` gates each unit at its cheapest member, and no
#: configuration variant and no delivery check; ``standard`` is the contract as
#: the spec states it; ``certified`` also turns on the witnesses, boot boundary
#: phases and the serve gate, and runs each unit's benchmark at every member. A
#: switch the spec sets itself keeps its value.
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


TARGET_KEYS: tuple[str, ...] = ("family", "checkpoint_segment", "keep_local")
#: The ``target`` keys that name a Python package directory.
TARGET_NAME_KEYS: tuple[str, ...] = ("family", "checkpoint_segment")

ACCURACY_KEYS: tuple[str, ...] = (
    # The absolute `acceptance_length` floor a unit falls back to when the
    # `=off` leg cannot exist. Also the acceptance gate's declared workload;
    # see `ACCEPTANCE_WORKLOAD_KEYS`.
    "acceptance",
    "benchmark",
    "reference_yaml",
    "model_key",
    "tolerance",
    "dataset_path",
    "llm_api_options",
    # Which ROW of reference_yaml this run anchors to. The references key a
    # model to a list discriminated by quant_algo / kv_cache_quant_algo /
    # spec_dec_algo, and the match is exact.
    "discriminators",
    # Per-unit configuration for the two-leg WITNESS gates.
    "witness",
    # `trtllm-eval` takes these two as its OWN flags rather than through the
    # options YAML, so they have to be declarable here.
    "max_seq_len",
    "max_batch_size",
    "benchmark_args",
    "floor",
    "boot",
    "envelope",
    "serve",
    # The benchmark's sample count, scalar or per unit, for the standard
    # error behind the statistical floor.
    "samples",
    "benchmark_members",
    "anchor",
    "logits",
)

# `accuracy.anchor: pending` (scalar, or per unit): no written anchor exists
# for the unit yet. Its accuracy benchmark is not a check; the logit gate
# against the HF reference (`accuracy.logits`) is the unit's correctness
# oracle, and delivery stays incomplete until a row is recorded from a written
# source.
ANCHOR_PENDING = "pending"
ANCHOR_RECORDED = "recorded"
ANCHOR_STATES: tuple[str, ...] = (ANCHOR_RECORDED, ANCHOR_PENDING)

# `accuracy.benchmark_members`: where each unit's accuracy benchmark runs.
#
#   sampled  (default) with the logit gate on: at one member, the cheapest
#            whose served engine the logit gate does not run whole (attention
#            DP; a speculator, unless the boot compares greedy tokens), else
#            the widest. The logit gate and the boot's greedy tokens, owed at
#            every member, prove the rest. Without the logit gate, at every
#            member.
#   every    at every member, whatever else the contract owes.
#   <member> at that member (`dep4`, ...), for each unit that has it.
BENCHMARK_MEMBERS_SAMPLED = "sampled"
BENCHMARK_MEMBERS_EVERY = "every"
BENCHMARK_MEMBERS: tuple[str, ...] = (BENCHMARK_MEMBERS_SAMPLED, BENCHMARK_MEMBERS_EVERY)

# Boot workloads allow run-wide keys and per-unit overlays. Keywords match
# generated text, never logs; absent multimodal images use the repository asset.
BOOT_KEYS: tuple[str, ...] = ("prompts", "prompts_file", "keywords", "max_tokens", "greedy")
BOOT_PROMPT_KEYS: tuple[str, ...] = ("prompt", "keywords", "keywords_any", "images", "render")
BOOT_RENDERS: tuple[str, ...] = ("loader", "raw")

# `accuracy.boot.greedy` (run-wide, or a unit's own entry): wherever the run has
# an HF logit reference (`accuracy.logits.reference`), the boot also sends its
# first text prompts as token ids and the engine's first greedy tokens must be
# the reference's, up to a near-tie (checks/boot.py, `GreedySetting`).
#
#   enabled   default true; false compares no tokens. A configuration variant
#             compares none unless its own entry says true.
#   prompts   how many of the reference's text prompts (default 4)
#   tokens    greedy tokens compared per prompt (default 24, at most the
#             reference's continuation)
GREEDY_KEYS: tuple[str, ...] = ("enabled", "prompts", "tokens")

# Optional runtime-boundary phases of ordinary boot (checks/boot.py).
# Disabled by default; certified enables them unless explicitly disabled.
# timeout bounds the complete boot invocation, not a second engine launch.
ENVELOPE_KEYS: tuple[str, ...] = (
    "enabled",
    "max_tokens",
    "agree_tokens",
    "long_prompts",
    "probes",
    "timeout",
)

# Serve defaults are derived from the declared sequence bound in derive_gate_defaults.
SERVE_KEYS: tuple[str, ...] = ("enabled", "prompt", "max_tokens", "timeout")

# Floor computation and systematic-shift findings are owned by checks/stats.py.
FLOOR_KEYS: tuple[str, ...] = ("rule", "k", "shift_min_members", "acknowledge")

ENVIRONMENT_KEYS: tuple[str, ...] = (
    "kind",
    "python",
    "env",
    "spool",
    "partition",
    "account",
    "container_image",
    "mounts",
    "time_limit",
    "gpus_per_node",
    # Pyxis reuses an already-imported container of this name on the node.
    "container_name",
    # A checkpoint the STOCK modeling path already supports, used by
    # `preflight.substrate_gate` when the run's own checkpoint has an
    # architecture that path has never heard of.
    "substrate_checkpoint",
    # The compute sessions the run stands on.
    "sessions",
    # How long the orchestrator waits for a gate session that stopped
    # heart-beating before it closes the run resumably.
    "session_wait",
)

ENVIRONMENT_KINDS: tuple[str, ...] = ("local", "session_spool")
ENVIRONMENT_REQUIRED_FIELDS: tuple[str, ...] = ("kind", "python")

# Accuracy tolerance, in benchmark points, under `accuracy.floor.rule:
# tolerance` only.
DEFAULT_ACCURACY_TOLERANCE = 5.0


class TaskSchemaError(ValueError):
    """Raised when ``task.yaml`` fails trtllm-modeling-v2 schema validation."""


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


def derive_family(architecture: str) -> str:
    """``GptOssForCausalLM`` -> ``gpt_oss``: the modeling_v2 family directory."""
    name = re.sub(r"For(Causal|Conditional|Seq2Seq|Sequence)?LM$", "", architecture)
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
    keep = block.get("keep_local")
    if keep is not None and (
        not isinstance(keep, list)
        or not all(isinstance(e, str) and re.match(r"^[a-z_]+/[a-z0-9_]+$", e) for e in keep)
    ):
        errors.append(
            f"'{TARGET_FIELD}.keep_local' lists shared catalog entries as "
            f"'<category>/<name>' (lowercase with underscores); got {keep!r}"
        )
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

    if len(set(world_sizes.values())) > 1 and environment_kind == "local":
        widest = max(world_sizes, key=lambda m: world_sizes[m])
        narrowest = min(world_sizes, key=lambda m: world_sizes[m])
        errors.append(
            f"'{MEMBERS_FIELD}' mixes world sizes: {narrowest} wants "
            f"{world_sizes[narrowest]} device(s) and {widest} wants {world_sizes[widest]}. "
            f"A local run claims one set of devices for every member, so they must all "
            f"declare the same world size; under `environment.kind: session_spool` each "
            f"gate asks the scheduler for its own member's devices."
        )


def _validate_source_embargo(block: Any, errors: list[str], repo: Any) -> None:
    """Validate ``source_embargo``: paths agents may not read.

    An entry is **repo-relative** (resolved under ``trtllm_repo_path``) or
    **absolute and outside the repo**. Either way it must **exist**; a
    **directory** entry embargoes the tree beneath it.
    """
    if not isinstance(block, list) or not all(isinstance(p, str) for p in block):
        raise TaskSchemaError(
            f"'{SOURCE_EMBARGO_FIELD}' must be a list of paths: repo-relative, or absolute for "
            f"a tree outside the repo"
        )
    if not isinstance(repo, str):
        return
    repo_root = Path(repo).resolve()
    for entry in block:
        if entry.startswith("/"):
            resolved = Path(entry).resolve()
            if resolved == repo_root or repo_root in resolved.parents:
                errors.append(
                    f"'{SOURCE_EMBARGO_FIELD}' names {entry!r}, which is inside "
                    f"{repo}. Spell an in-repo path repo-relative, so the list means "
                    f"the same thing from any working directory."
                )
            elif not resolved.exists():
                errors.append(
                    f"'{SOURCE_EMBARGO_FIELD}' names {entry!r}, which does not exist. "
                    f"An embargo on a path that is not there protects nothing."
                )
        elif not (Path(repo) / entry).exists():
            errors.append(
                f"'{SOURCE_EMBARGO_FIELD}' names {entry!r}, which does not exist under "
                f"{repo}. An embargo on a path that is not there protects nothing."
            )


def _validate_environment(block: Any, errors: list[str]) -> None:
    """Validate the ``environment`` block in place, appending to ``errors``."""
    if not isinstance(block, dict):
        raise TaskSchemaError(
            f"'{ENVIRONMENT_FIELD}' must be a mapping, got {type(block).__name__}"
        )

    reject_unknown_keys(block, ENVIRONMENT_KEYS, errors, where=ENVIRONMENT_FIELD)

    validate_sessions(block, errors, where=ENVIRONMENT_FIELD)

    for name in ENVIRONMENT_REQUIRED_FIELDS:
        value = block.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{ENVIRONMENT_FIELD}.{name}' must be a non-empty string")

    kind = block.get("kind")
    if isinstance(kind, str) and kind not in ENVIRONMENT_KINDS:
        errors.append(
            f"'{ENVIRONMENT_FIELD}.kind' must be one of {ENVIRONMENT_KINDS}; got {kind!r}"
        )

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

    proxy = block.get("substrate_checkpoint")
    if proxy is not None:
        if not isinstance(proxy, str) or not proxy.strip():
            errors.append(
                f"'{ENVIRONMENT_FIELD}.substrate_checkpoint' must be a path to a "
                f"checkpoint directory the stock modeling path supports"
            )
        elif not Path(proxy).exists():
            errors.append(
                f"'{ENVIRONMENT_FIELD}.substrate_checkpoint' does not exist: {proxy}. "
                f"The substrate gate falls back to it only after the run's own "
                f"checkpoint has already failed, so a missing proxy turns the "
                f"longest step in preflight into two failures instead of one."
            )

    if kind == "session_spool":
        spool = block.get("spool")
        if not isinstance(spool, str) or not spool.strip():
            errors.append(f"'{ENVIRONMENT_FIELD}.spool' is required when kind is 'session_spool'")
        elif not Path(spool).is_dir():
            errors.append(
                f"'{ENVIRONMENT_FIELD}.spool' does not exist: {spool}. Start the compute "
                f"session first — an unwatched spool accepts work and never completes it."
            )


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


def _validate_contrast_env(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    contrast_env = entry.get("contrast_env")
    if not isinstance(contrast_env, dict) or not contrast_env:
        errors.append(
            f"'{where}.contrast_env' is required and must be a non-empty mapping: "
            f"a witness with no changed variable compares a run to itself"
        )
    elif not all(isinstance(v, (str, int, float, bool)) for v in contrast_env.values()):
        errors.append(f"'{where}.contrast_env' values must be scalars the shell can carry")


def _validate_witness_timeout(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    timeout = entry.get("timeout")
    if timeout is not None and not _positive_int(timeout):
        errors.append(f"'{where}.timeout' must be a whole number of seconds >= 1, got {timeout!r}")


def _validate_witness_workload(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    """Validate a token-comparing witness block's declared workload."""
    _validate_acceptance_workload(
        {k: v for k, v in entry.items() if k in ("prompts", "prompts_file", "max_tokens")},
        errors,
        where,
    )
    minimum = entry.get("min_generated_tokens")
    if minimum is not None and not _positive_int(minimum):
        errors.append(f"'{where}.min_generated_tokens' must be an integer >= 1")


def _validate_witness_batch_size(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    value = entry.get(WITNESS_BATCH_SIZE_KEY)
    if value is not None and not _positive_int(value):
        errors.append(
            f"'{where}.{WITNESS_BATCH_SIZE_KEY}' must be an integer >= 1 (1 gives both legs "
            f"the same schedule), got {value!r}"
        )


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _validate_legs_env(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    """Validate the optional ``legs_env``: shell variables, none of them the contrast."""
    legs_env = entry.get(WITNESS_LEGS_ENV_KEY)
    if legs_env is None:
        return
    key = f"{where}.{WITNESS_LEGS_ENV_KEY}"
    if not isinstance(legs_env, dict) or not legs_env:
        errors.append(f"'{key}' must be a non-empty mapping from variable to value")
        return
    for name, value in legs_env.items():
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            errors.append(f"'{key}' names {name!r}, which is not a shell variable name")
        if not isinstance(value, (str, int, float, bool)):
            errors.append(f"'{key}.{name}' must be a scalar the shell can carry")
    contrast_env = entry.get("contrast_env")
    if isinstance(contrast_env, dict):
        shared = sorted(str(name) for name in set(legs_env) & set(contrast_env))
        if shared:
            errors.append(
                f"'{key}' and '{where}.contrast_env' both set {', '.join(shared)}: a variable "
                f"leg B sets differently is the contrast, not something the legs share"
            )


def _validate_equivalence_witness(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    """Validate an equivalence witness block: leg B is the run's own launcher."""
    reject_unknown_keys(entry, EQUIVALENCE_WITNESS_KEYS, errors, where=where)
    command = entry.get("contrast_command")
    if not isinstance(command, str) or not command.strip():
        errors.append(
            f"'{where}.contrast_command' is required and must be a non-empty string: declare "
            f"leg B's launcher, which starts a context and a generation engine and hands the "
            f"state between them"
        )
    _validate_contrast_env(entry, errors, where)
    _validate_legs_env(entry, errors, where)
    transceiver = entry.get(WITNESS_TRANSCEIVER_KEY)
    if transceiver is not None and (
        not isinstance(transceiver, dict)
        or not transceiver
        or not all(isinstance(k, str) for k in transceiver)
    ):
        errors.append(
            f"'{where}.{WITNESS_TRANSCEIVER_KEY}' must be a non-empty mapping of "
            f"CacheTransceiverConfig fields (e.g. backend: NIXL)"
        )
    lengths = entry.get("prompt_tokens")
    if lengths is not None:
        if not isinstance(lengths, list) or not lengths:
            errors.append(f"'{where}.prompt_tokens' must be a non-empty list of integers")
        else:
            for value in lengths:
                if not _positive_int(value):
                    errors.append(
                        f"'{where}.prompt_tokens' entries must be integers >= 1, got {value!r}"
                    )
                elif value > MAX_WITNESS_PROMPT_WORDS:
                    errors.append(
                        f"'{where}.prompt_tokens' entry {value} is over "
                        f"{MAX_WITNESS_PROMPT_WORDS}; each prompt travels as one argv string, "
                        f"which Linux caps at 128 KiB"
                    )
    _validate_witness_timeout(entry, errors, where)
    _validate_witness_batch_size(entry, errors, where)
    _validate_witness_workload(entry, errors, where)


def _validate_replay_witness(
    feature: str, entry: Mapping[str, Any], errors: list[str], where: str
) -> None:
    """Validate a bounded-replay witness block: the workload is DECLARED here."""
    reject_unknown_keys(
        entry,
        (*REPLAY_WITNESS_REQUIRED[feature], *REPLAY_WITNESS_OPTIONAL[feature]),
        errors,
        where=where,
    )
    for key in REPLAY_WITNESS_REQUIRED[feature]:
        if entry.get(key) is None:
            errors.append(
                f"'{where}.{key}' is required: the {feature} witness sizes its workload "
                f"from the spec, never from the run it judges"
            )
    _validate_contrast_env(entry, errors, where)

    window = entry.get("window")
    if window is not None and not _positive_int(window):
        errors.append(f"'{where}.window' must be an integer >= 1, got {window!r}")
        window = None
    for key in ("prompt_tokens", "prefix_tokens"):
        value = entry.get(key)
        if value is None or key not in REPLAY_WITNESS_REQUIRED[feature]:
            continue
        if not _positive_int(value):
            errors.append(f"'{where}.{key}' must be an integer >= 1, got {value!r}")
        elif window is not None and value <= window:
            errors.append(
                f"'{where}.{key}' is {value}, which does not exceed window={window}. At or "
                f"under the window, replay on and replay off do the same work, and the "
                f"witness passes for a target that implements nothing."
            )
        elif value > MAX_WITNESS_PROMPT_WORDS:
            errors.append(
                f"'{where}.{key}' is {value}; the witness prompt travels as one argv "
                f"string, which Linux caps at 128 KiB, so the limit is "
                f"{MAX_WITNESS_PROMPT_WORDS}"
            )
    alignment = entry.get("reuse_alignment")
    if alignment is not None and not _positive_int(alignment):
        errors.append(f"'{where}.reuse_alignment' must be an integer >= 1, got {alignment!r}")
    _validate_witness_timeout(entry, errors, where)


def _validate_kv_quant_witness(entry: dict[str, Any], errors: list[str], where: str) -> None:
    """Validate the ``kv_quant`` witness block, filling ``exact``'s default."""
    reject_unknown_keys(entry, KV_QUANT_WITNESS_KEYS, errors, where=where)
    expected = entry.get("expected_bytes_per_token")
    if not _positive_int(expected):
        errors.append(
            f"'{where}.expected_bytes_per_token' is required and must be an integer >= 1, "
            f"computed from the checkpoint's storage formats before the run; got {expected!r}"
        )
    exact = entry.setdefault("exact", True)
    if not isinstance(exact, bool):
        errors.append(f"'{where}.exact' must be true or false, got {exact!r}")
    elif exact or entry.get("contrast_env") is not None:
        _validate_contrast_env(entry, errors, where)
    tolerance = entry.get("bytes_tolerance")
    if tolerance is not None and (not _number(tolerance) or not 0 <= tolerance < 1):
        errors.append(f"'{where}.bytes_tolerance' must be a fraction in [0, 1), got {tolerance!r}")
    _validate_witness_timeout(entry, errors, where)
    _validate_witness_batch_size(entry, errors, where)
    _validate_witness_workload(entry, errors, where)


def load_prompts_file(path: str | Path) -> tuple[str, ...]:
    """Read a prompt list for the acceptance gate's declared workload.

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


def _validate_acceptance_workload(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    """Validate the workload keys one level of ``accuracy.acceptance`` declares."""
    prompts = entry.get("prompts")
    prompts_file = entry.get("prompts_file")
    if prompts is not None and prompts_file is not None:
        errors.append(
            f"'{where}' declares both 'prompts' and 'prompts_file'; keep one, so the "
            f"workload a verdict was measured on has a single source"
        )
    if prompts is not None and (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) and p.strip() for p in prompts)
    ):
        errors.append(f"'{where}.prompts' must be a non-empty list of non-empty strings")
    if prompts_file is not None:
        if not isinstance(prompts_file, str) or not prompts_file.strip():
            errors.append(f"'{where}.prompts_file' must be a non-empty path")
        elif not Path(prompts_file).is_file():
            errors.append(f"'{where}.prompts_file' does not exist: {prompts_file}")
        else:
            try:
                load_prompts_file(prompts_file)
            except TaskSchemaError as exc:
                errors.append(f"'{where}.prompts_file': {exc}")
    sources = [k for k in ("prompts", "prompts_file", "dataset") if entry.get(k) is not None]
    if "dataset" in sources and len(sources) > 1:
        errors.append(
            f"'{where}' declares 'dataset' beside {sources[0]!r}; keep one, so the workload "
            f"a verdict was measured on has a single source"
        )
    dataset = entry.get("dataset")
    if dataset is not None and (not isinstance(dataset, str) or not dataset.strip()):
        errors.append(
            f"'{where}.dataset' must be a local path or a cached dataset id (no download)"
        )
    _validate_fields(
        entry,
        dict.fromkeys(("text_field", "prompt_template", "split", "dataset_config"), _TEXT),
        errors,
        where,
    )
    template = entry.get("prompt_template")
    if isinstance(template, str) and "{text}" not in template:
        errors.append(f"'{where}.prompt_template' must name the question as {{text}}")
    _validate_fields(
        entry,
        dict.fromkeys(
            ("max_tokens", "min_draft_iterations", "num_prompts", "min_prompts"), _POS_INT
        ),
        errors,
        where,
    )


def _validate_acceptance_regression(block: Any, errors: list[str], where: str) -> None:
    """Validate ``accuracy.acceptance.regression``."""
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping with keys {list(ACCEPTANCE_REGRESSION_KEYS)}")
        return
    reject_unknown_keys(block, ACCEPTANCE_REGRESSION_KEYS, errors, where=where)
    alpha = block.get("alpha")
    if alpha is not None and (not _number(alpha) or not 0 < alpha < 1):
        errors.append(f"'{where}.alpha' must be a probability in (0, 1), got {alpha!r}")
    effect = block.get("min_effect")
    if effect is not None and (not _number(effect) or effect < 0):
        errors.append(f"'{where}.min_effect' must be a number >= 0, got {effect!r}")


def _validate_acceptance(block: Any, errors: list[str]) -> None:
    """Validate ``accuracy.acceptance``: per-unit floors plus the gate's workload.

    Every key that is not a workload key names a unit, and its value is the
    unit's floor -- a number, or a mapping with ``min_absolute`` -- optionally
    with that unit's own workload beside it.
    """
    where = f"{ACCURACY_FIELD}.acceptance"
    if not isinstance(block, dict):
        errors.append(
            f"'{where}' must be a mapping from unit to floor, plus the optional "
            f"workload keys {list(ACCEPTANCE_WORKLOAD_KEYS)}"
        )
        return
    _validate_acceptance_workload(block, errors, where)
    if block.get("regression") is not None:
        _validate_acceptance_regression(block["regression"], errors, f"{where}.regression")
    for unit, entry in block.items():
        if unit in ACCEPTANCE_BLOCK_KEYS:
            continue
        unit_where = f"{where}.{unit}"
        if isinstance(entry, dict):
            reject_unknown_keys(entry, ACCEPTANCE_UNIT_KEYS, errors, where=unit_where)
            _validate_acceptance_workload(entry, errors, unit_where)
            floor = entry.get("min_absolute")
            floor_where = f"{unit_where}.min_absolute"
            measured = entry.get("measured_reference")
            if measured is not None and (not _number(measured) or measured < 1.0):
                errors.append(
                    f"'{unit_where}.measured_reference' must be a number >= 1.0; got {measured!r}"
                )
            source = entry.get("floor_source")
            if source is not None and (not isinstance(source, str) or not source.strip()):
                errors.append(f"'{unit_where}.floor_source' must be a non-empty string")
        else:
            floor = entry
            floor_where = unit_where
        if floor is None:
            continue
        # Acceptance length counts the target's own token on every step, so
        # it is at least 1.0 by construction and a floor below that passes
        # everything.
        if not _number(floor) or floor < 1.0:
            errors.append(f"'{floor_where}' must be a number >= 1.0; got {floor!r}")


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
    _validate_fields(level, {"keywords": _STRINGS, "max_tokens": _POS_INT}, errors, where)
    greedy = level.get("greedy")
    if greedy is not None:
        at = f"{where}.greedy"
        if not isinstance(greedy, dict):
            errors.append(f"'{at}' must be a mapping")
            return
        reject_unknown_keys(greedy, GREEDY_KEYS, errors, where=at)
        _validate_fields(
            greedy, {"enabled": _FLAG, "prompts": _POS_INT, "tokens": _POS_INT}, errors, at
        )


def _validate_boot(block: Any, errors: list[str]) -> None:
    """Validate ``accuracy.boot``: the run-wide keys, then each unit's entry."""
    where = f"{ACCURACY_FIELD}.boot"
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping")
        return
    _validate_boot_level(block, errors, where)
    for unit, entry in block.items():
        if unit in BOOT_KEYS:
            continue
        if not isinstance(entry, dict):
            errors.append(
                f"'{where}.{unit}' is neither a boot key ({list(BOOT_KEYS)}) nor a unit's "
                f"mapping of them"
            )
            continue
        reject_unknown_keys(entry, BOOT_KEYS, errors, where=f"{where}.{unit}")
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


def anchor_pending(accuracy: Mapping[str, Any], unit: str | None) -> bool:
    """Whether ``unit`` has no written accuracy anchor yet (``accuracy.anchor: pending``)."""
    return resolve_per_unit(accuracy, "anchor", unit) == ANCHOR_PENDING


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


def _validate_envelope(block: Any, errors: list[str]) -> None:
    """Validate ``accuracy.envelope``."""
    where = f"{ACCURACY_FIELD}.envelope"
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping")
        return
    reject_unknown_keys(block, ENVELOPE_KEYS, errors, where=where)
    _validate_fields(
        block,
        {
            "enabled": _FLAG,
            "max_tokens": _POS_INT,
            "agree_tokens": _POS_INT,
            "timeout": _POS_INT,
            "long_prompts": (
                lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0,
                "must be an integer >= 0",
            ),
            "probes": (
                lambda v: bool(v) and _nonempty_strings(v),
                "must be a non-empty list of prompts",
            ),
        },
        errors,
        where,
    )
    max_tokens = block.get("max_tokens") or 16
    agree_tokens = block.get("agree_tokens") or 4
    if _positive_int(max_tokens) and _positive_int(agree_tokens) and agree_tokens > max_tokens:
        errors.append(f"'{where}.agree_tokens' must not exceed max_tokens")


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


# Logit agreement compares against the recorded HF reference, not the run's output.
LOGITS_KEYS: tuple[str, ...] = ("enabled", "reference", "prompts", "threshold_factor", "timeout")
#: The reference a spec that names none gets, when the file is beside the spec.
DEFAULT_LOGIT_REFERENCE = "logits.reference.json.gz"


def _validate_logits(block: Any, errors: list[str]) -> None:
    """Validate ``accuracy.logits``."""
    where = f"{ACCURACY_FIELD}.logits"
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping")
        return
    reject_unknown_keys(block, LOGITS_KEYS, errors, where=where)
    enabled = block.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        errors.append(f"'{where}.enabled' must be true or false")
    reference = block.get("reference")
    if reference is not None and (not isinstance(reference, str) or not reference.strip()):
        errors.append(f"'{where}.reference' must be a path")
    elif reference is None and enabled is not False:
        errors.append(
            f"'{where}.reference' names the reference file "
            f"(checks/logit_reference.py writes it); without one set enabled: false"
        )
    elif reference is not None and enabled is not False and not Path(reference).is_file():
        errors.append(f"'{where}.reference' does not exist: {reference}")
    prompts = block.get("prompts")
    if prompts is not None and (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) and p.strip() for p in prompts)
        or len(set(prompts)) != len(prompts)
    ):
        errors.append(f"'{where}.prompts' must be a non-empty list of distinct prompt ids")
    factor = block.get("threshold_factor")
    if factor is not None and (not _number(factor) or not factor > 0):
        errors.append(f"'{where}.threshold_factor' must be a number > 0")
    if block.get("timeout") is not None and not _positive_int(block["timeout"]):
        errors.append(f"'{where}.timeout' must be an integer >= 1")


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
    """The switches the run's ``profile`` sets where ``accuracy`` leaves them unsaid, in place.

    ``certified`` turns on boot boundary phases, the serve gate and the witnesses,
    and runs each unit's benchmark at every member; ``bringup`` and ``smoke``
    turn boot boundary phases and the serve gate off. ``standard`` sets nothing.
    """
    if profile not in (PROFILE_SMOKE, PROFILE_BRINGUP, PROFILE_CERTIFIED):
        return
    on = profile == PROFILE_CERTIFIED
    for key in ("envelope", "serve"):
        block = accuracy.get(key)
        if block is None:
            accuracy[key] = {"enabled": on}
        elif isinstance(block, dict):
            block.setdefault("enabled", on)
    if on:
        witness = accuracy.get("witness")
        if witness is None:
            accuracy["witness"] = {WITNESS_ENABLED_KEY: True}
        elif isinstance(witness, dict):
            witness.setdefault(WITNESS_ENABLED_KEY, True)
        accuracy.setdefault("benchmark_members", BENCHMARK_MEMBERS_EVERY)


def derive_gate_defaults(data: dict[str, Any], spec_dir: Path) -> None:
    """Turn on, in place, the gates the spec leaves unsaid and the run can derive.

    - ``accuracy.logits``: a block that names no reference and does not turn
      the gate off gets :data:`DEFAULT_LOGIT_REFERENCE` in ``spec_dir`` when
      that file is there;
    - ``accuracy.serve``: a block without ``enabled`` gets it, true wherever
      the spec bounds the server's sequence length (:func:`_bounds_server_length`).

    A spec that says ``enabled: false`` keeps either off. A ``profile`` sets its
    switches first (:func:`_profile_defaults`).
    """
    accuracy = data.get(ACCURACY_FIELD)
    if not isinstance(accuracy, dict):
        return
    _profile_defaults(profile_of(data), accuracy)
    logits = accuracy.get("logits")
    default = spec_dir / DEFAULT_LOGIT_REFERENCE
    unnamed = isinstance(logits, dict) and logits.get("reference") is None
    if (logits is None or (unnamed and logits.get("enabled") is not False)) and default.is_file():
        accuracy["logits"] = {**(logits or {}), "reference": str(default.resolve())}
    serve = accuracy.get("serve")
    if serve is None or (isinstance(serve, dict) and "enabled" not in serve):
        accuracy["serve"] = {**(serve or {}), "enabled": _bounds_server_length(accuracy)}


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
        block, dict.fromkeys(("max_seq_len", "max_batch_size"), _POS_INT), errors, ACCURACY_FIELD
    )

    if block.get("acceptance") is not None:
        _validate_acceptance(block["acceptance"], errors)

    _validate_floor(block, errors)

    if block.get("boot") is not None:
        _validate_boot(block["boot"], errors)

    if block.get("envelope") is not None:
        _validate_envelope(block["envelope"], errors)

    if block.get("logits") is not None:
        _validate_logits(block["logits"], errors)

    if block.get("serve") is not None:
        _validate_serve(block["serve"], errors)

    anchor = block.get("anchor")
    for where, state in (
        anchor.items() if isinstance(anchor, dict) else [("", anchor)] if anchor is not None else []
    ):
        if state not in ANCHOR_STATES:
            errors.append(
                f"'{ACCURACY_FIELD}.anchor{'.' + str(where) if where else ''}' must be one of "
                f"{list(ANCHOR_STATES)}; got {state!r}"
            )

    members = block.get("benchmark_members")
    named = isinstance(members, str) and PARALLEL_PATTERN.match(members)
    if members is not None and members not in BENCHMARK_MEMBERS and not named:
        errors.append(
            f"'{ACCURACY_FIELD}.benchmark_members' must be one of {list(BENCHMARK_MEMBERS)} "
            f"or a member (tp4, dep4, ...); got {members!r}"
        )

    witness = block.get("witness")
    if witness is not None:
        if not isinstance(witness, dict):
            errors.append(f"'{ACCURACY_FIELD}.witness' must be a mapping from unit to a block")
        else:
            for unit, entry in witness.items():
                where = f"{ACCURACY_FIELD}.witness.{unit}"
                if unit == WITNESS_ENABLED_KEY:
                    if not isinstance(entry, bool):
                        errors.append(f"'{where}' must be true or false")
                    continue
                if not isinstance(entry, dict):
                    errors.append(f"'{where}' must be a mapping")
                    continue
                # Closed per feature.
                if unit in REPLAY_WITNESS_REQUIRED:
                    _validate_replay_witness(str(unit), entry, errors, where)
                    continue
                if unit == "kv_quant":
                    _validate_kv_quant_witness(entry, errors, where)
                    continue
                if unit in EQUIVALENCE_WITNESS_FEATURES:
                    _validate_equivalence_witness(entry, errors, where)
                    continue
                reject_unknown_keys(entry, WITNESS_KEYS, errors, where=where)
                _validate_witness_timeout(entry, errors, where)
                _validate_witness_batch_size(entry, errors, where)
                _validate_witness_workload(entry, errors, where)
                command = entry.get("contrast_command")
                if command is not None and (not isinstance(command, str) or not command.strip()):
                    errors.append(f"'{where}.contrast_command' must be a non-empty string")
                _validate_contrast_env(entry, errors, where)
                predicted = entry.get("expected_mib")
                values = predicted.values() if isinstance(predicted, dict) else [predicted]
                if predicted is not None and (
                    not all(_number(v) and v > 0 for v in values)
                    or (isinstance(predicted, dict) and not predicted)
                ):
                    errors.append(
                        f"'{where}.expected_mib' must be a positive number, or a mapping from "
                        f"member (and `default`) to one"
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


def witness_enabled(task_data: Mapping[str, Any]) -> bool:
    """Whether the contract owes the witnesses (``accuracy.witness.enabled``); off by default."""
    accuracy = task_data.get(ACCURACY_FIELD)
    witness = accuracy.get("witness") if isinstance(accuracy, Mapping) else None
    return isinstance(witness, Mapping) and witness.get(WITNESS_ENABLED_KEY) is True


def witness_owed(task_data: Mapping[str, Any], unit: str) -> bool:
    """Whether the contract owes ``unit``'s witness.

    Always for a feature the equivalence witness proves (``disagg``): its boot
    and accuracy run an aggregated engine, so the witness is its only proof.
    Otherwise when the witnesses are on, or the unit's profile is ``certified``.
    """
    return (
        unit in EQUIVALENCE_WITNESS_FEATURES
        or witness_enabled(task_data)
        or profile_of(task_data, unit) == PROFILE_CERTIFIED
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


def _require_declared_witnesses(data: Mapping[str, Any], errors: list[str]) -> None:
    """A feature whose owed witness judges against a DECLARED value needs it here."""
    features = data.get(FEATURES_FIELD)
    accuracy = data.get(ACCURACY_FIELD)
    if not isinstance(features, list) or not isinstance(accuracy, dict):
        return
    witness = accuracy.get("witness")
    witness = witness if isinstance(witness, dict) else {}
    for feature in features:
        if feature not in DECLARED_WITNESS_FEATURES or feature in witness:
            continue
        if not witness_owed(data, feature):
            continue
        if feature in EQUIVALENCE_WITNESS_FEATURES:
            keys = EQUIVALENCE_WITNESS_REQUIRED
        else:
            keys = REPLAY_WITNESS_REQUIRED.get(feature) or ("expected_bytes_per_token", "exact")
        errors.append(
            f"feature {feature!r} needs 'accuracy.witness.{feature}' ({', '.join(keys)}): "
            f"a unit that implements nothing passes both boot and accuracy, so the "
            f"witness is the only gate that can tell, and what it judges against has to "
            f"be declared before the run."
        )


def _beside_spec(command: str, spec_dir: Path) -> str:
    """``command`` with each word naming a file in ``spec_dir`` made absolute; else unchanged."""
    try:
        words = shlex.split(command)
    except ValueError:
        return command
    moved = [
        str((spec_dir / word).resolve())
        if word and not Path(word).is_absolute() and (spec_dir / word).is_file()
        else word
        for word in words
    ]
    return shlex.join(moved) if moved != words else command


def _resolve_spec_relative_paths(data: dict[str, Any], spec_dir: Path) -> None:
    """Make paths the user writes NEXT TO the spec portable, in place.

    ``accuracy.llm_api_options``, the witness and acceptance prompt files, boot
    images and the logit reference are resolved against the **spec's own
    directory** and written back as absolute; of a witness's
    ``contrast_command``, each word naming a file the spec's directory holds is
    made absolute. Absolute paths are untouched, so a workspace copy of an
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

    witness = accuracy.get("witness")
    if isinstance(witness, dict):
        for block in witness.values():
            if isinstance(block, dict) and block.get("prompts_file") is not None:
                block["prompts_file"] = absolute(block["prompts_file"])
            if isinstance(block, dict) and isinstance(block.get("contrast_command"), str):
                block["contrast_command"] = _beside_spec(block["contrast_command"], spec_dir)

    # A `dataset` may be a cached dataset id ("openai/gsm8k"), which reads like
    # a relative path, so it is resolved only when the spec's directory holds it.
    acceptance = accuracy.get("acceptance")
    if isinstance(acceptance, dict):
        for level in (acceptance, *(v for v in acceptance.values() if isinstance(v, dict))):
            if level.get("prompts_file") is not None:
                level["prompts_file"] = absolute(level["prompts_file"])
            dataset = level.get("dataset")
            if isinstance(dataset, str) and dataset.strip() and not Path(dataset).is_absolute():
                if (spec_dir / dataset).exists():
                    level["dataset"] = absolute(dataset)

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

    logits = accuracy.get("logits")
    if isinstance(logits, dict) and logits.get("reference") is not None:
        logits["reference"] = absolute(logits["reference"])


def load_and_validate_task_yaml(path: str | Path) -> dict[str, Any]:
    """Parse ``path`` as YAML, validate it, and fill in the workflow's defaults.

    Returns the resolved mapping. Raises :class:`TaskSchemaError` with
    **every** detected problem batched into a single message.
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

    # BEFORE validation, deliberately. `accuracy.llm_api_options` has an
    # existence check, and that check is a *consumer* of the resolved path.
    _resolve_spec_relative_paths(data, task_path.resolve().parent)
    # After the paths, before validation: what it derives is validated like
    # what the spec says.
    derive_gate_defaults(data, task_path.resolve().parent)

    errors: list[str] = []

    validate_commit_policy_block(data.get(COMMIT_POLICY_FIELD), errors)

    errors.extend(validate_perf_block(data))
    reject_unknown_keys(data, TOP_LEVEL_KEYS, errors)
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

    if RUNTIME_FIELD not in data:
        data[RUNTIME_FIELD] = dict(RUNTIME_SWITCHES) | {"extra_flags": []}
    else:
        try:
            _validate_runtime(data[RUNTIME_FIELD], errors)
        except TaskSchemaError as exc:
            errors.append(str(exc))

    notes = data.get(NOTES_FIELD)
    if notes is not None and (not isinstance(notes, str) or not notes.strip()):
        errors.append(f"'{NOTES_FIELD}' must be a non-empty string")

    scope = data.get(SCOPE_FIELD)
    if scope is None:
        data[SCOPE_FIELD] = DEFAULT_SCOPE
    elif scope not in SCOPES:
        errors.append(f"'{SCOPE_FIELD}' must be one of {SCOPES}; got {scope!r}")

    if SOURCE_EMBARGO_FIELD not in data:
        data[SOURCE_EMBARGO_FIELD] = []
    else:
        try:
            _validate_source_embargo(
                data[SOURCE_EMBARGO_FIELD], errors, data.get("trtllm_repo_path")
            )
        except TaskSchemaError as exc:
            errors.append(str(exc))

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
        _require_declared_witnesses(data, errors)

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
    "ANCHOR_PENDING",
    "ANCHOR_RECORDED",
    "ANCHOR_STATES",
    "anchor_pending",
    "COVERAGE_FIELD",
    "COVERAGE_KEYS",
    "resolve_per_unit",
    "resolve_samples",
    "BENCHMARK_MEMBERS",
    "BENCHMARK_MEMBERS_EVERY",
    "BENCHMARK_MEMBERS_SAMPLED",
    "DEFAULT_LOGIT_REFERENCE",
    "GREEDY_KEYS",
    "derive_gate_defaults",
    "ACCEPTANCE_UNIT_KEYS",
    "ACCEPTANCE_WORKLOAD_KEYS",
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
    "DECLARED_WITNESS_FEATURES",
    "WITNESS_ENABLED_KEY",
    "DEFAULT_BYTES_TOLERANCE",
    "FEATURES_FIELD",
    "FEATURE_FAMILIES",
    "SPECULATORS",
    "PROFILES",
    "PROFILE_BRINGUP",
    "PROFILE_CERTIFIED",
    "PROFILE_FIELD",
    "PROFILE_SMOKE",
    "PROFILE_STANDARD",
    "profile_of",
    "KV_QUANT_WITNESS_KEYS",
    "LOGITS_KEYS",
    "MAX_WITNESS_PROMPT_WORDS",
    "MEMBERS_FIELD",
    "MTP_NAMESPACE_FEATURES",
    "REPLAY_WITNESS_OPTIONAL",
    "REPLAY_WITNESS_REQUIRED",
    "EQUIVALENCE_WITNESS_FEATURES",
    "EQUIVALENCE_WITNESS_KEYS",
    "EQUIVALENCE_WITNESS_REQUIRED",
    "WITNESS_BATCH_SIZE_KEY",
    "WITNESS_KEYS",
    "WITNESS_LEGS_ENV_KEY",
    "WITNESS_TRANSCEIVER_KEY",
    "PARALLEL_FIELD",
    "PARALLEL_UNIFIED",
    "REQUIRED_PATH_FIELDS",
    "RUNTIME_FIELD",
    "RUNTIME_SWITCHES",
    "SCOPES",
    "SCOPE_FIELD",
    "SERVE_KEYS",
    "reject_unknown_keys",
    "SOURCE_EMBARGO_FIELD",
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
    "widest_world_size",
    "witness_enabled",
    "witness_owed",
]
