# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema validation for trtllm-modeling-v2's ``task.yaml`` input.

The spec says which checkpoint, tree and architecture, how commands run
(``environment``), who works on it (``agents``), what judges accuracy by default
(``accuracy``) and what the run brings up: ``deployments``, the one way to say it
(``spec/deployments.py``). ``arch`` is **required and never inferred**: catalog receipts
are per-architecture. :func:`load_and_validate_task_yaml` batches every detected problem
into one message and fills in the workflow's defaults, so the resolved spec copied into
the workspace is fully explicit.

Only a spec that cannot be read as one stops a run here: a schema error, or options that
cannot resolve. What a spec is likely to cost or miss is advice (``spec/advice.py``),
printed by ``intake derive`` and ``--dry-run``.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from agent_flow.agent_runtime import AGENTS_FIELD, validate_agents

from ..agents.roles import ROLES
from ..checks import accuracy_policy
from ..infra.compute import pool as pool_mod
from ..infra.compute import slurm as slurm_mod
from ..infra.compute.environment import EnvironmentError, check_tmpdir
from ..infra.git import COMMIT_POLICY_FIELD, validate_commit_policy_block
from . import deployments as deployments_mod
from .deployments import DEPLOYMENTS_FIELD
from .schema_keys import reject_unknown_keys

REQUIRED_PATH_FIELDS: tuple[str, ...] = (
    "checkpoint_path",
    "trtllm_repo_path",
)

ARCH_FIELD = "arch"
ENVIRONMENT_FIELD = "environment"
ACCURACY_FIELD = "accuracy"

# ``sm_100`` (B200), ``sm_103`` (GB300), ... Matched rather than
# enumerated: preflight asserts the value against the host.
ARCH_PATTERN = re.compile(r"^sm_\d+$")

# A deployment's shape (``Deployment.shape``): ``tp4`` / ``tep4`` / ``dep4`` for one
# engine, ``pd6`` for a disaggregated deployment, each with the devices it holds in the
# trailing number. ``unified`` is the name of a target that serves every layout its
# route admits: the default last part of the target's directory.
PARALLEL_FIELD = "parallel"
PARALLEL_UNIFIED = "unified"
DEFAULT_PARALLEL = PARALLEL_UNIFIED

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
ACCEPTANCE_BLOCK_KEYS: tuple[str, ...] = (*ACCEPTANCE_WORKLOAD_KEYS, "timeout")
# What one deployment's entry in `accuracy.acceptance` may carry:
#   min_absolute        the absolute floor its drafter's acceptance is held to
#   measured_reference  what a reference drafter measured on this workload; the
#                       floor is this x 0.95 when `min_absolute` is not given
#   floor_source        where that measurement was taken. A floor with neither
#                       a measured reference nor a source is used, and every
#                       verdict says it is not tied to a measurement
ACCEPTANCE_UNIT_KEYS: tuple[str, ...] = (
    "min_absolute",
    "measured_reference",
    "floor_source",
    "timeout",
    *ACCEPTANCE_WORKLOAD_KEYS,
)

TARGET_FIELD = "target"
NOTES_FIELD = "notes"

TOP_LEVEL_KEYS: tuple[str, ...] = (
    *REQUIRED_PATH_FIELDS,
    # The onboarded ref this spec adds to (a feature on an onboarded model): its survey
    # writes the delta rows only and no deployment waits for a first one (run/execution/tasks.py).
    "onboarded",
    ARCH_FIELD,
    ENVIRONMENT_FIELD,
    ACCURACY_FIELD,
    TARGET_FIELD,
    NOTES_FIELD,
    AGENTS_FIELD,
    COMMIT_POLICY_FIELD,
    # What the run brings up, each as the native options its engine takes.
    DEPLOYMENTS_FIELD,
)

#: The parts of the target's directory: ``models/<family>/<checkpoint_segment>__<arch>__<parallel>``.
TARGET_NAME_KEYS: tuple[str, ...] = ("family", "checkpoint_segment", PARALLEL_FIELD)
#: Beside them: whether the target is held to the guide's import rule (it reads and may
#: copy Modeling V1 code, and does not import it). ``refused`` unless the spec says
#: ``allowed``: a requirement, changed like any other (``checks/delivery.py``).
TARGET_KEYS: tuple[str, ...] = (*TARGET_NAME_KEYS, "v1_imports")

ACCURACY_KEYS: tuple[str, ...] = (
    # The benchmark of a deployment that lists none, and what every benchmark takes
    # unless its own entry says otherwise (`deployments.BENCHMARK_DEFAULTS`).
    "benchmark",
    *deployments_mod.BENCHMARK_DEFAULTS,
    # The floor under a score (checks/accuracy_policy.py), and the points of its
    # `tolerance` rule.
    "floor",
    "tolerance",
    # Seconds one benchmark may take.
    "timeout",
    # The boot workload: run-wide keys, and an entry of the same keys per deployment.
    "boot",
    # A drafter's acceptance workload, and each deployment's floor for it.
    "acceptance",
    # Whether the final tests run the engine's accuracy tests they select (each case a
    # load and a whole benchmark) or collect them and leave them to CI, which is the
    # default: the deployment's session measured the benchmark.
)

# Boot workloads allow run-wide keys and per-deployment overlays. Keywords match
# generated text, never logs; absent images use the repository asset.
BOOT_KEYS: tuple[str, ...] = ("prompts", "prompts_file", "keywords", "max_tokens")
#: Tokens a boot prompt may generate when the spec states none, and what `intake derive`
#: writes: a model that says a few words before its number is cut short at 32.
DEFAULT_BOOT_MAX_TOKENS = 64
# `timeout` at either level of `accuracy.boot` and `accuracy.acceptance`: read by no
# check of a session (a deployment's own `timeout` bounds it), kept so a spec that
# states one still loads.
GATE_TIMEOUT_KEY = "timeout"
BOOT_PROMPT_KEYS: tuple[str, ...] = ("prompt", "keywords", "keywords_any", "images", "render")
BOOT_RENDERS: tuple[str, ...] = ("loader", "raw")

# Floor computation and systematic-shift findings are owned by checks/stats.py.
FLOOR_KEYS: tuple[str, ...] = ("rule", "k", "shift_min_members", "acknowledge")

ENVIRONMENT_KEYS: tuple[str, ...] = (
    "kind",
    "python",
    "env",
    "spool",
    # The image the GPU jobs run in, if any, and how host paths appear inside it
    # (`host:inside`).
    "container_image",
    "mounts",
    "gpus_per_node",
    # Each device's memory, GiB: the advice sizes engines by it.
    "gpu_memory_gib",
    # Slurm, the default environment: the values a holder is submitted with. Site facts,
    # from the user's spec or a site file (`infra/compute/slurm.py` says which keys).
    "slurm",
    # Where the run's trees go (`run/workspace/integration.py`): a place other than each
    # node's own disk, and how long an unused pin's tree is kept.
    "trees",
)

#: A spec that names no environment is Slurm (`pool.environment_kind`).
ENVIRONMENT_KINDS: tuple[str, ...] = pool_mod.KINDS
ENVIRONMENT_REQUIRED_FIELDS: tuple[str, ...] = ("python",)

# Accuracy tolerance, in benchmark points, under `accuracy.floor.rule:
# tolerance` only.
DEFAULT_ACCURACY_TOLERANCE = 5.0


class TaskSchemaError(ValueError):
    """Raised when ``task.yaml`` fails trtllm-modeling-v2 schema validation."""


# ------------------------------------------------ what the run deploys, by unit and member
#
# The run's records, its contract and its planner say "unit" and "member". A unit is a
# deployment, by its name; its one member is the deployment's shape. These read both
# from the spec's deployments, for the modules that still use the older words.


def assembly_units(data: Mapping[str, Any]) -> list[str]:
    """The run's units in dispatch order: its deployments' names, the first one first."""
    return [d.name for d in deployments_mod.resolve(data)]


def widest_world_size(data: Mapping[str, Any], default: int = 1) -> int:
    """The most devices any deployment of a validated spec holds; ``default`` with none."""
    return max((d.ranks for d in deployments_mod.resolve(data)), default=0) or default


def target_parallel(data: Mapping[str, Any]) -> str:
    """The last part of the target's directory name: ``target.parallel``, else ``unified``."""
    return str((data.get(TARGET_FIELD) or {}).get(PARALLEL_FIELD) or DEFAULT_PARALLEL)


def derive_slug(data: Mapping[str, Any]) -> str:
    """Return ``<checkpoint>-<arch>-<parallel>`` for a validated spec.

    The checkpoint segment is the checkpoint directory's own name,
    lowercased with non-alphanumerics folded to hyphens.
    """
    checkpoint = Path(str(data["checkpoint_path"]).rstrip("/")).name
    checkpoint = re.sub(r"[^a-zA-Z0-9]+", "-", checkpoint).strip("-").lower()
    return f"{checkpoint}-{data[ARCH_FIELD]}-{target_parallel(data)}"


def derive_family(architecture: str) -> str:
    """``GptOssForCausalLM`` -> ``gpt_oss``: a family directory for an architecture no family routes.

    An architecture a family already routes takes that family's directory
    (``run.startup.preflight.routed_family``); this names a new family.
    """
    name = re.sub(
        r"For(?:(?:Causal|Conditional|Seq2Seq|Sequence)?LM|ConditionalGeneration)$",
        "",
        architecture,
    )
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
    if block.get("v1_imports") not in (None, "refused", "allowed"):
        errors.append(
            f"'{TARGET_FIELD}.v1_imports' is `refused` (the guide's rule, and what is taken "
            f"when it is unsaid) or `allowed`; got {block.get('v1_imports')!r}"
        )


def _validate_environment(block: Any, errors: list[str]) -> None:
    """Validate the ``environment`` block in place, appending to ``errors``."""
    if not isinstance(block, dict):
        raise TaskSchemaError(
            f"'{ENVIRONMENT_FIELD}' must be a mapping, got {type(block).__name__}"
        )

    kind = block.get("kind")
    if block.get("pool") is not None or (kind is not None and kind not in ENVIRONMENT_KINDS):
        # Neither Slurm nor one host: one message, the same wherever this is found.
        named = f"kind: {kind}" if kind is not None and kind not in ENVIRONMENT_KINDS else "pool"
        errors.append(
            f"'{ENVIRONMENT_FIELD}' names an environment the flow has no launcher for "
            f"(`{ENVIRONMENT_FIELD}.{named}`): {pool_mod.OTHER_ENVIRONMENT}"
        )
        return
    if "sessions" in block:
        errors.append(
            f"'{ENVIRONMENT_FIELD}.sessions' is no key any more: the flow asks for its own "
            f"holders (`{ENVIRONMENT_FIELD}.slurm`) or runs on this one host "
            f"(`{ENVIRONMENT_FIELD}.kind: local`), and serves no watcher somebody else starts"
        )
        return
    reject_unknown_keys(block, ENVIRONMENT_KEYS, errors, where=ENVIRONMENT_FIELD)

    for name in ENVIRONMENT_REQUIRED_FIELDS:
        value = block.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{ENVIRONMENT_FIELD}.{name}' must be a non-empty string")

    named = kind is not None or block.get("slurm") is not None
    if not named and not shutil.which("sbatch"):
        # One host that is not Slurm: nothing to state. A block that names no environment
        # could not run on Slurm anyway (no account, no partition), so where there is no
        # `sbatch` it is this host, as `intake derive` writes it. The kind goes into the spec
        # the run reads, and the start's report names it with the devices it claimed.
        block["kind"] = pool_mod.KIND_LOCAL
    kind = pool_mod.environment_kind(block)
    if kind == pool_mod.KIND_SLURM:
        if block.get("slurm") is None:
            errors.append(
                f"'{ENVIRONMENT_FIELD}' names no environment, so it is Slurm: state "
                f"`{ENVIRONMENT_FIELD}.slurm`, or `{ENVIRONMENT_FIELD}.kind: local` to use this "
                f"one host's devices"
            )
        slurm_mod.validate(block.get("slurm") or {}, errors, where=f"{ENVIRONMENT_FIELD}.slurm")

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

    from ..run.workspace.integration import TREES_FIELD, validate_trees_block

    validate_trees_block(block.get(TREES_FIELD), errors, where=f"{ENVIRONMENT_FIELD}.{TREES_FIELD}")

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
            ("max_tokens", "min_draft_iterations", "num_prompts", "min_prompts", GATE_TIMEOUT_KEY),
            _POS_INT,
        ),
        errors,
        where,
    )


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
            named = entry.get("reference")
            if named is not None and (not isinstance(named, str) or not named.strip()):
                errors.append(f"'{unit_where}.reference' must name a deployment")
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


def _validate_floor(block: Mapping[str, Any], errors: list[str]) -> None:
    """Validate ``accuracy.floor``: the rule a score's floor is computed by."""
    where = f"{ACCURACY_FIELD}.floor"
    floor = block.get("floor")
    if floor is None:
        return
    if not isinstance(floor, dict):
        errors.append(f"'{where}' must be a mapping with keys {list(FLOOR_KEYS)}")
        return
    reject_unknown_keys(floor, FLOOR_KEYS, errors, where=where)
    rule = floor.get("rule")
    if rule is not None and rule not in accuracy_policy.FLOOR_RULES:
        errors.append(f"'{where}.rule' must be one of {accuracy_policy.FLOOR_RULES}, got {rule!r}")
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
            f"'{where}.acknowledge' must map a benchmark to the reason the systematic shift "
            f"of its deployments is accepted, as a non-empty string"
        )


def _validate_accuracy(block: Any, errors: list[str]) -> None:
    """Validate the ``accuracy`` block in place, appending to ``errors``."""
    if not isinstance(block, dict):
        errors.append(f"'{ACCURACY_FIELD}' must be a mapping, got {type(block).__name__}")
        return
    for gone, now in deployments_mod.REPLACED_ACCURACY_KEYS.items():
        if gone in block:
            errors.append(f"'{ACCURACY_FIELD}.{gone}' is no longer read: {now}")
    known = {k: v for k, v in block.items() if k not in deployments_mod.REPLACED_ACCURACY_KEYS}
    reject_unknown_keys(known, ACCURACY_KEYS, errors, where=ACCURACY_FIELD)

    benchmark = block.get("benchmark")
    if benchmark is not None and not _text(benchmark):
        errors.append(
            f"'{ACCURACY_FIELD}.benchmark' must be one benchmark's name: the one a deployment "
            f"that lists none is scored on; a deployment's own go in its `benchmarks`"
        )
    deployments_mod.validate_judging(block, errors, ACCURACY_FIELD)

    tolerance = block.get("tolerance")
    if tolerance is None:
        block["tolerance"] = DEFAULT_ACCURACY_TOLERANCE
    elif not _number(tolerance):
        errors.append(f"'{ACCURACY_FIELD}.tolerance' must be a number")
    elif tolerance < 0:
        errors.append(f"'{ACCURACY_FIELD}.tolerance' must be >= 0; got {tolerance}")
    _validate_fields(block, {"timeout": _POS_INT}, errors, ACCURACY_FIELD)

    if block.get("acceptance") is not None:
        _validate_acceptance(block["acceptance"], errors)

    _validate_floor(block, errors)
    if block.get("boot") is not None:
        _validate_boot(block["boot"], errors)


def gpus_per_node(data: Mapping[str, Any]) -> int | None:
    """The most devices one node of the run holds, as ``environment`` states it."""
    env = data.get(ENVIRONMENT_FIELD) or {}
    slurm = env.get("slurm") if isinstance(env.get("slurm"), Mapping) else {}
    sizes = [env.get("gpus_per_node"), slurm.get("gpus_per_node")]
    sizes = [s for s in sizes if isinstance(s, int) and not isinstance(s, bool) and s > 0]
    return max(sizes) if sizes else None


def _resolve_spec_relative_paths(data: dict[str, Any], spec_dir: Path) -> None:
    """Make paths the user writes NEXT TO the spec portable, in place.

    Each deployment's options file, the acceptance prompt files and boot images
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

    deployments_mod.resolve_paths(data, spec_dir)
    accuracy = data.get(ACCURACY_FIELD)
    if not isinstance(accuracy, dict):
        return

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
    # that does not is left relative: the session resolves it against the tree
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


def load_and_validate_task_yaml(path: str | Path) -> dict[str, Any]:
    """Parse ``path`` as YAML, validate it, and fill in the workflow's defaults.

    Returns the resolved mapping; ``spec.deployments.resolve`` reads what it deploys from
    it. Raises :class:`TaskSchemaError` with **every** detected problem batched into a
    single message.
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

    # BEFORE validation, deliberately: an options file's existence check is a
    # *consumer* of the resolved path.
    _resolve_spec_relative_paths(data, task_path.resolve().parent)

    errors: list[str] = []

    validate_commit_policy_block(data.get(COMMIT_POLICY_FIELD), errors)

    if "perf" in data:
        errors.append(
            "'perf' is not read: the flow has no tuning stage, and performance work runs "
            "outside it on the delivered target; drop the block"
        )
    # What said what runs before `deployments` did: one line each, with what replaced it.
    for gone, now in deployments_mod.REPLACED_KEYS.items():
        if gone in data:
            errors.append(f"'{gone}' is no longer read: {now}")
    unread = ("perf", *deployments_mod.REPLACED_KEYS)
    reject_unknown_keys({k: v for k, v in data.items() if k not in unread}, TOP_LEVEL_KEYS, errors)
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

    if data.get(DEPLOYMENTS_FIELD) is None:
        errors.append(
            f"missing required field '{DEPLOYMENTS_FIELD}': what the run brings up, each a "
            f"name and a native options file, the first one first "
            f"(`[{{name: tp1, options: engines/tp1.yaml}}]`); `trtllm-modeling-v2 intake "
            f'derive --checkpoint <dir> --repo <dir> --goal "<what to bring up>" --out '
            f"<dir>` writes them"
        )
    else:
        deployments_mod.validate(data[DEPLOYMENTS_FIELD], errors)

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

    # No `accuracy` block: every deployment is judged by its session's other checks.
    if data.get(ACCURACY_FIELD) is not None:
        _validate_accuracy(data[ACCURACY_FIELD], errors)

    errors = list(dict.fromkeys(errors))
    if errors:
        bullet = "\n  - "
        raise TaskSchemaError(
            f"{task_path} failed trtllm-modeling-v2 schema validation:{bullet}{bullet.join(errors)}"
        )

    return data


__all__ = [
    "ACCEPTANCE_UNIT_KEYS",
    "ACCEPTANCE_WORKLOAD_KEYS",
    "ACCURACY_FIELD",
    "ACCURACY_KEYS",
    "ARCH_FIELD",
    "ARCH_PATTERN",
    "DEFAULT_ACCURACY_TOLERANCE",
    "DEFAULT_BOOT_MAX_TOKENS",
    "DEFAULT_PARALLEL",
    "DEPLOYMENTS_FIELD",
    "ENVIRONMENT_FIELD",
    "ENVIRONMENT_KEYS",
    "ENVIRONMENT_KINDS",
    "PARALLEL_FIELD",
    "PARALLEL_UNIFIED",
    "REQUIRED_PATH_FIELDS",
    "TARGET_FIELD",
    "TARGET_KEYS",
    "TOP_LEVEL_KEYS",
    "TaskSchemaError",
    "assembly_units",
    "derive_checkpoint_segment",
    "derive_family",
    "derive_slug",
    "gpus_per_node",
    "load_and_validate_task_yaml",
    "load_prompts_file",
    "reject_unknown_keys",
    "resolve_per_unit",
    "target_parallel",
    "widest_world_size",
]
