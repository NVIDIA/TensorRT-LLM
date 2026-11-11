# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate trtllm-modeling-v2's ``task.yaml``.

The spec says which checkpoint, tree and architecture, how commands run
(``environment``), who works on it (``agents``), what judges accuracy by default
(``accuracy``) and what the run brings up: ``deployments``, or the shorthand keys that
expand into them. This module validates the top-level keys and hands the rest to
``spec/deployments.py`` and ``spec/shorthand.py``. Every problem is batched into one
:class:`TaskSchemaError`. A key an earlier version read and this one does not
(:data:`IGNORED_KEYS`) is dropped with one notice, never an error.
"""

from __future__ import annotations

import ast
import difflib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from agent_flow.agent_runtime import AGENTS_FIELD, validate_agents

from ..agents.roles import ROLES, unstated_models
from ..infra.compute.environment import EnvironmentError, check_tmpdir
from ..infra.git import COMMIT_POLICY_FIELD, validate_commit_policy_block
from ..operators.coverage import MODELING_V2_PACKAGE

REQUIRED_PATH_FIELDS: dict[str, str] = {
    "checkpoint_path": "the HuggingFace checkpoint directory",
    "trtllm_repo_path": "the TensorRT-LLM checkout the target is built in",
}
ARCH_FIELD = "arch"
ENVIRONMENT_FIELD = "environment"
ACCURACY_FIELD = "accuracy"
TARGET_FIELD = "target"
NOTES_FIELD = "notes"
#: ``sm_100`` (B200), ``sm_103`` (GB300), ... Matched, not enumerated.
ARCH_PATTERN = re.compile(r"^sm_\d+$")
#: The name segment of the target's directory when the spec gives none.
DEFAULT_PARALLEL = "unified"
SCOPES: tuple[str, ...] = ("full", "text_only")
TOP_LEVEL_KEYS: tuple[str, ...] = (
    *REQUIRED_PATH_FIELDS,
    ARCH_FIELD,
    ENVIRONMENT_FIELD,
    ACCURACY_FIELD,
    TARGET_FIELD,
    NOTES_FIELD,
    AGENTS_FIELD,
    COMMIT_POLICY_FIELD,
    # What the run brings up, as native options; or, below, the shorthand for them.
    "deployments",
    # A member name in shorthand, and the name segment of the target's directory.
    "parallel",
    "members",
    "features",
    "variants",
    "runtime",
    "scope",
)
TARGET_KEYS: tuple[str, ...] = ("family", "checkpoint_segment")
ENVIRONMENT_KEYS: tuple[str, ...] = (
    "kind",
    "python",
    # The site's prefix for a one-GPU command (an entry's test); a deployment has its own.
    "launch",
    "env",
    "container_image",
    "mounts",
    "gpus_per_node",
    "gpu_memory_gib",
)
ENVIRONMENT_KINDS: tuple[str, ...] = ("local",)
#: Keys of an executor the workflow no longer has: refused with why.
REMOVED_ENVIRONMENT_KEYS: tuple[str, ...] = (
    "spool",
    "sessions",
    "session_wait",
    "substrate_checkpoint",
)
#: The boot workload: run-wide keys, and an entry of the same keys per deployment name.
BOOT_KEYS: tuple[str, ...] = ("prompts", "prompts_file", "keywords", "max_tokens")
GATE_TIMEOUT_KEY = "timeout"
BOOT_PROMPT_KEYS: tuple[str, ...] = ("prompt", "keywords", "keywords_any", "images", "render")
BOOT_RENDERS: tuple[str, ...] = ("loader", "raw")
SERVE_KEYS: tuple[str, ...] = ("enabled", "prompt", "max_tokens", "timeout")

#: Keys an earlier version of the spec read and this one does not, by path (``*`` is any
#: key of a mapping), each with what holds instead. Dropped at load, one notice each.
IGNORED_KEYS: dict[str, str] = {
    "profile": "every deployment owes its boot and its benchmarks; there are no profiles",
    "coverage": "there is no coverage matrix; what runs is the deployments the spec comes to",
    "accuracy.envelope": "a boot runs its workload only; there are no boundary phases",
    "accuracy.reference_logits": "a boot keeps no divergence record against a reference's logits",
    "accuracy.acceptance": "a drafter's acceptance is judged by a native test, named in a "
    "deployment's `tests`; the flow measures no acceptance length of its own",
    "accuracy.floor": "the floor is the anchor less two standard errors",
    "accuracy.tolerance": "the floor is the anchor less two standard errors",
    "accuracy.benchmark_members": "every deployment is scored on its benchmarks",
    "accuracy.anchor": "an anchor is declared on a benchmark, read from a reference row, or "
    "another deployment's score (`relative_to`)",
    "accuracy.disagg.devices": "a disaggregated deployment takes its two servers' ranks",
    "accuracy.disagg.nodes": "where a server runs is its launch prefix's to say",
    "commit_policy.base_ref": "each task's commits are audited as they land; the branch is "
    "not audited as a whole",
    "commit_policy.allowed_paths": "no path is refused but the workspace and the judge's "
    "(the evaluator, the accuracy references); a landed commit is labelled target, shared "
    "or mixed",
    "agents.roles.planner": "there is no planner: the order of the tasks is fixed in code",
    "variants.*.gates": "every deployment boots and is scored on its benchmarks",
    "variants.*.members": "a variant is one deployment on the first member",
    "variants.*.accuracy.acceptance": "a drafter's acceptance is a native test in `tests`",
    "variants.*.accuracy.anchor": "an anchor is a reference row, else the first member's score",
}


class TaskSchemaError(ValueError):
    """Raised when ``task.yaml`` fails trtllm-modeling-v2 schema validation."""


def reject_unknown_keys(
    block: Mapping[str, Any], known: Sequence[str], errors: list[str], *, where: str = ""
) -> None:
    """Append an error for any key in ``block`` that ``known`` does not list.

    Every block is closed: a misspelled key that is silently ignored produces a run that
    checks a configuration nobody asked for. The nearest known key is suggested.
    """
    prefix = f"{where}." if where else ""
    for key in sorted(set(block) - set(known), key=str):
        close = difflib.get_close_matches(str(key), [str(k) for k in known], n=1, cutoff=0.6)
        hint = f" — did you mean '{prefix}{close[0]}'?" if close else "."
        errors.append(
            f"unknown key '{prefix}{key}'{hint} Known keys here: {sorted(str(k) for k in known)}"
        )


def missing(path: str, what: str, fix: str) -> str:
    """The one line a missing or wrong key gets: what it is, and what to write."""
    return f"missing {path}: {what}; fix: `{fix}`"


# -------------------------------------------------------------------------- the target


def derive_slug(data: Mapping[str, Any]) -> str:
    """``<checkpoint>-<arch>-<parallel>``: the run's name, from the checkpoint directory's."""
    checkpoint = Path(str(data["checkpoint_path"]).rstrip("/")).name
    checkpoint = re.sub(r"[^a-zA-Z0-9]+", "-", checkpoint).strip("-").lower()
    return f"{checkpoint}-{data[ARCH_FIELD]}-{data.get('parallel') or DEFAULT_PARALLEL}"


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
    """``DeepSeek-R1-0528-NVFP4`` -> ``deep_seek_r1_0528_nvfp4``; ``target.checkpoint_segment`` overrides."""
    name = Path(str(checkpoint_path).rstrip("/")).name
    # Only at a lowercase-to-capital boundary: a digit on the left would split `30B`.
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name)
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


# ------------------------------------------------------------------------- the blocks


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _strings(value: Any, *, nonempty: bool = False) -> bool:
    return isinstance(value, list) and all(
        isinstance(v, str) and (v.strip() or not nonempty) for v in value
    )


def _validate_target(block: Any, errors: list[str]) -> None:
    if not isinstance(block, dict):
        errors.append(f"'{TARGET_FIELD}' must be a mapping, got {type(block).__name__}")
        return
    reject_unknown_keys(block, TARGET_KEYS, errors, where=TARGET_FIELD)
    for name in TARGET_KEYS:
        value = block.get(name)
        if value is not None and not (isinstance(value, str) and re.match(r"^[a-z0-9_]+$", value)):
            errors.append(
                f"'{TARGET_FIELD}.{name}' is a Python package directory name: lowercase with "
                f"underscores; got {value!r}"
            )


def _validate_environment(block: Any, errors: list[str]) -> None:
    """Validate the ``environment`` block in place, filling ``kind`` and ``env``."""
    if not isinstance(block, dict):
        errors.append(
            missing(ENVIRONMENT_FIELD, "how commands run", "environment: {python: /path/to/python}")
        )
        return
    removed = [key for key in REMOVED_ENVIRONMENT_KEYS if key in block]
    if removed or block.get("kind") == "session_spool":
        named = (
            ", ".join(f"'{ENVIRONMENT_FIELD}.{key}'" for key in removed) or "kind: session_spool"
        )
        errors.append(
            f"{named}: the compute-session executor is no longer part of this workflow. Start "
            f"the orchestrator where the GPUs are, or give a deployment a `launch` prefix, and "
            f"drop these keys."
        )
    reject_unknown_keys(
        {k: v for k, v in block.items() if k not in REMOVED_ENVIRONMENT_KEYS},
        ENVIRONMENT_KEYS,
        errors,
        where=ENVIRONMENT_FIELD,
    )
    python = block.get("python")
    if not isinstance(python, str) or not python.strip():
        errors.append(
            missing(
                f"{ENVIRONMENT_FIELD}.python",
                "the interpreter every Python command runs under",
                "python: /path/to/TensorRT-LLM/.venv/bin/python",
            )
        )
    kind = block.setdefault("kind", ENVIRONMENT_KINDS[0])
    if kind not in (*ENVIRONMENT_KINDS, "session_spool"):
        errors.append(
            f"'{ENVIRONMENT_FIELD}.kind' must be one of {ENVIRONMENT_KINDS}; got {kind!r}"
        )
    launch = block.get("launch")
    if launch is not None and not isinstance(launch, str):
        errors.append(f"'{ENVIRONMENT_FIELD}.launch' must be a string: the site's launch prefix")
    env = block.setdefault("env", {})
    if not isinstance(env, dict):
        errors.append(f"'{ENVIRONMENT_FIELD}.env' must be a mapping, got {type(env).__name__}")
    else:
        for key, value in env.items():
            if not isinstance(value, (str, int, float)):
                errors.append(
                    f"'{ENVIRONMENT_FIELD}.env.{key}' must be a scalar the shell can carry"
                )
        try:
            check_tmpdir({str(k): str(v) for k, v in env.items()})
        except EnvironmentError as exc:
            errors.append(f"'{ENVIRONMENT_FIELD}.env': {exc}")
    mounts = block.get("mounts")
    if mounts is not None and not _strings(mounts):
        errors.append(f"'{ENVIRONMENT_FIELD}.mounts' must be a list of strings")
    memory = block.get("gpu_memory_gib")
    if memory is not None and not (
        isinstance(memory, (int, float)) and not isinstance(memory, bool) and memory > 0
    ):
        errors.append(f"'{ENVIRONMENT_FIELD}.gpu_memory_gib' must be a positive number of GiB")


def load_prompts_file(path: str | Path) -> tuple[str, ...]:
    """A boot workload's prompt list: a YAML or JSON list of strings, or a mapping with ``prompts``."""
    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TaskSchemaError(f"cannot read the prompts file {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TaskSchemaError(f"the prompts file {source} is not valid YAML/JSON: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("prompts")
    if not _strings(data, nonempty=True) or not data:
        raise TaskSchemaError(
            f"the prompts file {source} must hold a non-empty list of non-empty strings, or a "
            f"mapping with one under 'prompts'"
        )
    return tuple(data)


def _validate_boot_level(level: Mapping[str, Any], errors: list[str], where: str) -> None:
    prompts = level.get("prompts")
    if prompts is not None and (not isinstance(prompts, list) or not prompts):
        errors.append(f"'{where}.prompts' must be a non-empty list")
        prompts = None
    for n, entry in enumerate(prompts or ()):
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
        for key in ("keywords", "keywords_any", "images"):
            if entry.get(key) is not None and not _strings(entry[key], nonempty=True):
                errors.append(f"'{at}.{key}' must be a list of non-empty strings")
        if entry.get("render") is not None and entry["render"] not in BOOT_RENDERS:
            errors.append(f"'{at}.render' must be one of {BOOT_RENDERS}")
    prompts_file = level.get("prompts_file")
    if prompts_file is not None and not (
        isinstance(prompts_file, str) and Path(prompts_file).is_file()
    ):
        errors.append(f"'{where}.prompts_file' does not exist: {prompts_file}")
    if level.get("keywords") is not None and not _strings(level["keywords"]):
        errors.append(f"'{where}.keywords' must be a list of strings")
    for key in ("max_tokens", GATE_TIMEOUT_KEY):
        if level.get(key) is not None and not _positive_int(level[key]):
            errors.append(f"'{where}.{key}' must be an integer >= 1")


def _validate_boot(block: Any, errors: list[str]) -> None:
    """``accuracy.boot``: the run-wide keys, then an entry of the same keys per deployment."""
    where = f"{ACCURACY_FIELD}.boot"
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping")
        return
    _validate_boot_level(block, errors, where)
    for name, entry in block.items():
        if name in BOOT_KEYS or name == GATE_TIMEOUT_KEY:
            continue
        if not isinstance(entry, dict):
            errors.append(
                f"'{where}.{name}' is neither a boot key ({list(BOOT_KEYS)}) nor a "
                f"deployment's mapping of them"
            )
            continue
        reject_unknown_keys(entry, (*BOOT_KEYS, GATE_TIMEOUT_KEY), errors, where=f"{where}.{name}")
        _validate_boot_level(entry, errors, f"{where}.{name}")


def resolve_gate_timeout(accuracy: Mapping[str, Any], block: str, name: str | None) -> int | None:
    """``accuracy.<block>``'s ``timeout`` for deployment ``name``: its own, else the block's."""
    level = accuracy.get(block)
    if not isinstance(level, Mapping):
        return None
    for source in (level.get(name) if name is not None else None, level):
        if isinstance(source, Mapping) and source.get(GATE_TIMEOUT_KEY) is not None:
            return int(source[GATE_TIMEOUT_KEY])
    return None


def _validate_accuracy(block: Any, known: Sequence[str], errors: list[str]) -> None:
    """The ``accuracy`` block's own keys: ``boot``, ``serve``, ``timeout``, and that every key is known."""
    if not isinstance(block, dict):
        errors.append(f"'{ACCURACY_FIELD}' must be a mapping, got {type(block).__name__}")
        return
    reject_unknown_keys(block, known, errors, where=ACCURACY_FIELD)
    if block.get("boot") is not None:
        _validate_boot(block["boot"], errors)
    serve = block.get("serve")
    if serve is not None:
        where = f"{ACCURACY_FIELD}.serve"
        if not isinstance(serve, dict):
            errors.append(f"'{where}' must be a mapping")
        else:
            reject_unknown_keys(serve, SERVE_KEYS, errors, where=where)
            if serve.get("enabled") is not None and not isinstance(serve["enabled"], bool):
                errors.append(f"'{where}.enabled' must be true or false")
            for key in ("max_tokens", "timeout"):
                if serve.get(key) is not None and not _positive_int(serve[key]):
                    errors.append(f"'{where}.{key}' must be an integer >= 1")
            if serve.get("prompt") is not None and not (
                isinstance(serve["prompt"], str) and serve["prompt"].strip()
            ):
                errors.append(f"'{where}.prompt' must be a non-empty string")
    if block.get("timeout") is not None and not _positive_int(block["timeout"]):
        errors.append(f"'{ACCURACY_FIELD}.timeout' must be an integer >= 1")


def _resolve_spec_relative_paths(data: dict[str, Any], spec_dir: Path) -> None:
    """Make the paths a user writes beside the spec absolute, in place.

    ``accuracy.llm_api_options``, the boot prompt files and the boot images that sit
    beside the spec. Absolute paths are untouched, so the workspace's copy of a resolved
    spec resolves to itself. An image that is not beside the spec stays relative: the
    check resolves it against the tree under test.
    """

    def absolute(value: Any) -> Any:
        if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
            return value
        return str((spec_dir / value).resolve())

    accuracy = data.get(ACCURACY_FIELD)
    if not isinstance(accuracy, dict):
        return
    options = accuracy.get("llm_api_options")
    if isinstance(options, dict):
        accuracy["llm_api_options"] = {k: absolute(v) for k, v in options.items()}
    elif options is not None:
        accuracy["llm_api_options"] = absolute(options)
    boot = accuracy.get("boot")
    if isinstance(boot, dict):
        levels = (v for k, v in boot.items() if k not in BOOT_KEYS and isinstance(v, dict))
        for level in (boot, *levels):
            if level.get("prompts_file") is not None:
                level["prompts_file"] = absolute(level["prompts_file"])
            for entry in level.get("prompts") or ():
                if isinstance(entry, dict) and isinstance(entry.get("images"), list):
                    entry["images"] = [
                        absolute(i) if isinstance(i, str) and (spec_dir / i).exists() else i
                        for i in entry["images"]
                    ]


def drop_ignored_keys(data: dict[str, Any], notices: list[str] | None) -> None:
    """Take the keys this version no longer reads out of ``data``, one notice for each."""

    def drop(block: Any, parts: list[str], at: str, why: str) -> None:
        if not isinstance(block, dict):
            return
        head, rest = parts[0], parts[1:]
        for key in list(block) if head == "*" else [head]:
            if key not in block:
                continue
            where = f"{at}.{key}" if at else str(key)
            if rest:
                drop(block[key], rest, where, why)
            else:
                del block[key]
                if notices is not None:
                    notices.append(f"'{where}' is ignored: {why}; drop the key")

    for path, why in IGNORED_KEYS.items():
        drop(data, path.split("."), "", why)


def load_and_validate_task_yaml(
    path: str | Path, notices: list[str] | None = None
) -> dict[str, Any]:
    """Parse ``path`` as YAML, validate it, and fill in the workflow's defaults.

    Returns the resolved mapping; ``spec.deployments.resolve`` reads what it deploys from
    it, whichever way the spec states that. Raises :class:`TaskSchemaError` with every
    problem in one message. ``notices``, when given, gains one line for each ignored key,
    before any error is raised.
    """
    # Imported here: those modules read this one's helpers.
    from . import deployments as deployments_mod
    from . import shorthand

    task_path = Path(path)
    if not task_path.is_file():
        raise TaskSchemaError(f"task file not found: {task_path}")
    try:
        data = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TaskSchemaError(f"{task_path} is not valid YAML: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TaskSchemaError(
            f"{task_path} must be a YAML mapping at the top level, got {type(data).__name__}"
        )
    stated = deployments_mod.is_stated(data)
    spec_dir = task_path.resolve().parent
    drop_ignored_keys(data, notices)
    # Before validation: a path's existence check reads the resolved path.
    _resolve_spec_relative_paths(data, spec_dir)
    deployments_mod.resolve_paths(data, spec_dir)

    errors: list[str] = []
    validate_commit_policy_block(data.get(COMMIT_POLICY_FIELD), errors)
    if "perf" in data:
        errors.append(
            "'perf' is not read: the flow has no tuning stage, and performance work runs "
            "outside it on the delivered target; drop the block"
        )
    reject_unknown_keys({k: v for k, v in data.items() if k != "perf"}, TOP_LEVEL_KEYS, errors)
    errors.extend(validate_agents(data, ROLES))
    unstated = unstated_models(data)
    if unstated:
        errors.append(
            missing(
                "agents.defaults.model",
                f"the model the roles run on, never inherited from the launching session "
                f"(unstated for {', '.join(unstated)})",
                "agents: {defaults: {backend: claude-code, model: <model>}}",
            )
        )
    for name, what in REQUIRED_PATH_FIELDS.items():
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(missing(name, what, f"{name}: /path/to/it"))
        elif not Path(value).exists():
            errors.append(f"'{name}' points to a non-existent path: {value}")
    arch = data.get(ARCH_FIELD)
    if not isinstance(arch, str) or not ARCH_PATTERN.match(arch):
        errors.append(
            missing(
                ARCH_FIELD,
                "the GPU architecture, never inferred (sm_100 is B200, sm_103 is GB300)"
                + (f", got {arch!r}" if arch is not None else ""),
                "arch: sm_103",
            )
        )
    _validate_environment(data.get(ENVIRONMENT_FIELD), errors)
    data.setdefault(TARGET_FIELD, {})
    _validate_target(data[TARGET_FIELD], errors)
    notes = data.get(NOTES_FIELD)
    if notes is not None and (not isinstance(notes, str) or not notes.strip()):
        errors.append(f"'{NOTES_FIELD}' must be a non-empty string")
    if data.get("scope") is not None and data["scope"] not in SCOPES:
        errors.append(f"'scope' must be one of {SCOPES}; got {data['scope']!r}")

    accuracy = data.get(ACCURACY_FIELD)
    if stated:
        deployments_mod.validate(data.get(deployments_mod.DEPLOYMENTS_FIELD), errors)
        beside = [key for key in shorthand.SHORTHAND_KEYS if data.get(key)]
        if beside:
            errors.append(
                f"{', '.join(repr(key) for key in beside)} beside 'deployments': these keys are "
                f"shorthand for deployments, and this spec states its deployments itself. Put "
                f"what they ask for in a deployment's options, `inputs`, `benchmarks` or `tests`"
            )
        if accuracy is not None:
            _validate_accuracy(accuracy, deployments_mod.STATED_ACCURACY_KEYS, errors)
            if isinstance(accuracy, dict):
                deployments_mod.validate_judging(accuracy, errors, ACCURACY_FIELD)
    else:
        if accuracy is not None:
            known = (*deployments_mod.STATED_ACCURACY_KEYS, *shorthand.ACCURACY_KEYS)
            _validate_accuracy(accuracy, known, errors)
        if isinstance(accuracy, dict):
            # What is stated once for every deployment; a per-deployment mapping is judged
            # where it is resolved.
            flat = {
                key: value
                for key, value in accuracy.items()
                if not isinstance(value, dict)
                or (
                    key == "discriminators" and not any(isinstance(v, dict) for v in value.values())
                )
            }
            deployments_mod.validate_judging(flat, errors, ACCURACY_FIELD)
        shorthand.validate(data, errors)
    errors = list(dict.fromkeys(errors))
    if errors:
        bullet = "\n  - "
        raise TaskSchemaError(
            f"{task_path} failed trtllm-modeling-v2 schema validation:{bullet}{bullet.join(errors)}"
        )
    if not stated:
        # The shorthand is read here, once: below the loader a spec states its deployments.
        data[deployments_mod.DEPLOYMENTS_FIELD] = [
            {**found.as_dict(), "benchmarks": [b.as_dict() for b in found.benchmarks]}
            for found in shorthand.expand(data)
        ]
        for key in shorthand.SHORTHAND_KEYS:
            data.pop(key, None)
        if isinstance(accuracy, dict):
            # What was stated per deployment is now each benchmark's own.
            spent = (set(accuracy) - set(flat) - {"boot", "serve"}) | set(shorthand.ACCURACY_KEYS)
            for key in spent & set(accuracy):
                del accuracy[key]
    return data
