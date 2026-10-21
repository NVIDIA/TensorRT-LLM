# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The user-facing ``task.yaml`` and the per-cluster ``site.yaml``: closed keys, batched errors."""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..schema_keys import reject_unknown_keys

INTAKE_VERSION = 1

GOAL_ONBOARD = "onboard"
GOAL_ADD_FEATURES = "add_features"
GOAL_TUNE = "tune"
GOALS: tuple[str, ...] = (GOAL_ONBOARD, GOAL_ADD_FEATURES, GOAL_TUNE)

STATUS_REQUIRED = "required"
STATUS_NICE = "nice_to_have"
STATUS_OUT = "out_of_scope"
STATUSES: tuple[str, ...] = (STATUS_REQUIRED, STATUS_NICE, STATUS_OUT)

TASK_KEYS: tuple[str, ...] = (
    "version",
    "goals",
    "model",
    "device",
    "parallelism",
    "workload",
    "features",
    "accuracy",
    "perf",
    "constraints",
    "deliverable",
    "notes",
    "provenance",
    "open_questions",
    "confirmed",
    # Written by the fast path (`derive.py`): the checkpoint, site and goal the
    # request was derived from, and when.
    "derived",
)
MODEL_KEYS = (
    "checkpoint",
    "architecture",
    "quantization",
    "kind",
    "in_tree",
    "reference_impl",
    "base",
    "target",
)
DEVICE_KEYS = ("arch", "gpus_per_node", "nodes")
WORKLOAD_KEYS = ("isl", "osl", "max_seq_len", "max_batch_size", "concurrency")
ACCURACY_KEYS = (
    "benchmark",
    "protocol",
    "reference",
    "reference_yaml",
    "model_key",
    # The reference row's discriminators, when the checkpoint's own cannot be read.
    "discriminators",
    # The boot gate's prompts and the witness declarations, carried to the
    # resolved spec's `accuracy.boot` and `accuracy.witness` as they are.
    "boot",
    "witness",
)
PERF_KEYS = ("metric", "baseline", "goal", "configs")
CONSTRAINT_KEYS = ("embargo", "base_ref", "budget")
FEATURE_ENTRY_KEYS = ("status", "params", "members", "accept")
PARALLEL_KINDS = ("tp", "tep", "dep")
#: One parallel configuration, as a member is written: `tp1`, `tep4`, `dep8`.
PARALLEL_SEGMENT = re.compile(r"^(tp|tep|dep)([1-9][0-9]*)$")

#: Runtime behaviours every target has; their status decides the gate switches.
CORE_FEATURES = ("cuda_graph", "overlap_scheduler", "chunked_prefill", "prefix_reuse", "kv_cache")
#: Features that become a build increment of today's pipeline, by their pipeline name.
INCREMENT_FEATURES = {
    "kv_quant": "kv_quant",
    "decoder_replay": "decoder_replay",
    "encoder_replay": "encoder_replay",
    "host_offload": "host_offload",
    "disaggregated": "disagg",
    "multimodal": "multimodal",
}
SPECULATIVE_ALGOS = ("mtp", "eagle3", "dflash", "dspark")
#: Features recorded for the user but with no Step 2 consumer yet.
UNGATED_FEATURES = (
    "long_context",
    "guided_decoding",
    "logprobs",
    "reasoning_parser",
    "tool_parser",
    "lora",
)
KNOWN_FEATURES = (
    *CORE_FEATURES,
    "speculative",
    *INCREMENT_FEATURES,
    *UNGATED_FEATURES,
    "custom",
)

SITE_KEYS = (
    "trtllm_repo_path",
    "environment",
    "agents",
    "perf",
    "commit_policy",
    # The cluster's devices, and what a derived request takes when its goal is silent.
    "device",
    "defaults",
)
SITE_PERF_KEYS = ("spools", "dev_spools", "env_script")
#: `site.yaml defaults`: the members, workload, accuracy anchor, boot prompts and
#: witness declarations (by pipeline feature name) a derived request starts from.
SITE_DEFAULTS_KEYS = ("members", "workload", "accuracy", "boot", "witness")
SITE_DEFAULT_ACCURACY_KEYS = ("benchmark", "protocol", "reference", "reference_yaml", "model_key")


class IntakeError(ValueError):
    """A ``task.yaml`` or ``site.yaml`` that cannot be compiled, every problem listed."""


def feature_entry(value: Any) -> dict[str, Any]:
    """``required`` or ``{status: required, params: ...}`` as one shape."""
    if isinstance(value, str):
        return {"status": value}
    return dict(value or {})


def feature_status(task: Mapping[str, Any], name: str) -> str | None:
    """The declared status of ``name``, or None when the task does not mention it."""
    features = task.get("features") or {}
    if name not in features:
        return None
    return feature_entry(features[name]).get("status")


def _mapping(task: Mapping[str, Any], key: str, keys, errors: list[str]) -> Mapping[str, Any]:
    block = task.get(key)
    if block is None:
        return {}
    if not isinstance(block, Mapping):
        errors.append(f"'{key}' must be a mapping")
        return {}
    reject_unknown_keys(block, keys, errors, where=key)
    return block


def validate_task(task: Any) -> list[str]:
    """Every problem with the shape of a user-facing ``task.yaml``."""
    if not isinstance(task, Mapping):
        return ["task.yaml must be a mapping"]
    errors: list[str] = []
    reject_unknown_keys(task, TASK_KEYS, errors, where="task.yaml")
    if task.get("version") != INTAKE_VERSION:
        errors.append(f"'version' must be {INTAKE_VERSION}")
    goals = task.get("goals")
    if not isinstance(goals, list) or not all(g in GOALS for g in goals):
        errors.append(f"'goals' must be a list drawn from {list(GOALS)}")
    model = _mapping(task, "model", MODEL_KEYS, errors)
    if not model.get("checkpoint"):
        errors.append("'model.checkpoint' is required")
    device = _mapping(task, "device", DEVICE_KEYS, errors)
    if not device.get("arch"):
        errors.append("'device.arch' is required (sm_100, sm_103, ...)")
    parallelism = task.get("parallelism")
    if parallelism is not None:
        if not isinstance(parallelism, list) or not parallelism:
            errors.append("'parallelism' must be a non-empty list of {kind, world}")
        else:
            for i, entry in enumerate(parallelism):
                if not isinstance(entry, Mapping) or entry.get("kind") not in PARALLEL_KINDS:
                    errors.append(f"'parallelism[{i}].kind' must be one of {list(PARALLEL_KINDS)}")
                elif not isinstance(entry.get("world"), int) or entry["world"] < 1:
                    errors.append(f"'parallelism[{i}].world' must be an integer >= 1")
    workload = _mapping(task, "workload", WORKLOAD_KEYS, errors)
    for key in ("isl", "osl", "max_seq_len", "max_batch_size"):
        if key in workload and (not isinstance(workload[key], int) or workload[key] < 1):
            errors.append(f"'workload.{key}' must be an integer >= 1")
    features = task.get("features") or {}
    if not isinstance(features, Mapping):
        errors.append("'features' must be a mapping of feature -> status or entry")
        features = {}
    for name, value in features.items():
        if name not in KNOWN_FEATURES:
            errors.append(f"'features.{name}' is not a known feature: {sorted(KNOWN_FEATURES)}")
            continue
        if name == "custom":
            if not isinstance(value, list):
                errors.append("'features.custom' must be a list")
            continue
        entry = feature_entry(value)
        reject_unknown_keys(entry, FEATURE_ENTRY_KEYS, errors, where=f"features.{name}")
        if entry.get("status") not in STATUSES:
            errors.append(f"'features.{name}.status' must be one of {list(STATUSES)}")
        if name == "speculative" and entry.get("status") != STATUS_OUT:
            algo = (entry.get("params") or {}).get("algo")
            if algo not in SPECULATIVE_ALGOS:
                errors.append(
                    f"'features.speculative.params.algo' must be one of {list(SPECULATIVE_ALGOS)}"
                )
    _mapping(task, "accuracy", ACCURACY_KEYS, errors)
    _mapping(task, "perf", PERF_KEYS, errors)
    _mapping(task, "constraints", CONSTRAINT_KEYS, errors)
    questions = task.get("open_questions")
    if questions is not None and not isinstance(questions, list):
        errors.append("'open_questions' must be a list")
    derived = task.get("derived")
    if derived is not None and not isinstance(derived, Mapping):
        errors.append("'derived' must be a mapping")
    return errors


def validate_site(site: Any) -> list[str]:
    """Every problem with the shape of a ``site.yaml``."""
    if not isinstance(site, Mapping):
        return ["site.yaml must be a mapping"]
    errors: list[str] = []
    reject_unknown_keys(site, SITE_KEYS, errors, where="site.yaml")
    for key in ("trtllm_repo_path", "environment"):
        if not site.get(key):
            errors.append(f"site.yaml: '{key}' is required")
    perf = site.get("perf") or {}
    if isinstance(perf, Mapping):
        reject_unknown_keys(perf, SITE_PERF_KEYS, errors, where="site.yaml perf")
    device = site.get("device")
    if device is not None:
        if not isinstance(device, Mapping):
            errors.append("site.yaml: 'device' must be a mapping")
        else:
            reject_unknown_keys(device, DEVICE_KEYS, errors, where="site.yaml device")
    defaults = site.get("defaults")
    if defaults is not None:
        if not isinstance(defaults, Mapping):
            errors.append("site.yaml: 'defaults' must be a mapping")
        else:
            reject_unknown_keys(defaults, SITE_DEFAULTS_KEYS, errors, where="site.yaml defaults")
            members = defaults.get("members")
            if members is not None and (
                not isinstance(members, list)
                or not members
                or not all(isinstance(m, str) and PARALLEL_SEGMENT.match(m) for m in members)
            ):
                errors.append(
                    "site.yaml: 'defaults.members' must be a non-empty list of tp<N>, tep<N>, dep<N>"
                )
            for key in ("workload", "accuracy", "boot", "witness"):
                if defaults.get(key) is not None and not isinstance(defaults[key], Mapping):
                    errors.append(f"site.yaml: 'defaults.{key}' must be a mapping")
            if isinstance(defaults.get("workload"), Mapping):
                reject_unknown_keys(
                    defaults["workload"], WORKLOAD_KEYS, errors, where="site.yaml defaults.workload"
                )
            if isinstance(defaults.get("accuracy"), Mapping):
                reject_unknown_keys(
                    defaults["accuracy"],
                    SITE_DEFAULT_ACCURACY_KEYS,
                    errors,
                    where="site.yaml defaults.accuracy",
                )
    return errors


def is_intake_task(data: Any) -> bool:
    """A user-facing ``task.yaml`` (as opposed to an already resolved spec)."""
    return isinstance(data, Mapping) and "goals" in data and "version" in data
