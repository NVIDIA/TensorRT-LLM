# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration variants: a unit's target, served under another configuration.

A variant builds nothing of its own. It adds checks to the contract, and those
checks run with the unit's spec overlaid with the variant's switches::

    variants:
      kv_fp8:
        unit: core                       # whose target and settings it starts from
        members: [tep4]                  # default: the unit's members
        gates: [boot, accuracy]          # the default; `acceptance` needs a speculator
        runtime: {extra_flags: [--kv_cache_dtype=fp8]}      # boot: appended to the unit's
        llm_api_options: {kv_cache_config: {dtype: fp8}}    # merged into the unit's YAML
        env: {}                          # added to the gate command's environment
        accuracy: {discriminators: {}}   # accuracy keys that differ from the unit's
        note: KV cache stored as FP8

Each check is named by the variant (``kv_fp8:accuracy[gsm8k]@tep4``), so its
verdicts, per-prompt baselines and findings are its own, never the unit's.
``checks/check_config.variant_spec`` applies and materializes the overlay.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping

from .schema_keys import reject_unknown_keys

VARIANTS_FIELD = "variants"
VARIANT_KEYS: tuple[str, ...] = (
    "unit",
    "members",
    "gates",
    "runtime",
    "llm_api_options",
    "env",
    "accuracy",
    "note",
)
VARIANT_GATES: tuple[str, ...] = ("boot", "accuracy", "acceptance")
DEFAULT_VARIANT_GATES: tuple[str, ...] = ("boot", "accuracy")
#: Accuracy keys a variant may set for itself; every other key is the unit's.
VARIANT_ACCURACY_KEYS: tuple[str, ...] = (
    "benchmark",
    "reference_yaml",
    "model_key",
    "discriminators",
    "benchmark_args",
    "samples",
    "acceptance",
)
RUNTIME_OVERLAY_KEYS: tuple[str, ...] = (
    "cuda_graph",
    "chunked_prefill",
    "overlap_scheduler",
    "kv_cache_reuse",
    "extra_flags",
)
#: Where a variant's merged LLM-API options are written, under the workspace.
OPTIONS_DIRNAME = "variants"
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def variants(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The declared variants, by name, in declaration order."""
    block = data.get(VARIANTS_FIELD) or {}
    return {str(k): dict(v or {}) for k, v in block.items()} if isinstance(block, Mapping) else {}


def is_variant(data: Mapping[str, Any], name: str | None) -> bool:
    return bool(name) and str(name) in variants(data)


def base_unit(data: Mapping[str, Any], name: str) -> str:
    """The unit whose target a variant configures."""
    return str(variants(data)[name].get("unit") or "core")


def variant_gates(data: Mapping[str, Any], name: str) -> tuple[str, ...]:
    return tuple(variants(data)[name].get("gates") or DEFAULT_VARIANT_GATES)


def validate_variants(
    block: Any,
    *,
    units: list[str],
    speculators: set[str],
    unified: bool,
    errors: list[str],
) -> None:
    """Every problem with the ``variants`` block, batched into ``errors``."""
    if block is None:
        return
    if not isinstance(block, Mapping):
        errors.append(f"'{VARIANTS_FIELD}' must be a mapping from variant name to its entry")
        return
    for name, entry in block.items():
        where = f"{VARIANTS_FIELD}.{name}"
        if not isinstance(name, str) or not _NAME.match(name):
            errors.append(f"'{where}': a variant name is lowercase letters, digits and '_'")
            continue
        if name in units or name == "default":
            errors.append(f"'{where}': {name!r} is a unit name; a variant needs its own")
            continue
        if not isinstance(entry, Mapping):
            errors.append(f"'{where}' must be a mapping")
            continue
        reject_unknown_keys(entry, VARIANT_KEYS, errors, where=where)
        unit = str(entry.get("unit") or "core")
        if unit not in units:
            errors.append(f"'{where}.unit' is {unit!r}, not one of the units {units}")
            continue
        gates = entry.get("gates") or list(DEFAULT_VARIANT_GATES)
        if not isinstance(gates, list) or not gates or any(g not in VARIANT_GATES for g in gates):
            errors.append(
                f"'{where}.gates' must be a non-empty list drawn from {list(VARIANT_GATES)}"
            )
        elif "acceptance" in gates:
            built = set(units[1 : units.index(unit) + 1])
            if not built & speculators:
                errors.append(
                    f"'{where}.gates' asks for acceptance, but unit {unit!r} implements no "
                    f"speculator ({sorted(speculators)})"
                )
        members = entry.get("members")
        if members is not None:
            if not unified:
                errors.append(f"'{where}.members' is only legal with 'parallel: unified'")
            elif not isinstance(members, list) or not members:
                errors.append(f"'{where}.members' must be a non-empty list of parallel segments")
            else:
                from .task_schema import TaskSchemaError, parse_parallel_segment

                for member in members:
                    try:
                        _, size = parse_parallel_segment(str(member))
                    except TaskSchemaError as exc:
                        errors.append(f"'{where}.members': {exc}")
                        continue
                    if size is None:
                        errors.append(f"'{where}.members': {member!r} is not a concrete member")
        runtime = entry.get("runtime")
        if runtime is not None:
            if not isinstance(runtime, Mapping):
                errors.append(f"'{where}.runtime' must be a mapping")
            else:
                reject_unknown_keys(runtime, RUNTIME_OVERLAY_KEYS, errors, where=f"{where}.runtime")
                flags = runtime.get("extra_flags")
                if flags is not None and not (
                    isinstance(flags, list) and all(isinstance(f, str) for f in flags)
                ):
                    errors.append(f"'{where}.runtime.extra_flags' must be a list of strings")
        for key in ("llm_api_options", "env", "accuracy"):
            if entry.get(key) is not None and not isinstance(entry[key], Mapping):
                errors.append(f"'{where}.{key}' must be a mapping")
        if isinstance(entry.get("accuracy"), Mapping):
            reject_unknown_keys(
                entry["accuracy"], VARIANT_ACCURACY_KEYS, errors, where=f"{where}.accuracy"
            )
        if not (entry.get("runtime") or entry.get("llm_api_options") or entry.get("env")):
            errors.append(
                f"'{where}' changes nothing: give it `runtime`, `llm_api_options` or `env`"
            )


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def variant_anchor_pending(data: Mapping[str, Any], spec: Mapping[str, Any], name: str) -> bool:
    """Whether variant ``name`` has no written anchor (``accuracy.anchor``).

    Its own say where the anchor mapping names it, else its base unit's: a
    variant of a unit whose anchor is pending has none of its own either.
    """
    # task_schema validates the variants with this module, so it is read here, when called.
    from .task_schema import ANCHOR_PENDING, anchor_pending

    own = (spec.get("accuracy") or {}).get("anchor")
    if isinstance(own, Mapping) and name in own:
        return own[name] == ANCHOR_PENDING
    return anchor_pending(data.get("accuracy") or {}, base_unit(data, name))


def describe(data: Mapping[str, Any], name: str) -> str:
    """What a variant asks of the target, for the planner and the assembler."""
    entry = variants(data)[name]
    lines = [
        f"Variant **{name}** of unit `{base_unit(data, name)}`: {entry.get('note') or ''}".rstrip()
    ]
    if entry.get("runtime"):
        lines.append(f"- boot-gate switches added: `{dict(entry['runtime'])}`")
    if entry.get("llm_api_options"):
        lines.append(
            f"- LLM-API options merged into the unit's: `{dict(entry['llm_api_options'])}`"
        )
    if entry.get("env"):
        lines.append(f"- environment added: `{dict(entry['env'])}`")
    lines.append(f"- gates: {', '.join(entry.get('gates') or DEFAULT_VARIANT_GATES)}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_VARIANT_GATES",
    "OPTIONS_DIRNAME",
    "VARIANTS_FIELD",
    "VARIANT_GATES",
    "base_unit",
    "describe",
    "is_variant",
    "validate_variants",
    "variant_anchor_pending",
    "variant_gates",
    "variants",
]
