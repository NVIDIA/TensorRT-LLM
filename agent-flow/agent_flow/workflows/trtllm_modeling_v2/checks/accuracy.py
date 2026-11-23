# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy references: the row a unit's benchmark is anchored to, as the flow reads it.

The e2e test judges a benchmark itself (``accuracy_core``, ``checks/e2e.py``).
The flow reads the same rows to know before the run whether one exists
(preflight), whether it is a low placeholder its first measured score replaces
(``run/validation/anchors.py``), and to report it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ..spec.derive import SPEC_DEC_ALGO
from ..spec.task_schema import features_through, resolve_per_unit
from ..spec.variants import base_unit, is_variant


def resolve_discriminators(
    accuracy: Mapping[str, Any], unit: str | None
) -> Mapping[str, Any] | None:
    """Pick the accuracy-reference discriminators for ``unit``; ``None`` when there are none.

    Accepts either a flat mapping (the whole run) or a mapping from unit
    name to mapping, with ``default`` as the fallback key.
    """
    block = accuracy.get("discriminators")
    if not isinstance(block, dict):
        return None
    # Per-unit form is recognised by its values being mappings; a flat form
    # like {"quant_algo": "NVFP4"} has scalar values.
    if any(isinstance(v, dict) for v in block.values()):
        chosen = block.get(unit)
        if chosen is None:
            chosen = block.get("default")
        return chosen if isinstance(chosen, dict) else None
    return block


def speculation(task_data: Mapping[str, Any], feature: str) -> str | None:
    """The speculative decoding ``feature``'s engine runs (its ``decoding_type``); ``None`` for none.

    The algorithm of the first speculator among the features its target implements,
    a variant's those of its unit, as ``check_config.check_configuration`` sets it.
    """
    unit = base_unit(task_data, feature) if is_variant(task_data, feature) else feature
    names = {str(f).lower() for f in features_through(task_data, unit)}
    return next((algo for name, algo in SPEC_DEC_ALGO.items() if name in names), None)


def reference_discriminators(task_data: Mapping[str, Any], feature: str) -> dict[str, Any] | None:
    """What ``feature``'s references row is matched by, as ``accuracy_core`` matches it.

    The spec's ``accuracy.discriminators`` for it, and the speculative decoding its
    engine runs as ``spec_dec_algo``: ``accuracy_core`` matches a row against the
    engine's own ``speculative_config.decoding_type``, so a speculator's benchmark
    needs a row of its own. ``None`` when there is nothing to match by.

    Raises:
        ValueError: The spec's discriminators name another ``spec_dec_algo`` than the
            engine runs.
    """
    wanted = dict(resolve_discriminators(task_data.get("accuracy") or {}, feature) or {})
    algo = speculation(task_data, feature)
    if algo is not None:
        declared = wanted.get("spec_dec_algo")
        if declared is not None and str(declared) != algo:
            raise ValueError(
                f"{feature}: accuracy.discriminators names spec_dec_algo {declared!r}, but its "
                f"engine runs {algo!r}, the spec_dec_algo accuracy_core matches its row by"
            )
        wanted["spec_dec_algo"] = algo
    return wanted or None


def load_reference_score(
    reference_yaml: str | Path,
    model_key: str,
    *,
    discriminators: Mapping[str, Any] | None = None,
) -> float:
    """Read the anchor for one model from TensorRT-LLM's accuracy references.

    An entry matches when its discriminators (``quant_algo``,
    ``kv_cache_quant_algo``, ``spec_dec_algo``) are exactly the ones asked
    for, no more, no less. Raises ``KeyError`` when no entry, or more than
    one, matches: the anchor is never invented.
    """
    path = Path(reference_yaml)
    if not path.is_file():
        raise FileNotFoundError(f"accuracy reference file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if model_key not in data:
        raise KeyError(
            f"{path} has no entry for {model_key!r}. The anchor comes from a written "
            f"source or from the user — it is never invented. Add the entry (citing "
            f"where the score came from) before the benchmark can be judged."
        )

    wanted = {k: v for k, v in (discriminators or {}).items() if v is not None}
    entries = data[model_key] or []
    matches = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        present = {k: v for k, v in entry.items() if k != "accuracy"}
        if present == wanted:
            matches.append(entry)

    if not matches:
        available = [
            {k: v for k, v in e.items() if k != "accuracy"} for e in entries if isinstance(e, dict)
        ]
        raise KeyError(
            f"{path}:{model_key} has no entry matching {wanted or '{}'}. Available "
            f"discriminator sets: {available}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"{path}:{model_key} has {len(matches)} entries matching {wanted or '{}'}; "
            f"the anchor must be unambiguous"
        )
    return float(matches[0]["accuracy"])


def resolve_reference(
    task_data: Mapping[str, Any],
    *,
    model_key: str,
    discriminators: Mapping[str, Any] | None = None,
    unit: str | None = None,
) -> float:
    """Look up this run's anchor, resolving the reference path against the repo.

    ``discriminators`` overrides ``accuracy.discriminators`` for ``unit``.
    """
    accuracy = dict(task_data.get("accuracy") or {})
    reference_yaml = Path(str(resolve_per_unit(accuracy, "reference_yaml", unit)))
    if not reference_yaml.is_absolute():
        reference_yaml = Path(str(task_data["trtllm_repo_path"])) / reference_yaml
    if discriminators is None:
        discriminators = resolve_discriminators(accuracy, unit)
    return load_reference_score(reference_yaml, model_key, discriminators=discriminators)


def resolve_model_key(task_data: Mapping[str, Any], unit: str | None = None) -> str:
    """The HuggingFace repo name this unit's anchor row is keyed by.

    An explicit ``accuracy.model_key`` when the spec states one (per unit,
    :func:`~.spec.task_schema.resolve_per_unit`), otherwise the last two components of the
    checkpoint path (``<org>/<model>``). This is the one implementation: the
    workflow's ``_model_key`` delegates here, so preflight's anchor check and
    the gate resolve the same key.
    """
    accuracy = dict(task_data.get("accuracy") or {})
    explicit = resolve_per_unit(accuracy, "model_key", unit)
    if explicit:
        return str(explicit)
    parts = Path(str(task_data["checkpoint_path"]).rstrip("/")).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


__all__ = [
    "load_reference_score",
    "reference_discriminators",
    "resolve_discriminators",
    "resolve_model_key",
    "resolve_reference",
    "speculation",
]
