# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The accuracy tasks the in-tree accuracy tests run, and their per-model reference scores.

``tests/integration/defs/accuracy/accuracy_core.py`` declares one class per
task (``GSM8K``, ``MMLU`` ...) with its dataset, sample count and the
hypothesis-test parameters (``ALPHA``, ``BETA``, ``SIGMA``) its threshold is
computed from; ``references/<task>.yaml`` maps a model key to rows keyed by
quantization, KV-cache quantization and speculative-decoding algorithm. Both
are read as source: the classes' literal attributes, the YAML as data.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml

ACCURACY_CORE_REL = "tests/integration/defs/accuracy/accuracy_core.py"
REFERENCES_REL = "tests/integration/defs/accuracy/references"

_FIELDS = (
    "DATASET",
    "DATASET_DIR",
    "ALPHA",
    "BETA",
    "SIGMA",
    "NUM_SAMPLES",
    "MAX_INPUT_LEN",
    "MAX_OUTPUT_LEN",
    "MAX_BATCH_SIZE",
)


def _literal(node: ast.expr) -> Any:
    if isinstance(node, ast.JoinedStr):
        # f"{llm_models_root()}/datasets/x": keep the text, name the call.
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                func = value.value.func if isinstance(value.value, ast.Call) else value.value
                name = getattr(func, "id", None) or getattr(func, "attr", "")
                parts.append("$LLM_MODELS_ROOT" if name == "llm_models_root" else f"{{{name}}}")
        return "".join(parts)
    try:
        return ast.literal_eval(node)
    except ValueError:
        return None


def extract_accuracy_tasks(repo: Path) -> list[dict[str, Any]]:
    path = Path(repo) / ACCURACY_CORE_REL
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    tasks = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name == "AccuracyTask":
            continue
        values: dict[str, Any] = {}
        for item in node.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                target = item.targets[0]
                if isinstance(target, ast.Name) and target.id in _FIELDS:
                    values[target.id.lower()] = _literal(item.value)
        if "dataset" in values:
            tasks.append(
                {"task": node.name, "path": f"{ACCURACY_CORE_REL}:{node.lineno}", **values}
            )
    return tasks


def extract_references(repo: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """``{benchmark: {model_key: [row, ...]}}`` from ``references/*.yaml``."""
    references: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in sorted((Path(repo) / REFERENCES_REL).glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(data, dict):
            references[path.stem] = {
                str(model): [row for row in rows if isinstance(row, dict)]
                for model, rows in data.items()
                if isinstance(rows, list)
            }
    return references
