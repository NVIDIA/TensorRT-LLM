# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conservative, model-key-aware selection for the shared perf path table."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import yaml

from cbts.blocks import YAMLIndex, _entry_target, _strip_pytest_options, normalize_test_id

from ._helpers import reconstruct_diff_pre_image

MODEL_PATHS = "tests/integration/defs/perf/_model_paths.py"
_CONSUMERS = {"perf/test_perf.py", "perf/test_perf_sanity.py"}


def changed_model_keys(source: str, diff: str) -> set[str] | None:
    """Return changed literal keys only when the rest of the module is unchanged."""
    previous = reconstruct_diff_pre_image(source, diff)
    if previous is None:
        return None
    tables: list[dict[str, str]] = []
    skeletons: list[str] = []
    for text in (previous, source):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return None
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "MODEL_PATH_DICT"
        ]
        if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
            return None
        if any(
            isinstance(node, ast.Name)
            and node.id == "MODEL_PATH_DICT"
            and node is not assignments[0].targets[0]
            for node in ast.walk(tree)
        ):
            return None
        table = assignments[0].value
        values: dict[str, str] = {}
        for key, value in zip(table.keys, table.values):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and key.value not in values
            ):
                return None
            values[key.value] = value.value
        tables.append(values)
        assignments[0].value = ast.Dict(keys=[], values=[])
        skeletons.append(ast.dump(tree))
    if skeletons[0] != skeletons[1]:
        return None
    before, after = tables
    return {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}


def _uses_only_model_lookups(source: str) -> bool:
    """Reject enumeration, mutation, or passing the table to an unknown consumer."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    bindings = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {"_model_paths", "perf._model_paths", "defs.perf._model_paths"}
        for alias in node.names
        if alias.name == "MODEL_PATH_DICT"
    }
    # Older sanity modules copy the shared table and add a local override.
    copies: set[ast.Name] = set()
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            continue
        spreads = [v for k, v in zip(node.value.keys, node.value.values) if k is None]
        if len(spreads) == 1 and isinstance(spreads[0], ast.Name) and spreads[0].id in bindings:
            for key, value in zip(node.value.keys, node.value.values):
                if key is not None and not (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    return False
            copies.add(spreads[0])
            copies.add(node.targets[0])
            bindings.add(node.targets[0].id)
    if not bindings:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in bindings
        ):
            return False
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"globals", "locals", "eval", "exec"}
        ):
            return False
        if not isinstance(node, ast.Name) or node.id not in bindings or node in copies:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.attr == "keys":
            call = parents.get(parent)
            if not isinstance(call, ast.Call) or call.args or call.keywords:
                return False
            parent = parents.get(call)
        if isinstance(parent, ast.Subscript) and parent.value is node:
            if isinstance(parent.ctx, ast.Load) and _is_model_name(parent.slice):
                continue
        if (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.In, ast.NotIn))
            and _is_model_name(parent.left)
        ):
            continue
        return False
    return True


def _is_model_name(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "model_name") or (
        isinstance(node, ast.Attribute) and node.attr == "model_name"
    )


def _uses_default_config_folders(source: str) -> bool:
    """Decline the YAML adapter if the consumer changes its config locations."""
    expected = {
        "AGG_CONFIG_FOLDER": "aggregated",
        "DISAGG_CONFIG_FOLDER": "disaggregated",
    }
    tree = ast.parse(source)  # The lookup-use check has already parsed this source.
    matched: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Store):
            continue
        if node.id not in expected:
            continue
        assignment = next(
            (
                statement
                for statement in tree.body
                if isinstance(statement, ast.Assign) and statement.targets == [node]
            ),
            None,
        )
        if assignment is None or node.id in matched:
            return False
        value = f'os.environ.get("{node.id}", "tests/scripts/perf-sanity/{expected[node.id]}")'
        if ast.dump(assignment.value) != ast.dump(ast.parse(value, mode="eval").body):
            return False
        matched.add(node.id)
    return matched == set(expected)


def _sanity_models(repo_root: Path, parameter: str) -> set[str] | None:
    """Read all model names in the case's config, over-including server overrides."""
    labels = parameter.split("-")
    prefix = labels[0]
    if prefix not in {"aggr", "aggr_upload", "disagg", "disagg_upload"} or len(labels) < 2:
        return None
    disagg = prefix.startswith("disagg") or labels[1] in {"ctx_only", "gen_only_no_context"}
    if disagg:
        if labels[1] not in {"e2e", "gen_only", "ctx_only", "gen_only_no_context"}:
            return None
        stem = "-".join(labels[2:])
        env_name = "DISAGG_CONFIG_FOLDER"
        folder = "disaggregated"
    else:
        stem = labels[1]
        env_name = "AGG_CONFIG_FOLDER"
        folder = "aggregated"
    # Runtime overrides and unfamiliar grammar must keep the case.
    if env_name in os.environ or not stem or "/" in stem or "\\" in stem:
        return None
    if not stem.endswith(".yaml"):
        stem += ".yaml"
    path = repo_root / "tests/scripts/perf-sanity" / folder / stem
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    names: set[str] = set()
    pending: list[object] = [data]
    visited: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, (dict, list)):
            if id(item) in visited:
                continue
            visited.add(id(item))
        if isinstance(item, dict):
            for key, value in item.items():
                if key == "model_name":
                    if not isinstance(value, str) or not value or any(c in value for c in "${}"):
                        return None
                    names.add(value)
                pending.append(value)
        elif isinstance(item, list):
            pending.extend(item)
    return names or None


def model_path_filters(
    repo_root: Path,
    source: str,
    diff: str,
    consumers: list[str],
    yaml_index: YAMLIndex,
) -> dict[tuple[str, int], dict[str, set[str]]] | None:
    """Select known consumers' model cases; None requests the existing broad fallback."""
    keys = changed_model_keys(source, diff)
    if keys is None or not set(consumers) <= _CONSUMERS:
        return None
    safe_consumers: set[str] = set()
    for consumer in consumers:
        try:
            text = (repo_root / "tests/integration/defs" / consumer).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if _uses_only_model_lookups(text) and (
            consumer != "perf/test_perf_sanity.py" or _uses_default_config_folders(text)
        ):
            safe_consumers.add(consumer)
    filters: dict[tuple[str, int], dict[str, set[str]]] = {}
    for block in yaml_index.blocks:
        for raw in block.tests:
            target = _entry_target(raw)
            relevant = [
                consumer
                for consumer in consumers
                if target == consumer
                or target.startswith(consumer + "::")
                or consumer.startswith(target.rstrip("/") + "/")
            ]
            if not relevant:
                continue
            entry = _strip_pytest_options(normalize_test_id(raw))
            models: set[str] | None = None
            if (
                len(relevant) == 1
                and relevant[0] in safe_consumers
                and "[" in entry
                and entry.endswith("]")
            ):
                parameter = entry.split("[", 1)[1][:-1]
                if target == "perf/test_perf.py::test_perf":
                    models = {parameter.split("-", 1)[0]} if parameter else None
                elif target == "perf/test_perf_sanity.py::test_e2e":
                    models = _sanity_models(repo_root, parameter)
            if models is not None and keys.isdisjoint(models):
                continue
            prefix = entry if "[" in entry else target
            filters.setdefault((block.yaml_stem, block.block_index), {}).setdefault(
                prefix, set()
            ).add(raw)
    return filters
