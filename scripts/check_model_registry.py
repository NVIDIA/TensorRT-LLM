#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import collections
import pathlib
import sys
import typing

import yaml

MODEL_REGISTRY_PATH = pathlib.Path("examples/auto_deploy/model_registry/models.yaml")
REQUIRED_MODEL_KEYS = {"name", "yaml_extra"}
OPTIONAL_MODEL_KEYS = {"config_id"}
ALLOWED_MODEL_KEYS = REQUIRED_MODEL_KEYS | OPTIONAL_MODEL_KEYS
DEFAULT_CONFIG_ID = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the AutoDeploy model registry for duplicates and entry structure."
    )
    parser.add_argument(
        "--path",
        type=pathlib.Path,
        default=MODEL_REGISTRY_PATH,
        help="Path to the model registry YAML file.",
    )
    return parser.parse_args()


def load_registry(path: pathlib.Path) -> dict[str, typing.Any]:
    try:
        with path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise ValueError(f"Registry file does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Failed to parse YAML in {path}: {error}") from error

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected top-level mapping in {path}, got {type(loaded).__name__}.")

    return loaded


def validate_models(models: typing.Any) -> list[str]:
    if not isinstance(models, list):
        return [f"Expected 'models' to be a list, got {type(models).__name__}."]

    errors: list[str] = []
    seen_model_configs: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    seen_model_yaml_configs: dict[tuple[str, tuple[str, ...]], list[tuple[int, str]]] = (
        collections.defaultdict(list)
    )

    for index, model_entry in enumerate(models, start=1):
        entry_label = f"models[{index}]"
        if not isinstance(model_entry, dict):
            errors.append(
                f"{entry_label}: expected a mapping entry, got {type(model_entry).__name__}."
            )
            continue

        entry_keys = set(model_entry)
        missing_keys = sorted(REQUIRED_MODEL_KEYS - entry_keys)
        unexpected_keys = sorted(entry_keys - ALLOWED_MODEL_KEYS)
        if missing_keys or unexpected_keys:
            details: list[str] = []
            if missing_keys:
                details.append(f"missing keys {missing_keys}")
            if unexpected_keys:
                details.append(f"unexpected keys {unexpected_keys}")
            joined_details = ", ".join(details)
            errors.append(
                f"{entry_label}: expected keys ['name', 'yaml_extra'] and optional ['config_id']; "
                f"{joined_details}."
            )

        name = model_entry.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{entry_label}: missing non-empty string 'name'.")

        yaml_extra = model_entry.get("yaml_extra")
        if not isinstance(yaml_extra, list) or not all(
            isinstance(item, str) and item.strip() for item in yaml_extra
        ):
            errors.append(f"{entry_label}: 'yaml_extra' must be a list of non-empty strings.")

        config_id = model_entry.get("config_id", DEFAULT_CONFIG_ID)
        if not isinstance(config_id, str) or not config_id.strip():
            errors.append(f"{entry_label}: 'config_id' must be a non-empty string when provided.")

        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(config_id, str) or not config_id.strip():
            continue

        seen_model_configs[(name, config_id)].append(index)
        if isinstance(yaml_extra, list) and all(
            isinstance(item, str) and item.strip() for item in yaml_extra
        ):
            yaml_signature = tuple(yaml_extra)
            seen_model_yaml_configs[(name, yaml_signature)].append((index, config_id))

    for (name, config_id), indices in sorted(seen_model_configs.items()):
        if len(indices) > 1:
            joined_indices = ", ".join(str(index) for index in indices)
            errors.append(
                f"Duplicate model/config pair ({name!r}, {config_id!r}) found at entries: "
                f"{joined_indices}."
            )

    for (name, yaml_signature), records in sorted(seen_model_yaml_configs.items()):
        config_ids = sorted({cfg_id for _, cfg_id in records})
        if len(config_ids) > 1:
            joined_records = ", ".join(f"{idx}:{cfg_id}" for idx, cfg_id in records)
            errors.append(
                f"Model {name!r} has identical yaml_extra {list(yaml_signature)!r} "
                f"across different config_id values {config_ids!r} at entries: {joined_records}."
            )

    return errors


def main() -> int:
    args = parse_args()

    try:
        registry = load_registry(args.path)
    except ValueError as error:
        print(f"Model registry validation failed: {error}", file=sys.stderr)
        return 1

    errors = validate_models(registry.get("models"))
    if errors:
        print(f"Model registry validation failed for {args.path}:", file=sys.stderr)
        for error in errors:
            print(f" - {error}", file=sys.stderr)
        return 1

    print(f"Model registry validation passed for {args.path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
