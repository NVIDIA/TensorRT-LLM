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
EXPECTED_MODEL_KEYS = {"name", "yaml_extra"}


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
    seen_names: dict[str, list[int]] = collections.defaultdict(list)

    for index, model_entry in enumerate(models, start=1):
        entry_label = f"models[{index}]"
        if not isinstance(model_entry, dict):
            errors.append(
                f"{entry_label}: expected a mapping entry, got {type(model_entry).__name__}."
            )
            continue

        entry_keys = set(model_entry)
        missing_keys = sorted(EXPECTED_MODEL_KEYS - entry_keys)
        unexpected_keys = sorted(entry_keys - EXPECTED_MODEL_KEYS)
        if missing_keys or unexpected_keys:
            details: list[str] = []
            if missing_keys:
                details.append(f"missing keys {missing_keys}")
            if unexpected_keys:
                details.append(f"unexpected keys {unexpected_keys}")
            joined_details = ", ".join(details)
            errors.append(
                f"{entry_label}: expected exactly the keys ['name', 'yaml_extra']; {joined_details}."
            )

        name = model_entry.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{entry_label}: missing non-empty string 'name'.")
        yaml_extra = model_entry.get("yaml_extra")
        if not isinstance(yaml_extra, list) or not all(
            isinstance(item, str) and item.strip() for item in yaml_extra
        ):
            errors.append(f"{entry_label}: 'yaml_extra' must be a list of non-empty strings.")

        if not isinstance(name, str) or not name.strip():
            continue

        seen_names[name].append(index)

    for name, indices in sorted(seen_names.items()):
        if len(indices) > 1:
            joined_indices = ", ".join(str(index) for index in indices)
            errors.append(f"Duplicate model name {name!r} found at entries: {joined_indices}.")

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
