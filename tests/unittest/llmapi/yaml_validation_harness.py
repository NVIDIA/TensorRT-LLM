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
"""Shared YAML validation helpers for llmapi config tests."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from tensorrt_llm.llmapi import llm_args as llm_args_module


def collect_yaml_files(
    parent_dir: Path, include_glob: str, exclude_names: Iterable[str] = ()
) -> list[Path]:
    excluded = set(exclude_names)
    if not parent_dir.exists():
        return []

    return sorted(
        path
        for path in parent_dir.glob(include_glob)
        if path.is_file() and path.name not in excluded
    )


def load_yaml_dict(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"YAML at {path} must parse to a dict, got {type(loaded).__name__}")
    return loaded


def validate_torch_llm_args_config(cfg: dict[str, Any]) -> llm_args_module.TorchLlmArgs:
    base_args = llm_args_module.TorchLlmArgs(model="dummy/model", skip_tokenizer_init=True)
    merged = llm_args_module.update_llm_args_with_extra_dict(
        copy.deepcopy(base_args.model_dump(mode="json")), copy.deepcopy(cfg)
    )
    if "load_format" in merged:
        merged["load_format"] = llm_args_module.LoadFormat(merged["load_format"])
    return llm_args_module.TorchLlmArgs(**merged)


def assert_no_deprecated_keys(cfg: dict[str, Any], deprecated_map: dict[str, str]) -> None:
    violations: list[str] = []

    def walk_dict(node: dict[str, Any], path: tuple[str, ...]) -> None:
        for key, value in node.items():
            full_path = path + (str(key),)
            if key in deprecated_map:
                replacement = deprecated_map[key]
                violations.append(f"{'.'.join(full_path)} -> {replacement}")

            if isinstance(value, dict):
                walk_dict(value, full_path)

    walk_dict(cfg, ())
    if violations:
        raise AssertionError("Found deprecated config keys:\n" + "\n".join(sorted(violations)))


def assert_no_default_valued_leaves(
    cfg: dict[str, Any],
    default_cfg: dict[str, Any],
    allowlist: Iterable[str | tuple[str, ...]] = (),
) -> None:
    normalized_allowlist: set[tuple[str, ...]] = set()
    for item in allowlist:
        if isinstance(item, str):
            normalized_allowlist.add(tuple(item.split(".")))
        else:
            normalized_allowlist.add(tuple(item))

    violations: list[str] = []

    def walk(candidate: dict[str, Any], defaults: dict[str, Any], path: tuple[str, ...]) -> None:
        for key, value in candidate.items():
            full_path = path + (str(key),)
            if full_path in normalized_allowlist:
                continue

            if key not in defaults:
                continue
            default_value = defaults[key]

            if isinstance(value, dict) and isinstance(default_value, dict):
                walk(value, default_value, full_path)
                continue

            if value == default_value:
                violations.append(".".join(full_path))

    walk(cfg, default_cfg, ())
    if violations:
        raise AssertionError(
            "Found default-valued config leaves:\n" + "\n".join(sorted(violations))
        )


def assert_custom_policy(cfg: dict[str, Any], policy_fn: Callable[[dict[str, Any]], None]) -> None:
    policy_fn(cfg)


@contextmanager
def mock_cuda_for_schema_validation(
    device_count: int = 8, compute_capability_major: int = 8, is_available: bool = True
) -> Iterator[None]:
    mock_props = mock.Mock()
    mock_props.major = compute_capability_major

    with mock.patch("torch.cuda.device_count", return_value=device_count):
        with mock.patch("torch.cuda.get_device_properties", return_value=mock_props):
            with mock.patch("torch.cuda.is_available", return_value=is_available):
                yield
