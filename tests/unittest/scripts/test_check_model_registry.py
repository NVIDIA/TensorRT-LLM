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

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_model_registry.py"


@pytest.fixture()
def mod():
    spec = importlib.util.spec_from_file_location("check_model_registry", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_models_allows_same_name_with_different_config_id(mod):
    models = [
        {"name": "meta-llama/Llama-3.1-8B-Instruct", "yaml_extra": ["world_size_1.yaml"]},
        {
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "config_id": "fp8",
            "yaml_extra": ["world_size_1.yaml", "fp8.yaml"],
        },
    ]

    errors = mod.validate_models(models)

    assert errors == []


def test_validate_models_rejects_duplicate_default_config_id(mod):
    models = [
        {"name": "Qwen/Qwen3-8B", "yaml_extra": ["world_size_1.yaml"]},
        {"name": "Qwen/Qwen3-8B", "yaml_extra": ["world_size_2.yaml"]},
    ]

    errors = mod.validate_models(models)

    assert len(errors) == 1
    assert "Duplicate model/config pair" in errors[0]
    assert "'default'" in errors[0]


def test_validate_models_rejects_duplicate_explicit_config_id(mod):
    models = [
        {
            "name": "Qwen/Qwen3-8B",
            "config_id": "benchmark_a",
            "yaml_extra": ["world_size_1.yaml"],
        },
        {
            "name": "Qwen/Qwen3-8B",
            "config_id": "benchmark_a",
            "yaml_extra": ["world_size_2.yaml"],
        },
    ]

    errors = mod.validate_models(models)

    assert len(errors) == 1
    assert "Duplicate model/config pair" in errors[0]
    assert "'benchmark_a'" in errors[0]


def test_validate_models_rejects_empty_config_id(mod):
    models = [
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "config_id": "   ",
            "yaml_extra": ["world_size_1.yaml"],
        }
    ]

    errors = mod.validate_models(models)

    assert len(errors) == 1
    assert "'config_id' must be a non-empty string" in errors[0]


def test_validate_models_rejects_same_model_same_yaml_extra_different_config_id(mod):
    models = [
        {
            "name": "Qwen/Qwen3-8B",
            "config_id": "cfg_a",
            "yaml_extra": ["dashboard_default.yaml", "world_size_2.yaml"],
        },
        {
            "name": "Qwen/Qwen3-8B",
            "config_id": "cfg_b",
            "yaml_extra": ["dashboard_default.yaml", "world_size_2.yaml"],
        },
    ]

    errors = mod.validate_models(models)

    assert len(errors) == 1
    assert "identical yaml_extra" in errors[0]
    assert "'cfg_a'" in errors[0]
    assert "'cfg_b'" in errors[0]


@pytest.mark.parametrize("name", [" model", "model "])
def test_validate_models_rejects_padded_name(mod, name):
    errors = mod.validate_models([{"name": name, "yaml_extra": ["config.yaml"]}])

    assert "models[1]: 'name' must not have leading or trailing whitespace." in errors


@pytest.mark.parametrize("config_id", [" default", "default "])
def test_validate_models_rejects_padded_config_id(mod, config_id):
    errors = mod.validate_models(
        [{"name": "model", "yaml_extra": ["config.yaml"], "config_id": config_id}]
    )

    assert "models[1]: 'config_id' must not have leading or trailing whitespace." in errors


def test_invalid_padded_name_is_excluded_from_duplicate_tracking(mod):
    errors = mod.validate_models(
        [
            {"name": "model ", "yaml_extra": ["config.yaml"]},
            {"name": "model ", "yaml_extra": ["config.yaml"]},
        ]
    )

    assert errors.count("models[1]: 'name' must not have leading or trailing whitespace.") == 1
    assert errors.count("models[2]: 'name' must not have leading or trailing whitespace.") == 1
    assert not any("Duplicate model/config pair" in error for error in errors)
    assert not any("identical yaml_extra" in error for error in errors)


def test_invalid_padded_config_id_is_excluded_from_duplicate_tracking(mod):
    errors = mod.validate_models(
        [
            {
                "name": "model",
                "yaml_extra": ["config.yaml"],
                "config_id": "default ",
            },
            {
                "name": "model",
                "yaml_extra": ["config.yaml"],
                "config_id": "default ",
            },
        ]
    )

    assert errors.count(
        "models[1]: 'config_id' must not have leading or trailing whitespace."
    ) == 1
    assert errors.count(
        "models[2]: 'config_id' must not have leading or trailing whitespace."
    ) == 1
    assert not any("Duplicate model/config pair" in error for error in errors)
    assert not any("identical yaml_extra" in error for error in errors)
