#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_model_registry.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_model_registry", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [" model", "model "])
def test_validate_models_rejects_padded_name(name: str) -> None:
    module = _load_module()
    errors = module.validate_models([{"name": name, "yaml_extra": ["config.yaml"]}])

    assert "models[1]: 'name' must not have leading or trailing whitespace." in errors


@pytest.mark.parametrize("config_id", [" default", "default "])
def test_validate_models_rejects_padded_config_id(config_id: str) -> None:
    module = _load_module()
    errors = module.validate_models(
        [{"name": "model", "yaml_extra": ["config.yaml"], "config_id": config_id}]
    )

    assert "models[1]: 'config_id' must not have leading or trailing whitespace." in errors


def test_invalid_padded_name_is_excluded_from_duplicate_tracking() -> None:
    module = _load_module()
    errors = module.validate_models(
        [
            {"name": "model ", "yaml_extra": ["config.yaml"]},
            {"name": "model ", "yaml_extra": ["config.yaml"]},
        ]
    )

    assert errors.count("models[1]: 'name' must not have leading or trailing whitespace.") == 1
    assert errors.count("models[2]: 'name' must not have leading or trailing whitespace.") == 1
    assert not any("Duplicate model/config pair" in error for error in errors)
    assert not any("identical yaml_extra" in error for error in errors)


def test_invalid_padded_config_id_is_excluded_from_duplicate_tracking() -> None:
    module = _load_module()
    errors = module.validate_models(
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
