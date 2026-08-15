#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_model_registry.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_model_registry", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_models_rejects_padded_name():
    module = _load_module()
    errors = module.validate_models([{"name": "model ", "yaml_extra": ["config.yaml"]}])

    assert "models[1]: 'name' must not have leading or trailing whitespace." in errors


def test_validate_models_rejects_padded_config_id():
    module = _load_module()
    errors = module.validate_models(
        [{"name": "model", "yaml_extra": ["config.yaml"], "config_id": " default"}]
    )

    assert "models[1]: 'config_id' must not have leading or trailing whitespace." in errors
