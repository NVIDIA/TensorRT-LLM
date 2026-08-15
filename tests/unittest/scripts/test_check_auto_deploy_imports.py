#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_auto_deploy_imports.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_auto_deploy_imports", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_file_reports_read_errors(tmp_path):
    module = _load_module()
    source = tmp_path / "model.py"

    with mock.patch.object(Path, "read_text", side_effect=OSError("permission denied")):
        violations = module._check_file(source)

    assert violations == [(1, "failed to read file: permission denied")]
