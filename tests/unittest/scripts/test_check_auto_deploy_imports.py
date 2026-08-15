#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_auto_deploy_imports.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_auto_deploy_imports", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "error",
    [
        OSError("permission denied"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
)
def test_check_file_reports_read_errors(
    tmp_path: Path, error: OSError | UnicodeDecodeError
) -> None:
    module = _load_module()
    source = tmp_path / "model.py"

    with mock.patch.object(Path, "read_text", side_effect=error):
        violations = module._check_file(source)

    assert violations == [(1, f"failed to read file: {error}")]
