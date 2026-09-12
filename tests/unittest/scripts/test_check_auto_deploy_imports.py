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
from types import ModuleType
from unittest import mock

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_auto_deploy_imports.py"


@pytest.fixture()
def mod() -> ModuleType:
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
    mod: ModuleType, tmp_path: Path, error: OSError | UnicodeDecodeError
) -> None:
    source = tmp_path / "model.py"

    with mock.patch.object(Path, "read_text", side_effect=error):
        violations = mod._check_file(source)

    assert violations == [(1, f"failed to read file: {error}")]


@pytest.mark.parametrize(
    "error",
    [
        OSError("permission denied"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
)
def test_main_fails_when_auto_deploy_file_cannot_be_read(
    mod: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: OSError | UnicodeDecodeError,
) -> None:
    ad_root = tmp_path / "tensorrt_llm" / "_torch" / "auto_deploy"
    ad_root.mkdir(parents=True)
    source = ad_root / "model.py"
    source.touch()

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "AD_ROOT", ad_root)

    with mock.patch.object(Path, "read_text", side_effect=error):
        assert mod.main(["prog", str(source)]) == 1
