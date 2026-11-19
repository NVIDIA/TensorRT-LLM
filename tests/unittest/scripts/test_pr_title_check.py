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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "pr_title_check.py"


@pytest.fixture()
def mod():
    spec = importlib.util.spec_from_file_location("pr_title_check", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "title",
    [
        "[TRTLLM-1234][feat] Add new feature",
        "[TRTLLM-1234] [fix] Fix some bugs",
        "[https://nvbugs/1234567][fix] Fix some bugs",
        "[#1234][doc] Update documentation",
        "[None][chore] Minor clean-up",
    ],
)
def test_validate_pr_title_accepts_valid_titles(mod, title: str):
    assert mod.validate_pr_title(title) == []


@pytest.mark.parametrize(
    ("title", "expected_substring"),
    [
        ("", "empty"),
        ("[Fix][feat] Add feature", "where a ticket is expected"),
        ("[fix][feat] Add feature", "where a ticket is expected"),
        ("[TRTLLM-1234][feat]", "Missing summary"),
        (" [TRTLLM-1234][feat] Add feature", "leading whitespace"),
        ("[TRTLLM-1234][feat] Add feature ", "trailing whitespace"),
        ("[TRTLLM-1234][feat]  Add feature", "single space"),
        ("[trtllm-1234][feat] Add feature", "uppercase project key"),
        ("TRTLLM-1234][feat] Add feature", "square brackets"),
        ("[TRTLLM-1234] feat] Add feature", "type in square brackets"),
        ("[TRTLLM1234][feat] Add feature", "missing a hyphen"),
        ("[https://nvbugs/][fix] Fix bug", "NVBugs ticket must be the full URL"),
    ],
)
def test_validate_pr_title_reports_specific_errors(mod, title: str, expected_substring: str):
    errors = mod.validate_pr_title(title)
    assert errors
    assert any(expected_substring in error for error in errors)
