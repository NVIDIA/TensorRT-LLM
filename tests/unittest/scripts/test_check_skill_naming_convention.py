#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_skill_naming_convention.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_skill_naming_convention", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_frontmatter_name_accepts_crlf(tmp_path):
    module = _load_module()
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"---\r\nname: perf-example\r\n---\r\nBody\r\n")

    assert module.load_frontmatter_name(skill) == "perf-example"
