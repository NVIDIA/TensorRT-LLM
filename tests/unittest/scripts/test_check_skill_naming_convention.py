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


def test_main_reports_malformed_yaml_without_crashing(tmp_path, monkeypatch, capsys):
    module = _load_module()
    skill = tmp_path / "skills" / "perf-example" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: [unterminated\n---\nBody\n", encoding="utf-8")

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "NAMING_DOC", tmp_path / ".claude" / "README.md")
    monkeypatch.setattr(module, "collect_items", lambda: [(skill, "perf-example")])

    assert module.main() == 1
    assert "invalid YAML frontmatter" in capsys.readouterr().err
