#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_skill_naming_convention.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_skill_naming_convention", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_reports_malformed_yaml_without_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    malformed = tmp_path / "skills" / "perf-example" / "SKILL.md"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("---\nname: [unterminated\n---\nBody\n", encoding="utf-8")

    later = tmp_path / "skills" / "perf-later" / "SKILL.md"
    later.parent.mkdir(parents=True)
    later.write_text("---\nname: perf-wrong\n---\nBody\n", encoding="utf-8")

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "NAMING_DOC", tmp_path / ".claude" / "README.md")
    monkeypatch.setattr(
        module,
        "collect_items",
        lambda: [(malformed, "perf-example"), (later, "perf-later")],
    )

    assert module.main() == 1
    stderr = capsys.readouterr().err
    assert "invalid YAML frontmatter" in stderr
    assert "skills/perf-example/SKILL.md" in stderr
    assert "skills/perf-later/SKILL.md" in stderr


@pytest.mark.parametrize("frontmatter", ["- item", "scalar", "7"])
def test_main_reports_non_mapping_yaml_frontmatter(
    frontmatter: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    skill = tmp_path / "skills" / "perf-example" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(f"---\n{frontmatter}\n---\nBody\n", encoding="utf-8")

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "NAMING_DOC", tmp_path / ".claude" / "README.md")
    monkeypatch.setattr(module, "collect_items", lambda: [(skill, "perf-example")])

    assert module.main() == 1
    assert "invalid YAML frontmatter: frontmatter root must be a mapping" in (
        capsys.readouterr().err
    )
