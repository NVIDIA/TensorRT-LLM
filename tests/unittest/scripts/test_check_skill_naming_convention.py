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

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest import mock

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
    assert "frontmatter line" in stderr
    assert "skills/perf-example/SKILL.md" in stderr
    assert "skills/perf-later/SKILL.md" in stderr


@pytest.mark.parametrize(
    "frontmatter", ["- item", "scalar", "7", "null", "[]", "false", "0", '""']
)
def test_main_reports_non_mapping_frontmatter(
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
    assert "invalid frontmatter: frontmatter root must be a mapping" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize(
    "error",
    [
        OSError("permission denied"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
)
def test_main_reports_frontmatter_read_errors(
    error: OSError | UnicodeDecodeError,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    skill = tmp_path / "skills" / "perf-example" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.touch()

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "NAMING_DOC", tmp_path / ".claude" / "README.md")
    monkeypatch.setattr(module, "collect_items", lambda: [(skill, "perf-example")])

    with mock.patch.object(Path, "read_text", side_effect=error):
        assert module.main() == 1

    stderr = capsys.readouterr().err
    assert "skills/perf-example/SKILL.md" in stderr
    assert "failed to read frontmatter" in stderr
