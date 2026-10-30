# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the trtllm_modeling_v2 package tracks: no session placeholders, no one's home directory.

A sandboxed session leaves empty `.mcp.json` and `.claude/*` files in its
working directory, and a commit that stages by directory picks them up. A path
under one operator's home or project scratch works on exactly one machine.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[3] / "agent_flow" / "workflows" / "trtllm_modeling_v2"
TESTS = Path(__file__).resolve().parent

#: A path under a user's home, or under a named cluster's per-user storage.
PERSONAL_PATH = re.compile(r"/home/[a-z][a-z0-9_-]*/|/(?:scratch|lustre)/fsw/portfolios/[a-z]")


def _tracked(directory: Path) -> list[Path]:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "."],
        cwd=directory,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not inside a git checkout")
    return [directory / name for name in result.stdout.decode().split("\0") if name]


def test_no_session_placeholder_is_tracked():
    tracked = _tracked(PACKAGE)
    placeholders = [
        path.relative_to(PACKAGE)
        for path in tracked
        if path.name == ".mcp.json" or ".claude" in path.relative_to(PACKAGE).parts
    ]
    assert not placeholders, f"session placeholders tracked: {placeholders}"
    empty = [
        path.relative_to(PACKAGE)
        for path in tracked
        if path.is_file() and path.stat().st_size == 0 and path.name != "__init__.py"
    ]
    assert not empty, f"empty files tracked: {empty}"


@pytest.mark.parametrize("root", [PACKAGE, TESTS], ids=["package", "tests"])
def test_no_tracked_file_names_a_personal_path(root):
    offences = []
    for path in _tracked(root):
        if path.suffix not in {".py", ".md", ".yaml", ".sh", ".json"} or not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if PERSONAL_PATH.search(line):
                offences.append(f"{path.relative_to(root)}:{number}: {line.strip()[:100]}")
    assert not offences, "personal paths:\n  " + "\n  ".join(offences)


def test_raw_run_reports_do_not_ship_and_their_lessons_do():
    """Run reports stay out of the package; the lessons drawn from them are linked and shipped."""
    assert not (PACKAGE / "findings").exists()
    assert "(LESSONS.md)" in (PACKAGE / "README.md").read_text(encoding="utf-8")
    pyproject = PACKAGE.parents[2] / "pyproject.toml"
    assert '"LESSONS.md"' in pyproject.read_text(encoding="utf-8")


def test_the_readme_flags_table_is_the_cli():
    """Every flag the CLI accepts is in the README's table, and the table names no other."""
    import argparse

    from agent_flow.workflows.trtllm_modeling_v2 import cli

    captured: list[argparse.ArgumentParser] = []
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        captured.append(self)
        return original(self, ["--task", "t.yaml"], namespace)

    argparse.ArgumentParser.parse_args = capture
    try:
        cli._parse_args([])
    finally:
        argparse.ArgumentParser.parse_args = original
    accepted = {
        option
        for action in captured[0]._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }

    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    table = readme.split("## Flags", 1)[1].split("\n\n", 2)[1]
    documented = set(re.findall(r"`(--[a-z-]+)`", table))
    assert documented == accepted
    for subcommand in cli.SUBCOMMANDS:
        assert f"`trtllm-modeling-v2 {subcommand} " in readme


def test_the_readme_has_no_placeholder_left():
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    assert "To be written" not in readme
    assert "workstream" not in readme
