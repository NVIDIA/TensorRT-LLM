# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""modeling_v2 is ``tensorrt_llm/_torch/modeling_v2``; nothing names it under ``_torch/_experimental``.

A path or module name that points at ``_torch/_experimental/modeling_v2`` fails
only when the code that holds it runs: an import inside a target, a glob in a
source check that then matches nothing, a CODEOWNERS or CI rule that stops
applying. These checks read the checkout and import nothing, so they run on any
CPU machine. ``docs/`` is not read: release notes may name the old location.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_REPO = Path(__file__).resolve().parents[4]
_OLD_PACKAGE = _REPO / "tensorrt_llm/_torch/_experimental/modeling_v2"

# The old location with any separator (paths, dotted and relative module names,
# escaped regexes), an import of the package from its parent, and the two path
# components written as separate strings, on one line or across lines.
_STALE = re.compile(
    r"(?<!\w)_experimental[./\\]+modeling_v2(?!\w)"
    r"|from\s+\.+_experimental\s+import\s+modeling_v2(?!\w)"
    r"|[\"']_experimental[\"']\s*[,/]\s*[\"']modeling_v2[\"']"
)
_UNREAD_TOP_DIRS = ("3rdparty", "docs")
_UNREAD_DIR_NAMES = frozenset({".git", "__pycache__", "build", "node_modules"})
_LARGEST_READ = 4 << 20  # source and config files are far smaller; skips binaries in a snapshot
_SOURCE_SUFFIXES = frozenset({".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"})


def _is_git_work_tree_root() -> bool:
    try:
        top = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        return False
    return Path(top.strip()).resolve() == _REPO


def _files_mentioning_experimental():
    """Files that contain ``_experimental`` at all, outside ``_UNREAD_TOP_DIRS``.

    Tracked files when the checkout is a git work tree of its own, so a
    developer's untracked build and test outputs are not read; every file
    otherwise, as in a source snapshot without ``.git``.
    """
    if _is_git_work_tree_root():
        excludes = [f":(exclude){d}" for d in _UNREAD_TOP_DIRS]
        listed = subprocess.run(
            ["git", "-C", str(_REPO), "grep", "-l", "-z", "-I", "-F", "-e", "_experimental"]
            + ["--", "."]
            + excludes,
            capture_output=True,
        )
        assert listed.returncode in (0, 1), listed.stderr.decode()  # 1: no file matched
        return [_REPO / name for name in listed.stdout.decode().split("\0") if name]
    found = []
    for directory, subdirectories, files in os.walk(_REPO):
        at_top = Path(directory) == _REPO
        subdirectories[:] = [
            d
            for d in subdirectories
            if d not in _UNREAD_DIR_NAMES and not (at_top and d in _UNREAD_TOP_DIRS)
        ]
        for name in files:
            path = Path(directory, name)
            try:
                if path.stat().st_size <= _LARGEST_READ and b"_experimental" in path.read_bytes():
                    found.append(path)
            except OSError:
                continue
    return found


def test_no_source_file_is_left_under_the_old_location():
    assert (_REPO / "tensorrt_llm/_torch/modeling_v2/__init__.py").is_file()
    # A checkout updated across the move keeps empty directories, __pycache__
    # and build outputs there; only a source file is a problem.
    sources = sorted(
        str(path.relative_to(_REPO))
        for path in (_OLD_PACKAGE.rglob("*") if _OLD_PACKAGE.is_dir() else ())
        if path.is_file() and (path.suffix in _SOURCE_SUFFIXES or path.name == "CMakeLists.txt")
    )
    assert not sources, (
        "modeling_v2 is tensorrt_llm/_torch/modeling_v2; move these files there:\n"
        + "\n".join(sources)
    )


def test_nothing_names_the_old_location():
    this_file = Path(__file__).resolve()
    stale = []
    for path in _files_mentioning_experimental():
        if path.resolve() == this_file or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for match in _STALE.finditer(text):
            number = text.count("\n", 0, match.start()) + 1
            stale.append(f"{path.relative_to(_REPO)}:{number}: {lines[number - 1].strip()}")
    assert not stale, (
        "modeling_v2 is tensorrt_llm/_torch/modeling_v2 (module "
        "tensorrt_llm._torch.modeling_v2); these lines name the old location:\n" + "\n".join(stale)
    )
