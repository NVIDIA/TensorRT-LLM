# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the remote metadata command emitted by the Jenkins pipeline."""

import os
import re
import shlex
import subprocess
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_metadata(path: Path) -> subprocess.CompletedProcess[str]:
    source = (_REPO_ROOT / "jenkins" / "L0_Test.groovy").read_text()
    reader = source.split("def readSlurmWorkspaceFileResult(", 1)[1]
    reader = reader.split("def readSlurmWorkspaceFile(", 1)[0]
    command = re.search(r'Utils\.bashWrappedRemoteCmd\("([^"\n]*)"\)', reader)
    assert command is not None
    script = command.group(1).replace("${path}", shlex.quote(str(path)))
    script = script.replace("${quotedPath}", shlex.quote(str(path)))
    if "${metadataReader}" in script:
        program = re.search(r"def metadataReader = '''(.*?)'''", reader, re.S)
        assert program is not None
        code = textwrap.dedent(program.group(1))
        code = "\n".join(line.removeprefix("|") for line in code.splitlines()).strip()
        script = script.replace("${metadataReader}", code)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize("content", [None, "", "2\n", "node[1-2]\nnode3\n"])
def test_metadata_reader_accepts_missing_empty_and_readable_files(tmp_path, content):
    path = tmp_path / "restart marker's value.txt"
    if content is not None:
        path.write_text(content)
    result = _read_metadata(path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == (content or "")


@pytest.mark.skipif(os.geteuid() == 0, reason="Root bypasses file permission checks")
@pytest.mark.parametrize("unreadable_parent", [False, True])
def test_metadata_reader_propagates_permission_errors(tmp_path, unreadable_parent):
    parent = tmp_path / "metadata"
    parent.mkdir()
    path = parent / "slurm_restart_count.txt"
    path.write_text("1\n")
    blocked_path = parent if unreadable_parent else path
    blocked_path.chmod(0)
    try:
        result = _read_metadata(path)
    finally:
        blocked_path.chmod(0o700 if unreadable_parent else 0o600)
    assert result.returncode != 0
    assert result.stdout == ""


def test_metadata_reader_propagates_non_file_errors(tmp_path):
    result = _read_metadata(tmp_path)
    assert result.returncode != 0
    assert result.stdout == ""
