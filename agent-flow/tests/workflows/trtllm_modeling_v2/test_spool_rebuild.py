# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native rebuilds that do not crash the processes holding the old libraries.

A rebuild that rewrites ``tensorrt_llm/libs/*.so`` in place makes a process on
another session that has them mapped take SIGSEGV, taking its whole
allocation with it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import rebuild


def test_rebuild_markers_hold_each_spool_until_expiry_or_standalone_retraction(tmp_path):
    tree = tmp_path / "tree"
    package = tree / "tensorrt_llm"
    package.mkdir(parents=True)
    native = tmp_path / "native-libs"
    native.mkdir()
    (package / "libs").symlink_to(native, target_is_directory=True)
    spools = [tmp_path / "a", tmp_path / "b"]
    script = str(Path(rebuild.__file__))
    args = [item for spool in spools for item in ("--spool", str(spool))]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    begin = subprocess.run(
        [sys.executable, "-S", script, "begin", "--tree", str(tree), "--max-age", "60", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    token = begin.stdout.strip()
    records = [json.loads((spool / "rebuild" / f"{token}.json").read_text()) for spool in spools]
    assert records[0] == records[1]
    record = records[0]
    assert record["token"] == token and record["tree"] == str(tree)
    assert record["native"] == sorted([str(package), str(native)])
    for spool in spools:
        directory = spool / "rebuild"
        (directory / "broken.json").write_text("{")
        (directory / "list.json").write_text("[]")
        assert rebuild.active_announcements(spool, now=record["started"] + 60) == [record]
        assert rebuild.active_announcements(spool, now=record["started"] + 61) == []
        assert not list(directory.glob(".*.tmp"))
    subprocess.run(
        [sys.executable, "-S", script, "end", "--token", token, *args],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    for spool in spools:
        assert rebuild.active_announcements(spool, now=record["started"]) == []
        assert not (spool / "rebuild" / f"{token}.json").exists()


def test_a_fresh_inode_leaves_a_mapped_library_untouched(tmp_path):
    lib = tmp_path / "libs" / "libth_common.so"
    lib.parent.mkdir()
    lib.write_bytes(b"old-library-bytes")
    os.chmod(lib, 0o755)
    held = lib.open("rb")  # a running process's mapping
    before = os.stat(lib).st_ino
    try:
        refreshed = rebuild.refresh_inodes(tmp_path)
        assert refreshed == [lib]
        assert os.stat(lib).st_ino != before
        assert lib.read_bytes() == b"old-library-bytes"
        assert os.stat(lib).st_mode & 0o777 == 0o755
        # The build then rewrites the file in place -- the new inode, not the held one.
        with lib.open("r+b") as handle:
            handle.write(b"NEW")
        held.seek(0)
        assert held.read() == b"old-library-bytes"
    finally:
        held.close()
    assert not list(lib.parent.glob(".*.tmp"))
