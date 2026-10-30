# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native rebuilds that do not crash the processes holding the old libraries.

A rebuild that rewrites ``tensorrt_llm/libs/*.so`` in place makes a process on
another session that has them mapped take SIGSEGV, taking its whole
allocation with it.
"""

from __future__ import annotations

import os

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import rebuild


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
