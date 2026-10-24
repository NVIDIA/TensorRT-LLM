# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A resumed run keeps a current skills copy instead of rewriting it (it may be read-only)."""

from agent_flow.workflows.trtllm_modeling_v2.run.stages.common import _same_tree


def test_same_tree_compares_files_and_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        (d / "refs").mkdir(parents=True)
        (d / "SKILL.md").write_text("x\n")
        (d / "refs" / "r.md").write_text("y\n")
    assert _same_tree(a, b)
    (b / "refs" / "r.md").write_text("changed\n")
    assert not _same_tree(a, b)
    (b / "refs" / "r.md").write_text("y\n")
    (b / "extra.md").write_text("z\n")
    assert not _same_tree(a, b)
    assert not _same_tree(a, tmp_path / "missing")
