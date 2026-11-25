# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The node-share key of an Engram table names the checkpoint file that holds it, as stock's
``_engram_table_share_key``: rewriting that file at the same path changes the key (so a process never maps a table
another process built from different contents), and a checkpoint whose index does not name the file gets no key (a
private table)."""

import json
import os

import pytest

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
    table_share_key,
)

pytestmark = pytest.mark.cpu_only


def _checkpoint(root, layer=3, shard="model-00002-of-00048.safetensors", content=b"rows-v1"):
    weight_map = {f"layers.{layer}.engram.embed.weight": shard}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / shard).write_bytes(content)
    return root / shard


def test_the_key_names_the_file_and_layer(tmp_path):
    path = _checkpoint(tmp_path)
    key = table_share_key(str(tmp_path), 3)
    stat = path.stat()
    assert (
        key
        == f"{path.resolve()}:{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}:layer3"
    )


def test_rewriting_the_checkpoint_at_the_same_path_changes_the_key(tmp_path):
    path = _checkpoint(tmp_path)
    before = table_share_key(str(tmp_path), 3)
    # A new file under the old name (a re-download or re-export): new inode, size and mtime.
    replacement = tmp_path / "replacement.tmp"
    replacement.write_bytes(b"rows-v2, longer")
    stat = path.stat()
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    os.replace(replacement, path)
    after = table_share_key(str(tmp_path), 3)
    assert before is not None and after is not None and before != after


@pytest.mark.parametrize("case", ["no-dir", "no-index", "layer-not-indexed", "file-missing"])
def test_no_key_without_an_indexed_file(tmp_path, case):
    if case == "no-dir":
        assert table_share_key(None, 3) is None
        return
    if case != "no-index":
        path = _checkpoint(tmp_path)
        if case == "file-missing":
            path.unlink()
    layer = 4 if case == "layer-not-indexed" else 3
    assert table_share_key(str(tmp_path), layer) is None
