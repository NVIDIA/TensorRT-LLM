# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Engram host table: chunk geometry, the shard reader, and two ranks sharing one host copy."""

from __future__ import annotations

import threading
import uuid

import pytest
import torch
from cuda.bindings import runtime as cudart

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash import engram_host as EH
from tensorrt_llm._utils import CUASSERT

#: The checkpoint's Engram table shape.
ROWS, COLS = 164994380, 96
#: Fake tables: eight shards of one chunk each, so both of two ranks own non-empty chunks.
ROWS_PER_SHARD, SHARDS = 65536, 8


def _fake_shards(seed: int) -> tuple:
    """`(data, keys, weights)`: a table as equal contiguous row shards, in bytes."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(
        0, 256, (SHARDS * ROWS_PER_SHARD, COLS * 2), generator=g, dtype=torch.uint8
    )
    keys = [f"s{i}" for i in range(SHARDS)]
    weights = {
        k: data[i * ROWS_PER_SHARD : (i + 1) * ROWS_PER_SHARD].view(torch.bfloat16)
        for i, k in enumerate(keys)
    }
    return data, keys, weights


def _shard_rows(weights, keys):
    return EH.shard_rows(weights, keys, ROWS_PER_SHARD, lambda entry, cut: entry[cut])


@pytest.mark.cpu_only
def test_chunks_tile_the_table():
    ranges = [EH.chunk_range(k, ROWS, COLS) for k in range(EH.CHUNKS)]
    assert ranges[0][0] == 0 and ranges[-1][1] == ROWS
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert EH.chunk_rows(ROWS, COLS) * COLS * 2 % EH.CHUNK_ALIGN == 0
    for world in (1, 2, 4, 8):
        owned = [k for r in range(world) for k in EH.owned_chunks(r, world)]
        assert owned == list(range(EH.CHUNKS))


@pytest.mark.cpu_only
def test_shard_rows_crosses_shard_boundaries():
    data, keys, weights = _fake_shards(seed=3)
    copy, read = _shard_rows(weights, keys)
    lo, hi = ROWS_PER_SHARD - 5, 2 * ROWS_PER_SHARD + 7
    window = torch.zeros(hi - lo, COLS, dtype=torch.bfloat16)
    copy(window, lo, hi, lo)
    assert torch.equal(window.view(torch.uint8), data[lo:hi])
    assert torch.equal(read(lo, hi).view(torch.uint8), data[lo:hi])


def _vmm_host_numa() -> bool:
    try:
        EH._VmmChunks(torch.cuda.current_device(), EH.CHUNK_ALIGN, 1, 0).close()
    except (RuntimeError, AttributeError):
        return False
    return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host memory the GPU maps")
def test_two_ranks_share_one_host_table(monkeypatch):
    """Each rank creates its half and attaches the other's; both gather the checkpoint's rows
    from host memory."""
    if not _vmm_host_numa():
        pytest.skip("no CUDA VMM host NUMA allocation on this platform")
    monkeypatch.setattr(EH, "_ATTACH_TIMEOUT_S", 60.0)
    data, keys, weights = _fake_shards(seed=7)
    rows, name = data.shape[0], f"test-{uuid.uuid4().hex}"
    tables = [
        EH.EngramHostTable(
            rows, COLS, torch.bfloat16, torch.device("cuda"), rank=r, world=2, name=name
        )
        for r in range(2)
    ]
    threads = [
        threading.Thread(target=t.fill, args=_shard_rows(weights, keys), daemon=True)
        for t in tables
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(300)
    try:
        assert not any(thread.is_alive() for thread in threads), "a rank's fill did not finish"
        expected = data.cuda().view(torch.bfloat16)
        ids = torch.tensor(
            [[0, ROWS_PER_SHARD, rows // 2], [-1, rows - 1, 5 * ROWS_PER_SHARD]], device="cuda"
        )
        want = expected.index_select(0, ids.clamp_min(0).reshape(-1)).view(*ids.shape, COLS)
        for table in tables:
            (attributes,) = CUASSERT(cudart.cudaPointerGetAttributes(table.table.data_ptr()))
            assert int(attributes.type) == int(cudart.cudaMemoryType.cudaMemoryTypeHost)
            assert torch.equal(table.table.view(torch.uint8), expected.view(torch.uint8))
            assert torch.equal(table.gather(ids).view(torch.uint8), want.view(torch.uint8))
    finally:
        for table in tables:
            table.close()
