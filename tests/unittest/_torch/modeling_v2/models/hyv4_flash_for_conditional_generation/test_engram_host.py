# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Engram host tables: the chunk geometry, the shard reader, and on one device the VMM
mapping the ranks of a host share -- each rank's slice host-resident and not device memory,
the export/attach round trip serving another rank's rows, and the index-select gather over the
mapped table."""

from __future__ import annotations

import os
import threading
import uuid

import pytest
import torch
from cuda.bindings import runtime as cudart

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation import (
    engram_host as EH,
)
from tensorrt_llm._utils import CUASSERT

#: The checkpoint's four Engram tables: BF16 [164994380, 96], 31.7 GB each.
ROWS, COLS = 164994380, 96
#: The fake tables: eight shards of one chunk each (96 MiB at `COLS`), so every rank of two
#: owns real, non-empty chunks.
ROWS_PER_SHARD, SHARDS = 65536, 8


def _fake_shards(
    seed: int, cols: int = COLS, rows_per_shard: int = ROWS_PER_SHARD, shards: int = SHARDS
) -> tuple:
    """`(data, keys, weights)`: a table as `shards` equal contiguous row shards, in bytes."""
    g = torch.Generator().manual_seed(seed)
    itemsize = torch.empty(0, dtype=torch.bfloat16).element_size()
    data = torch.randint(
        0, 256, (shards * rows_per_shard, cols * itemsize), generator=g, dtype=torch.uint8
    )
    keys = [f"s{i}" for i in range(shards)]
    weights = {
        k: data[i * rows_per_shard : (i + 1) * rows_per_shard].view(torch.bfloat16)
        for i, k in enumerate(keys)
    }
    return data, keys, weights


def _read_entry(entry, cut):
    return entry[cut]


def _shard_rows(weights, keys, rows_per_shard):
    return EH.shard_rows(weights, keys, rows_per_shard, _read_entry)


# -------------------------------------------------------------------------- CPU --


@pytest.mark.cpu_only
def test_chunk_geometry_covers_the_checkpoint_table():
    """Eight aligned chunks tile the table's rows, and every supported host size owns whole,
    non-empty chunks."""
    per = EH.chunk_rows(ROWS, COLS)
    assert per == 20643840, per
    assert per * COLS * 2 == 3963617280
    assert per * COLS * 2 % EH.CHUNK_ALIGN == 0
    ranges = [EH.chunk_range(k, ROWS, COLS) for k in range(EH.CHUNKS)]
    assert ranges[0][0] == 0 and ranges[-1][1] == ROWS
    assert all(lo < hi for lo, hi in ranges)
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    for world in (1, 2, 4, 8):
        owned = [EH.owned_rows(r, world, ROWS, COLS) for r in range(world)]
        assert owned[0][0] == 0 and owned[-1][1] == ROWS
        assert all(lo < hi for lo, hi in owned)
        assert all(a[1] == b[0] for a, b in zip(owned, owned[1:]))
        assert all(len(EH.owned_chunks(r, world)) == EH.CHUNKS // world for r in range(world))


@pytest.mark.cpu_only
def test_shard_rows_reads_and_writes_the_checkpoint_layout():
    """`copy` writes global rows into a table whose row 0 is a given global row, and `read`
    returns them, across shard boundaries."""
    data, keys, weights = _fake_shards(seed=3)
    rows = ROWS_PER_SHARD * SHARDS
    copy, read = _shard_rows(weights, keys, ROWS_PER_SHARD)
    got = torch.zeros(rows, COLS, dtype=torch.bfloat16)
    copy(got, 0, rows, 0)
    assert torch.equal(got.view(torch.uint8), data)
    lo, hi = ROWS_PER_SHARD - 5, 2 * ROWS_PER_SHARD + 7
    window = torch.zeros(hi - lo, COLS, dtype=torch.bfloat16)
    copy(window, lo, hi, lo)
    assert torch.equal(window.view(torch.uint8), data[lo:hi])
    assert torch.equal(read(lo, hi).view(torch.uint8), data[lo:hi])


@pytest.mark.cpu_only
def test_gather_on_a_host_table_equals_torch_index_select():
    """The gather is the table's index-select, and a negative id reads row 0 for the caller to
    mask."""
    rows = 64
    data, keys, weights = _fake_shards(seed=1, rows_per_shard=rows, shards=1)
    table = EH.EngramHostTable(
        rows, COLS, torch.bfloat16, torch.device("cpu"), rank=0, world=1, name="cpu-test"
    )
    table.fill(*_shard_rows(weights, keys, rows))
    assert table.resident_on_host()
    ids = torch.tensor([[0, 1, rows - 1], [-1, rows // 2, 2]])
    got = table.gather(ids)
    want = table.table.index_select(0, ids.clamp_min(0).reshape(-1)).view(*ids.shape, COLS)
    assert got.shape == (*ids.shape, COLS)
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    table.close()


# -------------------------------------------------------------------------- GPU --


def _vmm_host_numa() -> bool:
    try:
        EH._VmmChunks(torch.cuda.current_device(), EH.CHUNK_ALIGN, 1, 0).close()
    except (RuntimeError, AttributeError):
        return False
    return True


def _host_tables(world: int, data, keys, weights, name: str) -> list:
    """`world` ranks' tables in one process, filled in threads (the broker is per host)."""
    rows, cols = data.shape[0], weights[keys[0]].shape[1]
    sources = [_shard_rows(weights, keys, ROWS_PER_SHARD)] * world
    tables = [
        EH.EngramHostTable(
            rows, cols, torch.bfloat16, torch.device("cuda"), rank=r, world=world, name=name
        )
        for r in range(world)
    ]
    threads = [
        threading.Thread(target=t.fill, args=sources[r], daemon=True) for r, t in enumerate(tables)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(300)
    assert not any(thread.is_alive() for thread in threads), "a rank's fill did not finish"
    return tables


def _assert_host_memory(table) -> None:
    """The table is host memory the device addresses, not device memory."""
    assert table.resident_on_host(), table.describe()
    (attributes,) = CUASSERT(cudart.cudaPointerGetAttributes(table.table.data_ptr()))
    assert int(attributes.type) == EH._CUDA_MEMORY_TYPE_HOST, table.describe()
    assert table.table.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host memory the GPU maps")
def test_export_attach_round_trip_serves_another_rank():
    """A chunk created and written by one process is read through another process's mapping of
    the descriptor, so a rank serves rows it does not hold."""
    if not _vmm_host_numa():
        pytest.skip("no CUDA VMM host NUMA allocation on this platform")
    chunk_bytes = 2 * EH.CHUNK_ALIGN
    rows, cols = 1024, COLS
    device = torch.device("cuda")
    owner = EH._VmmChunks(torch.cuda.current_device(), chunk_bytes, 1, 0)
    peer = EH._VmmChunks(torch.cuda.current_device(), chunk_bytes, 1, 0)
    try:
        owner.create(0)
        written = owner.view(rows, cols, torch.bfloat16, device)
        written.fill_(1.5)
        torch.cuda.synchronize()
        fd = owner.export(0)
        try:
            peer.attach(0, fd)
        finally:
            os.close(fd)
        assert owner.resident_on_host() and peer.resident_on_host()
        read_back = peer.view(rows, cols, torch.bfloat16, device)
        assert torch.equal(read_back, written)
    finally:
        owner.close()
        peer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host memory the GPU maps")
def test_every_rank_holds_its_slice_in_host_memory(monkeypatch):
    """Two ranks split the chunks; each creates its half, attaches the other's, and both views
    hold the checkpoint's bytes in host memory."""
    if not _vmm_host_numa():
        pytest.skip("no CUDA VMM host NUMA allocation on this platform")
    monkeypatch.setattr(EH, "_ATTACH_TIMEOUT_S", 60.0)
    data, keys, weights = _fake_shards(seed=5)
    tables = _host_tables(2, data, keys, weights, f"test-{uuid.uuid4().hex}")
    try:
        expected = data.cuda()
        for rank, table in enumerate(tables):
            _assert_host_memory(table)
            assert table.table.shape == (data.shape[0], COLS)
            assert torch.equal(table.table.view(torch.uint8), expected)
            assert sorted(table._created) == list(EH.owned_chunks(rank, 2))
            assert sorted(table._segments) == list(EH.owned_chunks(rank, 2))
            assert table._attached == EH.CHUNKS // 2
            assert table._vmm.complete()
            lo, hi = table.owned_rows
            assert lo < hi, table.describe()
            assert (lo, hi) == EH.owned_rows(rank, 2, data.shape[0], COLS)
    finally:
        for table in tables:
            table.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host memory the GPU maps")
def test_gather_over_the_mapped_table_equals_the_torch_gather(monkeypatch):
    """A gather over either rank's mapping returns the checkpoint's rows, including rows the
    other rank created."""
    if not _vmm_host_numa():
        pytest.skip("no CUDA VMM host NUMA allocation on this platform")
    monkeypatch.setattr(EH, "_ATTACH_TIMEOUT_S", 60.0)
    data, keys, weights = _fake_shards(seed=7)
    rows = data.shape[0]
    tables = _host_tables(2, data, keys, weights, f"test-{uuid.uuid4().hex}")
    try:
        expected = data.cuda().view(torch.bfloat16)
        ids = torch.tensor(
            [[0, ROWS_PER_SHARD, rows // 2], [-1, rows - 1, 5 * ROWS_PER_SHARD]], device="cuda"
        )
        want = expected.index_select(0, ids.clamp_min(0).reshape(-1)).view(*ids.shape, COLS)
        for table in tables:
            got = table.gather(ids)
            assert got.shape == (*ids.shape, COLS)
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), table.describe()
            # The gather clamps: a negative id reads row 0, which the caller masks.
            assert torch.equal(got[1, 0].view(torch.uint8), expected[0].view(torch.uint8))
            masked = got.masked_fill((ids < 0).unsqueeze(-1), 0.0)
            assert int(masked[1, 0].abs().sum()) == 0
    finally:
        for table in tables:
            table.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host memory the GPU maps")
def test_one_rank_owns_and_maps_all_chunks(monkeypatch):
    """`world` 1 needs no broker: the rank creates all chunks and maps the whole table."""
    if not _vmm_host_numa():
        pytest.skip("no CUDA VMM host NUMA allocation on this platform")
    data, keys, weights = _fake_shards(seed=9)
    table = EH.EngramHostTable(
        data.shape[0], COLS, torch.bfloat16, torch.device("cuda"), rank=0, world=1, name="solo"
    )
    try:
        table.fill(*_shard_rows(weights, keys, ROWS_PER_SHARD))
        _assert_host_memory(table)
        assert sorted(table._created) == list(range(EH.CHUNKS))
        assert not table._segments
        assert table._attached == 0
        assert torch.equal(table.table.view(torch.uint8), data.cuda())
    finally:
        table.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host memory the GPU maps")
def test_resident_on_host_distinguishes_device_memory():
    """The check is not vacuous: a device allocation reports device memory."""
    device_table = torch.zeros(8, COLS, dtype=torch.bfloat16, device="cuda")
    (attributes,) = CUASSERT(cudart.cudaPointerGetAttributes(device_table.data_ptr()))
    assert int(attributes.type) != EH._CUDA_MEMORY_TYPE_HOST
