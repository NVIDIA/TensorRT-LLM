# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SWA slot kernel's per-table constants (`swa_tables.swa_table_static`, every target's) are kept per table buffer
and its whole geometry: a table at the same address with the same request rows but another row stride (a rebuilt KV
cache manager with narrower block tables) gets its own descriptors. Keyed by `(address, rows)` before, it reused the old
row stride, and every SWA read and write went to another request's rows."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.swa_tables import swa_table_static

pytestmark = pytest.mark.cpu_only

LAYERS, ROWS = 4, 8


def test_same_address_and_rows_other_stride_gets_its_own_descriptors():
    buf = torch.full((LAYERS * ROWS * 24,), -1, dtype=torch.int32)
    wide = buf.view(LAYERS, ROWS, 24)[:, :, :16]  # the first manager's tables, 24 columns per row
    narrow = buf[: LAYERS * ROWS * 16].view(
        LAYERS, ROWS, 16
    )  # a rebuilt manager's: 16 columns per row
    assert wide.data_ptr() == narrow.data_ptr() and wide.shape == narrow.shape
    ratios = [1, 2, 4, 1]
    d_wide = swa_table_static(wide, ratios)[0]
    d_narrow = swa_table_static(narrow, ratios)[0]
    # (pointer, row stride, rows, columns) per layer
    assert d_wide[:, 1].tolist() == [24] * LAYERS
    assert d_narrow[:, 1].tolist() == [16] * LAYERS
    assert d_narrow[:, 0].tolist() == [narrow[i].data_ptr() for i in range(LAYERS)]


def test_the_same_buffer_keeps_its_constants():
    """The CUDA graphs read the descriptors, ratios and floors at fixed addresses: one build per buffer."""
    t = torch.full((LAYERS, ROWS, 16), -1, dtype=torch.int32)
    a, b = swa_table_static(t, [1, 2, 4, 1]), swa_table_static(t, [1, 2, 4, 1])
    assert all(x is y for x, y in zip(a, b))
    assert a[1].tolist() == [1, 2, 4, 1] and a[2].shape == (ROWS,)


def test_the_targets_read_the_engines_swa_tables_for_every_request():
    """Every target reads the step's SWA tables from the engine's metadata (`kv_cache_block_offsets[:layers, :, 0]`),
    for as many requests as the engine schedules: dep2 kept its own 512-row table and asserted at most 512 requests a
    step, so an engine with the default max_batch_size (2048) failed its first warm-up step."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.swa_tables import engine_swa_tables

    requests = 2048
    offsets = torch.arange(LAYERS * requests * 2 * 16, dtype=torch.int32).view(
        LAYERS, requests, 2, 16
    )
    tables = engine_swa_tables(SimpleNamespace(kv_cache_block_offsets=offsets), LAYERS)
    assert tables.shape == (LAYERS, requests, 16) and tables.data_ptr() == offsets.data_ptr()
    assert torch.equal(tables, offsets[:, :, 0, :])
    with pytest.raises(AssertionError, match="kv_cache_block_offsets"):
        engine_swa_tables(SimpleNamespace(kv_cache_block_offsets=None), LAYERS)
