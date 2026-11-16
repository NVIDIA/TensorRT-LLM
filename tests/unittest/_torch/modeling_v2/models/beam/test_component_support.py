# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the test harness's cache readback indexing."""

from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["."]

from _beam_component_support import cache_rows  # noqa: E402

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("num_layers,local_layer", [(1, 0), (2, 1), (3, 2)])
def test_cache_rows_reads_nonzero_slots(num_layers: int, local_layer: int) -> None:
    slots, heads, block, dim = 4, 2, 4, 3
    pool = torch.arange(slots * num_layers * 2 * heads * block * dim).reshape(
        slots, num_layers, 2, heads, block, dim
    )
    # get_buffers starts at the requested layer's K page; rows are K/V pairs.
    view = pool.flatten(0, 1)[local_layer:]

    def get_buffers(layer_idx: int, kv_layout: str) -> torch.Tensor:
        assert layer_idx == 7 and kv_layout == "HND"
        return view

    selected = [2, 0, 1]
    scale = 2 * num_layers
    offsets = torch.tensor(selected) * scale
    manager = SimpleNamespace(
        get_buffers=get_buffers, get_layer_page_index_scale=lambda layer: scale
    )
    md = SimpleNamespace(
        kv_cache_manager=manager, kv_cache_block_offsets=offsets.reshape(1, 1, 1, -1)
    )
    length = 2 * block + 2
    keys, values = cache_rows(md, length, 7, block)
    for actual, role in ((keys, 0), (values, 1)):
        expected = torch.cat([pool[slot, local_layer, role].permute(1, 0, 2) for slot in selected])[
            :length
        ]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
