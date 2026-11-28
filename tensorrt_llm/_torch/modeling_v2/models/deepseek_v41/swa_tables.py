# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-table constants of the SWA slot kernel (`ops/attention/csa2_swa_slots`), shared by every target.

`csa2_swa_slots` reads a step's `[layers, requests, blocks]` SWA page tables through descriptors that hold their raw
pointer, row stride, rows and columns. The descriptors, the layers' compression ratios and the per-request floors are
built once per table buffer, at its first (eager) use, and kept at fixed addresses for the CUDA graphs that read them.
A table buffer is identified by its address and its whole geometry (shape and strides), as the engine's own descriptor
cache is (`csa2/metadata.py`): a rebuilt KV cache manager (the engine replaces the estimation pass's manager, and a
clamped `max_seq_len` gives narrower block tables) can hand out a table at the same address with another row stride,
which must not reuse the old descriptors.
"""

from typing import Dict, Sequence, Tuple

import torch

from .ops.attention.csa2_swa_slots import csa2_swa_table_descriptors

_STATIC: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def swa_table_key(tables: torch.Tensor) -> tuple:
    """What identifies a table buffer for its descriptors: address, device, shape and strides."""
    return (tables.data_ptr(), tables.device.index, tuple(tables.shape), tuple(tables.stride()))


def swa_table_static(
    tables: torch.Tensor, ratios: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(descriptors [layers, 4] int64, ratios [L] int32, floors [requests] int64)` of `tables`, built on first use
    (an eager step: the engine warms up before it captures graphs)."""
    key = swa_table_key(tables)
    static = _STATIC.get(key)
    if static is None:
        dev = tables.device
        static = _STATIC[key] = (
            csa2_swa_table_descriptors(tables),
            torch.tensor(list(ratios), dtype=torch.int32, device=dev),
            torch.zeros(tables.shape[1], dtype=torch.int64, device=dev),
        )
    return static


def engine_swa_tables(attn_metadata, num_layers: int) -> torch.Tensor:
    """This step's sliding-window page tables `[layer, request, block]` (int32, unmapped entries invalid): the engine's.

    The attention metadata's `prepare` converts every SWA table of the batch on the device, straight into
    `kv_cache_block_offsets[:layers, :requests, 0, :width]` (`CSA2CacheManager.copy_batch_block_offsets`: one kernel,
    only the live columns; rows past the batch invalid; graph metadata at a fixed address, refreshed before every
    replay). The targets read that view, for every request the engine schedules (no row cap of their own) and with no
    host rebuild or copy of their own."""
    offsets = getattr(attn_metadata, "kv_cache_block_offsets", None)
    assert (
        offsets is not None
        and offsets.dim() == 4
        and offsets.shape[0] >= num_layers
        and offsets.dtype == torch.int32
    ), "the CSA2 attention metadata carries the step's SWA tables in kv_cache_block_offsets"
    return offsets[:num_layers, :, 0, :]
