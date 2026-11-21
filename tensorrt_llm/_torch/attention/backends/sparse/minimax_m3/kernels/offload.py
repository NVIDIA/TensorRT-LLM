# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map M3's token writes into sparse cache arenas.

The per-head selection merge and the ordinal-indexed arena block table are
model-agnostic and live in ``sparse/offload_kernels.py``.
"""

import torch
import triton
import triton.language as tl

from ...offload import SparseOffloadState

__all__ = ["prepare_write_slots"]


@triton.jit
def _write_slots_kernel(
    rows,
    active,
    history,
    writes,
    out,
    WRITE_STRIDE: tl.constexpr,
    SLOT: tl.constexpr,
    PAGE: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.arange(0, BLOCK)
    row = tl.load(rows + query, query < SIZE, other=-1)
    live = (query < tl.load(active)) & (row >= 0) & (query < SIZE)
    length = tl.load(history + row, live, other=0)
    page = tl.load(writes + row * WRITE_STRIDE + SLOT, live, other=-1)
    tl.store(
        out + query, tl.where(live & (page >= 0), page * PAGE + length % PAGE, -1), query < SIZE
    )


def prepare_write_slots(state: SparseOffloadState, layer_idx: int, out: torch.Tensor) -> None:
    """Convert K arena pages to token slots in input order, masking graph padding.

    ``out`` is CUDA int32 [N], one entry per input position: the arena cell of the
    request's in-flight unit times ``unit_tokens`` plus the token's offset in it,
    from the layer's int32 [B, buffers] write indices, or -1 for a padding position.
    Slots are int32; the cache manager rejects an arena whose cells times
    ``unit_tokens`` would overflow them.
    """
    descriptor = state.layers[layer_idx]
    writes = state.write_indices[len(descriptor.buffer_ids)]
    _write_slots_kernel[(1,)](
        state.batch_rows,
        state.active_request_count,
        state.history_lengths,
        writes,
        out,
        writes.stride(0),
        state.out_slots[layer_idx][descriptor.data_buffer_index],
        descriptor.unit_tokens,
        out.numel(),
        triton.next_power_of_2(out.numel()),
    )
