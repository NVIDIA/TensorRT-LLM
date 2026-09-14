# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-only resolution of logical selections to independent host/GPU locations."""

import triton as tr
import triton.language as tl


@tr.jit
def resolve_entries_kernel(
    Positions,
    RequestIndices,
    RequestIds,
    Lengths,
    Mask,
    HostRequestIds,
    HostSlots,
    HostLengths,
    GpuRequestIds,
    GpuEntries,
    Valid,
    HostValid,
    HostOffsets,
    GpuIndices,
    K: tl.constexpr,
    NREQ: tl.constexpr,
    NPAGES: tl.constexpr,
    GPU_CAPACITY: tl.constexpr,
    POS_ROW: tl.constexpr,
    POS_COL: tl.constexpr,
    REQ_INDEX_STRIDE: tl.constexpr,
    REQ_ID_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    MASK_ROW: tl.constexpr,
    MASK_COL: tl.constexpr,
    HOST_ID_STRIDE: tl.constexpr,
    HOST_ROW: tl.constexpr,
    HOST_COL: tl.constexpr,
    HOST_LEN_ROW: tl.constexpr,
    HOST_LEN_COL: tl.constexpr,
    GPU_ID_STRIDE: tl.constexpr,
    GPU_ROW: tl.constexpr,
    GPU_COL: tl.constexpr,
    ENTRIES_PER_PAGE: tl.constexpr,
    COMPONENTS: tl.constexpr,
    SLOT_BYTES: tl.constexpr,
    POOL_BYTES: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_output = col < K
    req = tl.load(RequestIndices + row * REQ_INDEX_STRIDE).to(tl.int64)
    req_valid = (req >= 0) & (req < NREQ)
    req_id = tl.load(RequestIds + req * REQ_ID_STRIDE, req_valid, other=0)
    length = tl.load(Lengths + row * LENGTH_STRIDE)
    pos = tl.load(Positions + row * POS_ROW + col * POS_COL, in_output, other=-1).to(tl.int64)
    valid = in_output & req_valid & (pos >= 0) & (pos < length)
    if HAS_MASK:
        valid &= tl.load(Mask + row * MASK_ROW + col * MASK_COL, in_output, other=False)
    page = pos // ENTRIES_PER_PAGE
    entry = pos % ENTRIES_PER_PAGE
    host_id = tl.load(HostRequestIds + req * HOST_ID_STRIDE, req_valid, other=0)
    host_valid = valid & (host_id == req_id) & (page < NPAGES)
    slot = tl.load(HostSlots + req * HOST_ROW + page * HOST_COL, host_valid, other=-1)
    host_length = tl.load(
        HostLengths + req * HOST_LEN_ROW + page * HOST_LEN_COL, host_valid, other=0
    )
    host_valid &= (slot >= 0) & (entry < host_length)
    # Bound the slot before multiplication, so an invalid int64 slot cannot wrap
    # around to a plausible byte offset. Every component must be available.
    for component in tl.static_range(len(COMPONENTS)):
        host_valid &= (
            slot < POOL_BYTES[COMPONENTS[component][0]] // SLOT_BYTES[COMPONENTS[component][0]]
        )
    slot = tl.where(host_valid, slot, 0)
    entry = tl.where(host_valid, entry, 0)
    for component in tl.static_range(len(COMPONENTS)):
        byte_offset = (
            slot * SLOT_BYTES[COMPONENTS[component][0]]
            + COMPONENTS[component][1]
            + entry * COMPONENTS[component][2]
        )
        tl.store(
            HostOffsets + (row * K + col) * len(COMPONENTS) + component,
            tl.where(host_valid, byte_offset, -1),
            in_output,
        )
    gpu_id = tl.load(GpuRequestIds + req * GPU_ID_STRIDE, req_valid, other=0)
    gpu_valid = valid & (gpu_id == req_id) & (pos < GPU_CAPACITY)
    gpu_index = tl.load(GpuEntries + req * GPU_ROW + pos * GPU_COL, gpu_valid, other=-1)
    tl.store(Valid + row * K + col, valid, in_output)
    tl.store(HostValid + row * K + col, host_valid, in_output)
    tl.store(GpuIndices + row * K + col, tl.maximum(gpu_index, -1), in_output)
