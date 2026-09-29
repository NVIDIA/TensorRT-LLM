# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental TP8 fabric gather for column-sharded MuseSpark projections.

Calls on a workspace must form one ordered CUDA execution chain, with identical
shapes and collective order on every rank. Separate workspaces are required for
independent streams. Consumed inbox words are retired before the next launch,
so publication tags may wrap without matching a stale payload.
"""

import torch
import triton
import triton.language as tl

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.mapping import Mapping

_PROGRAMS = 32
_WORLD_SIZE = 8
_MAX_TOKENS = 128
_MAX_COLUMNS = 8192
_PAYLOAD_WORD_BYTES = 8


@triton.jit
def _gather_columns(
    source,
    data,
    counters,
    output,
    DATA_STRIDE: tl.constexpr,
    SLOT_WORDS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    LOCAL_COLUMNS: tl.constexpr,
    ELEMENTS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    state = tl.load(counters + program)
    epoch = state.to(tl.uint32) + 1
    # Zero denotes an empty inbox, including after retirement. Skip it on wrap.
    epoch = tl.where(epoch == 0, 1, epoch)
    # Slot alternation is independent of the tag: skipping zero on tag wrap
    # must not make two consecutive launches reuse the same slot.
    slot_index = ((state >> 32) ^ 1) & 1
    slot = slot_index * WORLD * SLOT_WORDS
    offsets = program * BLOCK + tl.arange(0, BLOCK)
    # Each aligned 64-bit word carries two BF16 values and their publication
    # epoch. No floating-point conversion or separately ordered ready flag.
    for tile in range(tl.cdiv(ELEMENTS, PROGRAMS * BLOCK)):
        index = offsets + tile * PROGRAMS * BLOCK
        bits = tl.load(source + index, index < ELEMENTS, 0).to(tl.uint32, bitcast=True)
        tagged = (epoch.to(tl.uint64) << 32) | bits.to(tl.uint64)
        for peer in tl.static_range(WORLD):
            target = data + peer * DATA_STRIDE + slot + RANK * SLOT_WORDS + index
            tl.store(target, tagged, index < ELEMENTS, cache_modifier=".wt")
    for tile in range(tl.cdiv(ELEMENTS, PROGRAMS * BLOCK)):
        index = offsets + tile * PROGRAMS * BLOCK
        row = index // LOCAL_COLUMNS
        column = index % LOCAL_COLUMNS
        for peer in tl.static_range(WORLD):
            inbox = data + RANK * DATA_STRIDE + slot + peer * SLOT_WORDS + index
            tagged = tl.load(inbox, index < ELEMENTS, 0, volatile=True).to(tl.uint64, bitcast=True)
            while tl.sum(((index < ELEMENTS) & ((tagged >> 32) != epoch)).to(tl.int32), 0) != 0:
                tagged = tl.load(inbox, index < ELEMENTS, 0, volatile=True).to(
                    tl.uint64, bitcast=True
                )
            values = tagged.to(tl.uint32).to(tl.int32, bitcast=True)
            destination = row * (WORLD * LOCAL_COLUMNS) + peer * LOCAL_COLUMNS + column
            tl.store(output + destination, values, index < ELEMENTS)
            # A peer cannot reuse this slot until this rank has published the
            # next launch, which is ordered after all these retirements. Clear
            # consumed words even when subsequent launches use smaller shapes.
            tl.store(inbox, 0, index < ELEMENTS, cache_modifier=".wt")
    tl.store(counters + program, (slot_index << 32) | epoch.to(tl.uint64))


class _ProjectionMnnvlMemory(MnnvlMemory):
    """Independent TP allocation, separate from MoE finalize epochs."""

    force_fabric_handle = True

    @classmethod
    def comm_split_color_key(cls, mapping: Mapping) -> tuple[int, int]:
        return mapping.pp_rank * mapping.cp_size + mapping.cp_rank, mapping.tp_rank


class ProjectionGatherWorkspace:
    """Own TP8 payload and synchronization memory for ordered CUDA calls.

    All ranks must use identical shapes and collective order. CUDA dependencies
    must serialize calls, including graph replay and transitions between
    streams; independent concurrent calls require separate workspaces.
    Allocation supports up to 128 tokens for component validation, while the
    model enables sharded projections only for batches of at most eight tokens.
    """

    def __init__(self, mapping: Mapping) -> None:
        if mapping.tp_size != _WORLD_SIZE or mapping.enable_attention_dp or mapping.cp_size != 1:
            raise ValueError("Projection gather requires TP8 without attention DP or CP")
        if not all(Distributed.get(mapping).tp_allgather(MnnvlMemory.supports_mnnvl())):
            raise ValueError("Projection gather requires NVLink on every TP rank")
        MnnvlMemory.initialize()
        self.mapping = mapping
        self.device = torch.cuda.current_device()
        self.programs = _PROGRAMS
        # Double buffering: publishing epoch E+2 requires consuming E+1 from
        # every peer. Each peer can publish E+1 only after its entire E launch
        # finished on the owning stream, so E's inbox is then safe to reuse.
        self.slot_words = _MAX_TOKENS * (_MAX_COLUMNS // _WORLD_SIZE) // 2
        self.data_owner = _ProjectionMnnvlMemory(
            mapping, 2 * _WORLD_SIZE * self.slot_words * _PAYLOAD_WORD_BYTES
        )
        record = type(self.data_owner).allocated_map[self.data_owner.ptr]
        if record.comm_size != _WORLD_SIZE or record.comm_rank != mapping.tp_rank:
            raise RuntimeError("Projection gather allocation does not match TP group")
        self.data = self.data_owner.as_torch_strided_tensor(torch.int64)
        self.data[mapping.tp_rank].zero_()
        self.counters = torch.zeros(self.programs, dtype=torch.uint64, device="cuda")
        torch.cuda.synchronize()
        Distributed.get(mapping).tp_allgather(True)

    def gather(self, local: torch.Tensor) -> torch.Tensor:
        """Gather contiguous BF16 ``[M, N/8]`` columns, preserving rank order."""
        if not (
            local.is_cuda
            and local.device.index == self.device
            and local.dtype == torch.bfloat16
            and local.ndim == 2
            and local.is_contiguous()
            and 1 <= local.shape[0] <= _MAX_TOKENS
            and local.shape[1] in (4096 // _WORLD_SIZE, 8192 // _WORLD_SIZE)
        ):
            raise ValueError("Unsupported projection gather tensor contract")
        output = torch.empty(
            (local.shape[0], local.shape[1] * _WORLD_SIZE), dtype=local.dtype, device=local.device
        )
        _gather_columns[(self.programs,)](
            local.view(torch.int32),
            self.data,
            self.counters,
            output.view(torch.int32),
            self.data.stride(0),
            self.slot_words,
            self.mapping.tp_rank,
            _WORLD_SIZE,
            local.shape[1] // 2,
            local.numel() // 2,
            self.programs,
            256,
            num_warps=4,
        )
        return output
