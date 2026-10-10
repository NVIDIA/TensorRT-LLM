# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VC-Attention scale addressing for the context softmax.

Every dequant-scale load of the VC-Attention context kernel goes through this
module, the context counterpart of the decode kernels'
:mod:`..fmha_decode.fmha_decode_resources.sage_scales`. ``sfQ`` and ``sfK`` use
the trtllm-gen flat layout (``flat_scale_slot`` of
:mod:`flashinfer.attention.prims_ts.sage`): per head, sequence ``b`` of length
``S`` starts at slot ``(b * S >> log2(blk)) + b`` and token ``t`` uses slot
``t >> log2(blk)`` inside it. Q rows past the valid row count clamp to the
last valid slot, so masked rows keep a finite scale.

:class:`VCKScaleTable` is one softmax group's ``sfK`` words for the whole K/V
loop of a work tile: the group fills it with one coalesced pass before the
loop (published by a named barrier that also keeps a fast warp from refilling
while a peer still reads the previous work tile's words) and every softmax
step reads the next tile's word one iteration ahead. The words live behind the
group's SMEM P tile (``SmemPResource.kscale_table_offset``).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.experimental.task_scheduling.resources import StageInfo


@cute.jit
def vc_flat_q_slot(
    batch_coord: Int32,
    row: Int32,
    seq_len_q: cutlass.Constexpr[int],
    q_block_log2: cutlass.Constexpr[int],
) -> Int32:
    """Flat ``sfQ`` slot of query ``row`` of sequence ``batch_coord``."""
    lbq = Int32(q_block_log2)
    row_c = cute.math.min(row, Int32(seq_len_q - 1))
    return ((batch_coord * Int32(seq_len_q)) >> lbq) + batch_coord + (row_c >> lbq)


@cute.jit
def vc_flat_k_base(
    batch_coord: Int32,
    seq_len_k: cutlass.Constexpr[int],
    kv_tile_n: cutlass.Constexpr[int],
) -> Int32:
    """Flat ``sfK`` slot of K/V tile 0 of sequence ``batch_coord``; tile ``t`` is ``base + t``."""
    lbk = Int32(kv_tile_n.bit_length() - 1)
    return ((batch_coord * Int32(seq_len_k)) >> lbk) + batch_coord


class VCKScaleTable:
    """One softmax group's ``sfK`` table for the work tile, staged in SMEM."""

    def __init__(self, cfg, smem_p, q_half: int) -> None:
        self.cfg = cfg
        self.smem_p = smem_p
        self.q_half = q_half

    @property
    def words(self) -> int:
        return self.cfg.vc_kscale_table_bytes // 4

    @cute.jit
    def view(self, stage_info: StageInfo) -> cutlass.Array:
        """The table as an fp32 SMEM array."""
        context = stage_info.context
        return cutlass.Array(
            context.smem_base.data_ptr()
            + self.smem_p._alloc.offset
            + self.smem_p.kscale_table_offset,
            dtype=Float32,
            shape=(self.words,),
            addrspace=3,
        )

    @cute.jit
    def fill(
        self,
        stage_info: StageInfo,
        vc_k_scale: cute.Tensor,
        kv_head: Int32,
        k_base: Int32,
    ) -> Float32:
        """Gather this work tile's ``sfK`` words from the flat layout and publish
        the table to the group's four warps; returns tile 0's scale."""
        cfg = self.cfg
        num_threads = len(cfg.softmax0_warp_ids) * cute.arch.WARP_SIZE
        barrier_id = cfg.vc_kscale_barrier_id + self.q_half
        table = self.view(stage_info)
        num_tiles = Int32(cfg.vc_max_kv_tiles)
        warp_id_in_sg = cute.arch.warp_idx() % len(cfg.softmax0_warp_ids)
        thread = warp_id_in_sg * cute.arch.WARP_SIZE + cute.arch.lane_idx()
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)
        for pass_idx in cutlass.range_constexpr(
            (self.words + num_threads - 1) // num_threads
        ):
            tile_idx = Int32(pass_idx * num_threads) + thread
            if tile_idx < num_tiles:
                table[tile_idx] = Float32(vc_k_scale[(kv_head, k_base + tile_idx)])
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)
        return Float32(table[Int32(0)])

    @cute.jit
    def next_scale(self, stage_info: StageInfo, tile_idx: Int32) -> Float32:
        """``sfK`` of tile ``tile_idx + 1`` (clamped to the table), read one tile ahead."""
        table = self.view(stage_info)
        next_idx = cute.math.min(tile_idx + Int32(1), Int32(self.words) - Int32(1))
        return Float32(table[next_idx])
