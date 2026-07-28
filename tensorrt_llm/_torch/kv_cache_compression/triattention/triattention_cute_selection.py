# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM100 CuTe-DSL selection preparation for TriAttention scores."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace
from cutlass.cute.typing import Int32 as CuteInt32
from cutlass.cute.typing import Pointer as CutePointer
from cutlass.cutlass_dsl import T, dsl_user_op

# Single-sourced constants: the score file owns N and the stats layout; Triton owns the epsilon.
from .triattention_cute_score_fused import PADDED_HEAD_COLUMNS, STATS_M2, STATS_MEAN
from .triattention_cute_score_fused import STATS_FIELDS as _STATS_FIELDS
from .triattention_kernels import STD_EPSILON as _STD_EPSILON

_REDUCE_THREADS = 256
_WARP_SIZE = 32
_REDUCE_WARPS = _REDUCE_THREADS // _WARP_SIZE
_LARGE_TOKENS_PER_LANE = 4
_LARGE_TOKEN_SUBTILES = 2
_SMALL_TOKENS_PER_LANE = 2
_SMALL_TOKEN_SUBTILES = 1
_MAX_ROW_CLUSTER_CTAS = 4
_SMALL_TILE_RESIDENT_CTAS_PER_SM = 6


def _select_normalize_union_config(
    request_count: int,
    width: int,
    sm_count: int,
) -> tuple[int, int, int]:
    """Return (tokens_per_lane, token_subtiles, row_cluster_ctas)."""
    row_cluster_ctas = max(1, _MAX_ROW_CLUSTER_CTAS // request_count)
    small_token_tile = _WARP_SIZE * _SMALL_TOKENS_PER_LANE * _SMALL_TOKEN_SUBTILES
    token_tiles = (width + small_token_tile - 1) // small_token_tile
    grid_ctas = request_count * token_tiles * row_cluster_ctas
    if grid_ctas <= sm_count * _SMALL_TILE_RESIDENT_CTAS_PER_SM:
        return _SMALL_TOKENS_PER_LANE, _SMALL_TOKEN_SUBTILES, row_cluster_ctas
    return _LARGE_TOKENS_PER_LANE, _LARGE_TOKEN_SUBTILES, 1


@dsl_user_op
def _mapa_shared_cluster(
    smem_ptr: CutePointer,
    peer_rank: CuteInt32,
    *,
    loc=None,
    ip=None,
) -> CuteInt32:
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_shared_cluster_f32(
    mapped_addr: CuteInt32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [mapped_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _gmem_lane_tile(iterator, flat_index, tokens_per_lane, assumed_align):
    """One lane's fp32 gmem tile; folds the 64-bit index into the pointer before access."""
    return cute.make_tensor(
        cute.make_ptr(
            cutlass.Float32,
            (iterator + flat_index).toint(),
            AddressSpace.gmem,
            assumed_align=assumed_align,
        ),
        cute.make_layout(tokens_per_lane),
    )


class _TriAttentionNormalizeUnionKernel:
    """Merge row moments, normalize scores, and reduce their elementwise maximum."""

    def __init__(
        self,
        *,
        num_layers: int,
        score_token_capacity: int,
        num_q_heads: int,
        num_kv_heads: int,
        page_shards: int,
        tokens_per_lane: int,
        token_subtiles: int,
        row_cluster_ctas: int,
        output_row_stride: int,
    ) -> None:
        # Real head row q_head lives in score plane kv*8 + qg; partial-stats rows stay compact.
        self.score_group_size = num_q_heads // num_kv_heads
        self.score_head_pad = PADDED_HEAD_COLUMNS - self.score_group_size
        self.num_layers = num_layers
        self.score_token_capacity = score_token_capacity
        self.output_row_stride = output_row_stride
        self.num_q_heads = num_q_heads
        self.num_rows = num_layers * num_q_heads
        self.page_shards = page_shards
        self.tokens_per_lane = tokens_per_lane
        self.token_subtiles = token_subtiles
        self.subtile_token_tile = _WARP_SIZE * self.tokens_per_lane
        self.token_tile = self.subtile_token_tile * self.token_subtiles
        self.reduce_threads = _REDUCE_THREADS
        self.reduce_warps = _REDUCE_WARPS
        self.row_cluster_ctas = row_cluster_ctas
        # The widest score window (the whole bucket) sizes the token-tile grid;
        # output rows are the TopK selection rows with their own stride.
        self.num_token_tiles = (self.score_token_capacity + self.token_tile - 1) // self.token_tile

    @cute.jit
    def __call__(
        self,
        partial_stats: cute.Tensor,
        scores: cute.Tensor,
        source_lengths: cute.Tensor,
        seg_out_offset: cute.Tensor,
        prompt_lengths: cute.Tensor,
        union_scores: cute.Tensor,
        request_count: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        kernel = self.kernel(
            partial_stats,
            scores,
            source_lengths,
            seg_out_offset,
            prompt_lengths,
            union_scores,
            request_count,
        )
        if cutlass.const_expr(self.row_cluster_ctas == 1):
            kernel.launch(
                grid=(request_count, self.num_token_tiles, 1),
                block=(self.reduce_threads, 1, 1),
                stream=stream,
            )
        else:
            # Cluster peers are consecutive CTAs; the kernel decode must keep this factor order.
            kernel.launch(
                grid=(
                    request_count * self.num_token_tiles * self.row_cluster_ctas,
                    1,
                    1,
                ),
                block=(self.reduce_threads, 1, 1),
                cluster=(self.row_cluster_ctas, 1, 1),
                stream=stream,
            )

    @cute.jit
    def _reduce_and_store_union_rows(
        self,
        union_scores: cute.Tensor,
        union_values: cute.Tensor,
        warp_max: cute.Tensor,
        warp_max_ptr,
        score_copy_atom,
        request_idx: cutlass.Int32,
        decode_length: cutlass.Int32,
        first_token: cutlass.Int32,
        lane_idx: cutlass.Int32,
        from_cluster_peers: cutlass.Constexpr,
    ):
        """Final peer reduce and union-row store (the peer source is picked at trace time)."""
        for token_subtile in cutlass.range_constexpr(self.token_subtiles):
            reduced_values = cute.make_rmem_tensor((self.tokens_per_lane,), cutlass.Float32)
            for token_slot in cutlass.range_constexpr(self.tokens_per_lane):
                if cutlass.const_expr(from_cluster_peers):
                    union_value = warp_max[(0, token_subtile, token_slot, lane_idx)]
                    # Offset derived from the same layout the smem tensor was built with.
                    shared_offset = cute.crd2idx(
                        (0, token_subtile, token_slot, lane_idx), warp_max.layout
                    )
                    for peer_rank in cutlass.range_constexpr(1, self.row_cluster_ctas):
                        remote_addr = _mapa_shared_cluster(warp_max_ptr, cutlass.Int32(peer_rank))
                        union_value = cute.arch.fmax(
                            union_value,
                            _ld_shared_cluster_f32(
                                remote_addr + shared_offset * (cutlass.Float32.width // 8)
                            ),
                        )
                else:
                    union_value = union_values[(token_subtile, token_slot)]
                    for other_warp in cutlass.range_constexpr(1, self.reduce_warps):
                        union_value = cute.arch.fmax(
                            union_value,
                            warp_max[(other_warp, token_subtile, token_slot, lane_idx)],
                        )
                reduced_values[token_slot] = union_value
            subtile_first_token = first_token + token_subtile * self.subtile_token_tile
            # Straddling subtiles store per token; the selection rows stay < 2^31 so i32 cannot wrap.
            if cutlass.const_expr(
                self.output_row_stride % self.tokens_per_lane == 0
            ) and cutlass.dynamic_expr(subtile_first_token + self.tokens_per_lane <= decode_length):
                union_index = request_idx * self.output_row_stride + subtile_first_token
                union_tile = _gmem_lane_tile(
                    union_scores.iterator,
                    union_index,
                    self.tokens_per_lane,
                    self.tokens_per_lane * 4,
                )
                cute.copy(
                    score_copy_atom,
                    cute.coalesce(reduced_values),
                    cute.coalesce(union_tile),
                )
            else:
                union_index = request_idx * self.output_row_stride + subtile_first_token
                union_tile = _gmem_lane_tile(
                    union_scores.iterator,
                    union_index,
                    self.tokens_per_lane,
                    4,
                )
                for token_slot in cutlass.range_constexpr(self.tokens_per_lane):
                    token = subtile_first_token + token_slot
                    if cutlass.dynamic_expr(token < decode_length):
                        union_tile[token_slot] = reduced_values[token_slot]

    @cute.kernel
    def kernel(
        self,
        partial_stats: cute.Tensor,
        scores: cute.Tensor,
        source_lengths: cute.Tensor,
        seg_out_offset: cute.Tensor,
        prompt_lengths: cute.Tensor,
        union_scores: cute.Tensor,
        request_count: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx_x, block_idx_y, _ = cute.arch.block_idx()
        cta_rank = cutlass.Int32(0)
        if cutlass.const_expr(self.row_cluster_ctas == 1):
            request_idx = block_idx_x
            token_tile_idx = block_idx_y
        else:
            cta_rank = cute.arch.block_idx_in_cluster()
            cluster_idx = block_idx_x // self.row_cluster_ctas
            request_idx = cluster_idx // self.num_token_tiles
            token_tile_idx = cluster_idx - request_idx * self.num_token_tiles
        warp_idx = tidx // _WARP_SIZE
        lane_idx = tidx % _WARP_SIZE
        first_token = token_tile_idx * self.token_tile + lane_idx * self.tokens_per_lane
        first_segment = request_idx * self.num_layers
        # The normalization domain and the union output row both cover [0, valid - start).
        score_start = cutlass.Int32(prompt_lengths[request_idx])
        decode_length = source_lengths[request_idx] - score_start
        warp_max_ptr = cute.arch.alloc_smem(
            cutlass.Float32,
            self.reduce_threads * self.tokens_per_lane * self.token_subtiles,
        )
        warp_max = cute.make_tensor(
            warp_max_ptr,
            cute.make_layout(
                (
                    self.reduce_warps,
                    self.token_subtiles,
                    self.tokens_per_lane,
                    _WARP_SIZE,
                ),
                stride=(
                    self.token_subtiles * self.tokens_per_lane * _WARP_SIZE,
                    self.tokens_per_lane * _WARP_SIZE,
                    _WARP_SIZE,
                    1,
                ),
            ),
        )
        union_values = cute.make_rmem_tensor(
            (self.token_subtiles, self.tokens_per_lane),
            cutlass.Float32,
        )
        score_value_tiles = tuple(
            cute.make_rmem_tensor((self.tokens_per_lane,), cutlass.Float32)
            for _ in range(self.token_subtiles)
        )
        score_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=self.tokens_per_lane * cutlass.Float32.width,
        )
        for token_subtile in cutlass.range_constexpr(self.token_subtiles):
            for token_slot in cutlass.range_constexpr(self.tokens_per_lane):
                union_values[(token_subtile, token_slot)] = cutlass.Float32(float("-inf"))

        common_count = cutlass.Float32(0.0)
        mean_weight_1 = cutlass.Float32(0.0)
        m2_cross_weight = cutlass.Float32(0.0)
        shard_mean_weights = cute.make_rmem_tensor((self.page_shards,), cutlass.Float32)
        shard_m2_cross_weights = cute.make_rmem_tensor((self.page_shards,), cutlass.Float32)
        if cutlass.const_expr(self.page_shards == 2):
            first_stats_row = first_segment * self.num_q_heads
            first_stats_base = first_stats_row * 2 * _STATS_FIELDS
            count_0 = partial_stats[first_stats_base]
            count_1 = partial_stats[first_stats_base + _STATS_FIELDS]
            common_count = count_0 + count_1
            if cutlass.dynamic_expr(common_count > 0.0):
                mean_weight_1 = count_1 / common_count
                m2_cross_weight = count_0 * count_1 / common_count
        else:
            first_stats_row = first_segment * self.num_q_heads
            first_stats_base = first_stats_row * self.page_shards * _STATS_FIELDS
            for page_shard in cutlass.range_constexpr(self.page_shards):
                shard_count = partial_stats[first_stats_base + page_shard * _STATS_FIELDS]
                merged_count = common_count + shard_count
                mean_weight = cutlass.Float32(0.0)
                m2_cross = cutlass.Float32(0.0)
                if cutlass.dynamic_expr(merged_count > 0.0):
                    mean_weight = shard_count / merged_count
                    m2_cross = common_count * shard_count / merged_count
                shard_mean_weights[page_shard] = mean_weight
                shard_m2_cross_weights[page_shard] = m2_cross
                common_count = merged_count

        first_logical_row = warp_idx + cta_rank * self.reduce_warps
        logical_row_stride = self.reduce_warps * self.row_cluster_ctas
        for logical_row in cutlass.range(
            first_logical_row,
            self.num_rows,
            logical_row_stride,
            unroll=1,
        ):
            layer_slot = logical_row // self.num_q_heads
            q_head = logical_row - layer_slot * self.num_q_heads
            segment = first_segment + layer_slot
            stats_row = segment * self.num_q_heads + q_head
            # Map the real head row onto its padded score plane; padded planes are never visited.
            score_plane = q_head
            if cutlass.const_expr(self.score_head_pad > 0):
                score_plane = q_head + (q_head // self.score_group_size) * self.score_head_pad

            count = cutlass.Float32(0.0)
            mean = cutlass.Float32(0.0)
            m2 = cutlass.Float32(0.0)
            delta = cutlass.Float32(0.0)
            if cutlass.const_expr(self.page_shards == 2):
                stats_base = stats_row * 2 * _STATS_FIELDS
                mean_0 = partial_stats[stats_base + STATS_MEAN]
                m2_0 = partial_stats[stats_base + STATS_M2]
                mean_1 = partial_stats[stats_base + _STATS_FIELDS + STATS_MEAN]
                m2_1 = partial_stats[stats_base + _STATS_FIELDS + STATS_M2]
                delta = mean_1 - mean_0
                count = common_count
                mean = mean_0 + delta * mean_weight_1
                m2 = m2_0 + m2_1 + delta * delta * m2_cross_weight
            else:
                count = common_count
                for page_shard in cutlass.range_constexpr(self.page_shards):
                    stats_base = (stats_row * self.page_shards + page_shard) * _STATS_FIELDS
                    shard_mean = partial_stats[stats_base + STATS_MEAN]
                    shard_m2 = partial_stats[stats_base + STATS_M2]
                    delta = shard_mean - mean
                    mean = mean + delta * shard_mean_weights[page_shard]
                    m2 = m2 + shard_m2 + delta * delta * shard_m2_cross_weights[page_shard]
            inv_std = cutlass.Float32(0.0)
            if cutlass.dynamic_expr(count > 0.0):
                variance = m2 / count
                if cutlass.dynamic_expr(variance < _STD_EPSILON * _STD_EPSILON):
                    inv_std = cutlass.Float32(1.0 / _STD_EPSILON)
                else:
                    inv_std = cute.math.rsqrt(variance)

            for token_subtile in cutlass.range_constexpr(self.token_subtiles):
                subtile_first_token = first_token + token_subtile * self.subtile_token_tile
                score_index = (
                    cutlass.Int64(score_plane)
                    * request_count
                    * self.num_layers
                    * self.score_token_capacity
                    + seg_out_offset[segment]
                    + score_start
                    + subtile_first_token
                )
                # The vectorized load needs the runtime start aligned to the lane width.
                if cutlass.const_expr(
                    self.score_token_capacity % self.tokens_per_lane == 0
                ) and cutlass.dynamic_expr(
                    score_start % self.tokens_per_lane == 0
                    and subtile_first_token + self.tokens_per_lane <= decode_length
                ):
                    score_tile = _gmem_lane_tile(
                        scores.iterator,
                        score_index,
                        self.tokens_per_lane,
                        self.tokens_per_lane * 4,
                    )
                    cute.copy(
                        score_copy_atom,
                        cute.coalesce(score_tile),
                        cute.coalesce(score_value_tiles[token_subtile]),
                    )
                else:
                    # Fold the 64-bit index into the pointer; the scratch exceeds 2^31 elements.
                    score_tail = _gmem_lane_tile(
                        scores.iterator, score_index, self.tokens_per_lane, 4
                    )
                    for token_slot in cutlass.range_constexpr(self.tokens_per_lane):
                        token = subtile_first_token + token_slot
                        if cutlass.dynamic_expr(token < decode_length):
                            score_value_tiles[token_subtile][token_slot] = score_tail[token_slot]
                        else:
                            score_value_tiles[token_subtile][token_slot] = cutlass.Float32(
                                float("-inf")
                            )

            for token_subtile in cutlass.range_constexpr(self.token_subtiles):
                if cutlass.const_expr(self.tokens_per_lane >= 2):
                    normalized_01 = cute.arch.sub_packed_f32x2(
                        (
                            score_value_tiles[token_subtile][0],
                            score_value_tiles[token_subtile][1],
                        ),
                        (mean, mean),
                    )
                    normalized_01 = cute.arch.mul_packed_f32x2(
                        normalized_01,
                        (inv_std, inv_std),
                    )
                    union_values[(token_subtile, 0)] = cute.arch.fmax(
                        union_values[(token_subtile, 0)], normalized_01[0]
                    )
                    union_values[(token_subtile, 1)] = cute.arch.fmax(
                        union_values[(token_subtile, 1)], normalized_01[1]
                    )
                if cutlass.const_expr(self.tokens_per_lane == 4):
                    normalized_23 = cute.arch.sub_packed_f32x2(
                        (
                            score_value_tiles[token_subtile][2],
                            score_value_tiles[token_subtile][3],
                        ),
                        (mean, mean),
                    )
                    normalized_23 = cute.arch.mul_packed_f32x2(
                        normalized_23,
                        (inv_std, inv_std),
                    )
                    union_values[(token_subtile, 2)] = cute.arch.fmax(
                        union_values[(token_subtile, 2)], normalized_23[0]
                    )
                    union_values[(token_subtile, 3)] = cute.arch.fmax(
                        union_values[(token_subtile, 3)], normalized_23[1]
                    )
        for token_subtile in cutlass.range_constexpr(self.token_subtiles):
            for token_slot in cutlass.range_constexpr(self.tokens_per_lane):
                warp_max[(warp_idx, token_subtile, token_slot, lane_idx)] = union_values[
                    (token_subtile, token_slot)
                ]
        cute.arch.sync_threads()
        if cutlass.const_expr(self.row_cluster_ctas == 1):
            if warp_idx == 0:
                self._reduce_and_store_union_rows(
                    union_scores,
                    union_values,
                    warp_max,
                    warp_max_ptr,
                    score_copy_atom,
                    request_idx,
                    decode_length,
                    first_token,
                    lane_idx,
                    False,
                )
        else:
            # Warp 0 reduces its CTA's rows; CTA 0 combines cluster maxima via distributed smem.
            if warp_idx == 0:
                for token_subtile in cutlass.range_constexpr(self.token_subtiles):
                    for token_slot in cutlass.range_constexpr(self.tokens_per_lane):
                        union_value = union_values[(token_subtile, token_slot)]
                        for other_warp in cutlass.range_constexpr(1, self.reduce_warps):
                            union_value = cute.arch.fmax(
                                union_value,
                                warp_max[(other_warp, token_subtile, token_slot, lane_idx)],
                            )
                        warp_max[(0, token_subtile, token_slot, lane_idx)] = union_value
            cute.arch.sync_threads()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if cta_rank == 0 and warp_idx == 0:
                self._reduce_and_store_union_rows(
                    union_scores,
                    union_values,
                    warp_max,
                    warp_max_ptr,
                    score_copy_atom,
                    request_idx,
                    decode_length,
                    first_token,
                    lane_idx,
                    True,
                )
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
