/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace cutlass::fmha::collective {

using namespace cute;

enum class SparseAttnMode {
  Off,
  OnlyScore,
  Full,
  Sparse
};

struct NoMask {
  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                    ProblemSize const& problem_size) {
    return ceil_div(get<1>(problem_size), get<1>(tile_shape));
  }

  template <class ProblemSize>
  CUTLASS_DEVICE static int sparse_causal_bound(int q_s, ProblemSize const&) {
    return INT_MAX / 4;
  }

  // Per-warp: trip count doesn't depend on Q range for NoMask.
  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                    ProblemSize const& problem_size,
                                    int sub_q_offset, int sub_q_size) {
    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                           ProblemSize const& problem_size) {
    return 0;
  }

  // Per-warp: sub_q_offset/sub_q_size describe the Q sub-range within the tile.
  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                           ProblemSize const& problem_size,
                                           int sub_q_offset, int sub_q_size) {
    return 0;
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_unmasked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                             ProblemSize const& problem_size) {
    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template <class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE void apply_mask(AccQK& acc_qk, IndexQK const& index_qk,
                                 ProblemSize const& problem_size) {
    return;
  }
};

struct ResidualMask : NoMask {
  using Base = NoMask;

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                           ProblemSize const& problem_size) {
    if (get<1>(problem_size) % get<1>(tile_shape) != 0) {
      return 1;
    }
    return 0;
  }

  // Per-warp: residual mask doesn't depend on Q range.
  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                           ProblemSize const& problem_size,
                                           int sub_q_offset, int sub_q_size) {
    return get_masked_trip_count(blk_coord, tile_shape, problem_size);
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_unmasked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                             ProblemSize const& problem_size) {
    // if the sequence length does not divide the tile size evenly
    if (get<1>(problem_size) % get<1>(tile_shape) != 0) {
      return get_trip_count(blk_coord, tile_shape, problem_size) - 1;
    }
    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template <class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE void apply_mask(AccQK& acc_qk, IndexQK const& index_qk,
                                 ProblemSize const& problem_size) {
    // This is useful is seqlen_k % kBlockN != 0 since it masks
    // the remaining elements out from softmax.
    // d % kHeadDim != 0 or seqlen_q % kBlockM do not suffer from similar
    // issues as they are transparently taken care of by TMA and the
    // epilogue, if it is instantiated with predication support.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      if (get<1>(pos) >= get<1>(problem_size)) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

struct CausalMask : NoMask {
  using Base = NoMask;

  // Read qo_offset from 5th element of problem_size if available,
  // otherwise fall back to kv_len - qo_len (Q aligned to end of KV).
  template <class ProblemSize>
  CUTLASS_DEVICE static int get_qo_offset(ProblemSize const& problem_size) {
    if constexpr (cute::tuple_size_v<ProblemSize> > 4) {
      int offset = int(get<4>(problem_size));
      if (offset >= 0) return offset;
    }
    return int(get<1>(problem_size)) - int(get<0>(problem_size));
  }

  template <class ProblemSize>
  CUTLASS_DEVICE static int sparse_causal_bound(int q_s, ProblemSize const& ps) {
    return q_s + get_qo_offset(ps);
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                    ProblemSize const& problem_size) {
    int offset_q = get_qo_offset(problem_size);
    int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);
    int max_blocks_q =
        ceil_div((get<0>(blk_coord) + 1) * get<0>(tile_shape) + offset_q, get<1>(tile_shape));
    return std::min(max_blocks_k, max_blocks_q);
  }

  // Per-warp: trip count based on the last Q row of the sub-tile.
  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                    ProblemSize const& problem_size,
                                    int sub_q_offset, int sub_q_size) {
    int offset_q = get_qo_offset(problem_size);
    int n = int(get<1>(tile_shape));
    int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);
    int q_end = int(get<0>(blk_coord)) * int(get<0>(tile_shape)) + sub_q_offset + sub_q_size;
    return std::min(max_blocks_k, ceil_div(q_end + offset_q, n));
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                           ProblemSize const& problem_size) {
    int offset_q = get_qo_offset(problem_size);
    int n = int(get<1>(tile_shape));
    int offset_tile_q = ((offset_q % n) + n) % n;
    return min(get_trip_count(blk_coord, tile_shape, problem_size),
               int(ceil_div(int(get<0>(tile_shape)) + offset_tile_q, n)));
  }

  // Per-warp refinement: each warp handles sub_q_size Q rows starting at sub_q_offset
  // within the tile, so tiles fully below this warp's causal diagonal are unmasked.
  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                           ProblemSize const& problem_size,
                                           int sub_q_offset, int sub_q_size) {
    int offset_q = get_qo_offset(problem_size);
    int n = int(get<1>(tile_shape));
    int total_trip = get_trip_count(blk_coord, tile_shape, problem_size);

    // Per-warp trip count based on the last Q row of this sub-tile
    int q_end = int(get<0>(blk_coord)) * int(get<0>(tile_shape)) + sub_q_offset + sub_q_size;
    int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);
    int sub_trip = min(max_blocks_k, ceil_div(q_end + offset_q, n));

    // Per-warp masked count
    int q_start = q_end - sub_q_size;
    int sub_offset_tile_q = (((q_start + offset_q) % n) + n) % n;
    int sub_masked = min(sub_trip, ceil_div(sub_q_size + sub_offset_tile_q, n));

    return total_trip - (sub_trip - sub_masked);
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_unmasked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                             ProblemSize const& problem_size) {
    return get_trip_count(blk_coord, tile_shape, problem_size) -
           get_masked_trip_count(blk_coord, tile_shape, problem_size);
  }

  template <class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE void apply_mask(AccQK& acc_qk, IndexQK const& index_qk,
                                 ProblemSize const& problem_size) {
    int offset_q = get_qo_offset(problem_size);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      if ((get<0>(pos) + offset_q < get<1>(pos)) || (get<1>(pos) >= get<1>(problem_size))) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

template <class M> struct pack_factor_of { static constexpr int value = 1; };

template <int PackFactor>
struct PackedCausalMask : NoMask {
  using Base = NoMask;

  template <class ProblemSize>
  CUTLASS_DEVICE static int get_qo_offset(ProblemSize const& problem_size) {
    return CausalMask::get_qo_offset(problem_size);
  }

  template <class ProblemSize>
  CUTLASS_DEVICE static int sparse_causal_bound(int q_s, ProblemSize const& ps) {
    return logical_q(q_s) + get_qo_offset(ps);
  }  CUTLASS_DEVICE static int logical_q(int packed_q) {
    return packed_q / PackFactor;
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                    ProblemSize const& problem_size) {
    int offset_q = get_qo_offset(problem_size);
    int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);
    int packed_q_end = (int(get<0>(blk_coord)) + 1) * int(get<0>(tile_shape));
    int q_end = logical_q(packed_q_end - 1) + 1;
    return std::min(max_blocks_k, ceil_div(q_end + offset_q, int(get<1>(tile_shape))));
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                    ProblemSize const& problem_size,
                                    int sub_q_offset, int sub_q_size) {
    int offset_q = get_qo_offset(problem_size);
    int n = int(get<1>(tile_shape));
    int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);
    int packed_q_end = int(get<0>(blk_coord)) * int(get<0>(tile_shape)) + sub_q_offset + sub_q_size;
    int q_end = logical_q(packed_q_end - 1) + 1;
    return std::min(max_blocks_k, ceil_div(q_end + offset_q, n));
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                           ProblemSize const& problem_size) {
    int offset_q = get_qo_offset(problem_size);
    int n = int(get<1>(tile_shape));
    int packed_tile_start = int(get<0>(blk_coord)) * int(get<0>(tile_shape));
    int q_start = logical_q(packed_tile_start);
    int offset_tile_q = (((q_start + offset_q) % n) + n) % n;
    int packed_q_end = packed_tile_start + int(get<0>(tile_shape));
    int q_end = logical_q(packed_q_end - 1) + 1;
    int logical_tile_q = q_end - q_start;
    return min(get_trip_count(blk_coord, tile_shape, problem_size),
               int(ceil_div(logical_tile_q + offset_tile_q, n)));
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                           ProblemSize const& problem_size,
                                           int sub_q_offset, int sub_q_size) {
    int offset_q = get_qo_offset(problem_size);
    int n = int(get<1>(tile_shape));
    int total_trip = get_trip_count(blk_coord, tile_shape, problem_size);

    int packed_q_end = int(get<0>(blk_coord)) * int(get<0>(tile_shape)) + sub_q_offset + sub_q_size;
    int q_end = logical_q(packed_q_end - 1) + 1;
    int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);
    int sub_trip = min(max_blocks_k, ceil_div(q_end + offset_q, n));

    int packed_q_start = packed_q_end - sub_q_size;
    int q_start = logical_q(packed_q_start);
    int sub_offset_tile_q = (((q_start + offset_q) % n) + n) % n;
    int logical_sub_q = q_end - q_start;
    int sub_masked = min(sub_trip, ceil_div(logical_sub_q + sub_offset_tile_q, n));

    return total_trip - (sub_trip - sub_masked);
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_unmasked_trip_count(BlkCoord const& blk_coord, TileShape const& tile_shape,
                                             ProblemSize const& problem_size) {
    return get_trip_count(blk_coord, tile_shape, problem_size) -
           get_masked_trip_count(blk_coord, tile_shape, problem_size);
  }

  template <class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE void apply_mask(AccQK& acc_qk, IndexQK const& index_qk,
                                 ProblemSize const& problem_size) {
    int offset_q = get_qo_offset(problem_size);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      int q_logical = logical_q(int(get<0>(pos)));
      if ((q_logical + offset_q < int(get<1>(pos))) || (int(get<1>(pos)) >= int(get<1>(problem_size)))) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

template <int PF>
struct pack_factor_of<PackedCausalMask<PF>> { static constexpr int value = PF; };

struct VariableLength {
  int* segment_lens = nullptr;
  int* segment_offsets = nullptr;
};

// Per-batch offset for causal mask. Resolves to offsets[batch_idx] per batch,
// or -1 (sentinel for default kv_len - qo_len) if offsets is nullptr.
struct PerBatchOffset {
  int* offsets = nullptr;
};

template <class T>
struct is_variable_length : std::false_type {};
template <>
struct is_variable_length<VariableLength> : std::true_type {};
template <class T>
constexpr bool is_variable_length_v = is_variable_length<T>::value;

template <class T>
struct is_per_batch_offset : std::false_type {};
template <>
struct is_per_batch_offset<PerBatchOffset> : std::true_type {};
template <class T>
constexpr bool is_per_batch_offset_v = is_per_batch_offset<T>::value;

template <class Shape, class Idx>
CUTE_HOST_DEVICE constexpr auto apply_variable_length(Shape const& shape, Idx const& idx) {
  return transform_leaf(shape, [&](auto const& s) {
    if constexpr (is_variable_length_v<remove_cvref_t<decltype(s)>>) {
      return s.segment_lens[idx];
    } else if constexpr (is_per_batch_offset_v<remove_cvref_t<decltype(s)>>) {
      return s.offsets ? s.offsets[idx] : -1;
    } else {
      return s;
    }
  });
}

template <class Shape, class Coord, class Idx>
CUTE_HOST_DEVICE constexpr auto apply_variable_length(Shape const& shape, Coord const& coord,
                                                      Idx const& idx) {
  auto new_shape = apply_variable_length(shape, idx);
  auto new_coord = transform_leaf(shape, coord, [&](auto const& s, auto const& c) {
    if constexpr (is_variable_length_v<remove_cvref_t<decltype(s)>>) {
      return cute::make_tuple(c, s.segment_offsets[idx]);
    } else {
      return c;
    }
  });
  return cute::make_tuple(new_shape, new_coord);
}

}  // namespace cutlass::fmha::collective

namespace cute {

template <>
struct is_integral<cutlass::fmha::collective::VariableLength> : true_type {};

template <>
struct is_integral<cutlass::fmha::collective::PerBatchOffset> : true_type {};

CUTE_HOST_DEVICE
void print(cutlass::fmha::collective::VariableLength a) { printf("Varlen<lens=%p,offs=%p>", a.segment_lens, a.segment_offsets); }

CUTE_HOST_DEVICE
void print(cutlass::fmha::collective::PerBatchOffset a) { printf("PerBatchOffset<%p>", a.offsets); }

}  // namespace cute
