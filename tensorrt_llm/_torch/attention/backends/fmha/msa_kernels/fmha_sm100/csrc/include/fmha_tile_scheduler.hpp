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

#include <climits>

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"
#include "gmem_bounds_check.h"

namespace cutlass::fmha::kernel {

struct HostPrecomputedTileScheduler {
  // Packed layout (uint64_t):
  //   packed_work_info[i]:  [63:32] qo_tile_idx | [31:16] qo_head_idx | [15:0] batch_idx
  //   packed_work_range[i]: [63:32] work_ptr_end | [31:0] work_ptr
  // Widened from uint32_t so per-token sparse decode (batch_size = total_q, up to 65536)
  // and large prefill (qo_tile_idx can exceed 256 when q ≥ 64K) both fit without wrap.
  // See bug_sparse_decode_qo256.md.
  struct Arguments {
    uint64_t* packed_work_range;
    uint64_t* packed_work_info;
    int* kv_tile_begin_indices = nullptr;
    int* kv_tile_end_indices = nullptr;
    int* kv_split_indices = nullptr;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int packed_work_info_size = 0;
#endif
  };

  struct Params {
    uint64_t* packed_work_range;
    uint64_t* packed_work_info;
    int num_sm;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int packed_work_range_size;
    int packed_work_info_size;
#endif
  };

  Params params;
  int work_ptr;
  int work_ptr_end;
  int qo_tile_idx;
  int batch_idx;
  int qo_head_idx;
  bool is_valid_;

  CUTLASS_DEVICE
  void load_work_item() {
#ifdef FMHA_GMEM_BOUNDS_CHECK
    uint64_t packed = gmem_load_checked(params.packed_work_info, work_ptr,
                                         params.packed_work_info_size, "packed_work_info");
#else
    uint64_t packed = params.packed_work_info[work_ptr];
#endif
    qo_tile_idx = static_cast<int>(packed >> 32);
    qo_head_idx = static_cast<int>((packed >> 16) & 0xFFFF);
    batch_idx   = static_cast<int>(packed & 0xFFFF);
  }

  CUTLASS_DEVICE
  HostPrecomputedTileScheduler(Params const& params) {
    this->params = params;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    uint64_t range = gmem_load_checked(params.packed_work_range, (int)blockIdx.x,
                                        params.packed_work_range_size, "packed_work_range");
#else
    uint64_t range = params.packed_work_range[blockIdx.x];
#endif
    work_ptr     = static_cast<int>(range & 0xFFFFFFFF);
    work_ptr_end = static_cast<int>(range >> 32);
    if (work_ptr < work_ptr_end) {
      is_valid_ = true;
      load_work_item();
    } else {
      qo_tile_idx = 0;
      batch_idx = 0;
      qo_head_idx = 0;
      is_valid_ = false;
    }
  }

  static Params to_underlying_arguments(Arguments const& args, KernelHardwareInfo hw_info) {
    Params p{};
    p.packed_work_range = args.packed_work_range;
    p.packed_work_info = args.packed_work_info;
    p.num_sm = hw_info.sm_count;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    p.packed_work_range_size = hw_info.sm_count;
    p.packed_work_info_size = args.packed_work_info_size;
#endif
    return p;
  }

  static dim3 get_grid_shape(Params const& params) {
    dim3 grid(params.num_sm);
    return grid;
  }

  CUTLASS_DEVICE
  bool is_valid() const { return is_valid_; }

  CUTLASS_DEVICE
  auto get_block_coord() {
    return make_coord(qo_tile_idx, _0{}, make_coord(qo_head_idx, batch_idx));
  }

  CUTLASS_DEVICE int get_work_ptr() const { return work_ptr; }

  CUTLASS_DEVICE
  HostPrecomputedTileScheduler& operator++() {
    work_ptr++;
    is_valid_ = work_ptr < work_ptr_end;
    if (is_valid_) {
      load_work_item();
    }
    return *this;
  }
};

}  // namespace cutlass::fmha::kernel
