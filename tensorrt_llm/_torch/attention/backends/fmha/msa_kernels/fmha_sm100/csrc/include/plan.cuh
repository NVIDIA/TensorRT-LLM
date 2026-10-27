// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cub/block/block_scan.cuh>
#include "utils.cuh"

namespace flashinfer {

// ============================================================================
// Cost model constants
// ============================================================================

constexpr float kPerIterOverhead = 43;
constexpr float kTileGlobalOverhead = 110;
constexpr float kSMGlobalOverhead = 165;
constexpr float kSplitExtraOverhead = 150;


// ============================================================================
// Plan kernel constants
// ============================================================================

constexpr int MAX_SMS = 256;
constexpr int MAX_TASKS_PER_SM = 96;
constexpr int MAX_UNSPLIT = 4096;
constexpr int MAX_REBALANCE_ITERS = 1024;
constexpr int MIN_ITERS_PER_SPLIT = 4;

// ============================================================================
// Shared memory state
// ============================================================================

struct PlanSharedState {
  int     sm_cost[MAX_SMS];
  int16_t sm_task_count[MAX_SMS];

  int16_t task_kv_begin[MAX_SMS][MAX_TASKS_PER_SM];
  int16_t task_kv_end[MAX_SMS][MAX_TASKS_PER_SM];
  int16_t task_unsplit_id[MAX_SMS][MAX_TASKS_PER_SM];
  int8_t  task_sub_id[MAX_SMS][MAX_TASKS_PER_SM];

  int16_t unsplit_qo_tile[MAX_UNSPLIT];
  int8_t  unsplit_head[MAX_UNSPLIT];
  int16_t unsplit_batch[MAX_UNSPLIT];
  int16_t unsplit_kv_iters[MAX_UNSPLIT];
  int     unsplit_sub_counter[MAX_UNSPLIT];
  int     num_unsplit;

  uint     warp_scratch[8];  // for block reduce
  typename cub::BlockScan<int, MAX_SMS>::TempStorage scan_temp;
};

// ============================================================================
// Parallel min/max cost SM: pack (cost, idx) into int32
// cost in upper 24 bits, idx in lower 8 bits → min/max on int32 works
// ============================================================================

__device__ __forceinline__ uint pack_cost_idx(uint cost, uint idx) {
  return (cost << 8) | (idx & 0xff);
}
__device__ __forceinline__ uint unpack_idx(uint packed) { return packed & 0xff; }

// Layout: [63:32] qo_tile_idx (32b) | [31:16] qo_head_idx (16b) | [15:0] batch_idx (16b)
// 64-bit so per-token sparse decode (batch_size = total_q, up to 65536) can address batch
// without wrap, while large prefill keeps qo_tile_idx headroom (max_qo_len / qo_tile_size
// can exceed 256 for q ≥ 64K). See bug_sparse_decode_qo256.md.
__device__ __forceinline__ uint64_t pack_work_info(int qo_tile, int head, int batch) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(qo_tile)) << 32) |
         (static_cast<uint64_t>(head & 0xFFFF) << 16) |
         static_cast<uint64_t>(batch & 0xFFFF);
}

__device__ __forceinline__ uint64_t pack_work_range(int start, int end) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(end)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(start));
}

template <bool IsMin>
__device__ __forceinline__ int block_reduce_cost(PlanSharedState& s, int n) {
  int tid = static_cast<int>(threadIdx.x);
  uint val = (tid < n)
      ? pack_cost_idx(s.sm_cost[tid], tid)
      : pack_cost_idx(IsMin ? 0x7fffff : 0, tid);

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    uint other = __shfl_xor_sync(0xffffffff, val, offset);
    val = IsMin ? ::min(val, other) : ::max(val, other);
  }

  if (tid % 32 == 0) s.warp_scratch[tid / 32] = val;
  __syncthreads();

  if (tid < 8) {
    val = (tid * 32 < n) ? s.warp_scratch[tid]
        : pack_cost_idx(IsMin ? 0x7fffff : 0, tid * 32);
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      uint other = __shfl_xor_sync(0xff, val, offset);
      val = IsMin ? ::min(val, other) : ::max(val, other);
    }
    if (tid == 0) s.warp_scratch[0] = val;
  }
  __syncthreads();
  return s.warp_scratch[0];
}

__device__ __forceinline__ int find_min_cost_sm(PlanSharedState& s, int n) {
  return unpack_idx(block_reduce_cost<true>(s, n));
}

__device__ __forceinline__ int find_max_cost_sm(PlanSharedState& s, int n) {
  return unpack_idx(block_reduce_cost<false>(s, n));
}

// ============================================================================
// Helpers (thread 0 only)
// ============================================================================

__device__ __forceinline__ int compute_kv_iters(
    int batch_idx, int qo_tile_idx, int* qo_lens, int* kv_lens,
    int* qo_offsets, int qo_tile_size, int kv_tile_size, bool causal,
    int pack_factor = 1) {
  int kv_len = kv_lens[batch_idx];
  if (!causal) return (kv_len + kv_tile_size - 1) / kv_tile_size;
  int offset_q = qo_offsets ? qo_offsets[batch_idx] : (kv_len - qo_lens[batch_idx]);
  int packed_q_end = (qo_tile_idx + 1) * qo_tile_size;
  int q_end = (pack_factor > 1) ? (packed_q_end - 1) / pack_factor + 1 : packed_q_end;
  int eff_kv = q_end + offset_q;
  if (eff_kv > kv_len) eff_kv = kv_len;
  if (eff_kv <= 0) return 0;
  return (eff_kv + kv_tile_size - 1) / kv_tile_size;
}

__device__ __forceinline__ void add_task(
    PlanSharedState& s, int sm,
    int16_t kv_begin, int16_t kv_end, int16_t unsplit_id, int8_t sub_id) {
  int idx = s.sm_task_count[sm]++;
  s.task_kv_begin[sm][idx] = kv_begin;
  s.task_kv_end[sm][idx] = kv_end;
  s.task_unsplit_id[sm][idx] = unsplit_id;
  s.task_sub_id[sm][idx] = sub_id;
}

__device__ __forceinline__ bool split_task(
    PlanSharedState& s, int sm, int task_idx, int keep_iters, int target_sm, int max_splits) {
  int16_t uid = s.task_unsplit_id[sm][task_idx];

  int old_count = atomicAdd(&s.unsplit_sub_counter[uid], 1);
  if (old_count >= max_splits) {
    atomicAdd(&s.unsplit_sub_counter[uid], -1);
    return false;
  }

  int16_t orig_begin = s.task_kv_begin[sm][task_idx];
  int16_t split_point = orig_begin + keep_iters;
  int16_t orig_end = s.task_kv_end[sm][task_idx];
  s.task_kv_end[sm][task_idx] = split_point;

  add_task(s, target_sm, split_point, orig_end, uid, (int8_t)old_count);
  return true;
}

__device__ __forceinline__ int find_best_fit_task(
    PlanSharedState& s, int sm, int target_iters, int max_splits) {
  int n = s.sm_task_count[sm];
  int best_fit = -1, best_fit_iters = 0x7fff;
  int largest = -1, largest_iters = 0;
  int min_needed = target_iters + MIN_ITERS_PER_SPLIT;

  for (int t = 0; t < n; t++) {
    int iters = s.task_kv_end[sm][t] - s.task_kv_begin[sm][t];
    int16_t uid = s.task_unsplit_id[sm][t];
    if (s.unsplit_sub_counter[uid] >= max_splits) continue;
    if (iters < 2 * MIN_ITERS_PER_SPLIT) continue;

    if (iters >= min_needed && iters < best_fit_iters) {
      best_fit = t;
      best_fit_iters = iters;
    }
    if (iters > largest_iters) {
      largest = t;
      largest_iters = iters;
    }
  }
  return best_fit >= 0 ? best_fit : largest;
}

// ============================================================================
// Parallel output writing: each thread writes its own SM's tasks
// All threads participate (prefix sum for write offsets)
// ============================================================================

__device__ void parallel_write_output(
    PlanSharedState& s, int num_buckets,
    uint64_t* packed_work_range, uint64_t* packed_work_info,
    int* kv_tile_begin_indices = nullptr,
    int* kv_tile_end_indices = nullptr,
    int* kv_split_indices = nullptr) {

  int tid = static_cast<int>(threadIdx.x);
  int my_count = (tid < num_buckets) ? static_cast<int>(s.sm_task_count[tid]) : 0;

  int my_offset;
  cub::BlockScan<int, MAX_SMS>(s.scan_temp).ExclusiveSum(my_count, my_offset);
  __syncthreads();

  if (tid < num_buckets) {
    packed_work_range[tid] = pack_work_range(my_offset, my_offset + my_count);
    for (int t = 0; t < my_count; t++) {
      int pos = my_offset + t;
      int16_t uid = s.task_unsplit_id[tid][t];
      packed_work_info[pos] = pack_work_info(
          s.unsplit_qo_tile[uid], s.unsplit_head[uid], s.unsplit_batch[uid]);
      if (kv_tile_begin_indices) kv_tile_begin_indices[pos] = s.task_kv_begin[tid][t];
      if (kv_tile_end_indices) kv_tile_end_indices[pos] = s.task_kv_end[tid][t];
      if (kv_split_indices) kv_split_indices[pos] = s.task_sub_id[tid][t];
    }
  }
  __syncthreads();
}

// ============================================================================
// Direct nosplit: enumerate tiles + greedy assign + write output in one pass
// No smem unsplit table needed — for large tile counts (>MAX_UNSPLIT)
// ============================================================================

__device__ int direct_greedy_rank(
    PlanSharedState& s, int num_buckets,
    int* qo_lens, int* kv_lens, int* qo_offsets,
    int qo_tile_size, int kv_tile_size, int batch_size, int num_heads, bool causal,
    int num_kv_splits,
    uint64_t* packed_work_range, uint64_t* packed_work_info,
    int* kv_tile_begin_indices = nullptr, int* kv_tile_end_indices = nullptr,
    int* kv_split_indices = nullptr,
    int* num_kv_splits_per_row = nullptr, int* qo_segment_offsets = nullptr,
    int pack_factor = 1) {

  int tid = static_cast<int>(threadIdx.x);

  __shared__ int _max_qo_tiles;
  if (tid == 0) {
    int mq = 0;
    for (int b = 0; b < batch_size; b++) {
      int nt = ceil_div(qo_lens[b], qo_tile_size);
      if (nt > mq) mq = nt;
    }
    _max_qo_tiles = mq;
  }
  __syncthreads();
  int max_qo_tiles = _max_qo_tiles;

  if (tid < num_buckets) {
    s.sm_cost[tid] = kSMGlobalOverhead;
    s.sm_task_count[tid] = 0;
  }
  __syncthreads();

  // Pass 1: batch-assign heads in reverse QO tile order
  for (int qt = max_qo_tiles - 1; qt >= 0; qt--) {
    for (int b = 0; b < batch_size; b++) {
      if (qt >= ceil_div(qo_lens[b], qo_tile_size)) continue;
      int total_ki = compute_kv_iters(b, qt, qo_lens, kv_lens, qo_offsets,
                                       qo_tile_size, kv_tile_size, causal, pack_factor);
      if (total_ki <= 0) continue;
      int actual_splits = (num_kv_splits <= 1) ? 1
          : ::min(num_kv_splits, ::max(1, total_ki / MIN_ITERS_PER_SPLIT));
      for (int sp = 0; sp < actual_splits; sp++) {
        int kb = sp * ((total_ki + actual_splits - 1) / actual_splits);
        int ke = ::min(kb + (total_ki + actual_splits - 1) / actual_splits, total_ki);
        int piece_ki = ke - kb;
        int tile_cost = static_cast<int>(kPerIterOverhead * piece_ki + kTileGlobalOverhead);

        for (int h_off = 0; h_off < num_heads; ) {
          int batch_count = num_heads - h_off;
          if (batch_count > num_buckets) batch_count = num_buckets;

          int my_cost = (tid < num_buckets) ? s.sm_cost[tid] : 0x7fffffff;
          int my_rank = 0;
          if (tid < num_buckets) {
            for (int i = 0; i < num_buckets; i++) {
              int oc = s.sm_cost[i];
              if (oc < my_cost || (oc == my_cost && i < tid))
                my_rank++;
            }
          }
          bool i_get = (tid < num_buckets) && (my_rank < batch_count);
          __syncthreads();

          if (i_get) {
            s.sm_cost[tid] += tile_cost;
            s.sm_task_count[tid]++;
          }
          __syncthreads();
          h_off += batch_count;
        }
      }
    }
  }

  int max_sm = find_max_cost_sm(s, num_buckets);
  int max_cost = s.sm_cost[max_sm];

  // Compute work_indptr, reset for pass 2
  __shared__ int _work_offsets[MAX_SMS + 1];
  if (threadIdx.x == 0) {
    int pos = 0;
    for (int sm = 0; sm < num_buckets; sm++) {
      _work_offsets[sm] = pos;
      pos += s.sm_task_count[sm];
    }
    _work_offsets[num_buckets] = pos;
    for (int sm = 0; sm < num_buckets; sm++)
      s.sm_task_count[sm] = 0;
  }
  __syncthreads();

  // Pass 2: same order, batch-assign + write tile info
  if (tid < num_buckets) s.sm_cost[tid] = kSMGlobalOverhead;
  __syncthreads();

  for (int qt = max_qo_tiles - 1; qt >= 0; qt--) {
    for (int b = 0; b < batch_size; b++) {
      if (qt >= ceil_div(qo_lens[b], qo_tile_size)) continue;
      int total_ki = compute_kv_iters(b, qt, qo_lens, kv_lens, qo_offsets,
                                       qo_tile_size, kv_tile_size, causal, pack_factor);
      if (total_ki <= 0) continue;
      int actual_splits = (num_kv_splits <= 1) ? 1
          : ::min(num_kv_splits, ::max(1, total_ki / MIN_ITERS_PER_SPLIT));
      if (threadIdx.x == 0 && num_kv_splits_per_row && qo_segment_offsets) {
        int seg_off = qo_segment_offsets[b];
        int qo_len = qo_lens[b];
        int row_start = qt * qo_tile_size;
        int row_end = row_start + qo_tile_size < qo_len ? row_start + qo_tile_size : qo_len;
        for (int r = row_start; r < row_end; r++)
          num_kv_splits_per_row[seg_off + r] = actual_splits;
      }
      for (int sp = 0; sp < actual_splits; sp++) {
        int kb = sp * ((total_ki + actual_splits - 1) / actual_splits);
        int ke = ::min(kb + (total_ki + actual_splits - 1) / actual_splits, total_ki);
        int piece_ki = ke - kb;
        int tile_cost = static_cast<int>(kPerIterOverhead * piece_ki + kTileGlobalOverhead);

        for (int h_off = 0; h_off < num_heads; ) {
          int batch_count = num_heads - h_off;
          if (batch_count > num_buckets) batch_count = num_buckets;

          int my_cost = (tid < num_buckets) ? s.sm_cost[tid] : 0x7fffffff;
          int my_rank = 0;
          if (tid < num_buckets) {
            for (int i = 0; i < num_buckets; i++) {
              int oc = s.sm_cost[i];
              if (oc < my_cost || (oc == my_cost && i < tid))
                my_rank++;
            }
          }
          bool i_get = (tid < num_buckets) && (my_rank < batch_count);
          __syncthreads();

          if (i_get) {
            s.sm_cost[tid] += tile_cost;
            int pos = _work_offsets[tid] + s.sm_task_count[tid]++;
            packed_work_info[pos] = pack_work_info(qt, h_off + my_rank, b);
            if (kv_tile_begin_indices) kv_tile_begin_indices[pos] = kb;
            if (kv_tile_end_indices) kv_tile_end_indices[pos] = ke;
            if (kv_split_indices) kv_split_indices[pos] = sp;
          }
          __syncthreads();
          h_off += batch_count;
        }
      }
    }
  }
  __syncthreads();
  if (tid < num_buckets) {
    packed_work_range[tid] = pack_work_range(
        _work_offsets[tid], _work_offsets[tid] + s.sm_task_count[tid]);
  }
  __syncthreads();
  return max_cost;
}

// PLAN_PAR v2 (MSA, PREREG_M31_MSA_PLAN_PAR_v1 b5d18c02): the original direct_greedy's picks computed by warp 0 alone: each
// head-batch takes the batch_count lowest (cost, bucket index) buckets one at a time as a 64-bit warp argmin over
// (cost << 32 | index), already-picked buckets of the batch excluded — the same set, rank (head offset) and tie rule as the
// original per-thread rank scan, without the O(buckets) scan per thread and the 2 block barriers per pick.
__device__ __forceinline__ unsigned long long plan_warp_min_u64(unsigned long long v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    unsigned long long o = __shfl_xor_sync(0xffffffffu, v, off);
    v = o < v ? o : v;
  }
  return v;
}

__device__ int direct_greedy_warp(
    PlanSharedState& s, int num_buckets,
    int* qo_lens, int* kv_lens, int* qo_offsets,
    int qo_tile_size, int kv_tile_size, int batch_size, int num_heads, bool causal,
    int num_kv_splits,
    uint64_t* packed_work_range, uint64_t* packed_work_info,
    int* kv_tile_begin_indices = nullptr, int* kv_tile_end_indices = nullptr,
    int* kv_split_indices = nullptr,
    int* num_kv_splits_per_row = nullptr, int* qo_segment_offsets = nullptr,
    int pack_factor = 1) {

  int tid = static_cast<int>(threadIdx.x);
  __shared__ int _work_offsets[MAX_SMS + 1];
  __shared__ int _max_cost;
  constexpr int kPerLane = MAX_SMS / 32;  // buckets b = lane + 32 * j

  if (tid < num_buckets) {
    s.sm_cost[tid] = kSMGlobalOverhead;
    s.sm_task_count[tid] = 0;
  }
  __syncthreads();

  if (tid < 32) {
    int const lane = tid;
    int max_qo_tiles = 0;
    for (int b = 0; b < batch_size; b++) {
      int nt = ceil_div(qo_lens[b], qo_tile_size);
      if (nt > max_qo_tiles) max_qo_tiles = nt;
    }
    for (int pass = 0; pass < 2; pass++) {
      if (pass == 1) {
        __syncwarp();
        if (lane == 0) {
          // The original find_max_cost_sm, bit for bit: max over the 32-bit packed (cost << 8 | idx & 0xff) of all MAX_SMS
          // block threads (threads >= num_buckets contribute (0 << 8 | tid & 0xff)), then sm_cost at that index.
          unsigned int best = 0u;
          for (int t = 0; t < MAX_SMS; t++) {
            unsigned int v = (t < num_buckets) ? pack_cost_idx(static_cast<unsigned int>(s.sm_cost[t]), t)
                                               : pack_cost_idx(0u, t);
            if (v > best) best = v;
          }
          _max_cost = s.sm_cost[unpack_idx(best)];
          int pos = 0;
          for (int sm = 0; sm < num_buckets; sm++) {
            _work_offsets[sm] = pos;
            pos += s.sm_task_count[sm];
          }
          _work_offsets[num_buckets] = pos;
        }
        __syncwarp();
        for (int sm = lane; sm < num_buckets; sm += 32) {
          s.sm_task_count[sm] = 0;
          s.sm_cost[sm] = kSMGlobalOverhead;
        }
        __syncwarp();
      }
      for (int qt = max_qo_tiles - 1; qt >= 0; qt--) {
        for (int b = 0; b < batch_size; b++) {
          if (qt >= ceil_div(qo_lens[b], qo_tile_size)) continue;
          int total_ki = compute_kv_iters(b, qt, qo_lens, kv_lens, qo_offsets,
                                           qo_tile_size, kv_tile_size, causal, pack_factor);
          if (total_ki <= 0) continue;
          int actual_splits = (num_kv_splits <= 1) ? 1
              : ::min(num_kv_splits, ::max(1, total_ki / MIN_ITERS_PER_SPLIT));
          if (pass == 1 && num_kv_splits_per_row && qo_segment_offsets) {
            int seg_off = qo_segment_offsets[b];
            int qo_len = qo_lens[b];
            int row_start = qt * qo_tile_size;
            int row_end = row_start + qo_tile_size < qo_len ? row_start + qo_tile_size : qo_len;
            for (int r = row_start + lane; r < row_end; r += 32)
              num_kv_splits_per_row[seg_off + r] = actual_splits;
          }
          for (int sp = 0; sp < actual_splits; sp++) {
            int kb = sp * ((total_ki + actual_splits - 1) / actual_splits);
            int ke = ::min(kb + (total_ki + actual_splits - 1) / actual_splits, total_ki);
            int piece_ki = ke - kb;
            int tile_cost = static_cast<int>(kPerIterOverhead * piece_ki + kTileGlobalOverhead);
            for (int h_off = 0; h_off < num_heads; ) {
              int batch_count = num_heads - h_off;
              if (batch_count > num_buckets) batch_count = num_buckets;
              unsigned int taken = 0u;  // bit j: this lane's bucket lane + 32 j is picked in this batch
              for (int r = 0; r < batch_count; r++) {
                unsigned long long mine = ~0ull;
#pragma unroll
                for (int j = 0; j < kPerLane; j++) {
                  int bk = lane + 32 * j;
                  if (bk < num_buckets && !((taken >> j) & 1u)) {
                    // Signed cost order (a piece past total_ki has a negative cost in the original): bias the sign bit.
                    unsigned long long key =
                        (static_cast<unsigned long long>(static_cast<unsigned int>(s.sm_cost[bk]) ^ 0x80000000u) << 32) |
                        static_cast<unsigned long long>(static_cast<unsigned int>(bk));
                    mine = key < mine ? key : mine;
                  }
                }
                unsigned long long win = plan_warp_min_u64(mine);
                int bk = static_cast<int>(win & 0xffffffffull);
                if ((bk & 31) == lane) {
                  int j = bk >> 5;
                  taken |= 1u << j;
                  s.sm_cost[bk] += tile_cost;
                  if (pass == 1) {
                    int pos = _work_offsets[bk] + s.sm_task_count[bk];
                    packed_work_info[pos] = pack_work_info(qt, h_off + r, b);
                    if (kv_tile_begin_indices) kv_tile_begin_indices[pos] = kb;
                    if (kv_tile_end_indices) kv_tile_end_indices[pos] = ke;
                    if (kv_split_indices) kv_split_indices[pos] = sp;
                  }
                  s.sm_task_count[bk]++;
                }
                // The cost update above lands after this pick's argmin; later picks of the batch exclude bk via taken
                // (the original ranks all picks of a batch on the pre-batch costs).
              }
              __syncwarp();
              h_off += batch_count;
            }
          }
        }
      }
    }
  }
  __syncthreads();
  if (tid < num_buckets) {
    packed_work_range[tid] = pack_work_range(
        _work_offsets[tid], _work_offsets[tid] + s.sm_task_count[tid]);
  }
  __syncthreads();
  return _max_cost;
}

// PLAN_PAR v3: the warp argmin for few heads per batch (the prefill proxy plan: 1-4 index heads per rank); the original
// rank scan otherwise (many heads per batch make sequential argmins slower than one parallel rank). Both are the same greedy.
// planpar_max_heads = 0 (TRTLLM_M31_DISABLE_TP2CTX_PLAN_PAR=1, read by plan_kernel_wrapper) takes the original rank scan always.
__device__ int direct_greedy(
    PlanSharedState& s, int num_buckets,
    int* qo_lens, int* kv_lens, int* qo_offsets,
    int qo_tile_size, int kv_tile_size, int batch_size, int num_heads, bool causal,
    int num_kv_splits,
    uint64_t* packed_work_range, uint64_t* packed_work_info,
    int* kv_tile_begin_indices = nullptr, int* kv_tile_end_indices = nullptr,
    int* kv_split_indices = nullptr,
    int* num_kv_splits_per_row = nullptr, int* qo_segment_offsets = nullptr,
    int pack_factor = 1, int planpar_max_heads = 4) {
  if (num_heads <= planpar_max_heads) {
    return direct_greedy_warp(s, num_buckets, qo_lens, kv_lens, qo_offsets, qo_tile_size, kv_tile_size, batch_size,
                              num_heads, causal, num_kv_splits, packed_work_range, packed_work_info, kv_tile_begin_indices,
                              kv_tile_end_indices, kv_split_indices, num_kv_splits_per_row, qo_segment_offsets, pack_factor);
  }
  return direct_greedy_rank(s, num_buckets, qo_lens, kv_lens, qo_offsets, qo_tile_size, kv_tile_size, batch_size, num_heads,
                            causal, num_kv_splits, packed_work_range, packed_work_info, kv_tile_begin_indices,
                            kv_tile_end_indices, kv_split_indices, num_kv_splits_per_row, qo_segment_offsets, pack_factor);
}

// ============================================================================
// Enumerate unsplit tiles — all threads participate (parallel over heads)
// ============================================================================

__device__ void enumerate_unsplit_tiles(
    PlanSharedState& s, int* qo_lens, int* kv_lens, int* qo_offsets,
    int qo_tile_size, int kv_tile_size, int batch_size, int num_heads, bool causal,
    int pack_factor = 1) {
  int tid = static_cast<int>(threadIdx.x);
  int max_qt = 0;
  for (int b = 0; b < batch_size; b++) {
    int nt = ceil_div(qo_lens[b], qo_tile_size);
    if (nt > max_qt) max_qt = nt;
  }
  int uid = 0;
  for (int qt = max_qt - 1; qt >= 0; qt--) {
    for (int b = 0; b < batch_size; b++) {
      if (qt >= ceil_div(qo_lens[b], qo_tile_size)) continue;
      int ki = compute_kv_iters(b, qt, qo_lens, kv_lens, qo_offsets,
                                 qo_tile_size, kv_tile_size, causal, pack_factor);
      if (ki <= 0) continue;
      int count = num_heads;
      if (uid + count > MAX_UNSPLIT) count = MAX_UNSPLIT - uid;
      for (int h = tid; h < count; h += static_cast<int>(blockDim.x)) {
        s.unsplit_qo_tile[uid + h] = qt;
        s.unsplit_head[uid + h] = h;
        s.unsplit_batch[uid + h] = b;
        s.unsplit_kv_iters[uid + h] = ki;
        s.unsplit_sub_counter[uid + h] = 0;
      }
      uid += count;
    }
  }
  if (tid == 0) s.num_unsplit = uid;
}

// ============================================================================
// LPT sort: insertion sort unsplit tiles by kv_iters descending (thread 0 only)
// After reverse-order enumeration the table is nearly sorted, so O(n) expected.
// ============================================================================

__device__ void sort_unsplit_descending(PlanSharedState& s) {
  int n = s.num_unsplit;
  for (int i = 1; i < n; i++) {
    int16_t ki = s.unsplit_kv_iters[i];
    if (ki <= s.unsplit_kv_iters[i - 1]) continue;
    int16_t qt = s.unsplit_qo_tile[i];
    int8_t  h  = s.unsplit_head[i];
    int16_t b  = s.unsplit_batch[i];
    int j = i - 1;
    while (j >= 0 && s.unsplit_kv_iters[j] < ki) {
      s.unsplit_kv_iters[j + 1] = s.unsplit_kv_iters[j];
      s.unsplit_qo_tile[j + 1] = s.unsplit_qo_tile[j];
      s.unsplit_head[j + 1]    = s.unsplit_head[j];
      s.unsplit_batch[j + 1]   = s.unsplit_batch[j];
      j--;
    }
    s.unsplit_kv_iters[j + 1] = ki;
    s.unsplit_qo_tile[j + 1]  = qt;
    s.unsplit_head[j + 1]     = h;
    s.unsplit_batch[j + 1]    = b;
  }
}

// ============================================================================
// Phase 0: Nosplit greedy with batch assignment
// Tiles with same kv_iters are assigned in bulk via parallel rank computation.
// AssignTasks: populate smem task arrays (for subsequent rebalance)
// WriteGlobal: write final output to global memory
// ============================================================================

template <bool AssignTasks, bool WriteGlobal>
__device__ int phase0_nosplit_greedy(
    PlanSharedState& s, int num_buckets, int num_heads,
    uint64_t* packed_work_range, uint64_t* packed_work_info) {

  int tid = static_cast<int>(threadIdx.x);
  if (tid < num_buckets) {
    s.sm_cost[tid] = kSMGlobalOverhead;
    s.sm_task_count[tid] = 0;
  }
  __syncthreads();

  int n = s.num_unsplit;
  int step = num_heads > 0 ? num_heads : 1;
  int u = 0;
  while (u < n) {
    int ki = s.unsplit_kv_iters[u];
    int group_end = u + step;
    while (group_end < n && s.unsplit_kv_iters[group_end] == ki)
      group_end += step;
    if (group_end > n) group_end = n;
    int tile_cost = static_cast<int>(kPerIterOverhead * ki + kTileGlobalOverhead);

    while (u < group_end) {
      int batch_count = group_end - u;
      if (batch_count > num_buckets) batch_count = num_buckets;

      int my_cost = (tid < num_buckets) ? s.sm_cost[tid] : 0x7fffffff;
      int my_rank = 0;
      if (tid < num_buckets) {
        for (int i = 0; i < num_buckets; i++) {
          int oc = s.sm_cost[i];
          if (oc < my_cost || (oc == my_cost && i < tid))
            my_rank++;
        }
      }

      bool i_get_tile = (tid < num_buckets) && (my_rank < batch_count);
      __syncthreads();

      if (i_get_tile) {
        s.sm_cost[tid] += tile_cost;
        if constexpr (AssignTasks) {
          int tile_u = u + my_rank;
          add_task(s, tid, 0, ki, tile_u, 0);
          s.unsplit_sub_counter[tile_u] = 1;
        }
      }
      __syncthreads();
      u += batch_count;
    }
  }

  int max_sm = find_max_cost_sm(s, num_buckets);
  int max_cost = s.sm_cost[max_sm];

  if constexpr (WriteGlobal) {
    parallel_write_output(s, num_buckets,
        packed_work_range, packed_work_info);
  }
  __syncthreads();
  return max_cost;
}

// ============================================================================
// Phase 1: Linear scan fill with LPT-sorted input (thread 0 only)
// ============================================================================

__device__ void phase1_linear_fill(
    PlanSharedState& s, int num_buckets, int max_splits) {

  int tid = static_cast<int>(threadIdx.x);
  if (tid < num_buckets) {
    s.sm_cost[tid] = kSMGlobalOverhead;
    s.sm_task_count[tid] = 0;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int n = s.num_unsplit;
    int total_iters = 0;
    for (int u = 0; u < n; u++)
      total_iters += s.unsplit_kv_iters[u];

    int avg_iters = (total_iters + num_buckets - 1) / num_buckets;
    int target_budget = kTileGlobalOverhead + avg_iters * kPerIterOverhead * 1.02;
    int cur_sm = 0;
    int sm_budget = target_budget;
    int min_piece_cost = kTileGlobalOverhead + MIN_ITERS_PER_SPLIT * kPerIterOverhead;

    for (int u = 0; u < n; u++) {
      int tile_iters = s.unsplit_kv_iters[u];
      if (tile_iters <= 0) { s.unsplit_sub_counter[u] = 1; continue; }

      int tile_pos = 0;
      int sub_id = 0;

      while (tile_pos < tile_iters) {
        while (cur_sm < num_buckets && s.sm_task_count[cur_sm] >= MAX_TASKS_PER_SM) {
          cur_sm++;
          sm_budget = target_budget;
        }
        if (cur_sm >= num_buckets) {
          int best_sm = 0;
          for (int i = 1; i < num_buckets; i++)
            if (s.sm_task_count[i] < s.sm_task_count[best_sm]) best_sm = i;
          if (s.sm_task_count[best_sm] >= MAX_TASKS_PER_SM) break;
          cur_sm = best_sm;
          sm_budget = 0x7fffffff;
        }

        int avail = tile_iters - tile_pos;
        int max_iters = (sm_budget - kTileGlobalOverhead) / kPerIterOverhead;
        if (max_iters < MIN_ITERS_PER_SPLIT) max_iters = MIN_ITERS_PER_SPLIT;
        if (sub_id + 1 >= max_splits) max_iters = avail;
        int take = avail < max_iters ? avail : max_iters;

        add_task(s, cur_sm, tile_pos, tile_pos + take, u, sub_id++);
        int piece_cost = kPerIterOverhead * take + kTileGlobalOverhead;
        s.sm_cost[cur_sm] += piece_cost;
        sm_budget -= piece_cost;
        tile_pos += take;

        if (sm_budget < min_piece_cost) {
          cur_sm++;
          sm_budget = target_budget;
        }
      }
      s.unsplit_sub_counter[u] = sub_id;
    }
  }
  __syncthreads();
}

// ============================================================================
// Phase 1b: Proportional fill — each tile gets SMs proportional to its iters
// Avoids tile-boundary crossings (1 task per SM), near-optimal balance.
// ============================================================================

__device__ void phase1_proportional_fill(
    PlanSharedState& s, int num_buckets, int max_splits) {

  int tid = static_cast<int>(threadIdx.x);
  if (tid < num_buckets) {
    s.sm_cost[tid] = kSMGlobalOverhead;
    s.sm_task_count[tid] = 0;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int n = s.num_unsplit;
    int total_iters = 0;
    for (int u = 0; u < n; u++)
      total_iters += s.unsplit_kv_iters[u];

    if (total_iters == 0) { __syncthreads(); return; }

    int sm_pos = 0;
    long long cumul_iters = 0;

    for (int u = 0; u < n; u++) {
      int tile_iters = s.unsplit_kv_iters[u];
      if (tile_iters <= 0) { s.unsplit_sub_counter[u] = 1; continue; }

      cumul_iters += tile_iters;
      int target_end = (u == n - 1) ? num_buckets
          : static_cast<int>(cumul_iters * num_buckets / total_iters);
      int tile_sms = target_end - sm_pos;
      if (tile_sms < 1) tile_sms = 1;
      if (sm_pos + tile_sms > num_buckets) tile_sms = num_buckets - sm_pos;
      int actual_splits = tile_sms < max_splits ? tile_sms : max_splits;

      int base = tile_iters / actual_splits;
      int extra = tile_iters % actual_splits;
      int kv_pos = 0;
      for (int j = 0; j < actual_splits; j++) {
        int take = base + (j < extra ? 1 : 0);
        int sm = sm_pos + (j < actual_splits ? (long long)j * tile_sms / actual_splits : j);
        add_task(s, sm, kv_pos, kv_pos + take, u, j);
        s.sm_cost[sm] += kPerIterOverhead * take + kTileGlobalOverhead;
        kv_pos += take;
      }
      s.unsplit_sub_counter[u] = actual_splits;
      sm_pos += tile_sms;
    }
  }
  __syncthreads();
}

// ============================================================================
// Phase 2: Iterative rebalance
// ============================================================================

__device__ void phase2_rebalance(
    PlanSharedState& s, int num_buckets, int max_splits) {

  for (int iter = 0; iter < MAX_REBALANCE_ITERS; iter++) {
    int max_sm = find_max_cost_sm(s, num_buckets);
    int min_sm = find_min_cost_sm(s, num_buckets);

    __shared__ int _done;
    if (threadIdx.x == 0) {
      _done = 0;
      int gap = s.sm_cost[max_sm] - s.sm_cost[min_sm];
      if (gap <= kTileGlobalOverhead) { _done = 1; }
      else {
        int n = s.sm_task_count[max_sm];

        // Strategy 1: move a whole task if it reduces gap
        int best_move = -1;
        int best_new_gap = gap;
        for (int t = 0; t < n; t++) {
          int iters = s.task_kv_end[max_sm][t] - s.task_kv_begin[max_sm][t];
          int task_cost = kPerIterOverhead * iters + kTileGlobalOverhead;
          int new_max = s.sm_cost[max_sm] - task_cost;
          int new_min = s.sm_cost[min_sm] + task_cost;
          if (new_max >= new_min) {
            int new_gap = new_max - new_min;
            if (new_gap < best_new_gap) {
              best_move = t;
              best_new_gap = new_gap;
            }
          }
        }

        if (best_move >= 0) {
          // Move entire task
          int16_t kb = s.task_kv_begin[max_sm][best_move];
          int16_t ke = s.task_kv_end[max_sm][best_move];
          int16_t uid = s.task_unsplit_id[max_sm][best_move];
          int8_t sid = s.task_sub_id[max_sm][best_move];
          int iters = ke - kb;
          int task_cost = kPerIterOverhead * iters + kTileGlobalOverhead;

          // Remove from max_sm (swap with last)
          int last = s.sm_task_count[max_sm] - 1;
          if (best_move != last) {
            s.task_kv_begin[max_sm][best_move] = s.task_kv_begin[max_sm][last];
            s.task_kv_end[max_sm][best_move] = s.task_kv_end[max_sm][last];
            s.task_unsplit_id[max_sm][best_move] = s.task_unsplit_id[max_sm][last];
            s.task_sub_id[max_sm][best_move] = s.task_sub_id[max_sm][last];
          }
          s.sm_task_count[max_sm]--;
          add_task(s, min_sm, kb, ke, uid, sid);
          s.sm_cost[max_sm] -= task_cost;
          s.sm_cost[min_sm] += task_cost;
        } else {
          // Strategy 2: split — target half the gap
          int move_iters = (gap - kTileGlobalOverhead) / (2 * kPerIterOverhead);
          if (move_iters < MIN_ITERS_PER_SPLIT) { _done = 1; }
          else {
            int best = find_best_fit_task(s, max_sm, move_iters, max_splits);
            if (best < 0) { _done = 1; }
            else {
              int task_iters = s.task_kv_end[max_sm][best] - s.task_kv_begin[max_sm][best];
              int keep = task_iters - move_iters;
              if (keep < MIN_ITERS_PER_SPLIT) keep = MIN_ITERS_PER_SPLIT;
              int actual_move = task_iters - keep;
              if (actual_move < MIN_ITERS_PER_SPLIT) { _done = 1; }
              else {
                if (!split_task(s, max_sm, best, keep, min_sm, max_splits)) { _done = 1; }
                else {
                s.sm_cost[max_sm] -= actual_move * kPerIterOverhead;
                s.sm_cost[min_sm] += actual_move * kPerIterOverhead + kTileGlobalOverhead;
                }
              }
            }
          }
        }
      }
    }
    __syncthreads();
    if (_done) break;
  }
}

// ============================================================================
// Phase 2 (parallel): all threads participate, each handles its own SM
// Excess/deficit SMs paired via prefix sum, all splits happen simultaneously
// ============================================================================

__device__ void phase2_rebalance_parallel(
    PlanSharedState& s, int num_buckets, int max_splits) {

  int tid = static_cast<int>(threadIdx.x);
  __shared__ int16_t _deficit_sms[MAX_SMS];
  __shared__ int _n_deficit;
  __shared__ int _gap;

  constexpr int MAX_PARALLEL_ITERS = 8;
  for (int iter = 0; iter < MAX_PARALLEL_ITERS; iter++) {
    int max_sm = find_max_cost_sm(s, num_buckets);
    int min_sm = find_min_cost_sm(s, num_buckets);
    if (tid == 0) _gap = s.sm_cost[max_sm] - s.sm_cost[min_sm];
    __syncthreads();
    if (_gap <= kTileGlobalOverhead) break;

    // Parallel sum → average cost
    int my_cost = (tid < num_buckets) ? s.sm_cost[tid] : 0;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      my_cost += __shfl_xor_sync(0xffffffff, my_cost, offset);
    if (tid % 32 == 0) s.warp_scratch[tid / 32] = my_cost;
    __syncthreads();
    if (tid < 8) {
      int v = (tid * 32 < num_buckets) ? (int)s.warp_scratch[tid] : 0;
      #pragma unroll
      for (int offset = 4; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xff, v, offset);
      if (tid == 0) s.warp_scratch[0] = v / num_buckets;
    }
    __syncthreads();
    int avg_cost = (int)s.warp_scratch[0];

    // Parallel deficit collection via prefix sum
    int is_deficit = (tid < num_buckets && s.sm_cost[tid] < avg_cost) ? 1 : 0;
    int deficit_idx;
    cub::BlockScan<int, MAX_SMS>(s.scan_temp).ExclusiveSum(is_deficit, deficit_idx);
    __syncthreads();
    if (is_deficit) _deficit_sms[deficit_idx] = tid;
    if (tid == num_buckets - 1) _n_deficit = deficit_idx + is_deficit;
    __syncthreads();
    if (_n_deficit == 0) break;

    // Each above-average SM gets a unique index via prefix sum
    int is_excess = (tid < num_buckets && s.sm_cost[tid] > avg_cost) ? 1 : 0;
    int excess_idx;
    cub::BlockScan<int, MAX_SMS>(s.scan_temp).ExclusiveSum(is_excess, excess_idx);
    __syncthreads();

    // Each excess thread independently splits/moves to its unique deficit target
    if (is_excess && excess_idx < _n_deficit) {
      int sm = tid;
      int target_sm = _deficit_sms[excess_idx];
      int n_tasks = s.sm_task_count[sm];
      int sm_gap = s.sm_cost[sm] - s.sm_cost[target_sm];

      // Strategy 1: move a whole task if it reduces the pair gap
      int best_move = -1;
      int best_new_gap = sm_gap;
      for (int t = 0; t < n_tasks; t++) {
        int iters = s.task_kv_end[sm][t] - s.task_kv_begin[sm][t];
        int task_cost = static_cast<int>(kPerIterOverhead * iters + kTileGlobalOverhead);
        int new_src = s.sm_cost[sm] - task_cost;
        int new_dst = s.sm_cost[target_sm] + task_cost;
        if (new_src >= new_dst && (new_src - new_dst) < best_new_gap) {
          best_move = t;
          best_new_gap = new_src - new_dst;
        }
      }

      if (best_move >= 0) {
        int16_t kb = s.task_kv_begin[sm][best_move];
        int16_t ke = s.task_kv_end[sm][best_move];
        int16_t uid = s.task_unsplit_id[sm][best_move];
        int8_t sid = s.task_sub_id[sm][best_move];
        int task_cost = static_cast<int>(kPerIterOverhead * (ke - kb) + kTileGlobalOverhead);
        int last = s.sm_task_count[sm] - 1;
        if (best_move != last) {
          s.task_kv_begin[sm][best_move] = s.task_kv_begin[sm][last];
          s.task_kv_end[sm][best_move] = s.task_kv_end[sm][last];
          s.task_unsplit_id[sm][best_move] = s.task_unsplit_id[sm][last];
          s.task_sub_id[sm][best_move] = s.task_sub_id[sm][last];
        }
        s.sm_task_count[sm]--;
        add_task(s, target_sm, kb, ke, uid, sid);
        s.sm_cost[sm] -= task_cost;
        s.sm_cost[target_sm] += task_cost;
      } else {
        // Strategy 2: split
        int move_iters = static_cast<int>((sm_gap - kTileGlobalOverhead) / (2 * kPerIterOverhead));
        if (move_iters >= MIN_ITERS_PER_SPLIT) {
          int best = find_best_fit_task(s, sm, move_iters, max_splits);
          if (best >= 0) {
            int task_iters = s.task_kv_end[sm][best] - s.task_kv_begin[sm][best];
            int keep = task_iters - move_iters;
            if (keep < MIN_ITERS_PER_SPLIT) keep = MIN_ITERS_PER_SPLIT;
            int actual_move = task_iters - keep;
            if (actual_move >= MIN_ITERS_PER_SPLIT) {
              if (split_task(s, sm, best, keep, target_sm, max_splits)) {
              s.sm_cost[sm] -= static_cast<int>(actual_move * kPerIterOverhead);
              s.sm_cost[target_sm] += static_cast<int>(actual_move * kPerIterOverhead + kTileGlobalOverhead);
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }
}

// ============================================================================
// Phase 3: Write split output (all threads for parallel write, thread 0 for num_kv_splits_per_row)
// ============================================================================

__device__ void phase3_write_output(
    PlanSharedState& s, int num_buckets,
    uint64_t* packed_work_range, uint64_t* packed_work_info,
    int* kv_tile_begin_indices, int* kv_tile_end_indices,
    int* kv_split_indices, int* num_kv_splits_per_row,
    int* qo_segment_offsets, int* qo_lens, int qo_tile_size) {

  parallel_write_output(s, num_buckets,
      packed_work_range, packed_work_info,
      kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices);

  if (threadIdx.x == 0 && num_kv_splits_per_row && qo_segment_offsets && qo_lens) {
    int n = s.num_unsplit;
    for (int u = 0; u < n; u++) {
      int b = s.unsplit_batch[u];
      int qt = s.unsplit_qo_tile[u];
      int splits = s.unsplit_sub_counter[u];
      int seg_off = qo_segment_offsets[b];
      int qo_len = qo_lens[b];
      int row_start = qt * qo_tile_size;
      int row_end = row_start + qo_tile_size;
      if (row_end > qo_len) row_end = qo_len;
      for (int r = row_start; r < row_end; r++) {
        // Take max across heads for same (batch, qo_tile) row
        int cur = num_kv_splits_per_row[seg_off + r];
        if (splits > cur) num_kv_splits_per_row[seg_off + r] = splits;
      }
    }
  }
  __syncthreads();
}

// ============================================================================
// Main plan kernel
// ============================================================================

__global__ void plan_kernel(
    int* qo_segment_offsets, int* qo_lens, int* kv_lens,
    uint64_t* packed_work_range, uint64_t* packed_work_info,
    int qo_tile_size, int kv_tile_size, int batch_size, int num_heads,
    int num_buckets, bool causal, int* qo_offsets,
    int num_kv_splits, int* kv_tile_begin_indices, int* kv_tile_end_indices,
    int* kv_split_indices, int adaptive_chunk_size, float* out_max_sm_cost,
    int* num_kv_splits_per_row,
    float* workspace_lse = nullptr, int lse_total_size = 0,
    int pack_factor = 1, int planpar_max_heads = 4) {

  extern __shared__ char __plan_smem[];
  PlanSharedState& state = *reinterpret_cast<PlanSharedState*>(__plan_smem);

  if (num_kv_splits_per_row && qo_segment_offsets) {
    int total_rows = qo_segment_offsets[batch_size];
    for (int i = static_cast<int>(threadIdx.x); i < total_rows; i += static_cast<int>(blockDim.x))
      num_kv_splits_per_row[i] = 1;
    __syncthreads();
  }

  if (workspace_lse) {
    for (int i = static_cast<int>(threadIdx.x); i < lse_total_size; i += static_cast<int>(blockDim.x))
      workspace_lse[i] = -INFINITY;
  }

#ifdef SM_TIMING_ENABLED
  long long __plan_start = clock64();
#endif

  bool adaptive_mode = (num_kv_splits < 0);
  int max_splits = adaptive_mode ? -num_kv_splits : num_kv_splits;

  if (!adaptive_mode) {
    int cost = direct_greedy(state, num_buckets,
        qo_lens, kv_lens, qo_offsets, qo_tile_size, kv_tile_size,
        batch_size, num_heads, causal, num_kv_splits,
        packed_work_range, packed_work_info,
        kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
        num_kv_splits_per_row, qo_segment_offsets, pack_factor, planpar_max_heads);
    if (threadIdx.x == 0 && out_max_sm_cost) {
      out_max_sm_cost[0] = static_cast<float>(cost);
      out_max_sm_cost[1] = static_cast<float>(cost);
    }
  } else {
#ifdef SM_TIMING_ENABLED
    long long _t0 = clock64();
#endif
    // Adaptive split path
    enumerate_unsplit_tiles(state, qo_lens, kv_lens, qo_offsets,
                            qo_tile_size, kv_tile_size, batch_size, num_heads, causal,
                            pack_factor);
    __syncthreads();

#ifdef SM_TIMING_ENABLED
    long long _t1 = clock64();
#endif
    // Greedy nosplit: assign tasks + write nosplit output + return cost
    int nosplit_cost = phase0_nosplit_greedy<true, true>(
        state, num_buckets, num_heads,
        packed_work_range, packed_work_info);

#ifdef SM_TIMING_ENABLED
    long long _t2 = clock64();
#endif
    // Quick check: can split possibly beat nosplit?
    // Best-case split cost = total_cost/M + kSplitExtraOverhead
    // If that already >= nosplit_cost * threshold, skip rebalance entirely
    int min_sm = find_min_cost_sm(state, num_buckets);
    int min_cost = state.sm_cost[min_sm];
    int gap = nosplit_cost - min_cost;

    bool skip_rebalance = (gap <= kSplitExtraOverhead);
    if (!skip_rebalance) {
      // phase1_proportional_fill(state, num_buckets, max_splits);
      if (state.num_unsplit < num_buckets * 0.8)
        phase1_linear_fill(state, num_buckets, max_splits);
      phase2_rebalance_parallel(state, num_buckets, max_splits);
    }

#ifdef SM_TIMING_ENABLED
    long long _t3 = clock64();
#endif
    int split_max_sm = find_max_cost_sm(state, num_buckets);
    int split_cost = state.sm_cost[split_max_sm] + kSplitExtraOverhead;

    bool split = false;
    if (split_cost < nosplit_cost * 0.99) {
      phase3_write_output(state, num_buckets,
                          packed_work_range, packed_work_info,
                          kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
                          num_kv_splits_per_row, qo_segment_offsets, qo_lens, qo_tile_size);
      split = true;
    }
    if (threadIdx.x == 0 && out_max_sm_cost) {
      out_max_sm_cost[0] = split ? 1 : 0;
      out_max_sm_cost[1] = nosplit_cost / float(split_cost);
    }
#ifdef SM_TIMING_ENABLED
    long long _t4 = clock64();
    if (threadIdx.x == 0)
      printf("SM_PLAN_PHASES enumerate=%lld phase0=%lld rebalance=%lld decide+write=%lld n=%d\n",
             _t1-_t0, _t2-_t1, _t3-_t2, _t4-_t3, state.num_unsplit);
#endif
  }

#ifdef SM_TIMING_ENABLED
  if (threadIdx.x == 0) {
    long long __plan_end = clock64();
    printf("SM_PLAN_TIME duration=%lld\n", __plan_end - __plan_start);
    for (int sm = 0; sm < num_buckets; sm++) {
      if (state.sm_task_count[sm] > 0)
        printf("SM_PLAN sm=%d model_cost=%d num_tiles=%d kv_iters=0\n",
               sm, state.sm_cost[sm], static_cast<int>(state.sm_task_count[sm]));
    }
  }
#endif
}

// ============================================================================
// Host wrapper (unchanged signature)
// ============================================================================

cudaError_t plan_kernel_wrapper(int* qo_segment_offsets, int* qo_lens,
                                int* kv_lens, uint64_t* packed_work_range,
                                uint64_t* packed_work_info, int qo_tile_size,
                                int kv_tile_size, int batch_size, int num_heads, int num_buckets,
                                bool causal, int* qo_offsets, bool enable_pdl,
                                cudaStream_t stream,
                                int num_kv_splits = 1,
                                int* kv_tile_begin_indices = nullptr,
                                int* kv_tile_end_indices = nullptr,
                                int* kv_split_indices = nullptr,
                                int adaptive_chunk_size = 0,
                                float* out_max_sm_cost = nullptr,
                                int* num_kv_splits_per_row = nullptr,
                                float* workspace_lse = nullptr,
                                int lse_total_size = 0,
                                int pack_factor = 1) {
  int smem_size = sizeof(PlanSharedState);
  static bool smem_attr_set = false;
  if (!smem_attr_set && smem_size >= (48 << 10)) {
    cudaFuncSetAttribute(plan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    smem_attr_set = true;
  }
  // MiniMax-M3.1 PLAN_PAR (MSA): the warp-argmin greedy for <= 4 heads per batch, bitwise the original picks.
  // Kill: TRTLLM_M31_DISABLE_TP2CTX_PLAN_PAR=1 (the original rank scan for every plan).
  static int const planpar_max_heads = [] {
    char const* v = std::getenv("TRTLLM_M31_DISABLE_TP2CTX_PLAN_PAR");
    bool const off = v != nullptr && v[0] == '1' && v[1] == '\0';
    std::fprintf(stderr, "[fmha_sm100] plan: PLAN_PAR %s (TRTLLM_M31_DISABLE_TP2CTX_PLAN_PAR)\n",
                 off ? "off (original rank scan)" : "on (warp argmin, <= 4 heads)");
    return off ? 0 : 4;
  }();
  plan_kernel<<<1, MAX_SMS, smem_size, stream>>>(
      qo_segment_offsets, qo_lens, kv_lens,
      packed_work_range, packed_work_info,
      qo_tile_size, kv_tile_size, batch_size, num_heads,
      num_buckets, causal, qo_offsets, num_kv_splits,
      kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
      adaptive_chunk_size, out_max_sm_cost, num_kv_splits_per_row,
      workspace_lse, lse_total_size, pack_factor, planpar_max_heads);
  FLASHINFER_CUDA_CALL(cudaGetLastError());
  return cudaSuccess;
}

}  // namespace flashinfer
