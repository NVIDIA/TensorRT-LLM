// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace cg = cooperative_groups;

namespace py = pybind11;

namespace {

constexpr int64_t kTargetDecodeQHeadPerKv = 16;
constexpr int64_t kTargetDecodeHeadDim = 128;
constexpr int64_t kTargetDecodeTileM = 128;
constexpr int64_t kTargetDecodeKvBytes = 1;

template <typename T>
T ceil_div(T x, T y) {
  return (x + y - 1) / y;
}

// ---------------------------------------------------------------------------
// GPU schedule kernel: one CTA computes the full decode schedule in-place.
//
// Layout: single block of `tpb` threads (must be >= batch, power-of-two,
// ≤ 1024).  Each thread b handles one batch slot.
//
// Pipeline:
//   1. Per-thread: read seqused_k[b] → compute kv_pages[b]; write to output.
//   2. Block reduce: max_pages, min_pages, sum_pages.
//   3. Thread 0: decide kv_chunk_size_pages (binary search + variable-kv
//      load-balance heuristic) and split_kv flag.
//   4. Per-thread: compute split_counts[b], inclusive prefix-sum of
//      work-slots and partial-rows for offsets.
//   5. Block reduce: max_split_count.
//   6. Per-thread: write o_indptr, merge_indptr, request_indices,
//      qo_tile_indices, kv_tile_indices, block_valid_mask for batch b's
//      slot range.
//   7. Thread 0: write 5-i32 info_scalars = (split_kv, kv_chunk_size_pages,
//      padded_work_count, max_split_count, partial_rows).
//
// `padded_work_count_pad` is the host-side worst-case pad for the output
// arrays; the kernel writes block_valid_mask=0 for slots past work_count.
// Caller must ensure tpb >= batch and is a power of two.
// ---------------------------------------------------------------------------
// Warp-shuffle helpers (single-warp fast path).
__device__ __forceinline__ int warp_reduce_max(int v) {
  v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
  v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
  v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
  v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
  v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
  return v;
}
__device__ __forceinline__ int warp_reduce_min(int v) {
  v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
  v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
  v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
  v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
  v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
  return v;
}
__device__ __forceinline__ int warp_reduce_sum(int v) {
  v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
  v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
  v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
  v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
  v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
  return v;
}
// Inclusive scan within a warp using shuffle.
__device__ __forceinline__ int warp_inclusive_scan_sum(int v) {
  int n = __shfl_up_sync(0xFFFFFFFFu, v, 1);
  if ((threadIdx.x & 31) >= 1) v += n;
  n = __shfl_up_sync(0xFFFFFFFFu, v, 2);
  if ((threadIdx.x & 31) >= 2) v += n;
  n = __shfl_up_sync(0xFFFFFFFFu, v, 4);
  if ((threadIdx.x & 31) >= 4) v += n;
  n = __shfl_up_sync(0xFFFFFFFFu, v, 8);
  if ((threadIdx.x & 31) >= 8) v += n;
  n = __shfl_up_sync(0xFFFFFFFFu, v, 16);
  if ((threadIdx.x & 31) >= 16) v += n;
  return v;
}

__global__ void build_decode_schedule_gpu_kernel(
    const int32_t* __restrict__ seqused_k,
    int batch,
    int page_size,
    int seqlen_q,
    int num_q_tiles,
    int num_kv_heads,
    int q_tokens_per_group,
    int max_grid_size,
    int fixed_split_size,
    int disable_split_kv,
    int enable_cuda_graph,
    int padded_work_count_pad,
    // outputs:
    int32_t* __restrict__ kv_pages,
    int32_t* __restrict__ split_counts,
    int32_t* __restrict__ request_indices,
    int32_t* __restrict__ qo_tile_indices,
    int32_t* __restrict__ kv_tile_indices,
    int32_t* __restrict__ block_valid_mask,
    int32_t* __restrict__ merge_indptr,
    int32_t* __restrict__ o_indptr,
    int32_t* __restrict__ info_scalars) {
  // Multi-CTA cooperative kernel.  CTA 0 (warp 0) does the small
  // sequential decision phases (reductions, binary-search chunk pick,
  // prefix scan, info_scalars).  All CTAs then collaborate on the
  // scatter phase via grid-stride loop.  grid.sync() between phases.
  cg::grid_group grid = cg::this_grid();
  constexpr int kMaxBatch = 1024;
  constexpr int kMaxWarps = 32;  // tpb<=1024 → at most 32 warps
  __shared__ int s_kv_pages[kMaxBatch];
  __shared__ int s_split_counts[kMaxBatch];
  __shared__ int s_work_slots[kMaxBatch];      // inclusive prefix-sum
  __shared__ int s_partial_slots[kMaxBatch];   // inclusive prefix-sum
  __shared__ int s_chunk_size;
  __shared__ int s_split_kv_flag;
  __shared__ int s_work_count_shared;
  __shared__ int s_warp_max[kMaxWarps];
  __shared__ int s_warp_min[kMaxWarps];
  __shared__ int s_warp_sum[kMaxWarps];
  __shared__ int s_warp_max_split[kMaxWarps];

  const int tid = threadIdx.x;
  const int tpb = blockDim.x;
  const int bid = blockIdx.x;
  const int n_ctas = gridDim.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Single-CTA design (4 warps = 128 threads).  Decision in warp 0 via
  // shuffles; scatter across all warps via grid-stride loop within the
  // CTA.  No grid.sync() needed.
  {
    // Phase 1: per-thread read kv_pages, write to gmem + shmem.
    int kv_pages_b = 0;
    if (tid < batch) {
      int sk = seqused_k[tid];
      kv_pages_b = (sk + page_size - 1) / page_size;
      if (kv_pages_b < 1) kv_pages_b = 1;
      s_kv_pages[tid] = kv_pages_b;
      kv_pages[tid] = kv_pages_b;
    } else if (tid < kMaxBatch) {
      s_kv_pages[tid] = 0;
    }
    __syncthreads();

    // Phase 2-3: chunk-size decision.
    //
    // First, cross-warp reduction of max/min/sum over ALL batches.  Each
    // thread brings its own batch slot's kv_pages (or sentinel for
    // tid >= batch); each warp does a shuffle-based intra-warp reduce
    // and writes per-warp partials to shmem; warp 0 then combines the
    // warp partials with another shuffle reduce.
    //
    // (The old code only reduced lanes 0-31 of warp 0, which under-counted
    //  pages whenever batch > 32 — see correctness bug fix.)
    const int num_warps = tpb >> 5;
    int chunk = 0;
    int split_kv_flag = 0;
    int max_pages = 0;
    int min_pages = 0;
    int sum_pages = 0;
    {
      const int active_my_slot = (tid < batch) ? 1 : 0;
      int v_for_max = active_my_slot ? kv_pages_b : INT_MIN;
      int v_for_min = active_my_slot ? kv_pages_b : INT_MAX;
      int v_for_sum = active_my_slot ? kv_pages_b : 0;
      int warp_max = warp_reduce_max(v_for_max);
      int warp_min = warp_reduce_min(v_for_min);
      int warp_sum = warp_reduce_sum(v_for_sum);
      if (lane == 0) {
        s_warp_max[warp] = warp_max;
        s_warp_min[warp] = warp_min;
        s_warp_sum[warp] = warp_sum;
      }
    }
    __syncthreads();
    if (warp == 0) {
      int lv_max = (lane < num_warps) ? s_warp_max[lane] : INT_MIN;
      int lv_min = (lane < num_warps) ? s_warp_min[lane] : INT_MAX;
      int lv_sum = (lane < num_warps) ? s_warp_sum[lane] : 0;
      max_pages = warp_reduce_max(lv_max);
      min_pages = warp_reduce_min(lv_min);
      sum_pages = warp_reduce_sum(lv_sum);
    }
    if (warp == 0) {
      // Helper: compute work_x = sum_b(ceil(kv_pages_b / chunk_size)) * num_q_tiles
      // across ALL batches using lane-parallel iteration in groups of 32.
      // Each iteration of the outer loop covers 32 batches; warp_reduce_sum
      // gives the partial; accumulate into work_x in lockstep across lanes.
      auto compute_work_x = [&](int chunk_size) -> int {
        int sum = 0;
        for (int b_base = 0; b_base < batch; b_base += 32) {
          int b = b_base + lane;
          int kvp = (b < batch) ? s_kv_pages[b] : 0;
          int c_x = (b < batch) ? ((kvp + chunk_size - 1) / chunk_size) : 0;
          sum += warp_reduce_sum(c_x);
        }
        return sum * num_q_tiles;
      };

      int base_work_count = batch * num_q_tiles;
      int base_cta = base_work_count * num_kv_heads;
      // Quantum constants tuned from the Phase B 1346-case sweep
      // (bench_split_kv_strategy.py).  The schedule chooses chunk_pages
      // and split_kv_flag for the attn launch; these constants reflect
      // the observed crossover where split overhead (combine kernel +
      // O_partial / LSE_partial fp32 writes ~5-10us) exceeds the
      // parallel-attention savings.
      //
      //   kMinUsefulChunkPages = 16   chunks below ~2K tokens are below
      //                               the combine-overhead break-even.
      //                               B2 sweep: 57/150 over-split cases
      //                               all chose chunk ∈ [1, 8] and were
      //                               1.5-2x slower than no-split.
      //
      //   kTinyKvNoSplitPages  = 8    when max kv_pages ≤ 8 (≤ 1K tokens)
      //                               every batch's KV fits in well under
      //                               one combine-overhead window — split
      //                               is pure tax.
      constexpr int kMinUsefulChunkPages = 16;
      constexpr int kTinyKvNoSplitPages  = 8;
      int min_chunk_pages_floor = (128 / page_size);
      if (min_chunk_pages_floor < 1) min_chunk_pages_floor = 1;
      const int min_chunk_pages = max(min_chunk_pages_floor, kMinUsefulChunkPages);
      if (disable_split_kv != 0) {
        chunk = max_pages;
        split_kv_flag = 0;
      } else if (fixed_split_size > 0) {
        chunk = max(fixed_split_size, 1);
        int work_x = compute_work_x(chunk);
        split_kv_flag = (work_x != base_work_count) ? 1 : 0;
      } else if (base_cta >= max_grid_size) {
        chunk = max_pages;
        split_kv_flag = 0;
      } else if (max_pages <= kTinyKvNoSplitPages) {
        // KV per batch is too short for split to pay back the combine
        // overhead.  Skip the binary search entirely.
        chunk = max_pages;
        split_kv_flag = 0;
      } else {
        int low = min(min_chunk_pages, max_pages);
        int high = max_pages;
        while (low < high) {
          int mid = (low + high) >> 1;
          int work_x = compute_work_x(mid);
          if (work_x * num_kv_heads > max_grid_size) low = mid + 1;
          else high = mid;
        }
        chunk = low;
        // Variable-kv load-balance heuristic.  When kv-lengths span a
        // wide range (e.g. one long batch among many short ones) the
        // binary-search above picks a `chunk` that produces just enough
        // work-slots to fill the grid — but the long batch ends up
        // sequential on one CTA while short batches finish fast and
        // leave SMs idle.  Override with a smaller chunk (but still
        // ≥ min_chunk_pages — the 2K-token floor) so the long batch is
        // broken into ~4× more split slots:
        //
        //   trigger: max/avg ≥ 1.5  (i.e. max_pages*2 ≥ avg_pages*3)
        //   chunk  : max(min_chunk_pages, avg_pages / 4)
        //
        // The 1.5 ratio is the imbalance threshold below which the
        // extra split overhead outweighs the speedup; the avg/4 target
        // (≈ avg_pages / 4 splits per average batch) is tuned for SM100
        // with 148 SMs and our 1-CTA/SM attn kernel.  `avg_pages ≥ 4`
        // guards against tiny kv where splits would dominate.
        int avg_pages = (sum_pages + batch - 1) / batch;
        if (max_pages * 2 >= avg_pages * 3 && avg_pages >= 4) {
          int balance_chunk = max(min_chunk_pages, avg_pages >> 2);
          if (max_pages > balance_chunk && balance_chunk < chunk) {
            chunk = balance_chunk;
          }
        }
        // TODO(combine-256): the LDGSTS combine kernel caps max_splits at
        // 256 (combine.py:53; bound by sLSE smem allocation + per-thread
        // register pressure on the LSE reduction).  Auto round-up `chunk`
        // here so that no single batch ever produces > 256 splits, even
        // at kv > 512K.  This loses parallelism for ultra-long context
        // (e.g. at kv=1M the effective chunk floor jumps from 16 to 32),
        // which is acceptable until combine is rewritten with multi-pass
        // tree reduction (Phase E follow-up).  Remove this cap when
        // combine no longer enforces max_splits <= 256.
        constexpr int kCombineMaxSplits = 256;
        int min_chunk_for_combine = (max_pages + kCombineMaxSplits - 1) /
                                    kCombineMaxSplits;
        if (chunk < min_chunk_for_combine) chunk = min_chunk_for_combine;
        int work_x = compute_work_x(chunk);
        split_kv_flag = (enable_cuda_graph != 0 || work_x != base_work_count) ? 1 : 0;
        if (split_kv_flag == 0) chunk = max_pages;
      }
      if (lane == 0) {
        s_chunk_size = chunk;
        s_split_kv_flag = split_kv_flag;
      }
    }
    __syncthreads();

    // Phase 4-5: compute split_counts + parallel inclusive scan.
    int chunks_b = 0;
    int work_slots_b = 0;
    int partial_slots_b = 0;
    if (tid < batch) {
      chunks_b = s_split_kv_flag
                     ? ((s_kv_pages[tid] + s_chunk_size - 1) / s_chunk_size)
                     : 1;
      work_slots_b = chunks_b * num_q_tiles;
      partial_slots_b = chunks_b * num_q_tiles * q_tokens_per_group;
      s_split_counts[tid] = chunks_b;
      split_counts[tid] = chunks_b;
    } else if (tid < kMaxBatch) {
      s_split_counts[tid] = 0;
    }

    // Inclusive scan via warp shuffle for batch ≤ 32.  For batch > 32,
    // fall back to Hillis-Steele in shared memory (rare in production).
    if (batch <= 32) {
      int inc_w = (tid < batch) ? work_slots_b : 0;
      int inc_p = (tid < batch) ? partial_slots_b : 0;
      if (warp == 0) {
        inc_w = warp_inclusive_scan_sum(inc_w);
        inc_p = warp_inclusive_scan_sum(inc_p);
      }
      if (tid < batch) {
        s_work_slots[tid] = inc_w;
        s_partial_slots[tid] = inc_p;
      }
    } else {
      s_work_slots[tid] = (tid < batch) ? work_slots_b : 0;
      s_partial_slots[tid] = (tid < batch) ? partial_slots_b : 0;
      __syncthreads();
      for (int off = 1; off < tpb; off <<= 1) {
        int w_add = (tid >= off) ? s_work_slots[tid - off] : 0;
        int p_add = (tid >= off) ? s_partial_slots[tid - off] : 0;
        __syncthreads();
        s_work_slots[tid] += w_add;
        s_partial_slots[tid] += p_add;
        __syncthreads();
      }
    }
    __syncthreads();

    // Phase 6: max_split_count via cross-warp reduce (correctness fix —
    // old code only reduced lanes 0-31 of warp 0, undercounting for
    // batch > 32).
    int local_max_split = (tid < batch) ? chunks_b : INT_MIN;
    {
      int warp_max = warp_reduce_max(local_max_split);
      if (lane == 0) s_warp_max_split[warp] = warp_max;
    }
    __syncthreads();
    int max_split_count_local = 0;
    if (warp == 0) {
      int v = (lane < num_warps) ? s_warp_max_split[lane] : INT_MIN;
      max_split_count_local = warp_reduce_max(v);
      if (max_split_count_local < 1) max_split_count_local = 1;
    }

    // Phase 7-8: write o_indptr and merge_indptr in parallel.
    // o_indptr[0] = 0, o_indptr[b+1] = inclusive_partial[b]
    // merge_indptr[0] = 0
    // merge_indptr[b * seqlen_q + q + 1] =
    //     exclusive_prefix_chunks[b] * seqlen_q + (q + 1) * chunks[b]
    // exclusive_prefix_chunks[b] = (work_slots inclusive scan / num_q_tiles) - chunks[b]
    // ... actually simpler: chunks-prefix == work_slots-prefix / num_q_tiles, since
    // work_slots_b = chunks_b * num_q_tiles.  When num_q_tiles==1, they're equal.
    if (tid == 0) {
      o_indptr[0] = 0;
      merge_indptr[0] = 0;
    }
    if (tid < batch) {
      o_indptr[tid + 1] = s_partial_slots[tid];
      // Compute exclusive prefix sum of chunks for THIS batch tid.
      int incl_chunks = s_work_slots[tid] / max(num_q_tiles, 1);
      int excl_chunks = incl_chunks - chunks_b;
      // Parallel per-q write within this batch slot.
      for (int q = 0; q < seqlen_q; ++q) {
        merge_indptr[tid * seqlen_q + q + 1] =
            excl_chunks * seqlen_q + (q + 1) * chunks_b;
      }
    }

    // Phase 11: write info_scalars + s_work_count_shared so the scatter
    // phase across other warps can read it.  Thread (batch-1) holds the
    // inclusive scan total; broadcast via shared mem.
    if (tid == batch - 1) {
      s_work_count_shared = s_work_slots[batch - 1];
    } else if (tid == 0 && batch == 0) {
      s_work_count_shared = 0;
    }
    __syncthreads();
    int work_count = s_work_count_shared;
    int partial_rows = (batch > 0) ? s_partial_slots[batch - 1] : 0;
    if (warp == 0 && lane == 0) {
      int padded_wc = (enable_cuda_graph != 0 && s_split_kv_flag != 0)
                          ? max(work_count, max(1, max_grid_size / num_kv_heads))
                          : work_count;
      info_scalars[0] = s_split_kv_flag;
      info_scalars[1] = s_chunk_size;
      info_scalars[2] = padded_wc;
      info_scalars[3] = max_split_count_local;
      info_scalars[4] = partial_rows;
    }
  }

  // Sync so warps 1-3 see the shared-memory state written by warp 0 in
  // Part A (s_split_counts, s_work_slots, s_work_count_shared).
  __syncthreads();

  // Scatter (all 128 threads, intra-CTA grid-stride loop).
  // Note: s_split_counts and s_work_slots are valid in shared mem from
  // Part A.  Use them directly (no global reload).
  int work_count_total = s_work_count_shared;
  for (int idx = tid; idx < work_count_total; idx += tpb) {
    // Inverse map idx -> (b, q_tile, kv_tile) via linear search.
    int b_found = 0;
    int prev_prefix = 0;
    for (int j = 0; j < batch; ++j) {
      int p = s_split_counts[j] * num_q_tiles;
      if (idx < prev_prefix + p) { b_found = j; break; }
      prev_prefix += p;
    }
    int within = idx - prev_prefix;
    int chunks_at_b = s_split_counts[b_found];
    int q_tile = (chunks_at_b > 0) ? (within / chunks_at_b) : 0;
    int kv_tile = within - q_tile * chunks_at_b;
    request_indices[idx] = b_found;
    qo_tile_indices[idx] = q_tile;
    kv_tile_indices[idx] = kv_tile;
    block_valid_mask[idx] = 1;
  }
}

int64_t determine_cta_tile_q(int64_t packed_q_len, int64_t head_dim, int compute_major) {
  if (packed_q_len > 64 && head_dim < 256) {
    return 128;
  }
  if (compute_major >= 8) {
    return packed_q_len > 16 ? 64 : 16;
  }
  return 64;
}

// Decode attn kernel runs at 1 CTA/SM (UTCMMA + warp specialization
// holds ~240 reg/thread × 512 threads, saturating the register file).
// max_grid_size = num_sms is therefore the exact attainable grid; we
// don't probe occupancy because the CUTE DSL kernel's function pointer
// isn't reachable from C++ (would have required a proxy kernel, whose
// register pressure differs from the real one and gave misleading 8-16
// blocks/SM estimates that triggered over-splitting at small kv).
std::tuple<int64_t, int64_t, int64_t> estimate_decode_grid_size(
    int64_t /*num_qo_heads*/,
    int64_t /*num_kv_heads*/,
    int64_t /*head_dim*/,
    int64_t max_grid_size_override) {
  int dev_id = 0;
  AT_CUDA_CHECK(cudaGetDevice(&dev_id));
  int num_sms = 0;
  AT_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  if (max_grid_size_override > 0) {
    int64_t active_blocks = std::max<int64_t>(
        1, ceil_div(max_grid_size_override, std::max<int64_t>(num_sms, 1)));
    return {max_grid_size_override, active_blocks, num_sms};
  }
  // Hardcoded: 1 CTA/SM for the decode attn kernel.
  return {static_cast<int64_t>(num_sms), int64_t{1}, num_sms};
}

int64_t split_work_x(
    const std::vector<int32_t>& kv_pages,
    int64_t chunk_pages,
    int64_t num_q_tiles,
    bool split_kv) {
  int64_t work = 0;
  for (int32_t pages : kv_pages) {
    const int64_t chunks = split_kv ? ceil_div<int64_t>(std::max<int64_t>(pages, 1), chunk_pages) : 1;
    work += chunks * num_q_tiles;
  }
  return work;
}

}  // namespace

// ============================================================================
// GPU-only schedule launcher.  All schedule arrays are computed in a single
// CUDA kernel from seqused_k on GPU — no D2H copy of seqused_k, no H2D copy
// of index arrays.  Only the info_scalars summary tensor is read back to
// host (single small D2H sync) so the wrapper can size O_partial, launch
// the right kernel grid, and choose the split/non-split compile path.
// ============================================================================
py::dict build_decode_schedule(
    torch::Tensor seqused_k,
    int64_t page_size,
    int64_t seqlen_q,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t max_seqlen_k,
    bool enable_cuda_graph,
    int64_t max_grid_size_override,
    int64_t fixed_split_size,
    bool disable_split_kv) {
  TORCH_CHECK(seqused_k.is_cuda(), "seqused_k must be a CUDA tensor");
  TORCH_CHECK(seqused_k.scalar_type() == at::kInt, "seqused_k must be int32");
  TORCH_CHECK(seqused_k.dim() == 1, "seqused_k must have shape [B]");
  TORCH_CHECK(seqused_k.is_contiguous(), "seqused_k must be contiguous");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(seqlen_q > 0, "seqlen_q must be positive");
  TORCH_CHECK(num_qo_heads > 0 && num_kv_heads > 0, "head counts must be positive");
  TORCH_CHECK(num_qo_heads % num_kv_heads == 0,
              "num_qo_heads must be divisible by num_kv_heads");
  TORCH_CHECK(num_qo_heads / num_kv_heads == kTargetDecodeQHeadPerKv,
              "decode schedule currently supports only qhead_per_kv=16");
  TORCH_CHECK(head_dim == kTargetDecodeHeadDim,
              "decode schedule currently supports only head_dim=128");
  TORCH_CHECK(max_seqlen_k > 0, "max_seqlen_k must be positive");

  const int64_t batch = seqused_k.size(0);
  TORCH_CHECK(batch > 0, "seqused_k must contain at least one batch item");
  // Hard cap: single-CTA schedule kernel stores per-batch state in shared
  // memory (s_kv_pages[1024], s_split_counts[1024], etc.) and uses tpb
  // ≤ 1024 with one-thread-per-batch.  For batch > 1024 we need a
  // multi-CTA cooperative redesign (gmem state, CUB DeviceScan, grid.sync
  // between phases).  Tracked separately — for now, reject early with a
  // clear message so production code doesn't silently truncate batches.
  TORCH_CHECK(batch <= 1024,
              "build_decode_schedule currently supports batch <= 1024 (got ",
              batch,
              ").  Larger batches require the multi-CTA scheduler "
              "(unimplemented in this revision).");

  // Host-side derived constants (no D2H needed for these).
  int dev_id = 0;
  AT_CUDA_CHECK(cudaGetDevice(&dev_id));
  int compute_major = 0;
  AT_CUDA_CHECK(cudaDeviceGetAttribute(&compute_major,
                                       cudaDevAttrComputeCapabilityMajor,
                                       dev_id));
  const int64_t qhead_per_kv = num_qo_heads / num_kv_heads;
  const int64_t packed_q_len = seqlen_q * qhead_per_kv;
  const int64_t cta_tile_q = determine_cta_tile_q(packed_q_len, head_dim, compute_major);
  const int64_t num_q_tiles = ceil_div<int64_t>(packed_q_len, cta_tile_q);
  TORCH_CHECK(kTargetDecodeTileM % qhead_per_kv == 0,
              "decode tile_m must be divisible by qhead_per_kv");
  const int64_t q_tokens_per_group = kTargetDecodeTileM / qhead_per_kv;
  const auto [max_grid_size, active_blocks_per_sm, num_sms] =
      estimate_decode_grid_size(num_qo_heads, num_kv_heads, head_dim,
                                max_grid_size_override);

  // Worst-case padding for the work-tile arrays.  When the heuristic picks
  // the smallest possible chunk (min_chunk_pages = max(128/page_size, 1)),
  // a single batch can produce up to max_pages_global / min_chunk_pages
  // chunks; across batches this is bounded by sum-of-pages which is at most
  // batch × max_pages_global.
  const int64_t max_pages_global = ceil_div<int64_t>(max_seqlen_k, page_size);
  const int64_t pad_work = batch * num_q_tiles * std::max<int64_t>(max_pages_global, 1);
  const int64_t pad_partial = pad_work * q_tokens_per_group;

  const auto device = seqused_k.device();
  auto i32_options = torch::TensorOptions().dtype(torch::kInt32).device(device);

  // Allocate all output arrays on GPU.
  auto kv_pages_tensor = torch::empty({batch}, i32_options);
  auto split_counts_tensor = torch::empty({batch}, i32_options);
  auto request_tensor = torch::empty({pad_work}, i32_options);
  auto qo_tile_tensor = torch::empty({pad_work}, i32_options);
  auto kv_tile_tensor = torch::empty({pad_work}, i32_options);
  auto mask_tensor = torch::empty({pad_work}, i32_options);
  auto merge_indptr_tensor = torch::empty({batch * seqlen_q + 1}, i32_options);
  auto o_indptr_tensor = torch::empty({batch + 1}, i32_options);
  auto info_tensor = torch::empty({5}, i32_options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(seqused_k.get_device());

  // tpb: threads per CTA.  Use 128 (4 warps) so we have plenty of warps
  // for the per-CTA setup phase and for scatter work.  CTA 0's warp 0
  // does the decision phase; all warps in all CTAs collaborate on the
  // scatter phase via grid-stride loop.
  int tpb = 128;
  if (batch > 128) {
    // Cap to next power of two so single-CTA reductions still work.
    tpb = 1;
    while (tpb < static_cast<int>(batch)) tpb <<= 1;
    if (tpb > 1024) tpb = 1024;
  }

  // Single CTA — all decisions and scatter happen on one CTA with 4 warps.
  // No grid.sync(), no cooperative-launch overhead.
  build_decode_schedule_gpu_kernel<<<1, tpb, 0, stream>>>(
      seqused_k.data_ptr<int32_t>(),
      static_cast<int>(batch),
      static_cast<int>(page_size),
      static_cast<int>(seqlen_q),
      static_cast<int>(num_q_tiles),
      static_cast<int>(num_kv_heads),
      static_cast<int>(q_tokens_per_group),
      static_cast<int>(max_grid_size),
      static_cast<int>(fixed_split_size),
      static_cast<int>(disable_split_kv),
      static_cast<int>(enable_cuda_graph),
      static_cast<int>(pad_work),
      kv_pages_tensor.data_ptr<int32_t>(),
      split_counts_tensor.data_ptr<int32_t>(),
      request_tensor.data_ptr<int32_t>(),
      qo_tile_tensor.data_ptr<int32_t>(),
      kv_tile_tensor.data_ptr<int32_t>(),
      mask_tensor.data_ptr<int32_t>(),
      merge_indptr_tensor.data_ptr<int32_t>(),
      o_indptr_tensor.data_ptr<int32_t>(),
      info_tensor.data_ptr<int32_t>());
  AT_CUDA_CHECK(cudaGetLastError());

  // Single D2H sync for the 5 summary scalars.  Payload = 20 bytes.
  auto info_cpu = info_tensor.cpu();
  const int32_t* info_host = info_cpu.data_ptr<int32_t>();
  const int32_t split_kv_flag = info_host[0];
  const int32_t kv_chunk_size_pages = info_host[1];
  const int32_t padded_work_count = info_host[2];
  const int32_t max_split_count = info_host[3];
  const int32_t partial_rows = info_host[4];

  const int64_t base_work_count = batch * num_q_tiles;
  const int64_t base_cta = base_work_count * num_kv_heads;
  const int64_t work_count =
      (split_kv_flag != 0) ? static_cast<int64_t>(padded_work_count) : base_work_count;

  py::dict result;
  result["split_kv"] = (split_kv_flag != 0);
  result["cta_tile_q"] = cta_tile_q;
  result["num_q_tiles"] = num_q_tiles;
  result["kv_chunk_size_pages"] = static_cast<int64_t>(kv_chunk_size_pages);
  result["kv_chunk_size_tokens"] = static_cast<int64_t>(kv_chunk_size_pages) * page_size;
  result["work_count"] = work_count;
  result["padded_work_count"] = static_cast<int64_t>(padded_work_count);
  result["partial_rows"] = static_cast<int64_t>(partial_rows);
  result["max_split_count"] = static_cast<int64_t>(max_split_count);
  result["max_grid_size"] = max_grid_size;
  result["active_blocks_per_sm"] = active_blocks_per_sm;
  result["num_sms"] = num_sms;
  result["base_cta"] = base_cta;
  result["request_indices"] = request_tensor;
  result["qo_tile_indices"] = qo_tile_tensor;
  result["kv_tile_indices"] = kv_tile_tensor;
  result["block_valid_mask"] = mask_tensor;
  result["split_counts"] = split_counts_tensor;
  result["kv_pages"] = kv_pages_tensor;
  result["merge_indptr"] = merge_indptr_tensor;
  result["o_indptr"] = o_indptr_tensor;
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_decode_schedule", &build_decode_schedule,
        "Build paged decode split-KV schedule on GPU");
}
