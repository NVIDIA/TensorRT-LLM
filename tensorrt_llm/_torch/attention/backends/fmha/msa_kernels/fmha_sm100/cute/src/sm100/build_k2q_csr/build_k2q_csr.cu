// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

// CUDA C++ q2k -> k2q CSR builder.
//
// Five-stage pipeline. q-ascending order within each CSR row is preserved
// by partitioning q across (CTA, warp_in_CTA) units; each unit owns a
// contiguous q-sub-range and reserves a contiguous slot range per row via
// a precomputed exclusive prefix scan.
//
//   M:  build_row_map      -- round-robin packing of rows across batches
//   H:  histogram + tile_counts
//   PR: row prefix         -- single block per head, row_counts -> row_ptr
//   PT: tile prefix        -- multi-block, scan tile_counts along (c, w) axis
//   S:  scatter (sorted)   -- per-warp slot range, q-sequential within warp
//
// Per-warp partitioning: each CTA has kWarps warps; warp w of CTA c owns
// q-range [c*q_per_cta + w*q_per_warp, c*q_per_cta + (w+1)*q_per_warp).
// tile_counts is shaped [G * kWarps, H, total_rows]; the "row" dimension
// of the prefix scan is the flattened (c * kWarps + w) index, scanned in
// lexicographic order so that warp-local slot ranges concatenate to the
// global q-sorted output.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT(x)

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ void advance_batch_only(
    int const* __restrict__ cu_q, int B, int q_abs, int& bi)
{
    while (bi < B && cu_q[bi + 1] <= q_abs) ++bi;
}

// Atomic increment of a 16-bit half within a 32-bit SMEM word; returns the
// OLD 16-bit value (slot). Per-warp count must stay < 32768 so the low
// half does not carry into the high half.
//   base_int32 : int32 pointer; element i holds rows 2*i (low) and 2*i+1 (high).
__device__ __forceinline__ int atomic_inc_int16_packed(
    int* base_int32, int row)
{
    int idx = row >> 1;
    int shift = (row & 1) << 4;  // 0 or 16
    int delta = 1 << shift;
    int old = atomicAdd(&base_int32[idx], delta);
    return (old >> shift) & 0xFFFF;
}

// Read 16-bit half from packed int32 storage.
__device__ __forceinline__ int read_int16_packed(int const* base_int32, int row) {
    int v = base_int32[row >> 1];
    int shift = (row & 1) << 4;
    return (v >> shift) & 0xFFFF;
}

// ---------------------------------------------------------------------------
// M: round-robin row map.
// ---------------------------------------------------------------------------
template <int kBlockK>
__global__ void k2q_build_row_map_kernel(
    int const* __restrict__ cu_k,
    int* __restrict__ row_map,
    int* __restrict__ row_coords,
    int B,
    int max_kv_blocks)
{
    int level = blockIdx.x;
    if (level >= max_kv_blocks) return;
    if (threadIdx.x != 0) return;
    int rows_before = 0;
    for (int b = 0; b < B; ++b) {
        int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
        rows_before += (rb < level ? rb : level);
    }
    int active_before = 0;
    for (int b = 0; b < B; ++b) {
        int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
        if (rb > level) {
            int row_linear = rows_before + active_before;
            row_map[(size_t)b * max_kv_blocks + level] = row_linear;
            if (row_coords != nullptr) {
                row_coords[(size_t)row_linear * 2] = b;
                row_coords[(size_t)row_linear * 2 + 1] = level;
            }
            ++active_before;
        } else {
            row_map[(size_t)b * max_kv_blocks + level] = -1;
        }
    }
}

// ---------------------------------------------------------------------------
// H: per-warp histogram + tile_counts.
// kWarps warps per CTA, each owns q-sub-range = q_per_cta / kWarps.
// SMEM hist[kWarps, total_rows] int32 (stored as packed int16 cursor:
// 2 entries per int32 word). Each warp counts to its own row.
// At end-of-CTA, write tile_counts[c*kWarps + w, h, r] = smem_hist[w, r]
// and atomicAdd(row_counts[h, r], sum over w of smem_hist[w, r]).
// ---------------------------------------------------------------------------
template <int kTopK, int kBlockK, int kWarps>
__global__ void k2q_hist_kernel(
    int const* __restrict__ q2k,
    int const* __restrict__ cu_q,
    int const* __restrict__ row_map,
    int* __restrict__ row_counts,
    int* __restrict__ tile_counts,
    int H, int B, int S_Q,
    int total_rows, int max_kv_blocks,
    int q_per_cta, int q_per_warp)
{
    constexpr int kThreads = kWarps * kWarpSize;
    extern __shared__ int smem_hist_int[];
    int* smem_hist = smem_hist_int;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int c = blockIdx.x;
    int q_start_cta = c * q_per_cta;
    int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
    int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
    int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

    constexpr int kInt4PerToken = kTopK / 4;
    int packed_per_warp = (total_rows + 1) >> 1;
    int* my_hist = smem_hist + warp_id * packed_per_warp;

    for (int h = 0; h < H; ++h) {
        for (int i = lane; i < packed_per_warp; i += kWarpSize) my_hist[i] = 0;
        __syncthreads();

        if (q_start_warp < q_end_warp) {
            int bi = 0;
            int qi = q_start_warp + lane;
            advance_batch_only(cu_q, B, qi, bi);

            int4 const* head_topk4 =
                reinterpret_cast<int4 const*>(q2k + (size_t)h * S_Q * kTopK);

            for (; qi < q_end_warp; qi += kWarpSize) {
                advance_batch_only(cu_q, B, qi, bi);
                int const* my_row_map = row_map + (size_t)bi * max_kv_blocks;

                int4 buf[kInt4PerToken];
                #pragma unroll
                for (int v = 0; v < kInt4PerToken; ++v) {
                    buf[v] = head_topk4[(size_t)qi * kInt4PerToken + v];
                }
                #pragma unroll
                for (int t = 0; t < kTopK; ++t) {
                    int kvb_local = reinterpret_cast<int const*>(buf)[t];
                    if (kvb_local >= 0 && kvb_local < max_kv_blocks) {
                        int row = my_row_map[kvb_local];
                        if (row >= 0 && row < total_rows) {
                            atomic_inc_int16_packed(my_hist, row);
                        }
                    }
                }
            }
        }
        __syncthreads();

        int* head_row_counts = row_counts + (size_t)h * total_rows;
        // Each warp writes its own slice of tile_counts (full int32) by
        // unpacking int16 entries from SMEM.
        int* my_tile = tile_counts +
            ((size_t)(c * kWarps + warp_id) * H + h) * total_rows;
        for (int i = lane; i < total_rows; i += kWarpSize) {
            my_tile[i] = read_int16_packed(my_hist, i);
        }
        __syncthreads();

        // Sum across warps (int32 accumulator), atomicAdd to row_counts.
        for (int i = tid; i < total_rows; i += kThreads) {
            int sum = 0;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                sum += read_int16_packed(smem_hist + w * packed_per_warp, i);
            }
            if (sum > 0) atomicAdd(&head_row_counts[i], sum);
        }
        if (h + 1 < H) __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// PR: row prefix. One block per head.
// ---------------------------------------------------------------------------
template <int kThreads>
__global__ void k2q_row_prefix_agg_kernel(
    int const* __restrict__ row_counts,
    int* __restrict__ row_ptr,
    int const* __restrict__ row_coords,
    int* __restrict__ scheduler_metadata,
    int* __restrict__ work_count,
    int total_rows,
    int target_q_per_cta,
    int work_capacity)
{
    // Same row_ptr as k2q_row_prefix_kernel. Work items are claimed with one atomic per warp per 32-row group
    // (the order of scheduler_metadata entries was already atomic-order, i.e. unspecified).
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    __shared__ int scan_buf[kThreads];
    int const* head_counts = row_counts + (size_t)h * total_rows;
    int* head_rowptr = row_ptr + (size_t)h * (total_rows + 1);
    int chunk = (total_rows + kThreads - 1) / kThreads;
    int lo = tid * chunk;
    int hi = min(lo + chunk, total_rows);
    int local_sum = 0;
    for (int i = lo; i < hi; ++i) local_sum += head_counts[i];
    scan_buf[tid] = local_sum;
    __syncthreads();
    for (int off = 1; off < kThreads; off <<= 1) {
        int add = (tid >= off) ? scan_buf[tid - off] : 0;
        __syncthreads();
        scan_buf[tid] += add;
        __syncthreads();
    }
    int running = scan_buf[tid] - local_sum;
    for (int i = lo; i < hi; ++i) {
        running += head_counts[i];
        head_rowptr[i + 1] = running;
    }
    if (scheduler_metadata == nullptr || work_count == nullptr) return;
    // Work items: rows strided by thread so each warp instruction covers 32 consecutive rows.
    for (int i0 = 0; i0 < total_rows; i0 += kThreads) {
        int i = i0 + tid;
        int row_count = i < total_rows ? head_counts[i] : 0;
        int num_chunks = row_count > 0 ? (row_count + target_q_per_cta - 1) / target_q_per_cta : 0;
        int x = num_chunks;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int nbr = __shfl_up_sync(0xFFFFFFFFu, x, off);
            if (lane >= off) x += nbr;
        }
        int warp_total = __shfl_sync(0xFFFFFFFFu, x, 31);
        int base = 0;
        if (lane == 31 && warp_total > 0) base = atomicAdd(work_count, warp_total);
        base = __shfl_sync(0xFFFFFFFFu, base, 31) + x - num_chunks;
        if (num_chunks > 0) {
            int batch_idx = row_coords[(size_t)i * 2];
            int kv_block_idx = row_coords[(size_t)i * 2 + 1];
            for (int c = 0; c < num_chunks; ++c) {
                int work_idx = base + c;
                if (work_idx < work_capacity) {
                    int q_begin = c * target_q_per_cta;
                    int* meta = scheduler_metadata + (size_t)work_idx * 6;
                    meta[0] = h;
                    meta[1] = i;
                    meta[2] = q_begin;
                    meta[3] = min(target_q_per_cta, row_count - q_begin);
                    meta[4] = batch_idx;
                    meta[5] = kv_block_idx;
                }
            }
        }
    }
}

template <int kThreads>
__global__ void k2q_row_prefix_kernel(
    int const* __restrict__ row_counts,
    int* __restrict__ row_ptr,
    int const* __restrict__ row_coords,
    int* __restrict__ scheduler_metadata,
    int* __restrict__ work_count,
    int total_rows,
    int target_q_per_cta,
    int work_capacity)
{
    int h = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ int scan_buf[kThreads];

    int const* head_counts = row_counts + (size_t)h * total_rows;
    int* head_rowptr = row_ptr + (size_t)h * (total_rows + 1);
    int chunk = (total_rows + kThreads - 1) / kThreads;
    int lo = tid * chunk;
    int hi = min(lo + chunk, total_rows);

    int local_sum = 0;
    for (int i = lo; i < hi; ++i) local_sum += head_counts[i];
    scan_buf[tid] = local_sum;
    __syncthreads();

    for (int off = 1; off < kThreads; off <<= 1) {
        int add = (tid >= off) ? scan_buf[tid - off] : 0;
        __syncthreads();
        scan_buf[tid] += add;
        __syncthreads();
    }
    int running = scan_buf[tid] - local_sum;
    for (int i = lo; i < hi; ++i) {
        int row_count = head_counts[i];
        running += row_count;
        head_rowptr[i + 1] = running;
        if (scheduler_metadata != nullptr && work_count != nullptr && row_count > 0) {
            int num_chunks = (row_count + target_q_per_cta - 1) / target_q_per_cta;
            int base = atomicAdd(work_count, num_chunks);
            int batch_idx = row_coords[(size_t)i * 2];
            int kv_block_idx = row_coords[(size_t)i * 2 + 1];
            for (int c = 0; c < num_chunks; ++c) {
                int work_idx = base + c;
                if (work_idx < work_capacity) {
                    int q_begin = c * target_q_per_cta;
                    int q_count = min(target_q_per_cta, row_count - q_begin);
                    int* meta = scheduler_metadata + (size_t)work_idx * 6;
                    meta[0] = h;
                    meta[1] = i;
                    meta[2] = q_begin;
                    meta[3] = q_count;
                    meta[4] = batch_idx;
                    meta[5] = kv_block_idx;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MiniMax-M3 (MSA) edge-rank build (kill: FMHA_K2Q_EDGE_RANK=0): hist in the scatter's lane layout (lanes =
// (q pair, slot)); each edge's per-warp rank comes from __match_any_sync + popc (lane order = q order), not from
// the order of same-address atomics, which CUDA leaves unspecified (the stock scatter relies on it). Each edge keeps
// its per-warp rank in registers (q_per_warp <= 2 * kIters); after the CTA sync the per-warp counts become an exclusive
// cross-warp prefix in SMEM (warp order = q order), so rank + prefix = the edge's q-ordered rank inside the CTA
// tile. Stored per edge (int16); the scatter is then a flat per-edge kernel with no SMEM and no atomics.
// ---------------------------------------------------------------------------
template <int kTopK, int kWarps, int kIters>
__global__ void k2q_hist_rank_kernel(
    int const* __restrict__ q2k, int const* __restrict__ cu_q, int const* __restrict__ row_map,
    int* __restrict__ row_counts, int* __restrict__ tile_counts, short* __restrict__ edge_rank,
    int H, int B, int S_Q, int total_rows, int max_kv_blocks, int q_per_cta, int q_per_warp)
{
    constexpr int kQPerIter = kWarpSize / kTopK;
    constexpr int kThreads = kWarps * kWarpSize;
    extern __shared__ int smem_hist_int[];
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int c = blockIdx.x;
    int q_start_cta = c * q_per_cta;
    int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
    int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
    int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);
    int q_in_iter = lane / kTopK;
    int slot_in_q = lane % kTopK;
    int packed_per_warp = (total_rows + 1) >> 1;
    int* my_hist = smem_hist_int + warp_id * packed_per_warp;

    for (int h = 0; h < H; ++h) {
        for (int i = lane; i < packed_per_warp; i += kWarpSize) my_hist[i] = 0;
        __syncwarp();
        int const* head_q2k = q2k + (size_t)h * S_Q * kTopK;
        int row[kIters];
        int rank[kIters];
        int bi = 0;
        #pragma unroll
        for (int it = 0; it < kIters; ++it) {
            int qi = q_start_warp + it * kQPerIter + q_in_iter;
            row[it] = -1;
            rank[it] = 0;
            if (qi < q_end_warp) {
                advance_batch_only(cu_q, B, qi, bi);
                int kvb = head_q2k[(size_t)qi * kTopK + slot_in_q];
                if (kvb >= 0 && kvb < max_kv_blocks) {
                    int r = row_map[(size_t)bi * max_kv_blocks + kvb];
                    if (r >= 0 && r < total_rows) row[it] = r;
                }
            }
        }
        // Deterministic per-warp rank (no reliance on the order of same-address atomics): the lanes holding the
        // same row in this step are matched; rank = warp base for that row + earlier matching lanes (lane order =
        // q order). The group leader adds the group size with one atomic whose return is the base; __syncwarp
        // orders the warp's successive steps.
        unsigned int const lt_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
        #pragma unroll
        for (int it = 0; it < kIters; ++it) {
            int r = row[it];
            unsigned int peers = __match_any_sync(0xFFFFFFFFu, r >= 0 ? r : -1 - lane);
            int leader = __ffs(peers) - 1;
            int base = 0;
            if (r >= 0 && lane == leader) {
                int shift = (r & 1) << 4;
                base = (atomicAdd(&my_hist[r >> 1], __popc(peers) << shift) >> shift) & 0xFFFF;
            }
            base = __shfl_sync(0xFFFFFFFFu, base, leader);
            rank[it] = base + __popc(peers & lt_mask);
            __syncwarp();
        }
        __syncthreads();
        // Exclusive cross-warp prefix in place (packed int16 words add without carry: per-CTA row count
        // <= q_per_cta < 32768), CTA totals -> tile_counts [G, H, rows] + row_counts.
        int* cta_tile = tile_counts + ((size_t)c * H + h) * total_rows;
        int* head_row_counts = row_counts + (size_t)h * total_rows;
        for (int j = tid; j < packed_per_warp; j += kThreads) {
            int running = 0;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                int v = smem_hist_int[w * packed_per_warp + j];
                smem_hist_int[w * packed_per_warp + j] = running;
                running += v;
            }
            int r0 = 2 * j, s0 = running & 0xFFFF, s1 = (running >> 16) & 0xFFFF;
            cta_tile[r0] = s0;
            if (s0 > 0) atomicAdd(&head_row_counts[r0], s0);
            if (r0 + 1 < total_rows) {
                cta_tile[r0 + 1] = s1;
                if (s1 > 0) atomicAdd(&head_row_counts[r0 + 1], s1);
            }
        }
        __syncthreads();
        short* head_rank = edge_rank + (size_t)h * S_Q * kTopK;
        #pragma unroll
        for (int it = 0; it < kIters; ++it) {
            int qi = q_start_warp + it * kQPerIter + q_in_iter;
            if (row[it] >= 0)
                head_rank[(size_t)qi * kTopK + slot_in_q] = (short)(rank[it] + read_int16_packed(my_hist, row[it]));
        }
        if (h + 1 < H) __syncthreads();
    }
}

template <int kTopK>
__global__ void k2q_scatter_rank_kernel(
    int const* __restrict__ q2k, int const* __restrict__ cu_q, int const* __restrict__ row_map,
    int const* __restrict__ abs_base, short const* __restrict__ edge_rank,
    int* __restrict__ q_idx, int* __restrict__ qsplit_idx, int* __restrict__ split_counts,
    int H, int B, int S_Q, int total_rows, int max_kv_blocks, int q_per_cta)
{
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // e = (h * S_Q + q) * kTopK + slot
    long long n = (long long)H * S_Q * kTopK;
    int lane = threadIdx.x & 31;
    bool in = e < n;
    int slot = (int)(e % kTopK);
    long long hq = e / kTopK;
    int q = in ? (int)(hq % S_Q) : 0;
    int h = in ? (int)(hq / S_Q) : 0;
    int row = -1, qloc = 0;
    if (in) {
        int lo = 0, hi = B - 1;
        while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (cu_q[mid] <= q) lo = mid; else hi = mid - 1; }
        qloc = q - cu_q[lo];
        int kvb = q2k[e];
        if (kvb >= 0 && kvb < max_kv_blocks) {
            int r = row_map[(size_t)lo * max_kv_blocks + kvb];
            if (r >= 0 && r < total_rows) row = r;
        }
    }
    bool valid = row >= 0;
    unsigned int vmask = __ballot_sync(0xFFFFFFFFu, valid);
    unsigned int gmask = (kTopK >= 32) ? 0xFFFFFFFFu : (((1u << kTopK) - 1u) << ((lane / kTopK) * kTopK));
    unsigned int lower = lane == 0 ? 0u : ((1u << lane) - 1u);
    int split_slot = __popc(vmask & gmask & lower);
    if (!in) return;
    if (split_counts != nullptr && slot == 0) split_counts[(size_t)q * H + h] = __popc(vmask & gmask);
    if (!valid) return;
    int c = q / q_per_cta;
    int pos = abs_base[((size_t)c * H + h) * total_rows + row] + (int)edge_rank[e];
    size_t o = (size_t)h * S_Q * kTopK + pos;
    q_idx[o] = qloc;
    if (qsplit_idx != nullptr) qsplit_idx[o] = qloc | ((split_slot & 0xFF) << 24);
}

// MiniMax-M3 (MSA): the work-item emission of k2q_row_prefix_agg_kernel on a multi-CTA grid (rows x heads)
// instead of H CTAs. Same per-warp claims; only the (already atomic) order of the items changes.
__global__ void k2q_sched_emit_kernel(
    int const* __restrict__ row_counts,
    int const* __restrict__ row_coords,
    int* __restrict__ scheduler_metadata,
    int* __restrict__ work_count,
    int total_rows,
    int target_q_per_cta,
    int work_capacity)
{
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int row_count = i < total_rows ? row_counts[(size_t)h * total_rows + i] : 0;
    int num_chunks = row_count > 0 ? (row_count + target_q_per_cta - 1) / target_q_per_cta : 0;
    int x = num_chunks;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int nbr = __shfl_up_sync(0xFFFFFFFFu, x, off);
        if (lane >= off) x += nbr;
    }
    int warp_total = __shfl_sync(0xFFFFFFFFu, x, 31);
    int base = 0;
    if (lane == 31 && warp_total > 0) base = atomicAdd(work_count, warp_total);
    base = __shfl_sync(0xFFFFFFFFu, base, 31) + x - num_chunks;
    if (num_chunks == 0) return;
    int batch_idx = row_coords[(size_t)i * 2];
    int kv_block_idx = row_coords[(size_t)i * 2 + 1];
    for (int c = 0; c < num_chunks; ++c) {
        int work_idx = base + c;
        if (work_idx < work_capacity) {
            int q_begin = c * target_q_per_cta;
            int* meta = scheduler_metadata + (size_t)work_idx * 6;
            meta[0] = h;
            meta[1] = i;
            meta[2] = q_begin;
            meta[3] = min(target_q_per_cta, row_count - q_begin);
            meta[4] = batch_idx;
            meta[5] = kv_block_idx;
        }
    }
}

// ---------------------------------------------------------------------------
// PT_smem: SMEM-staged tile prefix scan.
// Each block handles kRowsPerBlock rows for one head h. Cooperative load
// of tile_counts[*, h, base_r..base_r+M) into SMEM (better coalescing
// than per-warp uncoalesced stride reads), then per-warp scan in SMEM,
// then cooperative store back. Fuses row_ptr into the base.
// ---------------------------------------------------------------------------
template <int kThreads, int kRowsPerBlock>
__global__ void k2q_tile_prefix_smem_kernel(
    int* __restrict__ tile_counts,
    int const* __restrict__ row_ptr,
    int H, int total_rows, int G_total)
{
    static_assert(kRowsPerBlock > 0, "kRowsPerBlock must be positive");
    extern __shared__ int smem_tprefix[];
    // smem layout: smem[r_off][g] for r_off in [0, M), g in [0, G_total).

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    // Grid: H * blocks_per_h. Each block stays within a single head h
    // and processes kRowsPerBlock contiguous rows starting at b_in_h *
    // kRowsPerBlock. (Earlier flat-grid mapping `h = block_job /
    // total_rows; base_r = block_job - h*total_rows` skipped rows when
    // total_rows was not a multiple of kRowsPerBlock and H > 1, because
    // the last partial block of head h-1 left blocks of head h starting
    // at a non-zero row offset.)
    int blocks_per_h = (total_rows + kRowsPerBlock - 1) / kRowsPerBlock;
    int h = blockIdx.x / blocks_per_h;
    int b_in_h = blockIdx.x - h * blocks_per_h;
    if (h >= H) return;
    int base_r = b_in_h * kRowsPerBlock;
    if (base_r >= total_rows) return;
    int actual_M = min(kRowsPerBlock, total_rows - base_r);

    size_t stride_g = (size_t)H * total_rows;
    int* base_ptr = tile_counts + (size_t)h * total_rows + base_r;
    int total_elems = G_total * actual_M;

    // Cooperative load. Pattern: thread tid -> (r_off=tid%M, g=tid/M),
    // then strided. 32 lanes hit M r's × (32/M) g's, giving 32/M cache
    // lines per warp (vs 32 in the naive stride-along-g pattern).
    for (int i = tid; i < total_elems; i += kThreads) {
        int r_off = i % actual_M;
        int g = i / actual_M;
        smem_tprefix[r_off * G_total + g] = base_ptr[g * stride_g + r_off];
    }
    __syncthreads();

    // Per-warp scan: warp w scans row (base_r + w) if w < actual_M.
    if (warp_id < actual_M) {
        int abs_r = base_r + warp_id;
        int rp = row_ptr[(size_t)h * (total_rows + 1) + abs_r];
        int* my_smem = smem_tprefix + warp_id * G_total;
        int running = rp;
        for (int g0 = 0; g0 < G_total; g0 += kWarpSize) {
            int g = g0 + lane;
            int v = (g < G_total) ? my_smem[g] : 0;
            int x = v;
            #pragma unroll
            for (int off = 1; off < kWarpSize; off <<= 1) {
                int nbr = __shfl_up_sync(0xFFFFFFFF, x, off);
                if (lane >= off) x += nbr;
            }
            int excl = running + x - v;
            if (g < G_total) my_smem[g] = excl;
            int chunk_sum = __shfl_sync(0xFFFFFFFF, x, 31);
            running += chunk_sum;
        }
    }
    __syncthreads();

    // Cooperative store back.
    for (int i = tid; i < total_elems; i += kThreads) {
        int r_off = i % actual_M;
        int g = i / actual_M;
        base_ptr[g * stride_g + r_off] = smem_tprefix[r_off * G_total + g];
    }
}

// ---------------------------------------------------------------------------
// S: scatter. kWarps warps per CTA, each owns q-sub-range. Per-warp SMEM
// cursor and per-warp tile_offset slot range. Within a warp, q's are
// processed sequentially; lanes 0..kTopK-1 handle the topK slots in
// lockstep. Across distinct q's in the same warp, the lockstep ordering
// guarantees q-monotonic atomicAdd on smem_cursor[r].
// ---------------------------------------------------------------------------
// kQPerIter * kTopK lanes are active per warp iter; remaining lanes idle.
// For kTopK=16, kQPerIter=2 uses all 32 lanes; for kTopK=8, kQPerIter=4.
// CORRECTNESS NOTE: relies on lane-ordered SMEM atomicAdd return values
// within a single warp instruction (verified on SM100; tests pass).
//
// SMEM cursor stored as packed int16 (two cursors per int32). Per-warp
// row count must stay < 32768 (~q_per_warp * kTopK at max sink), which
// holds for all task.md sizes up to 1024K.
template <int kTopK, int kBlockK, int kWarps>
__global__ void k2q_scatter_kernel(
    int const* __restrict__ q2k,
    int const* __restrict__ cu_q,
    int const* __restrict__ row_map,
    int const* __restrict__ abs_base,
    int* __restrict__ q_idx,
    int* __restrict__ qsplit_idx,
    int* __restrict__ split_counts,
    int H, int B, int S_Q,
    int total_rows, int max_kv_blocks,
    int q_per_cta, int q_per_warp,
    int max_seqlen_q)
{
    constexpr int kQPerIter = kWarpSize / kTopK > 0 ? kWarpSize / kTopK : 1;
    extern __shared__ int smem_cursor_int[];
    int* smem_cursor = smem_cursor_int;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int c = blockIdx.x;
    int q_start_cta = c * q_per_cta;
    int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
    int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
    int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

    int q_in_iter = lane / kTopK;
    int slot_in_q = lane % kTopK;
    bool lane_active = (lane < kQPerIter * kTopK);

    // Per-warp packed cursor: total_rows int16 entries -> ceil(total_rows/2) int32.
    int packed_per_warp = (total_rows + 1) >> 1;
    int* my_cursor = smem_cursor + warp_id * packed_per_warp;

    for (int h = 0; h < H; ++h) {
        for (int i = lane; i < packed_per_warp; i += kWarpSize) my_cursor[i] = 0;
        __syncwarp();

        if (q_start_warp < q_end_warp) {
            int bi = 0;
            advance_batch_only(cu_q, B, q_start_warp, bi);

            int const* head_q2k = q2k + (size_t)h * S_Q * kTopK;
            int const* my_abs_base =
                abs_base + ((size_t)(c * kWarps + warp_id) * H + h) * total_rows;
            int* head_qidx = q_idx + (size_t)h * S_Q * kTopK;

            // (Hot-row register cache experiment showed no measurable
            // benefit; relying on L1 to keep row 0 / row total_rows-1
            // hot since they're hit every iteration in sink workloads.)

            constexpr int kUnroll = 16;
            int qi_base = q_start_warp;
            for (; qi_base + kUnroll * kQPerIter <= q_end_warp;
                 qi_base += kUnroll * kQPerIter) {
                int kvb[kUnroll];
                int qloc[kUnroll];
                int batch[kUnroll];
                int const* rmap[kUnroll];

                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    int qi_u = qi_base + u * kQPerIter + q_in_iter;
                    kvb[u] = -1;
                    qloc[u] = 0;
                    batch[u] = 0;
                    if (lane_active) {
                        advance_batch_only(cu_q, B, qi_u, bi);
                        qloc[u] = qi_u - cu_q[bi];
                        batch[u] = bi;
                        kvb[u] = head_q2k[(size_t)qi_u * kTopK + slot_in_q];
                    }
                    rmap[u] = row_map + (size_t)bi * max_kv_blocks;
                }

                int row[kUnroll];
                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    row[u] = -1;
                    if (lane_active && kvb[u] >= 0 && kvb[u] < max_kv_blocks)
                        row[u] = rmap[u][kvb[u]];
                }

                // Pre-issue all kUnroll abs_base loads in parallel before
                // the atomic chain so memory pipeline runs concurrently
                // with SMEM atomic-adds.
                int abs_v[kUnroll];
                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    abs_v[u] = (row[u] >= 0 && row[u] < total_rows)
                        ? my_abs_base[row[u]] : 0;
                }

                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    int r = row[u];
                    bool valid_edge = r >= 0 && r < total_rows;
                    unsigned int valid_mask = __ballot_sync(0xFFFFFFFFu, valid_edge);
                    unsigned int group_mask = (kTopK == 32)
                        ? 0xFFFFFFFFu
                        : (((1u << kTopK) - 1u) << (q_in_iter * kTopK));
                    unsigned int lower_lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
                    int split_slot = __popc(valid_mask & group_mask & lower_lane_mask);
                    int valid_count = __popc(valid_mask & group_mask);
                    if (split_counts != nullptr && slot_in_q == 0) {
                        int q_abs = cu_q[batch[u]] + qloc[u];
                        split_counts[(size_t)q_abs * H + h] = valid_count;
                    }
                    if (valid_edge) {
                        int slot = atomic_inc_int16_packed(my_cursor, r);
                        int out_pos = abs_v[u] + slot;
                        head_qidx[out_pos] = qloc[u];
                        if (qsplit_idx != nullptr) {
                            qsplit_idx[(size_t)h * S_Q * kTopK + out_pos] =
                                qloc[u] | ((split_slot & 0xFF) << 24);
                        }
                    }
                }
            }
            // Tail: 1-3 iters left.
            for (; qi_base < q_end_warp; qi_base += kQPerIter) {
                int my_qi = qi_base + q_in_iter;
                bool valid_q = (my_qi < q_end_warp) && lane_active;
                int kvb_local = -1;
                int q_local = 0;
                int batch_local = 0;
                if (valid_q) {
                    advance_batch_only(cu_q, B, my_qi, bi);
                    batch_local = bi;
                    q_local = my_qi - cu_q[bi];
                    kvb_local = head_q2k[(size_t)my_qi * kTopK + slot_in_q];
                }
                int const* my_row_map = row_map + (size_t)bi * max_kv_blocks;
                int row = -1;
                if (valid_q && kvb_local >= 0 && kvb_local < max_kv_blocks) {
                    row = my_row_map[kvb_local];
                }
                bool valid_edge = row >= 0 && row < total_rows;
                unsigned int valid_mask = __ballot_sync(0xFFFFFFFFu, valid_edge);
                unsigned int group_mask = (kTopK == 32)
                    ? 0xFFFFFFFFu
                    : (((1u << kTopK) - 1u) << (q_in_iter * kTopK));
                unsigned int lower_lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
                int split_slot = __popc(valid_mask & group_mask & lower_lane_mask);
                int valid_count = __popc(valid_mask & group_mask);
                if (split_counts != nullptr && valid_q && slot_in_q == 0) {
                    split_counts[(size_t)my_qi * H + h] = valid_count;
                }
                if (valid_edge) {
                    int slot = atomic_inc_int16_packed(my_cursor, row);
                    int out_pos = my_abs_base[row] + slot;
                    head_qidx[out_pos] = q_local;
                    if (qsplit_idx != nullptr) {
                        qsplit_idx[(size_t)h * S_Q * kTopK + out_pos] =
                            q_local | ((split_slot & 0xFF) << 24);
                    }
                }
            }
        }
        if (h + 1 < H) __syncthreads();
    }
}

}  // anonymous namespace

// ===========================================================================
// Host orchestration
// ===========================================================================

// MiniMax-M3 (MSA) F2 host-cost cut (SM107 only, eager only): the pipeline's five scratch buffers (row_counts,
// row_map, row_coords, tile_counts, edge_rank) are slices of one per-device int32 arena that grows on demand instead of
// five allocations per call (row_counts zeroed by a memset instead of torch::zeros), the SM count is read once per
// device, and cudaFuncSetAttribute is called only when a kernel's requested dynamic SMEM differs from the value last set
// (stock sets it on every call; the attribute state is the same). Every user of the arena runs on the caller's current
// stream, so reuse is stream-ordered; a capturing stream takes the stock allocations. Same kernels, same launches.
// Kill: FMHA_K2Q_WS_ARENA=0.
static bool k2q_arena_enabled() {
    static bool const on = [] {
        char const* env = std::getenv("FMHA_K2Q_WS_ARENA");
        bool const v = env == nullptr || std::atoi(env) != 0;
        std::fprintf(stderr, "[fmha_sm100] k2q CSR build: scratch arena %s (FMHA_K2Q_WS_ARENA)\n", v ? "on" : "off (stock)");
        return v;
    }();
    return on;
}

// Leaked on purpose: a static tensor array would free CUDA memory during static destruction, after the context is gone.
static torch::Tensor* const k2q_arena_slots = new torch::Tensor[64];
static cudaStream_t k2q_arena_stream[64] = {};
static bool k2q_arena_bound[64] = {};  // the default stream is a null handle, so binding needs its own flag

static torch::Tensor k2q_arena_take(int dev, torch::TensorOptions const& opts, int64_t words) {
    torch::Tensor& a = k2q_arena_slots[dev & 63];
    if (!a.defined() || a.numel() < words) {
        int64_t const grow = a.defined() ? std::max<int64_t>(words, a.numel() + a.numel() / 2) : words;
        a = torch::empty({grow}, opts);
    }
    return a;
}

// Every set is recorded (cached or not), so a cached skip always matches the attribute's current value.
static void k2q_set_max_dyn_smem(bool cached, int dev, void const* fn, int bytes) {
    static std::mutex mu;
    static std::unordered_map<void const*, int> last[64];
    std::lock_guard<std::mutex> lock(mu);
    auto& m = last[dev & 63];
    auto it = m.find(fn);
    if (cached && it != m.end() && it->second == bytes) return;
    AT_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    m[fn] = bytes;
}

template <int kTopK, int kBlockK>
static void launch_pipeline(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    int total_rows,
    int max_kv_blocks,
    torch::Tensor scheduler_metadata = torch::Tensor(),
    torch::Tensor work_count = torch::Tensor(),
    torch::Tensor qsplit_idx = torch::Tensor(),
    torch::Tensor split_counts = torch::Tensor(),
    int target_q_per_cta = 1,
    int work_capacity = 0,
    int max_seqlen_q = 0)
{
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int topK = (int)q2k.size(2);
    TORCH_CHECK(topK == kTopK, "topK runtime != template kTopK");
    int B = (int)cu_q.size(0) - 1;
    auto device = q2k.device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_CUDA_CHECK(cudaMemsetAsync(
        row_ptr.data_ptr<int>(), 0,
        (size_t)H * (total_rows + 1) * sizeof(int), stream));
    AT_CUDA_CHECK(cudaMemsetAsync(
        q_idx.data_ptr<int>(), 0xFF,
        (size_t)H * S_Q * kTopK * sizeof(int), stream));

    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
    torch::Tensor row_counts, row_map, row_coords;  // allocated below, once the tile grid is known
    bool emit_schedule = scheduler_metadata.defined();
    int* scheduler_metadata_ptr = emit_schedule ? scheduler_metadata.data_ptr<int>() : nullptr;
    int* work_count_ptr = emit_schedule ? work_count.data_ptr<int>() : nullptr;
    int* qsplit_idx_ptr = emit_schedule ? qsplit_idx.data_ptr<int>() : nullptr;
    int* split_counts_ptr = emit_schedule ? split_counts.data_ptr<int>() : nullptr;
    int* row_coords_ptr = nullptr;
    if (emit_schedule) {
        AT_CUDA_CHECK(cudaMemsetAsync(work_count_ptr, 0, sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(
            scheduler_metadata_ptr, 0,
            (size_t)work_capacity * 6 * sizeof(int), stream));
    }

    int dev = q2k.get_device();
    // The MiniMax-M3 k2q changes below (min q per CTA, edge-rank, multi-CTA emission, row-prefix claims, the scratch
    // arena) are validated on SM107 only; SM100 / SM103 keep the stock pipeline and allocations. SM107 loads the sm_100
    // build, so this is a runtime check, resolved once per device (host code only; the device code is unchanged).
    static signed char sm107_by_dev[64] = {};  // 0 unknown, 1 SM107, 2 other
    signed char& sm107_state = sm107_by_dev[dev & 63];
    if (sm107_state == 0) {
        int cc_major = 0, cc_minor = 0;
        AT_CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev));
        AT_CUDA_CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev));
        sm107_state = (cc_major == 10 && cc_minor == 7) ? 1 : 2;
    }
    bool const is_sm107 = sm107_state == 1;
    bool use_arena = false;
    if (is_sm107 && k2q_arena_enabled()) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        AT_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        // The arena is bound to the first stream that uses it; any other stream takes the stock allocations, so
        // reuse stays stream-ordered.
        if (capture == cudaStreamCaptureStatusNone && !k2q_arena_bound[dev & 63]) {
            k2q_arena_stream[dev & 63] = stream;
            k2q_arena_bound[dev & 63] = true;
        }
        use_arena = capture == cudaStreamCaptureStatusNone && k2q_arena_stream[dev & 63] == stream;
    }
    int num_sms = 0;
    static int num_sms_by_dev[64] = {};
    if (use_arena && num_sms_by_dev[dev & 63] > 0) {
        num_sms = num_sms_by_dev[dev & 63];
    } else {
        AT_CUDA_CHECK(cudaDeviceGetAttribute(
            &num_sms, cudaDevAttrMultiProcessorCount, dev));
        num_sms_by_dev[dev & 63] = num_sms;
    }

    // -- Pick kWarps per CTA based on SMEM budget for cursor/hist ---------
    // SMEM per CTA = kWarps * total_rows * sizeof(int) (for both H and S).
    // Want at least 2 CTAs/SM for memory parallelism. SM100 SMEM = 228KB.
    // Pick the largest kWarps that fits two CTAs/SM, capped at 4.
    // SMEM cursor packed as int16 (2 entries per int32 word):
    int per_warp_smem = ((total_rows + 1) >> 1) * (int)sizeof(int);
    int kWarps_pick = 4;
    while (kWarps_pick > 1 && (kWarps_pick * per_warp_smem) * 2 > 228 * 1024) {
        kWarps_pick >>= 1;
    }
    if (kWarps_pick < 1) kWarps_pick = 1;

    // -- Pick G (CTAs) ----------------------------------------------------
    // For each (kWarps, per_warp_smem) pair, the SMEM-bound occupancy is
    // 228KB / (kWarps*per_warp_smem) CTAs/SM. We size G as
    // num_sms * occupancy so a single resident wave covers all CTAs and
    // the memory pipeline runs at peak.
    int per_cta_smem_bytes = kWarps_pick * per_warp_smem;
    int max_ctas_per_sm = std::max(
        1, (228 * 1024) / std::max(1, per_cta_smem_bytes));
    if (max_ctas_per_sm > 8) max_ctas_per_sm = 8;
    // MiniMax-M3 (MSA): 128 query tokens per CTA (stock 256) doubles the hist / scatter grid at long context, where
    // the per-warp row histograms make each CTA latency-bound; the CSR (row_ptr, q_idx, qsplit, split_counts) is
    // bitwise the stock one, since rows stay q-ascending whatever the tile split. Kill: FMHA_K2Q_MIN_Q_PER_CTA=256.
    static int const kMinQPerCta = [] {
        char const* env = std::getenv("FMHA_K2Q_MIN_Q_PER_CTA");
        int const v = (env != nullptr && std::atoi(env) > 0) ? std::atoi(env) : 128;
        std::fprintf(stderr, "[fmha_sm100] k2q CSR build: %d query tokens per CTA (FMHA_K2Q_MIN_Q_PER_CTA)\n", v);
        return v;
    }();
    // Cap target_g at num_sms * 3 — empirically this balances
    // per-CTA work-size against parallelism. Higher caps regress
    // mid-size cases due to row_counts atomicAdd contention and
    // smaller q_per_cta. SMEM-bound configurations naturally cap
    // lower if max_ctas_per_sm < 3.
    int target_g = num_sms * std::min(max_ctas_per_sm, 3);
    int const min_q_per_cta = is_sm107 ? kMinQPerCta : 256;  // SM107 only; stock 256 elsewhere
    int max_g_for_q = (S_Q + min_q_per_cta - 1) / min_q_per_cta;
    int G = std::min({target_g, max_g_for_q, S_Q});
    if (G < 1) G = 1;
    int q_per_cta = (S_Q + G - 1) / G;
    G = (S_Q + q_per_cta - 1) / q_per_cta;
    int q_per_warp = (q_per_cta + kWarps_pick - 1) / kWarps_pick;
    // MiniMax-M3 (MSA) edge-rank build: per-CTA tiles, deterministic per-edge ranks, flat scatter (see
    // k2q_hist_rank_kernel). Same CSR as the per-warp pipeline. Kill: FMHA_K2Q_EDGE_RANK=0.
    static bool const kEdgeRankOn = [] {
        char const* env = std::getenv("FMHA_K2Q_EDGE_RANK");
        bool const on = env == nullptr || std::atoi(env) != 0;
        std::fprintf(stderr, "[fmha_sm100] k2q CSR build: edge-rank %s (FMHA_K2Q_EDGE_RANK)\n", on ? "on" : "off (stock)");
        return on;
    }();
    // Work items emitted on a (rows x heads) grid instead of H CTAs; order was already atomic-order.
    // Kill: FMHA_K2Q_SCHED_PAR=0.
    static bool const kSchedPar = [] {
        char const* env = std::getenv("FMHA_K2Q_SCHED_PAR");
        bool const on = env == nullptr || std::atoi(env) != 0;
        std::fprintf(stderr, "[fmha_sm100] k2q CSR build: multi-CTA work emission %s (FMHA_K2Q_SCHED_PAR)\n",
                     on ? "on" : "off (stock)");
        return on;
    }();
    bool const kEdgeRank = is_sm107 && kEdgeRankOn && kTopK == 16 && kWarps_pick == 4 && q_per_warp <= 32;  // SM107 only
    int G_total = kEdgeRank ? G : G * kWarps_pick;

    torch::Tensor tile_counts, edge_rank_ws;
    if (use_arena) {
        auto pad = [](int64_t w) { return (w + 31) / 32 * 32; };  // 128-byte aligned slices
        int64_t const n_rc = (int64_t)H * total_rows;
        int64_t const n_rm = (int64_t)B * max_kv_blocks;
        int64_t const n_co = emit_schedule ? (int64_t)total_rows * 2 : 0;
        int64_t const n_tc = (int64_t)G_total * H * total_rows;
        int64_t const n_er = kEdgeRank ? ((int64_t)H * S_Q * kTopK + 1) / 2 : 0;  // int16 entries in int32 words
        int64_t const o_rm = pad(n_rc), o_co = o_rm + pad(n_rm), o_tc = o_co + pad(n_co), o_er = o_tc + pad(n_tc);
        torch::Tensor arena = k2q_arena_take(dev, opts, o_er + pad(n_er));
        row_counts = arena.narrow(0, 0, n_rc).view({H, total_rows});
        AT_CUDA_CHECK(cudaMemsetAsync(row_counts.data_ptr<int>(), 0, (size_t)n_rc * sizeof(int), stream));
        row_map = arena.narrow(0, o_rm, n_rm).view({B, max_kv_blocks});
        if (emit_schedule) row_coords = arena.narrow(0, o_co, n_co).view({total_rows, 2});
        tile_counts = arena.narrow(0, o_tc, n_tc).view({G_total, H, total_rows});
        if (kEdgeRank) edge_rank_ws = arena.narrow(0, o_er, n_er);
    } else {
        row_counts = torch::zeros({H, total_rows}, opts);
        row_map = torch::empty({B, max_kv_blocks}, opts);
        if (emit_schedule) row_coords = torch::empty({total_rows, 2}, opts);
        tile_counts = torch::empty({G_total, H, total_rows}, opts);
    }
    row_coords_ptr = emit_schedule ? row_coords.data_ptr<int>() : nullptr;

    // -- Compile-time switch on kWarps for the templated kernels ---------
    auto rmap_fn = k2q_build_row_map_kernel<kBlockK>;
    // MiniMax-M3 (MSA): work items are claimed with one atomic per warp (a warp prefix sum of the per-row chunk
    // counts) instead of one per row. Only the order of scheduler_metadata entries changes, which was already
    // atomic-order in stock; the attention consumer is order-invariant (O/LSE bitwise under permuted orders).
    // Kill: FMHA_K2Q_ROW_PREFIX_AGG=0.
    static bool const kRowPrefixAgg = [] {
        char const* env = std::getenv("FMHA_K2Q_ROW_PREFIX_AGG");
        bool const on = env == nullptr || std::atoi(env) != 0;
        std::fprintf(stderr, "[fmha_sm100] k2q CSR build: row-prefix work claims %s (FMHA_K2Q_ROW_PREFIX_AGG)\n",
                     on ? "warp-aggregated" : "per row (stock)");
        return on;
    }();
    auto rprefix_fn = (is_sm107 && kRowPrefixAgg) ? k2q_row_prefix_agg_kernel<1024> : k2q_row_prefix_kernel<1024>;  // SM107 only
    constexpr int kPtRowsPerBlock = 8;
    constexpr int kPtThreads = 256;
    auto tprefix_smem_fn = k2q_tile_prefix_smem_kernel<kPtThreads, kPtRowsPerBlock>;

    if (max_kv_blocks > 0) {
        rmap_fn<<<max_kv_blocks, 32, 0, stream>>>(
            cu_k.data_ptr<int>(), row_map.data_ptr<int>(), row_coords_ptr, B, max_kv_blocks);
    }

    auto launch_hist_scatter = [&](auto kWarps_const) {
        constexpr int W = decltype(kWarps_const)::value;
        size_t smem_bytes = (size_t)W * per_warp_smem;
        auto hist_fn = k2q_hist_kernel<kTopK, kBlockK, W>;
        auto scat_fn = k2q_scatter_kernel<kTopK, kBlockK, W>;
        k2q_set_max_dyn_smem(use_arena, dev, (void const*)hist_fn, (int)smem_bytes);
        k2q_set_max_dyn_smem(use_arena, dev, (void const*)scat_fn, (int)smem_bytes);

        torch::Tensor edge_rank;
        if constexpr (W == 4 && kTopK == 16) {
            if (kEdgeRank) {
                edge_rank = use_arena ? edge_rank_ws : torch::empty({(long long)H * S_Q * kTopK}, opts.dtype(torch::kInt16));
                // More warps per CTA (the tiles are per CTA, so the CSR does not depend on it): the rank pass is
                // latency-bound at 4 warps. 16 / 8 / 4 warps by SMEM (warps x packed rows) and q per CTA.
                int rw = 16;
                while (rw > 4 && ((size_t)rw * per_warp_smem > 200 * 1024 || (q_per_cta + rw - 1) / rw < 2)) rw >>= 1;
                int qpw = (q_per_cta + rw - 1) / rw;
                auto go = [&](auto kern, int warps) {
                    size_t sb = (size_t)warps * per_warp_smem;
                    k2q_set_max_dyn_smem(use_arena, dev, (void const*)kern, (int)sb);
                    kern<<<G, warps * kWarpSize, sb, stream>>>(
                        q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
                        row_counts.data_ptr<int>(), tile_counts.data_ptr<int>(),
                        reinterpret_cast<short*>(edge_rank.data_ptr()),
                        H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, (q_per_cta + warps - 1) / warps);
                };
                if (rw == 16 && qpw <= 8) go(k2q_hist_rank_kernel<kTopK, 16, 4>, 16);
                else if (rw >= 8 && (q_per_cta + 7) / 8 <= 16) go(k2q_hist_rank_kernel<kTopK, 8, 8>, 8);
                else go(k2q_hist_rank_kernel<kTopK, 4, 16>, 4);
            }
        }
        if (!kEdgeRank) {
            hist_fn<<<G, W * kWarpSize, smem_bytes, stream>>>(
                q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
                row_counts.data_ptr<int>(), tile_counts.data_ptr<int>(),
                H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp);
        }

        bool const par = is_sm107 && kSchedPar && emit_schedule;  // SM107 only
        rprefix_fn<<<H, 1024, 0, stream>>>(
            row_counts.data_ptr<int>(), row_ptr.data_ptr<int>(),
            emit_schedule ? row_coords.data_ptr<int>() : nullptr,
            par ? nullptr : scheduler_metadata_ptr,
            par ? nullptr : work_count_ptr,
            total_rows,
            target_q_per_cta,
            work_capacity);
        if (par) {
            dim3 eg((total_rows + 255) / 256, H);
            k2q_sched_emit_kernel<<<eg, 256, 0, stream>>>(
                row_counts.data_ptr<int>(), row_coords.data_ptr<int>(), scheduler_metadata_ptr, work_count_ptr,
                total_rows, target_q_per_cta, work_capacity);
        }

        // Grid is H * blocks_per_h so each block stays within a single
        // head; flat (H*total_rows) grid would skip rows when total_rows
        // is not a multiple of kPtRowsPerBlock.
        int blocks_per_h = (total_rows + kPtRowsPerBlock - 1) / kPtRowsPerBlock;
        int pt_grid = H * blocks_per_h;
        if (pt_grid < 1) pt_grid = 1;
        size_t pt_smem = (size_t)kPtRowsPerBlock * G_total * sizeof(int);
        k2q_set_max_dyn_smem(use_arena, dev, (void const*)tprefix_smem_fn, (int)pt_smem);
        tprefix_smem_fn<<<pt_grid, kPtThreads, pt_smem, stream>>>(
            tile_counts.data_ptr<int>(), row_ptr.data_ptr<int>(),
            H, total_rows, G_total);

        if (kEdgeRank) {
            long long ne = (long long)H * S_Q * kTopK;
            k2q_scatter_rank_kernel<kTopK><<<(unsigned)((ne + 255) / 256), 256, 0, stream>>>(
                q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
                tile_counts.data_ptr<int>(), reinterpret_cast<short const*>(edge_rank.data_ptr()),
                q_idx.data_ptr<int>(), qsplit_idx_ptr, split_counts_ptr,
                H, B, S_Q, total_rows, max_kv_blocks, q_per_cta);
        } else {
            scat_fn<<<G, W * kWarpSize, smem_bytes, stream>>>(
                q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
                tile_counts.data_ptr<int>(), q_idx.data_ptr<int>(),
                qsplit_idx_ptr, split_counts_ptr,
                H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp,
                max_seqlen_q);
        }
    };

    if (kWarps_pick == 4) {
        launch_hist_scatter(std::integral_constant<int, 4>{});
    } else if (kWarps_pick == 2) {
        launch_hist_scatter(std::integral_constant<int, 2>{});
    } else {
        launch_hist_scatter(std::integral_constant<int, 1>{});
    }
}

void run_build_k2q_csr(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks)
{
    CHECK_INPUT(q2k);
    CHECK_INPUT(cu_q);
    CHECK_INPUT(cu_k);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(q_idx);
    TORCH_CHECK(blk_kv == 128, "build_k2q_csr only supports blk_kv == 128");
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int tr = (int)total_rows;
    int mkv = (int)max_kv_blocks;
    TORCH_CHECK(tr >= 0 && mkv >= 0,
                "total_rows / max_kv_blocks must be non-negative");
    TORCH_CHECK(row_ptr.size(0) == H && row_ptr.size(1) == tr + 1,
                "row_ptr shape mismatch");
    TORCH_CHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * (int)topk,
                "q_idx shape mismatch");
    if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaMemsetAsync(
            row_ptr.data_ptr<int>(), 0,
            (size_t)H * (tr + 1) * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(
            q_idx.data_ptr<int>(), 0xFF,
            (size_t)H * S_Q * (int)topk * sizeof(int), stream));
        return;
    }

    if (topk == 16) {
        launch_pipeline<16, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv);
    } else if (topk == 8) {
        launch_pipeline<8, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv);
    } else if (topk == 32) {
        launch_pipeline<32, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv);
    } else if (topk == 4) {
        launch_pipeline<4, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv);
    } else {
        TORCH_CHECK(false, "unsupported topK ", topk, " (expected 4, 8, 16, or 32)");
    }
}

void run_build_k2q_csr_with_schedule(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    torch::Tensor scheduler_metadata,
    torch::Tensor work_count,
    torch::Tensor qsplit_idx,
    torch::Tensor split_counts,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks,
    int64_t target_q_per_cta,
    int64_t work_capacity,
    int64_t max_seqlen_q)
{
    CHECK_INPUT(q2k);
    CHECK_INPUT(cu_q);
    CHECK_INPUT(cu_k);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(q_idx);
    CHECK_INPUT(scheduler_metadata);
    CHECK_INPUT(work_count);
    CHECK_INPUT(qsplit_idx);
    CHECK_INPUT(split_counts);
    TORCH_CHECK(blk_kv == 128, "build_k2q_csr only supports blk_kv == 128");
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int tr = (int)total_rows;
    int mkv = (int)max_kv_blocks;
    int target = (int)target_q_per_cta;
    int capacity = (int)work_capacity;
    int max_sq = (int)max_seqlen_q;
    TORCH_CHECK(tr >= 0 && mkv >= 0 && target > 0 && capacity > 0 && max_sq >= 0,
                "invalid schedule sizing arguments");
    TORCH_CHECK(row_ptr.size(0) == H && row_ptr.size(1) == tr + 1,
                "row_ptr shape mismatch");
    TORCH_CHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * (int)topk,
                "q_idx shape mismatch");
    TORCH_CHECK(qsplit_idx.sizes() == q_idx.sizes(), "qsplit_idx shape mismatch");
    TORCH_CHECK(scheduler_metadata.size(0) == capacity && scheduler_metadata.size(1) == 6,
                "scheduler_metadata shape mismatch");
    TORCH_CHECK(work_count.numel() == 1, "work_count must have one int32 element");
    TORCH_CHECK(split_counts.dim() == 2 && split_counts.size(0) == S_Q
                && split_counts.size(1) == H,
                "split_counts shape mismatch");
    if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaMemsetAsync(
            row_ptr.data_ptr<int>(), 0,
            (size_t)H * (tr + 1) * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(
            q_idx.data_ptr<int>(), 0xFF,
            (size_t)H * S_Q * (int)topk * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(work_count.data_ptr<int>(), 0, sizeof(int), stream));
        if (split_counts.numel() > 0) {
            AT_CUDA_CHECK(cudaMemsetAsync(
                split_counts.data_ptr<int>(), 0,
                (size_t)split_counts.numel() * sizeof(int), stream));
        }
        return;
    }

    if (topk == 16) {
        launch_pipeline<16, 128>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else if (topk == 8) {
        launch_pipeline<8, 128>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else if (topk == 32) {
        launch_pipeline<32, 128>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else if (topk == 4) {
        launch_pipeline<4, 128>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else {
        TORCH_CHECK(false, "unsupported topK ", topk, " (expected 4, 8, 16, or 32)");
    }
}


// ---------------------------------------------------------------------------
// MiniMax-M3.1 R1-SCHED: heaviest-work-item-first (LPT) issue order for the sparse forward schedule, one launch, no
// host sync. One CTA bitonic-sorts packed int64 keys in SMEM -- (q_cap - q_count) | head_kv | row_linear | q_begin |
// row index (unique, so the order is a pure function of the rows) -- then gathers each live row of a pre-copied scratch
// into its sorted position. Rows at or past *work_count sort last. Slots (qsplit_idx) are not touched: they are the
// popc rank of the entry in its query's q2k row, so the output is bitwise for any item order.
// ---------------------------------------------------------------------------
__global__ void k2q_lpt_sort_kernel(int const* __restrict__ src, int* __restrict__ dst,
                                    int const* __restrict__ work_count, int n, int npad, long long qcap,
                                    int s_c, int s_h, int s_r, int s_q)
{
    extern __shared__ long long lpt_keys[];
    int const tid = threadIdx.x;
    int const wc = *work_count;
    for (int i = tid; i < npad; i += blockDim.x) {
        long long k;
        if (i < n && i < wc) {
            int const* r = src + (size_t)i * 6;
            k = ((qcap - (long long)r[3]) << s_c) | ((long long)r[0] << s_h) | ((long long)r[1] << s_r)
                | ((long long)r[2] << s_q) | (long long)i;
        } else if (i < n) {
            k = (1LL << 62) | (long long)i;
        } else {
            k = (1LL << 62) | (1LL << 61) | (long long)i;
        }
        lpt_keys[i] = k;
    }
    __syncthreads();
    for (int size = 2; size <= npad; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < npad; i += blockDim.x) {
                int const j = i ^ stride;
                if (j > i) {
                    long long const a = lpt_keys[i];
                    long long const b = lpt_keys[j];
                    bool const ascending = (i & size) == 0;
                    if ((a > b) == ascending) {
                        lpt_keys[i] = b;
                        lpt_keys[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    long long const idx_mask = (1LL << s_q) - 1;
    for (int i = tid; i < n; i += blockDim.x) {
        int const from = (int)(lpt_keys[i] & idx_mask);
        int const* r = src + (size_t)from * 6;
        int* d = dst + (size_t)i * 6;
#pragma unroll
        for (int c = 0; c < 6; ++c) d[c] = r[c];
    }
}


// MiniMax-M3.1 K2Q-FAST: the LPT issue order by RANK (the same unique key as k2q_lpt_sort_kernel). Each CTA loads every live
// key into SMEM; a WARP owns a row (< work_count): its lanes count the live keys smaller than the row's key, reduce, and
// lanes 0-5 copy the row to that rank. Keys are unique, so the ranks are the bitonic sort's permutation exactly; rows at or
// past work_count are not touched (the bitonic kernel rewrites them unchanged). 256-thread CTAs of 32 rows (8 warps x 4).
__global__ void __launch_bounds__(256) k2q_lpt_rank_sort_kernel(int const* __restrict__ src, int* __restrict__ dst,
    int const* __restrict__ work_count, int n, long long qcap, int s_c, int s_h, int s_r, int s_q)
{
    extern __shared__ long long lpt_rank_keys[];
    int const wc = min(*work_count, n);
    int const row0 = (int)blockIdx.x * 32;
    if (row0 >= wc) return;
    for (int i = threadIdx.x; i < wc; i += blockDim.x) {
        int const* r = src + (size_t)i * 6;
        lpt_rank_keys[i] = ((qcap - (long long)r[3]) << s_c) | ((long long)r[0] << s_h) | ((long long)r[1] << s_r)
            | ((long long)r[2] << s_q) | (long long)i;
    }
    __syncthreads();
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        int const i = row0 + warp * 4 + t;
        if (i >= wc) break;  // warp-uniform
        long long const mine = lpt_rank_keys[i];
        unsigned cnt = 0;
        for (int j = lane; j < wc; j += 32) cnt += lpt_rank_keys[j] < mine;
        int const rank = (int)__reduce_add_sync(0xffffffffu, cnt);
        if (lane < 6) dst[(size_t)rank * 6 + lane] = src[(size_t)i * 6 + lane];
    }
}

static bool k2q_fast_enabled()
{
    char const* v = std::getenv("TRTLLM_M31_DISABLE_K2Q_FAST");
    return !(v != nullptr && v[0] == '1' && v[1] == '\0');
}

// In place on scheduler_metadata [n, 6] int32 (CUDA, contiguous); work_count int32 [1] on device. Returns false (and
// does nothing) when n exceeds the single-CTA SMEM budget, so the caller can fall back.
bool run_lpt_reorder(torch::Tensor scheduler_metadata, torch::Tensor work_count, int64_t qcap,
                     int64_t s_c, int64_t s_h, int64_t s_r, int64_t s_q)
{
    TORCH_CHECK(scheduler_metadata.is_cuda() && scheduler_metadata.is_contiguous()
                && scheduler_metadata.scalar_type() == at::kInt && scheduler_metadata.dim() == 2
                && scheduler_metadata.size(1) == 6, "scheduler_metadata must be contiguous int32 [n, 6] on CUDA");
    TORCH_CHECK(work_count.is_cuda() && work_count.scalar_type() == at::kInt && work_count.numel() >= 1,
                "work_count must be int32 on CUDA");
    int const n = (int)scheduler_metadata.size(0);
    int npad = 2;
    while (npad < n) npad <<= 1;
    size_t const smem = (size_t)npad * sizeof(long long);
    if (smem > 200 * 1024) return false;
    c10::cuda::CUDAGuard guard(scheduler_metadata.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto scratch = torch::empty_like(scheduler_metadata);
    AT_CUDA_CHECK(cudaMemcpyAsync(scratch.data_ptr<int>(), scheduler_metadata.data_ptr<int>(),
                                  (size_t)n * 6 * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    if (smem > 48 * 1024) {
        AT_CUDA_CHECK(cudaFuncSetAttribute((void const*)k2q_lpt_sort_kernel,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    }
    if (k2q_fast_enabled()) {
        if (smem > 48 * 1024) {
            AT_CUDA_CHECK(cudaFuncSetAttribute((void const*)k2q_lpt_rank_sort_kernel,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        }
        k2q_lpt_rank_sort_kernel<<<(n + 31) / 32, 256, smem, stream>>>(scratch.data_ptr<int>(),
            scheduler_metadata.data_ptr<int>(), work_count.data_ptr<int>(), n, (long long)qcap,
            (int)s_c, (int)s_h, (int)s_r, (int)s_q);
    } else {
        k2q_lpt_sort_kernel<<<1, 1024, smem, stream>>>(scratch.data_ptr<int>(), scheduler_metadata.data_ptr<int>(),
                                                       work_count.data_ptr<int>(), n, npad, (long long)qcap,
                                                       (int)s_c, (int)s_h, (int)s_r, (int)s_q);
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return true;
}

// ===========================================================================
// MiniMax-M3.1 K2Q_FUSE (MSA, PREREG_M31_MSA_K2Q_FUSE_v1): the SM107 edge-rank + multi-CTA emission + LPT path of
// run_build_k2q_csr_with_schedule + run_lpt_reorder in 1 memset + 3 launches (K2 / K3 chained with PDL) instead of
// 4-5 memsets + 7 launches + a D2D copy. Every output (row_ptr, q_idx, qsplit_idx, split_counts, scheduler_metadata,
// work_count) is bitwise the two-call path's:
//   K1 = k2q_hist_rank (same tiles, same ranks) with the row map computed inline per lookup; its threads also write
//        the row map / row coords tables (for K2 / K3) and the fills the memsets did (q_idx = -1, scheduler_metadata = 0,
//        row_ptr[h][0] = 0, work_count = 0).
//   K2 = row prefix (heads) | work-item emission into the LPT scratch | tile prefix, by block index. The tile prefix
//        takes its row_ptr base from a local prefix of row_counts (the same integer sums), so it needs no barrier.
//   K3 = flat scatter | LPT rank sort (scratch -> scheduler_metadata), by block index.
// Only row_counts keeps a memset (K1 adds into it). Kill: TRTLLM_M31_DISABLE_TP2CTX_K2Q_FUSE=1.
// ===========================================================================
namespace {

__device__ __forceinline__ void k2q_pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void k2q_pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// row_map[b][level] of k2q_build_row_map_kernel, computed in place (O(B)).
template <int kBlockK>
__device__ __forceinline__ int k2q_row_of(int const* __restrict__ cu_k, int B, int b, int level)
{
    int rows_before = 0, active_before = 0, rb_self = 0;
    for (int bb = 0; bb < B; ++bb) {
        int rb = (cu_k[bb + 1] - cu_k[bb] + kBlockK - 1) / kBlockK;
        rows_before += (rb < level ? rb : level);
        if (bb < b && rb > level) ++active_before;
        if (bb == b) rb_self = rb;
    }
    return rb_self > level ? rows_before + active_before : -1;
}

template <int kTopK, int kBlockK, int kWarps, int kIters>
__global__ void k2q_fused_hist_kernel(
    int const* __restrict__ q2k, int const* __restrict__ cu_q, int const* __restrict__ cu_k,
    int* __restrict__ row_map, int* __restrict__ row_coords,
    int* __restrict__ row_counts, int* __restrict__ tile_counts, short* __restrict__ edge_rank,
    int* __restrict__ q_idx, int* __restrict__ row_ptr, int* __restrict__ scheduler_metadata,
    int* __restrict__ work_count, int work_capacity,
    int H, int B, int S_Q, int total_rows, int max_kv_blocks, int q_per_cta, int q_per_warp)
{
    constexpr int kQPerIter = kWarpSize / kTopK;
    constexpr int kThreads = kWarps * kWarpSize;
    extern __shared__ int smem_hist_int[];
    k2q_pdl_wait();
    k2q_pdl_trigger();
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int c = blockIdx.x;

    // The memsets' fills and the row map / row coords tables (k2q_build_row_map_kernel), grid-stride.
    {
        long long const gt = (long long)blockIdx.x * kThreads + tid;
        long long const gs = (long long)gridDim.x * kThreads;
        long long const nq = (long long)H * S_Q * kTopK;
        for (long long i = gt; i < nq; i += gs) q_idx[i] = -1;
        long long const nm = (long long)work_capacity * 6;
        for (long long i = gt; i < nm; i += gs) scheduler_metadata[i] = 0;
        if (gt < H) row_ptr[gt * (total_rows + 1)] = 0;
        if (gt == 0) *work_count = 0;
        for (long long level = gt; level < max_kv_blocks; level += gs) {
            int rows_before = 0;
            for (int b = 0; b < B; ++b) {
                int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
                rows_before += (rb < (int)level ? rb : (int)level);
            }
            int active_before = 0;
            for (int b = 0; b < B; ++b) {
                int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
                if (rb > (int)level) {
                    int row_linear = rows_before + active_before;
                    row_map[(size_t)b * max_kv_blocks + level] = row_linear;
                    row_coords[(size_t)row_linear * 2] = b;
                    row_coords[(size_t)row_linear * 2 + 1] = (int)level;
                    ++active_before;
                } else {
                    row_map[(size_t)b * max_kv_blocks + level] = -1;
                }
            }
        }
    }

    // k2q_hist_rank_kernel, with row_map[bi][kvb] computed in place.
    int q_start_cta = c * q_per_cta;
    int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
    int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
    int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);
    int q_in_iter = lane / kTopK;
    int slot_in_q = lane % kTopK;
    int packed_per_warp = (total_rows + 1) >> 1;
    int* my_hist = smem_hist_int + warp_id * packed_per_warp;

    for (int h = 0; h < H; ++h) {
        for (int i = lane; i < packed_per_warp; i += kWarpSize) my_hist[i] = 0;
        __syncwarp();
        int const* head_q2k = q2k + (size_t)h * S_Q * kTopK;
        int row[kIters];
        int rank[kIters];
        int bi = 0;
        #pragma unroll
        for (int it = 0; it < kIters; ++it) {
            int qi = q_start_warp + it * kQPerIter + q_in_iter;
            row[it] = -1;
            rank[it] = 0;
            if (qi < q_end_warp) {
                advance_batch_only(cu_q, B, qi, bi);
                int kvb = head_q2k[(size_t)qi * kTopK + slot_in_q];
                if (kvb >= 0 && kvb < max_kv_blocks) {
                    int r = k2q_row_of<kBlockK>(cu_k, B, bi, kvb);
                    if (r >= 0 && r < total_rows) row[it] = r;
                }
            }
        }
        unsigned int const lt_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
        #pragma unroll
        for (int it = 0; it < kIters; ++it) {
            int r = row[it];
            unsigned int peers = __match_any_sync(0xFFFFFFFFu, r >= 0 ? r : -1 - lane);
            int leader = __ffs(peers) - 1;
            int base = 0;
            if (r >= 0 && lane == leader) {
                int shift = (r & 1) << 4;
                base = (atomicAdd(&my_hist[r >> 1], __popc(peers) << shift) >> shift) & 0xFFFF;
            }
            base = __shfl_sync(0xFFFFFFFFu, base, leader);
            rank[it] = base + __popc(peers & lt_mask);
            __syncwarp();
        }
        __syncthreads();
        int* cta_tile = tile_counts + ((size_t)c * H + h) * total_rows;
        int* head_row_counts = row_counts + (size_t)h * total_rows;
        for (int j = tid; j < packed_per_warp; j += kThreads) {
            int running = 0;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                int v = smem_hist_int[w * packed_per_warp + j];
                smem_hist_int[w * packed_per_warp + j] = running;
                running += v;
            }
            int r0 = 2 * j, s0 = running & 0xFFFF, s1 = (running >> 16) & 0xFFFF;
            cta_tile[r0] = s0;
            if (s0 > 0) atomicAdd(&head_row_counts[r0], s0);
            if (r0 + 1 < total_rows) {
                cta_tile[r0 + 1] = s1;
                if (s1 > 0) atomicAdd(&head_row_counts[r0 + 1], s1);
            }
        }
        __syncthreads();
        short* head_rank = edge_rank + (size_t)h * S_Q * kTopK;
        #pragma unroll
        for (int it = 0; it < kIters; ++it) {
            int qi = q_start_warp + it * kQPerIter + q_in_iter;
            if (row[it] >= 0)
                head_rank[(size_t)qi * kTopK + slot_in_q] = (short)(rank[it] + read_int16_packed(my_hist, row[it]));
        }
        if (h + 1 < H) __syncthreads();
    }
}

constexpr int kFusedMidThreads = 1024;
constexpr int kFusedPtRows = 32;  // tile-prefix rows per 1024-thread block (one warp per row)

__global__ void __launch_bounds__(kFusedMidThreads) k2q_fused_mid_kernel(
    int const* __restrict__ row_counts, int* __restrict__ row_ptr, int const* __restrict__ row_coords,
    int* __restrict__ lpt_scratch, int* __restrict__ work_count, int* __restrict__ tile_counts,
    int H, int total_rows, int G_total, int target_q_per_cta, int work_capacity)
{
    extern __shared__ int smem_mid[];
    __shared__ int red[kFusedMidThreads / kWarpSize];
    k2q_pdl_wait();
    k2q_pdl_trigger();
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp_id = tid >> 5;
    int const emit_per_h = (total_rows + kFusedMidThreads - 1) / kFusedMidThreads;
    int const pt_per_h = (total_rows + kFusedPtRows - 1) / kFusedPtRows;
    int bid = blockIdx.x;

    if (bid < H) {
        // Row prefix: k2q_row_prefix_agg_kernel<1024> (no emission: the multi-CTA emission below does it).
        int h = bid;
        int* scan_buf = smem_mid;
        int const* head_counts = row_counts + (size_t)h * total_rows;
        int* head_rowptr = row_ptr + (size_t)h * (total_rows + 1);
        int chunk = (total_rows + kFusedMidThreads - 1) / kFusedMidThreads;
        int lo = tid * chunk;
        int hi = min(lo + chunk, total_rows);
        int local_sum = 0;
        for (int i = lo; i < hi; ++i) local_sum += head_counts[i];
        scan_buf[tid] = local_sum;
        __syncthreads();
        for (int off = 1; off < kFusedMidThreads; off <<= 1) {
            int add = (tid >= off) ? scan_buf[tid - off] : 0;
            __syncthreads();
            scan_buf[tid] += add;
            __syncthreads();
        }
        int running = scan_buf[tid] - local_sum;
        for (int i = lo; i < hi; ++i) {
            running += head_counts[i];
            head_rowptr[i + 1] = running;
        }
        return;
    }
    bid -= H;
    if (bid < H * emit_per_h) {
        // Work-item emission: k2q_sched_emit_kernel (one atomic claim per warp; the LPT sort fixes the order).
        int h = bid / emit_per_h;
        int i = (bid - h * emit_per_h) * kFusedMidThreads + tid;
        int row_count = i < total_rows ? row_counts[(size_t)h * total_rows + i] : 0;
        int num_chunks = row_count > 0 ? (row_count + target_q_per_cta - 1) / target_q_per_cta : 0;
        int x = num_chunks;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int nbr = __shfl_up_sync(0xFFFFFFFFu, x, off);
            if (lane >= off) x += nbr;
        }
        int warp_total = __shfl_sync(0xFFFFFFFFu, x, 31);
        int base = 0;
        if (lane == 31 && warp_total > 0) base = atomicAdd(work_count, warp_total);
        base = __shfl_sync(0xFFFFFFFFu, base, 31) + x - num_chunks;
        if (num_chunks == 0) return;
        int batch_idx = row_coords[(size_t)i * 2];
        int kv_block_idx = row_coords[(size_t)i * 2 + 1];
        for (int cc = 0; cc < num_chunks; ++cc) {
            int work_idx = base + cc;
            if (work_idx < work_capacity) {
                int q_begin = cc * target_q_per_cta;
                int* meta = lpt_scratch + (size_t)work_idx * 6;
                meta[0] = h;
                meta[1] = i;
                meta[2] = q_begin;
                meta[3] = min(target_q_per_cta, row_count - q_begin);
                meta[4] = batch_idx;
                meta[5] = kv_block_idx;
            }
        }
        return;
    }
    bid -= H * emit_per_h;
    // Tile prefix: k2q_tile_prefix_smem_kernel with 32 rows per block; row_ptr[h][r] = sum of row_counts[h][0..r).
    int h = bid / pt_per_h;
    if (h >= H) return;
    int base_r = (bid - h * pt_per_h) * kFusedPtRows;
    if (base_r >= total_rows) return;
    int actual_M = min(kFusedPtRows, total_rows - base_r);
    int const* head_counts = row_counts + (size_t)h * total_rows;
    int part = 0;
    for (int i = tid; i < base_r; i += kFusedMidThreads) part += head_counts[i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xFFFFFFFFu, part, off);
    if (lane == 0) red[warp_id] = part;
    size_t stride_g = (size_t)H * total_rows;
    int* base_ptr = tile_counts + (size_t)h * total_rows + base_r;
    int total_elems = G_total * actual_M;
    for (int i = tid; i < total_elems; i += kFusedMidThreads) {
        int r_off = i % actual_M;
        int g = i / actual_M;
        smem_mid[r_off * G_total + g] = base_ptr[g * stride_g + r_off];
    }
    __syncthreads();
    if (warp_id < actual_M) {
        int rp = 0;
        for (int w = 0; w < kFusedMidThreads / kWarpSize; ++w) rp += red[w];
        for (int r = 0; r < warp_id; ++r) rp += head_counts[base_r + r];
        int* my_smem = smem_mid + warp_id * G_total;
        int running = rp;
        for (int g0 = 0; g0 < G_total; g0 += kWarpSize) {
            int g = g0 + lane;
            int v = (g < G_total) ? my_smem[g] : 0;
            int x = v;
            #pragma unroll
            for (int off = 1; off < kWarpSize; off <<= 1) {
                int nbr = __shfl_up_sync(0xFFFFFFFF, x, off);
                if (lane >= off) x += nbr;
            }
            int excl = running + x - v;
            if (g < G_total) my_smem[g] = excl;
            int chunk_sum = __shfl_sync(0xFFFFFFFF, x, 31);
            running += chunk_sum;
        }
    }
    __syncthreads();
    for (int i = tid; i < total_elems; i += kFusedMidThreads) {
        int r_off = i % actual_M;
        int g = i / actual_M;
        base_ptr[g * stride_g + r_off] = smem_mid[r_off * G_total + g];
    }
}

template <int kTopK>
__global__ void __launch_bounds__(256) k2q_fused_tail_kernel(
    int const* __restrict__ q2k, int const* __restrict__ cu_q, int const* __restrict__ row_map,
    int const* __restrict__ abs_base, short const* __restrict__ edge_rank,
    int* __restrict__ q_idx, int* __restrict__ qsplit_idx, int* __restrict__ split_counts,
    int H, int B, int S_Q, int total_rows, int max_kv_blocks, int q_per_cta, int scatter_blocks,
    int const* __restrict__ lpt_src, int* __restrict__ lpt_dst, int const* __restrict__ work_count, int n,
    long long qcap, int s_c, int s_h, int s_r, int s_q)
{
    extern __shared__ long long lpt_keys_f[];
    k2q_pdl_wait();
    k2q_pdl_trigger();
    if ((int)blockIdx.x < scatter_blocks) {
        // k2q_scatter_rank_kernel
        long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        long long ne = (long long)H * S_Q * kTopK;
        int lane = threadIdx.x & 31;
        bool in = e < ne;
        int slot = (int)(e % kTopK);
        long long hq = e / kTopK;
        int q = in ? (int)(hq % S_Q) : 0;
        int h = in ? (int)(hq / S_Q) : 0;
        int row = -1, qloc = 0;
        if (in) {
            int lo = 0, hi = B - 1;
            while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (cu_q[mid] <= q) lo = mid; else hi = mid - 1; }
            qloc = q - cu_q[lo];
            int kvb = q2k[e];
            if (kvb >= 0 && kvb < max_kv_blocks) {
                int r = row_map[(size_t)lo * max_kv_blocks + kvb];
                if (r >= 0 && r < total_rows) row = r;
            }
        }
        bool valid = row >= 0;
        unsigned int vmask = __ballot_sync(0xFFFFFFFFu, valid);
        unsigned int gmask = (kTopK >= 32) ? 0xFFFFFFFFu : (((1u << kTopK) - 1u) << ((lane / kTopK) * kTopK));
        unsigned int lower = lane == 0 ? 0u : ((1u << lane) - 1u);
        int split_slot = __popc(vmask & gmask & lower);
        if (!in) return;
        if (split_counts != nullptr && slot == 0) split_counts[(size_t)q * H + h] = __popc(vmask & gmask);
        if (!valid) return;
        int c = q / q_per_cta;
        int pos = abs_base[((size_t)c * H + h) * total_rows + row] + (int)edge_rank[e];
        size_t o = (size_t)h * S_Q * kTopK + pos;
        q_idx[o] = qloc;
        if (qsplit_idx != nullptr) qsplit_idx[o] = qloc | ((split_slot & 0xFF) << 24);
        return;
    }
    // k2q_lpt_rank_sort_kernel
    int const wc = min(*work_count, n);
    int const row0 = ((int)blockIdx.x - scatter_blocks) * 32;
    if (row0 >= wc) return;
    for (int i = threadIdx.x; i < wc; i += blockDim.x) {
        int const* r = lpt_src + (size_t)i * 6;
        lpt_keys_f[i] = ((qcap - (long long)r[3]) << s_c) | ((long long)r[0] << s_h) | ((long long)r[1] << s_r)
            | ((long long)r[2] << s_q) | (long long)i;
    }
    __syncthreads();
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        int const i = row0 + warp * 4 + t;
        if (i >= wc) break;
        long long const mine = lpt_keys_f[i];
        unsigned cnt = 0;
        for (int j = lane; j < wc; j += 32) cnt += lpt_keys_f[j] < mine;
        int const rank = (int)__reduce_add_sync(0xffffffffu, cnt);
        if (lane < 6) lpt_dst[(size_t)rank * 6 + lane] = lpt_src[(size_t)i * 6 + lane];
    }
}

}  // anonymous namespace

static bool k2q_fuse_enabled()
{
    static bool const on = [] {
        char const* v = std::getenv("TRTLLM_M31_DISABLE_TP2CTX_K2Q_FUSE");
        bool const o = !(v != nullptr && v[0] == '1' && v[1] == '\0');
        std::fprintf(stderr, "[fmha_sm100] k2q CSR build: K2Q_FUSE %s (TRTLLM_M31_DISABLE_TP2CTX_K2Q_FUSE)\n",
                     o ? "on (1 memset + 3 launches)" : "off");
        return o;
    }();
    return on;
}

static torch::Tensor* const k2q_fused_arena_slots = new torch::Tensor[64];

template <typename Kern, typename... Args>
static void k2q_launch_pdl(Kern kern, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args... args)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    AT_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
}

// The fused build + LPT order. Returns false (nothing launched) when the call is outside the fused path's shapes; the
// caller then runs run_build_k2q_csr_with_schedule + run_lpt_reorder.
bool run_build_k2q_csr_lpt_fused(
    torch::Tensor q2k, torch::Tensor cu_q, torch::Tensor cu_k, torch::Tensor row_ptr, torch::Tensor q_idx,
    torch::Tensor scheduler_metadata, torch::Tensor work_count, torch::Tensor qsplit_idx, torch::Tensor split_counts,
    int64_t topk, int64_t blk_kv, int64_t total_rows, int64_t max_kv_blocks, int64_t target_q_per_cta,
    int64_t work_capacity, int64_t max_seqlen_q, int64_t qcap, int64_t s_c, int64_t s_h, int64_t s_r, int64_t s_q)
{
    if (!k2q_fuse_enabled() || topk != 16 || blk_kv != 128 || !k2q_fast_enabled()) return false;
    // The LPT key must fit 61 bits (the caller's `fits`): s_c + bit length of qcap (= 2^bc - 1).
    {
        int bc = 0;
        for (int64_t v = qcap; v > 0; v >>= 1) ++bc;
        if (qcap < 0 || s_c < 0 || s_c + bc > 61) return false;
    }
    int const H = (int)q2k.size(0);
    int const S_Q = (int)q2k.size(1);
    int const tr = (int)total_rows;
    int const mkv = (int)max_kv_blocks;
    int const target = (int)target_q_per_cta;
    int const capacity = (int)work_capacity;
    int const B = (int)cu_q.size(0) - 1;
    if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0 || B < 1 || target <= 0 || capacity <= 0) return false;
    if (!(q2k.is_cuda() && q2k.is_contiguous() && cu_q.is_contiguous() && cu_k.is_contiguous()
          && scheduler_metadata.is_contiguous() && scheduler_metadata.size(0) == capacity)) return false;
    int const dev = q2k.get_device();
    int cc_major = 0, cc_minor = 0, num_sms = 0;
    static int sm_by_dev[64] = {};
    static int smc_by_dev[64] = {};
    if (sm_by_dev[dev & 63] == 0) {
        AT_CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev));
        AT_CUDA_CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev));
        AT_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
        sm_by_dev[dev & 63] = (cc_major == 10 && cc_minor == 7) ? 1 : 2;
        smc_by_dev[dev & 63] = num_sms;
    }
    if (sm_by_dev[dev & 63] != 1 || !k2q_arena_enabled()) return false;
    num_sms = smc_by_dev[dev & 63];
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    AT_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) return false;
    // The tile grid of launch_pipeline<16, 128> (same G, q_per_cta, edge-rank variant).
    int per_warp_smem = ((tr + 1) >> 1) * (int)sizeof(int);
    int kWarps_pick = 4;
    while (kWarps_pick > 1 && (kWarps_pick * per_warp_smem) * 2 > 228 * 1024) kWarps_pick >>= 1;
    if (kWarps_pick != 4) return false;
    int per_cta_smem_bytes = kWarps_pick * per_warp_smem;
    int max_ctas_per_sm = std::max(1, (228 * 1024) / std::max(1, per_cta_smem_bytes));
    if (max_ctas_per_sm > 8) max_ctas_per_sm = 8;
    char const* mq = std::getenv("FMHA_K2Q_MIN_Q_PER_CTA");
    int const min_q_per_cta = (mq != nullptr && std::atoi(mq) > 0) ? std::atoi(mq) : 128;
    int target_g = num_sms * std::min(max_ctas_per_sm, 3);
    int max_g_for_q = (S_Q + min_q_per_cta - 1) / min_q_per_cta;
    int G = std::min({target_g, max_g_for_q, S_Q});
    if (G < 1) G = 1;
    int q_per_cta = (S_Q + G - 1) / G;
    G = (S_Q + q_per_cta - 1) / q_per_cta;
    int q_per_warp = (q_per_cta + kWarps_pick - 1) / kWarps_pick;
    if (q_per_warp > 32) return false;
    int rw = 16;
    while (rw > 4 && ((size_t)rw * per_warp_smem > 200 * 1024 || (q_per_cta + rw - 1) / rw < 2)) rw >>= 1;
    int qpw = (q_per_cta + rw - 1) / rw;
    int const n = capacity;
    {
        // run_lpt_reorder's budget (npad = next power of two >= n), so the fused entry declines exactly where it does
        int npad = 2;
        while (npad < n) npad <<= 1;
        if ((size_t)npad * sizeof(long long) > 200 * 1024) return false;
    }

    // Arena: row_counts | row_map | row_coords | tile_counts | edge_rank | lpt scratch.
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(q2k.device());
    auto pad = [](int64_t w) { return (w + 31) / 32 * 32; };
    int64_t const n_rc = (int64_t)H * tr, n_rm = (int64_t)B * mkv, n_co = (int64_t)tr * 2;
    int64_t const n_tc = (int64_t)G * H * tr, n_er = ((int64_t)H * S_Q * 16 + 1) / 2, n_ls = (int64_t)n * 6;
    int64_t const o_rm = pad(n_rc), o_co = o_rm + pad(n_rm), o_tc = o_co + pad(n_co), o_er = o_tc + pad(n_tc),
                  o_ls = o_er + pad(n_er), words = o_ls + pad(n_ls);
    torch::Tensor& a = k2q_fused_arena_slots[dev & 63];
    if (!a.defined() || a.numel() < words) {
        int64_t const grow = a.defined() ? std::max<int64_t>(words, a.numel() + a.numel() / 2) : words;
        a = torch::empty({grow}, opts);
    }
    int* base = a.data_ptr<int>();
    int* row_counts = base;
    int* row_map = base + o_rm;
    int* row_coords = base + o_co;
    int* tile_counts = base + o_tc;
    short* edge_rank = reinterpret_cast<short*>(base + o_er);
    int* lpt_scratch = base + o_ls;
    AT_CUDA_CHECK(cudaMemsetAsync(row_counts, 0, (size_t)n_rc * sizeof(int), stream));

    size_t sb = (size_t)rw * per_warp_smem;
    auto k1 = [&](auto kern, int warps) {
        k2q_set_max_dyn_smem(true, dev, (void const*)kern, (int)sb);
        kern<<<G, warps * kWarpSize, sb, stream>>>(
            q2k.data_ptr<int>(), cu_q.data_ptr<int>(), cu_k.data_ptr<int>(), row_map, row_coords, row_counts,
            tile_counts, edge_rank, q_idx.data_ptr<int>(), row_ptr.data_ptr<int>(), scheduler_metadata.data_ptr<int>(),
            work_count.data_ptr<int>(), n, H, B, S_Q, tr, mkv, q_per_cta, (q_per_cta + warps - 1) / warps);
    };
    if (rw == 16 && qpw <= 8) k1(k2q_fused_hist_kernel<16, 128, 16, 4>, 16);
    else if (rw >= 8 && (q_per_cta + 7) / 8 <= 16) k1(k2q_fused_hist_kernel<16, 128, 8, 8>, 8);
    else k1(k2q_fused_hist_kernel<16, 128, 4, 16>, 4);

    int const emit_per_h = (tr + kFusedMidThreads - 1) / kFusedMidThreads;
    int const pt_per_h = (tr + kFusedPtRows - 1) / kFusedPtRows;
    size_t const mid_smem = std::max((size_t)kFusedPtRows * G, (size_t)kFusedMidThreads) * sizeof(int);
    k2q_set_max_dyn_smem(true, dev, (void const*)k2q_fused_mid_kernel, (int)mid_smem);
    k2q_launch_pdl(k2q_fused_mid_kernel, dim3(H + H * emit_per_h + H * pt_per_h), dim3(kFusedMidThreads), mid_smem,
                   stream, (int const*)row_counts, row_ptr.data_ptr<int>(), (int const*)row_coords, lpt_scratch,
                   work_count.data_ptr<int>(), tile_counts, H, tr, G, target, n);

    long long const ne = (long long)H * S_Q * 16;
    int const scatter_blocks = (int)((ne + 255) / 256);
    size_t const tail_smem = (size_t)n * sizeof(long long);
    k2q_set_max_dyn_smem(true, dev, (void const*)k2q_fused_tail_kernel<16>, (int)std::max<size_t>(tail_smem, 1));
    // Up to 48 KB of LPT keys the scatter and the LPT sort share one launch; above it the scatter CTAs would carry the
    // keys' SMEM, so the two run as two PDL launches (scatter, then the sort).
    bool const one_tail = tail_smem <= 48 * 1024;
    auto tail = [&](int grid, int scat, size_t smem) {
        k2q_launch_pdl(k2q_fused_tail_kernel<16>, dim3(grid), dim3(256), smem, stream,
                       (int const*)q2k.data_ptr<int>(), (int const*)cu_q.data_ptr<int>(), (int const*)row_map,
                       (int const*)tile_counts, (short const*)edge_rank, q_idx.data_ptr<int>(),
                       qsplit_idx.data_ptr<int>(), split_counts.data_ptr<int>(), H, B, S_Q, tr, mkv, q_per_cta, scat,
                       (int const*)lpt_scratch, scheduler_metadata.data_ptr<int>(),
                       (int const*)work_count.data_ptr<int>(), n, (long long)qcap, (int)s_c, (int)s_h, (int)s_r,
                       (int)s_q);
    };
    if (one_tail) {
        tail(scatter_blocks + (n + 31) / 32, scatter_blocks, tail_smem);
    } else {
        tail(scatter_blocks, scatter_blocks, 0);
        tail((n + 31) / 32, 0, tail_smem);
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_build_k2q_csr_lpt_fused", &run_build_k2q_csr_lpt_fused,
          "MiniMax-M3.1 K2Q_FUSE: run_build_k2q_csr_with_schedule + run_lpt_reorder in 1 memset + 3 launches "
          "(returns false, launching nothing, outside its shapes)",
          pybind11::arg("q2k"), pybind11::arg("cu_q"), pybind11::arg("cu_k"), pybind11::arg("row_ptr"),
          pybind11::arg("q_idx"), pybind11::arg("scheduler_metadata"), pybind11::arg("work_count"),
          pybind11::arg("qsplit_idx"), pybind11::arg("split_counts"), pybind11::arg("topk"), pybind11::arg("blk_kv"),
          pybind11::arg("total_rows"), pybind11::arg("max_kv_blocks"), pybind11::arg("target_q_per_cta"),
          pybind11::arg("work_capacity"), pybind11::arg("max_seqlen_q"), pybind11::arg("qcap"), pybind11::arg("s_c"),
          pybind11::arg("s_h"), pybind11::arg("s_r"), pybind11::arg("s_q"));
    m.def("run_lpt_reorder", &run_lpt_reorder, "MiniMax-M3.1 R1-SCHED LPT re-sort of scheduler_metadata (in place)",
          pybind11::arg("scheduler_metadata"), pybind11::arg("work_count"), pybind11::arg("qcap"),
          pybind11::arg("s_c"), pybind11::arg("s_h"), pybind11::arg("s_r"), pybind11::arg("s_q"));
    m.def("run_build_k2q_csr", &run_build_k2q_csr,
          "q2k -> k2q CSR build (sorted within row)",
          pybind11::arg("q2k"),
          pybind11::arg("cu_q"),
          pybind11::arg("cu_k"),
          pybind11::arg("row_ptr"),
          pybind11::arg("q_idx"),
          pybind11::arg("topk"),
          pybind11::arg("blk_kv"),
          pybind11::arg("total_rows"),
          pybind11::arg("max_kv_blocks"));
    m.def("run_build_k2q_csr_with_schedule", &run_build_k2q_csr_with_schedule,
          "q2k -> k2q CSR build with fused attention schedule metadata",
          pybind11::arg("q2k"),
          pybind11::arg("cu_q"),
          pybind11::arg("cu_k"),
          pybind11::arg("row_ptr"),
          pybind11::arg("q_idx"),
          pybind11::arg("scheduler_metadata"),
          pybind11::arg("work_count"),
          pybind11::arg("qsplit_idx"),
          pybind11::arg("split_counts"),
          pybind11::arg("topk"),
          pybind11::arg("blk_kv"),
          pybind11::arg("total_rows"),
          pybind11::arg("max_kv_blocks"),
          pybind11::arg("target_q_per_cta"),
          pybind11::arg("work_capacity"),
          pybind11::arg("max_seqlen_q"));
}
