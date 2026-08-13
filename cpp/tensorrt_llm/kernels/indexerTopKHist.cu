/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from:
 * https://github.com/sgl-project/sglang/blob/d03c8cee8090bdfa63f6476c6f7e150ad4244f50/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh
 * Adapted from:
 * https://github.com/sgl-project/sglang/blob/d03c8cee8090bdfa63f6476c6f7e150ad4244f50/python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk_impl.cuh
 * SPDX-FileCopyrightText: Copyright contributors to the sglang project
 *
 * The upstream sources above are themselves adapted from, and preserve credit to:
 *   https://github.com/vllm-project/vllm/blob/a8c6ee9b787d273916206a29b77feebadb80c368/csrc/persistent_topk.cuh
 *   https://github.com/flashinfer-ai/flashinfer/blob/c2b4db2b1a84448d802f0e6ac445243312bd6a4c/include/flashinfer/topk.cuh
 */

// ============================================================================
// DSA-indexer decode top-k kernel, ported to TensorRT-LLM (v1: SELECTION
// ONLY -- emits the same LOCAL indices the stock topKPerRowDecode produces, no
// page-table gather fusion). See the file header above for upstream provenance.
//
// Ported entities: the host dispatch + __global__ kernels, the device-side
// TopKConfig / TopKRadixBase / TopKRegister / TopKStreaming / TopKCluster<8>
// classes, and ~10 small helpers (div_ceil, AlignedVector, warp sum-reduce,
// PDL wait/trigger, fp16 cast) inlined below.
//
// Algorithm (per row): fp16 coarse histogram -> threshold bin -> classify by two
// fp32 boundaries -> exact radix tie-break. The long-row path (rowEnd >
// cluster_floor) runs one 8-block cooperative-groups cluster per row, reducing
// the coarse histogram across the 8 ranks via distributed shared memory
// (map_shared_rank); short rows fall to the single-block Register4/Streaming
// paths, and rows with rowEnd <= topK take the trivial identity+(-1)-pad path.
//
// Deviations from the upstream source (all documented inline):
//   * TopKProblem carries only {in, out, topk, seq_len}; the page_table /
//     raw_out / transform_output machinery is dropped (v1 is selection-only).
//   * The fused kernel writes RAW selected indices straight to the global output
//     row (all cluster ranks share the same global pointer, so the original
//     shared-memory staging + page-table transform pass is unnecessary).
//   * usePDL is a caller-provided template bool, not the upstream hardcoded `true`.
//   * device::warp::reduce_sum / cast<fp16_t> are replaced by local minimal
//     equivalents (integer warp sum-reduce; __float2half_rn / __half2float).
//
// EXACT-MATCH row-length contract (see topKPerRowDecode in indexerTopK.cu):
//   seq_len       = seqLens[rowIdx / next_n]   (clamped >= 0, read as uint32)
//   actual_kv_len = seq_len - next_n + (rowIdx % next_n) + 1
//   rowEnd        = actual_kv_len / compressRatio
// ============================================================================

#include "tensorrt_llm/kernels/indexerTopKHist.h"

#include "tensorrt_llm/common/cudaUtils.h"

#include <cfloat>
#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace topk_hist
{

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Inlined minimal helpers (see the upstream provenance in the file header).
// ---------------------------------------------------------------------------

#define INDEXER_TOPK_HIST_DEVICE __forceinline__ __device__

using fp16_t = __half;

inline constexpr uint32_t kWarpSize = 32u;

// Ceiling-division helper.
template <typename T, typename U>
INDEXER_TOPK_HIST_DEVICE constexpr auto div_ceil(T a, U b)
{
    return (a + b - 1) / b;
}

// fp16 <-> float cast helpers.
INDEXER_TOPK_HIST_DEVICE fp16_t to_fp16(float x)
{
    return __float2half_rn(x);
}

INDEXER_TOPK_HIST_DEVICE float to_float(fp16_t x)
{
    return __half2float(x);
}

// PDL wait/trigger helpers (griddepcontrol on sm90+).
template <bool kUsePDL>
INDEXER_TOPK_HIST_DEVICE void pdlWaitPrimary()
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (kUsePDL)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#endif
}

template <bool kUsePDL>
INDEXER_TOPK_HIST_DEVICE void pdlTriggerSecondary()
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (kUsePDL)
    {
        asm volatile("griddepcontrol.launch_dependents;" :::);
    }
#endif
}

// Shared-memory spilling hint (CUDA 13+ only; no-op otherwise).
INDEXER_TOPK_HIST_DEVICE void enableSmemSpilling()
{
#if defined(__CUDA_ARCH__) && CUDART_VERSION >= 13000
    asm(".pragma \"enable_smem_spilling\";");
#endif
}

// Minimal port of device::warp::reduce_sum<kGroup> for 32-bit integers, kInner:
// each lane receives the sum over its kGroup-sized consecutive-lane group
// (kGroup == kWarpSize => full-warp sum). Matches the __shfl_xor_sync ladder in
// warp.cuh (width = full warp, mask halved each step).
template <uint32_t kGroup = kWarpSize>
INDEXER_TOPK_HIST_DEVICE uint32_t warpReduceSum(uint32_t value)
{
    static_assert(kGroup >= 1 && kGroup <= kWarpSize && (kGroup & (kGroup - 1)) == 0, "kGroup must be pow2 <= 32");
#pragma unroll
    for (uint32_t mask = kGroup / 2; mask >= 1; mask >>= 1)
    {
        value += __shfl_xor_sync(0xFFFFFFFFu, value, mask, kWarpSize);
    }
    return value;
}

// Minimal port of device::AlignedVector<T, N> (vectorized 16-byte loads).
template <typename T, uint32_t N>
struct alignas(sizeof(T) * N) AlignedVector
{
    static_assert((N > 0) && ((N & (N - 1)) == 0) && sizeof(T) * N <= 16, "AlignedVector: N pow2, <= 16 bytes");

    T m_data[N];

    INDEXER_TOPK_HIST_DEVICE void load(void const* ptr, int64_t offset = 0)
    {
        *this = reinterpret_cast<AlignedVector const*>(ptr)[offset];
    }

    INDEXER_TOPK_HIST_DEVICE void fill(T value)
    {
#pragma unroll
        for (uint32_t i = 0; i < N; ++i)
        {
            m_data[i] = value;
        }
    }

    INDEXER_TOPK_HIST_DEVICE T& operator[](uint32_t idx)
    {
        return m_data[idx];
    }

    INDEXER_TOPK_HIST_DEVICE T operator[](uint32_t idx) const
    {
        return m_data[idx];
    }
};

// ---------------------------------------------------------------------------
// Shared-memory storage sized/aligned for several impl Smem types (MaxSmem).
// ---------------------------------------------------------------------------

template <typename T>
constexpr T ct_max(T a)
{
    return a;
}

template <typename T, typename... Ts>
constexpr T ct_max(T a, Ts... rest)
{
    T const m = ct_max(rest...);
    return a > m ? a : m;
}

template <typename... Smems>
struct MaxSmem
{
    static constexpr size_t kSize = ct_max(sizeof(Smems)...);
    static constexpr size_t kAlign = ct_max(alignof(Smems)...);
    alignas(kAlign) uint8_t storage[kSize];
};

// ---------------------------------------------------------------------------
// Order-preserving float -> integer key extraction (topk_impl.cuh L72-135)
// ---------------------------------------------------------------------------

INDEXER_TOPK_HIST_DEVICE uint32_t extract_exact_bin(float x)
{
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// ceil(log2(x)) for x >= 1 (0 for x <= 1). Used to size 10-bit range-relative
// radix digits in the long-context exact-refinement fallback.
INDEXER_TOPK_HIST_DEVICE uint32_t ceilLog2U64(uint64_t x)
{
    if (x <= 1ull)
        return 0u;
    return 64u - static_cast<uint32_t>(__clzll(static_cast<long long>(x - 1ull)));
}

template <uint32_t kBits>
INDEXER_TOPK_HIST_DEVICE uint32_t extract_coarse_bin(float x)
{
    static_assert(0 < kBits && kBits < 15);
    auto const hx = to_fp16(x);
    uint16_t const bits = *reinterpret_cast<uint16_t const*>(&hx);
    uint16_t const key = (bits & 0x8000) ? ~bits : bits | 0x8000;
    return key >> (16 - kBits);
}

// Lower fp32 boundary of coarse bin `bin`; see topk_impl.cuh for the derivation.
template <uint32_t kBits>
INDEXER_TOPK_HIST_DEVICE float coarse_bin_lower_bound(uint32_t bin)
{
    constexpr uint32_t kShift = 16 - kBits;
    uint32_t const key = bin << kShift;
    auto const to_finite_val = [](uint32_t okey) -> float
    {
        uint16_t const ob = static_cast<uint16_t>(okey);
        uint16_t const hb = (ob & 0x8000) ? static_cast<uint16_t>(ob ^ 0x8000) : static_cast<uint16_t>(~ob);
        return to_float(*reinterpret_cast<fp16_t const*>(&hb));
    };
    if (key - 0x0401u <= 0xFBFFu - 0x0401u && bin < (1u << kBits))
    {
        return 0.5f * (to_finite_val(key) + to_finite_val(key - 1));
    }
    if (bin == 0)
        return -FLT_MAX;
    if (bin >= (1u << kBits))
        return FLT_MAX;
    auto const to_val = [&](uint32_t okey) -> float
    {
        constexpr float k_Inf = std::numeric_limits<float>::infinity();
        if (okey < 0x03FFu)
            return -k_Inf;
        if (okey == 0x03FFu)
            return -65536.0f;
        if (okey == 0xFC00u)
            return 65536.0f;
        if (okey > 0xFC00u)
            return FLT_MAX;
        return to_finite_val(okey);
    };
    return 0.5f * (to_val(key) + to_val(key - 1));
}

INDEXER_TOPK_HIST_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val)
{
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset *= 2)
    {
        uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset)
            val += n;
    }
    return val;
}

INDEXER_TOPK_HIST_DEVICE uint32_t warp_sum_bool(bool pred, uint32_t mask = 0xFFFFFFFF)
{
    return __popc(__ballot_sync(mask, pred));
}

struct alignas(8) TieValue
{
    float value;
    uint32_t idx;

    static constexpr TieValue invalid()
    {
        return TieValue{-FLT_MAX, 0xFFFFFFFFu};
    }
};

// ---------------------------------------------------------------------------
// Per-row problem description (selection-only: no page-table transform).
// ---------------------------------------------------------------------------

struct TopKProblem
{
    float const* __restrict__ in;
    int32_t* __restrict__ out; // [topk] local indices, -1 padded
    uint32_t topk;
    uint32_t seq_len;

    // Write the raw selected local index. (The upstream kernel applies a page-table transform
    // in a later pass; v1 keeps the raw index, matching topKPerRowDecode.)
    INDEXER_TOPK_HIST_DEVICE void emit(uint32_t pos, uint32_t raw_idx) const
    {
        out[pos] = static_cast<int32_t>(raw_idx);
    }
};

// ---------------------------------------------------------------------------
// Shared configuration + tie handling (exact radix select on the threshold bin)
// ---------------------------------------------------------------------------

struct TopKConfig
{
    static constexpr uint32_t kMaxTopK = 2048;
    static constexpr uint32_t kBlockSize = 1024;
    static constexpr uint32_t kOccupancy = 2;
    static constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;
    static constexpr uint32_t kMaxNumTie = 2048;
    static constexpr uint32_t kRadixSize = 1 << 8;
    static constexpr uint32_t kTopKItems = (kMaxTopK + kBlockSize - 1) / kBlockSize;
    static constexpr uint32_t kTieItems = kMaxNumTie / kBlockSize;
    static_assert(kMaxNumTie >= kMaxTopK && kMaxNumTie % kBlockSize == 0 && kBlockSize % kNumWarps == 0);

    struct TieHandleSmem
    {
        struct alignas(16) MatchBin
        {
            uint32_t bin;
            uint32_t above_count;
            uint32_t equal_count;
            uint32_t _pad = 0;
        };

        alignas(128) uint32_t counter;
        alignas(128) uint32_t counter_final;
        MatchBin match;
        uint32_t warp_sum[kNumWarps];
        uint32_t histogram[2][kRadixSize];
    };

    INDEXER_TOPK_HIST_DEVICE static void handle_tie( //
        TieValue const* tie_buffer, TopKProblem const& problem, uint32_t const base, uint32_t const num_ties,
        uint32_t const topk, TieHandleSmem* smem)
    {
        constexpr auto is_greater = [](TieValue const& a, TieValue const& b)
        { return (a.value > b.value) || (a.value == b.value && a.idx < b.idx); };
        auto const tx = threadIdx.x;
        auto const lane_id = tx % kWarpSize;
        auto const warp_id = tx / kWarpSize;
        static_assert(kNumWarps == kWarpSize);

        if (num_ties <= topk)
        {
            for (uint32_t t = tx; t < num_ties; t += kBlockSize)
            {
                problem.emit(base + t, tie_buffer[t].idx);
            }
            for (uint32_t t = num_ties + tx; t < topk; t += kBlockSize)
            {
                problem.emit(base + t, base + t);
            }
        }
        else if (num_ties <= kWarpSize)
        {
            if (lane_id >= num_ties || warp_id >= num_ties)
                return;
            uint32_t const mask = (1ull << num_ties) - 1u;
            auto const tie = tie_buffer[lane_id];
            auto const target = tie_buffer[warp_id];
            auto const rank = warp_sum_bool(is_greater(tie, target), mask);
            if (lane_id == 0 && rank < topk)
                problem.emit(base + rank, target.idx);
        }
        else if (num_ties <= kWarpSize * 2)
        {
            auto const warp_id_0 = warp_id;
            auto const warp_id_1 = warp_id + kWarpSize;
            auto const lane_id_1 = lane_id + kWarpSize;
            auto const invalid = TieValue::invalid();
            auto const tie_0 = tie_buffer[lane_id];
            auto const tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
            auto const target_0 = tie_buffer[warp_id_0];
            auto const target_1 = tie_buffer[warp_id_1];
            if (true)
            {
                auto const rank_0 = warp_sum_bool(is_greater(tie_0, target_0));
                auto const rank_1 = warp_sum_bool(is_greater(tie_1, target_0));
                auto const rank = rank_0 + rank_1;
                if (lane_id == 0 && rank < topk)
                    problem.emit(base + rank, target_0.idx);
            }
            if (warp_id_1 < num_ties)
            {
                auto const rank_0 = warp_sum_bool(is_greater(tie_0, target_1));
                auto const rank_1 = warp_sum_bool(is_greater(tie_1, target_1));
                auto const rank = rank_0 + rank_1;
                if (lane_id == 0 && rank < topk)
                    problem.emit(base + rank, target_1.idx);
            }
        }
        else if (num_ties <= kWarpSize * 4)
        {
            auto const invalid = TieValue::invalid();
            TieValue const tie[] = {
                tie_buffer[lane_id + 0 * kWarpSize],
                tie_buffer[lane_id + 1 * kWarpSize],
                lane_id + 2 * kWarpSize < num_ties ? tie_buffer[lane_id + 2 * kWarpSize] : invalid,
                lane_id + 3 * kWarpSize < num_ties ? tie_buffer[lane_id + 3 * kWarpSize] : invalid,
            };
            TieValue const target[] = {
                tie_buffer[warp_id + 0 * kWarpSize],
                tie_buffer[warp_id + 1 * kWarpSize],
                tie_buffer[warp_id + 2 * kWarpSize],
                tie_buffer[warp_id + 3 * kWarpSize],
            };
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                if (i >= 2 && warp_id + i * kWarpSize >= num_ties)
                    break;
                uint32_t rank = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    rank += warp_sum_bool(is_greater(tie[j], target[i]));
                }
                if (lane_id == 0 && rank < topk)
                    problem.emit(base + rank, target[i].idx);
            }
        }
        else if (num_ties <= kBlockSize)
        {
            radix_tie_select<1>(tie_buffer, problem, base, num_ties, topk, smem);
        }
        else
        {
            radix_tie_select<kTieItems>(tie_buffer, problem, base, num_ties, topk, smem);
        }
    }

    template <uint32_t kItems>
    INDEXER_TOPK_HIST_DEVICE static void radix_tie_select( //
        TieValue const* tie_buffer, TopKProblem const& problem, uint32_t const base, uint32_t const num_ties,
        uint32_t const topk, TieHandleSmem* smem)
    {
        auto const tx = threadIdx.x;
        auto const lane_id = tx % kWarpSize;
        auto const warp_id = tx / kWarpSize;

        bool active[kItems];
        uint32_t key[kItems];
        uint32_t idx[kItems];
        uint32_t write_pos[kItems];
#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i)
        {
            auto const t = tx + i * kBlockSize;
            active[i] = t < num_ties;
            auto const tie = active[i] ? tie_buffer[t] : TieValue::invalid();
            key[i] = extract_exact_bin(tie.value);
            idx[i] = tie.idx;
            write_pos[i] = topk;
        }
        uint32_t topk_remain = topk;
        if (tx < kRadixSize)
            smem->histogram[0][tx] = 0;
        if (tx == kRadixSize)
            smem->counter = smem->counter_final = 0;
        __syncthreads();
        uint32_t total_active = num_ties;

#pragma unroll
        for (int round = 0; round < 4; round++)
        {
            uint32_t const shift = 24 - round * 8;
            auto const hist_idx = round % 2;
            auto const histogram = smem->histogram[hist_idx];

#pragma unroll
            for (uint32_t i = 0; i < kItems; ++i)
            {
                if (active[i])
                    atomicAdd(&histogram[(key[i] >> shift) & 0xFFu], 1);
            }
            if (round < 3 && tx < kRadixSize)
            {
                smem->histogram[hist_idx ^ 1][tx] = 0;
            }
            __syncthreads();

            uint32_t hist_val = 0;
            uint32_t warp_inc = 0;
            if (tx < kRadixSize)
            {
                hist_val = histogram[tx];
                warp_inc = warp_inclusive_sum(lane_id, hist_val);
                if (lane_id == kWarpSize - 1)
                    smem->warp_sum[warp_id] = warp_inc;
            }
            __syncthreads();
            if (tx < kRadixSize)
            {
                auto const inter = warpReduceSum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
                auto const prefix = inter + warp_inc;
                auto const above = total_active - prefix;
                if (above < topk_remain && above + hist_val >= topk_remain)
                {
                    smem->match = {tx, above, hist_val};
                }
            }
            __syncthreads();

            auto const [threshold_bin, above_count, equal_count, __] = smem->match;
            if (round < 3)
                total_active = equal_count;
            topk_remain -= above_count;

#pragma unroll
            for (uint32_t i = 0; i < kItems; ++i)
            {
                if (!active[i])
                    continue;
                uint32_t const bin = (key[i] >> shift) & 0xFFu;
                if (bin > threshold_bin)
                {
                    write_pos[i] = atomicAdd(&smem->counter, 1);
                    active[i] = false;
                }
                else if (bin < threshold_bin)
                {
                    active[i] = false;
                }
                else if (round == 3)
                {
                    write_pos[i] = topk - topk_remain + atomicAdd(&smem->counter_final, 1);
                }
            }

            if (round == 3 || topk_remain == 0)
                break;
        }

#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i)
        {
            if (write_pos[i] < topk)
                problem.emit(base + write_pos[i], idx[i]);
        }
    }
};

// ---------------------------------------------------------------------------
// Radix base: histogram storage + input iteration + threshold-bin search
// ---------------------------------------------------------------------------

template <uint32_t kHistBits_>
struct TopKRadixBase : TopKConfig
{
    static constexpr uint32_t kVecSize = 4;
    static constexpr uint32_t kHistBits = kHistBits_;
    static constexpr uint32_t kHistSize = 1 << kHistBits;
    using vec_t = AlignedVector<float, kVecSize>;

    struct Smem
    {
        using kHistVec = AlignedVector<uint32_t, kHistSize / kBlockSize>;
        alignas(128) uint32_t count_eq;
        alignas(128) uint32_t count_gt;
        uint32_t threshold_bin;
        uint32_t warp_sum[kNumWarps];
        // Long-context exact-refinement scratch (overflow fallback).
        // Separate members -- they do NOT alias the histogram/tie union below.
        uint32_t refine_bin, refine_above, refine_equal, emit_counter;

        union
        {
            uint32_t histogram[kHistSize];
            kHistVec hist_vecs[kBlockSize];

            struct
            {
                TieHandleSmem handle;
                TieValue values[kMaxNumTie];
            } tie;
        };
    };

protected:
    template <typename F>
    INDEXER_TOPK_HIST_DEVICE static void for_each_input(float const* __restrict__ in, uint32_t seq_len, F&& fn)
    {
        auto const tx = threadIdx.x;
        uint32_t const num_full = seq_len / kVecSize;

        vec_t next_vec;
        uint32_t vi = tx;
        if (vi < num_full)
            next_vec.load(in, vi);
        while (vi < num_full)
        {
            auto const cur = next_vec;
            auto const base = vi * kVecSize;
            vi += kBlockSize;
            if (vi < num_full)
                next_vec.load(in, vi);
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
            {
                fn(cur[j], base + j);
            }
        }

        static_assert(kVecSize <= kBlockSize);
        uint32_t const tail_start = num_full * kVecSize;
        if (tx < seq_len - tail_start)
        {
            auto const idx = tail_start + tx;
            fn(in[idx], idx);
        }
    }

    INDEXER_TOPK_HIST_DEVICE static void find_threshold(uint32_t const topk, uint32_t const seq_len, Smem* smem)
    {
        auto const tx = threadIdx.x;
        constexpr uint32_t kItems = kHistSize / kBlockSize;
        uint32_t orig[kItems];
        auto const hist_vec = smem->hist_vecs[tx];
        uint32_t tmp_local_sum = 0;

#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i)
        {
            orig[i] = hist_vec[i];
            tmp_local_sum += orig[i];
        }

        auto const lane_id = tx % kWarpSize;
        auto const warp_id = tx / kWarpSize;
        auto const warp_inc = warp_inclusive_sum(lane_id, tmp_local_sum);
        auto const warp_exc = warp_inc - tmp_local_sum;
        if (lane_id == kWarpSize - 1)
            smem->warp_sum[warp_id] = warp_inc;

        __syncthreads();

        auto const tmp = smem->warp_sum[lane_id];
        uint32_t prefix_sum = warpReduceSum(lane_id < warp_id ? tmp : 0);
        prefix_sum += warp_exc;
#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i)
        {
            prefix_sum += orig[i];
            auto const above = seq_len - prefix_sum;
            if (above < topk && above + orig[i] >= topk)
            {
                smem->threshold_bin = tx * kItems + i;
            }
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // Long-context / peaked-row correctness fallback (shared by all tiers).
    //
    // The coarse fp16 histogram can leave a threshold
    // bin holding FAR more than kMaxNumTie elements -- fp16 rounding collapses
    // many distinct fp32 values into one ~1/16-binade bin -- so the fixed 2048-
    // entry tie buffer TRUNCATES and handle_tie then selects from an incomplete
    // set (wrong top-K). This happens for any row whose top coarse bin exceeds
    // kMaxNumTie ties (uniform data: rowLen > ~65k -> the whole cluster domain).
    //
    // Fix: detect overflow from find_threshold's exact (DSMEM-summed, pre-scatter)
    // bin population, and on overflow DISCARD the coarse result and redo the row
    // with an exact-key radix select. Mirrors the stock kernel's own fp16-bin-
    // overflow -> exact fp32 restart (indexerTopK.cu). Runs single-block on the
    // calling CTA (Register4 / Streaming), or on the cluster's elected primary
    // while the other ranks return to the outer barrier -- no cluster.sync here.
    // ------------------------------------------------------------------------

    // Block-wide threshold search over the 1024-bucket histogram (one bucket per
    // thread): find the bucket holding the `remaining`-th largest element and
    // publish {refine_bin, refine_above (count in higher buckets), refine_equal}.
    INDEXER_TOPK_HIST_DEVICE static void findThresholdFull(uint32_t remaining, Smem* smem)
    {
        auto const tx = threadIdx.x;
        auto const lane = tx % kWarpSize;
        auto const warp = tx / kWarpSize;
        static_assert(kHistSize >= kBlockSize, "findThresholdFull uses the first kBlockSize buckets");
        static_assert(kNumWarps == kWarpSize, "findThresholdFull assumes kNumWarps == warpSize");
        uint32_t const h = smem->histogram[tx];
        uint32_t const winc = warp_inclusive_sum(lane, h);
        if (lane == kWarpSize - 1)
            smem->warp_sum[warp] = winc;
        __syncthreads();
        uint32_t const wprefix = warpReduceSum(lane < warp ? smem->warp_sum[lane] : 0u);
        uint32_t const prefix_inc = wprefix + winc; // buckets [0..tx]
        uint32_t const total = warpReduceSum(lane < kNumWarps ? smem->warp_sum[lane] : 0u);
        uint32_t const above = total - prefix_inc;  // buckets > tx (higher value)
        if (above < remaining && above + h >= remaining)
        {
            smem->refine_bin = tx;
            smem->refine_above = above;
            smem->refine_equal = h;
        }
        __syncthreads();
    }

    // Emit up to `count` indices whose exact key == `key` into out[base..base+count)
    // (value-set correct; identical-value ties resolved by atomic arrival order).
    INDEXER_TOPK_HIST_DEVICE static void emitExactKey(float const* __restrict__ in, uint32_t n, uint32_t key,
        uint32_t base, uint32_t count, uint32_t topk, int32_t* __restrict__ out, Smem* smem)
    {
        auto const tx = threadIdx.x;
        if (tx == 0)
            smem->emit_counter = 0u;
        __syncthreads();
        for_each_input(in, n,
            [&](float val, uint32_t idx)
            {
                if (extract_exact_bin(val) == key)
                {
                    uint32_t const slot = atomicAdd(&smem->emit_counter, 1u);
                    if (slot < count && base + slot < topk)
                        out[base + slot] = static_cast<int32_t>(idx);
                }
            });
        __syncthreads();
    }

    // Single-CTA exact-key radix top-K over in[0..n) (requires n > topk). Emits
    // exactly `topk` unique indices into out[0..topk). Block-level syncs only --
    // NO cluster ops, so it is safe to run on one rank while the others wait.
    // NOT force-inlined (INDEXER_TOPK_HIST_DEVICE): keep this off the hot Register4/Streaming
    // register budget so the overflow fallback can't spill the fast paths.
    __noinline__ __device__ static void singleBlockExactTopK(
        float const* __restrict__ in, uint32_t n, uint32_t topk, int32_t* __restrict__ out, Smem* smem)
    {
        auto const tx = threadIdx.x;
        uint32_t klo = 0u, khi = 0xFFFFFFFFu; // active ordered-key range [klo, khi)
        uint32_t base = 0u, remaining = topk;

        for (uint32_t guard = 0; guard < 8u && remaining > 0u; ++guard)
        {
            uint64_t const range = static_cast<uint64_t>(khi) - klo;
            // 10-bit range-relative digit: s = max(0, ceil_log2(range) - 10) so at
            // most 1024 buckets, reusing the existing 1024-bucket machinery.
            uint32_t const cl = ceilLog2U64(range);
            uint32_t const s = (range > 1024ull && cl > 10u) ? cl - 10u : 0u;

            smem->histogram[tx] = 0u;
            __syncthreads();
            for_each_input(in, n,
                [&](float val, uint32_t)
                {
                    uint32_t const key = extract_exact_bin(val);
                    if (key >= klo && key < khi)
                    {
                        uint32_t const b = static_cast<uint32_t>((static_cast<uint64_t>(key) - klo) >> s);
                        atomicAdd(&smem->histogram[b], 1u);
                    }
                });
            __syncthreads();

            findThresholdFull(remaining, smem);
            uint32_t const b = smem->refine_bin;
            uint32_t const above_b = smem->refine_above;

            // Emit elements strictly above bucket b (guaranteed in the top-K).
            if (tx == 0)
                smem->emit_counter = 0u;
            __syncthreads();
            for_each_input(in, n,
                [&](float val, uint32_t idx)
                {
                    uint32_t const key = extract_exact_bin(val);
                    if (key >= klo && key < khi)
                    {
                        uint32_t const bb = static_cast<uint32_t>((static_cast<uint64_t>(key) - klo) >> s);
                        if (bb > b)
                        {
                            uint32_t const pos = base + atomicAdd(&smem->emit_counter, 1u);
                            if (pos < topk)
                                out[pos] = static_cast<int32_t>(idx);
                        }
                    }
                });
            __syncthreads();

            base += above_b;
            remaining -= above_b;
            if (remaining == 0u)
                break;

            // Narrow to bucket b.
            uint64_t const new_klo = static_cast<uint64_t>(klo) + (static_cast<uint64_t>(b) << s);
            uint64_t new_khi = new_klo + (1ull << s);
            if (new_khi > khi)
                new_khi = khi;
            klo = static_cast<uint32_t>(new_klo);
            khi = static_cast<uint32_t>(new_khi);

            if (s == 0u)
            {
                // Bucket b is exactly one key wide: the `remaining` still needed all
                // share this exact value (>= remaining of them exist by construction).
                emitExactKey(in, n, klo, base, remaining, topk, out, smem);
                remaining = 0u;
                break;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Register path: scores stay resident in registers across both passes.
// ---------------------------------------------------------------------------

template <uint32_t kLocalVecs_>
struct TopKRegister : TopKRadixBase<12>
{
    static constexpr uint32_t kLocalVecs = kLocalVecs_;
    static constexpr uint32_t kMaxSeqLen = kBlockSize * kVecSize * kLocalVecs;
    using Smem = typename TopKRadixBase<12>::Smem;

    template <bool kUsePDL>
    INDEXER_TOPK_HIST_DEVICE static void forward(TopKProblem const problem, void* _smem)
    {
        auto const tx = threadIdx.x;
        auto const smem = static_cast<Smem*>(_smem);

        {
            typename Smem::kHistVec hist_vec;
            hist_vec.fill(0);
            smem->hist_vecs[tx] = hist_vec;
        }
        if (tx == 0)
        {
            smem->count_eq = 0;
            smem->count_gt = 0;
        }

        __syncthreads();
        pdlWaitPrimary<kUsePDL>();

        uint32_t const num_full = problem.seq_len / kVecSize;
        uint32_t const tail_start = num_full * kVecSize;
        uint32_t const tail = problem.seq_len - tail_start;

        vec_t local_vecs[kLocalVecs];
#pragma unroll
        for (uint32_t i = 0; i < kLocalVecs; ++i)
        {
            auto const vi = tx + kBlockSize * i;
            if (vi >= num_full)
                break;
            local_vecs[i].load(problem.in, vi);
        }
#pragma unroll
        for (uint32_t i = 0; i < kLocalVecs; ++i)
        {
            auto const vi = tx + kBlockSize * i;
            if (vi >= num_full)
                break;
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
                atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(local_vecs[i][j])], 1);
        }
        if (tx >= kBlockSize - tail)
        {
            uint32_t const idx = tail_start + tx - (kBlockSize - tail);
            atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(problem.in[idx])], 1);
        }
        __syncthreads();

        find_threshold(problem.topk, problem.seq_len, smem);

        // Overflow fallback: if the coarse threshold bin holds more than the fixed tie
        // buffer keeps, the collect + handle_tie path below would truncate to an arbitrary
        // subset -> wrong top-K. Redo the row exactly (stock's fp16-bin-overflow -> fp32
        // restart). threshold_bin is in smem (post-syncthreads) -> block-uniform branch.
        uint32_t const tiePop = smem->histogram[smem->threshold_bin];
        __syncthreads(); // snapshot before the histogram/tie union is repurposed
        if (tiePop + 256u > kMaxNumTie)
        {
            singleBlockExactTopK(problem.in, problem.seq_len, problem.topk, problem.out, smem);
            return;
        }

        auto const topk = problem.topk;
        auto const threshold_bin = smem->threshold_bin;
        auto const v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
        auto const v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin);
        auto const collect = [&](float val, uint32_t idx)
        {
            if (val >= v_hi)
            {
                auto const pos = atomicAdd(&smem->count_gt, 1);
                if (pos < topk)
                    problem.emit(pos, idx);
            }
            else if (val >= v_lo)
            {
                auto const count_eq = atomicAdd(&smem->count_eq, 1);
                if (count_eq < kMaxNumTie)
                    smem->tie.values[count_eq] = {val, idx};
            }
        };
#pragma unroll
        for (uint32_t i = 0; i < kLocalVecs; ++i)
        {
            auto const vi = tx + kBlockSize * i;
            auto const base = vi * kVecSize;
            if (vi >= num_full)
                break;
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
                collect(local_vecs[i][j], base + j);
        }
        if (tx >= kBlockSize - tail)
        {
            uint32_t const idx = tail_start + tx - (kBlockSize - tail);
            collect(problem.in[idx], idx);
        }

        __syncthreads();
        auto const above_count = smem->count_gt;
        auto const equal_count = smem->count_eq;
        // Exact-count backstop: the coarse guard can under-count the fp32-
        // classified sets at exact fp16 midpoints. count_gt/count_eq are the EXACT
        // uncapped counts -> if the tie buffer truncated (equal_count > kMaxNumTie)
        // or count_gt overshot topK, redo the row exactly (any preliminary output is
        // overwritten). Uniform branch: both counts are final and untouched by the fallback.
        if (equal_count > kMaxNumTie || above_count > topk)
        {
            singleBlockExactTopK(problem.in, problem.seq_len, problem.topk, problem.out, smem);
            return;
        }
        auto const remain_topk = above_count < topk ? topk - above_count : 0;
        auto const tie_count = min(equal_count, kMaxNumTie);
        handle_tie(smem->tie.values, problem, above_count, tie_count, remain_topk, &smem->tie.handle);
    }
};

// ---------------------------------------------------------------------------
// Streaming path: seq_len > 8192 -- two vectorized passes over global memory
// ---------------------------------------------------------------------------

struct TopKStreaming : TopKRegister<2>
{
    static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();

    template <bool kUsePDL>
    INDEXER_TOPK_HIST_DEVICE static void forward(TopKProblem const problem, void* _smem)
    {
        auto const tx = threadIdx.x;
        auto const smem = static_cast<Smem*>(_smem);

        {
            typename Smem::kHistVec hist_vec;
            hist_vec.fill(0);
            smem->hist_vecs[tx] = hist_vec;
        }
        if (tx == 0)
        {
            smem->count_eq = 0;
            smem->count_gt = 0;
        }
        __syncthreads();
        pdlWaitPrimary<kUsePDL>();

        for_each_input(problem.in, problem.seq_len,
            [&](float val, uint32_t)
            {
                auto const bin = extract_coarse_bin<kHistBits>(val);
                atomicAdd(&smem->histogram[bin], 1);
            });
        __syncthreads();

        find_threshold(problem.topk, problem.seq_len, smem);

        // Overflow fallback: see Register path -- the coarse threshold bin can
        // exceed the fixed tie buffer -> truncation -> wrong top-K; redo the row exactly.
        uint32_t const tiePop = smem->histogram[smem->threshold_bin];
        __syncthreads(); // snapshot before the histogram/tie union is repurposed
        if (tiePop + 256u > kMaxNumTie)
        {
            singleBlockExactTopK(problem.in, problem.seq_len, problem.topk, problem.out, smem);
            return;
        }

        auto const threshold_bin = smem->threshold_bin;
        float const v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
        float const v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin);
        auto const topk = problem.topk;
        for_each_input(problem.in, problem.seq_len,
            [&](float val, uint32_t idx)
            {
                if (val >= v_hi)
                {
                    auto const pos = atomicAdd(&smem->count_gt, 1);
                    if (pos < topk)
                    {
                        problem.emit(pos, idx);
                    }
                }
                else if (val >= v_lo)
                {
                    auto const count_eq = atomicAdd(&smem->count_eq, 1);
                    if (count_eq < kMaxNumTie)
                    {
                        smem->tie.values[count_eq] = {val, idx};
                    }
                }
            });

        __syncthreads();
        auto const above_count = smem->count_gt;
        auto const equal_count = smem->count_eq;
        // Exact-count backstop: the coarse guard can under-count the fp32-
        // classified sets at exact fp16 midpoints. count_gt/count_eq are the EXACT
        // uncapped counts -> if the tie buffer truncated (equal_count > kMaxNumTie)
        // or count_gt overshot topK, redo the row exactly (any preliminary output is
        // overwritten). Uniform branch: both counts are final and untouched by the fallback.
        if (equal_count > kMaxNumTie || above_count > topk)
        {
            singleBlockExactTopK(problem.in, problem.seq_len, problem.topk, problem.out, smem);
            return;
        }
        auto const remain_topk = above_count < topk ? topk - above_count : 0;
        auto const tie_count = min(equal_count, kMaxNumTie);
        handle_tie(smem->tie.values, problem, above_count, tie_count, remain_topk, &smem->tie.handle);
    }
};

// ---------------------------------------------------------------------------
// Cluster path: very long seq_len, small batch. kClusterSize blocks cooperate
// on one row via distributed shared memory (one cluster per row).
// ---------------------------------------------------------------------------

template <uint32_t kClusterSize_>
struct TopKCluster : TopKRadixBase<10>
{
    static constexpr uint32_t kClusterSize = kClusterSize_;
    static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();
    using Base = TopKRadixBase<10>;

    struct Smem : Base::Smem
    {
        using kHistVec = Base::Smem::kHistVec;
        uint32_t start_eq_local, start_gt_local;
        int32_t tmp_out[kMaxTopK];
    };

    // Process ONE row (one cluster). All ranks share the SAME global output row
    // pointer (problem.out), so the raw indices land directly in global memory --
    // no shared-memory staging + page-table transform pass (v1 = selection only).
    template <bool kUsePDL>
    INDEXER_TOPK_HIST_DEVICE static void forward(TopKProblem problem, void* _smem)
    {
        auto const tx = threadIdx.x;
        auto const smem = static_cast<Smem*>(_smem);
        auto const cluster = cg::this_cluster();
        auto const this_rank = blockIdx.y;
        bool const is_primary = (this_rank == 0);

        constexpr uint32_t kAlignElems = kWarpSize * kVecSize;
        uint32_t const chunk_size = div_ceil(problem.seq_len, kClusterSize * kAlignElems) * kAlignElems;
        uint32_t const chunk_start = min(this_rank * chunk_size, problem.seq_len);
        uint32_t const chunk_finish = min(chunk_start + chunk_size, problem.seq_len);
        uint32_t const local_seq_len = chunk_finish - chunk_start;
        problem.in += chunk_start;

        {
            typename Smem::kHistVec hist_vec;
            hist_vec.fill(0);
            smem->hist_vecs[tx] = hist_vec;
        }
        if (tx == 0)
        {
            smem->count_eq = 0;
            smem->count_gt = 0;
        }
        __syncthreads();
        pdlWaitPrimary<kUsePDL>();

        for_each_input(problem.in, local_seq_len,
            [&](float val, uint32_t)
            {
                auto const bin = extract_coarse_bin<kHistBits>(val);
                atomicAdd(&smem->histogram[bin], 1);
            });
        __syncthreads();

        // Phase 1.5: 1-shot cross-cluster histogram all-reduce via DSMEM.
        {
            cluster.sync();
            static_assert(kHistSize == kBlockSize);
            constexpr uint32_t kPartition = kHistSize / kClusterSize;
            auto const start = this_rank * kPartition;
            auto const which = start + tx / kClusterSize;
            auto const peer_rank = tx % kClusterSize;
            auto const addr = cluster.map_shared_rank(&smem->histogram[which], peer_rank);
            auto const value = *addr;
            *addr = warpReduceSum<kClusterSize>(value);
            cluster.sync();
        }

        find_threshold(problem.topk, problem.seq_len, smem);

        // Long-context overflow guard. histogram[threshold_bin] is the coarse bin's exact
        // population (DSMEM-summed across all 8 ranks). If it approaches the fixed tie-buffer
        // capacity the fast path below would truncate -> wrong top-K, so on overflow the
        // elected primary redoes the row with an exact-key radix and the others return.
        // (The guard can under-detect at exact fp16 midpoints -- measure-zero for real logits;
        // the single-block tiers add an exact-count backstop, the cluster one is a follow-up.)
        uint32_t const tiePop = smem->histogram[smem->threshold_bin];
        // Cluster-wide barrier (not __syncthreads): tie.values aliases the histogram union, and
        // peer ranks scatter their tie merges into rank 0's union remotely with no other barrier
        // since the all-reduce. Fencing all ranks past this read keeps the overflow guard
        // cluster-uniform, so no rank takes a divergent early return -> no cluster.sync deadlock.
        cluster.sync();
        if (tiePop + 256u > kMaxNumTie)
        {
            if (is_primary)
                singleBlockExactTopK(problem.in - chunk_start, problem.seq_len, problem.topk, problem.out, smem);
            return;
        }

        auto const topk = problem.topk;
        auto const threshold_bin = smem->threshold_bin;
        float const v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
        float const v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin);
        auto const cur_out = is_primary ? problem.out : smem->tmp_out;
        for_each_input(problem.in, local_seq_len,
            [&](float val, uint32_t local_idx)
            {
                auto const idx = chunk_start + local_idx;
                if (val >= v_hi)
                {
                    auto const pos = atomicAdd(&smem->count_gt, 1);
                    if (pos < topk)
                    {
                        cur_out[pos] = idx;
                    }
                }
                else if (val >= v_lo)
                {
                    auto const count_eq = atomicAdd(&smem->count_eq, 1);
                    if (count_eq < kMaxNumTie)
                    {
                        smem->tie.values[count_eq] = {val, idx};
                    }
                }
            });

        uint32_t start_write = 0;
        uint32_t num_write = 0;
        if (!is_primary)
        {
            __syncthreads();
            auto const local_above_count = smem->count_gt;
            auto const local_equal_count = min(smem->count_eq, kMaxNumTie);
            auto const smem_0 = cluster.map_shared_rank(smem, 0);
            if (tx == 0)
            {
                auto const gt = atomicAdd(&smem_0->count_gt, local_above_count);
                auto const eq = atomicAdd(&smem_0->count_eq, local_equal_count);
                smem->start_gt_local = gt;
                smem->start_eq_local = eq;
            }
            __syncthreads();
            auto const start_gt_local = smem->start_gt_local;
            auto const start_eq_local = smem->start_eq_local;
#pragma unroll
            for (uint32_t i = 0; i < kTieItems; ++i)
            {
                auto const t = tx + i * kBlockSize;
                if (t < local_equal_count && start_eq_local + t < kMaxNumTie)
                {
                    smem_0->tie.values[start_eq_local + t] = smem->tie.values[t];
                }
            }
            start_write = start_gt_local;
            num_write = local_above_count;
        }

        cluster.sync();
        if (!is_primary)
        {
#pragma unroll
            for (uint32_t i = 0; i < kTopKItems; ++i)
            {
                if (auto const t = tx + i * kBlockSize; t < num_write && start_write + t < topk)
                {
                    problem.emit(start_write + t, smem->tmp_out[t]);
                }
            }
        }
        else
        {
            auto const above_count = smem->count_gt;
            auto const equal_count = smem->count_eq;
            auto const remain_topk = above_count < topk ? topk - above_count : 0;
            auto const tie_count = min(equal_count, kMaxNumTie);
            handle_tie(smem->tie.values, problem, above_count, tie_count, remain_topk, &smem->tie.handle);
        }
    }
};

// ---------------------------------------------------------------------------
// Host-side constants + the fused selection kernel (ports topk_small_batch_kernel)
// ---------------------------------------------------------------------------

using Register4 = TopKRegister<4>; // <= 16384
using Streaming = TopKStreaming;
using Cluster = TopKCluster<8>;

constexpr uint32_t kBlockSize = TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = TopKConfig::kOccupancy;
constexpr uint32_t kClusterSize = Cluster::kClusterSize;
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen; // 16384

// Cluster-floor selection (topk_v2.cuh TopKKernel::transform L410-423): batch
// <= 15 stays latency-bound so the 8-way split beats streaming from a lower seq.
constexpr uint32_t kClusterFloor = 65536;
constexpr uint32_t kClusterFloorSmall = 32768;
constexpr uint32_t kSmallBatchLowFloor = 15;
// One 8-block cluster per row; bound how many rows the small-batch launch maps.
constexpr uint32_t kClusterMaxBatch = 512;

struct SelectParams
{
    float const* __restrict__ logits;
    int const* __restrict__ seqLens;
    int32_t* __restrict__ outIndices;
    int64_t stride0;
    uint32_t topK;
    uint32_t next_n;
    uint32_t compressRatio;
    uint32_t clusterFloor;
};

// Per-row length, matching topKPerRowDecode exactly (indexerTopK.cu L667-669),
// with the requested safety clamp of padded/negative rows to 0.
INDEXER_TOPK_HIST_DEVICE uint32_t computeRowEnd(SelectParams const& p, uint32_t rowIdx)
{
    // seqLens read as uint32 (matching the upstream convention); a padded/garbage negative
    // length would make actual_kv_len negative -> clamp rowEnd to 0.
    int const seq_len = p.seqLens[rowIdx / p.next_n];
    int const actual_kv_len = seq_len - static_cast<int>(p.next_n) + static_cast<int>(rowIdx % p.next_n) + 1;
    int const rowEnd = actual_kv_len > 0 ? actual_kv_len / static_cast<int>(p.compressRatio) : 0;
    return static_cast<uint32_t>(rowEnd < 0 ? 0 : rowEnd);
}

// The cluster size is supplied at launch via cudaLaunchAttributeClusterDimension
// (see mnnvlAllreduceKernels.cu), NOT the compile-time __cluster_dims__ attribute
// -- that keeps this TU compilable for sm<90 targets in the multi-arch build,
// where the cluster branch below is #if'd out and long rows fall to Streaming.
// __grid_constant__ const T is the repo-standard param form (see flashMLA).
template <bool kPDL>
__global__ __launch_bounds__(kBlockSize, kOccupancy) void topkSelectSmallBatchKernel(
    __grid_constant__ SelectParams const params)
{
    enableSmemSpilling();
    // The PDL wait must precede the seqLens read: computeRowEnd() loads seqLens, which the
    // overlap scheduler mutates in place each step. Reading it before griddepcontrol.wait
    // races that write, so the 8 blocks of a row's cluster could see different rowEnd, diverge
    // on tier/overflow branches, and mismatch their cluster.sync() counts -> deadlock. (The
    // stock kernel likewise waits before this read; the per-tier waits below become no-ops.)
    pdlWaitPrimary<kPDL>();
    uint32_t const rowIdx = blockIdx.x;
    uint32_t const rowEnd = computeRowEnd(params, rowIdx);

    TopKProblem problem;
    problem.in = params.logits + static_cast<int64_t>(rowIdx) * params.stride0;
    problem.out = params.outIndices + static_cast<int64_t>(rowIdx) * params.topK;
    problem.topk = params.topK;
    problem.seq_len = rowEnd;

    __shared__ MaxSmem<Streaming::Smem, Cluster::Smem> smem;

    // Randomly elect one worker rank per row to balance the single-block paths.
    auto const worker_rank = rowIdx % kClusterSize;

    if (rowEnd <= problem.topk)
    {
        // Trivial: identity local indices then -1 padding (matches the
        // rowLen <= topK short-row branch of topKPerRowJob). One rank writes.
        pdlWaitPrimary<kPDL>();
        if (blockIdx.y == worker_rank)
        {
            for (uint32_t t = threadIdx.x; t < problem.topk; t += kBlockSize)
            {
                problem.out[t] = (t < rowEnd) ? static_cast<int32_t>(t) : -1;
            }
        }
    }
    else if (rowEnd <= kReg4MaxSeqLen)
    {
        if (blockIdx.y == worker_rank)
            Register4::forward<kPDL>(problem, &smem);
    }
    else if (rowEnd <= params.clusterFloor)
    {
        if (blockIdx.y == worker_rank)
            Streaming::forward<kPDL>(problem, &smem);
    }
    else
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        // All 8 ranks cooperate; rowEnd depends only on blockIdx.x so every rank
        // of this cluster takes this same branch (required for cluster.sync()).
        Cluster::forward<kPDL>(problem, &smem);
        cg::this_cluster().sync();
#else
        // sm<90: no distributed-shared-memory clusters. Fall back to the
        // single-block Streaming path (correct for any seq_len, just slower).
        // The env-gated fast path targets sm90+/sm103, so this is compile-only.
        if (blockIdx.y == worker_rank)
            Streaming::forward<kPDL>(problem, &smem);
#endif
    }

    __syncthreads();
    pdlTriggerSecondary<kPDL>();
}

} // namespace topk_hist

// ---------------------------------------------------------------------------
// Public launcher
// ---------------------------------------------------------------------------

bool indexerTopKDecodeHistSupported(int numRows, int topK, int stride1, int compressRatio)
{
    bool const topKOk = (topK == 512 || topK == 1024 || topK == 2048);
    bool const strideOk = (stride1 == 1);
    bool const compressOk = (compressRatio == 1 || compressRatio == 4);
    bool const rowsOk = (numRows > 0 && numRows <= static_cast<int>(topk_hist::kClusterMaxBatch));
    return topKOk && strideOk && compressOk && rowsOk;
}

void invokeIndexerTopKDecodeHist(float const* logits, int const* seqLens, int* outIndices, int numRows, int numColumns,
    int stride0, int next_n, int topK, int compressRatio, bool usePDL, cudaStream_t stream)
{
    using namespace topk_hist;
    (void) numColumns; // per-row length is recomputed on-device; kept for API fidelity.

    // The caller (invokeIndexerTopKDecode) gates on stride1 == 1 (unit inner
    // stride); the launcher assumes it and passes 1 to the support predicate.
    TLLM_CHECK_WITH_INFO(indexerTopKDecodeHistSupported(numRows, topK, /*stride1=*/1, compressRatio),
        "invokeIndexerTopKDecodeHist called with an unsupported shape");
    // Vectorized 16-byte row loads require the row stride to be a multiple of 4
    // floats (matches the upstream score_stride % 4 == 0 runtime check).
    TLLM_CHECK_WITH_INFO(stride0 % 4 == 0, "invokeIndexerTopKDecodeHist: stride0 must be a multiple of 4");
    TLLM_CHECK_WITH_INFO(next_n > 0, "invokeIndexerTopKDecodeHist: next_n must be > 0");

    SelectParams params;
    params.logits = logits;
    params.seqLens = seqLens;
    params.outIndices = outIndices;
    params.stride0 = static_cast<int64_t>(stride0);
    params.topK = static_cast<uint32_t>(topK);
    params.next_n = static_cast<uint32_t>(next_n);
    params.compressRatio = static_cast<uint32_t>(compressRatio);
    params.clusterFloor = (numRows <= static_cast<int>(kSmallBatchLowFloor)) ? kClusterFloorSmall : kClusterFloor;

    cudaLaunchConfig_t config;
    config.gridDim = dim3(static_cast<unsigned>(numRows), kClusterSize, 1);
    config.blockDim = dim3(kBlockSize, 1, 1);
    config.dynamicSmemBytes = 0; // Smem is static (MaxSmem<...>).
    config.stream = stream;

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = kClusterSize;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = usePDL ? 1 : 0;
    config.numAttrs = 2;
    config.attrs = attrs;

    if (usePDL)
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, topkSelectSmallBatchKernel<true>, params));
    }
    else
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, topkSelectSmallBatchKernel<false>, params));
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
