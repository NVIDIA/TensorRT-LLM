/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
 */

#pragma once

#include "sm120_utils.cuh"

using namespace cute;

namespace sm120_blockscaled_gemm
{

// Find max element within 8-lane group (for 128-element quantization block)
__device__ __forceinline__ float find_max_elem_in_8_lanes(float value)
{
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFF, value, 4, 8));
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFF, value, 2, 8));
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFF, value, 1, 8));
    return value;
}

// Compute reciprocal of 2^(exp-127) for UE8M0 scale
__device__ __forceinline__ float exp2f_rcp(uint8_t exp)
{
    constexpr uint32_t FP32_EXPONENT_BIAS = 127;
    return (exp == 0) ? 1.0f : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(exp));
}

inline __device__ float reciprocal_approximate_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

template <typename InputType, typename OutputType, int WarpsPerBlock = 4>
__global__ void scale_1x128_kernel_sm120(OutputType* __restrict__ fp8_output, int32_t* __restrict__ scale_output,
    InputType const* __restrict__ input, int64_t const* __restrict__ token_offset, int64_t num_experts, int64_t size_k,
    int64_t scale_leading_dim)
{

    extern __shared__ char shared_memory[];
    int64_t* smem_token_offset = reinterpret_cast<int64_t*>(shared_memory);

    // Load token_offset into shared memory
    for (int i = threadIdx.x; i <= num_experts; i += blockDim.x)
    {
        smem_token_offset[i] = token_offset[i];
    }
    __syncthreads();

    // Get actual token_num from token_offset[num_experts]
    const int64_t token_num = smem_token_offset[num_experts];

    int const warp_id = threadIdx.x >> 5;
    int const lane_id = threadIdx.x & 31;

    const int64_t k_block_idx = blockIdx.x;
    const int64_t grid_stride = static_cast<int64_t>(gridDim.y) * WarpsPerBlock;

    for (int64_t token_idx = static_cast<int64_t>(blockIdx.y) * WarpsPerBlock + warp_id; token_idx < token_num;
         token_idx += grid_stride)
    {

        // Binary search to find expert_idx: token_offset[expert_idx] <= token_idx < token_offset[expert_idx + 1]
        int64_t expert_idx = 0;
        {
            int left = 0;
            int right = num_experts - 1;
            while (left < right)
            {
                int mid = (left + right + 1) >> 1;
                if (smem_token_offset[mid] <= token_idx)
                {
                    left = mid;
                }
                else
                {
                    right = mid - 1;
                }
            }
            expert_idx = left;
        }

        // Local token index within this expert
        const int64_t local_token_idx = token_idx - smem_token_offset[expert_idx];

        // Check if this thread's data is within k bounds
        int const k_offset = (k_block_idx * 512 + lane_id * 16);

        // 1. Load 16 BF16 elements per thread (512 per warp)
        auto const cur_input_ptr = reinterpret_cast<double4 const*>(input + token_idx * size_k + k_offset);

        constexpr int kLoadNumElems = sizeof(double4) / sizeof(InputType); // 16 for BF16

        union LoadTrick
        {
            double4 pack;
            InputType v[kLoadNumElems];
        };

        LoadTrick load_trick;

        // Conditional load: zero-fill if out of bounds
        load_trick.pack = k_offset < size_k ? cur_input_ptr[0] : double4{};

        // 2.1 Find max abs element in 16 elements per thread
        InputType max_elem = InputType(0.0f);
#pragma unroll
        for (int i = 0; i < kLoadNumElems; i++)
        {
            max_elem = __hmax(max_elem, __habs(load_trick.v[i]));
        }

        // 2.2 Find max in 8-lane group (128 elements = 1 quantization block)
        float amax = float(max_elem);
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 4, 8));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 2, 8));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1, 8));
        amax = fmaxf(amax, 1e-10f);

        // 3. Compute UE8M0 scale and quant_scale
        float dequant_scale_raw = amax * reciprocal_approximate_ftz(448.0f);
        __nv_fp8_e8m0 ue8m0_scale;
        ue8m0_scale.__x = __nv_cvt_float_to_e8m0(dequant_scale_raw, __NV_SATFINITE, cudaRoundPosInf);
        float quant_scale = exp2f_rcp(ue8m0_scale.__x);

        // 4.1 Quantize and store FP8 output
        constexpr int kStoreNumElems = sizeof(float4) / sizeof(OutputType); // 16 for FP8

        union StoreTrick
        {
            float4 pack;
            OutputType v[kStoreNumElems];
        };

        StoreTrick store_trick;
        store_trick.pack = float4{};

#pragma unroll
        for (int i = 0; i < kStoreNumElems; i++)
        {
            store_trick.v[i] = OutputType(float(load_trick.v[i]) * quant_scale);
        }

        auto cur_output_ptr = reinterpret_cast<float4*>(fp8_output + token_idx * size_k + k_offset);

        if (k_offset < size_k)
        {
            cur_output_ptr[0] = store_trick.pack;
        }

        // 4.2 Pack scales from lane 0, 8, 16, 24 and store
        uint32_t s0 = __shfl_sync(0xFFFFFFFF, (uint32_t) ue8m0_scale.__x, 0);
        uint32_t s1 = __shfl_sync(0xFFFFFFFF, (uint32_t) ue8m0_scale.__x, 8);
        uint32_t s2 = __shfl_sync(0xFFFFFFFF, (uint32_t) ue8m0_scale.__x, 16);
        uint32_t s3 = __shfl_sync(0xFFFFFFFF, (uint32_t) ue8m0_scale.__x, 24);

        if (lane_id == 0)
        {
            uint32_t packed_scale = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);

            const int64_t scale_padded_offset
                = compute_padded_offset(static_cast<int64_t>(smem_token_offset[expert_idx]), expert_idx);

            auto cur_scale_ptr = scale_output + k_block_idx * scale_leading_dim + scale_padded_offset;
            *reinterpret_cast<uint32_t*>(&cur_scale_ptr[local_token_idx]) = packed_scale;
        }
    }
}

// ============================================================================
// MOE Scheduler: inspired by deep_gemm's GroupedWithOffsetScheduler
// Stores per-expert element-level offsets, provides coordinate getters.
// No alignment constraint on token_offset — uses domain_offset for
// element-level TMA addressing.
// ============================================================================
template <int BlockM, int BlockN>
struct SM120BlockScaledMoeScheduler
{

    int32_t current_iter = -1;
    int32_t curr_group_idx = 0; // current expert index
    int32_t cur_m_cumsum = 0;   // cumulative M blocks across processed experts
    int32_t num_m_blocks = 0;   // M blocks for current expert
    int32_t num_n_blocks = 0;
    int32_t num_groups = 0;     // num_experts
    int64_t m_offset = 0;       // token_offset[expert] — actual row in A
    int64_t m_boundary = 0;     // token_offset[expert + 1]
    int64_t* token_offset = nullptr;

    __device__ __forceinline__ explicit SM120BlockScaledMoeScheduler(
        int shape_n, int num_groups_, int64_t* token_offset_)
        : num_n_blocks((shape_n + BlockN - 1) / BlockN)
        , num_groups(num_groups_)
        , token_offset(token_offset_)
    {
    }

    __device__ __forceinline__ bool get_next_block(int32_t& m_block_idx, int32_t& n_block_idx)
    {
        auto next_block_idx = ++current_iter * gridDim.x + blockIdx.x;
        while (true)
        {
            if (curr_group_idx >= num_groups)
                return false;

            m_offset = token_offset[curr_group_idx];
            m_boundary = token_offset[curr_group_idx + 1];
            auto shape_m = m_boundary - m_offset;
            num_m_blocks = (shape_m + BlockM - 1) / BlockM;

            auto next_m_cumsum = cur_m_cumsum + num_m_blocks;
            if (next_block_idx < next_m_cumsum * num_n_blocks)
                break;

            ++curr_group_idx;
            cur_m_cumsum = next_m_cumsum;
        }

        auto remain_blocks = next_block_idx - cur_m_cumsum * num_n_blocks;
        m_block_idx = remain_blocks / num_n_blocks;
        n_block_idx = remain_blocks % num_n_blocks;
        return true;
    }

    __device__ __forceinline__ int32_t get_m_offset() const
    {
        return m_offset;
    }

    __device__ __forceinline__ int32_t get_m_boundary() const
    {
        return m_boundary;
    }

    __device__ __forceinline__ int32_t get_expert_idx() const
    {
        return curr_group_idx;
    }
};

// ============================================================================
// MOE Kernel: Independent kernel for MOE GEMM, fully decoupled from
// SM120BlockScaledKernel. Reuses SM120BlockScaledBuilder type definitions.
//
// Data layout:
//   A:   [M, K]              — 2D, TMA L=1
//   B:   [num_experts, N, K]            — 3D, TMA L=num_experts
//   SFA: [sf_K, align(M,4)]  — 2D, TMA L=1
//   SFB: [num_experts, align(N,4), sf_K]— 3D, TMA L=num_experts
//   D:   [M, N]              — 2D, TMA L=1
// ============================================================================
template <typename KT>
struct SM120BlockScaledMoeKernel
{

    static constexpr int kNumTMAThreads = 128;
    static constexpr int kNumMathThreads = KT::kNumMathThreads;
    static constexpr int MaxThreadsPerBlock = kNumTMAThreads + kNumMathThreads;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    using Scheduler = SM120BlockScaledMoeScheduler<KT::kTileM, KT::kTileN>;
    using ProblemShape = typename KT::ProblemShape;

    struct Params
    {
        typename KT::TMA_A tma_load_a;
        typename KT::TMA_B tma_load_b;
        typename KT::TMA_SFA tma_load_sfa;
        typename KT::TMA_SFB tma_load_sfb;
        typename KT::ElementD* ptr_D;
        int M;
        int N;
        int K;
        int num_experts;
        int64_t* token_offset; // device ptr, size = num_experts + 1
    };

    struct Arguments
    {
        typename KT::ElementA* ptr_A;
        typename KT::StrideA dA;
        typename KT::ElementB* ptr_B;
        typename KT::StrideB dB;
        typename KT::ElementSFLoad* ptr_SFA;
        typename KT::StrideSFA dSFA;
        typename KT::ElementSFLoad* ptr_SFB;
        typename KT::StrideSFB dSFB;
        typename KT::ElementD* ptr_D;
        typename KT::StrideD dD;
        int64_t* token_offset;
    };

    static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args)
    {
        auto [M, N, K, num_experts] = problem_shape;

        // A TMA: (M, K, 1) — A is 2D with 1 as L
        auto tensor_A = make_tensor(make_gmem_ptr(args.ptr_A), make_layout(make_shape(M, K, 1), args.dA));
        typename KT::TMA_A tma_load_a
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_A, typename KT::SmemLayoutA{}(_, _, Int<0>{}),
                make_shape(shape<0>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});

        // B TMA: (N, K, num_experts) — B is 3D with experts as batch
        auto tensor_B = make_tensor(make_gmem_ptr(args.ptr_B), make_layout(make_shape(N, K, num_experts), args.dB));
        typename KT::TMA_B tma_load_b
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_B, typename KT::SmemLayoutB{}(_, _, Int<0>{}),
                make_shape(shape<1>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});

        // SFA TMA: uses m_padded to match quantization kernel's scale storage layout
        int m_padded = (M + num_experts * 3) / 4 * 4;
        auto sfa_shape = make_shape(m_padded, N, K, 1);
        auto sfa_layout = KT::deduce_sfa_layout(sfa_shape);
        auto tensor_sfa = make_tensor(make_gmem_ptr(args.ptr_SFA), sfa_layout);
        typename KT::TMA_SFA tma_load_sfa
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_sfa, typename KT::SmemLayoutSFA{}(_, _, Int<0>{}),
                make_shape(shape<0>(typename KT::ScaleTileShape{}), shape<2>(typename KT::ScaleTileShape{})), _1{});

        // SFB TMA: uses B-side problem shape (L=num_experts)
        auto sfb_shape = make_shape(M, N, K, num_experts);
        auto sfb_layout = KT::deduce_sfb_layout(sfb_shape);
        auto tensor_sfb = make_tensor(make_gmem_ptr(args.ptr_SFB), sfb_layout);
        typename KT::TMA_SFB tma_load_sfb
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_sfb, typename KT::SmemLayoutSFB{}(_, _, Int<0>{}),
                make_shape(shape<1>(typename KT::ScaleTileShape{}), shape<2>(typename KT::ScaleTileShape{})), _1{});

        return {
            tma_load_a, tma_load_b, tma_load_sfa, tma_load_sfb, args.ptr_D, M, N, K, num_experts, args.token_offset};
    }

    static dim3 get_grid_shape(Params const& params)
    {
        int device;
        cudaGetDevice(&device);
        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        return dim3(sm_count, 1, 1);
    }

    static dim3 get_block_shape()
    {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    CUTE_DEVICE
    static void prefetch_tma_descriptors(Params const& params)
    {
        cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
    }

    using TensorStorage = typename KT::TensorStorageMoe;
    using BarrierStorage = typename KT::BarrierStorage;

    struct SharedStorage
    {
        TensorStorage tensors;
        alignas(16) BarrierStorage barriers;
    };

    static constexpr int kSmemSize = int(sizeof(SharedStorage));

    using FullBarrier = typename KT::FullBarrier;
    using EmptyBarrier = typename KT::EmptyBarrier;
    using ProducerBarrierType = typename FullBarrier::ValueType;
    using ConsumerBarrierType = typename EmptyBarrier::ValueType;

    CUTE_DEVICE
    static auto get_mbarriers(SharedStorage& shared_storage)
    {
        using FullBarrier = typename KT::FullBarrier;
        using EmptyBarrier = typename KT::EmptyBarrier;
        using ProducerBarrierType = typename FullBarrier::ValueType;
        using ConsumerBarrierType = typename EmptyBarrier::ValueType;
        auto* ab_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.ab_full_mbar[0]);
        auto* ab_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.ab_empty_mbar[0]);
        auto* sf_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.sf_full_mbar[0]);
        auto* sf_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.sf_empty_mbar[0]);
        auto* store_full_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.store_full_mbar[0]);
        auto* store_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.store_empty_mbar[0]);
        return cute::make_tuple(
            ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar, store_empty_mbar);
    }

    CUTE_DEVICE
    static void load_sf(Params const& params, SharedStorage& shared_storage, int32_t m_offset, int32_t m_block_idx,
        int32_t n_block_idx, int32_t expert_idx, int32_t sf_tile_count, uint32_t& sf_phase, uint32_t& store_phase)
    {
        using X = Underscore;

        // SFA: shift origin to m_offset, then tile from there (no alignment needed)
        // Use m_padded instead of params.M to match quantization kernel's scale storage layout
        int32_t m_padded = (params.M + params.num_experts * 3) / 4 * 4;
        auto sfa_shape = make_shape(m_padded, params.N, params.K, 1);
        auto mSFA_full = params.tma_load_sfa.get_tma_tensor(shape(KT::deduce_sfa_layout(sfa_shape)));
        int32_t sf_m_offset = compute_padded_offset(m_offset, expert_idx);
        auto mSFA_mkl = cute::domain_offset(make_coord(sf_m_offset, 0, 0), mSFA_full);

        // SFB: standard tile-level indexing with expert as L
        auto sfb_shape = make_shape(params.M, params.N, params.K, params.num_experts);
        auto mSFB_nkl = params.tma_load_sfb.get_tma_tensor(shape(KT::deduce_sfb_layout(sfb_shape)));

        auto gSFA_mkl = local_tile(mSFA_mkl, typename KT::ScaleTileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
        auto gSFB_nkl = local_tile(mSFB_nkl, typename KT::ScaleTileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

        auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
        auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

        // SFA: m_block_idx from shifted origin, l=0;  SFB: n_block_idx, l=expert_idx
        auto gSFA = gSFA_mkl(_, _, m_block_idx, _, 0);
        auto gSFB = gSFB_nkl(_, _, n_block_idx, _, expert_idx);

        auto tAgSFA = block_tma_sfa.partition_S(gSFA);
        auto tBgSFB = block_tma_sfb.partition_S(gSFB);

        auto sSFA_
            = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()), typename KT::SmemLayoutSFA{});
        auto sSFB_
            = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()), typename KT::SmemLayoutSFB{});
        auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
        auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

        auto tAsSFA = block_tma_sfa.partition_D(sSFA);
        auto tBsSFB = block_tma_sfb.partition_D(sSFB);

        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        store_empty_mbar[0].wait(store_phase);
        store_phase ^= 1;

        for (int32_t sf_tile_idx = 0; sf_tile_idx < sf_tile_count; ++sf_tile_idx)
        {
            sf_empty_mbar[0].wait(sf_phase);
            auto& sf_full_barrier = sf_full_mbar[0];
            auto tma_copy_sfa = params.tma_load_sfa.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
            cute::copy(tma_copy_sfa, tAgSFA(_, _, _, sf_tile_idx), tAsSFA(_, _, _, Int<0>{}));
            auto tma_copy_sfb = params.tma_load_sfb.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
            cute::copy(tma_copy_sfb, tBgSFB(_, _, _, sf_tile_idx), tBsSFB(_, _, _, Int<0>{}));
            sf_full_mbar[0].arrive_and_expect_tx(KT::TmaSFTransactionBytes);
            sf_phase ^= 1;
        }
    }

    CUTE_DEVICE
    static void load_ab(Params const& params, SharedStorage& shared_storage, int32_t m_offset, int32_t m_block_idx,
        int32_t n_block_idx, int32_t expert_idx, int32_t sf_tile_count, uint32_t& ab_phase, uint32_t& store_phase)
    {
        using X = Underscore;

        // A: shift origin to m_offset, then tile from there (no alignment needed)
        auto mA_full = params.tma_load_a.get_tma_tensor(make_shape(params.M, params.K, 1));
        auto mA_mkl = cute::domain_offset(make_coord(m_offset, 0, 0), mA_full);

        // B: standard tile-level indexing with expert as L
        auto mB_nkl = params.tma_load_b.get_tma_tensor(make_shape(params.N, params.K, params.num_experts));

        auto gA_mkl = local_tile(mA_mkl, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
        auto gB_nkl = local_tile(mB_nkl, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

        auto block_tma_a = params.tma_load_a.get_slice(0);
        auto block_tma_b = params.tma_load_b.get_slice(0);

        // A: m_block_idx from shifted origin, l=0;  B: n_block_idx, l=expert_idx
        auto gA = gA_mkl(_, _, m_block_idx, _, 0);
        auto gB = gB_nkl(_, _, n_block_idx, _, expert_idx);

        auto tAgA = block_tma_a.partition_S(gA);
        auto tBgB = block_tma_b.partition_S(gB);

        auto sA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()), typename KT::SmemLayoutA{});
        auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()), typename KT::SmemLayoutB{});
        auto sA = as_position_independent_swizzle_tensor(sA_);
        auto sB = as_position_independent_swizzle_tensor(sB_);

        auto tAsA = block_tma_a.partition_D(sA);
        auto tBsB = block_tma_b.partition_D(sB);

        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        store_empty_mbar[0].wait(store_phase);
        store_phase ^= 1;

        int32_t k_tile_count = sf_tile_count * KT::kNumTileKPerSF;
        for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; k_tile_idx += KT::AB_Stages)
        {
            cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                [&](auto write_stage)
                {
                    ab_empty_mbar[write_stage].wait(ab_phase);
                    auto& ab_full_barrier = ab_full_mbar[write_stage];
                    auto tma_copy_a = params.tma_load_a.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
                    cute::copy(tma_copy_a, tAgA(_, _, _, k_tile_idx + write_stage), tAsA(_, _, _, write_stage));
                    auto tma_copy_b = params.tma_load_b.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
                    cute::copy(tma_copy_b, tBgB(_, _, _, k_tile_idx + write_stage), tBsB(_, _, _, write_stage));
                    ab_full_mbar[write_stage].arrive_and_expect_tx(KT::TmaABTransactionBytes);
                });
            ab_phase ^= 1;
        }
    }

    CUTE_DEVICE
    static void mma(Params const& params, SharedStorage& shared_storage, int32_t sf_tile_count, int32_t m_offset,
        int32_t m_boundary, int32_t m_block_idx, int32_t n_block_idx, uint32_t& sf_phase, uint32_t& ab_phase)
    {
        int thread_idx = int(threadIdx.x);

        auto sA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()), typename KT::SmemLayoutA{});
        auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()), typename KT::SmemLayoutB{});
        auto sSFA_
            = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()), typename KT::SmemLayoutSFA{});
        auto sSFB_
            = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()), typename KT::SmemLayoutSFB{});
        auto sA = as_position_independent_swizzle_tensor(sA_);
        auto sB = as_position_independent_swizzle_tensor(sB_);
        auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
        auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

        typename KT::TiledMma mma;
        auto tile_shape_mnk = tile_shape(mma);
        auto thr_mma = mma.get_thread_slice(thread_idx);
        auto accum = partition_fragment_C(mma, cute::take<0, 2>(typename KT::TileShape{}));
        auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
        auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));

        // A smem -> reg
        auto s2r_copy_A = make_tiled_copy_A(typename KT::SmemCopyAtomA{}, mma);
        auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
        auto tXsA = s2r_thr_copy_A.partition_S(sA);
        auto tXrA = s2r_thr_copy_A.retile_D(tCrA);
        // B smem -> reg
        auto s2r_copy_B = make_tiled_copy_B(typename KT::SmemCopyAtomB{}, mma);
        auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
        auto tXsB = s2r_thr_copy_B.partition_S(sB);
        auto tXrB = s2r_thr_copy_B.retile_D(tCrB);

        // SFA smem -> reg
        auto s2r_copy_SFA = make_tiled_copy_impl(
            typename KT::SmemCopyAtomSF{}, KT::get_layoutSFA_TV(mma), make_shape(size<0>(tile_shape(mma)), _1{}));
        auto s2r_thr_copy_SFA = s2r_copy_SFA.get_thread_slice(thread_idx);
        auto tXsSFA = s2r_thr_copy_SFA.partition_S(sSFA);
        auto tCrSFA = KT::partition_fragment_SFA(sSFA(_, _, Int<0>{}), thr_mma);
        auto tXrSFA = s2r_thr_copy_SFA.retile_D(tCrSFA);
        auto tCrSFA_frg = KT::transform_fragment_for_qmma(tCrSFA);

        // SFB smem -> reg
        auto s2r_copy_SFB = make_tiled_copy_impl(
            typename KT::SmemCopyAtomSF{}, KT::get_layoutSFB_TV(mma), make_shape(size<1>(tile_shape(mma)), _1{}));
        auto s2r_thr_copy_SFB = s2r_copy_SFB.get_thread_slice(thread_idx);
        auto tXsSFB = s2r_thr_copy_SFB.partition_S(sSFB);
        auto tCrSFB = KT::partition_fragment_SFB(sSFB(_, _, Int<0>{}), thr_mma);
        auto tXrSFB = s2r_thr_copy_SFB.retile_D(tCrSFB);
        auto tCrSFB_frg = KT::transform_fragment_for_qmma(tCrSFB);

        cute::clear(accum);
        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);

        // Main MMA loop: sf_tile_count - 1 iterations
        for (int32_t sf_tile_idx = 0; sf_tile_idx < sf_tile_count - 1; ++sf_tile_idx)
        {
            sf_full_mbar[0].wait(sf_phase);
            cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, Int<0>{}), tXrSFA);
            cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, Int<0>{}), tXrSFB);
            sf_empty_mbar[0].arrive();

            cute::for_each(cute::make_int_sequence<KT::kNumStagePerSF>{},
                [&](auto iter)
                {
                    cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                        [&](auto read_stage)
                        {
                            ab_full_mbar[read_stage].wait(ab_phase);
                            cute::copy(s2r_copy_A, tXsA(_, _, _, read_stage), tXrA);
                            cute::copy(s2r_copy_B, tXsB(_, _, _, read_stage), tXrB);
                            ab_empty_mbar[read_stage].arrive();

                            auto tCrSFA_stage = tCrSFA_frg(_, _, _, iter * KT::AB_Stages + read_stage);
                            auto tCrSFB_stage = tCrSFB_frg(_, _, _, iter * KT::AB_Stages + read_stage);
                            cute::gemm(
                                mma, make_zip_tensor(tCrA, tCrSFA_stage), make_zip_tensor(tCrB, tCrSFB_stage), accum);
                        });
                    ab_phase ^= 1;
                });
            sf_phase ^= 1;
        }

        // Last SF tile iteration with sync barrier
        sf_full_mbar[0].wait(sf_phase);
        cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, Int<0>{}), tXrSFA);
        cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, Int<0>{}), tXrSFB);
        sf_empty_mbar[0].arrive();

        cute::for_each(cute::make_int_sequence<KT::kNumStagePerSF>{},
            [&](auto iter)
            {
                cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                    [&](auto read_stage)
                    {
                        ab_full_mbar[read_stage].wait(ab_phase);
                        cute::copy(s2r_copy_A, tXsA(_, _, _, read_stage), tXrA);
                        cute::copy(s2r_copy_B, tXsB(_, _, _, read_stage), tXrB);
                        ab_empty_mbar[read_stage].arrive();
                        if constexpr (iter == KT::kNumStagePerSF - 1 && read_stage == KT::AB_Stages - 1)
                        {
                            cutlass::arch::NamedBarrier::sync(KT::kNumMathThreads, 0);
                        }
                        auto tCrSFA_stage = tCrSFA_frg(_, _, _, iter * KT::AB_Stages + read_stage);
                        auto tCrSFB_stage = tCrSFB_frg(_, _, _, iter * KT::AB_Stages + read_stage);
                        cute::gemm(
                            mma, make_zip_tensor(tCrA, tCrSFA_stage), make_zip_tensor(tCrB, tCrSFB_stage), accum);
                    });
                ab_phase ^= 1;
            });
        sf_phase ^= 1;

        // Epilogue: convert accum to output type and write to smem
        auto accum_frg = recast<Array<typename KT::ElementAccum, 2>>(accum);
        auto epi = make_fragment_like<typename KT::ElementD>(accum);
        auto epi_frg = recast<Array<typename KT::ElementD, 2>>(epi);
        cutlass::NumericArrayConverter<typename KT::ElementD, typename KT::ElementAccum, 2> converter;
        cute::for_each(
            cute::make_int_sequence<cute::size(epi_frg)>{}, [&](auto i) { epi_frg(i) = converter(accum_frg(i)); });

        auto sD_ = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.tensors.store.smem_O.begin()), typename KT::SmemLayoutO{});
        auto sD = as_position_independent_swizzle_tensor(sD_);
        // copy rf -> smem
        auto tiled_copy_R2S = cute::make_tiled_copy_C(typename KT::SmemCopyAtomR2S{}, mma);
        auto thr_copy_R2S = tiled_copy_R2S.get_slice(thread_idx);
        auto tRS_rD = thr_copy_R2S.retile_S(epi);
        auto tRS_sD = thr_copy_R2S.partition_D(sD);
        cute::copy(tiled_copy_R2S, tRS_rD, tRS_sD);
        cutlass::arch::NamedBarrier::sync(KT::kNumMathThreads, 0);

        // copy smem -> rf
        typename KT::TiledCopyS2G tiled_copy_S2G;
        auto thr_copy_S2G = tiled_copy_S2G.get_slice(thread_idx);
        auto tSR_sD = thr_copy_S2G.partition_S(sD);
        auto tSR_rD = cute::make_tensor<typename KT::ElementD>(cute::shape(tSR_sD));

        cute::copy(tiled_copy_S2G, tSR_sD, tSR_rD);
        cutlass::arch::NamedBarrier::sync(KT::kNumMathThreads, 0);
        store_empty_mbar[0].arrive();

        // copy rf -> gmem
        auto mD_full = cute::make_tensor(cute::make_gmem_ptr(params.ptr_D), cute::make_shape(params.M, params.N),
            cute::make_stride(params.N, cute::_1{}));
        auto mD_mn = cute::domain_offset(make_coord(m_offset, 0), mD_full);
        auto cta_coord = cute::make_coord(m_block_idx, n_block_idx);
        auto gD = cute::local_tile(mD_mn, make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kTileN>{}), cta_coord);
        auto cD = cute::make_identity_tensor(cute::make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kTileN>{}));
        auto tRG_rD = thr_copy_S2G.retile_S(tSR_rD);
        auto tRG_gD = thr_copy_S2G.partition_D(gD);
        auto tRG_cD = thr_copy_S2G.partition_D(cD);

        int residue_m = m_boundary - m_offset - KT::kTileM * m_block_idx;
        int residue_n = params.N - KT::kTileN * n_block_idx;
        CUTE_UNROLL
        for (int m = 0; m < cute::size<1>(tRG_gD); ++m)
        {
            CUTE_UNROLL
            for (int n = 0; n < cute::size<2>(tRG_gD); ++n)
            {
                if (cute::get<0>(tRG_cD(0, m, n)) < residue_m && cute::get<1>(tRG_cD(0, m, n)) < residue_n)
                {
                    cute::copy(typename KT::GmemCopyAtomR2G{}, tRG_rD(cute::_, m, n), tRG_gD(cute::_, m, n));
                }
            }
        }
    }

    CUTE_DEVICE
    void operator()(Params const& params, char* smem_buf)
    {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        int warp_idx = canonical_warp_idx_sync();
        int lane_predicate = cute::elect_one_sync();
        bool is_tma_thread = warp_idx == 0 && lane_predicate;

        if (is_tma_thread)
        {
            prefetch_tma_descriptors(params);
        }
        __syncthreads();

        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        // Init barriers
        if (is_tma_thread)
        {
#pragma unroll
            for (uint32_t i = 0; i < KT::SF_Stages; ++i)
            {
                sf_full_mbar[i].init(1);
                sf_empty_mbar[i].init(KT::kNumMathThreads);
            }
#pragma unroll
            for (uint32_t i = 0; i < KT::AB_Stages; ++i)
            {
                ab_full_mbar[i].init(1);
                ab_empty_mbar[i].init(KT::kNumMathThreads);
            }
            store_empty_mbar[0].init(KT::kNumMathThreads);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        int32_t sf_tile_count = (params.K + 511) / 512;

        if (warp_idx >= KT::kNumMathWarps)
        {
            constexpr int epi_warp_idx = KT::kNumMathWarps;
            constexpr int ab_warp_idx = epi_warp_idx + 1;
            constexpr int sf_warp_idx = ab_warp_idx + 1;

            if (warp_idx == ab_warp_idx)
            {
                uint32_t ab_phase = 1;
                uint32_t store_phase = 1;
                if (lane_predicate)
                {
                    Scheduler scheduler(params.N, params.num_experts, params.token_offset);
                    int32_t m_block_idx, n_block_idx;
                    while (scheduler.get_next_block(m_block_idx, n_block_idx))
                    {
                        load_ab(params, shared_storage, scheduler.get_m_offset(), m_block_idx, n_block_idx,
                            scheduler.get_expert_idx(), sf_tile_count, ab_phase, store_phase);
                    }
                }
                __syncwarp();
            }
            if (warp_idx == sf_warp_idx)
            {
                uint32_t sf_phase = 1;
                uint32_t store_phase = 1;
                if (lane_predicate)
                {
                    Scheduler scheduler(params.N, params.num_experts, params.token_offset);
                    int32_t m_block_idx, n_block_idx;
                    while (scheduler.get_next_block(m_block_idx, n_block_idx))
                    {
                        load_sf(params, shared_storage, scheduler.get_m_offset(), m_block_idx, n_block_idx,
                            scheduler.get_expert_idx(), sf_tile_count, sf_phase, store_phase);
                    }
                }
                __syncwarp();
            }
        }
        else
        {
            uint32_t sf_phase = 0;
            uint32_t ab_phase = 0;
            Scheduler scheduler(params.N, params.num_experts, params.token_offset);
            int32_t m_block_idx, n_block_idx;
            while (scheduler.get_next_block(m_block_idx, n_block_idx))
            {
                mma(params, shared_storage, sf_tile_count, scheduler.get_m_offset(), scheduler.get_m_boundary(),
                    m_block_idx, n_block_idx, sf_phase, ab_phase);
            }
        }
    }
};

} // namespace sm120_blockscaled_gemm
