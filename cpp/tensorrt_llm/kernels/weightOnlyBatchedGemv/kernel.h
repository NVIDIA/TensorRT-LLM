/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/common.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/converter.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/details.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/utility.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace weight_only
{
template <typename Details, int CtaM, int CtaN, int Threads, int GroupSize, bool EnableActScale, bool EnableZero,
    bool EnableBias, bool ApplyAlphaInAdvance, typename TypeA = typename Details::TypeDetailsA::Type>
__global__ void kernel(TypeA* act, TypeA* act_scale, uint8_t* weight, TypeA* scales, TypeA* zeros, TypeA* bias,
    TypeA* out, float alpha, int m, int n, int k)
{
    // clang-format off
    // ArgType          ArgName          DataType               Shape                           Layout
    //
    // input            act              fp16/bf16              [m, k]                          RowMajor
    // input            act_scale        fp16/bf16              [1, k]                          RowMajor
    // input            weight           int4b/int8b            [k, n]                          ColumnMajor or ColumnMajorInterleaved
    // input            scales           fp16/bf16              [k / GroupSize, n] or [1, n]    RowMajor
    // input            zeros            fp16/bf16              [k / GroupSize, n] or [1, n]    RowMajor
    // input            bias             fp16/bf16              [1, n]                          RowMajor
    // output           out              fp16/bf16              [m, n]                          RowMajor
    // clang-format on
    using AccessTypeA = typename Details::AccessTypeA;
    using AccessTypeW = typename Details::AccessTypeW;

    static constexpr bool Mandatory = true;
    static constexpr int StepK = Details::kStepK;
    static constexpr int CtaK = StepK * Threads;
    static_assert(CtaN % 2 == 0);
    if constexpr (GroupSize != 0)
    {
        static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
    }

    int const origin_k = k, interleaved_k = k * Details::kInterleave;

    int const tile_id_m = blockIdx.x, tile_id_n = blockIdx.y, tid = threadIdx.x;
    int const offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
    int const blk_offset_n = interleaved_offset_n * Details::kInterleave;
    int const thr_offset_n = (tid / Details::kThreadsPerInterleavedTile) % Details::kInterleave;
    int const real_offset_k
        = (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize
        + ((tid * StepK) % Details::LayoutDetails::kTileSize);
    int const offset_k_group = (GroupSize != 0 ? real_offset_k / GroupSize : 0);
    
    // number of threads with neighboring scale or zero values
    static constexpr int near_sz_group = (GroupSize != 0 ? GroupSize / Details::kInterleave / Details::kThreadsPerInterleavedTile : 32);
    // number of scale or zero values needed in a single block per k iterations
    static constexpr int sz_per_iter = CtaN * Details::kInterleave;
    // number of k-iterations which would be loaded in a single shared memory block load
    static constexpr int sh_sz_group = (GroupSize != 0 ? near_sz_group / (sz_per_iter > (sizeof(AccessTypeA) / sizeof(TypeA)) ? 
                                sz_per_iter / (sizeof(AccessTypeA) / sizeof(TypeA)) : 1) : 1);
    static constexpr int CtaKInterleave = CtaK / Details::kInterleave;
    
    __shared__ TypeA shmem_sz[sz_per_iter * (GroupSize != 0 ? Threads / near_sz_group : 1) * sh_sz_group * (EnableZero ? 2 : 1)];
    __shared__ uint8_t shmem_w[CtaK / Details::kElemsPerByteW * CtaN];
    __shared__ TypeA shmem_a[CtaKInterleave * (EnableActScale ? CtaM + 1 : CtaM)];

    TypeA* sh_scale = shmem_sz + sz_per_iter * offset_k_group;
    TypeA* sh_zero = nullptr;
    TypeA* sh_actscale = nullptr;

    if constexpr (EnableZero)
    {
        sh_zero = shmem_sz + sz_per_iter * (GroupSize != 0 ? Threads / near_sz_group * sh_sz_group : 1);
        sh_zero += sz_per_iter * offset_k_group;
    }

    if constexpr (EnableActScale)
    {
        sh_actscale = shmem_a + CtaKInterleave * CtaM;
    }
    
    SHMemIterator<Mandatory, 0,
    ShMemOptimizer<true, 0, CtaKInterleave, CtaKInterleave, StepK, AccessTypeA, TypeA>> act_iterator(
            act, 0, shmem_a, real_offset_k, CtaKInterleave, origin_k, interleaved_k / CtaK);
    SHMemIterator<EnableActScale, 0,
    ShMemOptimizer<false, 0, 0, CtaKInterleave, StepK, AccessTypeA, TypeA>> act_scale_iterator(
            act_scale, 0, sh_actscale, real_offset_k, CtaKInterleave, 0, interleaved_k / CtaK);
    SHMemIterator<Mandatory, 0,  
    ShMemOptimizer<true, 0, CtaK / Details::kElemsPerByteW, CtaK / Details::kElemsPerByteW, StepK / Details::kElemsPerByteW, AccessTypeW, uint8_t>> weight_iterator(
            weight, interleaved_offset_n * interleaved_k / Details::kElemsPerByteW, shmem_w, tid * StepK / Details::kElemsPerByteW,
            CtaK / Details::kElemsPerByteW, interleaved_k / Details::kElemsPerByteW, interleaved_k / CtaK);
    SHMemIterator<Mandatory, near_sz_group, 
    ShMemOptimizer<false, (GroupSize != 0 ? sz_per_iter * Threads / near_sz_group : 0), Details::kInterleave, sz_per_iter, 1, AccessTypeA, TypeA>> scales_iterator(
            scales, offset_k_group * n + blk_offset_n, sh_scale, thr_offset_n, (GroupSize != 0 ? CtaKInterleave / GroupSize * n : 0),
            0, (GroupSize != 0 ? interleaved_k / CtaK : 1));
    SHMemIterator<EnableZero, near_sz_group, 
    ShMemOptimizer<false, (GroupSize != 0 ? sz_per_iter * Threads / near_sz_group : 0), Details::kInterleave, sz_per_iter, 1, AccessTypeA, TypeA>> zeros_iterator(
            zeros, offset_k_group * n + blk_offset_n, sh_zero, thr_offset_n, (GroupSize != 0 ? CtaKInterleave / GroupSize * n : 0),
            0, (GroupSize != 0 ? interleaved_k / CtaK : 1));

    out += offset_m * n + tile_id_n * CtaN * Details::kInterleave;
    if constexpr (EnableBias)
    {
        bias += tile_id_n * CtaN * Details::kInterleave;
    }

// Prefetch stage 0
#pragma unroll
    for (int i = 0; i < CtaN; ++i)
    {
        weight_iterator.copy_to_shmem(0, i);
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i)
    {
        act_iterator.copy_to_shmem(0, i);
    }
    act_scale_iterator.copy_to_shmem(0);
    __pipeline_commit();

    TypeA tile_acc[CtaM * CtaN];
    fill<CtaM * CtaN>(tile_acc, static_cast<TypeA>(0.f));

    for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter)
    {
        TypeA vec_act_scale[StepK];
        TypeA vec_scale[CtaN], vec_zero[CtaN];
        TypeA tile_a[StepK * CtaM], tile_w[StepK], tile_w_pack2[CtaN * StepK];
        uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW * CtaN];

        if (iter % sh_sz_group == 0)
        {
            scales_iterator.copy_to_shmem(iter);
            zeros_iterator.copy_to_shmem(iter);
            __pipeline_commit();
        }
        __pipeline_wait_prior(0);

#pragma unroll
        for (int i = 0; i < CtaN; ++i)
        {
            scales_iterator.load(vec_scale + i, i, iter % sh_sz_group);
            zeros_iterator.load(vec_zero + i, i, iter % sh_sz_group);
            weight_iterator.load(tile_w_quantized + i * StepK / Details::kElemsPerByteW, i);
        }
#pragma unroll
        for (int i = 0; i < CtaM; ++i)
        {
            act_iterator.load(tile_a + i * StepK, i);
        }    
        act_scale_iterator.load(vec_act_scale);

        // Prefetch next stage
        if (idx_k + CtaK < interleaved_k)
        {
#pragma unroll
            for (int i = 0; i < CtaN; ++i)
            {
                weight_iterator.copy_to_shmem(iter + 1, i);
            }
#pragma unroll
            for (int i = 0; i < CtaM; ++i)
            {
                act_iterator.copy_to_shmem(iter + 1, i);
            }
            act_scale_iterator.copy_to_shmem(iter + 1);
            __pipeline_commit();
        }

#pragma unroll
        for (int i = 0; i < CtaN; ++i)
        {
            dequantize<Details, 1, StepK, EnableZero, ApplyAlphaInAdvance>(
                tile_w, tile_w_quantized + i * StepK / Details::kElemsPerByteW, vec_scale + i, vec_zero + i, alpha);
            pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
        }
#pragma unroll
        for (int i = 0; i < CtaM; ++i)
        {
            apply_scale<Details, 1, StepK, EnableActScale>(tile_a + i * StepK, vec_act_scale);
            mma<Details, 1, CtaN, StepK>(tile_acc + i * CtaN, tile_w_pack2, tile_a + i * StepK);
        }
    }
    epilogue<Details, CtaM, CtaN, Threads, EnableBias, ApplyAlphaInAdvance>(out, n, tile_acc, bias, alpha);
}

template <typename Details, int CtaM, int CtaN, int Threads, int GroupSize, bool EnableActScale, bool EnableZero,
    bool EnableBias, bool ApplyAlphaInAdvance>
void exec_kernel(Params& params, cudaStream_t s)
{
    using T = typename Details::TypeDetailsA::Type;
    if (params.m % CtaM || params.n % (CtaN * Details::kInterleave))
    {
        throw std::runtime_error("launch failed");
    }
    dim3 grid(params.m / CtaM, params.n / (CtaN * Details::kInterleave));
    dim3 block(Threads);
    // clang-format off
    kernel<Details, CtaM, CtaN, Threads, GroupSize, EnableActScale, EnableZero, EnableBias, ApplyAlphaInAdvance><<<grid, block, 0, s>>>(
        reinterpret_cast<T*>(params.act),
        reinterpret_cast<T*>(params.act_scale),
        reinterpret_cast<uint8_t*>(params.weight),
        reinterpret_cast<T*>(params.scales),
        reinterpret_cast<T*>(params.zeros),
        reinterpret_cast<T*>(params.bias),
        reinterpret_cast<T*>(params.out),
        params.alpha,
        params.m, params.n, params.k
    );
    // clang-format on
}

} // namespace weight_only
} // namespace kernels
} // namespace tensorrt_llm
