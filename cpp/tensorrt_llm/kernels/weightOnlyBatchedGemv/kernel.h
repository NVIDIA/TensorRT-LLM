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

    extern __shared__ TypeA shmem_sz[];
    // dimension of each is [kInterleave * CtaN, {Threads * kInterleave * kThreadsPerInterleavedTile / GroupSize or 1}]
    TypeA* vec_scale = shmem_sz + CtaN * Details::kInterleave * offset_k_group;
    TypeA* vec_zero = nullptr;
    if constexpr (EnableZero)
    {
        vec_zero = shmem_sz + CtaN * Details::kInterleave * (GroupSize != 0 ? Threads * Details::kInterleave * Details::kThreadsPerInterleavedTile / GroupSize : 1);
        vec_zero += CtaN * Details::kInterleave * offset_k_group;
    }
    GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
        act, offset_m * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
    GMemIterator<EnableActScale, AccessTypeA, 1, Details::kAccessNumA, TypeA> act_scale_iterator(
        act_scale, real_offset_k, CtaK / Details::kInterleave, 0);
    GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(weight,
        (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW, CtaK / Details::kElemsPerByteW,
        interleaved_k / Details::kElemsPerByteW);
    SHMemIterator<Mandatory, CtaN * Details::kInterleave * sizeof(TypeA), TypeA> scales_iterator(scales,
        offset_k_group * n + blk_offset_n, vec_scale, thr_offset_n,
        (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave,
        (GroupSize != 0 ? GroupSize / Details::kInterleave / Details::kThreadsPerInterleavedTile : CtaN * Details::kInterleave)
        );
    SHMemIterator<EnableZero, CtaN * Details::kInterleave * sizeof(TypeA), TypeA> zeros_iterator(zeros,
        offset_k_group * n + blk_offset_n, vec_zero, thr_offset_n,
        (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave,
        (GroupSize != 0 ? GroupSize / Details::kInterleave / Details::kThreadsPerInterleavedTile : CtaN * Details::kInterleave)
        );

    out += offset_m * n + tile_id_n * CtaN * Details::kInterleave;
    if constexpr (EnableBias)
    {
        bias += tile_id_n * CtaN * Details::kInterleave;
    }

    TypeA tile_acc[CtaM * CtaN];
    fill<CtaM * CtaN>(tile_acc, static_cast<TypeA>(0.f));

    for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter)
    {
        TypeA vec_act_scale[StepK];
        TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
        uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];

        scales_iterator.load(iter);
        zeros_iterator.load(iter);
        act_scale_iterator.load(vec_act_scale, iter);
#pragma unroll
        for (int i = 0; i < CtaN; ++i)
        {
            weight_iterator.load(tile_w_quantized, iter, i);
            dequantize<Details, 1, StepK, EnableZero, ApplyAlphaInAdvance>(
                tile_w, tile_w_quantized, scales_iterator.iter(i),
                zeros_iterator.iter(i), alpha);
            pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
        }
#pragma unroll
        for (int i = 0; i < CtaM; ++i)
        {
            act_iterator.load(tile_a, iter, i);
            apply_scale<Details, 1, StepK, EnableActScale>(tile_a, vec_act_scale);
            mma<Details, 1, CtaN, StepK>(tile_acc + i * CtaN, tile_w_pack2, tile_a);
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
    int const shmem_size = (EnableZero ? 2 : 1) * CtaN * Details::kInterleave * sizeof(T)
        * (GroupSize != 0 ? Threads * Details::kInterleave * Details::kThreadsPerInterleavedTile / GroupSize : 1);
    // clang-format off
    kernel<Details, CtaM, CtaN, Threads, GroupSize, EnableActScale, EnableZero, EnableBias, ApplyAlphaInAdvance><<<
        grid, block, shmem_size, s>>>(
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
