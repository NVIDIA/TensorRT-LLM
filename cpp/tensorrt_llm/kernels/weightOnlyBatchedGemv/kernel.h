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
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/utility.h"

namespace tensorrt_llm
{
namespace kernels
{
template <typename ActType>
struct ActTypeDetails;

template <>
struct ActTypeDetails<half>
{
    using CutlassType = cutlass::half_t;
    using Vec2 = half2;

    __device__ __forceinline__ static Vec2 to_vec2(half v)
    {
        return __half2half2(v);
    }
};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
template <>
struct ActTypeDetails<__nv_bfloat16>
{
    using CutlassType = cutlass::bfloat16_t;
    using Vec2 = __nv_bfloat162;

    __device__ __forceinline__ static Vec2 to_vec2(__nv_bfloat16 v)
    {
        return __bfloat162bfloat162(v);
    }
};
#endif

template <typename ActType, WeightOnlyQuantType QType>
struct ConverterSelector
{
    static_assert(QType == WeightOnlyQuantType::Int4b || QType == WeightOnlyQuantType::Int8b);

    using WeiType = std::conditional_t<QType == WeightOnlyQuantType::Int4b, cutlass::uint4b_t, uint8_t>;
    static constexpr int kConvertCount = QType == WeightOnlyQuantType::Int4b ? 8 : 4;
    using Converter
        = cutlass::FastInterleavedAndBiasedNumericArrayConverter<typename ActTypeDetails<ActType>::CutlassType, WeiType,
            kConvertCount>;
};

template <typename ActType, WeightOnlyQuantType QType>
struct WeightOnlyDetails;

template <typename ActType>
struct WeightOnlyDetails<ActType, WeightOnlyQuantType::Int4b>
{
    // Every four rows of the original weights are interleaved into a row with stride of 64, so if each thread
    // processes 32 elements(for int4, we can use ldg.128 to load weights), then every group of two adjacent threads
    // will alternately process four different row weights
    // for example
    // every 256 consecutive int4 elements [256*i, 256*(i+1)-1] of row N under interleave layout,
    // the first 64 are from [64*i, 64*(i+1)-1] of row 4N before interleaving,
    // and the second 64 are from [64*i, 64*(i+1)-1] of row 4N+1 before interleaving, and so on.
    // So if each thread loads 32 int4 elements, then the elements of each 2 adjacent threads of each 8
    // consecutive threads will come from row 4N ~ 4N+3 respectively before interleaving.
    static constexpr int kElemBits = 4;
    static constexpr int kInterleave = 4;
    static constexpr int kStride = 64;

    // The index remapping here is to counteracts the effect of cutlass::permute_B_rows_for_mixed_gemm
    // input 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ... 31
    // weight 0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5 12 13 20 21 28 29 6 7 14 15 22 23 30 31
    static constexpr int kShuffleSize = 32;
    static constexpr int kShuffleBasicTile = 2;
    static constexpr int kShuffleContinous = 4;
    static constexpr int kShuffleStrided = 4;

    // Each warp completes the internal reduce and writes the [Batch * NPerBlock * Interleave] results to the
    // corresponding address in shared memory
    template <int Num, int WarpSize>
    __device__ __forceinline__ static void sync(float* res, float (*sm)[Num * kInterleave])
    {
#pragma unroll
        for (int i = 0; i < Num; ++i)
        {
            res[i] += __shfl_xor_sync(~0, res[i], 16);
            res[i] += __shfl_xor_sync(~0, res[i], 8);
            res[i] += __shfl_xor_sync(~0, res[i], 1);
        }
        __syncthreads();
        int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
        if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
        {
#pragma unroll
            for (int i = 0; i < Num; ++i)
            {
                sm[warp][i * kInterleave + lane / 2] = res[i];
            }
        }
        __syncthreads();
    }
};

template <typename ActType>
struct WeightOnlyDetails<ActType, WeightOnlyQuantType::Int8b>
{
    // Every two rows of the original weights are interleaved into a row with stride of 64, so if each thread
    // processes 16 elements(for int8, we can use ldg.128 to load weights), then every group of four adjacent threads
    // will alternately process two different row weights
    // for example
    // every 128 consecutive int8 elements [128*i, 128*(i+1)-1] of row N under interleave layout,
    // the first 64 are from [64*i, 64*(i+1)-1] of row 2N before interleaving,
    // and the last 64 are from [64*i, 64*(i+1)-1] of row 2N+1 before interleaving.
    // So if each thread loads 16 int8 elements, then the elements of the first four and last four threads of each 8
    // consecutive threads will come from row 2N and row 2N+1 respectively before interleaving.
    static constexpr int kElemBits = 8;
    static constexpr int kInterleave = 2;
    static constexpr int kStride = 64;

    // The index remapping here is to counteracts the effect of cutlass::permute_B_rows_for_mixed_gemm
    // input 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    // weight 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
    static constexpr int kShuffleSize = 16;
    static constexpr int kShuffleBasicTile = 2;
    static constexpr int kShuffleContinous = 2;
    static constexpr int kShuffleStrided = 4;

    // Each warp completes the internal reduce and writes the [Batch * NPerBlock * Interleave] results to the
    // corresponding address in shared memory
    template <int Num, int WarpSize>
    __device__ __forceinline__ static void sync(float* res, float (*sm)[Num * kInterleave])
    {
#pragma unroll
        for (int i = 0; i < Num; ++i)
        {
            res[i] += __shfl_xor_sync(~0, res[i], 16);
            res[i] += __shfl_xor_sync(~0, res[i], 8);
            res[i] += __shfl_xor_sync(~0, res[i], 2);
            res[i] += __shfl_xor_sync(~0, res[i], 1);
        }
        __syncthreads();
        int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
        if (lane == 0 || lane == 4)
        {
#pragma unroll
            for (int i = 0; i < Num; ++i)
            {
                sm[warp][i * kInterleave + lane / 4] = res[i];
            }
        }
        __syncthreads();
    }
};

template <typename ActType, WeightOnlyQuantType QType>
struct WeightOnlyKernelDetails
{
    using Layout = WeightOnlyDetails<ActType, QType>;

    static constexpr int kElemBits = Layout::kElemBits;
    static constexpr int kInterleave = Layout::kInterleave;
    static constexpr int kStride = Layout::kStride;

    static constexpr int kShuffleSize = Layout::kShuffleSize;
    static constexpr int kShuffleBasicTile = Layout::kShuffleBasicTile;
    static constexpr int kShuffleContinous = Layout::kShuffleContinous;
    static constexpr int kShuffleStrided = Layout::kShuffleStrided;

    // The rearrangement here counteracts the effect of cutlass::add_bias_and_interleave_int4/8s_inplace
    // Input int8 data layout
    //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)
    //
    // Converted fp16/bf16 data layout
    //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 16 bits)

    // Input int8 data layout
    //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt occupies 4 bits)
    //
    // Converted fp16/bf16 data layout
    //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt occupies 16 bits)
    static constexpr int kConvertCount = ConverterSelector<ActType, QType>::kConvertCount;
    using Converter = typename ConverterSelector<ActType, QType>::Converter;

    // Use ldg128 load data from global memory
    static constexpr int kAccessSize = 128;
    using AccessType = uint4;

    static constexpr int kElemsPerByte = 8 / kElemBits;
    static constexpr int kElemsPerThread = kAccessSize / kElemBits;
    static constexpr int kBytePerThread = kElemsPerThread / kElemsPerByte;
    static constexpr int kThreadsNumPerTile = kStride / kElemsPerThread;
    static constexpr int kThreadsNumPerInterleave = kThreadsNumPerTile * kInterleave;

    static constexpr int kConvertIters = kElemsPerThread / kConvertCount;

    // Each thread loads 16(int8b)/32(int4b) quantized weight elements each time through ldg128
    // So more times of ldg128 are needed to load the same number of fp16/bf16 activation elements.
    static constexpr int kActivationElemNumPerAccess = kAccessSize / (sizeof(ActType) * 8);
    static constexpr int kActivationAccessNum = kElemsPerThread / kActivationElemNumPerAccess;
};

template <typename WeightOnlyFlag>
struct WeightOnlyProperties;

template <>
struct WeightOnlyProperties<WeightOnlyPerChannel>
{
    static constexpr bool kIsFineGrained = false;
    static constexpr int kGroupSize = 0;
};

template <int GS>
struct WeightOnlyProperties<WeightOnlyGroupWise<GS>>
{
    static constexpr bool kIsFineGrained = true;
    static constexpr int kGroupSize = GS;
};

template <typename ActType, WeightOnlyQuantType QType, typename WeightOnlyFlag, bool Zero, int BlockSize>
struct WeightOnlyScaleLoader
{
    using ElemType = ActType;
    using Details = WeightOnlyKernelDetails<ActType, QType>;
    static constexpr bool kIsFineGrained = WeightOnlyProperties<WeightOnlyFlag>::kIsFineGrained;
    static constexpr int kGroupSize = WeightOnlyProperties<WeightOnlyFlag>::kGroupSize;

private:
    const ElemType* _scales;
    const ElemType* _zeros;
    int _stride;
    int _offset;

public:
    __device__ __forceinline__ WeightOnlyScaleLoader(
        const ElemType* scales, const ElemType* zeros, int initial_offset, int stride)
        : _scales(scales)
        , _zeros(zeros)
        , _stride(stride)
    {
        _scales += initial_offset;
        if constexpr (Zero)
        {
            _zeros += initial_offset;
        }
        // Calculate the k dimension index of the element processed by the current thread of layout before interleave
        // Used to load scales and zeros in groupwise weight only quant
        _offset = threadIdx.x / Details::kThreadsNumPerInterleave * Details::kStride
            + (threadIdx.x % Details::kThreadsNumPerTile) * Details::kElemsPerThread;
    }

    __device__ __forceinline__ void load(ElemType& scale, ElemType& zero, int nid)
    {
        int offset = nid * Details::kInterleave;
        if constexpr (kIsFineGrained)
        {
            offset += _offset / kGroupSize * _stride;
        }
        scale = _scales[offset];
        if constexpr (Zero)
        {
            zero = _zeros[offset];
        }
        else
        {
            zero = static_cast<ElemType>(0.f);
        }
    }

    __device__ __forceinline__ void advance()
    {
        _offset += BlockSize * Details::kElemsPerThread / Details::kInterleave;
    }

    __device__ __forceinline__ int offset()
    {
        return _offset;
    }
};

template <typename ActType, WeightOnlyQuantType QType, typename WeightOnlyFlag, template <typename T> class ActOp,
    bool Zero, bool Bias, bool ActScale, int NPerBlock, int Batch, int BlockSize>
__device__ void weight_only_batched_gemv(const uint8_t* qweight, const ActType* scales, const ActType* zeros,
    const ActType* in, const ActType* act_scale, const ActType* bias, ActType* out, const int n, const int k)
{
    static_assert(NPerBlock == 1 || (NPerBlock % 2 == 0));
    using ActType2 = typename ActTypeDetails<ActType>::Vec2;
    using Details = WeightOnlyKernelDetails<ActType, QType>;

    using Converter = typename Details::Converter;
    using AccType = typename Details::AccessType;
    using CvtSrcType = typename Converter::source_type;
    using CvtResType = typename Converter::result_type;
    using ScaleLoader = WeightOnlyScaleLoader<ActType, QType, WeightOnlyFlag, Zero, BlockSize>;
    extern __shared__ uint8_t shmem[];
    constexpr int Interleave = Details::kInterleave;
    constexpr int WarpSize = 32;
    constexpr int Num = Batch * NPerBlock;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n_start_id = bid * NPerBlock * Interleave;
    // Calculate the n-dimensional index of the data processed by the current thread in the interleave tile
    const int interleave_n_id = (tid / Details::kThreadsNumPerTile) % Interleave;

    qweight += n_start_id * k / Details::kElemsPerByte;
    ScaleLoader scale_loader(scales, zeros, n_start_id + interleave_n_id, n);

    float(*sm)[Num * Interleave] = reinterpret_cast<float(*)[Num * Interleave]>(shmem);

    // In order to take advantage of hfma2, we use fp16/bf16 for accumulation within threads and fp32 for accumulation
    // between threads.
    ActType accumulator[Num];
    for (int i = 0; i < Num; ++i)
    {
        accumulator[i] = static_cast<ActType>(0.f);
    }

    // Iteration in k dimensions
    for (int local_k = tid * Details::kElemsPerThread; local_k < k * Interleave;
         local_k += BlockSize * Details::kElemsPerThread)
    {
        ActType weights_f16[Details::kElemsPerThread * NPerBlock];
        ActType scale[NPerBlock], zero[NPerBlock];
#pragma unroll
        for (int idx = 0; idx < NPerBlock; ++idx)
        {
            // Load quantized weight and scales/zeros
            uint8_t weights_quantized[Details::kBytePerThread];
            load<AccType>(weights_quantized,
                qweight + idx * Interleave * k / Details::kElemsPerByte + local_k / Details::kElemsPerByte);
            scale_loader.load(scale[idx], zero[idx], idx);
            ActType weights_vec[Details::kElemsPerThread];
#pragma unroll
            for (int i = 0; i < Details::kConvertIters; ++i)
            {
                // Use cutlass::FastInterleavedAndBiasedNumericArrayConverter for I2F type conversion
                assign<CvtResType>(weights_vec + i * Details::kConvertCount,
                    Converter::convert(*reinterpret_cast<CvtSrcType*>(
                        weights_quantized + i * Details::kConvertCount / Details::kElemsPerByte)));
            }
#pragma unroll
            for (int i = 0; i < Details::kShuffleContinous; ++i)
            {
#pragma unroll
                for (int j = 0; j < Details::kShuffleStrided; ++j)
                {
                    // Dequantize the weights and arrange the shuffled elements back to the correct order in the
                    // register array
                    ActType2 v = *reinterpret_cast<ActType2*>(weights_vec + i * Details::kShuffleBasicTile
                        + j * Details::kShuffleContinous * Details::kShuffleBasicTile);
                    v = __hfma2(
                        v, ActTypeDetails<ActType>::to_vec2(scale[idx]), ActTypeDetails<ActType>::to_vec2(zero[idx]));
                    weights_f16[(i * Details::kShuffleStrided * Details::kShuffleBasicTile
                                    + j * Details::kShuffleBasicTile + 0)
                            * NPerBlock
                        + idx]
                        = v.x;
                    weights_f16[(i * Details::kShuffleStrided * Details::kShuffleBasicTile
                                    + j * Details::kShuffleBasicTile + 1)
                            * NPerBlock
                        + idx]
                        = v.y;
                }
            }
        }
        ActType act_scale_v[Details::kElemsPerThread];
        if constexpr (ActScale)
        {
#pragma unroll
            for (int idx = 0; idx < Details::kActivationAccessNum; ++idx)
            {
                load<AccType>(act_scale_v + idx * Details::kActivationElemNumPerAccess,
                    act_scale + scale_loader.offset() + idx * Details::kActivationElemNumPerAccess);
            }
        }
#pragma unroll
        for (int b = 0; b < Batch; ++b)
        {
            ActType in_v[Details::kElemsPerThread];
#pragma unroll
            for (int idx = 0; idx < Details::kActivationAccessNum; ++idx)
            {
                // load activation elements
                load<AccType>(in_v + idx * Details::kActivationElemNumPerAccess,
                    in + b * k + scale_loader.offset() + idx * Details::kActivationElemNumPerAccess);
                if constexpr (ActScale)
                {
#pragma unroll
                    for (int i = 0; i < Details::kActivationElemNumPerAccess; i += 2)
                    {
                        *reinterpret_cast<ActType2*>(in_v + idx * Details::kActivationElemNumPerAccess + i) = __hmul2(
                            *reinterpret_cast<ActType2*>(in_v + idx * Details::kActivationElemNumPerAccess + i),
                            *reinterpret_cast<ActType2*>(act_scale_v + idx * Details::kActivationElemNumPerAccess + i));
                    }
                }
            }
            // Perform vector inner product and accumulate
            if constexpr (NPerBlock == 1)
            {
                ActType2 v = ActTypeDetails<ActType>::to_vec2(static_cast<ActType>(0.f));
#pragma unroll
                for (int y = 0; y < Details::kElemsPerThread; y += 2)
                {
                    v = __hfma2(
                        *reinterpret_cast<ActType2*>(weights_f16 + y), *reinterpret_cast<ActType2*>(in_v + y), v);
                }
                accumulator[b] += __hadd(v.x, v.y);
            }
            else
            {
#pragma unroll
                for (int x = 0; x < NPerBlock / 2; ++x)
                {
#pragma unroll
                    for (int y = 0; y < Details::kElemsPerThread; ++y)
                    {
                        *reinterpret_cast<ActType2*>(accumulator + b * NPerBlock + x * 2)
                            = __hfma2(*reinterpret_cast<ActType2*>(weights_f16 + y * NPerBlock + x * 2),
                                ActTypeDetails<ActType>::to_vec2(in_v[y]),
                                *reinterpret_cast<ActType2*>(accumulator + b * NPerBlock + x * 2));
                    }
                }
            }
        }
        scale_loader.advance();
    }
    float reses[Num];
#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
        reses[i] = static_cast<float>(accumulator[i]);
    }

    // Each warp completes the internal reduce and writes the [Batch * NPerBlock * Interleave] results to the
    // corresponding address in shared memory
    Details::Layout::sync<Num, WarpSize>(reses, sm);

    // Each thread is responsible for the accumulation and store to global memory of one element
    for (int i = tid; i < Num * Interleave; i += BlockSize)
    {
        int nid = i % (NPerBlock * Interleave);
        float v = 0.f;
        for (int j = 0; j < BlockSize / WarpSize; ++j)
        {
            v += sm[j][i];
        }
        float bias_v = 0.f;
        if constexpr (Bias)
        {
            bias_v = static_cast<float>(bias[n_start_id + nid]);
        }
        int b = i / NPerBlock / Interleave;
        out[b * n + n_start_id + nid] = static_cast<ActType>(ActOp<float>::apply(v + bias_v));
    }
}

template <typename ActType, WeightOnlyQuantType QType, typename WeightOnlyFlag, template <typename T> class ActOp,
    bool Zero, bool Bias, bool ActScale, int NPerBlock, int Batch, int BlockSize>
__global__ void weight_only_batched_gemv_wrapper(const uint8_t* qweight, const ActType* scales, const ActType* zeros,
    const ActType* in, const ActType* act_scale, const ActType* bias, ActType* out, const int n, const int k)
{
    if constexpr (std::is_same_v<ActType, half>)
    {
        weight_only_batched_gemv<ActType, QType, WeightOnlyFlag, ActOp, Zero, Bias, ActScale, NPerBlock, Batch,
            BlockSize>(qweight, scales, zeros, in, act_scale, bias, out, n, k);
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
    else if (std::is_same_v<ActType, nv_bfloat16>)
    {
        weight_only_batched_gemv<ActType, QType, WeightOnlyFlag, ActOp, Zero, Bias, ActScale, NPerBlock, Batch,
            BlockSize>(qweight, scales, zeros, in, act_scale, bias, out, n, k);
    }
#endif
}

template <WeightOnlyQuantType QType, typename WeightOnlyFlag, template <typename T> class ActOp, bool Zero, bool Bias,
    int NPerBlock, int Batch, int BlockSize>
struct WeightOnlyBatchedGemvKernelLauncher
{
    static void run(const WeightOnlyParams& params, cudaStream_t stream)
    {
        if (params.act_type == WeightOnlyActivationType::FP16)
        {
            constexpr int kInterleave = WeightOnlyDetails<half, QType>::kInterleave;
            dim3 grid(params.n / NPerBlock / kInterleave);
            dim3 block(BlockSize);
            int size = sizeof(float) * BlockSize / 32 * Batch * NPerBlock * kInterleave;
            if (params.act_scale != nullptr)
            {
                weight_only_batched_gemv_wrapper<half, QType, WeightOnlyFlag, ActOp, Zero, Bias, true, NPerBlock, Batch,
                    BlockSize><<<grid, block, size, stream>>>(params.qweight,
                    reinterpret_cast<const half*>(params.scales), reinterpret_cast<const half*>(params.zeros),
                    reinterpret_cast<const half*>(params.in), reinterpret_cast<const half*>(params.act_scale),
                    reinterpret_cast<const half*>(params.bias), reinterpret_cast<half*>(params.out), params.n,
                    params.k);
            }
            else
            {
                weight_only_batched_gemv_wrapper<half, QType, WeightOnlyFlag, ActOp, Zero, Bias, false, NPerBlock,
                    Batch, BlockSize><<<grid, block, size, stream>>>(params.qweight,
                    reinterpret_cast<const half*>(params.scales), reinterpret_cast<const half*>(params.zeros),
                    reinterpret_cast<const half*>(params.in), reinterpret_cast<const half*>(params.act_scale),
                    reinterpret_cast<const half*>(params.bias), reinterpret_cast<half*>(params.out), params.n,
                    params.k);
            }
        }
#if defined(ENABLE_BF16)
        else if (params.act_type == WeightOnlyActivationType::BF16)
        {
            constexpr int kInterleave = WeightOnlyDetails<nv_bfloat16, QType>::kInterleave;
            dim3 grid(params.n / NPerBlock / kInterleave);
            dim3 block(BlockSize);
            int size = sizeof(float) * BlockSize / 32 * Batch * NPerBlock * kInterleave;
            if (params.act_scale != nullptr)
            {
                weight_only_batched_gemv_wrapper<__nv_bfloat16, QType, WeightOnlyFlag, ActOp, Zero, Bias, true,
                    NPerBlock, Batch, BlockSize><<<grid, block, size, stream>>>(params.qweight,
                    reinterpret_cast<const __nv_bfloat16*>(params.scales),
                    reinterpret_cast<const __nv_bfloat16*>(params.zeros),
                    reinterpret_cast<const __nv_bfloat16*>(params.in),
                    reinterpret_cast<const __nv_bfloat16*>(params.act_scale),
                    reinterpret_cast<const __nv_bfloat16*>(params.bias), reinterpret_cast<__nv_bfloat16*>(params.out),
                    params.n, params.k);
            }
            else
            {
                weight_only_batched_gemv_wrapper<__nv_bfloat16, QType, WeightOnlyFlag, ActOp, Zero, Bias, false,
                    NPerBlock, Batch, BlockSize><<<grid, block, size, stream>>>(params.qweight,
                    reinterpret_cast<const __nv_bfloat16*>(params.scales),
                    reinterpret_cast<const __nv_bfloat16*>(params.zeros),
                    reinterpret_cast<const __nv_bfloat16*>(params.in),
                    reinterpret_cast<const __nv_bfloat16*>(params.act_scale),
                    reinterpret_cast<const __nv_bfloat16*>(params.bias), reinterpret_cast<__nv_bfloat16*>(params.out),
                    params.n, params.k);
            }
        }
#endif
    }
};
} // namespace kernels
} // namespace tensorrt_llm
