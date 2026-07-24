/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "rmsNormFp4QuantKernels.h"
#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Self-contained device helpers for this kernel. These mirror the small
// reduce_fusion utilities in customAllReduceKernels.cu but are copied here (in a
// private namespace) so this translation unit does not depend on the AllReduce
// files at all. Kept byte-identical to the originals.
namespace rms_norm_fp4_quant
{

static constexpr int kBytesPerAccess = 16;
static constexpr int kWarpSize = 32;
static constexpr int kMaxCtaSize = 1024;

// Type converter that packs data format to 128-bit data type.
using PackedFloat = union
{
    int4 packed;
    float unpacked[4];
};

using PackedHalf = union
{
    int4 packed;
    half2 unpacked[4];
};

template <typename T>
struct PackedOn16Bytes
{
};

template <>
struct PackedOn16Bytes<float>
{
    using Type = PackedFloat;
};

template <>
struct PackedOn16Bytes<half>
{
    using Type = PackedHalf;
};

#ifdef ENABLE_BF16
using PackedBFloat16 = union
{
    int4 packed;
    __nv_bfloat162 unpacked[4];
};

template <>
struct PackedOn16Bytes<__nv_bfloat16>
{
    using Type = PackedBFloat16;
};
#endif

// add two 128b data
template <typename T>
inline __device__ int4 add128b(T& a, T& b)
{
    T c;
    c.unpacked[0] = a.unpacked[0] + b.unpacked[0];
    c.unpacked[1] = a.unpacked[1] + b.unpacked[1];
    c.unpacked[2] = a.unpacked[2] + b.unpacked[2];
    c.unpacked[3] = a.unpacked[3] + b.unpacked[3];
    return c.packed;
}

inline __device__ float warp_reduce_sum(float val)
{
    val += __shfl_xor_sync(~0, val, 16);
    val += __shfl_xor_sync(~0, val, 8);
    val += __shfl_xor_sync(~0, val, 4);
    val += __shfl_xor_sync(~0, val, 2);
    val += __shfl_xor_sync(~0, val, 1);
    return val;
}

inline __device__ float block_reduce_sum(float val)
{
    __shared__ float smem[kWarpSize];
    int lane_id = threadIdx.x % kWarpSize, warp_id = threadIdx.x / kWarpSize, warp_num = blockDim.x / kWarpSize;
    val = warp_reduce_sum(val);
    if (lane_id == 0)
    {
        smem[warp_id] = val;
    }
    __syncthreads();
    val = lane_id < warp_num ? smem[lane_id] : 0.f;
    val = warp_reduce_sum(val);
    return val;
}

template <typename T, typename PackedStruct>
inline __device__ float accumulate(float acc, PackedStruct& vec)
{
    static constexpr int kLoopNum = sizeof(PackedStruct) / sizeof(T);
#pragma unroll
    for (int i = 0; i < kLoopNum; ++i)
    {
        float v = static_cast<float>(reinterpret_cast<T*>(vec.unpacked)[i]);
        acc += v * v;
    }
    return acc;
}

template <typename T, bool Affine, typename PackedStruct>
inline __device__ int4 rms_norm(float denom, PackedStruct& vec, PackedStruct& weight)
{
    static constexpr int kLoopNum = sizeof(PackedStruct) / sizeof(T);
    PackedStruct ret;
#pragma unroll
    for (int i = 0; i < kLoopNum; ++i)
    {
        float v1 = static_cast<float>(reinterpret_cast<T*>(vec.unpacked)[i]);
        if constexpr (Affine)
        {
            float v2 = static_cast<float>(reinterpret_cast<T*>(weight.unpacked)[i]);
            reinterpret_cast<T*>(ret.unpacked)[i] = static_cast<T>(v1 * denom * v2);
        }
        else
        {
            reinterpret_cast<T*>(ret.unpacked)[i] = static_cast<T>(v1 * denom);
        }
    }
    return ret.packed;
}

// Fused (optional residual-add +) RMSNorm + NVFP4 input-quantize. Invoked by
// the standalone thop ops fused_add_rmsnorm_fp4_quantize (Residual=true) and
// fused_rmsnorm_fp4_quantize (Residual=false) on the attention-DP path.
// Performs the residual-add and reduction, then emits a per-block
// (SF_VEC_SIZE=16) NVFP4 representation to (quant_out, scale_out). When
// OutNorm=true, BF16 norm_out is also written (so a downstream consumer can read
// the un-quantized value).
//
// Layout assumptions (match cvt_warp_fp16_to_fp4):
//   - Each thread accesses kPackedSize=8 BF16/half elements (one int4 = 16 B).
//   - Two adjacent threads cover one SF_VEC_SIZE=16 block; SF is computed
//     warp-cooperatively via __shfl_xor_sync inside cvt_warp_fp16_to_fp4.
//   - Caller guarantees hidden_size % SF_VEC_SIZE == 0.
template <typename T, bool Bias = false, bool Residual = false, bool Affine = false, bool UseSmem = false,
    bool OutNorm = false>
__global__ void rmsNormFp4QuantKernel(RmsNormFp4QuantParams params)
{
    static constexpr int kPackedSize = kBytesPerAccess / sizeof(T);
    static constexpr int kSfVecSize = 16;
    using PackedStruct = typename PackedOn16Bytes<T>::Type;

    extern __shared__ uint8_t smem_ptr[];
    T* smem = reinterpret_cast<T*>(smem_ptr);

    int const bid = blockIdx.x;
    int const tid = threadIdx.x;

    T const* bias_buffer = reinterpret_cast<T const*>(params.bias_buffer);
    T const* residual_buffer = reinterpret_cast<T const*>(params.residual_buffer);
    T const* weight_buffer = reinterpret_cast<T const*>(params.weight_buffer);
    T const* intermediate_buffer = reinterpret_cast<T const*>(params.intermediate_buffer);
    T* residual_out_buffer = reinterpret_cast<T*>(params.residual_out_buffer);
    T* norm_out = reinterpret_cast<T*>(params.norm_out);

    int const block_offset = bid * params.hidden_size;
    // Input rows may be strided (e.g. a column slice of a wider projection,
    // such as the leading q_lora_rank columns of kv_a_proj_with_mqa). When
    // input_row_stride <= 0 it defaults to hidden_size (packed rows), making
    // this byte-identical to all existing callers. Only the INPUT read offset
    // uses the stride; every output (residual/norm/quant/scale) stays packed.
    int const input_block_offset = bid * (params.input_row_stride > 0 ? params.input_row_stride : params.hidden_size);
    int const thread_offset = tid * kPackedSize;

    if constexpr (Residual)
    {
        residual_buffer += block_offset;
        // residual_out is packed [m, hidden_size], so it uses the dense block
        // offset (not the possibly-strided input offset).
        residual_out_buffer += block_offset;
    }
    intermediate_buffer += input_block_offset;
    if constexpr (OutNorm)
    {
        norm_out += block_offset;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaGridDependencySynchronize();
#endif

    PackedStruct inter_vec, weight_vec;
    float acc = 0.f;
    for (int offset = thread_offset; offset < params.hidden_size; offset += blockDim.x * kPackedSize)
    {
        inter_vec.packed = *reinterpret_cast<int4 const*>(intermediate_buffer + offset);
        if constexpr (Bias)
        {
            PackedStruct bias_vec;
            bias_vec.packed = *reinterpret_cast<int4 const*>(bias_buffer + offset);
            inter_vec.packed = add128b(inter_vec, bias_vec);
        }
        if constexpr (Residual)
        {
            PackedStruct residual_vec;
            residual_vec.packed = *reinterpret_cast<int4 const*>(residual_buffer + offset);
            inter_vec.packed = add128b(inter_vec, residual_vec);
            // Write the residual sum to a distinct output buffer (packed offset),
            // leaving the input intermediate_buffer untouched.
            *reinterpret_cast<int4*>(residual_out_buffer + offset) = inter_vec.packed;
        }
        acc = accumulate<T>(acc, inter_vec);
        if constexpr (UseSmem)
        {
            *reinterpret_cast<int4*>(&smem[offset]) = inter_vec.packed;
        }
    }
    acc = block_reduce_sum(acc);
    float const denom = rsqrtf(acc / params.hidden_size + params.eps);

    float const sf_scale = params.scale_factor_ptr ? *params.scale_factor_ptr : 1.f;
    int const hidden_dim_packed = params.hidden_size / kSfVecSize;

    // cvt_warp_fp16_to_fp4 performs full-mask __shfl_xor_sync exchanges, which
    // require every lane of the warp to execute the call. When hidden_size /
    // kPackedSize is not a multiple of kWarpSize (e.g. hidden_size = 32, 128,
    // 8208), the tail warp would otherwise be only partially active in this
    // loop -- undefined behavior. Iterate with a warp-uniform bound (the warp's
    // lane-0 offset) so all 32 lanes stay converged through the shuffle, and
    // mask the per-lane loads/stores instead. Out-of-range lanes feed zeros to
    // the shuffle; this is safe because hidden_size % kSfVecSize == 0 makes the
    // active region end on an SF-pair boundary, so a padding lane's xor-1
    // partner is always another padding lane.
    int const lane_id = tid % kWarpSize;
    for (int offset = thread_offset; offset - lane_id * kPackedSize < params.hidden_size;
         offset += blockDim.x * kPackedSize)
    {
        bool const valid = offset < params.hidden_size;
        if (valid)
        {
            if constexpr (UseSmem)
            {
                inter_vec.packed = *reinterpret_cast<int4 const*>(&smem[offset]);
            }
            if constexpr (Affine)
            {
                weight_vec.packed = *reinterpret_cast<int4 const*>(weight_buffer + offset);
            }
            inter_vec.packed = rms_norm<T, Affine>(denom, inter_vec, weight_vec);
            if constexpr (OutNorm)
            {
                *reinterpret_cast<int4*>(norm_out + offset) = inter_vec.packed;
            }
        }
        else
        {
            // Benign values for the warp-cooperative SF exchange below (a
            // padding lane's first-loop inter_vec may be uninitialized).
            inter_vec.packed = make_int4(0, 0, 0, 0);
        }

        // FP4 quantize this 8-element packed vec; warp-cooperate with the
        // neighbour thread (offset ^ kPackedSize) to compute a single SF for
        // their joint 16-element block. All lanes (incl. padding) must execute
        // this call; padding lanes pass a null SF pointer and drop the result.
        ::tensorrt_llm::kernels::PackedVec<T> pv
            = *reinterpret_cast<::tensorrt_llm::kernels::PackedVec<T>*>(&inter_vec);
        uint8_t* sf_out_ptr = nullptr;
        if (valid)
        {
            int const access_id_in_token = offset / kPackedSize;
            sf_out_ptr = ::tensorrt_llm::kernels::cvt_quant_get_sf_out_offset<uint32_t, 2>(std::nullopt, bid,
                access_id_in_token, std::nullopt, hidden_dim_packed, reinterpret_cast<uint32_t*>(params.scale_out),
                params.sf_layout);
        }
        uint32_t const quant_val = ::tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<T, kSfVecSize, /*UE8M0_SF=*/false>(
            pv, sf_scale, sf_out_ptr);
        if (valid)
        {
            int const access_id = bid * (params.hidden_size / kPackedSize) + (offset / kPackedSize);
            reinterpret_cast<uint32_t*>(params.quant_out)[access_id] = quant_val;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, bool OutNorm = false>
void launchRmsNormFp4QuantKernel(RmsNormFp4QuantParams const& params, cudaStream_t stream)
{
    static constexpr int kPackedSize = kBytesPerAccess / sizeof(T);
    TLLM_CHECK(params.hidden_size % kPackedSize == 0);
    TLLM_CHECK(params.hidden_size % 16 == 0); // SF_VEC_SIZE
    int need_threads = params.hidden_size / kPackedSize;
    int cta_size = need_threads <= kMaxCtaSize ? (need_threads + kWarpSize - 1) / kWarpSize * kWarpSize : kMaxCtaSize;
    int cta_num = params.elts_total / params.hidden_size;
    bool const need_smem = (cta_size * kBytesPerAccess / sizeof(T) < params.hidden_size);
    int smem_size = need_smem ? params.hidden_size * sizeof(T) : 0;
    bool const use_smem = need_smem;

    bool const has_bias = params.bias_buffer != nullptr;
    bool const has_residual = params.residual_buffer != nullptr;
    bool const has_weight = params.weight_buffer != nullptr;

    // Macro-dispatch over Bias/Residual/Affine/UseSmem and the OutNorm template arg.
    // Launch through launchWithPdlWhenEnabled so the kernel's PDL primitives
    // (cudaGridDependencySynchronize / cudaTriggerProgrammaticLaunchCompletion)
    // are actually enabled when TLLM_ENABLE_PDL is set.
#define DISPATCH_FP4_QUANT(BIAS, RESIDUAL, AFFINE, SMEM)                                                               \
    if (use_smem == SMEM)                                                                                              \
    {                                                                                                                  \
        tensorrt_llm::common::launchWithPdlWhenEnabled("rmsNormFp4Quant",                                              \
            rmsNormFp4QuantKernel<T, BIAS, RESIDUAL, AFFINE, SMEM, OutNorm>, dim3(cta_num), dim3(cta_size),            \
            static_cast<size_t>(smem_size), stream, params);                                                           \
    }

    auto launch = [&]()
    {
        if (has_bias && has_residual && has_weight)
        {
            DISPATCH_FP4_QUANT(true, true, true, true)
            else DISPATCH_FP4_QUANT(true, true, true, false)
        }
        else if (!has_bias && has_residual && has_weight)
        {
            DISPATCH_FP4_QUANT(false, true, true, true)
            else DISPATCH_FP4_QUANT(false, true, true, false)
        }
        else if (has_bias && !has_residual && has_weight)
        {
            DISPATCH_FP4_QUANT(true, false, true, true)
            else DISPATCH_FP4_QUANT(true, false, true, false)
        }
        else if (!has_bias && !has_residual && has_weight)
        {
            DISPATCH_FP4_QUANT(false, false, true, true)
            else DISPATCH_FP4_QUANT(false, false, true, false)
        }
        else if (has_bias && has_residual && !has_weight)
        {
            DISPATCH_FP4_QUANT(true, true, false, true)
            else DISPATCH_FP4_QUANT(true, true, false, false)
        }
        else if (!has_bias && has_residual && !has_weight)
        {
            DISPATCH_FP4_QUANT(false, true, false, true)
            else DISPATCH_FP4_QUANT(false, true, false, false)
        }
        else if (has_bias && !has_residual && !has_weight)
        {
            DISPATCH_FP4_QUANT(true, false, false, true)
            else DISPATCH_FP4_QUANT(true, false, false, false)
        }
        else
        {
            DISPATCH_FP4_QUANT(false, false, false, true)
            else DISPATCH_FP4_QUANT(false, false, false, false)
        }
    };
    launch();
#undef DISPATCH_FP4_QUANT
}

} // namespace rms_norm_fp4_quant

void residualRmsNormFp4Quant(RmsNormFp4QuantParams const& params, nvinfer1::DataType dataType, cudaStream_t stream)
{
    // The NVFP4 epilogue (cvt_warp_fp16_to_fp4) is compiled only for
    // __CUDA_ARCH__ >= 1000 and emits zeros otherwise, so this kernel is correct
    // only on SM 10.x (Blackwell). Fail fast on unsupported archs rather than
    // silently producing wrong FP4 (the Python dispatch in rms_norm.py already
    // routes those to the unfused path).
    int const sm = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(sm >= 100 && sm < 120,
        "residualRmsNormFp4Quant requires SM 10.x (Blackwell); got SM %d. The fused NVFP4 epilogue is unsupported on "
        "this arch.",
        sm);
    TLLM_CHECK_WITH_INFO(params.quant_out != nullptr && params.scale_out != nullptr,
        "residualRmsNormFp4Quant requires quant_out and scale_out output buffers.");
    sync_check_cuda_error(stream);
    bool const out_norm = (params.norm_out != nullptr);
    if (out_norm)
    {
        switch (dataType)
        {
#ifdef ENABLE_BF16
        case nvinfer1::DataType::kBF16:
            rms_norm_fp4_quant::launchRmsNormFp4QuantKernel<__nv_bfloat16, /*OutNorm=*/true>(params, stream);
            break;
#endif
        case nvinfer1::DataType::kHALF:
            rms_norm_fp4_quant::launchRmsNormFp4QuantKernel<half, /*OutNorm=*/true>(params, stream);
            break;
        default: TLLM_THROW("Unsupported dataType for residualRmsNormFp4Quant");
        }
    }
    else
    {
        switch (dataType)
        {
#ifdef ENABLE_BF16
        case nvinfer1::DataType::kBF16:
            rms_norm_fp4_quant::launchRmsNormFp4QuantKernel<__nv_bfloat16, /*OutNorm=*/false>(params, stream);
            break;
#endif
        case nvinfer1::DataType::kHALF:
            rms_norm_fp4_quant::launchRmsNormFp4QuantKernel<half, /*OutNorm=*/false>(params, stream);
            break;
        default: TLLM_THROW("Unsupported dataType for residualRmsNormFp4Quant");
        }
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
