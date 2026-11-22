// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <vector>

namespace
{
constexpr int kCtaSize = 256;
constexpr int kVecSize = 8;
constexpr int kPairsPerVec = 4;
constexpr int64_t kMaxGridDim = 65536;

int64_t gridSize(int64_t numel)
{
    int64_t vecElements = numel / kVecSize;
    int64_t blocks = (vecElements + kCtaSize - 1) / kCtaSize;
    return std::max(int64_t(1), std::min(blocks, kMaxGridDim));
}

__device__ __forceinline__ float2 mulPairRn(float2 value, float factor)
{
    // Each b64 low/high word is the corresponding original FP32 component.
    // No .ftz, no intervening FP16/BF16 round, no changed F8 conversion.
    float lo, hi;
    asm("{ .reg .b64 a, b, c;\n"
        "  mov.b64 a, {%2, %3};\n"
        "  mov.b64 b, {%4, %4};\n"
        "  mul.rn.f32x2 c, a, b;\n"
        "  mov.b64 {%0, %1}, c;\n"
        "}"
        : "=f"(lo), "=f"(hi)
        : "f"(value.x), "f"(value.y), "f"(factor));
    return make_float2(lo, hi);
}

// mode0: copied native scalar arithmetic; mode1: same 1/s + packed RN;
// mode2: packed RN with an independently qualified immutable GPU RN factor.
template <int Mode>
__global__ void h3QuantVec8(
    __nv_fp8_e4m3* output, float const* inputScale, float const* inputFactor, __nv_bfloat16 const* input, int64_t numel)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    float factor;
    if constexpr (Mode == 2)
    {
        factor = inputFactor[0];
    }
    else
    {
        float const s = inputScale[0];
        factor = 1.0f / s;
    }
    int64_t const vecElements = numel / kVecSize;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t vi = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x; vi < vecElements; vi += stride)
    {
        int64_t const base = vi * kVecSize;
        float4 raw = *reinterpret_cast<float4 const*>(input + base);
        __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raw);
        __nv_fp8x2_storage_t fp8_pairs[kPairsPerVec];
#pragma unroll
        for (int p = 0; p < kPairsPerVec; ++p)
        {
            float2 f2 = __bfloat1622float2(pairs[p]);
            if constexpr (Mode == 0)
            {
                f2.x *= factor;
                f2.y *= factor;
            }
            else
            {
                f2 = mulPairRn(f2, factor);
            }
            fp8_pairs[p] = __nv_cvt_float2_to_fp8x2(f2, __NV_SATFINITE, __NV_E4M3);
        }
        *reinterpret_cast<uint2*>(output + base) = *reinterpret_cast<uint2 const*>(fp8_pairs);
    }
    int64_t const tailStart = vecElements * kVecSize;
    for (int64_t i = tailStart + threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x; i < numel; i += stride)
    {
        output[i] = __nv_fp8_e4m3(static_cast<float>(input[i]) * factor);
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <bool ExactRN>
__global__ void h3QuantFactor(float* output, float const* scales, int64_t count)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    int64_t i = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (i < count)
    {
        if constexpr (ExactRN)
        {
            float factor;
            asm("div.rn.f32 %0, %1, %2;" : "=f"(factor) : "f"(1.0f), "f"(scales[i]));
            output[i] = factor;
        }
        else
        {
            output[i] = 1.0f / scales[i];
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

cudaLaunchConfig_t launchConfig(int64_t blocks, cudaStream_t stream, bool pdl, cudaLaunchAttribute* attr)
{
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned>(blocks));
    config.blockDim = dim3(kCtaSize);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    attr->id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr->val.programmaticStreamSerializationAllowed = pdl;
    config.numAttrs = 1;
    config.attrs = attr;
    return config;
}

void checkScale(at::Tensor const& t, at::Device device, bool scalar)
{
    TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == at::kFloat && t.is_contiguous()
            && t.numel() > 0 && (!scalar || t.numel() == 1),
        "scale/factor must be same-device contiguous FP32, scalar when requested");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0, "unaligned FP32 pointer");
}

bool overlaps(at::Tensor const& a, at::Tensor const& b)
{
    auto ap = reinterpret_cast<uintptr_t>(a.data_ptr()), bp = reinterpret_cast<uintptr_t>(b.data_ptr());
    return ap < bp + b.numel() * b.element_size() && bp < ap + a.numel() * a.element_size();
}

void checkIo(
    at::Tensor const& input, at::Tensor const& scale, at::Tensor const& factor, at::Tensor const& output, int64_t mode)
{
    TORCH_CHECK(mode >= 0 && mode <= 2, "mode must be0,1,2");
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 && input.is_contiguous() && input.dim() >= 2
            && input.numel() > 0,
        "input must be nonempty contiguous CUDA BF16, rank>=2");
    checkScale(scale, input.device(), true);
    checkScale(factor, input.device(), true);
    TORCH_CHECK(output.is_cuda() && output.device() == input.device()
            && output.scalar_type() == at::ScalarType::Float8_e4m3fn && output.is_contiguous()
            && output.sizes() == input.sizes(),
        "output must be same-shape contiguous CUDA E4M3FN");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(output.data_ptr()) % 8 == 0,
        "native Vec8 requires16B input and8B output alignment");
    TORCH_CHECK(!overlaps(input, output) && !overlaps(scale, output) && !overlaps(factor, output),
        "output may not overlap input/scale/factor");
}

void const* symbol(int64_t mode)
{
    switch (mode)
    {
    case 0: return reinterpret_cast<void const*>(h3QuantVec8<0>);
    case 1: return reinterpret_cast<void const*>(h3QuantVec8<1>);
    case 2: return reinterpret_cast<void const*>(h3QuantVec8<2>);
    case 3: return reinterpret_cast<void const*>(h3QuantFactor<true>);
    case 4: return reinterpret_cast<void const*>(h3QuantFactor<false>);
    default: TORCH_CHECK(false, "mode0..4 required");
    }
}
} // namespace

void quantOut(at::Tensor input, at::Tensor scale, at::Tensor factor, at::Tensor output, int64_t mode, bool pdl)
{
    checkIo(input, scale, factor, output, mode);
    c10::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr{};
    auto config = launchConfig(gridSize(input.numel()), stream, pdl, &attr);
    auto out = reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr());
    auto in = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr<at::BFloat16>());
    auto s = scale.data_ptr<float>();
    auto f = factor.data_ptr<float>();
    auto count = input.numel();
    if (mode == 0)
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, h3QuantVec8<0>, out, s, f, in, count));
    else if (mode == 1)
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, h3QuantVec8<1>, out, s, f, in, count));
    else
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, h3QuantVec8<2>, out, s, f, in, count));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor quantize(at::Tensor input, at::Tensor scale, at::Tensor factor, int64_t mode, bool pdl)
{
    auto output = at::empty(input.sizes(), input.options().dtype(at::ScalarType::Float8_e4m3fn));
    quantOut(input, scale, factor, output, mode, pdl);
    return output;
}

at::Tensor makeFactors(at::Tensor scales, bool exact_rn, bool pdl)
{
    checkScale(scales, scales.device(), false);
    c10::cuda::CUDAGuard guard(scales.device());
    auto out = at::empty_like(scales);
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr{};
    auto config = launchConfig((scales.numel() + kCtaSize - 1) / kCtaSize, stream, pdl, &attr);
    if (exact_rn)
        C10_CUDA_CHECK(cudaLaunchKernelEx(
            &config, h3QuantFactor<true>, out.data_ptr<float>(), scales.data_ptr<float>(), scales.numel()));
    else
        C10_CUDA_CHECK(cudaLaunchKernelEx(
            &config, h3QuantFactor<false>, out.data_ptr<float>(), scales.data_ptr<float>(), scales.numel()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<int64_t> quantInfo(int64_t mode, int64_t device)
{
    c10::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, device));
    void const* sym = symbol(mode);
    cudaFuncAttributes a{};
    cudaFunction_t f{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, sym));
    // CUDA runtime docs: cudaFunction_t and CUfunction are interoperable handles.
    C10_CUDA_CHECK(cudaGetFuncBySymbol(&f, sym));
    return {static_cast<int64_t>(reinterpret_cast<uintptr_t>(f)), a.numRegs, static_cast<int64_t>(a.localSizeBytes),
        a.maxThreadsPerBlock, static_cast<int64_t>(a.sharedSizeBytes), a.ptxVersion, a.binaryVersion};
}

std::vector<int64_t> quantPlan(int64_t numel, bool pdl)
{
    TORCH_CHECK(numel > 0, "numel must be positive");
    cudaLaunchAttribute attr{};
    auto config = launchConfig(gridSize(numel), nullptr, pdl, &attr);
    return {config.gridDim.x, config.blockDim.x, static_cast<int64_t>(config.dynamicSmemBytes),
        attr.val.programmaticStreamSerializationAllowed, numel / 8, numel % 8, int64_t(attr.id)};
}

// The launch geometry is supplied by the frozen U1 runtime.
void quantOutConfig(at::Tensor input, at::Tensor scale, at::Tensor factor, at::Tensor output, int64_t mode, bool pdl,
    int64_t blocks, int64_t threads)
{
    checkIo(input, scale, factor, output, mode);
    TORCH_CHECK(blocks > 0, "blocks must be positive");
    TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "threads must be128,256,512");
    c10::cuda::CUDAGuard guard(input.device());
    cudaFuncAttributes attrFn{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attrFn, symbol(mode)));
    int gridLimit = 0, threadLimit = 0, registerLimit = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&gridLimit, cudaDevAttrMaxGridDimX, input.get_device()));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&threadLimit, cudaDevAttrMaxThreadsPerBlock, input.get_device()));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&registerLimit, cudaDevAttrMaxRegistersPerBlock, input.get_device()));
    TORCH_CHECK(blocks <= gridLimit, "blocks exceeds actual device grid limit");
    TORCH_CHECK(threads <= threadLimit && threads <= attrFn.maxThreadsPerBlock,
        "threads exceeds actual device/function maximum");
    TORCH_CHECK(
        static_cast<int64_t>(attrFn.numRegs) * threads <= registerLimit, "register budget exceeds actual block limit");
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr{};
    // Reuse the original PDL launch config unchanged, then change only block.x.
    auto config = launchConfig(blocks, stream, pdl, &attr);
    config.blockDim = dim3(static_cast<unsigned>(threads));
    auto out = reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr());
    auto in = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr<at::BFloat16>());
    auto s = scale.data_ptr<float>();
    auto f = factor.data_ptr<float>();
    auto count = input.numel();
    if (mode == 0)
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, h3QuantVec8<0>, out, s, f, in, count));
    else if (mode == 1)
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, h3QuantVec8<1>, out, s, f, in, count));
    else
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, h3QuantVec8<2>, out, s, f, in, count));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor quantizeConfig(
    at::Tensor input, at::Tensor scale, at::Tensor factor, int64_t mode, bool pdl, int64_t blocks, int64_t threads)
{
    auto output = at::empty(input.sizes(), input.options().dtype(at::ScalarType::Float8_e4m3fn));
    quantOutConfig(input, scale, factor, output, mode, pdl, blocks, threads);
    return output;
}
