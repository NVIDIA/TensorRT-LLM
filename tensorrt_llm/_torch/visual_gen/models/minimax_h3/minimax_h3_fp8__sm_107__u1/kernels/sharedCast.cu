// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
// Exact BF16-to-FP32 conversion, preserving Torch .float() bit patterns.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>
#include <vector>

namespace
{
template <int Mode>
__global__ void h3CastVec8(float* output, __nv_bfloat16 const* input, int64_t count)
{
    int64_t const first = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t const vectors = count / 8;
    for (int64_t v = first; v < vectors; v += stride)
    {
        int64_t const base = v * 8;
        if constexpr (Mode == 0)
        {
            // Scalar intrinsic control in the SAME Vec8 ownership/grid scaffold.
#pragma unroll
            for (int j = 0; j < 8; ++j)
                output[base + j] = __bfloat162float(input[base + j]);
        }
        else
        {
            uint4 raw = *reinterpret_cast<uint4 const*>(input + base);
            auto const* pair = reinterpret_cast<__nv_bfloat162 const*>(&raw);
            float2 p0 = __bfloat1622float2(pair[0]);
            float2 p1 = __bfloat1622float2(pair[1]);
            float2 p2 = __bfloat1622float2(pair[2]);
            float2 p3 = __bfloat1622float2(pair[3]);
            *reinterpret_cast<float4*>(output + base) = make_float4(p0.x, p0.y, p1.x, p1.y);
            *reinterpret_cast<float4*>(output + base + 4) = make_float4(p2.x, p2.y, p3.x, p3.y);
        }
    }
    // The same scalar intrinsic handles all0..7 tail elements; no OOB vectorload.
    for (int64_t i = vectors * 8 + first; i < count; i += stride)
        output[i] = __bfloat162float(input[i]);
}

bool overlaps(at::Tensor const& a, at::Tensor const& b)
{
    auto ap = reinterpret_cast<uintptr_t>(a.data_ptr()), bp = reinterpret_cast<uintptr_t>(b.data_ptr());
    return ap < bp + b.numel() * b.element_size() && bp < ap + a.numel() * a.element_size();
}

void checkIo(at::Tensor const& input, at::Tensor const& output, int64_t mode)
{
    TORCH_CHECK(mode == 0 || mode == 1, "cast mode must be0 or1");
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 && input.is_contiguous() && input.dim() >= 1
            && input.numel() > 0,
        "cast input must be nonempty contiguous CUDA BF16 rank>=1");
    TORCH_CHECK(output.is_cuda() && output.device() == input.device() && output.scalar_type() == at::kFloat
            && output.is_contiguous() && output.sizes() == input.sizes(),
        "cast output must be same-shape contiguous same-device CUDA FP32");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
        "cast pointers need16B alignment");
    TORCH_CHECK(!overlaps(input, output), "cast output must not overlap BF16 input");
}

void const* symbol(int64_t mode)
{
    TORCH_CHECK(mode == 0 || mode == 1, "cast mode must be0 or1");
    return mode == 0 ? reinterpret_cast<void const*>(h3CastVec8<0>) : reinterpret_cast<void const*>(h3CastVec8<1>);
}
} // namespace

void castOut(at::Tensor input, at::Tensor output, int64_t mode, int64_t blocks, int64_t threads)
{
    checkIo(input, output, mode);
    TORCH_CHECK(blocks > 0, "cast blocks must be positive");
    TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "cast threads must be128/256/512");
    c10::cuda::CUDAGuard guard(input.device());
    cudaFuncAttributes attr{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, symbol(mode)));
    int gridLimit = 0, threadLimit = 0, registerLimit = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&gridLimit, cudaDevAttrMaxGridDimX, input.get_device()));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&threadLimit, cudaDevAttrMaxThreadsPerBlock, input.get_device()));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&registerLimit, cudaDevAttrMaxRegistersPerBlock, input.get_device()));
    TORCH_CHECK(blocks <= gridLimit, "cast grid exceeds actual device limit");
    TORCH_CHECK(threads <= threadLimit && threads <= attr.maxThreadsPerBlock, "cast threads exceeds actual limit");
    TORCH_CHECK(attr.numRegs > 0 && int64_t(attr.numRegs) * threads <= registerLimit,
        "cast register budget exceeds block limit");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto in = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr<at::BFloat16>());
    auto out = output.data_ptr<float>();
    auto count = input.numel();
    // Ordinary current-stream copy, with NO PDL launch attribute or device hooks.
    if (mode == 0)
        h3CastVec8<0><<<blocks, threads, 0, stream>>>(out, in, count);
    else
        h3CastVec8<1><<<blocks, threads, 0, stream>>>(out, in, count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor cast(at::Tensor input, int64_t mode, int64_t blocks, int64_t threads)
{
    auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat));
    castOut(input, output, mode, blocks, threads);
    return output;
}

std::vector<int64_t> castInfo(int64_t mode, int64_t device)
{
    int count = 0;
    C10_CUDA_CHECK(cudaGetDeviceCount(&count));
    TORCH_CHECK(device >= 0 && device < count && device <= std::numeric_limits<c10::DeviceIndex>::max(),
        "invalid cast info device");
    c10::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device)));
    cudaFuncAttributes attr{};
    cudaFunction_t function{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, symbol(mode)));
    C10_CUDA_CHECK(cudaGetFuncBySymbol(&function, symbol(mode)));
    return {static_cast<int64_t>(reinterpret_cast<uintptr_t>(function)), attr.numRegs,
        static_cast<int64_t>(attr.localSizeBytes), attr.maxThreadsPerBlock, static_cast<int64_t>(attr.sharedSizeBytes),
        attr.ptxVersion, attr.binaryVersion};
}
