/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// BF16 -> NVFP4 activation quantization (E2M1 codes, one E4M3 scale per 16 elements, linear scale layout) that writes
// the bytes of invokeFP4Quantization<__nv_bfloat16, 16>(..., useUE8M0 = false, QuantizationSFLayout::LINEAR, ...) on
// finite inputs. Kimi K3 quantizes the input of every routed-expert layer this way (fp4_quantize(x, gs, 16, false,
// false) on the 3584-wide latent), and there the production kernel, one CTA per row, reaches only a fraction of the
// HBM speed of light.
//
// The kernel comes from an automated kernel search. Here it gains
// PDL, arch guards and any K that is a multiple of 16, and it takes the two reciprocals with rcp.approx.ftz exactly
// like the production kernel (the original refined them to correctly rounded values, which can differ in the last
// bit). It drops the original's evict_last hint on the prefill stores: the evict_first input loads already leave the
// output in L2, and the hint would keep up to an L2's worth of activations ahead of the following kernels' data.
//
// Rows are contiguous and K is a multiple of 16, so the tensor is a flat stream of 16-element blocks: block b reads
// input bytes [32 b, 32 b + 32) with one 256-bit load and writes output bytes [8 b, 8 b + 8) and scale byte b. A warp
// owns 32 * U consecutive blocks, lane-interleaved so that each load and store instruction of the warp is contiguous.
//
// Per block, as cvt_warp_fp16_to_fp4 computes it: amax = max |x|; SF = e4m3(gs * (amax * rcp(6))) (round to nearest
// even, saturating at 448); os = amax != 0 ? rcp(float(SF) * rcp(gs)) : 0 with rcp = rcp.approx.ftz; codes =
// e2m1(x * os) (round to nearest even, saturating at 6), the even element in the low nibble.

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/nvfp4QuantKf.h"
#include "tensorrt_llm/kernels/nvfp4QuantKfBlock.cuh"

#include <cuda_fp16.h>

#include <cstdint>
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

using nvfp4_quant_kf::kBlockElems;
using nvfp4_quant_kf::kBlockInputBytes;
constexpr uintptr_t kInputAlignment = 32;
constexpr uintptr_t kOutputAlignment = 8;

// Up to 1024 rows of the 3584-wide Kimi K3 MoE input the work is latency bound: one block per thread.
constexpr int64_t kDecodeMaxBlocks = 1024 * (3584 / kBlockElems);
constexpr int kDecodeThreads = 128;
// Above that, two blocks (64 B) in flight per thread at full occupancy (32 registers, eight 256-thread CTAs per SM).
constexpr int kPrefillThreads = 256;
constexpr int kPrefillUnroll = 2;
constexpr int kPrefillMinCtasPerSm = 8;

#if TLLM_NVFP4_QUANT_KF_SM10X

// Stores one quantized block: codes at output[block], its scale byte at scales[block].
__device__ __forceinline__ void storeBlock(nvfp4_quant_kf::QuantizedBlock const& q, uint2* __restrict__ output,
    uint8_t* __restrict__ scales, int block, int numBlocks)
{
    if (block < numBlocks)
    {
        output[block] = make_uint2(q.codesLo, q.codesHi);
        scales[block] = q.scale;
    }
}

#endif // TLLM_NVFP4_QUANT_KF_SM10X

template <int kUnroll, int kThreads, int kMinCtasPerSm>
__global__ void __launch_bounds__(kThreads, kMinCtasPerSm) nvfp4QuantKfKernel(uint8_t const* __restrict__ input,
    float const* __restrict__ globalScale, uint2* __restrict__ output, uint8_t* __restrict__ scales, int numBlocks)
{
#if TLLM_NVFP4_QUANT_KF_SM10X
    int const lane = static_cast<int>(threadIdx.x) & 31;
    int const firstBlock
        = ((static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x)) >> 5) * (32 * kUnroll) + lane;

    // The global scale is the next GEMM's input scale, a model constant that the previous kernel does not write: it
    // may be read before the wait, as quantize_with_block_size reads it (the compiler may still place the load after).
    float const gs = __ldg(globalScale);

    cudaGridDependencySynchronize();

    uint32_t r[kUnroll][8];
    if constexpr (kUnroll == 1)
    {
        // Straight-line code up to the stores: out-of-range lanes load a valid (clamped) block and skip the stores.
        nvfp4_quant_kf::loadBlock(input + static_cast<size_t>(min(firstBlock, numBlocks - 1)) * kBlockInputBytes, r[0]);
    }
    else
    {
        // Blocks past the end (in the grid's last CTA only) are not loaded: quantizeBlock runs on their unwritten
        // registers and skips their stores. Clamped loads as above make this path slower in an sm_100f build.
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
        {
            int const block = firstBlock + u * 32;
            if (block < numBlocks)
            {
                nvfp4_quant_kf::loadBlock(input + static_cast<size_t>(block) * kBlockInputBytes, r[u]);
            }
        }
    }
    nvfp4_quant_kf::BlockConsts const consts = nvfp4_quant_kf::makeBlockConsts(gs);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
    {
        storeBlock(nvfp4_quant_kf::quantizeBlock(r[u], consts), output, scales, firstBlock + u * 32, numBlocks);
    }
    // No explicit PDL trigger: the dependents launch when the grid completes. A trigger would only launch them sooner:
    // their griddepcontrol.wait returns once the grid has completed and its stores are visible either way, and
    // the fence and CTA barrier that the production kernel puts before its trigger cost noticeably here.
#endif
}

template <int kUnroll, int kThreads, int kMinCtasPerSm>
void launch(uint8_t const* input, float const* globalScale, uint8_t* output, uint8_t* scales, int numBlocks,
    cudaStream_t stream)
{
    constexpr int kBlocksPerCta = kThreads * kUnroll;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3((numBlocks + kBlocksPerCta - 1) / kBlocksPerCta);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, nvfp4QuantKfKernel<kUnroll, kThreads, kMinCtasPerSm>, input,
        globalScale, reinterpret_cast<uint2*>(output), scales, numBlocks));
}

} // namespace

bool isNvfp4QuantKfSupported(void const* input, void const* output, int64_t m, int64_t n)
{
    if (m <= 0 || n <= 0 || n % kBlockElems != 0)
    {
        return false;
    }
    // Block indices and the grid stay in int32, with room for the last CTA's unrolled blocks.
    int64_t const maxBlocks = std::numeric_limits<int32_t>::max() - kPrefillThreads * kPrefillUnroll;
    if (m > maxBlocks / (n / kBlockElems))
    {
        return false;
    }
    int const smVersion = tensorrt_llm::common::getSMVersion();
    return smVersion >= 100 && smVersion < 110 && reinterpret_cast<uintptr_t>(input) % kInputAlignment == 0
        && reinterpret_cast<uintptr_t>(output) % kOutputAlignment == 0;
}

void invokeNvfp4QuantKf(__nv_bfloat16 const* input, float const* globalScale, uint8_t* output, uint8_t* scales,
    int64_t m, int64_t n, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(isNvfp4QuantKfSupported(input, output, m, n),
        "invokeNvfp4QuantKf: needs SM10x, n %% 16 == 0, m * n / 16 within int32, a 32-byte aligned input and an "
        "8-byte aligned output (m = %ld, n = %ld)",
        static_cast<long>(m), static_cast<long>(n));
    TLLM_CHECK_WITH_INFO(globalScale != nullptr, "invokeNvfp4QuantKf: the global scale is required");
    int64_t const numBlocks = m * (n / kBlockElems);
    auto const* in = reinterpret_cast<uint8_t const*>(input);
    if (numBlocks <= kDecodeMaxBlocks)
    {
        launch<1, kDecodeThreads, 1>(in, globalScale, output, scales, static_cast<int>(numBlocks), stream);
    }
    else
    {
        launch<kPrefillUnroll, kPrefillThreads, kPrefillMinCtasPerSm>(
            in, globalScale, output, scales, static_cast<int>(numBlocks), stream);
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
