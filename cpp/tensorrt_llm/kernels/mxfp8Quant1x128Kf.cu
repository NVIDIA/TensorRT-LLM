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

// BF16 -> MXFP8 activation quantization with 1x128 UE8M0 scales in the R128c4 layout, byte-identical to
// invokeMxFP8Quantization<__nv_bfloat16, 128, 32>(..., QuantizationSFLayout::SWIZZLED, ...) on finite inputs.
//
// The kernel comes from an automated kernel search against a byte-exact PyTorch reference of
// fp8_quantize_1x128_packed_ue8m0(x, True). Here it gains PDL, arch guards and support for any K that is a multiple of
// 128, and converts to FP8 from FP32 instead of BF16x2 (same bytes; not every toolchain accepts the BF16x2 source
// form). It drops the original's evict_last store hint: worth little in isolation, but it would keep up to an L2's
// worth of activations ahead of the following kernels' data. The outputs were re-validated byte for byte against the
// production kernel on Rubin.
//
// The work is tiled like the scale layout [ceil(M / 128), KB, 32, 4, 4]: a work item is the row32-th row of the four
// 32-row sub-tiles of a 128-row tile, for a pair of adjacent K128 blocks (8 blocks of 128 elements). Four lanes own a
// block, 32 contiguous elements each (two 32-byte loads, one 32-byte store), so the block amax takes two xor-shuffles.
// A warp takes two adjacent work items, so each half-warp ends with the four sub-tile scales of two adjacent row32
// records of one K128 block: 32 contiguous scale bytes, written with one store. (Variants that walk the blocks in
// row-major order and store each scale as its own 4-byte word get noticeably less of the HBM speed of light.) Rows
// past M quantize nothing and get zero scales.
//
// Per block: amax = max |x| (at least 1e-10); sf = amax * R with R the FP32 constant 0x3B124925 (rcp.approx(448));
// e8 = the biased exponent of sf rounded up, capped at 254; q = E4M3(rn, satfinite) of x * 2^(127 - e8). The scaling
// is exact in BF16 for finite inputs (a power of two; results below the BF16 normal range are far below the E4M3 range
// and round to a signed zero either way) and BF16 -> FP32 is exact, so the FP8 conversion is the only rounding, as in
// the reference. Each scale byte is replicated into the block's four K32 slots.

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/mxfp8Quant1x128Kf.h"

#include <cstdint>
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kThreads = 256;
constexpr int kItemsPerWarp = 2;                            // adjacent row32 records, one 32-byte scale store
constexpr int kItemsPerCta = kThreads / 32 * kItemsPerWarp; // 16
constexpr int kAlignment = 32;                              // bytes, for the 32-byte loads and stores

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)

constexpr uint32_t kFullMask = 0xffffffffu;
constexpr uint32_t kRcp448Bits = 0x3B124925u; // rcp.approx(448.0f)

__device__ __forceinline__ void loadStreaming256(void const* ptr, uint32_t* r)
{
    asm volatile("ld.global.nc.L1::no_allocate.L2::evict_first.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "l"(ptr));
}

__device__ __forceinline__ void store256(void* ptr, uint32_t const* r)
{
    asm volatile("st.global.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(ptr), "r"(r[0]), "r"(r[1]), "r"(r[2]),
                 "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])
                 : "memory");
}

// max(|a|, |b|) per BF16 lane; the sign bits of the result are the xor of the input signs and are masked later.
__device__ __forceinline__ uint32_t absMaxBf16x2(uint32_t a, uint32_t b)
{
    uint32_t d;
    asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ uint32_t mulBf16x2(uint32_t a, uint32_t b)
{
    uint32_t d;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

// Four BF16 values (a.lo, a.hi, b.lo, b.hi) -> four E4M3 bytes in that order.
__device__ __forceinline__ uint32_t toE4m3x4(uint32_t a, uint32_t b)
{
    uint16_t lo;
    uint16_t hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(lo)
        : "f"(__uint_as_float(a & 0xffff0000u)), "f"(__uint_as_float(a << 16)));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(hi)
        : "f"(__uint_as_float(b & 0xffff0000u)), "f"(__uint_as_float(b << 16)));
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// UE8M0 exponent of a block whose amax has the FP32 bit pattern amaxBits.
__device__ __forceinline__ uint32_t ue8m0Exponent(uint32_t amaxBits)
{
    float const amax = fmaxf(__uint_as_float(amaxBits), 1e-10f);
    uint32_t const sfBits = __float_as_uint(__fmul_rn(amax, __uint_as_float(kRcp448Bits)));
    return min(((sfBits >> 23) & 0xffu) + ((sfBits & 0x7fffffu) != 0u ? 1u : 0u), 254u);
}

// BF16 bits of 2^(127 - e8) in both halves (e8 == 254: the BF16 subnormal 2^-127).
__device__ __forceinline__ uint32_t inverseScaleBf16x2(uint32_t e8)
{
    uint32_t const bf16 = e8 == 254u ? 0x0040u : ((254u - e8) << 7);
    return bf16 * 0x00010001u;
}

#endif // SM10x

__global__ void __launch_bounds__(kThreads) mxfp8Quant1x128KfKernel(uint16_t const* __restrict__ input,
    uint8_t* __restrict__ output, uint8_t* __restrict__ scales, int numRows, int numKBlocks)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100) // SM10x
    int const numKPairs = (numKBlocks + 1) / 2;
    int64_t const rowElements = static_cast<int64_t>(numKBlocks) * 128;
    int const lane = static_cast<int>(threadIdx.x) & 31;
    int const group = lane >> 2; // block of the work item: sub-tile (group & 3), K128 block parity (group >> 2)
    int const sub = lane & 3;
    int const firstItem
        = static_cast<int>(blockIdx.x) * kItemsPerCta + (static_cast<int>(threadIdx.x) >> 5) * kItemsPerWarp;
    int const numItems = (numRows + 127) / 128 * numKPairs * 32;

    cudaGridDependencySynchronize();

    uint32_t vals[kItemsPerWarp][16];
    int64_t offsets[kItemsPerWarp];
    bool valid[kItemsPerWarp];
#pragma unroll
    for (int s = 0; s < kItemsPerWarp; ++s)
    {
        int const item = firstItem + s;
        int const tile = item >> 5;
        int const kPair = tile % numKPairs;
        int const row = (tile / numKPairs) * 128 + (item & 31) + ((group & 3) << 5);
        int const kBlock = 2 * kPair + (group >> 2);
        valid[s] = item < numItems && row < numRows && kBlock < numKBlocks;
        offsets[s] = row * rowElements + kBlock * 128 + sub * 32;
        if (valid[s])
        {
            loadStreaming256(input + offsets[s], vals[s]);
            loadStreaming256(input + offsets[s] + 16, vals[s] + 8);
        }
        else
        {
#pragma unroll
            for (int j = 0; j < 16; ++j)
            {
                vals[s][j] = 0u;
            }
        }
    }

    uint32_t e8s[kItemsPerWarp];
#pragma unroll
    for (int s = 0; s < kItemsPerWarp; ++s)
    {
        uint32_t m = absMaxBf16x2(vals[s][0], vals[s][1]);
#pragma unroll
        for (int j = 2; j < 16; ++j)
        {
            m = absMaxBf16x2(m, vals[s][j]);
        }
        m = absMaxBf16x2(m, __shfl_xor_sync(kFullMask, m, 1));
        m = absMaxBf16x2(m, __shfl_xor_sync(kFullMask, m, 2));
        uint32_t const amaxBits = max((m << 16) & 0x7fff0000u, m & 0x7fff0000u);
        e8s[s] = valid[s] ? ue8m0Exponent(amaxBits) : 0u;
        if (valid[s])
        {
            uint32_t const invScale = inverseScaleBf16x2(e8s[s]);
            uint32_t q[8];
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                q[j] = toE4m3x4(mulBf16x2(vals[s][2 * j], invScale), mulBf16x2(vals[s][2 * j + 1], invScale));
            }
            store256(output + offsets[s], q);
        }
    }

    // Each half-warp gathers the four sub-tile scales of its K128 block for both row32 records: 32 contiguous bytes of
    // the scale layout, zeros for rows past M. When K / 128 is odd, the half-warp of the missing last block stores
    // nothing.
    uint32_t scaleWords[8];
#pragma unroll
    for (int s = 0; s < kItemsPerWarp; ++s)
    {
#pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            scaleWords[4 * s + g] = __shfl_sync(kFullMask, e8s[s], 4 * g, 16) * 0x01010101u;
        }
    }
    if ((lane & 15) == 0)
    {
        int const tile = firstItem >> 5;
        int const kBlock = 2 * (tile % numKPairs) + (lane >> 4);
        if (firstItem < numItems && kBlock < numKBlocks)
        {
            int64_t const record
                = (static_cast<int64_t>(tile / numKPairs) * numKBlocks + kBlock) * 32 + (firstItem & 31);
            store256(scales + record * 16, scaleWords); // 16 bytes = the four sub-tile words of one row32 record
        }
    }
    // No explicit PDL trigger: the dependents launch when the grid completes. A trigger would only launch them sooner:
    // their griddepcontrol.wait returns once the grid has completed and its stores are visible either way, and
    // the fence and CTA barrier that the production kernel puts before its trigger cost noticeably here.
#endif
}

} // namespace

bool isMxFP8Quant1x128KfSupported(void const* input, void const* output, void const* scales, int64_t m, int64_t n)
{
    if (m <= 0 || n <= 0 || n % 128 != 0 || n > std::numeric_limits<int32_t>::max())
    {
        return false;
    }
    int64_t const numItems = (m + 127) / 128 * ((n / 128 + 1) / 2) * 32;
    int const smVersion = tensorrt_llm::common::getSMVersion();
    return smVersion >= 100 && smVersion < 110 && m <= std::numeric_limits<int32_t>::max() - 127
        && numItems <= std::numeric_limits<int32_t>::max() && reinterpret_cast<uintptr_t>(input) % kAlignment == 0
        && reinterpret_cast<uintptr_t>(output) % kAlignment == 0
        && reinterpret_cast<uintptr_t>(scales) % kAlignment == 0;
}

void invokeMxFP8Quant1x128Kf(
    __nv_bfloat16 const* input, uint8_t* output, uint8_t* scales, int64_t m, int64_t n, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(isMxFP8Quant1x128KfSupported(input, output, scales, m, n),
        "invokeMxFP8Quant1x128Kf: needs SM10x, n %% 128 == 0, m and n within int32 and 32-byte aligned input, output "
        "and "
        "scales (m = %ld, n = %ld)",
        static_cast<long>(m), static_cast<long>(n));
    int const numRows = static_cast<int>(m);
    int const numKBlocks = static_cast<int>(n / 128);
    int const numItems = (numRows + 127) / 128 * ((numKBlocks + 1) / 2) * 32;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(numItems / kItemsPerCta);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, mxfp8Quant1x128KfKernel, reinterpret_cast<uint16_t const*>(input),
        output, scales, numRows, numKBlocks));
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
