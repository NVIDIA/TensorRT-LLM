/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/fusedMoeCommKernels.h"

#include <type_traits>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/quantization.cuh"

namespace tensorrt_llm
{
namespace kernels
{

using tensorrt_llm::common::launchWithPdlWhenEnabled;

// Quantize a contiguous shared-memory buffer containing elements of DType into NVFP4 with per-16-element FP8 scales.
// Output layout (repeated per 16-element group per lane), followed by one global scale float:
//   [WARP_SIZE * 8 bytes packed e2m1 values] [WARP_SIZE * 1 byte E4M3 per-group scales] ... [global_scale (4 bytes)]
// Each lane writes one 64-bit packed e2m1 for its 16 values and one 1-byte E4M3 scale per group.
// Global scale is computed as (448*6)/absmax and written once at the end of the buffer.
template <typename DType>
__device__ __forceinline__ void quantize_nvfp4_sharedmem(uint8_t* compact_ptr, int sizeInBytes, int laneId)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    int const numElems = sizeInBytes / sizeof(DType);
    assert(numElems % 2 == 0);
    if (numElems <= 0)
    {
        return;
    }

    DType const* in = reinterpret_cast<DType const*>(compact_ptr);

    // 1) Global absmax across the field (warp reduce) in original dtype precision when possible
    float threadMaxFloat = 0.f;
    if constexpr (std::is_same_v<DType, half> || std::is_same_v<DType, __nv_bfloat16>)
    {
        using DType2 = typename tensorrt_llm::common::packed_as<DType, 2>::type;
        DType2 const* in2 = reinterpret_cast<DType2 const*>(in);
        int const numPairs = numElems / 2;

        // Initialize to zero to avoid a concentrated shared-memory read from index 0 across all lanes
        DType2 localMax2;
        localMax2.x = DType(0.);
        localMax2.y = DType(0.);
        // stride over pairs
        for (int i = laneId; i < numPairs; i += WARP_SIZE)
        {
            DType2 v2 = in2[i];
            localMax2 = tensorrt_llm::common::cuda_max(localMax2, tensorrt_llm::common::cuda_abs(v2));
        }
        // Reduce vector to scalar float in-thread
        DType localMax = tensorrt_llm::common::cuda_max<DType, DType2>(localMax2);
        threadMaxFloat = tensorrt_llm::common::cuda_cast<float>(localMax);
    }
    else
    {
        float localMax = 0.f;
        for (int i = laneId; i < numElems; i += WARP_SIZE)
        {
            float v = fabsf(tensorrt_llm::common::cuda_cast<float>(in[i]));
            localMax = fmaxf(localMax, v);
        }
        threadMaxFloat = localMax;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        threadMaxFloat = fmaxf(threadMaxFloat, __shfl_xor_sync(0xffffffff, threadMaxFloat, offset));
    }
    float const eps = 1e-12f;
    float const globalAbsMax = fmaxf(threadMaxFloat, eps);

    // 2) Global scale
    float const SFScaleVal = (448.0f * 6.0f) * (1.0f / globalAbsMax);

    // 3) Output layout
    int const numGroups = (numElems + WARP_SIZE * 16 - 1) / (WARP_SIZE * 16);

    // 8 bytes for e2m1, 1 byte for scale
    int const outputBlockSizeInBytes = 8 * WARP_SIZE + WARP_SIZE;
    uint8_t* const globalScaleOutBytes = compact_ptr + numGroups * outputBlockSizeInBytes;

    // 4) Per-16 group quantization
    int const swizzle_idy = laneId / 4;
    int const swizzle_idx = (laneId % 4) * 8;

    for (int groupId = 0; groupId < numGroups; groupId++)
    {
        int groupStart = groupId * (WARP_SIZE * 16);
        float vecMax = 0.f;
        float2 raw[8];

        if constexpr (std::is_same_v<DType, half> || std::is_same_v<DType, __nv_bfloat16>)
        {
            using DType2 = typename tensorrt_llm::common::packed_as<DType, 2>::type;
            int const numPairs = numElems / 2;
            DType2 const* in2Ptr = reinterpret_cast<DType2 const*>(in);
            int const pairBase = groupStart >> 1;

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                int const pi = pairBase + swizzle_idy * 32 + swizzle_idx + (i + swizzle_idy) % 8;
                if (pi < numPairs)
                {
                    DType2 v2 = in2Ptr[pi];
                    float x = tensorrt_llm::common::cuda_cast<float>(v2.x);
                    float y = tensorrt_llm::common::cuda_cast<float>(v2.y);
                    raw[i] = make_float2(x, y);
                    vecMax = fmaxf(vecMax, fmaxf(fabsf(x), fabsf(y)));
                }
                else
                {
                    raw[i] = make_float2(0.0f, 0.0f);
                }
            }
        }
        else
        {
            groupStart += laneId * 16;
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                int idx = groupStart + (i << 1);
                if (idx < numElems)
                {
                    float x = tensorrt_llm::common::cuda_cast<float>(in[idx]);
                    float y = (idx + 1 < numElems) ? tensorrt_llm::common::cuda_cast<float>(in[idx + 1]) : 0.0f;
                    raw[i] = make_float2(x, y);
                    vecMax = fmaxf(vecMax, fmaxf(fabsf(x), fabsf(y)));
                }
                else
                {
                    raw[i] = make_float2(0.0f, 0.0f);
                }
            }
        }

        // SF from vecMax and global scale; write as E4M3
        float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
        __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(SFValue);
        float SFValueNarrow = static_cast<float>(sf8);
        float const outputScale = (vecMax != 0.f)
            ? reciprocal_approximate_ftz(SFValueNarrow * reciprocal_approximate_ftz(SFScaleVal))
            : 0.0f;

        // Pack 16 values -> 8 bytes e2m1 (use raw[] read above to avoid a second shared-memory read)
        float2 fp2Vals[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            fp2Vals[i] = make_float2(raw[i].x * outputScale, raw[i].y * outputScale);
        }
        uint64_t const e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

        uint8_t* const outValPtr = compact_ptr + groupId * outputBlockSizeInBytes + laneId * sizeof(uint64_t);
        uint8_t* const outScalePtr
            = compact_ptr + groupId * outputBlockSizeInBytes + WARP_SIZE * sizeof(uint64_t) + laneId * sizeof(uint8_t);

        if (laneId < 16)
        {
            reinterpret_cast<uint64_t*>(outValPtr)[0] = e2m1Vec;
        }
        __syncwarp();
        if (laneId >= 16)
        {
            reinterpret_cast<uint64_t*>(outValPtr)[0] = e2m1Vec;
        }
        outScalePtr[0] = sf8.__x;
    }

    // Store global scale (fp32) once with a single 32-bit store. Use lane 0 to avoid races.
    if (laneId == 0)
    {
        *reinterpret_cast<float*>(globalScaleOutBytes) = SFScaleVal;
    }
#endif
}

// Convert one lane's packed 16 e2m1 values (in a 64-bit word) into eight float2 values (16 floats).
// Uses 8 cvt.rn.f16x2.e2m1x2 instructions, one per input byte, to produce eight half2 which are cast to float2.
inline __device__ void e2m1_to_fp32_vec(uint64_t e2m1Vec, float2 (&array)[8])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t out_fp16[8];
    asm volatile(
        "{\n"
        ".reg .b8 b0;\n"
        ".reg .b8 b1;\n"
        ".reg .b8 b2;\n"
        ".reg .b8 b3;\n"
        ".reg .b8 b4;\n"
        ".reg .b8 b5;\n"
        ".reg .b8 b6;\n"
        ".reg .b8 b7;\n"
        ".reg .b32 lo;\n"
        ".reg .b32 hi;\n"
        "mov.b64 {lo, hi}, %8;\n"
        "mov.b32 {b0, b1, b2, b3}, lo;\n"
        "mov.b32 {b4, b5, b6, b7}, hi;\n"
        "cvt.rn.f16x2.e2m1x2   %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2   %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2   %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2   %3, b3;\n"
        "cvt.rn.f16x2.e2m1x2   %4, b4;\n"
        "cvt.rn.f16x2.e2m1x2   %5, b5;\n"
        "cvt.rn.f16x2.e2m1x2   %6, b6;\n"
        "cvt.rn.f16x2.e2m1x2   %7, b7;\n"
        "}"
        : "=r"(out_fp16[0]), "=r"(out_fp16[1]), "=r"(out_fp16[2]), "=r"(out_fp16[3]), "=r"(out_fp16[4]),
        "=r"(out_fp16[5]), "=r"(out_fp16[6]), "=r"(out_fp16[7])
        : "l"(e2m1Vec));

    array[0] = __half22float2(reinterpret_cast<__half2&>(out_fp16[0]));
    array[1] = __half22float2(reinterpret_cast<__half2&>(out_fp16[1]));
    array[2] = __half22float2(reinterpret_cast<__half2&>(out_fp16[2]));
    array[3] = __half22float2(reinterpret_cast<__half2&>(out_fp16[3]));
    array[4] = __half22float2(reinterpret_cast<__half2&>(out_fp16[4]));
    array[5] = __half22float2(reinterpret_cast<__half2&>(out_fp16[5]));
    array[6] = __half22float2(reinterpret_cast<__half2&>(out_fp16[6]));
    array[7] = __half22float2(reinterpret_cast<__half2&>(out_fp16[7]));
#endif
}

template <typename DType>
__device__ __forceinline__ void dequantize_nvfp4_sharedmem(uint8_t* compact_ptr, int sizeInBytes, int laneId)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    int const numElems = sizeInBytes / sizeof(DType);
    if (numElems <= 0)
    {
        return;
    }

    int const numGroups = (numElems + WARP_SIZE * 16 - 1) / (WARP_SIZE * 16);

    // New layout matches quantize: per-group blocks of [8*WARP_SIZE bytes values][WARP_SIZE bytes scales],
    // followed by a single 4-byte global scale at the end.
    int const inputBlockSizeInBytes = 8 * WARP_SIZE + WARP_SIZE;
    uint8_t* const globalScaleOutBytes = compact_ptr + numGroups * inputBlockSizeInBytes;
    float const SFScaleVal = reciprocal_approximate_ftz(*reinterpret_cast<float const*>(globalScaleOutBytes));
    __syncwarp();

    DType* out = reinterpret_cast<DType*>(compact_ptr);

    // Process groups in reverse order to avoid overwriting packed input before it is read
    for (int groupId = numGroups - 1; groupId >= 0; --groupId)
    {
        int const groupStart = laneId * 16 + groupId * (WARP_SIZE * 16);
        // Conflict-free read of packed 64-bit e2m1 values from shared memory:
        // serialize half-warps to avoid lane i and i+16 hitting the same bank in the same cycle.
        uint8_t const* const valBase = compact_ptr + groupId * inputBlockSizeInBytes;
        uint64_t packed = 0ull;
        if (laneId < 16)
        {
            packed = reinterpret_cast<uint64_t const*>(valBase)[laneId];
        }
        __syncwarp();
        if (laneId >= 16)
        {
            packed = reinterpret_cast<uint64_t const*>(valBase)[laneId];
        }

        // Read per-lane 1-byte scales to match quantize access pattern
        uint8_t const* const scalesBase = compact_ptr + groupId * inputBlockSizeInBytes + WARP_SIZE * sizeof(uint64_t);
        uint8_t sfByte = scalesBase[laneId];
        __nv_fp8_e4m3 sf8;
        sf8.__x = sfByte;
        float const SFValueNarrow = static_cast<float>(sf8);
        float const dequantScale = SFScaleVal * SFValueNarrow;
        __syncwarp();

        float2 tmp[8];
        e2m1_to_fp32_vec(packed, tmp);

        // Vectorized stores with swizzle to avoid bank conflicts, matching quantize path
        if constexpr (std::is_same_v<DType, half> || std::is_same_v<DType, __nv_bfloat16>)
        {
            using DType2 = typename tensorrt_llm::common::packed_as<DType, 2>::type;
            DType2* out2 = reinterpret_cast<DType2*>(out);
            int const numPairs = numElems / 2;
            int const pairBase = (groupId * (WARP_SIZE * 16)) >> 1;
            int const swizzle_idy = laneId / 4;
            int const swizzle_idx = (laneId % 4) * 8;

#pragma unroll
            for (int t = 0; t < 8; ++t)
            {
                int const pi = pairBase + swizzle_idy * 32 + swizzle_idx + (t + swizzle_idy) % 8;
                if (pi < numPairs)
                {
                    DType2 v2;
                    v2.x = tensorrt_llm::common::cuda_cast<DType>(tmp[t].x * dequantScale);
                    v2.y = tensorrt_llm::common::cuda_cast<DType>(tmp[t].y * dequantScale);
                    out2[pi] = v2;
                }
            }
        }
        else
        {
            // Fallback linear layout for non-16-bit types
#pragma unroll
            for (int t = 0; t < 8; ++t)
            {
                int idx0 = groupStart + (t << 1);
                if (idx0 < numElems)
                {
                    using DType2 = typename tensorrt_llm::common::packed_as<DType, 2>::type;
                    DType2 v2;
                    v2.x = tensorrt_llm::common::cuda_cast<DType>(tmp[t].x * dequantScale);
                    v2.y = tensorrt_llm::common::cuda_cast<DType>(tmp[t].y * dequantScale);
                    reinterpret_cast<DType2*>(out + idx0)[0] = v2;
                }
            }
        }
        __syncwarp();
    }
#endif
}

static __device__ __forceinline__ uint32_t __as_ptr_smem(void const* __ptr)
{
    // Consider adding debug asserts here.
    return static_cast<uint32_t>(__cvta_generic_to_shared(__ptr));
}

static __device__ __forceinline__ uint64_t __as_ptr_gmem(void const* __ptr)
{
    // Consider adding debug asserts here.
    return static_cast<uint64_t>(__cvta_generic_to_global(__ptr));
}

__device__ __forceinline__ void fence_release_sys()
{
    asm volatile("fence.release.sys;" : : : "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t* addr, uint32_t const& count)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm("mbarrier.init.shared.b64 [%0], %1;" : : "r"(__as_ptr_smem(addr)), "r"(count) : "memory");
#endif
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* addr, const uint32_t txCount)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :
        : "r"(__as_ptr_smem(addr)), "r"(txCount)
        : "memory");
#endif
}

__device__ __forceinline__ uint64_t mbarrier_arrive(uint64_t* addr)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    uint64_t state;
    asm("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(__as_ptr_smem(addr)) : "memory");
    return state;
#else
    return 0;
#endif
}

__device__ __forceinline__ uint64_t mbarrier_arrive_expect_tx(uint64_t* addr, const uint32_t txCount)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    uint64_t state;
    asm("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
        : "=l"(state)
        : "r"(__as_ptr_smem(addr)), "r"(txCount)
        : "memory");
    return state;
#else
    return 0;
#endif
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint64_t* addr, uint32_t const& phaseParity)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    uint32_t waitComplete;
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(waitComplete)
        : "r"(__as_ptr_smem(addr)), "r"(phaseParity)
        : "memory");
    return static_cast<bool>(waitComplete);
#else
    return false;
#endif
}

template <int COPY_SIZE = 4>
__device__ __forceinline__ void ldgsts(int* dstShm, int const* srcMem, bool predGuard)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int) predGuard),
        "r"(__as_ptr_smem(dstShm)), "l"(__as_ptr_gmem(srcMem)), "n"(COPY_SIZE));
#endif
}

__device__ __forceinline__ void cp_async_commit_group()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" : : :);
#endif
}

template <int N = 0>
__device__ __forceinline__ void cp_async_wait_group()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" : : "n"(N) : "memory");
#endif
}

__device__ __forceinline__ void cp_async_bulk_g2s(void* dstMem, void const* srcMem, int copySize, uint64_t* smemBar)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(dstMem)), "l"(__as_ptr_gmem(srcMem)), "r"(copySize), "r"(__as_ptr_smem(smemBar))
        : "memory");
#endif
}

__device__ __forceinline__ void cp_async_bulk_s2g(void* dstMem, void const* srcMem, int copySize)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(dstMem)), "r"(__as_ptr_smem(srcMem)), "r"(copySize)
        : "memory");
#endif
}

__device__ __forceinline__ void cp_async_bulk_commit_group()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.commit_group;" : : :);
#endif
}

template <int N = 0>
__device__ __forceinline__ void cp_async_bulk_wait_group()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
#endif
}

template <int N = 0>
__device__ __forceinline__ void cp_async_bulk_wait_group_read()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(N) : "memory");
#endif
}

__host__ void MoeCommFieldInfo::fillFieldInfo(
    uint8_t* dataPtr, size_t elementSize, int vectorSize, int stride, cudaDataType_t dataType)
{
    TLLM_CHECK(elementSize == 1 || elementSize == 2 || elementSize == 4 || elementSize == 8 || elementSize == 16);

    dataPtrBase = dataPtr;

    uint64_t dataPtrU64 = reinterpret_cast<uint64_t>(dataPtr);

    while (elementSize < 16 && dataPtrU64 % (elementSize * 2) == 0 && vectorSize % 2 == 0 && stride % 2 == 0)
    {
        elementSize *= 2;
        vectorSize /= 2;
        stride /= 2;
    }

    if (elementSize == 16)
    {
        alignedUnitBit = 4;
    }
    else if (elementSize == 8)
    {
        alignedUnitBit = 3;
    }
    else if (elementSize == 4)
    {
        alignedUnitBit = 2;
    }
    else if (elementSize == 2)
    {
        alignedUnitBit = 1;
    }
    else
    {
        alignedUnitBit = 0;
    }

    alignedUnitCount = vectorSize;
    alignedUnitStride = stride;
    originalDataType = dataType;
}

class Ll128Proto
{
public:
    static constexpr uint32_t INITIALIZED_VALUE = 0xFFFFFFFFU;

    template <bool USE_FINISH>
    static __device__ __forceinline__ int checkDataReceivedInShm(uint8_t* sharedMemoryBase, uint64_t step,
        int countIn128Bytes, int fifoEntry128ByteIndexBase, int loaded128ByteCount, int warpId, int laneId)
    {
        // return value should be how many package already been received.
        // 0 means no data received, -1 means has received finish package(should be the very first 128 Byte).
        uint64_t* aligned128BytesShm = reinterpret_cast<uint64_t*>(sharedMemoryBase);
        int totalValidCount = 0;
        for (int idxBase = loaded128ByteCount; idxBase < countIn128Bytes; idxBase += WARP_SIZE)
        {
            int idx = idxBase + laneId;
            bool valid = false;
            bool finish = false;
            if (idx < countIn128Bytes)
            {
                int indexInFifoEntry = fifoEntry128ByteIndexBase + idx;
                uint64_t value = aligned128BytesShm[idx * MoeCommFieldInfo::UINT64_PER_128B_BLOCK
                    + indexInFifoEntry % MoeCommFieldInfo::UINT64_PER_128B_BLOCK];
                if (USE_FINISH)
                {
                    finish = (value == (step & (1ULL << 63ULL)));
                    valid = (value == step) || finish;
                }
                else
                {
                    valid = (value == step);
                }
            }
            __syncwarp();
            unsigned validMask = __ballot_sync(WARP_MASK, valid);
            // here we check valid in order, if previous valid is not true, we ignore the current valid.
            int validCount = (validMask == WARP_MASK) ? WARP_SIZE : (__ffs(~validMask) - 1);
            if (USE_FINISH)
            {
                unsigned finishedMask = __ballot_sync(WARP_MASK, finish);
                // finish should be the very first 128 Byte.
                if (finishedMask & 0x1)
                {
                    return -1;
                }
            }
            totalValidCount += validCount;

            if (validCount != WARP_SIZE)
            {
                break;
            }
        }
        return totalValidCount;
    }

    static __device__ __forceinline__ void protoPack(uint8_t* sharedMemoryBase, uint64_t step, int countIn128Bytes,
        int fifoEntry128ByteIndexBase, int warpId, int laneId)
    {
        uint64_t* aligned128BytesShm = reinterpret_cast<uint64_t*>(sharedMemoryBase);
        int halfLaneId = laneId % 16;
        int halfIndex = laneId / 16;
        int tailOffsetIn128Bytes = countIn128Bytes + halfIndex;
        // for LL128 15 * 128 Bytes will be packed to 16 * 128 Bytes, each 16 threads is used for one 15 * 128 bytes.
        for (int idxIn128BytesBase = halfIndex * 15; idxIn128BytesBase < countIn128Bytes; idxIn128BytesBase += 30)
        {
            int tailFlagIndexFromFifoEntry = fifoEntry128ByteIndexBase + tailOffsetIn128Bytes;
            int tailFlagInnerIndex = tailFlagIndexFromFifoEntry % MoeCommFieldInfo::UINT64_PER_128B_BLOCK;
            int idxIn128Bytes = idxIn128BytesBase + halfLaneId;
            int idxFromFifoEntry = fifoEntry128ByteIndexBase + idxIn128Bytes;
            uint64_t tailValue = step;
            uint64_t tailInnerIndex = (halfLaneId >= tailFlagInnerIndex) ? halfLaneId + 1 : halfLaneId;
            if (halfLaneId == 15)
            {
                tailInnerIndex = tailFlagInnerIndex;
            }
            int targetTailIndex = tailOffsetIn128Bytes * MoeCommFieldInfo::UINT64_PER_128B_BLOCK + tailInnerIndex;
            if (idxIn128Bytes < countIn128Bytes && halfLaneId < 15)
            {
                int flagIndex = idxIn128Bytes * MoeCommFieldInfo::UINT64_PER_128B_BLOCK
                    + idxFromFifoEntry % MoeCommFieldInfo::UINT64_PER_128B_BLOCK;
                tailValue = aligned128BytesShm[flagIndex];
                aligned128BytesShm[flagIndex] = step;
            }
            aligned128BytesShm[targetTailIndex] = tailValue;
            tailOffsetIn128Bytes += 2;
        }
        __syncwarp();
    }

    static __device__ __forceinline__ void protoUnpack(uint8_t* sharedMemoryBase, uint64_t step, int countIn128Bytes,
        int fifoEntry128ByteIndexBase, int loaded128ByteCount, int warpId, int laneId)
    {
        uint64_t* aligned128BytesShm = reinterpret_cast<uint64_t*>(sharedMemoryBase);
        int halfLaneId = laneId % 16;
        int halfIndex = laneId / 16;
        int tailOffsetIn128Bytes = countIn128Bytes + halfIndex;
        for (int idxIn128BytesBase = halfIndex * 15; idxIn128BytesBase < countIn128Bytes; idxIn128BytesBase += 30)
        {
            int tailFlagIndexFromFifoEntry = fifoEntry128ByteIndexBase + tailOffsetIn128Bytes;
            int tailFlagInnerIndex = tailFlagIndexFromFifoEntry % MoeCommFieldInfo::UINT64_PER_128B_BLOCK;
            int idxIn128Bytes = idxIn128BytesBase + halfLaneId;
            int idxFromFifoEntry = fifoEntry128ByteIndexBase + idxIn128Bytes;
            uint64_t tailValue = 0;
            int tailInnerIndex = (halfLaneId >= tailFlagInnerIndex) ? halfLaneId + 1 : halfLaneId;
            int targetTailIndex = tailOffsetIn128Bytes * MoeCommFieldInfo::UINT64_PER_128B_BLOCK + tailInnerIndex;
            if (halfLaneId < 15)
            {
                tailValue = aligned128BytesShm[targetTailIndex];
            }
            if (idxIn128Bytes < countIn128Bytes && halfLaneId < 15)
            {
                int flagIndex = idxIn128Bytes * MoeCommFieldInfo::UINT64_PER_128B_BLOCK
                    + idxFromFifoEntry % MoeCommFieldInfo::UINT64_PER_128B_BLOCK;
                aligned128BytesShm[flagIndex] = tailValue;
            }
            tailOffsetIn128Bytes += 2;
        }
        __syncwarp();
    }

    static __device__ __forceinline__ void rearm(
        uint32_t* u32FifoPtr, uint64_t step, int countIn128Bytes, int fifoEntry128ByteIndexBase, int warpId, int laneId)
    {
        // LL128 don't need rearm
    }

    static __device__ __host__ __forceinline__ int computeProtoTransfer128ByteAlignedSize(
        int compact128ByteSizeBeforeProto)
    {
        // each 15 * 128 byte need one tail 128 byte
        int tail128ByteSize = (compact128ByteSizeBeforeProto + 15 * 128 - 1) / (15 * 128) * 128;
        return compact128ByteSizeBeforeProto + tail128ByteSize;
    }
};

using FusedMoeProto = Ll128Proto;

// using FusedMoeProto = LamportProto;

namespace fused_moe_impl
{

// returns copy size for txCount
__device__ __forceinline__ int startFieldG2S(MoeCommFieldInfo const& fieldInfo, int dataIndex,
    uint8_t* sharedMemoryBase, int warpId, int laneId, uint64_t* smemBar)
{
    // we can copy more data than needed, just align to 16 bytes.
    int alignedShmLoadOffset = fieldInfo.getUncompactShmOffset();
    uint8_t* sharedMemoryLoadPtr = sharedMemoryBase + alignedShmLoadOffset;
    int copyByteCount = 0;
    uint8_t* loadPtr = fieldInfo.get16BAlignedLoadCopyRange(dataIndex, &copyByteCount);
    if (laneId == 0 && copyByteCount > 0)
    {
        cp_async_bulk_g2s(sharedMemoryLoadPtr, loadPtr, copyByteCount, smemBar);
    }
    return copyByteCount;
}

__device__ __forceinline__ void startFieldS2G(
    MoeCommFieldInfo const& fieldInfo, int dataIndex, uint8_t* sharedMemoryBase, int warpId, int laneId)
{
    int alignedShmStoreOffset = fieldInfo.getUncompactShmOffset();
    uint8_t* sharedMemoryStorePtr = sharedMemoryBase + alignedShmStoreOffset;
    int copyByteCount = 0;
    int headTailShmIdx;
    int headTailGlobalIdx;
    uint8_t* storePtr
        = fieldInfo.get16BAlignedStoreCopyRange(dataIndex, &copyByteCount, laneId, &headTailShmIdx, &headTailGlobalIdx);
    if (copyByteCount > 0 && laneId == 0)
    {
        cp_async_bulk_s2g(storePtr, sharedMemoryStorePtr + MoeCommFieldInfo::BYTES_PER_16B_BLOCK, copyByteCount);
    }
    if (headTailGlobalIdx >= 0)
    {
        // copy head and tail
        fieldInfo.getRawPtr(dataIndex, nullptr)[headTailGlobalIdx] = sharedMemoryStorePtr[headTailShmIdx];
    }
    __syncwarp();
}

// SRC_AFTER_DST is true, if src > dst, pack will use this,
// SRC_AFTER_DST is false, if src < dst, unpack will use this
template <typename T, bool SRC_AFTER_DST = true>
__device__ __forceinline__ void memmoveSharedMemory(uint8_t* dst, uint8_t const* src, int copySize, int laneId)
{
    int count = (copySize + sizeof(T) - 1) / sizeof(T);
    int warpLoopStart = SRC_AFTER_DST ? 0 : (count + WARP_SIZE - 1) / WARP_SIZE - 1;
    int warpLoopEnd = SRC_AFTER_DST ? (count + WARP_SIZE - 1) / WARP_SIZE : -1;
    int warpLoopUpdate = SRC_AFTER_DST ? 1 : -1;
    for (int i = warpLoopStart; i != warpLoopEnd; i += warpLoopUpdate)
    {
        int idx = laneId + i * WARP_SIZE;
        T data = T{};
        if (idx < count)
        {
            data = reinterpret_cast<T const*>(src)[idx];
        }
        __syncwarp();
        if (idx < count)
        {
            reinterpret_cast<T*>(dst)[idx] = data;
        }
        __syncwarp();
    }
}

template <bool IS_PACK = true>
__device__ __forceinline__ void memmoveFieldOnSharedMemory(
    MoeCommFieldInfo const& fieldInfo, int dataIndex, uint8_t* sharedMemoryBase, int laneId)
{
    int movOffset = fieldInfo.getMemmoveOffsets(dataIndex);
    if (movOffset == 0)
    {
        // if movOffset is 0, src and dst are the same, don't need memmove.
        return;
    }
    int alignedBytes = 1 << fieldInfo.alignedUnitBit;
    int copySize = fieldInfo.alignedUnitCount * alignedBytes;
    uint8_t* sharedMemoryCompact = sharedMemoryBase + fieldInfo.getCompactShmOffset();
    uint8_t* sharedMemoryUncompact = sharedMemoryCompact + movOffset;
    uint8_t* sharedMemoryDst = IS_PACK ? sharedMemoryCompact : sharedMemoryUncompact;
    uint8_t* sharedMemorySrc = IS_PACK ? sharedMemoryUncompact : sharedMemoryCompact;

    if (movOffset % 16 == 0)
    {
        memmoveSharedMemory<int4, IS_PACK>(sharedMemoryDst, sharedMemorySrc, copySize, laneId);
    }
    else if (movOffset % 8 == 0)
    {
        memmoveSharedMemory<int64_t, IS_PACK>(sharedMemoryDst, sharedMemorySrc, copySize, laneId);
    }
    else if (movOffset % 4 == 0)
    {
        memmoveSharedMemory<int, IS_PACK>(sharedMemoryDst, sharedMemorySrc, copySize, laneId);
    }
    else if (movOffset % 2 == 0)
    {
        memmoveSharedMemory<int16_t, IS_PACK>(sharedMemoryDst, sharedMemorySrc, copySize, laneId);
    }
    else
    {
        memmoveSharedMemory<int8_t, IS_PACK>(sharedMemoryDst, sharedMemorySrc, copySize, laneId);
    }
}

template <int FIELD_COUNT = MOE_COMM_FIELD_MAX_COUNT>
__device__ __forceinline__ void packAllFields(
    FusedMoeFieldInfo const& sendFieldInfo, int dataIndex, uint8_t* sharedMemoryBase, int laneId)
{
#pragma unroll
    for (int i = 0; i < FIELD_COUNT; i++)
    {
        memmoveFieldOnSharedMemory<true>(sendFieldInfo.fieldsInfo[i], dataIndex, sharedMemoryBase, laneId);
    }
    __syncwarp();
}

template <int FIELD_COUNT = MOE_COMM_FIELD_MAX_COUNT>
__device__ __forceinline__ void unpackAllFields(
    FusedMoeFieldInfo const& recvFieldInfo, int dataIndex, uint8_t* sharedMemoryBase, int laneId)
{
#pragma unroll
    for (int i = FIELD_COUNT - 1; i >= 0; i--)
    {
        memmoveFieldOnSharedMemory<false>(recvFieldInfo.fieldsInfo[i], dataIndex, sharedMemoryBase, laneId);
    }
    __syncwarp();
}

__device__ __forceinline__ void initSmemBar(uint64_t* smemBar, int laneId)
{
    if (laneId == 0)
    {
        mbarrier_init(smemBar, WARP_SIZE);
    }
    __syncwarp();
}

__device__ __forceinline__ void smemBarWait(uint64_t* smemBar, uint32_t* phaseParity)
{
    while (!mbarrier_try_wait_parity(smemBar, *phaseParity))
    {
    }
    *phaseParity = 1 - *phaseParity;
}

__device__ __forceinline__ void startWorkspaceS2G(
    uint64_t* fifoEntry, uint8_t* sharedMemoryBase, int send128ByteCount, int fifo128ByteOffset, int warpId, int laneId)
{
    int copyByteCount = send128ByteCount * MoeCommFieldInfo::BYTES_PER_128B_BLOCK;
    if (laneId == 0)
    {
        cp_async_bulk_s2g(fifoEntry + fifo128ByteOffset * MoeCommFieldInfo::BYTES_PER_128B_BLOCK / sizeof(int64_t),
            sharedMemoryBase, copyByteCount);
    }
    __syncwarp();
    cp_async_bulk_commit_group();
}

__device__ __forceinline__ void startWorkspaceS2GReg(
    uint64_t* fifoEntry, uint8_t* sharedMemoryBase, int send128ByteCount, int fifo128ByteOffset, int warpId, int laneId)
{
    int copyInt4Count = send128ByteCount * MoeCommFieldInfo::BYTES_PER_128B_BLOCK / sizeof(int4);
    int4* sharedMemoryInt4 = reinterpret_cast<int4*>(sharedMemoryBase);
    uint64_t* fifoPtr = fifoEntry + fifo128ByteOffset * MoeCommFieldInfo::BYTES_PER_128B_BLOCK / sizeof(int64_t);
    int4* fifoPtrInt4 = reinterpret_cast<int4*>(fifoPtr);
#pragma unroll 4
    for (int i = laneId; i < copyInt4Count; i += WARP_SIZE)
    {
        fifoPtrInt4[i] = sharedMemoryInt4[i];
    }
    __syncwarp();
}

__device__ __forceinline__ uint64_t startWorkspaceG2S(uint8_t* sharedMemoryBase, uint64_t* fifoEntry,
    int allLoad128ByteCount, int fifo128ByteOffset, int loaded128ByteCount, uint64_t* smemBar, int warpId, int laneId)
{
    int copyByteCount = (allLoad128ByteCount - loaded128ByteCount) * MoeCommFieldInfo::BYTES_PER_128B_BLOCK;
    if (laneId == 0)
    {
        cp_async_bulk_g2s(sharedMemoryBase + loaded128ByteCount * MoeCommFieldInfo::BYTES_PER_128B_BLOCK,
            fifoEntry
                + (fifo128ByteOffset + loaded128ByteCount) * MoeCommFieldInfo::BYTES_PER_128B_BLOCK / sizeof(int64_t),
            copyByteCount, smemBar);
    }
    return mbarrier_arrive_expect_tx(smemBar, laneId == 0 ? copyByteCount : 0);
}

__device__ __forceinline__ void g2sBasicFields(FusedMoeFieldInfo const& sendFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int dataIndex, uint8_t* sharedMemoryBase, int laneId)
{
    int topK = expertParallelInfo.topK;
    int* tokenSelectedSlotsPtr = sendFieldInfo.getTokenSelectedSlotsPtr(dataIndex, laneId, topK);
    float* scalePtr = sendFieldInfo.getScalePtr(dataIndex, laneId, topK);
    ldgsts<4>(reinterpret_cast<int*>(sharedMemoryBase) + laneId, tokenSelectedSlotsPtr, laneId < topK);
    ldgsts<4>(reinterpret_cast<int*>(sharedMemoryBase) + laneId + topK, reinterpret_cast<int*>(scalePtr),
        laneId < topK && sendFieldInfo.expertScales != nullptr);
}

// May commit 1 group for basic fields(tokenSelectedSlots and scales) if HAS_BASIC_FIELDS is true
// For other fields, use smemBar.
template <bool HAS_BASIC_FIELDS = true, int FIELD_COUNT = MOE_COMM_FIELD_MAX_COUNT>
__device__ __forceinline__ uint64_t g2sAllFields(FusedMoeFieldInfo const& sendFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int dataIndex, uint8_t* sharedMemoryBase, int warpId, int laneId,
    uint64_t* smemBar)
{
    if (HAS_BASIC_FIELDS)
    {
        g2sBasicFields(sendFieldInfo, expertParallelInfo, dataIndex, sharedMemoryBase, laneId);
        cp_async_commit_group();
    }
    int asyncLoadSize = 0;
#pragma unroll
    for (int i = 0; i < FIELD_COUNT; i++)
    {
        asyncLoadSize
            += startFieldG2S(sendFieldInfo.fieldsInfo[i], dataIndex, sharedMemoryBase, warpId, laneId, smemBar);
    }
    return mbarrier_arrive_expect_tx(smemBar, laneId == 0 ? asyncLoadSize : 0);
}

template <bool HAS_BASIC_FIELDS = true>
__device__ __forceinline__ void waitG2SBasicFields()
{
    if (HAS_BASIC_FIELDS)
    {
        cp_async_wait_group<0>();
        __syncwarp();
    }
}

__device__ __forceinline__ void waitG2SOtherFields(uint64_t* memBar, uint32_t* phaseParity)
{
    tensorrt_llm::kernels::fused_moe_impl::smemBarWait(memBar, phaseParity);
}

template <bool HAS_BASIC_FIELDS = true>
__device__ __forceinline__ void waitG2SAllFields(uint64_t* memBar, uint32_t* phaseParity)
{
    waitG2SBasicFields<HAS_BASIC_FIELDS>();
    waitG2SOtherFields(memBar, phaseParity);
}

__device__ __forceinline__ void waitS2GBulkRead()
{
    cp_async_bulk_wait_group_read<0>();
    __syncwarp();
}

__device__ __forceinline__ void s2gBasicFields(FusedMoeFieldInfo const& recvFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int dataIndex, uint8_t* sharedMemoryBase, int warpId, int laneId)
{
    int topK = expertParallelInfo.topK;
    int* tokenSelectedSlotsPtr = recvFieldInfo.getTokenSelectedSlotsPtr(dataIndex, laneId, topK);
    float* scalePtr = recvFieldInfo.getScalePtr(dataIndex, laneId, topK);
    if (laneId < topK)
    {
        int selectedSlot = reinterpret_cast<int*>(sharedMemoryBase)[laneId];
        *tokenSelectedSlotsPtr = selectedSlot;
        if (recvFieldInfo.expertScales != nullptr)
        {
            float scale = reinterpret_cast<float*>(sharedMemoryBase)[laneId + topK];
            *scalePtr = scale;
        }
    }
}

// Will commit 1 group, for all non-basic fields
template <bool HAS_BASIC_FIELDS = true, int FIELD_COUNT = MOE_COMM_FIELD_MAX_COUNT>
__device__ __forceinline__ void s2gAllFields(FusedMoeFieldInfo const& recvFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int dataIndex, uint8_t* sharedMemoryBase, int warpId, int laneId)
{
    if (HAS_BASIC_FIELDS)
    {
        s2gBasicFields(recvFieldInfo, expertParallelInfo, dataIndex, sharedMemoryBase, warpId, laneId);
        __syncwarp();
    }
#pragma unroll
    for (int i = 0; i < FIELD_COUNT; i++)
    {
        startFieldS2G(recvFieldInfo.fieldsInfo[i], dataIndex, sharedMemoryBase, warpId, laneId);
    }
    cp_async_bulk_commit_group();
}

template <int FIELD_COUNT, bool HAS_BASIC_FIELD = true, bool LOW_PRECISION = false>
class SingleChannelCommunicator
{
public:
    __device__ __forceinline__ SingleChannelCommunicator(FusedMoeFieldInfo const& fieldInfo,
        MoeExpertParallelInfo const& expertParallelInfo, MoeSingleCommMeta const& commMeta,
        FusedMoeWorkspace const& workspace, FusedMoeWorldInfo const& worldInfo, FusedMoePairInfo const& pairInfo,
        uint64_t* smemBar, uint8_t* shmemBase)
        : mFieldInfo(fieldInfo)
        , mExpertParallelInfo(expertParallelInfo)
        , mCommMeta(commMeta)
        , mWorkspace(workspace)
        , mWorldInfo(worldInfo)
        , mPairInfo(pairInfo)
        , mSmemBar(smemBar)
        , mShmemBase(shmemBase)
    {
        if constexpr (LOW_PRECISION)
        {
            static_assert(FIELD_COUNT == 1, "Low precision alltoall only support 1 field");
        }

        mWarpId = threadIdx.x / WARP_SIZE;
        mLaneId = threadIdx.x % WARP_SIZE;

        mFifoBasePtr = mWorkspace.getFifoBasePtr(mWorldInfo, mPairInfo);
        mSenderSideFifoInfo = mWorkspace.getSenderSideFifoInfo(mWorldInfo, mPairInfo);
        mReceiverSideFifoInfo = mWorkspace.getReceiverSideFifoInfo(mWorldInfo, mPairInfo);

        mSingleTransfer128ByteCount = mCommMeta.getTransfer128ByteCount();
        mSingleCompactData128ByteCount = mCommMeta.getCompactData128ByteCount();
        // initialize as need new Entry first
        mFifoEntry128ByteIndexBase = kFifoEntry128ByteCount;
        mFifoEntryIndex = -1;

        tensorrt_llm::kernels::fused_moe_impl::initSmemBar(mSmemBar, mLaneId);
    }

    __device__ __forceinline__ uint64_t* getFifoEntryPtr() const
    {
        return mFifoBasePtr + mFifoEntryIndex * kFifoEntrySizeInU64;
    }

    __device__ __forceinline__ bool needNewEntry() const
    {
        return mFifoEntry128ByteIndexBase + mSingleTransfer128ByteCount > kFifoEntry128ByteCount;
    }

    __device__ __forceinline__ void nextToken()
    {
        mFifoEntry128ByteIndexBase += mSingleTransfer128ByteCount;
    }

    __device__ __forceinline__ void senderInitFifo()
    {
        mHead = mSenderSideFifoInfo->head;
        mTail = mSenderSideFifoInfo->tail;
    }

    __device__ __forceinline__ void receiverInitFifo()
    {
        mHead = mReceiverSideFifoInfo->head;
        mTail = mReceiverSideFifoInfo->tail;
    }

    /*
     * Head     | 0 | 1 | 2 | 3 | 4 | 4 | 4 | 4 | 4 | 5 |
     * Tail     | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 3 | 4 | 4 |
     * Writable | Y | Y | Y | Y | N | Y | Y | Y | Y | Y |
     * Readable | N | Y | Y | Y | Y | Y | Y | Y | N | Y |
     */

    __device__ __forceinline__ void waitEntryWritable()
    {
        while (mTail + kFifoDepth <= mHead)
        {
            mTail = mSenderSideFifoInfo->tail;
        }
    }

    __device__ __forceinline__ void updateWriteEntry()
    {
        __syncwarp();
        mSenderSideFifoInfo->head = mHead;
    }

    __device__ __forceinline__ void waitEntryReadable()
    {
        // always readable as long as flag matches.
    }

    __device__ __forceinline__ void updateReadEntry()
    {
        mReceiverSideFifoInfo->tail = mTail;
        mSenderSideFifoInfo->tail = mTail;
    }

    __device__ __forceinline__ void newSendEntry()
    {
        mFifoEntryIndex = mHead % kFifoDepth;
        mFifoEntry128ByteIndexBase = 0;
        waitEntryWritable();
        __syncwarp();
    }

    __device__ __forceinline__ void newReceiveEntry()
    {
        mFifoEntryIndex = mTail % kFifoDepth;
        mFifoEntry128ByteIndexBase = 0;
        waitEntryReadable();
        __syncwarp();
    }

    __device__ __forceinline__ void doSend(int tokenCount, int* sendIndexMapping)
    {
        senderInitFifo();

        int sendIndex = mPairInfo.channel;
        uint32_t phaseParity = 0;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        for (; sendIndex < tokenCount; sendIndex += mPairInfo.runChannelCount)
        {
            int tokenIndex = sendIndexMapping == nullptr ? sendIndex : sendIndexMapping[sendIndex];
            tensorrt_llm::kernels::fused_moe_impl::g2sAllFields<HAS_BASIC_FIELD, FIELD_COUNT>(
                mFieldInfo, mExpertParallelInfo, tokenIndex, mShmemBase, mWarpId, mLaneId, mSmemBar);
            if (needNewEntry())
            {
                if (mFifoEntryIndex >= 0)
                {
                    // not first entry, update FIFO info from last entry.
                    mHead++;
                    updateWriteEntry();
                }
                newSendEntry();
            }
            tensorrt_llm::kernels::fused_moe_impl::waitG2SAllFields<HAS_BASIC_FIELD>(mSmemBar, &phaseParity);
            tensorrt_llm::kernels::fused_moe_impl::packAllFields<FIELD_COUNT>(
                mFieldInfo, tokenIndex, mShmemBase, mLaneId);

            if constexpr (LOW_PRECISION)
            {
                // quantize here.
                int alignedUnitBit = mFieldInfo.fieldsInfo[0].alignedUnitBit;
                int alignedUnitCount = mFieldInfo.fieldsInfo[0].alignedUnitCount;
                int sizeInBytes = alignedUnitCount * (1 << alignedUnitBit);
                uint8_t* sharedMemoryCompact = mShmemBase + mFieldInfo.fieldsInfo[0].getCompactShmOffset();
                cudaDataType_t originalDataType = mFieldInfo.fieldsInfo[0].originalDataType;

                switch (originalDataType)
                {
                case CUDA_R_16BF:
                    quantize_nvfp4_sharedmem<__nv_bfloat16>(sharedMemoryCompact, sizeInBytes, mLaneId);
                    break;
                case CUDA_R_16F: quantize_nvfp4_sharedmem<half>(sharedMemoryCompact, sizeInBytes, mLaneId); break;
                default: break;
                }
            }

            FusedMoeProto::protoPack(
                mShmemBase, mHead, mSingleCompactData128ByteCount, mFifoEntry128ByteIndexBase, mWarpId, mLaneId);

            tensorrt_llm::kernels::fused_moe_impl::startWorkspaceS2GReg(getFifoEntryPtr(), mShmemBase,
                mSingleTransfer128ByteCount, mFifoEntry128ByteIndexBase, mWarpId, mLaneId);

            // tensorrt_llm::kernels::fused_moe_impl::waitS2GBulkRead();

            nextToken();
        }
        if (mFifoEntry128ByteIndexBase > 0)
        {
            mHead++;
            updateWriteEntry();
        }
    }

    __device__ __forceinline__ void rearmFifoBuffer()
    {
        constexpr int kUint32CountPer128Byte = 128 / sizeof(uint32_t);
        uint32_t* fifoPtr = reinterpret_cast<uint32_t*>(getFifoEntryPtr());
        fifoPtr += mFifoEntry128ByteIndexBase * kUint32CountPer128Byte;

        FusedMoeProto::rearm(fifoPtr, mTail, mSingleTransfer128ByteCount, mFifoEntry128ByteIndexBase, mWarpId, mLaneId);
        __syncwarp();
    }

    __device__ __forceinline__ void doReceive(int tokenCount, int* recvIndexMapping)
    {
        receiverInitFifo();
        int recvIndex = mPairInfo.channel;
        uint32_t phaseParity = 0;
        bool needRelease = false;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        for (; recvIndex < tokenCount; recvIndex += mPairInfo.runChannelCount)
        {
            int tokenIndex = recvIndexMapping == nullptr ? recvIndex : recvIndexMapping[recvIndex];
            int loaded128ByteCount = 0;
            if (needNewEntry())
            {
                if (mFifoEntryIndex >= 0)
                {
                    // not first entry, update FIFO info from last entry.
                    mTail++;
                    needRelease = true;
                }
                newReceiveEntry();
            }
            while (loaded128ByteCount < mSingleTransfer128ByteCount)
            {
                tensorrt_llm::kernels::fused_moe_impl::startWorkspaceG2S(mShmemBase, getFifoEntryPtr(),
                    mSingleTransfer128ByteCount, mFifoEntry128ByteIndexBase, loaded128ByteCount, mSmemBar, mWarpId,
                    mLaneId);
                if (needRelease)
                {
                    updateReadEntry();
                    needRelease = false;
                }
                tensorrt_llm::kernels::fused_moe_impl::smemBarWait(mSmemBar, &phaseParity);
                loaded128ByteCount += FusedMoeProto::template checkDataReceivedInShm<false>(mShmemBase, mTail,
                    mSingleTransfer128ByteCount, mFifoEntry128ByteIndexBase, loaded128ByteCount, mWarpId, mLaneId);
            }

            FusedMoeProto::protoUnpack(mShmemBase, mTail, mSingleCompactData128ByteCount, mFifoEntry128ByteIndexBase,
                loaded128ByteCount, mWarpId, mLaneId);

            if constexpr (LOW_PRECISION)
            {
                int alignedUnitBit = mFieldInfo.fieldsInfo[0].alignedUnitBit;
                int alignedUnitCount = mFieldInfo.fieldsInfo[0].alignedUnitCount;
                int sizeInBytes = alignedUnitCount * (1 << alignedUnitBit);
                uint8_t* sharedMemoryCompact = mShmemBase + mFieldInfo.fieldsInfo[0].getCompactShmOffset();
                cudaDataType_t originalDataType = mFieldInfo.fieldsInfo[0].originalDataType;

                switch (originalDataType)
                {
                case CUDA_R_16BF:
                    dequantize_nvfp4_sharedmem<__nv_bfloat16>(sharedMemoryCompact, sizeInBytes, mLaneId);
                    break;
                case CUDA_R_16F: dequantize_nvfp4_sharedmem<half>(sharedMemoryCompact, sizeInBytes, mLaneId); break;
                default: break;
                }
            }

            tensorrt_llm::kernels::fused_moe_impl::unpackAllFields<FIELD_COUNT>(
                mFieldInfo, tokenIndex, mShmemBase, mLaneId);
            tensorrt_llm::kernels::fused_moe_impl::s2gAllFields<HAS_BASIC_FIELD, FIELD_COUNT>(
                mFieldInfo, mExpertParallelInfo, tokenIndex, mShmemBase, mWarpId, mLaneId);
            tensorrt_llm::kernels::fused_moe_impl::waitS2GBulkRead();

            rearmFifoBuffer();
            nextToken();
        }
        if (mFifoEntry128ByteIndexBase > 0)
        {
            mTail++;
            updateReadEntry();
        }
    }

private:
    static constexpr int kFifoEntrySizeInU64 = FusedMoeCommunicator::FIFO_ENTRY_BYTES / sizeof(uint64_t);
    static constexpr int kFifoEntry128ByteCount = FusedMoeCommunicator::FIFO_ENTRY_128_BYTE_COUNT;
    static constexpr int kFifoDepth = FusedMoeCommunicator::FIFO_DEPTH;

    FusedMoeFieldInfo mFieldInfo;
    MoeExpertParallelInfo mExpertParallelInfo;
    MoeSingleCommMeta mCommMeta;
    FusedMoeWorkspace mWorkspace;
    FusedMoeWorldInfo mWorldInfo;
    FusedMoePairInfo mPairInfo;
    uint64_t* mSmemBar;
    uint8_t* mShmemBase;

    int mLaneId;
    int mWarpId;

    uint64_t* mFifoBasePtr;
    SenderSideFifoInfo* mSenderSideFifoInfo;
    ReceiverSideFifoInfo* mReceiverSideFifoInfo;

    int64_t mHead;
    int64_t mTail;

    int mSingleTransfer128ByteCount;
    int mSingleCompactData128ByteCount;
    int mFifoEntry128ByteIndexBase;
    int mFifoEntryIndex;
};

template <int FIELD_COUNT = MOE_COMM_FIELD_MAX_COUNT, bool LOW_PRECISION = false>
__global__ void moeAllToAllKernel(FusedMoeCommKernelParam params, FusedMoeWorkspace workspace, bool hasBasicFields)
{
    __shared__ uint64_t allWarpSmemBar[32];
    extern __shared__ int4 allWarpShm[];

    bool isSender = blockIdx.z == 0;
    int runChannelCount = gridDim.y;
    int group = threadIdx.y;
    SendRecvIndices dataIndices = isSender ? params.sendIndices : params.recvIndices;

    FusedMoePairInfo pairInfo;
    int peerRank = blockIdx.x * blockDim.y + group;
    if (peerRank >= params.worldInfo.epInfo.epSize)
    {
        return;
    }
    int tokenCount;
    int* groupStartPtr = dataIndices.getGroupStart(peerRank, tokenCount);
    if (tokenCount == 0)
    {
        return;
    }

    pairInfo.channel = blockIdx.y;
    pairInfo.runChannelCount = runChannelCount;
    pairInfo.senderRank = isSender ? params.worldInfo.epInfo.epRank : peerRank;
    pairInfo.receiverRank = isSender ? peerRank : params.worldInfo.epInfo.epRank;

    if (isSender)
    {
        int singleShmSize = params.sendCommMeta.getSingleShmSize();
        if (hasBasicFields)
        {
            SingleChannelCommunicator<FIELD_COUNT, true, LOW_PRECISION> comm(params.sendFieldInfo,
                params.expertParallelInfo, params.sendCommMeta, workspace, params.worldInfo, pairInfo,
                allWarpSmemBar + group, reinterpret_cast<uint8_t*>(allWarpShm) + singleShmSize * group);
            comm.doSend(tokenCount, groupStartPtr);
        }
        else
        {
            SingleChannelCommunicator<FIELD_COUNT, false, LOW_PRECISION> comm(params.sendFieldInfo,
                params.expertParallelInfo, params.sendCommMeta, workspace, params.worldInfo, pairInfo,
                allWarpSmemBar + group, reinterpret_cast<uint8_t*>(allWarpShm) + singleShmSize * group);
            comm.doSend(tokenCount, groupStartPtr);
        }
    }
    else
    {
        int singleShmSize = params.recvCommMeta.getSingleShmSize();
        if (hasBasicFields)
        {
            SingleChannelCommunicator<FIELD_COUNT, true, LOW_PRECISION> comm(params.recvFieldInfo,
                params.expertParallelInfo, params.recvCommMeta, workspace, params.worldInfo, pairInfo,
                allWarpSmemBar + group, reinterpret_cast<uint8_t*>(allWarpShm) + singleShmSize * group);
            comm.doReceive(tokenCount, groupStartPtr);
        }
        else
        {
            SingleChannelCommunicator<FIELD_COUNT, false, LOW_PRECISION> comm(params.recvFieldInfo,
                params.expertParallelInfo, params.recvCommMeta, workspace, params.worldInfo, pairInfo,
                allWarpSmemBar + group, reinterpret_cast<uint8_t*>(allWarpShm) + singleShmSize * group);
            comm.doReceive(tokenCount, groupStartPtr);
        }
    }
}

int computeMoeAlltoallMaxDynamicSharedMemorySize()
{
    int devId = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&devId));
    cudaFuncAttributes attr{};
    TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, (void const*) moeAllToAllKernel<1>));
    int staticSmem = static_cast<int>(attr.sharedSizeBytes);
    int maxPerBlockShmOptin = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&maxPerBlockShmOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
    return maxPerBlockShmOptin - staticSmem;
}

} // namespace fused_moe_impl

void FusedMoeFieldInfo::fillMetaInfo(
    MoeSingleCommMeta* singleCommMeta, int topK, bool hasScales, bool hasBasicFields, bool isLowPrecision) const
{
    singleCommMeta->singleUncompactAlignedSize = computeSingleUncompactSize(topK, hasScales, hasBasicFields);

    if (isLowPrecision)
    {
        assert(fieldCount == 1);
        assert(fieldsInfo[0].originalDataType == CUDA_R_16F || fieldsInfo[0].originalDataType == CUDA_R_16BF);

        auto alignment128 = MoeCommFieldInfo::BYTES_PER_128B_BLOCK;

        auto alignedUnitBit = fieldsInfo[0].alignedUnitBit;
        auto alignedUnitCount = fieldsInfo[0].alignedUnitCount;
        auto originalFieldSize = alignedUnitCount * (1 << alignedUnitBit);

        int numElements = originalFieldSize / 2;
        int numGroups = (numElements + WARP_SIZE * 16 - 1) / (WARP_SIZE * 16);
        int sizePerGroupInBytes = (WARP_SIZE * 16 / 2 + WARP_SIZE * 1);

        int totalSize = numGroups * sizePerGroupInBytes + 4;
        int compactSize = (totalSize + alignment128 - 1) / alignment128 * alignment128;

        singleCommMeta->singleCompactAlignedSize = compactSize;
        singleCommMeta->singleTransferAlignedSize
            = FusedMoeProto::computeProtoTransfer128ByteAlignedSize(singleCommMeta->singleCompactAlignedSize);
        return;
    }

    singleCommMeta->singleCompactAlignedSize = computeSingleCompactSize(topK, hasScales, hasBasicFields);
    singleCommMeta->singleTransferAlignedSize
        = FusedMoeProto::computeProtoTransfer128ByteAlignedSize(singleCommMeta->singleCompactAlignedSize);
}

void FusedMoeFieldInfo::fillFieldPlacementInfo(int topK, bool hasBasicFields)
{
    int basicFieldSize = 0;
    if (hasBasicFields)
    {
        basicFieldSize = topK * sizeof(int) + (expertScales != nullptr ? topK * sizeof(float) : 0);
        // align to 16 bytes
        basicFieldSize = (basicFieldSize + MoeCommFieldInfo::BYTES_PER_16B_BLOCK - 1)
            / MoeCommFieldInfo::BYTES_PER_16B_BLOCK * MoeCommFieldInfo::BYTES_PER_16B_BLOCK;
    }
    int offset = basicFieldSize;
    int unalignedFieldIndex = 0;
    for (int i = 0; i < fieldCount; i++)
    {
        fieldsInfo[i].compact16BOffset = offset / MoeCommFieldInfo::BYTES_PER_16B_BLOCK;
        offset += fieldsInfo[i].getFieldCompactSize();
        fieldsInfo[i].unalignedFieldIndex = unalignedFieldIndex;
        if (fieldsInfo[i].alignedUnitBit < 4)
        {
            unalignedFieldIndex++;
        }
    }
    for (int i = fieldCount; i < MOE_COMM_FIELD_MAX_COUNT; i++)
    {
        fieldsInfo[i].setUnused();
    }
}

void FusedMoeWorkspace::initializeLocalWorkspace(FusedMoeWorldInfo const& worldInfo)
{
    int epSize = worldInfo.epInfo.epSize;
    int epRank = worldInfo.epInfo.epRank;
    size_t fifoSize = static_cast<size_t>(FusedMoeCommunicator::FIFO_TOTAL_BYTES) * epSize * channelCount;
    size_t senderSideInfoSize = sizeof(SenderSideFifoInfo) * epSize * channelCount;
    size_t receiverSideInfoSize = sizeof(ReceiverSideFifoInfo) * epSize * channelCount;
    uint64_t* localWorkspacePtr = workspacePtr + epRank * rankStrideInU64;
    TLLM_CU_CHECK(cuMemsetD32(reinterpret_cast<CUdeviceptr>(localWorkspacePtr), FusedMoeProto::INITIALIZED_VALUE,
        fifoSize / sizeof(uint32_t)));
    TLLM_CUDA_CHECK(cudaMemset(
        reinterpret_cast<uint8_t*>(localWorkspacePtr) + fifoSize, 0, senderSideInfoSize + receiverSideInfoSize));
}

void moeAllToAll(FusedMoeCommKernelParam params, FusedMoeWorkspace workspace, cudaStream_t stream)
{
    bool hasBasicFields = params.sendFieldInfo.tokenSelectedSlots != nullptr;
    int warpSendShmSize = params.sendCommMeta.getSingleShmSize();
    int warpRecvShmSize = params.recvCommMeta.getSingleShmSize();
    int warpShmSize = warpSendShmSize;
    int epSize = params.worldInfo.epInfo.epSize;
    TLLM_CHECK_WITH_INFO(warpSendShmSize == warpRecvShmSize, "warpSendShmSize(%d) not same as warpRecvShmSize(%d)",
        warpSendShmSize, warpRecvShmSize);
    int maxGroupCountPerCta = std::min(params.worldInfo.epInfo.epSize, FusedMoeCommunicator::MAX_GROUP_COUNT_PER_BLOCK);
    static int maxDynamicShmSize = fused_moe_impl::computeMoeAlltoallMaxDynamicSharedMemorySize();
    int groupCountPerCta = std::min(maxGroupCountPerCta, maxDynamicShmSize / warpShmSize);

    int maxFieldCount = std::max(params.sendFieldInfo.fieldCount, params.recvFieldInfo.fieldCount);
    TLLM_CHECK_WITH_INFO(params.isLowPrecision == false || maxFieldCount == 1, "low precision only support 1 field");

    auto getFunc = [](int fieldCount, bool lowPrecision)
    {
        switch (fieldCount)
        {
        case 1:
            if (lowPrecision)
                return fused_moe_impl::moeAllToAllKernel<1, true>;
            else
                return fused_moe_impl::moeAllToAllKernel<1>;
        case 2: return fused_moe_impl::moeAllToAllKernel<2>;
        case 3: return fused_moe_impl::moeAllToAllKernel<3>;
        case 4: return fused_moe_impl::moeAllToAllKernel<4>;
        case 5: return fused_moe_impl::moeAllToAllKernel<5>;
        case 6: return fused_moe_impl::moeAllToAllKernel<6>;
        case 7: return fused_moe_impl::moeAllToAllKernel<7>;
        case 8: return fused_moe_impl::moeAllToAllKernel<8>;
        default: return fused_moe_impl::moeAllToAllKernel<8>;
        }
        return fused_moe_impl::moeAllToAllKernel<8>;
    };
    auto* kernelFn = getFunc(maxFieldCount, params.isLowPrecision);

    if (groupCountPerCta * warpShmSize > 48 * 1024)
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            kernelFn, cudaFuncAttributeMaxDynamicSharedMemorySize, groupCountPerCta * warpShmSize));
    }
    for (; groupCountPerCta > 0; groupCountPerCta--)
    {
        int dynamicShmSize = groupCountPerCta * warpShmSize;
        int numBlocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &numBlocks, kernelFn, WARP_SIZE * groupCountPerCta, dynamicShmSize)
            != cudaSuccess)
        {
            continue;
        }
        if (numBlocks >= 1)
        {
            break;
        }
    }
    TLLM_CHECK_WITH_INFO(
        groupCountPerCta >= 1, "computed groupCount=%d, warpShmSize=%d", groupCountPerCta, warpShmSize);
    int ctaPerChannel = (epSize + groupCountPerCta - 1) / groupCountPerCta;
    groupCountPerCta = (epSize + ctaPerChannel - 1) / ctaPerChannel;
    int totalDynamicShmSize = warpShmSize * groupCountPerCta;

    dim3 block = FusedMoeCommunicator::getLaunchBlockDim(groupCountPerCta);
    dim3 grid = FusedMoeCommunicator::getLaunchGridDim(params.worldInfo.epInfo.epSize, groupCountPerCta);
    launchWithPdlWhenEnabled(
        "moeAllToAll", kernelFn, grid, block, totalDynamicShmSize, stream, params, workspace, hasBasicFields);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

int FusedMoeCommunicator::maxSmCount = -1;
bool FusedMoeCommunicator::maxSmCountUsed = false;

void setMaxUsableSmCount(int smCount)
{
    FusedMoeCommunicator::setMaxUsableSmCount(smCount);
}

size_t getFusedMoeCommWorkspaceSize(int epSize)
{
    int channelCount = FusedMoeCommunicator::getMoeCommChannelCount(epSize);
    size_t workspaceSize = FusedMoeWorkspace::computeWorkspaceSizePreRank(epSize, channelCount);
    return workspaceSize;
}

void constructWorkspace(FusedMoeWorkspace* workspace, uint64_t* workspacePtr, size_t rankStrideInU64, int epSize)
{
    workspace->workspacePtr = workspacePtr;
    workspace->rankStrideInU64 = rankStrideInU64;
    workspace->channelCount = FusedMoeCommunicator::getMoeCommChannelCount(epSize);
}

void initializeFusedMoeLocalWorkspace(FusedMoeWorkspace* workspace, FusedMoeWorldInfo const& worldInfo)
{
    workspace->initializeLocalWorkspace(worldInfo);
}

namespace fused_moe_comm_tests
{

__global__ void g2sKernel(FusedMoeFieldInfo allFieldInfo, MoeExpertParallelInfo expertParallelInfo,
    MoeSingleCommMeta singleCommMeta, int tokenCount, int* shmDump, bool hasBasicFields)
{
    __shared__ uint64_t allWarpSmemBar[32];
    extern __shared__ int4 allWarpShm[];
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int warpCount = blockDim.x / WARP_SIZE;
    int tokenIndex = warpId + blockIdx.x * warpCount;
    if (tokenIndex >= tokenCount)
    {
        return;
    }

    int singleShmSize = singleCommMeta.singleUncompactAlignedSize;

    tensorrt_llm::kernels::fused_moe_impl::initSmemBar(&allWarpSmemBar[warpId], laneId);
    uint32_t phaseParity = 0;

    uint8_t* sharedMemoryBase = reinterpret_cast<uint8_t*>(allWarpShm) + singleShmSize * warpId;

    if (hasBasicFields)
    {
        tensorrt_llm::kernels::fused_moe_impl::g2sAllFields<true>(
            allFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId, &allWarpSmemBar[warpId]);
        tensorrt_llm::kernels::fused_moe_impl::waitG2SAllFields<true>(&allWarpSmemBar[warpId], &phaseParity);
    }
    else
    {
        tensorrt_llm::kernels::fused_moe_impl::g2sAllFields<false>(
            allFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId, &allWarpSmemBar[warpId]);
        tensorrt_llm::kernels::fused_moe_impl::waitG2SAllFields<false>(&allWarpSmemBar[warpId], &phaseParity);
    }

    for (int offset = laneId; offset < singleShmSize / sizeof(int); offset += WARP_SIZE)
    {
        shmDump[tokenIndex * singleShmSize / sizeof(int) + offset] = reinterpret_cast<int*>(sharedMemoryBase)[offset];
    }
}

void launchSingleG2S(FusedMoeFieldInfo const& sendFieldInfo, MoeExpertParallelInfo const& expertParallelInfo,
    int tokenCount, int* shmDump, int warpsPerBlock, bool hasBasicFields, cudaStream_t stream)
{
    int warpShmSize = sendFieldInfo.computeSingleUncompactSize(
        expertParallelInfo.topK, sendFieldInfo.expertScales != nullptr, hasBasicFields);
    dim3 blockDim(WARP_SIZE * warpsPerBlock, 1, 1);
    dim3 gridDim((tokenCount + warpsPerBlock - 1) / warpsPerBlock, 1, 1);
    MoeSingleCommMeta singleCommMeta;
    sendFieldInfo.fillMetaInfo(
        &singleCommMeta, expertParallelInfo.topK, sendFieldInfo.expertScales != nullptr, hasBasicFields, false);
    TLLM_CUDA_CHECK(
        cudaFuncSetAttribute(g2sKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, warpShmSize * warpsPerBlock));
    g2sKernel<<<gridDim, blockDim, warpShmSize * warpsPerBlock, stream>>>(
        sendFieldInfo, expertParallelInfo, singleCommMeta, tokenCount, shmDump, hasBasicFields);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

__global__ void s2gKernel(FusedMoeFieldInfo recvFieldInfo, MoeExpertParallelInfo expertParallelInfo,
    MoeSingleCommMeta singleCommMeta, int tokenCount, int* shmPreload, bool hasBasicFields)
{
    extern __shared__ int4 allWarpShm[];
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int warpCount = blockDim.x / WARP_SIZE;
    int tokenIndex = warpId + blockIdx.x * warpCount;
    if (tokenIndex >= tokenCount)
    {
        return;
    }
    int singleShmSize = singleCommMeta.singleUncompactAlignedSize;
    uint8_t* sharedMemoryBase = reinterpret_cast<uint8_t*>(allWarpShm) + singleShmSize * warpId;

    for (int offset = laneId; offset < singleShmSize / sizeof(int); offset += WARP_SIZE)
    {
        reinterpret_cast<int*>(sharedMemoryBase)[offset]
            = shmPreload[tokenIndex * singleShmSize / sizeof(int) + offset];
    }
    __syncwarp();

    if (hasBasicFields)
    {
        tensorrt_llm::kernels::fused_moe_impl::s2gAllFields<true>(
            recvFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId);
    }
    else
    {
        tensorrt_llm::kernels::fused_moe_impl::s2gAllFields<false>(
            recvFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId);
    }

    tensorrt_llm::kernels::fused_moe_impl::waitS2GBulkRead();
}

void launchSingleS2G(FusedMoeFieldInfo const& recvFieldInfo, MoeExpertParallelInfo const& expertParallelInfo,
    int tokenCount, int* shmPreload, int warpsPerBlock, bool hasBasicFields, cudaStream_t stream)
{
    int warpShmSize = recvFieldInfo.computeSingleUncompactSize(
        expertParallelInfo.topK, recvFieldInfo.expertScales != nullptr, hasBasicFields);
    dim3 blockDim(WARP_SIZE * warpsPerBlock, 1, 1);
    dim3 gridDim((tokenCount + warpsPerBlock - 1) / warpsPerBlock, 1, 1);
    MoeSingleCommMeta singleCommMeta;
    recvFieldInfo.fillMetaInfo(
        &singleCommMeta, expertParallelInfo.topK, recvFieldInfo.expertScales != nullptr, hasBasicFields, false);
    TLLM_CUDA_CHECK(
        cudaFuncSetAttribute(s2gKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, warpShmSize * warpsPerBlock));
    s2gKernel<<<gridDim, blockDim, warpShmSize * warpsPerBlock, stream>>>(
        recvFieldInfo, expertParallelInfo, singleCommMeta, tokenCount, shmPreload, hasBasicFields);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

__global__ void loopbackKernel(FusedMoeFieldInfo sendFieldInfo, FusedMoeFieldInfo recvFieldInfo,
    MoeExpertParallelInfo expertParallelInfo, MoeSingleCommMeta sendCommMeta, MoeSingleCommMeta recvCommMeta,
    int* recvIndexMapping, int tokenCount, bool hasBasicFields)
{
    __shared__ uint64_t allWarpSmemBar[32];
    extern __shared__ int4 allWarpShm[];
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int warpCount = blockDim.x / WARP_SIZE;
    int tokenIndex = warpId + blockIdx.x * warpCount;
    if (tokenIndex >= tokenCount)
    {
        return;
    }

    int recvTokenIndex = recvIndexMapping[tokenIndex];

    tensorrt_llm::kernels::fused_moe_impl::initSmemBar(&allWarpSmemBar[warpId], laneId);
    uint32_t phaseParity = 0;

    int singleShmSize = sendCommMeta.getSingleShmSize();

    uint8_t* sharedMemoryBase = reinterpret_cast<uint8_t*>(allWarpShm) + singleShmSize * warpId;

    if (hasBasicFields)
    {
        tensorrt_llm::kernels::fused_moe_impl::g2sAllFields<true>(
            sendFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId, &allWarpSmemBar[warpId]);
    }
    else
    {
        tensorrt_llm::kernels::fused_moe_impl::g2sAllFields<false>(
            sendFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId, &allWarpSmemBar[warpId]);
    }

    if (hasBasicFields)
    {
        tensorrt_llm::kernels::fused_moe_impl::waitG2SAllFields<true>(&allWarpSmemBar[warpId], &phaseParity);
    }
    else
    {
        tensorrt_llm::kernels::fused_moe_impl::waitG2SAllFields<false>(&allWarpSmemBar[warpId], &phaseParity);
    }

    tensorrt_llm::kernels::fused_moe_impl::packAllFields(sendFieldInfo, tokenIndex, sharedMemoryBase, laneId);

    tokenIndex = recvTokenIndex; // switch to recvTokenIndex;

    tensorrt_llm::kernels::fused_moe_impl::unpackAllFields(recvFieldInfo, tokenIndex, sharedMemoryBase, laneId);

    if (hasBasicFields)
    {
        tensorrt_llm::kernels::fused_moe_impl::s2gAllFields<true>(
            recvFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId);
    }
    else
    {
        tensorrt_llm::kernels::fused_moe_impl::s2gAllFields<false>(
            recvFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId);
    }

    cp_async_bulk_wait_group_read<0>();
    __syncwarp();
}

// G2S -> Pack -> Unpack -> S2G
void launchLoopback(FusedMoeFieldInfo const& sendFieldInfo, FusedMoeFieldInfo const& recvFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int* recvIndexMapping, int tokenCount, int warpsPerBlock,
    bool hasBasicFields, cudaStream_t stream)
{
    MoeSingleCommMeta sendCommMeta, recvCommMeta;
    sendFieldInfo.fillMetaInfo(
        &sendCommMeta, expertParallelInfo.topK, sendFieldInfo.expertScales != nullptr, hasBasicFields, false);
    recvFieldInfo.fillMetaInfo(
        &recvCommMeta, expertParallelInfo.topK, recvFieldInfo.expertScales != nullptr, hasBasicFields, false);
    int warpSendShmSize = sendCommMeta.getSingleShmSize();
    int warpRecvShmSize = recvCommMeta.getSingleShmSize();
    int warpShmSize = warpSendShmSize;
    TLLM_CHECK_WITH_INFO(warpSendShmSize == warpRecvShmSize, "warpSendShmSize(%d) not same as warpRecvShmSize(%d)",
        warpSendShmSize, warpRecvShmSize);
    dim3 blockDim(WARP_SIZE * warpsPerBlock, 1, 1);
    dim3 gridDim((tokenCount + warpsPerBlock - 1) / warpsPerBlock, 1, 1);
    TLLM_CUDA_CHECK(
        cudaFuncSetAttribute(loopbackKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, warpShmSize * warpsPerBlock));
    loopbackKernel<<<gridDim, blockDim, warpShmSize * warpsPerBlock, stream>>>(sendFieldInfo, recvFieldInfo,
        expertParallelInfo, sendCommMeta, recvCommMeta, recvIndexMapping, tokenCount, hasBasicFields);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <bool HAS_BASIC_FIELD = true>
__global__ void localFifoSendRecvKernel(FusedMoeFieldInfo sendFieldInfo, FusedMoeFieldInfo recvFieldInfo,
    MoeExpertParallelInfo expertParallelInfo, MoeSingleCommMeta sendCommMeta, MoeSingleCommMeta recvCommMeta,
    FusedMoeWorkspace fusedMoeWorkspace, int* sendIndexMapping, int* recvIndexMapping, int tokenCount)
{
    __shared__ uint64_t allWarpSmemBar[32];
    extern __shared__ int4 allWarpShm[];

    FusedMoeWorldInfo worldInfo;
    worldInfo.epInfo.epRank = 0;
    worldInfo.epInfo.epSize = 1;

    int warpId = threadIdx.x / WARP_SIZE;
    int warpCount = blockDim.x / WARP_SIZE;

    FusedMoePairInfo pairInfo;
    pairInfo.senderRank = 0;
    pairInfo.receiverRank = 0;
    pairInfo.channel = blockIdx.z * warpCount + warpId;
    pairInfo.runChannelCount = gridDim.z * warpCount;

    if (blockIdx.y == 0)
    {
        tensorrt_llm::kernels::fused_moe_impl::SingleChannelCommunicator<MOE_COMM_FIELD_MAX_COUNT, HAS_BASIC_FIELD>
            senderComm(sendFieldInfo, expertParallelInfo, sendCommMeta, fusedMoeWorkspace, worldInfo, pairInfo,
                &allWarpSmemBar[warpId],
                reinterpret_cast<uint8_t*>(&allWarpShm[0]) + warpId * sendCommMeta.getSingleShmSize());
        senderComm.doSend(tokenCount, sendIndexMapping);
    }
    else
    {
        tensorrt_llm::kernels::fused_moe_impl::SingleChannelCommunicator<MOE_COMM_FIELD_MAX_COUNT, HAS_BASIC_FIELD>
            recverComm(recvFieldInfo, expertParallelInfo, recvCommMeta, fusedMoeWorkspace, worldInfo, pairInfo,
                &allWarpSmemBar[warpId],
                reinterpret_cast<uint8_t*>(&allWarpShm[0]) + warpId * recvCommMeta.getSingleShmSize());
        recverComm.doReceive(tokenCount, recvIndexMapping);
    }
}

void launchLocalFifoSendRecv(FusedMoeFieldInfo const& sendFieldInfo, FusedMoeFieldInfo const& recvFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int* sendIndexMapping, int* recvIndexMapping,
    FusedMoeWorkspace fusedMoeWorkspace, int tokenCount, int warpsPerBlock, int blockChannelCount, bool hasBasicFields,
    cudaStream_t stream)
{
    MoeSingleCommMeta sendCommMeta, recvCommMeta;
    sendFieldInfo.fillMetaInfo(
        &sendCommMeta, expertParallelInfo.topK, sendFieldInfo.expertScales != nullptr, hasBasicFields, false);
    recvFieldInfo.fillMetaInfo(
        &recvCommMeta, expertParallelInfo.topK, recvFieldInfo.expertScales != nullptr, hasBasicFields, false);
    int warpSendShmSize = sendCommMeta.getSingleShmSize();
    int warpRecvShmSize = recvCommMeta.getSingleShmSize();
    int warpShmSize = warpSendShmSize;
    TLLM_CHECK_WITH_INFO(warpSendShmSize == warpRecvShmSize, "warpSendShmSize(%d) not same as warpRecvShmSize(%d)",
        warpSendShmSize, warpRecvShmSize);
    dim3 blockDim(WARP_SIZE * warpsPerBlock, 1, 1);
    dim3 gridDim(1, 2, blockChannelCount);
    auto* kernelFn = localFifoSendRecvKernel<>;
    if (hasBasicFields)
    {
        kernelFn = localFifoSendRecvKernel<true>;
    }
    else
    {
        kernelFn = localFifoSendRecvKernel<false>;
    }
    TLLM_CUDA_CHECK(
        cudaFuncSetAttribute(kernelFn, cudaFuncAttributeMaxDynamicSharedMemorySize, warpShmSize * warpsPerBlock));
    kernelFn<<<gridDim, blockDim, warpShmSize * warpsPerBlock, stream>>>(sendFieldInfo, recvFieldInfo,
        expertParallelInfo, sendCommMeta, recvCommMeta, fusedMoeWorkspace, sendIndexMapping, recvIndexMapping,
        tokenCount);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace fused_moe_comm_tests

} // namespace kernels
} // namespace tensorrt_llm
