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

#include "minimax_m3/csrc/m31/swigluOaiBf16x2.cuh"
#include "minimax_m3/csrc/m31/swigluOaiMxfp8Sm103.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <optional>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

//! One row per CTA iteration (rows strided over the grid), one 8-column chunk per thread: the 8 gate and 8 up values
//! (16 B each) -> 8 bf16 activations -> cvt_warp_fp16_to_mxfp8<bf16, 32> over the 4 lanes that share a K32 scale.
template <bool kWriteH>
__global__ void swigluOaiMxfp8Kernel(uint4 const* __restrict__ x, uint64_t* __restrict__ qOut, uint32_t* sfOut,
    uint4* __restrict__ hOut, int rows, int d)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static constexpr int kThreadsPerSf = 32 / tensorrt_llm::kernels::CVT_ELTS_PER_THREAD;
    int const colIdx = threadIdx.x;
    int const colThreads = d / tensorrt_llm::kernels::CVT_ELTS_PER_THREAD;
    std::optional<int> const batchIdx = 0;
    std::optional<int> const numRows = rows;

    cudaGridDependencySynchronize();

    for (int row = blockIdx.x; row < rows; row += gridDim.x)
    {
        // x row = [gate (d) | up (d)] = 2 * colThreads chunks of 8 bf16.
        int64_t const xRow = static_cast<int64_t>(row) * 2 * colThreads;
        uint4 const gate = x[xRow + colIdx];
        uint4 const up = x[xRow + colThreads + colIdx];
        uint4 h;
        h.x = swigluOaiBf16x2(gate.x, up.x);
        h.y = swigluOaiBf16x2(gate.y, up.y);
        h.z = swigluOaiBf16x2(gate.z, up.z);
        h.w = swigluOaiBf16x2(gate.w, up.w);
        int64_t const outIdx = static_cast<int64_t>(row) * colThreads + colIdx;
        if constexpr (kWriteH)
        {
            hOut[outIdx] = h;
        }
        tensorrt_llm::kernels::PackedVec<__nv_bfloat16> quantIn;
        *reinterpret_cast<uint4*>(&quantIn) = h;
        auto sf = tensorrt_llm::kernels::cvt_quant_get_sf_out_offset<uint32_t, kThreadsPerSf>(
            batchIdx, row, colIdx, numRows, d / 32, sfOut, QuantizationSFLayout::SWIZZLED);
        qOut[outIdx] = tensorrt_llm::kernels::cvt_warp_fp16_to_mxfp8<__nv_bfloat16, 32>(quantIn, sf);
    }
    int const paddedRows = (rows + 127) / 128 * 128;
    for (int padRow = rows + blockIdx.x; padRow < paddedRows; padRow += gridDim.x)
    {
        auto sf = tensorrt_llm::kernels::cvt_quant_get_sf_out_offset<uint32_t, kThreadsPerSf>(
            batchIdx, padRow, colIdx, numRows, d / 32, sfOut, QuantizationSFLayout::SWIZZLED);
        tensorrt_llm::kernels::cvt_store_replicated_sf<1>(sf, 0x00);
    }
    // Every thread's stores are visible before any thread reports PDL completion (as in quantize_with_block_size).
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

//! SWIGLU_MXFP8_ROWS: swigluOaiMxfp8Kernel with kRows rows in flight per CTA. An iteration takes rows
//! base + i * gridDim.x (i < kRows), issues all 2 * kRows 16-byte loads first, then runs each row exactly as
//! swigluOaiMxfp8Kernel does (the same swigluOaiBf16x2 sequence, the same cvt_warp_fp16_to_mxfp8 over the same 4 lanes
//! of one K32 block). CTA c still visits rows c, c + gridDim.x, ... in order, only kRows at a time, so the row-to-CTA
//! map is the only change and no arithmetic depends on it: every output byte is swigluOaiMxfp8Kernel's. The row
//! bound is uniform over the CTA, so the 4-lane scale shuffle always runs on converged lanes.
template <bool kWriteH, int kRows>
__global__ void swigluOaiMxfp8RowsKernel(uint4 const* __restrict__ x, uint64_t* __restrict__ qOut, uint32_t* sfOut,
    uint4* __restrict__ hOut, int rows, int d)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static constexpr int kThreadsPerSf = 32 / tensorrt_llm::kernels::CVT_ELTS_PER_THREAD;
    int const colIdx = threadIdx.x;
    int const colThreads = d / tensorrt_llm::kernels::CVT_ELTS_PER_THREAD;
    int const stride = gridDim.x;
    std::optional<int> const batchIdx = 0;
    std::optional<int> const numRows = rows;

    cudaGridDependencySynchronize();

    for (int base = blockIdx.x; base < rows; base += stride * kRows)
    {
        uint4 gate[kRows] = {};
        uint4 up[kRows] = {};
#pragma unroll
        for (int i = 0; i < kRows; ++i)
        {
            int const row = base + i * stride;
            if (row < rows)
            {
                int64_t const xRow = static_cast<int64_t>(row) * 2 * colThreads;
                gate[i] = x[xRow + colIdx];
                up[i] = x[xRow + colThreads + colIdx];
            }
        }
#pragma unroll
        for (int i = 0; i < kRows; ++i)
        {
            int const row = base + i * stride;
            if (row >= rows)
            {
                break;
            }
            uint4 h;
            h.x = swigluOaiBf16x2(gate[i].x, up[i].x);
            h.y = swigluOaiBf16x2(gate[i].y, up[i].y);
            h.z = swigluOaiBf16x2(gate[i].z, up[i].z);
            h.w = swigluOaiBf16x2(gate[i].w, up[i].w);
            int64_t const outIdx = static_cast<int64_t>(row) * colThreads + colIdx;
            if constexpr (kWriteH)
            {
                hOut[outIdx] = h;
            }
            tensorrt_llm::kernels::PackedVec<__nv_bfloat16> quantIn;
            *reinterpret_cast<uint4*>(&quantIn) = h;
            auto sf = tensorrt_llm::kernels::cvt_quant_get_sf_out_offset<uint32_t, kThreadsPerSf>(
                batchIdx, row, colIdx, numRows, d / 32, sfOut, QuantizationSFLayout::SWIZZLED);
            qOut[outIdx] = tensorrt_llm::kernels::cvt_warp_fp16_to_mxfp8<__nv_bfloat16, 32>(quantIn, sf);
        }
    }
    // The pad rows and the PDL tail are swigluOaiMxfp8Kernel's.
    int const paddedRows = (rows + 127) / 128 * 128;
    for (int padRow = rows + blockIdx.x; padRow < paddedRows; padRow += gridDim.x)
    {
        auto sf = tensorrt_llm::kernels::cvt_quant_get_sf_out_offset<uint32_t, kThreadsPerSf>(
            batchIdx, padRow, colIdx, numRows, d / 32, sfOut, QuantizationSFLayout::SWIZZLED);
        tensorrt_llm::kernels::cvt_store_replicated_sf<1>(sf, 0x00);
    }
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

//! The rows form engages from this many rows (below it the one-row launch already has every row in flight).
constexpr int64_t kSwigluRowsMinRows = 1024;

//! Resident CTAs per SM of `kernel` at `block` threads on the current device (the occupancy API, so registers and the
//! device's thread limit both count: sm_107 holds 1024 threads per SM, not the 2048 the one-row grid assumes), cached
//! per (device, kernel, block): this thread's last hit first, then a locked table. `fresh` is set when this call added
//! the table entry, so a caller can report each (device, kernel, block) once.
cudaError_t swigluRowsResidentCtas(void const* kernel, int block, int* ctas, bool* fresh)
{
    struct Entry
    {
        int device;
        void const* kernel;
        int block;
        int ctas;
    };

    static std::mutex sMutex;
    static Entry sCache[64] = {};
    static int sCount = 0;
    thread_local Entry tLast{-1, nullptr, 0, 0};
    *fresh = false;
    int device = 0;
    cudaError_t e = cudaGetDevice(&device);
    if (e != cudaSuccess)
    {
        return e;
    }
    if (tLast.device == device && tLast.kernel == kernel && tLast.block == block)
    {
        *ctas = tLast.ctas;
        return cudaSuccess;
    }
    std::lock_guard<std::mutex> const lock(sMutex);
    for (int i = 0; i < sCount; ++i)
    {
        if (sCache[i].device == device && sCache[i].kernel == kernel && sCache[i].block == block)
        {
            *ctas = sCache[i].ctas;
            tLast = sCache[i];
            return cudaSuccess;
        }
    }
    if ((e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(ctas, kernel, block, 0)) != cudaSuccess)
    {
        return e;
    }
    Entry const entry{device, kernel, block, *ctas};
    if (sCount < static_cast<int>(sizeof(sCache) / sizeof(sCache[0])))
    {
        sCache[sCount++] = entry;
        *fresh = true;
    }
    tLast = entry;
    return cudaSuccess;
}

//! Launches the rows form with grid = min(pad128(rows), resident CTAs x SMs): every SM gets resident CTAs at any row
//! count, and a CTA with fewer than kRows rows left batches the rows it has. Sets `taken` only when it launched. A
//! failed occupancy query (its error cleared) or no resident CTA at this block size leaves the launch to the one-row
//! form; both are logged, the latter once per (device, kernel, block).
template <bool kWriteH, int kRows>
cudaError_t launchSwigluRows(cudaLaunchConfig_t cfg, uint4 const* x, uint64_t* qOut, uint32_t* sfOut, uint4* hOut,
    int rows, int d, int64_t paddedRows, int multiProcessorCount, bool* taken)
{
    auto kernel = &swigluOaiMxfp8RowsKernel<kWriteH, kRows>;
    int ctas = 0;
    bool fresh = false;
    cudaError_t const e = swigluRowsResidentCtas(
        reinterpret_cast<void const*>(kernel), static_cast<int>(cfg.blockDim.x), &ctas, &fresh);
    if (e != cudaSuccess)
    {
        // The one-row launch does not need the query: clear its error so it is not reported by a later check.
        static_cast<void>(cudaGetLastError());
        static std::once_flag warned;
        std::call_once(warned,
            [&]
            {
                TLLM_LOG_WARNING(
                    "invokeSwigluOaiMxfp8: SWIGLU_MXFP8 rows form NOT ENGAGED at d %d (kRows=%d): occupancy query "
                    "failed (%s), one-row launch",
                    d, kRows, cudaGetErrorString(e));
            });
        return cudaSuccess;
    }
    if (ctas < 1)
    {
        if (fresh)
        {
            TLLM_LOG_INFO(
                "invokeSwigluOaiMxfp8: SWIGLU_MXFP8 rows form NOT ENGAGED at d %d (kRows=%d, h %s): no resident CTA, "
                "one-row launch",
                d, kRows, kWriteH ? "written" : "elided");
        }
        return cudaSuccess;
    }
    int64_t const grid = std::max<int64_t>(1, std::min<int64_t>(paddedRows, int64_t{multiProcessorCount} * ctas));
    cfg.gridDim = dim3(static_cast<unsigned>(grid));
    if (fresh)
    {
        TLLM_LOG_INFO(
            "invokeSwigluOaiMxfp8: SWIGLU_MXFP8 rows form kRows=%d at d %d (h %s): %d CTAs/SM, grid %d at %d rows",
            kRows, d, kWriteH ? "written" : "elided", ctas, static_cast<int>(grid), rows);
    }
    *taken = true;
    return cudaLaunchKernelEx(&cfg, kernel, x, qOut, sfOut, hOut, rows, d);
}

//! The rows form at the requested kRows (2 / 4 / 8; other values: 4).
template <bool kWriteH>
cudaError_t dispatchSwigluRows(int kRows, cudaLaunchConfig_t const& cfg, uint4 const* x, uint64_t* qOut,
    uint32_t* sfOut, uint4* hOut, int rows, int d, int64_t paddedRows, int multiProcessorCount, bool* taken)
{
    switch (kRows)
    {
    case 2:
        return launchSwigluRows<kWriteH, 2>(cfg, x, qOut, sfOut, hOut, rows, d, paddedRows, multiProcessorCount, taken);
    case 8:
        return launchSwigluRows<kWriteH, 8>(cfg, x, qOut, sfOut, hOut, rows, d, paddedRows, multiProcessorCount, taken);
    default:
        return launchSwigluRows<kWriteH, 4>(cfg, x, qOut, sfOut, hOut, rows, d, paddedRows, multiProcessorCount, taken);
    }
}

} // namespace

cudaError_t invokeSwigluOaiMxfp8(uint4 const* x, uint64_t* qOut, uint32_t* sfOut, uint4* hOut, int64_t rows, int64_t d,
    int multiProcessorCount, bool pdl, cudaStream_t stream, bool rowsForm, int kRows)
{
    if (d % 256 != 0 || d > 8192 || rows > (int64_t{1} << 30))
    {
        return cudaErrorInvalidValue;
    }
    if (rows == 0)
    {
        return cudaSuccess;
    }
    int const threads = static_cast<int>(d / tensorrt_llm::kernels::CVT_ELTS_PER_THREAD);
    int64_t const paddedRows = (rows + 127) / 128 * 128;
    cudaLaunchConfig_t cfg{};
    cfg.blockDim = dim3(threads);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int const r = static_cast<int>(rows);
    int const dd = static_cast<int>(d);
    if (rowsForm && rows >= kSwigluRowsMinRows)
    {
        bool taken = false;
        auto const dispatch = hOut != nullptr ? &dispatchSwigluRows<true> : &dispatchSwigluRows<false>;
        cudaError_t const e
            = dispatch(kRows, cfg, x, qOut, sfOut, hOut, r, dd, paddedRows, multiProcessorCount, &taken);
        if (e != cudaSuccess || taken)
        {
            return e;
        }
    }
    // The one-row launch (below 1024 rows, the kill switch, or no resident rows-form CTA at this width):
    // quantize_with_block_size's grid, one CTA per (padded) row, capped at 2048 / threads CTAs per SM.
    int const blocksPerSm = std::max(1, 2048 / threads);
    cfg.gridDim
        = dim3(static_cast<unsigned>(std::min<int64_t>(paddedRows, int64_t{multiProcessorCount} * blocksPerSm)));
    return hOut != nullptr ? cudaLaunchKernelEx(&cfg, swigluOaiMxfp8Kernel<true>, x, qOut, sfOut, hOut, r, dd)
                           : cudaLaunchKernelEx(&cfg, swigluOaiMxfp8Kernel<false>, x, qOut, sfOut, hOut, r, dd);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
