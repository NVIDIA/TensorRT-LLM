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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/swaDecodeAttention.h"
#include "tensorrt_llm/kernels/swaDecodeAttentionKernel.cuh"
#include "tensorrt_llm/kernels/swaDecodeAttentionKernelV2.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// Once per process and instantiation, before its first launch: prefer the largest shared-memory carveout (each CTA
// uses 32 KB at MT=1, 39 KB at MT=2, so the carveout decides how many short windows are resident per SM) and print the
// compiled resources and the resulting occupancy (runtime evidence for the register / CTA-per-SM budget).
template <int MT>
void prepareSwaDecodeKernelOnce()
{
    static bool const done = []()
    {
        auto const kernel = swa_decode::swaDecodeFmhaKernel<MT, KVBlockArray>;
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        cudaFuncAttributes attr{};
        TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
        int ctasPerSm = 0;
        TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctasPerSm, kernel, swa_decode::kThreads, 0));
        std::fprintf(stderr,
            "[TRT-LLM] [INFO] swaDecodeFmhaKernel<%d>: regs=%d local=%zu B static smem=%zu B, %d CTAs/SM at max "
            "shared-memory carveout\n",
            MT, attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes, ctasPerSm);
        return true;
    }();
    (void) done;
}

// TLLM_FMHA_SWA_DECODE_V2=1 (exactly "1"; read once per process; only consulted once TLLM_FMHA_SWA_DECODE_KERNEL=1
// has routed a launch here): launches with at most swa_decode2::kRows rows (query tokens x Q heads per KV head) and
// at most swa_decode2::kMaxKeys keys (window + query tokens - 1) run swaDecodeFmhaKernelV2 (same attention, fewer
// instructions per key; swaDecodeAttentionKernelV2.cuh). Every other launch keeps swaDecodeFmhaKernel. Default off.
bool swaDecodeV2FromEnv()
{
    static bool const value = []()
    {
        char const* env = std::getenv("TLLM_FMHA_SWA_DECODE_V2");
        return env != nullptr && std::strcmp(env, "1") == 0;
    }();
    return value;
}

// Once per process, before the first launch of swaDecodeFmhaKernelV2: the largest shared-memory carveout, and one line
// with the compiled resources, the resulting occupancy and the first launch's shape (runtime evidence).
void prepareSwaDecodeV2Once(SwaDecodeAttentionParams const& params, int headsPerKv)
{
    static bool const done = [&]()
    {
        auto const kernel = swa_decode2::swaDecodeFmhaKernelV2<KVBlockArray>;
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        cudaFuncAttributes attr{};
        TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
        int ctasPerSm = 0;
        TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctasPerSm, kernel, swa_decode2::kThreads, 0));
        std::fprintf(stderr,
            "[TRT-LLM] [INFO] TLLM_FMHA_SWA_DECODE_V2=1 active: swaDecodeFmhaKernelV2 (regs=%d local=%zu B static "
            "smem=%zu B, %d CTAs/SM at max shared-memory carveout; first launch batch=%d maxSeqLenQ=%d headsPerKv=%d "
            "window=%d grid %dx%d)\n",
            attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes, ctasPerSm, params.batch, params.maxSeqLenQ,
            headsPerKv, params.window, params.numHeadsKv, params.batch);
        return true;
    }();
    (void) done;
}

} // namespace

bool swaDecodeAttentionSupported(int headDim, int headsPerKv, int maxSeqLenQ, int window)
{
    return headDim == swa_decode::kHeadDim && headsPerKv >= 1 && maxSeqLenQ >= 1 && headsPerKv * maxSeqLenQ <= 32
        && window >= 1 && window + maxSeqLenQ - 1 <= swa_decode::kMaxKeys;
}

void launchSwaDecodeAttention(SwaDecodeAttentionParams const& params, cudaStream_t stream)
{
    int const headsPerKv = params.numHeadsQ / params.numHeadsKv;
    TLLM_CHECK_WITH_INFO(params.numHeadsKv > 0 && headsPerKv * params.numHeadsKv == params.numHeadsQ,
        "swaDecodeAttention: numHeadsQ must be a multiple of numHeadsKv.");
    TLLM_CHECK_WITH_INFO(
        swaDecodeAttentionSupported(swa_decode::kHeadDim, headsPerKv, params.maxSeqLenQ, params.window),
        "swaDecodeAttention: unsupported shape (headsPerKv=%d maxSeqLenQ=%d window=%d).", headsPerKv, params.maxSeqLenQ,
        params.window);
    TLLM_CHECK_WITH_INFO(params.scaleSoftmaxLog2 != nullptr && params.outputScale != nullptr,
        "swaDecodeAttention: the device scales are required.");
    if (params.batch <= 0)
    {
        return;
    }
    swa_decode::Params<KVBlockArray> p;
    p.q = static_cast<uint8_t const*>(params.q);
    p.out = static_cast<uint16_t*>(params.out);
    p.kv = params.kv;
    p.seqLensKv = params.seqLensKv;
    p.cuSeqLensQ = params.cuSeqLensQ;
    p.scaleSoftmaxLog2 = params.scaleSoftmaxLog2;
    p.outputScale = params.outputScale;
    p.sinks = params.sinks;
    p.numHeadsQ = params.numHeadsQ;
    p.headsPerKv = headsPerKv;
    p.maxSeqLenQ = params.maxSeqLenQ;
    p.window = params.window;

    dim3 const grid(params.numHeadsKv, params.batch);
    if (swaDecodeV2FromEnv())
    {
        if (headsPerKv * params.maxSeqLenQ <= swa_decode2::kRows
            && params.window + params.maxSeqLenQ - 1 <= swa_decode2::kMaxKeys)
        {
            swa_decode2::Params<KVBlockArray> p2;
            p2.q = p.q;
            p2.out = p.out;
            p2.kv = p.kv;
            p2.seqLensKv = p.seqLensKv;
            p2.cuSeqLensQ = p.cuSeqLensQ;
            p2.scaleSoftmaxLog2 = p.scaleSoftmaxLog2;
            p2.outputScale = p.outputScale;
            p2.sinks = p.sinks;
            p2.numHeadsQ = p.numHeadsQ;
            p2.headsPerKv = p.headsPerKv;
            p2.maxSeqLenQ = p.maxSeqLenQ;
            p2.window = p.window;
            prepareSwaDecodeV2Once(params, headsPerKv);
            tensorrt_llm::common::launchWithPdlWhenEnabled("swaDecodeFmhaKernelV2",
                swa_decode2::swaDecodeFmhaKernelV2<KVBlockArray>, grid, dim3(swa_decode2::kThreads), 0, stream, p2);
            sync_check_cuda_error(stream);
            return;
        }
        static bool loggedKeep = false;
        if (!loggedKeep)
        {
            loggedKeep = true;
            std::fprintf(stderr,
                "[TRT-LLM] [INFO] TLLM_FMHA_SWA_DECODE_V2=1: a launch keeps swaDecodeFmhaKernel (rows %d > %d or keys "
                "%d > %d); printed once\n",
                headsPerKv * params.maxSeqLenQ, swa_decode2::kRows, params.window + params.maxSeqLenQ - 1,
                swa_decode2::kMaxKeys);
        }
    }
    dim3 const block(swa_decode::kThreads);
    if (headsPerKv * params.maxSeqLenQ <= 16)
    {
        prepareSwaDecodeKernelOnce<1>();
        tensorrt_llm::common::launchWithPdlWhenEnabled(
            "swaDecodeFmhaKernel", swa_decode::swaDecodeFmhaKernel<1, KVBlockArray>, grid, block, 0, stream, p);
    }
    else
    {
        prepareSwaDecodeKernelOnce<2>();
        tensorrt_llm::common::launchWithPdlWhenEnabled(
            "swaDecodeFmhaKernel", swa_decode::swaDecodeFmhaKernel<2, KVBlockArray>, grid, block, 0, stream, p);
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
