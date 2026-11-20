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

// The XQA decode kernel compiled ahead of time for this target. The macros below are the ones TensorRT-LLM's XQA JIT
// (decoderXQAImplJIT/compileEngine.cpp, nvrtcWrapper.cpp) passes for this configuration; GENERATE_CUBIN stays
// undefined, so the kernels and their shared-memory sizes are local to this translation unit. NDEBUG selects the
// kernel_mha wrapper that takes the shared-prefix arguments. This file is compiled with --use_fast_math, as the JIT.

#ifndef NDEBUG
#define NDEBUG 1
#endif
#define HEAD_ELEMS 256
#define BEAM_WIDTH 1
#define CACHE_ELEM_ENUM 2
#define TOKENS_PER_PAGE 32
#define HEAD_GRP_SIZE 6
#define M_TILESIZE 6
#define USE_CUSTOM_BARRIER 1
#define SLIDING_WINDOW 1
#define LOW_PREC_OUTPUT 0
#define USE_INPUT_KV 0
#define ROPE_STYLE 0
#define ROPE_ELEMS 256
#define IS_SPEC_DEC_TREE 0
#define SKIP_SOFTMAX_ATTN 0
#define INPUT_FP16 0
#define DTYPE __nv_bfloat16

#include "xqa/mha.cu"

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/xqaDecodeSm120.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mutex>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

static_assert(TOKENS_PER_PAGE == kXqaSm120TokensPerPage, "");

// TensorRT-LLM's XQA constants (decoderXQAConstants.h).
constexpr int kMinHistoryTokensPerBlock = 128;
constexpr double kTargetWaveFactor = 8;
constexpr int kMaxNbSubSeq = 6000;
// The multi-block scratch of XqaDispatcher::getWorkspaceSize (multiBlockMode), for head size 256 and 6 query heads per
// KV head: semaphores, row max, row sum, partial tiles, each rounded to the 12288-byte alignment, plus one alignment.
constexpr size_t kMbAlign = 12288; // roundUp(2 * 4 * 6 * 256, 128)

constexpr size_t roundUpSize(size_t a, size_t b)
{
    return (a + b - 1) / b * b;
}

constexpr size_t kScratchBytes = roundUpSize(sizeof(uint32_t) * (kMaxNbSubSeq / 2), kMbAlign)
    + 2 * roundUpSize(sizeof(float) * 32 * kMaxNbSubSeq, kMbAlign) + kMbAlign * kMaxNbSubSeq + kMbAlign;

// The shared-prefix pass (kernel_mha_prefix, csrc/xqa/mha.cu): 4 KiB of cascade info words,
// then one FP32 record per (request, KV head, prefix chunk), then kernel_mha's own multi-block scratch, within the
// budget getWorkspaceSize reserves for kernel_mha alone.
constexpr uint32_t kCascadeRowsPerCta = 64;
constexpr int32_t kCascadeMaxChunks = 63;
constexpr size_t kCascadeInfoBytes = 4096;
constexpr size_t kMhaBytesPerSubSeq = 4096;
constexpr size_t kCascadeBudget = size_t{kMaxNbSubSeq} * 3328;

struct KernelSmem
{
    uint32_t mha = 0;
    uint32_t prefix = 0;
};

KernelSmem const& kernelSmem()
{
    static KernelSmem smem;
    static std::once_flag once;
    std::call_once(once,
        []()
        {
            TLLM_CUDA_CHECK(cudaMemcpyFromSymbol(&smem.mha, smemSize, sizeof(smemSize)));
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize, smem.mha));
            TLLM_CUDA_CHECK(cudaMemcpyFromSymbol(&smem.prefix, prefixSmemSize, sizeof(prefixSmemSize)));
            TLLM_CUDA_CHECK(
                cudaFuncSetAttribute(kernel_mha_prefix, cudaFuncAttributeMaxDynamicSharedMemorySize, smem.prefix));
        });
    return smem;
}

cudaLaunchConfig_t launchConfig(
    dim3 grid, dim3 block, uint32_t smem, cudaStream_t stream, bool pdl, cudaLaunchAttribute* attr)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cfg;
}

} // namespace

size_t xqaSm120ScratchBytes()
{
    return kScratchBytes;
}

size_t xqaSm120SemaphoreWords(uint32_t maxBatch, int multiProcessorCount)
{
    // As the stock op: max(numQHeads * maxBatch, SMs), rounded up to 8.
    size_t const n = std::max<size_t>(size_t{24} * maxBatch, static_cast<size_t>(multiProcessorCount));
    return roundUpSize(n, 8);
}

uint32_t xqaSm120MultiBlockCount(uint32_t batch, uint32_t maxPastKvLength, int multiProcessorCount)
{
    auto const userSpecified = tensorrt_llm::common::getEnvXqaBlocksPerSequence();
    if (userSpecified.has_value())
    {
        return static_cast<uint32_t>(userSpecified.value());
    }
    int const batchSize = static_cast<int>(batch);
    int mb = static_cast<int>(maxPastKvLength) / kMinHistoryTokensPerBlock;
    mb = std::min(mb, static_cast<int>(std::round(std::sqrt(mb * 8.F))));
    mb = std::max(mb, 1);
    double const waveCount = static_cast<double>(batchSize) * kXqaSm120KvHeads * mb / multiProcessorCount;
    double const adjFactor = waveCount / kTargetWaveFactor;
    if (adjFactor > 1.0)
    {
        mb = static_cast<int>(std::floor(mb / adjFactor));
    }
    mb = std::max(mb, 1);
    mb = std::max(std::min(mb, kMaxNbSubSeq / batchSize), 1);
    // One CTA wave: kernel_mha runs at one CTA per SM here, so all CTAs of a call stay co-resident.
    int const oneWave = std::max(1, multiProcessorCount / (batchSize * static_cast<int>(kXqaSm120KvHeads)));
    return static_cast<uint32_t>(std::min(mb, oneWave));
}

void xqaSm120Decode(XqaSm120DecodeArgs const& a, cudaStream_t stream)
{
    TLLM_CHECK(a.batch >= 1 && a.multiBlock >= 1);
    TLLM_CHECK_WITH_INFO(static_cast<size_t>(a.multiBlock) * a.batch <= kMaxNbSubSeq || a.multiBlock == 1,
        "XQA multi-block count %u x batch %u exceeds the scratch budget", a.multiBlock, a.batch);
    KernelSmem const& smem = kernelSmem();
    KVCacheList<true> const cacheList{static_cast<GMemKVCacheHead*>(a.kvPool),
        reinterpret_cast<KVCachePageIndex const*>(a.kvPageList), a.seqLens, a.maxNbPagesPerSeq};
    auto const* q = static_cast<IOHead const*>(a.q);
    auto* const output = static_cast<OutputHead*>(a.output);
    uint32_t const nbKHeads = kXqaSm120KvHeads;
    void* scratch = a.scratch;
    uint32_t* cascadeInfo = nullptr;
    uint32_t nbPrefixParts = 0;

    uint32_t const minBatch = a.sharedPrefixMinBatch == 0 ? 0 : std::max(2U, a.sharedPrefixMinBatch);
    if (minBatch != 0 && a.batch >= minBatch && a.multiBlock >= 1)
    {
        size_t const nbSeq = static_cast<size_t>(a.batch) * nbKHeads;
        size_t const recordBytes
            = sizeof(float) * (size_t{HEAD_GRP_SIZE} * HEAD_ELEMS + 2 * roundUpSize(HEAD_GRP_SIZE, 4));
        size_t const mhaBytes = static_cast<size_t>(a.multiBlock) * nbSeq * kMhaBytesPerSubSeq;
        uint32_t const nbRowGrps = (a.batch * HEAD_GRP_SIZE + kCascadeRowsPerCta - 1) / kCascadeRowsPerCta;
        int32_t chunks
            = std::min<int32_t>(kCascadeMaxChunks, a.multiProcessorCount / static_cast<int32_t>(nbRowGrps * nbKHeads));
        int64_t const fit = kCascadeBudget > mhaBytes + 2 * kCascadeInfoBytes
            ? static_cast<int64_t>((kCascadeBudget - mhaBytes - 2 * kCascadeInfoBytes) / (nbSeq * recordBytes))
            : 0;
        chunks = static_cast<int32_t>(std::min<int64_t>(chunks, fit));
        if (chunks >= 1)
        {
            auto* const base = static_cast<std::byte*>(a.scratch);
            cascadeInfo = reinterpret_cast<uint32_t*>(base);
            nbPrefixParts = static_cast<uint32_t>(chunks);
            scratch = base + kCascadeInfoBytes + roundUpSize(nbSeq * chunks * recordBytes, kCascadeInfoBytes);
            cudaLaunchAttribute attr[1];
            auto const cfg = launchConfig(dim3(static_cast<uint32_t>(chunks), nbRowGrps, nbKHeads),
                dim3(cascade::ctaThreads), smem.prefix, stream, a.enablePdl, attr);
            TLLM_CUDA_CHECK(cudaLaunchKernelEx(
                &cfg, kernel_mha_prefix, nbKHeads, a.qScale, q, cacheList, a.batch, a.kvCacheScale, cascadeInfo));
        }
    }

    cudaLaunchAttribute attr[1];
    dim3 const block(warp_size * ctaShapeInWarps.x, ctaShapeInWarps.y, ctaShapeInWarps.z);
    auto const cfg = launchConfig(dim3(a.multiBlock, nbKHeads, a.batch), block, smem.mha, stream, a.enablePdl, attr);
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel_mha, nbKHeads, a.slidingWindowSize, a.qScale, output, q,
        static_cast<float const*>(nullptr), cacheList, a.batch, a.kvCacheScale, a.semaphores, scratch,
        static_cast<uint32_t const*>(cascadeInfo), nbPrefixParts));
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
