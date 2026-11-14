/*
 * MIT License
 *
 * Copyright (c) 2025 DeepSeek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
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
 *
 * reference: https://github.com/deepseek-ai/FlashMLA (csrc/sm100/decode/head64)
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/csa2SparseDecode/csa2SparseDecode.h"
#include "tensorrt_llm/kernels/csa2SparseDecode/csa2SparseDecodeKernel.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cuda.h>
#include <initializer_list>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::csa2SparseDecode
{

namespace
{

constexpr int kBlockKeys = 64;
constexpr int64_t kMaxTmaRows = int64_t{1} << 31;
constexpr int kMaxDevices = 64;

int blocksPerToken(int swaTopk, int mainTopk)
{
    return std::max((swaTopk + kBlockKeys - 1) / kBlockKeys, 1) + (mainTopk + kBlockKeys - 1) / kBlockKeys;
}

// ceil(2^64 / d) for 2 <= d < 2^32 (0 encodes d == 1); see divMagic().
uint64_t divisionMagic(int64_t d)
{
    return d <= 1 ? 0 : ~uint64_t{0} / static_cast<uint64_t>(d) + 1;
}

size_t partialOutputBytes(int numCtas)
{
    size_t const bytes = size_t{2} * numCtas * kNumHeads * kHeadDim * sizeof(float);
    return (bytes + kWorkspaceAlignment - 1) / kWorkspaceAlignment * kWorkspaceAlignment;
}

size_t partialLseBytes(int numCtas)
{
    return size_t{2} * numCtas * kNumHeads * sizeof(float);
}

CUtensorMap encodeTensorMap(int rank, uint64_t const* dims, uint64_t const* strideBytes, uint32_t const* box,
    void const* base, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion)
{
    CUtensorMap tensorMap{};
    uint32_t const elementStrides[3] = {1, 1, 1};
    auto const driver = common::CUDADriverWrapper::getInstance();
    TLLM_CU_CHECK(driver->cuTensorMapEncodeTiled(&tensorMap, dataType, rank, const_cast<void*>(base), dims, strideBytes,
        box, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensorMap;
}

// Whole packed rows as INT64 elements: one gather4 copies four complete rows.
CUtensorMap makeRowMap(void const* base, int64_t rows, int rowBytes)
{
    uint64_t const dims[2] = {static_cast<uint64_t>(rowBytes / 8), static_cast<uint64_t>(std::max<int64_t>(rows, 1))};
    uint64_t const strides[1] = {static_cast<uint64_t>(rowBytes)};
    uint32_t const box[2] = {static_cast<uint32_t>(rowBytes / 8), 1};
    return encodeTensorMap(2, dims, strides, box, base, CU_TENSOR_MAP_DATA_TYPE_INT64, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
}

// [tokens, 64, 512] with element strides; strides are in elements of elementBytes.
CUtensorMap makeHeadMap(void const* base, int64_t tokens, int64_t tokenStride, int64_t headStride, int elementBytes,
    uint32_t boxChannels, uint32_t boxHeads, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion)
{
    uint64_t const dims[3] = {kHeadDim, kNumHeads, static_cast<uint64_t>(tokens)};
    uint64_t const strides[2]
        = {static_cast<uint64_t>(headStride * elementBytes), static_cast<uint64_t>(tokenStride * elementBytes)};
    uint32_t const box[3] = {boxChannels, boxHeads, 1};
    return encodeTensorMap(3, dims, strides, box, base, dataType, swizzle, l2Promotion);
}

bool isAligned(void const* pointer, size_t alignment)
{
    return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

void checkParams(Csa2SparseDecodeParams const& p, Csa2SparseDecodeSchedule const& s)
{
    TLLM_CHECK_WITH_INFO(p.numTokens > 0, "CSA2 sparse decode needs at least one query row");
    TLLM_CHECK_WITH_INFO(
        s.numCtas >= 1 && s.fullPerCta >= 0 && s.remCtas >= 0, "CSA2 sparse decode schedule is invalid");
    TLLM_CHECK_WITH_INFO(
        int64_t{s.fullPerCta} * s.numCtas <= p.numTokens, "CSA2 sparse decode schedule does not match the token count");
    TLLM_CHECK_WITH_INFO(p.q != nullptr && isAligned(p.q, 16) && p.qTokenStride % (p.qFp8 ? 16 : 8) == 0
            && p.qHeadStride % (p.qFp8 ? 16 : 8) == 0,
        "CSA2 sparse decode Q must be 16-B aligned with 16-B multiple strides");
    if (p.outFp8 != nullptr)
    {
        TLLM_CHECK_WITH_INFO(isAligned(p.outFp8, 32) && p.outScale != nullptr && isAligned(p.outScale, 4)
                && p.positions != nullptr && p.ropeCosSin != nullptr && isAligned(p.ropeCosSin, 16),
            "CSA2 sparse decode MXFP8 output must be 32-B aligned with scales, positions and a 16-B aligned "
            "cos/sin table");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            p.out != nullptr && isAligned(p.out, 16) && p.outTokenStride % 8 == 0 && p.outHeadStride % 8 == 0,
            "CSA2 sparse decode output must be 16-B aligned with strides that are multiples of 8 elements");
    }
    TLLM_CHECK_WITH_INFO(p.swaPool != nullptr && isAligned(p.swaPool, 16) && p.swaRows >= 0 && p.swaRows < kMaxTmaRows,
        "CSA2 sparse decode SWA pool must be 16-B aligned with fewer than 2^31 rows");
    TLLM_CHECK_WITH_INFO(p.swaTopk > 0 && p.swaTopk % 2 == 0, "CSA2 sparse decode SWA width must be even and positive");
    TLLM_CHECK_WITH_INFO(p.swaIndices != nullptr && isAligned(p.swaIndices, 16) && p.swaIndexStride % 2 == 0,
        "CSA2 sparse decode SWA indices must be 16-B aligned with an even row stride");
    TLLM_CHECK_WITH_INFO(p.mainTopk >= 0 && p.mainTopk % kBlockKeys == 0,
        "CSA2 sparse decode main width must be a multiple of 64 (or 0 for SWA-only layers)");
    if (p.mainTopk > 0)
    {
        TLLM_CHECK_WITH_INFO(
            p.mainPool != nullptr && isAligned(p.mainPool, 16) && p.mainRows >= 0 && p.mainRows < kMaxTmaRows,
            "CSA2 sparse decode main pool must be 16-B aligned with fewer than 2^31 rows");
        TLLM_CHECK_WITH_INFO(p.mainIndices != nullptr && isAligned(p.mainIndices, 8) && p.mainIndexStride % 2 == 0,
            "CSA2 sparse decode main indices must be 8-B aligned with an even row stride");
        if (p.mainPageTable != nullptr)
        {
            TLLM_CHECK_WITH_INFO(p.tokenRequests != nullptr && p.tokenVisible != nullptr,
                "CSA2 sparse decode logical main indices need per-token requests and visible lengths");
            TLLM_CHECK_WITH_INFO(p.mainPageShift >= 0 && p.mainPageShift < 31 && p.mainPageTableRows >= 0
                    && p.mainPageTableCols >= 0 && p.mainPageTableStride >= 0,
                "CSA2 sparse decode page mapping is invalid");
        }
    }
    int64_t const remTokens = p.numTokens - int64_t{s.fullPerCta} * s.numCtas;
    if (remTokens > 0)
    {
        // Every remainder chunk must hold a block and run on a launched CTA, every remainder token needs an arrival
        // counter, and no token may be cut into more parts than the combine kernel merges.
        int64_t const remBlocks = remTokens * blocksPerToken(p.swaTopk, p.mainTopk);
        bool const mergeable
            = s.remCtas <= remTokens || (s.remCtas % remTokens == 0 && s.remCtas / remTokens <= kMaxCombineSplits);
        TLLM_CHECK_WITH_INFO(s.remCtas > 0 && s.remCtas <= s.numCtas && s.remCtas <= remBlocks
                && remTokens <= kMaxSplitTokens && mergeable,
            "CSA2 sparse decode schedule does not match the token count and top-k widths");
    }
    TLLM_CHECK_WITH_INFO(p.attnSinks != nullptr, "CSA2 sparse decode needs the attention sinks");
    TLLM_CHECK_WITH_INFO(p.workspace != nullptr && isAligned(p.workspace, kWorkspaceAlignment),
        "CSA2 sparse decode workspace must be 256-B aligned");
}

// The kernel needs more than the default 48 KB of dynamic shared memory; the attribute is set once per device.
void ensureSharedMemoryAttribute()
{
    static std::array<std::atomic<bool>, kMaxDevices> configured{};
    int const device = common::getDevice();
    if (device < kMaxDevices && configured[device].load(std::memory_order_acquire))
    {
        return;
    }
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(
        csa2SparseDecodeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(Csa2Kernel::SMEM_SIZE)));
    if (device < kMaxDevices)
    {
        configured[device].store(true, std::memory_order_release);
    }
}

void launchAttention(Csa2SparseDecodeParams const& p, KernelParams const& kp, cudaStream_t stream)
{
    Csa2Kernel::TmaParams tma{};
    tma.tensor_map_swa = makeRowMap(p.swaPool, p.swaRows, kSwaRowBytes);
    tma.tensor_map_main = p.mainTopk > 0 ? makeRowMap(p.mainPool, p.mainRows, kMainRowBytes) : tma.tensor_map_swa;
    tma.tensor_map_q = p.qFp8
        ? makeHeadMap(p.q, p.numTokens, p.qTokenStride, p.qHeadStride, 1, 128, 64, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
        : makeHeadMap(p.q, p.numTokens, p.qTokenStride, p.qHeadStride, 2, 64, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    // An MXFP8 output stores no BF16 tile; its maps then only need a valid address.
    bool const bf16Out = p.outFp8 == nullptr;
    void* const out = bf16Out ? p.out : p.workspace;
    int64_t const outTokenStride = bf16Out ? p.outTokenStride : int64_t{kNumHeads} * kHeadDim;
    int64_t const outHeadStride = bf16Out ? p.outHeadStride : kHeadDim;
    tma.tensor_map_out = makeHeadMap(out, p.numTokens, outTokenStride, outHeadStride, 2, 32, 32,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    tma.tensor_map_oaccum = makeHeadMap(kp.o_accum, int64_t{2} * kp.num_ctas, int64_t{kNumHeads} * kHeadDim, kHeadDim,
        4, 16, 32, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    tma.tensor_map_out_wide = makeHeadMap(out, p.numTokens, outTokenStride, outHeadStride, 2, 64, 32,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    tma.tensor_map_oaccum_wide
        = makeHeadMap(kp.o_accum, int64_t{2} * kp.num_ctas, int64_t{kNumHeads} * kHeadDim, kHeadDim, 4, 32, 32,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE);

    auto const kernel = &csa2SparseDecodeKernel;
    ensureSharedMemoryAttribute();
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = p.enablePdl ? 1 : 0;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1, kp.num_ctas, 1);
    config.blockDim = dim3(Csa2Kernel::NUM_THREADS, 1, 1);
    config.dynamicSmemBytes = Csa2Kernel::SMEM_SIZE;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, kp, tma));
}

void launchCombine(Csa2SparseDecodeParams const& p, KernelParams const& kp, cudaStream_t stream)
{
    int const remTokens = kp.num_tokens - kp.full_per_cta * kp.num_ctas;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = p.enablePdl ? 1 : 0;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(remTokens, kNumHeads / kCombineHeadsPerCta, 1);
    config.blockDim = dim3(256, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, csa2SparseDecodeCombineKernel, kp));
}

} // namespace

Csa2SparseDecodeSchedule makeSchedule(int numTokens, int swaTopk, int mainTopk, int numSms)
{
    TLLM_CHECK_WITH_INFO(numTokens >= 0 && numSms > 0 && numSms <= kMaxSplitTokens,
        "CSA2 sparse decode schedule needs tokens >= 0 and 0 < SMs <= kMaxSplitTokens");
    int const nb = blocksPerToken(swaTopk, mainTopk);
    // Every CTA takes `full` whole tokens. The rem remaining tokens are either given whole to rem CTAs, or each of them
    // is cut into s near-equal parts processed by s CTAs (remCtas = s * rem, so every chunk boundary is a token
    // boundary: no CTA holds parts of two tokens, which would cost a second FP32 partial epilogue and a token switch).
    int const full = numTokens / numSms;
    int const rem = numTokens - full * numSms;
    Csa2SparseDecodeSchedule best{full > 0 ? numSms : std::max(rem, 1), full, rem};
    if (rem == 0)
    {
        return best;
    }
    // Estimated critical path in 64-key-block periods: the busiest CTA's blocks, plus the cost of a split schedule
    // (FP32 partial epilogue, partial traffic and the combine kernel): 3 periods when no CTA owns whole tokens (the
    // combine then runs after every CTA), 2 when the split parts are processed ahead of whole tokens (the combine
    // overlaps their tail under PDL), plus 3 periods per CTA-wide wave of parts. Fitted to graph-replay measurements on
    // GB300; costs are scaled by numSms to stay integral. Ties keep fewer parts.
    int64_t bestCost = int64_t{numSms} * (full + 1) * nb;
    for (int s = 2; s <= std::min(kMaxCombineSplits, nb) && int64_t{s} * rem <= numSms; ++s)
    {
        int64_t const cost
            = int64_t{numSms} * (int64_t{full} * nb + (nb + s - 1) / s + (full > 0 ? 2 : 3)) + int64_t{3} * s * rem;
        if (cost < bestCost)
        {
            best = Csa2SparseDecodeSchedule{full > 0 ? numSms : s * rem, full, s * rem};
            bestCost = cost;
        }
    }
    return best;
}

bool scheduleHasSplit(int numTokens, Csa2SparseDecodeSchedule const& schedule)
{
    int const rem = numTokens - schedule.fullPerCta * schedule.numCtas;
    return rem > 0 && rem % std::max(schedule.remCtas, 1) != 0;
}

size_t getWorkspaceSize(Csa2SparseDecodeSchedule const& schedule)
{
    return partialOutputBytes(schedule.numCtas) + partialLseBytes(schedule.numCtas);
}

size_t getMaxWorkspaceSize(int numSms)
{
    // makeSchedule never uses more CTAs than SMs.
    return getWorkspaceSize(Csa2SparseDecodeSchedule{numSms, 0, 0});
}

bool isSupported()
{
#if defined(PLACEHOLDER_KERNELS)
    return false;
#else
    // Per device: 0 not queried yet, 1 supported, 2 unsupported.
    static std::array<std::atomic<int>, kMaxDevices> cached{};
    int const device = common::getDevice();
    if (device < kMaxDevices)
    {
        int const known = cached[device].load(std::memory_order_acquire);
        if (known != 0)
        {
            return known == 1;
        }
    }
    bool supported = common::isSM100Family();
    if (supported)
    {
        // The family builds carry no image for some members; ask the runtime for one.
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(&attributes, csa2SparseDecodeKernel) != cudaSuccess)
        {
            static_cast<void>(cudaGetLastError());
            supported = false;
        }
        else
        {
            supported = attributes.sharedSizeBytes + Csa2Kernel::SMEM_SIZE
                <= static_cast<size_t>(common::getMaxSharedMemoryPerBlockOptin());
        }
    }
    if (device < kMaxDevices)
    {
        cached[device].store(supported ? 1 : 2, std::memory_order_release);
    }
    return supported;
#endif
}

void run(Csa2SparseDecodeParams const& params, Csa2SparseDecodeSchedule const& schedule, cudaStream_t stream)
{
    checkParams(params, schedule);
    KernelParams kp{};
    kp.num_tokens = params.numTokens;
    kp.swa_indices = static_cast<int64_t const*>(params.swaIndices);
    kp.stride_swa_idx = params.swaIndexStride;
    kp.swa_topk = params.swaTopk;
    kp.swa_rows = params.swaRows;
    kp.main_indices = params.mainTopk > 0 ? params.mainIndices : nullptr;
    kp.stride_main_idx = params.mainIndexStride;
    kp.main_topk = params.mainTopk;
    kp.main_rows = params.mainTopk > 0 ? params.mainRows : 0;
    bool const logical = params.mainTopk > 0 && params.mainPageTable != nullptr;
    kp.main_page_table = logical ? params.mainPageTable : nullptr;
    kp.main_page_table_stride = logical ? params.mainPageTableStride : 0;
    kp.main_page_table_rows = logical ? params.mainPageTableRows : 0;
    kp.main_page_table_cols = logical ? params.mainPageTableCols : 0;
    kp.main_page_shift = logical ? params.mainPageShift : 0;
    kp.main_max_positions = logical ? params.mainMaxPositions : 0;
    kp.token_requests = logical ? params.tokenRequests : nullptr;
    kp.token_visible = logical ? params.tokenVisible : nullptr;
    kp.attn_sink = params.attnSinks;
    kp.sm_scale_log2 = params.softmaxScale * 1.4426950408889634F;
    kp.q_fp8 = params.qFp8 ? 1 : 0;
    kp.q = params.q;
    kp.stride_q_tok = params.qTokenStride;
    kp.stride_q_head = params.qHeadStride;
    kp.out = static_cast<bf16*>(params.out);
    kp.stride_o_tok = params.outTokenStride;
    kp.stride_o_head = params.outHeadStride;
    kp.out_fp8 = static_cast<uint8_t*>(params.outFp8);
    kp.out_scale = static_cast<uint8_t*>(params.outScale);
    kp.positions = params.positions;
    kp.rope_cos_sin = params.ropeCosSin;
    auto* const workspace = static_cast<char*>(params.workspace);
    kp.o_accum = reinterpret_cast<float*>(workspace);
    kp.lse_accum = reinterpret_cast<float*>(workspace + partialOutputBytes(schedule.numCtas));
    kp.num_ctas = schedule.numCtas;
    kp.full_per_cta = schedule.fullPerCta;
    kp.rem_ctas = schedule.remCtas;
    kp.pdl = params.enablePdl ? 1 : 0;
    int const nb = blocksPerToken(params.swaTopk, params.mainTopk);
    int64_t const remBlocks = int64_t{params.numTokens - schedule.fullPerCta * schedule.numCtas} * nb;
    int const remCtas = std::max(schedule.remCtas, 1);
    TLLM_CHECK_WITH_INFO(remBlocks * remCtas < (int64_t{1} << 32), "CSA2 sparse decode schedule is too large");
    kp.magic_rem_ctas = divisionMagic(remCtas);
    kp.magic_rem_blocks = divisionMagic(std::max<int64_t>(remBlocks, 1));
    kp.magic_nb = divisionMagic(nb);

    launchAttention(params, kp, stream);
    if (scheduleHasSplit(params.numTokens, schedule))
    {
        launchCombine(params, kp, stream);
    }
}

} // namespace kernels::csa2SparseDecode

TRTLLM_NAMESPACE_END
