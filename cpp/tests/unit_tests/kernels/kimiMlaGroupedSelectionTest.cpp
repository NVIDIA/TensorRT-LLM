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
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunnerParams.h"

namespace
{

using namespace tensorrt_llm::kernels;

// Reinterpret a BF16 bit pattern (uint16_t) as the corresponding FP32 value
// by zero-extending into the high half of a 32-bit float. Lossless because
// BF16's exponent + mantissa share FP32's high 16 bits.
inline float bf16BitsToFloat(uint16_t bits)
{
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

// BF16 1.0 bit pattern (sign=0, exp=127=0x7F, mantissa=0): 0x3F80.
constexpr uint16_t kBf16One = 0x3F80;
// BF16 2.0 bit pattern (sign=0, exp=128=0x80, mantissa=0): 0x4000.
constexpr uint16_t kBf16Two = 0x4000;

// BF16 NaN-like sentinel used to pre-fill the output buffer. cudaMemset(..., 0xFF)
// puts 0xFFFF in every BF16 slot (sign=1, exp=0xFF, mantissa nonzero -> NaN).
constexpr uint16_t kBf16NaNSentinel = 0xFFFF;

// Multi-CTA-KV semaphore/counter buffer: sized generously so the test does not
// depend on the exact per-CTA semaphore layout of the current kernels.
constexpr size_t kCounterBytes = size_t{1} << 20;

// Convert FP32 to BF16 with simple truncation (top 16 bits of the FP32 word).
// Matches what TensorRT-LLM's BF16 storage convention is for these tests.
inline uint16_t floatToBf16(float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

// FP32 CPU reference for MLA generation with the Kimi shape (B=1, q_len=4, 16 heads,
// HQk=576, HV=512, paged KV). Causal spec-decode mask: query token qi sees kv indices
// [0, seqLenKv - q_len + 1 + qi).
//
// MLA softmax scale is 1/sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) = 1/sqrt(192), not
// 1/sqrt(headDimQk=576). The caller must plumb the same scale to the kernel: pass
// scaleSoftmaxLog2 = (1/sqrt(192)) * log2(e) and set params.mScaleQ = sqrt(192/576) so the
// host-side softmaxScale = (1 / (sqrt(headDimQk) * mScaleQ)) * log2(e) resolves identically.
void mlaReferenceCpu(int seqLenQ, int seqLenKv, int numHeadsQ, int headDimQk, int headDimV, int numTokensPerPage,
    std::vector<uint16_t> const& Q, std::vector<uint16_t> const& KV, float softmaxScale, std::vector<uint16_t>& Out)
{
    // Q shape: [seqLenQ * numHeadsQ * headDimQk]
    // KV shape (paged, single page-table batch): [seqLenKv * headDimQk] when
    //   page table is identity. We assume the caller has built KV with pages
    //   in sequential order so logical KV index kv just indexes KV[kv*headDimQk + d].
    // Out shape: [seqLenQ * numHeadsQ * headDimV]
    auto qAt = [&](int q, int h, int d) -> float
    { return bf16BitsToFloat(Q[static_cast<size_t>(q) * numHeadsQ * headDimQk + h * headDimQk + d]); };
    auto kvAt = [&](int kv, int d) -> float
    { return bf16BitsToFloat(KV[static_cast<size_t>(kv) * headDimQk + d]); };
    (void) numTokensPerPage; // page-table identity is captured by the caller's hPageIdx[i]=i.

    std::vector<float> scores(seqLenKv);
    std::vector<float> weights(seqLenKv);
    for (int q = 0; q < seqLenQ; ++q)
    {
        // Spec-decode causal upper bound for this query position.
        int const validEnd = seqLenKv - seqLenQ + 1 + q;
        for (int h = 0; h < numHeadsQ; ++h)
        {
            // Scores Q[q,h] dot K[kv]; K spans the full headDimQk (kv_lora_rank + qk_rope).
            for (int kv = 0; kv < seqLenKv; ++kv)
            {
                float s = 0.f;
                for (int d = 0; d < headDimQk; ++d)
                {
                    s += qAt(q, h, d) * kvAt(kv, d);
                }
                scores[kv] = s * softmaxScale;
            }
            // Apply spec-decode causal mask: positions >= validEnd are -inf.
            for (int kv = validEnd; kv < seqLenKv; ++kv)
            {
                scores[kv] = -std::numeric_limits<float>::infinity();
            }
            // Numerically stable softmax.
            float maxScore = -std::numeric_limits<float>::infinity();
            for (int kv = 0; kv < seqLenKv; ++kv)
            {
                maxScore = std::max(maxScore, scores[kv]);
            }
            float sumExp = 0.f;
            for (int kv = 0; kv < seqLenKv; ++kv)
            {
                weights[kv] = (scores[kv] == -std::numeric_limits<float>::infinity()) ? 0.f
                                                                                      : std::exp(scores[kv] - maxScore);
                sumExp += weights[kv];
            }
            float const invSum = (sumExp > 0.f) ? (1.f / sumExp) : 0.f;
            for (int kv = 0; kv < seqLenKv; ++kv)
            {
                weights[kv] *= invSum;
            }
            // Output Out[q,h,d_v] = sum_kv weights[kv] * V[kv, d_v], V is first kv_lora_rank
            // of headDimQk.
            for (int d_v = 0; d_v < headDimV; ++d_v)
            {
                float o = 0.f;
                for (int kv = 0; kv < seqLenKv; ++kv)
                {
                    o += weights[kv] * kvAt(kv, d_v);
                }
                Out[static_cast<size_t>(q) * numHeadsQ * headDimV + h * headDimV + d_v] = floatToBf16(o);
            }
        }
    }
}

// Selection + correctness tests for the SM103 grouped-token MLA generation cubin
// (tileSizeQ=64 groups tokensQ and headsQ into one CTA) for the Kimi K2.5/K2.6 EAGLE-3
// decode shape. Selection is driven by the trtllm-gen static-library autotuner and the
// grouped kernelMetaInfo.h row; the TLLM_FMHA_TEST_HOOKS probe asserts the resolved cubin.

class KimiMlaGroupedSelectionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int sm = tensorrt_llm::common::getSMVersion();
        if (sm != kSM_103)
        {
            GTEST_SKIP() << "Kimi grouped MLA cubin is registered only for SM103 (B300). "
                         << "Current SM=" << sm << ". Skipping selection test.";
        }
        // Establish a CUDA context so cuModuleLoadData inside the runner ctor
        // can resolve. cudaFree(0) is the canonical way to lazy-init the
        // primary context for the current device without allocating. Check
        // both return values so environment failures surface clearly.
        ASSERT_EQ(cudaSuccess, cudaSetDevice(0)) << "cudaSetDevice(0) failed";
        ASSERT_EQ(cudaSuccess, cudaFree(nullptr)) << "cudaFree(nullptr) failed (CUDA context init)";
    }

    // Build a RunnerParams that mirrors the Kimi K2.5/K2.6 EAGLE-3 MLA decode
    // call shape (B=1, q_len=4, 16 heads, HQk=576, HV=512, paged KV P32, BF16).
    // mIsMlaGen is auto-derived from mHeadDimQk==576 && mHeadDimV==512 inside
    // parseOptionsFromRunnerParams (see fmhaKernels.h:1209). Same for
    // mIsCausalSpecDecodingGen from mMaxSeqLenQ>1 && !mIsSpecDecTree.
    static void buildKimiParams(TllmGenFmhaRunnerParams& p, int maxSeqLenQ = 4, int headDimQk = 576, int headDimV = 512,
        int numHeadsQ = 16)
    {
        std::memset(&p, 0, sizeof(p));
        p.mQkvLayout = QkvLayout::PagedKv;
        // Callers pass Dense for generation; the autotuner rewrites the mask for
        // causal spec-decode generation kernels.
        p.mMaskType = TrtllmGenAttentionMaskType::Dense;
        p.mIsSpecDecTree = false;
        p.mKernelType = FmhaKernelType::Generation;
        p.mTileScheduler = TileScheduler::Static;
        p.mMultiCtasKvMode = true;
        p.mHeadDimQk = headDimQk;
        p.mHeadDimV = headDimV;
        p.mHeadDimQkNope = 512;
        p.mNumHeadsQ = numHeadsQ;
        p.mNumHeadsKv = 1;
        p.mNumHeadsQPerKv = numHeadsQ;
        p.mBatchSize = 1;
        p.mMaxSeqLenQ = maxSeqLenQ;
        p.mMaxSeqLenKv = 1024;
        p.mNumTokensPerPage = 32;
        p.mChunkedAttentionSize = INT_MAX;
        p.mAttentionWindowSize = INT_MAX;
        p.mScaleQ = 1.f;
        p.mSparseAttention = SparseType::None;
        p.mSparseTopK = 0;
        // Query the actual device's SM count instead of hard-coding B300=148;
        // the autotuner's CTA heuristics consume this value and a wrong count
        // can perturb the selected tuple.
        p.mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
        p.mSumOfSeqLensQ = maxSeqLenQ * p.mBatchSize;
        p.mSumOfSeqLensKv = p.mMaxSeqLenKv * p.mBatchSize;
        p.mMaxNumPagesPerSeqKv = (p.mMaxSeqLenKv + p.mNumTokensPerPage - 1) / p.mNumTokensPerPage;
    }
};

// The Kimi shape must resolve to the SM103 grouped cubin.
TEST_F(KimiMlaGroupedSelectionTest, KimiShape_SelectsGroupedCubin)
{
    TllmGenFmhaRunner runner(DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16);
    TllmGenFmhaRunnerParams params;
    buildKimiParams(params);

    auto [supported, info] = runner.isSupportedWithInfo(params);
    EXPECT_TRUE(supported) << "Cubin lookup failed for Kimi MLA grouped shape: " << info;

    // Exact selected-cubin proof via the test-only probe: consults the matched
    // mFunctions[hashId] -> mKernelMeta entry directly.
    auto selected = runner.probeKernelSelectionForTesting(params);
    EXPECT_TRUE(selected.mFound) << "probeKernelSelectionForTesting reported no match";
    EXPECT_FALSE(selected.mUsedNvrtc) << "probe took the NVRTC path instead of the grouped cubin";
    // The autotuner picks the headDimPerCtaV / reduction variant; assert the grouped MLA family.
    EXPECT_NE(selected.mFuncName.find("HQk576HV512"), std::string::npos) << selected.mFuncName;
    EXPECT_NE(selected.mFuncName.find("Q64Kv128"), std::string::npos) << selected.mFuncName;
    EXPECT_NE(selected.mFuncName.find("GroupedKeepsAbForGen"), std::string::npos) << selected.mFuncName;
    EXPECT_TRUE(selected.mGroupsHeadsQ);
    EXPECT_TRUE(selected.mGroupsTokensHeadsQ);
}

// q_len=1 (plain decode, M=16) must NOT select the grouped Q64 cubin.
TEST_F(KimiMlaGroupedSelectionTest, NonKimiShape_DoesNotSelectGroupedCubin)
{
    TllmGenFmhaRunner runner(DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16);
    TllmGenFmhaRunnerParams params;
    buildKimiParams(params, /*maxSeqLenQ=*/1);

    auto [supported, info] = runner.isSupportedWithInfo(params);
    EXPECT_TRUE(supported) << "MLA decode path reported unsupported with q_len=1: " << info;

    auto selected = runner.probeKernelSelectionForTesting(params);
    EXPECT_FALSE(selected.mGroupsTokensHeadsQ) << "Grouped cubin selected for a non-spec-decode shape: "
                                               << selected.mFuncName;
}

// Smoke run with Q=K=V=zero: both the FMHA and separate-reduction kernels must launch
// cleanly, every output slot must be written (no surviving 0xFFFF sentinel), and the
// output must be exactly zero.
TEST_F(KimiMlaGroupedSelectionTest, KimiShape_RunSmokeSucceeds)
{
    TllmGenFmhaRunner runner(DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16);
    TllmGenFmhaRunnerParams params;
    buildKimiParams(params);

    int const seqLenQ = params.mMaxSeqLenQ;                       // 4
    int const seqLenKv = params.mMaxSeqLenKv;                     // 1024
    int const batchSize = params.mBatchSize;                      // 1
    int const numHeadsQ = params.mNumHeadsQ;                      // 16
    int const headDimQk = params.mHeadDimQk;                      // 576
    int const headDimV = params.mHeadDimV;                        // 512
    int const numTokensPerPage = params.mNumTokensPerPage;        // 32
    int const maxNumPagesPerSeqKv = params.mMaxNumPagesPerSeqKv;  // ceilDiv(1024,32)=32
    int const numPages = maxNumPagesPerSeqKv * batchSize;         // 32

    constexpr size_t kBf16 = sizeof(uint16_t);

    // Allocate device buffers. Sizes mirror the Kimi MLA paged-KV layout:
    //   Q   : [batch * seqLenQ, numHeadsQ, headDimQk]
    //   KV  : [num_pages, page_size, headDimQk] - shared K/V cache, MLA layout
    //   O   : [batch * seqLenQ, numHeadsQ, headDimV]
    size_t const qBytes = static_cast<size_t>(batchSize) * seqLenQ * numHeadsQ * headDimQk * kBf16;
    size_t const kvBytes = static_cast<size_t>(numPages) * numTokensPerPage * headDimQk * kBf16;
    size_t const oBytes = static_cast<size_t>(batchSize) * seqLenQ * numHeadsQ * headDimV * kBf16;
    // multiCtasKv scratch holds partialStats (float2 per CTA) + partialO
    // (float per Q tile slot * headDimV). 64 MB is comfortably above what
    // the SM103 grouped grid requires (~4 MB) for this shape.
    size_t const scratchBytes = static_cast<size_t>(64) * 1024 * 1024;

    void* dQ = nullptr;
    void* dKV = nullptr;
    void* dO = nullptr;
    void* dScratch = nullptr;
    int32_t* dCounter = nullptr;
    int32_t* dPageIdx = nullptr;
    int32_t* dSeqLensKv = nullptr;
    int32_t* dCumSeqLensQ = nullptr;
    int32_t* dCumSeqLensKv = nullptr;
    float* dScaleSoftmaxLog2 = nullptr;
    float* dOutputScale = nullptr;
    cudaStream_t stream = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dQ, qBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dKV, kvBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dO, oBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dScratch, scratchBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCounter, kCounterBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dPageIdx, batchSize * maxNumPagesPerSeqKv * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dSeqLensKv, batchSize * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCumSeqLensQ, (batchSize + 1) * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCumSeqLensKv, (batchSize + 1) * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dScaleSoftmaxLog2, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dOutputScale, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // Zero the input + scratch buffers. With Q=K=V=zero, the expected output
    // is zero (uniform softmax over zero scores -> uniform attention; uniform
    // weights * V=zero -> zero per element). Pre-fill dO with a NaN sentinel
    // (BF16 bits 0xFFFF = sign=1, exp=255, mantissa=0x7F -> NaN) so we can
    // detect whether the kernel actually wrote every output position. Any
    // 0xFFFF that survives after run() means the kernel skipped that slot.
    ASSERT_EQ(cudaSuccess, cudaMemset(dQ, 0, qBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dKV, 0, kvBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dO, 0xFF, oBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dScratch, 0, scratchBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dCounter, 0, kCounterBytes));

    // Page table: sequential page indices [0, 1, ..., numPages-1].
    std::vector<int32_t> hPageIdx(batchSize * maxNumPagesPerSeqKv);
    for (int i = 0; i < static_cast<int>(hPageIdx.size()); ++i)
    {
        hPageIdx[i] = i;
    }
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dPageIdx, hPageIdx.data(), hPageIdx.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // seqLensKv = [seqLenKv]; cumSeqLensQ = [0, seqLenQ]; cumSeqLensKv = [0, seqLenKv].
    std::vector<int32_t> const hSeqLensKv = {seqLenKv};
    std::vector<int32_t> const hCumSeqLensQ = {0, seqLenQ};
    std::vector<int32_t> const hCumSeqLensKv = {0, seqLenKv};
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(
            dSeqLensKv, hSeqLensKv.data(), batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dCumSeqLensQ, hCumSeqLensQ.data(), (batchSize + 1) * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dCumSeqLensKv, hCumSeqLensKv.data(), (batchSize + 1) * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));

    // softmaxScale_log2 = (1 / sqrt(headDimQk)) * log2(e). Mirrors the host
    // value setFmhaData computes from params.mScaleQ=1 and mHeadDimQk.
    float const hScaleSoftmaxLog2 = (1.f / std::sqrt(static_cast<float>(headDimQk))) * static_cast<float>(M_LOG2E);
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dScaleSoftmaxLog2, &hScaleSoftmaxLog2, sizeof(float), cudaMemcpyHostToDevice, stream));
    float const hOutputScale = 1.0f;
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dOutputScale, &hOutputScale, sizeof(float), cudaMemcpyHostToDevice, stream));

    // Wire up RunnerParams pointers.
    params.qPtr = dQ;
    params.kvPtr = dKV;
    params.oPtr = dO;
    params.kvPageIdxPtr = dPageIdx;
    params.seqLensKvPtr = dSeqLensKv;
    params.cumSeqLensQPtr = dCumSeqLensQ;
    params.cumSeqLensKvPtr = dCumSeqLensKv;
    params.scaleSoftmaxLog2Ptr = dScaleSoftmaxLog2;
    params.outputScalePtr = dOutputScale;
    params.multiCtasKvScratchPtr = dScratch;
    params.multiCtasKvCounterPtr = dCounter;
    params.stream = stream;

    // Sync any pending host-to-device copies so the kernel reads valid data.
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    ASSERT_NO_THROW(runner.run(params));

    // Make sure both the FMHA kernel and the separate reduction kernel
    // finished without runtime errors.
    cudaError_t const syncErr = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, syncErr) << "cudaStreamSynchronize after run: " << cudaGetErrorString(syncErr);
    cudaError_t const lastErr = cudaGetLastError();
    EXPECT_EQ(cudaSuccess, lastErr) << "cudaGetLastError after run: " << cudaGetErrorString(lastErr);

    // Output equivalence: pull dO back and verify (a) every BF16 element was
    // overwritten (no 0xFFFF NaN sentinel survives -> the kernel actually
    // wrote every output slot), (b) no NaN or Inf was produced, (c) every
    // element equals exactly 0.0 because softmax(QK^T)=uniform and V=zero
    // implies output=zero with no accumulation error.
    size_t const numOutputElems = oBytes / sizeof(uint16_t);
    std::vector<uint16_t> hO(numOutputElems);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(hO.data(), dO, oBytes, cudaMemcpyDeviceToHost));

    int numSentinelRemnants = 0;
    int numNaN = 0;
    int numInf = 0;
    int numNonZero = 0;
    for (uint16_t bits : hO)
    {
        if (bits == kBf16NaNSentinel)
        {
            ++numSentinelRemnants;
        }
        float const f = bf16BitsToFloat(bits);
        if (std::isnan(f))
        {
            ++numNaN;
        }
        if (std::isinf(f))
        {
            ++numInf;
        }
        if (f != 0.0f)
        {
            ++numNonZero;
        }
    }
    EXPECT_EQ(0, numSentinelRemnants)
        << "0xFFFF NaN sentinels remain: " << numSentinelRemnants << "/" << numOutputElems
        << " elements not written by the kernel (zero input should still cover every output position)";
    EXPECT_EQ(0, numNaN) << "Found " << numNaN << "/" << numOutputElems << " NaN elements in output";
    EXPECT_EQ(0, numInf) << "Found " << numInf << "/" << numOutputElems << " Inf elements in output";
    EXPECT_EQ(0, numNonZero) << "Zero input must produce zero output; found " << numNonZero << "/"
                             << numOutputElems << " non-zero elements";

    // Cleanup.
    cudaFree(dQ);
    cudaFree(dKV);
    cudaFree(dO);
    cudaFree(dScratch);
    cudaFree(dCounter);
    cudaFree(dPageIdx);
    cudaFree(dSeqLensKv);
    cudaFree(dCumSeqLensQ);
    cudaFree(dCumSeqLensKv);
    cudaFree(dScaleSoftmaxLog2);
    cudaFree(dOutputScale);
    cudaStreamDestroy(stream);
}

// Q=zero (uniform softmax) + KV=1.0 everywhere: output must be ~1.0 per element.
// Unlike the all-zero case this exercises V reads from the paged cache and cross-tile
// aggregation through the separate GMEM reduction kernel.
TEST_F(KimiMlaGroupedSelectionTest, KimiShape_RunSmokeConstantKVOutputsOne)
{
    TllmGenFmhaRunner runner(DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16);
    TllmGenFmhaRunnerParams params;
    buildKimiParams(params);

    int const seqLenQ = params.mMaxSeqLenQ;
    int const seqLenKv = params.mMaxSeqLenKv;
    int const batchSize = params.mBatchSize;
    int const numHeadsQ = params.mNumHeadsQ;
    int const headDimQk = params.mHeadDimQk;
    int const headDimV = params.mHeadDimV;
    int const numTokensPerPage = params.mNumTokensPerPage;
    int const maxNumPagesPerSeqKv = params.mMaxNumPagesPerSeqKv;
    int const numPages = maxNumPagesPerSeqKv * batchSize;

    constexpr size_t kBf16 = sizeof(uint16_t);
    size_t const qBytes = static_cast<size_t>(batchSize) * seqLenQ * numHeadsQ * headDimQk * kBf16;
    size_t const kvBytes = static_cast<size_t>(numPages) * numTokensPerPage * headDimQk * kBf16;
    size_t const oBytes = static_cast<size_t>(batchSize) * seqLenQ * numHeadsQ * headDimV * kBf16;
    size_t const scratchBytes = static_cast<size_t>(64) * 1024 * 1024;

    void* dQ = nullptr;
    void* dKV = nullptr;
    void* dO = nullptr;
    void* dScratch = nullptr;
    int32_t* dCounter = nullptr;
    int32_t* dPageIdx = nullptr;
    int32_t* dSeqLensKv = nullptr;
    int32_t* dCumSeqLensQ = nullptr;
    int32_t* dCumSeqLensKv = nullptr;
    float* dScaleSoftmaxLog2 = nullptr;
    float* dOutputScale = nullptr;
    cudaStream_t stream = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dQ, qBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dKV, kvBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dO, oBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dScratch, scratchBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCounter, kCounterBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dPageIdx, batchSize * maxNumPagesPerSeqKv * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dSeqLensKv, batchSize * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCumSeqLensQ, (batchSize + 1) * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCumSeqLensKv, (batchSize + 1) * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dScaleSoftmaxLog2, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dOutputScale, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // Q is all zeros. KV cache is BF16 1.0 across every entry (both the K
    // slice the kernel reads for scores and the V slice it reads for the
    // weighted sum). dO starts as NaN-sentinel so we can detect un-written
    // slots after the kernel runs.
    ASSERT_EQ(cudaSuccess, cudaMemset(dQ, 0, qBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dO, 0xFF, oBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dScratch, 0, scratchBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dCounter, 0, kCounterBytes));
    std::vector<uint16_t> const hKV(kvBytes / sizeof(uint16_t), kBf16One);
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dKV, hKV.data(), kvBytes, cudaMemcpyHostToDevice, stream));

    std::vector<int32_t> hPageIdx(batchSize * maxNumPagesPerSeqKv);
    for (int i = 0; i < static_cast<int>(hPageIdx.size()); ++i)
    {
        hPageIdx[i] = i;
    }
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dPageIdx, hPageIdx.data(), hPageIdx.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    std::vector<int32_t> const hSeqLensKv = {seqLenKv};
    std::vector<int32_t> const hCumSeqLensQ = {0, seqLenQ};
    std::vector<int32_t> const hCumSeqLensKv = {0, seqLenKv};
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dSeqLensKv, hSeqLensKv.data(), batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dCumSeqLensQ, hCumSeqLensQ.data(), (batchSize + 1) * sizeof(int32_t), cudaMemcpyHostToDevice,
            stream));
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dCumSeqLensKv, hCumSeqLensKv.data(), (batchSize + 1) * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
    float const hScaleSoftmaxLog2 = (1.f / std::sqrt(static_cast<float>(headDimQk))) * static_cast<float>(M_LOG2E);
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dScaleSoftmaxLog2, &hScaleSoftmaxLog2, sizeof(float), cudaMemcpyHostToDevice, stream));
    float const hOutputScale = 1.0f;
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dOutputScale, &hOutputScale, sizeof(float), cudaMemcpyHostToDevice, stream));

    params.qPtr = dQ;
    params.kvPtr = dKV;
    params.oPtr = dO;
    params.kvPageIdxPtr = dPageIdx;
    params.seqLensKvPtr = dSeqLensKv;
    params.cumSeqLensQPtr = dCumSeqLensQ;
    params.cumSeqLensKvPtr = dCumSeqLensKv;
    params.scaleSoftmaxLog2Ptr = dScaleSoftmaxLog2;
    params.outputScalePtr = dOutputScale;
    params.multiCtasKvScratchPtr = dScratch;
    params.multiCtasKvCounterPtr = dCounter;
    params.stream = stream;

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_NO_THROW(runner.run(params));
    cudaError_t const syncErr = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, syncErr) << "cudaStreamSynchronize after run: " << cudaGetErrorString(syncErr);
    cudaError_t const lastErr = cudaGetLastError();
    EXPECT_EQ(cudaSuccess, lastErr) << "cudaGetLastError after run: " << cudaGetErrorString(lastErr);

    size_t const numOutputElems = oBytes / sizeof(uint16_t);
    std::vector<uint16_t> hO(numOutputElems);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(hO.data(), dO, oBytes, cudaMemcpyDeviceToHost));

    int numSentinelRemnants = 0;
    int numNaN = 0;
    int numInf = 0;
    int numOutOfTolerance = 0;
    float minVal = std::numeric_limits<float>::infinity();
    float maxVal = -std::numeric_limits<float>::infinity();
    constexpr float kExpectedValue = 1.0f;
    constexpr float kAbsTolerance = 0.03f; // ~3% of |1.0|
    for (uint16_t bits : hO)
    {
        if (bits == kBf16NaNSentinel)
        {
            ++numSentinelRemnants;
        }
        float const f = bf16BitsToFloat(bits);
        if (std::isnan(f))
        {
            ++numNaN;
        }
        if (std::isinf(f))
        {
            ++numInf;
        }
        if (std::isfinite(f))
        {
            minVal = std::min(minVal, f);
            maxVal = std::max(maxVal, f);
            if (std::abs(f - kExpectedValue) > kAbsTolerance)
            {
                ++numOutOfTolerance;
            }
        }
    }
    EXPECT_EQ(0, numSentinelRemnants) << "0xFFFF NaN sentinels remain: " << numSentinelRemnants << "/" << numOutputElems;
    EXPECT_EQ(0, numNaN) << "Found " << numNaN << "/" << numOutputElems << " NaN elements";
    EXPECT_EQ(0, numInf) << "Found " << numInf << "/" << numOutputElems << " Inf elements";
    EXPECT_EQ(0, numOutOfTolerance) << "Output elements outside |x - 1.0| <= " << kAbsTolerance << ": "
                                    << numOutOfTolerance << "/" << numOutputElems << "; min=" << minVal
                                    << " max=" << maxVal;

    cudaFree(dQ);
    cudaFree(dKV);
    cudaFree(dO);
    cudaFree(dScratch);
    cudaFree(dCounter);
    cudaFree(dPageIdx);
    cudaFree(dSeqLensKv);
    cudaFree(dCumSeqLensQ);
    cudaFree(dCumSeqLensKv);
    cudaFree(dScaleSoftmaxLog2);
    cudaFree(dOutputScale);
    cudaStreamDestroy(stream);
}

// Q=zero + KV split into two halves (pages 0..15 -> 1.0, pages 16..31 -> 2.0):
// uniform attention over 1024 positions gives output ~1.5 per element. Catches
// dropped/double-counted halves of the multi-CTA reduction; per-position indexing
// is covered by the patterned-V and CPU-reference tests below. (In causal
// spec-decode the 4 packed Q tokens have slightly different visible lengths near
// the tail — far smaller than the tolerance, so no exact 1.5 check.)
TEST_F(KimiMlaGroupedSelectionTest, KimiShape_RunSmokeSplitKVOutputsOnePointFive)
{
    TllmGenFmhaRunner runner(DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16);
    TllmGenFmhaRunnerParams params;
    buildKimiParams(params);

    int const seqLenQ = params.mMaxSeqLenQ;
    int const seqLenKv = params.mMaxSeqLenKv;
    int const batchSize = params.mBatchSize;
    int const numHeadsQ = params.mNumHeadsQ;
    int const headDimQk = params.mHeadDimQk;
    int const headDimV = params.mHeadDimV;
    int const numTokensPerPage = params.mNumTokensPerPage;
    int const maxNumPagesPerSeqKv = params.mMaxNumPagesPerSeqKv;
    int const numPages = maxNumPagesPerSeqKv * batchSize;
    ASSERT_EQ(0, numPages % 2) << "split-V test assumes an even number of pages";

    constexpr size_t kBf16 = sizeof(uint16_t);
    size_t const qBytes = static_cast<size_t>(batchSize) * seqLenQ * numHeadsQ * headDimQk * kBf16;
    size_t const kvBytes = static_cast<size_t>(numPages) * numTokensPerPage * headDimQk * kBf16;
    size_t const oBytes = static_cast<size_t>(batchSize) * seqLenQ * numHeadsQ * headDimV * kBf16;
    size_t const scratchBytes = static_cast<size_t>(64) * 1024 * 1024;

    void* dQ = nullptr;
    void* dKV = nullptr;
    void* dO = nullptr;
    void* dScratch = nullptr;
    int32_t* dCounter = nullptr;
    int32_t* dPageIdx = nullptr;
    int32_t* dSeqLensKv = nullptr;
    int32_t* dCumSeqLensQ = nullptr;
    int32_t* dCumSeqLensKv = nullptr;
    float* dScaleSoftmaxLog2 = nullptr;
    float* dOutputScale = nullptr;
    cudaStream_t stream = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dQ, qBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dKV, kvBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dO, oBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dScratch, scratchBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCounter, kCounterBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dPageIdx, batchSize * maxNumPagesPerSeqKv * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dSeqLensKv, batchSize * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCumSeqLensQ, (batchSize + 1) * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCumSeqLensKv, (batchSize + 1) * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dScaleSoftmaxLog2, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dOutputScale, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    ASSERT_EQ(cudaSuccess, cudaMemset(dQ, 0, qBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dO, 0xFF, oBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dScratch, 0, scratchBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dCounter, 0, kCounterBytes));

    // Build the split KV cache on host. Layout: [page, slot_in_page, headDim].
    // Every BF16 element of a slot gets the same value (1.0 for the first half
    // of pages, 2.0 for the second half) so both the K slice (used for scores)
    // and the V slice (used for the weighted sum) are uniform within a page.
    std::vector<uint16_t> hKV(kvBytes / sizeof(uint16_t));
    size_t const elemsPerPage = static_cast<size_t>(numTokensPerPage) * headDimQk;
    int const halfPages = numPages / 2;
    for (int page = 0; page < numPages; ++page)
    {
        uint16_t const value = page < halfPages ? kBf16One : kBf16Two;
        size_t const pageStart = static_cast<size_t>(page) * elemsPerPage;
        std::fill_n(hKV.begin() + pageStart, elemsPerPage, value);
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dKV, hKV.data(), kvBytes, cudaMemcpyHostToDevice, stream));

    std::vector<int32_t> hPageIdx(batchSize * maxNumPagesPerSeqKv);
    for (int i = 0; i < static_cast<int>(hPageIdx.size()); ++i)
    {
        hPageIdx[i] = i;
    }
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dPageIdx, hPageIdx.data(), hPageIdx.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    std::vector<int32_t> const hSeqLensKv = {seqLenKv};
    std::vector<int32_t> const hCumSeqLensQ = {0, seqLenQ};
    std::vector<int32_t> const hCumSeqLensKv = {0, seqLenKv};
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dSeqLensKv, hSeqLensKv.data(), batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dCumSeqLensQ, hCumSeqLensQ.data(), (batchSize + 1) * sizeof(int32_t), cudaMemcpyHostToDevice,
            stream));
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dCumSeqLensKv, hCumSeqLensKv.data(), (batchSize + 1) * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
    float const hScaleSoftmaxLog2 = (1.f / std::sqrt(static_cast<float>(headDimQk))) * static_cast<float>(M_LOG2E);
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dScaleSoftmaxLog2, &hScaleSoftmaxLog2, sizeof(float), cudaMemcpyHostToDevice, stream));
    float const hOutputScale = 1.0f;
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dOutputScale, &hOutputScale, sizeof(float), cudaMemcpyHostToDevice, stream));

    params.qPtr = dQ;
    params.kvPtr = dKV;
    params.oPtr = dO;
    params.kvPageIdxPtr = dPageIdx;
    params.seqLensKvPtr = dSeqLensKv;
    params.cumSeqLensQPtr = dCumSeqLensQ;
    params.cumSeqLensKvPtr = dCumSeqLensKv;
    params.scaleSoftmaxLog2Ptr = dScaleSoftmaxLog2;
    params.outputScalePtr = dOutputScale;
    params.multiCtasKvScratchPtr = dScratch;
    params.multiCtasKvCounterPtr = dCounter;
    params.stream = stream;

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_NO_THROW(runner.run(params));
    cudaError_t const syncErr = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, syncErr) << "cudaStreamSynchronize after run: " << cudaGetErrorString(syncErr);
    cudaError_t const lastErr = cudaGetLastError();
    EXPECT_EQ(cudaSuccess, lastErr) << "cudaGetLastError after run: " << cudaGetErrorString(lastErr);

    size_t const numOutputElems = oBytes / sizeof(uint16_t);
    std::vector<uint16_t> hO(numOutputElems);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(hO.data(), dO, oBytes, cudaMemcpyDeviceToHost));

    int numSentinelRemnants = 0;
    int numNaN = 0;
    int numInf = 0;
    int numOutOfTolerance = 0;
    float minVal = std::numeric_limits<float>::infinity();
    float maxVal = -std::numeric_limits<float>::infinity();
    constexpr float kExpectedValue = 1.5f;
    constexpr float kAbsTolerance = 0.03f;
    for (uint16_t bits : hO)
    {
        if (bits == kBf16NaNSentinel)
        {
            ++numSentinelRemnants;
        }
        float const f = bf16BitsToFloat(bits);
        if (std::isnan(f))
        {
            ++numNaN;
        }
        if (std::isinf(f))
        {
            ++numInf;
        }
        if (std::isfinite(f))
        {
            minVal = std::min(minVal, f);
            maxVal = std::max(maxVal, f);
            if (std::abs(f - kExpectedValue) > kAbsTolerance)
            {
                ++numOutOfTolerance;
            }
        }
    }
    EXPECT_EQ(0, numSentinelRemnants) << "0xFFFF NaN sentinels remain: " << numSentinelRemnants << "/" << numOutputElems;
    EXPECT_EQ(0, numNaN) << "Found " << numNaN << "/" << numOutputElems << " NaN elements";
    EXPECT_EQ(0, numInf) << "Found " << numInf << "/" << numOutputElems << " Inf elements";
    EXPECT_EQ(0, numOutOfTolerance) << "Output elements outside |x - 1.5| <= " << kAbsTolerance << ": "
                                    << numOutOfTolerance << "/" << numOutputElems << "; min=" << minVal
                                    << " max=" << maxVal;

    cudaFree(dQ);
    cudaFree(dKV);
    cudaFree(dO);
    cudaFree(dScratch);
    cudaFree(dCounter);
    cudaFree(dPageIdx);
    cudaFree(dSeqLensKv);
    cudaFree(dCumSeqLensQ);
    cudaFree(dCumSeqLensKv);
    cudaFree(dScaleSoftmaxLog2);
    cudaFree(dOutputScale);
    cudaStreamDestroy(stream);
}

// Q=zero + V[kv, d_v] = (d_v % 8) + 1 (identical across kv, exactly representable in
// BF16): uniform attention gives output[q, h, d_v] = (d_v % 8) + 1. Complements the
// split-half test by varying V along the d_v axis, so d_v mis-indexing (transposed or
// shifted strides) trips the per-element compare.
TEST_F(KimiMlaGroupedSelectionTest, KimiShape_RunSmokePatternedVOutputsPerDim)
{
    TllmGenFmhaRunner runner(DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16);
    TllmGenFmhaRunnerParams params;
    buildKimiParams(params);

    int const seqLenQ = params.mMaxSeqLenQ;
    int const seqLenKv = params.mMaxSeqLenKv;
    int const batchSize = params.mBatchSize;
    int const numHeadsQ = params.mNumHeadsQ;
    int const headDimQk = params.mHeadDimQk;
    int const headDimV = params.mHeadDimV;
    int const numTokensPerPage = params.mNumTokensPerPage;
    int const maxNumPagesPerSeqKv = params.mMaxNumPagesPerSeqKv;
    int const numPages = maxNumPagesPerSeqKv * batchSize;

    constexpr size_t kBf16 = sizeof(uint16_t);
    size_t const qBytes = static_cast<size_t>(batchSize) * seqLenQ * numHeadsQ * headDimQk * kBf16;
    size_t const kvBytes = static_cast<size_t>(numPages) * numTokensPerPage * headDimQk * kBf16;
    size_t const oBytes = static_cast<size_t>(batchSize) * seqLenQ * numHeadsQ * headDimV * kBf16;
    size_t const scratchBytes = static_cast<size_t>(64) * 1024 * 1024;

    void* dQ = nullptr;
    void* dKV = nullptr;
    void* dO = nullptr;
    void* dScratch = nullptr;
    int32_t* dCounter = nullptr;
    int32_t* dPageIdx = nullptr;
    int32_t* dSeqLensKv = nullptr;
    int32_t* dCumSeqLensQ = nullptr;
    int32_t* dCumSeqLensKv = nullptr;
    float* dScaleSoftmaxLog2 = nullptr;
    float* dOutputScale = nullptr;
    cudaStream_t stream = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dQ, qBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dKV, kvBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dO, oBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dScratch, scratchBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCounter, kCounterBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dPageIdx, batchSize * maxNumPagesPerSeqKv * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dSeqLensKv, batchSize * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCumSeqLensQ, (batchSize + 1) * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCumSeqLensKv, (batchSize + 1) * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dScaleSoftmaxLog2, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dOutputScale, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    ASSERT_EQ(cudaSuccess, cudaMemset(dQ, 0, qBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dO, 0xFF, oBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dScratch, 0, scratchBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dCounter, 0, kCounterBytes));

    // Build the patterned KV cache. Same value V_d = (d % 8) + 1 for every
    // kv across the full head_dim_qk = 576. BF16 encoding for integer N in
    // [1, 8] is float-to-BF16 of the integer (top 16 bits of the FP32 word).
    auto floatToBf16 = [](float v) -> uint16_t {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(float));
        return static_cast<uint16_t>(bits >> 16);
    };
    std::vector<uint16_t> hKV(kvBytes / sizeof(uint16_t));
    size_t const totalKvSlots = static_cast<size_t>(numPages) * numTokensPerPage;
    for (size_t kv = 0; kv < totalKvSlots; ++kv)
    {
        for (int d = 0; d < headDimQk; ++d)
        {
            float const value = static_cast<float>((d % 8) + 1); // 1..8
            hKV[kv * headDimQk + d] = floatToBf16(value);
        }
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dKV, hKV.data(), kvBytes, cudaMemcpyHostToDevice, stream));

    std::vector<int32_t> hPageIdx(batchSize * maxNumPagesPerSeqKv);
    for (int i = 0; i < static_cast<int>(hPageIdx.size()); ++i)
    {
        hPageIdx[i] = i;
    }
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dPageIdx, hPageIdx.data(), hPageIdx.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    std::vector<int32_t> const hSeqLensKv = {seqLenKv};
    std::vector<int32_t> const hCumSeqLensQ = {0, seqLenQ};
    std::vector<int32_t> const hCumSeqLensKv = {0, seqLenKv};
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dSeqLensKv, hSeqLensKv.data(), batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dCumSeqLensQ, hCumSeqLensQ.data(), (batchSize + 1) * sizeof(int32_t), cudaMemcpyHostToDevice,
            stream));
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dCumSeqLensKv, hCumSeqLensKv.data(), (batchSize + 1) * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
    float const hScaleSoftmaxLog2 = (1.f / std::sqrt(static_cast<float>(headDimQk))) * static_cast<float>(M_LOG2E);
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dScaleSoftmaxLog2, &hScaleSoftmaxLog2, sizeof(float), cudaMemcpyHostToDevice, stream));
    float const hOutputScale = 1.0f;
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dOutputScale, &hOutputScale, sizeof(float), cudaMemcpyHostToDevice, stream));

    params.qPtr = dQ;
    params.kvPtr = dKV;
    params.oPtr = dO;
    params.kvPageIdxPtr = dPageIdx;
    params.seqLensKvPtr = dSeqLensKv;
    params.cumSeqLensQPtr = dCumSeqLensQ;
    params.cumSeqLensKvPtr = dCumSeqLensKv;
    params.scaleSoftmaxLog2Ptr = dScaleSoftmaxLog2;
    params.outputScalePtr = dOutputScale;
    params.multiCtasKvScratchPtr = dScratch;
    params.multiCtasKvCounterPtr = dCounter;
    params.stream = stream;

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_NO_THROW(runner.run(params));
    cudaError_t const syncErr = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, syncErr) << "cudaStreamSynchronize after run: " << cudaGetErrorString(syncErr);
    cudaError_t const lastErr = cudaGetLastError();
    EXPECT_EQ(cudaSuccess, lastErr) << "cudaGetLastError after run: " << cudaGetErrorString(lastErr);

    size_t const numOutputElems = oBytes / sizeof(uint16_t);
    std::vector<uint16_t> hO(numOutputElems);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(hO.data(), dO, oBytes, cudaMemcpyDeviceToHost));

    // Output layout: [batch, q_len, num_heads, head_dim_v] = [1, 4, 16, 512]
    // contiguous; element (q, h, d_v) lives at index q*numHeadsQ*headDimV +
    // h*headDimV + d_v. For each (q, h), the d_v dimension cycles 1..8 every
    // 8 elements.
    int numSentinelRemnants = 0;
    int numNaN = 0;
    int numInf = 0;
    int numOutOfTolerance = 0;
    float minVal = std::numeric_limits<float>::infinity();
    float maxVal = -std::numeric_limits<float>::infinity();
    constexpr float kAbsTolerance = 0.05f;
    for (int q = 0; q < seqLenQ; ++q)
    {
        for (int h = 0; h < numHeadsQ; ++h)
        {
            for (int d_v = 0; d_v < headDimV; ++d_v)
            {
                size_t const idx
                    = static_cast<size_t>(q) * numHeadsQ * headDimV + static_cast<size_t>(h) * headDimV + d_v;
                uint16_t const bits = hO[idx];
                if (bits == kBf16NaNSentinel)
                {
                    ++numSentinelRemnants;
                }
                float const f = bf16BitsToFloat(bits);
                if (std::isnan(f))
                {
                    ++numNaN;
                }
                if (std::isinf(f))
                {
                    ++numInf;
                }
                if (std::isfinite(f))
                {
                    minVal = std::min(minVal, f);
                    maxVal = std::max(maxVal, f);
                    float const expected = static_cast<float>((d_v % 8) + 1);
                    if (std::abs(f - expected) > kAbsTolerance)
                    {
                        ++numOutOfTolerance;
                    }
                }
            }
        }
    }
    EXPECT_EQ(0, numSentinelRemnants) << "0xFFFF NaN sentinels remain: " << numSentinelRemnants << "/" << numOutputElems;
    EXPECT_EQ(0, numNaN) << "Found " << numNaN << "/" << numOutputElems << " NaN elements";
    EXPECT_EQ(0, numInf) << "Found " << numInf << "/" << numOutputElems << " Inf elements";
    EXPECT_EQ(0, numOutOfTolerance) << "Output elements outside |x - expected(d_v)| <= " << kAbsTolerance << ": "
                                    << numOutOfTolerance << "/" << numOutputElems << "; min=" << minVal
                                    << " max=" << maxVal;

    cudaFree(dQ);
    cudaFree(dKV);
    cudaFree(dO);
    cudaFree(dScratch);
    cudaFree(dCounter);
    cudaFree(dPageIdx);
    cudaFree(dSeqLensKv);
    cudaFree(dCumSeqLensQ);
    cudaFree(dCumSeqLensKv);
    cudaFree(dScaleSoftmaxLog2);
    cudaFree(dOutputScale);
    cudaStreamDestroy(stream);
}

// Random-input vs CPU FP32 reference (mlaReferenceCpu) diff — covers the
// non-uniform-softmax, causal-spec-decode-mask, and arbitrary-V cases the hand-derived
// tests cannot. Inputs are uniform [-1, 1] so post-scale logits have stddev ~0.58 and
// softmax weights are non-uniform enough (~33x max/min) for the wrong-scale negative
// control below to fire. Tolerance is 0.03 absolute (BF16 precision + 576-dim dot
// product + 1024-position softmax rounding).
//
// Negative control: re-run the CPU reference with the softmax scale doubled and expect
// >= 1% of elements to diverge from the cubin output. The cubin bakes the softmax scale
// in for this shape (perturbing mScaleQ or scaleSoftmaxLog2Ptr does not change its
// output), so the reference is the only scale lever; the control proves the inputs are
// non-uniform enough to catch a baked-in-scale drift via the positive case.
TEST_F(KimiMlaGroupedSelectionTest, KimiShape_RunMatchesCpuMlaReference)
{
    TllmGenFmhaRunner runner(DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16);
    TllmGenFmhaRunnerParams params;
    buildKimiParams(params);

    int const seqLenQ = params.mMaxSeqLenQ;
    int const seqLenKv = params.mMaxSeqLenKv;
    int const batchSize = params.mBatchSize;
    int const numHeadsQ = params.mNumHeadsQ;
    int const headDimQk = params.mHeadDimQk;
    int const headDimV = params.mHeadDimV;
    int const numTokensPerPage = params.mNumTokensPerPage;
    int const maxNumPagesPerSeqKv = params.mMaxNumPagesPerSeqKv;
    int const numPages = maxNumPagesPerSeqKv * batchSize;
    ASSERT_EQ(1, batchSize) << "reference assumes batch=1";

    // MLA softmax scale: 1/sqrt(QK_NOPE + QK_ROPE) = 1/sqrt(128 + 64) = 1/sqrt(192).
    constexpr int kQkNopeHeadDim = 128;
    constexpr int kQkRopeHeadDim = 64;
    float const softmaxScale = 1.f / std::sqrt(static_cast<float>(kQkNopeHeadDim + kQkRopeHeadDim));

    constexpr size_t kBf16 = sizeof(uint16_t);
    size_t const qBytes = static_cast<size_t>(batchSize) * seqLenQ * numHeadsQ * headDimQk * kBf16;
    size_t const kvBytes = static_cast<size_t>(numPages) * numTokensPerPage * headDimQk * kBf16;
    size_t const oBytes = static_cast<size_t>(batchSize) * seqLenQ * numHeadsQ * headDimV * kBf16;
    size_t const scratchBytes = static_cast<size_t>(64) * 1024 * 1024;

    // Build random Q and KV on host. Deterministic seed so a failure is
    // reproducible. Uniform [-1.0, 1.0] makes scores have stddev ~8 and
    // post-scale logits stddev ~0.58, so softmax weights are non-uniform
    // (max/min ratio ~33x) and the wrong-scale negative control below
    // actually fires.
    std::mt19937 rng(/*seed=*/42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> hQ(qBytes / sizeof(uint16_t));
    for (auto& q : hQ)
    {
        q = floatToBf16(dist(rng));
    }
    std::vector<uint16_t> hKV(kvBytes / sizeof(uint16_t));
    for (auto& kv : hKV)
    {
        kv = floatToBf16(dist(rng));
    }

    // CPU reference: pre-compute expected output. Use the same SOFTMAX_SCALE
    // we plumb into the cubin so internal-consistency holds.
    size_t const numOutputElems = oBytes / sizeof(uint16_t);
    std::vector<uint16_t> hOutRef(numOutputElems);
    mlaReferenceCpu(seqLenQ, seqLenKv, numHeadsQ, headDimQk, headDimV, numTokensPerPage, hQ, hKV, softmaxScale, hOutRef);

    void* dQ = nullptr;
    void* dKV = nullptr;
    void* dO = nullptr;
    void* dScratch = nullptr;
    int32_t* dCounter = nullptr;
    int32_t* dPageIdx = nullptr;
    int32_t* dSeqLensKv = nullptr;
    int32_t* dCumSeqLensQ = nullptr;
    int32_t* dCumSeqLensKv = nullptr;
    float* dScaleSoftmaxLog2 = nullptr;
    float* dOutputScale = nullptr;
    cudaStream_t stream = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dQ, qBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dKV, kvBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dO, oBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dScratch, scratchBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCounter, kCounterBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dPageIdx, batchSize * maxNumPagesPerSeqKv * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dSeqLensKv, batchSize * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCumSeqLensQ, (batchSize + 1) * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dCumSeqLensKv, (batchSize + 1) * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dScaleSoftmaxLog2, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dOutputScale, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    ASSERT_EQ(cudaSuccess, cudaMemset(dO, 0xFF, oBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dScratch, 0, scratchBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dCounter, 0, kCounterBytes));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dQ, hQ.data(), qBytes, cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dKV, hKV.data(), kvBytes, cudaMemcpyHostToDevice, stream));

    std::vector<int32_t> hPageIdx(batchSize * maxNumPagesPerSeqKv);
    for (int i = 0; i < static_cast<int>(hPageIdx.size()); ++i)
    {
        hPageIdx[i] = i;
    }
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dPageIdx, hPageIdx.data(), hPageIdx.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    std::vector<int32_t> const hSeqLensKv = {seqLenKv};
    std::vector<int32_t> const hCumSeqLensQ = {0, seqLenQ};
    std::vector<int32_t> const hCumSeqLensKv = {0, seqLenKv};
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dSeqLensKv, hSeqLensKv.data(), batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dCumSeqLensQ, hCumSeqLensQ.data(), (batchSize + 1) * sizeof(int32_t), cudaMemcpyHostToDevice,
            stream));
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dCumSeqLensKv, hCumSeqLensKv.data(), (batchSize + 1) * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));

    // Plumb the same softmax scale to both the device pointer and via mScaleQ:
    // setFmhaData computes softmaxScale = (1 / (sqrt(headDimQk) * mScaleQ)) * log2(e),
    // so mScaleQ = sqrt(192/576) resolves it to (1/sqrt(192)) * log2(e).
    params.mScaleQ = std::sqrt(static_cast<float>(kQkNopeHeadDim + kQkRopeHeadDim) / static_cast<float>(headDimQk));
    float const hScaleSoftmaxLog2 = softmaxScale * static_cast<float>(M_LOG2E);
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dScaleSoftmaxLog2, &hScaleSoftmaxLog2, sizeof(float), cudaMemcpyHostToDevice, stream));
    float const hOutputScale = 1.0f;
    ASSERT_EQ(cudaSuccess,
        cudaMemcpyAsync(dOutputScale, &hOutputScale, sizeof(float), cudaMemcpyHostToDevice, stream));

    params.qPtr = dQ;
    params.kvPtr = dKV;
    params.oPtr = dO;
    params.kvPageIdxPtr = dPageIdx;
    params.seqLensKvPtr = dSeqLensKv;
    params.cumSeqLensQPtr = dCumSeqLensQ;
    params.cumSeqLensKvPtr = dCumSeqLensKv;
    params.scaleSoftmaxLog2Ptr = dScaleSoftmaxLog2;
    params.outputScalePtr = dOutputScale;
    params.multiCtasKvScratchPtr = dScratch;
    params.multiCtasKvCounterPtr = dCounter;
    params.stream = stream;

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_NO_THROW(runner.run(params));
    cudaError_t const syncErr = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, syncErr) << "cudaStreamSynchronize after run: " << cudaGetErrorString(syncErr);
    cudaError_t const lastErr = cudaGetLastError();
    EXPECT_EQ(cudaSuccess, lastErr) << "cudaGetLastError after run: " << cudaGetErrorString(lastErr);

    std::vector<uint16_t> hOutCubin(numOutputElems);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(hOutCubin.data(), dO, oBytes, cudaMemcpyDeviceToHost));

    // Element-by-element diff. Track count + max abs diff for diagnosis on failure.
    int numSentinelRemnants = 0;
    int numNaN = 0;
    int numInf = 0;
    int numOutOfTolerance = 0;
    float maxAbsDiff = 0.f;
    float maxRefMag = 0.f;
    constexpr float kAbsTolerance = 0.03f;
    for (size_t i = 0; i < numOutputElems; ++i)
    {
        uint16_t const cubinBits = hOutCubin[i];
        if (cubinBits == kBf16NaNSentinel)
        {
            ++numSentinelRemnants;
        }
        float const fCubin = bf16BitsToFloat(cubinBits);
        float const fRef = bf16BitsToFloat(hOutRef[i]);
        if (std::isnan(fCubin))
        {
            ++numNaN;
        }
        if (std::isinf(fCubin))
        {
            ++numInf;
        }
        if (std::isfinite(fCubin) && std::isfinite(fRef))
        {
            float const absDiff = std::abs(fCubin - fRef);
            maxAbsDiff = std::max(maxAbsDiff, absDiff);
            maxRefMag = std::max(maxRefMag, std::abs(fRef));
            if (absDiff > kAbsTolerance)
            {
                ++numOutOfTolerance;
            }
        }
    }
    EXPECT_EQ(0, numSentinelRemnants) << numSentinelRemnants << "/" << numOutputElems << " sentinels remain";
    EXPECT_EQ(0, numNaN) << numNaN << "/" << numOutputElems << " NaN elements in cubin output";
    EXPECT_EQ(0, numInf) << numInf << "/" << numOutputElems << " Inf elements in cubin output";
    EXPECT_EQ(0, numOutOfTolerance)
        << "cubin vs CPU reference disagrees by > " << kAbsTolerance << " on " << numOutOfTolerance << "/"
        << numOutputElems << " elements; maxAbsDiff=" << maxAbsDiff << " maxRefMag=" << maxRefMag;

    // Negative control (see test header comment): a 2x-wrong-scale reference must
    // diverge from the cubin output, proving the positive case is scale-sensitive.
    std::vector<uint16_t> hOutRefWrong(numOutputElems);
    mlaReferenceCpu(seqLenQ, seqLenKv, numHeadsQ, headDimQk, headDimV, numTokensPerPage, hQ, hKV, softmaxScale * 2.f,
        hOutRefWrong);
    int numWrongScaleDivergent = 0;
    float maxWrongScaleAbsDiff = 0.f;
    for (size_t i = 0; i < numOutputElems; ++i)
    {
        float const fCubin = bf16BitsToFloat(hOutCubin[i]);
        float const fRefWrong = bf16BitsToFloat(hOutRefWrong[i]);
        if (std::isfinite(fCubin) && std::isfinite(fRefWrong))
        {
            float const absDiff = std::abs(fCubin - fRefWrong);
            maxWrongScaleAbsDiff = std::max(maxWrongScaleAbsDiff, absDiff);
            if (absDiff > kAbsTolerance)
            {
                ++numWrongScaleDivergent;
            }
        }
    }
    // Expect the 2x-wrong reference to disagree with the cubin on many
    // elements. Threshold is conservative (>= 1% of elements) so this is
    // robust to the exact random seed.
    int const kMinDivergent = static_cast<int>(numOutputElems / 100);
    EXPECT_GE(numWrongScaleDivergent, kMinDivergent)
        << "Negative control: 2x-wrong-scale CPU reference should diverge from cubin on >= " << kMinDivergent
        << " elements (>" << kAbsTolerance << " abs); only " << numWrongScaleDivergent
        << " did. Random inputs may be too uniform for scale sensitivity. maxWrongScaleAbsDiff="
        << maxWrongScaleAbsDiff;

    cudaFree(dQ);
    cudaFree(dKV);
    cudaFree(dO);
    cudaFree(dScratch);
    cudaFree(dCounter);
    cudaFree(dPageIdx);
    cudaFree(dSeqLensKv);
    cudaFree(dCumSeqLensQ);
    cudaFree(dCumSeqLensKv);
    cudaFree(dScaleSoftmaxLog2);
    cudaFree(dOutputScale);
    cudaStreamDestroy(stream);
}

} // namespace
