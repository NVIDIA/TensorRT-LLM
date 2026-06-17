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

// Regression tests for encoder-decoder beam-search fixes.
//
// Fix 1 — cross-KV copyBlockOffsets shares beam-0 blocks across all beams.
//   Before: beams 1..N-1 received uninitialised physical blocks, causing
//   degenerate output ("happ happ happ") when the decoder cross-attended to
//   garbage encoder features.
//   After:  KVCacheManager::copyBlockOffsets uses beam-0's block IDs for every
//   beam when isCrossKv() is true, since the encoder output is identical for
//   all beams of a request.
//
// Fix 3 — fragmentPointerDevice is now per-batch-slot ([maxBatchSize, kCACHE_LENGTH]).
//   Before: fragmentPointerDevice was a single shared row ([kCACHE_LENGTH]); sequential
//   flushes from different requests in the same batch clobbered each other's GPU pointer
//   arrays, causing the mergeLogitsFragmentsKernel to read stale fragment addresses and
//   produce degenerate output with gather_generation_logits=True.
//   After:  each request gets its own device-side pointer row via getFragmentPointerSlot(),
//   eliminating cross-request interference.

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "gtest/gtest.h"
#include <memory>

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
namespace tr = tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
using SizeType32 = tr::SizeType32;

// ============================================================================
// Fix 1: KVCacheManager::copyBlockOffsets cross-KV beam sharing
// ============================================================================

// Verify that copyBlockOffsets normalises all beam slots to beam-0's value on the
// cross-KV path even when per-beam source rows differ.  The test writes distinct
// sentinel values into each beam row of the source tensor before calling
// copyBlockOffsets so that the old bug (srcBeamIdx = beamIdx) would leave beams
// 1..N-1 with different values, while the fix (srcBeamIdx = 0 for isCrossKv())
// produces equal values across all beams.
TEST(CrossKvBeamSharingTest, CopyBlockOffsetsNormalisesAllBeamsToBeam0)
{
    auto stream = std::make_shared<tr::CudaStream>();

    SizeType32 constexpr numLayers = 1;
    SizeType32 constexpr numHeads = 1;
    SizeType32 constexpr sizePerHead = 4;
    SizeType32 constexpr tokensPerBlock = 8;
    SizeType32 constexpr maxNumSequences = 1;
    SizeType32 constexpr beamWidth = 3;
    SizeType32 constexpr maxAttentionWindow = 16;
    SizeType32 constexpr encoderLen = 16; // 2 blocks

    SizeType32 constexpr numBlocks = maxNumSequences * beamWidth * (encoderLen / tokensPerBlock);
    BlocksPerWindow const blocksPerWindow{{maxAttentionWindow, {numBlocks, 0}}};

    KVCacheManager crossKvMgr(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, {maxAttentionWindow}, nvinfer1::DataType::kHALF,
        /*sinkTokenLength=*/0, stream, maxAttentionWindow, maxAttentionWindow,
        /*enableBlockReuse=*/false, CacheType::kCROSS);
    crossKvMgr.allocatePools(false);

    RequestIdType constexpr requestId = 1;
    auto inputTokens = std::make_shared<VecTokens>(encoderLen, 0);
    tr::SamplingConfig const samplingConfig{beamWidth};
    auto llmReq = std::make_shared<LlmRequest>(requestId, /*maxNewTokens=*/0, inputTokens, samplingConfig, false);
    crossKvMgr.addSequenceBatch({{{requestId, encoderLen, beamWidth}}}, {std::ref(*llmReq)});

    // Write distinct per-beam values into the source cacheBlockIndices tensor so
    // that the old code (copying each beam's own row) would produce different
    // outputs, while the fixed code (always copying beam-0's row) produces equal.
    auto& seq = crossKvMgr.getSequence(requestId);
    auto& srcTensor = seq.getCacheBlockIndices(maxAttentionWindow);
    auto const& srcShape = srcTensor.getShape();
    auto* const srcPtr = tr::bufferCast<tk::KVCacheIndex>(srcTensor);
    for (SizeType32 beam = 0; beam < beamWidth; ++beam)
    {
        for (SizeType32 kv = 0; kv < 2; ++kv)
        {
            for (SizeType32 block = 0; block < srcShape.d[3]; ++block)
            {
                auto const idx = tc::flat_index(srcShape.d, /*pool=*/0, beam, kv, block);
                // Beam b gets value (b*100 + kv*10 + block), all non-zero and distinct.
                srcPtr[idx]
                    = tk::KVCacheIndex{static_cast<tk::KVCacheIndex::UnderlyingType>(beam * 100 + kv * 10 + block + 1)};
            }
        }
    }

    auto const dims = crossKvMgr.getOffsetTableDimensions();
    SizeType32 const numPools = dims.numPools;
    SizeType32 const maxBlocksPerSeq = dims.maxBlocksPerSeq;
    auto blockOffsets
        = tr::BufferManager::cpu(tr::ITensor::makeShape({numPools, maxNumSequences * beamWidth, 2, maxBlocksPerSeq}),
            tr::TRTDataType<tk::KVCacheIndex>::value);

    auto* const raw = tr::bufferCast<tk::KVCacheIndex>(*blockOffsets);
    std::fill(raw, raw + blockOffsets->getSize(), tk::KVCacheIndex{tk::KVCacheIndex::kInvalidPoolIndex});

    crossKvMgr.copyBlockOffsets(*blockOffsets, /*outputSlotOffset=*/0, requestId);

    auto const& shape = blockOffsets->getShape();
    for (SizeType32 pool = 0; pool < numPools; ++pool)
    {
        for (SizeType32 kv = 0; kv < 2; ++kv)
        {
            for (SizeType32 block = 0; block < maxBlocksPerSeq; ++block)
            {
                auto idx = [&](SizeType32 beam) { return tc::flat_index(shape.d, pool, beam, kv, block); };

                // Beam 0 must have been written (non-sentinel) and reflect its source value.
                EXPECT_NE(raw[idx(0)].get(), tk::KVCacheIndex::kInvalidPoolIndex)
                    << "pool=" << pool << " beam=0 kv=" << kv << " block=" << block << ": not written";

                // All other beams must equal beam 0 — the fix normalises them.
                for (SizeType32 beam = 1; beam < beamWidth; ++beam)
                {
                    EXPECT_EQ(raw[idx(beam)].get(), raw[idx(0)].get())
                        << "pool=" << pool << " beam=" << beam << " kv=" << kv << " block=" << block
                        << ": differs from beam 0 — cross-KV beam sharing is broken";
                }
            }
        }
    }
}

// ============================================================================
// Fix 3: copyGenerationLogits per-slot fragmentPointerDevice
// ============================================================================

// Verify that copyGenerationLogits correctly assembles the host logits buffer
// using the real kernel merge path, and that two back-to-back calls (simulating
// two requests flushing in the same batch) use distinct fragmentPointerDevice
// slots so their pointer arrays do not clobber each other.
TEST(CopyGenerationLogitsTest, KernelMergePathProducesCorrectHostLayoutAndSlotsAreIsolated)
{
    SizeType32 constexpr beamWidth = 2;
    SizeType32 constexpr numSteps = RuntimeBuffers::GenerationLogitsCache::kCACHE_LENGTH; // full flush
    SizeType32 constexpr vocabSize = 8;
    SizeType32 constexpr promptLen = 1;
    SizeType32 constexpr maxBatchSize = 4; // must be >= 2 to test slot isolation

    auto stream = std::make_shared<tr::CudaStream>();
    tr::BufferManager bufferMgr{stream};

    // Build a real GenerationLogitsCache so that transposedLogits,
    // fragmentPointerDevice and fragmentPointerHost are all properly allocated.
    // cache.logits uses pinned memory so the test can fill it from the CPU while
    // the GPU kernel can still read from it via DMA.
    RuntimeBuffers::GenerationLogitsCache cache;
    cache.logits = tr::BufferManager::pinnedPool(
        tr::ITensor::makeShape({numSteps, maxBatchSize * beamWidth, vocabSize}), nvinfer1::DataType::kFLOAT);
    cache.transposedLogits
        = bufferMgr.gpu(tr::ITensor::makeShape({beamWidth, numSteps, vocabSize}), nvinfer1::DataType::kFLOAT);
    cache.fragmentPointerDevice
        = bufferMgr.gpu(tr::ITensor::makeShape({maxBatchSize, numSteps}), nvinfer1::DataType::kINT64);
    cache.fragmentPointerHost
        = tr::BufferManager::pinnedPool(tr::ITensor::makeShape({maxBatchSize, numSteps}), nvinfer1::DataType::kINT64);

    // Helper: build one LlmRequest that has numSteps fragments pointing into
    // cache.logits[0..numSteps-1][logitsIndex:logitsIndex+beamWidth].
    // Each fragment is filled with sentinel value (step*100 + beam + reqOffset).
    auto makeRequest = [&](RequestIdType reqId, SizeType32 logitsIndex, float reqOffset) -> std::shared_ptr<LlmRequest>
    {
        auto tokens = std::make_shared<VecTokens>(promptLen, 0);
        tr::SamplingConfig sc{beamWidth};
        auto req = std::make_shared<LlmRequest>(reqId, numSteps, tokens, sc, false);

        LlmRequest::BeamTokens gen(beamWidth, VecTokens(numSteps, 1));
        req->setGeneratedTokens(gen);
        req->allocGenerationLogitsHost(vocabSize, nvinfer1::DataType::kFLOAT);

        // Write known values into the logits cache slots for this request and
        // create matching fragment slice views.
        for (SizeType32 step = 0; step < numSteps; ++step)
        {
            // cache.logits shape: [numSteps, maxBatchSize*beamWidth, vocabSize]
            // Slice to [1, maxBS*bw, vocab], squeeze to [maxBS*bw, vocab].
            tr::ITensor::SharedPtr slot = tr::ITensor::slice(cache.logits, step, 1);
            slot->squeeze(0); // [maxBS*bw, vocab]
            auto* slotPtr = tr::bufferCast<float>(*slot);
            for (SizeType32 beam = 0; beam < beamWidth; ++beam)
            {
                float const val = reqOffset + static_cast<float>(step * 100 + beam);
                for (SizeType32 v = 0; v < vocabSize; ++v)
                {
                    slotPtr[(logitsIndex + beam) * vocabSize + v] = val;
                }
            }

            // Fragment matches HandleGenerationLogits: slice [logitsIndex:logitsIndex+beamWidth]
            // from the step slot, then unsqueeze(0) → [1, beamWidth, vocab].
            tr::ITensor::SharedPtr fragView = tr::ITensor::slice(slot, logitsIndex, beamWidth);
            fragView->unsqueeze(0); // [1, beamWidth, vocab]
            req->addGenerationLogitsFragment(fragView);
        }
        return req;
    };

    // Request 0 occupies logitsIndex=0 in the batch slot.
    auto req0 = makeRequest(1, /*logitsIndex=*/0, /*reqOffset=*/0.0f);
    // Request 1 occupies logitsIndex=beamWidth in the batch slot.
    auto req1 = makeRequest(2, /*logitsIndex=*/beamWidth, /*reqOffset=*/1000.0f);

    // Flush request 0 — uses workIdx=0.
    utils::copyGenerationLogits(cache, bufferMgr, *req0, /*beforeDecoder=*/false, {});
    // Flush request 1 — uses workIdx=1 (different slot → no pointer clobbering).
    utils::copyGenerationLogits(cache, bufferMgr, *req1, /*beforeDecoder=*/false, {});

    ASSERT_EQ(cudaStreamSynchronize(stream->get()), cudaSuccess);

    // Verify req0 host buffer: host[beam, step, v] == step*100 + beam
    auto const* host0 = tr::bufferCast<float>(*req0->getGenerationLogitsHost());
    for (SizeType32 beam = 0; beam < beamWidth; ++beam)
    {
        for (SizeType32 step = 0; step < numSteps; ++step)
        {
            float const expected = static_cast<float>(step * 100 + beam);
            for (SizeType32 v = 0; v < vocabSize; ++v)
            {
                SizeType32 const idx = (beam * numSteps + step) * vocabSize + v;
                EXPECT_FLOAT_EQ(host0[idx], expected) << "req0 host[beam=" << beam << ",step=" << step << ",v=" << v
                                                      << "]=" << host0[idx] << " expected " << expected;
            }
        }
    }

    // Verify req1 host buffer: host[beam, step, v] == 1000 + step*100 + beam
    auto const* host1 = tr::bufferCast<float>(*req1->getGenerationLogitsHost());
    for (SizeType32 beam = 0; beam < beamWidth; ++beam)
    {
        for (SizeType32 step = 0; step < numSteps; ++step)
        {
            float const expected = 1000.0f + static_cast<float>(step * 100 + beam);
            for (SizeType32 v = 0; v < vocabSize; ++v)
            {
                SizeType32 const idx = (beam * numSteps + step) * vocabSize + v;
                EXPECT_FLOAT_EQ(host1[idx], expected) << "req1 host[beam=" << beam << ",step=" << step << ",v=" << v
                                                      << "]=" << host1[idx] << " expected " << expected;
            }
        }
    }

    // Both requests must have had their fragments cleared.
    EXPECT_EQ(req0->getGenerationLogitsFragmentsSize(), 0);
    EXPECT_EQ(req1->getGenerationLogitsFragmentsSize(), 0);
}
