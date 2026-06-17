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
// Fix 3 — copyGenerationLogits direct-copy places each (beam, step) logit at
//   the correct host offset.
//   Before: the mergeLogitsFragmentsKernel pointer-indirection caused
//   intermittent GPU-state corruption when gather_generation_logits=True was
//   combined with concurrent mixed beam-width requests.
//   After:  each fragment is copied directly to the host without the kernel.

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

// Verify that for a cross-KV cache with beam width > 1, copyBlockOffsets
// places the same physical block IDs in every beam slot.
//
// This tests the isCrossKv() branch added by fix 1: when all beams share the
// same encoder output, the output offset table must reflect that.  In the
// production Whisper case the allocator gives each beam its own physical
// blocks, so beams 1..N-1 would reference uninitialised GPU memory without
// the fix.  This unit test uses a simple context-only sequence where the
// allocator happens to share blocks across beams; the observable property
// (all beam slots equal beam-0) still holds and constitutes a sanity check.
TEST(CrossKvBeamSharingTest, CopyBlockOffsetsAllBeamsShareBeam0Blocks)
{
    // Encoder-decoder setup: 1 layer, 1 KV head, sizePerHead=4,
    // tokensPerBlock=8, encoder output length=16 → 2 blocks per sequence,
    // beam width=3.
    auto stream = std::make_shared<tr::CudaStream>();

    SizeType32 constexpr numLayers = 1;
    SizeType32 constexpr numHeads = 1;
    SizeType32 constexpr sizePerHead = 4;
    SizeType32 constexpr tokensPerBlock = 8;
    SizeType32 constexpr maxNumSequences = 1;
    SizeType32 constexpr beamWidth = 3;
    SizeType32 constexpr maxAttentionWindow = 16;
    SizeType32 constexpr encoderLen = 16; // 2 blocks

    // Reserve enough blocks for 1 sequence × beamWidth × (encoderLen / tokensPerBlock).
    SizeType32 constexpr numBlocks = maxNumSequences * beamWidth * (encoderLen / tokensPerBlock);
    BlocksPerWindow const blocksPerWindow{{maxAttentionWindow, {numBlocks, 0}}};

    KVCacheManager crossKvMgr(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, {maxAttentionWindow}, nvinfer1::DataType::kHALF,
        /*sinkTokenLength=*/0, stream, maxAttentionWindow, maxAttentionWindow,
        /*enableBlockReuse=*/false, CacheType::kCROSS);
    crossKvMgr.allocatePools(false);

    // Build a minimal LlmRequest and allocate a cross-KV sequence.
    RequestIdType constexpr requestId = 1;
    auto inputTokens = std::make_shared<VecTokens>(encoderLen, 0);
    tr::SamplingConfig const samplingConfig{beamWidth};
    auto llmReq = std::make_shared<LlmRequest>(requestId, /*maxNewTokens=*/0, inputTokens, samplingConfig, false);
    crossKvMgr.addSequenceBatch({{{requestId, encoderLen, beamWidth}}}, {std::ref(*llmReq)});

    // Allocate CPU output tensor: [numPools, maxNumSeq*beamWidth, 2, maxBlocksPerSeq].
    auto const dims = crossKvMgr.getOffsetTableDimensions();
    SizeType32 const numPools = dims.numPools;
    SizeType32 const maxBlocksPerSeq = dims.maxBlocksPerSeq;
    auto blockOffsets
        = tr::BufferManager::cpu(tr::ITensor::makeShape({numPools, maxNumSequences * beamWidth, 2, maxBlocksPerSeq}),
            tr::TRTDataType<tk::KVCacheIndex>::value);

    // Fill with sentinel so we can detect un-written slots.
    auto* const raw = tr::bufferCast<tk::KVCacheIndex>(*blockOffsets);
    std::fill(raw, raw + blockOffsets->getSize(), tk::KVCacheIndex{tk::KVCacheIndex::kInvalidPoolIndex});

    crossKvMgr.copyBlockOffsets(*blockOffsets, /*outputSlotOffset=*/0, requestId);

    // Post-condition: for every (pool, K/V, block), beams 1..beamWidth-1
    // must hold the same physical block index as beam 0, and beam 0 itself
    // must be a valid (non-sentinel) index.
    auto const& shape = blockOffsets->getShape();
    for (SizeType32 pool = 0; pool < numPools; ++pool)
    {
        for (SizeType32 kv = 0; kv < 2; ++kv)
        {
            for (SizeType32 block = 0; block < maxBlocksPerSeq; ++block)
            {
                auto idx = [&](SizeType32 beam) { return tc::flat_index(shape.d, pool, beam, kv, block); };

                EXPECT_NE(raw[idx(0)].get(), tk::KVCacheIndex::kInvalidPoolIndex)
                    << "pool=" << pool << " beam=0 kv=" << kv << " block=" << block << ": not initialised";

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
// Fix 3: copyGenerationLogits direct-copy correctness
// ============================================================================

// Verify that the direct-copy implementation of copyGenerationLogits writes
// each step's logits for each beam to the correct slot in the host buffer.
TEST(CopyGenerationLogitsTest, DirectCopyPlacesEachBeamStepAtCorrectHostOffset)
{
    // Parameters.
    SizeType32 constexpr beamWidth = 2;
    SizeType32 constexpr numSteps = 4; // one full cache-length flush
    SizeType32 constexpr vocabSize = 8;
    SizeType32 constexpr promptLen = 1;

    auto stream = std::make_shared<tr::CudaStream>();
    tr::BufferManager bufferMgr{stream};

    // Create a request: promptLen=1, maxNewTokens=numSteps.
    RequestIdType constexpr requestId = 1;
    auto inputTokens = std::make_shared<VecTokens>(promptLen, 0);
    tr::SamplingConfig const samplingConfig{beamWidth};
    auto llmReq = std::make_shared<LlmRequest>(requestId, numSteps, inputTokens, samplingConfig, /*isStreaming=*/false);

    // Advance internal token count to simulate numSteps tokens generated so
    // that (with beforeDecoder=false):
    //   numGenerationToken = getNumTokens(beam) - mPromptLen = numSteps
    //   hostOffset         = numGenerationToken - fragmentSize = 0
    LlmRequest::BeamTokens const generatedTokens(beamWidth, VecTokens(numSteps, /*token=*/1));
    llmReq->setGeneratedTokens(generatedTokens);
    llmReq->allocGenerationLogitsHost(vocabSize, nvinfer1::DataType::kFLOAT);

    // Build numSteps logit fragments, each of shape [1, beamWidth, vocabSize],
    // filled with a unique per-(step, beam) sentinel value: step*100 + beam.
    for (SizeType32 step = 0; step < numSteps; ++step)
    {
        tr::ITensor::SharedPtr frag = tr::BufferManager::pinnedPool(
            tr::ITensor::makeShape({1, beamWidth, vocabSize}), nvinfer1::DataType::kFLOAT);
        auto* const fragData = tr::bufferCast<float>(*frag);
        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            float const val = static_cast<float>(step * 100 + beam);
            for (SizeType32 v = 0; v < vocabSize; ++v)
            {
                // Flat layout: [1][beam][v] → beam * vocabSize + v
                fragData[beam * vocabSize + v] = val;
            }
        }
        llmReq->addGenerationLogitsFragment(frag);
    }
    ASSERT_EQ(llmReq->getGenerationLogitsFragmentsSize(), numSteps);

    // Dummy cache — not accessed by the direct-copy implementation.
    RuntimeBuffers::GenerationLogitsCache dummyCache;

    utils::copyGenerationLogits(dummyCache, bufferMgr, *llmReq, /*beforeDecoder=*/false, /*numDroppedTokens=*/{});

    ASSERT_EQ(cudaStreamSynchronize(stream->get()), cudaSuccess);

    // Post-condition: generationLogitsHost[beam, step, v] == step*100 + beam
    // for all (beam, step, v).  Host shape: [beamWidth, maxNewTokens, vocab].
    auto const* const hostData = tr::bufferCast<float>(*llmReq->getGenerationLogitsHost());
    for (SizeType32 beam = 0; beam < beamWidth; ++beam)
    {
        for (SizeType32 step = 0; step < numSteps; ++step)
        {
            float const expected = static_cast<float>(step * 100 + beam);
            for (SizeType32 v = 0; v < vocabSize; ++v)
            {
                SizeType32 const flatIdx = (beam * numSteps + step) * vocabSize + v;
                EXPECT_FLOAT_EQ(hostData[flatIdx], expected) << "host[beam=" << beam << ", step=" << step << ", v=" << v
                                                             << "]=" << hostData[flatIdx] << " expected " << expected;
            }
        }
    }

    // copyGenerationLogits must clear fragments after flushing.
    EXPECT_EQ(llmReq->getGenerationLogitsFragmentsSize(), 0);
}
