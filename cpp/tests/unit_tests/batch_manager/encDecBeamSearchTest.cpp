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

// Regression tests for encoder-decoder beam-search logits gathering.
//
// fragmentPointerDevice is per-batch-slot ([maxBatchSize, kCACHE_LENGTH]).
// Before the fix, fragmentPointerDevice was a single shared row ([kCACHE_LENGTH]); sequential
// flushes from different requests in the same batch clobbered each other's GPU pointer
// arrays, causing the mergeLogitsFragmentsKernel to read stale fragment addresses and
// produce degenerate output with gather_generation_logits=True.
// After the fix, each request gets its own device-side pointer row via getFragmentPointerSlot(),
// eliminating cross-request interference.

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
