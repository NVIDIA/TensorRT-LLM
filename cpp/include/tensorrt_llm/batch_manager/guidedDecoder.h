/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <dlpack/dlpack.h>
#include <optional>

namespace tensorrt_llm::batch_manager
{

class IGrammarMatcher
{
public:
    virtual ~IGrammarMatcher() = default;
    virtual bool AcceptToken(int32_t tokenId) = 0;
    virtual void FillNextTokenBitmask(DLTensor* nextTokenBitmask) = 0;
};

class IGrammarMatcherFactory
{
public:
    virtual ~IGrammarMatcherFactory() = default;
    virtual std::shared_ptr<IGrammarMatcher> Create(
        tensorrt_llm::executor::GuidedDecodingParams const& guidedDecodingParams)
        = 0;
};

class GuidedDecoder
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using BitmaskT = uint32_t;

    GuidedDecoder(executor::GuidedDecodingConfig const& guidedDecodingConfig, SizeType32 maxNumSequences,
        SizeType32 vocabSizePadded, nvinfer1::DataType logitsDtype, runtime::BufferManager const& runtimeBufferManager);
    void build(ScheduledRequests const& scheduledRequests);
    void execute(ScheduledRequests const& scheduledRequests, runtime::BufferManager const& runtimeBufferManager,
        std::vector<TensorPtr> const& decoderBuffersLogits);

private:
    executor::GuidedDecodingConfig::GuidedDecodingBackend mGuidedDecodingBackend;
    std::shared_ptr<IGrammarMatcherFactory> mGrammarMatcherFactory;
    std::vector<std::shared_ptr<IGrammarMatcher>> mGrammarMatchers;

    SizeType32 mMaxNumSequences;
    SizeType32 mVocabSizePadded;
    SizeType32 mBitmaskSize; // CeilDiv(vocabSizePadded, 32)
    nvinfer1::DataType mLogitsDtype;

    TensorPtr mLogitsBitmask;           // [mMaxNumRequests, mBitmaskSize]
    TensorPtr mLogitsBitmaskHost;       // [mMaxNumRequests, mBitmaskSize]
    TensorPtr mLogitsBitmaskPtrVec;     // [mMaxNumRequests], pointers to the logitsBitmask in a batch
    TensorPtr mLogitsBitmaskPtrVecHost; // [mMaxNumRequests]
    TensorPtr mLogitsPtrVec;            // [mMaxNumRequests], pointers to the logits in a batch
    TensorPtr mLogitsPtrVecHost;        // [mMaxNumRequests]

    // BufferManager with a dedicated stream for async copy of buffers for guided decoding.
    runtime::BufferManager mCopyBufferManager;
};

} // namespace tensorrt_llm::batch_manager
