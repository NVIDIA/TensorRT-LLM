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

namespace xgrammar
{
class GrammarMatcher;
class GrammarCompiler;
} // namespace xgrammar

namespace tensorrt_llm::batch_manager
{
class DecoderInputBuffers;

class GuidedDecoder
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using BitmaskT = uint32_t;

    GuidedDecoder(executor::GuidedDecodingConfig const& guidedDecodingConfig, SizeType32 maxNumSequences,
        SizeType32 vocabSizePadded, nvinfer1::DataType logitsDtype, runtime::BufferManager const& runtimeBufferManager);
    void build(ScheduledRequests const& scheduledRequests);
    void execute(DecoderInputBuffers const& decoderInputBuffers, runtime::BufferManager const& runtimeBufferManager);

private:
    executor::GuidedDecodingConfig::GuidedDecodingBackend mGuidedDecodingBackend;
    std::vector<std::shared_ptr<xgrammar::GrammarMatcher>> mXGrammarMatchers;
    std::shared_ptr<xgrammar::GrammarCompiler> mXGrammarCompiler;

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
