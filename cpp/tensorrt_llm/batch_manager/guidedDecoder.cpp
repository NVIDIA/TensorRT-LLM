/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/guidedDecoder.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/kernels/logitsBitmask.h"

#include <xgrammar/xgrammar.h>

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

GuidedDecoder::GuidedDecoder(executor::GuidedDecodingConfig const& guidedDecodingConfig, SizeType32 maxNumSequences,
    SizeType32 vocabSizePadded, nvinfer1::DataType logitsDtype, BufferManager const& runtimeBufferManager)
    : mGuidedDecodingBackend{guidedDecodingConfig.getBackend()}
    , mMaxNumSequences{maxNumSequences}
    , mVocabSizePadded{vocabSizePadded}
    , mBitmaskSize{common::ceilDiv(mVocabSizePadded, 32)}
    , mLogitsDtype{logitsDtype}
    , mCopyBufferManager{std::make_shared<CudaStream>()}
{
    TLLM_CHECK_WITH_INFO(mGuidedDecodingBackend != executor::GuidedDecodingConfig::GuidedDecodingBackend::kLLGUIDANCE,
        "LLGuidance is not supported for guided decoding in C++ runtime.");
    if (mGuidedDecodingBackend == executor::GuidedDecodingConfig::GuidedDecodingBackend::kXGRAMMAR)
    {
        mXGrammarMatchers.resize(mMaxNumSequences);
        auto const& tokenizerStr = guidedDecodingConfig.getTokenizerStr();
        if (tokenizerStr)
        {
            auto const& tokenizerInfo = xgrammar::TokenizerInfo::FromHuggingFace(
                guidedDecodingConfig.getEncodedVocab().value(), guidedDecodingConfig.getTokenizerStr().value(),
                mVocabSizePadded, guidedDecodingConfig.getStopTokenIds());
            mXGrammarCompiler = std::make_shared<xgrammar::GrammarCompiler>(tokenizerInfo);
        }
        else
        {
            auto const& tokenizerInfo = xgrammar::TokenizerInfo(guidedDecodingConfig.getEncodedVocab().value(),
                xgrammar::VocabType::RAW, mVocabSizePadded, guidedDecodingConfig.getStopTokenIds());
            mXGrammarCompiler = std::make_shared<xgrammar::GrammarCompiler>(tokenizerInfo);
        }

        auto const logitsPtrDtype = BufferDataType{mLogitsDtype, false, true};
        auto constexpr bitmaskDtype = TRTDataType<BitmaskT>::value;
        auto constexpr bitmaskPtrDtype = TRTDataType<BitmaskT*>::value;

        mLogitsBitmask = runtimeBufferManager.gpu(ITensor::makeShape({mMaxNumSequences, mBitmaskSize}), bitmaskDtype);
        mLogitsBitmaskHost = BufferManager::pinned(ITensor::makeShape({mMaxNumSequences, mBitmaskSize}), bitmaskDtype);
        mLogitsBitmaskPtrVec = runtimeBufferManager.gpu(ITensor::makeShape({mMaxNumSequences}), bitmaskPtrDtype);
        mLogitsBitmaskPtrVecHost = BufferManager::pinned(ITensor::makeShape({mMaxNumSequences}), bitmaskPtrDtype);
        mLogitsPtrVec = runtimeBufferManager.gpu(ITensor::makeShape({mMaxNumSequences}), logitsPtrDtype);
        mLogitsPtrVecHost = BufferManager::pinned(ITensor::makeShape({mMaxNumSequences}), logitsPtrDtype);
    }
}

void GuidedDecoder::build(ScheduledRequests const& scheduledRequests)
{
    if (mGuidedDecodingBackend == executor::GuidedDecodingConfig::GuidedDecodingBackend::kXGRAMMAR)
    {
        for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
        {
            for (auto const& llmReq : requests)
            {
                auto const& guidedDecodingParams = llmReq->getGuidedDecodingParams();
                if (!guidedDecodingParams.has_value())
                {
                    continue;
                }
                auto const seqSlot = llmReq->mSeqSlot.value();
                if (llmReq->isContextInitState()
                    && llmReq->getContextCurrentPosition() == llmReq->getPrepopulatedPromptLen())
                {
                    // The request is in the first context forward step (considering kv cache reuse).
                    auto const& guideType = guidedDecodingParams->getGuideType();
                    auto const& guide = guidedDecodingParams->getGuide();
                    if (guideType == executor::GuidedDecodingParams::GuideType::kJSON)
                    {
                        mXGrammarMatchers.at(seqSlot) = std::make_shared<xgrammar::GrammarMatcher>(
                            mXGrammarCompiler->CompileBuiltinJSONGrammar());
                    }
                    else if (guideType == executor::GuidedDecodingParams::GuideType::kJSON_SCHEMA)
                    {
                        mXGrammarMatchers.at(seqSlot) = std::make_shared<xgrammar::GrammarMatcher>(
                            mXGrammarCompiler->CompileJSONSchema(guide.value()));
                    }
                    else if (guideType == executor::GuidedDecodingParams::GuideType::kREGEX)
                    {
                        auto const& grammar = xgrammar::Grammar::FromRegex(guide.value());
                        mXGrammarMatchers.at(seqSlot)
                            = std::make_shared<xgrammar::GrammarMatcher>(mXGrammarCompiler->CompileGrammar(grammar));
                    }
                    else if (guideType == executor::GuidedDecodingParams::GuideType::kEBNF_GRAMMAR)
                    {
                        auto const& grammar = xgrammar::Grammar::FromEBNF(guide.value());
                        mXGrammarMatchers.at(seqSlot)
                            = std::make_shared<xgrammar::GrammarMatcher>(mXGrammarCompiler->CompileGrammar(grammar));
                    }
                }
                else if (llmReq->isGenerationInProgressState())
                {
                    // The request is in a generation forward step.
                    // Currently, guided decoding does not support with beam search.
                    mXGrammarMatchers.at(seqSlot)->AcceptToken(llmReq->getLastTokens(0));
                }
                else
                {
                    continue;
                }

                // Fill the bitmask on host and asynchorously copy to device using mCopyBufferManager.
                auto const logitsBitmask = ITensor::at(mLogitsBitmask, {seqSlot});
                auto const logitsBitmaskHost = ITensor::at(mLogitsBitmaskHost, {seqSlot});

                std::array<int64_t, 1> bitmaskShape{mBitmaskSize};
                DLTensor logitsBitmaskDlt{logitsBitmaskHost->data(), DLDevice{kDLCPU, 0}, 1, DLDataType{kDLInt, 32, 1},
                    bitmaskShape.data(), nullptr, 0};
                mXGrammarMatchers.at(seqSlot)->FillNextTokenBitmask(&logitsBitmaskDlt);
                mCopyBufferManager.copy(*logitsBitmaskHost, *logitsBitmask);
            }
        }
    }
}

void GuidedDecoder::execute(ScheduledRequests const& scheduledRequests, BufferManager const& runtimeBufferManager,
    std::vector<TensorPtr> const& decoderBuffersLogits)
{
    auto const& stream = runtimeBufferManager.getStream();

    // Wait for mCopyBufferManager finishing the H2D copy of logitsBitmask
    // TODO(enweiz): Move the H2D copy of logitsBitmaskPtrVec to buildGuidedDecoding.
    // This may not bring too much perf gain because of the small size of logitsBitmaskPtrVec.
    // TODO(enweiz): For chunked context, we currently build mask cache at the first context chunk, and apply
    // the mask at the last context chunk. So, ideally we should sync the stream at the last context chunk.
    CudaEvent event{};
    mCopyBufferManager.getStream().record(event);
    stream.wait(event);

    SizeType32 batchIdx{0};
    if (mGuidedDecodingBackend == executor::GuidedDecodingConfig::GuidedDecodingBackend::kXGRAMMAR)
    {
        for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
        {
            for (auto const& llmReq : requests)
            {
                if (llmReq->isContextInitState() && !llmReq->isLastContextChunk())
                {
                    continue;
                }
                auto const& guidedDecodingParams = llmReq->getGuidedDecodingParams();
                if (guidedDecodingParams.has_value())
                {
                    auto const seqSlot = llmReq->mSeqSlot.value();

                    auto const& logits = decoderBuffersLogits.at(seqSlot);
                    auto const logitsBitmask = ITensor::at(mLogitsBitmask, {seqSlot});

                    // Use void* to unify the code for different mLogitsDtype
                    *reinterpret_cast<void**>(ITensor::at(mLogitsPtrVecHost, {batchIdx})->data()) = logits->data();
                    *reinterpret_cast<void**>(ITensor::at(mLogitsBitmaskPtrVecHost, {batchIdx})->data())
                        = logitsBitmask->data();

                    ++batchIdx;
                }
            }
        }
        if (batchIdx > 0)
        {
            runtimeBufferManager.copy(
                *ITensor::slice(mLogitsPtrVecHost, 0, batchIdx), *ITensor::slice(mLogitsPtrVec, 0, batchIdx));
            runtimeBufferManager.copy(*ITensor::slice(mLogitsBitmaskPtrVecHost, 0, batchIdx),
                *ITensor::slice(mLogitsBitmaskPtrVec, 0, batchIdx));

            auto logitsBitmaskPtrVec = bufferCast<BitmaskT const*>(*mLogitsBitmaskPtrVec);
            if (mLogitsDtype == nvinfer1::DataType::kFLOAT)
            {
                auto logitsPtrVec = bufferCast<float*>(*mLogitsPtrVec);
                tensorrt_llm::kernels::invokeLogitsBitmask<float>(
                    logitsPtrVec, logitsBitmaskPtrVec, batchIdx, mVocabSizePadded, stream.get());
            }
            else if (mLogitsDtype == nvinfer1::DataType::kHALF)
            {
                auto logitsPtrVec = bufferCast<half*>(*mLogitsPtrVec);
                tensorrt_llm::kernels::invokeLogitsBitmask<half>(
                    logitsPtrVec, logitsBitmaskPtrVec, batchIdx, mVocabSizePadded, stream.get());
            }
            else
            {
                TLLM_THROW("Unsupported logits data type.");
            }
        }
    }
}

} // namespace tensorrt_llm::batch_manager
