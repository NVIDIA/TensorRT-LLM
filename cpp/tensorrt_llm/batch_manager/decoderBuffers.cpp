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

#include "tensorrt_llm/batch_manager/decoderBuffers.h"

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/mpiTags.h"

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

DecoderInputBuffers::DecoderInputBuffers(
    SizeType32 maxBatchSize, SizeType32 maxDecoderSteps, BufferManager const& manager)
{
    auto const maxBatchSizeShape = ITensor::makeShape({maxBatchSize});
    auto const nvSizeType = TRTDataType<SizeType32>::value;

    inputsIds = BufferManager::pinnedPool(ITensor::makeShape({0}), TRTDataType<TokenIdType>::value);

    setupBatchSlots = BufferManager::pinnedPool(maxBatchSizeShape, nvSizeType);
    setupBatchSlotsDevice = manager.gpu(maxBatchSizeShape, nvSizeType);

    fillValues = tensorrt_llm::runtime::BufferManager::pinnedPool(maxBatchSizeShape, nvSizeType);
    fillValuesDevice = manager.gpu(maxBatchSizeShape, nvSizeType);

    forwardBatchSlots.reserve(maxDecoderSteps);
    for (SizeType32 i = 0; i < maxDecoderSteps; ++i)
    {
        forwardBatchSlots.emplace_back(BufferManager::pinnedPool(ITensor::makeShape({maxBatchSize}), nvSizeType));
    }
}

DecoderOutputBuffers::DecoderOutputBuffers(SizeType32 maxNumSequences, SizeType32 maxBeamWidth, SizeType32 maxSeqLen,
    SizeType32 maxTokensPerStep, BufferManager const& manager)
{
    auto constexpr TRTTokenIdType = runtime::TRTDataType<runtime::TokenIdType>::value;

    sequenceLengthsHost
        = BufferManager::pinned(ITensor::makeShape({maxNumSequences, maxBeamWidth}), nvinfer1::DataType::kINT32);

    finishedSumHost = BufferManager::pinned(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);

    newOutputTokensHost
        = BufferManager::pinned(ITensor::makeShape({maxTokensPerStep, maxNumSequences, maxBeamWidth}), TRTTokenIdType);

    cumLogProbsHost
        = BufferManager::pinned(ITensor::makeShape({maxNumSequences, maxBeamWidth}), nvinfer1::DataType::kFLOAT);

    logProbsHost = BufferManager::pinned(
        ITensor::makeShape({maxNumSequences, maxBeamWidth, maxSeqLen}), nvinfer1::DataType::kFLOAT);

    finishReasonsHost
        = BufferManager::pinned(ITensor::makeShape({maxNumSequences, maxBeamWidth}), nvinfer1::DataType::kUINT8);
}

void DecoderOutputBuffers::enableLookaheadDecoding(SizeType32 maxNumSequences, SizeType32 maxTokensPerStep)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    newOutputTokensHost->reshape(ITensor::makeShape({maxTokensPerStep, maxNumSequences, 1}));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderOutputBuffers::disableLookaheadDecoding(SizeType32 maxNumSequences)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    newOutputTokensHost->reshape(ITensor::makeShape({1, maxNumSequences, 1}));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

DecoderBuffers::DecoderBuffers(SizeType32 maxNumSequences, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
    SizeType32 maxTokensPerStep, BufferManager const& manager, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    if (worldConfig.isLastPipelineParallelRank())
    {
        logits.resize(maxNumSequences);
    }

    cacheIndirectionInput = manager.gpu(
        ITensor::makeShape({maxNumSequences, maxBeamWidth, maxAttentionWindow}), nvinfer1::DataType::kINT32);
    cacheIndirectionOutput = manager.gpu(
        ITensor::makeShape({maxNumSequences, maxBeamWidth, maxAttentionWindow}), nvinfer1::DataType::kINT32);

    if (modelConfig.getSpeculativeDecodingMode().needsKVCacheRewind()
        || modelConfig.getSpeculativeDecodingMode().hasDraftLogits()
        || modelConfig.getSpeculativeDecodingMode().predictsDraftTokens())
    {
        draftBuffers.create(maxNumSequences, maxTokensPerStep, manager, modelConfig);
    }
}

void DecoderBuffers::DraftBuffers::create(SizeType32 maxNumSequences, SizeType32 maxTokensPerStep,
    BufferManager const& manager, ModelConfig const& modelConfig)
{
    auto const speculativeDecodingMode = modelConfig.getSpeculativeDecodingMode();

    auto constexpr TRTTokenIdType = runtime::TRTDataType<runtime::TokenIdType>::value;

    if (speculativeDecodingMode.predictsDraftTokens())
    {
        nextDraftTokensHost
            = BufferManager::pinned(ITensor::makeShape({maxNumSequences, maxTokensPerStep - 1}), TRTTokenIdType);
        if (speculativeDecodingMode.variableDraftLength())
        {
            nextDraftTokensLengthsHost
                = BufferManager::pinned(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
            prevDraftTokensLengthsHost
                = BufferManager::pinned(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
        }
    }

    if (speculativeDecodingMode.isMedusa())
    {
        auto const maxDraftPathLen = modelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen();
        predictedDraftLogits.resize(maxNumSequences);
        for (auto& medusaLogitsHead : predictedDraftLogits)
        {
            medusaLogitsHead.resize(maxDraftPathLen);
        }
    }

    if (speculativeDecodingMode.needsKVCacheRewind())
    {
        auto const maxDraftPathLen = modelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen();
        acceptedLengthsCumSumDevice
            = manager.gpu(ITensor::makeShape({maxNumSequences + 1}), nvinfer1::DataType::kINT32);
        acceptedPackedPathsDevice
            = manager.gpu(ITensor::makeShape({maxNumSequences, maxDraftPathLen}), nvinfer1::DataType::kINT32);
    }
}

DecoderStepAsyncSend::DecoderStepAsyncSend(DecoderOutputBuffers const& decoderOutputBuffers,
    DecoderBuffers const& decoderBuffers, bool const returnLogProbs, SizeType32 const maxBeamWidth,
    bool const useMedusa, mpi::MpiComm const& commSession, int peer)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_DEBUG("start send outputs of DecoderBuffers to rank %d", peer);

    mRequest1 = commSession.sendAsync(
        *decoderOutputBuffers.newOutputTokensHost, peer, mpi::MpiTag::kDecoderStepNewOutputTokensHost);
    mRequest2
        = commSession.sendAsync(*decoderOutputBuffers.finishedSumHost, peer, mpi::MpiTag::kDecoderStepFinishedSumHost);
    mRequest3 = commSession.sendAsync(
        *decoderOutputBuffers.sequenceLengthsHost, peer, mpi::MpiTag::kDecoderStepSequenceLengthsHost);
    mRequest4 = returnLogProbs
        ? commSession.sendAsync(*decoderOutputBuffers.cumLogProbsHost, peer, mpi::MpiTag::kDecoderStepCumLogProbsHost)
        : nullptr;
    mRequest5 = returnLogProbs
        ? commSession.sendAsync(*decoderOutputBuffers.logProbsHost, peer, mpi::MpiTag::kDecoderStepLogProbsHost)
        : nullptr;
    mRequest6 = maxBeamWidth > 1 ? commSession.sendAsync(
                    *decoderBuffers.cacheIndirectionOutput, peer, mpi::MpiTag::kDecoderStepCacheIndirectionOutput)
                                 : nullptr;
    mRequest7 = useMedusa ? commSession.sendAsync(*decoderBuffers.draftBuffers.acceptedLengthsCumSumDevice, peer,
                    mpi::MpiTag::kDecoderStepAcceptedLengthsCumSumDevice)
                          : nullptr;
    mRequest8 = useMedusa ? commSession.sendAsync(*decoderBuffers.draftBuffers.acceptedPackedPathsDevice, peer,
                    mpi::MpiTag::kDecoderStepAcceptedPackedPathsDevice)
                          : nullptr;
    mRequest9 = commSession.sendAsync(
        *decoderOutputBuffers.finishReasonsHost, peer, mpi::MpiTag::kDecoderStepFinishReasonsHost);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderStepAsyncSend::recv(DecoderOutputBuffers const& decoderOutputBuffers, DecoderBuffers const& decoderBuffers,
    bool const returnLogProbs, SizeType32 const maxBeamWidth, bool const useMedusa, mpi::MpiComm const& commSession,
    int const peer)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_DEBUG("start recv outputs of DecoderBuffers from rank %d", peer);

    commSession.recv(*decoderOutputBuffers.newOutputTokensHost, peer, mpi::MpiTag::kDecoderStepNewOutputTokensHost);
    commSession.recv(*decoderOutputBuffers.finishedSumHost, peer, mpi::MpiTag::kDecoderStepFinishedSumHost);
    commSession.recv(*decoderOutputBuffers.sequenceLengthsHost, peer, mpi::MpiTag::kDecoderStepSequenceLengthsHost);
    if (returnLogProbs)
    {
        commSession.recv(*decoderOutputBuffers.cumLogProbsHost, peer, mpi::MpiTag::kDecoderStepCumLogProbsHost);
        commSession.recv(*decoderOutputBuffers.logProbsHost, peer, mpi::MpiTag::kDecoderStepLogProbsHost);
    }
    if (maxBeamWidth > 1)
    {
        commSession.recv(*decoderBuffers.cacheIndirectionOutput, peer, mpi::MpiTag::kDecoderStepCacheIndirectionOutput);
    }
    if (useMedusa)
    {
        commSession.recv(*decoderBuffers.draftBuffers.acceptedLengthsCumSumDevice, peer,
            mpi::MpiTag::kDecoderStepAcceptedLengthsCumSumDevice);
        commSession.recv(*decoderBuffers.draftBuffers.acceptedPackedPathsDevice, peer,
            mpi::MpiTag::kDecoderStepAcceptedPackedPathsDevice);
    }
    commSession.recv(*decoderOutputBuffers.finishReasonsHost, peer, mpi::MpiTag::kDecoderStepFinishReasonsHost);

    TLLM_LOG_DEBUG("end recv outputs of DecoderBuffers from rank %d", peer);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

DecoderStepAsyncSend::~DecoderStepAsyncSend()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mRequest1->wait();
    mRequest2->wait();
    mRequest3->wait();
    if (mRequest4)
        mRequest4->wait();
    if (mRequest5)
        mRequest5->wait();
    if (mRequest6)
        mRequest6->wait();
    if (mRequest7)
        mRequest7->wait();
    if (mRequest8)
        mRequest8->wait();
    mRequest9->wait();

    TLLM_LOG_DEBUG("end send outputs of DecoderBuffers");
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderStepAsyncSend::bcast(DecoderOutputBuffers const& decoderOutputBuffers, DecoderBuffers const& decoderBuffers,
    bool const returnLogProbs, SizeType32 const maxBeamWidth, bool const useMedusa, mpi::MpiComm const& commSession,
    int const root)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_DEBUG("start bcast outputs of DecoderBuffers from rank %d", root);

    auto request1 = commSession.bcastAsync(*decoderOutputBuffers.newOutputTokensHost, root);
    auto request2 = commSession.bcastAsync(*decoderOutputBuffers.finishedSumHost, root);
    auto request3 = commSession.bcastAsync(*decoderOutputBuffers.sequenceLengthsHost, root);
    auto request4 = returnLogProbs ? commSession.bcastAsync(*decoderOutputBuffers.cumLogProbsHost, root) : nullptr;
    auto request5 = returnLogProbs ? commSession.bcastAsync(*decoderOutputBuffers.logProbsHost, root) : nullptr;
    auto request6 = maxBeamWidth > 1 ? commSession.bcastAsync(*decoderBuffers.cacheIndirectionOutput, root) : nullptr;
    auto request7
        = useMedusa ? commSession.bcastAsync(*decoderBuffers.draftBuffers.acceptedLengthsCumSumDevice, root) : nullptr;
    auto request8
        = useMedusa ? commSession.bcastAsync(*decoderBuffers.draftBuffers.acceptedPackedPathsDevice, root) : nullptr;
    auto request9 = commSession.bcastAsync(*decoderOutputBuffers.finishReasonsHost, root);

    request1->wait();
    request2->wait();
    request3->wait();
    if (request4)
        request4->wait();
    if (request5)
        request5->wait();
    if (request6)
        request6->wait();
    if (request7)
        request7->wait();
    if (request8)
        request8->wait();
    request9->wait();

    TLLM_LOG_DEBUG("end bcast outputs of DecoderBuffers from rank %d", root);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

DecoderSlotAsyncSend::DecoderSlotAsyncSend(TensorPtr const& outputIds, TensorPtr const& sequenceLengths,
    TensorPtr const& cumLogProbs, TensorPtr const& logProbs, bool const returnLogProbs, mpi::MpiComm const& commSession,
    int const peer)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_DEBUG("start send outputs of SlotDecoderBuffers to rank %d", peer);

    mRequest1 = commSession.sendAsync(*outputIds, peer, mpi::MpiTag::kDecoderSlotOutputIds);
    mRequest2 = commSession.sendAsync(*sequenceLengths, peer, mpi::MpiTag::kDecoderSlotSequenceLengths);
    mRequest3
        = returnLogProbs ? commSession.sendAsync(*cumLogProbs, peer, mpi::MpiTag::kDecoderSlotCumLogProbs) : nullptr;
    mRequest4 = returnLogProbs ? commSession.sendAsync(*logProbs, peer, mpi::MpiTag::kDecoderSlotLogProbs) : nullptr;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

DecoderSlotAsyncSend::DecoderSlotAsyncSend(SlotDecoderBuffers const& slotDecoderBuffers, bool const returnLogProbs,
    mpi::MpiComm const& commSession, int const peer)
    : DecoderSlotAsyncSend(slotDecoderBuffers.outputIds, slotDecoderBuffers.sequenceLengths,
        slotDecoderBuffers.cumLogProbs, slotDecoderBuffers.logProbs, returnLogProbs, commSession, peer)
{
}

void DecoderSlotAsyncSend::recv(SlotDecoderBuffers const& slotDecoderBuffers, bool const returnLogProbs,
    mpi::MpiComm const& commSession, int const peer)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_DEBUG("start recv outputs of SlotDecoderBuffers from rank %d", peer);

    commSession.recv(*slotDecoderBuffers.outputIds, peer, mpi::MpiTag::kDecoderSlotOutputIds);
    commSession.recv(*slotDecoderBuffers.sequenceLengths, peer, mpi::MpiTag::kDecoderSlotSequenceLengths);
    if (returnLogProbs)
    {
        commSession.recv(*slotDecoderBuffers.cumLogProbs, peer, mpi::MpiTag::kDecoderSlotCumLogProbs);
        commSession.recv(*slotDecoderBuffers.logProbs, peer, mpi::MpiTag::kDecoderSlotLogProbs);
    }

    TLLM_LOG_DEBUG("end recv outputs of SlotDecoderBuffers from rank %d", peer);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

DecoderSlotAsyncSend::~DecoderSlotAsyncSend()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mRequest1->wait();
    mRequest2->wait();
    if (mRequest3)
        mRequest3->wait();
    if (mRequest4)
        mRequest4->wait();

    TLLM_LOG_DEBUG("end send outputs of SlotDecoderBuffers");
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

SlotDecoderBuffers::SlotDecoderBuffers(SizeType32 maxBeamWidth, SizeType32 maxSeqLen, BufferManager const& manager)
{
    outputIds = manager.gpu(ITensor::makeShape({maxBeamWidth, maxSeqLen}), nvinfer1::DataType::kINT32);
    outputIdsHost = BufferManager::pinned(ITensor::makeShape({maxBeamWidth, maxSeqLen}), nvinfer1::DataType::kINT32);

    sequenceLengths = manager.gpu(ITensor::makeShape({maxBeamWidth}), nvinfer1::DataType::kINT32);
    sequenceLengthsHost = BufferManager::pinned(ITensor::makeShape({maxBeamWidth}), nvinfer1::DataType::kINT32);

    cumLogProbs = manager.gpu(ITensor::makeShape({maxBeamWidth}), nvinfer1::DataType::kFLOAT);
    cumLogProbsHost = BufferManager::pinned(ITensor::makeShape({maxBeamWidth}), nvinfer1::DataType::kFLOAT);

    logProbs = manager.gpu(ITensor::makeShape({maxBeamWidth, maxSeqLen}), nvinfer1::DataType::kFLOAT);
    logProbsHost = BufferManager::pinned(ITensor::makeShape({maxBeamWidth, maxSeqLen}), nvinfer1::DataType::kFLOAT);
}

} // namespace tensorrt_llm::batch_manager
