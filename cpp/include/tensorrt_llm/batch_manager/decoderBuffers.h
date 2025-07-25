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
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <vector>

namespace tensorrt_llm::runtime::decoder
{
class DecoderState;
}

namespace tensorrt_llm::batch_manager
{

class DecoderInputBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;

    explicit DecoderInputBuffers(
        SizeType32 maxBatchSize, SizeType32 maxDecoderSteps, runtime::BufferManager const& manager);

    void setupMedusaLogits(SizeType32 maxNumSequences, runtime::ModelConfig const& modelConfig);

    //! Buffers for decoder setup

    //! Input IDs of new requests, [maxBatchSize]
    TensorPtr inputsIds;
    //! Batch slots for setup step, [maxBatchSize]
    TensorPtr setupBatchSlots;
    TensorPtr setupBatchSlotsDevice;
    //! Helper buffer for copying sequence lengths, [maxBatchSize]
    TensorPtr fillValues;
    TensorPtr fillValuesDevice;

    //! Buffers for decoder forward

    //! Requests for considered in decoder forward
    RequestVector decoderRequests;

    //! Batch slots for all decoder steps, [maxDecoderSteps][maxBatchSize]
    std::vector<TensorPtr> forwardBatchSlots;

    //! Logits of decoder requests
    std::vector<TensorPtr> logits;

    //! Logits for speculative decoding (Medusa)
    //! [maxBatchSize][maxAcceptedDraftTokensPerStep][maxDraftTokens + 1, vocabSizePadded]
    std::vector<std::vector<runtime::ITensor::SharedPtr>> predictedDraftLogits;
};

class DecoderOutputBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;

    DecoderOutputBuffers(SizeType32 maxNumSequences, SizeType32 maxBeamWidth, SizeType32 maxSeqLen,
        SizeType32 maxTokensPerStep, runtime::BufferManager const& manager);

    void enableLookaheadDecoding(SizeType32 maxNumSequences, SizeType32 maxTokensPerStep);
    void disableLookaheadDecoding(SizeType32 maxNumSequences);

    void setupSpeculativeDecoding(
        SizeType32 maxNumSequences, SizeType32 maxTokensPerStep, runtime::ModelConfig const& modelConfig);

    TensorPtr sequenceLengthsHost; // [mMaxNumRequests, beamWidth], pinned host tensor
    TensorPtr newOutputTokensHost; // [maxTokensPerStep, mMaxNumRequests, beamWidth]
    TensorPtr cumLogProbsHost;     // [mMaxNumRequests, beamWidth]
    TensorPtr logProbsHost;        // [mMaxNumRequests, beamWidth, maxSeqLen]
    TensorPtr finishedSumHost;     // [mMaxNumRequests], pinned host tensor
    TensorPtr finishReasonsHost;   // [mMaxNumRequests, beamWidth], pinned host tensor

    // speculative decoding buffers
    TensorPtr nextDraftTokensHost;        // [mMaxNumRequests, maxTokensPerStep-1]
    TensorPtr prevDraftTokensLengthsHost; // [mMaxNumRequests]
    TensorPtr nextDraftTokensLengthsHost; // [mMaxNumRequests]
};

class DecoderStepAsyncSend
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;

    DecoderStepAsyncSend(DecoderOutputBuffers const& decoderOutputBuffers,
        runtime::decoder::DecoderState const& decoderState, bool returnLogProbs, SizeType32 maxBeamWidth,
        bool useMedusa, mpi::MpiComm const& commSession, int peer);

    ~DecoderStepAsyncSend();

    static void recv(DecoderOutputBuffers const& decoderOutputBuffers,
        runtime::decoder::DecoderState const& decoderState, bool returnLogProbs, SizeType32 maxBeamWidth,
        bool useMedusa, mpi::MpiComm const& commSession, int peer);

    static void bcast(DecoderOutputBuffers const& decoderOutputBuffers,
        runtime::decoder::DecoderState const& decoderState, bool returnLogProbs, SizeType32 maxBeamWidth,
        bool useMedusa, mpi::MpiComm const& commSession, int root);

private:
    std::unique_ptr<mpi::MpiRequest> mRequest1;
    std::unique_ptr<mpi::MpiRequest> mRequest2;
    std::unique_ptr<mpi::MpiRequest> mRequest3;
    std::unique_ptr<mpi::MpiRequest> mRequest4;
    std::unique_ptr<mpi::MpiRequest> mRequest5;
    std::unique_ptr<mpi::MpiRequest> mRequest6;
    std::unique_ptr<mpi::MpiRequest> mRequest7;
    std::unique_ptr<mpi::MpiRequest> mRequest8;
    std::unique_ptr<mpi::MpiRequest> mRequest9;
};

class SlotDecoderBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;

    TensorPtr outputIds;           // [beamWidth, maxSeqLen], outputIds of single batch slot
    TensorPtr outputIdsHost;       // [beamWidth, maxSeqLen], outputIds of single batch slot
    TensorPtr sequenceLengths;     // [beamWidth]
    TensorPtr sequenceLengthsHost; // [beamWidth]
    TensorPtr cumLogProbs;         // [beamWidth]
    TensorPtr cumLogProbsHost;     // [beamWidth]
    TensorPtr logProbs;            // [beamWidth, maxSeqLen]
    TensorPtr logProbsHost;        // [beamWidth, maxSeqLen]
    TensorPtr finishReasonsHost;   // [beamWidth]

    SlotDecoderBuffers(SizeType32 maxBeamWidth, SizeType32 maxSeqLen, runtime::BufferManager const& manager);
};

class DecoderSlotAsyncSend
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;

    DecoderSlotAsyncSend(TensorPtr const& outputIds, TensorPtr const& sequenceLengths, TensorPtr const& cumLogProbs,
        TensorPtr const& logProbs, bool returnLogProbs, mpi::MpiComm const& commSession, int peer);

    DecoderSlotAsyncSend(
        SlotDecoderBuffers const& slotDecoderBuffers, bool returnLogProbs, mpi::MpiComm const& commSession, int peer);

    ~DecoderSlotAsyncSend();

    static void recv(
        SlotDecoderBuffers const& slotDecoderBuffers, bool returnLogProbs, mpi::MpiComm const& commSession, int peer);

private:
    std::unique_ptr<mpi::MpiRequest> mRequest1;
    std::unique_ptr<mpi::MpiRequest> mRequest2;
    std::unique_ptr<mpi::MpiRequest> mRequest3;
    std::unique_ptr<mpi::MpiRequest> mRequest4;
};

} // namespace tensorrt_llm::batch_manager
