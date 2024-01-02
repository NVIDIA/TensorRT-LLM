/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

using namespace tensorrt_llm::runtime;

void DecodingOutput::BeamHypotheses::empty(BufferManager& manager)
{
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType>::value;
    auto constexpr nvFloatType = TRTDataType<float>::value;
    auto constexpr nvBoolType = TRTDataType<bool>::value;

    outputIdsTgt = manager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    sequenceLengthsTgt = manager.emptyTensor(MemoryType::kGPU, nvSizeType);
    cumLogProbs = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
    normedScores = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
    logProbs = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
    minNormedScores = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
    numBeams = manager.emptyTensor(MemoryType::kGPU, nvSizeType);
    isDone = manager.emptyTensor(MemoryType::kGPU, nvBoolType);
}

void DecodingOutput::BeamHypotheses::reshape(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength)
{
    outputIdsTgt->reshape(ITensor::makeShape({batchSize, 2 * beamWidth, maxSequenceLength}));
    sequenceLengthsTgt->reshape(ITensor::makeShape({batchSize, 2 * beamWidth}));
    cumLogProbs->reshape(ITensor::makeShape({batchSize, 2 * beamWidth}));
    normedScores->reshape(ITensor::makeShape({batchSize, 2 * beamWidth}));
    logProbs->reshape(ITensor::makeShape({batchSize, 2 * beamWidth, maxSequenceLength}));
    minNormedScores->reshape(ITensor::makeShape({batchSize}));
    numBeams->reshape(ITensor::makeShape({batchSize}));
    isDone->reshape(ITensor::makeShape({batchSize}));
}

void DecodingOutput::BeamHypotheses::init(BufferManager& manager, TokenIdType endId)
{
    kernels::invokeFill(*outputIdsTgt, endId, manager.getStream());
    manager.setZero(*sequenceLengthsTgt);
    manager.setZero(*cumLogProbs);
    manager.setZero(*normedScores);
    manager.setZero(*logProbs);
    manager.setZero(*minNormedScores);
    manager.setZero(*numBeams);
    manager.setZero(*isDone);
}

DecodingOutput::BeamHypotheses DecodingOutput::BeamHypotheses::slice(SizeType batchIndex, SizeType size) const
{
    DecodingOutput::BeamHypotheses bh{};
    bh.outputIdsTgt = ITensor::slice(outputIdsTgt, batchIndex, size);
    bh.sequenceLengthsTgt = ITensor::slice(sequenceLengthsTgt, batchIndex, size);
    bh.cumLogProbs = ITensor::slice(cumLogProbs, batchIndex, size);
    bh.normedScores = ITensor::slice(normedScores, batchIndex, size);
    bh.logProbs = ITensor::slice(logProbs, batchIndex, size);
    bh.minNormedScores = ITensor::slice(minNormedScores, batchIndex, size);
    bh.numBeams = ITensor::slice(numBeams, batchIndex, size);
    bh.isDone = ITensor::slice(isDone, batchIndex, size);
    return bh;
}

void DecodingOutput::BeamHypotheses::release()
{
    outputIdsTgt->release();
    sequenceLengthsTgt->release();
    cumLogProbs->release();
    normedScores->release();
    logProbs->release();
    minNormedScores->release();
    numBeams->release();
    isDone->release();
}
