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

void DecodingOutput::BeamHypotheses::empty(BufferManager const& manager)
{
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
    auto constexpr nvFloatType = TRTDataType<float>::value;
    auto constexpr nvBoolType = TRTDataType<bool>::value;

    outputIdsCBA = manager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    logProbsCBA = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
    sequenceLengthsCBA = manager.emptyTensor(MemoryType::kGPU, nvSizeType);
    cumLogProbsCBA = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
    normedScoresCBA = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
    numBeamsCBA = manager.emptyTensor(MemoryType::kGPU, nvSizeType);
    minNormedScoresCBA = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
    batchDones = manager.emptyTensor(MemoryType::kGPU, nvBoolType);
}

void DecodingOutput::BeamHypotheses::reshape(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxSequenceLength)
{
    outputIdsCBA->reshape(ITensor::makeShape({batchSize, 2 * beamWidth, maxSequenceLength}));
    logProbsCBA->reshape(ITensor::makeShape({batchSize, 2 * beamWidth, maxSequenceLength}));
    sequenceLengthsCBA->reshape(ITensor::makeShape({batchSize, 2 * beamWidth}));
    cumLogProbsCBA->reshape(ITensor::makeShape({batchSize, 2 * beamWidth}));
    normedScoresCBA->reshape(ITensor::makeShape({batchSize, 2 * beamWidth}));
    numBeamsCBA->reshape(ITensor::makeShape({batchSize}));
    minNormedScoresCBA->reshape(ITensor::makeShape({batchSize}));
    batchDones->reshape(ITensor::makeShape({batchSize}));
}

void DecodingOutput::BeamHypotheses::init(BufferManager const& manager, TokenIdType endId)
{
    kernels::invokeFill(*outputIdsCBA, endId, manager.getStream());
    manager.setZero(*logProbsCBA);
    manager.setZero(*sequenceLengthsCBA);
    manager.setZero(*cumLogProbsCBA);
    manager.setZero(*normedScoresCBA);
    manager.setZero(*numBeamsCBA);
    manager.setZero(*minNormedScoresCBA);
    manager.setZero(*batchDones);
}

DecodingOutput::BeamHypotheses DecodingOutput::BeamHypotheses::slice(SizeType32 batchIndex, SizeType32 size) const
{
    DecodingOutput::BeamHypotheses bh{};
    bh.outputIdsCBA = ITensor::slice(outputIdsCBA, batchIndex, size);
    bh.logProbsCBA = ITensor::slice(logProbsCBA, batchIndex, size);
    bh.sequenceLengthsCBA = ITensor::slice(sequenceLengthsCBA, batchIndex, size);
    bh.cumLogProbsCBA = ITensor::slice(cumLogProbsCBA, batchIndex, size);
    bh.normedScoresCBA = ITensor::slice(normedScoresCBA, batchIndex, size);
    bh.numBeamsCBA = ITensor::slice(numBeamsCBA, batchIndex, size);
    bh.minNormedScoresCBA = ITensor::slice(minNormedScoresCBA, batchIndex, size);
    bh.batchDones = ITensor::slice(batchDones, batchIndex, size);
    return bh;
}
