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

#include "tensorrt_llm/runtime/decodingLayerWorkspace.h"

#include <utility>

tensorrt_llm::runtime::DecodingLayerWorkspace::DecodingLayerWorkspace(std::shared_ptr<BufferManager> bufferManager,
    tensorrt_llm::layers::DecoderDomain const& decoderDomain, nvinfer1::DataType logitsType,
    size_t workspaceBufferSizeInBytes)
    : mBufferManager(std::move(bufferManager))
    , mBatchSlotsDevice(
          mBufferManager->gpu(ITensor::makeShape({decoderDomain.getBatchSize()}), TRTDataType<SizeType32>::value))
    , mRuntimeLogitsDevice(
          mBufferManager->gpu(ITensor::makeShape({decoderDomain.getBatchSize(), decoderDomain.getMaxDecodingTokens(),
                                  decoderDomain.getBeamWidth(), decoderDomain.getVocabSizePadded()}),
              logitsType))
    , mCurandStatesDevice(
          mBufferManager->gpu(ITensor::makeShape({decoderDomain.getBatchSize(), sizeof(curandState_t)})))
    , mWorkspaceDeviceBuffer(mBufferManager->gpu(workspaceBufferSizeInBytes))
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_DEBUG("Creating decoding workspace for a maximum batch size of %i, with a scratch space of %lu bytes",
        decoderDomain.getBatchSize(), workspaceBufferSizeInBytes);
    mBufferManager->getStream().synchronize();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void* tensorrt_llm::runtime::DecodingLayerWorkspace::getRawWorkspaceDevicePtr() const
{
    return mWorkspaceDeviceBuffer->data();
}

tensorrt_llm::runtime::DecodingLayerWorkspace::BufferPtr
tensorrt_llm::runtime::DecodingLayerWorkspace::getWorkspaceDeviceBuffer() const
{
    return mWorkspaceDeviceBuffer;
}

void tensorrt_llm::runtime::DecodingLayerWorkspace::setDeviceBatchSlots(TensorConstPtr const& newBatchSlots)
{
    mBatchSlotsDevice->reshape(newBatchSlots->getShape());
    mBufferManager->copy(*newBatchSlots, *mBatchSlotsDevice);
}

tensorrt_llm::runtime::SizeType32 const* tensorrt_llm::runtime::DecodingLayerWorkspace::getDeviceBatchSlotsPtr() const
{
    return tensorrt_llm::runtime::bufferCast<tensorrt_llm::runtime::SizeType32>(*mBatchSlotsDevice);
}

tensorrt_llm::runtime::DecodingLayerWorkspace::TensorConstPtr
tensorrt_llm::runtime::DecodingLayerWorkspace::getDeviceBatchSlots() const
{
    return mBatchSlotsDevice;
}

tensorrt_llm::runtime::DecodingLayerWorkspace::TensorPtr
tensorrt_llm::runtime::DecodingLayerWorkspace::getDeviceRuntimeLogits() const
{
    return mRuntimeLogitsDevice;
}

void tensorrt_llm::runtime::DecodingLayerWorkspace::resize(size_t minSize)
{
    if (mWorkspaceDeviceBuffer->getSizeInBytes() < minSize)
    {
        mWorkspaceDeviceBuffer->resize(minSize);
    }
}

tensorrt_llm::runtime::DecodingLayerWorkspace::TensorPtr
tensorrt_llm::runtime::DecodingLayerWorkspace::getWorkspaceAsDeviceTensor(ITensor::Shape shape, nvinfer1::DataType type)
{
    auto const sizeInBytes = ITensor::volume(shape) * BufferDataType(type).getSize();
    return std::make_shared<GenericTensor<BorrowingAllocator<MemoryType::kGPU>>>(
        shape, type, BorrowingAllocator<MemoryType::kGPU>{mWorkspaceDeviceBuffer->data(), sizeInBytes});
}

void tensorrt_llm::runtime::DecodingLayerWorkspace::initializeDeviceCurandStates(
    std::optional<std::vector<uint64_t>> const& randomSeed, tensorrt_llm::runtime::SizeType32 batchSize,
    tensorrt_llm::runtime::DecodingLayerWorkspace::TensorConstPtr const& batchSlots,
    tensorrt_llm::runtime::DecodingLayerWorkspace::TensorPtr& statesDevice)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // If runtime argument has single random seed, using this random seed to
    // initialize the random table of all sentences. If the argument has
    // [batchSize] random seeds, initializing the random table by different
    // random seeds respectively. If no random seed, initialize the random table
    // of all sentences by 0 directly.
    auto const* batchSlotsPtr = tensorrt_llm::runtime::bufferCast<tensorrt_llm::runtime::SizeType32>(*batchSlots);
    auto* curandStateDevicePtr = reinterpret_cast<curandState_t*>(statesDevice->data());
    if (randomSeed)
    {
        if (randomSeed->size() == 1)
        {
            tensorrt_llm::kernels::invokeCurandInitialize(
                curandStateDevicePtr, batchSlotsPtr, batchSize, randomSeed->front(), getStream());
        }
        else
        {
            TLLM_CHECK_WITH_INFO(static_cast<tensorrt_llm::runtime::SizeType32>(randomSeed->size()) == batchSize,
                "Random seed vector size mismatch.");
            auto randomSeedsDevice = copyToWorkspace(randomSeed.value());
            auto const* randomSeedsDevicePtr = tensorrt_llm::runtime::bufferCast<uint64_t>(*randomSeedsDevice);
            tensorrt_llm::kernels::invokeCurandBatchInitialize(
                curandStateDevicePtr, batchSlotsPtr, batchSize, randomSeedsDevicePtr, getStream());
        }
    }
    else
    {
        // Initialize curand states using the default seed 0.
        tensorrt_llm::kernels::invokeCurandInitialize(curandStateDevicePtr, batchSlotsPtr, batchSize, 0, getStream());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

cudaStream_t tensorrt_llm::runtime::DecodingLayerWorkspace::getStream()
{
    return mBufferManager->getStream().get();
}
