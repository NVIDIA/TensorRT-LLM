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

#include "tensorrt_llm/batch_manager/transformerBuffers.h"

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/kernels/attentionMask.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaPackedMask.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmBuffers.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include <cstdint>

using namespace tensorrt_llm::runtime;
namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::batch_manager
{

TransformerBuffers::TransformerBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
    runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig)
    : maxInputLen(modelConfig.getMaxInputLen())
    , maxEncoderOutputLen(modelConfig.getMaxEncoderLen())
{
    auto const& manager = runtime.getBufferManager();
    auto const& engine = runtime.getEngine();

    positionIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    auto const localNbAttnLayers
        = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank());
    // find the index of the first attention layer in the current rank
    auto const firstLayerId = modelConfig.countLowerRankLayers(runtime::ModelConfig::LayerType::kATTENTION,
        worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank());

    cacheIndirection
        = manager.gpu(ITensor::makeShape({maxBatchSize, maxBeamWidth, maxAttentionWindow}), nvinfer1::DataType::kINT32);

    if (!modelConfig.getMaxNumTokens().has_value())
    {
        TLLM_THROW("Model must configure a max number of tokens.");
    }
    maxNumTokens = modelConfig.getMaxNumTokens().value();

    if (modelConfig.isKVCacheEnabled())
    {
        auto const kvCacheBlockOffsetsType = engine.getTensorDataType("kv_cache_block_offsets");
        kvCacheBlockOffsetsHost = manager.emptyTensor(MemoryType::kPINNEDPOOL, kvCacheBlockOffsetsType);
        kvCacheBlockOffsetsDevice = manager.emptyTensor(MemoryType::kGPU, kvCacheBlockOffsetsType);

        if (modelConfig.useCrossAttention())
        {
            crossKvCacheBlockOffsetsHost = manager.emptyTensor(MemoryType::kPINNEDPOOL, kvCacheBlockOffsetsType);
            crossKvCacheBlockOffsetsDevice = manager.emptyTensor(MemoryType::kGPU, kvCacheBlockOffsetsType);
            crossAttentionMaskDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kBOOL);
            crossAttentionMaskPinnedHost = tensorrt_llm::runtime::BufferManager::pinnedPool(
                ITensor::makeShape({maxNumTokens, maxEncoderOutputLen}), nvinfer1::DataType::kBOOL);
            crossAttentionPackedMaskDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
            crossAttentionCuQSeqLensDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
            crossAttentionPackedMaskCuMaskRowsDevice
                = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

            // Pinned memory for batch copy of attention masks.
            // There will be paddings in the dim1, so copy it by tokens.
            crossAttentionMaskCopySrcOffsets = tensorrt_llm::runtime::BufferManager::pinnedPool(
                ITensor::makeShape({maxNumTokens}), nvinfer1::DataType::kINT64);
            crossAttentionMaskCopyDstOffsets = tensorrt_llm::runtime::BufferManager::pinnedPool(
                ITensor::makeShape({maxNumTokens}), nvinfer1::DataType::kINT64);
            crossAttentionMaskCopySizes = tensorrt_llm::runtime::BufferManager::pinnedPool(
                ITensor::makeShape({maxNumTokens}), nvinfer1::DataType::kINT64);
        }
    }

    fillValuesAlt = tensorrt_llm::runtime::BufferManager::pinnedPool(
        ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    fillValuesAltDevice = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    seqSlotsAlt = tensorrt_llm::runtime::BufferManager::pinnedPool(
        ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    seqSlotsAltDevice = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

    cacheIndirBatchedCopySrcOffsets = tensorrt_llm::runtime::BufferManager::pinnedPool(
        ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT64);
    cacheIndirBatchedCopyDstOffsets = tensorrt_llm::runtime::BufferManager::pinnedPool(
        ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT64);
    cacheIndirBatchedCopySizes = tensorrt_llm::runtime::BufferManager::pinnedPool(
        ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT64);
    skipCrossAttnBlocks
        = tensorrt_llm::runtime::BufferManager::pinnedPool(ITensor::makeShape({1}), nvinfer1::DataType::kBOOL);

    pastKeyValueLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);

    maxAttentionWindows = BufferManager::cpu(ITensor::makeShape({localNbAttnLayers}), nvinfer1::DataType::kINT32);
    auto* maxAttentionWindowsPtr = bufferCast<SizeType32>(*maxAttentionWindows);
    auto const attentionWindowLength = maxAttentionWindowVec.size();
    for (SizeType32 i = 0; i < localNbAttnLayers; ++i)
    {
        maxAttentionWindowsPtr[i] = maxAttentionWindowVec[(firstLayerId + i) % attentionWindowLength];
    }

    sinkTokenLengths = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    bufferCast<SizeType32>(*sinkTokenLengths)[0] = sinkTokenLen;

    contextProgressHost = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT64);
    bufferCast<int64_t>(*contextProgressHost)[0] = 0;

    if (modelConfig.useGemmAllReducePlugin() && worldConfig.isTensorParallel())
    {
        nvinfer1::DataType ARType = modelConfig.getGemmAllReduceDtype();

        auto hiddenSize = modelConfig.getHiddenSize() * worldConfig.getTensorParallelism();

        auto tpGroup = worldConfig.getTensorParallelGroup();
        std::set<int> tpGroupSet(tpGroup.begin(), tpGroup.end());

        auto outputDims = ITensor::makeShape({modelConfig.getMaxNumTokens().value() * hiddenSize});

        gemmAllReduceOutput = std::make_shared<MulticastTensor>(outputDims, ARType, tpGroupSet);
    }
}

void TransformerBuffers::reshape(SizeType32 numSequences, SizeType32 numInputTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    pastKeyValueLengths->reshape(ITensor::makeShape({numSequences}));

    if (kvCacheBlockOffsetsHost)
    {
        auto cacheBlockOffsetsShape = kvCacheBlockOffsetsHost->getShape();
        if (cacheBlockOffsetsShape.nbDims > 0)
        {
            cacheBlockOffsetsShape.d[1] = numSequences;
            kvCacheBlockOffsetsHost->reshape(cacheBlockOffsetsShape);
            kvCacheBlockOffsetsDevice->reshape(cacheBlockOffsetsShape);
        }
        else
        {
            TLLM_LOG_DEBUG("kvCacheBlockOffsets not allocated yet");
        }
    }

    if (crossKvCacheBlockOffsetsHost)
    {
        TLLM_CHECK_WITH_INFO(
            crossKvCacheBlockOffsetsDevice, "crossKvCacheBlockOffsetsDevice is empty for model with cross attention!");
        auto crossCacheBlockOffsetsShape = crossKvCacheBlockOffsetsHost->getShape();
        if (crossCacheBlockOffsetsShape.nbDims > 0)
        {
            crossCacheBlockOffsetsShape.d[1] = numSequences;
            crossKvCacheBlockOffsetsHost->reshape(crossCacheBlockOffsetsShape);
            crossKvCacheBlockOffsetsDevice->reshape(crossCacheBlockOffsetsShape);
        }
        else
        {
            TLLM_LOG_DEBUG("crossKvCacheBlockOffsets not allocated yet");
        }
    }

    if (crossAttentionMaskDevice)
    {
        auto crossAttentionMaskShape = crossAttentionMaskDevice->getShape();
        if (crossAttentionMaskShape.nbDims > 0)
        {
            crossAttentionMaskShape.d[0] = numInputTokens;
            crossAttentionMaskDevice->reshape(crossAttentionMaskShape);
            crossAttentionMaskPinnedHost->reshape(crossAttentionMaskShape);
            crossAttentionMaskCopySrcOffsets->reshape(ITensor::makeShape({numInputTokens}));
            crossAttentionMaskCopyDstOffsets->reshape(ITensor::makeShape({numInputTokens}));
            crossAttentionMaskCopySizes->reshape(ITensor::makeShape({numInputTokens}));
        }
        else
        {
            TLLM_LOG_DEBUG("crossAttentionMaskDevice not allocated yet");
        }
    }

    if (crossAttentionPackedMaskDevice)
    {
        auto crossAttentionMaskPackedShape = crossAttentionPackedMaskDevice->getShape();
        if (crossAttentionMaskPackedShape.nbDims > 0)
        {
            crossAttentionMaskPackedShape.d[0] = numInputTokens;
            crossAttentionPackedMaskDevice->reshape(crossAttentionMaskPackedShape);
        }
        else
        {
            TLLM_LOG_DEBUG("crossAttentionPackedMaskDevice not allocated yet");
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::reshapeKvTensors(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxBlocksPerSeq,
    kv_cache_manager::CacheType kvCacheType, SizeType32 numPools, BufferManager const& manager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // allocate with max shape during init
    if (kvCacheType == kv_cache_manager::CacheType::kSELF)
    {
        auto const cacheBlockOffsetsShape
            = ITensor::makeShape({numPools, maxBatchSize * maxBeamWidth, 2, maxBlocksPerSeq});

        kvCacheBlockOffsetsHost->reshape(cacheBlockOffsetsShape);
        manager.setZero(*kvCacheBlockOffsetsHost);

        kvCacheBlockOffsetsDevice->reshape(cacheBlockOffsetsShape);
        manager.setZero(*kvCacheBlockOffsetsDevice);
    }
    else if (kvCacheType == kv_cache_manager::CacheType::kCROSS)
    {
        auto const crossCacheBlockOffsetsShape
            = ITensor::makeShape({numPools, maxBatchSize * maxBeamWidth, 2, maxBlocksPerSeq});

        crossKvCacheBlockOffsetsHost->reshape(crossCacheBlockOffsetsShape);
        manager.setZero(*crossKvCacheBlockOffsetsHost);

        crossKvCacheBlockOffsetsDevice->reshape(crossCacheBlockOffsetsShape);
        manager.setZero(*crossKvCacheBlockOffsetsDevice);

        crossAttentionMaskDevice->reshape(ITensor::makeShape({maxNumTokens, maxEncoderOutputLen}));
        manager.setZero(*crossAttentionMaskDevice);
        manager.setZero(*crossAttentionMaskPinnedHost);

        // Only context attention needs this, so allocate it by shape [maxBatchSize, maxInputLen, maxEncoderOutputLen].
        auto [packedMaskM, packedMaskN] = tk::roundUpPackedMaskMNDims(maxInputLen, maxEncoderOutputLen);
        crossAttentionPackedMaskDevice->reshape(ITensor::makeShape({maxBatchSize * packedMaskM, packedMaskN}));
        manager.setZero(*crossAttentionPackedMaskDevice);

        crossAttentionCuQSeqLensDevice->reshape(ITensor::makeShape({maxBatchSize + 1}));
        manager.setZero(*crossAttentionCuQSeqLensDevice);

        crossAttentionPackedMaskCuMaskRowsDevice->reshape(ITensor::makeShape({maxBatchSize + 1}));
        manager.setZero(*crossAttentionPackedMaskCuMaskRowsDevice);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::getBuffers(
    TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::ModelConfig const& modelConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(transformerBuffersGetBuffers);

    inputBuffers.insert_or_assign(kPositionIdsTensorName, positionIds);
    inputBuffers.insert_or_assign(kHostPastKeyValueLengthsTensorName, pastKeyValueLengths);
    inputBuffers.insert_or_assign(kCacheIndirectionsTensorName, cacheIndirection);
    inputBuffers.insert_or_assign(kHostSinkTokenLengthTensorName, sinkTokenLengths);

    inputBuffers.insert_or_assign(kHostMaxAttentionWindowSizesTensorName, maxAttentionWindows);
    inputBuffers.insert_or_assign(kKvCacheBlockOffsetsTensorName, kvCacheBlockOffsetsDevice);
    inputBuffers.insert_or_assign(kHostKvCacheBlockOffsetsTensorName, kvCacheBlockOffsetsHost);
    inputBuffers.insert_or_assign(kHostContextProgressTensorName, contextProgressHost);

    if (crossKvCacheBlockOffsetsHost)
    {
        inputBuffers.insert_or_assign(kCrossKvCacheBlockOffsetsTensorName, crossKvCacheBlockOffsetsDevice);
        inputBuffers.insert_or_assign(kHostCrossKvCacheBlockOffsetsTensorName, crossKvCacheBlockOffsetsHost);
        inputBuffers.insert_or_assign(kHostCrossKvCachePoolPointersTensorName, crossKvCacheBlockPoolPointers);
        inputBuffers.insert_or_assign(kHostCrossKvCachePoolMappingTensorName, crossKvCacheBlockPoolMapping);
        inputBuffers.insert_or_assign(kCrossAttentionMaskTensorName, crossAttentionMaskDevice);
        inputBuffers.insert_or_assign(kCrossAttentionPackedMaskTensorName, crossAttentionPackedMaskDevice);
    }

    if (skipCrossAttnBlocks)
    {
        inputBuffers.insert_or_assign(kSkipCrossAttentionBlocksTensorName, skipCrossAttnBlocks);
    }

    if (modelConfig.useGemmAllReducePlugin())
    {
        for (int idx = 0; idx < modelConfig.getNbAttentionLayers() * 2; ++idx)
        {
            // XXX (xsimmons): this is a bit hacky as it assumes
            // 2x RowLinear layers per attention block.
            // This will be fixed soon when I remove coupling between model
            // and runtime.
            auto gemmARViewUC = gemmAllReduceOutput->getTensorView(MulticastTensorView::ViewType::kUNICAST);
            auto gemmARViewMC = gemmAllReduceOutput->getTensorView(MulticastTensorView::ViewType::kMULTICAST);
            auto gemmARViewIpc = gemmAllReduceOutput->getTensorView(MulticastTensorView::ViewType::kIPC_LIST);

            outputBuffers.insert_or_assign("gemm_allreduce_uc_out_" + std::to_string(idx), gemmARViewUC);
            outputBuffers.insert_or_assign("gemm_allreduce_mc_out_" + std::to_string(idx), gemmARViewMC);
            outputBuffers.insert_or_assign("gemm_allreduce_ipc_out_" + std::to_string(idx), gemmARViewIpc);
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::copyPositionIds(runtime::TllmRuntime const& runtime,
    std::vector<SizeType32> const& positionIdsHost, bool isChatGlm, TensorPtr const& decoderPositionIds)
{
    auto const& manager = runtime.getBufferManager();
    if (isChatGlm)
    {
        positionIds->reshape(ITensor::makeShape({2, static_cast<int>(positionIdsHost.size()) / 2}));
        manager.copy(positionIdsHost.data(), *positionIds);
    }
    else if (decoderPositionIds == nullptr)
    {
        positionIds->reshape(ITensor::makeShape({static_cast<int>(positionIdsHost.size())}));
        manager.copy(positionIdsHost.data(), *positionIds);
    }
    else
    {
        // concat context phase and generation phase positionIds.
        auto const contextPositionIdsLen = static_cast<ITensor::DimType64>(positionIdsHost.size());
        auto const generationPositionIdsLen = ITensor::volume(decoderPositionIds->getShape());
        positionIds->reshape(ITensor::makeShape({contextPositionIdsLen + generationPositionIdsLen}));
        manager.copy(positionIdsHost.data(), *ITensor::slice(positionIds, 0, contextPositionIdsLen));
        manager.copy(*decoderPositionIds, *ITensor::slice(positionIds, contextPositionIdsLen));
    }
}

void TransformerBuffers::copyKvBlockOffsets(RequestVector const& contextRequests, RequestVector const& genRequests,
    kv_cache_manager::BaseKVCacheManager const* kvCacheManager,
    kv_cache_manager::BaseKVCacheManager const* crossKvCacheManager, BufferManager const& manager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(copyKvBlockOffsets);

    auto const& cudaStream = manager.getStream();

    SizeType32 constexpr contextBeamWidth{1};
    SizeType32 numSequences{0};
    SizeType32 maxBlockCount{0};
    SizeType32 maxCrossBlockCount{0};
    for (auto const& requests : {contextRequests, genRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const requestId = llmReq->mRequestId;
            auto const isContextRequest = llmReq->isContextInitState();
            auto const beamWidth = isContextRequest ? contextBeamWidth : llmReq->getBeamWidthByIter();
            auto const maxBeamBlockCount
                = kvCacheManager->copyBlockOffsets(*kvCacheBlockOffsetsHost, numSequences, requestId);
            maxBlockCount = std::max(maxBlockCount, maxBeamBlockCount);
            if (crossKvCacheBlockOffsetsHost)
            {
                auto const maxCrossBeamBlockCount
                    = crossKvCacheManager->copyBlockOffsets(*crossKvCacheBlockOffsetsHost, numSequences, requestId);
                maxCrossBlockCount = std::max(maxCrossBlockCount, maxCrossBeamBlockCount);
            }
            numSequences += beamWidth;
        }
    }

    // requests' block offsets collected as [totalNumSequences, 2, maxBlocksPerSeq], copy to device
    auto copyOffsetsToDevice = [&cudaStream](TensorPtr& offsetsHost, TensorPtr& offsetsDevice, SizeType32 maxBlockCount)
    {
        // shape should be [totalNumSequences, 2, maxBlocksPerSeq]
        auto const& offsetsShape = offsetsHost->getShape();
        auto const maxBlocksPerSeq = offsetsShape.d[3];
        auto const offsetsTypeSize = tensorrt_llm::common::getDTypeSize(offsetsHost->getDataType());
        auto const copyPitch = maxBlocksPerSeq * offsetsTypeSize;
        auto const copyHeight = offsetsShape.d[0] * offsetsShape.d[1] * offsetsShape.d[2];
        auto const copyWidth = maxBlockCount * offsetsTypeSize;
        auto* srcPtr = bufferCast<tk::KVCacheIndex>(*offsetsHost);
        auto* dstPtr = bufferCast<tk::KVCacheIndex>(*offsetsDevice);

        TLLM_CUDA_CHECK(cudaMemcpy2DAsync(
            dstPtr, copyPitch, srcPtr, copyPitch, copyWidth, copyHeight, cudaMemcpyHostToDevice, cudaStream.get()));
    };

    copyOffsetsToDevice(kvCacheBlockOffsetsHost, kvCacheBlockOffsetsDevice, maxBlockCount);
    if (crossKvCacheBlockOffsetsHost)
    {
        copyOffsetsToDevice(crossKvCacheBlockOffsetsHost, crossKvCacheBlockOffsetsDevice, maxCrossBlockCount);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::copyCacheIndirection(
    RequestVector const& genRequests, TensorPtr const& decoderCacheIndirectionOutput, CudaStream const& stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(copyCacheIndirection);

    auto const numGenerationRequests = genRequests.size();

    auto batchedCopySrcOffsets = BufferRange<SizeType64>(*cacheIndirBatchedCopySrcOffsets);
    auto batchedCopyDstOffsets = BufferRange<SizeType64>(*cacheIndirBatchedCopyDstOffsets);
    auto batchedCopySizes = BufferRange<SizeType64>(*cacheIndirBatchedCopySizes);

    auto cacheIndirShape = decoderCacheIndirectionOutput->getShape();

    // At present, all requests of a batch must have the same beam width in one generation step (or they will not
    // be batched together). So, the beam width of the first request is taken here to reshape the buffer.
    // Corresponding changes must be done if Diverse-Beam-Width-Search (DBWS, requests with diverse beam width in
    // a batch in one generation step) is supported in the future.
    auto reqBeamWidth = genRequests[0]->getBeamWidthByIter();

    // Get size of copying from shape of `CacheIndirectionOutput`
    cacheIndirShape.d[0] = 1;
    cacheIndirShape.d[1] = reqBeamWidth; // Use beam width of current step rather than max beam width as dst offset
    auto const copySize = static_cast<SizeType64>(ITensor::volume(cacheIndirShape));

    std::transform(genRequests.begin(), genRequests.end(), batchedCopySrcOffsets.begin(),
        [copySize](auto const& llmReq) { return llmReq->mSeqSlot.value() * copySize; });
    std::generate_n(
        batchedCopyDstOffsets.begin(), numGenerationRequests, [copySize, i = 0]() mutable { return (i++) * copySize; });
    std::fill_n(batchedCopySizes.begin(), numGenerationRequests, copySize);

    auto const batchedCopySrcOffsetsSlice = ITensor::slice(cacheIndirBatchedCopySrcOffsets, 0, numGenerationRequests);
    auto const batchedCopyDstOffsetsSlice = ITensor::slice(cacheIndirBatchedCopyDstOffsets, 0, numGenerationRequests);
    auto const batchedCopySizesSlice = ITensor::slice(cacheIndirBatchedCopySizes, 0, numGenerationRequests);
    runtime::kernels::invokeCopyBatch(*decoderCacheIndirectionOutput, *cacheIndirection, *batchedCopySrcOffsetsSlice,
        *batchedCopyDstOffsetsSlice, *batchedCopySizesSlice, copySize, stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::copyCrossAttentionMasks(RequestVector const& contextRequests, RequestVector const& genRequests,
    TensorPtr const& decoderContextLengthsDevice, TensorPtr const& encoderInputLengths,
    SizeType32 maxDecoderContextLength, SizeType32 maxEncoderInputLengthInBatch, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& manager = runtime.getBufferManager();

    // Reshape the tensor to make sure the dim1 matches maxEncoderInputLengthInBatch.
    auto crossAttentionMaskShape = crossAttentionMaskDevice->getShape();
    crossAttentionMaskShape.d[1] = maxEncoderInputLengthInBatch;
    crossAttentionMaskDevice->reshape(crossAttentionMaskShape);
    // Set crossAttentionMask to true by default if it is not provided.
    manager.setMem(*crossAttentionMaskDevice, 1);

    // Check if all context requests have cross attention mask.
    bool allContextCrossAttentionMaskProvided = true;
    for (auto const& llmReq : contextRequests)
    {
        auto const& crossAttentionMaskRequest = llmReq->getCrossAttentionMask();
        if (bufferCastOrNull<bool>(crossAttentionMaskRequest) == nullptr)
        {
            allContextCrossAttentionMaskProvided = false;
            break;
        }
    }
    // If not all requests have cross attention mask, let us create the default ones.
    auto const& stream = runtime.getStream();
    if (!allContextCrossAttentionMaskProvided)
    {
        TLLM_LOG_WARNING("Default padding attention mask will be used as not all requests have cross attention mask.");
        tk::AttentionMaskParams<bool> attentionMaskParams;
        memset((void*) &attentionMaskParams, 0, sizeof(attentionMaskParams));
        // Set parameters.
        attentionMaskParams.mask = bufferCastOrNull<bool>(crossAttentionMaskDevice);
        attentionMaskParams.cuQSeqLens = bufferCastOrNull<SizeType32>(crossAttentionCuQSeqLensDevice);
        attentionMaskParams.actualQSeqLens = bufferCastOrNull<SizeType32>(decoderContextLengthsDevice);
        attentionMaskParams.actualKvSeqLens = bufferCastOrNull<SizeType32>(encoderInputLengths);
        attentionMaskParams.attentionMaskType = tk::AttentionMaskType::PADDING;
        attentionMaskParams.batchSize = static_cast<SizeType32>(contextRequests.size());
        attentionMaskParams.maxQSeqLen = maxDecoderContextLength;
        attentionMaskParams.maxKvSeqLen = maxEncoderInputLengthInBatch;
        // Launch the kernel.
        tk::invokeBuildAttentionMask(attentionMaskParams, stream.get());
        sync_check_cuda_error(stream.get());
    }
    // Use the first request's cross attention mask tensor's pointer address as the primary source pointer.
    auto const& attentionMaskSrc = !contextRequests.empty() ? contextRequests[0]->getCrossAttentionMask()
                                                            : genRequests[0]->getCrossAttentionMask();
    bool const* primarySrcPtr = bufferCastOrNull<bool>(attentionMaskSrc);

    // Pinned-memory buffer preparation for batch copy.
    auto batchedCopySrcOffsets = BufferRange<SizeType64>(*crossAttentionMaskCopySrcOffsets);
    auto batchedCopyDstOffsets = BufferRange<SizeType64>(*crossAttentionMaskCopyDstOffsets);
    auto batchedCopySizes = BufferRange<SizeType64>(*crossAttentionMaskCopySizes);
    // Requests with cross-attention-mask don't need to copy.
    manager.setZero(*crossAttentionMaskCopySizes);
    sync_check_cuda_error(stream.get());

    SizeType32 numTokens = 0;
    SizeType32 numCopiedTokens = 0;
    bool* pinnedMemPtr = bufferCastOrNull<bool>(crossAttentionMaskPinnedHost);
    for (auto const& llmReq : contextRequests)
    {
        auto const& crossAttentionMaskRequest = llmReq->getCrossAttentionMask();
        auto const position = llmReq->getContextCurrentPosition();
        auto const size = llmReq->getContextChunkSize();
        if (bufferCastOrNull<bool>(crossAttentionMaskRequest) != nullptr)
        {
            auto memType = crossAttentionMaskRequest->getMemoryType();
            auto const crossAttentionMaskRequestDim0
                = static_cast<SizeType64>(crossAttentionMaskRequest->getShape().d[0]);
            auto const crossAttentionMaskRequestDim1
                = static_cast<SizeType64>(crossAttentionMaskRequest->getShape().d[1]);
            TLLM_LOG_DEBUG("copyCrossAttentionMasks (shape [%d, %d]) from contextRequests position %d chunkSize %d",
                crossAttentionMaskRequestDim0, crossAttentionMaskRequestDim1, position, size);
            if ((position + size - 1) >= crossAttentionMaskRequestDim0)
            {
                TLLM_LOG_WARNING(
                    "The provided crossAttentionMask input is not complete for context phases, the last row "
                    "will be "
                    "used by default.");
            }
            // copy it to pinned memory if it is a cpu tensor.
            if (memType == MemoryType::kCPU)
            {
                TLLM_LOG_DEBUG("CrossAttentionMask tensor is on CPU.");
                auto const copiedPosition
                    = std::min(crossAttentionMaskRequestDim0 - 1, static_cast<SizeType64>(position));
                auto const copiedSize
                    = std::min(crossAttentionMaskRequestDim0 - copiedPosition, static_cast<SizeType64>(size));
                SizeType64 inputMaskOffset = (copiedPosition * crossAttentionMaskRequestDim1);
                SizeType64 inputMaskSize = (copiedSize * crossAttentionMaskRequestDim1);
                std::memcpy(
                    pinnedMemPtr, bufferCastOrNull<bool>(crossAttentionMaskRequest) + inputMaskOffset, inputMaskSize);
                pinnedMemPtr += inputMaskSize;
                for (SizeType32 tokenId = position; tokenId < position + size; tokenId++)
                {
                    SizeType64 tokenIdInPinnedMem
                        = std::min(copiedSize - 1, static_cast<SizeType64>(tokenId - position));
                    batchedCopySrcOffsets.begin()[numCopiedTokens]
                        = (pinnedMemPtr - primarySrcPtr) + tokenIdInPinnedMem * crossAttentionMaskRequestDim1;
                    batchedCopyDstOffsets.begin()[numCopiedTokens]
                        = numTokens * static_cast<SizeType64>(maxEncoderInputLengthInBatch);
                    batchedCopySizes.begin()[numCopiedTokens] = crossAttentionMaskRequestDim1;
                    numCopiedTokens++;
                    numTokens++;
                }
            }
            else
            {
                TLLM_LOG_DEBUG("CrossAttentionMask tensor is on GPU.");
                for (SizeType32 tokenId = position; tokenId < position + size; tokenId++)
                {
                    batchedCopySrcOffsets.begin()[numCopiedTokens]
                        = static_cast<SizeType64>(bufferCastOrNull<bool>(crossAttentionMaskRequest) - primarySrcPtr)
                        + std::min(crossAttentionMaskRequestDim0 - 1, static_cast<SizeType64>(tokenId))
                            * crossAttentionMaskRequestDim1;
                    batchedCopyDstOffsets.begin()[numCopiedTokens]
                        = numTokens * static_cast<SizeType64>(maxEncoderInputLengthInBatch);
                    batchedCopySizes.begin()[numCopiedTokens] = crossAttentionMaskRequestDim1;
                    numCopiedTokens++;
                    numTokens++;
                }
            }
        }
        else
        {
            numTokens += size;
            TLLM_LOG_WARNING(
                "CrossAttentionMask is not provided for the request. Default padding attention mask will be "
                "created.");
        }
    }
    sync_check_cuda_error(stream.get());

    for (auto const& llmReq : genRequests)
    {
        auto const promptLen = llmReq->mPromptLen;
        auto const decodingIter = llmReq->getDecodingIter();
        auto const& crossAttentionMaskRequest = llmReq->getCrossAttentionMask();
        if (bufferCastOrNull<bool>(crossAttentionMaskRequest) != nullptr)
        {
            auto const memType = crossAttentionMaskRequest->getMemoryType();
            auto const crossAttentionMaskRequestDim0
                = static_cast<SizeType64>(crossAttentionMaskRequest->getShape().d[0]);
            auto const crossAttentionMaskRequestDim1
                = static_cast<SizeType64>(crossAttentionMaskRequest->getShape().d[1]);
            TLLM_LOG_DEBUG("copyCrossAttentionMasks (shape [%d, %d]) from genRequests decodingIter %d",
                crossAttentionMaskRequestDim0, crossAttentionMaskRequestDim1, decodingIter);
            if (promptLen + decodingIter - 1 >= crossAttentionMaskRequestDim0)
            {
                TLLM_LOG_WARNING(
                    "The provided crossAttentionMask input is not complete for generation phases, the last row "
                    "will be "
                    "used by default.");
            }
            // copy it to pinned memory if it is a cpu tensor.
            if (memType == MemoryType::kCPU)
            {
                TLLM_LOG_DEBUG("CrossAttentionMask tensor is on CPU.");
                SizeType64 copiedPosition = std::min(
                    crossAttentionMaskRequestDim0 - 1, static_cast<SizeType64>(promptLen + decodingIter - 1));
                SizeType64 inputMaskOffset = (copiedPosition * crossAttentionMaskRequestDim1);
                SizeType64 inputMaskSize = crossAttentionMaskRequestDim1;
                std::memcpy(
                    pinnedMemPtr, bufferCastOrNull<bool>(crossAttentionMaskRequest) + inputMaskOffset, inputMaskSize);
                pinnedMemPtr += inputMaskSize;
                batchedCopySrcOffsets.begin()[numCopiedTokens] = static_cast<SizeType64>(pinnedMemPtr - primarySrcPtr);
                batchedCopyDstOffsets.begin()[numCopiedTokens]
                    = numTokens * static_cast<SizeType64>(maxEncoderInputLengthInBatch);
                batchedCopySizes.begin()[numCopiedTokens] = crossAttentionMaskRequestDim1;
            }
            else
            {
                TLLM_LOG_DEBUG("CrossAttentionMask tensor is on GPU.");
                batchedCopySrcOffsets.begin()[numCopiedTokens]
                    = static_cast<SizeType64>(bufferCastOrNull<bool>(crossAttentionMaskRequest) - primarySrcPtr)
                    + std::min(crossAttentionMaskRequestDim0 - 1, static_cast<SizeType64>(promptLen + decodingIter - 1))
                        * crossAttentionMaskRequestDim1;
                batchedCopyDstOffsets.begin()[numCopiedTokens]
                    = numTokens * static_cast<SizeType64>(maxEncoderInputLengthInBatch);
                batchedCopySizes.begin()[numCopiedTokens] = crossAttentionMaskRequestDim1;
            }
            numCopiedTokens++;
            numTokens++;
        }
        else
        {
            numTokens++;
            TLLM_LOG_WARNING(
                "CrossAttentionMask is not provided for the generation request. Full valid attentionMask will "
                "be used "
                "by default.");
        }
    }
    sync_check_cuda_error(stream.get());

    // Copy all requests' attention mask in one kernel.
    if (attentionMaskSrc != nullptr)
    {
        crossAttentionMaskCopySrcOffsets->reshape(ITensor::makeShape({numCopiedTokens}));
        crossAttentionMaskCopyDstOffsets->reshape(ITensor::makeShape({numCopiedTokens}));
        crossAttentionMaskCopySizes->reshape(ITensor::makeShape({numCopiedTokens}));
        runtime::kernels::invokeCopyBatch(*attentionMaskSrc, *crossAttentionMaskDevice,
            *crossAttentionMaskCopySrcOffsets, *crossAttentionMaskCopyDstOffsets, *crossAttentionMaskCopySizes,
            maxEncoderInputLengthInBatch, stream);
    }
    sync_check_cuda_error(stream.get());

    // The packed mask is only needed by context requests now.
    if (!contextRequests.empty())
    {
        // Set the parameters for creating packed mask for context FMHA.
        tk::PackedMaskParams<bool> maskParams{};
        maskParams.maskInput = bufferCastOrNull<bool>(crossAttentionMaskDevice);
        maskParams.cuQSeqLens = bufferCastOrNull<SizeType32>(crossAttentionCuQSeqLensDevice);
        maskParams.packedMask = bufferCastOrNull<uint32_t>(crossAttentionPackedMaskDevice);
        maskParams.cuMaskRows = bufferCastOrNull<SizeType32>(crossAttentionPackedMaskCuMaskRowsDevice);
        maskParams.actualQSeqLens = bufferCastOrNull<SizeType32>(decoderContextLengthsDevice);
        maskParams.actualKvSeqLens = bufferCastOrNull<SizeType32>(encoderInputLengths);
        maskParams.batchSize = contextRequests.size();
        maskParams.maxQSeqLen = maxDecoderContextLength;
        maskParams.maxKvSeqLen = maxEncoderInputLengthInBatch;
        maskParams.attentionMaskType = tk::ContextAttentionMaskType::CUSTOM_MASK;
        maskParams.validPosVal = true;

        // Launch the pack mask kernel.
        tk::invokeBuildPackedMask(maskParams, stream.get());
        sync_check_cuda_error(stream.get());
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::copySkipCrossAttnBlocks(bool const& _skipCrossAttnBlocks, runtime::TllmRuntime const& runtime)
{
    auto const& manager = runtime.getBufferManager();
    manager.copy(&_skipCrossAttnBlocks, *skipCrossAttnBlocks);
}

} // namespace tensorrt_llm::batch_manager
