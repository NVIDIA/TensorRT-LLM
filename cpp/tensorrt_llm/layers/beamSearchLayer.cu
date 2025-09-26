/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "beamSearchLayer.h"
#include "tensorrt_llm/kernels/beamSearchKernels/beamSearchKernelsTemplate.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"

#include <limits>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm::layers
{

#define GET_INFO_STAGE1(paddedBeamWidth)                                                                               \
    {                                                                                                                  \
        int constexpr nBlock = (paddedBeamWidth < 16) ? ((paddedBeamWidth < 8) ? kThreadForSmallBeamWidth : 128) : 64; \
        TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                 \
            &nMaxActiveBlock, beamStage1Kernel<T, 2 * paddedBeamWidth, nBlock>, nBlock, 0));                           \
        TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage1Kernel<T, 2 * paddedBeamWidth, nBlock>));               \
        break;                                                                                                         \
    }

#define GET_INFO_STAGE2(paddedBeamWidth)                                                                               \
    {                                                                                                                  \
        if (nByteDynamicSharedMemoryStage2 > nByteMaxSharedMemoryPerBlock)                                             \
        {                                                                                                              \
            TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, paddedBeamWidth, 128, false>));           \
        }                                                                                                              \
        else if (nVPart <= 32)                                                                                         \
        {                                                                                                              \
            TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, paddedBeamWidth, 32, true>));             \
        }                                                                                                              \
        else if (nVPart <= 64)                                                                                         \
        {                                                                                                              \
            TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, paddedBeamWidth, 64, true>));             \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, paddedBeamWidth, 128, true>));            \
        }                                                                                                              \
        break;                                                                                                         \
    }

#define GET_INFO_STAGE3(paddedBeamWidth, isV2)                                                                         \
    {                                                                                                                  \
        int constexpr nThreadStage3 = (paddedBeamWidth + 31) / 32 * 32;                                                \
        TLLM_CUDA_CHECK(                                                                                               \
            cudaFuncGetAttributes(&attr, beamStage3Kernel<T, paddedBeamWidth, nThreadStage3, true, isV2>));            \
        break;                                                                                                         \
    }

template <typename T>
BeamSearchLayer<T>::BeamSearchLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 const batchSize{mDecoderDomain.getBatchSize()};
    SizeType32 const beamWidth{mDecoderDomain.getBeamWidth()};
    SizeType32 const vocabSize{mDecoderDomain.getVocabSize()};
    TLLM_CHECK_WITH_INFO(beamWidth <= kMaxBeamWidth, "Beam width is larger than the maximum supported (%d > %d)",
        int(beamWidth), int(kMaxBeamWidth));
    this->mVBWS = mode.isUseVariableBeamWidthSearch();

    allocateBuffer();
    configureBeamSearchLayer();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BeamSearchLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 const batchSize{mDecoderDomain.getBatchSize()};
    auto const batchSizeShape{ITensor::makeShape({batchSize})};
    auto const batchSizeXBeamWidthArraySizeShape{
        ITensor::makeShape({batchSize * static_cast<SizeType32>(kMaxBeamWidthArrayLength)})};

    mBeamSearchDiversityRateHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mBeamSearchDiversityRateDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);

    mLengthPenaltyHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mLengthPenaltyDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);

    mEarlyStoppingHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<int>::value);
    mEarlyStoppingDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<int>::value);

    if (this->mVBWS)
    {
        mBeamWidthArrayHost = mBufferManager->pinnedPool(batchSizeXBeamWidthArraySizeShape, TRTDataType<int>::value);
        mBeamWidthArrayDevice = mBufferManager->gpu(batchSizeXBeamWidthArraySizeShape, TRTDataType<int>::value);

        mBeamWidthIn = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<int>::value);
        mBeamWidthOut = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<int>::value);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BeamSearchLayer<T>::configureBeamSearchLayer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 const batchSize{mDecoderDomain.getBatchSize()};
    SizeType32 const beamWidth{mDecoderDomain.getBeamWidth()};
    SizeType32 const vocabSize{mDecoderDomain.getVocabSize()};
    SizeType32 const paddedBeamWidth{padToNextPowerOfTwo(beamWidth)};
    cudaFuncAttributes attr;

    // Find device information to determine `nVPart`.
    int const nByteMaxSharedMemoryPerSM = getMaxSharedMemoryPerSM();
    int const nByteMaxSharedMemoryPerBlock = getMaxSharedMemoryPerBlockOptin();
    int const nByteReservedSharedMemoryPerBlock = nByteMaxSharedMemoryPerSM - nByteMaxSharedMemoryPerBlock;
    this->mByteMaxSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock;

    if (beamWidth <= kMaxBeamWidthForV1 && !(this->mVBWS))
    {
        // V1 workflow for small beam width and non-VBWS
        // Stage 1
        int nMaxActiveBlock = -1;
        switch (paddedBeamWidth)
        {
        case 1: GET_INFO_STAGE1(1);
        case 2: GET_INFO_STAGE1(2);
        case 4: GET_INFO_STAGE1(4);
        case 8: GET_INFO_STAGE1(8);
        default: break;
        }
        int nByteStaticSharedMemory = attr.sharedSizeBytes;
        int nByteMaxDynamicSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock - nByteStaticSharedMemory;
        // Find the maximum of `nBlock` (maximum of `nVPart`, minimum of `nByteDynamicSharedMemoryStage1`), s.t.
        // `nVPart <= kMaxVPartStage1 && nByteDynamicSharedMemoryStage1 * nVPart >= sizeof(T) * vocabSize`
        TLLM_CHECK_WITH_INFO(nByteMaxDynamicSharedMemoryPerBlock * kMaxVPartStage1 >= sizeof(T) * vocabSize,
            "vocab_size is too large for Beam search.");
        int nByteExtralSharedMemory = nByteReservedSharedMemoryPerBlock + nByteStaticSharedMemory;
        int nBlock = nMaxActiveBlock;
        int nVPart = kMaxVPartStage1 + 1;
        for (; nBlock > 0 && nVPart > kMaxVPartStage1; --nBlock)
        {
            int nByteDynamicSharedMemoryStage1 = nByteMaxSharedMemoryPerSM / nBlock - nByteExtralSharedMemory;
            nByteDynamicSharedMemoryStage1 -= nByteDynamicSharedMemoryStage1 % sizeof(T);
            nVPart = ceilDiv(sizeof(T) * vocabSize, nByteDynamicSharedMemoryStage1);
        }
        TLLM_CHECK_WITH_INFO(nBlock >= 0, "No enough active blocks for Beam Search stage 1 kernel.");

        int const nByteDynamicSharedMemoryStage1 = sizeof(T) * ceilDiv(vocabSize, nVPart);
        this->mVPart = nVPart;
        this->mByteSharedMemoryStage1 = nByteDynamicSharedMemoryStage1; // Only dynamic shared memory

        // Stage 2
        TLLM_CHECK_WITH_INFO(batchSize * beamWidth * paddedBeamWidth < (1 << 21),
            "max_batch_size or max_beam_width of TRT-LLM engine is too large for Beam search, try to decrease the "
            "parameters while building.");
        size_t const nByteDynamicSharedMemoryStage2 = common::roundUp(
            sizeof(float) * nVPart * (paddedBeamWidth * 4) + sizeof(cub::KeyValuePair<int, T>) * paddedBeamWidth * 2,
            4);
        switch (paddedBeamWidth)
        {
        case 1: GET_INFO_STAGE2(1);
        case 2: GET_INFO_STAGE2(2);
        case 4: GET_INFO_STAGE2(4);
        case 8: GET_INFO_STAGE2(8);
        default: break;
        }
        nByteStaticSharedMemory = attr.sharedSizeBytes;
        nByteMaxDynamicSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock - nByteStaticSharedMemory;
        nByteExtralSharedMemory = nByteReservedSharedMemoryPerBlock + nByteStaticSharedMemory;
        bool const bUseGlobalMemoryStage2 = (nByteDynamicSharedMemoryStage2 > nByteMaxDynamicSharedMemoryPerBlock);

        // Stage 3
        // Keep top 2K candidates in case of k candidates finishes in one iteration
        size_t const nByteDynamicSharedMemoryStage3
            = common::roundUp(sizeof(T) * paddedBeamWidth * paddedBeamWidth * 2, 4);
        switch (paddedBeamWidth)
        {
        case 1: GET_INFO_STAGE3(1, false);
        case 2: GET_INFO_STAGE3(2, false);
        case 4: GET_INFO_STAGE3(4, false);
        case 8: GET_INFO_STAGE3(8, false);
        }
        nByteStaticSharedMemory = attr.sharedSizeBytes;
        nByteMaxDynamicSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock - nByteStaticSharedMemory;
        nByteExtralSharedMemory = nByteReservedSharedMemoryPerBlock + nByteStaticSharedMemory;
        bool const bUseGlobalMemoryStage3 = (nByteDynamicSharedMemoryStage3 > nByteMaxDynamicSharedMemoryPerBlock);
        this->mByteSharedMemoryStage3 = nByteStaticSharedMemory; // Only static shared memory

        // Compute workspace size, see `beamSearchKernelsTemplate.h` for detailed information
        // |<----- Workspace ----->|
        // |<- A ->|<- B ->|<- C ->|
        //         |<---- D ---->|
        // A for data exchange between stage 2 and 3
        // B for data exchange between stage 1 and 2, can be reuse for stage 3
        // C for stage 2 if `bUseGlobalMemoryStage2 == true`, can be reuse for stage 3
        // D for stage 3 if `bUseGlobalMemoryStage3 == true`
        size_t const nByteA = common::roundUp(sizeof(T) * batchSize * paddedBeamWidth * paddedBeamWidth * 4, 4);
        size_t const nByteB
            = common::roundUp(sizeof(T) * batchSize * paddedBeamWidth * kMaxVPartStage1 * paddedBeamWidth * 4, 4);
        size_t const nByteC = (bUseGlobalMemoryStage2) ? nByteDynamicSharedMemoryStage2 : 0;
        size_t const nByteD = (bUseGlobalMemoryStage3) ? nByteDynamicSharedMemoryStage3 : 0;
        this->mWorkspaceSize = nByteA + std::max(nByteB + nByteC, nByteD);
    }
    else // V2 workflow for large beam width or VBWS
    {
        this->mV2 = true;
        switch (paddedBeamWidth)
        {
        case 1: GET_INFO_STAGE3(1, true);
        case 2: GET_INFO_STAGE3(2, true);
        case 4: GET_INFO_STAGE3(4, true);
        case 8: GET_INFO_STAGE3(8, true);
        case 16: GET_INFO_STAGE3(16, true);
        case 32: GET_INFO_STAGE3(32, true);
        case 64: GET_INFO_STAGE3(64, true);
        case 128: GET_INFO_STAGE3(128, true);
        case 256: GET_INFO_STAGE3(256, true);
        case 512: GET_INFO_STAGE3(512, true);
        case 1024: GET_INFO_STAGE3(1024, true);
        }
        this->mByteSharedMemoryStage3 = attr.sharedSizeBytes; // Only static shared memory

        // Compute shared memory size for stage 3
        // Compute workspace size, see `beamSearchKernelsTemplate.h` for detailed information
        // |<----------------------------------------- Workspace ------------------------------------------>|
        // |<- Stage2Ids ->|<- Stage2LogProbs ->|<- Stage1Ids ->|<- Stage1LogProbs ->|<---- Stage1TopK ---->|
        //                                                                           |<- stage2TopK ->|
        //                                      |<------------------ Stage3 ------------------>|
        SizeType32 const batchSize{mDecoderDomain.getBatchSize()};
        SizeType32 const beamWidth{mDecoderDomain.getBeamWidth()};
        SizeType32 const vocabSize{mDecoderDomain.getVocabSize()};
        SizeType32 const paddedBeamWidth{padToNextPowerOfTwo(beamWidth)};
        size_t const nByteStage1LogProbs = roundUp(sizeof(T) * batchSize * paddedBeamWidth * paddedBeamWidth * 2, 4);
        size_t const nByteStage1Ids = roundUp(sizeof(int) * batchSize * paddedBeamWidth * paddedBeamWidth * 2, 4);
        size_t const nByteStage2LogProbs = roundUp(sizeof(T) * batchSize * paddedBeamWidth * 2, 4);
        size_t const nByteStage2Ids = roundUp(sizeof(int) * batchSize * paddedBeamWidth * 2, 4);
        size_t const nByteStage1TopK
            = invokeComputeTopkLastDimWorkspaceSize<T>(batchSize * beamWidth, vocabSize, paddedBeamWidth * 2, true);
        size_t const nByteStage2TopK = invokeComputeTopkLastDimWorkspaceSize<T>(
            batchSize, paddedBeamWidth * paddedBeamWidth * 2, beamWidth * 2, true);
        size_t const nByteStage3 = sizeof(T) * beamWidth * beamWidth * 2;
        this->mWorkspaceSize = nByteStage2LogProbs + nByteStage2Ids
            + max(nByteStage1LogProbs + nByteStage1Ids + max(nByteStage1TopK, nByteStage2TopK), nByteStage3);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t BeamSearchLayer<T>::getWorkspaceSize() const noexcept
{
    return mWorkspaceSize;
}

template <typename T>
void BeamSearchLayer<T>::setup(SizeType32 const batchSize, SizeType32 const beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(BeamSearchLayer_setup);

    SizeType32 const maxBamWidth{mDecoderDomain.getBeamWidth()};
    TLLM_CHECK_WITH_INFO(beamWidth <= maxBamWidth, "Beam width is larger than the constructed for (%d > %d).",
        int(beamWidth), int(maxBamWidth));

    auto setupParams = std::dynamic_pointer_cast<BeamSearchSetupParams>(baseSetupParams);
    auto constexpr fltMax = std::numeric_limits<float>::max();
    auto constexpr fltMin = std::numeric_limits<float>::lowest();
    auto constexpr fltEpsilon = std::numeric_limits<float>::epsilon();
    auto constexpr int32Max = std::numeric_limits<int32_t>::max();
    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};
    fillBuffers(setupParams->beamSearchDiversityRate, DefaultDecodingParams::getBeamSearchDiversity(),
        mBeamSearchDiversityRateHost, mBeamSearchDiversityRateDevice, batchSlots, std::make_pair(-fltEpsilon, fltMax),
        "diversity rate");
    fillBuffers(setupParams->lengthPenalty, DefaultDecodingParams::getLengthPenalty(), mLengthPenaltyHost,
        mLengthPenaltyDevice, batchSlots, std::make_pair(fltMin, fltMax), "length penalty");
    fillBuffers(setupParams->earlyStopping, DefaultDecodingParams::getEarlyStopping(), mEarlyStoppingHost,
        mEarlyStoppingDevice, batchSlots, std::make_pair(-fltEpsilon, int32Max), "early stopping");

    if (this->mVBWS)
    {
        fillBuffers(setupParams->beamWidthArray, DefaultDecodingParams::getBeamWidthArray(), mBeamWidthArrayHost,
            mBeamWidthArrayDevice, batchSlots, std::make_pair(-fltEpsilon, kMaxBeamWidth), "beam width array");
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BeamSearchLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(BeamSearchLayer_forwardAsync);

    auto ip = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);
    auto op = std::dynamic_pointer_cast<BeamSearchOutputs>(baseOutputs);
    auto const localDecoderDomain = getLocalDecoderDomain(ip, mDecoderDomain);

    TLLM_CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() > 1, "Use beamWidth <= 1 (%d <= 1) in Beam Search mode",
        localDecoderDomain.getBeamWidth());
    TLLM_CHECK_WITH_INFO(ip->srcCacheIndirection.has_value(), "srcCacheIndirection is mandatory in beam search.");
    TLLM_CHECK_WITH_INFO(op->parentIds.has_value(), "parentIds tensor is mandatory in beam search.");
    TLLM_CHECK_WITH_INFO(op->finished.has_value(), "finished tensor is mandatory in beam search.");
    TLLM_CHECK_WITH_INFO(op->cumLogProbs.has_value(), "cumLogProbs tensor is mandatory in beam search.");
    TLLM_CHECK_WITH_INFO(op->beamHypotheses, "Output BeamHypotheses is not set.");
    TLLM_CHECK_WITH_INFO(bufferCastOrNull<int>(*op->sequenceLength) != nullptr || mLengthPenaltyDevice == nullptr,
        "Current sequence lengths must be set for length penalty computation.");
    TLLM_CHECK_WITH_INFO(ip->ite == 0, "Pipeline Parallelism is not supported yet!");

    BeamHypotheses bh;
    // bh's members not used in this function: outputIds, logProbs, outputIdsUnfinish, parentIdsUnfinish
    bh.bVBWS = this->mVBWS;
    bh.nMaxBatchSize = static_cast<std::int32_t>(op->outputIdsPtr->getDimension<0>());
    bh.nBatchSize = ip->localBatchSize;
    bh.nBeamWidth = op->outputIds->getDimension<1>();
    bh.nMaxSeqLen = op->outputIds->getDimension<2>();
    bh.nVocabSize = mDecoderDomain.getVocabSizePadded();
    bh.nVPart = this->mVPart;
    bh.nByteMaxSharedMemoryPerBlock = this->mByteMaxSharedMemoryPerBlock;
    bh.nByteSharedMemoryStage1 = this->mByteSharedMemoryStage1;
    bh.nByteSharedMemoryStage3 = this->mByteSharedMemoryStage3;
    bh.diversityRates = bufferCast<float>(*mBeamSearchDiversityRateDevice);
    bh.lengthPenalties = bufferCast<float>(*mLengthPenaltyDevice);
    bh.earlyStoppings = bufferCast<int>(*mEarlyStoppingDevice);

    if (this->mVBWS)
    {
        bh.beamWidthArraysHost = bufferCast<int>(*mBeamWidthArrayHost);
        bh.nBeamWidthInHost = bufferCast<int>(*mBeamWidthIn);
        bh.nBeamWidthOutHost = bufferCast<int>(*mBeamWidthOut);
        int const* batchSlotsHost = bufferCast<int>(*ip->batchSlots);
        for (int i = 0; i < ip->localBatchSize; ++i)
        {
            auto const slot = batchSlotsHost[i];
            auto const step = ip->beamSearchSteps.value()[slot];
            // Clamp `step` to [0, kMaxBeamWidthArrayLength - 1], and set `indexInput=0` when step = 0 or 1
            auto const indexOutput = std::min(step, static_cast<SizeType32>(kMaxBeamWidthArrayLength) - 1);
            auto const indexInput = std::max(indexOutput - 1, 0);
            bh.nBeamWidthInHost[i] = bh.beamWidthArraysHost[slot * kMaxBeamWidthArrayLength + indexInput];
            bh.nBeamWidthOutHost[i] = bh.beamWidthArraysHost[slot * kMaxBeamWidthArrayLength + indexOutput];
        }
        // At present, all requests of a batch must have the same beam width in one generation step (or they will not
        // be batched together). So, the beam width of the first request is taken here to reshape the buffer.
        // Corresponding changes must be done if Diverse-Beam-Width-Search (DBWS, requests with diverse beam width in
        // a batch in one generation step) is supported in the future.
        op->beamWidth = bh.nBeamWidthOutHost[0];
    }
    else
    {
        op->beamWidth = bh.nBeamWidth;
    }

    bh.inputLengths = bufferCast<SizeType32>(*ip->inputLengths.value());
    bh.endIds = bufferCast<TokenIdType>(*ip->endIds);
    bh.batchSlots = workspace->getDeviceBatchSlotsPtr(); // Device copy of `ip->batchSlots`
    bh.logProbsTiled = bufferCastOrNull<float>(op->outputLogProbsTiled);
    bh.sequenceLengths = bufferCast<SizeType32>(*op->sequenceLength.value());
    bh.cumLogProbs = bufferCast<float>(*op->cumLogProbs.value());
    bh.outputIdsCBA = op->beamHypotheses->outputIdsCBA;
    bh.logProbsCBA = op->beamHypotheses->logProbsCBA;
    bh.sequenceLengthsCBA = op->beamHypotheses->sequenceLengthsCBA;
    bh.cumLogProbsCBA = op->beamHypotheses->cumLogProbsCBA;
    bh.normedScoresCBA = op->beamHypotheses->normedScoresCBA;
    bh.numBeamsCBA = op->beamHypotheses->numBeamsCBA;
    bh.minNormedScoresCBA = op->beamHypotheses->minNormedScoresCBA;
    bh.batchDones = op->beamHypotheses->batchDones;
    bh.finished = reinterpret_cast<FinishedState*>(bufferCast<FinishedState::UnderlyingType>(*op->finished.value()));
    bh.outputIdsPtr = bufferCast<TokenIdType*>(*op->outputIdsPtr);
    bh.parentIdsPtr = bufferCast<TokenIdType*>(*op->parentIdsPtr);

    T const* logProbs = bufferCast<T>(*workspace->getDeviceRuntimeLogits());
    T const* bias = static_cast<T const*>(nullptr);
    TLLM_CHECK_WITH_INFO(getWorkspaceSize() >= 2 * bh.nBatchSize * bh.nBeamWidth * bh.nBeamWidth * 2,
        "Workspace size (%lu) is not enough for topk softmax required (%lu).", (uint64_t) getWorkspaceSize(),
        (uint64_t) (2 * bh.nMaxBatchSize * bh.nBeamWidth * bh.nBeamWidth * 2));

    if (this->mV2 || this->mVBWS)
    {
        invokeTopkBeamSearch<T, true>(logProbs, bias, workspace->getRawWorkspaceDevicePtr(), bh, getStream());
    }
    else
    {
        invokeTopkBeamSearch<T, false>(logProbs, bias, workspace->getRawWorkspaceDevicePtr(), bh, getStream());
    }

    int* tgtCI = bufferCast<int>(*op->tgtCacheIndirection);
    int* srcCI = bufferCast<int>(*ip->srcCacheIndirection.value());
    invokeUpdateCacheIndirection(tgtCI, srcCI, bh, ip->maxAttentionWindow, ip->sinkTokenLength, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class BeamSearchLayer<float>;
template class BeamSearchLayer<half>;

} // namespace tensorrt_llm::layers
