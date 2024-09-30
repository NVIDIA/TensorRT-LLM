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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"
#include <limits>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm::layers
{

template <typename T>
size_t BeamSearchLayer<T>::getWorkspaceSize() const noexcept
{
    return mWorkspaceSize;
}

template <typename T>
BeamSearchLayer<T>::BeamSearchLayer(DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer(mDecoderDomain.getBatchSize(), mDecoderDomain.getBeamWidth());

    TLLM_CHECK_WITH_INFO(mDecoderDomain.getBeamWidth() <= nMaxBeamWidth,
        std::string("Beam width is larger than the maximum supported (" + std::to_string(mDecoderDomain.getBeamWidth())
            + " > " + std::to_string(nMaxBeamWidth) + ")."));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BeamSearchLayer<T>::setup(SizeType32 const batchSize, SizeType32 const beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(beamWidth <= mDecoderDomain.getBeamWidth(),
        std::string("Beam width is larger than the constructed for (" + std::to_string(beamWidth) + " > "
            + std::to_string(mDecoderDomain.getBeamWidth()) + ")."));

    auto setupParams = std::dynamic_pointer_cast<BeamSearchSetupParams>(baseSetupParams);

    auto constexpr fltMax = std::numeric_limits<float>::max();
    auto constexpr fltMin = std::numeric_limits<float>::lowest();
    auto constexpr fltEpsilon = std::numeric_limits<float>::epsilon();

    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};
    fillBuffers(setupParams->beamSearchDiversityRate, DefaultDecodingParams::getBeamSearchDiversity(),
        mBeamSearchDiversityRateHost, mBeamSearchDiversityRateDevice, batchSlots, std::make_pair(-fltEpsilon, fltMax),
        "diversity rate");
    fillBuffers(setupParams->lengthPenalty, DefaultDecodingParams::getLengthPenalty(), mLengthPenaltyHost,
        mLengthPenaltyDevice, batchSlots, std::make_pair(fltMin, fltMax), "length penalty");
    fillBuffers(setupParams->earlyStopping, DefaultDecodingParams::getEarlyStopping(), mEarlyStoppingHost,
        mEarlyStoppingDevice, batchSlots, std::make_pair(-fltEpsilon, std::numeric_limits<int>::max()),
        "early stopping");

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

__global__ void updateCacheIndirectionKernel(
    int* tgtCI, int const* srcCI, BeamHypotheses bh, int const nMaxAttentionWindow, int const nSinkTokenLength)
{
    // Update indirections from steps `bh.inputLength[indexBatchBeam]` to step `sequenceLengths[indexBatchBeam]`
    int const step = threadIdx.x + blockIdx.x * blockDim.x;
    int const nBM{bh.nBeamWidth};
    int const nMSL{bh.nMaxSeqLen};
    int const indexBatch = blockIdx.y;
    int const batchSlot = bh.batchSlots[indexBatch];
    int const indexBeam = blockIdx.z;
    int const indexBatchBeam = batchSlot * nBM + indexBeam;
    int const lastStep{bh.sequenceLengths[indexBatchBeam] - 1}; // the sequenceLengths is updated, need to minus 1

    // Return early when the indexBatchBeam or step is out of the bound
    // No update for the indices of context part since KV Cache is shared
    if (step >= nMSL || step < bh.inputLengths[indexBatchBeam] || step < (nMSL - nMaxAttentionWindow)
        || bh.finished[indexBatchBeam].isFinished())
    {
        return;
    }

    // Keep all past tokens by parentIdsPtr
    int const indexBeamSrc = bh.parentIdsPtr[batchSlot][indexBeam * nMSL + lastStep];
    int const stepCirc = (step >= nSinkTokenLength)
        ? nSinkTokenLength + (step - nSinkTokenLength) % (nMaxAttentionWindow - nSinkTokenLength)
        : step;
    // Consider cyclic kv cache for the indir tables
    uint32_t const tgtOffset = batchSlot * nBM * nMaxAttentionWindow + indexBeam * nMaxAttentionWindow + stepCirc;
    uint32_t const srcOffset = batchSlot * nBM * nMaxAttentionWindow + indexBeamSrc * nMaxAttentionWindow + stepCirc;
    tgtCI[tgtOffset] = (step == lastStep) ? indexBeam : srcCI[srcOffset];
}

template <typename T>
void BeamSearchLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto ip = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);
    auto op = std::dynamic_pointer_cast<BeamSearchOutputs>(baseOutputs);
    auto const localDecoderDomain = getLocalDecoderDomain(ip, mDecoderDomain);

    TLLM_CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() > 1,
        "Decoding mode is beam search, but beamWidth <= 1 (%d <= 1)", localDecoderDomain.getBeamWidth());
    TLLM_CHECK_WITH_INFO(ip->srcCacheIndirection.has_value(), "srcCacheIndirection is mandatory in beam search.");
    TLLM_CHECK_WITH_INFO(op->parentIds.has_value(), "parentIds tensor is mandatory in beam search.");
    TLLM_CHECK_WITH_INFO(op->finished.has_value(), "finished tensor is mandatory in beam search.");
    TLLM_CHECK_WITH_INFO(op->cumLogProbs.has_value(), "cumLogProbs tensor is mandatory in beam search.");
    TLLM_CHECK_WITH_INFO(op->beamHypotheses, std::string("Output BeamHypotheses is not set."));
    TLLM_CHECK_WITH_INFO(bufferCastOrNull<int>(*op->sequenceLength) != nullptr || mLengthPenaltyDevice == nullptr,
        std::string("Current sequence lengths must be set for length penalty computation."));
    TLLM_CHECK_WITH_INFO(ip->ite == 0, "Pipeline Parallelism is not supported yet !");

    BeamHypotheses bh;
    // bh's members not used in function: outputIds, logProbs, outputIdsUnfinish, parentIdsUnfinish
    bh.outputIdsCBA = op->beamHypotheses->outputIdsCBA;
    bh.logProbsCBA = op->beamHypotheses->logProbsCBA;
    bh.sequenceLengthsCBA = op->beamHypotheses->sequenceLengthsCBA;
    bh.cumLogProbsCBA = op->beamHypotheses->cumLogProbsCBA;
    bh.normedScoresCBA = op->beamHypotheses->normedScoresCBA;
    bh.numBeamsCBA = op->beamHypotheses->numBeamsCBA;
    bh.minNormedScoresCBA = op->beamHypotheses->minNormedScoresCBA;
    bh.batchDones = op->beamHypotheses->batchDones;
    bh.nMaxBatchSize = static_cast<std::int32_t>(op->outputIdsPtr->getDimension<0>());
    bh.nBatchSize = ip->localBatchSize;
    bh.batchSlots = workspace->getDeviceBatchSlotsPtr();
    bh.nBeamWidth = op->outputIds->getDimension<1>();
    bh.nMaxSeqLen = op->outputIds->getDimension<2>();
    bh.nVocabSize = mDecoderDomain.getVocabSizePadded();
    bh.diversityRates = bufferCast<float>(*mBeamSearchDiversityRateDevice);
    bh.lengthPenalties = bufferCast<float>(*mLengthPenaltyDevice);
    bh.earlyStoppings = bufferCast<int>(*mEarlyStoppingDevice);
    bh.inputLengths = bufferCast<SizeType32>(*ip->inputLengths.value());
    bh.endIds = bufferCast<TokenIdType>(*ip->endIds);
    bh.logProbsTiled = bufferCastOrNull<float>(op->outputLogProbsTiled);
    bh.sequenceLengths = bufferCast<SizeType32>(*op->sequenceLength.value());
    bh.cumLogProbs = bufferCast<float>(*op->cumLogProbs.value());
    bh.finished = reinterpret_cast<FinishedState*>(bufferCast<FinishedState::UnderlyingType>(*op->finished.value()));
    bh.outputIdsPtr = bufferCast<TokenIdType*>(*op->outputIdsPtr);
    bh.parentIdsPtr = bufferCast<TokenIdType*>(*op->parentIdsPtr);

    T const* logits = bufferCast<T>(*workspace->getDeviceRuntimeLogits());
    T const* bias = static_cast<T const*>(nullptr);
    TLLM_CHECK_WITH_INFO(getWorkspaceSize() >= 2 * bh.nBatchSize * bh.nBeamWidth * bh.nBeamWidth * 2,
        common::fmtstr("Workspace size (%lu) is not enough for topk softmax required (%lu).",
            (uint64_t) getWorkspaceSize(), (uint64_t) (2 * bh.nMaxBatchSize * bh.nBeamWidth * bh.nBeamWidth * 2)));

    invokeTopkSoftMax(logits, bias, workspace->getRawWorkspaceDevicePtr(), bh, getStream());
    sync_check_cuda_error();

    if (bh.nBeamWidth > 1)
    {
        auto tgtCI = bufferCast<int>(*op->tgtCacheIndirection);
        auto srcCI = bufferCast<int>(*ip->srcCacheIndirection.value());
        dim3 const grid(common::roundUp(bh.nMaxSeqLen, 32), bh.nBatchSize, bh.nBeamWidth);
        updateCacheIndirectionKernel<<<grid, 32, 0, getStream()>>>(
            tgtCI, srcCI, bh, ip->maxAttentionWindow, ip->sinkTokenLength);
        sync_check_cuda_error();
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BeamSearchLayer<T>::allocateBuffer(runtime::SizeType32 const batchSize, runtime::SizeType32 const beamWidth)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const nPadBeamWidth = padToNextPowerOfTwo(beamWidth);
    auto const nTopK = batchSize * nPadBeamWidth * nPadBeamWidth * 2;
    auto const nTempBuffer
        = batchSize * nPadBeamWidth * nMaxVocabPartForStage1FastKernel * (2 * (nPadBeamWidth * 2) + 2);
    auto const batchSizeShape = ITensor::makeShape({mDecoderDomain.getBatchSize()});
    mBeamSearchDiversityRateHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mLengthPenaltyHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mEarlyStoppingHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<int>::value);

    // Unit of workspaceSize is number of elements (not Byte), align to 4 for further optimization
    mWorkspaceSize = common::roundUp(nTopK, 4) * 2 + common::roundUp(nTempBuffer, 4);

    mBeamSearchDiversityRateDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mLengthPenaltyDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mEarlyStoppingDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<int>::value);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class BeamSearchLayer<float>;
template class BeamSearchLayer<half>;

} // namespace tensorrt_llm::layers
