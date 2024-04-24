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

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include "tensorrt_llm/layers/beamSearchLayer.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/fillBuffers.h"
#include <limits>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{

template <typename T>
BeamSearchLayer<T>::BeamSearchLayer(runtime::SizeType vocab_size, runtime::SizeType vocab_size_padded,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
    : BaseLayer(stream, std::move(allocator), nullptr)
    , mVocabSize(vocab_size)
    , mVocabSizePadded(vocab_size_padded)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
}

template <typename T>
BeamSearchLayer<T>::BeamSearchLayer(BeamSearchLayer<T> const& beam_search_layer)
    : BaseLayer(beam_search_layer)
    , mVocabSize(beam_search_layer.mVocabSize)
    , mVocabSizePadded(beam_search_layer.mVocabSizePadded)
    , mWorkspaceSize(beam_search_layer.mWorkspaceSize)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
}

template <typename T>
BeamSearchLayer<T>::~BeamSearchLayer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
}

template <typename T>
void BeamSearchLayer<T>::setup(
    runtime::SizeType const batch_size, runtime::SizeType const beam_width, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(
        beam_width <= nMaxBeamWidth, std::string("Beam width is larger than the maximum supported (64)."));

    mDiversityRateHost.resize(batch_size);
    mLengthPenaltyHost.resize(batch_size);
    mEarlyStoppingHost.resize(batch_size);
    allocateBuffer(batch_size, beam_width);

    auto constexpr fltMax = std::numeric_limits<float>::max();
    auto constexpr fltMin = std::numeric_limits<float>::lowest();
    auto constexpr fltEpsilon = std::numeric_limits<float>::epsilon();

    FillBuffers const fillBuffers{batch_size, batch_size, mStream};
    fillBuffers(setupParams.beam_search_diversity_rate, DefaultDecodingParams::getBeamSearchDiversity(),
        mDiversityRateHost, mDiversityRateDevice, (int*) nullptr, std::make_pair(-fltEpsilon, fltMax),
        "diveristy rate");
    fillBuffers(setupParams.length_penalty, DefaultDecodingParams::getLengthPenalty(), mLengthPenaltyHost,
        mLengthPenaltyDevice, (int*) nullptr, std::make_pair(fltMin, fltMax), "length penalty");
    fillBuffers(setupParams.early_stopping, DefaultDecodingParams::getEarlyStopping(), mEarlyStoppingHost,
        mEarlyStoppingDevice, (int*) nullptr, std::make_pair(fltMin, fltMax), "early stopping");
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

__global__ void updateIndirCacheKernel(int* tgt_indir_cache, int const* src_indir_cache, BeamHypotheses bh,
    int const* inputLengths, int max_attention_window, int sink_token_length)
{
    // Update indirections from steps `nInputLength[nBBId]` to step `sequence_lengths[nBBId]`
    int const time_step = threadIdx.x + blockIdx.x * blockDim.x;
    int const nBBId = blockIdx.y;
    int const nBM{bh.nBeamWidth};
    int const batch_id = nBBId / nBM;
    int const beam_id = nBBId % nBM;
    int const current_step{bh.sequenceLengths[nBBId] - 1}; // the sequence_lengths is updated, need to minus 1
    int const nInputLength{inputLengths == nullptr ? 0 : inputLengths[nBBId]};

    // Return early when the nBBId or timestep is out of the bound
    // No update for the indices of context part since KV Cache is shared and fixed for context part
    if (nBBId >= nBM * bh.nBatchSizeLocal || time_step >= bh.nMaxSeqLen || time_step < nInputLength
        || time_step < (bh.nMaxSeqLen - max_attention_window) || bh.finished[nBBId].isFinished())
    {
        return;
    }
    int time_step_circ = time_step;
    if (time_step_circ >= sink_token_length)
    {
        time_step_circ
            = sink_token_length + (time_step - sink_token_length) % (max_attention_window - sink_token_length);
    }

    // for the parent_ids, we will still keep it for all past tokens (i.e. bh.nMaxSeqLen)
    int const src_beam = bh.parentIdsPtr[batch_id][beam_id * bh.nMaxSeqLen + current_step];

    // for the indir tables, we have the cyclic kv cache.
    uint32_t const tgt_offset = batch_id * nBM * max_attention_window + beam_id * max_attention_window + time_step_circ;
    uint32_t const src_offset
        = batch_id * nBM * max_attention_window + src_beam * max_attention_window + time_step_circ;

    tgt_indir_cache[tgt_offset] = (time_step == current_step) ? beam_id : src_indir_cache[src_offset];
}

void updateIndirCacheKernelLauncher(int* tgt_cache_indirection, int const* src_cache_indirection, BeamHypotheses& bh,
    int const* inputLengths, int max_attention_window, int sink_token_length, cudaStream_t stream)
{
    int const max_seq_len_aligned = (bh.nMaxSeqLen + 31) / 32;
    dim3 const grid(max_seq_len_aligned, bh.nBatchSizeLocal * bh.nBeamWidth);
    updateIndirCacheKernel<<<grid, 32, 0, stream>>>(
        tgt_cache_indirection, src_cache_indirection, bh, inputLengths, max_attention_window, sink_token_length);
}

template <typename T>
void BeamSearchLayer<T>::forward(OutputParams& op, ForwardParams const& fp)
{
    TLLM_LOG_TRACE("%s", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(op.beamHypotheses, std::string("Output BeamHypotheses is not set."));
    TLLM_CHECK_WITH_INFO(op.sequence_length->template getPtr<int>() != nullptr || mLengthPenaltyDevice == nullptr,
        std::string("Current sequence lengths must be set for length penalty computation."));
    TLLM_CHECK_WITH_INFO(fp.ite == 0, "Pipeline Parallelism is not supported yet !");

    BeamHypotheses& bh{*op.beamHypotheses};
    bh.nBatchSize = static_cast<std::int32_t>(op.output_ids_ptr.shape[0]);
    bh.nBeamWidth = static_cast<std::int32_t>(op.output_ids_ptr.shape[1]);
    bh.nIte = fp.ite;
    bh.nBatchSizeLocal = fp.logits.shape[0];
    bh.nMaxSeqLen = static_cast<std::int32_t>(op.output_ids_ptr.shape[2]);
    bh.nVocabSize = mVocabSizePadded;
    bh.diversityRates = mDiversityRateDevice;
    bh.lengthPenalties = mLengthPenaltyDevice;
    bh.earlyStoppings = mEarlyStoppingDevice;
    // bh.inputLengths = (fp.input_lengths) ? fp.input_lengths->template getPtr<int const>() : nullptr;
    // TODO: unify the assignment of inputLengths
    bh.endIds = fp.end_ids.template getPtr<int const>();
    bh.logProbs = (op.output_log_probs) ? op.output_log_probs->template getPtr<float>() : nullptr;
    // TODO (wili): here is a error in C++ workflow
    // In Python workflow, `op.output_log_probs` here is assigned by `outputs.output_log_probs_tiled` [MSL, BS, BM]
    // (function layersForward in file cpp/tensorrt_llm/layers/dynamicDecodeLayer.cpp)
    // But in C++ workflow, `op.output_log_probs` here is assigned by `output.logProbs` [BS, BM, MSL]
    // (function prepareOutputs in file cpp/tensorrt_llm/runtime/gptDecoder.cpp)
    bh.sequenceLengths = op.sequence_length->template getPtr<int>();
    bh.cumLogProbs = op.cum_log_probs->template getPtr<float>();
    bh.finished = reinterpret_cast<FinishedState*>(op.finished->template getPtr<FinishedState::UnderlyingType>());
    bh.outputIdsPtr = op.output_ids_ptr.template getPtr<int*>();
    bh.parentIdsPtr = op.parent_ids_ptr.template getPtr<int*>();

    T const* logits = fp.logits.template getPtr<T>();
    T const* bias = static_cast<T const*>(nullptr);
    TLLM_CHECK_WITH_INFO(mWorkspaceSize >= 2 * bh.nBatchSize * bh.nBeamWidth * bh.nBeamWidth * 2,
        std::string("Workspace size is not enough for topk softmax."));

    invokeTopkSoftMax(logits, bias, mWorkspace, bh, mStream);
    sync_check_cuda_error();

    if (bh.nBeamWidth > 1)
    {
        auto* const inputLengths = fp.input_lengths ? fp.input_lengths->template getPtr<int const>() : nullptr;
        auto tgt_ci = op.tgt_cache_indirection.template getPtr<int>();
        auto src_ci = fp.src_cache_indirection.template getPtr<int const>();

        updateIndirCacheKernelLauncher(
            tgt_ci, src_ci, bh, inputLengths, fp.max_attention_window, fp.sink_token_length, mStream);
        sync_check_cuda_error();
    }
}

template <typename T>
void BeamSearchLayer<T>::allocateBuffer(runtime::SizeType const batch_size, runtime::SizeType const beam_width)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    int const nPadBeamWidth = padToNextPowerOfTwo(beam_width);
    // Unit of mWorkspaceSize is number of elements (not Byte), align to 4 for further optimization
    size_t nTopK = batch_size * nPadBeamWidth * nPadBeamWidth * 2;
    size_t nTempBuffer = batch_size * nPadBeamWidth * nMaxVocabPartForStage1FastKernel * (2 * (nPadBeamWidth * 2) + 2);
    mWorkspaceSize = roundUp(nTopK, 4) * 2 + roundUp(nTempBuffer, 4);
    mWorkspace = mAllocator->reMalloc(mWorkspace, sizeof(float) * mWorkspaceSize, true);
    mDiversityRateDevice = mAllocator->reMalloc(mDiversityRateDevice, sizeof(float) * batch_size, false);
    mLengthPenaltyDevice = mAllocator->reMalloc(mLengthPenaltyDevice, sizeof(float) * batch_size, false);
    mEarlyStoppingDevice = mAllocator->reMalloc(mEarlyStoppingDevice, sizeof(int) * batch_size, false);
    mIsAllocateBuffer = true;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BeamSearchLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (mIsAllocateBuffer)
    {
        mAllocator->free((void**) (&mWorkspace));
        mAllocator->free((void**) (&mDiversityRateDevice));
        mAllocator->free((void**) (&mLengthPenaltyDevice));
        mAllocator->free((void**) (&mEarlyStoppingDevice));
        mIsAllocateBuffer = false;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class BeamSearchLayer<float>;
template class BeamSearchLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
