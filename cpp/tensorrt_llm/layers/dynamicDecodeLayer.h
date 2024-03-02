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

#pragma once

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/kernels/beamSearchTopkKernels.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/onlineBeamSearchLayer.h"
#include "tensorrt_llm/layers/samplingLayer.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/decodingMode.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
struct BeamHypotheses;
}

namespace layers
{

template <typename T>
class DynamicDecodeLayer : public BaseLayer
{
public:
    DynamicDecodeLayer(runtime::DecodingMode const& mode, size_t max_batch_size, size_t max_beam_width,
        size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream, std::shared_ptr<tc::IAllocator> allocator,
        cudaDeviceProp* cuda_device_prop);

    ~DynamicDecodeLayer() override;
    DynamicDecodeLayer(DynamicDecodeLayer const& dynamic_decode_layer);

    class SetupParams
    {
    public:
        std::optional<std::vector<float>> temperature;        // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> repetition_penalty; // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> presence_penalty;   // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> frequency_penalty;  // [1] or [batch_size] on cpu
        std::optional<std::vector<std::int32_t>> min_length;  // [1] or [batch_size] on cpu

        // baseSamplingLayer
        std::optional<std::vector<std::uint32_t>> runtime_top_k; // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> runtime_top_p;         // [1] or [batch_size] on cpu
        std::optional<std::vector<uint64_t>> randomSeed;         // [1] or [batch_size] on cpu

        // topPSamplingLayer
        std::optional<std::vector<float>> top_p_decay;            // [batch_size], must between [0, 1]
        std::optional<std::vector<float>> top_p_min;              // [batch_size], must between [0, 1]
        std::optional<std::vector<std::int32_t>> top_p_reset_ids; // [batch_size]

        // omlineBeamSearchLayer
        std::optional<std::vector<float>> beam_search_diversity_rate;
        std::optional<std::vector<float>> length_penalty;
        std::optional<std::vector<int>> early_stopping;

        std::optional<bool> normalize_log_probs;
    };

    void setup(size_t batch_size, size_t beam_width, int const* batch_slots, SetupParams const& setupParams);

    class ForwardParams
    {
    public:
        ForwardParams(int step, int ite, int maxInputLength, int maxAttentionWindow, int sinkTokenLength,
            int localBatchSize, tc::Tensor endIds)
            : step{step}
            , ite{ite}
            , max_input_length{maxInputLength}
            , max_attention_window{maxAttentionWindow}
            , sink_token_length{sinkTokenLength}
            , local_batch_size{localBatchSize}
            , max_stop_words_len{0}
            , max_bad_words_len{0}
            , end_ids{std::move(endIds)}
        {
        }

        // mandatory parameters
        int step;
        int ite;
        int max_input_length;
        int max_attention_window;
        int sink_token_length;
        int local_batch_size;
        int max_stop_words_len;
        int max_bad_words_len;
        tc::Tensor end_ids; // [batch_size], on gpu

        // One of these two fields has to be set
        // DynamicDecodeLayer::forward checks for it
        // Need both of these fields to support legacy code during transition period to the batched decoder
        std::optional<tc::Tensor> logits;                  // [batch_size, beam_width, vocab_size_padded], on gpu
        std::optional<std::vector<tc::Tensor>> logits_vec; // [batch_size], on gpu

        // optional parameters
        std::optional<tc::Tensor> finished;              // [batch_size * beam_width], optional
        std::optional<tc::Tensor> src_cache_indirection; // [local_batch_size, beam_width, max_seq_len] - the k/v cache
                                                         // index for beam search, mandatory for beam search, on gpu
        std::optional<tc::Tensor> sequence_limit_length; // [batch_size], on gpu
        std::optional<tc::Tensor> embedding_bias;        // [vocab_size_padded], on gpu
        std::optional<tc::Tensor> input_lengths;         // [batch_size, beam_width], on gpu
        std::optional<tc::Tensor> bad_words_ptr; // [2, bad_words_length] or [batch_size, 2, bad_words_length], on gpu
        std::optional<tc::Tensor> bad_words_lengths;    // [batch_size], on gpu
        std::optional<tc::Tensor> stop_words_ptr;       // [batch_size][2, stop_words_length], on gpu
        std::optional<tc::Tensor> stop_words_lengths;   // [batch_size], on gpu
        std::optional<tc::Tensor> no_repeat_ngram_size; // [batch_size], optional
        std::optional<tc::Tensor> batch_slots;          // [batch_size], optional, in pinned memory
    };

    class OutputParams
    {
    public:
        explicit OutputParams(tc::Tensor outputIds)
            : output_ids{std::move(outputIds)}
        {
        }

        // mandatory parameters
        tc::Tensor output_ids; // [batch_size, beam_width. max_seq_len]
        tc::Tensor newTokens;  // [batch_size, beam_width]
        // optional parameters
        std::optional<tc::Tensor> finished;        // [batch_size * beam_width], optional
        std::optional<tc::Tensor> finished_sum;    // [1], optional, in pinned host memory
        std::optional<tc::Tensor> cum_log_probs;   // [batch_size * beam_width], necessary in beam search
        std::optional<tc::Tensor> parent_ids;      // [max_seq_len, batch_size * beam_width], necessary in beam search
        std::optional<tc::Tensor> sequence_length; // [batch_size * beam_width], optional
        std::optional<tc::Tensor>
            output_log_probs_tiled; // [request_output_length, batch_size, beam_width], must be float*, optional
        std::optional<tc::Tensor>
            output_log_probs;       // [batch_size, beam_width, request_output_length], must be float*, optional
        std::optional<tc::Tensor>
            tgt_cache_indirection;  // [local_batch_size, beam_width, max_seq_len], the k/v cache index for beam search
        std::shared_ptr<kernels::BeamHypotheses>
            beamHypotheses;         // a special structure which maintains some pointers of beam search

        tc::Tensor output_ids_ptr;  // [batch_size] int* (2-d array), each int* has [beam_width, max_seq_len]
        tc::Tensor parent_ids_ptr;  // [batch_size] int* (2-d array), each int* has [beam_width, max_seq_len]
    };

    void forward(OutputParams& outputs, ForwardParams const& params);
    void allocateBuffer();
    void freeBuffer();

    T* getRuntimeLogitsDevice()
    {
        return mRuntimeLogitsDevice;
    }

private:
    void initialize();
    void initializeLayers();

    void setupLayers(size_t batchSize, size_t beamWidth, int32_t const* batchSlots, SetupParams const& setupParams);
    void setupPenalties(size_t batchSize, int32_t const* batchSlots, SetupParams const& setupParams);

    void layersForward(tc::Tensor& logits, OutputParams& outputs, ForwardParams const& params,
        int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen);

    void applyPenalties(OutputParams& outputs, ForwardParams const& params, int32_t const* batchSlotsHost,
        int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen);

    static void banWords(tc::Tensor& logits, OutputParams& outputs, ForwardParams const& params,
        int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen, size_t vocabSizePadded,
        cudaStream_t stream);
    static void banRepeatNGrams(tc::Tensor& logits, OutputParams& outputs, ForwardParams const& params,
        int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen, size_t vocabSizePadded,
        cudaStream_t stream);
    static void banBadWords(tc::Tensor& logits, OutputParams& outputs, ForwardParams const& params,
        int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen, size_t vocabSizePadded,
        cudaStream_t stream);

    static void checkStopCriteria(OutputParams& outputs, ForwardParams const& params, int32_t const* batchSlots,
        size_t batchSize, size_t beamWidth, size_t maxSeqLen, cudaStream_t stream);
    static void checkMaxLengthStopCriteria(OutputParams& outputs, ForwardParams const& params,
        int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen, cudaStream_t stream);
    static void checkStopWordsStopCriteria(OutputParams& outputs, ForwardParams const& params,
        int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen, cudaStream_t stream);

    void prepareIdsPtrs(
        OutputParams& outputs, int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen);
    static void prepareOutputData(OutputParams& outputs, ForwardParams const& params,
        runtime::ITensor::SharedPtr const& idsPtrsHost, int32_t const* batchSlots, size_t batchSize,
        size_t maxBatchSize, size_t beamWidth, size_t maxSeqLen, int32_t cyclicStep, cudaStream_t stream);

private:
    std::unique_ptr<OnlineBeamSearchLayer<T>> mOnlineBeamSearchDecode;
    std::unique_ptr<SamplingLayer<T>> mSamplingLayer;

    runtime::DecodingMode mDecodingMode;
    size_t mMaxBatchSize;
    size_t mMaxBeamWidth;
    size_t mVocabSize;
    size_t mVocabSizePadded;

    cudaDeviceProp* mCudaDeviceProp;

    int32_t* mZeroParentIdsDevice = nullptr;
    int32_t* mPenaltyWorkspaceDevice = nullptr;
    int32_t* mPenaltyWorkspacePrevDevice = nullptr;
    runtime::ITensor::SharedPtr mIdsPtrHost;
    runtime::ITensor::SharedPtr mLogitsPtrsHost;

    float* mTemperatureDevice = nullptr;
    float* mRepetitionPenaltyDevice = nullptr;
    float* mPresencePenaltyDevice = nullptr;
    float* mFrequencyPenaltyDevice = nullptr;
    int32_t* mMinLengthDevice = nullptr;
    T* mRuntimeLogitsDevice = nullptr;

    std::vector<float> mTemperature;
    std::vector<float> mRepetitionPenalty;
    std::vector<float> mPresencePenalty;
    std::vector<float> mFrequencyPenalty;
    std::vector<int32_t> mMinLength;

    bool mUseTemperature = false;
    bool mUseRepetitionPenalty = false;
    bool mUsePresencePenalty = false;
    bool mUseFrequencyPenalty = false;
    bool mUseMinLength = false;

    bool mHasDiffRuntimeArgs = false;
    int* h_pinned_finished_sum_ = nullptr;

    int32_t mCyclicStep = 0;
    int32_t mRuntimeMaxSeqLen = 0;
    int32_t mConfiguredBeamWidth = -1;
};

} // namespace layers
} // namespace tensorrt_llm
