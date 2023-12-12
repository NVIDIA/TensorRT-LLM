/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager
{

namespace inference_request
{
// Input tensors
auto constexpr kInputIdsTensorName = "input_ids";
auto constexpr kDraftInputIdsTensorName = "draft_input_ids";
auto constexpr kDraftLogitsTensorName = "draft_logits";
auto constexpr kMaxNewTokensTensorName = "request_output_len";
auto constexpr kBeamWidthTensorName = "beam_width";
auto constexpr kEndIdTensorName = "end_id";
auto constexpr kPadIdTensorName = "pad_id";
auto constexpr kBadWordsListTensorName = "bad_words_list";
auto constexpr kStopWordsListTensorName = "stop_words_list";
auto constexpr kEmbeddingBiasTensorName = "embedding_bias";
auto constexpr kTemperatureTensorName = "temperature";
auto constexpr kRuntimeTopKTensorName = "runtime_top_k";
auto constexpr kRuntimeTopPTensorName = "runtime_top_p";
auto constexpr kLengthPenaltyTensorName = "len_penalty";
auto constexpr kRepetitionPenaltyTensorName = "repetition_penalty";
auto constexpr kMinLengthTensorName = "min_length";
auto constexpr kPresencePenaltyTensorName = "presence_penalty";
auto constexpr kRandomSeedTensorName = "random_seed";
auto constexpr kReturnLogProbsTensorName = "return_log_probs";
auto constexpr kPromptEmbeddingTableName = "prompt_embedding_table";
auto constexpr kPromptVocabSizeName = "prompt_vocab_size";

// Obsolete names for backward compatibility
auto constexpr kInputLengthsTensorName = "input_lengths";

// Output tensors
auto constexpr kOutputIdsTensorName = "output_ids";
auto constexpr kSequenceLengthTensorName = "sequence_length";
auto constexpr kLogProbsTensorName = "output_log_probs";
auto constexpr kCumLogProbsTensorName = "cum_log_probs";
} // namespace inference_request

template <typename TTensor, typename TNamedTensor>
class GenericInferenceRequest
{
public:
    using TensorPtr = TTensor;
    using NamedTensorType = TNamedTensor;
    using TensorMap = std::unordered_map<std::string, TTensor>;

    explicit GenericInferenceRequest(uint64_t requestId)
        : mRequestId{requestId}
        , mIsStreaming{false}
    {
    }

    GenericInferenceRequest(uint64_t requestId, TensorMap&& tensorMap)
        : mRequestId{requestId}
        , mIsStreaming{false}
        , mInputTensors{std::move(tensorMap)}
    {
        for (auto const& [name, tensor] : mInputTensors)
        {
            validateTensorName(name);
        }
    }

    GenericInferenceRequest(uint64_t requestId, TensorMap const& tensorMap)
        : GenericInferenceRequest(requestId, TensorMap{tensorMap})
    {
    }

    void setIsStreaming(bool isStreaming)
    {
        mIsStreaming = isStreaming;
    }

    [[nodiscard]] bool isStreaming() const
    {
        return mIsStreaming;
    }

    [[nodiscard]] uint64_t getRequestId() const
    {
        return mRequestId;
    }

    static std::array constexpr kTensorNames = {
        inference_request::kInputIdsTensorName,
        inference_request::kDraftInputIdsTensorName,
        inference_request::kDraftLogitsTensorName,
        inference_request::kMaxNewTokensTensorName,
        inference_request::kBeamWidthTensorName,
        inference_request::kEndIdTensorName,
        inference_request::kPadIdTensorName,
        inference_request::kBadWordsListTensorName,
        inference_request::kStopWordsListTensorName,
        inference_request::kEmbeddingBiasTensorName,
        inference_request::kTemperatureTensorName,
        inference_request::kRuntimeTopKTensorName,
        inference_request::kRuntimeTopPTensorName,
        inference_request::kLengthPenaltyTensorName,
        inference_request::kRepetitionPenaltyTensorName,
        inference_request::kMinLengthTensorName,
        inference_request::kPresencePenaltyTensorName,
        inference_request::kRandomSeedTensorName,
        inference_request::kReturnLogProbsTensorName,
        inference_request::kPromptEmbeddingTableName,
        inference_request::kPromptVocabSizeName,
        // obsolete names for backward compatibility
        inference_request::kInputLengthsTensorName,
    };

#define TENSOR_GETTER_SETTER(funcName, tensorName)                                                                     \
                                                                                                                       \
    [[nodiscard]] bool has##funcName() const                                                                           \
    {                                                                                                                  \
        return mInputTensors.find(tensorName) != mInputTensors.end();                                                  \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] TensorPtr const& get##funcName() const                                                               \
    {                                                                                                                  \
        auto it = mInputTensors.find(tensorName);                                                                      \
        TLLM_CHECK_WITH_INFO(it != mInputTensors.end(), "Undefined tensor: %s", tensorName);                           \
        return it->second;                                                                                             \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] TensorPtr get##funcName##Unchecked() const                                                           \
    {                                                                                                                  \
        auto it = mInputTensors.find(tensorName);                                                                      \
        return it != mInputTensors.end() ? it->second : TensorPtr{};                                                   \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] NamedTensorType get##funcName##Named() const                                                         \
    {                                                                                                                  \
        auto it = mInputTensors.find(tensorName);                                                                      \
        return it != mInputTensors.end() ? NamedTensorType{it->second, tensorName} : NamedTensor{tensorName};          \
    }                                                                                                                  \
                                                                                                                       \
    void set##funcName(TensorPtr const& tensor)                                                                        \
    {                                                                                                                  \
        mInputTensors[tensorName] = tensor;                                                                            \
    }

    TENSOR_GETTER_SETTER(InputIds, inference_request::kInputIdsTensorName)
    TENSOR_GETTER_SETTER(DraftInputIds, inference_request::kDraftInputIdsTensorName)
    TENSOR_GETTER_SETTER(DraftLogits, inference_request::kDraftLogitsTensorName)
    TENSOR_GETTER_SETTER(MaxNewTokens, inference_request::kMaxNewTokensTensorName)
    TENSOR_GETTER_SETTER(BeamWidth, inference_request::kBeamWidthTensorName)
    TENSOR_GETTER_SETTER(EndId, inference_request::kEndIdTensorName)
    TENSOR_GETTER_SETTER(PadId, inference_request::kPadIdTensorName)
    TENSOR_GETTER_SETTER(BadWordsList, inference_request::kBadWordsListTensorName)
    TENSOR_GETTER_SETTER(StopWordsList, inference_request::kStopWordsListTensorName)
    TENSOR_GETTER_SETTER(EmbeddingBias, inference_request::kEmbeddingBiasTensorName)
    TENSOR_GETTER_SETTER(Temperature, inference_request::kTemperatureTensorName)
    TENSOR_GETTER_SETTER(RuntimeTopK, inference_request::kRuntimeTopKTensorName)
    TENSOR_GETTER_SETTER(RuntimeTopP, inference_request::kRuntimeTopPTensorName)
    TENSOR_GETTER_SETTER(LengthPenalty, inference_request::kLengthPenaltyTensorName)
    TENSOR_GETTER_SETTER(RepetitionPenalty, inference_request::kRepetitionPenaltyTensorName)
    TENSOR_GETTER_SETTER(MinLength, inference_request::kMinLengthTensorName)
    TENSOR_GETTER_SETTER(PresencePenalty, inference_request::kPresencePenaltyTensorName)
    TENSOR_GETTER_SETTER(RandomSeed, inference_request::kRandomSeedTensorName)
    TENSOR_GETTER_SETTER(ReturnLogProbs, inference_request::kReturnLogProbsTensorName)
    TENSOR_GETTER_SETTER(PromptEmbeddingTable, inference_request::kPromptEmbeddingTableName)
    TENSOR_GETTER_SETTER(PromptVocabSize, inference_request::kPromptVocabSizeName)

#undef TENSOR_GETTER_SETTER

protected:
    static void validateTensorName(std::string const& tensorName)
    {
        TLLM_CHECK_WITH_INFO(std::find(kTensorNames.begin(), kTensorNames.end(), tensorName) != kTensorNames.end(),
            "Invalid tensor name: %s", tensorName.c_str());
    }

    uint64_t mRequestId;
    bool mIsStreaming;
    TensorMap mInputTensors;
};

class InferenceRequest : public GenericInferenceRequest<tensorrt_llm::runtime::ITensor::SharedPtr, NamedTensor>
{
public:
    using Base = GenericInferenceRequest<tensorrt_llm::runtime::ITensor::SharedPtr, NamedTensor>;
    using TensorPtr = Base::TensorPtr;
    using TensorMap = Base::TensorMap;

    explicit InferenceRequest(uint64_t requestId)
        : Base(requestId)
    {
    }

    InferenceRequest(TensorMap const& inputTensors, uint64_t requestId)
        : Base(requestId, inputTensors)
    {
    }

    InferenceRequest(TensorMap&& inputTensors, uint64_t requestId)
        : Base(requestId, std::move(inputTensors))
    {
    }

    [[deprecated("Use direct tensor access instead")]] [[nodiscard]] TensorPtr const& getInputTensor(
        std::string const& inputTensorName) const
    {
        auto it = Base::mInputTensors.find(inputTensorName);
        TLLM_CHECK_WITH_INFO(it != Base::mInputTensors.end(), "Invalid input tensor name: %s", inputTensorName.c_str());
        return it->second;
    }

    [[deprecated("Use direct tensor access instead")]] void emplaceInputTensor(
        std::string const& inputTensorName, TensorPtr inputTensor)
    {
        validateTensorName(inputTensorName);
        Base::mInputTensors[inputTensorName] = std::move(inputTensor);
    }

    [[nodiscard]] std::vector<int64_t> serialize() const;

    static std::shared_ptr<InferenceRequest> deserialize(const std::vector<int64_t>& packed);

    static std::shared_ptr<InferenceRequest> deserialize(const int64_t* packed_ptr);
};

} // namespace tensorrt_llm::batch_manager
