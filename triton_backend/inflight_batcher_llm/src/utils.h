// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "NvInfer.h"
#include "namedTensor.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"
#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace tensorrt_llm;

namespace triton::backend::inflight_batcher_llm
{

/// @brief Names of input fields
struct InputFieldsNames
{
    static constexpr char const* inputTokens = "input_ids";
    static constexpr char const* encoderInputFeatures = "encoder_input_features";
    static constexpr char const* encoderOutputLengths = "encoder_output_lengths";
    static constexpr char const* inputLengths = "input_lengths";
    static constexpr char const* decoderInputTokens = "decoder_input_ids";
    static constexpr char const* maxNewTokens = "request_output_len";
    static constexpr char const* noRepeatNgramSize = "no_repeat_ngram_size";
    static constexpr char const* numReturnSequences = "num_return_sequences";
    static constexpr char const* endId = "end_id";
    static constexpr char const* padId = "pad_id";
    static constexpr char const* badWords = "bad_words_list";
    static constexpr char const* stopWords = "stop_words_list";
    static constexpr char const* embeddingBias = "embedding_bias";
    static constexpr char const* contextPhaseParams = "context_phase_params";
    static constexpr char const* crossAttentionMask = "cross_attention_mask";
    static constexpr char const* skipCrossAttnBlocks = "skip_cross_attn_blocks";
    static constexpr char const* multimodalEmbedding = "multimodal_embedding";

    // OutputConfig
    static constexpr char const* returnLogProbs = "return_log_probs";
    static constexpr char const* returnGenerationLogits = "return_generation_logits";
    static constexpr char const* returnContextLogits = "return_context_logits";
    static constexpr char const* excludeInputFromOutput = "exclude_input_in_output";
    static constexpr char const* returnPerfMetrics = "return_perf_metrics";
    static constexpr char const* returnNumInputTokens = "return_num_input_tokens";
    static constexpr char const* returnNumOutputTokens = "return_num_output_tokens";

    // Deprecated
    static constexpr char const* returnKvCacheReuseStats = "return_kv_cache_reuse_stats";

    // SamplingConfig
    static constexpr char const* beamWidth = "beam_width";
    static constexpr char const* topK = "runtime_top_k";
    static constexpr char const* topP = "runtime_top_p";
    static constexpr char const* topPMin = "runtime_top_k_min";
    static constexpr char const* topPDecay = "runtime_top_p_decay";
    static constexpr char const* topPResetIds = "runtime_top_p_reset_ids";
    static constexpr char const* temperature = "temperature";
    static constexpr char const* lengthPenalty = "len_penalty";
    static constexpr char const* earlyStopping = "early_stopping";
    static constexpr char const* repetitionPenalty = "repetition_penalty";
    static constexpr char const* minTokens = "min_tokens";
    static constexpr char const* beamSearchDiversityRate = "beam_search_diversity_rate";
    static constexpr char const* presencePenalty = "presence_penalty";
    static constexpr char const* frequencyPenalty = "frequency_penalty";
    static constexpr char const* seed = "seed";

    // PromptTuningConfig
    static constexpr char const* promptEmbeddingTable = "prompt_embedding_table";
    static constexpr char const* InputTokenExtraIds = "prompt_table_extra_ids";

    // MropeConfig
    static constexpr char const* mropeRotaryCosSin = "mrope_rotary_cos_sin";
    static constexpr char const* mropePositionDeltas = "mrope_position_deltas";

    // MultimodalInput
    static constexpr char const* multimodalHashes = "multimodal_hashes";
    static constexpr char const* multimodalPositions = "multimodal_positions";
    static constexpr char const* multimodalLengths = "multimodal_lengths";

    // LoraConfig
    static constexpr char const* loraTaskId = "lora_task_id";
    static constexpr char const* loraWeights = "lora_weights";
    static constexpr char const* loraConfig = "lora_config";

    // ExternalDraftTokensConfig
    static constexpr char const* draftInputs = "draft_input_ids";
    static constexpr char const* draftLogits = "draft_logits";
    static constexpr char const* draftAcceptanceThreshold = "draft_acceptance_threshold";

    // KvCacheRetentionConfig
    static constexpr char const* retentionTokenRangeStarts = "retention_token_range_starts";
    static constexpr char const* retentionTokenRangeEnds = "retention_token_range_ends";
    static constexpr char const* retentionTokenRangePriorities = "retention_token_range_priorities";
    static constexpr char const* retentionTokenRangeDurations = "retention_token_range_durations_ms";
    static constexpr char const* retentionDecodePriority = "retention_decode_priority";
    static constexpr char const* retentionDecodeDuration = "retention_decode_duration_ms";

    // GuidedDecodingParams
    static constexpr char const* guidedDecodingGuideType = "guided_decoding_guide_type";
    static constexpr char const* guidedDecodingGuide = "guided_decoding_guide";

    // LookaheadDecodingConfig
    static constexpr char const* requestLookaheadDecodingWindowSize = "lookahead_window_size";
    static constexpr char const* requestLookaheadDecodingNgramSize = "lookahead_ngram_size";
    static constexpr char const* requestLookaheadDecodingVerificationSetSize = "lookahead_verification_set_size";
};

/// @brief Names of output fields
struct OutputFieldsNames
{
    static constexpr char const* outputIds = "output_ids";
    static constexpr char const* sequenceLength = "sequence_length";
    static constexpr char const* contextLogits = "context_logits";
    static constexpr char const* generationLogits = "generation_logits";
    static constexpr char const* outputLogProbs = "output_log_probs";
    static constexpr char const* cumLogProbs = "cum_log_probs";
    static constexpr char const* batchIndex = "batch_index";
    static constexpr char const* sequenceIndex = "sequence_index";
    static constexpr char const* contextPhaseParams = "context_phase_params";
    static constexpr char const* kvCacheAllocNewBlocks = "kv_cache_alloc_new_blocks";
    static constexpr char const* kvCacheReusedBlocks = "kv_cache_reused_blocks";
    static constexpr char const* kvCacheAllocTotalBlocks = "kv_cache_alloc_total_blocks";
    static constexpr char const* arrivalTime = "arrival_time_ns";
    static constexpr char const* firstScheduledTime = "first_scheduled_time_ns";
    static constexpr char const* firstTokenTime = "first_token_time_ns";
    static constexpr char const* lastTokenTime = "last_token_time_ns";
    static constexpr char const* acceptanceRate = "acceptance_rate";
    static constexpr char const* totalAcceptedDraftTokens = "total_accepted_draft_tokens";
    static constexpr char const* totalDraftTokens = "total_draft_tokens";
    static constexpr char const* numInputTokens = "num_input_tokens";
    static constexpr char const* numOutputTokens = "num_output_tokens";
};

inline static std::string const kStopInputTensorName = "stop";
inline static std::string const kStreamingInputTensorName = "streaming";
inline static std::string const kRequestTypeParameterName = "request_type";
inline static std::unordered_map<std::string, executor::RequestType> stringToRequestType
    = {{"context_and_generation", executor::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION},
        {"context_only", executor::RequestType::REQUEST_TYPE_CONTEXT_ONLY},
        {"generation_only", executor::RequestType::REQUEST_TYPE_GENERATION_ONLY}};

namespace utils
{

/// @brief  Convert Triton datatype to TRT datatype
nvinfer1::DataType to_trt_datatype(TRITONSERVER_DataType data_type);

/// @brief Convert executor datatype to Triton datatype
TRITONSERVER_DataType to_triton_datatype(executor::DataType data_type);

using InputTensors = std::unordered_map<std::string, NamedTensor>;

/// @brief Split batched input tensors into bs==1 tensors.
/// @return Vector of maps of bs==1 tensors keyed on tensor name.
std::vector<InputTensors> splitBatchInputsTensors(InputTensors const& inputsTensors);

/// @brief Gather input tenors in a Triton request
/// @return An unordered map with key being input name and value being input tensor for each batch sample
std::vector<InputTensors> readInputsTensors(TRITONBACKEND_Request* request);

/// @brief Construct executor::SampleConfig from input tensors
executor::SamplingConfig getSamplingConfigFromTensors(InputTensors const& inputsTensors);

/// @brief Construct executor::OutputConfig from input tensors
executor::OutputConfig getOutputConfigFromTensors(InputTensors const& inputsTensors);

/// @brief Construct executor::ExternalDraftTokensConfig from input tensors
std::optional<executor::ExternalDraftTokensConfig> getExternalDraftTokensConfigFromTensors(
    InputTensors const& inputsTensors, bool const fastLogits);

/// @brief Construct executor::PromptTuningConfig from input tensors
std::optional<executor::PromptTuningConfig> getPromptTuningConfigFromTensors(
    InputTensors const& inputsTensors, size_t inputLen);

/// @brief Construct executor::LoraConfig from input tensors
std::optional<executor::LoraConfig> getLoraConfigFromTensors(InputTensors const& inputsTensors);

/// @brief Construct executor::KvCacheRetentionConfig from input tensors
std::optional<executor::KvCacheRetentionConfig> getKvCacheRetentionConfigFromTensors(InputTensors const& inputsTensors);

/// @brief Construct executor::GuidedDecodingParams from input tensors
std::optional<executor::GuidedDecodingParams> getGuidedDecodingParamsFromTensors(InputTensors const& inputsTensors);

/// @brief Construct executor::LookaheadDecodingConfig from input tensors for requests
/// @note Let executor_lookahead_config as (W, N, G). Each request can specify a Lookahead configuration, noted as (w,
/// n, g). Ensure the Lookahead configuration for each request satisfies w <= W, n <= N, g <= G.
std::optional<executor::LookaheadDecodingConfig> getLookaheadDecodingFromTensors(
    InputTensors const& inputsTensors, std::optional<executor::LookaheadDecodingConfig> const& executorLookaheadConfig);

/// @brief Construct executor::Request from input tensors
std::vector<executor::Request> createRequestsFromInputTensors(std::vector<InputTensors> const& inputsTensors,
    bool excludeInputFromOutput, bool isDecoupled, bool streaming, executor::ModelType modelType,
    executor::RequestType requestType, bool isOrchestrator, bool specDecFastLogits,
    std::optional<executor::LookaheadDecodingConfig> const& executorLookaheadConfig);

/// @brief get the requestId of the request and update requestIdStrMap
/// @return Returns 0 if not specified. Throws an error if request_id cannot be convert to uint64_t
uint64_t getRequestId(TRITONBACKEND_Request* request, std::unordered_map<uint64_t, std::string>& requestIdStrMap);

/// @brief Get the requested output names
std::unordered_set<std::string> getRequestOutputNames(TRITONBACKEND_Request* request);

/// @brief Get the value of a boolean tensor
bool getRequestBooleanInputTensor(TRITONBACKEND_Request* request, std::string const& inputTensorName);

/// @brief Get a single value tensor from the input tensors
/// @return true if the value is found else false
template <typename Value>
bool extractSingleton(InputTensors const& params, std::string const& name, Value& value)
{
    if (!params.count(name))
    {
        return false;
    }
    auto const& tensor = params.at(name);
    TLLM_CHECK_WITH_INFO(tensor.tensor->getSize() == 1, "Invalid size for tensor " + name);
    value = *(static_cast<Value*>(tensor.tensor->data()));
    return true;
}

/// @brief Get a single value tensor from the input tensors and put it into an optional. Set to std::nullopt if it's not
/// found.
template <typename Value>
void extractOptionalSingleton(InputTensors const& params, std::string const& name, std::optional<Value>& optionalValue)
{
    Value value;
    if (extractSingleton<Value>(params, name, value))
    {
        optionalValue = value;
    }
    else
    {
        optionalValue = std::nullopt;
    }
}

/// @brief Get a 1d tensor from the input tensors
/// @return true if the tensor is found else false
template <typename Value>
bool extractVector(InputTensors const& params, std::string const& name, std::vector<Value>& value)
{
    if (!params.count(name))
    {
        return false;
    }
    auto const& tensor = params.at(name);
    int64_t n = tensor.tensor->getSize();
    value.resize(n);
    for (int64_t i = 0; i < n; ++i)
    {
        value[i] = static_cast<Value*>(tensor.tensor->data())[i];
    }
    return true;
}

int64_t numElements(std::vector<int64_t> const& shape);

/// @brief Flatten the vector and copy into the buffer
template <typename T>
void flatten(std::vector<T> const& vec, void* buffer, std::vector<int64_t> const& expectedShape)
{
    TLLM_CHECK_WITH_INFO(static_cast<int64_t>(vec.size()) == numElements(expectedShape),
        "Trying to flatten a tensor with unexpected size");
    T* typedBuffer = static_cast<T*>(buffer);
    std::copy(vec.begin(), vec.end(), typedBuffer);
}

/// @brief Flatten the vector of vector and copy into the buffer
template <typename T>
void flatten(std::vector<std::vector<T>> const& vec, void* buffer, std::vector<int64_t> const& expectedShape)
{
    T* typedBuffer = static_cast<T*>(buffer);
    int64_t copiedSize = 0;
    for (auto const& innerVec : vec)
    {
        TLLM_CHECK_WITH_INFO(innerVec.size() == vec.at(0).size(),
            "The vector of vector to be flattened has mismatched sizes in its inner vectors");
        copiedSize += innerVec.size();
        typedBuffer = std::copy(innerVec.begin(), innerVec.end(), typedBuffer);
    }
    TLLM_CHECK_WITH_INFO(copiedSize == numElements(expectedShape), "Trying to flatten a tensor with unexpected size");
}

/// @brief Flatten the tensor and copy into the buffer
template <typename Value>
void flatten(tensorrt_llm::executor::Tensor const& tensor, void* buffer, std::vector<int64_t> const& expectedShape)
{
    TLLM_CHECK_WITH_INFO(static_cast<int64_t>(tensor.getSize()) == numElements(expectedShape),
        "Trying to flatten a tensor with unexpected size");
    Value* typedBuffer = static_cast<Value*>(buffer);
    Value const* ptr = static_cast<Value const*>(tensor.getData());
    std::copy(ptr, ptr + tensor.getSize(), typedBuffer);
}

/// @brief Query Triton for a buffer that can be used to pass the output tensors
template <typename T>
void* getResponseBuffer(TRITONBACKEND_Response* tritonResponse, std::vector<int64_t> const& shape,
    TRITONSERVER_DataType dtype, std::string const& name)
{
    TRITONBACKEND_Output* output;
    TRITONSERVER_Error* err{nullptr};
    err = TRITONBACKEND_ResponseOutput(tritonResponse, &output, name.c_str(), dtype, shape.data(), shape.size());
    if (err != nullptr)
    {
        auto errMsg = TRITONSERVER_ErrorMessage(err);
        TLLM_THROW("Could not get response output for output tensor %s: %s", name.c_str(), errMsg);
    }

    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    uint64_t size = 1;
    for (auto s : shape)
    {
        size *= s;
    }
    auto buffersize = size * sizeof(T);
    void* tritonBuffer = 0L;
    err = TRITONBACKEND_OutputBuffer(output, &tritonBuffer, buffersize, &memory_type, &memory_type_id);
    if (err != nullptr)
    {
        auto errMsg = TRITONSERVER_ErrorMessage(err);
        TLLM_THROW("Could not get output buffer for output tensor %s: %s", name.c_str(), errMsg);
    }
    return tritonBuffer;
}

template <typename T>
struct ParameterTypeMap
{
    static const TRITONSERVER_ParameterType parameter_type;
};

template <>
const TRITONSERVER_ParameterType ParameterTypeMap<int32_t>::parameter_type;

template <>
const TRITONSERVER_ParameterType ParameterTypeMap<std::string>::parameter_type;

template <>
const TRITONSERVER_ParameterType ParameterTypeMap<bool>::parameter_type;

template <>
const TRITONSERVER_ParameterType ParameterTypeMap<double>::parameter_type;

template <typename T>
std::optional<T> getRequestParameter(TRITONBACKEND_Request* request, std::string const& name)
{
    uint32_t parameter_count;
    TRITONBACKEND_RequestParameterCount(request, &parameter_count);
    for (size_t i = 0; i < parameter_count; i++)
    {
        char const* request_key;
        TRITONSERVER_ParameterType parameter_type;
        void const* value;
        TRITONBACKEND_RequestParameter(request, i, &request_key, &parameter_type, &value);
        if (parameter_type == ParameterTypeMap<T>::parameter_type && request_key == name)
        {
            if (std::is_same<T, std::string>::value)
            {
                return reinterpret_cast<char const*>(value);
            }
            else
            {
                return *reinterpret_cast<T const*>(value);
            }
        }
    }

    // If the parameter is not found, we would return a nullopt.
    return std::nullopt;
}

/// @brief Convert a sparse tensor to a list of VecTokens
std::list<executor::VecTokens> convertWordList(executor::VecTokens const& sparseList);

/// @brief Remove the additional size 1 dimension for tensor
void squeezeTensor(std::shared_ptr<runtime::ITensor> const& tensor, int32_t expectedNumDims);

/// Helper functions to parse a csv delimited string to a vector ints
std::vector<int32_t> csvStrToVecInt(std::string const& str);

/// Helper functions to parse a csv delimited string to a vector of vector ints
std::vector<std::vector<int32_t>> csvStrToVecVecInt(std::string const& str);

/// Split a string by a delimiter and return the tokens in a vector of strings.
std::vector<std::string> split(std::string const& str, char delimiter);

/// @brief Get the TRTLLM request type from the request parameters.
executor::RequestType getRequestType(TRITONBACKEND_Request* request);

struct MemoryBuffer : std::streambuf
{
    MemoryBuffer(char* base, size_t size)
    {
        this->setg(base, base, base + size);
    }
};

struct InMemoryStreamBuffer : virtual MemoryBuffer, std::istream
{
    InMemoryStreamBuffer(char* base, size_t size)
        : MemoryBuffer(base, size)
        , std::istream(static_cast<std::streambuf*>(this))
    {
    }
};

} // namespace utils
} // namespace triton::backend::inflight_batcher_llm
