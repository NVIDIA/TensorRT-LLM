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

#include "utils.h"
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include <cassert>

using namespace tensorrt_llm::batch_manager;

namespace triton::backend::inflight_batcher_llm::utils
{

auto constexpr kRetentionNoExpirationTime = -1;
auto constexpr kRetentionNoRangeEnd = -1;

nvinfer1::DataType to_trt_datatype(TRITONSERVER_DataType data_type)
{
    if (data_type == TRITONSERVER_TYPE_INVALID)
    {
        assert(false);
    }
    else if (data_type == TRITONSERVER_TYPE_BOOL)
    {
        return nvinfer1::DataType::kBOOL;
    }
    else if (data_type == TRITONSERVER_TYPE_UINT8)
    {
        return nvinfer1::DataType::kUINT8;
    }
    else if (data_type == TRITONSERVER_TYPE_UINT16)
    {
        assert(false);
    }
    else if (data_type == TRITONSERVER_TYPE_UINT32)
    {
        return nvinfer1::DataType::kINT32;
    }
    else if (data_type == TRITONSERVER_TYPE_UINT64)
    {
        return nvinfer1::DataType::kINT64;
    }
    else if (data_type == TRITONSERVER_TYPE_INT8)
    {
        return nvinfer1::DataType::kINT8;
    }
    else if (data_type == TRITONSERVER_TYPE_INT16)
    {
        assert(false);
    }
    else if (data_type == TRITONSERVER_TYPE_INT32)
    {
        return nvinfer1::DataType::kINT32;
    }
    else if (data_type == TRITONSERVER_TYPE_INT64)
    {
        return nvinfer1::DataType::kINT64;
    }
    else if (data_type == TRITONSERVER_TYPE_FP16)
    {
        return nvinfer1::DataType::kHALF;
    }
    else if (data_type == TRITONSERVER_TYPE_FP32)
    {
        return nvinfer1::DataType::kFLOAT;
    }
    else if (data_type == TRITONSERVER_TYPE_FP64)
    {
        assert(false);
    }
    else if (data_type == TRITONSERVER_TYPE_BYTES)
    {
        return nvinfer1::DataType::kINT8;
    }
    else if (data_type == TRITONSERVER_TYPE_BF16)
    {
        return nvinfer1::DataType::kBF16;
    }
    else
    {
        assert(false);
    }
    return nvinfer1::DataType(0);
}

TRITONSERVER_DataType to_triton_datatype(executor::DataType data_type)
{
    if (data_type == executor::DataType::kBOOL)
    {
        return TRITONSERVER_TYPE_BOOL;
    }
    else if (data_type == executor::DataType::kUINT8)
    {
        return TRITONSERVER_TYPE_UINT8;
    }
    else if (data_type == executor::DataType::kINT8)
    {
        return TRITONSERVER_TYPE_INT8;
    }
    else if (data_type == executor::DataType::kINT32)
    {
        return TRITONSERVER_TYPE_INT32;
    }
    else if (data_type == executor::DataType::kINT64)
    {
        return TRITONSERVER_TYPE_INT64;
    }
    else if (data_type == executor::DataType::kBF16)
    {
        return TRITONSERVER_TYPE_BF16;
    }
    else if (data_type == executor::DataType::kFP8)
    {
        assert(false);
    }
    else if (data_type == executor::DataType::kFP16)
    {
        return TRITONSERVER_TYPE_FP16;
    }
    else if (data_type == executor::DataType::kFP32)
    {
        return TRITONSERVER_TYPE_FP32;
    }
    else if (data_type == executor::DataType::kUNKNOWN)
    {
        assert(false);
    }
    else
    {
        assert(false);
    }
    return TRITONSERVER_TYPE_INVALID;
}

std::vector<InputTensors> splitBatchInputsTensors(InputTensors const& inputsTensors)
{
    auto inputTokensIt = inputsTensors.find(InputFieldsNames::inputTokens);
    auto encoderInputFeaturesIt = inputsTensors.find(InputFieldsNames::encoderInputFeatures);
    auto batchSize = 1;
    if (inputTokensIt != inputsTensors.end())
    {
        auto inputTokensTensor = inputTokensIt->second.tensor;
        auto inputTokensShape = inputTokensTensor->getShape();
        TLLM_CHECK_WITH_INFO(inputTokensShape.nbDims == 2, "Expected inputTokens tensors to have 2 dimensions");
        batchSize = inputTokensShape.d[0];
    }
    else if (encoderInputFeaturesIt != inputsTensors.end())
    {
        auto encoderInputFeaturesTensor = encoderInputFeaturesIt->second.tensor;
        auto encoderInputFeaturesShape = encoderInputFeaturesTensor->getShape();
        batchSize = encoderInputFeaturesShape.d[0];
    }
    else
    {
        TLLM_THROW("inputTokens or encoderInputFeatures tensor not provided");
    }

    if (batchSize > 1)
    {
        // Split tensors along batch dimension. All tensors have same batch size.
        auto itrInputLengths = inputsTensors.find(InputFieldsNames::inputLengths);
        TLLM_CHECK_WITH_INFO(itrInputLengths != inputsTensors.end(), "input lengths tensor not provided");
        auto inputLengthsTensor = itrInputLengths->second.tensor;
        auto inputLengthsShape = inputLengthsTensor->getShape();
        TLLM_CHECK_WITH_INFO(inputLengthsShape.nbDims == 1, "Expected inputLengths tensors to have 1 dimension");

        using typedPtr = typename runtime::DataTypeTraits<nvinfer1::DataType::kINT32>::type*;
        auto inputLengths = static_cast<typedPtr>(inputLengthsTensor->data());
        std::vector<InputTensors> rval(batchSize);
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            auto inputLength = inputLengths[batchIdx];
            for (auto const& [name, batchedTensor] : inputsTensors)
            {
                std::shared_ptr<runtime::ITensor> slicedTensor
                    = std::move(runtime::ITensor::slice(batchedTensor.tensor, batchIdx, 1));
                // Use input_length to strip off padding for inputTokens
                if (name == InputFieldsNames::inputTokens || name == InputFieldsNames::encoderInputFeatures)
                {
                    slicedTensor->squeeze(0);
                    slicedTensor = runtime::ITensor::slice(slicedTensor, 0, static_cast<int>(inputLength));
                    slicedTensor->unsqueeze(0);
                }
                NamedTensor namedTensor(slicedTensor, name);
                rval[batchIdx].insert({name, namedTensor});
            }
        }
        return rval;
    }
    else
    {
        return {inputsTensors};
    }
}

std::vector<InputTensors> readInputsTensors(TRITONBACKEND_Request* request)
{
    InputTensors inputsTensors;
    uint32_t num_inputs;
    LOG_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &num_inputs), "Error getting input count");
    auto const stream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
    auto const manager = tensorrt_llm::runtime::BufferManager{std::move(stream)};
    for (uint32_t idx = 0; idx < num_inputs; ++idx)
    {
        TRITONBACKEND_Input* input = nullptr;
        LOG_IF_ERROR(TRITONBACKEND_RequestInputByIndex(request, idx, &input), "Error getting input index");

        char const* input_name = nullptr;
        TRITONSERVER_DataType data_type = TRITONSERVER_TYPE_INVALID;
        int64_t const* shape = nullptr;
        uint32_t dims_count = 0;
        uint64_t byte_size = 0;
        uint32_t buffer_count = 0;
        LOG_IF_ERROR(TRITONBACKEND_InputProperties(
                         input, &input_name, &data_type, &shape, &dims_count, &byte_size, &buffer_count),
            "Error getting input properties");

        if (std::string(input_name) == "START" || std::string(input_name) == "CORRID"
            || std::string(input_name) == "END" || std::string(input_name) == kStopInputTensorName
            || std::string(input_name) == kStreamingInputTensorName)
        {
            continue;
        }

        std::vector<int64_t> shapev;
        for (uint32_t i = 0; i < dims_count; ++i)
        {
            shapev.push_back(shape[i]);
            // NOTE
            // To handle the encoded string, we create placeholder shapes for string input.
            // The encoded string is passed with a shape of [1] and a data type of int8, as determined by
            // utils::to_trt_datatype. This makes that the tensor size in bytes is 1, even if the actual string size in
            // bytes is larger. In orer to address this, we intentionally set the last dimension of the tensor's shape
            // to match the byte size obtained from TRITONBACKEND_InputProperties. This allows us to accurately recover
            // the actual byte size of the tensor.
            if (data_type == TRITONSERVER_TYPE_BYTES && i == dims_count - 1)
            {
                shapev[dims_count - 1] = byte_size / shapev[0];
            }
        }

        NamedTensor t(utils::to_trt_datatype(data_type), shapev, input_name);
        uint64_t buffer_offset = 0;
        for (int64_t buffer_id = 0; buffer_id < buffer_count; ++buffer_id)
        {
            void const* buffer = nullptr;
            uint64_t buffer_byte_size = 0;
            TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
            int64_t memory_type_id = 0;
            LOG_IF_ERROR(
                TRITONBACKEND_InputBuffer(input, buffer_id, &buffer, &buffer_byte_size, &memory_type, &memory_type_id),
                "failed to get input buffer");
            if (memory_type == TRITONSERVER_MEMORY_GPU)
            {
                if (buffer_id == 0)
                {
                    t.tensor = manager.gpu(t.tensor->getShape(), utils::to_trt_datatype(data_type));
                }
                TLLM_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(t.tensor->data()) + buffer_offset, buffer,
                    buffer_byte_size, cudaMemcpyDeviceToDevice, manager.getStream().get()));
            }
            else
            {
                std::memcpy(static_cast<char*>(t.tensor->data()) + buffer_offset, buffer, buffer_byte_size);
            }
            buffer_offset += buffer_byte_size;
        }
        inputsTensors.insert(make_pair(t.name, std::move(t)));
    }
    manager.getStream().synchronize();
    return splitBatchInputsTensors(inputsTensors);
}

uint64_t getRequestId(TRITONBACKEND_Request* request, std::unordered_map<uint64_t, std::string>& requestIdStrMap)
{
    char const* charRequestId;
    TRITONBACKEND_RequestId(request, &charRequestId);
    uint64_t requestId = 0;
    if (charRequestId != nullptr)
    {
        std::string strRequestId(charRequestId);
        if (!strRequestId.empty())
        {
            try
            {
                requestId = stoul(strRequestId);
            }
            catch (std::exception const& e)
            {
                std::hash<std::string> hasher;
                requestId = hasher(strRequestId);

                // Check for hash collisions
                // If requestID already exists in the map with the same string, increment the ID and check again
                for (auto it = requestIdStrMap.find(requestId);
                     it != requestIdStrMap.end() && it->second != strRequestId;)
                {
                    requestId++;
                }
            }
            requestIdStrMap.insert({requestId, strRequestId});
        }
    }

    return requestId;
}

executor::RequestType getRequestType(TRITONBACKEND_Request* request)
{
    executor::RequestType requestType = executor::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION;
    auto requestTypeStr = getRequestParameter<std::string>(request, kRequestTypeParameterName);
    if (requestTypeStr)
    {
        if (stringToRequestType.count(requestTypeStr.value()) > 0)
        {
            requestType = stringToRequestType.at(requestTypeStr.value());
        }
        else
        {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                (std::string("Unexpected parameter value for 'triton_trtllm_request_type'. Found: ")
                    + requestTypeStr.value())
                    .c_str());
        }
    }

    return requestType;
}

std::unordered_set<std::string> getRequestOutputNames(TRITONBACKEND_Request* request)
{
    std::unordered_set<std::string> outputNames;
    uint32_t outputCount;
    LOG_IF_ERROR(TRITONBACKEND_RequestOutputCount(request, &outputCount), "Error getting request output count");
    for (size_t i = 0; i < outputCount; ++i)
    {
        char const* name;
        LOG_IF_ERROR(TRITONBACKEND_RequestOutputName(request, i, &name), "Error getting request output name");
        std::string name_s(name);
        outputNames.insert(std::move(name_s));
    }
    return outputNames;
}

bool getRequestBooleanInputTensor(TRITONBACKEND_Request* request, std::string const& inputTensorName)
{
    // Get stop signal from the request
    TRITONBACKEND_Input* input;
    TRITONSERVER_Error* error = TRITONBACKEND_RequestInput(request, inputTensorName.c_str(), &input);
    if (error)
    {
        // If the user does not provide input "stop", then regard the request as
        // unstopped
        std::string msg
            = "ModelInstanceState::getRequestBooleanInputTensor: user "
              "did not not provide "
            + inputTensorName + " input for the request";
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, msg.c_str());
        TRITONSERVER_ErrorDelete(error);
        return false;
    }

    uint64_t input_byte_size = 0;
    uint32_t buffer_count = 0;
    TRITONBACKEND_InputProperties(input, nullptr, nullptr, nullptr, nullptr, &input_byte_size, &buffer_count);

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
        ("ModelInstanceState::getRequestStopSignal: buffer_count = " + std::to_string(buffer_count)).c_str());

    void const* buffer = 0L;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    TRITONBACKEND_InputBuffer(input, 0, &buffer, &buffer_byte_size, &memory_type, &memory_type_id);

    assert((memory_type == TRITONSERVER_MEMORY_CPU) || (memory_type == TRITONSERVER_MEMORY_CPU_PINNED));

    bool boolean = *reinterpret_cast<bool const*>(buffer);

    return boolean;
}

std::string sparseListToStr(executor::VecTokens const& sparseList)
{
    std::string buffer;
    for (auto v : sparseList)
    {
        buffer.append(std::to_string(v) + " ");
    }
    return buffer;
}

std::list<executor::VecTokens> convertWordList(executor::VecTokens const& sparseList)
{
    std::list<executor::VecTokens> convertedList;
    int32_t n = sparseList.size();
    TLLM_CHECK_WITH_INFO(n % 2 == 0, "Sparse list must not have odd length: " + sparseListToStr(sparseList));
    int32_t numTokens = n / 2;
    int32_t currentIndex = 0;
    for (auto i = numTokens; i < n; ++i)
    {
        if (sparseList[i] == -1)
        {
            for (auto j = i + 1; j < n; ++j)
            {
                TLLM_CHECK_WITH_INFO(
                    sparseList[j] == -1, "Sparse list must not have additional -1s: " + sparseListToStr(sparseList));
            }
            break;
        }
        TLLM_CHECK_WITH_INFO(sparseList[i] <= numTokens,
            "Sparse list must not have out-of-bound offsets: " + sparseListToStr(sparseList));
        if (i != numTokens)
        {
            TLLM_CHECK_WITH_INFO(sparseList[i] > sparseList[i - 1],
                "Sparse list must not have non-increasing offsets: " + sparseListToStr(sparseList));
        }
        executor::VecTokens currentWords;
        while (currentIndex < sparseList[i])
        {
            currentWords.push_back(sparseList[currentIndex]);
            ++currentIndex;
        }
        convertedList.push_back(currentWords);
    }
    return convertedList;
}

void squeezeTensor(std::shared_ptr<runtime::ITensor> const& tensor, int32_t expectedNumDims)
{
    auto shape = tensor->getShape();
    if (shape.nbDims == expectedNumDims)
    {
        return;
    }
    if (shape.nbDims == expectedNumDims + 1 && shape.d[0] == 1)
    {
        --shape.nbDims;
        for (int32_t i = 0; i < expectedNumDims; ++i)
        {
            shape.d[i] = shape.d[i + 1];
        }
        tensor->reshape(shape);
    }
    else
    {
        TLLM_LOG_ERROR("Unexpected prompt tensor shape");
    }
}

std::vector<int32_t> csvStrToVecInt(std::string const& str)
{
    TLLM_CHECK_WITH_INFO(!str.empty(), "Cannot convert empty string to vector of vector of ints");

    std::vector<int32_t> output;
    std::stringstream ss(str);
    while (ss.good())
    {
        std::string substr;
        ss >> std::ws;
        getline(ss, substr, ',');
        if (substr.empty())
        {
            break;
        }
        output.push_back(std::stoi(substr));
    }
    TLLM_CHECK_WITH_INFO(!output.empty(), "Empty vector");
    return output;
}

std::vector<std::vector<int32_t>> csvStrToVecVecInt(std::string const& str)
{
    TLLM_CHECK_WITH_INFO(!str.empty(), "Cannot convert empty string to vector of vector of ints");

    std::vector<std::vector<int32_t>> output;
    std::stringstream ss(str);

    while (true)
    {
        std::string substr;
        getline(ss, substr, '}');
        if (substr.empty() || ss.eof())
        {
            break;
        }
        if (substr[0] == '{')
        {
            // Remove the opening bracket from the content
            substr = substr.substr(1);
        }
        output.push_back(csvStrToVecInt(substr));
        // Ignore the comma and any whitespace
        ss >> std::ws;
        ss.ignore(std::numeric_limits<std::streamsize>::max(), ',');
        ss >> std::ws;
    }
    TLLM_CHECK_WITH_INFO(!output.empty(), "Empty vector of vector");
    return output;
}

std::vector<std::string> split(std::string const& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(str);

    while (std::getline(ss, token, delimiter))
    {
        tokens.push_back(token);
    }

    return tokens;
}

int64_t numElements(std::vector<int64_t> const& shape)
{
    int64_t n = 1;
    for (auto d : shape)
    {
        n *= d;
    }
    return n;
}

executor::SamplingConfig getSamplingConfigFromTensors(InputTensors const& inputsTensors)
{
    int32_t beamWidth = 1;
    // If beam_width is specified, set it from config.pbtxt
    extractSingleton<int32_t>(inputsTensors, InputFieldsNames::beamWidth, beamWidth);

    std::optional<executor::SizeType32> topK{std::nullopt};
    extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::topK, topK);

    std::optional<float> topP{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::topP, topP);
    if (topP.has_value() && topP.value() <= 0.F)
    {
        topP.reset();
    }

    std::optional<float> topPMin{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::topPMin, topPMin);

    std::optional<float> topPDecay{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::topPDecay, topPDecay);

    std::optional<int32_t> topPResetIds{std::nullopt};
    extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::topPResetIds, topPResetIds);

    std::optional<float> temperature{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::temperature, temperature);

    std::optional<float> lengthPenalty{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::lengthPenalty, lengthPenalty);

    std::optional<int32_t> earlyStopping{std::nullopt};
    extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::earlyStopping, earlyStopping);

    std::optional<float> repetitionPenalty{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::repetitionPenalty, repetitionPenalty);

    std::optional<int32_t> minTokens{std::nullopt};
    extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::minTokens, minTokens);

    std::optional<float> beamSearchDiversityRate{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::beamSearchDiversityRate, beamSearchDiversityRate);

    std::optional<float> presencePenalty{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::presencePenalty, presencePenalty);

    std::optional<float> frequencyPenalty{std::nullopt};
    extractOptionalSingleton<float>(inputsTensors, InputFieldsNames::frequencyPenalty, frequencyPenalty);

    std::optional<uint64_t> seed{std::nullopt};
    extractOptionalSingleton<uint64_t>(inputsTensors, InputFieldsNames::seed, seed);

    std::optional<int32_t> noRepeatNgramSize{std::nullopt};
    extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::noRepeatNgramSize, noRepeatNgramSize);

    std::optional<int32_t> numReturnSequences{std::nullopt};
    extractOptionalSingleton<int32_t>(inputsTensors, InputFieldsNames::numReturnSequences, numReturnSequences);

    return executor::SamplingConfig(beamWidth, topK, topP, topPMin, topPResetIds, topPDecay, seed, temperature,
        minTokens, beamSearchDiversityRate, repetitionPenalty, presencePenalty, frequencyPenalty, lengthPenalty,
        earlyStopping, noRepeatNgramSize, numReturnSequences);
}

executor::OutputConfig getOutputConfigFromTensors(InputTensors const& inputsTensors)
{
    bool returnLogProbs{false};
    extractSingleton<bool>(inputsTensors, InputFieldsNames::returnLogProbs, returnLogProbs);

    bool returnGenerationLogits{false};
    extractSingleton<bool>(inputsTensors, InputFieldsNames::returnGenerationLogits, returnGenerationLogits);

    bool returnContextLogits{false};
    extractSingleton<bool>(inputsTensors, InputFieldsNames::returnContextLogits, returnContextLogits);

    bool returnPerfMetrics{false};
    extractSingleton<bool>(inputsTensors, InputFieldsNames::returnPerfMetrics, returnPerfMetrics);

    return executor::OutputConfig(returnLogProbs, returnContextLogits, returnGenerationLogits,
        false /* excludeInputFromOutput */, false /* returnEncoderOutput */, returnPerfMetrics);
}

std::optional<executor::ExternalDraftTokensConfig> getExternalDraftTokensConfigFromTensors(
    InputTensors const& inputsTensors, bool const fastLogits)
{
    std::optional<executor::ExternalDraftTokensConfig> externalDraftTokensConfig = std::nullopt;

    if (inputsTensors.count(InputFieldsNames::draftInputs))
    {
        executor::VecTokens draftInputs;
        extractVector<int32_t>(inputsTensors, InputFieldsNames::draftInputs, draftInputs);

        std::optional<executor::Tensor> draftLogits = std::nullopt;
        if (inputsTensors.count(InputFieldsNames::draftLogits))
        {
            std::shared_ptr<runtime::ITensor> originaldraftLogitsTensor
                = inputsTensors.at(InputFieldsNames::draftLogits).tensor;
            utils::squeezeTensor(originaldraftLogitsTensor, 2);
            draftLogits = executor::detail::ofITensor(originaldraftLogitsTensor);
        }

        std::optional<float> draftAcceptanceThreshold{std::nullopt};
        utils::extractOptionalSingleton<float>(
            inputsTensors, InputFieldsNames::draftAcceptanceThreshold, draftAcceptanceThreshold);

        externalDraftTokensConfig
            = executor::ExternalDraftTokensConfig(draftInputs, draftLogits, draftAcceptanceThreshold, fastLogits);
    }
    return externalDraftTokensConfig;
}

std::optional<executor::PromptTuningConfig> getPromptTuningConfigFromTensors(
    InputTensors const& inputsTensors, size_t inputlen)
{
    std::optional<executor::PromptTuningConfig> pTuningConfig = std::nullopt;
    if (inputsTensors.count(InputFieldsNames::promptEmbeddingTable))
    {
        std::shared_ptr<runtime::ITensor> originalTensor
            = inputsTensors.at(InputFieldsNames::promptEmbeddingTable).tensor;
        utils::squeezeTensor(originalTensor, 2);
        auto const& executorTensor = executor::detail::ofITensor(originalTensor);

        std::optional<executor::VecTokenExtraIds> optionalInputTokenExtraIds{std::nullopt};
        executor::VecTokenExtraIds InputTokenExtraIds;
        if (utils::extractVector<uint64_t>(inputsTensors, InputFieldsNames::InputTokenExtraIds, InputTokenExtraIds)
            && InputTokenExtraIds.size() == inputlen)
        {
            optionalInputTokenExtraIds = InputTokenExtraIds;
            pTuningConfig = executor::PromptTuningConfig(executorTensor, optionalInputTokenExtraIds);
        }
        else
            pTuningConfig = executor::PromptTuningConfig(executorTensor);
    }
    return pTuningConfig;
}

std::optional<executor::MropeConfig> getMropeConfigFromTensors(InputTensors const& inputsTensors)
{
    std::optional<executor::MropeConfig> mropeConfig = std::nullopt;
    if (inputsTensors.count(InputFieldsNames::mropeRotaryCosSin)
        && inputsTensors.count(InputFieldsNames::mropePositionDeltas))
    {
        std::shared_ptr<runtime::ITensor> originalMropeRotaryCosSinTensor
            = inputsTensors.at(InputFieldsNames::mropeRotaryCosSin).tensor;
        utils::squeezeTensor(originalMropeRotaryCosSinTensor, 1);
        auto const& mropeRotaryCosSinTensor = executor::detail::ofITensor(originalMropeRotaryCosSinTensor);

        executor::SizeType32 mropePositionDeltas;
        utils::extractSingleton<executor::SizeType32>(
            inputsTensors, InputFieldsNames::mropePositionDeltas, mropePositionDeltas);
        mropeConfig = executor::MropeConfig(mropeRotaryCosSinTensor, mropePositionDeltas);
    }
    return mropeConfig;
}

std::optional<executor::LoraConfig> getLoraConfigFromTensors(InputTensors const& inputsTensors)
{
    std::optional<executor::LoraConfig> loraConfig = std::nullopt;
    if (inputsTensors.count(InputFieldsNames::loraTaskId))
    {
        uint64_t taskId;
        if (!utils::extractSingleton<uint64_t>(inputsTensors, InputFieldsNames::loraTaskId, taskId))
        {
            throw std::runtime_error("failed to extract lora task id");
        }

        std::optional<executor::Tensor> loraConfigTensor{std::nullopt};
        if (inputsTensors.count(InputFieldsNames::loraConfig))
        {
            std::shared_ptr<runtime::ITensor> originalLoraConfigTensor
                = inputsTensors.at(InputFieldsNames::loraConfig).tensor;
            utils::squeezeTensor(originalLoraConfigTensor, 2);
            loraConfigTensor = executor::detail::ofITensor(originalLoraConfigTensor);
        }

        std::optional<executor::Tensor> loraWeightsTensor{std::nullopt};
        if (inputsTensors.count(InputFieldsNames::loraWeights))
        {
            std::shared_ptr<runtime::ITensor> originalLoraWeightsTensor
                = inputsTensors.at(InputFieldsNames::loraWeights).tensor;
            utils::squeezeTensor(originalLoraWeightsTensor, 2);
            loraWeightsTensor = executor::detail::ofITensor(originalLoraWeightsTensor);
        }

        loraConfig = executor::LoraConfig(taskId, loraWeightsTensor, loraConfigTensor);
    }
    return loraConfig;
}

std::optional<executor::KvCacheRetentionConfig> getKvCacheRetentionConfigFromTensors(InputTensors const& inputsTensors)
{

    if (inputsTensors.count(InputFieldsNames::retentionTokenRangeStarts))
    {
        std::vector<executor::SizeType32> tokenRangeStarts;
        std::vector<executor::SizeType32> tokenRangeEnds;

        utils::extractVector<executor::SizeType32>(
            inputsTensors, InputFieldsNames::retentionTokenRangeStarts, tokenRangeStarts);

        if (!utils::extractVector<executor::SizeType32>(
                inputsTensors, InputFieldsNames::retentionTokenRangeEnds, tokenRangeEnds)
            || tokenRangeStarts.size() != tokenRangeEnds.size())
        {
            throw std::runtime_error(
                "retention_token_range_ends must be provided, and have the same length as "
                "retention_token_range_starts");
        }

        std::vector<executor::RetentionPriority> priorities;

        if (!utils::extractVector<executor::SizeType32>(
                inputsTensors, InputFieldsNames::retentionTokenRangePriorities, priorities)
            || priorities.size() != tokenRangeStarts.size())
        {
            throw std::runtime_error(
                "retention_token_range_priorities must be provided, and have the same length as "
                "retention_token_range_starts");
        }

        std::vector<std::optional<std::chrono::milliseconds>> durationsMs(tokenRangeStarts.size(), std::nullopt);

        if (inputsTensors.count(InputFieldsNames::retentionTokenRangeDurations))
        {
            std::vector<executor::SizeType32> durationVector;
            utils::extractVector<executor::SizeType32>(
                inputsTensors, InputFieldsNames::retentionTokenRangeDurations, durationVector);

            if (durationVector.size() != tokenRangeStarts.size())
            {
                throw std::runtime_error(
                    "retention_token_range_durations_ms must have the same length as retention_token_range_starts");
            }

            for (size_t i = 0; i < durationVector.size(); i++)
            {
                durationsMs[i] = durationVector[i] != kRetentionNoExpirationTime
                    ? std::optional(std::chrono::milliseconds(durationVector[i]))
                    : std::nullopt;
            }
        }

        auto decodePriority = executor::KvCacheRetentionConfig::kDefaultRetentionPriority;
        std::optional<executor::SizeType32> decodeDurationMs = std::nullopt;

        utils::extractSingleton<executor::RetentionPriority>(
            inputsTensors, InputFieldsNames::retentionDecodePriority, decodePriority);

        utils::extractOptionalSingleton<executor::SizeType32>(
            inputsTensors, InputFieldsNames::retentionDecodeDuration, decodeDurationMs);

        auto decodeDuration
            = decodeDurationMs.has_value() ? std::optional(std::chrono::milliseconds(*decodeDurationMs)) : std::nullopt;

        std::vector<executor::KvCacheRetentionConfig::TokenRangeRetentionConfig> tokenRanges;

        for (size_t i = 0; i < tokenRangeStarts.size(); i++)
        {
            tokenRanges.emplace_back(tokenRangeStarts[i],
                tokenRangeEnds[i] == kRetentionNoRangeEnd ? std::nullopt : std::optional(tokenRangeEnds[i]),
                priorities[i], durationsMs[i]);
        }

        return executor::KvCacheRetentionConfig(tokenRanges, decodePriority, decodeDuration);
    }
    return std::nullopt;
}

std::optional<executor::GuidedDecodingParams> getGuidedDecodingParamsFromTensors(InputTensors const& inputsTensors)
{
    std::unordered_map<std::string, executor::GuidedDecodingParams::GuideType> guideTypeMapping
        = {{"json", executor::GuidedDecodingParams::GuideType::kJSON},
            {"json_schema", executor::GuidedDecodingParams::GuideType::kJSON_SCHEMA},
            {"regex", executor::GuidedDecodingParams::GuideType::kREGEX},
            {"ebnf_grammar", executor::GuidedDecodingParams::GuideType::kEBNF_GRAMMAR}};

    std::optional<executor::GuidedDecodingParams::GuideType> guideTypeOpt = std::nullopt;
    if (inputsTensors.count(InputFieldsNames::guidedDecodingGuideType))
    {
        auto guideTypeTensor = inputsTensors.at(InputFieldsNames::guidedDecodingGuideType).tensor;
        utils::squeezeTensor(guideTypeTensor, 2);
        char* encodedGuideType = static_cast<char*>(guideTypeTensor->data());
        std::string guideType(encodedGuideType, guideTypeTensor->getSizeInBytes());
        // NOTE
        // The first 4 bytes is null character and need to be removed for mapping.
        // Same happens for guideOpt below.
        guideType.erase(0, 4);
        if (guideTypeMapping.count(guideType))
        {
            guideTypeOpt = guideTypeMapping[guideType];
        }
    }

    std::optional<std::string> guideOpt = std::nullopt;
    if (inputsTensors.count(InputFieldsNames::guidedDecodingGuide))
    {
        auto guideTensor = inputsTensors.at(InputFieldsNames::guidedDecodingGuide).tensor;

        char* encodedGuide = static_cast<char*>(guideTensor->data());
        guideOpt = std::string(encodedGuide, guideTensor->getSizeInBytes());
        if (guideOpt.has_value())
        {
            guideOpt->erase(0, 4);
        }
    }

    std::optional<executor::GuidedDecodingParams> guidedDecodingParams = std::nullopt;
    if (guideTypeOpt.has_value())
    {
        guidedDecodingParams = executor::GuidedDecodingParams(guideTypeOpt.value(), guideOpt);
    }

    return guidedDecodingParams;
}

std::optional<executor::LookaheadDecodingConfig> getLookaheadDecodingFromTensors(
    InputTensors const& inputsTensors, std::optional<executor::LookaheadDecodingConfig> const& executorLookaheadConfig)
{
    std::optional<executor::LookaheadDecodingConfig> requestLookaheadConfig = std::nullopt;
    if (inputsTensors.count(InputFieldsNames::requestLookaheadDecodingWindowSize))
    {
        executor::SizeType32 windowSize = 0, ngramSize = 0, verificationSetSize = 0;
        if (!utils::extractSingleton<int32_t>(
                inputsTensors, InputFieldsNames::requestLookaheadDecodingWindowSize, windowSize))
        {
            throw std::runtime_error("Failed to extract lookahead_window_size");
        }
        if (!utils::extractSingleton<int32_t>(
                inputsTensors, InputFieldsNames::requestLookaheadDecodingNgramSize, ngramSize))
        {
            throw std::runtime_error("Failed to extract lookahead_ngram_size");
        }
        if (!utils::extractSingleton<int32_t>(
                inputsTensors, InputFieldsNames::requestLookaheadDecodingVerificationSetSize, verificationSetSize))
        {
            throw std::runtime_error("Failed to extract lookahead_verification_set_size");
        }

        requestLookaheadConfig = executor::LookaheadDecodingConfig{windowSize, ngramSize, verificationSetSize};

        TLLM_CHECK_WITH_INFO(executorLookaheadConfig.has_value(),
            "Cannot set the request lookahead decoding configuration when model instance lookahead parameters are not "
            "set.");
    }
    return requestLookaheadConfig;
}

std::vector<executor::Request> createRequestsFromInputTensors(std::vector<InputTensors> const& inputsTensors,
    bool paramExcludeInputFromOutput, bool isDecoupled, bool streaming, executor::ModelType modelType,
    executor::RequestType requestType, bool isOrchestrator, bool specDecFastLogits,
    std::optional<executor::LookaheadDecodingConfig> const& executorLookaheadConfig)
{
    if (!isDecoupled && inputsTensors.size() > 1)
    {
        TLLM_THROW("Batch size > 1 requires the tensorrt_llm backend to be using decoupled transaction policy");
    }

    if (streaming && !isDecoupled)
    {
        TLLM_THROW("Streaming is only supported if model is deployed using decoupled mode.");
    }

    std::vector<executor::Request> requests;
    for (auto const& inputTensors : inputsTensors)
    {
        executor::OutputConfig outConfig = utils::getOutputConfigFromTensors(inputTensors);

        std::optional<bool> reqExcludeInputFromOutput{std::nullopt};
        extractOptionalSingleton<bool>(
            inputTensors, InputFieldsNames::excludeInputFromOutput, reqExcludeInputFromOutput);

        // If specified in request, set from request
        if (reqExcludeInputFromOutput != std::nullopt)
        {
            outConfig.excludeInputFromOutput = reqExcludeInputFromOutput.value();
        }
        else // Set from parameter
        {
            outConfig.excludeInputFromOutput = paramExcludeInputFromOutput;
        }

        executor::VecTokens inputTokens;
        std::optional<executor::Tensor> encoderInputFeatures{std::nullopt};
        if (inputTensors.count(InputFieldsNames::encoderInputFeatures))
        {
            std::shared_ptr<runtime::ITensor> originalTensor
                = inputTensors.at(InputFieldsNames::encoderInputFeatures).tensor;
            utils::squeezeTensor(originalTensor, 2);
            encoderInputFeatures = executor::detail::ofITensor(originalTensor);
        }
        if (!utils::extractVector<int32_t>(inputTensors, InputFieldsNames::inputTokens, inputTokens))
        {
            if (!encoderInputFeatures.has_value())
            {
                TLLM_THROW("%s or %s is not present in the request.", InputFieldsNames::inputTokens,
                    InputFieldsNames::encoderInputFeatures);
            }
        }
        executor::SizeType32 maxNewTokens;
        if (!utils::extractSingleton<int32_t>(inputTensors, InputFieldsNames::maxNewTokens, maxNewTokens))
        {
            throw std::runtime_error("request_output_len is not present in the request");
        }

        std::optional<executor::SizeType32> endId{std::nullopt};
        utils::extractOptionalSingleton<int32_t>(inputTensors, InputFieldsNames::endId, endId);

        std::optional<executor::SizeType32> padId{std::nullopt};
        utils::extractOptionalSingleton<int32_t>(inputTensors, InputFieldsNames::padId, padId);

        std::optional<executor::VecTokens> encoderInputTokens{std::nullopt};
        if (modelType == executor::ModelType::kENCODER_ONLY || modelType == executor::ModelType::kENCODER_DECODER)
        {
            if (inputTensors.count(InputFieldsNames::inputTokens))
            {
                encoderInputTokens = inputTokens;
            }
            // If encoder-decoder, check if decoder tokens are specified
            if (modelType == executor::ModelType::kENCODER_DECODER)
            {
                if (!utils::extractVector<int32_t>(inputTensors, InputFieldsNames::decoderInputTokens, inputTokens))
                {
                    if (padId)
                    {
                        TLLM_LOG_WARNING(
                            "%s is not present in the request for encoder-decoder model. The decoder input tokens will "
                            "be "
                            "set to "
                            "[padId]",
                            InputFieldsNames::decoderInputTokens);
                        inputTokens = {padId.value()};
                    }
                    else
                    {
                        TLLM_LOG_WARNING("%s is not present in the request for encoder-decoder model",
                            InputFieldsNames::decoderInputTokens);
                        inputTokens.clear();
                    }
                }
            }
        }

        auto samplingConfig = utils::getSamplingConfigFromTensors(inputTensors);

        std::optional<std::list<executor::VecTokens>> badWords = std::nullopt;
        executor::VecTokens badWordsRaw;
        if (utils::extractVector<int32_t>(inputTensors, InputFieldsNames::badWords, badWordsRaw))
        {
            badWords = utils::convertWordList(badWordsRaw);
        }

        std::optional<std::list<executor::VecTokens>> stopWords = std::nullopt;
        executor::VecTokens stopWordsRaw;
        if (utils::extractVector<int32_t>(inputTensors, InputFieldsNames::stopWords, stopWordsRaw))
        {
            stopWords = utils::convertWordList(stopWordsRaw);
        }

        std::optional<executor::Tensor> embeddingBias{std::nullopt};
        if (inputTensors.count(InputFieldsNames::embeddingBias))
        {
            std::shared_ptr<runtime::ITensor> originalTensor = inputTensors.at(InputFieldsNames::embeddingBias).tensor;
            utils::squeezeTensor(originalTensor, 1);
            auto newShape = originalTensor->getShape();
            if (!(newShape.nbDims == 1 && newShape.d[0] == 0))
            {
                embeddingBias = executor::detail::ofITensor(originalTensor);
            }
        }

        auto pTuningConfig = utils::getPromptTuningConfigFromTensors(inputTensors, inputTokens.size());

        std::optional<executor::Tensor> multimodalEmbedding{std::nullopt};
        if (inputTensors.count(InputFieldsNames::multimodalEmbedding))
        {
            std::shared_ptr<runtime::ITensor> originalTensor
                = inputTensors.at(InputFieldsNames::multimodalEmbedding).tensor;
            utils::squeezeTensor(originalTensor, 2);
            multimodalEmbedding = executor::detail::ofITensor(originalTensor);
        }

        auto mropeConfig = utils::getMropeConfigFromTensors(inputTensors);

        auto loraConfig = utils::getLoraConfigFromTensors(inputTensors);

        auto kvCacheRetentionConfig = utils::getKvCacheRetentionConfigFromTensors(inputTensors);

        auto externalDraftTokensConfig
            = utils::getExternalDraftTokensConfigFromTensors(inputTensors, specDecFastLogits);

        auto requestLookaheadConfig = getLookaheadDecodingFromTensors(inputTensors, executorLookaheadConfig);

        // Multimodal input construction
        std::optional<executor::MultimodalInput> multimodalInputOpt{std::nullopt};
        if (inputTensors.count(InputFieldsNames::multimodalHashes)
            && inputTensors.count(InputFieldsNames::multimodalPositions)
            && inputTensors.count(InputFieldsNames::multimodalLengths))
        {
            std::vector<std::vector<executor::SizeType32>> multimodalHashes;
            std::vector<executor::SizeType32> multimodalPositions;
            std::vector<executor::SizeType32> multimodalLengths;
            // Extract multimodalHashes as a vector of vectors (2D tensor with [num_tokens, hash_len])
            auto const& hashesTensor = inputTensors.at(InputFieldsNames::multimodalHashes).tensor;
            auto hashesShape = hashesTensor->getShape();
            TLLM_CHECK_WITH_INFO(hashesShape.nbDims == 2, "multimodal_hashes tensor must be 2D (num_tokens, hash_len)");
            int64_t numTokens = hashesShape.d[0];
            int64_t hashLen = hashesShape.d[1];
            auto* data = static_cast<executor::SizeType32*>(hashesTensor->data());
            multimodalHashes.resize(numTokens);
            for (int64_t i = 0; i < numTokens; ++i)
            {
                multimodalHashes[i].resize(hashLen);
                std::memcpy(multimodalHashes[i].data(), data + i * hashLen, hashLen * sizeof(executor::SizeType32));
            }
            // Extract positions and lengths as 1D vectors
            utils::extractVector<executor::SizeType32>(
                inputTensors, InputFieldsNames::multimodalPositions, multimodalPositions);
            utils::extractVector<executor::SizeType32>(
                inputTensors, InputFieldsNames::multimodalLengths, multimodalLengths);
            multimodalInputOpt = executor::MultimodalInput(multimodalHashes, multimodalPositions, multimodalLengths);
        }

        auto request = executor::Request(inputTokens, maxNewTokens, streaming, samplingConfig, outConfig, endId, padId,
            /*positionIds*/ std::nullopt, badWords, stopWords, embeddingBias, externalDraftTokensConfig,
            /*PromptTuningConfig*/ pTuningConfig, /*multimodalInput*/ multimodalInputOpt,
            /*multimodalEmbedding*/ multimodalEmbedding,
            /*MropeConfig*/ mropeConfig, loraConfig, requestLookaheadConfig, kvCacheRetentionConfig,
            /*logitsPostProcessorName*/ std::nullopt, /*logitsPostProcessor*/ std::nullopt, encoderInputTokens);

        if (encoderInputFeatures.has_value())
        {
            executor::Tensor encoderInputFeaturesTensor = encoderInputFeatures.value();
            request.setEncoderInputFeatures(encoderInputFeaturesTensor);
        }

        executor::SizeType32 encoderOutputLengths;
        if (utils::extractSingleton<int32_t>(
                inputTensors, InputFieldsNames::encoderOutputLengths, encoderOutputLengths))
        {
            request.setEncoderOutputLength(encoderOutputLengths);
        }

        request.setRequestType(requestType);
        auto contextPhaseParamsIt = inputTensors.find(InputFieldsNames::contextPhaseParams);
        if (contextPhaseParamsIt != inputTensors.end())
        {
            auto& contextPhaseParams = contextPhaseParamsIt->second();
            InMemoryStreamBuffer buffer(
                reinterpret_cast<char*>(contextPhaseParams->data()), contextPhaseParams->getSize());

            auto requestContextPhase = executor::Serialization::deserializeContextPhaseParams(buffer);
            request.setContextPhaseParams(requestContextPhase);
        }

        if (inputTensors.count(InputFieldsNames::crossAttentionMask))
        {
            std::shared_ptr<runtime::ITensor> originalTensor
                = inputTensors.at(InputFieldsNames::crossAttentionMask).tensor;
            utils::squeezeTensor(originalTensor, 2);
            request.setCrossAttentionMask(executor::detail::ofITensor(originalTensor));
        }

        if (inputTensors.count(InputFieldsNames::skipCrossAttnBlocks))
        {
            std::shared_ptr<runtime::ITensor> originalTensor
                = inputTensors.at(InputFieldsNames::skipCrossAttnBlocks).tensor;
            request.setSkipCrossAttnBlocks(executor::detail::ofITensor(originalTensor));
        }

        auto guidedDecodingParams = utils::getGuidedDecodingParamsFromTensors(inputTensors);
        if (guidedDecodingParams.has_value())
        {
            request.setGuidedDecodingParams(guidedDecodingParams.value());
        }
        requests.emplace_back(std::move(request));
    }
    return requests;
}

template <>
const TRITONSERVER_ParameterType ParameterTypeMap<int32_t>::parameter_type = TRITONSERVER_PARAMETER_INT;

template <>
const TRITONSERVER_ParameterType ParameterTypeMap<std::string>::parameter_type = TRITONSERVER_PARAMETER_STRING;

template <>
const TRITONSERVER_ParameterType ParameterTypeMap<bool>::parameter_type = TRITONSERVER_PARAMETER_BOOL;

template <>
const TRITONSERVER_ParameterType ParameterTypeMap<double>::parameter_type = TRITONSERVER_PARAMETER_DOUBLE;

} // namespace triton::backend::inflight_batcher_llm::utils
