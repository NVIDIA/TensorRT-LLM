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

#include "model_state.h"

#include "utils.h"

#include <algorithm>

namespace triton::backend::inflight_batcher_llm
{

TRITONSERVER_Error* ModelState::Create(
    TRITONBACKEND_Model* triton_model, std::string const& name, uint64_t const version, ModelState** state)
{
    TRITONSERVER_Message* config_message;
    RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1 /* config_version */, &config_message));
    // We can get the model configuration as a json string from
    // config_message, parse it with our favorite json parser to create
    // DOM that we can access when we need to example the
    // configuration. We use TritonJson, which is a wrapper that returns
    // nice errors (currently the underlying implementation is
    // rapidjson... but others could be added). You can use any json
    // parser you prefer.
    char const* buffer;
    size_t byte_size;
    RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

    common::TritonJson::Value model_config;
    TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
    RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
    RETURN_IF_ERROR(err);

    try
    {
        *state = new ModelState(triton_model, name, version, std::move(model_config));
    }
    catch (std::exception const& ex)
    {
        std::string errStr = std::string("unexpected error when creating modelState: ") + ex.what();
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errStr.c_str());
    }

    return nullptr; // success
}

void ModelState::LoadParameters()
{
    // Check if model is in decoupled mode:
    triton::common::TritonJson::Value transaction_policy;
    model_config_.MemberAsObject("model_transaction_policy", &transaction_policy);
    transaction_policy.MemberAsBool("decoupled", &is_decoupled_);

    // Retrieve logits datatype
    triton::common::TritonJson::Value outputs;
    model_config_.MemberAsArray("output", &outputs);
    for (size_t i = 0; i < outputs.ArraySize(); ++i)
    {
        triton::common::TritonJson::Value output;
        std::string dtype_str, output_name;
        outputs.IndexAsObject(i, &output);
        output.MemberAsString("name", &output_name);
        if (output_name == "generation_logits" || output_name == "context_logits")
        {
            output.MemberAsString("data_type", &dtype_str);
            dtype_str.erase(0, 5); // Remove the 'TYPE_' prefix
            mLogitsDataType = TRITONSERVER_StringToDataType(dtype_str.c_str());
            break;
        }
    }

    try
    {
        auto gpuDeviceIds = GetParameter<std::string>("gpu_device_ids");

        auto deviceIdsList = utils::split(gpuDeviceIds, ';');

        for (auto const& deviceIds : deviceIdsList)
        {
            if (!mGpuDeviceIds)
            {
                mGpuDeviceIds = std::vector<std::vector<int32_t>>{};
            }
            mGpuDeviceIds.value().emplace_back(utils::csvStrToVecInt(deviceIds));
        }

        if (deviceIdsList.size() > 0)
        {
            auto deviceIdInfo = std::string{"Using GPU device ids: " + gpuDeviceIds};
            TLLM_LOG_INFO(deviceIdInfo);
        }
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("gpu_device_ids is not specified, will be automatically set");
    }

    try
    {
        auto participantIds = GetParameter<std::string>("participant_ids");

        auto participantIdsList = utils::split(participantIds, ';');

        if (!participantIdsList.empty())
        {
            mParticipantIds = std::vector<std::vector<int32_t>>{};
        }
        for (auto const& participantList : participantIdsList)
        {
            mParticipantIds.value().emplace_back(utils::csvStrToVecInt(participantList));
        }

        if (!participantIdsList.empty())
        {
            auto participantIdsInfo = std::string{"Using participant ids: " + participantIds};
            TLLM_LOG_INFO(participantIdsInfo);
        }
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("participant_ids is not specified, will be automatically set");
    }
}

common::TritonJson::Value& ModelState::GetModelConfig()
{
    return model_config_;
}

std::string const& ModelState::GetModelName() const
{
    return model_name_;
}

uint64_t ModelState::GetModelVersion() const
{
    return model_version_;
}

std::string const ModelState::GetExecutorWorkerPath()
{

    // Check if worker_path is specified, if so throw an error
    try
    {
        auto workerPath = GetParameter<std::string>("worker_path");
        TLLM_THROW(
            "worker_path parameter is specified, but this is no longer supported. Please specify executor_worker_path "
            "instead to specify the location of the trtllmExecutorWorker executable.");
    }
    catch (std::exception const& e)
    {
    }

    std::string executorWorkerPath = "/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker";
    try
    {
        executorWorkerPath = GetParameter<std::string>("executor_worker_path");
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("executor_worker_path is not specified, will use default value");
    }

    return executorWorkerPath;
}

std::vector<int64_t> ModelState::serialize() const
{
    // model name
    // model version
    // model config
    size_t totalSize = 3;

    int nameSize = (model_name_.size() + sizeof(int64_t)) / sizeof(int64_t);
    totalSize += nameSize;

    TritonJson::WriteBuffer buffer;
    model_config_.Write(&buffer);

    totalSize += buffer.Size();

    std::vector<int64_t> packed(totalSize);
    int64_t* ptr = packed.data();

    *ptr++ = model_name_.size();
    std::memcpy(ptr, model_name_.c_str(), model_name_.size());
    ptr += nameSize;

    *ptr++ = model_version_;
    *ptr++ = buffer.Size();
    std::memcpy(ptr, buffer.Base(), buffer.Size());

    return packed;
}

ModelState ModelState::deserialize(int64_t const* packed_ptr)
{
    auto const nameSize = *packed_ptr++;
    char const* cname = reinterpret_cast<char const*>(packed_ptr);
    packed_ptr += (nameSize + sizeof(int64_t)) / sizeof(int64_t);

    uint64_t const version = *packed_ptr++;

    auto const jsonSize = *packed_ptr++;
    char const* jsonBuffer = reinterpret_cast<char const*>(packed_ptr);
    common::TritonJson::Value model_config;
    TRITONSERVER_Error* err = model_config.Parse(jsonBuffer, jsonSize);
    if (err)
    {
        TRITONSERVER_ErrorDelete(err);
        throw std::runtime_error("Failed to parse model config");
    }

    return ModelState{nullptr, cname, version, std::move(model_config)};
}

ModelState ModelState::deserialize(std::vector<int64_t> const& packed)
{
    return ModelState::deserialize(packed.data());
}

template <>
std::string ModelState::GetParameter<std::string>(std::string const& name)
{
    TritonJson::Value parameters;
    TRITONSERVER_Error* err = model_config_.MemberAsObject("parameters", &parameters);
    if (err != nullptr)
    {
        TRITONSERVER_ErrorDelete(err);
        throw std::runtime_error("Model config doesn't have a parameters section");
    }
    TritonJson::Value value;
    std::string str_value;
    err = parameters.MemberAsObject(name.c_str(), &value);
    if (err != nullptr)
    {
        TRITONSERVER_ErrorDelete(err);
        std::string errStr = "Cannot find parameter with name: " + name;
        throw std::runtime_error(errStr);
    }
    value.MemberAsString("string_value", &str_value);
    return str_value;
}

template <>
int32_t ModelState::GetParameter<int32_t>(std::string const& name)
{
    return std::stoi(GetParameter<std::string>(name));
}

template <>
std::vector<int32_t> ModelState::GetParameter<std::vector<int32_t>>(std::string const& name)
{
    auto deviceIdsStr = GetParameter<std::string>(name);
    // Parse as comma delimited string
    return utils::csvStrToVecInt(deviceIdsStr);
}

template <>
uint32_t ModelState::GetParameter<uint32_t>(std::string const& name)
{
    return (uint32_t) std::stoul(GetParameter<std::string>(name));
}

template <>
int64_t ModelState::GetParameter<int64_t>(std::string const& name)
{
    return std::stoll(GetParameter<std::string>(name));
}

template <>
uint64_t ModelState::GetParameter<uint64_t>(std::string const& name)
{
    return std::stoull(GetParameter<std::string>(name));
}

template <>
float ModelState::GetParameter<float>(std::string const& name)
{
    return std::stof(GetParameter<std::string>(name));
}

template <>
bool ModelState::GetParameter<bool>(std::string const& name)
{
    auto val = GetParameter<std::string>(name);
    if (val == "True" || val == "true" || val == "TRUE" || val == "1")
    {
        return true;
    }
    else if (val == "False" || val == "false" || val == "FALSE" || val == "0")
    {
        return false;
    }
    else
    {
        std::string err = "Cannot convert " + val + " to a boolean.";
        throw std::runtime_error(err);
    }
}

template <>
std::vector<std::vector<int32_t>> ModelState::GetParameter<std::vector<std::vector<int32_t>>>(std::string const& name)
{
    auto str = GetParameter<std::string>(name);
    // Parse as comma delimited string and {} as array bounders
    return utils::csvStrToVecVecInt(str);
}

} // namespace triton::backend::inflight_batcher_llm
