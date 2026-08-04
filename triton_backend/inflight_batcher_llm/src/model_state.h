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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"
#include <atomic>
#include <optional>

using namespace ::triton::common; // TritonJson

namespace triton::backend::inflight_batcher_llm
{

// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.

class ModelState
{
public:
    static TRITONSERVER_Error* Create(
        TRITONBACKEND_Model* triton_model, std::string const& name, uint64_t const version, ModelState** state);

    template <typename T>
    T GetParameter(std::string const& name)
    {
        assert(false);
        auto dummy = T();
        return dummy;
    }

    virtual ~ModelState() = default;

    common::TritonJson::Value& GetModelConfig();
    std::string const& GetModelName() const;
    uint64_t GetModelVersion() const;
    std::string const GetExecutorWorkerPath();

    std::optional<std::vector<std::vector<int32_t>>> getDeviceIds()
    {
        return mGpuDeviceIds;
    }

    std::optional<std::vector<std::vector<int32_t>>> getParticipantIds()
    {
        return mParticipantIds;
    }

    bool IsDecoupled() const
    {
        return is_decoupled_;
    }

    TRITONSERVER_DataType getLogitsDataType() const
    {
        return mLogitsDataType;
    }

    uint32_t getAndIncrementInstanceIndex()
    {
        return mInstanceIndex++;
    }

    [[nodiscard]] std::vector<int64_t> serialize() const;

    static ModelState deserialize(int64_t const* packed_ptr);

    static ModelState deserialize(std::vector<int64_t> const& packed);

private:
    std::string const model_name_;
    uint64_t model_version_;
    common::TritonJson::Value model_config_;
    std::shared_ptr<nvinfer1::ILogger> mTrtLogger{};
    std::atomic<uint32_t> mInstanceIndex{0};

    // model parameters
    std::optional<std::vector<std::vector<int32_t>>> mGpuDeviceIds;
    std::optional<std::vector<std::vector<int32_t>>> mParticipantIds;
    bool is_decoupled_ = false;
    TRITONSERVER_DataType mLogitsDataType = TRITONSERVER_TYPE_INVALID;

    void LoadParameters();

public:
    ModelState(
        TRITONBACKEND_Model* triton_model, std::string const& name, uint64_t version, TritonJson::Value&& model_config)
        : model_name_(name)
        , model_version_(version)
        , model_config_(std::move(model_config))
    {
        mTrtLogger = std::make_shared<tensorrt_llm::runtime::TllmLogger>();
        initTrtLlmPlugins(mTrtLogger.get());

        LoadParameters();
    }
};

template <>
std::string ModelState::GetParameter<std::string>(std::string const& name);

template <>
int32_t ModelState::GetParameter<int32_t>(std::string const& name);

template <>
uint32_t ModelState::GetParameter<uint32_t>(std::string const& name);

template <>
int64_t ModelState::GetParameter<int64_t>(std::string const& name);

template <>
uint64_t ModelState::GetParameter<uint64_t>(std::string const& name);

template <>
float ModelState::GetParameter<float>(std::string const& name);

template <>
bool ModelState::GetParameter<bool>(std::string const& name);

template <>
std::vector<int32_t> ModelState::GetParameter<std::vector<int32_t>>(std::string const& name);

template <>
std::vector<std::vector<int32_t>> ModelState::GetParameter<std::vector<std::vector<int32_t>>>(std::string const& name);

} // namespace triton::backend::inflight_batcher_llm
