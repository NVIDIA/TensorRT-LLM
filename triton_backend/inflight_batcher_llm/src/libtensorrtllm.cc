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

#include <cassert>
#include <chrono>
#include <fstream>
#include <list>
#include <memory>
#include <thread>

// Triton headers
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

// trtllm backend headers
#include "model_instance_state.h"
#include "model_state.h"

namespace triton::backend::inflight_batcher_llm
{

extern "C"
{

    TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
    {
        char const* cname;
        RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
        std::string name(cname);

        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

        // We should check the backend API version that Triton supports
        // vs. what this backend was compiled against.
        uint32_t api_version_major, api_version_minor;
        RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
            (std::string("Triton TRITONBACKEND API version: ") + std::to_string(api_version_major) + "."
                + std::to_string(api_version_minor))
                .c_str());
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
            (std::string("'") + name + "' TRITONBACKEND API version: " + std::to_string(TRITONBACKEND_API_VERSION_MAJOR)
                + "." + std::to_string(TRITONBACKEND_API_VERSION_MINOR))
                .c_str());

        if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR)
            || (api_version_minor < TRITONBACKEND_API_VERSION_MINOR))
        {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED, "triton backend API version does not support this backend");
        }

        // The backend configuration may contain information needed by the
        // backend, such as command-line arguments.
        TRITONSERVER_Message* backend_config_message;
        RETURN_IF_ERROR(TRITONBACKEND_BackendConfig(backend, &backend_config_message));

        char const* buffer;
        size_t byte_size;
        RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message, &buffer, &byte_size));
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("backend configuration:\n") + buffer).c_str());

        return nullptr; // success
    }

    // Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
    // to allow the backend to create any state associated with the model,
    // and to also examine the model configuration to determine if the
    // configuration is suitable for the backend. Any errors reported by
    // this function will prevent the model from loading.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
    {
        // Create a ModelState object and associate it with the
        // TRITONBACKEND_Model. If anything goes wrong with initialization
        // of the model state then an error is returned and Triton will fail
        // to load the model.
        char const* cname;
        RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
        const std::string name(cname);

        uint64_t version;
        RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

        ModelState* model_state;
        RETURN_IF_ERROR(ModelState::Create(model, name, version, &model_state));
        RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
            (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " + std::to_string(version) + ")")
                .c_str());

        return nullptr; // success
    }

    // Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
    // needed. The backend should cleanup any state associated with the
    // model. This function will not be called until all model instances
    // of the model have been finalized.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
    {
        void* vstate;
        RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
        ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
        delete model_state;

        return nullptr; // success
    }

    // Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
    // instance is created to allow the backend to initialize any state
    // associated with the instance.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
    {
        // Get the model state associated with this instance's model.
        TRITONBACKEND_Model* model;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

        void* vmodelstate;
        RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
        ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

        // Create a ModelInstanceState object and associate it with the
        // TRITONBACKEND_ModelInstance.
        ModelInstanceState* instance_state;
        RETURN_IF_ERROR(ModelInstanceState::Create(model_state, instance, &instance_state));
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(instance_state)));

        char const* cname;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
        std::string name(cname);

        int32_t device_id;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

        std::string gpus = "";
        if (instance_state->getGpuDeviceIds())
        {
            for (auto gpu : instance_state->getGpuDeviceIds().value())
            {
                gpus += std::to_string(gpu) + ",";
            }
            // Remove the last comma.
            if (!gpus.empty())
            {
                gpus = gpus.substr(0, gpus.size() - 1);
            }
        }

        if (!gpus.empty())
        {
            LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (devices: " + gpus + ").").c_str());
        }
        else
        {
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name).c_str());
        }

        return nullptr; // success
    }

    // Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
    // instance is no longer needed. The backend should cleanup any state
    // associated with the model instance.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
    {
        TRITONBACKEND_Model* model;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

        void* vstate;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
        ModelInstanceState* instance_state = reinterpret_cast<ModelInstanceState*>(vstate);
        delete instance_state;

        return nullptr; // success
    }

    // When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
    // that a backend create a response for each request in the batch. A
    // response may be the output tensors required for that request or may
    // be an error that is returned in the response.
    //
    TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
        TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests, const uint32_t request_count)
    {
        ModelInstanceState* instance_state;
        RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&instance_state)));

        instance_state->enqueue(requests, request_count);

        return nullptr; // success
    }

} // extern "C"

} // namespace triton::backend::inflight_batcher_llm
