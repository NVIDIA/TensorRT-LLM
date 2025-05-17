/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"

namespace tensorrt_llm::executor::kv_cache
{

[[nodiscard]] nixl_mem_t NixlHelper::convert(MemoryType type)
{
    switch (type)
    {
    case MemoryType::kDRAM: return DRAM_SEG;
    case MemoryType::kVRAM: return VRAM_SEG;
    case MemoryType::kBLK: return BLK_SEG;
    case MemoryType::kOBJ: return OBJ_SEG;
    case MemoryType::kFILE: return FILE_SEG;
    default: TLLM_THROW("Unknown MemoryType value");
    }
}

[[nodiscard]] nixlBasicDesc NixlHelper::convert(MemoryDesc const& desc)
{
    return nixlBasicDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()};
}

[[nodiscard]] nixl_reg_dlist_t NixlHelper::convertRegDlist(RegisterDescs const& descs)
{
    nixl_reg_dlist_t list{convert(descs.getType())};
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBlobDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()});
    }
    return list;
}

[[nodiscard]] nixl_xfer_op_t NixlHelper::convert(TransferOp const& op)
{
    switch (op)
    {
    case TransferOp::kREAD: return NIXL_READ;
    case TransferOp::kWRITE: return NIXL_WRITE;
    default: TLLM_THROW("Unknown TransferOp value");
    }
}

[[nodiscard]] nixl_xfer_dlist_t NixlHelper::convertXferDist(TransferDescs const& descs)
{
    nixl_xfer_dlist_t list{convert(descs.getType())};
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBasicDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()});
    }
    return list;
}

NixlTransferStatus::NixlTransferStatus(nixlAgent* agent, nixlXferReqH* handle)
    : mRawAgent{agent}
    , mHandle{handle}
{
    TLLM_CHECK(mRawAgent);
    TLLM_CHECK(mHandle);
}

void NixlTransferStatus::wait() const
{
    while (!isCompleted())
        ;
}

[[nodiscard]] bool NixlTransferStatus::isCompleted() const
{
    return mRawAgent->getXferStatus(mHandle) == NIXL_SUCCESS;
}

NixlTransferAgent::NixlTransferAgent(BaseAgentConfig const& config, AgentRegistrar* registrar)
    : mRegistrar{registrar}
    , mName{config.mName}
{
    nixl_status_t status;
    TLLM_CHECK(mRegistrar);
    nixlAgentConfig nixlConfig{config.useProgThread};
    mRawAgent = std::make_unique<nixlAgent>(config.mName, std::move(nixlConfig));

    nixl_b_params_t init1;
    nixl_mem_list_t mems1;
    status = mRawAgent->getPluginParams("UCX", mems1, init1);
    TLLM_CHECK(status == NIXL_SUCCESS);

    status = mRawAgent->createBackend("UCX", init1, mRawBackend);
    if (status != NIXL_SUCCESS || !mRawBackend)
    {
        TLLM_THROW("Failed to create NIXL backend");
    }
    mExtraParams.backends.push_back(mRawBackend);
}

void NixlTransferAgent::registerMemory(RegisterDescs const& descs)
{
    nixl_status_t status;
    status = mRawAgent->registerMem(NixlHelper::convertRegDlist(descs), &mExtraParams);
    TLLM_CHECK(status == NIXL_SUCCESS);

    std::string localMD;
    status = mRawAgent->getLocalMD(localMD);
    TLLM_CHECK(status == NIXL_SUCCESS);
    mRegistrar->addAgentDesc(mName.c_str(), std::vector<char>(localMD.begin(), localMD.end()));
}

void NixlTransferAgent::deregisterMemory(RegisterDescs const& descs)
{
    nixl_status_t status;
    status = mRawAgent->deregisterMem(NixlHelper::convertRegDlist(descs), &mExtraParams);
    TLLM_CHECK(status == NIXL_SUCCESS);
}

void NixlTransferAgent::loadRemoteAgent(char const* name)
{
    nixl_status_t status;
    auto const* desc = mRegistrar->getAgentDesc(name);
    TLLM_CHECK(desc);
    std::string remoteName;
    auto backendDesc = desc->getBackendAgentDesc();
    status = mRawAgent->loadRemoteMD(std::string(backendDesc.begin(), backendDesc.end()), remoteName);
    TLLM_CHECK(status == NIXL_SUCCESS);
    TLLM_CHECK_WITH_INFO(
        name == remoteName, "loadRemoteAgent gets error agent name: %s != %s", name, remoteName.c_str());
}

void NixlTransferAgent::invalidateRemoteAgent(char const* name)
{
    mRawAgent->invalidateRemoteMD(name);
}

[[nodiscard]] std::unique_ptr<TransferStatus> NixlTransferAgent::submitTransferRequests(TransferRequest const& request)
{
    nixl_status_t status;
    nixlXferReqH* handle;
    status = mRawAgent->createXferReq(NixlHelper::convert(request.getOp()),
        NixlHelper::convertXferDist(request.getSrcDescs()), NixlHelper::convertXferDist(request.getDstDescs()),
        request.getRemoteName(), handle, &mExtraParams);
    TLLM_CHECK(status == NIXL_SUCCESS);

    status = mRawAgent->postXferReq(handle, &mExtraParams);
    return std::make_unique<NixlTransferStatus>(mRawAgent.get(), handle);
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    std::unique_ptr<BaseTransferAgent> createNixlTransferAgent(BaseAgentConfig const* config, AgentRegistrar* registrar)
    {
        TLLM_CHECK(config);
        return std::make_unique<NixlTransferAgent>(*config, registrar);
    }
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache
