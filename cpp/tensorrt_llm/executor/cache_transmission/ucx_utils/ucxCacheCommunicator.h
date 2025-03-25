/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h" //TODO: remove when progressing to standalone UCX stack

#include "ucxx/api.h"
#include "ucxx/utils/sockaddr.h"
#include "ucxx/utils/ucx.h"
#if __linux__
#include <arpa/inet.h>
#include <ifaddrs.h>
#endif
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/ucx_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/ucx_utils/ucxCacheCommunicator.h"

namespace tensorrt_llm::executor::kv_cache
{

class UcxConnectionManager : public ConnectionManager, public std::enable_shared_from_this<UcxConnectionManager>
{
public:
    UcxConnectionManager(mpi::MpiComm const* comm);
    void addConnection(ucp_conn_request_h conn_request);

    Connection const* recvConnect(DataContext const& ctx, void* data, size_t size) override
    {
        // Guard to ensure CUDA context is initialized for UCX ops
        TLLM_CUDA_CHECK(cudaFree(0));
        // TLLM_CHECK_WITH_INFO((mEndpoint), "recvBuffer called without established communicator channel.");
        uint64_t sender_tag;
        ucp_request_param_t tag_recv_params;
        tag_recv_params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
        tag_recv_params.cb.recv = [](void* request, ucs_status_t status, ucp_tag_recv_info_t const* info,
                                      void* user_data) -> void { *(uint64_t*) user_data = info->sender_tag; };
        tag_recv_params.user_data = &sender_tag;
        // auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
        auto request = ucp_tag_recv_nbx(mWorkersPool.front().get()->getHandle(), data, size, 1, 0, &tag_recv_params);
        // auto req = mWorkersPool->tagRecv(data, size, ucxx::Tag(ctx.getTag()), 0, false, completionCallback);
        // std::unique_lock<std::mutex> lk(mMtx);
        // mCv.wait(lk, [&req]() { return req->isCompleted(); });
        // throw if there is error
        // req->checkError();
        while (ucp_request_check_status(request) != UCS_INPROGRESS)
            ;

        return mConnections[sender_tag].get();
    }

    std::vector<Connection const*> getConnections(CommState const& state) override
    {
        std::vector<Connection const*> ret;
        for (auto const& [id, ucxConnection] : mConnections)
        {
            ret.emplace_back(ucxConnection.get());
        }
        return ret;
    }

private:
    mpi::MpiComm const* mComm;
    std::shared_ptr<ucxx::Context> mUcxCtx;
    std::vector<std::shared_ptr<ucxx::Worker>> mWorkersPool;
    std::map<uint64_t, std::shared_ptr<UcxConnection>> mConnections;
    std::shared_ptr<ucxx::Listener> mListener;

    uint64_t getNewConnectionId(std::shared_ptr<ucxx::Endpoint> newEp)
    {
        ucp_ep_attr_t ep_attr;
        ep_attr.field_mask = UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR | UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR;
        uint64_t remotePort, localPort;
        uint32_t remoteIp;
        char lIpStr[INET6_ADDRSTRLEN];
        char rIpStr[INET6_ADDRSTRLEN];
        char portStr[INET6_ADDRSTRLEN];
        ucs_status_t status = ucp_ep_query(newEp->getHandle(), &ep_attr);
        if (status == UCS_OK)
        {

            ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.remote_sockaddr, rIpStr, portStr, INET6_ADDRSTRLEN);
            remotePort = static_cast<ucxx::Tag>(std::stoull(portStr));
            remoteIp = std::stoull(lIpStr);
            ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.local_sockaddr, lIpStr, portStr, INET6_ADDRSTRLEN);
            localPort = static_cast<ucxx::Tag>(std::stoull(portStr));

            return ((remotePort << (32 + 16)) | (localPort << 32) | remoteIp);
        }
        else
        {
            // [FIXME] better message
            if (status == UCS_ERR_NOT_CONNECTED)
            {
                TLLM_LOG_ERROR("UCX connection has not been established yet");
            }
        }
        return 0;
        // return reinterpret_cast<uint64_t>(newEp.get()->getHandle());
    }

    void addConnection(std::string Ip);
};
} // namespace tensorrt_llm::executor::kv_cache
