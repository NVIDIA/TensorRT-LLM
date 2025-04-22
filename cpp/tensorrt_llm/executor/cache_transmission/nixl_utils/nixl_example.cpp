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
#include <cassert>
#include <iostream>

#include <sys/time.h>

#include "nixl.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

std::string agent1("Agent001");
std::string agent2("Agent002");

void check_buf(void* buf, size_t len)
{

    // Do some checks on the data.
    for (size_t i = 0; i < len; i++)
    {
        assert(((uint8_t*) buf)[i] == 0xbb);
    }
}

bool equal_buf(void* buf1, void* buf2, size_t len)
{

    // Do some checks on the data.
    for (size_t i = 0; i < len; i++)
        if (((uint8_t*) buf1)[i] != ((uint8_t*) buf2)[i])
            return false;
    return true;
}

void printParams(nixl_b_params_t const& params, nixl_mem_list_t const& mems)
{
    if (params.empty())
    {
        std::cout << "Parameters: (empty)" << std::endl;
        return;
    }

    std::cout << "Parameters:" << std::endl;
    for (auto const& pair : params)
    {
        std::cout << "  " << pair.first << " = " << pair.second << std::endl;
    }

    if (mems.empty())
    {
        std::cout << "Mems: (empty)" << std::endl;
        return;
    }

    std::cout << "Mems:" << std::endl;
    for (auto const& elm : mems)
    {
        std::cout << "  " << nixlEnumStrings::memTypeStr(elm) << std::endl;
    }
}

int main()
{
    nixl_status_t ret1, ret2;
    std::string ret_s1, ret_s2;

    // Example: assuming two agents running on the same machine,
    // with separate memory regions in DRAM

    nixlAgentConfig cfg(true);
    nixl_b_params_t init1, init2;
    nixl_mem_list_t mems1, mems2;

    // populate required/desired inits
    nixlAgent A1(agent1, cfg);
    nixlAgent A2(agent2, cfg);

    std::vector<nixl_backend_t> plugins;

    ret1 = A1.getAvailPlugins(plugins);
    assert(ret1 == NIXL_SUCCESS);

    std::cout << "Available plugins:\n";

    for (nixl_backend_t b : plugins)
        std::cout << b << "\n";

    ret1 = A1.getPluginParams("UCX", mems1, init1);
    ret2 = A2.getPluginParams("UCX", mems2, init2);

    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    std::cout << "Params before init:\n";
    printParams(init1, mems1);
    printParams(init2, mems2);

    nixlBackendH *ucx1, *ucx2;
    ret1 = A1.createBackend("UCX", init1, ucx1);
    ret2 = A2.createBackend("UCX", init2, ucx2);

    nixl_opt_args_t extra_params1, extra_params2;
    extra_params1.backends.push_back(ucx1);
    extra_params2.backends.push_back(ucx2);

    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    ret1 = A1.getBackendParams(ucx1, mems1, init1);
    ret2 = A2.getBackendParams(ucx2, mems2, init2);

    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    std::cout << "Params after init:\n";
    printParams(init1, mems1);
    printParams(init2, mems2);

    // // One side gets to listen, one side to initiate. Same string is passed as the last 2 steps
    // ret1 = A1->makeConnection(agent2, 0);
    // ret2 = A2->makeConnection(agent1, 1);

    // assert (ret1 == NIXL_SUCCESS);
    // assert (ret2 == NIXL_SUCCESS);

    // User allocates memories, and passes the corresponding address
    // and length to register with the backend
    nixlBlobDesc buff1, buff2, buff3;
    nixl_reg_dlist_t dlist1(DRAM_SEG), dlist2(DRAM_SEG);
    size_t len = 256;
    void* addr1 = calloc(1, len);
    void* addr2 = calloc(1, len);

    memset(addr1, 0xbb, len);
    memset(addr2, 0, len);

    buff1.addr = (uintptr_t) addr1;
    buff1.len = len;
    buff1.devId = 0;
    dlist1.addDesc(buff1);

    buff2.addr = (uintptr_t) addr2;
    buff2.len = len;
    buff2.devId = 0;
    dlist2.addDesc(buff2);

    // dlist1.print();
    // dlist2.print();

    // sets the metadata field to a pointer to an object inside the ucx_class
    ret1 = A1.registerMem(dlist1, &extra_params1);
    ret2 = A2.registerMem(dlist2, &extra_params2);

    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    std::string meta1;
    ret1 = A1.getLocalMD(meta1);
    std::string meta2;
    ret2 = A2.getLocalMD(meta2);

    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    std::cout << "Agent1's Metadata: " << meta1 << "\n";
    std::cout << "Agent2's Metadata: " << meta2 << "\n";

    ret1 = A1.loadRemoteMD(meta2, ret_s1);

    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    size_t req_size = 8;
    size_t dst_offset = 8;

    nixl_xfer_dlist_t req_src_descs(DRAM_SEG);
    nixlBasicDesc req_src;
    req_src.addr = (uintptr_t) (((char*) addr1) + 16); // random offset
    req_src.len = req_size;
    req_src.devId = 0;
    req_src_descs.addDesc(req_src);

    nixl_xfer_dlist_t req_dst_descs(DRAM_SEG);
    nixlBasicDesc req_dst;
    req_dst.addr = (uintptr_t) ((char*) addr2) + dst_offset; // random offset
    req_dst.len = req_size;
    req_dst.devId = 0;
    req_dst_descs.addDesc(req_dst);

    std::cout << "Transfer request from " << addr1 << " to " << addr2 << "\n";
    nixlXferReqH* req_handle;

    extra_params1.notifMsg = "notification";
    extra_params1.hasNotif = true;
    ret1 = A1.createXferReq(NIXL_WRITE, req_src_descs, req_dst_descs, agent2, req_handle, &extra_params1);
    assert(ret1 == NIXL_SUCCESS);

    nixl_status_t status = A1.postXferReq(req_handle);

    std::cout << "Transfer was posted\n";

    nixl_notifs_t notif_map;
    int n_notifs = 0;

    while (status != NIXL_SUCCESS || n_notifs == 0)
    {
        if (status != NIXL_SUCCESS)
            status = A1.getXferStatus(req_handle);
        if (n_notifs == 0)
            ret2 = A2.getNotifs(notif_map);
        assert(status >= 0);
        assert(ret2 == NIXL_SUCCESS);
        n_notifs = notif_map.size();
    }

    std::vector<std::string> agent1_notifs = notif_map[agent1];
    assert(agent1_notifs.size() == 1);
    assert(agent1_notifs.front() == "notification");
    notif_map[agent1].clear(); // Redundant, for testing
    notif_map.clear();
    n_notifs = 0;

    std::cout << "Transfer verified\n";

    ret1 = A1.releaseXferReq(req_handle);
    assert(ret1 == NIXL_SUCCESS);

    ret1 = A1.deregisterMem(dlist1, &extra_params1);
    ret2 = A2.deregisterMem(dlist2, &extra_params2);
    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    // only initiator should call invalidate
    ret1 = A1.invalidateRemoteMD(agent2);
    assert(ret1 == NIXL_SUCCESS);

    free(addr1);
    free(addr2);

    std::cout << "Test done\n";
}
