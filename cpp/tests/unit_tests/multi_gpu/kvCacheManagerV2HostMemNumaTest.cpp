/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! Placement of host-tier pages across NUMA nodes.
//!
//! Lives with the multi-GPU tests because it needs two GPUs on different NUMA
//! nodes to say anything: with one node, or with every GPU on the same node,
//! page placement is whatever first touch would have produced and the case
//! cannot distinguish a working policy from a missing one.

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMemBacking.h"

#include "tensorrt_llm/runtime/moeLoadBalancer/topologyDetector.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <numa.h>
#include <numaif.h>

#include <cstdint>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

//! Writes a per-offset pattern, which is also what faults the pages in.
void stamp(MemAddress base, size_t offset, size_t size, uint8_t salt)
{
    auto* bytes = reinterpret_cast<uint8_t*>(base + offset);
    for (size_t i = 0; i < size; ++i)
    {
        bytes[i] = static_cast<uint8_t>((offset + i + salt) & 0xFF);
    }
}

//! Reports the NUMA node each page of [base, base + size) actually landed on.
//!
//! MPOL_F_NODE together with MPOL_F_ADDR turns get_mempolicy() into a placement
//! query: it yields the node the page at that address is on, not the policy the
//! range carries. move_pages() answers the same question but is unavailable
//! here, since a container's default seccomp profile rejects it.
std::vector<int> pageNodes(MemAddress base, size_t size, size_t stride)
{
    std::vector<int> nodes;
    for (size_t offset = 0; offset < size; offset += stride)
    {
        int node = -1;
        if (::get_mempolicy(&node, nullptr, 0, reinterpret_cast<void*>(base + offset), MPOL_F_NODE | MPOL_F_ADDR) != 0)
        {
            return {};
        }
        nodes.push_back(node);
    }
    return nodes;
}

//! Physical pages must land on the node the GPU attaches to, not on the node
//! whichever thread faulted them happens to run on.
//!
//! The two are deliberately different here: the thread is pinned to one node and
//! the GPU is chosen from the other. Without a policy on the range, first touch
//! would place every page on the thread's node, so this case can actually fail.
//! Where no such pair of nodes exists there is nothing to distinguish and the
//! case skips rather than passing vacuously.
//!
//! Parametrized on allowRemoteNumaFallback. The two modes place identically
//! while the node has room, so placement alone cannot tell them apart; the
//! policy in effect is read back to check the flag reached the kernel.
class HostMemNumaTest : public ::testing::TestWithParam<bool>
{
};

TEST_P(HostMemNumaTest, MmapPagesLandOnTheGpusNode)
{
    bool const allowRemoteFallback = GetParam();
    if (numa_available() < 0 || numa_max_node() < 1)
    {
        GTEST_SKIP() << "needs more than one NUMA node";
    }

    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);

    // A GPU whose node differs from the node this thread will be pinned to.
    int localNode = -1;
    int remoteDevice = -1;
    int remoteNode = -1;
    for (int device = 0; device < deviceCount; ++device)
    {
        ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        int const node = tensorrt_llm::runtime::TopologyDetector::getInstance().getCurrentGpuNumaId();
        if (node < 0)
        {
            continue;
        }
        if (localNode < 0)
        {
            localNode = node;
        }
        else if (node != localNode)
        {
            remoteDevice = device;
            remoteNode = node;
            break;
        }
    }
    if (remoteDevice < 0)
    {
        GTEST_SKIP() << "every GPU reports the same NUMA node";
    }

    // Fault from the other node, so an unbound range would land there.
    ASSERT_EQ(numa_run_on_node(localNode), 0);
    ASSERT_EQ(cudaSetDevice(remoteDevice), cudaSuccess);
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);

    constexpr size_t kSize = size_t{8} << 20;
    HostMemBackingOptions options;
    options.allowRemoteNumaFallback = allowRemoteFallback;
    auto backing = createHostMemBacking(HostMemBackingKind::kMmap, options);
    MemAddress const base = backing->reserve(kSize);
    ASSERT_NE(base, 0U);

    // The flag picks between a strict binding and a preference. Both place the
    // same way while the node has room, so the mode itself is what distinguishes
    // them.
    int mode = -1;
    ASSERT_EQ(::get_mempolicy(&mode, nullptr, 0, reinterpret_cast<void*>(base), MPOL_F_ADDR), 0);
    EXPECT_EQ(mode, allowRemoteFallback ? MPOL_PREFERRED : MPOL_BIND);

    backing->commit(0, kSize);
    stamp(base, 0, kSize, 0);

    auto const nodes = pageNodes(base, kSize, size_t{2} << 20);
    ASSERT_FALSE(nodes.empty()) << "page placement query failed";
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        EXPECT_EQ(nodes[i], remoteNode) << "page " << i << " landed on node " << nodes[i]
                                        << " rather than the node of GPU " << remoteDevice;
    }
}

INSTANTIATE_TEST_SUITE_P(NumaPolicies, HostMemNumaTest, ::testing::Values(false, true),
    [](::testing::TestParamInfo<bool> const& info) { return info.param ? "RemoteFallback" : "Strict"; });

} // namespace
