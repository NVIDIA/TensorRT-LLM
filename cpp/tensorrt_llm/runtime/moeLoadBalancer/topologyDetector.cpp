/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/moeLoadBalancer/topologyDetector.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <algorithm> // For std::for_each, std::sort, std::unique
#include <filesystem>
#include <fstream>
#include <limits> // For std::numeric_limits
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#ifdef __linux__
#include <cerrno>   // For errno
#include <cstring>  // For strerror
#include <numa.h>   // For libnuma
#include <numaif.h> // For struct bitmask definition if not in numa.h
#include <pthread.h>
#include <sched.h>
#endif

namespace tensorrt_llm::runtime
{

TopologyDetector::TopologyDetector()
{
    std::lock_guard<std::mutex> lock(mDetectionMutex);
    if (!mTopologyDetected)
    {
        detectCpuTopology();
        detectGpuTopology();
#ifdef __linux__
        if (numa_available() != -1)
        { // Only precompute if libnuma is usable
            precomputeCpuAffinityMasks();
        }
#endif
        mTopologyDetected = true;
    }
}

TopologyDetector::~TopologyDetector()
{
#ifdef __linux__
    auto free_mask_map = [](std::map<int, struct bitmask*>& mask_map)
    {
        for (auto const& [id, mask] : mask_map)
        {
            if (mask)
            {
                numa_free_cpumask(mask);
            }
        }
        mask_map.clear();
    };
    free_mask_map(mGpuStrictCpuMasks);
#endif
}

void TopologyDetector::detectCpuTopology()
{
    // Detect CPU architecture
#if defined(__x86_64__) || defined(_M_X64)
    mCpuArchitecture = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    mCpuArchitecture = "aarch64";
#elif defined(__powerpc64__)
    mCpuArchitecture = "ppc64";
#else
    mCpuArchitecture = "unknown";
#endif

    // Detect NUMA topology on Linux systems using libnuma
#ifdef __linux__
    if (numa_available() == -1)
    {
        // libnuma not available, fall back to default behavior
        TLLM_LOG_WARNING("libnuma not available. Falling back to default CPU topology detection.");
        mNumaToCpuCountMap[0] = std::thread::hardware_concurrency();
        return;
    }

    int maxNode = numa_max_node();
    if (maxNode < 0)
    {
        // Failed to get max node, fall back to default behavior
        TLLM_LOG_WARNING("Failed to get max NUMA node. Falling back to default CPU topology detection.");
        mNumaToCpuCountMap[0] = std::thread::hardware_concurrency();
        return;
    }

    mNumaToCpuCountMap.clear(); // Clear before re-populating
    std::map<int, int> tempNumaToCpuCountMap;
    for (int i = 0; i <= maxNode; ++i)
    {
        struct bitmask* cpus = numa_allocate_cpumask();
        if (!cpus)
        {
            TLLM_LOG_WARNING("Failed to allocate cpumask for NUMA node query. Skipping node %d.", i);
            continue; // Skip to the next node if allocation fails
        }

        // Attempt to get CPUs for node i. If numa_node_to_cpus returns 0, it's successful.
        if (numa_node_to_cpus(i, cpus) == 0)
        {
            int cpuCount = 0;
            for (int cpu_idx = 0; cpu_idx < numa_num_possible_cpus(); ++cpu_idx)
            {
                if (numa_bitmask_isbitset(cpus, cpu_idx))
                {
                    cpuCount++;
                }
            }
            if (cpuCount > 0)
            { // Only add NUMA nodes with actual CPUs
                tempNumaToCpuCountMap[i] = cpuCount;
            }
        }
        // If numa_node_to_cpus failed (returned -1), node 'i' might be invalid or an error occurred.
        // In this case, we simply don't add it to our map, effectively skipping it.

        numa_free_cpumask(cpus); // Always free the allocated mask
    }
    mNumaToCpuCountMap = tempNumaToCpuCountMap;

    if (mNumaToCpuCountMap.empty())
    {
        // If no NUMA nodes with CPUs were detected (e.g. libnuma error or unusual configuration),
        // default to a single NUMA node with all hardware concurrency.
        TLLM_LOG_WARNING(
            "No NUMA nodes with CPUs detected via libnuma, or libnuma error. Defaulting to single NUMA node.");
        mNumaToCpuCountMap[0] = std::thread::hardware_concurrency();
    }

#else
    // For non-Linux systems, assume a single NUMA node
    mNumaToCpuCountMap[0] = std::thread::hardware_concurrency();
#endif
}

void TopologyDetector::detectGpuTopology()
{
    int deviceCount = 0;
    cudaError_t result = cudaGetDeviceCount(&deviceCount);
    if (result != cudaSuccess || deviceCount == 0)
    {
        return;
    }
    mGpuToNumaMap.clear();       // Clear before re-populating
    mGpuMemoryToNumaMap.clear(); // Clear before re-populating
    mNumaToGpuMap.clear();       // Clear before re-populating

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
    {
        int numaNode = 0;        // Default NUMA node
        int numaMemoryNode = -1; // Default Memory NUMA node

#ifdef __linux__
        if (numa_available() != -1)
        {
            char pciPath[256];
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess)
            {
                // Construct PCI path to find NUMA node
                snprintf(pciPath, sizeof(pciPath), "/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node", prop.pciDomainID,
                    prop.pciBusID, prop.pciDeviceID);
                std::ifstream numaFile(pciPath);
                if (numaFile.is_open())
                {
                    numaFile >> numaNode;
                    numaFile.close();
                    // If NUMA node is -1, it means no specific NUMA information, use node 0
                    if (numaNode < 0)
                    {
                        numaNode = 0;
                    }
                }
                else
                {
                    // Fallback if sysfs path is not available or readable
                    TLLM_LOG_DEBUG("Could not open %s to determine NUMA node for GPU %d. Defaulting to node 0.",
                        pciPath, deviceId);
                    numaNode = 0;
                }
                TLLM_LOG_INFO("GPU %d is on NUMA node %d", deviceId, numaNode);
            }
            else
            {
                TLLM_LOG_WARNING("Failed to get properties for GPU %d. Defaulting to NUMA node 0.", deviceId);
                numaNode = 0;
            }
        }
        else
        {
            // libnuma not available, default GPU to NUMA node 0
            numaNode = 0;
        }
        int hasMemoryNumaConfig = 0;
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&hasMemoryNumaConfig, cudaDevAttrNumaConfig, deviceId));
        if (hasMemoryNumaConfig == cudaDeviceNumaConfigNumaNode)
        {
            TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&numaMemoryNode, cudaDevAttrNumaId, deviceId));
        }
#endif

        mGpuToNumaMap[deviceId] = numaNode;
        mGpuMemoryToNumaMap[deviceId] = numaMemoryNode;
        mNumaToGpuMap[numaNode].push_back(deviceId);
    }
}

#ifdef __linux__

static void bitmask_copy_manual(struct bitmask* dst, const struct bitmask* src)
{
    if (!dst || !src)
        return;
    numa_bitmask_clearall(dst);
    for (int i = 0; i < numa_num_possible_cpus(); ++i)
    {
        if (numa_bitmask_isbitset(src, i))
        {
            numa_bitmask_setbit(dst, i);
        }
    }
}

static void bitmask_or_manual(struct bitmask* dst, const struct bitmask* src)
{
    if (!dst || !src)
        return;
    for (int i = 0; i < numa_num_possible_cpus(); ++i)
    {
        if (numa_bitmask_isbitset(src, i))
        {
            numa_bitmask_setbit(dst, i);
        }
    }
}

void TopologyDetector::precomputeCpuAffinityMasks()
{
    int num_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus == 0)
    {
        return;
    }

    for (int gpuId = 0; gpuId < num_gpus; ++gpuId)
    {
        auto itGpuNuma = mGpuToNumaMap.find(gpuId);
        if (itGpuNuma == mGpuToNumaMap.end())
        {
            TLLM_LOG_WARNING("GPU %d not found in mGpuToNumaMap during mask precomputation. Skipping.", gpuId);
            continue;
        }
        int gpuNumaNode = itGpuNuma->second;

        // Strict Mask: CPUs on the GPU's direct NUMA node
        struct bitmask* strictMask = numa_allocate_cpumask(); // Uses numa_bitmask_alloc internally
        if (strictMask)
        {
            numa_bitmask_clearall(strictMask); // Initialize to empty
            if (mNumaToCpuCountMap.count(gpuNumaNode) && mNumaToCpuCountMap.at(gpuNumaNode) > 0)
            {
                if (numa_node_to_cpus(gpuNumaNode, strictMask) != 0)
                {
                    TLLM_LOG_WARNING(
                        "Failed to get CPUs for GPU %d's NUMA node %d for strict mask. Strict mask will be empty.",
                        gpuId, gpuNumaNode);
                    numa_bitmask_clearall(strictMask); // Ensure it's empty on failure
                }
            }
            mGpuStrictCpuMasks[gpuId] = strictMask;
        }
        else
        {
            TLLM_LOG_WARNING("Failed to allocate strict CPU mask for GPU %d.", gpuId);
        }
    }
}

const struct bitmask* TopologyDetector::getStrictCpuMaskForGpu(int gpuId) const
{
    auto it = mGpuStrictCpuMasks.find(gpuId);
    if (it != mGpuStrictCpuMasks.end())
    {
        return it->second;
    }
    return nullptr;
}

#endif

void TopologyDetector::bindThreadByCurrentGpu()
{
#ifdef __linux__
    if (numa_available() == -1)
    {
        TLLM_LOG_WARNING("libnuma not available. Cannot bind thread to NUMA node.");
        return;
    }

    int currentDevice = -1;
    if (cudaGetDevice(&currentDevice) != cudaSuccess)
    {
        TLLM_LOG_WARNING("Failed to get current CUDA device. Cannot bind thread.");
        return;
    }

    const struct bitmask* targetMask = nullptr;
    targetMask = getStrictCpuMaskForGpu(currentDevice);

    if (targetMask)
    {
        // Check if the mask is not all clear before attempting to set affinity
        bool maskIsClear = true;
        for (int k = 0; k < numa_num_possible_cpus(); ++k)
        {
            if (numa_bitmask_isbitset(targetMask, k))
            {
                maskIsClear = false;
                break;
            }
        }

        if (!maskIsClear)
        {
            // Create a mutable copy of the targetMask to pass to numa_sched_setaffinity
            struct bitmask* mutableCopyForAffinity = numa_allocate_cpumask();
            if (mutableCopyForAffinity)
            {
                bitmask_copy_manual(mutableCopyForAffinity, targetMask);
                if (numa_sched_setaffinity(0, mutableCopyForAffinity) == -1)
                { // 0 refers to the current thread
                    TLLM_LOG_WARNING("Failed to set thread affinity for GPU %d using precomputed mask. Error: %s",
                        currentDevice, strerror(errno));
                }
                numa_free_cpumask(mutableCopyForAffinity);
            }
            else
            {
                TLLM_LOG_WARNING(
                    "Failed to allocate temporary bitmask for setting affinity. Cannot bind thread for GPU %d.",
                    currentDevice);
            }
        }
        else
        {
            TLLM_LOG_DEBUG("Target affinity mask for GPU %d is empty. Not setting affinity.", currentDevice);
        }
    }
    else
    {
        TLLM_LOG_WARNING("Precomputed CPU affinity mask not found for GPU %d. Cannot bind thread.", currentDevice);
    }

#else
    TLLM_LOG_DEBUG("Thread binding by GPU NUMA node is only supported on Linux with libnuma.");
#endif
}

int TopologyDetector::getCurrentGpuNumaCpuCount()
{
    int numaId = getCurrentGpuNumaId();
    if (numaId >= 0)
    {
        auto it = mNumaToCpuCountMap.find(numaId);
        if (it != mNumaToCpuCountMap.end())
        {
            return it->second;
        }
    }
    TLLM_LOG_DEBUG(
        "CPU count for GPU's NUMA node %d not found or node invalid. Returning total hardware concurrency.", numaId);
    return std::thread::hardware_concurrency();
}

int TopologyDetector::getCurrentGpuNumaId()
{
    int currentDevice = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&currentDevice));

    auto it = mGpuToNumaMap.find(currentDevice);
    if (it != mGpuToNumaMap.end())
    {
        return it->second;
    }
    TLLM_LOG_WARNING("NUMA node for current GPU %d not found in map. Defaulting to node 0.", currentDevice);
    return 0;
}

int TopologyDetector::getCurrentGpuMemoryNumaId()
{
    int currentDevice = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&currentDevice));

    auto it = mGpuMemoryToNumaMap.find(currentDevice);
    if (it != mGpuMemoryToNumaMap.end())
    {
        return it->second;
    }
    TLLM_LOG_WARNING(
        "NUMA node for current GPU Memory %d not found in map. Defaulting to node -1 (No Memory Node).", currentDevice);
    return -1;
}

int TopologyDetector::getGpuCountUnderNuma(int numaId)
{
    auto it = mNumaToGpuMap.find(numaId);
    if (it != mNumaToGpuMap.end())
    {
        return it->second.size();
    }
    return 0;
}

void* TopologyDetector::allocateCurrentGpuNumaMemory(size_t memorySize)
{
    int currentDevice = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&currentDevice));
    int numaId = getCurrentGpuMemoryNumaId();
    TLLM_CHECK_WITH_INFO(numaId >= 0, "Current GPU memory has no NUMA ID. Cannot allocate memory.");
    void* ptr = numa_alloc_onnode(memorySize, numaId);
    TLLM_CUDA_CHECK(cudaHostRegister(ptr, memorySize, cudaHostRegisterDefault));
    return ptr;
}

void TopologyDetector::freeCurrentGpuNumaMemory(void* ptr, size_t memorySize)
{
    TLLM_CUDA_CHECK(cudaHostUnregister(ptr));
    numa_free(ptr, memorySize);
}

std::string TopologyDetector::getCpuArchitecture()
{
    return mCpuArchitecture;
}

} // namespace tensorrt_llm::runtime
