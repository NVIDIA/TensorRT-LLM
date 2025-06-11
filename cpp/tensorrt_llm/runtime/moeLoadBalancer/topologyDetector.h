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

#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>
#ifdef __linux__
#include <numa.h> // For libnuma
#endif

// Forward declaration for struct bitmask to avoid including numaif.h if numa.h already covers it,
// or if only numa.h is intended to be the public include for this header's users.
#ifdef __linux__
struct bitmask;
#endif

namespace tensorrt_llm::runtime
{

class TopologyDetector
{
public:
    static TopologyDetector& getInstance()
    {
        static TopologyDetector instance;
        return instance;
    }

    ~TopologyDetector();

    // Binds the current thread to the CPU cores of the NUMA node associated with the current GPU.
    void bindThreadByCurrentGpu();

    // Returns the number of CPU cores on the NUMA node associated with the current GPU.
    // Returns total hardware concurrency as a fallback if specific count cannot be determined.
    int getCurrentGpuNumaCpuCount();

    // Returns the ID of the NUMA node associated with the current GPU.
    // Returns 0 as a default or -1 on error.
    int getCurrentGpuNumaId();

    // Returns the ID of the NUMA node that current GPU's memory is assigned.
    // GPUs using C2C link with CPU may have assigned NUMA ID for its memory, like GB200.
    // Returns -1 if it doesn't have NUMA ID.
    int getCurrentGpuMemoryNumaId();

    // Returns the number of GPUs associated with the given NUMA node ID.
    int getGpuCountUnderNuma(int numaId);

    // Returns the number of GPUs which have same NUMA node ID with the current GPU.
    int getGpuCountUnderSameNuma()
    {
        return getGpuCountUnderNuma(getCurrentGpuNumaId());
    }

    // Returns a pointer to a memory region on the current GPU's NUMA node.
    void* allocateCurrentGpuNumaMemory(size_t memorySize);

    // Frees a memory region allocated by allocateCurrentGpuNumaMemory.
    void freeCurrentGpuNumaMemory(void* ptr, size_t memorySize);

    // Returns the detected CPU architecture (e.g., "x86_64", "aarch64").
    std::string getCpuArchitecture();

#ifdef __linux__
    // Getters for precomputed CPU affinity masks
    const struct bitmask* getStrictCpuMaskForGpu(int gpuId) const;
#endif

private:
    TopologyDetector();
    void detectCpuTopology();          // Detects CPU NUMA topology and CPU counts per node.
    void detectGpuTopology();          // Detects GPU to NUMA node mapping.
#ifdef __linux__
    void precomputeCpuAffinityMasks(); // Precomputes CPU masks for each GPU
#endif

    // Member variables
    std::map<int, int> mGpuToNumaMap;              // GPU ID -> NUMA Node ID
    std::map<int, int> mGpuMemoryToNumaMap;        // GPU ID -> Memory NUMA Node ID
    std::map<int, std::vector<int>> mNumaToGpuMap; // NUMA Node ID -> List of GPU IDs
    std::map<int, int> mNumaToCpuCountMap;         // NUMA Node ID -> CPU Core Count
    std::string mCpuArchitecture;
    bool mTopologyDetected = false;
    std::mutex mDetectionMutex; // Mutex to protect topology detection process

#ifdef __linux__
    // Precomputed CPU affinity masks
    std::map<int, struct bitmask*> mGpuStrictCpuMasks; // GPU ID -> Strict CPU mask
#endif
};

} // namespace tensorrt_llm::runtime
