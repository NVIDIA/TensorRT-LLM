/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/plugins/common/plugin.h"

namespace tensorrt_llm::plugins
{

struct GemmDims
{
    int32_t minM;
    int32_t maxM;
    int32_t n;
    int32_t k;

    GemmDims()
        : minM(-1)
        , maxM(-1)
        , n(-1)
        , k(-1)
    {
    }

    GemmDims(int32_t minM_, int32_t maxM_, int32_t n_, int32_t k_)
        : minM(minM_)
        , maxM(maxM_)
        , n(n_)
        , k(k_)
    {
    }

    bool isInitialized() const
    {
        return minM >= 0 && maxM >= 0 && n >= 0 && k >= 0;
    }
};

// Unique ID of GEMM
// In our case GEMM is uniqly identified by N and K
class GemmIdCore
{
public:
    int n;
    int k;
    nvinfer1::DataType dtype;

    GemmIdCore(int n_, int k_, const nvinfer1::DataType& dtype_)
        : n(n_)
        , k(k_)
        , dtype(dtype_)
    {
    }

    GemmIdCore()
        : n(-1)
        , k(-1)
        , dtype(nvinfer1::DataType::kFLOAT) // dtype does not matter here
    {
    }

    bool operator==(const GemmIdCore& id) const
    {
        return isEqual(id);
    }

    friend std::ostream& operator<<(std::ostream& out, const GemmIdCore& id)
    {
        out << "(N;K)=(" << id.n << ";" << id.k << "),";
        out << " type=" << static_cast<int>(id.dtype);
        return out;
    }

protected:
    bool isEqual(const GemmIdCore& id) const
    {
        return n == id.n && k == id.k && dtype == id.dtype;
    }
};

// Hash of GemmId
struct GemmIdCoreHash
{
    std::size_t operator()(const GemmIdCore& id) const
    {
        auto h1 = std::hash<int>{}(id.n);
        auto h2 = std::hash<int>{}(id.k);
        auto h3 = std::hash<int>{}(static_cast<int>(id.dtype));
        return h1 ^ h2 ^ h3;
    }
};

class GemmIdCublas : public GemmIdCore
{
public:
    bool transA{};
    bool transB{};

    GemmIdCublas(int n_, int k_, const nvinfer1::DataType& dtype_, bool transA_, bool transB_)
        : GemmIdCore(n_, k_, dtype_)
        , transA(transA_)
        , transB(transB_)
    {
    }

    GemmIdCublas() {}

    bool operator==(const GemmIdCublas& id) const
    {
        return isEqual(id) && transA == id.transA && transB == id.transB;
    }

    friend std::ostream& operator<<(std::ostream& out, const GemmIdCublas& id)
    {
        out << "(N;K)=(" << id.n << ";" << id.k << "),";
        out << " type=" << static_cast<int>(id.dtype);
        out << " transA=" << id.transA;
        out << " transB=" << id.transB;
        return out;
    }
};

// Hash of GemmIdCublas
struct GemmIdCublasHash
{
    std::size_t operator()(const GemmIdCublas& id) const
    {
        auto h1 = std::hash<int>{}(id.n);
        auto h2 = std::hash<int>{}(id.k);
        auto h3 = std::hash<int>{}(static_cast<int>(id.dtype));
        auto h4 = std::hash<bool>{}(id.transA);
        auto h5 = std::hash<bool>{}(id.transB);
        return h1 ^ h2 ^ h3 ^ h4 ^ h5;
    }
};

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
class GemmPluginProfiler
{
public:
    static constexpr int MAX_PROFILE_M = 8192;

    // Map for single GEMM for different Ms (GEMM dimension) to the best config for particular M
    using MProfileMap = std::unordered_map<int, std::optional<Config>>;
    using MProfileMapPtr = std::shared_ptr<MProfileMap>;

    // requires exclusive ownership to write to *this
    using reader_lock = std::unique_lock<std::shared_timed_mutex>;
    // requires shared ownership to read from other
    using writer_lock = std::shared_lock<std::shared_timed_mutex>;

    // Struct of continuing map if GEMMs to the best profiles for different Ms
    struct MNKProfileMap
    {
        // Mutex guarding map
        std::shared_timed_mutex mutex;
        // Map from GEMM Id to profile for particular GEMM
        std::unordered_map<GemmIdType, MProfileMapPtr, GemmIdHashType> profileMap;

        bool existsMProfileMap(const GemmIdType& id)
        {
            const auto iter = profileMap.find(id);
            return iter != profileMap.end();
        }

        void createMProfileMap(const GemmIdType& id)
        {
            profileMap[id] = std::make_shared<MProfileMap>();
        }

        MProfileMapPtr getMProfileMap(const GemmIdType& id)
        {
            const auto iter = profileMap.find(id);
            if (iter == profileMap.end())
            {
                std::ostringstream msg;
                msg << "Cannot find ID (" << id << ") in the profile map. Abort.";
                TLLM_LOG_ERROR(msg.str());
            }
            return iter->second;
        }
    };

    using MNKProfileMapPtr = std::shared_ptr<MNKProfileMap>;

    GemmPluginProfiler();

    void serialize(char*& buffer, const GemmIdType& gemmId) const;

    void deserialize(const char*& data, GemmDims& dims, const GemmIdType& gemmId);
    size_t getSerializationSize(const GemmIdType& gemmId) const;

    void profileTactics(
        const RunnerPtr& runner, const nvinfer1::DataType& type, const GemmDims& dims, const GemmIdType& gemmId);

    void setSelectionTactics(const MNKProfileMapPtr& map)
    {
        mMNKProfileMap = map;
    }

    void setTmpWorkspaceSizeInBytes(size_t bytes)
    {
        mTmpWorkspaceSizeInBytes = bytes;
    }

    void setSkip(bool skip)
    {
        mSkip = mSkip || skip;
    }

    std::optional<Config> getBestConfig(int m, const GemmIdType& gemmId) const;

protected:
    virtual void runTactic(int m, int n, int k, const Config& tactic, char* workspace, const cudaStream_t& stream) = 0;

    virtual void computeTmpSize(int maxM, int n, int k) = 0;

    virtual bool checkTactic(int m, int n, int k, const Config& tactic) const
    {
        return true;
    }

    virtual std::vector<Config> getTactics(int m, int n, int k) const = 0;

    virtual void initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream){};

private:
    void allocateTmpData();

    void freeTmpData();

    std::optional<Config> profileTacticsForProblem(int m, int n, int k, const std::vector<Config>& tactics);

    float profileTacticForProblem(int m, int n, int k, const Config& tactic);

    int nextPowerOfTwo(int v) const
    {
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return ++v;
    }

protected:
    RunnerPtr mRunner{nullptr};

    nvinfer1::DataType mType{};

private:
    MNKProfileMapPtr mMNKProfileMap{};

    size_t mTmpWorkspaceSizeInBytes{0};

    char* mWorkspaceTmp{nullptr};

    GemmDims mDims{};

    bool mSkip{false};
};

template <typename GemmPluginProfilerType>
class GemmPluginProfilerManager
{
public:
    using MNKProfileMap = typename GemmPluginProfilerType::MNKProfileMap;
    using MNKProfileMapPtr = typename GemmPluginProfilerType::MNKProfileMapPtr;
    using GemmPluginProfilerPtr = std::shared_ptr<GemmPluginProfilerType>;

    GemmPluginProfilerManager()
    {
        mMNKProfileMap = std::make_shared<MNKProfileMap>();
    }

    GemmPluginProfilerPtr createGemmPluginProfiler(bool inference, bool skip = false)
    {
        auto profiler = std::make_shared<GemmPluginProfilerType>();
        profiler->setSkip(skip);
        // If the profiler is created during the engine build,
        // mMNKProfileMap is shared between different profilers to minimize the time spent on the profiling
        // and do not repeat profiling for the GEMMs of the same shape.
        if (!inference)
        {
            profiler->setSelectionTactics(mMNKProfileMap);
        }
        return profiler;
    }

private:
    MNKProfileMapPtr mMNKProfileMap{};
};

} // namespace tensorrt_llm::plugins
