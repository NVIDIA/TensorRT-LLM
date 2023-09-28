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
        return n == id.n && k == id.k && dtype == id.dtype;
    }

    friend std::ostream& operator<<(std::ostream& out, const GemmIdCore& id)
    {
        out << "(N;K)=(" << id.n << ";" << id.k << "),";
        out << " type=" << static_cast<int>(id.dtype);
        return out;
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

    // Struct of contining map if GEMMs to the best profiles for different Ms
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

    GemmPluginProfiler()
    {
        mMNKProfileMap = std::make_shared<MNKProfileMap>();

        // set SKIP_GEMM_PLUGIN_PROFILINGS=1 to avoid tactics profilings
        const auto skip = std::getenv("SKIP_GEMM_PLUGIN_PROFILINGS");
        mSkip = (skip != NULL && std::stoi(skip));
        if (mSkip)
        {
            TLLM_LOG_DEBUG(
                "SKIP_GEMM_PLUGIN_PROFILINGS is set. Skipping GEMM plugin profilings. It could result in runtime error "
                "if default tactic is not defined.");
        }
    }

    void serialize(char* buffer, const GemmIdType& gemmId) const
    {
        auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

        // Save number of profiles for given GEMM ID
        write(buffer, static_cast<int>(mProfileMap->size()));
        for (const auto& pair : *mProfileMap)
        {
            // Save pair of M to the best GEMM config
            write(buffer, pair);
        }
    }

    void deserialize(const char*& data, GemmDims& dims, const GemmIdType& gemmId)
    {
        // NOTE(nkorobov): this mutex is not needed since each thread owns its own map, but will put here for
        // consistency
        writer_lock lock(mMNKProfileMap->mutex);

        mDims = dims;

        // GemmId gemmId(dims.n, dims.k);
        if (!mMNKProfileMap->existsMProfileMap(gemmId))
        {
            // Create GEMM with GEMM ID if it does not exist
            mMNKProfileMap->createMProfileMap(gemmId);
        }
        // Populate map with profiles of GEMM ID
        auto profileMap = mMNKProfileMap->getMProfileMap(gemmId);
        int selectedMapSize;
        read(data, selectedMapSize);
        for (int ii = 0; ii < selectedMapSize; ++ii)
        {
            std::pair<int, std::optional<Config>> config;
            read(data, config);
            profileMap->insert(config);
        }
    }

    size_t getSerializationSize(const GemmIdType& gemmId) const
    {
        reader_lock lock(mMNKProfileMap->mutex);
        return sizeof(int) +                                 // size of the tactics map
            mMNKProfileMap->getMProfileMap(gemmId)->size()
            * sizeof(std::pair<int, std::optional<Config>>); // size of the tactics map
    }

    void profileTactics(const std::vector<Config>& tactics, const RunnerPtr& runner, const nvinfer1::DataType& type,
        const GemmDims& dims, const GemmIdType& gemmId)
    {
        writer_lock lock(mMNKProfileMap->mutex);

        if (!dims.isInitialized())
        {
            return;
        }

        mRunner = runner;
        mType = type;

        const int maxM = std::min(nextPowerOfTwo(dims.maxM), MAX_PROFILE_M);
        computeTmpSize(maxM, dims.n, dims.k);

        if (!mMNKProfileMap->existsMProfileMap(gemmId))
        {
            // Create map for GEMM ID
            mMNKProfileMap->createMProfileMap(gemmId);
        }

        if (mSkip)
        {
            return;
        }

        auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

        auto profileTactics = [&tactics, &mProfileMap, this](int m, int n, int k)
        {
            if (mProfileMap->count(m) == 0)
            {
                // Profile different tactics for particular m and insert best config to the map
                mProfileMap->insert({m, this->profileTacticsForProblem(m, n, k, tactics)});
            }
        };

        // Allocate tmp data to run GEMMs
        allocateTmpData();
        const int startMinMRounded = nextPowerOfTwo(dims.minM);
        for (int m = startMinMRounded; m < maxM; m *= 2)
        {
            profileTactics(m, dims.n, dims.k);
        }

        profileTactics(maxM, dims.n, dims.k);
        // Free tmp data
        freeTmpData();
    }

    void setSelectionTactics(const MNKProfileMapPtr& map)
    {
        mMNKProfileMap = map;
    }

    void setTmpWorkspaceSizeInBytes(size_t bytes)
    {
        mTmpWorkspaceSizeInBytes = bytes;
    }

    std::optional<Config> getBestConfig(int m, const GemmIdType& gemmId) const
    {
        reader_lock lock(mMNKProfileMap->mutex);

        if (mSkip)
        {
            return std::nullopt;
        }

        const int mRounded = std::min(nextPowerOfTwo(m), MAX_PROFILE_M);
        return mMNKProfileMap->getMProfileMap(gemmId)->at(mRounded);
    }

protected:
    virtual void runTactic(int m, int n, int k, const Config& tactic, char* workspace, const cudaStream_t& stream) = 0;

    virtual void computeTmpSize(int maxM, int n, int k) = 0;

    virtual bool checkTactic(int m, int n, int k, const Config& tactic) const
    {
        return true;
    }

private:
    void allocateTmpData()
    {
        TLLM_CHECK_WITH_INFO(mTmpWorkspaceSizeInBytes > 0, "tmpWorkspaceSizeInBytes must be larger than 0");
        const auto status = cudaMalloc(&mWorkspaceTmp, mTmpWorkspaceSizeInBytes);
        TLLM_CHECK_WITH_INFO(status == cudaSuccess, "Can't allocate tmp workspace for GEMM tactics profiling.");
    }

    void freeTmpData()
    {
        const auto status = cudaFree(mWorkspaceTmp);
        TLLM_CHECK_WITH_INFO(status == cudaSuccess, "Can't free tmp workspace for GEMM tactics profiling.");
    }

    std::optional<Config> profileTacticsForProblem(int m, int n, int k, const std::vector<Config>& tactics)
    {
        TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

        float bestTime = std::numeric_limits<float>::max();
        Config bestConfig;
        bool foundOne = false;

        // Iterate over all tactics for given M, N and K
        for (int ii = 0; ii < tactics.size(); ++ii)
        {
            const Config& candidateConfig = tactics[ii];
            float time = std::numeric_limits<float>::max();
            try
            {
                if (!checkTactic(m, n, k, candidateConfig))
                {
                    continue;
                }
                // Profile particualar tactic for given M, N and K
                time = profileTacticForProblem(m, n, k, candidateConfig);
                foundOne = true;
            }
            catch (const std::exception& e)
            {
                std::ostringstream msg;
                msg << "Cannot profile configuration " << ii << " (for"
                    << " m=" << m << ", n=" << n << ", k=" << k << "). Skipped";
                TLLM_LOG_WARNING(msg.str());
                continue;
            }

            // Choose the fastest tactic
            if (time < bestTime)
            {
                bestConfig = candidateConfig;
                bestTime = time;
            }
        }

        if (!foundOne)
        {
            std::ostringstream msg;
            msg << "Have not found any valid GEMM config for shape ("
                << "m=" << m << ", n=" << n << ", k=" << k << "). Will try to use default or fail at runtime";
            TLLM_LOG_WARNING(msg.str());
            return std::nullopt;
        }
        return {bestConfig};
    }

    float profileTacticForProblem(int m, int n, int k, const Config& tactic)
    {
        constexpr int warmup = 5;
        constexpr int runs = 10;

        cudaStream_t stream = cudaStreamDefault;
        // Warmup the execution
        for (int i = 0; i < warmup; ++i)
        {
            runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
        }

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaDeviceSynchronize();
        cudaEventRecord(start, 0);

        // Profile GEMM
        for (int i = 0; i < runs; ++i)
        {
            runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
        }

        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);

        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return elapsed / runs;
    }

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

    GemmPluginProfilerPtr createGemmPluginProfiler(bool inference)
    {
        auto profiler = std::make_shared<GemmPluginProfilerType>();
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
