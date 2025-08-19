/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "envUtils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/stringUtils.h"
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>

namespace tensorrt_llm::common
{

std::optional<int32_t> getIntEnv(char const* name)
{
    char const* const env = std::getenv(name);
    if (env == nullptr)
    {
        return std::nullopt;
    }
    int32_t const val = std::stoi(env);
    return {val};
};

std::optional<size_t> getUInt64Env(char const* name)
{
    char const* const env = std::getenv(name);
    if (env == nullptr)
    {
        return std::nullopt;
    }
    size_t const val = std::stoull(env);
    return {val};
};

std::optional<float> getFloatEnv(char const* name)
{
    char const* const env = std::getenv(name);
    if (env == nullptr)
    {
        return std::nullopt;
    }
    float const val = std::stof(env);
    return {val};
}

std::optional<std::string> getStrEnv(char const* name)
{
    char const* const env = std::getenv(name);
    if (env == nullptr)
    {
        return std::nullopt;
    }
    return std::string(env);
}

// Returns true if the env variable exists and is set to "1"
bool getBoolEnv(char const* name)
{
    char const* env = std::getenv(name);
    return env && env[0] == '1' && env[1] == '\0';
}

static std::string trim(std::string const& str)
{
    size_t start = str.find_first_not_of(" \t\n\r");
    size_t end = str.find_last_not_of(" \t\n\r");
    return (start == std::string::npos || end == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

// Parse memory size
static size_t parseMemorySize(std::string const& input)
{
    std::string str = trim(input);

    size_t unitPos = 0;
    while (unitPos < str.size() && (std::isdigit(str[unitPos]) || str[unitPos] == '.'))
    {
        ++unitPos;
    }

    // Split the numeric part and the unit part
    std::string numberPart = str.substr(0, unitPos);
    std::string unitPart = str.substr(unitPos);

    double value = 0;
    try
    {
        value = std::stod(numberPart);
    }
    catch (std::invalid_argument const& e)
    {
        throw std::invalid_argument("Invalid number format in memory size: " + input);
    }

    toLower(unitPart);
    size_t multiplier = 1;
    if (unitPart == "b")
    {
        multiplier = 1;
    }
    else if (unitPart == "kb")
    {
        multiplier = 1024;
    }
    else if (unitPart == "mb")
    {
        multiplier = 1024 * 1024;
    }
    else if (unitPart == "gb")
    {
        multiplier = 1024 * 1024 * 1024;
    }
    else if (unitPart == "tb")
    {
        multiplier = static_cast<size_t>(pow(1024.0, 4));
    }
    else
    {
        throw std::invalid_argument("Unknown unit in memory size: " + unitPart);
    }

    return static_cast<size_t>(value * multiplier);
}

// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels()
{
    return (getIntEnv("TRTLLM_FORCE_XQA").value_or(0) != 0) || getEnvForceDeterministicAttention();
}

std::optional<bool> getEnvEnableXQAJIT()
{
    auto const tmp = getIntEnv("TRTLLM_ENABLE_XQA_JIT");
    if (tmp.has_value())
    {
        return static_cast<bool>(tmp.value());
    }
    return std::nullopt;
}

std::optional<int> getEnvXqaBlocksPerSequence()
{
    auto const val = getIntEnv("TRTLLM_XQA_BLOCKS_PER_SEQUENCE");
    return (val.has_value() && *val <= 0) ? std::nullopt : val;
}

// Tune the number of blocks per sequence for accuracy/performance purpose.
bool getEnvMmhaMultiblockDebug()
{
    char const* enable_mmha_debug_var = std::getenv("TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG");
    return enable_mmha_debug_var && enable_mmha_debug_var[0] == '1' && enable_mmha_debug_var[1] == '\0';
}

int getEnvMmhaBlocksPerSequence()
{
    char const* mmhaBlocksPerSequenceEnv = std::getenv("TRTLLM_MMHA_BLOCKS_PER_SEQUENCE");
    if (mmhaBlocksPerSequenceEnv)
    {
        int mmhaBlocksPerSequence = std::atoi(mmhaBlocksPerSequenceEnv);
        if (mmhaBlocksPerSequence <= 0)
        {
            TLLM_LOG_WARNING("Invalid value for TRTLLM_MMHA_BLOCKS_PER_SEQUENCE. Will use default values instead!");
            return 0;
        }
        return mmhaBlocksPerSequence;
    }
    return 0;
}

int getEnvMmhaKernelBlockSize()
{
    char const* mmhaKernelBlockSizeEnv = std::getenv("TRTLLM_MMHA_KERNEL_BLOCK_SIZE");
    if (mmhaKernelBlockSizeEnv)
    {
        int mmhaKernelBlockSize = std::atoi(mmhaKernelBlockSizeEnv);
        if (mmhaKernelBlockSize <= 0)
        {
            TLLM_LOG_WARNING("Invalid value for TRTLLM_MMHA_KERNEL_BLOCK_SIZE. Will use default values instead!");
            return 0;
        }
        return mmhaKernelBlockSize;
    }
    return 0;
}

bool getEnvUseTileSizeKv64ForTrtllmGen()
{
    return getBoolEnv("TRTLLM_GEN_ENABLE_TILE_SIZE_KV64");
}

bool getEnvEnablePDL()
{
    if (getSMVersion() >= 90)
    {
        // PDL will be enabled by setting the env variables `TRTLLM_ENABLE_PDL` to `1`
        return getBoolEnv("TRTLLM_ENABLE_PDL");
    }
    return false;
}

bool getEnvUseUCXKvCache()
{
    return getBoolEnv("TRTLLM_USE_UCX_KVCACHE");
}

bool getEnvUseMPIKvCache()
{
    return getBoolEnv("TRTLLM_USE_MPI_KVCACHE");
}

bool getEnvUseNixlKvCache()
{
    return getBoolEnv("TRTLLM_USE_NIXL_KVCACHE");
}

std::string getEnvUCXInterface()
{
    char const* ucx_interface = std::getenv("TRTLLM_UCX_INTERFACE");
    return ucx_interface ? std::string(ucx_interface) : std::string();
}

std::string getEnvNixlInterface()
{
    static std::once_flag flag;
    static std::string nixlInterface;

    std::call_once(flag,
        [&]()
        {
            char const* nixl_interface = std::getenv("TRTLLM_NIXL_INTERFACE");
            if (nixl_interface)
            {
                nixlInterface = nixl_interface;
            }
        });
    return nixlInterface;
}

bool getEnvDisaggLayerwise()
{
    return getBoolEnv("TRTLLM_DISAGG_LAYERWISE");
}

bool getEnvDisableSelectiveCacheTransfer()
{
    return getBoolEnv("TRTLLM_DISABLE_SELECTIVE_CACHE_TRANSFER");
}

bool getEnvParallelCacheSend()
{
    return getBoolEnv("TRTLLM_PARALLEL_CACHE_SEND");
}

bool getEnvRequestKVCacheConcurrent()
{
    return getBoolEnv("TRTLLM_REQUEST_KV_CACHE_CONCURRENT");
}

bool getEnvDisableKVCacheTransferOverlap()
{
    return getBoolEnv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP");
}

bool getEnvEnableReceiveKVCacheParallel()
{
    return getBoolEnv("TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL");
}

bool getEnvTryZCopyForKVCacheTransfer()
{
    return getBoolEnv("TRTLLM_TRY_ZCOPY_FOR_KVCACHE_TRANSFER");
}

bool getEnvForceDeterministic()
{
    return getBoolEnv("FORCE_DETERMINISTIC");
}

bool getEnvForceDeterministicMOE()
{
    return getBoolEnv("FORCE_MOE_KERNEL_DETERMINISTIC") || getEnvForceDeterministic();
}

bool getEnvMOEDisableFinalizeFusion()
{
    static bool const moeDisableFinalizeFusion = getBoolEnv("TRTLLM_MOE_DISABLE_FINALIZE_FUSION");
    return moeDisableFinalizeFusion;
}

bool getEnvForceDeterministicAttention()
{
    return getBoolEnv("FORCE_ATTENTION_KERNEL_DETERMINISTIC") || getEnvForceDeterministic();
}

bool getEnvForceDeterministicAllReduce()
{
    return getBoolEnv("FORCE_ALL_REDUCE_DETERMINISTIC") || getEnvForceDeterministic();
}

size_t getEnvAllReduceWorkspaceSize()
{
    return getUInt64Env("FORCE_ALLREDUCE_KERNEL_WORKSPACE_SIZE").value_or(1000 * 1000 * 1000);
}

std::string const& getEnvKVCacheTransferOutputPath()
{
    return getStrEnv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH").value_or("");
}

bool getEnvKVCacheTransferUseAsyncBuffer()
{
    return getBoolEnv("TRTLLM_KVCACHE_TRANSFER_USE_ASYNC_BUFFER");
}

bool getEnvKVCacheTransferUseSyncBuffer()
{
    return getBoolEnv("TRTLLM_KVCACHE_TRANSFER_USE_SYNC_BUFFER");
}

size_t getEnvKVCacheSendMaxConcurrenceNum()
{
    return getUInt64Env("TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM").value_or(2);
}

size_t getEnvKVCacheRecvBufferCount()
{
    return getUInt64Env("TRTLLM_KVCACHE_RECV_BUFFER_COUNT").value_or(2);
}

size_t getEnvMemSizeForKVCacheTransferBuffer()
{
    char const* memSizeForKVCacheTransferBufferEnv = std::getenv("TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE");
    if (memSizeForKVCacheTransferBufferEnv)
    {
        return parseMemorySize(memSizeForKVCacheTransferBufferEnv);
    }
    else
    {
        return parseMemorySize("512MB");
    }
}

uint16_t getEnvNixlPort()
{
    return getUInt64Env("TRTLLM_NIXL_PORT").value_or(0);
}

bool getEnvDisaggBenchmarkGenOnly()
{
    return getBoolEnv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY");
}

bool getEnvDisableChunkedAttentionInGenPhase()
{
    return getBoolEnv("TRTLLM_DISABLE_CHUNKED_ATTENTION_IN_GEN_PHASE");
}

} // namespace tensorrt_llm::common
