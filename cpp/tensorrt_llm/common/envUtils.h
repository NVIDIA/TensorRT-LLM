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

#pragma once
#include "tensorrt_llm/common/cudaUtils.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <optional>
#include <string>

namespace tensorrt_llm::common
{
// Useful when you want to inject some debug code controllable with env var.
std::optional<int32_t> getIntEnv(char const* name);

std::optional<size_t> getUInt64Env(char const* name);

std::optional<float> getFloatEnv(char const* name);

bool getBoolEnv(char const* name);

// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels();

// Whether XQA JIT is enabled.
//
// Returns the value of TRTLLM_ENABLE_XQA_JIT env var. If such env var doesn't exist, std::nullopt is returned.
std::optional<bool> getEnvEnableXQAJIT();

// 0 means to use heuristics.
std::optional<int32_t> getEnvXqaBlocksPerSequence();

// Whether use tileSizeKv64 for multiCtasKvMode of trtllm-gen kernels.
bool getEnvUseTileSizeKv64ForTrtllmGen();

// Tune the number of blocks per sequence for accuracy/performance purpose.
bool getEnvMmhaMultiblockDebug();

int getEnvMmhaBlocksPerSequence();

int getEnvMmhaKernelBlockSize();

// Whether PDL is enabled.
bool getEnvEnablePDL();

template <typename KernelFn, typename... Args>
inline void launchWithPdlWhenEnabled(char const* name, KernelFn kernelFn, dim3 grid, dim3 block, size_t dynamicShmSize,
    cudaStream_t stream, Args&&... args)
{
    TLLM_LOG_DEBUG("Enable PDL in %s", name);
    cudaLaunchConfig_t kernelConfig;
    kernelConfig.gridDim = grid;
    kernelConfig.blockDim = block;
    kernelConfig.dynamicSmemBytes = dynamicShmSize;
    kernelConfig.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    kernelConfig.attrs = attrs;
    kernelConfig.numAttrs = 1;

    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&kernelConfig, kernelFn, std::forward<Args>(args)...));
}

bool getEnvUseUCXKvCache();

bool getEnvUseMPIKvCache();
bool getEnvUseNixlKvCache();

std::string getEnvUCXInterface();

std::string getEnvNixlInterface();

bool getEnvDisaggLayerwise();

bool getEnvParallelCacheSend();

bool getEnvRequestKVCacheConcurrent();

bool getEnvDisableKVCacheTransferOverlap();

bool getEnvEnableReceiveKVCacheParallel();

std::string const& getEnvKVCacheTransferOutputPath();

bool getEnvTryZCopyForKVCacheTransfer();

// Force deterministic behavior for all kernels.
bool getEnvForceDeterministic();

// Force deterministic behavior for MoE plugin.
bool getEnvForceDeterministicMOE();

// Disable finalize fusion in MoE plugin
bool getEnvMOEDisableFinalizeFusion();

// Force deterministic behavior for attention plugin.
bool getEnvForceDeterministicAttention();

// Force deterministic behavior for all reduce plugin.
bool getEnvForceDeterministicAllReduce();

// Return the workspace size for custom all reduce kernels.
// This only works when force deterministic is enabled.
size_t getEnvAllReduceWorkspaceSize();

size_t getEnvKVCacheRecvBufferCount();

bool getEnvKVCacheTransferUseAsyncBuffer();

bool getEnvKVCacheTransferUseSyncBuffer();

size_t getEnvKVCacheSendMaxConcurrenceNum();

size_t getEnvMemSizeForKVCacheTransferBuffer();

uint16_t getEnvNixlPort();

bool getEnvDisaggBenchmarkGenOnly();

// Whether to disable the chunked-attention in the generation phase.
bool getEnvDisableChunkedAttentionInGenPhase();

bool getEnvKVCacheTransferAllBlocksForWindow();

} // namespace tensorrt_llm::common
