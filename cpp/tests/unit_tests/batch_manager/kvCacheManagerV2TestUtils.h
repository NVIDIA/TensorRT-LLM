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

#pragma once

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"

#include <optional>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2::test
{

inline KVCacheManagerConfig makeConfig(bool enableStats = true)
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    AttentionLayerConfig layer;
    layer.layerId = 0;
    layer.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(layer));
    config.enableStats = enableStats;
    return config;
}

inline KVCacheManagerConfig makeTieredConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    AttentionLayerConfig layer;
    layer.layerId = 0;
    layer.buffers.push_back(BufferConfig{"key", 2 << 20, std::nullopt});
    config.layers.emplace_back(std::move(layer));
    return config;
}

//! Attention and SSM life cycles side by side, over a GPU and a host tier.
//!
//! Life cycles are registered in layer order, so the attention layer is LifeCycleId{0} and
//! the SSM layer is LifeCycleId{1}. The buffer sizes differ so per-life-cycle byte counters
//! identify which life cycle they came from.
//!
//! The quotas give attention 4 GPU and 2 host slots of 1 MiB, and SSM 1 GPU and 1 host slot
//! of 2 MiB. A three-block sequence therefore fits on the GPU but a second sequence evicts
//! it, and a second eviction round overflows the host pools.
inline KVCacheManagerConfig makeHybridTieredConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{6UL << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{4UL << 20});

    AttentionLayerConfig attention;
    attention.layerId = 0;
    attention.buffers.push_back(BufferConfig{"key", 1UL << 20, std::nullopt});
    config.layers.emplace_back(std::move(attention));

    SsmLayerConfig ssm;
    ssm.layerId = 1;
    ssm.buffers.push_back(BufferConfig{"ssm_state", 2UL << 20, std::nullopt});
    config.layers.emplace_back(std::move(ssm));

    // KVCacheManagerConfig::validate() rejects an SSM layer without this.
    config.commitMinSnapshot = true;
    return config;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2::test
