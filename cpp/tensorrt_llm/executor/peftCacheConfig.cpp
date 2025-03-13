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

#include "tensorrt_llm/executor/executor.h"

namespace tensorrt_llm::executor
{
PeftCacheConfig::PeftCacheConfig(SizeType32 numHostModuleLayer, SizeType32 numDeviceModuleLayer,
    SizeType32 optimalAdapterSize, SizeType32 maxAdapterSize, SizeType32 numPutWorkers, SizeType32 numEnsureWorkers,
    SizeType32 numCopyStreams, SizeType32 maxPagesPerBlockHost, SizeType32 maxPagesPerBlockDevice,
    std::optional<FloatType> const& deviceCachePercent, std::optional<size_t> const& hostCacheSize,
    std::optional<std::string> const& loraPrefetchDir)
    : mNumHostModuleLayer(numHostModuleLayer)
    , mNumDeviceModuleLayer(numDeviceModuleLayer)
    , mOptimalAdapterSize(optimalAdapterSize)
    , mMaxAdapterSize(maxAdapterSize)
    , mNumPutWorkers(numPutWorkers)
    , mNumEnsureWorkers(numEnsureWorkers)
    , mNumCopyStreams(numCopyStreams)
    , mMaxPagesPerBlockHost(maxPagesPerBlockHost)
    , mMaxPagesPerBlockDevice(maxPagesPerBlockDevice)
    , mDeviceCachePercent(deviceCachePercent)
    , mHostCacheSize(hostCacheSize)
    , mLoraPrefetchDir(loraPrefetchDir)
{
}

bool PeftCacheConfig::operator==(PeftCacheConfig const& other) const
{
    return mNumHostModuleLayer == other.mNumHostModuleLayer && mNumDeviceModuleLayer == other.mNumDeviceModuleLayer
        && mOptimalAdapterSize == other.mOptimalAdapterSize && mMaxAdapterSize == other.mMaxAdapterSize
        && mNumPutWorkers == other.mNumPutWorkers && mNumEnsureWorkers == other.mNumEnsureWorkers
        && mNumCopyStreams == other.mNumCopyStreams && mMaxPagesPerBlockHost == other.mMaxPagesPerBlockHost
        && mMaxPagesPerBlockDevice == other.mMaxPagesPerBlockDevice && mDeviceCachePercent == other.mDeviceCachePercent
        && mHostCacheSize == other.mHostCacheSize && mLoraPrefetchDir == other.mLoraPrefetchDir;
}

SizeType32 PeftCacheConfig::getNumHostModuleLayer() const
{
    return mNumHostModuleLayer;
}

SizeType32 PeftCacheConfig::getNumDeviceModuleLayer() const
{
    return mNumDeviceModuleLayer;
}

SizeType32 PeftCacheConfig::getOptimalAdapterSize() const
{
    return mOptimalAdapterSize;
}

SizeType32 PeftCacheConfig::getMaxAdapterSize() const
{
    return mMaxAdapterSize;
}

SizeType32 PeftCacheConfig::getNumPutWorkers() const
{
    return mNumPutWorkers;
}

SizeType32 PeftCacheConfig::getNumEnsureWorkers() const
{
    return mNumEnsureWorkers;
}

SizeType32 PeftCacheConfig::getNumCopyStreams() const
{
    return mNumCopyStreams;
}

SizeType32 PeftCacheConfig::getMaxPagesPerBlockHost() const
{
    return mMaxPagesPerBlockHost;
}

SizeType32 PeftCacheConfig::getMaxPagesPerBlockDevice() const
{
    return mMaxPagesPerBlockDevice;
}

std::optional<FloatType> PeftCacheConfig::getDeviceCachePercent() const
{
    return mDeviceCachePercent;
}

std::optional<size_t> PeftCacheConfig::getHostCacheSize() const
{
    return mHostCacheSize;
}

std::optional<std::string> PeftCacheConfig::getLoraPrefetchDir() const
{
    return mLoraPrefetchDir;
}

} // namespace tensorrt_llm::executor
