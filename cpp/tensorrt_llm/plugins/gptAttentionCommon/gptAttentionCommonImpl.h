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

#include "gptAttentionCommon.h"

namespace tensorrt_llm::plugins
{
template <typename T>
T* GPTAttentionPluginCommon::cloneImpl() const noexcept
{
    static_assert(std::is_base_of_v<GPTAttentionPluginCommon, T>);
    auto* plugin = new T(static_cast<T const&>(*this));
    plugin->setPluginNamespace(mNamespace.c_str());

    // Cloned plugins should be in initialized state with correct resources ready to be enqueued.
    plugin->initialize();
    return plugin;
}

template <typename T>
T* GPTAttentionPluginCreatorCommon::deserializePluginImpl(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GPTAttentionPluginCommon::destroy()
    try
    {
        auto* obj = new T(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
} // namespace tensorrt_llm::plugins
