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

#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/common/logger.h"

#include <dlfcn.h>

namespace tensorrt_llm::executor::kv_cache
{

[[nodiscard]] DynLibLoader& DynLibLoader::getInstance()
{
    static DynLibLoader instance;
    return instance;
}

[[nodiscard]] void* DynLibLoader::getHandle(std::string const& name)
{
    std::lock_guard<std::mutex> lock(mDllMutex);
    auto it = mHandlers.find(name);
    if (it != mHandlers.end())
    {
        return it->second;
    }
    TLLM_LOG_INFO("dlopen: " + name);
    void* handler = dlopen(name.c_str(), RTLD_LAZY);
    TLLM_CHECK_WITH_INFO(handler, "%s can not be loaded correctly: %s", name.c_str(), dlerror());
    mHandlers[name] = handler;
    return handler;
}

[[nodiscard]] void* DynLibLoader::dlSym(void* handle, char const* symbol)
{
    return dlsym(handle, symbol);
}

DynLibLoader::~DynLibLoader()
{
    std::lock_guard<std::mutex> lock(mDllMutex);
    for (auto const& pair : mHandlers)
    {
        dlclose(pair.second);
    }
}

} // namespace tensorrt_llm::executor::kv_cache
