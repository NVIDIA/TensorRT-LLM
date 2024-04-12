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
#if !defined(_WIN32)
#include <dlfcn.h>
#endif // !defined(_WIN32)
#include <iostream>
#include <stdexcept>
#include <string>

#include "NvInfer.h"

template <typename tSymbolSignature>
tSymbolSignature getTrtLLMFunction(std::string libFileSoName, std::string symbol)
{
#if !defined(_WIN32)
    std::cout << "Trying to load " << libFileSoName << " ..." << std::endl;

    // 1. Defining a handle to the library
    void* handle = dlopen(libFileSoName.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    // 2. Check for errors
    char const* dl_error1 = dlerror();
    if (!handle)
    {
        throw std::runtime_error("Cannot open library: " + std::string(dl_error1));
    }

    // 3. Load actual queried `symbol`
    std::cout << "Loading symbol `" << symbol << "` ..." << std::endl;

    tSymbolSignature symbolFctn = nullptr;
    *(void**) (&symbolFctn) = dlsym(handle, symbol.c_str());

    // 4. Check for errors
    char const* dl_error2 = dlerror();
    if (dl_error2)
    {
        dlclose(handle);
        throw std::runtime_error("Cannot load symbol '" + symbol + "': " + std::string(dl_error2));
    }

    return symbolFctn;
#else  // on windows
    throw std::runtime_error(
        "`tSymbolSignature getTrtLLMFunction(std::string, std::string)` is not implemented on Windows.");
    return nullptr;
#endif // !defined(_WIN32)
}
