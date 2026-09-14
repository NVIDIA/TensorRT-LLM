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

#include "nvrtcLinkage.h"

#if defined(__linux__) && !TRTLLM_NVRTC_DYNAMIC_LINKING
#include <dlfcn.h>
#include <nvrtc.h>
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

std::string getNvrtcLinkageError()
{
#if defined(__linux__) && !TRTLLM_NVRTC_DYNAMIC_LINKING
    // A local object identifies this DSO without itself being preemptible.
    static char const kModuleAnchor = 0;
    Dl_info expected{};
    if (dladdr(&kModuleAnchor, &expected) == 0)
    {
        return "Cannot determine the XQA NVRTC wrapper's shared library.";
    }

    struct Symbol
    {
        char const* name;
        void const* address;
    };

    // Inspect the addresses used by this PIC object, not RTLD_DEFAULT: another
    // library may legitimately use dynamic NVRTC without interposing these APIs.
    Symbol const symbols[] = {
        {"nvrtcCreateProgram", reinterpret_cast<void const*>(&nvrtcCreateProgram)},
        {"nvrtcCompileProgram", reinterpret_cast<void const*>(&nvrtcCompileProgram)},
        {"nvrtcGetProgramLogSize", reinterpret_cast<void const*>(&nvrtcGetProgramLogSize)},
        {"nvrtcGetProgramLog", reinterpret_cast<void const*>(&nvrtcGetProgramLog)},
        {"nvrtcGetCUBINSize", reinterpret_cast<void const*>(&nvrtcGetCUBINSize)},
        {"nvrtcGetCUBIN", reinterpret_cast<void const*>(&nvrtcGetCUBIN)},
        {"nvrtcDestroyProgram", reinterpret_cast<void const*>(&nvrtcDestroyProgram)},
    };
    for (auto const& symbol : symbols)
    {
        Dl_info actual{};
        if (dladdr(symbol.address, &actual) == 0)
        {
            return std::string("Cannot determine the library providing ") + symbol.name + ".";
        }
        if (actual.dli_fbase != expected.dli_fbase)
        {
            return std::string("XQA was built with NVRTC_DYNAMIC_LINKING=OFF, but ") + symbol.name + " resolves to "
                + actual.dli_fname + " instead of the statically linked NVRTC in " + expected.dli_fname
                + ". Mixing NVRTC implementations can cause compilation failures or invalid program handles. "
                  "Remove the conflicting dynamic NVRTC dependency/LD_PRELOAD, or rebuild TensorRT-LLM with "
                  "--configure_cmake --extra-cmake-vars NVRTC_DYNAMIC_LINKING=ON to use dynamic NVRTC intentionally.";
        }
    }
#endif
    return {};
}

} // namespace kernels

TRTLLM_NAMESPACE_END
