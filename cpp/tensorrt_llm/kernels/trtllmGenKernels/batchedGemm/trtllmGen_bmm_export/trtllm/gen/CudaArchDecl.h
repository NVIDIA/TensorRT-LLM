/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <cstring>
#include <string>

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Be careful when modifying this file as it is included by the generated kernels. For example, do
// not add TLLM_CHECK_* constructs in this file. Thanks!
//
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace batchedGemm
{

namespace trtllm
{
namespace gen
{

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class CudaArch
{
    // Hopper
    Sm90a = 0,
    // Blackwell
    Sm100a,
    // Blackwell-family
    Sm100f,
    // Blackwell Ultra
    Sm103a,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool isArchHopper(CudaArch cudaArch)
{
    return cudaArch == CudaArch::Sm90a;
}

inline bool isArchBlackwell(CudaArch cudaArch)
{
    return cudaArch == CudaArch::Sm100a || cudaArch == CudaArch::Sm100f || cudaArch == CudaArch::Sm103a;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string cudaArchToString(CudaArch cudaArch, bool isFull = true)
{
    switch (cudaArch)
    {
    case CudaArch::Sm90a: return isFull ? "90a" : "90";
    case CudaArch::Sm100a: return isFull ? "100a" : "100";
    case CudaArch::Sm100f: return isFull ? "100f" : "100";
    case CudaArch::Sm103a: return isFull ? "103a" : "103";
    default: assert(false); return "";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline CudaArch stringToCudaArch(std::string const& str)
{
    if (str == "90a")
    {
        return CudaArch::Sm90a;
    }
    else if (str == "100a")
    {
        return CudaArch::Sm100a;
    }
    else if (str == "100f")
    {
        return CudaArch::Sm100f;
    }
    else if (str == "103a")
    {
        return CudaArch::Sm103a;
    }
    else
    {
        assert(false);
        return CudaArch::Sm100a;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm
} // namespace batchedGemm
