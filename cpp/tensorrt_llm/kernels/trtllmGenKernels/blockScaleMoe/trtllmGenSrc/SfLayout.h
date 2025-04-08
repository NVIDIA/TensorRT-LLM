/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "SfLayoutDecl.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include <string>

namespace trtllm
{
namespace gen
{

////////////////////////////////////////////////////////////////////////////////////////////////////

inline SfLayout sfLayoutFromString(std::string const& str)
{
    if (str == "linear")
    {
        return SfLayout::Linear;
    }
    else if (str == "8x4")
    {
        return SfLayout::R8c4;
    }
    else if (str == "8x16")
    {
        return SfLayout::R8c16;
    }
    else if (str == "128x4")
    {
        return SfLayout::R128c4;
    }
    else
    {
        TLLM_THROW("Unknown SfLayout %s", str.c_str());
    }
}

inline std::string sfLayoutToString(SfLayout layout)
{
    switch (layout)
    {
    case SfLayout::Linear: return "linear";
    case SfLayout::R8c4: return "8x4";
    case SfLayout::R8c16: return "8x16";
    case SfLayout::R128c4: return "128x4";
    default: TLLM_LOG_ERROR("Unsupported layout"); return "error";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm
