/***************************************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
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
