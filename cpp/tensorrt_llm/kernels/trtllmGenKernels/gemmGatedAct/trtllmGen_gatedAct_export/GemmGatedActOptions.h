/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
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

#include "GemmOptions.h"

#ifdef TLLM_GEN_EXPORT_INTERFACE
#include <iostream>

#define TLLM_CHECK_ERROR(cond, ...)                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        printArgs(__VA_ARGS__);                                                                                        \
        return false;                                                                                                  \
    }

#define TLLM_LOG_ERROR(...) TLLM_CHECK_ERROR(false, __VA_ARGS__)

#define TLLM_CHECK_ERROR_FMT(...) TLLM_CHECK_ERROR(false, __VA_ARGS__)

#define TLLM_CHECK_WARNING(cond, ...)                                                                                  \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        printArgs(__VA_ARGS__);                                                                                        \
        return false;                                                                                                  \
    }

#define TLLM_LOG_WARNING(...) TLLM_CHECK_WARNING(false, __VA_ARGS__)

#define TLLM_LOG_INFO(...) TLLM_CHECK_WARNING(false, __VA_ARGS__)

#endif

namespace gemmGatedAct
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

// Type of the gated activation
enum class ActType
{
    // silu(x) = x * sigmoid(x) = x * (1 / (1 + e^(-x)))
    Silu = 0
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to check the ActType type.

#define TLLM_ACT_TYPE_FUNCTION(actType)                                                                                \
    inline bool is##actType(ActType type)                                                                              \
    {                                                                                                                  \
        return (type == ActType::actType);                                                                             \
    }

TLLM_ACT_TYPE_FUNCTION(Silu)

#undef TLLM_ACT_TYPE_FUNCTION

////////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmGatedActOptions : virtual public gemm::GemmOptions
{
    GemmGatedActOptions() = default;

    GemmGatedActOptions(gemm::GemmOptions const& options, ActType actType)
        : gemm::GemmOptions(options)
        , mActType(actType)
    {
    }

    // Type of the gated activation.
    ActType mActType{ActType::Silu};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Check if the options are valid or not.
inline bool checkAndUpdateGemmGatedActOptions(
    gemmGatedAct::GemmGatedActOptions& options, bool isBlackwell, bool updateOptions = true)
{

    // tmpOut is already transposed at this stage
    auto const hiddenSizeStr = options.mTransposeMmaOutput ? "M" : "N";
    auto const hiddenSize = options.mTransposeMmaOutput ? options.mM : options.mN;
    auto const hiddenEpilogueTileSize = options.mTransposeMmaOutput ? options.mEpilogueTileM : options.mEpilogueTileN;

    TLLM_CHECK_ERROR(hiddenSize % 2 == 0, hiddenSizeStr, " must be a multiple of 2.");

    TLLM_CHECK_ERROR((options.mTransposeMmaOutput ^ options.mUseShuffledMatrixA) == 0,
        "Transpose mma output can only be used with shuffled A matrix. And vice versa.");

    if (options.mUseTmaStore)
    {
        TLLM_CHECK_ERROR(hiddenEpilogueTileSize * tg::dtypeGetNumBits(options.mDtypeElt) / /* bits */ 8 % 32 == 0,
            "Unsupported output hidden tile size");
    }

    if (options.mUseDeepSeekFp8)
    {
        TLLM_CHECK_ERROR(hiddenSize % 256 == 0, "Output hidden size must be a multiple of 256");
    }

    if (options.mDtypeC == tg::Dtype::E2m1 || options.mDtypeC == tg::Dtype::MxE4m3)
    {
        int const outHiddenSize = (options.mTransposeMmaOutput ? options.mM : options.mN) / 2;
        int const hiddenGranularity = 4 * tg::dtypeNumEltsPerSf(options.mDtypeC);
        TLLM_CHECK_ERROR(outHiddenSize % hiddenGranularity == 0, "Output hidden size (", outHiddenSize,
            ") must be a multiple of ", hiddenGranularity, " for block-scaled outputs.");
    }

    auto isValid = gemm::checkAndUpdateGemmOptions(options, isBlackwell,
        /* tpGrpSize */ 1, updateOptions);

    if (!isValid)
    {
        return false;
    }

    if (options.mNumSlicesForSplitK > 1)
    {
        TLLM_CHECK_ERROR(doesSplitKUseDsmem(options.mSplitK), "Split-k GMEM and GemmGatedAct are not supported yet.");
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string dumpOptions(GemmGatedActOptions const& options)
{
    std::stringstream ss;
    ss << gemm::dumpOptions(options) << ", ";
    ss << "mActType="
       << "gemmGatedAct::ActType(" << static_cast<int32_t>(options.mActType) << ")" << std::endl;
    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GemmGatedActConfig
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmGatedActConfig
{
    // When TRT-LLM Gen is exported to the other frameworks, the TLLM_GEN_EXPORT_INTERFACE must be
    // defined. In this case, the cubins will be loaded from the provided data and function name.
    // Otherwise, the kernel will be loaded from the CudaRunner.
#ifdef TLLM_GEN_EXPORT_INTERFACE
    uint8_t const* mData{nullptr};
    uint32_t const mSize{0};
    uint32_t const mSharedMemSize{0};
    char const* mFunctionName{nullptr};
    uint32_t const mNumThreadsPerCTA{0};
#else
    trtllm::gen::CudaRunner* mCudaRunner{nullptr};
#endif

    GemmGatedActOptions mOptions{};
    gemm::SmVersion mSm{gemm::SmVersion::Sm100a};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemmGatedAct

#ifdef TLLM_GEN_EXPORT_INTERFACE

#undef TLLM_CHECK_ERROR
#undef TLLM_CHECK_ERROR_FMT
#undef TLLM_CHECK_WARNING
#undef TLLM_LOG_WARNING
#undef TLLM_LOG_INFO
#undef TLLM_LOG_ERROR
#endif // TLLM_GEN_EXPORT_INTERFACE
