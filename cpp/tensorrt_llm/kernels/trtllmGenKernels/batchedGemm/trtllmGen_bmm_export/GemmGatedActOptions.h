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

namespace batchedGemm
{

namespace trtllm
{
namespace gen
{
class CudaRunner;
class GenCfg;
} // namespace gen
} // namespace trtllm

namespace gemmGatedAct
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

// Type of the gated activation
enum class ActType
{
    // clang-format off
  // For ActType == SwiGlu, ideally we would like to have something like
  //    gatedAct = quantScaleC * (x0 * dequantScaleAb + beta) * ((x1 * scaleGate) * sigmoid(alpha * x1 * scaleGate)).
  // But for now, we use the simplified version
  //    gatedAct = scaleC      * (x0                 + beta') * ((x1 * scaleGate) * sigmoid(alpha * x1 * scaleGate)),
  // where x0 and x1 are the raw numbers from Gemm, while scaleC and scaleGate are input scales,
  // beta' = beta / dequantScaleAb, scaleC = quantScaleC * dequantScaleAb.
  //
  // GatedSilu is a special case of SwiGlu where the alpha is 1.0 and the beta is 0.0.
    // clang-format on
    SwiGlu,
    // For ActType == GeGlu, we use the simplified version
    //    gatedAct = scaleC' * (x0 + beta') * ((x1 * scaleGate) * phi(alpha * x1 * scaleGate)),
    // where x0 and x1 are the raw numbers from Gemm, while scaleC and scaleGate are input scales,
    // beta' = beta / scaleAb, scaleC' = scaleC * scaleAb.
    GeGlu,
    // Placeholder for no activation; not implemented in codegen
    None,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to check the ActType type.

#define TLLM_ACT_TYPE_FUNCTION(actType)                                                                                \
    inline bool is##actType(ActType type)                                                                              \
    {                                                                                                                  \
        return (type == ActType::actType);                                                                             \
    }

TLLM_ACT_TYPE_FUNCTION(SwiGlu)
TLLM_ACT_TYPE_FUNCTION(GeGlu)

#undef TLLM_ACT_TYPE_FUNCTION

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string getActTypeName(ActType type)
{
    switch (type)
    {
    case ActType::SwiGlu: return "SwiGlu";
    case ActType::GeGlu: return "GeGlu";
    default: return "Unknown type";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmGatedActOptions : public gemm::GemmOptions
{
    GemmGatedActOptions() = default;

    GemmGatedActOptions(gemm::GemmOptions options, ActType actType, bool clampBeforeAct)
        : gemm::GemmOptions(options)
        , mActType(actType)
        , mClampBeforeAct(clampBeforeAct)
    {
    }

    // Type of the gated activation.
    ActType mActType{ActType::SwiGlu};
    // Clamp the dequantized values to the range [-limit, limit].
    bool mClampBeforeAct{false};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Check if the options are valid or not.
inline bool checkAndUpdateGemmGatedActOptions(
    gemmGatedAct::GemmGatedActOptions& options, tg::CudaArch cudaArch, bool updateOptions = true)
{
    auto isValid = gemm::checkAndUpdateGemmOptions(options, cudaArch,
        /* tpGrpSize */ 1, updateOptions);
    if (!isValid)
    {
        return false;
    }

    if (options.mActType == gemmGatedAct::ActType::None)
    {
        TLLM_CHECK_ERROR(false, "ActType None is not supported");
    }
    // tmpOut is already transposed at this stage
    auto const hiddenSizeStr = options.mTransposeMmaOutput ? "M" : "N";
    auto const hiddenSize = options.mTransposeMmaOutput ? options.mM : options.mN;
    auto const hiddenEpilogueTileSize = options.mTransposeMmaOutput ? options.mEpilogueTileM : options.mEpilogueTileN;

    TLLM_CHECK_ERROR(hiddenSize % 2 == 0, hiddenSizeStr, " must be a multiple of 2.");

    TLLM_CHECK_ERROR((options.mTransposeMmaOutput && !options.mUseShuffledMatrix) == false,
        "Transpose mma output can only be used with shuffled matrix.");

    if (options.mUseTmaStore)
    {
        TLLM_CHECK_ERROR(hiddenEpilogueTileSize * tg::dtypeGetNumBits(options.mDtypeC) / /* bits */ 8 % 32 == 0,
            "Unsupported output hidden tile size");
    }

    if (options.mDtypeC == tg::Dtype::E2m1 || options.mDtypeC == tg::Dtype::MxE4m3)
    {
        int const outHiddenSize = (options.mTransposeMmaOutput ? options.mM : options.mN) / 2;
        int const hiddenGranularity = 4 * options.mSfBlockSizeC;
        TLLM_CHECK_ERROR(outHiddenSize % hiddenGranularity == 0, "Output hidden size (", outHiddenSize,
            ") must be a multiple of ", hiddenGranularity, " for block-scaled outputs.");
    }

    auto const validHiddenSize = options.mTransposeMmaOutput ? options.mValidM : options.mValidN;
    if (options.mUseDeepSeekFp8)
    {
        TLLM_CHECK_ERROR(hiddenSize % 256 == 0 && validHiddenSize % 256 == 0, "Hidden size (", hiddenSize,
            ") and valid hidden size (", validHiddenSize, ") must be a multiple of 256");
    }

    //
    if (options.mUseShuffledMatrix)
    {
        auto const shuffleBlockSize = gemm::getShuffleBlockSize(options.mEpilogueTileM);
        TLLM_CHECK_ERROR(hiddenSize % (2 * shuffleBlockSize) == 0 && validHiddenSize % (2 * shuffleBlockSize) == 0,
            "M/validM must be a multiple of 2 * shuffle block size (", 2 * shuffleBlockSize,
            ") when useShuffledMatrix");
    }
    if (options.mNumSlicesForSplitK > 1)
    {
        TLLM_CHECK_ERROR(doesSplitKUseDsmem(options.mSplitK), "Split-k GMEM and GemmGatedAct are not supported yet.");
    }

    if (gemm::isBiasTypeMn(options.mBiasType))
    {
        TLLM_CHECK_ERROR(options.mTransposeMmaOutput, "Bias type Mn is not supported with not transpose mma output.");
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string dumpOptions(GemmGatedActOptions const& options, bool dumpRuntimeParams = true)
{
    std::stringstream ss;
    ss << gemm::dumpOptions(options, dumpRuntimeParams) << ", ";
    ss << "mActType="
       << "gemmGatedAct::ActType(" << static_cast<int32_t>(options.mActType) << ")," << std::endl;
    ss << "mClampBeforeAct=" << options.mClampBeforeAct << "" << std::endl;
    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GemmGatedActConfig
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmGatedActConfig
{
    uint8_t const* mData{nullptr};
    uint32_t mSize{0};
    uint32_t mSharedMemSize{0};
    char const* mFunctionName{nullptr};
    uint32_t mNumThreadsPerCTA{0};
    char const* mHash{nullptr};

    std::string mGenCfgJsonStr{""};
    char const* mExecPath{nullptr};
    trtllm::gen::CudaRunner* mCudaRunner{nullptr};
    trtllm::gen::GenCfg* mGenCfg{nullptr};
    int32_t mInstanceIdx{0};

    GemmGatedActOptions mOptions{};
    tg::CudaArch mSm{tg::CudaArch::Sm100a};
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

} // namespace batchedGemm
