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

// Force the order of headers to avoid definition not found error
// clang-format off
#include <numeric>
#include <set>
#include "trtllmGenSrc/KernelParams.h"
#include "trtllmGenSrc/Enums.h"
#include "trtllmGenSrc/KernelTraits.h"
#include "trtllmGenSrc/SfLayout.h"
#include "gemmList.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/envUtils.h"
// clang-format on

#define TLLM_CHECK_ERROR_FMT(cond, ...)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "TRTLLM-GEN kernel launch failed");                                            \
        }                                                                                                              \
    } while (0)
#define TLLM_CHECK_WARNING TLLM_CHECK_ERROR_FMT

namespace tensorrt_llm
{
namespace kernels
{
namespace trtllmGenFp8BlockScaleMoe
{
namespace gemmCommon
{

struct MyOptions
{
    int mTopK;
    bool mBatchM;
    bool mTransposeMmaOutput;
    bool mUseDeepSeekFp8;
    bool mUseRouteAct;
    bool mRouteAct;
    bool mUseShuffledMatrixA;
    int mTileM;
    int mTileN;
    int mTileK;
    int mEpilogueTileM;
    int mEpilogueTileN;
    int mMmaM;
    int mMmaN;
    int mMmaK;
    int mNumExperts;
    int mNumTokens;
    int mM;
    int mN;
    int mK;
    tg::Dtype mDtypeElt;
    tg::Dtype mDtypeC;
    tg::Dtype mDtypeAcc;
    bool mUseTmaStore;
    bool mUseTwoTmaLoadWarps;
    float mAtol;
    float mRtol;
    int mNumSlicesForSplitK;
    int mClusterDimX;
    int mClusterDimY;
    int mClusterDimZ;
    ::gemm::AllReduceAlgo mAllReduceAlgo;
    ::gemm::SplitK mSplitK;
    ::gemm::KernelTraits mKernelTraits;
    int mNumStages;
    int mNumStagesMma;
    bool mUseFusedAct;
    bool mGateUseClusterSplitK;
    int mProjGateNumSplitKSlices;
    int32_t* mPtrNumNonExitingCtas{nullptr};
    int32_t* mPtrTotalNumPaddedTokens{nullptr};
    int32_t* mPtrCtaIdxXyToBatchIdx{nullptr};
    int32_t* mPtrCtaIdxXyToMnLimit{nullptr};
    tg::SfLayout mSfLayoutB;
    tg::SfLayout mSfLayoutC;
    bool mUseCustomLowLatencyImpl;
    bool mSliceK;
    int mNumSlicesForSliceK;
    std::vector<int> mBatchedM;
    std::vector<int> mBatchedN;
    int mNumBatches;
    bool mAllToAllRouteAct;
    bool mIsStaticBatch;
};

void copyKernelInfoToOptions(GemmInfo const& kernelInfo, MyOptions& options)
{
    options.mUseDeepSeekFp8 = kernelInfo.blockScale && kernelInfo.dtypeElt == tg::Dtype::E4m3;
    options.mUseRouteAct = kernelInfo.permuteFusion;
    options.mRouteAct = kernelInfo.permuteFusion;
    options.mUseShuffledMatrixA = kernelInfo.shuffledMatrixA;
    options.mTileM = kernelInfo.tileM;
    options.mTileN = kernelInfo.tileN;
    options.mTileK = kernelInfo.tileK;
    options.mEpilogueTileM = kernelInfo.epilogueTileM;
    options.mEpilogueTileN = kernelInfo.epilogueTileN;
    options.mMmaM = kernelInfo.mmaM;
    options.mMmaN = kernelInfo.mmaN;
    options.mMmaK = kernelInfo.mmaK;
    options.mNumSlicesForSplitK = kernelInfo.numSlicesForSplitK;
    options.mDtypeElt = kernelInfo.dtypeElt;
    options.mDtypeC = kernelInfo.dtypeC;
    options.mDtypeAcc = kernelInfo.dtypeAcc;
    options.mUseTmaStore = kernelInfo.useTmaStore;
    options.mNumStagesMma = kernelInfo.numStagesMma;
    options.mNumStages = kernelInfo.numStages;
    options.mNumStagesMma = kernelInfo.numStagesMma;
    options.mUseFusedAct = kernelInfo.useFusedAct;
    options.mGateUseClusterSplitK = kernelInfo.gateUseClusterSplitK;
    options.mProjGateNumSplitKSlices = kernelInfo.projGateNumSplitKSlices;
    options.mSliceK = kernelInfo.sliceK;
    options.mNumSlicesForSliceK = kernelInfo.mNumSlicesForSliceK;
    options.mSfLayoutB = kernelInfo.mSfLayoutB;
    options.mSfLayoutC = kernelInfo.mSfLayoutC;
}

namespace gemm
{
template <typename T>
inline std::string toString(T e)
{
    return std::to_string(e);
}

template <>
inline std::string toString(trtllm::gen::Dtype e)
{
    return trtllm::gen::dtypeToString(e);
}

inline int32_t divUp(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

inline int32_t getShuffleBlockSize(int epilogueTileM)
{
    int shuffleBlockSize = 16;
    if (epilogueTileM % 128 == 0)
    {
        shuffleBlockSize = 32;
    }
    return shuffleBlockSize;
}

using namespace ::gemm;

inline void checkAndUpdateGemmOptions(
    MyOptions& options, bool isBlackwell, bool usesAtolFromArg, bool usesRtolFromArg, int /* tpGrpSize */)
{
    if (options.mUseCustomLowLatencyImpl)
    {
        TLLM_CHECK_ERROR(isBlackwell, "LowLatencyGemm is only supported on Blackwell");
        TLLM_CHECK_ERROR(options.mDtypeElt == tg::Dtype::E2m1, "LowLatencyGemm only supports nvfp4 (for now).");
        TLLM_CHECK_ERROR(options.mDtypeAcc == tg::Dtype::Fp32, "LowLatencyGemm accumulates in fp32.");
        TLLM_CHECK_ERROR(options.mTileN == 32 || options.mTileN == 64 || options.mTileN == 128,
            "LowLatencyGemm supports tileN of 32, 64, 128..");
        if (options.mTileN == 32)
        {
            TLLM_CHECK_ERROR(options.mNumSlicesForSplitK == 1 || options.mNumSlicesForSplitK == 2
                    || options.mNumSlicesForSplitK == 4,
                "LowLatencyGemm accumulates in fp32.");
        }
        else
        {
            TLLM_CHECK_ERROR(
                options.mNumSlicesForSplitK == 1, "LowLatencyGemm does not use SplitK > 1 with tileN > 32.");
        }
        return;
    }
    if (options.mDtypeElt == tg::Dtype::E4m3 && options.mMmaK != 32)
    {
        TLLM_LOG_WARNING("Unsupported MmaK (", options.mMmaK, ") for ", gemm::toString(options.mDtypeElt).c_str(),
            ". Setting MmaK to 32");
        options.mMmaK = 32;
        options.mTileK = std::max(options.mMmaK, options.mTileK);
    }

    // NvFp4 constraints
    if (options.mDtypeElt == tg::Dtype::E2m1)
    {
        TLLM_CHECK_ERROR(isBlackwell, "FP4 is only supported on Blackwell");
        TLLM_CHECK_ERROR(options.mSfLayoutB == tg::SfLayout::R128c4 || options.mSfLayoutB == tg::SfLayout::R8c4,
            "Only the 128x4 and 8x4 SF layouts are supported for B, got ", tg::sfLayoutToString(options.mSfLayoutB));
        if (options.mMmaK != 64)
        {
            int newTileK = 64 * divUp(options.mTileK, 64);
            TLLM_LOG_WARNING("Unsupported MmaK (", options.mMmaK, ") for ", gemm::toString(options.mDtypeElt).c_str(),
                ". Setting MmaK to 64 and TileK to ", newTileK);
            options.mMmaK = 64;
            options.mTileK = newTileK;
        }
        if (options.mMmaM != 128)
        {
            int newTileM = 128 * divUp(options.mTileM, 128);
            TLLM_LOG_WARNING("Unsupported MmaM (", options.mMmaM, ") for ", gemm::toString(options.mDtypeElt).c_str(),
                ". Setting MmaM to 128 and TileM to ", newTileM);
            options.mMmaM = 128;
            options.mTileM = newTileM;
        }
        // TileN must be a multiple of the number of rows per SF tile.
        // TODO: relax this restriction.
        int const numSfTileRowsB = options.mSfLayoutB == tg::SfLayout::R128c4 ? 128 : 8;
        TLLM_CHECK_ERROR(options.mTileN % numSfTileRowsB == 0, "TileN (", options.mTileN, ") must be a multiple of ",
            numSfTileRowsB, " for B SF layout ", tg::sfLayoutToString(options.mSfLayoutB));
        // The MMA N may only be smaller than 64 if it is equal to the tile N.
        TLLM_CHECK_ERROR(options.mMmaN >= 64 || options.mMmaN == options.mTileN, "MmaN (", options.mMmaN,
            ") must be >= 64 or equal to TileN (", options.mTileN, ") for ", gemm::toString(options.mDtypeElt));
    }
    if (options.mDtypeC == tg::Dtype::E2m1)
    {
        TLLM_CHECK_ERROR(isBlackwell, "FP4 is only supported on Blackwell");

        TLLM_CHECK_ERROR(options.mSfLayoutC == tg::SfLayout::R128c4 || options.mSfLayoutC == tg::SfLayout::R8c4,
            "Only the 128x4 and 8x4 SF layouts are supported for C.");
        int const numSfTileRowsC = options.mSfLayoutC == tg::SfLayout::R128c4 ? 128 : 8;
        TLLM_CHECK_ERROR(options.mTileN % numSfTileRowsC == 0, "TileN (", options.mTileN, ") must be a multiple of ",
            numSfTileRowsC, " for C SF layout ", tg::sfLayoutToString(options.mSfLayoutC));

        int const hiddenDim = options.mTransposeMmaOutput ? options.mM : options.mN;
        TLLM_CHECK_ERROR(hiddenDim % 64 == 0, "Hidden dim (", hiddenDim, ") must be a multiple of 64 for FP4 outputs.");
        TLLM_CHECK_ERROR(!options.mTransposeMmaOutput || options.mUseShuffledMatrixA,
            "Transposing FP4 outputs requires shuffled A.");
    }

    // If dtypeC is unspecified (Dtype::Void), assign to the input dtype.
    if (options.mDtypeC == tg::Dtype::Void)
    {
        options.mDtypeC = options.mDtypeElt;
    }

    // Set epilogue tile sizes to the output tile sizes, when epilogue tile sizes are incorrect.
    if (options.mTileM % options.mEpilogueTileM != 0)
    {
        TLLM_LOG_WARNING("TileM (", options.mTileM, ") must be divisible by EpilogueTileM (", options.mEpilogueTileM,
            "). Setting EpilogueTileM to TileM");
        options.mEpilogueTileM = options.mTileM;
    }

    if (options.mTileN % options.mEpilogueTileN != 0)
    {
        TLLM_LOG_WARNING("TileN (", options.mTileN, ") must be divisible by EpilogueTileN (", options.mEpilogueTileN,
            "). Setting EpilogueTileN to TileN");
        options.mEpilogueTileN = options.mTileN;
    }

    // On Hopper, epilogue tile sizes are the same as output tiles.
    if (!isBlackwell)
    {
        options.mEpilogueTileM = options.mTileM;
        options.mEpilogueTileN = options.mTileN;
        TLLM_LOG_WARNING("Overwriting epilogueTileM and epilogueTileN to match tileM and tileN respectively");
    }

    // Unsupported epilogue tile size.
    if (options.mMmaM == 128 && options.mEpilogueTileM != options.mTileM)
    {
        TLLM_LOG_WARNING("When MmaM = 128, EpilogueTileM must be equal to TileM. Setting EpilogueTileM to TileM");
        options.mEpilogueTileM = options.mTileM;
    }

    TLLM_CHECK_ERROR(options.mM > 0 && options.mN > 0 && options.mK > 0, "M, N and K must be larger than 0");
    TLLM_CHECK_ERROR(options.mNumSlicesForSplitK > 0, "Split K must be larger than 0.");
    TLLM_CHECK_ERROR(options.mK % options.mNumSlicesForSplitK == 0, "K must be divisible by NumSlicesForSplitK.");
    TLLM_CHECK_ERROR((options.mK / options.mNumSlicesForSplitK) % options.mTileK == 0,
        "K / NumSlicesForSplitK must be divisible by TileK. Found TileK=", options.mTileK, " and K=", options.mK,
        " and NumSlicesForSplitK=", options.mNumSlicesForSplitK);

    if (options.mUseShuffledMatrixA)
    {
        auto const shuffleBlockSize = getShuffleBlockSize(options.mEpilogueTileM);
        TLLM_CHECK_ERROR(options.mM % shuffleBlockSize == 0, "M must be a multiple of shuffle block size (",
            shuffleBlockSize, ") when useShuffledMatrixA");
    }

    TLLM_CHECK_ERROR(options.mMmaM <= options.mEpilogueTileM && options.mMmaN <= options.mEpilogueTileN,
        "EpilogueTileM and EpilogueTileN must be larger or equal than the respective atom sizes.");
    TLLM_CHECK_ERROR(options.mTileM % options.mEpilogueTileM == 0 && options.mTileN % options.mEpilogueTileN == 0,
        "TileM and TileN must be divisible by EpilogueTileM and EpilogueTileN respectively.");
    TLLM_CHECK_ERROR(
        options.mClusterDimX == 1 && options.mClusterDimY == 1, "GEMM does not support cluster in X and Y dimensions.");
    TLLM_CHECK_ERROR(
        options.mClusterDimZ == 1 || options.mNumSlicesForSplitK > 1, "Cluster DimZ is only allowed for split-k.");
    TLLM_CHECK_ERROR(options.mTileM <= 128, "GEMM does not support TileM > 128.");

    // Force the tolerance to a slightly higher value for FP16/BF16.
    if (options.mDtypeElt == tg::Dtype::Fp16 || options.mDtypeElt == tg::Dtype::Bfloat16)
    {
        if (!usesAtolFromArg)
        {
            options.mAtol = (options.mAllReduceAlgo != gemm::AllReduceAlgo::None) ? 4e-3f : 2e-4f;
        }
    }

    // Force the tolerance to a slightly higher value DeepSeek with for FP16/BF16 output.
    if (options.mUseDeepSeekFp8 && (options.mDtypeC == tg::Dtype::Fp16 || options.mDtypeC == tg::Dtype::Bfloat16))
    {
        if (!usesAtolFromArg)
        {
            options.mAtol = 1e-2f;
        }
        if (!usesRtolFromArg)
        {
            options.mRtol = 1e-2f;
        }
    }

    // When the A-matrix is shuffled, the output must be transposed.
    if (options.mUseShuffledMatrixA)
    {
        // TODO add matrix shuffle for N-major epilogue.
        TLLM_CHECK_ERROR(options.mTransposeMmaOutput,
            "Shuffled matrix A is only supported with M-major epilogue. Set -transposeMmaOutput");
    }

    // Check all-reduce options.
    if (options.mAllReduceAlgo == AllReduceAlgo::OneShot)
    {
        // One shot is implemented with PTX cp.reduce.async.bulk.tensor which supports only the
        // following types for reduce add: u32, s32, u64, f32, f16, bf16.
        //
        // See: https://docs.nvidia.com/cuda/parallel-thread-execution/
        //   #data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor
        std::set<tg::Dtype> dtypeSupported{tg::Dtype::UInt32, tg::Dtype::Int32, tg::Dtype::UInt64, tg::Dtype::Fp32,
            tg::Dtype::Fp16, tg::Dtype::Bfloat16};
        TLLM_CHECK_ERROR(dtypeSupported.find(options.mDtypeC) != dtypeSupported.end(), "Unsupported output dtype ",
            tg::dtypeToString(options.mDtypeC));
    }
    else if (options.mAllReduceAlgo == AllReduceAlgo::TwoShot)
    {
        // TODO(anchengc):
        // Input dtype == output dtype -> can perform all-reduce in-place.
        // Input dtype != output dtype -> must perform all-reduce out of place.
        TLLM_CHECK_ERROR_FMT(options.mDtypeC == options.mDtypeAcc,
            "Not implemented - mixed dtype (dtypeC (%s) != dtypeAcc (%s)) requires out of place update",
            tg::dtypeToString(options.mDtypeC).c_str(), tg::dtypeToString(options.mDtypeAcc).c_str());
    }
    if (options.mAllReduceAlgo != AllReduceAlgo::None)
    {
        TLLM_CHECK_ERROR(options.mUseTmaStore, "Non-TMA store with all-reduce is not implemented");
    }

    if (options.mNumSlicesForSplitK == 1)
    {
        // No split-k.
        options.mSplitK = SplitK::None;
    }
    else if (options.mNumSlicesForSplitK > 1 && options.mClusterDimZ == 1)
    {
        // Split-k with exchange through gmem.
        options.mSplitK = SplitK::Gmem;
    }
    else
    {
        // Split-k with exchange through Dsmem.
        options.mSplitK = SplitK::Dsmem;
    }
    // For GMEM-based split-K, we write 4 elements at once.
    if (options.mSplitK == SplitK::Gmem)
    {
        TLLM_CHECK_ERROR((options.mM * options.mN) % 4 == 0, "M * N must be a multiple of 4 for Split-K");
    }

    if (options.mNumSlicesForSplitK > 1)
    {
        if ((options.mEpilogueTileM != options.mTileM || options.mEpilogueTileN != options.mTileN)
            && !options.mUseDeepSeekFp8)
        {
            options.mEpilogueTileM = options.mTileM;
            options.mEpilogueTileN = options.mTileN;
            TLLM_LOG_WARNING("Overwriting epilogueTileM and epilogueTileN to match tileM and tileN respectively");
        }
    }
    if (options.mSplitK == SplitK::Dsmem)
    {
        TLLM_CHECK_ERROR(options.mClusterDimZ == options.mNumSlicesForSplitK,
            "CGA size must be equal to the number of slices in split-k");
    }

    // DeepSeek Fp8
    if (!options.mUseDeepSeekFp8)
    {
        TLLM_CHECK_ERROR(options.mNumStagesMma == 1, "Non-DeepSeekFp8 requires numStagesMma == 1");
    }
    if (options.mUseDeepSeekFp8)
    {
        TLLM_CHECK_ERROR(options.mDtypeElt == tg::Dtype::E4m3, "A and B dtype must be E4m3 for DeepSeek Fp8. Found ",
            tg::dtypeToString(options.mDtypeElt));

        TLLM_CHECK_ERROR(isBlackwell, "DeepSeek Fp8 is not supported for Hopper");
        TLLM_CHECK_ERROR(options.mAllReduceAlgo == AllReduceAlgo::None, "DeepSeek Fp8 does not support AllReduce");

        // Check that TileK = 128 for correct scaling of every 128 channels.
        TLLM_CHECK_ERROR(options.mTileK == 128, "Tile-K must be equal to 128 for DeepSeek Fp8");
        // Tile sizes of the output hidden dimension.
        auto hiddenDim = options.mTransposeMmaOutput ? options.mM : options.mN;
        auto hiddenDimPerOutputTile = options.mTransposeMmaOutput ? options.mTileM : options.mTileN;
        auto hiddenDimPerEpilogueTile = options.mTransposeMmaOutput ? options.mEpilogueTileM : options.mEpilogueTileN;
        auto hiddenDimPerMma = options.mTransposeMmaOutput ? options.mMmaM : options.mMmaN;
        // auto hiddenDimName = options.mTransposeMmaOutput ? "M" : "N";
        // TLLM_CHECK_WARNING(options.mNumStagesMma > 1, "DeepSeekFp8 recommends >1 MMA accumulator stages.");
        // Update the number of stages of the MMA accumulator pipeline. TODO: enable by default for
        // deepseek.
        // options.mNumStagesMma = 2;
        // Use two MMA warps to reduce mbar trywait latency. TODO: enable by default for deepseek.
        // options.mUseTwoMmaWarps = true;

        // Make sure the GEMM-M/N dimension is a multiple of 128 when using DeepSeek FP8.
        TLLM_CHECK_ERROR(hiddenDim % 128 == 0, "GEMM-", hiddenDimName,
            " must be a multiple of 128 when using DeepSeek Fp8. Found ", hiddenDim);
        // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
        TLLM_CHECK_ERROR(
            options.mK % 128 == 0, "GEMM-K must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mK);

        // Check that the output tile N can be processed with the epilogue tile granularity.
        TLLM_CHECK_ERROR((hiddenDimPerOutputTile / 2) % hiddenDimPerEpilogueTile == 0, "DeepSeek Fp8 requires Tile",
            hiddenDimName, " / 2 (", hiddenDimPerOutputTile / 2, ") being a multiple of EpilogueTile", hiddenDimName,
            " (", hiddenDimPerEpilogueTile, ")");
        // Check that the output tile N can be processed with the epilogue tile granularity.
        TLLM_CHECK_ERROR((hiddenDimPerOutputTile / 2) % hiddenDimPerMma == 0, "DeepSeek Fp8 requires Tile",
            hiddenDimName, " / 2 (", hiddenDimPerOutputTile / 2, ") being a multiple of mma", hiddenDimName, " (",
            hiddenDimPerMma, ")");
    }

    if (options.mSliceK)
    {
        TLLM_CHECK_ERROR(isBlackwell, "Slice-K is not supported on Hopper");

        TLLM_CHECK_ERROR(!options.mUseDeepSeekFp8, "DeepSeek Fp8 GEMM is not supported for slice-K");
        TLLM_CHECK_ERROR(options.mUseTwoTmaLoadWarps, "Slice-K requires two warp load for A and B");
        TLLM_CHECK_ERROR(options.mTransposeMmaOutput, "Slice-K requires transpose mma output");
        TLLM_CHECK_ERROR(options.mUseShuffledMatrixA, "Slice-K requires shuffled matrix A");
        TLLM_CHECK_ERROR(options.mTileK % 128 == 0, "Slice-K requires TileK be a multiple of 128");
        TLLM_CHECK_ERROR(options.mMmaM == 128, "Slice-K requires MmaM == 128");
        TLLM_CHECK_ERROR(options.mTileN == options.mEpilogueTileN, "TileN must be equal to EpilogueTileN for slice-K");

        TLLM_LOG_WARNING("Overwriting TileM and EpilogueTileM to 32 for slice-K");
        // FIXME: it is possible to remove this restriction.
        options.mTileM = 32;
        options.mEpilogueTileM = 32;
        TLLM_CHECK_ERROR(options.mDtypeElt == tg::Dtype::E4m3, "Slice-K requires e4m3 input dtype");

        options.mNumSlicesForSliceK = 4;
        TLLM_CHECK_ERROR((options.mTileK / options.mMmaK) % options.mNumSlicesForSliceK == 0, "TileK (", options.mTileK,
            ") / MmaK (", options.mMmaK, ") must be a multiple of mNumSlicesForSliceK (", options.mNumSlicesForSliceK,
            ")");
    }

    // Init kernel traits.
    options.mKernelTraits = KernelTraits(options.mDtypeElt, options.mDtypeC, options.mDtypeAcc, options.mTileM,
        options.mTileN, options.mTileK, options.mEpilogueTileM, options.mEpilogueTileN, options.mNumStages,
        options.mNumStagesMma, options.mNumSlicesForSplitK, options.mNumSlicesForSliceK, options.mSplitK,
        options.mUseTmaStore, options.mTransposeMmaOutput, options.mAllReduceAlgo, options.mUseDeepSeekFp8);
}
} // namespace gemm

namespace gemmGatedAct
{

// Check if the options are valid or not.
inline void checkAndUpdateGemmGatedActOptions(
    MyOptions& options, bool isBlackwell, bool usesAtolFromArg, bool usesRtolFromArg)
{

    // tmpOut is already transposed at this stage
    auto const hiddenSizeStr = options.mTransposeMmaOutput ? "M" : "N";
    auto const hiddenSize = options.mTransposeMmaOutput ? options.mM : options.mN;
    auto const hiddenEpilogueTileSize = options.mTransposeMmaOutput ? options.mEpilogueTileM : options.mEpilogueTileN;

    TLLM_CHECK_WITH_INFO(hiddenSize % 2 == 0, "%s must be a multiple of 2.", hiddenSizeStr);

    TLLM_CHECK_WITH_INFO((options.mTransposeMmaOutput ^ options.mUseShuffledMatrixA) == 0,
        "Transpose mma output can only be used with shuffled A matrix. And vice versa.");

    if (options.mUseTmaStore)
    {
        TLLM_CHECK_WITH_INFO(
            hiddenEpilogueTileSize * trtllm::gen::dtypeGetNumBits(options.mDtypeElt) / /* bits */ 8 % 32 == 0,
            "Unsupported output hidden tile size");
    }

    if (options.mUseDeepSeekFp8)
    {
        TLLM_CHECK_WITH_INFO(hiddenSize % 256 == 0, "Output hidden size must be a multiple of 256");
    }

    gemm::checkAndUpdateGemmOptions(options, isBlackwell, usesAtolFromArg, usesRtolFromArg,
        /* tpGrpSize */ 1);

    if (options.mNumSlicesForSplitK > 1)
    {
        TLLM_CHECK_WITH_INFO(
            doesSplitKUseDsmem(options.mSplitK), "Split-k GMEM and GemmGatedAct are not supported yet.");
    }
}
} // namespace gemmGatedAct

namespace batchedGemm
{
void checkAndUpdateGemmOptions(MyOptions& options, bool isBlackwell, bool usesAtolFromArg, bool usesRtolFromArg)
{
    if (options.mUseFusedAct)
    {
        gemmGatedAct::checkAndUpdateGemmGatedActOptions(options, isBlackwell, usesAtolFromArg, usesRtolFromArg);
    }
    else
    {
        gemm::checkAndUpdateGemmOptions(options, isBlackwell, usesAtolFromArg, usesRtolFromArg, 1 /* tpGrpSize */);
    }

    bool batchM = options.mBatchM;
    if (batchM)
    {
        if (options.mBatchedM.empty())
        {
            options.mBatchedM.push_back(128);
            options.mBatchedM.push_back(256);
        }
        options.mNumBatches = options.mBatchedM.size();
    }
    else
    {
        if (options.mBatchedN.empty())
        {
            options.mBatchedN.push_back(128);
            options.mBatchedN.push_back(256);
        }
        options.mNumBatches = options.mBatchedN.size();
    }

    for (int b = 0; b < options.mNumBatches; b++)
    {
        if (batchM)
        {
            TLLM_CHECK_ERROR(options.mN > 0 && options.mK > 0, "N and K must be larger than 0");
            TLLM_CHECK_ERROR(options.mN >= options.mTileN && options.mK >= options.mTileK,
                "N and K must be equal or larger than TileN and TileK respectively.");
            TLLM_CHECK_ERROR(options.mN % options.mTileN == 0 && options.mK % options.mTileK == 0,
                "N and K must be divisible by TileN and TileK respectively.");
            TLLM_CHECK_ERROR(!options.mTransposeMmaOutput, "When batchM the MMA output has to be in row-major.");
        }
        else
        {
            TLLM_CHECK_ERROR(options.mM > 0 && options.mK > 0, "M and K must be larger than 0");
            TLLM_CHECK_ERROR(options.mM >= options.mTileM && options.mK >= options.mTileK,
                "N and K must be equal or larger than tileN and tileK respectively.");
            TLLM_CHECK_ERROR(options.mM % options.mTileM == 0 && options.mK % options.mTileK == 0,
                "M and K must be divisible by TileM and TileK respectively.");
            TLLM_CHECK_ERROR(options.mTransposeMmaOutput, "When batchN the MMA output has to be in column-major.");
        }
    }

    if (options.mUseDeepSeekFp8)
    {
        if (batchM)
        {
            // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
            TLLM_CHECK_ERROR(
                options.mN % 128 == 0, "GEMM-N must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mN);
        }
        else
        {
            // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
            TLLM_CHECK_ERROR(
                options.mM % 128 == 0, "GEMM-N must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mN);
        }
        // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
        TLLM_CHECK_ERROR(
            options.mK % 128 == 0, "GEMM-K must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mK);

        TLLM_CHECK_ERROR(options.mDtypeC != tg::Dtype::E2m1 && options.mDtypeElt != tg::Dtype::E2m1,
            "E2m1 is not supported with DeepSeek FP8");
    }

    if (options.mAllToAllRouteAct)
    {
        if (batchM)
        {
            TLLM_CHECK_ERROR(options.mNumTokens <= options.mTileM, "Max number of tokens per expert (",
                options.mNumTokens, ") must be smaller than TileM (", options.mTileN, ")");
        }
        else
        {
            TLLM_CHECK_ERROR(options.mNumTokens <= options.mTileN, "Max number of tokens per expert (",
                options.mNumTokens, ") must be smaller than TileN (", options.mTileN, ")");
        }
    }

    TLLM_CHECK_ERROR(options.mUseTmaStore, "Only TMA store is supported.");

    if (options.mAllToAllRouteAct && options.mRouteAct)
    {
        // Turning off mRouteAct when we do mAllToAllRouteAct
        TLLM_LOG_INFO("Turning off mRouteAct when we do mAllToAllRouteAct");
        options.mRouteAct = false;
    }
}
} // namespace batchedGemm

class BatchedGemmData
{
public:
    int mNumCtaX;
    int mNumCtaY;
    int mNumCtaZ;
    void* mA;
    void* mB;
    void* mC;
    float* mScaleC;
    // For DeepSeek FP8 block scaling
    float* mDqSfsA;
    float* mDqSfsB;
    float* mDqSfsC;
    float* mScaleGate;

    // For blocking scaling factors for NVFP4
    void* mSfA;
    void* mSfB;
    void* mSfC;

    void* mSfTokens;

    void* mTokens;
    int32_t const* mRouteMap;

    float* mDqSfsTokens;

    // Pointer for partial row max for DeepSeek computation.
    float* mPtrPartialRowMax;
    // Flags in global memory that sync on "exit" for row max computation.
    uint32_t* mPtrRowMaxCompletionBars;
    // Number of CTAs that do not exit early. For when the workload is smaller than the CTA grid.
    int32_t* mPtrNumNonExitingCtas;
    // Pointer to total number of padded tokens
    int32_t* mPtrTotalNumPaddedTokens;
    // Pointer to CTA index X/Y to batch index
    int32_t* mPtrCtaIdxXyToBatchIdx;
    // Pointer to CTA index X/Y to tile index **expanded** M/N for batched dimension
    int32_t* mPtrCtaIdxXyToMnLimit;
};

void setSingleBatchedGemmData(void* A, void* B, void* C, float* scaleC, float* scaleGate, float* dqSfsA, float* dqSfsB,
    float* dqSfsC, void* sfA, void* sfB, void* sfC, int32_t* permutedIdxToTokenIdx, float* ptrPartialRowMax,
    uint32_t* ptrRowMaxCompletionBars,
    // const bool projUp,
    int const numSlicesForSplitK, MyOptions& args, BatchedGemmData& data, int32_t maxNumPaddedTokens)
{

    data.mB = B;
    data.mA = A;
    data.mC = C;

    if (args.mUseDeepSeekFp8)
    {
        data.mDqSfsA = dqSfsA;
        data.mDqSfsB = dqSfsB;
        data.mDqSfsC = dqSfsC;

        // Avoid illegal read when compiling with debug info
        data.mScaleC = dqSfsC;
        data.mScaleGate = dqSfsC;
    }
    else
    {
        data.mScaleC = scaleC;
        data.mScaleGate = scaleGate;
        data.mSfA = sfA;
        data.mSfB = sfB;
        data.mSfC = sfC;
    }

    if (args.mUseRouteAct)
    {
        data.mTokens = B;
        data.mSfTokens = sfB;
        data.mDqSfsTokens = dqSfsB;
        data.mRouteMap = permutedIdxToTokenIdx;
    }

    // Used in fused activation, will just be nullptrs if not being used
    data.mPtrPartialRowMax = ptrPartialRowMax;
    data.mPtrRowMaxCompletionBars = ptrRowMaxCompletionBars;
    data.mPtrNumNonExitingCtas = args.mPtrNumNonExitingCtas;

    /*
    const int tileN = (projUp) ? args.mProjUpTileN : args.mProjDownTileN;
    const int tileM = (projUp) ? args.mProjUpTileM : args.mProjDownTileM;
    const int m = (projUp) ? 2 * args.mIntermediateDim : args.mHiddenDim;
    */
    int const tileN = args.mTileN;
    int const tileM = args.mTileM;
    int const m = args.mM;

    // The number of tokens per expert and number of active experts are known only at runtime. We
    // launch the max possible number of CTAs and use ptrNumNonExitingCtas to decide if a CTA must
    // run (if its ID < this number) or early-exit.

    // Get maximum number of CTAs in batch dim per expert.
    auto maxCtasInBatchDimPerExpert = gemm::divUp(args.mNumTokens, tileN);
    // Get maximum enabled experts.
    auto maxEnabledExperts = std::min(args.mNumTokens * args.mTopK, args.mNumExperts);
    // Get maximum number of CTAs in batch dim.
    auto maxNumCtasInBatchDim = maxEnabledExperts * maxCtasInBatchDimPerExpert;
    data.mNumCtaY = std::min(maxNumCtasInBatchDim, gemm::divUp(maxNumPaddedTokens, tileN));
    data.mNumCtaX = gemm::divUp(m, tileM);
    data.mNumCtaZ = numSlicesForSplitK;

    // Pointer to total number of padded tokens
    data.mPtrTotalNumPaddedTokens = args.mPtrTotalNumPaddedTokens;
    data.mPtrCtaIdxXyToBatchIdx = args.mPtrCtaIdxXyToBatchIdx;
    data.mPtrCtaIdxXyToMnLimit = args.mPtrCtaIdxXyToMnLimit;
}

void launchGemmFromData(GemmInfo const& kernelInfo, MyOptions const& options, BatchedGemmData const& batchedGemmData,
    CUstream stream, bool usePDL = true)
{
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> cuDriver(
        tensorrt_llm::common::CUDADriverWrapper::getInstance());
    CUmodule cuModule;
    CUfunction cuFunction;
    TLLM_CU_CHECK(cuDriver->cuModuleLoadData(&cuModule, kernelInfo.data));
    TLLM_CU_CHECK(cuDriver->cuModuleGetFunction(&cuFunction, cuModule, kernelInfo.functionName));
    if (kernelInfo.sharedMemSize >= 48 * 1024)
    {
        TLLM_CU_CHECK(cuDriver->cuFuncSetAttribute(
            cuFunction, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelInfo.sharedMemSize));
    }

    auto params = KernelParams::setKernelParams(options, batchedGemmData.mA, batchedGemmData.mB, batchedGemmData.mC,
        batchedGemmData.mScaleC, batchedGemmData.mDqSfsA, batchedGemmData.mDqSfsB, batchedGemmData.mDqSfsC,
        batchedGemmData.mSfA, batchedGemmData.mSfB, batchedGemmData.mSfTokens, batchedGemmData.mSfC,
        batchedGemmData.mScaleGate, batchedGemmData.mTokens, batchedGemmData.mRouteMap, batchedGemmData.mDqSfsTokens,
        batchedGemmData.mPtrPartialRowMax, batchedGemmData.mPtrRowMaxCompletionBars,
        batchedGemmData.mPtrNumNonExitingCtas, batchedGemmData.mPtrTotalNumPaddedTokens,
        batchedGemmData.mPtrCtaIdxXyToBatchIdx, batchedGemmData.mPtrCtaIdxXyToMnLimit);
    CUlaunchConfig launch_config;
    launch_config.blockDimX = kernelInfo.threadsPerCTA;
    launch_config.blockDimY = 1;
    launch_config.blockDimZ = 1;
    launch_config.gridDimX = batchedGemmData.mNumCtaX;
    launch_config.gridDimY = batchedGemmData.mNumCtaY;
    launch_config.gridDimZ = batchedGemmData.mNumCtaZ;
    launch_config.hStream = stream;
    launch_config.sharedMemBytes = kernelInfo.sharedMemSize;
    CUlaunchAttribute launch_attribute[3];
    launch_attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    launch_attribute[0].value.clusterDim.x = 1;
    launch_attribute[0].value.clusterDim.y = 1;
    launch_attribute[0].value.clusterDim.z = 1;
    launch_attribute[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    launch_attribute[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;
    launch_attribute[2].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    launch_attribute[2].value.programmaticStreamSerializationAllowed = static_cast<int>(usePDL);
    if (!usePDL)
    {
        launch_attribute[2].value.programmaticStreamSerializationAllowed = 0;
    }
    else
    {
        // Even if true, respect the PDL environment variable
        launch_attribute[2].value.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    }
    launch_config.attrs = launch_attribute;
    launch_config.numAttrs = 3;

    // std::cout << "kernelInfo.data: " << kernelInfo.data << std::endl;
    // std::cout << "kernelInfo.functionName: " << kernelInfo.functionName << std::endl;

    TLLM_CHECK_WITH_INFO(kernelInfo.paramsStructSize == sizeof(params), "Alignment issue detected");
    void* kernelParamsList[] = {&params};
    TLLM_CU_CHECK(cuDriver->cuLaunchKernelEx(&launch_config, cuFunction, kernelParamsList, nullptr));
}
} // namespace gemmCommon
} // namespace trtllmGenFp8BlockScaleMoe
} // namespace kernels
} // namespace tensorrt_llm
