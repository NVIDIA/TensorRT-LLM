/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <set>
#include <unistd.h>
#include <vector>

#include "KernelRunner.h"
#include "trtllmGen_bmm_export/BatchedGemmInterface.h"
#include "trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
// DO NOT include cudaUtils.h and logger.h before BatchedGemmInterface.h as it #undef TLLM_LOG_INFO and co.
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

using namespace batchedGemm::batchedGemm;
using namespace batchedGemm::gemm;
using namespace batchedGemm::trtllm::gen;

static BatchedGemmInterface::ModuleCache globalTrtllmGenBatchedGemmModuleCache;
static std::set<std::string> printedBmmNames;

using tensorrt_llm::common::fmtstr;

constexpr bool isSMCompatible(int gpuSM, SmVersion kernelSM)
{
    if (gpuSM == 103)
    {
        return kernelSM == SmVersion::Sm100f || kernelSM == SmVersion::Sm103a;
    }
    else if (gpuSM == 100)
    {
        return kernelSM == SmVersion::Sm100f || kernelSM == SmVersion::Sm100a;
    }
    else if (gpuSM == 90)
    {
        return kernelSM == SmVersion::Sm90a;
    }

    TLLM_THROW("Unexpected gpuSM %d", gpuSM);
    return false;
}

static inline bool skipQuirks(BatchedGemmConfig const& config)
{
    // Skip kernels that are known to hang/crash. Keep a record here for future reference.
    auto const& options = config.mOptions;

    // BatchM 2CTA kernel hangs randomly for both FC1 and FC2. Skip for now. Including:
    bool const hang_c2x1_bM = options.mTileM == 128 && options.mClusterDimX == 2 && options.mClusterDimY == 1
        && options.mClusterDimZ == 1 && !options.mTransposeMmaOutput;

    // Static scheduler + TmaOobOpt + TileN=64. Skip for now. Including:
    // bmm_E2m1_E2m1E2m1_Fp32_t128x64x256_s6_et128x64_m128x64x64_cga1x1x1_16dp256b_TN_transOut_schedS_bN_ldgsts_tmaOpt_clmp_swiGlu_dynBatch_sm100a
    // bmm_MxE4m3_MxE2m1MxE4m3_Fp32_t128x64x256_s3_et128x64_m128x64x32_cga1x1x1_16dp256b_TN_transOut_schedS_biasM_bN_ldgsts_tmaOpt_clmp_swiGlu_dynBatch_sm100f
    bool const hang_schedS_tmaOob_tileN64
        = options.mTileScheduler == TileScheduler::Static && options.mUseTmaOobOpt && options.mTileN == 64;

    return hang_c2x1_bM || hang_schedS_tmaOob_tileN64;
}

void setProblemDimensions(BatchedGemmData& gemmData, bool transposeMmaOutput, int32_t m, int32_t n, int32_t k,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim,
    int32_t validM, int32_t validN, int32_t validK)
{
    gemmData.mProblemDimensions.mNumBatches = numBatches;
    gemmData.mProblemDimensions.mNumTokens = numTokens;
    gemmData.mProblemDimensions.mBatchM = !transposeMmaOutput;
    gemmData.mProblemDimensions.mBatchedM = transposeMmaOutput ? std::vector<int32_t>{} : batchedTokens;
    gemmData.mProblemDimensions.mBatchedN = transposeMmaOutput ? batchedTokens : std::vector<int32_t>{};
    gemmData.mProblemDimensions.mM = transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mValidM = transposeMmaOutput ? validN : validM;
    gemmData.mProblemDimensions.mValidN = transposeMmaOutput ? validM : validN;
    gemmData.mProblemDimensions.mValidK = validK;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;
    gemmData.mProblemDimensions.mMaxNumCtasInTokenDim = maxNumCtasInBatchDim;
}

TrtllmGenBatchedGemmRunner::TrtllmGenBatchedGemmRunner(TrtllmGenBatchedGemmRunnerOptions const& options_)
    : mOptions(options_)
{
    // Select a GEMM kernel config to use
    auto const bmm = BatchedGemmInterface();
    auto const configs = bmm.getBatchedGemmConfigs();

    mPassingConfigIndices.clear();

    // Check if detailed kernel rejection logging is enabled
    bool enableRejectLog = tensorrt_llm::common::getIntEnv("TLLM_BATCHED_GEMM_LOG_REJECTION").value_or(0) != 0;

    std::vector<std::string> rejectReason;
    if (enableRejectLog)
    {
        rejectReason.resize(bmm.getNumBatchedGemmConfigs());
    }

    int gpuSM = tensorrt_llm::common::getSMVersion();
    for (size_t i = 0; i < bmm.getNumBatchedGemmConfigs(); ++i)
    {
        auto acceptIf = [i, &rejectReason, enableRejectLog](bool condition, std::string const& reason) -> bool
        {
            if (condition)
            {
                return true;
            }
            else if (enableRejectLog)
            {
                rejectReason[i] = reason;
            }
            return false;
        };
        auto const& config = configs[i];
        auto const& options = config.mOptions;
        auto const tileSize = options.mTransposeMmaOutput ? options.mTileN * options.mClusterDimY
                                                          : options.mTileM * options.mClusterDimX;

        // Transpose kernels swap A/B operands relative to the canonical convention (A is activation, B is weight).
        bool const dtypeMatch = options.mTransposeMmaOutput
            // SwapAB: A is weight, B is activation. Opposite of canonical convention.
            ? (options.mDtypeA == mOptions.dtypeB && options.mDtypeB == mOptions.dtypeA)
            // NoSwap: A is activation, B is weight.
            : (options.mDtypeA == mOptions.dtypeA && options.mDtypeB == mOptions.dtypeB);

        if (!acceptIf(!skipQuirks(config), "skipQuirks: known hanging kernel"))
        {
            continue;
        }

        if (!acceptIf(dtypeMatch,
                fmtstr("dtypeAB mismatch (kernel: %s/%s, expected canonical: %s/%s)",
                    tg::dtypeToString(options.mDtypeA).c_str(), tg::dtypeToString(options.mDtypeB).c_str(),
                    tg::dtypeToString(mOptions.dtypeA).c_str(), tg::dtypeToString(mOptions.dtypeB).c_str())))
        {
            continue;
        }

        if (!acceptIf(options.mDtypeC == mOptions.dtypeC,
                fmtstr("dtypeC mismatch (kernel: %s, expected: %s)", tg::dtypeToString(options.mDtypeC).c_str(),
                    tg::dtypeToString(mOptions.dtypeC).c_str())))
        {
            continue;
        }

        if (!acceptIf(options.mUseDeepSeekFp8 == mOptions.deepSeekFp8,
                fmtstr(
                    "deepSeekFp8 mismatch (kernel: %d, expected: %d)", options.mUseDeepSeekFp8, mOptions.deepSeekFp8)))
        {
            continue;
        }

        if (!acceptIf((!doesRouteImplUseNoRoute(options.mRouteImpl)) == mOptions.routeAct,
                fmtstr("routeAct mismatch (kernel: %d, expected: %d)", !doesRouteImplUseNoRoute(options.mRouteImpl),
                    mOptions.routeAct)))
        {
            continue;
        }

        if (!acceptIf(options.mFusedAct == mOptions.fusedAct,
                fmtstr("fusedAct mismatch (kernel: %d, expected: %d)", options.mFusedAct, mOptions.fusedAct)))
        {
            continue;
        }

        if (!acceptIf(options.mIsStaticBatch == mOptions.staticBatch,
                fmtstr(
                    "staticBatch mismatch (kernel: %d, expected: %d)", options.mIsStaticBatch, mOptions.staticBatch)))
        {
            continue;
        }

        if (!acceptIf(tileSize == mOptions.tileSize,
                fmtstr("tileSize mismatch (kernel: %d, expected: %d)", tileSize, mOptions.tileSize)))
        {
            continue;
        }

        if (!acceptIf(isSMCompatible(gpuSM, configs[i].mSm),
                fmtstr("SM not compatible (gpuSM: %d, kernelSM: %d)", gpuSM, static_cast<int>(configs[i].mSm))))
        {
            continue;
        }

        auto sm = configs[i].mSm;
        if (sm != SmVersion::Sm100f)
        {
            int smVersion = tensorrt_llm::common::getSMVersion();
            if (smVersion == 100)
            {
                if (!acceptIf(sm == SmVersion::Sm100a,
                        fmtstr("SM version 100 requires Sm100a (kernel has: %d)", static_cast<int>(sm))))
                {
                    continue;
                }
            }
            else if (smVersion == 103)
            {
                if (!acceptIf(sm == SmVersion::Sm103a,
                        fmtstr("SM version 103 requires Sm103a (kernel has: %d)", static_cast<int>(sm))))
                {
                    continue;
                }
            }
        }

        if (options.mUseDeepSeekFp8)
        {
            if (!acceptIf(options.mUseShuffledMatrix == false, "useShuffledMatrixA should be false for DeepSeek Fp8"))
            {
                continue;
            }
        }

        if (options.mFusedAct)
        {
            if (!acceptIf(options.mActType == static_cast<batchedGemm::gemmGatedAct::ActType>(mOptions.actType),
                    fmtstr("actType mismatch (kernel: %d, expected: %d)", static_cast<int>(options.mActType),
                        static_cast<int>(mOptions.actType))))
            {
                continue;
            }
        }
        if (!acceptIf((int64_t) options.mEltwiseActType == (int64_t) mOptions.eltwiseActType,
                fmtstr("eltwiseActType mismatch (kernel: %ld, expected: %ld)", (int64_t) options.mEltwiseActType,
                    (int64_t) mOptions.eltwiseActType)))
        {
            continue;
        }

        if (!acceptIf(options.mEpilogueTileM == mOptions.epilogueTileM,
                fmtstr("epilogueTileM mismatch (kernel: %d, expected: %d)", options.mEpilogueTileM,
                    mOptions.epilogueTileM)))
        {
            continue;
        }

        // Kernel passed all filters
        mPassingConfigIndices.push_back(i);
    }

    if (mPassingConfigIndices.empty())
    {

        auto errMsg = fmtstr(
            "No kernel found for the given options: mDtypeA: %s, mDtypeB: %s, mDtypeC: %s, mUseDeepSeekFp8: %d, "
            "mTransposeMmaOutput: %s, mRouteAct: %d, mFusedAct: %d, mIsStaticBatch: %d, mTileSize: %d, "
            "mEltwiseActType: %ld, mEpilogueTileM: %d",
            tg::dtypeToString(mOptions.dtypeA).c_str(), tg::dtypeToString(mOptions.dtypeB).c_str(),
            tg::dtypeToString(mOptions.dtypeC).c_str(), mOptions.deepSeekFp8, "auto-tuned", mOptions.routeAct,
            mOptions.fusedAct, mOptions.staticBatch, mOptions.tileSize, (long) mOptions.eltwiseActType,
            mOptions.epilogueTileM);

        if (enableRejectLog)
        {
            // Show all rejection reasons for all kernels
            errMsg += fmtstr("\n\nRejection details for all %zu kernel(s):\n", bmm.getNumBatchedGemmConfigs());

            for (size_t i = 0; i < bmm.getNumBatchedGemmConfigs(); ++i)
            {
                errMsg += fmtstr("\n[%zu] %s\n    ", i, configs[i].mFunctionName);

                if (rejectReason[i] == "")
                {
                    errMsg += "PASSED\n";
                }
                else
                {
                    errMsg += fmtstr("REJECTED: %s\n", rejectReason[i].c_str());
                }
            }
        }
        else
        {
            errMsg
                += "\n\nTo see detailed rejection reasons, set environment variable: TLLM_BATCHED_GEMM_LOG_REJECTION=1";
        }

        TLLM_CHECK_WITH_INFO(false, errMsg);
    }
}

size_t TrtllmGenBatchedGemmRunner::getWorkspaceSizeInBytes(int32_t m, int32_t n, int32_t k,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim,
    int32_t configIndex) const
{
    BatchedGemmData gemmData;

    auto bmm = BatchedGemmInterface();

    auto const configs = bmm.getBatchedGemmConfigs();

    auto const& config = configs[configIndex];
    auto const transposeMmaOutput = config.mOptions.mTransposeMmaOutput;

    setProblemDimensions(
        gemmData, transposeMmaOutput, m, n, k, batchedTokens, numTokens, numBatches, maxNumCtasInBatchDim, m, n, k);

    return bmm.getWorkspaceSizeInBytes(config, gemmData);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, int32_t validM, int32_t validN, int32_t validK,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim,
    void const* a, void const* sfA, void const* b, void const* sfB, void const* perTokensSfA, void const* perTokensSfB,
    float const* scaleC, float const* scaleGateC, float const* ptrBias, float const* ptrAlpha, float const* ptrBeta,
    float const* ptrClampLimit, void* c, void* outSfC, int32_t const* routeMap, int32_t const* totalNumPaddedTokens,
    int32_t const* ctaIdxXyToBatchIdx, int32_t const* ctaIdxXyToMnLimit, int32_t const* numNonExitingCtas,
    void* workspace, CUstream stream, int device, int32_t configIndex)
{
    auto bmm = BatchedGemmInterface();

    BatchedGemmData gemmData;

    auto const configs = bmm.getBatchedGemmConfigs();

    auto const& config = configs[configIndex];
    auto const transposeMmaOutput = config.mOptions.mTransposeMmaOutput;

    TLLM_CHECK_WITH_INFO(numBatches > 0, "Batched GEMM requires numBatches > 0");
    if (!mOptions.staticBatch)
    {
        TLLM_CHECK_WITH_INFO(totalNumPaddedTokens, "Batched GEMM with dynamic batching requires totalNumPaddedTokens");
        TLLM_CHECK_WITH_INFO(ctaIdxXyToBatchIdx, "Batched GEMM with dynamic batching requires ctaIdxXyToBatchIdx");
        TLLM_CHECK_WITH_INFO(ctaIdxXyToMnLimit, "Batched GEMM with dynamic batching requires ctaIdxXyToMnLimit");
        TLLM_CHECK_WITH_INFO(numNonExitingCtas, "Batched GEMM with dynamic batching requires numNonExitingCtas");
    }

    if (!mOptions.staticBatch && numTokens != 0)
    {
        TLLM_CHECK_WITH_INFO(
            maxNumCtasInBatchDim > 0, "Batched GEMM with dynamic batching requires maxNumCtasInBatchDim > 0");
    }

    if (mOptions.routeAct)
    {
        TLLM_CHECK_WITH_INFO(routeMap, "Batched GEMM with routeAct requires routeMap");
        TLLM_CHECK_WITH_INFO(numTokens > 0, "Batched GEMM with routeAct requires numTokens > 0");
    }

    // Sanitize optional valid dimensions
    validM = validM <= 0 ? m : validM;
    validN = validN <= 0 ? n : validN;
    validK = validK <= 0 ? k : validK;

    // Dims
    setProblemDimensions(gemmData, transposeMmaOutput, m, n, k, batchedTokens, numTokens, numBatches,
        maxNumCtasInBatchDim, validM, validN, validK);

    // Inputs
    gemmData.mInputBuffers.mPtrA = transposeMmaOutput ? b : a;
    gemmData.mInputBuffers.mPtrSfA = transposeMmaOutput ? sfB : sfA;
    gemmData.mInputBuffers.mPtrB = transposeMmaOutput ? a : b;
    gemmData.mInputBuffers.mPtrSfB = transposeMmaOutput ? sfA : sfB;
    gemmData.mInputBuffers.mPtrScaleC = scaleC;
    gemmData.mInputBuffers.mPtrScaleGate = scaleGateC;
    gemmData.mInputBuffers.mPtrScaleAct = scaleGateC;
    gemmData.mInputBuffers.mPtrPerTokenSfA = transposeMmaOutput ? perTokensSfB : perTokensSfA;
    gemmData.mInputBuffers.mPtrPerTokenSfB = transposeMmaOutput ? perTokensSfA : perTokensSfB;
    gemmData.mInputBuffers.mPtrBias = ptrBias;
    gemmData.mInputBuffers.mPtrGatedActAlpha = ptrAlpha;
    gemmData.mInputBuffers.mPtrGatedActBeta = ptrBeta;
    gemmData.mInputBuffers.mPtrClampLimit = ptrClampLimit;

    gemmData.mInputBuffers.mPtrRouteMap = routeMap;

    // Pointer to total number of padded tokens
    gemmData.mInputBuffers.mPtrTotalNumPaddedTokens = totalNumPaddedTokens;
    gemmData.mInputBuffers.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    gemmData.mInputBuffers.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    gemmData.mInputBuffers.mPtrNumNonExitingCtas = numNonExitingCtas;

    // Outputs
    gemmData.mOutputBuffers.mPtrC = c;
    gemmData.mOutputBuffers.mPtrSfC = outSfC;

    int32_t multiProcessorCount;
    cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device);

    auto envVarVal = std::getenv("TLLM_BATCHED_GEMM_PRINT_NAME");
    if (envVarVal && std::atoi(envVarVal) == 1)
    {
        auto msg = fmtstr(
            "[PID %d] NumBatches %d, MaxNumCtasInBatchDim %d, ShapeMNK %d %d %d, ValidShapeMNK %d %d %d, Kernel %s",
            getpid(), numBatches, maxNumCtasInBatchDim, m, n, k, validM, validN, validK, config.mFunctionName);
        if (printedBmmNames.find(msg) == printedBmmNames.end())
        {
            printedBmmNames.insert(msg);
            TLLM_LOG_INFO(msg);
        }
    }
    // FIXME once we start using all-reduce in the epilogue of the bmm this can be moved elsewhere
    bmm.runInitBeforeWorldSync(config, gemmData, static_cast<void*>(stream));

    auto const err = bmm.run(config, workspace, gemmData, static_cast<void*>(stream), multiProcessorCount,
        tensorrt_llm::common::getEnvEnablePDL(), nullptr, globalTrtllmGenBatchedGemmModuleCache);

    CUresult cuErr = static_cast<CUresult>(err);
    char const* cuErrStr = nullptr;
    cuGetErrorString(cuErr, &cuErrStr);
    char const* cuErrName = nullptr;
    cuGetErrorName(cuErr, &cuErrName);
    TLLM_CHECK_WITH_INFO(cuErr == CUDA_SUCCESS,
        "Error occurred when running GEMM! Error %s (%s)"
        " (numBatches: %d, GemmMNK: %d %d %d, Kernel: %s)",
        cuErrName ? cuErrName : "UNKNOWN", cuErrStr ? cuErrStr : "Unknown error", numBatches, m, n, k,
        config.mFunctionName);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens,
    void const* a, void const* sfA, void const* b, void const* sfB, void* c, void* outSfC, void* workspace,
    CUstream stream, int device, int32_t configIndex, int32_t validM, int32_t validN, int32_t validK)
{
    // Dispatch with block scaling factors and with static batching.
    run(m, n, k, validM, validN, validK, batchedTokens, /* numTokens */ 0, batchedTokens.size(),
        /* maxNumCtasInBatchDim */ 0, a, sfA, b, sfB,
        /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr,
        /* scaleC */ nullptr, /* scaleGateC */ nullptr, /* ptrBias */ nullptr, /* ptrAlpha */ nullptr,
        /* ptrBeta */ nullptr, /* ptrClampLimit */ nullptr, c, outSfC,
        /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
        /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
        /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens,
    void const* a, void const* sfA, void const* b, void const* sfB, float const* ptrBias, float const* ptrAlpha,
    float const* ptrBeta, float const* ptrClampLimit, void* c, void* outSfC, void* workspace, CUstream stream,
    int device, int32_t configIndex, int32_t validM, int32_t validN, int32_t validK)
{
    // Dispatch with block scaling factors and with static batching.
    run(m, n, k, validM, validN, validK, batchedTokens, /* numTokens */ 0, batchedTokens.size(),
        /* maxNumCtasInBatchDim */ 0, a, sfA, b, sfB,
        /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr,
        /* scaleC */ nullptr, /* scaleGateC */ nullptr, ptrBias, ptrAlpha, ptrBeta, ptrClampLimit, c, outSfC,
        /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
        /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
        /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens,
    void const* a, void const* b, float const* scaleC, float const* scaleGateC, void* c, void* workspace,
    CUstream stream, int device, int32_t configIndex, int32_t validM, int32_t validN, int32_t validK)
{
    // Dispatch with block scaling factors and with static batching.
    run(m, n, k, validM, validN, validK, batchedTokens, /* numTokens */ 0, batchedTokens.size(),
        /* maxNumCtasInBatchDim */ 0, a,
        /* sfA */ nullptr, b, /* sfB */ nullptr, /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr, scaleC,
        scaleGateC, /* ptrBias */ nullptr, /* ptrAlpha */ nullptr, /* ptrBeta */ nullptr, /* ptrClampLimit */ nullptr,
        c,
        /* outSfC */ nullptr,
        /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
        /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
        /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex);
}

std::string TrtllmGenBatchedGemmRunner::getKernelNameFromConfigIndex(int32_t configIndex) const
{
    auto const bmm = BatchedGemmInterface();
    auto const configs = bmm.getBatchedGemmConfigs();
    return configs[configIndex].mFunctionName;
}

std::vector<int64_t> TrtllmGenBatchedGemmRunner::getValidConfigIndices(int32_t m, int32_t n, int32_t k,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim,
    int32_t validM, int32_t validN, int32_t validK) const
{
    auto const bmm = BatchedGemmInterface();
    auto const configs = bmm.getBatchedGemmConfigs();

    int32_t multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    BatchedGemmData gemmData;

    // Sanitize optional valid dimensions
    validM = validM <= 0 ? m : validM;
    validN = validN <= 0 ? n : validN;
    validK = validK <= 0 ? k : validK;

    setProblemDimensions(gemmData, /* transposeMmaOutput = */ false, m, n, k, batchedTokens, numTokens, numBatches,
        maxNumCtasInBatchDim, validM, validN, validK);
    auto cmpFunc = [&configs, &bmm, &multiProcessorCount, &m, &n, &k, &batchedTokens, &numTokens, &numBatches,
                       &maxNumCtasInBatchDim, &validM, &validN, &validK](int64_t idx0, int64_t idx1)
    {
        auto const& optionsA = configs[idx0].mOptions;
        auto const& optionsB = configs[idx1].mOptions;
        int32_t sizeK = k;

        if (optionsA.mTransposeMmaOutput != optionsB.mTransposeMmaOutput)
        {
            return optionsA.mTransposeMmaOutput;
        }

        // Tier 0: K < tileK, prefer higher efficiency.
        if (optionsA.mTileK != optionsB.mTileK)
        {
            // Both waste computation, prefer higher efficiency.
            if (sizeK <= optionsA.mTileK && sizeK <= optionsB.mTileK)
            {
                double eff_a = (double) sizeK / optionsA.mTileK;
                double eff_b = (double) sizeK / optionsB.mTileK;
                return eff_a > eff_b;
            }
            // If either can be utilized, sort by tileK.
            else
            {
                return optionsA.mTileK > optionsB.mTileK;
            }
        }

        // Tier 1: When tileK is the same, prefer unroll loop 2x for mma.
        if (optionsA.mUseUnrollLoop2xForMma != optionsB.mUseUnrollLoop2xForMma)
        {
            return optionsA.mUseUnrollLoop2xForMma;
        }

        // Tier 2+: When previous comparators are the same, prefer higher tileM.
        if (optionsA.mTileM != optionsB.mTileM)
        {
            return optionsA.mTileM > optionsB.mTileM;
        }

        // Tier 2+: When previous comparators are the same, prefer higher tileN.
        if (optionsA.mTileN != optionsB.mTileN)
        {
            return optionsA.mTileN > optionsB.mTileN;
        }

        // Tier 2+: When previous comparators are the same, and when the number of estimated CTAs is on the larger side,
        // prefer persistent tile scheduler.
        if (optionsA.mTileScheduler != optionsB.mTileScheduler)
        {
            BatchedGemmData gemmData;
            setProblemDimensions(gemmData, optionsA.mTransposeMmaOutput, m, n, k, batchedTokens, numTokens, numBatches,
                maxNumCtasInBatchDim, validM, validN, validK);
            auto options = bmm.getOptionsFromConfigAndData(configs[idx0], gemmData);
            auto numCtas = bmm.getNumCtas(options, gemmData.mProblemDimensions.mMaxNumCtasInTokenDim);
            if (numCtas > multiProcessorCount)
            {
                return optionsA.mTileScheduler == batchedGemm::gemm::TileScheduler::Persistent;
            }
            else
            {
                return optionsB.mTileScheduler == batchedGemm::gemm::TileScheduler::Persistent;
            }
        }

        return false;
    };
    // Sort configs by options.
    std::vector<int64_t> sortedIndices = mPassingConfigIndices;
    std::sort(sortedIndices.begin(), sortedIndices.end(), cmpFunc);

    // Filter out invalid configs.
    std::vector<int64_t> validConfigIndices;
    for (auto const& configIndex : sortedIndices)
    {
        auto const& config = configs[configIndex];
        setProblemDimensions(gemmData, config.mOptions.mTransposeMmaOutput, m, n, k, batchedTokens, numTokens,
            numBatches, maxNumCtasInBatchDim, validM, validN, validK);
        auto isValidConfig = bmm.isValidConfig(config, gemmData);
        if (isValidConfig)
        {
            validConfigIndices.push_back(configIndex);
        }
    }

    TLLM_CHECK_WITH_INFO(!validConfigIndices.empty(),
        "No valid config found for the given problem shape MNK %d %d %d and effective MNK range %d %d %d", m, n, k,
        validM, validN, validK);

    return validConfigIndices;
}

int64_t TrtllmGenBatchedGemmRunner::getDefaultValidConfigIndex(int32_t m, int32_t n, int32_t k,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim,
    int32_t validM, int32_t validN, int32_t validK) const
{
    auto const validConfigIndices = getValidConfigIndices(
        m, n, k, batchedTokens, numTokens, numBatches, maxNumCtasInBatchDim, validM, validN, validK);

    return validConfigIndices[0];
}

bool TrtllmGenBatchedGemmRunner::isValidConfigIndex(int32_t configIndex, int32_t m, int32_t n, int32_t k,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim,
    int32_t validM, int32_t validN, int32_t validK) const
{
    auto const bmm = BatchedGemmInterface();
    auto const configs = bmm.getBatchedGemmConfigs();

    BatchedGemmData gemmData;

    // Sanitize optional valid dimensions
    validM = validM <= 0 ? m : validM;
    validN = validN <= 0 ? n : validN;
    validK = validK <= 0 ? k : validK;

    // Dims
    auto const& config = configs[configIndex];
    setProblemDimensions(gemmData, config.mOptions.mTransposeMmaOutput, m, n, k, batchedTokens, numTokens, numBatches,
        maxNumCtasInBatchDim, validM, validN, validK);

    return bmm.isValidConfig(config, gemmData);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
