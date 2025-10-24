/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cublasVersionCheck.h"
#include <algorithm>
#include <unordered_map>

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace tensorrt_llm
{
namespace common
{

CublasMMWrapper::CublasMMWrapper(std::shared_ptr<cublasHandle_t> cublasHandle,
    std::shared_ptr<cublasLtHandle_t> cublasltHandle, cudaStream_t stream, void* workspace)
    : mCublasHandle(cublasHandle)
    , mCublasLtHandle(cublasltHandle)
    , mStream(stream)
    , mCublasWorkspace(workspace)
{
}

CublasMMWrapper::~CublasMMWrapper() {}

CublasMMWrapper::CublasMMWrapper(CublasMMWrapper const& wrapper)
    : mCublasHandle(wrapper.mCublasHandle)
    , mCublasLtHandle(wrapper.mCublasLtHandle)
    , mStream(wrapper.mStream)
{
}

void CublasMMWrapper::createDescriptors(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n,
    int const k, int const lda, int const ldb, int const ldc, int8_t fastAcc)
{
    // --------------------------------------
    // Create descriptors for the original matrices
    check_cuda_error(
        cublasLtMatrixLayoutCreate(&mADesc, mAType, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    check_cuda_error(
        cublasLtMatrixLayoutCreate(&mBDesc, mBType, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    check_cuda_error(cublasLtMatrixLayoutCreate(&mCDesc, mCType, m, n, ldc));
    check_cuda_error(cublasLtMatmulDescCreate(&mOperationDesc, mComputeType, mScaleType));
    check_cuda_error(cublasLtMatmulDescSetAttribute(
        mOperationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    check_cuda_error(cublasLtMatmulDescSetAttribute(
        mOperationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(mOperationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAcc, sizeof(int8_t)));

#ifdef ENABLE_CUBLASLT_FP4_GEMM
    // Set pointer mode for FP4 GEMM
    if (mAType == CUDA_R_4F_E2M1)
    {
        cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            mOperationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));
    }
#endif
}

void CublasMMWrapper::setScaleDescriptors(void* scale_a, void* scale_b)
{
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(mOperationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(void*)));
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(mOperationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(void*)));

    // Set scaling modes for FP4 GEMM
    if (mAType == CUDA_R_4F_E2M1)
    {
        // Set scaling mode - cuBLASLt requires e4m3 format scaling factors
        cublasLtMatmulMatrixScale_t AScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulMatrixScale_t BScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulMatrixScale_t CScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        cublasLtMatmulMatrixScale_t DScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        cublasLtMatmulMatrixScale_t DOutScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;

        check_cuda_error(cublasLtMatmulDescSetAttribute(
            mOperationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &AScaleMode, sizeof(AScaleMode)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            mOperationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &BScaleMode, sizeof(BScaleMode)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            mOperationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, &CScaleMode, sizeof(CScaleMode)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            mOperationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &DScaleMode, sizeof(DScaleMode)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            mOperationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &DOutScaleMode, sizeof(DOutScaleMode)));

        // Set C/D matrix scale pointers to nullptr
        void const* c_scale_ptr = nullptr;
        void const* d_scale_ptr = nullptr;
        void const* d_out_scale_ptr = nullptr;
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            mOperationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale_ptr, sizeof(c_scale_ptr)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            mOperationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_ptr, sizeof(d_scale_ptr)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            mOperationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_out_scale_ptr, sizeof(d_out_scale_ptr)));
    }
}

void CublasMMWrapper::setBiasDescriptor(void* bias)
{
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(mOperationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*)));

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(mOperationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
}

void CublasMMWrapper::destroyDescriptors()
{
    check_cuda_error(cublasLtMatmulDescDestroy(mOperationDesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(mADesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(mBDesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(mCDesc));
    mOperationDesc = NULL;
    mADesc = NULL;
    mBDesc = NULL;
    mCDesc = NULL;
}

void CublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
    void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc)
{
    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f);
}

void CublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
    void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc,
    std::optional<cublasLtMatmulHeuristicResult_t> const& heuristic)
{
    if (heuristic)
    {
        Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f, /* hasAlgo */ (*heuristic).algo,
            (*heuristic).state == CUBLAS_STATUS_SUCCESS && (*heuristic).workspaceSize < CUBLAS_WORKSPACE_SIZE,
            /* usingCublasLt */ true);
    }
    else
    {
        Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f, {}, /* hasAlgo */ false,
            /* usingCublasLt */ true);
    }
}

void CublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
    void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc, float f_alpha, float f_beta,
    std::optional<cublasLtMatmulHeuristicResult_t> const& heuristic)
{
    if (heuristic)
    {
        Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, f_alpha, f_beta, /* hasAlgo */ (*heuristic).algo,
            (*heuristic).state == CUBLAS_STATUS_SUCCESS && (*heuristic).workspaceSize < CUBLAS_WORKSPACE_SIZE,
            /* usingCublasLt */ true);
    }
    else
    {
        Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, f_alpha, f_beta, {}, /* hasAlgo */ false,
            /* usingCublasLt */ true);
    }
}

void CublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
    void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc, float f_alpha, float f_beta)
{
    bool usingCublasLt = mAType == CUDA_R_16F || mAType == CUDA_R_8F_E4M3;

    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, f_alpha, f_beta, {}, /* hasAlgo */ false,
        /* usingCublasLt */ usingCublasLt);
}

void CublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
    void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc, float f_alpha, float f_beta,
    cublasLtMatmulAlgo_t const& algo, bool hasAlgo, bool usingCublasLt)
{
    half h_alpha = (half) (f_alpha);
    half h_beta = (half) (f_beta);

    // TODO: default cublas libs
    usingCublasLt = usingCublasLt && (mAType == CUDA_R_16F || mAType == CUDA_R_8F_E4M3 || mAType == CUDA_R_16BF);
    bool isFp16ComputeType = mComputeType == CUBLAS_COMPUTE_16F;
    // fp32 use cublas as default
    // fp16 use cublasLt as default
    void const* alpha = isFp16ComputeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    void const* beta = isFp16ComputeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);
    int workspaceSize = mCublasWorkspace == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

    if (usingCublasLt)
    {
        if (hasAlgo)
        {
            hasAlgo = checkTactic(transa, transb, m, n, k, lda, ldb, ldc, algo);
        }

        check_cuda_error(cublasLtMatmul(getCublasLtHandle(), mOperationDesc, alpha, A, mADesc, B, mBDesc, beta, C,
            mCDesc, C, mCDesc, (hasAlgo ? (&algo) : NULL), mCublasWorkspace, workspaceSize, mStream));

        sync_check_cuda_error(mStream);
    }
    else
    {
        check_cuda_error(cublasSetStream(getCublasHandle(), mStream));
        check_cuda_error(cublasSetWorkspace(getCublasHandle(), mCublasWorkspace, workspaceSize));
        // Go with default heuristic to choose tactic as cuBLAS does not allow to choose tactics in Ampere+
        cublasGemmAlgo_t cublasAlgo = CUBLAS_GEMM_DEFAULT;
        check_cuda_error(cublasGemmEx(getCublasHandle(), transa, transb, m, n, k, alpha, A, mAType, lda, B, mBType, ldb,
            beta, C, mCType, ldc, mComputeType, static_cast<cublasGemmAlgo_t>(cublasAlgo)));
        sync_check_cuda_error(mStream);
    }
}

void CublasMMWrapper::stridedBatchedGemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n,
    int const k, void const* A, int const lda, const int64_t strideA, void const* B, int const ldb,
    const int64_t strideB, void* C, int const ldc, const int64_t strideC, int const batchCount, float const f_alpha,
    float const f_beta)
{
    half h_alpha = (half) f_alpha;
    half h_beta = (half) f_beta;

    int isFp16ComputeType = mComputeType == CUBLAS_COMPUTE_16F ? 1 : 0;
    void const* alpha = isFp16ComputeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void const*>(&f_alpha);
    void const* beta = isFp16ComputeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void const*>(&f_beta);

    check_cuda_error(cublasGemmStridedBatchedEx(getCublasHandle(), transa, transb, m, n, k, alpha, A, mAType, lda,
        strideA, B, mBType, ldb, strideB, beta, C, mCType, ldc, strideC, batchCount, mComputeType,
        mAType == CUDA_R_32F ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void CublasMMWrapper::stridedBatchedGemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n,
    int const k, float const f_alpha, void const* A, cudaDataType_t AType, int const lda, const int64_t strideA,
    void const* B, cudaDataType_t BType, int const ldb, const int64_t strideB, float const f_beta, void* C,
    cudaDataType_t CType, int const ldc, const int64_t strideC, int const batchCount, cudaDataType_t computeType)
{
    half h_alpha = (half) f_alpha;
    half h_beta = (half) f_beta;

    bool isFp16ComputeType = mComputeType == CUBLAS_COMPUTE_16F ? 1 : 0;
    void const* alpha = isFp16ComputeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void const*>(&f_alpha);
    void const* beta = isFp16ComputeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void const*>(&f_beta);

    check_cuda_error(cublasGemmStridedBatchedEx(getCublasHandle(), transa, transb, m, n, k, alpha, A, AType, lda,
        strideA, B, BType, ldb, strideB, beta, C, CType, ldc, strideC, batchCount, computeType,
        mAType == CUDA_R_32F ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void CublasMMWrapper::setWorkspace(void* workspace)
{
    mCublasWorkspace = workspace;
}

void CublasMMWrapper::setFP32GemmConfig()
{
    setGemmConfig(CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F);
}

void CublasMMWrapper::setFP16GemmConfig(cudaDataType_t outputType)
{
    setGemmConfig(CUDA_R_16F, CUDA_R_16F, outputType, CUDA_R_32F);
}

#ifdef ENABLE_BF16
void CublasMMWrapper::setBF16GemmConfig(cudaDataType_t outputType)
{
    setGemmConfig(CUDA_R_16BF, CUDA_R_16BF, outputType, CUDA_R_32F);
}
#endif

#ifdef ENABLE_FP8
void CublasMMWrapper::setFP8GemmConfig(cudaDataType_t outputType)
{
    setGemmConfig(CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, outputType, CUDA_R_32F);
}
#endif

#ifdef ENABLE_CUBLASLT_FP4_GEMM
void CublasMMWrapper::setFP4GemmConfig(cudaDataType_t outputType)
{
    setGemmConfig(CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, outputType, CUDA_R_32F);
}
#endif

void CublasMMWrapper::setGemmConfig(
    cudaDataType_t aType, cudaDataType_t bType, cudaDataType_t cType, cudaDataType_t computeType)
{
    mAType = aType;
    mBType = bType;
    mCType = cType;
    bool isFp16ComputeType = computeType == CUDA_R_16F;
    if (mAType == CUDA_R_4F_E2M1)
    {
        // for cublaslt nvfp4 gemm, fp32 compute type and fp32 scale type are required
        mComputeType = CUBLAS_COMPUTE_32F;
        mScaleType = CUDA_R_32F;
    }
    else if (isFp16ComputeType)
    {
        mComputeType = CUBLAS_COMPUTE_16F;
        mScaleType = CUDA_R_16F;
    }
    else
    {
        mComputeType = CUBLAS_COMPUTE_32F;
        mScaleType = CUDA_R_32F;
    }
}

CublasDataType CublasMMWrapper::getCublasDataType(cudaDataType_t data_type)
{
    if (data_type == CUDA_R_16F)
    {
        return HALF_DATATYPE;
    }
    else if (data_type == CUDA_R_32F)
    {
        return FLOAT_DATATYPE;
    }
    else if (data_type == CUDA_R_8I)
    {
        return INT8_DATATYPE;
    }
#ifdef ENABLE_BF16
    else if (data_type == CUDA_R_16BF)
    {
        return BFLOAT16_DATATYPE;
    }
#endif
    return FLOAT_DATATYPE;
}

void CublasMMWrapper::setStream(cudaStream_t stream)
{
    mStream = stream;
}

namespace
{

static inline char const* mmaToString(uint16_t mma)
{
    static char const* mmaStr[] = {
        "UNDEF", //
        "MMA884",
        "MMA1684",
        "MMA1688",
        "MMA16816",
    };

    static_assert(sizeof(mmaStr) / sizeof(mmaStr[0]) == CUBLASLT_MATMUL_INNER_SHAPE_END,
        "all mma configs must be listed in the metadata table");

    if (mma >= sizeof(mmaStr) / sizeof(mmaStr[0]))
        return "UNKNOWN";
    return mmaStr[mma];
}

static inline char const* cgaToString(uint16_t cga)
{
    // clang-format off
  static const char* cgaStr[] = {"AUTO",
                                 "ILLEGAL",
                                 "1x1x1",
                                 "1x2x1",
                                 "1x4x1",
                                 "2x1x1",
                                 "2x2x1",
                                 "2x4x1",
                                 "4x1x1",
                                 "4x2x1",
                                 "4x4x1",
                                 "1x8x1",
                                 "8x1x1",
                                 "2x8x1",
                                 "8x2x1",
                                 "1x16x1",
                                 "16x1x1",
                                 "1x3x1",
                                 "1x5x1",
                                 "1x6x1",
                                 "1x7x1",
                                 "1x9x1",
                                 "1x10x1",
                                 "1x11x1",
                                 "1x12x1",
                                 "1x13x1",
                                 "1x14x1",
                                 "1x15x1",
                                 "2x3x1",
                                 "2x5x1",
                                 "2x6x1",
                                 "2x7x1",
                                 "3x1x1",
                                 "3x2x1",
                                 "3x3x1",
                                 "3x4x1",
                                 "3x5x1",
                                 "4x3x1",
                                 "5x1x1",
                                 "5x2x1",
                                 "5x3x1",
                                 "6x1x1",
                                 "6x2x1",
                                 "7x1x1",
                                 "7x2x1",
                                 "9x1x1",
                                 "10x1x1",
                                 "11x1x1",
                                 "12x1x1",
                                 "13x1x1",
                                 "14x1x1",
                                 "15x1x1",
                                 };
    // clang-format on

    static_assert(sizeof(cgaStr) / sizeof(cgaStr[0]) == CUBLASLT_CLUSTER_SHAPE_END,
        "all cga configs must be listed in the metadata table");

    if (cga >= sizeof(cgaStr) / sizeof(cgaStr[0]))
        return "UNKNOWN";
    return cgaStr[cga];
}

static void print_algo(cublasLtMatmulAlgo_t const* matmulAlgo)
{
    int algoId, tile, stages, swizzle, customOption, numSplitsK, reductionScheme;
    uint16_t mma, cga;

    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);

    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &mma, sizeof(mma), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cga, sizeof(cga), NULL);

    TLLM_LOG_DEBUG(
        "algo={ %d %d %d splitK=%d reduc=%d swizzle=%d custom=%d mma=%s cga=%s}"
        " [-algo%d -m_tile%d -m_stages%d -m_numsK%d -m_reduction%d -m_swizzle%d -m_custom%d -m_mma%d -m_cga%d "
        "\n",
        algoId, tile, stages, numSplitsK, reductionScheme, swizzle, customOption, mmaToString(mma), cgaToString(cga),
        algoId, tile, stages, numSplitsK, reductionScheme, swizzle, customOption, mma, cga);
}

} // namespace

bool CublasMMWrapper::checkTactic(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n,
    int const k, int const lda, int const ldb, int const ldc, cublasLtMatmulAlgo_t const& algo)
{
    TLLM_CHECK_WITH_INFO(
        descriptorsCreated(), "Descriptors are not created! Call createDescriptors before calling this function");

    cublasLtMatmulHeuristicResult_t heurResult;
    cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(
        getCublasLtHandle(), mOperationDesc, mADesc, mBDesc, mCDesc, mCDesc, &algo, &heurResult);

    if (algoStatus != CUBLAS_STATUS_SUCCESS || heurResult.state != CUBLAS_STATUS_SUCCESS
        || heurResult.workspaceSize > CUBLAS_WORKSPACE_SIZE)
    {
        TLLM_LOG_WARNING("CheckTactic failed with status: %d and heuristic status: %d with workspace size: %d.\n",
            algoStatus, heurResult.state, heurResult.workspaceSize);
        return false;
    }

    sync_check_cuda_error(mStream);

    return true;
}

std::vector<cublasLtMatmulHeuristicResult_t> CublasMMWrapper::getTactics(cublasOperation_t transa,
    cublasOperation_t transb, int const m, int const n, int const k, int const lda, int const ldb, int const ldc)
{
    TLLM_CHECK_WITH_INFO(
        descriptorsCreated(), "Descriptors are not created! Call createDescriptors before calling this function");

    auto const heuristics = getTactics(getCublasLtHandle(), mOperationDesc, mADesc, mBDesc, mCDesc, mCDesc);

    sync_check_cuda_error(mStream);

    return heuristics;
}

std::vector<cublasLtMatmulHeuristicResult_t> CublasMMWrapper::getTactics(cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc)
{
#if TLLM_CUBLAS_VER_LE(11, 4, 2)
    TLLM_CHECK_WITH_INFO(false, "CUBLAS version too low, must be > 11.4.2.");
    return {};
#else
    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(200);
    cublasLtMatmulPreference_t preference;
    check_cuda_error(cublasLtMatmulPreferenceCreate(&preference));
    check_cuda_error(cublasLtMatmulPreferenceInit(preference));
    uint64_t workspace_size = CUBLAS_WORKSPACE_SIZE;
    check_cuda_error(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
    // Restrict reduction algorithms for numerical stability and better determinism
    uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_MASK;
    check_cuda_error(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_mask, sizeof(reduction_mask)));
#if TLLM_CUBLAS_VER_LT(12, 0, 0)
    uint32_t pointer_mode_mask = 0;
    check_cuda_error(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK, &pointer_mode_mask, sizeof(pointer_mode_mask)));
#endif

    int return_count = 0;
    check_cuda_error(cublasLtMatmulAlgoGetHeuristic(lightHandle, computeDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
        heuristics.size(), heuristics.data(), &return_count));
    heuristics.resize(return_count);

    return heuristics;
#endif
}

#ifdef ENABLE_CUBLASLT_FP4_GEMM

namespace
{
// Helper function: Get or create a zero beta tensor on GPU for the given device
// Beta is always 0 for FP4 GEMM and is allocated once per device per thread
float const* getBetaDevicePointer()
{
    thread_local static std::unordered_map<int, float*> beta_per_device;

    int current_device;
    cudaGetDevice(&current_device);

    auto it = beta_per_device.find(current_device);
    if (it == beta_per_device.end())
    {
        // Allocate GPU memory for beta and initialize to 0
        float* d_beta;
        cudaMalloc(&d_beta, sizeof(float));
        cudaMemset(d_beta, 0, sizeof(float));
        beta_per_device[current_device] = d_beta;
        return d_beta;
    }

    return it->second;
}
} // namespace

// BlockScaleGemm Version 1: Default algorithm (uses first valid heuristic)
void CublasMMWrapper::BlockScaleGemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n,
    int const k, void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc, void const* a_sf,
    void const* b_sf, float const* alpha)
{
    // Forward to the overloaded version with nullptr (use default algorithm)
    BlockScaleGemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, a_sf, b_sf, alpha, nullptr);
}

// BlockScaleGemm Version 2: Specified algorithm (unified implementation)
void CublasMMWrapper::BlockScaleGemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n,
    int const k, void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc, void const* a_sf,
    void const* b_sf, float const* alpha, cublasLtMatmulAlgo_t const* algo)
{
    // Verify input data types (currently supports FP4, can be extended to more formats in the future)
    TLLM_CHECK_WITH_INFO(mAType == CUDA_R_4F_E2M1 && mBType == CUDA_R_4F_E2M1,
        "BlockScaleGemm currently requires FP4 input types. "
        "Future versions may support other quantized formats with block-wise scaling.");

    // Validate input pointers
    TLLM_CHECK_WITH_INFO(A != nullptr, "A pointer is null");
    TLLM_CHECK_WITH_INFO(B != nullptr, "B pointer is null");
    TLLM_CHECK_WITH_INFO(C != nullptr, "C pointer is null");
    TLLM_CHECK_WITH_INFO(a_sf != nullptr, "a_sf (A scale factor) pointer is null");
    TLLM_CHECK_WITH_INFO(b_sf != nullptr, "b_sf (B scale factor) pointer is null");
    TLLM_CHECK_WITH_INFO(alpha != nullptr, "alpha pointer is null");

    // Beta is always 0 for FP4 GEMM, get per-device GPU pointer
    float const* beta = getBetaDevicePointer();

    // Create descriptors for block-scaled GEMM
    createDescriptors(transa, transb, m, n, k, lda, ldb, ldc, 0);

    // Create D descriptor for output matrix
    cublasLtMatrixLayout_t Ddesc = NULL;
    check_cuda_error(cublasLtMatrixLayoutCreate(&Ddesc, mCType, m, n, ldc));

    // Set block-wise scaling descriptors
    setScaleDescriptors(const_cast<void*>(a_sf), const_cast<void*>(b_sf));

    // Validate cuBLASLt handle
    TLLM_CHECK_WITH_INFO(mCublasLtHandle != nullptr, "cuBLASLt handle is null");

    // Determine which algorithm to use
    cublasLtMatmulAlgo_t const* selected_algo = algo;
    cublasLtMatmulAlgo_t default_algo;

    if (algo == nullptr)
    {
        // No algorithm specified, use heuristic (default behavior)
        auto heuristics = getTactics(getCublasLtHandle(), mOperationDesc, mADesc, mBDesc, mCDesc, Ddesc);

        if (heuristics.empty())
        {
            if (Ddesc)
                cublasLtMatrixLayoutDestroy(Ddesc);
            destroyDescriptors();
            throw std::runtime_error("No suitable cuBLASLt algorithm found for block-scaled GEMM");
        }

        // Use the first valid heuristic
        auto const& heuristic = heuristics[0];
        bool hasAlgo = heuristic.state == CUBLAS_STATUS_SUCCESS && heuristic.workspaceSize <= CUBLAS_WORKSPACE_SIZE;

        if (hasAlgo)
        {
            default_algo = heuristic.algo;
            selected_algo = &default_algo;
        }
        else
        {
            selected_algo = nullptr; // No valid algorithm, let cuBLASLt choose
        }
    }

    int workspaceSize = mCublasWorkspace == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

    // Call cuBLASLt matmul with selected or default algorithm
    check_cuda_error(cublasLtMatmul(getCublasLtHandle(), mOperationDesc, alpha, A, mADesc, B, mBDesc, beta, C, mCDesc,
        C, Ddesc, selected_algo, // nullptr or specific algorithm
        mCublasWorkspace, workspaceSize, mStream));

    // Synchronize stream
    sync_check_cuda_error(mStream);

    // Clean up descriptors
    if (Ddesc)
        cublasLtMatrixLayoutDestroy(Ddesc);
    destroyDescriptors();
}

#endif

} // namespace common

} // namespace tensorrt_llm
