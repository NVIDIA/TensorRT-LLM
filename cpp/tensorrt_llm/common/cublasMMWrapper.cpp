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

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace tensorrt_llm
{
namespace common
{

cublasMMWrapper::cublasMMWrapper(std::shared_ptr<cublasHandle_t> cublasHandle,
    std::shared_ptr<cublasLtHandle_t> cublasltHandle, cudaStream_t stream, cublasAlgoMap* cublas_algo_map,
    std::mutex* mu, void* workspace)
    : cublas_handle_(cublasHandle)
    , cublaslt_handle_(cublasltHandle)
    , stream_(stream)
    , cublas_algo_map_(cublas_algo_map)
    , mu_(mu)
    , cublas_workspace_(workspace)
{
}

cublasMMWrapper::~cublasMMWrapper()
{
    mu_ = nullptr;
}

cublasMMWrapper::cublasMMWrapper(const cublasMMWrapper& wrapper)
    : cublas_handle_(wrapper.cublas_handle_)
    , cublaslt_handle_(wrapper.cublaslt_handle_)
    , stream_(wrapper.stream_)
    , cublas_algo_map_(wrapper.cublas_algo_map_)
    , mu_(wrapper.mu_)
{
}

void cublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k,
    const void* alpha, const void* A, cudaDataType_t Atype, int lda, const void* B, cudaDataType_t Btype, int ldb,
    const void* beta, void* C, cudaDataType_t Ctype, int ldc, cudaDataType_t computeType, cublasGemmAlgo_t algo)
{
    mu_->lock();
    check_cuda_error(cublasGemmEx(*cublas_handle_, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta,
        C, Ctype, ldc, computeType, algo));
    sync_check_cuda_error();
    mu_->unlock();
}

void cublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k,
    const void* A, const int lda, const void* B, const int ldb, void* C, const int ldc)
{
    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f);
}

void cublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k,
    const void* A, const int lda, const void* B, const int ldb, void* C, const int ldc,
    const std::optional<cublasLtMatmulHeuristicResult_t>& heuristic)
{
    if (heuristic)
    {
        Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f, (*heuristic).algo,
            (*heuristic).state == CUBLAS_STATUS_SUCCESS && (*heuristic).workspaceSize < CUBLAS_WORKSPACE_SIZE);
    }
    else
    {
        Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f, {}, false);
    }
}

void cublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k,
    const void* A, const int lda, const void* B, const int ldb, void* C, const int ldc, float f_alpha, float f_beta)
{
    bool usingCublasLt = Atype_ == CUDA_R_16F;
    bool isFp16ComputeType = computeType_ == CUDA_R_16F;

    int batch_count = 1;
    int findAlgo = cublas_algo_map_->isExist(batch_count, m, n, k, getCublasDataType(Atype_));

    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));

    cublasLtMatmulAlgo_t algo;
    void* workSpace = cublas_workspace_;
    int workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
    if (findAlgo)
    {
        if (info.stages != -1)
        {
            usingCublasLt = true;
        }
        else
        {
            usingCublasLt = false;
        }
    }

    if (usingCublasLt)
    {
        cublasLtMatmulDesc_t operationDesc = NULL;
        cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
        cudaDataType_t scaleType;
#if (CUDART_VERSION >= 11000)
        cublasComputeType_t computeType;
#else
        cudaDataType_t computeType;
#endif

        if (isFp16ComputeType)
        {
#if (CUDART_VERSION >= 11000)
            computeType = CUBLAS_COMPUTE_16F;
#else
            computeType = CUDA_R_16F;
#endif
            scaleType = CUDA_R_16F;
        }
        else
        {
#if (CUDART_VERSION >= 11000)
            computeType = CUBLAS_COMPUTE_32F;
#else
            computeType = CUDA_R_32F;
#endif
            scaleType = CUDA_R_32F;
        }

        if (findAlgo)
        {
            if (info.workspaceSize > workspaceSize)
            {
                findAlgo = 0;
            }
            else
            {
                cublasLtMatmulAlgoInit(
                    *cublaslt_handle_, computeType, scaleType, Atype_, Btype_, Ctype_, Ctype_, info.algoId, &algo);
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption), sizeof(info.customOption));
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(info.tile), sizeof(info.tile));
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val), sizeof(info.splitK_val));
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle), sizeof(info.swizzle));
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(info.reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages), sizeof(info.stages));
#endif
            }
        }
    }

    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, f_alpha, f_beta, algo, findAlgo);
}

void cublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k,
    const void* A, const int lda, const void* B, const int ldb, void* C, const int ldc, float f_alpha, float f_beta,
    const cublasLtMatmulAlgo_t& algo, bool hasAlgo)
{
    half h_alpha = (half) (f_alpha);
    half h_beta = (half) (f_beta);

    std::lock_guard<std::mutex> lock(*mu_);

    // TODO: default cublas libs
    bool usingCublasLt = Atype_ == CUDA_R_16F;
    bool isFp16ComputeType = computeType_ == CUDA_R_16F;
    int batch_count = 1;
    // fp32 use cublas as default
    // fp16 use cublasLt as default
    const void* alpha = isFp16ComputeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta = isFp16ComputeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);
    if (hasAlgo)
    {
        int32_t stages;
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);
        if (stages != -1)
        {
            usingCublasLt = true;
        }
        else
        {
            usingCublasLt = false;
        }
    }

    if (usingCublasLt)
    {
        cublasLtMatmulDesc_t operationDesc = NULL;
        cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
        cudaDataType_t scaleType;
#if (CUDART_VERSION >= 11000)
        cublasComputeType_t computeType;
#else
        cudaDataType_t computeType;
#endif

        if (isFp16ComputeType)
        {
#if (CUDART_VERSION >= 11000)
            computeType = CUBLAS_COMPUTE_16F;
#else
            computeType = CUDA_R_16F;
#endif
            scaleType = CUDA_R_16F;
        }
        else
        {
#if (CUDART_VERSION >= 11000)
            computeType = CUBLAS_COMPUTE_32F;
#else
            computeType = CUDA_R_32F;
#endif
            scaleType = CUDA_R_32F;
        }
        // --------------------------------------
        // Create descriptors for the original matrices
        cublasLtMatrixLayoutCreate(&Adesc, Atype_, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
        cublasLtMatrixLayoutCreate(&Bdesc, Btype_, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
        cublasLtMatrixLayoutCreate(&Cdesc, Ctype_, m, n, ldc);
#if (CUDART_VERSION >= 11000)
        cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
        cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));

        void* workSpace = cublas_workspace_;
        int workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
        if (hasAlgo)
        {
            cublasLtMatmulHeuristicResult_t heurResult;
            // We have to check if the heruistic is correct given current shape size
            cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(
                getCublasLtHandle(), operationDesc, Adesc, Bdesc, Cdesc, Cdesc, &algo, &heurResult);

            if (algoStatus != CUBLAS_STATUS_SUCCESS || heurResult.state != CUBLAS_STATUS_SUCCESS
                || heurResult.workspaceSize > CUBLAS_WORKSPACE_SIZE)
            {
                // Rely on runtime based heruistic
                hasAlgo = false;
            }
        }

        check_cuda_error(cublasLtMatmul(*cublaslt_handle_, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C,
            Cdesc, (hasAlgo ? (&algo) : NULL), workSpace, workspaceSize, stream_));

        cublasLtMatmulDescDestroy(operationDesc);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
        sync_check_cuda_error();
    }
    else
    {
        // Go with default heruistic to choose tactic as cuBLAS does not allow to choose tactics in Ampere+
        cublasGemmAlgo_t cublasAlgo = CUBLAS_GEMM_DEFAULT;
        check_cuda_error(cublasGemmEx(*cublas_handle_, transa, transb, m, n, k, alpha, A, Atype_, lda, B, Btype_, ldb,
            beta, C, Ctype_, ldc, computeType_, static_cast<cublasGemmAlgo_t>(cublasAlgo)));
        sync_check_cuda_error();
    }
}

void cublasMMWrapper::setWorkspace(void* workspace)
{
    cublas_workspace_ = workspace;
}

void cublasMMWrapper::setFP32GemmConfig()
{
    setGemmConfig(CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F);
}

void cublasMMWrapper::setFP16GemmConfig()
{
    setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
}

#ifdef ENABLE_BF16
void cublasMMWrapper::setBF16GemmConfig()
{
    setGemmConfig(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F);
}
#endif

#ifdef ENABLE_FP8
void cublasMMWrapper::setFP8GemmConfig(cudaDataType_t outputType)
{
    setGemmConfig(CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, outputType, CUDA_R_32F);
}
#endif

void cublasMMWrapper::setGemmConfig(
    cudaDataType_t aType, cudaDataType_t bType, cudaDataType_t cType, cudaDataType_t computeType)
{
    Atype_ = aType;
    Btype_ = bType;
    Ctype_ = cType;
    computeType_ = computeType;
}

CublasDataType cublasMMWrapper::getCublasDataType(cudaDataType_t data_type)
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

#if (CUDART_VERSION >= 11000)
// input, weight, output are row-major
// only works for cublas 11.x
void cublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k,
    const void* A, const int lda, const void* B, const int ldb, const void* bias, void* C, const int ldc)
{
    cudaDataType_t Atype, Btype, Ctype;
    cublasComputeType_t computeType;
    cudaDataType_t scaleType;
    float alpha_float = 1.0f;
    float beta_float = 0.0f;
    half alpha_half = half(1.0f);
    half beta_half = half(0.0f);
    void *alpha, *beta;

    // int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    if (Atype_ == CUDA_R_32F)
    {
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
        Atype = CUDA_R_32F;
        Btype = CUDA_R_32F;
        Ctype = CUDA_R_32F;
        scaleType = CUDA_R_32F;
        alpha = &alpha_float;
        beta = &beta_float;
    }
    else if (Atype_ == CUDA_R_16BF)
    {
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
        Atype = CUDA_R_16BF;
        Btype = CUDA_R_16BF;
        Ctype = CUDA_R_16BF;
        scaleType = CUDA_R_32F;
        alpha = &alpha_float;
        beta = &beta_float;
    }
    else
    {
        computeType = CUBLAS_COMPUTE_16F;
        Atype = CUDA_R_16F;
        Btype = CUDA_R_16F;
        Ctype = CUDA_R_16F;
        scaleType = CUDA_R_16F;
        alpha = &alpha_half;
        beta = &beta_half;
    }

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatrixLayoutCreate(&Adesc, Atype, (transa == CUBLAS_OP_N) ? m : k, (transa == CUBLAS_OP_N) ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, Btype, (transb == CUBLAS_OP_N) ? k : n, (transb == CUBLAS_OP_N) ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc);

    cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(cublasLtEpilogue_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(const void*));
    check_cuda_error(cublasLtMatmul(
        *cublaslt_handle_, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, NULL, 0, stream_));
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
}
#endif
void cublasMMWrapper::setStream(cudaStream_t stream)
{
    stream_ = stream;
}

void cublasMMWrapper::stridedBatchedGemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n,
    const int k, const void* A, const int lda, const int64_t strideA, const void* B, const int ldb,
    const int64_t strideB, void* C, const int ldc, const int64_t strideC, const int batch_count, const float f_alpha,
    const float f_beta)
{
    half h_alpha = (half) f_alpha;
    half h_beta = (half) f_beta;

    mu_->lock();
    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    const void* alpha
        = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<const void*>(&f_beta);
    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));

    check_cuda_error(cublasGemmStridedBatchedEx(*cublas_handle_, transa, transb, m, n, k, alpha, A, Atype_, lda,
        strideA, B, Btype_, ldb, strideB, beta, C, Ctype_, ldc, strideC, batch_count, computeType_,
        static_cast<cublasGemmAlgo_t>(info.algoId)));

    mu_->unlock();
}

void cublasMMWrapper::stridedBatchedGemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n,
    const int k, const float f_alpha, const void* A, cudaDataType_t AType, const int lda, const int64_t strideA,
    const void* B, cudaDataType_t BType, const int ldb, const int64_t strideB, const float f_beta, void* C,
    cudaDataType_t CType, const int ldc, const int64_t strideC, const int batch_count, cudaDataType_t computeType)
{
    half h_alpha = (half) f_alpha;
    half h_beta = (half) f_beta;

    mu_->lock();
    int is_fp16_computeType = computeType == CUDA_R_16F ? 1 : 0;
    const void* alpha
        = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<const void*>(&f_beta);
    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));

    check_cuda_error(cublasGemmStridedBatchedEx(*cublas_handle_, transa, transb, m, n, k, alpha, A, AType, lda, strideA,
        B, BType, ldb, strideB, beta, C, CType, ldc, strideC, batch_count, computeType,
        static_cast<cublasGemmAlgo_t>(info.algoId)));

    mu_->unlock();
}

void cublasMMWrapper::batchedGemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n,
    const int k, const void* const* A, const int lda, const void* const* B, const int ldb, void* const* C,
    const int ldc, const int batch_count)
{
    float f_alpha = static_cast<float>(1.0f);
    float f_beta = static_cast<float>(0.0f);

    half h_alpha = (half) 1.0f;
    half h_beta = (half) 0.0f;

    mu_->lock();
    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);
    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));

    check_cuda_error(cublasGemmBatchedEx(*cublas_handle_, transa, transb, m, n, k, alpha, A, Atype_, lda, B, Btype_,
        ldb, beta, C, Ctype_, ldc, batch_count, computeType_, static_cast<cublasGemmAlgo_t>(info.algoId)));
    mu_->unlock();
}

bool cublasMMWrapper::isFuseBatchGemm(const int batch_count, const int m, const int k, const int n)
{
    CublasDataType data_type = getCublasDataType(Atype_);

    if (cublas_algo_map_->isExist(batch_count, m, k, n, data_type) == false
        || cublas_algo_map_->isExist(1, m, k, n, data_type) == false)
    {
        return false;
    }
    else
    {
        return cublas_algo_map_->getAlgo(batch_count, m, k, n, data_type).exec_time
            < 3 * cublas_algo_map_->getAlgo(1, m, k, n, data_type).exec_time;
    }
}

std::vector<cublasLtMatmulHeuristicResult_t> cublasMMWrapper::getTactics(cublasOperation_t transa,
    cublasOperation_t transb, const int m, const int n, const int k, const int lda, const int ldb, const int ldc)
{
    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cudaDataType_t scaleType;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if (is_fp16_computeType)
    {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
#else
        computeType = CUDA_R_16F;
#endif
        scaleType = CUDA_R_16F;
    }
    else
    {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
#else
        computeType = CUDA_R_32F;
#endif
        scaleType = CUDA_R_32F;
    }

    // --------------------------------------
    // Create descriptors for the original matrices
    check_cuda_error(
        cublasLtMatrixLayoutCreate(&Adesc, Atype_, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    check_cuda_error(
        cublasLtMatrixLayoutCreate(&Bdesc, Btype_, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    check_cuda_error(cublasLtMatrixLayoutCreate(&Cdesc, Ctype_, m, n, ldc));
#if (CUDART_VERSION >= 11000)
    check_cuda_error(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
#else
    check_cuda_error(cublasLtMatmulDescCreate(&operationDesc, computeType));
#endif

    check_cuda_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));

    void* workSpace = cublas_workspace_;
    int workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

    const auto heuristics = getTactics(getCublasLtHandle(), operationDesc, Adesc, Bdesc, Cdesc, Cdesc);

    check_cuda_error(cublasLtMatmulDescDestroy(operationDesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(Adesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(Bdesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(Cdesc));
    sync_check_cuda_error();

    return heuristics;
}

bool cublasMMWrapper::checkTactic(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n,
    const int k, const int lda, const int ldb, const int ldc, const cublasLtMatmulHeuristicResult_t& heuristic) const
{
    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cudaDataType_t scaleType;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if (is_fp16_computeType)
    {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
#else
        computeType = CUDA_R_16F;
#endif
        scaleType = CUDA_R_16F;
    }
    else
    {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
#else
        computeType = CUDA_R_32F;
#endif
        scaleType = CUDA_R_32F;
    }

    // --------------------------------------
    // Create descriptors for the original matrices
    check_cuda_error(
        cublasLtMatrixLayoutCreate(&Adesc, Atype_, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    check_cuda_error(
        cublasLtMatrixLayoutCreate(&Bdesc, Btype_, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    check_cuda_error(cublasLtMatrixLayoutCreate(&Cdesc, Ctype_, m, n, ldc));
#if (CUDART_VERSION >= 11000)
    check_cuda_error(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
#else
    check_cuda_error(cublasLtMatmulDescCreate(&operationDesc, computeType));
#endif

    check_cuda_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));

    void* workSpace = cublas_workspace_;
    int workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

    cublasLtMatmulHeuristicResult_t heurResult;
    cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(
        getCublasLtHandle(), operationDesc, Adesc, Bdesc, Cdesc, Cdesc, &heuristic.algo, &heurResult);

    if (algoStatus != CUBLAS_STATUS_SUCCESS || heurResult.state != CUBLAS_STATUS_SUCCESS
        || heurResult.workspaceSize > CUBLAS_WORKSPACE_SIZE)
    {
        return false;
    }

    check_cuda_error(cublasLtMatmulDescDestroy(operationDesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(Adesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(Bdesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(Cdesc));
    sync_check_cuda_error();

    return true;
}

std::vector<cublasLtMatmulHeuristicResult_t> cublasMMWrapper::getTactics(cublasLtHandle_t lightHandle,
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
    // Restrict reduction algorithms for numerical stability and better determenism
    uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_INPLACE;
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

std::pair<bool, cublasLtMatmulAlgo_t> cublasMMWrapper::findBestAlgo(cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc, const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* B,
    cublasLtMatrixLayout_t Bdesc, const void* beta, const void* C, cublasLtMatrixLayout_t Cdesc, void* D,
    cublasLtMatrixLayout_t Ddesc, cudaStream_t stream)
{
#if TLLM_CUBLAS_VER_LE(11, 4, 2)
    TLLM_CHECK_WITH_INFO(false, "CUBLAS version too low, must be > 11.4.2.");
    return {false, cublasLtMatmulAlgo_t{}};
#else
    size_t returnSize;
    int32_t pointer_mode;
    cublasLtMatmulDescGetAttribute(
        computeDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode), &returnSize);

    const auto heuristics = getTactics(lightHandle, computeDesc, Adesc, Bdesc, Cdesc, Ddesc);

    std::map<int, std::vector<float>> algo_results;
    for (const auto& heuristic : heuristics)
    {
        cublasLtMatmulAlgo_t algo = heuristic.algo;
        int32_t algo_id;
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize);

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        float my_alpha = 1.0f;
        float my_beta = 0.0f;

        for (int i = 0; i < 11; i++)
        {
            float duration_ms;
            cudaEventRecord(start_event, stream);
            check_cuda_error(cublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D,
                Ddesc, &algo, cublas_workspace_, CUBLAS_WORKSPACE_SIZE, stream));
            cudaEventRecord(stop_event, stream);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&duration_ms, start_event, stop_event);

            algo_results[algo_id].push_back(duration_ms);
        }
        std::sort(algo_results[algo_id].begin(), algo_results[algo_id].end());
    }

    cublasLtMatmulHeuristicResult_t result;
    float best_time = INFINITY;
    for (const auto& heuristic : heuristics)
    {
        cublasLtMatmulAlgo_t algo = heuristic.algo;
        int32_t algo_id;
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize);
        const auto& results = algo_results[algo_id];

        if (results.size() > 0 && results[5] < best_time)
        {
            best_time = results[5];
            result = heuristic;
        }
    }

    return {best_time != INFINITY, result.algo};
#endif
}

cublasMMWrapper::MatrixLayout cublasMMWrapper::createMatrixLayout(cublasLtMatrixLayout_t Mdesc)
{
    size_t returnSize;
    MatrixLayout m_layout;

    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &std::get<0>(m_layout), sizeof(std::get<0>(m_layout)), &returnSize);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &std::get<1>(m_layout), sizeof(std::get<1>(m_layout)), &returnSize);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &std::get<2>(m_layout), sizeof(std::get<2>(m_layout)), &returnSize);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &std::get<3>(m_layout), sizeof(std::get<3>(m_layout)), &returnSize);

    return m_layout;
}

cublasStatus_t cublasMMWrapper::cublasLtMatmulWrapper(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
    const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* B, cublasLtMatrixLayout_t Bdesc,
    const void* beta, const void* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream)
{
    cache_idx_t cache_idx{computeDesc,
        {createMatrixLayout(Adesc), createMatrixLayout(Bdesc), createMatrixLayout(Cdesc), createMatrixLayout(Ddesc)}};

    cublasLtMatmulAlgo_t algo_value;
    bool found_algo = false;
    if (algo == nullptr)
    {
        if (algo_cache.find(cache_idx) == algo_cache.end())
        {
            auto result
                = findBestAlgo(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, stream);
            if (result.first)
            {
                algo_cache[cache_idx] = result.second;
                algo_value = result.second;
                found_algo = true;
            }
        }
        else
        {
            algo_value = algo_cache[cache_idx];
            found_algo = true;
        }
    }

    return cublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc,
        found_algo ? &algo_value : algo, workspace, workspaceSizeInBytes, stream);
}

void cublasMMWrapper::_Int8Gemm(const int m, const int n, const int k, const int8_t* A, const int lda, const int8_t* B,
    const int ldb, void* C, const int ldc, const void* alpha, const int mode, const bool per_column_scaling)
{
    /* mode:
     *  - 0: int8 * int8 -> int32 -> int8
     *  - 1: int8 * int8 -> int32 -> int32
     */
#if TLLM_CUBLAS_VER_LE(11, 4, 2)
    TLLM_CHECK_WITH_INFO(false, "CUBLAS version too low, must be > 11.4.2.");
#else

    mu_->lock();
    const auto op_a = CUBLAS_OP_T;
    const auto op_b = CUBLAS_OP_N;
    const auto dataType = CUDA_R_8I;
    const auto resultType = mode == 0 ? CUDA_R_8I : CUDA_R_32I;
    const auto computeType = CUBLAS_COMPUTE_32I;
    const auto scaleType = mode == 0 ? CUDA_R_32F : CUDA_R_32I;
    const int batch_count = 1;
    const void* beta;

    int findAlgo = cublas_algo_map_->isExist(batch_count, m, n, k, getCublasDataType(dataType));

    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(dataType));

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

    // --------------------------------------
    // Create descriptors for the original matrices
    check_cuda_error(cublasLtMatrixLayoutCreate(&Adesc, dataType, k, m, lda));
    check_cuda_error(cublasLtMatrixLayoutCreate(&Bdesc, dataType, k, n, ldb));
    check_cuda_error(cublasLtMatrixLayoutCreate(&Cdesc, resultType, m, n, ldc));

    check_cuda_error(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));

    auto pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    if (mode == 0)
    {
        pointer_mode
            = per_column_scaling ? CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST : CUBLASLT_POINTER_MODE_DEVICE;
    }
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(cublasOperation_t)));
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(cublasOperation_t)));
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSC, &op_b, sizeof(cublasOperation_t)));
    check_cuda_error(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

    const int32_t int_one = 1;
    const int32_t int_zero = 0;
    const float float_zero = 0;
    if (mode == 0)
    {
        beta = per_column_scaling ? &float_zero : NULL;
    }
    else
    {
        alpha = &int_one;
        beta = &int_zero;
    }

    void* workSpace = cublas_workspace_;
    int workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

    sync_check_cuda_error();
    auto ret = cublasLtMatmulWrapper(*cublaslt_handle_, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C,
        Cdesc, NULL, workSpace, workspaceSize, stream_);
    check_cuda_error(ret);
    sync_check_cuda_error();

    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    sync_check_cuda_error();
    mu_->unlock();
#endif
}

void cublasMMWrapper::Int8Gemm(const int m, const int n, const int k, const int8_t* A, const int lda, const int8_t* B,
    const int ldb, int8_t* C, const int ldc, const float* alpha, const bool per_column_scaling)
{
    return _Int8Gemm(m, n, k, A, lda, B, ldb, C, ldc, alpha, 0, per_column_scaling);
}

void cublasMMWrapper::Int8Gemm(const int m, const int n, const int k, const int8_t* A, const int lda, const int8_t* B,
    const int ldb, int32_t* C, const int ldc)
{
    return _Int8Gemm(m, n, k, A, lda, B, ldb, C, ldc, (float*) nullptr, 1, false);
}
} // namespace common

} // namespace tensorrt_llm
