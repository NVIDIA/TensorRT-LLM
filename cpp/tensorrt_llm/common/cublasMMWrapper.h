/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <optional>
#include <string>

namespace tensorrt_llm
{
namespace common
{

class CublasMMWrapper
{
protected:
    std::shared_ptr<cublasHandle_t> mCublasHandle;
    std::shared_ptr<cublasLtHandle_t> mCublasLtHandle;

    cudaDataType_t mAType{};
    cudaDataType_t mBType{};
    cudaDataType_t mCType{};
    cublasComputeType_t mComputeType{};
    cudaDataType_t mScaleType{};

    cublasLtMatmulDesc_t mOperationDesc{NULL};
    cublasLtMatrixLayout_t mADesc{NULL};
    cublasLtMatrixLayout_t mBDesc{NULL};
    cublasLtMatrixLayout_t mCDesc{NULL};

    cudaStream_t mStream;
    //@fixme: we may not need the mutex if we copy the wrapper instead of sharing in GemmPlugin::clone()
    std::shared_ptr<std::mutex> mMutex{std::make_shared<std::mutex>()};

    void* mCublasWorkspace = nullptr;

private:
    bool descriptorsCreated() const
    {
        return mOperationDesc != NULL && mADesc != NULL && mBDesc != NULL && mCDesc != NULL;
    }

public:
    CublasMMWrapper(std::shared_ptr<cublasHandle_t> cublasHandle, std::shared_ptr<cublasLtHandle_t> cublasLtHandle,
        cudaStream_t stream, void* workspace);

    ~CublasMMWrapper();

    CublasMMWrapper(const CublasMMWrapper& wrapper);

    /********************** GEMMs **********************/
    void Gemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k, const void* A,
        const int lda, const void* B, const int ldb, void* C, const int ldc);

    void Gemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k, const void* A,
        const int lda, const void* B, const int ldb, void* C, const int ldc,
        const std::optional<cublasLtMatmulHeuristicResult_t>& algo);

    void Gemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k, const void* A,
        const int lda, const void* B, const int ldb, void* C, const int ldc, float f_alpha, float f_beta);

    void Gemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k, const void* A,
        const int lda, const void* B, const int ldb, void* C, const int ldc, float f_alpha, float f_beta,
        const cublasLtMatmulAlgo_t& algo, bool hasAlgo, bool usingCublasLt);

    void stridedBatchedGemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k,
        const void* A, const int lda, const int64_t strideA, const void* B, const int ldb, const int64_t strideB,
        void* C, const int ldc, const int64_t strideC, const int batchCount, const float f_alpha = 1.0f,
        const float f_beta = 0.0f);

    void stridedBatchedGemm(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k,
        const float f_alpha, const void* A, cudaDataType_t AType, const int lda, const int64_t strideA, const void* B,
        cudaDataType_t BType, const int ldb, const int64_t strideB, const float f_beta, void* C, cudaDataType_t CType,
        const int ldc, const int64_t strideC, const int batchCount, cudaDataType_t computeType);

    /********************** Tactic selection helpers **********************/
    bool checkTactic(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k,
        const int lda, const int ldb, const int ldc, const cublasLtMatmulAlgo_t& algo);

    std::vector<cublasLtMatmulHeuristicResult_t> getTactics(cublasOperation_t transa, cublasOperation_t transb,
        const int m, const int n, const int k, const int lda, const int ldb, const int ldc);

    std::vector<cublasLtMatmulHeuristicResult_t> getTactics(cublasLtHandle_t lightHandle,
        cublasLtMatmulDesc_t computeDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
        cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc);

    using MatrixLayout = std::tuple<cudaDataType_t, cublasLtOrder_t, uint64_t, uint64_t>;
    using cache_idx_t = std::tuple<cublasLtMatmulDesc_t, std::array<MatrixLayout, 4>>;

    MatrixLayout createMatrixLayout(cublasLtMatrixLayout_t Mdesc);

    /********************** Utils **********************/
    void setWorkspace(void* workspace);

    void setFP32GemmConfig();
    void setFP16GemmConfig(cudaDataType_t outputType = CUDA_R_16F);
#ifdef ENABLE_BF16
    void setBF16GemmConfig(cudaDataType_t outputType = CUDA_R_16BF);
#endif
#ifdef ENABLE_FP8
    void setFP8GemmConfig(cudaDataType_t outputType = CUDA_R_16F);
#endif

    void setStream(cudaStream_t stream);

    void setGemmConfig(cudaDataType_t aType, cudaDataType_t bType, cudaDataType_t cType, cudaDataType_t computeType);

    CublasDataType getCublasDataType(cudaDataType_t data_type);

    void createDescriptors(cublasOperation_t transa, cublasOperation_t transb, const int m, const int n, const int k,
        const int lda, const int ldb, const int ldc);
    void destroyDescriptors();

    cublasHandle_t getCublasHandle()
    {
        return *(this->mCublasHandle);
    }

    cublasLtHandle_t getCublasLtHandle() const
    {
        return *(this->mCublasLtHandle);
    }
};

} // namespace common

} // namespace tensorrt_llm
