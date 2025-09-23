/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "fp4_gemm.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/opUtils.h"
#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace tensorrt_llm
{
namespace kernels
{
namespace cublaslt_kernels
{

// Base class definition
class CublasLtFp4GemmRunnerInterface
{
public:
    virtual ~CublasLtFp4GemmRunnerInterface() = default;
    
    virtual void gemm(void* D, void const* A, void const* B, 
                     void const* input_sf, void const* weight_sf,
                     float const* global_sf, int m, int n, int k, 
                     int batch_count, char* workspace, const size_t workspaceBytes, 
                     cudaStream_t stream) = 0;
    
    virtual size_t getWorkspaceSize(int const m, int const n, int const k, int batch_count) = 0;
};

// Template class definition
template <typename T>
class CublasLtFp4GemmRunner : public CublasLtFp4GemmRunnerInterface
{
public:
    CublasLtFp4GemmRunner();
    ~CublasLtFp4GemmRunner();
    
    void gemm(void* D, void const* A, void const* B, 
             void const* input_sf, void const* weight_sf,
             float const* global_sf, int m, int n, int k, 
             int batch_count, char* workspace, const size_t workspaceBytes, 
             cudaStream_t stream) override;
    
    // New overload: support different scaling factor types
    void gemm(void* D, void const* A, void const* B, 
             void const* input_sf, void const* weight_sf,
             float const* global_sf, int m, int n, int k, 
             int batch_count, char* workspace, const size_t workspaceBytes, 
             cudaStream_t stream, bool input_sf_is_uint8 = true);
    
    size_t getWorkspaceSize(int const m, int const n, int const k, int batch_count) override;

private:
    void executeCublasLtGemm(void* D, void const* A, void const* B, 
                            void const* input_sf, void const* weight_sf,
                            float const* global_sf, int m, int n, int k, 
                            char* workspace, const size_t workspaceBytes, cudaStream_t stream,
                            bool input_sf_is_uint8 = true);
                            // Note: C matrix support can be added later for D = α * A * B + β * C
    
    cublasLtHandle_t mCublasLtHandle;
    int mSm;
};

// Template class implementation
template <typename T>
CublasLtFp4GemmRunner<T>::CublasLtFp4GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TLLM_CUDA_CHECK(cublasLtCreate(&mCublasLtHandle));
    mSm = tensorrt_llm::common::getSMVersion();
}

template <typename T>
CublasLtFp4GemmRunner<T>::~CublasLtFp4GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (mCublasLtHandle)
    {
        cublasLtDestroy(mCublasLtHandle);
    }
}

template <typename T>
void CublasLtFp4GemmRunner<T>::gemm(void* D, void const* A, void const* B,
                                    void const* input_sf, void const* weight_sf,
                                    float const* global_sf, int m, int n, int k,
                                    int batch_count, char* workspace, const size_t workspaceBytes,
                                    cudaStream_t stream)
{
    // Default assumption: scaling factor is uint8 type (consistent with CUTLASS)
    gemm(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, workspace, workspaceBytes, stream, true);
}

template <typename T>
void CublasLtFp4GemmRunner<T>::gemm(void* D, void const* A, void const* B,
                                    void const* input_sf, void const* weight_sf,
                                    float const* global_sf, int m, int n, int k,
                                    int batch_count, char* workspace, const size_t workspaceBytes,
                                    cudaStream_t stream, bool input_sf_is_uint8)
{
    
    // Execute cuBLASLt GEMM
    executeCublasLtGemm(D, A, B, input_sf, weight_sf, global_sf, m, n, k, workspace, workspaceBytes, stream, input_sf_is_uint8);
    
}


template <typename T>
void CublasLtFp4GemmRunner<T>::executeCublasLtGemm(void* D, void const* A, void const* B,
                                                   void const* input_sf, void const* weight_sf,
                                                   float const* global_sf, int m, int n, int k,
                                                   char* workspace, const size_t workspaceBytes, cudaStream_t stream,
                                                   bool input_sf_is_uint8)
{
    
    // Support fp16, bf16, and fp32 output types
    cudaDataType_t output_dtype;

    if (std::is_same<T, half>::value) {
        output_dtype = CUDA_R_16F;
    } else if (std::is_same<T, __nv_bfloat16>::value) {
        output_dtype = CUDA_R_16BF;
    } else if (std::is_same<T, float>::value) {
        output_dtype = CUDA_R_32F;
    } else {
        throw std::runtime_error("CublasLtFp4GemmRunner: Unsupported output type");
    }

    
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    
    try
    {
        // Create operation descriptor
        TLLM_CUDA_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, 
                                                     &transa, sizeof(transa)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, 
                                                     &transb, sizeof(transb)));
        
        // Set scaling mode - cuBLASLt requires e4m3 format scaling factors
        cublasLtMatmulMatrixScale_t AScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulMatrixScale_t BScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulMatrixScale_t CScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        cublasLtMatmulMatrixScale_t DScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        cublasLtMatmulMatrixScale_t DOutScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, 
                                                     &AScaleMode, sizeof(AScaleMode)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, 
                                                     &BScaleMode, sizeof(BScaleMode)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, 
                                                     &CScaleMode, sizeof(CScaleMode)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, 
                                                     &DScaleMode, sizeof(DScaleMode)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, 
                                                     &DOutScaleMode, sizeof(DOutScaleMode)));
        
        // Set scaling pointers - cuBLASLt expects e4m3 format scaling factors
        const void* input_sf_ptr = input_sf;
        const void* weight_sf_ptr = weight_sf;
        
        if (input_sf_is_uint8) {
            // Input is uint8 type, cuBLASLt expects __nv_fp8_e4m3 type
            // Since bit patterns are the same, we can directly reinterpret_cast to the correct type
            input_sf_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(input_sf);
            weight_sf_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(weight_sf);
        } else {
            // Input is already __nv_fp8_e4m3 type, use directly
            input_sf_ptr = input_sf;
            weight_sf_ptr = weight_sf;
        }
        
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, 
                                                     &input_sf_ptr, sizeof(input_sf_ptr)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, 
                                                     &weight_sf_ptr, sizeof(weight_sf_ptr)));
        
        // Set scaling pointers for C and D matrices (required by cuBLASLt samples)
        // Note: We use nullptr here because current implementation doesn't need C and D matrix scaling
        const void* c_scale_ptr = nullptr;
        const void* d_scale_ptr = nullptr;
        const void* d_out_scale_ptr = nullptr;
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, 
                                                     &c_scale_ptr, sizeof(c_scale_ptr)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, 
                                                     &d_scale_ptr, sizeof(d_scale_ptr)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, 
                                                     &d_out_scale_ptr, sizeof(d_out_scale_ptr)));
        
        // Create matrix descriptors
        // Create correct matrix descriptors based on transpose operations:
        // - transa = CUBLAS_OP_T, so A is [k, m] (transposed [m, k])
        // - transb = CUBLAS_OP_N, so B is [k, n]
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, k, m, k));  // A: act_fp4 [k, m] (transposed)
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, k, n, k));  // B: weight [k, n]
        
        
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, output_dtype, m, n, m));
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, output_dtype, m, n, m));
        
        // Create preference descriptor
        TLLM_CUDA_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        TLLM_CUDA_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                                           &workspaceBytes, sizeof(workspaceBytes)));
        
        // Get heuristic algorithm
        TLLM_CUDA_CHECK(cublasLtMatmulAlgoGetHeuristic(mCublasLtHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, 
                                                     preference, 1, &heuristicResult, &returnedResults));
        
        if (returnedResults == 0) {
            throw std::runtime_error("No suitable cuBLASLt algorithm found for FP4 GEMM");
        }
        
        // Execute matmul
        float alpha = 1.0f;
        float beta = 0.0f;
        
        TLLM_CUDA_CHECK(cublasLtMatmul(mCublasLtHandle,
                                     operationDesc,
                                     &alpha,
                                     A, Adesc,  // A: act_fp4 [k, m] - Python side has swapped order
                                     B, Bdesc,  // B: weight [k, n] - Python side has swapped order
                                     &beta,
                                     nullptr, Cdesc,  // No C input needed (β = 0)
                                     D, Ddesc,  // Output to D buffer using Ddesc (bfloat16) layout
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceBytes,
                                     0));
        
        
        // Clean up resources
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
        if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
        if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
        if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
        if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
    }
    catch (...)
    {
        // Clean up resources
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
        if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
        if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
        if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
        if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
        throw;
    }
}

template <typename T>
size_t CublasLtFp4GemmRunner<T>::getWorkspaceSize(int const m, int const n, int const k, int batch_count)
{
    // 32MB
    return 33554432;
}

} // namespace cublaslt_kernels
} // namespace kernels
} // namespace tensorrt_llm