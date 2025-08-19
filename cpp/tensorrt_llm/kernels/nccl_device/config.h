/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef TRTLLM_NCCL_DEVICE_CONFIG_H
#define TRTLLM_NCCL_DEVICE_CONFIG_H

#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "vector_types.h"
#include "constants.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/runtime/iBuffer.h"

namespace tensorrt_llm::kernels::nccl_device {

    // Kernel launch information helper class
    class LaunchConfig {
    public:
        const int hidden_dim;
        const int num_tokens;
        const int nRanks;
        const int rank;
        const bool useResidual;
        const bool useBias;
        const bool unshardResidualOut;
    protected:
        int token_per_rank;
        int start_token;
        bool valid;
        int threadsPerBlock;
        int unrollFactor;

        std::pair<int, int> pickLaunchCombo(const std::vector<std::pair<int,int>>& options);

    public:
        // Constructor with dynamic block size calculation
        LaunchConfig(const int hidden_dim, const int num_tokens, const int rank, const int nRanks, bool useResidual, bool useBias, bool unshardResidualOut);

        inline int getThreadsPerBlock() const { return this->threadsPerBlock; }
        int getUnrollFactor() const{ return this->unrollFactor;}
        virtual bool getValid()const=0;
        int getBlocksPerRank() const {return this->token_per_rank;}
        int getStartToken()const {return this->start_token;}
        virtual int getElementsPerVector()const = 0;
        virtual nvinfer1::DataType getDataType()const =0;
        virtual void* getKernelPtr() const = 0;
        virtual bool isValidConfig(int threadsPerBlock, int unrollFactor, int blocksPerRank) const = 0;
        
        // Launcher functions as member functions
        void launchRMSNorm(ncclWindow_t inWindow, ncclWindow_t outWindow,
                          const void* const residual, ncclWindow_t residualOutWindow,
                          const void* const weight, const void* const bias,
                          ncclDevComm devComm, const float eps, cudaStream_t stream) const;
        
        bool supportsMultimem() const;
    
    protected:
        // Pure virtual launch function that must be implemented by derived classes
        virtual void launchKernel(ncclWindow_t inWindow, ncclWindow_t outWindow,
                                 const void* const residual, ncclWindow_t residualOutWindow,
                                 const void* const weight, const void* const bias,
                                 ncclDevComm devComm, const float eps, cudaStream_t stream) const = 0;
    
        // Logging output
        std::string getLoggingString() const;
    };

  
    // Kernel launch information helper class
    template<typename T>
    class TypedLaunchConfig : public LaunchConfig {
    private:
        nvinfer1::DataType mType;
        
        // Private templated helper function to get kernel pointer for specific unroll factor
        template<int Nunroll>
        void* getKernelPtrForUnroll() const;
        
        // Private helper function to get kernel pointer for any unroll factor
        void* getKernelPtrForUnrollFactor(int unrollFactor) const;
        
        // Private helper function to launch kernel for any unroll factor
        void launchKernelForUnrollFactor(ncclWindow_t inWindow, ncclWindow_t outWindow,
                                        const void* const residual, ncclWindow_t residualOutWindow,
                                        const void* const weight, const void* const bias,
                                        ncclDevComm devComm, const float eps, cudaStream_t stream,
                                        const dim3& gridDim, const dim3& blockDim, const size_t sharedMemSize) const;
        
        // Private templated helper function to launch kernel for specific unroll factor
        template<int Nunroll>
        void launchKernelForUnrollImpl(ncclWindow_t inWindow, ncclWindow_t outWindow,
                                       const void* const residual, ncclWindow_t residualOutWindow,
                                       const void* const weight, const void* const bias,
                                       ncclDevComm devComm, const float eps, cudaStream_t stream,
                                       const dim3& gridDim, const dim3& blockDim, const size_t sharedMemSize,
                                       bool useResidual, bool useBias, bool unshardResidualOut,
                                       int startToken, int hiddenDim, int numTokens) const;
        
    public:
        using TN = typename VectorType<T>::type;
        constexpr static int elementsPerVector = sizeof(TN) / sizeof(T);
    public:

        virtual int getElementsPerVector() const {return this->elementsPerVector;}
        virtual void* getKernelPtr() const override { return getKernelPtrForUnrollFactor(this->unrollFactor); }
        virtual bool isValidConfig(int threadsPerBlock, int unrollFactor, int blocksPerRank) const override;
        
        // Launch function that handles all the type-specific logic internally
        virtual void launchKernel(ncclWindow_t inWindow, ncclWindow_t outWindow,
                                 const void* const residual, ncclWindow_t residualOutWindow,
                                 const void* const weight, const void* const bias,
                                 ncclDevComm devComm, const float eps, cudaStream_t stream) const override;
  
        // Constructor with dynamic block size calculation
        TypedLaunchConfig(const int hidden_dim, const int num_tokens, const int rank, const int nRanks, bool useResidual, bool useBias, bool unshardResidualOut);
        nvinfer1::DataType getDataType()const{return tensorrt_llm::runtime::TRTDataType<T>::value;}
        virtual bool getValid()const{ return this->valid;}

    };

    std::shared_ptr<LaunchConfig> makeLaunchConfig(nvinfer1::DataType dataType, const int hidden_dim, const int num_tokens, const int rank, const int nRanks, bool useResidual, bool useBias, bool unshardResidualOut);

} // namespace tensorrt_llm::kernels::nccl_device

#endif // TRTLLM_NCCL_DEVICE_CONFIG_H 
