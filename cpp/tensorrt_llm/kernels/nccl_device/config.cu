/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "config.h"
#include "kernels.h"
#include "vector_types.h"
#include <memory>
#include <cuda_runtime.h>
#include <iostream>
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::kernels::nccl_device {

    std::pair<int, int> LaunchConfig::pickLaunchCombo(const std::vector<std::pair<int,int>>& options){
        return options.at(0); // Experimenting found that using less unroll and more threads per block is better
    }

    LaunchConfig::LaunchConfig(const int hidden_dim, const int num_tokens, const int rank, const int nRanks, bool useResidual, bool useBias, bool unshardResidualOut)
        : hidden_dim(hidden_dim), num_tokens(num_tokens), rank(rank), nRanks(nRanks), useResidual(useResidual), useBias(useBias), unshardResidualOut(unshardResidualOut), token_per_rank(-1), start_token(-1), valid(false), threadsPerBlock(0), unrollFactor(0){

        token_per_rank = num_tokens/nRanks;
        int remainder = num_tokens % nRanks;
        if (remainder > 0)
        {
            token_per_rank += 1;
        }
        start_token = token_per_rank * rank;    
    }

    std::string LaunchConfig::getLoggingString() const{
        std::ostringstream oss;
        if (this->valid)
        {
            oss<< "Launching Kernel: NCCL fused AR kernel!\n";
        }
        else
        {
            oss<< "Unable to Launch Kernel: NCCL fused AR kernel ";
        }
        oss<< "\tConfiguration:\n";
        oss<< "\t\t ThreadsPerBlock: "<<this->getThreadsPerBlock()<<"\n";
        oss<< "\t\t UnrollFactor: "<<this->getUnrollFactor()<<"\n";
        oss<< "\t\t BlocksPerRank: "<<this->getBlocksPerRank()<< " = " <<this->token_per_rank << "\n";
        oss<< "\t\t VectorInfo: " <<this->getElementsPerVector()<<"\n";
        oss<< "\t\t HiddenDim: " <<this->getElementsPerVector() * this->getUnrollFactor() * this->getThreadsPerBlock() << " = " << this->hidden_dim<<"\n";
        oss<< "\t\t NumTokens: " <<this->num_tokens << "\n";
        oss<< "\t\t StartToken: "<<this->getStartToken()<<"\n";
  
        return oss.str();
    }

    // Template class implementation
    template<typename T>
    TypedLaunchConfig<T>::TypedLaunchConfig(const int hidden_dim, const int num_tokens, const int rank, const int nRanks, bool useResidual, bool useBias, bool unshardResidualOut)
        : LaunchConfig(hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, unshardResidualOut)
    {
        
        // Calculate optimal block size to achieve better coverage
        const int maxThreadsPerBlock = kMaxThreadsPerBlock; // Maximum allowed block size
        const int minThreadsPerBlock = kMinThreadsPerBlock;   // Minimum block size (warp size)

        std::vector<std::pair<int,int>> valid_launch_combo;    
    
        // Try to find a block size that gives optimal coverage
        for (int testThreadsPerBlock = maxThreadsPerBlock; testThreadsPerBlock >= minThreadsPerBlock; testThreadsPerBlock -= minThreadsPerBlock) {
            for (int testUnrollFactor=1; testUnrollFactor <= kMaxUnrollFactor; testUnrollFactor += 1){
                const size_t elementsProcessedPerBlock = elementsPerVector * testUnrollFactor * testThreadsPerBlock;

                if(elementsProcessedPerBlock == hidden_dim) {
                    // Validate that this configuration can actually be launched
                    if (isValidConfig(testThreadsPerBlock, testUnrollFactor, this->token_per_rank)) {
                        valid_launch_combo.push_back(std::make_pair(testThreadsPerBlock, testUnrollFactor));
                    }
                }
            }
        }

        if (valid_launch_combo.size() > 0)
        {
            std::pair<int,int> optimal_launch_combo = pickLaunchCombo(valid_launch_combo);

            // Set the calculated optimal values
            this->threadsPerBlock = optimal_launch_combo.first;
            this->unrollFactor = optimal_launch_combo.second;
            this->valid = true;
            
        }
    }

    std::shared_ptr<LaunchConfig> makeLaunchConfig(nvinfer1::DataType dataType, const int hidden_dim, const int num_tokens, const int rank, const int nRanks, bool useResidual, bool useBias, bool unshardResidualOut)
    {
        switch(dataType){
        case nvinfer1::DataType::kHALF:
            return std::make_shared<TypedLaunchConfig<half>>(hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, unshardResidualOut);
        case nvinfer1::DataType::kBF16:
            return std::make_shared<TypedLaunchConfig<__nv_bfloat16>>(hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, unshardResidualOut);
        case nvinfer1::DataType::kFLOAT:
            return std::make_shared<TypedLaunchConfig<float>>(hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, unshardResidualOut);
        default:
            TLLM_THROW("Unimplemented data type for fused NCCL AllReduce launches.");
        }
        return nullptr;
    }

    // Explicit template instantiations
    template class TypedLaunchConfig<half>;
    template class TypedLaunchConfig<__nv_bfloat16>;
    template class TypedLaunchConfig<float>;


    // Implementation of launch configuration validation
    template<typename T>
    bool TypedLaunchConfig<T>::isValidConfig(int threadsPerBlock, int unrollFactor, int blocksPerRank) const {
        // Get CUDA device properties
        cudaDeviceProp deviceProp;
        cudaError_t cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
        if (cudaStatus != cudaSuccess) {
            TLLM_LOG_ERROR("Failed to get CUDA device properties: " + std::string(cudaGetErrorString(cudaStatus)));
            return false;
        }

        // 1. Check threads per block limits
        if (threadsPerBlock <= 0 || threadsPerBlock > deviceProp.maxThreadsPerBlock) {
            return false;
        }

        // 2. Check warp size alignment
        if (threadsPerBlock % deviceProp.warpSize != 0) {
            return false;
        }

        // 3. Check unroll factor validity
        if (unrollFactor <= 0 || unrollFactor > kMaxUnrollFactor) {
            return false;
        }

        // 4. Check blocks per rank
        if (blocksPerRank <= 0) {
            return false;
        }

        // 6. Query actual kernel resource usage from kernel pointer for the specific unroll factor
        void* kernelPtr = this->getKernelPtrForUnrollFactor(unrollFactor);
        if (kernelPtr == nullptr) {
            return false;
        }

        // Get actual register and shared memory usage from the kernel
        cudaFuncAttributes funcAttrib;
        cudaError_t attrStatus = cudaFuncGetAttributes(&funcAttrib, reinterpret_cast<const void*>(kernelPtr));
        if (attrStatus != cudaSuccess) {
            TLLM_LOG_WARNING("Failed to get kernel attributes for validation: " + std::string(cudaGetErrorString(attrStatus)));
            return false;
        }

        // Check register usage
        const int totalRegistersPerBlock = funcAttrib.numRegs * threadsPerBlock;
        if (totalRegistersPerBlock > deviceProp.regsPerBlock) {
            return false;
        }

        // Check shared memory usage
        if (funcAttrib.sharedSizeBytes > deviceProp.sharedMemPerBlock) {
            return false;
        }

        // 8. Check occupancy
        const int warpsPerBlock = threadsPerBlock / deviceProp.warpSize;
        const int maxWarpsPerSM = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
        const int maxBlocksPerSM = deviceProp.maxThreadsPerMultiProcessor / threadsPerBlock;
        
        if (warpsPerBlock > maxWarpsPerSM) {
            return false;
        }

        if (maxBlocksPerSM <= 0) {
            return false;
        }
        return true;
    }

    // Template function implementations
    template<typename T>
    template<int Nunroll>
    void* TypedLaunchConfig<T>::getKernelPtrForUnroll() const {
        using TN = typename VectorType<T>::type;
        
        
        void* result = nullptr;
        if (useResidual && useBias && unshardResidualOut) {
            result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, true>);
        } else if (useResidual && useBias && !unshardResidualOut) {
            result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, false>);
        } else if (useResidual && !useBias && unshardResidualOut) {
            result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, false, true>);
        } else if (useResidual && !useBias && !unshardResidualOut) {
            result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, false, false>);
        } else if (!useResidual && useBias) {
            result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, true, false>);
        } else {
            result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, false, false>);
        }
        
        return result;
    }

    template<typename T>
    void* TypedLaunchConfig<T>::getKernelPtrForUnrollFactor(int unrollFactor) const {
        
        void* result = nullptr;
        switch(unrollFactor) {
            case 1: result = getKernelPtrForUnroll<1>(); break;
            case 2: result = getKernelPtrForUnroll<2>(); break;
            case 3: result = getKernelPtrForUnroll<3>(); break;
            case 4: result = getKernelPtrForUnroll<4>(); break;
            case 5: result = getKernelPtrForUnroll<5>(); break;
            case 6: result = getKernelPtrForUnroll<6>(); break;
            case 7: result = getKernelPtrForUnroll<7>(); break;
            case 8: result = getKernelPtrForUnroll<8>(); break;
            default: result = nullptr; break;
        }
        
        return result;
    }


    // Function to launch kernel for any unroll factor (shares logic with getKernelPtrForUnrollFactor)
    template<typename T>
    void TypedLaunchConfig<T>::launchKernelForUnrollFactor(ncclWindow_t inWindow, ncclWindow_t outWindow,
                                                                          const void* const residual, ncclWindow_t residualOutWindow,
                                                                          const void* const weight, const void* const bias,
                                                                          ncclDevComm devComm, const float eps, cudaStream_t stream,
                                                                          const dim3& gridDim, const dim3& blockDim, const size_t sharedMemSize) const {
        using TN = typename VectorType<T>::type;
        
        
        // Use the same logic as getKernelPtrForUnrollFactor but launch the kernel directly
        switch(this->unrollFactor) {
            case 1: this->launchKernelForUnrollImpl<1>(inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream, gridDim, blockDim, sharedMemSize, useResidual, useBias, unshardResidualOut, this->getStartToken(), this->hidden_dim, this->num_tokens); break;
            case 2: this->launchKernelForUnrollImpl<2>(inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream, gridDim, blockDim, sharedMemSize, useResidual, useBias, unshardResidualOut, this->getStartToken(), this->hidden_dim, this->num_tokens); break;
            case 3: this->launchKernelForUnrollImpl<3>(inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream, gridDim, blockDim, sharedMemSize, useResidual, useBias, unshardResidualOut, this->getStartToken(), this->hidden_dim, this->num_tokens); break;
            case 4: this->launchKernelForUnrollImpl<4>(inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream, gridDim, blockDim, sharedMemSize, useResidual, useBias, unshardResidualOut, this->getStartToken(), this->hidden_dim, this->num_tokens); break;
            case 5: this->launchKernelForUnrollImpl<5>(inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream, gridDim, blockDim, sharedMemSize, useResidual, useBias, unshardResidualOut, this->getStartToken(), this->hidden_dim, this->num_tokens); break;
            case 6: this->launchKernelForUnrollImpl<6>(inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream, gridDim, blockDim, sharedMemSize, useResidual, useBias, unshardResidualOut, this->getStartToken(), this->hidden_dim, this->num_tokens); break;
            case 7: this->launchKernelForUnrollImpl<7>(inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream, gridDim, blockDim, sharedMemSize, useResidual, useBias, unshardResidualOut, this->getStartToken(), this->hidden_dim, this->num_tokens); break;
            case 8: this->launchKernelForUnrollImpl<8>(inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream, gridDim, blockDim, sharedMemSize, useResidual, useBias, unshardResidualOut, this->getStartToken(), this->hidden_dim, this->num_tokens); break;
            default: 
                TLLM_CHECK_WITH_INFO(false, "Invalid unroll factor %d for %s precision. Supported values: 1-8", 
                                     this->unrollFactor, typeid(T).name());
        }
    }

    // Template implementation that shares the exact same logic as getKernelPtrForUnroll
    template<typename T>
    template<int Nunroll>
    void TypedLaunchConfig<T>::launchKernelForUnrollImpl(ncclWindow_t inWindow, ncclWindow_t outWindow,
                                   const void* const residual, ncclWindow_t residualOutWindow,
                                   const void* const weight, const void* const bias,
                                   ncclDevComm devComm, const float eps, cudaStream_t stream,
                                   const dim3& gridDim, const dim3& blockDim, const size_t sharedMemSize,
                                   bool useResidual, bool useBias, bool unshardResidualOut,
                                   int startToken, int hiddenDim, int numTokens) const {
        using TN = typename VectorType<T>::type;
        
        
        // Use the exact same logic as getKernelPtrForUnroll but launch the kernel
        if (useResidual && useBias && unshardResidualOut) {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, true><<<gridDim, blockDim, sharedMemSize, stream>>>(
                inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow,
                static_cast<const TN*>(weight), static_cast<const TN*>(bias),
                startToken, hiddenDim, numTokens, devComm, eps);
        } else if (useResidual && useBias && !unshardResidualOut) {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, false><<<gridDim, blockDim, sharedMemSize, stream>>>(
                inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow,
                static_cast<const TN*>(weight), static_cast<const TN*>(bias),
                startToken, hiddenDim, numTokens, devComm, eps);
        } else if (useResidual && !useBias && unshardResidualOut) {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, false, true><<<gridDim, blockDim, sharedMemSize, stream>>>(
                inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow,
                static_cast<const TN*>(weight), static_cast<const TN*>(bias),
                startToken, hiddenDim, numTokens, devComm, eps);
        } else if (useResidual && !useBias && !unshardResidualOut) {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, false, false><<<gridDim, blockDim, sharedMemSize, stream>>>(
                inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow,
                static_cast<const TN*>(weight), static_cast<const TN*>(bias),
                startToken, hiddenDim, numTokens, devComm, eps);
        } else if (!useResidual && useBias) {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, true, false><<<gridDim, blockDim, sharedMemSize, stream>>>(
                inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow,
                static_cast<const TN*>(weight), static_cast<const TN*>(bias),
                startToken, hiddenDim, numTokens, devComm, eps);
        } else {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, false, false><<<gridDim, blockDim, sharedMemSize, stream>>>(
                inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow,
                static_cast<const TN*>(weight), static_cast<const TN*>(bias),
                startToken, hiddenDim, numTokens, devComm, eps);
        }
    }

    // Implementation of launch function that handles all type-specific logic
    template<typename T>
    void TypedLaunchConfig<T>::launchKernel(ncclWindow_t inWindow, ncclWindow_t outWindow,
                                                          const void* const residual, ncclWindow_t residualOutWindow,
                                                          const void* const weight, const void* const bias,
                                                          ncclDevComm devComm, const float eps, cudaStream_t stream) const {
        using TN = typename VectorType<T>::type;
        
        
        const dim3 gridDim(this->getBlocksPerRank(), 1, 1);
        const dim3 blockDim(this->getThreadsPerBlock(), 1, 1);
        const size_t sharedMemSize = 0;
        
        
        // Get the kernel pointer
        void* kernelPtr = getKernelPtrForUnrollFactor(this->unrollFactor);
        
        // Note: Kernel pointer obtained for potential direct launch, but using launchKernelForUnrollFactor instead
        
        
        // Launch kernel using runtime template parameter selection (shared with getKernelPtrForUnroll logic)
        launchKernelForUnrollFactor(inWindow, outWindow, residual, residualOutWindow, 
                                   weight, bias, devComm, eps, stream, gridDim, blockDim, sharedMemSize);
        
    }

    // Member function implementations
    void LaunchConfig::launchRMSNorm(ncclWindow_t inWindow, ncclWindow_t outWindow,
                                                                const void* const residual, ncclWindow_t residualOutWindow,
                                                                const void* const weight, const void* const bias,
                                                                ncclDevComm devComm, const float eps, cudaStream_t stream) const
    {
        // Input validation
        TLLM_CHECK_WITH_INFO(inWindow != nullptr, "NCCL inWindow needs to be initialized.");
        TLLM_CHECK_WITH_INFO(outWindow != nullptr, "NNCL outWindow needs to be initialized.");

        TLLM_CHECK_WITH_INFO(eps >= 0.0f, "Epsilon must be non-negative, got %f", eps);
        TLLM_CHECK_WITH_INFO(weight != nullptr, "Weight pointer cannot be null");
        TLLM_CHECK_WITH_INFO(residualOutWindow != nullptr, "Residual output pointer cannot be null");
        TLLM_CHECK_WITH_INFO(residual != nullptr, "Residual needs to be a valid pointer");
        TLLM_CHECK_WITH_INFO(this->getValid(), "LaunchConfig invalid");
        TLLM_CHECK_WITH_INFO(bias == nullptr, "we are not supporting a bias here.");

        // Delegate all launch logic to the config class
        this->launchKernel(inWindow, outWindow, residual, residualOutWindow, 
                          weight, bias, devComm, eps, stream);
    }

    // Runtime function to check if multimem is supported for a given data type
    bool LaunchConfig::supportsMultimem() const {
        nvinfer1::DataType dataType = this->getDataType();
#ifdef ARCH_HAS_MULTIMEM
        if(this->nRanks <= 2)
            return false; // Current NCCL requires at least 3 ranks for multimem support

        // Basic types are always supported on SM90+
        switch (dataType)
        {
        case nvinfer1::DataType::kFLOAT:   // float
        {
            return this->getValid();
        }
        // Half and BFloat16 with .acc::f32 qualifier (SM90+)
#ifdef ARCH_HAS_MULTIMEM_ACC_F32
        case nvinfer1::DataType::kHALF:    // half
        {
            return this->getValid();
        }
        case nvinfer1::DataType::kBF16:    // __nv_bfloat16
        {
            return this->getValid();
        }
#endif // ARCH_HAS_MULTIMEM_ACC_F32

        // FP8 types with .acc::f16 qualifier (SM100+)
#ifdef ARCH_HAS_MULTIMEM_FP8
        case nvinfer1::DataType::kFP8:     // FP8 (either E5M2 or E4M3)
        {
            return this->getValid();
        }
#endif // ARCH_HAS_MULTIMEM_FP8
        }
#endif //ARCH_HAS_MULTIMEM
        return false;
    }

} // namespace tensorrt_llm::kernels::nccl_device
