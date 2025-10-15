/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "config.h"
#include "nccl.h"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#include "kernels.h"
#endif
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "vector_types.h"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

namespace tensorrt_llm::kernels::nccl_device
{

std::pair<int, int> LaunchConfig::pickLaunchCombo(std::vector<std::pair<int, int>> const& options)
{
    return options.at(0); // Experimenting found that using less unroll and more threads per block is better
}

LaunchConfig::LaunchConfig(int const hidden_dim, int const num_tokens, int const rank, int const nRanks,
    bool useResidual, bool useBias, bool unshardResidualOut, int const num_sms)
    : hidden_dim(hidden_dim)
    , num_tokens(num_tokens)
    , rank(rank)
    , nRanks(nRanks)
    , useResidual(useResidual)
    , useBias(useBias)
    , unshardResidualOut(unshardResidualOut)
    , token_per_rank(-1)
    , start_token(-1)
    , num_sms(num_sms)
    , valid(false)
    , threadsPerBlock(0)
    , unrollFactor(0)
{
    // Distribute tokens across ranks: first 'remainder' ranks get one extra token
    int const base_tokens = num_tokens / nRanks;
    int const remainder = num_tokens % nRanks;
    token_per_rank = base_tokens + (rank < remainder ? 1 : 0);
    start_token = rank * base_tokens + std::min(rank, remainder);

    // Query GPU SM count if not specified
    if (this->num_sms == -1)
    {
        int local_sms = 1;
        int dev = -1;
        cudaError_t cudaStatus = cudaGetDevice(&dev);
        if (cudaStatus == cudaSuccess)
        {
            cudaDeviceProp deviceProp;
            cudaStatus = cudaGetDeviceProperties(&deviceProp, dev);
            if (cudaStatus == cudaSuccess)
            {
                local_sms = deviceProp.multiProcessorCount;
            }
            else
            {
                TLLM_LOG_WARNING("Failed to get device properties for SM count: %s. Using default num_sms=1.",
                    cudaGetErrorString(cudaStatus));
            }
        }
        else
        {
            TLLM_LOG_WARNING(
                "Failed to get CUDA device for SM count: %s. Using default num_sms=1.", cudaGetErrorString(cudaStatus));
        }

        // Coordinate SM count across all ranks using MPI_Allreduce with MIN operation
        // This ensures all ranks use the same (minimum) number of SMs
#if ENABLE_MULTI_DEVICE
        try
        {
            COMM_SESSION.allreduce(
                &local_sms, &this->num_sms, 1, tensorrt_llm::mpi::MpiType::kINT32, tensorrt_llm::mpi::MpiOp::MIN);
            TLLM_LOG_DEBUG("Coordinated num_sms across ranks: local=%d, min=%d", local_sms, this->num_sms);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING("Failed to coordinate SM count via MPI: %s. Using local value: %d", e.what(), local_sms);
            this->num_sms = local_sms;
        }
#else
        this->num_sms = local_sms;
#endif
    }
}

std::string LaunchConfig::getLoggingString() const
{
    std::ostringstream oss;
    if (this->valid)
    {
        oss << "Launching Kernel: NCCL fused AR kernel!\n";
    }
    else
    {
        oss << "Unable to Launch Kernel: NCCL fused AR kernel ";
    }
    oss << "\tConfiguration:\n";
    oss << "\t\t ThreadsPerBlock: " << this->getThreadsPerBlock() << "\n";
    oss << "\t\t UnrollFactor: " << this->getUnrollFactor() << "\n";
    oss << "\t\t BlocksPerRank (gridDim.x): " << this->getNumSMs() << "\n";
    oss << "\t\t TokensPerRank: " << this->token_per_rank << "\n";
    oss << "\t\t NumSMs: " << this->getNumSMs() << "\n";
    oss << "\t\t VectorInfo: " << this->getElementsPerVector() << "\n";
    oss << "\t\t HiddenDim: " << this->getElementsPerVector() * this->getUnrollFactor() * this->getThreadsPerBlock()
        << " = " << this->hidden_dim << "\n";
    oss << "\t\t NumTokens: " << this->num_tokens << "\n";
    oss << "\t\t StartToken: " << this->getStartToken() << "\n";

    return oss.str();
}

// Template class implementation
template <typename T>
TypedLaunchConfig<T>::TypedLaunchConfig(int const hidden_dim, int const num_tokens, int const rank, int const nRanks,
    bool useResidual, bool useBias, bool unshardResidualOut, int const num_sms)
    : LaunchConfig(hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, unshardResidualOut, num_sms)
{

    // Calculate optimal block size to achieve better coverage
    int const maxThreadsPerBlock = kMaxThreadsPerBlock; // Maximum allowed block size
    int const minThreadsPerBlock = kMinThreadsPerBlock; // Minimum block size (warp size)

    std::vector<std::pair<int, int>> valid_launch_combo;

    // Try to find a block size that gives optimal coverage
    for (int testThreadsPerBlock = maxThreadsPerBlock; testThreadsPerBlock >= minThreadsPerBlock;
         testThreadsPerBlock -= minThreadsPerBlock)
    {
        for (int testUnrollFactor = 1; testUnrollFactor <= kMaxUnrollFactor; testUnrollFactor += 1)
        {
            size_t const elementsProcessedPerBlock = elementsPerVector * testUnrollFactor * testThreadsPerBlock;

            if (elementsProcessedPerBlock == hidden_dim)
            {
                // Validate that this configuration can actually be launched
                if (isValidConfig(testThreadsPerBlock, testUnrollFactor, this->token_per_rank))
                {
                    valid_launch_combo.push_back(std::make_pair(testThreadsPerBlock, testUnrollFactor));
                }
            }
        }
    }

    if (valid_launch_combo.size() > 0)
    {
        std::pair<int, int> optimal_launch_combo = pickLaunchCombo(valid_launch_combo);

        // Set the calculated optimal values
        this->threadsPerBlock = optimal_launch_combo.first;
        this->unrollFactor = optimal_launch_combo.second;
        this->valid = true;
    }
}

std::shared_ptr<LaunchConfig> makeLaunchConfig(nvinfer1::DataType dataType, int const hidden_dim, int const num_tokens,
    int const rank, int const nRanks, bool useResidual, bool useBias, bool unshardResidualOut, int const num_sms)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kHALF:
        return std::make_shared<TypedLaunchConfig<half>>(
            hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, unshardResidualOut, num_sms);
    case nvinfer1::DataType::kBF16:
        return std::make_shared<TypedLaunchConfig<__nv_bfloat16>>(
            hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, unshardResidualOut, num_sms);
    case nvinfer1::DataType::kFLOAT:
        return std::make_shared<TypedLaunchConfig<float>>(
            hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, unshardResidualOut, num_sms);
    default: TLLM_THROW("Unimplemented data type for fused NCCL AllReduce launches.");
    }
    return nullptr;
}

// Explicit template instantiations
template class TypedLaunchConfig<half>;
template class TypedLaunchConfig<__nv_bfloat16>;
template class TypedLaunchConfig<float>;

// Implementation of launch configuration validation
template <typename T>
bool TypedLaunchConfig<T>::isValidConfig(int threadsPerBlock, int unrollFactor, int blocksPerRank) const
{
    // Get CUDA device properties
    int dev = -1;
    cudaError_t cudaStatus = cudaGetDevice(&dev);
    if (cudaStatus != cudaSuccess)
    {
        TLLM_LOG_ERROR("Failed to get CUDA device: " + std::string(cudaGetErrorString(cudaStatus)));
        return false;
    }
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, dev);
    if (cudaStatus != cudaSuccess)
    {
        TLLM_LOG_ERROR("Failed to get CUDA device properties: " + std::string(cudaGetErrorString(cudaStatus)));
        return false;
    }

    // 1. Check threads per block limits
    if (threadsPerBlock <= 0 || threadsPerBlock > deviceProp.maxThreadsPerBlock)
    {
        return false;
    }

    // 2. Check warp size alignment
    if (threadsPerBlock % deviceProp.warpSize != 0)
    {
        return false;
    }

    // 3. Check unroll factor validity
    if (unrollFactor <= 0 || unrollFactor > kMaxUnrollFactor)
    {
        return false;
    }

    // 4. Check blocks per rank
    if (blocksPerRank <= 0)
    {
        return false;
    }

    // 6. Query actual kernel resource usage from kernel pointer for the specific unroll factor
    void* kernelPtr = this->getKernelPtrForUnrollFactor(unrollFactor);
    if (kernelPtr == nullptr)
    {
        return false;
    }

    // Get actual register and shared memory usage from the kernel
    cudaFuncAttributes funcAttrib;
    cudaError_t attrStatus = cudaFuncGetAttributes(&funcAttrib, reinterpret_cast<void const*>(kernelPtr));
    if (attrStatus != cudaSuccess)
    {
        TLLM_LOG_WARNING(
            "Failed to get kernel attributes for validation: " + std::string(cudaGetErrorString(attrStatus)));
        return false;
    }

    // Check register usage
    int const totalRegistersPerBlock = funcAttrib.numRegs * threadsPerBlock;
    if (totalRegistersPerBlock > deviceProp.regsPerBlock)
    {
        return false;
    }

    // Check shared memory usage
    if (funcAttrib.sharedSizeBytes > deviceProp.sharedMemPerBlock)
    {
        return false;
    }

    // 8. Check occupancy
    int const warpsPerBlock = threadsPerBlock / deviceProp.warpSize;
    int const maxWarpsPerSM = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
    int const maxBlocksPerSM = deviceProp.maxThreadsPerMultiProcessor / threadsPerBlock;

    if (warpsPerBlock > maxWarpsPerSM)
    {
        return false;
    }

    if (maxBlocksPerSM <= 0)
    {
        return false;
    }
    return true;
}

// Template function implementations
template <typename T>
template <int Nunroll>
void* TypedLaunchConfig<T>::getKernelPtrForUnroll() const
{
    using TN = typename VectorType<T>::type;

    void* result = nullptr;
    if (useResidual && useBias && unshardResidualOut)
    {
        result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, true>);
    }
    else if (useResidual && useBias && !unshardResidualOut)
    {
        result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, false>);
    }
    else if (useResidual && !useBias && unshardResidualOut)
    {
        result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, false, true>);
    }
    else if (useResidual && !useBias && !unshardResidualOut)
    {
        result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, false, false>);
    }
    else if (!useResidual && useBias)
    {
        result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, true, false>);
    }
    else
    {
        result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, false, false>);
    }

    return result;
}

template <typename T>
void* TypedLaunchConfig<T>::getKernelPtrForUnrollFactor(int unrollFactor) const
{
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    void* result = nullptr;
    switch (unrollFactor)
    {
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
#else
    return nullptr;
#endif
}

// Function to launch kernel for any unroll factor (shares logic with getKernelPtrForUnrollFactor)
template <typename T>
void TypedLaunchConfig<T>::launchKernelForUnrollFactor(ncclWindow_t inWindow, ncclWindow_t outWindow,
    void const* const residual, ncclWindow_t residualOutWindow, void const* const weight, void const* const bias,
    ncclDevComm devComm, float const eps, cudaStream_t stream) const
{
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    // Use the same logic as getKernelPtrForUnrollFactor but launch the kernel directly
    switch (this->unrollFactor)
    {
    case 1:
        this->launchKernelForUnrollImpl<1>(
            inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream);
        break;
    case 2:
        this->launchKernelForUnrollImpl<2>(
            inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream);
        break;
    case 3:
        this->launchKernelForUnrollImpl<3>(
            inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream);
        break;
    case 4:
        this->launchKernelForUnrollImpl<4>(
            inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream);
        break;
    case 5:
        this->launchKernelForUnrollImpl<5>(
            inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream);
        break;
    case 6:
        this->launchKernelForUnrollImpl<6>(
            inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream);
        break;
    case 7:
        this->launchKernelForUnrollImpl<7>(
            inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream);
        break;
    case 8:
        this->launchKernelForUnrollImpl<8>(
            inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream);
        break;
    default:
        TLLM_CHECK_WITH_INFO(false, "Invalid unroll factor %d for %s precision. Supported values: 1-8",
            this->unrollFactor, typeid(T).name());
    }
#else
    TLLM_THROW("NCCL device kernels not available (NCCL version < 2.28). Cannot launch kernel.");
#endif
}

// Template implementation that shares the exact same logic as getKernelPtrForUnroll
template <typename T>
template <int Nunroll>
void TypedLaunchConfig<T>::launchKernelForUnrollImpl(ncclWindow_t inWindow, ncclWindow_t outWindow,
    void const* const residual, ncclWindow_t residualOutWindow, void const* const weight, void const* const bias,
    ncclDevComm devComm, float const eps, cudaStream_t stream) const
{
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    using TN = typename VectorType<T>::type;

    // Calculate grid and block dimensions from config members
    // Use num_sms for grid dimension to match available hardware parallelism
    dim3 const gridDim(this->num_sms, 1, 1);
    dim3 const blockDim(this->threadsPerBlock, 1, 1);
    size_t const sharedMemSize = 0;

    // Use the exact same logic as getKernelPtrForUnroll but launch the kernel
    if (this->useResidual && this->useBias && this->unshardResidualOut)
    {
        fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, true><<<gridDim, blockDim, sharedMemSize, stream>>>(
            inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow, static_cast<const TN*>(weight),
            static_cast<const TN*>(bias), this->start_token, this->hidden_dim, this->token_per_rank, devComm, eps);
    }
    else if (this->useResidual && this->useBias && !this->unshardResidualOut)
    {
        fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, false><<<gridDim, blockDim, sharedMemSize, stream>>>(
            inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow, static_cast<const TN*>(weight),
            static_cast<const TN*>(bias), this->start_token, this->hidden_dim, this->token_per_rank, devComm, eps);
    }
    else if (this->useResidual && !this->useBias && this->unshardResidualOut)
    {
        fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, false, true><<<gridDim, blockDim, sharedMemSize, stream>>>(
            inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow, static_cast<const TN*>(weight),
            static_cast<const TN*>(bias), this->start_token, this->hidden_dim, this->token_per_rank, devComm, eps);
    }
    else if (this->useResidual && !this->useBias && !this->unshardResidualOut)
    {
        fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, false, false><<<gridDim, blockDim, sharedMemSize, stream>>>(
            inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow, static_cast<const TN*>(weight),
            static_cast<const TN*>(bias), this->start_token, this->hidden_dim, this->token_per_rank, devComm, eps);
    }
    else if (!this->useResidual && this->useBias)
    {
        fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, true, false><<<gridDim, blockDim, sharedMemSize, stream>>>(
            inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow, static_cast<const TN*>(weight),
            static_cast<const TN*>(bias), this->start_token, this->hidden_dim, this->token_per_rank, devComm, eps);
    }
    else
    {
        fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, false, false><<<gridDim, blockDim, sharedMemSize, stream>>>(
            inWindow, outWindow, static_cast<const TN*>(residual), residualOutWindow, static_cast<const TN*>(weight),
            static_cast<const TN*>(bias), this->start_token, this->hidden_dim, this->token_per_rank, devComm, eps);
    }
#else
    TLLM_THROW("NCCL device kernels not available (NCCL version < 2.28). Cannot launch kernel.");
#endif
}

// Implementation of launch function that handles all type-specific logic
template <typename T>
void TypedLaunchConfig<T>::launchKernel(ncclWindow_t inWindow, ncclWindow_t outWindow, void const* const residual,
    ncclWindow_t residualOutWindow, void const* const weight, void const* const bias, ncclDevComm devComm,
    float const eps, cudaStream_t stream) const
{
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    using TN = typename VectorType<T>::type;

    // Launch kernel using runtime template parameter selection
    launchKernelForUnrollFactor(inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream);
#else
    TLLM_THROW("NCCL device kernels not available (NCCL version < 2.28). Cannot launch kernel.");
#endif
}

// Member function implementations
void LaunchConfig::launchRMSNorm(ncclWindow_t inWindow, ncclWindow_t outWindow, void const* const residual,
    ncclWindow_t residualOutWindow, void const* const weight, void const* const bias, ncclDevComm devComm,
    float const eps, cudaStream_t stream) const
{
    // Input validation
    TLLM_CHECK_WITH_INFO(inWindow != nullptr, "NCCL inWindow needs to be initialized.");
    TLLM_CHECK_WITH_INFO(outWindow != nullptr, "NCCL outWindow needs to be initialized.");

    TLLM_CHECK_WITH_INFO(eps >= 0.0f, "Epsilon must be non-negative, got %f", eps);
    TLLM_CHECK_WITH_INFO(weight != nullptr, "Weight pointer cannot be null");
    TLLM_CHECK_WITH_INFO(residualOutWindow != nullptr, "Residual output pointer cannot be null");
    TLLM_CHECK_WITH_INFO(residual != nullptr, "Residual needs to be a valid pointer");
    TLLM_CHECK_WITH_INFO(this->getValid(), "LaunchConfig invalid");
    TLLM_CHECK_WITH_INFO(bias == nullptr, "we are not supporting a bias here.");

    // Delegate all launch logic to the config class
    this->launchKernel(inWindow, outWindow, residual, residualOutWindow, weight, bias, devComm, eps, stream);
}

// Runtime function to check if multimem is supported for a given data type
bool LaunchConfig::supportsMultimem() const
{
    nvinfer1::DataType dataType = this->getDataType();
#ifdef ARCH_HAS_MULTIMEM
    if (this->nRanks <= 2)
        return false; // Current NCCL requires at least 3 ranks for multimem support

    // Basic types are always supported on SM90+
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT: // float
    {
        return this->getValid();
    }
    // Half and BFloat16 with .acc::f32 qualifier (SM90+)
#ifdef ARCH_HAS_MULTIMEM_ACC_F32
    case nvinfer1::DataType::kHALF: // half
    {
        return this->getValid();
    }
    case nvinfer1::DataType::kBF16: // __nv_bfloat16
    {
        return this->getValid();
    }
#endif // ARCH_HAS_MULTIMEM_ACC_F32

    // FP8 types with .acc::f16 qualifier (SM100+)
#ifdef ARCH_HAS_MULTIMEM_FP8
    case nvinfer1::DataType::kFP8: // FP8 (either E5M2 or E4M3)
    {
        return this->getValid();
    }
#endif // ARCH_HAS_MULTIMEM_FP8
    }
#endif // ARCH_HAS_MULTIMEM
    return false;
}

} // namespace tensorrt_llm::kernels::nccl_device
