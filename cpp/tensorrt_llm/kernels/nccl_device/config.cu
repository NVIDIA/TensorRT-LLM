/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "config.h"
#include "nccl.h"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#include "kernels.cuh"
#endif
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
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
    bool useResidual, bool useBias, int const num_sms)
    : hidden_dim(hidden_dim)
    , num_tokens(num_tokens)
    , rank(rank)
    , nRanks(nRanks)
    , useResidual(useResidual)
    , useBias(useBias)
    , oneShot(false)
    , token_per_rank(-1)
    , start_token(-1)
    , num_sms(num_sms)
    , valid(false)
    , threadsPerBlock(0)
    , unrollFactor(0)
{
    // No grid-stride
    int const base_tokens = num_tokens / nRanks;
    this->num_sms = base_tokens;
    int const remainder = num_tokens % nRanks;
    if (remainder > 0)
        this->num_sms += 1;

    // TODO hard coded value for now. Maybe some tuning possible
    if (num_tokens <= 32)
        this->oneShot = true;

    if (this->oneShot)
    {
        // In one shot mode, each rank processes all tokens
        this->token_per_rank = num_tokens;
        this->start_token = 0;
        this->num_sms = num_tokens;
    }
    else
    {
        // Distribute tokens across ranks: first 'remainder' ranks get one extra token
        this->token_per_rank = base_tokens + (rank < remainder ? 1 : 0);
        this->start_token = rank * base_tokens + std::min(rank, remainder);
    }

    auto maxCTAEnv = tensorrt_llm::common::getIntEnv("TLLM_NCCL_DEVICE_AR_RMS_MAX_CTA");
    if (maxCTAEnv.has_value())
    {
        if (maxCTAEnv.value() > 0)
            this->num_sms = maxCTAEnv.value();
        else
        {
            TLLM_LOG_WARNING("TLLM_NCCL_DEVICE_AR_RMS_MAX_CTA was detected as <= 0 and is ignored.");
        }
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
    bool useResidual, bool useBias, int const num_sms)
    : LaunchConfig(hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, num_sms)
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
                if (isValidConfig(testThreadsPerBlock, testUnrollFactor))
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
    int const rank, int const nRanks, bool useResidual, bool useBias, int const num_sms)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kHALF:
        return std::make_shared<TypedLaunchConfig<half>>(
            hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, num_sms);
    case nvinfer1::DataType::kBF16:
        return std::make_shared<TypedLaunchConfig<__nv_bfloat16>>(
            hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, num_sms);
    case nvinfer1::DataType::kFLOAT:
        return std::make_shared<TypedLaunchConfig<float>>(
            hidden_dim, num_tokens, rank, nRanks, useResidual, useBias, num_sms);
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
bool TypedLaunchConfig<T>::isValidConfig(int threadsPerBlock, int unrollFactor) const
{
    // Get CUDA device properties
    int dev = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp deviceProp;
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // Check threads per block limits
    if (threadsPerBlock <= 0 || threadsPerBlock > deviceProp.maxThreadsPerBlock)
    {
        return false;
    }

    // Check warp size alignment
    if (threadsPerBlock % deviceProp.warpSize != 0)
    {
        return false;
    }

    // Check unroll factor validity
    if (unrollFactor <= 0 || unrollFactor > kMaxUnrollFactor)
    {
        return false;
    }

    // Query actual kernel resource usage from kernel pointer for the specific unroll factor
    void* kernelPtr = this->getKernelPtrForUnrollFactor(unrollFactor);
    if (kernelPtr == nullptr)
    {
        return false;
    }

    // Get actual register and shared memory usage from the kernel
    cudaFuncAttributes funcAttrib;
    TLLM_CUDA_CHECK(cudaFuncGetAttributes(&funcAttrib, reinterpret_cast<void const*>(kernelPtr)));

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

    // Check occupancy
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
    if (oneShot)
    {
        if (useResidual && useBias)
        {
            result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, true>);
        }
        else if (useResidual && !useBias)
        {
            result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, false, true>);
        }
        else if (!useResidual && useBias)
        {
            result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, true, true>);
        }
        else
        {
            result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, false, true>);
        }
    }
    else
    {
        if (useResidual && useBias)
        {
            result = reinterpret_cast<void*>(fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, false>);
        }
        else if (useResidual && !useBias)
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
    if (this->oneShot)
    {
        if (this->useResidual && this->useBias)
        {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, true>
                <<<gridDim, blockDim, sharedMemSize, stream>>>(inWindow, outWindow, static_cast<const TN*>(residual),
                    residualOutWindow, static_cast<const TN*>(weight), static_cast<const TN*>(bias), this->start_token,
                    this->hidden_dim, this->token_per_rank, devComm, eps);
        }
        else if (this->useResidual && !this->useBias)
        {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, false, true>
                <<<gridDim, blockDim, sharedMemSize, stream>>>(inWindow, outWindow, static_cast<const TN*>(residual),
                    residualOutWindow, static_cast<const TN*>(weight), static_cast<const TN*>(bias), this->start_token,
                    this->hidden_dim, this->token_per_rank, devComm, eps);
        }
        else if (!this->useResidual && this->useBias)
        {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, true, true>
                <<<gridDim, blockDim, sharedMemSize, stream>>>(inWindow, outWindow, static_cast<const TN*>(residual),
                    residualOutWindow, static_cast<const TN*>(weight), static_cast<const TN*>(bias), this->start_token,
                    this->hidden_dim, this->token_per_rank, devComm, eps);
        }
        else
        {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, false, true>
                <<<gridDim, blockDim, sharedMemSize, stream>>>(inWindow, outWindow, static_cast<const TN*>(residual),
                    residualOutWindow, static_cast<const TN*>(weight), static_cast<const TN*>(bias), this->start_token,
                    this->hidden_dim, this->token_per_rank, devComm, eps);
        }
    }
    else
    {
        if (this->useResidual && this->useBias)
        {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, true, false>
                <<<gridDim, blockDim, sharedMemSize, stream>>>(inWindow, outWindow, static_cast<const TN*>(residual),
                    residualOutWindow, static_cast<const TN*>(weight), static_cast<const TN*>(bias), this->start_token,
                    this->hidden_dim, this->token_per_rank, devComm, eps);
        }
        else if (this->useResidual && !this->useBias)
        {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, true, false, false>
                <<<gridDim, blockDim, sharedMemSize, stream>>>(inWindow, outWindow, static_cast<const TN*>(residual),
                    residualOutWindow, static_cast<const TN*>(weight), static_cast<const TN*>(bias), this->start_token,
                    this->hidden_dim, this->token_per_rank, devComm, eps);
        }
        else if (!this->useResidual && this->useBias)
        {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, true, false>
                <<<gridDim, blockDim, sharedMemSize, stream>>>(inWindow, outWindow, static_cast<const TN*>(residual),
                    residualOutWindow, static_cast<const TN*>(weight), static_cast<const TN*>(bias), this->start_token,
                    this->hidden_dim, this->token_per_rank, devComm, eps);
        }
        else
        {
            fusedAllReduceRMSNormKernel<T, TN, Nunroll, false, false, false>
                <<<gridDim, blockDim, sharedMemSize, stream>>>(inWindow, outWindow, static_cast<const TN*>(residual),
                    residualOutWindow, static_cast<const TN*>(weight), static_cast<const TN*>(bias), this->start_token,
                    this->hidden_dim, this->token_per_rank, devComm, eps);
        }
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
