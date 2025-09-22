/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef TRTLLM_NCCL_DEVICE_CONFIG_H
#define TRTLLM_NCCL_DEVICE_CONFIG_H

#include "constants.h"
#include "nccl.h"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#include "nccl_device.h"
#endif
#if NCCL_VERSION_CODE <= NCCL_VERSION(2, 28, 0)
using ncclDevComm = void*;
#endif
#if NCCL_VERSION_CODE <= NCCL_VERSION(2, 27, 0)
using ncclWindow_t = void*;
#endif

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "vector_types.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

namespace tensorrt_llm::kernels::nccl_device
{

// Kernel launch information helper class
class LaunchConfig
{
public:
    int const hidden_dim;
    int const num_tokens;
    int const nRanks;
    int const rank;
    bool const useResidual;
    bool const useBias;

protected:
    bool oneShot;
    int token_per_rank;
    int start_token;
    int num_sms;
    bool valid;
    int threadsPerBlock;
    int unrollFactor;

    std::pair<int, int> pickLaunchCombo(std::vector<std::pair<int, int>> const& options);

public:
    // Constructor with dynamic block size calculation
    LaunchConfig(int const hidden_dim, int const num_tokens, int const rank, int const nRanks, bool useResidual,
        bool useBias, int const num_sms = -1);

    inline int getThreadsPerBlock() const
    {
        return this->threadsPerBlock;
    }

    int getUnrollFactor() const
    {
        return this->unrollFactor;
    }

    int getNumSMs() const
    {
        return this->num_sms;
    }

    virtual bool getValid() const = 0;

    int getBlocksPerRank() const
    {
        return this->token_per_rank;
    }

    int getStartToken() const
    {
        return this->start_token;
    }

    virtual int getElementsPerVector() const = 0;
    virtual nvinfer1::DataType getDataType() const = 0;
    virtual bool isValidConfig(int threadsPerBlock, int unrollFactor) const = 0;

    // Launcher functions as member functions
    void launchRMSNorm(ncclWindow_t inWindow, ncclWindow_t outWindow, void const* const residual,
        ncclWindow_t residualOutWindow, void const* const weight, void const* const bias, ncclDevComm devComm,
        float const eps, cudaStream_t stream) const;

    bool supportsMultimem() const;

    // Logging output
    std::string getLoggingString() const;

protected:
    // Pure virtual launch function that must be implemented by derived classes
    virtual void launchKernel(ncclWindow_t inWindow, ncclWindow_t outWindow, void const* const residual,
        ncclWindow_t residualOutWindow, void const* const weight, void const* const bias, ncclDevComm devComm,
        float const eps, cudaStream_t stream) const
        = 0;
};

// Kernel launch information helper class
template <typename T>
class TypedLaunchConfig : public LaunchConfig
{
private:
    nvinfer1::DataType mType;

    // Private templated helper function to get kernel pointer for specific unroll factor
    template <int Nunroll>
    void* getKernelPtrForUnroll() const;

    // Private helper function to get kernel pointer for any unroll factor
    void* getKernelPtrForUnrollFactor(int unrollFactor) const;

    // Private helper function to launch kernel for any unroll factor
    void launchKernelForUnrollFactor(ncclWindow_t inWindow, ncclWindow_t outWindow, void const* const residual,
        ncclWindow_t residualOutWindow, void const* const weight, void const* const bias, ncclDevComm devComm,
        float const eps, cudaStream_t stream) const;

    // Private templated helper function to launch kernel for specific unroll factor
    template <int Nunroll>
    void launchKernelForUnrollImpl(ncclWindow_t inWindow, ncclWindow_t outWindow, void const* const residual,
        ncclWindow_t residualOutWindow, void const* const weight, void const* const bias, ncclDevComm devComm,
        float const eps, cudaStream_t stream) const;

public:
    using TN = typename VectorType<T>::type;
    constexpr static int elementsPerVector = sizeof(TN) / sizeof(T);

public:
    virtual int getElementsPerVector() const
    {
        return this->elementsPerVector;
    }

    virtual bool isValidConfig(int threadsPerBlock, int unrollFactor) const override;

    // Launch function that handles all the type-specific logic internally
    virtual void launchKernel(ncclWindow_t inWindow, ncclWindow_t outWindow, void const* const residual,
        ncclWindow_t residualOutWindow, void const* const weight, void const* const bias, ncclDevComm devComm,
        float const eps, cudaStream_t stream) const override;

    // Constructor with dynamic block size calculation
    TypedLaunchConfig(int const hidden_dim, int const num_tokens, int const rank, int const nRanks, bool useResidual,
        bool useBias, int const num_sms = -1);

    nvinfer1::DataType getDataType() const
    {
        return tensorrt_llm::runtime::TRTDataType<T>::value;
    }

    virtual bool getValid() const
    {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
        return this->valid;
#else
        TLLM_LOG_WARNING("NCCL device kernels not available (NCCL version < 2.28). LaunchConfig will be invalid.");
        return false;
#endif
    }
};

std::shared_ptr<LaunchConfig> makeLaunchConfig(nvinfer1::DataType dataType, int const hidden_dim, int const num_tokens,
    int const rank, int const nRanks, bool useResidual, bool useBias, int const num_sms = -1);

} // namespace tensorrt_llm::kernels::nccl_device

#endif // TRTLLM_NCCL_DEVICE_CONFIG_H
