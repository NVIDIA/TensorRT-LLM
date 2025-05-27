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

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>

#include "ada_blockwise_gemm_kernel.cuh"

#define CUTLASS_HOST_TRACE(x)                                                                                          \
    {                                                                                                                  \
        std::cout << __FILE__ << ":" << __LINE__ << "  " << x << std::endl;                                            \
    }

namespace ada_blockwise_gemm
{

template <typename GemmKernel>
CUTLASS_GLOBAL void run_global(typename GemmKernel::Params params)
{
    // Dynamic shared memory base pointer
    extern __shared__ int SharedStorageBase[];
    // Declare pointer to dynamic shared memory.
    typename GemmKernel::SharedStorage* shared_storage
        = reinterpret_cast<typename GemmKernel::SharedStorage*>(SharedStorageBase);

    GemmKernel::invoke(params, *shared_storage);
}

using namespace cutlass;

template <typename KT>
struct AdaBlockwiseGemm
{

    using GemmKernel = AdaBlockwiseGemmKernel<KT>;

    static constexpr int kSmemSize = GemmKernel::kSmemSize;
    static constexpr int kThreadCount = GemmKernel::kThreadCount;

    /// Kernel parameters object
    typename GemmKernel::Params params_;

    AdaBlockwiseGemm()
        : params_()
    {
    }

    using Arguments = typename GemmKernel::Arguments;

    /// Computes the maximum number of active blocks per multiprocessor
    static int maximum_active_blocks(int smem_capacity = -1)
    {

        CUTLASS_TRACE_HOST("AdaBlockwiseGemmKernel::maximum_active_blocks()");

        CUTLASS_TRACE_HOST("  kSmemSize: " << kSmemSize << " bytes");

        cudaError_t result;
        if (kSmemSize > (48 << 10))
        {
            result
                = cudaFuncSetAttribute(run_global<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);

            if (result != cudaSuccess)
            {
                // Call cudaGetLastError() to clear the error bit
                result = cudaGetLastError();
                CUTLASS_HOST_TRACE("  cudaFuncSetAttribute() returned error " << cudaGetErrorString(result));
                return -1;
            }
        }

        int max_active_blocks = -1;
        result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks, run_global<GemmKernel>, kThreadCount, kSmemSize);

        if (result != cudaSuccess)
        {
            // Call cudaGetLastError() to clear the error bit
            result = cudaGetLastError();
            CUTLASS_HOST_TRACE(
                "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned "
                "error "
                << cudaGetErrorString(result));
            return -1;
        }

        CUTLASS_HOST_TRACE("  max_active_blocks: " << max_active_blocks);
        return max_active_blocks;
    }

    Status can_implement(Arguments const& args)
    {
        if (kSmemSize > (48 << 10))
        {
            cudaError_t result
                = cudaFuncSetAttribute(run_global<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);

            if (result != cudaSuccess)
            {
                // Call cudaGetLastError() to clear the error bit
                result = cudaGetLastError();
                CUTLASS_HOST_TRACE("  cudaFuncSetAttribute() returned error " << cudaGetErrorString(result));
                return Status::kInvalid;
            }
        }

        if (args.problem_size.n() % KT::kTileN != 0)
        {
            CUTLASS_HOST_TRACE("  n:" << args.problem_size.n() << " % kTileN:" << KT::kTileN << " != 0");
            return Status::kInvalid;
        }

        if (args.problem_size.k() % KT::kTileK != 0)
        {
            CUTLASS_HOST_TRACE("  k:" << args.problem_size.k() << " % kTileK:" << KT::kTileK << " != 0");
            return Status::kInvalid;
        }

        return Status::kSuccess;
    }

    Status initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr)
    {

        params_ = GemmKernel::to_underlying_arguments(args);

        return Status::kSuccess;
    }

    Status run(cudaStream_t stream = nullptr)
    {

        // Configure grid and block dimensions

        dim3 grid = GemmKernel::get_grid_shape(params_.problem_size);
        dim3 block = GemmKernel::get_block_shape();

        // Launch kernel
        run_global<GemmKernel><<<grid, block, kSmemSize, stream>>>(params_);

        // Query for errors
        cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
        {
            CUTLASS_HOST_TRACE("  grid launch failed with error " << cudaGetErrorString(result));
            return Status::kErrorInternal;
        }

        return Status::kSuccess;
    }
};

} // namespace ada_blockwise_gemm
