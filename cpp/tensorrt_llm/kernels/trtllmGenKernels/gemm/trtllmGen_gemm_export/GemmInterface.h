/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <numeric>
#include <optional>
#include <unordered_map>

#include "GemmOptions.h"
#include "KernelParams.h"
#include "trtllm/gen/CudaKernelLauncher.h"

#ifdef TLLM_GEN_EXPORT_INTERFACE
#ifdef TLLM_GEN_EXPORT_FLASHINFER
#include "flashinferMetaInfo.h"
#else
#include "KernelMetaInfo.h"
#endif // TLLM_GEN_EXPORT_FLASHINFER
#endif // TLLM_GEN_EXPORT_INTERFACE

namespace gemm
{

namespace gemm
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GemmData
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmData
{
    struct ProblemDimensions
    {
        // The M dimension.
        // It is the total number of tokens if A is the activation matrix.
        // It is the total number of output channels if A is the weight matrix.
        // ValidM/N/K by default assumes to be full range of M/N/K respectively. If we pad M/N/K due to
        // alignment of other constraints, then we can specify ValidM/N/K to indicate the valid range.
        int32_t mM{0};
        int32_t mValidM{0};
        // The N dimension.
        // It is the total number of tokens if B is the activation matrix.
        // It is the total number of output channels if B is the weight matrix.
        int32_t mN{0};
        int32_t mValidN{0};
        // The K dimension. It is the hidden dimension of the input matrices.
        int32_t mK{0};
        int32_t mValidK{0};
        // The rank id of the current device in the multi-gpu space.
        int32_t mRank{0};
        // The number of devices in tensor-parallel group.
        int32_t mWorldSize{0};
    };

    struct InputBuffers
    {
        // The matrix A. The data type is controlled by options.mDtypeA.
        //
        // If S is the sparsity ratio (1 for dense, 2 for sparse):
        // When layoutA is MatrixLayout::MajorK, the shape is [M, K / S].
        // When LayoutA is MatrixLayout::MajorMn, the shape is [K, M] (sparsity not supported)
        // When LayoutA is MatrixLayout::BlockMajorK, the shape is [K / S / blockK, M, blockK] where
        // blockK is 128B.
        // The rightmost dimension is contiguous in memory.
        void const* mPtrA{nullptr};

        // The block scaling factors to dequantize A.
        //
        // If DeepSeek FP8 recipe is used:
        //    If transposeMmaOutput is false, shape is [K / 128, M].
        //    Otherwise, shape is [M / 128, K / 128].
        //  The rightmost dimension is contiguous in memory.
        //
        // If DeepSeek FP8 recipe is not used, but for MxFp{4,8}, MxInt4 and NvFp4 formats:
        //  If the layout is R128c4,
        //    M must be a multiple of 128.
        //    K must be a multiple of 4 * P, where P is the scaling block size.
        //    The "logical" shape is: [M, K / P].
        //    The R128c4 layout is: [M / 128, K / P / 4, 512].
        //    The shape we use for TMA is: [M / 128, K / P / 4, 2, 256].
        //  Dtype is E4m3 for NvFp4, UE8m0 for MxFp{4,8} formats, Bfloat16 for MxInt4.
        //
        // Otherwise should be set to nullptr.
        void const* mPtrSfA{nullptr};

        // The per-token scaling factors from scale A.
        //
        // This is used for either:
        //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is
        //   Dtype::Float32
        //   * One-sided per-token scaling. The dtype is Dtype::Bfloat16 by default.
        //
        // The shape is [M]
        void const* mPtrPerTokenSfA{nullptr};

        // The sparsity information of A, if structured sparsity is used.
        //
        // When sparsityA is Any_2_4:
        //     2 elements are non-zero in any chunk of 4 elements.
        //     A 4-bit index indicates the position of the non-zero elements.
        //     The shape in Uint8 is: [M, K / 8] (two 4-bit indices packed into one UInt8)
        //
        // When sparsityA is Pairwise_4_8:
        //     4 elements are non-zero in any chunk of 8 elements.
        //     The zero and non-zero elements are grouped in pairs.
        //     A 4-bit index indicates the position of the non-zero pairs.
        //     The shape in Uint8 is: [M, K / 16] (two 4-bit indices packed into one UInt8)
        //
        // If sparsityA is Dense, this should be set to nullptr.
        void const* mPtrSparsityInfoA{nullptr};

        // The matrix B. The data type is controlled by options.mDtypeB.
        //
        // When layoutB is MatrixLayout::MajorK, the shape is [N, K].
        // When layoutB is MatrixLayout::MajorMn, the shape is [K, N].
        // When layoutB is MatrixLayout::BlockMajorK, the shape is [K / blockK, N, blockK] where blockK
        // is 128B.
        // The rightmost dimension is contiguous in memory.
        void const* mPtrB{nullptr};

        // The scaling factors to dequantize B.
        //
        // If DeepSeek FP8 recipe is used:
        //    If transposeMmaOutput is false, shape is [N / 128, K / 128].
        //    Otherwise, shape is [K / 128, N].
        //    The rightmost dimension is contiguous in memory.
        //
        // If DeepSeek FP8 recipe is not used, but for MxFp{4,8} and NvFp4 formats:
        //  If the layout is R128c4,
        //     N must be a multiple of 128.
        //     K must be a multiple of 64.
        //     The R128c4 layout is: [N / 128, K / P / 4, 512], where P is the scaling block size.
        //     The shape we use for TMA is: [N / 128, K / P / 4, 2, 256]
        //
        //  If the layout is R8c4,
        //     N must be a multiple of 8.
        //     K must be a multiple of 64.
        //     The R8c4 layout is: [N / 8, K / P / 4, 32], where P is the scaling block size.
        //     The shape we use for TMA is: [N / 8, K / P / 4 / repeats, repeats * 32]
        //     where repeats = min(tileK / P / 4, 8)
        //
        // Dtype is E4m3 for NvFp4, UE8m0 for MxFp{4,8} formats.
        //
        // Otherwise should be set to nullptr.
        void const* mPtrSfB{nullptr};

        // The per-token scaling factors from scale B.
        //
        // This is used for either:
        //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is
        //   Dtype::Float32
        //   * One-sided per-token scaling. The dtype is Dtype::Bfloat16 by default.
        //
        // The shape is [N]
        void const* mPtrPerTokenSfB{nullptr};

        // The bias applied after the GEMM.
        // The bias is applied before applying the global scaling factor. I.e.
        // C' = (A * B + bias') * scaleC
        // scaleC = dequantA * dequantB * quantC
        // Thus, the bias' = bias / (dequantA * dequantB), where the bias is the original bias.
        //
        // if BiasType is N, the shape is [N].
        // The bias is broadcasted along the M dimension.
        //
        // if BiasType is M, the shape is [M].
        // The bias is broadcasted along the N dimension.
        //
        // The dtype is float32.
        void const* mPtrBias{nullptr};

        // The output tensor scaling factor for Fp8 (not DeepSeek FP8) and NvFp4 quantization.
        // TensorRT-LLM API requires a scaling factor on the device.
        // scaleC = dequantA * dequantB * quantC,
        // where dequantA is global dequantization scaling factor of A
        //    if dtypeA is FP8, it transforms the range from [-448, 448] to [-amaxA, amaxA]
        //    if dtypeA is NvFp4, it transforms the range from [-448 * 6, 448 * 6] to [-amaxA, amaxA],
        //    otherwise it is 1.
        // dequantB is defined similarly to dequantA.
        // quantC is the quantization scaling factor of C.
        //    if dtypeC is FP8, it transforms the range from [-amaxC, amaxC] to [-448, 448]
        //    if dtypeC is NvFp4, it transforms the range from [-amaxC, amaxC] to [-448 * 6, 448 * 6],
        //    otherwise it is 1.
        // Shape is [1].
        void* mPtrScaleC{nullptr};

        // The pre-activation scaling factor (typically dequantA * dequantB) for non-linear activation.
        // Only used when non-linear activation is applied (e.g., GELU, Relu2, Silu).
        // When used, scaleC should be quantScaleC only, and this scale is applied before the
        // activation. Shape is [1].
        void* mPtrScaleAct{nullptr};
    };

    struct OutputBuffers
    {
        // The output matrix C. The data type is controlled by options.mDtypeC.
        //
        // When transposeMmaOutput is true, the shape is [N, M].
        // Otherwise, the shape is [M, N].
        // Elements in a given row are stored contiguously in memory (row-major).
        void* mPtrC{nullptr};

        // Pointer for output with multicast mapping. It is used by the "reduce" op (LDGMC.ADD) of the
        // two-shot reduce-scatter phase. Otherwise, it should be set to nullptr.
        // The shape is [M, N] and the dtype is float.
        void* mPtrMultiMemC{nullptr};

        // The scaling factors calculated when quantizing C, for MxFp{4,8} and NvFp4 formats, also
        // used for the DeepSeek FP8 recipe.
        //
        // For DeepSeek FP8 recipe:
        //    If transposeMmaOutput is false, shape is [N / 128, M].
        //    Otherwise, shape is [M / 128, N].
        //    The rightmost dimension is contiguous in memory.
        //
        // For MxFp{4,8} and NvFp4 formats:
        //    If transposeMmaOutput is false, shape is [M, N / P], where P is the scaling block size.
        //    Otherwise, shape is [N, M / P].
        //    The layout is controlled by options.mSfLayoutC (either R128c4 or R8c4).
        //    The layout (R128c4 and R8c4) is the same as explained in mPtrSfB.
        //
        // Otherwise should be set to nullptr.
        void* mPtrSfC{nullptr};
    };

    struct AllReduceBuffers
    {
        // The barriers in global memory.
        //
        // The kernel arrives at (with release ordering) the multicast mapping of the barrier to
        // broadcast amongst peer devices. It then waits (with acquire ordering) for the unicast mapping
        // of the barrier.
        //
        // Flags in global memory that sync on "entrance" of reduce-scatter phase in two-shot
        // all-reduce. The shape is [numTilesM * numTilesN] and the dtype is uint32_t. The pointer to
        // the unicast memory created with IpcNvlsHandle. Must be set to 0 before the kernel launch.
        void* mPtrTileBars{nullptr};

        // The shape is [numTilesM * numTilesN] and the dtype is uint32_t.
        // The pointer to the multicast memory created with IpcNvlsHandle.
        void* mPtrMultiMemTileBars{nullptr};

        // Flags in global memory that sync on "exit" after the all-reduce finishes.
        // The shape is [numTilesM * numTilesN] and the dtype is uint32_t.
        // The pointer to the unicast memory created with IpcNvlsHandle.
        // Must be set to 0 before the kernel launch.
        void* mPtrCompletionBars{nullptr};

        // The shape is [numTilesM * numTilesN] and the dtype is uint32_t.
        // The pointer to the multicast memory created with IpcNvlsHandle.
        void* mPtrMultiMemCompletionBars{nullptr};
    };

    ProblemDimensions mProblemDimensions;
    InputBuffers mInputBuffers;
    OutputBuffers mOutputBuffers;
    AllReduceBuffers mAllReduceBuffers;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GemmInterface
//
////////////////////////////////////////////////////////////////////////////////////////////////////

class GemmInterface
{
public:
    using ModuleCache = std::unordered_map<std::string, std::tuple<CUmodule, CUfunction>>;

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    GemmInterface(int32_t rankId = 0, bool exportsCubin = false, int32_t numRotations = 1)
        : mRankId{rankId}
        , mExportsCubin{exportsCubin}
        , mNumRotations{numRotations}
    {
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef TLLM_GEN_EXPORT_INTERFACE
    // Generates and compiles the kernel using either nvcc or nvrtc.
    GemmConfig generateAndCompileKernel(GemmConfig const& gemmConfig) const;
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Returns the size of the workspace buffers in bytes
    size_t getWorkspaceSizeInBytes(GemmConfig const& config, GemmData const& data) const
    {
        auto workspaceSizes = getWorkspaceSizesInBytes(config, data);
        auto size = std::accumulate(workspaceSizes.begin(), workspaceSizes.end(), 0);
        // Additional 1023 bytes to align the pointer to 1024
        return size > 0 ? size + 1023 : 0;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Returns the list of all available cubin configurations
    GemmConfig const* getGemmConfigs() const
    {
#ifdef TLLM_GEN_EXPORT_INTERFACE
        return tensorrt_llm::kernels::tllmGenGemmList;
#else
        return nullptr;
#endif
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Returns the number of available cubin configurations
    size_t getNumGemmConfigs() const
    {
#ifdef TLLM_GEN_EXPORT_INTERFACE
        return sizeof(tensorrt_llm::kernels::tllmGenGemmList) / sizeof(tensorrt_llm::kernels::tllmGenGemmList[0]);
#else
        return 0;
#endif
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Creates GemmOptions from kernel and data.
    GemmOptions getOptionsFromConfigAndData(GemmConfig const& config, GemmData const& data) const
    {
        // Create options from config and data.
        GemmOptions options;
        options = config.mOptions;
        options.mM = data.mProblemDimensions.mM;
        options.mN = data.mProblemDimensions.mN;
        options.mK = data.mProblemDimensions.mK;
        options.mValidM = data.mProblemDimensions.mValidM;
        options.mValidN = data.mProblemDimensions.mValidN;
        options.mValidK = data.mProblemDimensions.mValidK;
        return options;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Returns true if the configuration of the cubin can be executed for the given params.
    bool isValidConfig(GemmConfig const& config, GemmData const& data) const
    {
        // Get options from config and data.
        auto options = getOptionsFromConfigAndData(config, data);

        if (usesFusedBiasReorder(options.mFusedBiasShuffleMode))
        {
            return false;
        }

        // Check options without modifications.
        return checkAndUpdateGemmOptions(options, config.mSm, data.mProblemDimensions.mWorldSize,
            /* updateOptions */ false);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // If config.mData is specified, it launches the cubin from the provided config.
    // Otherwise, it generates and compiles the kernel using either nvcc or nvrtc.
    // Launch the cubin from the provided config.
    // It calls all necessary memsets for internal buffers.
    // Provided config must be validated with isValidConfig before the call.
    int32_t run(GemmConfig const& config, void* workspace, GemmData const& data, void* cudaStream,
        int32_t multiProcessorCount, bool usePdl = true,
        std::optional<std::reference_wrapper<ModuleCache>> moduleCache = std::nullopt) const
    {
        // Get options from config and data.
        auto options = getOptionsFromConfigAndData(config, data);

        auto workspaceSizes = getWorkspaceSizesInBytes(config, data);
        void* dSplitKSlices{nullptr};
        void* dPtrSplitKCompletionBars{nullptr};

        // Set the completion barriers to 0 if needed.
        if (doesSplitKUseGmem(options.mSplitK))
        {
            dSplitKSlices = alignPtr(reinterpret_cast<char*>(workspace), 1024);
            dPtrSplitKCompletionBars = reinterpret_cast<uint32_t*>(
                alignPtr(reinterpret_cast<char*>(dSplitKSlices) + workspaceSizes[0], 1024));
            auto err = cudaMemsetAsync(
                (void*) dPtrSplitKCompletionBars, 0x00, workspaceSizes[1], reinterpret_cast<cudaStream_t>(cudaStream));
            if (err != cudaSuccess)
            {
                return 1;
            }
        }

        // Determine if the scheduler requires a fixed grid dimension.
        bool const isFixedGridDim = (options.mTileScheduler == gemm::TileScheduler::StaticPersistent
            || options.mTileScheduler == gemm::TileScheduler::PersistentSm90);
        // Get the number of tiles and number of CTAs for Z dimension.
        auto [gridDimX, gridDimY, gridDimZ] = isFixedGridDim
            ? getFixedGridSize(
                  options.mClusterDimX, options.mClusterDimY, options.mNumSlicesForSplitK, multiProcessorCount)
            : getGridSize(options.mM, options.mN, options.mTileM, options.mTileN, options.mClusterDimX,
                  options.mClusterDimY, options.mNumSlicesForSplitK);

        // Create kernel params.
        auto kernelParams = gemm::KernelParamsSetup::setKernelParams(options, data.mInputBuffers.mPtrA,
            data.mInputBuffers.mPtrSfA, data.mInputBuffers.mPtrPerTokenSfA, data.mInputBuffers.mPtrB,
            data.mInputBuffers.mPtrSfB, data.mInputBuffers.mPtrPerTokenSfB, data.mInputBuffers.mPtrSparsityInfoA,
            data.mInputBuffers.mPtrBias, data.mOutputBuffers.mPtrC, data.mOutputBuffers.mPtrSfC,
            data.mOutputBuffers.mPtrMultiMemC, (float*) data.mInputBuffers.mPtrScaleC,
            (float*) data.mInputBuffers.mPtrScaleAct, dSplitKSlices, data.mAllReduceBuffers.mPtrTileBars,
            data.mAllReduceBuffers.mPtrMultiMemTileBars, data.mAllReduceBuffers.mPtrCompletionBars,
            data.mAllReduceBuffers.mPtrMultiMemCompletionBars, dPtrSplitKCompletionBars,
            /* dPtrNumNonExitingCtas */ nullptr, data.mProblemDimensions.mRank, data.mProblemDimensions.mWorldSize);
        // The size of the grid.
        std::vector<int32_t> grid{gridDimX, gridDimY, gridDimZ};

        // When split-k is enabled and to guarantee the forward progress, we must ensure that the number
        // of tiles is less than number of SMs. This way, at least one CTA in the grid can make forward.
        if (doesSplitKUseGmem(options.mSplitK))
        {
            if (grid[0] * grid[1] >= multiProcessorCount)
            {
                // The number of MN tiles in Split-K (grid[0] * grid[1]) must be less than the number of
                // SMs.
                return 2;
            }
        }

        GemmConfig gemmConfig = config;

#ifndef TLLM_GEN_EXPORT_INTERFACE
        // Generate and compile the kernel if data is not provided.
        if (config.mData == nullptr)
        {
            gemmConfig = generateAndCompileKernel(gemmConfig);
            TLLM_CHECK_ERROR(gemmConfig.mCudaRunner != nullptr, "CudaRunner is not set");
            gemmConfig.mCudaRunner->run((void*) &kernelParams, (void*) cudaStream, grid,
                /*cluster*/ {},
                /*instanceId*/ gemmConfig.mInstanceIdx);
            return 0;
        }
#endif

        // Load from cubin if data is provided.
        CUmodule cuModule;
        CUfunction cuFunction;

        if (moduleCache.has_value())
        {
            ModuleCache& moduleCacheRef = moduleCache.value().get();

            // Modules are associated with a specific context, so the context is included in the key
            CUcontext ctx;
            unsigned long long ctxId;
            cuCtxGetCurrent(&ctx);
            cuCtxGetId(ctx, &ctxId);

            // Reinterpret the ctxId as a string to avoid needing a custom hash or converting it to a
            // string in decimal representation.
            std::string const ctxName
                = std::string(reinterpret_cast<char*>(&ctxId), sizeof(unsigned long long) / sizeof(char));
            std::string const funcName = std::string(gemmConfig.mFunctionName);
            auto const moduleKey = ctxName + funcName;
            auto module = moduleCacheRef.find(moduleKey);

            // Use cache if module is found, otherwise load and insert into cache
            if (module != moduleCacheRef.end())
            {
                cuFunction = std::get<1>(module->second);
            }
            else
            {
                loadCubinData(&cuModule, gemmConfig);
                cuModuleGetFunction(&cuFunction, cuModule, gemmConfig.mFunctionName);
                moduleCacheRef.insert(std::make_pair(moduleKey, std::make_tuple(cuModule, cuFunction)));
            }
        }
        else
        {
            loadCubinData(&cuModule, gemmConfig);
            cuModuleGetFunction(&cuFunction, cuModule, gemmConfig.mFunctionName);
        }

        // Prepare the grid/block.
        dim3 block3{
            static_cast<uint32_t>(gemmConfig.mNumThreadsPerCTA), static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
        dim3 grid3{(grid.size() > 0 ? static_cast<uint32_t>(grid[0]) : 1u),
            (grid.size() > 1 ? static_cast<uint32_t>(grid[1]) : 1u),
            (grid.size() > 2 ? static_cast<uint32_t>(grid[2]) : 1u)};
        // Prepare the cluster size.
        dim3 cluster3{static_cast<uint32_t>(options.mClusterDimX), static_cast<uint32_t>(options.mClusterDimY),
            static_cast<uint32_t>(options.mClusterDimZ)};

        // Whether PDL can safely be enabled
        bool const pdlSafe = gemmConfig.mOptions.mGridWaitForPrimaryEarlyExit
            || gemmConfig.mOptions.mGridWaitForPrimaryA || gemmConfig.mOptions.mGridWaitForPrimaryB;

        // Run the kernel.
        auto result = trtllm::gen::launchKernel((void*) &kernelParams, cudaStream, gemmConfig.mSharedMemSize,
            cuFunction, block3, grid3, cluster3, usePdl && pdlSafe);
        // If a module cache has not been given, unload the module to avoid leaking
        if (!moduleCache.has_value())
        {
            cuModuleUnload(cuModule);
        }
        if (result != CUDA_SUCCESS)
        {
            return result;
        }

        return 0;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Initializes the buffers before the world sync. Must be called before run.
    int32_t runInitBeforeWorldSync(GemmConfig const& config, GemmData const& data, void* cudaStream) const
    {
        if (data.mProblemDimensions.mWorldSize > 1)
        {
            // Get options from config and data.
            auto options = getOptionsFromConfigAndData(config, data);
            if (options.mAllReduceAlgo == gemm::AllReduceAlgo::OneShot)
            {
                // The size of each element of C in bits.
                int64_t const numBitsPerEltC = options.mAllReduceAlgo == gemm::AllReduceAlgo::TwoShot
                    ? tg::dtypeGetNumBits(options.mDtypeAcc)
                    : tg::dtypeGetNumBits(options.mDtypeC);
                // The number of bytes for C.
                int64_t const numBytesC
                    = data.mProblemDimensions.mM * data.mProblemDimensions.mN * numBitsPerEltC / /*bits*/ 8;
                // Reset the output buffer as one-shot uses UTMAREDG at multicast memory for reduction.
                auto err = cudaMemsetAsync(
                    data.mOutputBuffers.mPtrC, 0x00, numBytesC, reinterpret_cast<cudaStream_t>(cudaStream));
                if (err != cudaSuccess)
                {
                    return 1;
                }
            }

            // Get the number of tiles and number of CTAs for Z dimension.
            // mTileM/mTileN are already correctly set for both transpose modes.
            auto [numTilesM, numTilesN, gridDimZ] = getGridSize(options.mM, options.mN, options.mTileM, options.mTileN,
                options.mClusterDimX, options.mClusterDimY, options.mNumSlicesForSplitK);
            // The number of bytes for the tile barriers.
            int32_t numBytesTileBars = numTilesM * numTilesN * sizeof(uint32_t);
            // Sanitize system barriers.
            auto err = cudaMemsetAsync((void*) data.mAllReduceBuffers.mPtrTileBars, 0x00, numBytesTileBars,
                reinterpret_cast<cudaStream_t>(cudaStream));
            if (err != cudaSuccess)
            {
                return 2;
            }
            err = cudaMemsetAsync((void*) data.mAllReduceBuffers.mPtrCompletionBars, 0x00, numBytesTileBars,
                reinterpret_cast<cudaStream_t>(cudaStream));
            if (err != cudaSuccess)
            {
                return 3;
            }
        }
        return 0;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

private:
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Aligns the pointer to the alignment
    template <typename Dtype>
    inline Dtype* alignPtr(Dtype* ptr, int64_t alignment) const
    {
        assert((alignment & (alignment - 1)) == 0 && "Alignment must be a power of 2");
        return reinterpret_cast<Dtype*>((reinterpret_cast<uintptr_t>(ptr) + alignment - 1) & ~(alignment - 1));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Returns the number of tiles and number of CTAs for Z dimension.
    std::tuple<int32_t, int32_t, int32_t> getFixedGridSize(
        int32_t clusterDimX, int32_t clusterDimY, int32_t numSlicesForSplitK, int32_t multiProcessorCount) const
    {

        assert(multiProcessorCount > 0 && "multiProcessorCount must be provided when using StaticPersistent scheduler");
        // The cluster size spanned in the XY dimension.
        auto clusterSizeXy = clusterDimX * clusterDimY;
        // The maximum number of CTAs a GPU can run across the XY dimension.
        auto numCtasXy = multiProcessorCount / numSlicesForSplitK;
        // Round down to the nearest multiple of the cluster size.
        numCtasXy = (numCtasXy / clusterSizeXy) * clusterSizeXy;

        return std::make_tuple(numCtasXy, 1, numSlicesForSplitK);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Returns the number of tiles and number of CTAs for Z dimension.
    std::tuple<int32_t, int32_t, int32_t> getGridSize(int32_t M, int32_t N, int32_t tileM, int32_t tileN,
        int32_t clusterDimX, int32_t clusterDimY, int32_t numSlicesForSplitK) const
    {
        // The number of tiles in the M dimension.
        auto numTilesM = gemm::divUpMul(gemm::divUp(M, tileM), clusterDimX);
        // The number of tiles in the N dimension.
        auto numTilesN = gemm::divUpMul(gemm::divUp(N, tileN), clusterDimY);
        return std::make_tuple(numTilesM, numTilesN, numSlicesForSplitK);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Returns the size of the workspace buffers in bytes
    std::vector<size_t> getWorkspaceSizesInBytes(GemmConfig const& config, GemmData const& data) const
    {
        // Get options from config.
        auto& options = config.mOptions;

        // Get the number of tiles and cluster dimension Z.
        // mTileM/mTileN are already correctly set for both transpose modes.
        auto [numTilesM, numTilesN, gridDimZ] = getGridSize(data.mProblemDimensions.mM, data.mProblemDimensions.mN,
            options.mTileM, options.mTileN, options.mClusterDimX, options.mClusterDimY, options.mNumSlicesForSplitK);

        std::vector<size_t> workspaceSizes;

        int64_t numBytesSplitK{0}, numBytesSplitKBars{0};
        if (doesSplitKUseGmem(options.mSplitK))
        {

            // The number of elements for intermediate split-k buffer that contains K slices padded to
            // TileM/TileN sizes to avoid OOB accesses during the reduction.
            // FIXME: Split-K has excessive memory traffic when combined with slice-K.
            // Currently, data for all slice-K slices is sent, even though the slice-K reduction
            // has already been performed.
            // This should be optimized to send data for only one reduced slice.
            auto numEltsSplitK = options.mNumSlicesForSplitK * numTilesM * numTilesN * options.mTileM * options.mTileN;

            // The number of bytes for intermediate split-k buffer that contains K slices.
            numBytesSplitK = numEltsSplitK * tg::dtypeGetNumBits(tg::Dtype::Fp32) / /* bits */ 8;
            // The number of bytes for the split-k completion barriers.
            numBytesSplitKBars = numTilesM * numTilesN * sizeof(uint32_t);
            // Two epilogue warps do GMEM split-k in DS GEMM.
            if (options.mUseDeepSeekFp8)
            {
                numBytesSplitKBars *= 2;
            }

            // TODO: do we need to pad to 1024?
            workspaceSizes.push_back(getSizePaddedToAlignment(numBytesSplitK, 1024));
            workspaceSizes.push_back(getSizePaddedToAlignment(numBytesSplitKBars, 1024));
        }

        return workspaceSizes;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Returns the size padded to the alignment
    size_t getSizePaddedToAlignment(size_t size, size_t alignment) const
    {
        assert((alignment & (alignment - 1)) == 0);
        return (size + alignment - 1) & ~(alignment - 1);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

private:
    // The rank id of the current device in the multi-gpu space.
    int32_t mRankId;
    // Whether to export the cubin file.
    bool mExportsCubin;
    // The number of rotations.
    int32_t mNumRotations;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
