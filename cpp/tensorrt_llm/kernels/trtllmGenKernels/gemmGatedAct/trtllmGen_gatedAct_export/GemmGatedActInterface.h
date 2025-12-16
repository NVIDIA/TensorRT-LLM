/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
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

#include "GemmGatedActOptions.h"
#include "KernelParams.h"
#include "trtllm/gen/CudaKernelLauncher.h"

#ifdef TLLM_GEN_EXPORT_INTERFACE
#include "KernelMetaInfo.h"
#endif // TLLM_GEN_EXPORT_INTERFACE

namespace gemmGatedAct
{

namespace gemmGatedAct
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GemmGatedActData
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmGatedActData
{
    struct ProblemDimensions
    {
        // The M dimension.
        // It is the total number of tokens if A is the activation matrix.
        // It is the total number of output channels multiplied by 2 if A is the weight matrix.
        int32_t mM{0};
        // The N dimension.
        // It is the total number of tokens if B is the activation matrix.
        // It is the total number of output channels multiplied by 2 if B is the weight matrix.
        int32_t mN{0};
        // The K dimension. It is the hidden dimension of the input matrices.
        int32_t mK{0};
        // The rank id of the current device in the multi-gpu space.
        int32_t mRank{0};
        // The number of devices in tensor-parallel group.
        int32_t mWorldSize{0};
    };

    struct InputBuffers
    {
        // The matrix A. The data type is controlled by options.mDtypeA.
        //
        // When layoutA is MatrixLayout::MajorK, the shape is [M, K].
        // When LayoutA is MatrixLayout::MajorMn, the shape is [K, M].
        // When LayoutA is MatrixLayout::BlockMajorK, the shape is [K / blockK, M, blockK] where blockK
        // is 128B.
        // The rightmost dimension is contiguous in memory.
        void const* mPtrA{nullptr};

        // The block scaling factors to dequantize A.
        //
        // If DeepSeek FP8 recipe is used:
        //    If transposeMmaOutput is false, shape is [K / 128, M].
        //    Otherwise, shape is [M / 128, K / 128].
        //  The rightmost dimension is contiguous in memory.
        //
        // If DeepSeek FP8 recipe is not used, but for MxFp{4,8} and NvFp4 formats:
        //    The layout of scaling factors for A is always R128c4
        //    M must be a multiple of 128.
        //    K must be a multiple of 64.
        //    The "logical" shape is: [M, K / 16].
        //    The R128c4 layout is: [M / 128, K / 16 / 4, 512].
        //    The shape we use for TMA is: [M / 128, K / 16 / 4, 2, 256].
        //    Dtype is Dtype::E4m3.
        //
        // Otherwise should be set to nullptr.
        void const* mPtrSfA{nullptr};

        // The per-token scaling factors from scale A.
        //
        // This is used for either:
        //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is
        //   Dtype::Float32
        //   * When the routing scales are applied to the input activations (only when output is not
        //   transposed). The dtype is Dtype::Bfloat16
        //
        // The shape is [M]
        void const* mPtrPerTokenSfA{nullptr};

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
        //     The R128c4 layout is: [N / 128, K / 16 / 4, 512]
        //     The shape we use for TMA is: [N / 128, K / 16 / 4, 2, 256]
        //
        //  If the layout is R8c4,
        //     N must be a multiple of 8.
        //     K must be a multiple of 64.
        //     The R8c4 layout is: [N / 8, K / 16 / 4, 32]
        //     The shape we use for TMA is: [N / 8, K / 16 / 4 / repeats, repeats * 32]
        //     where repeats = min(tileK / 16 / 4, 8)
        //
        //  Dtype is Dtype::E4m3.
        //
        // Otherwise should be set to nullptr.
        void const* mPtrSfB{nullptr};

        // The per-token scaling factors from scale B.
        //
        // This is used for either:
        //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is
        //   Dtype::Float32
        //   * When the routing scales are applied to the input activations (only when output is
        //   transposed). The dtype is Dtype::Bfloat16
        //
        // The shape is [N]
        void const* mPtrPerTokenSfB{nullptr};

        // The bias applied after the GEMM and before the activation function.
        // The bias is applied before the global scaling factor. I.e.
        // C = act(A * B + bias') * scaleC
        // scaleC = dequantA * dequantB * quantC
        // Thus, the bias' = bias / (dequantA * dequantB), where the bias is the original bias.
        //
        // if BiasType is N, the shape is [N]
        // The bias is broadcasted along the M dimension.
        //
        // if BiasType is M, the shape is [M]
        // The bias is broadcasted along the N dimension.
        //
        // The dtype is float32.
        void const* mPtrBias{nullptr};

        // The output tensor scaling factor for MxFp{4,8}, Fp8, NvFp4 and DeepSeek FP8 quantization.
        // TensorRT LLM API requires a scaling factor on the device.
        // Shape is [1].
        void const* mPtrScaleC{nullptr};
        // The output gate scale for MxFp{4,8}, NvFp4 and DeepSeek FP8 quantization.
        // TensorRT LLM API requires a scaling factor on the device.
        // Shape is [1].
        void const* mPtrScaleGate{nullptr};
        // The alpha for SwiGlu or GeGlu.
        // Alpha is 1.f if nullptr.
        // Shape is [1].
        void const* mPtrGatedActAlpha{nullptr};
        // The beta for SwiGlu or GeGlu.
        // Beta is 0.f if nullptr.
        // Shape is [1].
        void const* mPtrGatedActBeta{nullptr};
        // The clamp limit before the activation.
        // Clamp limit is FLT_MAX if nullptr.
        // When the input is FP8 or NVFP4, the clamp has to be scaled by limit' = limit / dequantAb.
        // Shape is [1].
        //
        // The given clamp limit applies to the dequantized values, so the order of operations would
        // look something like this:
        //
        // x0 = x0 * dqAb
        // x0 = clamp(x0, none, limit)
        // x0 = x0 * sigmoid(alpha * x0)
        // x1 = dqAb * x1
        // x1 = clamp(x1, -limit, limit)
        // out = qC * (x1 + beta) * x0
        //
        // Given that the dqAb and qC are combined into scaleC, we can bring the dqAb into the clamp
        // limit and apply the clamping prior to dequantization:
        //
        // x0 = clamp(x0, none, limit / dqAb)
        // x0 = x0 * dqAb
        // x0 = x0 * sigmoid(alpha * x0)
        // x1 = clamp(x1, -limit / dqAb, limit / dqAb)
        // scaleC = dqAb * qC
        // beta' = beta / dqAb
        // out = scaleC * (x1 + beta') * x0
        //
        // Note this assumes that scaleAb == scaleGate which is true in TRT-LLM MoE use-case
        //
        void const* mPtrClampLimit{nullptr};
    };

    struct OutputBuffers
    {
        // The output matrix C. The data type is controlled by options.mDtypeC.
        //
        // When transposeMmaOutput is true, the shape is [N, M / 2].
        // Otherwise, the shape is [M, N / 2].
        // Elements in a given row are stored contiguously in memory (row-major).
        void* mPtrC{nullptr};

        // The scaling factors calculated when quantizing C, for MxFp{4,8} and NvFp4 formats, also
        // used for the DeepSeek FP8 recipe.
        //
        // For DeepSeek FP8 recipe:
        //    If transposeMmaOutput is false, shape is [N / 2 / 128, M].
        //    Otherwise, shape is [M / 2 / 128, N].
        //    The rightmost dimension is contiguous in memory.
        //
        // For MxFp{4,8} and NvFp4 formats:
        //    If transposeMmaOutput is false, shape is [M, N / 2 / 16].
        //    Otherwise, shape is [N, M / 2 / 16].
        //    The layout is controlled by options.mSfLayoutC (either R128c4 or R8c4).
        //    The layout (R128c4 and R8c4) is the same as explained in mPtrSfB.
        //
        // Otherwise should be set to nullptr.
        void* mPtrSfC{nullptr};
    };

    ProblemDimensions mProblemDimensions;
    InputBuffers mInputBuffers;
    OutputBuffers mOutputBuffers;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GemmGatedActInterface
//
////////////////////////////////////////////////////////////////////////////////////////////////////

class GemmGatedActInterface
{
public:
    using ModuleCache = std::unordered_map<std::string, std::tuple<CUmodule, CUfunction>>;

    GemmGatedActInterface() {}

    // Launch the cubin from the provided config. It calls all necessary memsets for internal buffers.
    // Provided config must be validated with isValidConfig before the call.
    int32_t run(GemmGatedActConfig const& config, void* workspace, GemmGatedActData const& data, void* cudaStream,
        int32_t multiProcessorCount, bool usePdl = true,
        std::optional<std::reference_wrapper<ModuleCache>> moduleCache = std::nullopt) const;

    // Initializes the buffers before the world sync. Must be called before run.
    int32_t runInitBeforeWorldSync(
        GemmGatedActConfig const& config, GemmGatedActData const& data, void* cudaStream) const;

    // Returns the size of the workspace buffers in bytes
    size_t getWorkspaceSizeInBytes(GemmGatedActConfig const& config, GemmGatedActData const& data) const;

    // Returns the list of all available cubin configurations
    GemmGatedActConfig const* getGemmConfigs() const;

    // Returns the number of available cubin configurations
    size_t getNumGemmConfigs() const;

    // Returns true if the configuration of the cubin can be executed for the given params.
    bool isValidConfig(GemmGatedActConfig const& config, GemmGatedActData const& data) const;

private:
    // Aligns the pointer to the alignment
    template <typename Dtype>
    inline Dtype* alignPtr(Dtype* ptr, int64_t alignment) const;

    // Creates GemmGatedActOptions from kernel and data.
    GemmGatedActOptions getOptionsFromConfigAndData(
        GemmGatedActConfig const& config, GemmGatedActData const& data) const;

    // Returns the size of the workspace buffers in bytes
    std::vector<size_t> getWorkspaceSizesInBytes(GemmGatedActConfig const& config, GemmGatedActData const& data) const;

    // Returns the size padded to the alignment
    size_t getSizePaddedToAlignment(size_t size, size_t alignment) const;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
inline Dtype* GemmGatedActInterface::alignPtr(Dtype* ptr, int64_t alignment) const
{
    assert((alignment & (alignment - 1)) == 0 && "Alignment must be a power of 2");
    return reinterpret_cast<Dtype*>((reinterpret_cast<uintptr_t>(ptr) + alignment - 1) & ~(alignment - 1));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GemmGatedActConfig const* GemmGatedActInterface::getGemmConfigs() const
{
#ifdef TLLM_GEN_EXPORT_INTERFACE
    return tensorrt_llm::kernels::tllmGenGemmGatedActList;
#else
    return nullptr;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t GemmGatedActInterface::getNumGemmConfigs() const
{
#ifdef TLLM_GEN_EXPORT_INTERFACE
    return tensorrt_llm::kernels::tllmGenGemmGatedActListLen;
#else
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GemmGatedActOptions GemmGatedActInterface::getOptionsFromConfigAndData(
    GemmGatedActConfig const& config, GemmGatedActData const& data) const
{
    // Create options from config and data.
    GemmGatedActOptions options;
    options = config.mOptions;
    options.mM = data.mProblemDimensions.mM;
    options.mN = data.mProblemDimensions.mN;
    options.mK = data.mProblemDimensions.mK;
    return options;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t GemmGatedActInterface::getSizePaddedToAlignment(size_t size, size_t alignment) const
{
    assert((alignment & (alignment - 1)) == 0);
    return (size + alignment - 1) & ~(alignment - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t GemmGatedActInterface::getWorkspaceSizeInBytes(
    GemmGatedActConfig const& config, GemmGatedActData const& data) const
{
    auto workspaceSizes = getWorkspaceSizesInBytes(config, data);
    auto size = std::accumulate(workspaceSizes.begin(), workspaceSizes.end(), 0);
    // Additional 1023 bytes to align the pointer to 1024
    return size > 0 ? size + 1023 : 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<size_t> GemmGatedActInterface::getWorkspaceSizesInBytes(
    GemmGatedActConfig const& config, GemmGatedActData const& data) const
{
    // Get options from config.
    auto& options = config.mOptions;

    // The number of tiles in the M dimension.
    int32_t numTilesM = gemm::divUp(data.mProblemDimensions.mM, options.mTileM);
    // The number of tiles in the N dimension.
    int32_t numTilesN = gemm::divUp(data.mProblemDimensions.mN, options.mTileN);

    std::vector<size_t> workspaceSizes;

    int64_t numBytesRowMax{0}, numBytesRowMaxBars{0};
    if (options.mUseDeepSeekFp8)
    {

        // The number of bytes for intermediate row max results.
        // numElts = M * N
        // numDqSfsC = numElts / 128
        // ctasPerTileN128 = 2
        // numBytesRowMax = ctasPerTileN128 * numDqSfsC
        numBytesRowMax = 2 * options.mM * options.mN / 128 * sizeof(float);
        // The number of bytes for the row max completion barriers.
        numBytesRowMaxBars = numTilesM * numTilesN / 2 * sizeof(uint32_t);

        // TODO: do we need to pad to 1024?
        workspaceSizes.push_back(getSizePaddedToAlignment(numBytesRowMax, 1024));
        workspaceSizes.push_back(getSizePaddedToAlignment(numBytesRowMaxBars, 1024));
    }

    return workspaceSizes;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool GemmGatedActInterface::isValidConfig(GemmGatedActConfig const& config, GemmGatedActData const& data) const
{
    // Get options from config and data.
    auto options = getOptionsFromConfigAndData(config, data);

    // Is Blackwell?
    bool isBlackwell = gemm::isSmVersionBlackwell(config.mSm);

    // Check options without modifications.
    return checkAndUpdateGemmGatedActOptions(options, isBlackwell,
        /* updateOptions */ false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t GemmGatedActInterface::run(GemmGatedActConfig const& config, void* workspace, GemmGatedActData const& data,
    void* cudaStream, int32_t multiProcessorCount, bool usePdl,
    std::optional<std::reference_wrapper<ModuleCache>> moduleCache) const
{
    // Might be used.
    (void) usePdl;
    (void) moduleCache;
    // Get options from config and data.
    auto options = getOptionsFromConfigAndData(config, data);

    auto workspaceSizes = getWorkspaceSizesInBytes(config, data);
    void* dRowMax{nullptr};
    void* dRowMaxBars{nullptr};

    // Set the completion barriers to 0 if needed.
    if (options.mUseDeepSeekFp8)
    {
        dRowMax = alignPtr(reinterpret_cast<char*>(workspace), 1024);
        dRowMaxBars = reinterpret_cast<uint32_t*>(alignPtr(reinterpret_cast<char*>(dRowMax) + workspaceSizes[0], 1024));
        auto err
            = cudaMemsetAsync((void*) dRowMaxBars, 0x00, workspaceSizes[1], reinterpret_cast<cudaStream_t>(cudaStream));
        if (err != cudaSuccess)
        {
            return 1;
        }
    }

    // The number of tiles in the M dimension.
    int numTilesM = gemm::divUp(options.mM, options.mTileM);
    // The number of tiles in the N dimension.
    int numTilesN = gemm::divUp(options.mN, options.mTileN);

    // Create kernel params.
    auto kernelParams = gemmGatedAct::KernelParams::setKernelParams(options, data.mInputBuffers.mPtrA,
        data.mInputBuffers.mPtrSfA, data.mInputBuffers.mPtrPerTokenSfA, data.mInputBuffers.mPtrB,
        data.mInputBuffers.mPtrSfB, data.mInputBuffers.mPtrPerTokenSfB, data.mInputBuffers.mPtrBias,
        data.mOutputBuffers.mPtrC, reinterpret_cast<float const*>(data.mInputBuffers.mPtrScaleC),
        data.mOutputBuffers.mPtrSfC, reinterpret_cast<float const*>(data.mInputBuffers.mPtrScaleGate),
        reinterpret_cast<float const*>(data.mInputBuffers.mPtrClampLimit),
        reinterpret_cast<float const*>(data.mInputBuffers.mPtrGatedActAlpha),
        reinterpret_cast<float const*>(data.mInputBuffers.mPtrGatedActBeta), reinterpret_cast<float*>(dRowMax),
        reinterpret_cast<uint32_t*>(dRowMaxBars));

    // The size of the grid.
    std::vector<int32_t> grid{numTilesM, numTilesN, options.mNumSlicesForSplitK};

    // When split-k is enabled and to guarantee the forward progress, we must ensure that the number
    // of tiles is less than number of SMs. This way, at least one CTA in the grid can make forward.
    if (options.mUseDeepSeekFp8)
    {
        if (grid[0] * grid[1] >= multiProcessorCount)
        {
            // The number of MN tiles in Split-K (grid[0] * grid[1]) must be less than the number of SMs.
            return 2;
        }
    }

#ifdef TLLM_GEN_EXPORT_INTERFACE
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
        std::string const funcName = std::string(config.mFunctionName);
        auto const moduleKey = ctxName + funcName;
        auto module = moduleCacheRef.find(moduleKey);

        // Use cache if module is found, otherwise load and insert into cache
        if (module != moduleCacheRef.end())
        {
            cuFunction = std::get<1>(module->second);
        }
        else
        {
            cuModuleLoadData(&cuModule, config.mData);
            cuModuleGetFunction(&cuFunction, cuModule, config.mFunctionName);
            moduleCacheRef.insert(std::make_pair(moduleKey, std::make_tuple(cuModule, cuFunction)));
        }
    }
    else
    {
        cuModuleLoadData(&cuModule, config.mData);
        cuModuleGetFunction(&cuFunction, cuModule, config.mFunctionName);
    }

    // Prepare the grid/block.
    dim3 block3{static_cast<uint32_t>(config.mNumThreadsPerCTA), static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
    dim3 grid3{(grid.size() > 0 ? static_cast<uint32_t>(grid[0]) : 1u),
        (grid.size() > 1 ? static_cast<uint32_t>(grid[1]) : 1u),
        (grid.size() > 2 ? static_cast<uint32_t>(grid[2]) : 1u)};
    // Prepare the cluster size.
    dim3 cluster3{static_cast<uint32_t>(options.mClusterDimX), static_cast<uint32_t>(options.mClusterDimY),
        static_cast<uint32_t>(options.mClusterDimZ)};

    // Run the kernel.
    auto result = trtllm::gen::launchKernel((void*) &kernelParams, cudaStream, config.mSharedMemSize, cuFunction,
        block3, grid3, cluster3,
        usePdl
            && (config.mOptions.mGridWaitForPrimaryEarlyExit | config.mOptions.mGridWaitForPrimaryA
                | config.mOptions.mGridWaitForPrimaryB));
    if (result != CUDA_SUCCESS)
    {
        return -1;
    }
    // If a module cache has not been given, unload the module to avoid leaking
    if (!moduleCache.has_value())
    {
        cuModuleUnload(cuModule);
    }
#else
    config.mCudaRunner->run((void*) &kernelParams, (void*) cudaStream, grid);
#endif

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t GemmGatedActInterface::runInitBeforeWorldSync(GemmGatedActConfig const&, GemmGatedActData const&, void*) const
{
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemmGatedAct

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemmGatedAct
