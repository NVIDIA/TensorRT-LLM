/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "DevKernel.h"

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include <cub/cub.cuh>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function for array conversion
template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input)
{
    cutlass::NumericArrayConverter<typename U::Element, typename T::Element, U::kElements> converter;
    return converter(input);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace moe::dev
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace activation
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float silu(float x)
{
    return x / (1.0f + expf(-x));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void activationKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // immediately trigger the secondary kernel when using PDL, then wait on primary
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    for (int tokenIdx = blockIdx.z; tokenIdx < params.numTokens; tokenIdx += gridDim.z)
    {
        // Look over experts per token
        for (int k = blockIdx.y; k < params.topK; k += gridDim.y)
        {
            int const expandedIdx = tokenIdx * params.topK + k;
            int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
            if (permutedIdx == -1)
                continue;

            // Loop over hidden dim
            for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.innerDim / 2;
                 hiddenIdx += blockDim.x * gridDim.x)
            {
                int const baseIdx = permutedIdx * params.innerDim + hiddenIdx;

                float x1 = (float) params.inPtr[baseIdx];
                float x2 = (float) params.inPtr[baseIdx + params.innerDim / 2];

                float act = silu(x2);
                Type out = (Type) (act * x1);

                int const outIdx = permutedIdx * (params.innerDim / 2) + hiddenIdx;
                params.outPtr[outIdx] = out;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void activationDeepSeekKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using BlockReduce = cub::BlockReduce<float, 128>;

    __shared__ float s_scaleOut;
    __shared__ typename BlockReduce::TempStorage temp_storage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // immediately trigger the secondary kernel when using PDL, then wait on primary
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif
    // Loop over tokens
    for (int tokenIdx = blockIdx.z; tokenIdx < params.numTokens; tokenIdx += gridDim.z)
    {
        // Look over experts per token
        for (int k = blockIdx.y; k < params.topK; k += gridDim.y)
        {
            int const expandedIdx = tokenIdx * params.topK + k;
            int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

            // Needed for expert parallelism
            if (permutedIdx == -1)
                continue;

            // Loop over hidden dim
            for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.innerDim / 2;
                 hiddenIdx += blockDim.x * gridDim.x)
            {
                int const baseIdx = permutedIdx * params.innerDim + hiddenIdx;

                int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];

                int const scale1_idx = permutedIdx + totalNumPaddedTokens * (hiddenIdx / 128);
                int const scale2_idx
                    = permutedIdx + totalNumPaddedTokens * ((hiddenIdx / 128) + (params.innerDim / 2 / 128));
                float const scale1 = params.inDqSfsPtr[scale1_idx];
                float const scale2 = params.inDqSfsPtr[scale2_idx];

                float x1 = scale1 * (float) params.inPtr[baseIdx];
                float x2 = scale2 * (float) params.inPtr[baseIdx + params.innerDim / 2];

                float act = silu(x2);
                float out = act * x1;

                // The largest (finite) value that can be represented using E4m3.
                float constexpr E4m3MaxVal{448.f};

                // Compute the absolute max
                float aMax = BlockReduce(temp_storage).Reduce(fabsf(out), cuda::maximum<>());
                if (threadIdx.x == 0)
                {
                    s_scaleOut = aMax / E4m3MaxVal;
                    int const scaleOut_idx = permutedIdx + totalNumPaddedTokens * (hiddenIdx / 128);
                    params.outDqSfsPtr[scaleOut_idx] = aMax / E4m3MaxVal;
                }
                __syncthreads();
                float const scaleOut = s_scaleOut;
                __syncthreads();
                int const outIdx = permutedIdx * (params.innerDim / 2) + hiddenIdx;
                params.outPtr[outIdx] = (Type) (out / scaleOut);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    if (data.mDtypeElt == tg::Dtype::E2m1)
    {
        // Note: this should be unreachable because the options are checked beforehand.
        // E2m1 requires using higher-precision intermediate data (bf16).
        TLLM_CHECK_WITH_INFO(false, "Activation with E2m1_t isn't supported.");
        return;
    }

    if (data.mUseDeepSeekFp8)
    {
        int const numThreads = 128;
        const dim3 grid(data.innerDim / 128, data.topK, std::min(8192, data.numTokens));

        LAUNCH(data, activationDeepSeekKernel, grid, numThreads, 0, stream);
    }
    else
    {
        int const numThreads = 256;
        const dim3 grid(data.innerDim / 128, data.topK, std::min(8192, data.numTokens));

        LAUNCH(data, activationKernel, grid, numThreads, 0, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace activation

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace convertsf
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

namespace dev
{
// Compute the offset that corresponds to (dataRowIdx, dataBlkColIdx) in the SF tensor where
// dataRowIdx and dataBlkColIdx are the respective indices of the row and the block of 16 elts
// from the K dim in the tensor of data.
inline __device__ int64_t getSfOffset(int32_t dataRowIdx, int32_t dataBlkColIdx, int32_t numDataBlksPerRow)
{

    // The number of rows of SF per block.
    static int32_t constexpr NumRowsPerSfBlock = 128;
    // The number of cols of SF per block.
    static int32_t constexpr NumColsPerSfBlock = 4;
    // The size of each SF block.
    static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;

    // The number of rows of data per SF block.
    static int32_t constexpr NumDataRowsPerSfBlock = NumRowsPerSfBlock;
    // The number of cols of blocks of data per SF block.
    static int32_t constexpr NumDataBlkColsPerSfBlock = NumColsPerSfBlock;

    // The row of the SF block in the SF tensor.
    int sfBlkRowIdx = dataRowIdx / NumDataRowsPerSfBlock;
    // The col of the SF block in the SF tensor.
    int sfBlkColIdx = dataBlkColIdx / NumDataBlkColsPerSfBlock;
    // The blocks are stored row-major in the tensor of scaling factors.
    int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / NumDataBlkColsPerSfBlock + sfBlkColIdx;

    // Find the row in the SF block.
    int sfRowIdx = (dataRowIdx % 32) * 4 + (dataRowIdx % NumDataRowsPerSfBlock) / 32;
    // Find the col in the SF block.
    int sfColIdx = (dataBlkColIdx % 4);

    // Compute the offset in bytes.
    return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock + sfColIdx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Given the GMEM address of an output element, compute the offset of the corresponding scaling
// factor in the SF tensor. Optionally, a startTokenIndex can be provided if the first token is not
// the start token in the SF tensor. This is useful when inflight batching is enabled in TRT-LLM,
// where the context and generation output are stored as one output tensor. In this case, the
// generation output may not start with zero offset in the SF output tensor.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(int64_t gmemOffsetInBytes, int32_t hiddenDim, int32_t startTokenIdx = 0)
{
    // The number of elements per sf.
    int32_t constexpr NumEltsPerSf = 16;
    // The GMEM offset of the output element.
    int64_t gmemOffset = gmemOffsetInBytes * 8 /*bits*/ / NumBitsPerElt;
    // The row/col indices of the corresponding SF element.
    int32_t sfRowIdx = gmemOffset / hiddenDim + startTokenIdx;
    int32_t sfColIdx = (gmemOffset % hiddenDim) / NumEltsPerSf;
    // Compute the SF offset.
    return getSfOffset(sfRowIdx, sfColIdx, hiddenDim / NumEltsPerSf);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO(tizheng): Refactor to track gmem offset instead of doing pointer subtraction.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(
    void const* gmemOutPtr, void const* gmemBasePtr, int32_t hiddenDim, int32_t startTokenIdx = 0)
{
    return getSfOffset<NumBitsPerElt>(
        reinterpret_cast<char const*>(gmemOutPtr) - reinterpret_cast<char const*>(gmemBasePtr), hiddenDim,
        startTokenIdx);
}

} // namespace dev

// TODO: it would be nice to move some of that logic to Fp4Utils.h
template <tg::SfLayout Layout>
inline __device__ int32_t getSfOffset(int32_t dataRowIdx, int32_t dataBlkColIdx, int32_t numDataBlksPerRow)
{
    if constexpr (Layout == tg::SfLayout::Linear)
    {
        return numDataBlksPerRow * dataRowIdx + dataBlkColIdx;
    }
    else if constexpr (Layout == tg::SfLayout::R128c4)
    {
        return static_cast<int32_t>(dev::getSfOffset(dataRowIdx, dataBlkColIdx, numDataBlksPerRow));
    }
    else if constexpr (Layout == tg::SfLayout::R8c4 || Layout == tg::SfLayout::R8c16)
    {
        static int32_t constexpr NumRowsPerSfBlock = 8;
        static int32_t constexpr NumColsPerSfBlock = (Layout == tg::SfLayout::R8c4) ? 4 : 16;
        static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;
        int sfBlkRowIdx = dataRowIdx / NumRowsPerSfBlock;
        int sfBlkColIdx = dataBlkColIdx / NumColsPerSfBlock;
        int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / NumColsPerSfBlock + sfBlkColIdx;
        int sfRowIdx = dataRowIdx % NumRowsPerSfBlock;
        int sfColIdx = dataBlkColIdx % NumColsPerSfBlock;
        return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock + sfColIdx;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <tg::SfLayout LayoutSrc, tg::SfLayout LayoutDst, typename KernelParams>
__device__ void convertSfCommon(KernelParams params)
{
    // Note: it's assumed that the number of scaling factors per row is a multiple of 4.
    constexpr int VecSize = 4;
    using VecType = uint32_t;
    static_assert(sizeof(VecType) == VecSize);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Immediately trigger the secondary kernel when using PDL, then wait on primary.
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    // TODO: consider optimizing if used in production.
    // This is a naive kernel. It's not doing coalesced loads.

    int const numSfPerRow = params.hiddenDimSf;

    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y)
    {
        for (int hiddenSfVecIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenSfVecIdx < numSfPerRow / VecSize;
             hiddenSfVecIdx += blockDim.x * gridDim.x)
        {
            // Index of the first SF in the vector.
            int const hiddenSfIdx = VecSize * hiddenSfVecIdx;

            // Load scale factors.
            int sfIdxIn = getSfOffset<LayoutSrc>(tokenIdx, hiddenSfIdx, numSfPerRow);
            const VecType sfVec = reinterpret_cast<VecType const*>(params.inSfPtr)[sfIdxIn / VecSize];

            // Store scale factors.
            int const sfIdxOut = getSfOffset<LayoutDst>(tokenIdx, hiddenSfIdx, numSfPerRow);
            reinterpret_cast<VecType*>(params.outSfPtr)[sfIdxOut / VecSize] = sfVec;
        }
    }
}

#define CONVERT_FP4_SF_KERNEL(LayoutSrc, LayoutDst)                                                                    \
    template <typename KernelParams>                                                                                   \
    __global__ void convertSf##LayoutSrc##To##LayoutDst##Kernel(KernelParams params)                                   \
    {                                                                                                                  \
        convertSfCommon<tg::SfLayout::LayoutSrc, tg::SfLayout::LayoutDst>(params);                                     \
    }
// We only need a conversion to the linear layout.
CONVERT_FP4_SF_KERNEL(R128c4, Linear);
CONVERT_FP4_SF_KERNEL(R8c4, Linear);
CONVERT_FP4_SF_KERNEL(R8c16, Linear);
#undef CONVERT_FP4_SF_KERNEL

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    constexpr int VecSize = 4;
    int const numThreads = 128;
    int const numBlocksX = (data.hiddenDimSf / VecSize - 1 + numThreads) / numThreads;
    int const numBlocksY = std::min(8192, data.numTokens);
    dim3 numBlocks(numBlocksX, numBlocksY);
#define CONVERT_FP4_SF_LAUNCH(LayoutSrc, LayoutDst)                                                                    \
    if (data.sfLayoutSrc == tg::SfLayout::LayoutSrc && data.sfLayoutDst == tg::SfLayout::LayoutDst)                    \
    {                                                                                                                  \
        LAUNCH_PDL(data, false, cutlass::float_e4m3_t, convertSf##LayoutSrc##To##LayoutDst##Kernel, numBlocks,         \
            numThreads, 0, stream);                                                                                    \
        return;                                                                                                        \
    }
    CONVERT_FP4_SF_LAUNCH(R128c4, Linear);
    CONVERT_FP4_SF_LAUNCH(R8c4, Linear);
    CONVERT_FP4_SF_LAUNCH(R8c16, Linear);
#undef CONVERT_FP4_SF_LAUNCH
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace convertsf

namespace permute
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void permuteKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // immediately trigger the secondary kernel when using PDL, then wait on primary
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y)
    {
        // Loop over hidden dim
        for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
             hiddenIdx += blockDim.x * gridDim.x)
        {

            // Load chunk of token into registers
            const Type data = params.inPtr[tokenIdx * params.hiddenDim + hiddenIdx];

            // Write to topK places
            for (int k = 0; k < params.topK; k++)
            {
                int const expandedIdx = tokenIdx * params.topK + k;
                int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
                params.outPtr[permutedIdx * params.hiddenDim + hiddenIdx] = data;
            }
        }
        if (params.useDeepSeekFp8)
        {
            for (int scaleIdx = threadIdx.x + blockDim.x * blockIdx.x; scaleIdx < params.hiddenDim / 128;
                 scaleIdx += blockDim.x * gridDim.x)
            {
                for (int k = 0; k < params.topK; k++)
                {
                    int const expandedIdx = tokenIdx * params.topK + k;
                    int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

                    int const idx_in = tokenIdx + params.numTokens * scaleIdx;
                    int const idx_out = permutedIdx + params.totalNumPaddedTokens[0] * scaleIdx;

                    params.outDqSfsPtr[idx_out] = params.inDqSfsPtr[idx_in];
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    int const numThreads = 256;
    int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
    int const numBlocksY = std::min(8192, data.numTokens);
    dim3 numBlocks(numBlocksX, numBlocksY);

    LAUNCH(data, permuteKernel, numBlocks, numThreads, 0, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace permute

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace finalize
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using TypeExpW = typename KernelParams::TypeExpW;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // wait on primary kernel when using PDL
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y)
    {
        // Loop over hidden dim
        for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
             hiddenIdx += blockDim.x * gridDim.x)
        {

            // Accumulate chunk of token into registers
            float data = 0.0F;

            // Write to topK places
            for (int k = 0; k < params.topK; k++)
            {
                int const expandedIdx = tokenIdx * params.topK + k;
                int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

                if (permutedIdx == -1)
                {
                    continue;
                }

                if (params.expertWeightsPtr != nullptr)
                {
                    TypeExpW const scale = params.expertWeightsPtr[expandedIdx];
                    data += float{scale} * float{params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
                }
                else
                {
                    data += float{params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
                }
            }

            params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] = static_cast<Type>(data);
        }
    }
}

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

__device__ float4 vectorizedLoadPtx(float4 const* ptr)
{
    float4 ret;
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
                 : "l"(ptr));
    return ret;
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.

template <typename KernelParams>
__global__ void finalizeKernelVecLoad(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using TypeExpW = typename KernelParams::TypeExpW;

    int const hiddenDimPaddedBits = params.hiddenDimPadded * cutlass::sizeof_bits<Type>::value;
    int const hiddenDimBits = params.hiddenDim * cutlass::sizeof_bits<Type>::value;
    assert(hiddenDimPaddedBits % 128 == 0);
    assert(hiddenDimBits % 128 == 0);

    // Load 128-bits per thread, according to the smallest data type we read/write
    constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<Type>::value;
    using InputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;

    int64_t const tokenIdx = blockIdx.x;
    int64_t const startOffset = threadIdx.x;
    int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
    int64_t const numElemsInPaddedCol = params.hiddenDimPadded / FINALIZE_ELEM_PER_THREAD;
    int64_t const numElemsInCol = params.hiddenDim / FINALIZE_ELEM_PER_THREAD;

    auto const offset = tokenIdx * params.hiddenDim;
    Type* outputPtr = params.outPtr + offset;
    auto* outElemPtr = reinterpret_cast<OutputElem*>(outputPtr);
    auto const* inElemPtr = reinterpret_cast<InputElem const*>(params.inPtr);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // wait on primary kernel when using PDL
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    for (int elemIndex = startOffset; elemIndex < numElemsInCol; elemIndex += stride)
    {
        ComputeElem threadOutput;
        threadOutput.fill(0);
        for (int k = 0; k < params.topK; ++k)
        {
            int const expandedIdx = tokenIdx * params.topK + k;
            int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
            if (permutedIdx == -1)
            {
                continue;
            }

            float const scale
                = (params.expertWeightsPtr != nullptr) ? static_cast<float>(params.expertWeightsPtr[expandedIdx]) : 1.f;

            auto const* inputPermutedPtr = inElemPtr + permutedIdx * numElemsInPaddedCol;

            float4 input = vectorizedLoadPtx(reinterpret_cast<float4 const*>(&inputPermutedPtr[elemIndex]));
            InputElem inputPermutedElem = *reinterpret_cast<InputElem const*>(&input);
            ComputeElem expertResult = arrayConvert<InputElem, ComputeElem>(inputPermutedElem);

            threadOutput = threadOutput + scale * expertResult;
        }

        OutputElem outputElem = arrayConvert<ComputeElem, OutputElem>(threadOutput);
        outElemPtr[elemIndex] = outputElem;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeDeepSeekKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using BlockReduce = cub::BlockReduce<float, 128>;

    __shared__ float s_scaleOut;
    __shared__ typename BlockReduce::TempStorage temp_storage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // wait on primary kernel when using PDL
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y)
    {
        // Loop over hidden dim
        for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
             hiddenIdx += blockDim.x * gridDim.x)
        {

            // Accumulate chunk of token into registers
            float acc = 0.0f;

            for (int k = 0; k < params.topK; k++)
            {
                int const expandedIdx = tokenIdx * params.topK + k;
                int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
                if (permutedIdx == -1)
                {
                    continue;
                }
                int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
                int const scaleIdx = permutedIdx + totalNumPaddedTokens * (hiddenIdx / 128);
                float const blockScale = params.inDqSfsPtr ? params.inDqSfsPtr[scaleIdx] : 1;

                float const expertProb = (float) params.expertWeightsPtr[tokenIdx * params.topK + k];

                float const scale = expertProb * blockScale;
                acc += scale * static_cast<float>(params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]);
            }

            // The largest (finite) value that can be represented using E4m3.
            float constexpr E4m3MaxVal{448.f};

            // Compute the absolute max
            float aMax = BlockReduce(temp_storage).Reduce(fabsf(acc), cuda::maximum<>());

            if (threadIdx.x == 0)
            {
                if (params.outDqSfsPtr)
                {
                    s_scaleOut = aMax / E4m3MaxVal;
                    int const scaleOut_idx = tokenIdx + hiddenIdx / 128 * params.numTokens;
                    params.outDqSfsPtr[scaleOut_idx] = aMax / E4m3MaxVal;
                }
                else
                {
                    s_scaleOut = 1.0f;
                }
            }
            __syncthreads();
            float const scaleOut = s_scaleOut;
            __syncthreads();
            params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] = (Type) (acc / scaleOut);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void run(Data const& data, void* stream)
{
    if (data.mUseDeepSeekFp8)
    {
        int const numThreads = 128;
        int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
        // Capped at rather arbitrary 8192 to avoid gridDim exceeding 65535 specified by CUDA.
        int const numBlocksY = std::min(8192, data.numTokens);
        dim3 numBlocks(numBlocksX, numBlocksY);

        LAUNCH_EXPW(data, finalizeDeepSeekKernel, numBlocks, numThreads, 0, stream);
    }
    else
    {
        int const numThreads = 256;
        int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
        // Capped at rather arbitrary 8192 to avoid gridDim exceeding 65535 specified by CUDA.
        int const numBlocksY = std::min(8192, data.numTokens);

        if (numBlocksX * numBlocksY < 1184)
        {
            // The number 1184 comes from 148 * 8, where 148 is the number of SMs (Streaming Multiprocessors) in the
            // Blackwell architecture,
            // and the value 8 means that each Streaming Multiprocessor (SM) can hold up to 8 blocks for this kernel.
            // This limitation is intended to ensure that when the number of waves is greater than 1, we choose to use
            // the kernel with vectorized loading.
            dim3 numBlocks(numBlocksX, numBlocksY);
            LAUNCH_EXPW(data, finalizeKernel, numBlocks, numThreads, 0, stream);
        }
        else
        {
            LAUNCH_EXPW(data, finalizeKernelVecLoad, /*numBlocks=*/data.numTokens,
                /*numThreads=*/FINALIZE_THREADS_PER_BLOCK, 0, stream);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace finalize

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace moe::dev
