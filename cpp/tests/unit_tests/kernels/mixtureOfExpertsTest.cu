/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

#ifdef USING_OSS_CUTLASS_MOE_GEMM
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
#include <tensorrt_llm/kernels/quantization.h>
#else
#include "moe_kernels.h"
#include "quantization.h"
#endif
#include "tensorrt_llm/kernels/cutlass_kernels/include/cutlass_kernel_selector.h"

#include "tensorrt_llm/runtime/bufferManager.h"

#include <tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h>

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

using namespace CUTLASS_MOE_GEMM_KERNELS_NAMESPACE;
using CUTLASS_MOE_GEMM_NAMESPACE::TmaWarpSpecializedGroupedGemmInput;
using CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::CutlassMoeFCRunner;
using CUTLASS_MOE_GEMM_NAMESPACE::ActivationType;
using CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::ActivationParams;
using CUTLASS_MOE_GEMM_NAMESPACE::isGatedActivation;

constexpr static float FP8_MAX = 448.f;
constexpr static float FP4_MAX = 6.f;

__host__ __device__ constexpr float applyExpertShift(float weight_value, int expert)
{
    float lookup_table[] = {0.5f, 1.0f, 2.0f};
    return weight_value * lookup_table[expert % 3];
}

template <class T>
__global__ void initWeightsKernel(T* data, int64_t w, int64_t h, float base, float scale)
{
    size_t expert_id = blockIdx.z;
    T* start_offset = data + expert_id * w * h;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h)
    {
        start_offset[y * w + x] = (x == y) ? T(applyExpertShift(base * scale, expert_id)) : T(0.f);
    }
}

template <class T>
__global__ void initWeightsGatedKernel(T* data, int64_t w, int64_t h, float base_1, float base_2, float scale)
{
    size_t expert_id = blockIdx.z;
    T* start_offset = data + expert_id * w * h * 2;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h)
    {
        start_offset[y * w + x] = (x == y) ? T(applyExpertShift(base_1 * scale, expert_id)) : T(0.f);
        start_offset[(y + h) * w + x] = (x == y) ? T(applyExpertShift(base_2 * scale, expert_id)) : T(0.f);
    }
}

template <class T>
__global__ void initBiasToExpertIdKernel(T* data, int64_t w)
{
    size_t expert_id = blockIdx.y;
    T* start_offset = data + expert_id * w;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < w)
        start_offset[x] = T(expert_id);
}

template <class T>
__global__ void initBiasToExpertIdGatedKernel(T* data, int64_t w)
{
    size_t expert_id = blockIdx.y;
    T* start_offset = data + expert_id * w * 2;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < w)
    {
        start_offset[x] = T(expert_id);
        start_offset[x + w] = T(expert_id + 1);
    }
}

template <class T>
using sizeof_bits = cutlass::sizeof_bits<typename cutlass_kernels::TllmToCutlassTypeAdapter<std::remove_cv_t<T>>::type>;

#ifdef ENABLE_FP8
using SafeFP8 = __nv_fp8_e4m3;
using SafeFP8E8 = __nv_fp8_e8m0;
#else
using SafeFP8 = void;
using SafeFP8E8 = void;
#endif
#ifdef ENABLE_FP4
using SafeFP4 = __nv_fp4_e2m1;

namespace cutlass
{
template <>
struct sizeof_bits<SafeFP4>
{
    static constexpr int value = 4;
};
} // namespace cutlass

static_assert(sizeof_bits<SafeFP4>::value == 4, "SafeFP4 is not 4 bits");
#else
using SafeFP4 = void;
#endif

template <class TypeTuple_>
class MixtureOfExpertsTest : public ::testing::Test
{
protected:
    using GemmDataType = typename TypeTuple_::DataType;
    using WeightType = typename TypeTuple_::WeightType;
    using OutputType = typename TypeTuple_::OutputType;
    using ActivationScale = typename TypeTuple_::ActivationScale;
    using WeightScale = typename TypeTuple_::WeightScale;

    using BackBoneType = OutputType;
    constexpr static bool INT4 = std::is_same_v<WeightType, cutlass::uint4b_t>;
    constexpr static bool ACT_FP8 = std::is_same_v<GemmDataType, SafeFP8>;
    constexpr static bool WEIGHT_FP8 = std::is_same_v<WeightType, SafeFP8>;
    constexpr static bool FP8 = ACT_FP8 && WEIGHT_FP8;
    constexpr static bool ACT_FP4 = std::is_same_v<GemmDataType, SafeFP4>;
    constexpr static bool WEIGHT_FP4 = std::is_same_v<WeightType, SafeFP4>;

    constexpr static bool MX_QUANT_ACT = std::is_same_v<ActivationScale, SafeFP8E8>;
    constexpr static bool MX_QUANT_WEIGHT = std::is_same_v<WeightScale, SafeFP8E8>;
    static_assert(!MX_QUANT_ACT || MX_QUANT_WEIGHT, "MX quantized act implies MX quantized weight");

    constexpr static bool NVFP4 = ACT_FP4 && WEIGHT_FP4 && !MX_QUANT_ACT && !MX_QUANT_WEIGHT;
    static_assert(!ACT_FP4 || NVFP4, "FP4 activations is only supported with NVFP4");

    constexpr static bool MXFP8_MXFP4 = ACT_FP8 && WEIGHT_FP4 && MX_QUANT_ACT && MX_QUANT_WEIGHT;
    constexpr static bool FP8_MXFP4 = ACT_FP8 && WEIGHT_FP4 && !MX_QUANT_ACT && MX_QUANT_WEIGHT;

    constexpr static bool ANY_FP4 = WEIGHT_FP4 || ACT_FP4;
    constexpr static bool ANY_FPX = ANY_FP4 || FP8;

    constexpr static bool INT_QUANT = !std::is_same_v<GemmDataType, WeightType> && std::is_integral_v<WeightType>;
    constexpr static int64_t WEIGHT_ELEM_PER_BYTE = (INT4 || WEIGHT_FP4) ? 2 : 1;
    using InputType = std::conditional_t<NVFP4 || MXFP8_MXFP4, OutputType, GemmDataType>;
    using WeightStorage = std::conditional_t<WEIGHT_ELEM_PER_BYTE == 2, uint8_t, WeightType>;
    constexpr static int64_t HIDDEN_SIZE_MULTIPLIER = 16;
    constexpr static int64_t MINIMUM_BYTE_ALIGNMENT
        = MX_QUANT_WEIGHT ? 64 : 128 / 8; // TMA requires 128 bits alignment, MX quant requires 64 bytes
    constexpr static int64_t MINIMUM_ALIGNMENT_CONST
        = MINIMUM_BYTE_ALIGNMENT * WEIGHT_ELEM_PER_BYTE / sizeof(WeightStorage);
    constexpr static int64_t DEFAULT_HIDDEN_SIZE = HIDDEN_SIZE_MULTIPLIER * MINIMUM_ALIGNMENT_CONST;
    int64_t mDeviceMinimumAlignment = MINIMUM_ALIGNMENT_CONST; // SM103 has different minimum alignment

    // FP4 uses the unquantized data type for inputs and quantizes on the fly
    using DataType = std::conditional_t<NVFP4 || MXFP8_MXFP4, OutputType, GemmDataType>;

    // FP8_MXFP4 quantizes just the weights on the fly
    using WeightRawType = std::conditional_t<FP8_MXFP4, OutputType, DataType>;

    static BufferManager::CudaStreamPtr mStream;
    static std::unique_ptr<BufferManager> mBufferManager;
    static int mDeviceCount;

    std::vector<BufferManager::IBufferPtr> managed_buffers;
    DataType* mInputTensor{};

    int64_t mHiddenSize{};
    int64_t mUnpaddedHiddenSize{}; // If 0, defaults to mHiddenSize
    int64_t mNumExperts{};
    int64_t mK{};

    float getTolerance(float scale = 1.f)
    {
        bool loose_tol = mActType != ActivationType::Relu || mUseBias;
        float tol = std::is_same_v<WeightType, uint8_t>     ? 0.1
            : std::is_same_v<WeightType, cutlass::uint4b_t> ? 0.1
            : std::is_same_v<GemmDataType, float>           ? 0.001
            : std::is_same_v<GemmDataType, half>            ? 0.005
            : std::is_same_v<GemmDataType, __nv_bfloat16>   ? 0.05
            : (MXFP8_MXFP4 || FP8_MXFP4)                    ? (loose_tol ? 0.1 : 0.01)
            : std::is_same_v<GemmDataType, SafeFP8>         ? (loose_tol ? 0.06 : 0.001)
            : NVFP4                                         ? 0.05
                                                            : 0.0;

        // Keep the scale in a sane range
        return std::max(tol, scale * tol);
    }

    static bool shouldSkip()
    {
#ifndef ENABLE_FP8
        static_assert(!FP8, "FP8 Tests enabled on unsupported CUDA version");
#endif
        bool should_skip_no_device = mDeviceCount <= 0;
        bool should_skip_unsupported_fp8 = getSMVersion() < 89 && FP8;
        bool should_skip_unsupported_fp4 = (getSMVersion() < 100) && ANY_FP4;
        return should_skip_no_device || should_skip_unsupported_fp8 || should_skip_unsupported_fp4;
    }

    static void SetUpTestCase()
    {
        mDeviceCount = getDeviceCount();
        if (shouldSkip())
        {
            GTEST_SKIP() << "Skipping due to no/unsupported GPU";
        }

        mStream = std::make_shared<CudaStream>();
        mBufferManager = std::make_unique<BufferManager>(mStream);
    }

    static void TearDownTestCase()
    {
        mBufferManager.reset();
        mStream.reset();
    }

    void SetUp() override
    {
        if (shouldSkip())
        {
            GTEST_SKIP() << "Skipping due to no/unsupported GPU";
        }
        assert(mBufferManager);

        int sm = getSMVersion();
        if (sm >= 100)
        {
            mDeviceMinimumAlignment
                = std::max(MINIMUM_ALIGNMENT_CONST, int64_t(WEIGHT_ELEM_PER_BYTE * 32 / sizeof(WeightStorage)));
        }
    }

    void TearDown() override
    {
        managed_buffers.clear();
        mUnpaddedHiddenSize = 0; // Reset for next test
        ASSERT_EQ(cudaStreamSynchronize(mStream->get()), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }

    void initWeights(WeightRawType* buffer, int64_t w, int64_t h, float base, float scalar)
    {
        dim3 block(16, 16, 1);
        dim3 grid(divUp(w, block.x), divUp(h, block.y), mNumExperts);
        initWeightsKernel<WeightRawType><<<grid, block, 0, mStream->get()>>>(buffer, w, h, base, scalar);
    }

    void initBias(BackBoneType* buffer, int64_t w)
    {
        dim3 block(256, 1, 1);
        dim3 grid(divUp(w, block.x), mNumExperts);
        initBiasToExpertIdKernel<BackBoneType><<<grid, block, 0, mStream->get()>>>(buffer, w);
    }

    void initWeightsGated(WeightRawType* buffer, int64_t w, int64_t h, float base_1, float base_2, float scalar)
    {
        if (!mIsGated)
            return initWeights(buffer, w, h, base_1, scalar);

        h /= 2;
        dim3 block(16, 16, 1);
        dim3 grid(divUp(w, block.x), divUp(h, block.y), mNumExperts);
        initWeightsGatedKernel<WeightRawType><<<grid, block, 0, mStream->get()>>>(buffer, w, h, base_1, base_2, scalar);
    }

    void initBiasGated(BackBoneType* buffer, int64_t w)
    {
        if (!mIsGated)
            return initBias(buffer, w);

        w /= 2;
        dim3 block(256, 1, 1);
        dim3 grid(divUp(w, block.x), mNumExperts);
        initBiasToExpertIdGatedKernel<BackBoneType><<<grid, block, 0, mStream->get()>>>(buffer, w);
    }

    CutlassMoeFCRunner<GemmDataType, WeightType, OutputType, InputType, BackBoneType> mMoERunner{};
    char* mWorkspace{};
    int* mSelectedExpert;
    float* mTokenFinalScales{};
    WeightRawType* mRawExpertWeight1{};
    WeightRawType* mRawExpertWeight2{};
    WeightStorage* mExpertWeight1{};
    WeightStorage* mExpertWeight2{};

    float mSwigluAlphaValue{0.5f};
    float mSwigluBetaValue{MX_QUANT_ACT ? 0.0f : 1.f};
    float mSwigluLimitValue{MX_QUANT_ACT ? FP8_MAX / 4 : NVFP4 ? 2.f : 0.5f};
    float* mSwigluAlpha{};
    float* mSwigluBeta{};
    float* mSwigluLimit{};

    DataType* mExpertIntScale1{};
    DataType* mExpertIntScale2{};

    float mFP8WeightScalar1{1.f};
    float mFP8WeightScalar2{1.f};
    float* mExpertFPXScale1{};
    float* mExpertFPXScale2{};
    float* mExpertFPXScale3{};

    float* mExpertFP4ActGlobalScale1{};
    float* mExpertFP4WeightGlobalScale1{};
    float* mExpertFP4WeightGlobalScale2{};

    using ElementSF = TmaWarpSpecializedGroupedGemmInput::ElementSF;
    constexpr static int FP4VecSize = MX_QUANT_WEIGHT ? TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize
                                                      : TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize;
#ifdef USING_OSS_CUTLASS_MOE_GEMM
    constexpr static int MinNDimAlignmentFP4 = MX_QUANT_WEIGHT
        ? TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX
        : TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4;
    constexpr static int MinKDimAlignmentFP4 = MX_QUANT_WEIGHT
        ? TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX
        : TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4;
#else
    constexpr static int MinNDimAlignmentFP4 = MX_QUANT_WEIGHT
        ? TmaWarpSpecializedGroupedGemmInput::MinNumRowsAlignmentMXFPX
        : TmaWarpSpecializedGroupedGemmInput::MiNumRowsAlignmentNVFP4;
    constexpr static int MinKDimAlignmentFP4 = FP4VecSize * 4; // Hardcode the correct value
#endif
    ElementSF* mFP4ScalingFactorsW1 = nullptr;
    ElementSF* mFP4ScalingFactorsW2 = nullptr;

    BackBoneType* mExpertBias1{};
    BackBoneType* mExpertBias2{};

    void* mTpExpertScratch{}; // Copy the experts here when slicing up inputs
    size_t mTpExpertScratchSize{};

    OutputType* mFinalOutput{};
    int* mSourceToExpandedMap;

    float mInterSizeFraction = 4.f;
    int64_t mInterSize{};
    int64_t mTotalTokens{};

    bool mUseBias = true;
    bool mUseLora = false;
    bool mUsePrequantScale = false;

    // Run tests with per-expert act scale
    bool mUsePerExpertActScale = true;

    bool mIsGated = false;
    int64_t mGatedMultiplier = 1;
    int64_t mGroupSize = -1;

    ActivationType mActType = ActivationType::Relu;

    float mSparseMixerEpsilon = 0.2f;

    // Default this to false. This only matters for K>2, and so by doing this we will test the fused and unfused paths
    bool mUseFusedFinalize = false;
    // The internal fused finalize variable, true if k < 3 or mUseFusedFinalize is true
    bool mUseFusedFinalizeInternal = false;

    // Default this to TMA. This only matters for SM10x.
    tensorrt_llm::cutlass_extensions::EpilogueScheduleType mEpilogueSchedule
        = tensorrt_llm::cutlass_extensions::EpilogueScheduleType::TMA;

    // Disable this for long running tests to speed up runtime
    bool mIsLongTest = false;

    // If the test sets mOverrideSelectedConfig1 the BasicPermuteTest and *ParallelTests will use that instead of
    // looping over samples for the different architectures we support.
    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> mOverrideSelectedConfig1 = std::nullopt;
    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> mOverrideSelectedConfig2 = std::nullopt;

    // This is the actual tactic we use internally in runMoePermute
    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> mInternalSelectedConfig1 = std::nullopt;
    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> mInternalSelectedConfig2 = std::nullopt;

    // Keep to simple power of two so we can have tight bounds on precision for quantized modes
    float const mExpertWDiag1{0.5};
    float const mExpertWDiagGated{1};
    float const mExpertWDiag2{2};

    float mMaxInput{};

    char mMemsetValue = 0xD5;

    template <class AllocType>
    AllocType* allocBuffer(size_t size)
    {
        size_t size_bytes = cute::ceil_div(size * sizeof_bits<AllocType>::value, 8);
        managed_buffers.emplace_back(mBufferManager->gpu(size_bytes));
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "Error allocating buffer of size: " << size;
        AllocType* ptr = static_cast<AllocType*>(managed_buffers.back()->data());
        // Memset to an obviously incorrect value, so we detect any issues with uninitialised fields
        check_cuda_error(cudaMemsetAsync(ptr, mMemsetValue, size_bytes, mStream->get()));
        return ptr;
    }

    bool checkSufficientTestMemory(
        int64_t num_tokens, int64_t hidden_size, int64_t num_experts, int64_t k, bool parallel = false)
    {
        this->managed_buffers.clear();             // Make sure all the previous buffers are freed
        check_cuda_error(cudaDeviceSynchronize()); // Sync to make sure all previous operations are resolved

        // Calculate the size contributions for all the large buffers to check if the GPU has enough space
        bool const is_gated = isGatedActivation(mActType);
        size_t const num_gemms = 2 + is_gated;
        bool const useDeepseek = false;

        // Expert weights
        size_t const weight_elems = hidden_size * (hidden_size * mInterSizeFraction) * num_experts * num_gemms;
        size_t const weight_size = weight_elems * sizeof(WeightStorage) / WEIGHT_ELEM_PER_BYTE;

        // Workspace size
        size_t const workspace_size = this->mMoERunner.getWorkspaceSize(num_tokens, hidden_size, hidden_size * 4,
            num_experts, k, this->mActType, {}, mUseLora, useDeepseek, false, mUsePrequantScale);
        // The input/output buffers
        size_t const in_out_size = 2 * num_tokens * hidden_size * sizeof(DataType);

        // This should be correct to within 100MiB (on tests with 30GiB total)
        size_t total_size = workspace_size + weight_size + in_out_size;

        // We allocate an extra shard of the weights for the parallel case, divide by 2 for when TP2
        if (parallel)
        {
            total_size += weight_size / 2;
        }
        // Quantized data types use a second scratch buffer for the weights before quantizing
        if (ANY_FPX || INT_QUANT)
        {
            total_size += weight_elems * sizeof(DataType);
        }

        size_t const memory_pool_free_mem_size = mBufferManager->memoryPoolFree();
        auto const [freeMem, totalMem] = tensorrt_llm::common::getDeviceMemoryInfo(false);
        float const freeMemBuffer = 0.9f; // Add some buffer so we aren't completely pushing the limits
        std::cout << "Free memory is: " << freeMem << ", memory pool size is: " << memory_pool_free_mem_size
                  << ", required memory is: " << total_size << ", device total memory capacity: " << totalMem
                  << std::endl;
        return (freeMem + memory_pool_free_mem_size) * freeMemBuffer >= total_size;
    }

    void initLocals(int64_t hidden_size, int64_t num_experts, int64_t k, int64_t num_tokens)
    {
        mHiddenSize = hidden_size;
        mInterSize = hidden_size * mInterSizeFraction;
        mNumExperts = num_experts;
        mK = k;
        mIsGated = isGatedActivation(mActType);
        mGatedMultiplier = mIsGated ? 2 : 1;
        mUseFusedFinalizeInternal = mUseFusedFinalize || k < 3;
        mMoERunner.use_fused_finalize_ = mUseFusedFinalizeInternal;
        mTotalTokens = num_tokens;
    }

    void initBuffersPermute(std::vector<DataType> h_hidden_states, std::vector<int> h_token_selected_experts,
        std::vector<float> h_token_final_scales, int64_t hidden_size, int64_t num_experts, int64_t k,
        MOEParallelismConfig parallelism_config)
    {
        managed_buffers.clear();

        auto const gated_inter = mInterSize * mGatedMultiplier;

        EXPECT_EQ(h_hidden_states.size() / hidden_size, mTotalTokens);
        EXPECT_EQ(h_token_selected_experts.size(), mTotalTokens * mK);
        EXPECT_EQ(h_token_final_scales.size(), mTotalTokens * mK);

        bool const useDeepseek = false;
        size_t workspace_size = mMoERunner.getWorkspaceSize(mTotalTokens, mHiddenSize, mInterSize, mNumExperts, mK,
            mActType, parallelism_config, mUseLora, useDeepseek, false, mUsePrequantScale);

        auto const stream = mStream->get();

        mWorkspace = allocBuffer<char>(workspace_size);

        size_t const expert_matrix_size = mNumExperts * mHiddenSize * mInterSize;

        mRawExpertWeight1 = allocBuffer<WeightRawType>(expert_matrix_size * mGatedMultiplier);
        mRawExpertWeight2 = allocBuffer<WeightRawType>(expert_matrix_size);

        size_t const experts_per_node = mNumExperts / parallelism_config.ep_size;
        int const moe_parallel_size = parallelism_config.tp_size * parallelism_config.ep_size;

        using SliceWeightType = std::conditional_t<WEIGHT_FP4, WeightRawType, WeightStorage>;
        mTpExpertScratchSize = sizeof(SliceWeightType) * expert_matrix_size * mGatedMultiplier / moe_parallel_size;
        mTpExpertScratchSize += sizeof(SliceWeightType) * expert_matrix_size / moe_parallel_size;

        mExpertBias1 = nullptr;
        mExpertBias2 = nullptr;
        if (mUseBias)
        {
            // Allow space for the slice of bias1 in the scratch
            mTpExpertScratchSize += sizeof(BackBoneType) * experts_per_node * gated_inter / parallelism_config.tp_size;
            mExpertBias1 = allocBuffer<BackBoneType>(mNumExperts * gated_inter);
            mExpertBias2 = allocBuffer<BackBoneType>(mNumExperts * mHiddenSize);

            check_cuda_error(
                cudaMemsetAsync(mExpertBias1, 0x0, mNumExperts * gated_inter * sizeof(BackBoneType), stream));
            check_cuda_error(
                cudaMemsetAsync(mExpertBias2, 0x0, mNumExperts * mHiddenSize * sizeof(BackBoneType), stream));
        }

        if constexpr (INT_QUANT)
        {
            mExpertWeight1 = allocBuffer<WeightStorage>(expert_matrix_size * mGatedMultiplier / WEIGHT_ELEM_PER_BYTE);
            mExpertWeight2 = allocBuffer<WeightStorage>(expert_matrix_size / WEIGHT_ELEM_PER_BYTE);

            mExpertIntScale1 = allocBuffer<DataType>(mNumExperts * gated_inter);
            mExpertIntScale2 = allocBuffer<DataType>(mNumExperts * mHiddenSize);
        }
        else if constexpr (ANY_FP4)
        {
            mExpertWeight1 = allocBuffer<WeightStorage>(
                expert_matrix_size * mGatedMultiplier / WEIGHT_ELEM_PER_BYTE / moe_parallel_size);
            mExpertWeight2 = allocBuffer<WeightStorage>(expert_matrix_size / WEIGHT_ELEM_PER_BYTE / moe_parallel_size);

            size_t const padded_fc1_size = mNumExperts
                * TmaWarpSpecializedGroupedGemmInput::alignToSfDim(mHiddenSize, MinKDimAlignmentFP4)
                * TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    mInterSize / parallelism_config.tp_size, MinNDimAlignmentFP4)
                * mGatedMultiplier / parallelism_config.ep_size;
            size_t const padded_fc2_size = mNumExperts
                * TmaWarpSpecializedGroupedGemmInput::alignToSfDim(mInterSize, MinKDimAlignmentFP4)
                * TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    mHiddenSize / parallelism_config.tp_size, MinNDimAlignmentFP4)
                / parallelism_config.ep_size;
            mFP4ScalingFactorsW1 = allocBuffer<ElementSF>(padded_fc1_size / FP4VecSize);
            mFP4ScalingFactorsW2 = allocBuffer<ElementSF>(padded_fc2_size / FP4VecSize);
        }
        else
        {
            mExpertWeight1 = mRawExpertWeight1;
            mExpertWeight2 = mRawExpertWeight2;
        }

        if constexpr (ANY_FPX)
        {
            // FP4 uses the same logic as FP8 to generate the global scales
            mExpertFPXScale1 = allocBuffer<float>(mNumExperts);
            mExpertFPXScale2 = allocBuffer<float>(mNumExperts); // mNumExperts or 1
            mExpertFPXScale3 = allocBuffer<float>(mNumExperts);

            if (ANY_FP4)
            {
                mExpertFP4ActGlobalScale1 = allocBuffer<float>(mNumExperts); // mNumExperts or 1
                mExpertFP4WeightGlobalScale1 = allocBuffer<float>(mNumExperts);
                mExpertFP4WeightGlobalScale2 = allocBuffer<float>(mNumExperts);
            }

            EXPECT_NE(mMaxInput, 0.0f);
            initFPQuantScales(mMaxInput);
        }

        if (parallelism_config.tp_size > 1 || parallelism_config.ep_size > 1)
        {
            mTpExpertScratch = allocBuffer<char>(mTpExpertScratchSize);
        }

        mTokenFinalScales = allocBuffer<float>(mTotalTokens * mK);
        mSelectedExpert = allocBuffer<int>(mTotalTokens * mK);

        mInputTensor = allocBuffer<DataType>(mTotalTokens * mHiddenSize);
        mFinalOutput = allocBuffer<OutputType>(mTotalTokens * mHiddenSize);

        mSourceToExpandedMap = allocBuffer<int>(mTotalTokens * mK);

        if (mActType == ActivationType::SwigluBias)
        {
            mSwigluAlpha = allocBuffer<float>(mNumExperts);
            mSwigluBeta = allocBuffer<float>(mNumExperts);
            mSwigluLimit = allocBuffer<float>(mNumExperts);
            std::vector<float> h_swiglu_alpha(mNumExperts, mSwigluAlphaValue);
            std::vector<float> h_swiglu_beta(mNumExperts, mSwigluBetaValue);
            std::vector<float> h_swiglu_limit(mNumExperts, mSwigluLimitValue);
            check_cuda_error(cudaMemcpyAsync(
                mSwigluAlpha, h_swiglu_alpha.data(), mNumExperts * sizeof(float), cudaMemcpyHostToDevice, stream));
            check_cuda_error(cudaMemcpyAsync(
                mSwigluBeta, h_swiglu_beta.data(), mNumExperts * sizeof(float), cudaMemcpyHostToDevice, stream));
            check_cuda_error(cudaMemcpyAsync(
                mSwigluLimit, h_swiglu_limit.data(), mNumExperts * sizeof(float), cudaMemcpyHostToDevice, stream));
        }

        check_cuda_error(cudaMemcpyAsync(mSelectedExpert, h_token_selected_experts.data(),
            mTotalTokens * mK * sizeof(int), cudaMemcpyHostToDevice, stream));
        check_cuda_error(cudaMemcpyAsync(mTokenFinalScales, h_token_final_scales.data(),
            mTotalTokens * mK * sizeof(float), cudaMemcpyHostToDevice, stream));
        check_cuda_error(cudaMemcpyAsync(mInputTensor, h_hidden_states.data(),
            h_hidden_states.size() * sizeof(DataType), cudaMemcpyHostToDevice, stream));

        check_cuda_error(cudaStreamSynchronize(stream));

        // Init the diagonals of our matrix, this will set to the scalar value
        initWeightsGated(
            mRawExpertWeight1, mHiddenSize, gated_inter, mExpertWDiag1, mExpertWDiagGated, mFP8WeightScalar1);
        initWeights(mRawExpertWeight2, mInterSize, mHiddenSize, mExpertWDiag2, mFP8WeightScalar2);

        if (mUseBias)
        {
            initBiasGated(mExpertBias1, gated_inter);
            initBias(mExpertBias2, mHiddenSize);
        }

        if constexpr (INT_QUANT)
        {
            cutlass_kernels::QuantType quant_type
                = INT4 ? cutlass_kernels::QuantType::W4_A16 : cutlass_kernels::QuantType::W8_A16;

            std::vector<size_t> shape1{(size_t) mNumExperts, (size_t) mHiddenSize, (size_t) gated_inter};
            std::vector<size_t> shape2{(size_t) mNumExperts, (size_t) mInterSize, (size_t) mHiddenSize};

            doIntQuant(quant_type, shape1, mRawExpertWeight1, mExpertIntScale1, mExpertWeight1);
            doIntQuant(quant_type, shape2, mRawExpertWeight2, mExpertIntScale2, mExpertWeight2);
        }

        check_cuda_error(cudaStreamSynchronize(stream));
    }

    void doIntQuant(cutlass_kernels::QuantType quant_type, std::vector<size_t> shape, DataType* inputs,
        DataType* scales, uint8_t* outputs)
    {
        // Runs on the CPU, must be after stream sync
        if constexpr (INT_QUANT)
        {
            check_cuda_error(cudaStreamSynchronize(mStream->get()));

            size_t elems = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
            std::vector<int8_t> h_out(elems);
            std::vector<DataType> h_input(elems);
            std::vector<DataType> h_scales(shape[0] * shape[2]);

            check_cuda_error(cudaMemcpy(h_input.data(), inputs, elems * sizeof(DataType), cudaMemcpyDeviceToHost));

            cutlass_kernels::symmetric_quantize(h_out.data(), h_scales.data(), h_input.data(), shape, quant_type, true);

            check_cuda_error(cudaMemcpy(
                outputs, h_out.data(), elems * sizeof(int8_t) / WEIGHT_ELEM_PER_BYTE, cudaMemcpyHostToDevice));
            check_cuda_error(
                cudaMemcpy(scales, h_scales.data(), h_scales.size() * sizeof(DataType), cudaMemcpyHostToDevice));
        }
    }

    void doFP4Quant(WeightRawType const* raw_weights, WeightStorage* quant_weights, float const* global_scales,
        ElementSF* scaling_factors, int in_shape, int out_shape, int num_experts)
    {
        int64_t const mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
        int64_t padded_out_dim = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(out_shape, MinNDimAlignmentFP4);
        int64_t padded_in_dim = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(in_shape, MinKDimAlignmentFP4);
        check_cuda_error(cudaMemsetAsync(scaling_factors, 0x00,
            num_experts * padded_out_dim * padded_in_dim / FP4VecSize * sizeof(ElementSF), mStream->get()));
#ifdef USING_OSS_CUTLASS_MOE_GEMM
        invokeFP4Quantization<WeightRawType, FP4VecSize>(num_experts, out_shape, in_shape, raw_weights, global_scales,
            reinterpret_cast<int64_t*>(quant_weights), reinterpret_cast<int32_t*>(scaling_factors), MX_QUANT_WEIGHT,
            tensorrt_llm::QuantizationSFLayout::SWIZZLED, mMultiProcessorCount, mStream->get());
#else
        invokeBatchedFP4Quantization<WeightRawType, FP4VecSize>(num_experts, out_shape, in_shape, raw_weights,
            global_scales, reinterpret_cast<int64_t*>(quant_weights), reinterpret_cast<int32_t*>(scaling_factors),
            MX_QUANT_WEIGHT, mMultiProcessorCount, mStream->get());
#endif

        // auto sf_data = getDataFromDevice<ElementSF>(scaling_factors, num_experts * padded_out_dim * padded_in_dim /
        // FP4VecSize); auto unquant_data = getDataFromDevice<WeightRawType>(raw_weights, num_experts * out_shape *
        // in_shape); auto quant_data = getDataFromDevice<uint32_t>((uint32_t*)quant_weights, num_experts * out_shape *
        // in_shape / 8); for(int expert = 0; expert < num_experts; expert++) {
        //     for(int i = 0; i < out_shape; i++) {
        //         for(int j = 0; j < in_shape / FP4VecSize; j++) {
        //             printf("quant_weights[(%d, %d, %d)]: ", expert, i, j * FP4VecSize);
        //             for(int k = 0; k < FP4VecSize / 8; k++) {
        //                 printf("0x%08x, ", quant_data[(expert * out_shape * in_shape + i * in_shape + j * FP4VecSize)
        //                 / 8 + k]);
        //             }
        //             printf("scaling_factors: %e, ",
        //             (float)sf_data[tensorrt_llm::kernels::get_sf_out_offset_128x4<FP4VecSize>(expert, i, j,
        //             out_shape, in_shape)]); printf("original: "); for(int k = 0; k < FP4VecSize; k++) {
        //                 printf("%e, ", (float)unquant_data[expert * out_shape * in_shape + i * in_shape + j *
        //                 FP4VecSize + k]);
        //             }
        //             printf("\n");
        //         }
        //     }
        // }
    }

    constexpr static float getFPXActScalar(float in)
    {
        // Our FP8 x MXFP4 implementation uses a global scale factor
        if (FP8 || FP8_MXFP4)
            return FP8_MAX / in;
        if (NVFP4)
            // We need to represent the block SF using FP8, so the largest value should be at most FP4_MAX * FP8_MAX
            // return FP8_MAX * FP4_MAX / in;
            // We carefully control precision in FP4. We want to avoid introducing any non-powers of two
            return 2.0f;

        // MX quant does not have a global scale factor
        return 1.0f;
    }

    constexpr static float getFPXWeightScalar(float in)
    {
        if (FP8)
            return FP8_MAX / in;
        if (NVFP4)
            // We need to represent the block SF using FP8, so the largest value should be at most FP4_MAX * FP8_MAX
            // return FP8_MAX * FP4_MAX / in;
            // We carefully control precision in FP4. We want to avoid introducing any non-powers of two
            return 2.0f;

        // MX quant does not have a global scale factor
        return 1.0f;
    }

    void initFPQuantScales(float max_input)
    {
        check_cuda_error(cudaStreamSynchronize(mStream->get()));

        // Add shift to the max because we add an adjustment for each expert so they get different results.
        float maxW1 = 0.f;
        int maxIndex = 0;
        float maxW2 = 0.f;
        float const maxW1GatedVal = mIsGated ? std::max(mExpertWDiag1, mExpertWDiagGated) : mExpertWDiag1;
        for (int i = 0; i < mNumExperts; i++)
        {
            float w1 = applyExpertShift(maxW1GatedVal, i);
            float w2 = applyExpertShift(mExpertWDiag2, i);
            if (w1 > maxW1)
            {
                maxW1 = w1;
                maxW2 = w2;
                maxIndex = i;
            }
        }

        // Weight scales are well-behaved powers of two so we use a power of two to improve our FP8 precision
        float scaleW1 = getFPXWeightScalar(maxW1);
        float scaleW2 = getFPXWeightScalar(maxW2);
        float scaleAct1 = getFPXActScalar(max_input);

        float maxFC1Output = calcMLPVal(max_input, maxIndex) / maxW2;

        std::vector<float> scales_1;
        std::vector<float> scales_2;
        std::vector<float> scales_3;
        if (mUsePerExpertActScale)
        {
            scales_2 = std::vector<float>(mNumExperts);
            for (int i = 0; i < mNumExperts; i++)
            {
                float maxExpertOutput = calcMLPVal(max_input, i) / applyExpertShift(mExpertWDiag2, i);
                float scaleAct2 = getFPXActScalar(maxExpertOutput);
                scales_2[i] = scaleAct2;
            }
        }
        else
        {
            float scaleAct2 = getFPXActScalar(maxFC1Output);
            scales_2 = std::vector<float>(mNumExperts, scaleAct2);
        }

        ASSERT_NE(mExpertFPXScale1, nullptr);
        ASSERT_NE(mExpertFPXScale2, nullptr);
        ASSERT_NE(mExpertFPXScale3, nullptr);

        if (ANY_FP4)
        {
            std::vector<float> scale_global_w1(mNumExperts);
            std::vector<float> scale_global_w2(mNumExperts);

            std::vector<float> scales_0(mUsePerExpertActScale && NVFP4 ? mNumExperts : 1, scaleAct1);
            scales_1 = std::vector<float>(mNumExperts);
            scales_3 = std::vector<float>(mNumExperts);

            for (int i = 0; i < mNumExperts; i++)
            {
                float maxW1 = applyExpertShift(maxW1GatedVal, i);
                float maxW2 = applyExpertShift(mExpertWDiag2, i);
                float scaleW1 = getFPXWeightScalar(maxW1);
                float scaleW2 = getFPXWeightScalar(maxW2);
                scale_global_w1[i] = scaleW1;
                scale_global_w2[i] = scaleW2;

                // TODO Per expert scaling factors
                scales_1[i] = 1.f / (scaleAct1 * scaleW1);
                scales_3[i] = 1.f / (scales_2[i] * scaleW2);
            }

            ASSERT_NE(mExpertFP4ActGlobalScale1, nullptr);
            ASSERT_NE(mExpertFP4WeightGlobalScale1, nullptr);
            ASSERT_NE(mExpertFP4WeightGlobalScale2, nullptr);
            check_cuda_error(cudaMemcpyAsync(mExpertFP4ActGlobalScale1, scales_0.data(),
                scales_0.size() * sizeof(float), cudaMemcpyHostToDevice, mStream->get()));
            check_cuda_error(cudaMemcpyAsync(mExpertFP4WeightGlobalScale1, scale_global_w1.data(),
                scale_global_w1.size() * sizeof(float), cudaMemcpyHostToDevice, mStream->get()));
            check_cuda_error(cudaMemcpyAsync(mExpertFP4WeightGlobalScale2, scale_global_w2.data(),
                scale_global_w2.size() * sizeof(float), cudaMemcpyHostToDevice, mStream->get()));
        }
        else
        {
            mFP8WeightScalar1 = scaleW1;
            mFP8WeightScalar2 = scaleW2;
            scales_1 = std::vector<float>(mNumExperts, 1.f / (scaleW1 * scaleAct1));
            scales_3 = std::vector<float>(mNumExperts);

            for (int i = 0; i < mNumExperts; i++)
            {
                scales_3[i] = 1.f / (scaleW2 * scales_2[i]);
            }
        }

        if (!mUsePerExpertActScale)
        {
            scales_2.resize(1);
        }

        check_cuda_error(cudaMemcpyAsync(mExpertFPXScale1, scales_1.data(), scales_1.size() * sizeof(float),
            cudaMemcpyHostToDevice, mStream->get()));
        check_cuda_error(cudaMemcpyAsync(mExpertFPXScale2, scales_2.data(), scales_2.size() * sizeof(float),
            cudaMemcpyHostToDevice, mStream->get()));
        check_cuda_error(cudaMemcpyAsync(mExpertFPXScale3, scales_3.data(), scales_3.size() * sizeof(float),
            cudaMemcpyHostToDevice, mStream->get()));

        check_cuda_error(cudaStreamSynchronize(mStream->get()));
    }

    void resetOutBuffers()
    {
        auto stream = mStream->get();

        check_cuda_error(cudaStreamSynchronize(stream));

        if (mTpExpertScratch)
            check_cuda_error(cudaMemsetAsync(mTpExpertScratch, 0x0, mTpExpertScratchSize, stream));
        check_cuda_error(cudaMemsetAsync(mFinalOutput, 0x0, mTotalTokens * mHiddenSize * sizeof(OutputType), stream));
        check_cuda_error(cudaMemsetAsync(mSourceToExpandedMap, 0x0, sizeof(int) * mTotalTokens * mK, stream));

        check_cuda_error(cudaStreamSynchronize(stream));
    }

    template <class T>
    auto populateTokens(std::vector<T>& hidden_states)
    {
        if constexpr (MX_QUANT_ACT) // MXFP8_MXFP4
        {
            int const max_order_of_magnitude = 4;
            std::vector<float> base(hidden_states.size());
            std::mt19937 gen(0xD5);
            // Filthy hack to make GELU/SiLu be not introduce large quantization errors
            float min = mIsGated ? 4.f : 0;
            float max = FP8_MAX;
            std::uniform_int_distribution<int> is_negative(0, 10);
            std::uniform_real_distribution<float> dist(min, max);
            std::generate(base.begin(), base.end(),
                [&]()
                {
                    if (is_negative(gen) == 0)
                    {
                        return float(__nv_fp8_e4m3(-dist(gen)));
                    }
                    else
                    {
                        return float(__nv_fp8_e4m3(dist(gen)));
                    }
                });

            // Avoid small values for gated activation
            int adjustment = max_order_of_magnitude / 2;
            for (int i = 0; i < hidden_states.size() / FP4VecSize; i++)
            {
                auto block_scale = mIsGated ? 1.f : exp2f(i % max_order_of_magnitude - adjustment);
                hidden_states[i * FP4VecSize] = T(FP8_MAX * block_scale);
                for (int j = 1; j < FP4VecSize; j++)
                {
                    hidden_states[i * FP4VecSize + j] = T(base[i * FP4VecSize + j] * block_scale);
                }
                mMaxInput = std::max(mMaxInput, FP8_MAX * block_scale);
            }
            return hidden_states;
        }
        // Use the actual template param because we recurse with a different type
        else if constexpr (std::is_same_v<T, SafeFP8>) // FP8, FP8_MXFP4
        {
            // Call the standard setup and then perform the quantization manually
            std::vector<OutputType> internal_states(hidden_states.size());
            populateTokens(internal_states);

            mMaxInput = *std::max_element(internal_states.begin(), internal_states.end());
            float scalar = getFPXActScalar(mMaxInput);
            std::transform(internal_states.begin(), internal_states.end(), hidden_states.begin(),
                [scalar](OutputType in) -> T { return static_cast<T>((float) in * scalar); });
            // Do the reverse transformation since we only have so much precision and this is a pretty broad range
            std::transform(hidden_states.begin(), hidden_states.end(), internal_states.begin(),
                [scalar](T in) -> OutputType { return static_cast<OutputType>(((float) in) / scalar); });
            return internal_states;
        }
        else if constexpr (ACT_FP4) // NVFP4
        {
            float const max_scale = 1.0f;
            mMaxInput = FP4_MAX * max_scale;
            // Excludes 0.75 as this causes increased quantization error
            std::array allowed_values{-6.f, -4.f, -3.f, -2.f, -1.5f, -1.f, 0.0f, 1.f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
            float scale = 1.f / 32.f;
            int stride = FP4VecSize;
            for (int i = 0; i < hidden_states.size(); i += stride)
            {
                hidden_states[i] = FP4_MAX * scale;
                for (int j = 1; j < stride; j++)
                {
                    hidden_states[i + j] = allowed_values[(i / stride + j) % allowed_values.size()] * scale;
                }
                mMaxInput = std::max(mMaxInput, FP4_MAX * scale);
                scale *= 2.f;
                if (scale >= max_scale)
                {
                    scale = 1 / 32.f;
                }
            }
            return hidden_states;
        }
        else // FP16, BF16, FP32, (recurse) FP8
        {
            // Generates numbers in increments of 1/max_order_of_magnitude in the range [0, 1)
            constexpr int max_order_of_magnitude = 256;
            std::vector<int> base(hidden_states.size());
            // Start from the near largest value so we always have some large values even for small hidden sizes
            std::iota(base.begin(), base.end(), max_order_of_magnitude - 4);
            std::mt19937 gen(0xD5);
            std::shuffle(base.begin(), base.end(), gen);
            // Lambda subtracts a small value so we have some < 0 to test the activation for negatives
            std::transform(base.begin(), base.end(), hidden_states.begin(),
                [l = hidden_states.size(), max_order_of_magnitude](auto a) {
                    return T(float(a % max_order_of_magnitude) / float(max_order_of_magnitude))
                        - T(4.f / max_order_of_magnitude);
                });
            mMaxInput = *std::max_element(hidden_states.begin(), hidden_states.end());
            return hidden_states;
        }
    }

    std::pair<std::vector<int>, std::vector<float>> populateRouting(int num_experts, int total_tokens, int k)
    {
        // Scratch buffer for generating random experts
        std::mt19937 gen(0xD5);
        std::vector<int> src_experts(num_experts);
        std::iota(src_experts.begin(), src_experts.end(), 0);

        // Generate a random selection of experts for each token
        std::vector<std::vector<int>> expected_experts_tiered(total_tokens);
        std::generate(expected_experts_tiered.begin(), expected_experts_tiered.end(),
            [&]()
            {
                // Shuffle and pick the first k experts
                std::shuffle(src_experts.begin(), src_experts.end(), gen);
                std::vector<int> selected_experts(k);
                std::copy(src_experts.begin(), src_experts.begin() + k, selected_experts.begin());
                return selected_experts;
            });
        // Flatten the tiered experts into a single vector
        auto expected_experts = flatten(expected_experts_tiered);
        EXPECT_EQ(expected_experts.size(), total_tokens * k);

        // These don't affect control flow so we just use some well behaved scales
        std::vector<float> token_final_scales = {1.f / 8, 5.f / 8, 1.f / 16, 3.f / 4, 3.f / 16};
        token_final_scales = expand(token_final_scales, expected_experts.size());

        return {expected_experts, token_final_scales};
    }

    void runMoEPermute(std::vector<DataType> h_hidden_states, std::vector<int> h_token_selected_experts,
        std::vector<float> h_token_final_scales, int64_t hidden_size, int64_t num_experts, int64_t k,
        MOEParallelismConfig parallelism_config = {}, bool enable_alltoall = false)
    {
        initBuffersPermute(std::move(h_hidden_states), std::move(h_token_selected_experts),
            std::move(h_token_final_scales), hidden_size, num_experts, k, parallelism_config);
        runMoEPermute(parallelism_config, enable_alltoall);
    }

    auto getWeights(MOEParallelismConfig parallelism_config)
    {
        constexpr bool has_fpx_scales = ANY_FPX;
        void* ep_scale_1 = has_fpx_scales ? (void*) mExpertFPXScale1 : (void*) mExpertIntScale1;
        void* ep_scale_2 = has_fpx_scales ? (void*) mExpertFPXScale2 : (void*) mExpertIntScale2;
        void* ep_scale_3 = has_fpx_scales ? mExpertFPXScale3 : nullptr;

        using SliceWeightType = std::conditional_t<WEIGHT_FP4, WeightRawType, WeightStorage>;
        // FP4 accesses the unquantized weight, so WEIGHT_ELEM_PER_BYTE is ignored in this context
        constexpr int SLICED_WEIGHT_ELEM_PER_BYTE = WEIGHT_FP4 ? 1 : WEIGHT_ELEM_PER_BYTE;
        SliceWeightType* slice_weight_1{};
        SliceWeightType* slice_weight_2{};
        if constexpr (WEIGHT_FP4)
        {
            slice_weight_1 = mRawExpertWeight1;
            slice_weight_2 = mRawExpertWeight2;
        }
        else
        {
            slice_weight_1 = mExpertWeight1;
            slice_weight_2 = mExpertWeight2;
        }

        // Handle the case with no parallelism to not require the extra alloc
        if (parallelism_config.tp_size == 1 && parallelism_config.ep_size == 1)
        {
            return std::tuple{(void*) slice_weight_1, (void*) slice_weight_2, mExpertBias1, mExpertBias2, ep_scale_1,
                ep_scale_2, ep_scale_3};
        }

        // Slice weights for EP
        size_t const gated_inter = mInterSize * mGatedMultiplier;
        size_t const experts_per_node = mNumExperts / parallelism_config.ep_size;
        size_t const weight_matrix_size = mHiddenSize * mInterSize * experts_per_node / SLICED_WEIGHT_ELEM_PER_BYTE;
        size_t const bias_fc1_size = gated_inter * experts_per_node;
        size_t const bias_fc2_size = mHiddenSize * experts_per_node;
        size_t const scale1_size = gated_inter * experts_per_node;
        size_t const scale2_size = mHiddenSize * experts_per_node;
        auto* weight1_ptr = slice_weight_1 + weight_matrix_size * mGatedMultiplier * parallelism_config.ep_rank;
        auto* weight2_ptr = slice_weight_2 + weight_matrix_size * parallelism_config.ep_rank;
        auto* bias1_ptr = mUseBias ? mExpertBias1 + bias_fc1_size * parallelism_config.ep_rank : nullptr;
        auto* bias2_ptr = mUseBias ? mExpertBias2 + bias_fc2_size * parallelism_config.ep_rank : nullptr;

        if (INT_QUANT)
        {
            ep_scale_1 = mExpertIntScale1 + scale1_size * parallelism_config.ep_rank;
            ep_scale_2 = mExpertIntScale2 + scale2_size * parallelism_config.ep_rank;
        }
        if constexpr (has_fpx_scales)
        {
            ep_scale_1 = mExpertFPXScale1 + experts_per_node * parallelism_config.ep_rank;
            ep_scale_3 = mExpertFPXScale3 + experts_per_node * parallelism_config.ep_rank;
        }
        if (mUsePerExpertActScale)
        {
            ep_scale_2 = mExpertFPXScale2 + experts_per_node * parallelism_config.ep_rank;
        }

        // Slice weights for TP
        void* scale_1 = ep_scale_1;
        void* scale_2 = ep_scale_2;
        void* scale_3 = ep_scale_3;

        int const tp_size = parallelism_config.tp_size;
        int const tp_rank = parallelism_config.tp_rank;

        size_t const matrix_size = mHiddenSize * mInterSize / tp_size;
        size_t const gated_matrix_size = mHiddenSize * mInterSize * mGatedMultiplier / tp_size;
        size_t const row_size_inter = mInterSize / tp_size;

        auto* weight_1 = reinterpret_cast<SliceWeightType*>(mTpExpertScratch);
        auto* weight_2 = weight_1 + experts_per_node * gated_matrix_size / SLICED_WEIGHT_ELEM_PER_BYTE;
        auto* bias_1
            = reinterpret_cast<BackBoneType*>(weight_2 + experts_per_node * matrix_size / SLICED_WEIGHT_ELEM_PER_BYTE);

        // 2D memcpy just the slices we care about
        // TODO Re-quantize here with matrices divided
        size_t const row_size_1 = matrix_size * sizeof(SliceWeightType) / SLICED_WEIGHT_ELEM_PER_BYTE;
        check_cuda_error(
            cudaMemcpy2DAsync(weight_1, row_size_1, (uint8_t*) weight1_ptr + row_size_1 * tp_rank, row_size_1 * tp_size,
                row_size_1, experts_per_node * mGatedMultiplier, cudaMemcpyDeviceToDevice, mStream->get()));

        size_t const row_size_2 = row_size_inter * sizeof(SliceWeightType) / SLICED_WEIGHT_ELEM_PER_BYTE;
        check_cuda_error(
            cudaMemcpy2DAsync(weight_2, row_size_2, (uint8_t*) weight2_ptr + row_size_2 * tp_rank, row_size_2 * tp_size,
                row_size_2, experts_per_node * mHiddenSize, cudaMemcpyDeviceToDevice, mStream->get()));

        if (mUseBias)
        {
            size_t const row_size_bias = row_size_inter * sizeof(BackBoneType);
            check_cuda_error(cudaMemcpy2DAsync(bias_1, row_size_bias, (uint8_t*) bias1_ptr + row_size_bias * tp_rank,
                row_size_bias * tp_size, row_size_bias, experts_per_node * mGatedMultiplier, cudaMemcpyDeviceToDevice,
                mStream->get()));
        }

        if constexpr (INT_QUANT)
        {
            scale_2 = ep_scale_2;
            size_t const row_size_scale = row_size_inter * sizeof(DataType);
            check_cuda_error(cudaMemcpy2DAsync(scale_1, row_size_scale,
                (uint8_t*) ep_scale_1 + row_size_scale * tp_rank, row_size_scale * tp_size, row_size_scale,
                experts_per_node * mGatedMultiplier, cudaMemcpyDeviceToDevice, mStream->get()));
        }

        bias_1 = mUseBias ? bias_1 : nullptr;

        return std::tuple{(void*) weight_1, (void*) weight_2, bias_1, bias2_ptr, scale_1, scale_2, scale_3};
    }

    auto getFilteredConfigs(int sm, MoeGemmId gemm_id)
    {
        auto tactics = mMoERunner.getTactics(gemm_id);
        if (sm == 89 || sm >= 120)
        {
            // Filter some unsupported configs for L40S
            auto it = std::remove_if(tactics.begin(), tactics.end(),
                [&](auto conf)
                {
                    using tensorrt_llm::cutlass_extensions::CutlassTileConfig;
                    auto checks = std::vector{
                        // Fail for BF16/FP16
                        conf.tile_config_sm80 == CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64,
                        conf.tile_config_sm80 == CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64
                            && conf.stages == 4,
                        // Fail for FP8
                        FP8 && conf.tile_config_sm80 == CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128
                            && conf.stages >= 3,
                    };

                    return std::any_of(checks.begin(), checks.end(), [](auto v) { return v; });
                });
            tactics.erase(it, tactics.end());
        }

        EXPECT_FALSE(tactics.empty());
        return tactics;
    }

    auto selectTacticsForArch(int sm, bool exact_match = false)
    {
        bool is_tma_warp_specialized = sm >= 90 && !INT_QUANT;
        auto epilogue_fusion_type = (is_tma_warp_specialized && mUseFusedFinalizeInternal)
            ? tensorrt_llm::cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE
            : tensorrt_llm::cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::NONE;

        auto smExact = [exact_match, sm](auto& c) { return !exact_match || c.sm_version == sm; };
        auto epilogueMatch = [this](auto& c)
        {
            return c.sm_version < 100 || c.sm_version >= 120
                || c.epilogue_fusion_type
                == tensorrt_llm::cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE
                || (c.sm_version == 100 && this->ANY_FP4) || c.epilogue_schedule == this->mEpilogueSchedule;
        };
        auto epilogueFusionMatch
            = [epilogue_fusion_type](auto& c) { return c.epilogue_fusion_type == epilogue_fusion_type; };

        auto tactics1 = getFilteredConfigs(sm, MoeGemmId::GEMM_1);
        auto tactics2 = getFilteredConfigs(sm, MoeGemmId::GEMM_2);
        auto it1 = std::find_if(tactics1.begin(), tactics1.end(),
            [is_tma_warp_specialized, epilogueMatch, smExact](auto& c)
            { return c.is_tma_warp_specialized == is_tma_warp_specialized && epilogueMatch(c) && smExact(c); });
        auto it2 = std::find_if(tactics2.begin(), tactics2.end(),
            [is_tma_warp_specialized, epilogueMatch, epilogueFusionMatch, smExact](auto& c)
            {
                return c.is_tma_warp_specialized == is_tma_warp_specialized && epilogueFusionMatch(c)
                    && epilogueMatch(c) && smExact(c);
            });
        if (it1 == tactics1.end() || it2 == tactics2.end())
        {
            // Fall back to any tactic
            std::cout << "WARNING: Could not find config for sm version " << sm << std::endl;
            it1 = (it1 == tactics1.end()) ? tactics1.begin() : it1;
            it2 = (it2 == tactics2.end()) ? tactics2.begin() : it2;
        }

        return std::pair(*it1, *it2);
    }

    using ConfigsToTestVec = std::vector<std::pair<tensorrt_llm::cutlass_extensions::CutlassGemmConfig,
        tensorrt_llm::cutlass_extensions::CutlassGemmConfig>>;

    auto getAllTileConfigsToTest()
    {
        if (mOverrideSelectedConfig1 && mOverrideSelectedConfig2)
        {
            return ConfigsToTestVec{std::pair{*mOverrideSelectedConfig1, *mOverrideSelectedConfig2}};
        }

        int sm = getSMVersion();
        bool needs_exact_match = sm == 103 && NVFP4;
        ConfigsToTestVec tactics = {selectTacticsForArch(sm, needs_exact_match)};
        if (sm == 103 && NVFP4)
        {
            // SM103 NVFP4 should also test SM100f kernels
            tactics.push_back(selectTacticsForArch(100, true));
        }
        if (sm >= 90 && !ANY_FPX)
        {
            // SM90+ should also grab some configs for SM80 to test them
            tactics.push_back(selectTacticsForArch(80, true));
        }
        return tactics;
    }

    void runMoEPermute(MOEParallelismConfig parallelism_config, bool enable_alltoall = false)
    {
        // Clear the buffers to blank so we can assume zero if not written
        resetOutBuffers();

        auto [weight1_ptr, weight2_ptr, bias1_ptr, bias2_ptr, scale1_ptr, scale2_ptr, scale3_ptr]
            = getWeights(parallelism_config);

        auto stream = mStream->get();
        auto tactic1 = mInternalSelectedConfig1;
        auto tactic2 = mInternalSelectedConfig2;
        if (!tactic1 || !tactic2)
        {
            int sm = getSMVersion();
            std::tie(tactic1, tactic2) = selectTacticsForArch(sm);
        }
        ASSERT_TRUE(tactic1.has_value());
        ASSERT_TRUE(tactic2.has_value());

        QuantParams quant_params;
        if constexpr (INT_QUANT)
        {
            ASSERT_TRUE(scale1_ptr && scale2_ptr);
            quant_params = QuantParams::Int(scale1_ptr, scale2_ptr);
        }
        else if (FP8)
        {
            ASSERT_TRUE(scale1_ptr && scale2_ptr && scale3_ptr);
            quant_params
                = QuantParams::FP8(static_cast<float const*>(scale1_ptr), static_cast<float const*>(scale2_ptr),
                    static_cast<float const*>(scale3_ptr), nullptr, nullptr, mUsePerExpertActScale);
        }
        else if (ANY_FP4)
        {
            ASSERT_TRUE(mExpertFP4ActGlobalScale1);
            ASSERT_TRUE(mFP4ScalingFactorsW1 && mFP4ScalingFactorsW2);
            ASSERT_TRUE(scale1_ptr && scale2_ptr && scale3_ptr);
            if constexpr (NVFP4 || FP8_MXFP4)
            {
                auto fc1_sf_offset = mUsePerExpertActScale && NVFP4
                    ? mNumExperts / parallelism_config.ep_size * parallelism_config.ep_rank
                    : 0;
                auto constructor = NVFP4 ? &QuantParams::FP4 : &QuantParams::FP8MXFP4;
                quant_params = constructor(mExpertFP4ActGlobalScale1 + fc1_sf_offset, mFP4ScalingFactorsW1,
                    static_cast<float const*>(scale1_ptr), static_cast<float const*>(scale2_ptr), mFP4ScalingFactorsW2,
                    static_cast<float const*>(scale3_ptr), mUsePerExpertActScale && NVFP4, mUsePerExpertActScale);
            }
            else if constexpr (MXFP8_MXFP4)
            {
                quant_params = QuantParams::MXFP8MXFP4(mFP4ScalingFactorsW1, static_cast<float const*>(scale1_ptr),
                    mFP4ScalingFactorsW2, static_cast<float const*>(scale3_ptr));
            }
        }

        if constexpr (WEIGHT_FP4)
        {
            // Dynamically quantize using the proper tp slice
            doFP4Quant(static_cast<WeightRawType const*>(weight1_ptr), mExpertWeight1, mExpertFP4WeightGlobalScale1,
                mFP4ScalingFactorsW1, mHiddenSize, mGatedMultiplier * mInterSize / parallelism_config.tp_size,
                mNumExperts / parallelism_config.ep_size);
            doFP4Quant(static_cast<WeightRawType const*>(weight2_ptr), mExpertWeight2, mExpertFP4WeightGlobalScale2,
                mFP4ScalingFactorsW2, mInterSize / parallelism_config.tp_size, mHiddenSize,
                mNumExperts / parallelism_config.ep_size);
            weight1_ptr = mExpertWeight1;
            weight2_ptr = mExpertWeight2;
        }

        LoraParams lora_params;
        bool const useFp8BlockScales = false;
        bool const minLatencyMode = false;
        MoeMinLatencyParams min_latency_params;
        mMoERunner.setTactic(tactic1, tactic2);
#ifdef USING_OSS_CUTLASS_MOE_GEMM
        mMoERunner.runMoe(mInputTensor, nullptr, true, mSelectedExpert, mTokenFinalScales, weight1_ptr, bias1_ptr,
            ActivationParams(mActType, mSwigluAlpha, mSwigluBeta, mSwigluLimit), weight2_ptr, bias2_ptr, quant_params,
            mTotalTokens, mHiddenSize, mUnpaddedHiddenSize > 0 ? mUnpaddedHiddenSize : mHiddenSize,
            mInterSize / parallelism_config.tp_size, mNumExperts, mK, mWorkspace, mFinalOutput, mSourceToExpandedMap,
            parallelism_config, enable_alltoall, mUseLora, lora_params, useFp8BlockScales, minLatencyMode,
            min_latency_params, stream);
#else
        mMoERunner.runMoe(mInputTensor, nullptr, true, mSelectedExpert, mTokenFinalScales, weight1_ptr, bias1_ptr,
            ActivationParams(mActType, mSwigluAlpha, mSwigluBeta, mSwigluLimit), weight2_ptr, bias2_ptr, quant_params,
            mTotalTokens, mHiddenSize, mInterSize / parallelism_config.tp_size, mNumExperts, mK, mWorkspace,
            mFinalOutput, mSourceToExpandedMap, parallelism_config, mUseLora, lora_params, useFp8BlockScales,
            minLatencyMode, min_latency_params, stream);
#endif

        check_cuda_error(cudaStreamSynchronize(stream));
    }

    template <class T>
    std::vector<T> getDataFromDevice(T const* in, size_t length)
    {
        std::vector<T> data(length);

        auto const stream = mStream->get();
        check_cuda_error(cudaMemcpyAsync(data.data(), in, length * sizeof(T), cudaMemcpyDeviceToHost, stream));
        check_cuda_error(cudaStreamSynchronize(mStream->get()));

        return data;
    }

    auto maskSelectedExpertsForTP(std::vector<int> const& vector, int tp_size, int tp_rank)
    {
        std::vector<int> result;
        int num_experts_per_node = mNumExperts / tp_size;
        std::transform(vector.begin(), vector.end(), std::back_inserter(result),
            [=](int entry)
            {
                if (entry >= num_experts_per_node * tp_rank && entry < num_experts_per_node * (tp_rank + 1))
                    return entry;
                return (int) mNumExperts;
            });
        return result;
    }

    void debugPrint()
    {
#define PRINT_CAST(array, size, cast)                                                                                  \
    do                                                                                                                 \
        if (array)                                                                                                     \
        {                                                                                                              \
            auto data = getDataFromDevice(array, size);                                                                \
            std::cout << #array << ": ";                                                                               \
            for (auto v : data)                                                                                        \
            {                                                                                                          \
                if (cast(v))                                                                                           \
                    std::cout << cast(v) << ", ";                                                                      \
                else                                                                                                   \
                    std::cout << "., ";                                                                                \
            }                                                                                                          \
            std::cout << std::endl;                                                                                    \
        }                                                                                                              \
    while (0)
#define PRINT(array, size) PRINT_CAST(array, size, )

        using WeightPrintType = std::conditional_t<INT_QUANT, uint8_t, WeightStorage>;
        PRINT_CAST((WeightPrintType*) mExpertWeight1,
            mNumExperts * mHiddenSize * mInterSize * mGatedMultiplier / WEIGHT_ELEM_PER_BYTE, float);
        PRINT_CAST(
            (WeightPrintType*) mExpertWeight2, mNumExperts * mHiddenSize * mInterSize / WEIGHT_ELEM_PER_BYTE, float);
        // PRINT_CAST(mRawExpertWeight1, mNumExperts * mHiddenSize * mInterSize * mGatedMultiplier, float);
        // PRINT_CAST(mRawExpertWeight2, mNumExperts * mHiddenSize * mInterSize, float);
        PRINT_CAST(mExpertBias1, mNumExperts * mInterSize * mGatedMultiplier, float);
        PRINT_CAST(mExpertBias2, mNumExperts * mHiddenSize, float);
        PRINT_CAST(mExpertIntScale1, mNumExperts * mInterSize * mGatedMultiplier, float);
        PRINT_CAST(mExpertIntScale2, mNumExperts * mHiddenSize, float);
        PRINT(mFinalOutput, mTotalTokens * mHiddenSize);
        PRINT(mSelectedExpert, mTotalTokens * mK);
        PRINT(mTokenFinalScales, mTotalTokens * mK);
        PRINT_CAST(mInputTensor, mTotalTokens * mHiddenSize, float);
        PRINT(mSourceToExpandedMap, mTotalTokens * mK);

#undef PRINT_CAST
#undef PRINT
    }

    template <class T>
    T actfn(T gate, T linear = T(0.0f), ActivationType act_type = ActivationType::InvalidType)
    {
        if (act_type == ActivationType::InvalidType)
            act_type = mActType;

        switch (act_type)
        {
        case ActivationType::Identity: return gate;
        case ActivationType::Relu: return std::max(gate, T(0.0f));
        case ActivationType::Gelu: return ((std::erf(float(gate) * float(sqrt(0.5))) + 1) * 0.5f * float(gate));
        case ActivationType::Silu: return (float(gate) / (1.f + std::exp(-(gate))));
        case ActivationType::Geglu: return actfn(gate, 0.0f, ActivationType::Gelu) * linear;
        case ActivationType::Swiglu: return actfn(gate, 0.0f, ActivationType::Silu) * linear;
        case ActivationType::SwigluBias:
            linear = std::min(std::max(linear, -mSwigluLimitValue), mSwigluLimitValue);
            gate = std::min(gate, mSwigluLimitValue);
            // silu(gate * alpha) / alpha = gate * sigmoid(gate * alpha)
            return actfn(gate * mSwigluAlphaValue, 0.0f, ActivationType::Silu) / mSwigluAlphaValue
                * (linear + mSwigluBetaValue);
        default: assert(false); return gate;
        }
    }

    float quantAct(float in, float block_max)
    {
        if (MX_QUANT_ACT)
        {
            float scale = std::exp2f(std::ceil(std::log2f(block_max / FP8_MAX)));
            return float(__nv_fp8_e4m3(in / scale)) * scale;
        }
        // TODO Handle NVFP4 too so we can test non-relu actfns
        return in;
    }

    float calcMLPVal(float input, int expert_id, bool final_bias = false, float block_max = 1.f)
    {
        if (expert_id >= mNumExperts)
            return 0;

        float w1_bias = mUseBias ? expert_id : 0.f;
        float activated = 0;
        if (mIsGated)
        {
            float scalar = applyExpertShift(mExpertWDiag1, expert_id);
            float linear = input * scalar + w1_bias;
            float gated_scalar = applyExpertShift(mExpertWDiagGated, expert_id);
            float gated_bias = mUseBias ? w1_bias + 1.f : 0.f;
            float gate = input * gated_scalar + gated_bias;

            activated = actfn(gate, linear);

            block_max = actfn(block_max * gated_scalar + gated_bias, block_max * scalar + w1_bias);
        }
        else
        {
            float scalar = applyExpertShift(mExpertWDiag1, expert_id);
            float fc1 = input * scalar + w1_bias;
            activated = actfn(fc1);

            block_max = actfn(block_max * scalar + w1_bias);
        }

        activated = quantAct(activated, block_max);

        EXPECT_TRUE(mUseBias || !final_bias);
        float result = activated * applyExpertShift(mExpertWDiag2, expert_id) + (float) (final_bias ? expert_id : 0);
        return result;
    }

    float calcMLPValWithFinalBias(float input, int expert_id, float block_max = 1.f)
    {
        return calcMLPVal(input, expert_id, mUseBias, block_max);
    }

    template <class T>
    [[nodiscard]] auto repeat(std::vector<T> const& vector, int64_t repetitions)
    {
        return repeat_blocks(vector, vector.size(), repetitions);
    }

    template <class T>
    [[nodiscard]] auto repeat_blocks(std::vector<T> const& vector, int64_t block_size, int64_t repetitions)
    {
        std::vector<T> output;
        output.reserve(vector.size() * repetitions);
        for (int64_t block = 0; block < vector.size(); block += block_size)
        {
            for (int rep = 0; rep < repetitions; rep++)
            {
                output.insert(output.end(), vector.begin() + block, vector.begin() + block + block_size);
            }
        }
        return output;
    }

    template <class T>
    [[nodiscard]] auto expand(std::vector<T> const& vector, size_t target_size)
    {
        std::vector<T> output;
        output.reserve(target_size);
        for (size_t i = 0; i < target_size; i += vector.size())
        {
            auto len = std::min(vector.size(), target_size - i);
            output.insert(output.end(), vector.begin(), vector.begin() + len);
        }
        return output;
    }

    template <class T>
    [[nodiscard]] auto flatten(std::vector<std::vector<T>> const& vector)
    {
        std::vector<T> output;
        for (auto& v : vector)
        {
            output.insert(output.end(), v.begin(), v.end());
        }
        return output;
    }

    void compareSourceToExpandedMap(std::vector<int> const& expected_experts,
        std::vector<int> const& source_to_expanded_map, std::vector<int> const& reference_map)
    {
        ASSERT_EQ(expected_experts.size(), source_to_expanded_map.size());
        ASSERT_EQ(expected_experts.size(), reference_map.size());
        for (size_t i = 0; i < expected_experts.size(); i++)
        {
            // Note: Only check valid positions (expert ids on the current rank).
            if (expected_experts[i] < mNumExperts)
            {
                int token_id = i / mK;
                int expert_id = i % mK;
                int interleaved_index = expert_id * mTotalTokens + token_id;
                ASSERT_EQ(source_to_expanded_map[interleaved_index], reference_map[interleaved_index])
                    << "Incorrect source_to_expanded_map for token: " << token_id << " expert: " << expert_id;
            }
        }
    }

    void compareFinal(std::vector<int> const& expected_experts, std::vector<float> const& token_final_scales,
        std::vector<OutputType> const& input_data, std::vector<OutputType> final_results = {})
    {
        if (mActType == ActivationType::SwigluBias)
        {
            ASSERT_GT(mMaxInput * std::max(mExpertWDiag1, mExpertWDiagGated), mSwigluLimitValue)
                << "SwigluBias limit values don't change the result";
        }

        ASSERT_EQ(expected_experts.size(), token_final_scales.size());
        ASSERT_EQ(expected_experts.size() / mK, input_data.size() / mHiddenSize);
        if (final_results.empty())
            final_results = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);

        // Use unpadded size for validation if set
        int64_t hidden_size_to_check = mUnpaddedHiddenSize > 0 ? mUnpaddedHiddenSize : mHiddenSize;

        for (int64_t token_id = 0; token_id < mTotalTokens; token_id++)
        {
            float block_max = 1.f;
            // NOTE: When mInterSize < mHiddenSize, those values get zeroed out by fc1 and lost
            for (int64_t hidden_id = 0; hidden_id < std::min(hidden_size_to_check, mInterSize); hidden_id++)
            {
                if (MX_QUANT_ACT && hidden_id % FP4VecSize == 0)
                {
                    block_max = input_data[token_id * mHiddenSize + hidden_id];
                }
                float sum = 0.0f;
                // Loop for the number of times each token is duplicated
                for (int k_idx = 0; k_idx < mK; k_idx++)
                {
                    int selected_expert = expected_experts[token_id * mK + k_idx];
                    float final_scale_value = token_final_scales[token_id * mK + k_idx];

                    float final_value = float(
                        calcMLPValWithFinalBias(static_cast<float>(input_data[token_id * mHiddenSize + hidden_id]),
                            selected_expert, block_max));
                    sum += final_value * final_scale_value;
                }

                ASSERT_NEAR(
                    OutputType{sum}, final_results[token_id * hidden_size_to_check + hidden_id], getTolerance(sum))
                    << "Incorrect final value at for token: " << token_id << "/" << mTotalTokens
                    << " offset: " << hidden_id << " hidden_size: " << mHiddenSize
                    << " unpadded_hidden_size: " << mUnpaddedHiddenSize << " inter_size: " << mInterSize;
            }
        }
    }

    void BasicPermuteTest(
        int k = 1, int64_t hidden_size = DEFAULT_HIDDEN_SIZE, int64_t num_experts = 4, int64_t num_tokens = 3);

    std::vector<int> calcPermuteMapExpertParallel(std::vector<int> const& expected_experts);

    void ExpertParallelTest(int k = 1, int64_t hidden_size = DEFAULT_HIDDEN_SIZE, int64_t num_experts = 4,
        int64_t num_tokens = 3, float inter_size_fraction = 4.0f)
    {
        mInterSizeFraction = inter_size_fraction;
        // 2 experts per rank
        ParallelismTest(k, 1, num_experts / 2, hidden_size, num_experts, num_tokens);
        // 1 expert per rank
        ParallelismTest(k, 1, num_experts, hidden_size, num_experts, num_tokens);

        // 2 expert per rank, enable alltoall optimised finalize
        ParallelismTest(k, 1, num_experts / 2, hidden_size, num_experts, num_tokens, true);
    }

    // Tensor parallel tests default to inter_size_fraction = 1.0f so that all ranks have interesting values (i.e. a
    // diagonal non-square matrix would be all zeros for the last rank)
    void TensorParallelTest(int k = 1, int64_t hidden_size = DEFAULT_HIDDEN_SIZE, int64_t num_experts = 4,
        int64_t num_tokens = 3, float inter_size_fraction = 1.0f)
    {
        // Ensure we dont drop below the minimum alignment
        mInterSizeFraction = std::max(inter_size_fraction, mDeviceMinimumAlignment * 8.0f / hidden_size);
        ParallelismTest(k, 2, 1, hidden_size, num_experts, num_tokens);
        ParallelismTest(k, 4, 1, hidden_size, num_experts, num_tokens);
        ParallelismTest(k, 8, 1, hidden_size, num_experts, num_tokens);
    }

    void MixedParallelTest(int k = 1, int64_t hidden_size = DEFAULT_HIDDEN_SIZE, int64_t num_experts = 4,
        int64_t num_tokens = 3, float inter_size_fraction = 1.0f)
    {
        mInterSizeFraction = std::max(inter_size_fraction, mDeviceMinimumAlignment * 8.0f / hidden_size);

        // 2 experts per rank
        ParallelismTest(k, 2, num_experts / 2, hidden_size, num_experts, num_tokens);
        ParallelismTest(k, 8, num_experts / 2, hidden_size, num_experts, num_tokens);

        // 1 expert per rank
        ParallelismTest(k, 2, num_experts, hidden_size, num_experts, num_tokens);
        ParallelismTest(k, 8, num_experts, hidden_size, num_experts, num_tokens);
    }

    void ParallelismTest(int k = 1, int tp_size = 4, int ep_size = 2, int64_t hidden_size = DEFAULT_HIDDEN_SIZE,
        int64_t num_experts = 4, int64_t num_tokens = 3, bool enable_alltoall = false);
};

template <class WeightParams>
using LargeMixtureOfExpertsTest = MixtureOfExpertsTest<WeightParams>;

template <class DataType_, class WeightType_ = DataType_, class OutputType_ = DataType_, class ActivationScale_ = void,
    class WeightScale_ = void>
struct WeightParams
{
    using DataType = DataType_;
    using WeightType = WeightType_;
    using OutputType = OutputType_;
    using ActivationScale = ActivationScale_;
    using WeightScale = WeightScale_;
};

// TODO Fix int quantized
using Types = ::testing::Types<
#ifdef ENABLE_BF16
    WeightParams<__nv_bfloat16>,
#endif
#ifdef ENABLE_FP8
    WeightParams<SafeFP8, SafeFP8, half>,
#endif
#ifdef ENABLE_FP4
    WeightParams<SafeFP4, SafeFP4, __nv_bfloat16, SafeFP8, SafeFP8>,
    WeightParams<SafeFP8, SafeFP4, __nv_bfloat16, void, SafeFP8E8>,

#ifdef USING_OSS_CUTLASS_MOE_GEMM
    WeightParams<SafeFP8, SafeFP4, __nv_bfloat16, SafeFP8E8, SafeFP8E8>,
#endif
#endif

    WeightParams<half>, WeightParams<float>

    //  , WeightParams<half, uint8_t>, WeightParams<half, cutlass::uint4b_t>

    >;
TYPED_TEST_SUITE(MixtureOfExpertsTest, Types);

// Have a separate test with only FP4, FP8 and half data type because this test is long
using LargeTestTypes = ::testing::Types<
#ifdef ENABLE_FP4
    WeightParams<SafeFP4, SafeFP4, half>,
#endif
#ifdef ENABLE_FP8
    WeightParams<SafeFP8, SafeFP8, half>,
#endif
    WeightParams<half>>;
TYPED_TEST_SUITE(LargeMixtureOfExpertsTest, LargeTestTypes);

template <class TypeParam_>
BufferManager::CudaStreamPtr MixtureOfExpertsTest<TypeParam_>::mStream{};
template <class TypeParam_>
std::unique_ptr<BufferManager> MixtureOfExpertsTest<TypeParam_>::mBufferManager{};
template <class TypeParam_>
int MixtureOfExpertsTest<TypeParam_>::mDeviceCount{};

template <class TypeParam_>
void MixtureOfExpertsTest<TypeParam_>::BasicPermuteTest(
    int k, int64_t hidden_size, int64_t num_experts, int64_t num_tokens)
{
    if (NVFP4 || (MXFP8_MXFP4 && isGatedActivation(mActType)))
    {
        // TODO Remove this when bias + FPX is supported
        mUseBias = false;
    }

    if (NVFP4)
    {
        if (mActType != ActivationType::Relu)
        {
            // FP4 has far too little precision to get any sort of consistency with non-relu actfn
            GTEST_SKIP() << "Skipping FP4 test with non-relu actfn";
            return;
        }
    }

    initLocals(hidden_size, num_experts, k, num_tokens);

    auto test_archs = getAllTileConfigsToTest();
    for (auto [gemm1, gemm2] : test_archs)
    {
        mInternalSelectedConfig1 = gemm1;
        mInternalSelectedConfig2 = gemm2;

        // Input data for each sequence
        std::vector<DataType> hidden_input(hidden_size * num_tokens);
        auto raw_unquant_input = populateTokens(hidden_input);

        auto [expected_experts, token_final_scales] = populateRouting(num_experts, num_tokens, k);

        runMoEPermute(hidden_input, expected_experts, token_final_scales, hidden_size, num_experts, k);
        bool is_finalize_fusion = gemm2.epilogue_fusion_type
            == tensorrt_llm::cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
        bool should_be_deterministic = !is_finalize_fusion || mK < 3;
        if (should_be_deterministic && !mIsLongTest)
        {
            auto first_iter = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);
            mMemsetValue = ~mMemsetValue; // Also check it doesn't depend on uninitialised memory
            runMoEPermute(hidden_input, expected_experts, token_final_scales, hidden_size, num_experts, k);
            auto second_iter = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);
            ASSERT_TRUE(std::equal(first_iter.begin(), first_iter.end(), second_iter.begin()))
                << "Running permute twice does not generate the same results";
        }

        auto proj_map = getDataFromDevice(mSourceToExpandedMap, mTotalTokens * k);
        auto permute_map = calcPermuteMapExpertParallel(expected_experts);
        compareSourceToExpandedMap(expected_experts, proj_map, permute_map);
        compareFinal(expected_experts, token_final_scales, raw_unquant_input);
    }
}

TYPED_TEST(MixtureOfExpertsTest, Permute)
{
    this->BasicPermuteTest();
}

TYPED_TEST(MixtureOfExpertsTest, PermuteK2)
{
    this->BasicPermuteTest(2);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteK3)
{
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteSweepNumTokens)
{
    this->mIsLongTest = true;
    for (int num_tokens : {2, 8, 15, 19, 64, 73, 256})
    {
        this->BasicPermuteTest(1, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);
        this->BasicPermuteTest(2, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);
        this->BasicPermuteTest(3, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);
    }
}

TYPED_TEST(MixtureOfExpertsTest, PermuteSweepNumTokensGeglu)
{
    this->mIsLongTest = true;
    this->mActType = ActivationType::Geglu;
    for (int num_tokens : {2, 8, 15, 19, 64, 73, 256})
    {
        this->BasicPermuteTest(1, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);
        this->BasicPermuteTest(2, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);
        this->BasicPermuteTest(3, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);
    }
}

TYPED_TEST(MixtureOfExpertsTest, PermuteNoBias)
{
    this->mUseBias = false;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteSingletonScale)
{
    if (!this->ANY_FPX)
    {
        GTEST_SKIP() << "Only FPX cares about per-expert act scale";
        return;
    }
    this->mUsePerExpertActScale = false;
    this->BasicPermuteTest(1);
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteGelu)
{
    this->mActType = ActivationType::Gelu;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteSilu)
{
    this->mActType = ActivationType::Silu;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteGeglu)
{
    this->mActType = ActivationType::Geglu;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteSwiglu)
{
    this->mActType = ActivationType::Swiglu;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteSwigluBias)
{
    this->mActType = ActivationType::SwigluBias;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteNoSmemEpilogueSchedule)
{
    if (getSMVersion() < 100 || getSMVersion() >= 120 || (getSMVersion() == 100 && this->NVFP4))
    {
        GTEST_SKIP() << "NoSmem is only supported for SM10x and SM100 without NVFP4";
        return;
    }

    this->mEpilogueSchedule = tensorrt_llm::cutlass_extensions::EpilogueScheduleType::NO_SMEM;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteSwigluNoSmemEpilogueSchedule)
{
    if (getSMVersion() < 100 || getSMVersion() >= 120 || (getSMVersion() == 100 && this->NVFP4))
    {
        GTEST_SKIP() << "NoSmem is only supported for SM10x and SM100 without NVFP4";
        return;
    }

    this->mActType = ActivationType::Swiglu;
    this->mEpilogueSchedule = tensorrt_llm::cutlass_extensions::EpilogueScheduleType::NO_SMEM;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteNonDeterministic)
{
    this->mUseFusedFinalize = true;
    // Just test case 3, cases 1&2 always use the fused paths
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteVerySmall)
{
    for (int i = 1; i <= 3; i++)
    {
        this->BasicPermuteTest(1, this->mDeviceMinimumAlignment * i);
        this->BasicPermuteTest(2, this->mDeviceMinimumAlignment * i);
        this->BasicPermuteTest(3, this->mDeviceMinimumAlignment * i);
    }
}

TYPED_TEST(MixtureOfExpertsTest, PermuteNonPowerOfTwo)
{
    this->BasicPermuteTest(1, this->DEFAULT_HIDDEN_SIZE, 10);
    this->BasicPermuteTest(2, this->DEFAULT_HIDDEN_SIZE, 10);
    this->BasicPermuteTest(3, this->DEFAULT_HIDDEN_SIZE, 10);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteNonPowerOfTwoSwiglu)
{
    this->mActType = ActivationType::Swiglu;
    this->BasicPermuteTest(1, this->DEFAULT_HIDDEN_SIZE, 10);
    this->BasicPermuteTest(2, this->DEFAULT_HIDDEN_SIZE, 10);
    this->BasicPermuteTest(3, this->DEFAULT_HIDDEN_SIZE, 10);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteManyExperts)
{
    this->mIsLongTest = true;
    /* This test is very slow. Only do one k value */
    this->BasicPermuteTest(2, this->mDeviceMinimumAlignment, 512);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteSwigluVerySmall)
{
    this->mActType = ActivationType::Swiglu;
    for (int i = 1; i <= 3; i++)
    {
        this->BasicPermuteTest(1, this->mDeviceMinimumAlignment * i);
        this->BasicPermuteTest(2, this->mDeviceMinimumAlignment * i);
        this->BasicPermuteTest(3, this->mDeviceMinimumAlignment * i);
    }
}

TYPED_TEST(MixtureOfExpertsTest, PermuteMixtral8x7b)
{
    this->mIsLongTest = true;
    this->mUseBias = false;
    this->mActType = ActivationType::Swiglu;
    this->BasicPermuteTest(2, 4096, 8);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteDeepSeekV3)
{
    this->mIsLongTest = true;
    this->mUseBias = false;
    this->mActType = ActivationType::Swiglu;
    size_t hidden_size = 7168;
    size_t inter_size = 2048;
    this->mInterSizeFraction = float(inter_size) / hidden_size;

    if (!this->checkSufficientTestMemory(100, hidden_size, 256, 8))
    {
        GTEST_SKIP() << "Insufficient free memory for test";
    }

    this->BasicPermuteTest(8, hidden_size, 256, 100);
}

TYPED_TEST(MixtureOfExpertsTest, MinimumAlignment)
{
    this->mInterSizeFraction = 1;
    this->BasicPermuteTest(1, this->DEFAULT_HIDDEN_SIZE + this->mDeviceMinimumAlignment);
}

template <class TypeParam_>
std::vector<int> MixtureOfExpertsTest<TypeParam_>::calcPermuteMapExpertParallel(
    std::vector<int> const& expected_experts)
{
    std::vector<int> map(expected_experts.size());
    auto getInterleavedIndex = [this](int i) { return (i % mK) * mTotalTokens + i / mK; };
    int map_idx = 0;
    for (int expert = 0; expert < mNumExperts; expert++)
    {
        for (int i = 0; i < map.size(); i++)
        {
            if (expected_experts[i] == expert)
                map[getInterleavedIndex(i)] = map_idx++;
        }
    }

    return map;
}

template <class TypeParam_>
void MixtureOfExpertsTest<TypeParam_>::ParallelismTest(
    int k, int tp_size, int ep_size, int64_t hidden_size, int64_t num_experts, int64_t num_tokens, bool enable_alltoall)
{
    if (NVFP4 || (MXFP8_MXFP4 && isGatedActivation(mActType)))
    {
        // TODO Remove this when bias + FPX is supported
        mUseBias = false;
    }

    if (NVFP4)
    {
        if (mActType != ActivationType::Relu)
        {
            // FP4 has too little precision to get any sort of consistency with non-relu actfn
            GTEST_SKIP();
            return;
        }
    }

    ASSERT_LE(ep_size, num_experts);
    if (tp_size == 1)
    {
        // Only the first 4 experts are ever used. They should be split across at least 2 ranks
        ASSERT_LT(num_experts / ep_size, 4)
            << "Expert parallelism must have less than 4 experts per rank or the test is ineffective";
    }

    initLocals(hidden_size, num_experts, k, num_tokens);

    auto test_archs = getAllTileConfigsToTest();
    for (auto [gemm1, gemm2] : test_archs)
    {
        mInternalSelectedConfig1 = gemm1;
        mInternalSelectedConfig2 = gemm2;

        std::vector<DataType> hidden_input(hidden_size * num_tokens);
        auto raw_unquant_input = populateTokens(hidden_input);

        auto [expected_experts, token_final_scales] = populateRouting(num_experts, num_tokens, k);

        std::vector<OutputType> results(hidden_input.size(), 0);
        for (int i = 0; i < tp_size; i++)
        {
            for (int j = 0; j < ep_size; j++)
            {
                if (i == 0 && j == 0)
                {
                    // Only need to init the inputs on the first iteration
                    runMoEPermute(hidden_input, expected_experts, token_final_scales, hidden_size, num_experts, k,
                        MOEParallelismConfig{tp_size, i, ep_size, j}, enable_alltoall);
                    bool is_finalize_fusion = gemm2.epilogue_fusion_type
                        == tensorrt_llm::cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
                    bool should_be_deterministic
                        = !is_finalize_fusion || mK < 3 || getSMVersion() < 90 || getSMVersion() >= 120;
                    if (should_be_deterministic && !mIsLongTest)
                    {
                        auto first_iter = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);
                        mMemsetValue = ~mMemsetValue; // Also check it doesn't depend on uninitialised memory
                        runMoEPermute(hidden_input, expected_experts, token_final_scales, hidden_size, num_experts, k,
                            MOEParallelismConfig{tp_size, i, ep_size, j}, enable_alltoall);
                        auto second_iter = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);
                        ASSERT_TRUE(std::equal(first_iter.begin(), first_iter.end(), second_iter.begin()))
                            << "Running permute a second time does not generate the same results";
                    }
                }
                else
                {
                    runMoEPermute(MOEParallelismConfig{tp_size, i, ep_size, j}, enable_alltoall);
                    bool is_finalize_fusion = gemm2.epilogue_fusion_type
                        == tensorrt_llm::cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
                    bool should_be_deterministic
                        = !is_finalize_fusion || mK < 3 || getSMVersion() < 90 || getSMVersion() >= 120;
                    if (should_be_deterministic && !mIsLongTest)
                    {
                        auto first_iter = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);
                        runMoEPermute(MOEParallelismConfig{tp_size, i, ep_size, j}, enable_alltoall);
                        auto second_iter = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);
                        ASSERT_TRUE(std::equal(first_iter.begin(), first_iter.end(), second_iter.begin()))
                            << "Running permute a second time does not generate the same results";
                    }
                }

                auto masked_expected_experts = maskSelectedExpertsForTP(expected_experts, ep_size, j);
                auto proj_map = getDataFromDevice(mSourceToExpandedMap, mTotalTokens * k);
                auto permute_map = calcPermuteMapExpertParallel(masked_expected_experts);
                compareSourceToExpandedMap(masked_expected_experts, proj_map, permute_map);

                // Do the final reduce
                // Note: For enable_alltoall=false, the invalid positions (expert ids outside the current rank) are
                // filled with 0 by mMoERunner.runMoe. For enable_alltoall=true, the invalid positions are untouched by
                // mMoERunner.runMoe, but they are filled with 0 by resetOutBuffers.
                auto iter_results = getDataFromDevice(mFinalOutput, mTotalTokens * hidden_size);
                std::transform(
                    iter_results.cbegin(), iter_results.cend(), results.cbegin(), results.begin(), std::plus<>{});
            }
        }

        compareFinal(expected_experts, token_final_scales, raw_unquant_input, results);
    }
}

#define PARALLEL_TEST_SUITE(ParallelismType)                                                                           \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType)                                                                  \
    {                                                                                                                  \
        this->ParallelismType##Test();                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##K2)                                                              \
    {                                                                                                                  \
        this->ParallelismType##Test(2);                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##K3)                                                              \
    {                                                                                                                  \
        this->ParallelismType##Test(3);                                                                                \
    }                                                                                                                  \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##SweepNumTokens)                                                  \
    {                                                                                                                  \
        this->mIsLongTest = true;                                                                                      \
        for (int num_tokens : {2, 8, 15, 64, 73, 256})                                                                 \
        {                                                                                                              \
            this->ParallelismType##Test(1, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);                                  \
            this->ParallelismType##Test(2, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);                                  \
            this->ParallelismType##Test(3, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);                                  \
        }                                                                                                              \
    }                                                                                                                  \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##SweepNumTokensGeglu)                                             \
    {                                                                                                                  \
        this->mIsLongTest = true;                                                                                      \
        this->mActType = ActivationType::Geglu;                                                                        \
        for (int num_tokens : {2, 8, 15, 64, 73, 256})                                                                 \
        {                                                                                                              \
            this->ParallelismType##Test(1, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);                                  \
            this->ParallelismType##Test(2, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);                                  \
            this->ParallelismType##Test(3, this->DEFAULT_HIDDEN_SIZE, 4, num_tokens);                                  \
        }                                                                                                              \
    }                                                                                                                  \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##NoBias)                                                          \
    {                                                                                                                  \
        this->mUseBias = false;                                                                                        \
        this->ParallelismType##Test();                                                                                 \
        this->ParallelismType##Test(2);                                                                                \
        this->ParallelismType##Test(3);                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##Gelu)                                                            \
    {                                                                                                                  \
        this->mActType = ActivationType::Gelu;                                                                         \
        this->ParallelismType##Test();                                                                                 \
        this->ParallelismType##Test(2);                                                                                \
        this->ParallelismType##Test(3);                                                                                \
    }                                                                                                                  \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##Silu)                                                            \
    {                                                                                                                  \
        this->mActType = ActivationType::Silu;                                                                         \
        this->ParallelismType##Test();                                                                                 \
        this->ParallelismType##Test(2);                                                                                \
        this->ParallelismType##Test(3);                                                                                \
    }                                                                                                                  \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##Geglu)                                                           \
    {                                                                                                                  \
        this->mActType = ActivationType::Geglu;                                                                        \
        this->ParallelismType##Test();                                                                                 \
        this->ParallelismType##Test(2);                                                                                \
        this->ParallelismType##Test(3);                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##Swiglu)                                                          \
    {                                                                                                                  \
        this->mActType = ActivationType::Swiglu;                                                                       \
        this->ParallelismType##Test();                                                                                 \
        this->ParallelismType##Test(2);                                                                                \
        this->ParallelismType##Test(3);                                                                                \
    }                                                                                                                  \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##SwigluBias)                                                      \
    {                                                                                                                  \
        this->mActType = ActivationType::SwigluBias;                                                                   \
        this->ParallelismType##Test();                                                                                 \
        this->ParallelismType##Test(2);                                                                                \
        this->ParallelismType##Test(3);                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##Mixtral8x7b)                                                     \
    {                                                                                                                  \
        this->mIsLongTest = true;                                                                                      \
        this->mUseBias = false;                                                                                        \
        this->mActType = ActivationType::Swiglu;                                                                       \
        this->ParallelismType##Test(2, 4096, 8, 8, 14336.f / 4096.f);                                                  \
    }                                                                                                                  \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##DeepSeekV3)                                                      \
    {                                                                                                                  \
        this->mIsLongTest = true;                                                                                      \
        this->mUseBias = false;                                                                                        \
        this->mActType = ActivationType::Swiglu;                                                                       \
        size_t hidden_size = 7168;                                                                                     \
        size_t inter_size = 2048;                                                                                      \
        float inter_size_fraction = float(inter_size) / hidden_size;                                                   \
                                                                                                                       \
        if (!this->checkSufficientTestMemory(75, hidden_size, 256, 8, true))                                           \
        {                                                                                                              \
            GTEST_SKIP() << "Insufficient free memory for test";                                                       \
        }                                                                                                              \
                                                                                                                       \
        this->ParallelismType##Test(8, hidden_size, 256, 75, inter_size_fraction);                                     \
    }                                                                                                                  \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##GptOss120b)                                                      \
    {                                                                                                                  \
        this->mIsLongTest = true;                                                                                      \
        this->mUseBias = true;                                                                                         \
        this->mActType = ActivationType::Swiglu;                                                                       \
        size_t hidden_size = 2944;                                                                                     \
        size_t inter_size = 2944;                                                                                      \
        if (std::string(#ParallelismType) != "ExpertParallel")                                                         \
        {                                                                                                              \
            /* If with TP, the inter_size should also be padded, */                                                    \
            /* so that after TP split the alignment requirement is still met */                                        \
            inter_size = 3072;                                                                                         \
        }                                                                                                              \
        float inter_size_fraction = float(inter_size) / hidden_size;                                                   \
        this->mUnpaddedHiddenSize = 2880;                                                                              \
                                                                                                                       \
        if (!this->checkSufficientTestMemory(75, hidden_size, 128, 4, true))                                           \
        {                                                                                                              \
            GTEST_SKIP() << "Insufficient free memory for test";                                                       \
        }                                                                                                              \
                                                                                                                       \
        this->ParallelismType##Test(4, hidden_size, 128, 75, inter_size_fraction);                                     \
    }                                                                                                                  \
                                                                                                                       \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##NonPowerOfTwo)                                                   \
    {                                                                                                                  \
        this->ParallelismType##Test(1, this->DEFAULT_HIDDEN_SIZE, 10);                                                 \
        this->ParallelismType##Test(2, this->DEFAULT_HIDDEN_SIZE, 10);                                                 \
        this->ParallelismType##Test(3, this->DEFAULT_HIDDEN_SIZE, 10);                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##NonPowerOfTwoSwiglu)                                             \
    {                                                                                                                  \
        this->mActType = ActivationType::Swiglu;                                                                       \
        this->ParallelismType##Test(1, this->DEFAULT_HIDDEN_SIZE, 10);                                                 \
        this->ParallelismType##Test(2, this->DEFAULT_HIDDEN_SIZE, 10);                                                 \
        this->ParallelismType##Test(3, this->DEFAULT_HIDDEN_SIZE, 10);                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TYPED_TEST(MixtureOfExpertsTest, ParallelismType##ManyExperts)                                                     \
    {                                                                                                                  \
        this->mIsLongTest = true;                                                                                      \
        /* This test is very slow. Only do one k value */                                                              \
        this->ParallelismType##Test(2, this->mDeviceMinimumAlignment, 512, 3, this->ANY_FP4 ? 8.0f : 4.0f);            \
    }

PARALLEL_TEST_SUITE(ExpertParallel)
PARALLEL_TEST_SUITE(TensorParallel)
PARALLEL_TEST_SUITE(MixedParallel)

TYPED_TEST(MixtureOfExpertsTest, ConfigSweep)
{
    this->mIsLongTest = true;
    this->mUseFusedFinalize = true; // True for all cases because we sweep both
    auto genConfigName = [](auto conf) -> std::string
    {
        using namespace tensorrt_llm::cutlass_extensions;
        std::stringstream tactic;
        tactic << "sm" << conf.sm_version << " tactic with tile shape ";
        if (conf.is_tma_warp_specialized)
        {
            tactic << conf.getTileConfigAsInt() << " and cluster shape " << (int) conf.cluster_shape
                   << " dynamic cluster shape " << (int) conf.dynamic_cluster_shape << " mainloop sched "
                   << (int) conf.mainloop_schedule << " epi sched " << (int) conf.epilogue_schedule
                   << " epilogue fusion " << (int) conf.epilogue_fusion_type << " swap ab " << (int) conf.swap_ab;
        }
        else if (conf.tile_config_sm80 != CutlassTileConfig::ChooseWithHeuristic)
        {
            tactic << (int) conf.getTileConfigAsInt() << " and stages " << (int) conf.stages << " split k "
                   << (int) conf.split_k_factor;
        }
        else
        {
            return {};
        }
        return tactic.str();
    };

    auto activation_pool = std::vector{ActivationType::Relu, ActivationType::Swiglu, ActivationType::SwigluBias};
    if (this->NVFP4)
        activation_pool = {ActivationType::Relu};
    auto configs1 = this->getFilteredConfigs(getSMVersion(), MoeGemmId::GEMM_1);
    auto configs2 = this->getFilteredConfigs(getSMVersion(), MoeGemmId::GEMM_2);
    for (auto const activation_type : activation_pool)
    {
        for (auto conf1 : configs1)
        {
            if (conf1.dynamic_cluster_shape != tensorrt_llm::cutlass_extensions::ClusterShape::Undefined
                && conf1.dynamic_cluster_shape != tensorrt_llm::cutlass_extensions::ClusterShape::ClusterShape_4x1x1)
                continue; // To reduce the number of iterations we only test one dynamic cluster shape
            for (auto conf2 : configs2)
            {
                if (conf2.dynamic_cluster_shape != tensorrt_llm::cutlass_extensions::ClusterShape::Undefined
                    && conf2.dynamic_cluster_shape
                        != tensorrt_llm::cutlass_extensions::ClusterShape::ClusterShape_4x1x1)
                    continue; // To reduce the number of iterations we only test one dynamic cluster shape
                auto name1 = genConfigName(conf1);
                auto name2 = genConfigName(conf2);
                if (name1.empty() || name2.empty())
                {
                    FAIL() << "Uninitialised tactic encountered";
                }
                ASSERT_NO_THROW({
                    this->mActType = activation_type;
                    for (auto k : {2, 3})
                    {
                        this->mOverrideSelectedConfig1 = conf1;
                        this->mOverrideSelectedConfig2 = conf2;
                        this->BasicPermuteTest(k, this->mDeviceMinimumAlignment);
                        if (::testing::Test::HasFailure()) // Throw on test failure so we get the print message
                            throw std::runtime_error("Test k=" + std::to_string(k) + " Failed");
                    }
                }) << "Failed\nTactic 1: "
                   << name1 << "\nTactic 2: " << name2 << " and activation type: " << static_cast<int>(activation_type);
            }
        }
    }
}

TYPED_TEST(LargeMixtureOfExpertsTest, PermuteVeryLargeExperts)
{
    this->mIsLongTest = true;

    // Chosen so that hidden_size * inter_size * num_experts >> 2^32, but we can still fit in 80GB for `half`
    // Uses a non-power of two so any integer overflow will have bad alignment
    int64_t hidden_size = 31 * 1024;
    ASSERT_GT(hidden_size * hidden_size * 4, (int64_t) std::numeric_limits<int>::max() + 1ull);

    int64_t k = 2; // Use k=2 so all experts get a value, with high probability
    int64_t num_tokens = 10;
    int64_t num_experts = 4;
    if (!this->checkSufficientTestMemory(num_tokens, hidden_size, num_experts, k))
    {
        GTEST_SKIP() << "Insufficient free memory for test";
    }

    this->BasicPermuteTest(k, hidden_size, num_experts, num_tokens); // 4 x 32k x 128K experts
}

TYPED_TEST(LargeMixtureOfExpertsTest, PermuteVeryLongSequence)
{
    this->mIsLongTest = true;
    this->mUseBias = !this->NVFP4;

    using DataType = typename MixtureOfExpertsTest<TypeParam>::DataType;
    // Sequence * hidden size > INT32_MAX
    int64_t hidden_size = 2048ll;
    int64_t num_experts = 4;
    int64_t k = 1;
    int64_t tokens_to_test = 100;
    int64_t num_tokens = 2ll * 1024ll * 1024ll + tokens_to_test + 1ll;
    ASSERT_GT(hidden_size * (num_tokens - tokens_to_test), (uint64_t) std::numeric_limits<uint32_t>::max() + 1ull);

    if (!this->checkSufficientTestMemory(num_tokens, hidden_size, num_experts, k))
    {
        GTEST_SKIP() << "Insufficient free memory for test";
    }

    std::vector<DataType> hidden_states(hidden_size * num_tokens);
    this->mMaxInput = 1.f; // Any arbitrary non-zero value

    // All tokens to expert 0, so we catch the case where an expert has more than 2^32 tokens
    std::vector<int> token_selected_experts(num_tokens, 0);
    std::vector<float> token_final_scales(num_tokens, 1.f);
    // Override the first few tokens to go to different experts.
    // This covers the regression case where an overflow only impacts one of the last experts
    // In particular the case when there are more than 2^32 elements before the last expert
    for (int i = 0; i < tokens_to_test; i++)
    {
        token_selected_experts[i] = i % num_experts;
    }

    this->initLocals(hidden_size, num_experts, k, num_tokens);
    this->runMoEPermute(hidden_states, token_selected_experts, token_final_scales, hidden_size, num_experts, k);

    // Just look at the first few tokens
    this->mTotalTokens = tokens_to_test;

    token_selected_experts.resize(this->mTotalTokens * this->mK);
    token_final_scales.resize(this->mTotalTokens * this->mK);
    hidden_states.resize(hidden_size * this->mTotalTokens);

    // Create a default vector for the reference outputs of the correct type for FP8
    std::vector<typename TypeParam::OutputType> unquant_states(this->mTotalTokens * hidden_size);
    this->compareFinal(token_selected_experts, token_final_scales, unquant_states);
}

template <class T>
constexpr static auto typeToDtypeID()
{
    if constexpr (std::is_same_v<T, SafeFP8>)
    {
        return nvinfer1::DataType::kFP8;
    }
    else if constexpr (std::is_same_v<T, SafeFP4>)
    {
        return nvinfer1::DataType::kFP4;
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
        return nvinfer1::DataType::kINT8;
    }
    else if constexpr (std::is_same_v<T, cutlass::uint4b_t>)
    {
        return nvinfer1::DataType::kINT4;
    }
    else if constexpr (std::is_same_v<T, nv_bfloat16>)
    {
        return nvinfer1::DataType::kBF16;
    }
    else if constexpr (std::is_same_v<T, half>)
    {
        return nvinfer1::DataType::kHALF;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return nvinfer1::DataType::kFLOAT;
    }
    else
    {
        // sizeof(T) to make the static assert dependent on the template
        static_assert(sizeof(T) == 0, "Unrecognised data type");
    }
}

TYPED_TEST(MixtureOfExpertsTest, RunProfiler)
{
    auto test_func = [this](GemmProfilerBackend::GemmToProfile gemm_to_profile)
    {
        int64_t num_experts = 4;
        int64_t k = 2;

        GemmProfilerBackend backend;
#ifdef USING_OSS_CUTLASS_MOE_GEMM
        backend.init(this->mMoERunner, gemm_to_profile, typeToDtypeID<typename TypeParam::DataType>(),
            typeToDtypeID<typename TypeParam::WeightType>(), typeToDtypeID<typename TypeParam::OutputType>(),
            num_experts, k, this->DEFAULT_HIDDEN_SIZE, this->DEFAULT_HIDDEN_SIZE, this->DEFAULT_HIDDEN_SIZE * 4,
            this->mGroupSize, ActivationType::Geglu, false, this->mUseLora, /*min_latency_mode=*/false,
            /*need_weights=*/true, MOEParallelismConfig{}, /*enable_alltoall=*/false);
#else
        backend.init(this->mMoERunner, gemm_to_profile, typeToDtypeID<typename TypeParam::DataType>(),
            typeToDtypeID<typename TypeParam::WeightType>(), typeToDtypeID<typename TypeParam::OutputType>(),
            num_experts, k, this->DEFAULT_HIDDEN_SIZE, this->DEFAULT_HIDDEN_SIZE * 4, this->mGroupSize,
            ActivationType::Geglu, false, this->mUseLora, /*min_latency_mode=*/false,
            /*need_weights=*/true, MOEParallelismConfig{});
#endif

        auto ws_size = backend.getWorkspaceSize(128);

        auto workspace = this->template allocBuffer<char>(ws_size);

        for (int64_t num_tokens : {1, 128})
        {
            backend.prepare(num_tokens, workspace, /*expert_weights=*/nullptr, this->mStream->get());
            for (auto const& tactic : this->getAllTileConfigsToTest())
            {
                backend.runProfiler(num_tokens,
                    gemm_to_profile == GemmProfilerBackend::GemmToProfile::GEMM_1 ? tactic.first : tactic.second,
                    workspace, /*expert_weights=*/nullptr, this->mStream->get());
            }
        }
        ASSERT_EQ(cudaStreamSynchronize(this->mStream->get()), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    };
    ASSERT_NO_THROW(test_func(GemmProfilerBackend::GemmToProfile::GEMM_1)) << "Failed to profile GEMM_1";
    ASSERT_NO_THROW(test_func(GemmProfilerBackend::GemmToProfile::GEMM_2)) << "Failed to profile GEMM_2";
}

// Data types don't matter for the distribution
using MixtureOfExpertsProfilerTest = MixtureOfExpertsTest<WeightParams<half, half>>;

TEST_F(MixtureOfExpertsProfilerTest, TestGeneratedProfilerDistribution)
{
    //    int64_t num_tokens = 128;
    int64_t num_experts = 8;
    int64_t k = 2;

    GemmProfilerBackend backend;

    // We need to test different EP values to ensure the tokens are properly assigned
    for (int64_t num_tokens : {1, 128})
    {
        int64_t expanded_num_tokens = num_tokens * k;
        for (int ep : {1, 4, 8})
        {
#ifdef USING_OSS_CUTLASS_MOE_GEMM
            backend.init(this->mMoERunner, GemmProfilerBackend::GemmToProfile::GEMM_1, nvinfer1::DataType::kHALF,
                nvinfer1::DataType::kHALF, nvinfer1::DataType::kHALF, num_experts, k, 1024, 1024, 4096, mGroupSize, {},
                false, mUseLora, /*min_latency_mode=*/false, /*need_weights=*/true, MOEParallelismConfig{1, 0, ep, 0},
                /*enable_alltoall=*/false);
#else
            backend.init(this->mMoERunner, GemmProfilerBackend::GemmToProfile::GEMM_1, nvinfer1::DataType::kHALF,
                nvinfer1::DataType::kHALF, nvinfer1::DataType::kHALF, num_experts, k, 1024, 4096, mGroupSize, {}, false,
                mUseLora, /*min_latency_mode=*/false, /*need_weights=*/true, MOEParallelismConfig{1, 0, ep, ep - 1});
#endif

            auto ws_size = backend.getWorkspaceSize(num_tokens);
            auto workspace = this->allocBuffer<char>(ws_size);
            int64_t num_experts_per_node = num_experts / ep;

            backend.prepare(num_tokens, workspace, /*expert_weights=*/nullptr, mStream->get());

            auto workspaces = backend.getProfilerWorkspaces(num_tokens, getSMVersion() >= 90 && getSMVersion() < 120);
#define GET_WS_PTR(type, name) auto* name = reinterpret_cast<type>(workspace + workspaces.at(#name).second)

            GET_WS_PTR(int64_t*, expert_first_token_offset);
            GET_WS_PTR(int*, unpermuted_row_to_permuted_row);
            GET_WS_PTR(int*, permuted_row_to_unpermuted_row);
#ifdef USING_OSS_CUTLASS_MOE_GEMM
            GET_WS_PTR(int*, token_selected_experts);
#else
            GET_WS_PTR(int*, unpermuted_selected_experts);
#endif
#undef GET_WS_PTR

            for (int sample = 0; sample < backend.NUM_ROUTING_SAMPLES; sample++)
            {
                auto host_expert_first_token_offset_size = getDataFromDevice(
                    expert_first_token_offset + sample * (num_experts_per_node + 1), num_experts_per_node + 1);
                auto host_unpermuted_row_to_permuted_row_map = getDataFromDevice(
                    unpermuted_row_to_permuted_row + sample * expanded_num_tokens, expanded_num_tokens);
                auto host_permuted_row_to_unpermuted_row_map = getDataFromDevice(
                    permuted_row_to_unpermuted_row + sample * expanded_num_tokens, expanded_num_tokens);
#ifdef USING_OSS_CUTLASS_MOE_GEMM
                auto host_token_selected_experts
                    = getDataFromDevice(token_selected_experts + sample * expanded_num_tokens, expanded_num_tokens);
#else
                auto host_token_selected_experts = getDataFromDevice(
                    unpermuted_selected_experts + sample * expanded_num_tokens, expanded_num_tokens);
#endif

                std::vector<int64_t> calculated_routing_values(num_experts_per_node + 1, 0);
                int skipped = 0;
                for (auto v : host_token_selected_experts)
                {
#ifndef USING_OSS_CUTLASS_MOE_GEMM
                    ASSERT_TRUE(v < num_experts_per_node || (v == num_experts_per_node && ep > 1))
                        << "v " << v << " num_experts_per_node " << num_experts_per_node << " ep " << ep;
#endif
                    if (v < num_experts_per_node)
                    {
                        calculated_routing_values[v]++;
                    }
                    else
                    {
                        skipped++;
                    }
                }

                if (num_tokens > 1)
                {
                    // Check tokens are distributed between all EP ranks
                    // Statistically possible, but so unlikely that it should be considered a bug
                    ASSERT_TRUE(ep == 1 || skipped > 0);
                    // Check all experts get some tokens
                    ASSERT_EQ(std::count(calculated_routing_values.begin(), calculated_routing_values.end() - 1, 0), 0);

                    float p = 1.f / num_experts;
                    float variance = expanded_num_tokens * p * (1 - p);
                    float stddev = sqrt(variance);
                    float mean = expanded_num_tokens * p;
                    for (int i = 0; i < num_experts_per_node; i++)
                    {
                        // All values should be within three standard deviations of the mean
                        // 99.7% of values should fall within this range.
                        // We have NUM_ROUTING_SAMPLES * (8 + 2 + 1) = 176 cases so this is unlikely
                        // If the test changes to have a much larger number of cases this will need revisited
                        EXPECT_LE(abs(calculated_routing_values[i] - mean), 3 * stddev)
                            << "Expert " << i << " for sample " << sample << " has unbalanced token count "
                            << calculated_routing_values[i] << " vs mean value " << mean << " with standard deviation "
                            << stddev;
                    }
                }
                ASSERT_EQ(host_expert_first_token_offset_size.back(), expanded_num_tokens - skipped)
                    << "Num expanded tokens " << expanded_num_tokens << " num skipped " << skipped;

                std::exclusive_scan(calculated_routing_values.begin(), calculated_routing_values.end(),
                    calculated_routing_values.begin(), 0);
                ASSERT_TRUE(std::equal(calculated_routing_values.begin(), calculated_routing_values.end(),
                    host_expert_first_token_offset_size.begin()));

                std::fill(calculated_routing_values.begin(), calculated_routing_values.end(), 0);
                for (int64_t token_idx = 0; token_idx < num_tokens; token_idx++)
                {
                    for (int64_t k_idx = 0; k_idx < k; k_idx++)
                    {
                        int64_t idx = token_idx * k + k_idx;
                        int64_t expert_idx = host_token_selected_experts[idx];

#ifdef USING_OSS_CUTLASS_MOE_GEMM
                        if (expert_idx < num_experts_per_node)
#else
                        if (expert_idx < num_experts)
#endif
                        {
                            int64_t unpermuted_row = k_idx * num_tokens + token_idx;
                            int64_t permuted_row = host_expert_first_token_offset_size[expert_idx]
                                + calculated_routing_values[expert_idx];

                            ASSERT_EQ(host_unpermuted_row_to_permuted_row_map[unpermuted_row], permuted_row);
                            ASSERT_EQ(host_permuted_row_to_unpermuted_row_map[permuted_row], unpermuted_row);

                            calculated_routing_values[expert_idx]++;
                        }
                    }
                }
            }
        }
    }
}
