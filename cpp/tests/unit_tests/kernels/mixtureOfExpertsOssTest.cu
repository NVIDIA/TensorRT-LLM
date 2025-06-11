#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

//  Temporary opend-sourced version. Will be daleted when open-sourced moe_gemm support MXFP4

// #if defined(USING_OSS_CUTLASS_MOE_GEMM)
#if true
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
#else
#include "moe_kernels.h"
#endif

#include "tensorrt_llm/runtime/bufferManager.h"

#include <tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h>
#include <tensorrt_llm/kernels/quantization.h>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;
using LoraParams = tensorrt_llm::kernels::LoraParams;
// #if defined(USING_OSS_CUTLASS_MOE_GEMM)
#if true
namespace kernels = tensorrt_llm::kernels::cutlass_kernels;
using GemmProfilerBackend = tensorrt_llm::kernels::cutlass_kernels::GemmProfilerBackend;
using MoeMinLatencyParams = tensorrt_llm::kernels::cutlass_kernels::MoeMinLatencyParams;
using QuantParams = tensorrt_llm::kernels::cutlass_kernels::QuantParams;
using MOEParallelismConfig = tensorrt_llm::kernels::cutlass_kernels::MOEParallelismConfig;
using ActivationType = tensorrt_llm::kernels::cutlass_kernels::ActivationType;
using TmaWarpSpecializedGroupedGemmInput = tensorrt_llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;
using tensorrt_llm::kernels::cutlass_kernels::isGatedActivation;
#else
namespace kernels = tensorrt_llm::kernels;
using GemmProfilerBackend = tensorrt_llm::kernels::GemmProfilerBackend;
using MoeMinLatencyParams = tensorrt_llm::kernels::MoeMinLatencyParams;
using QuantParams = tensorrt_llm::kernels::QuantParams;
using MOEParallelismConfig = tensorrt_llm::kernels::MOEParallelismConfig;
using ActivationType = tensorrt_llm::ActivationType;
using TmaWarpSpecializedGroupedGemmInput = tensorrt_llm::TmaWarpSpecializedGroupedGemmInput;
using tensorrt_llm::isGatedActivation;
#endif
namespace cutlass_kernels = tensorrt_llm::kernels::cutlass_kernels;

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
#else
using SafeFP8 = void;
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
#else
using SafeFP4 = void;
#endif

static_assert(sizeof_bits<SafeFP4>::value == 4, "SafeFP4 is not 4 bits");

template <class TypeTuple_>
class MixtureOfExpertsTest : public ::testing::Test
{
protected:
    using GemmDataType = typename TypeTuple_::DataType;
    using WeightType = typename TypeTuple_::WeightType;
    using OutputType = typename TypeTuple_::OutputType;
    constexpr static bool INT4 = std::is_same_v<WeightType, cutlass::uint4b_t>;
    constexpr static bool FP8 = std::is_same_v<GemmDataType, SafeFP8>;
    constexpr static bool FP4 = std::is_same_v<GemmDataType, SafeFP4>;
    constexpr static bool INT_QUANT = !std::is_same_v<GemmDataType, WeightType>;
    constexpr static int WEIGHT_ELEM_PER_BYTE = (INT4 || FP4) ? 2 : 1;
    using InputType = std::conditional_t<FP4, OutputType, GemmDataType>;
    using WeightStorage = std::conditional_t<WEIGHT_ELEM_PER_BYTE == 2, uint8_t, WeightType>;
    constexpr static int64_t HIDDEN_SIZE_MULTIPLIER = 16;
    constexpr static int64_t MINIMUM_BYTE_ALIGNMENT = 64;
    constexpr static int64_t MINIMUM_ALIGNMENT = MINIMUM_BYTE_ALIGNMENT * WEIGHT_ELEM_PER_BYTE / sizeof(WeightStorage);
    constexpr static int64_t DEFAULT_HIDDEN_SIZE = HIDDEN_SIZE_MULTIPLIER * MINIMUM_ALIGNMENT;

    // FP4 uses the unquantized data type for inputs and quantizes on the fly
    using DataType = std::conditional_t<FP4, OutputType, GemmDataType>;

    static BufferManager::CudaStreamPtr mStream;
    static std::unique_ptr<BufferManager> mBufferManager;
    static int mDeviceCount;

    std::vector<BufferManager::IBufferPtr> managed_buffers;
    DataType* mInputTensor{};

    int64_t mHiddenSize{};
    int64_t mNumExperts{};
    int64_t mK{};

    float getTolerance(float scale = 1.f)
    {
        bool loose_fp8 = mActType != ActivationType::Relu;
        float tol = std::is_same_v<WeightType, uint8_t>     ? 0.1
            : std::is_same_v<WeightType, cutlass::uint4b_t> ? 0.1
            : std::is_same_v<GemmDataType, float>           ? 0.001
            : std::is_same_v<GemmDataType, half>            ? 0.005
            : std::is_same_v<GemmDataType, __nv_bfloat16>   ? 0.05
            : std::is_same_v<GemmDataType, SafeFP8>         ? (loose_fp8 ? 0.06 : 0.001)
            : std::is_same_v<GemmDataType, SafeFP4>         ? 0.05
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
        bool should_skip_unsupported_fp4 = (getSMVersion() < 100 || getSMVersion() >= 120) && FP4;
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
    }

    void TearDown() override
    {
        managed_buffers.clear();
        ASSERT_EQ(cudaStreamSynchronize(mStream->get()), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }

    void initWeights(DataType* buffer, int64_t w, int64_t h, float base, float scalar)
    {
        dim3 block(16, 16, 1);
        dim3 grid(divUp(w, block.x), divUp(h, block.y), mNumExperts);
        initWeightsKernel<DataType><<<grid, block, 0, mStream->get()>>>(buffer, w, h, base, scalar);
    }

    void initBias(DataType* buffer, int64_t w)
    {
        dim3 block(256, 1, 1);
        dim3 grid(divUp(w, block.x), mNumExperts);
        initBiasToExpertIdKernel<DataType><<<grid, block, 0, mStream->get()>>>(buffer, w);
    }

    void initWeightsGated(DataType* buffer, int64_t w, int64_t h, float base_1, float base_2, float scalar)
    {
        if (!mIsGated)
            return initWeights(buffer, w, h, base_1, scalar);

        h /= 2;
        dim3 block(16, 16, 1);
        dim3 grid(divUp(w, block.x), divUp(h, block.y), mNumExperts);
        initWeightsGatedKernel<DataType><<<grid, block, 0, mStream->get()>>>(buffer, w, h, base_1, base_2, scalar);
    }

    void initBiasGated(DataType* buffer, int64_t w)
    {
        if (!mIsGated)
            return initBias(buffer, w);

        w /= 2;
        dim3 block(256, 1, 1);
        dim3 grid(divUp(w, block.x), mNumExperts);
        initBiasToExpertIdGatedKernel<DataType><<<grid, block, 0, mStream->get()>>>(buffer, w);
    }

    kernels::CutlassMoeFCRunner<GemmDataType, WeightType, OutputType, InputType> mMoERunner{};
    char* mWorkspace{};
    int* mSelectedExpert;
    float* mTokenFinalScales{};
    DataType* mRawExpertWeight1{};
    DataType* mRawExpertWeight2{};
    WeightStorage* mExpertWeight1{};
    WeightStorage* mExpertWeight2{};
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
    constexpr static int FP4VecSize = TmaWarpSpecializedGroupedGemmInput::BlockScaleVectorSize;
    ElementSF* mFP4ScalingFactorsW1 = nullptr;
    ElementSF* mFP4ScalingFactorsW2 = nullptr;

    DataType* mExpertBias1{};
    DataType* mExpertBias2{};

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

    bool mIsGated = false;
    int64_t mGatedMultiplier = 1;
    int64_t mGroupSize = -1;

    ActivationType mActType = ActivationType::Relu;

    float mSparseMixerEpsilon = 0.2f;

    // Default this to true. This only matters for K>2, and so by doing this we will test the fused and unfused paths
    bool mUseDeterminsiticHopperReduce = true;

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
        size_t const weight_size = hidden_size * (hidden_size * mInterSizeFraction) * num_experts
            * sizeof(WeightStorage) * num_gemms / WEIGHT_ELEM_PER_BYTE;
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

        size_t const memory_pool_free_mem_size = mBufferManager->memoryPoolFree();
        auto const [freeMem, totalMem] = tensorrt_llm::common::getDeviceMemoryInfo(false);
        float const freeMemBuffer = 0.9f; // Add some buffer so we aren't completely pushing the limits
        std::cout << "Free memory is: " << freeMem << ", memory pool size is: " << memory_pool_free_mem_size
                  << ", required memory is: " << total_size << ", device total memory capacity: " << totalMem
                  << std::endl;
        return (freeMem + memory_pool_free_mem_size) * freeMemBuffer >= total_size;
    }

    void initBuffersPermute(std::vector<DataType> h_hidden_states, std::vector<int> h_token_selected_experts,
        std::vector<float> h_token_final_scales, int64_t hidden_size, int64_t num_experts, int64_t k,
        MOEParallelismConfig parallelism_config)
    {
        managed_buffers.clear();

        mMoERunner.use_deterministic_hopper_reduce_ = k > 2 && mUseDeterminsiticHopperReduce;

        mHiddenSize = hidden_size;
        mInterSize = hidden_size * mInterSizeFraction;
        mNumExperts = num_experts;
        mK = k;
        mIsGated = isGatedActivation(mActType);
        mGatedMultiplier = mIsGated ? 2 : 1;
        auto const gated_inter = mInterSize * mGatedMultiplier;

        mTotalTokens = h_hidden_states.size() / hidden_size;
        EXPECT_EQ(h_token_selected_experts.size(), mTotalTokens * mK);
        EXPECT_EQ(h_token_final_scales.size(), mTotalTokens * mK);

        bool const useDeepseek = false;
        size_t workspace_size = mMoERunner.getWorkspaceSize(mTotalTokens, mHiddenSize, mInterSize, mNumExperts, mK,
            mActType, parallelism_config, mUseLora, useDeepseek, false, mUsePrequantScale);

        auto const stream = mStream->get();

        mWorkspace = allocBuffer<char>(workspace_size);

        size_t const expert_matrix_size = mNumExperts * mHiddenSize * mInterSize;

        mRawExpertWeight1 = allocBuffer<DataType>(expert_matrix_size * mGatedMultiplier);
        mRawExpertWeight2 = allocBuffer<DataType>(expert_matrix_size);

        size_t const experts_per_node = mNumExperts / parallelism_config.ep_size;
        int const moe_parallel_size = parallelism_config.tp_size * parallelism_config.ep_size;

        mTpExpertScratchSize = expert_matrix_size * mGatedMultiplier / moe_parallel_size;
        mTpExpertScratchSize += expert_matrix_size / moe_parallel_size;

        mExpertBias1 = nullptr;
        mExpertBias2 = nullptr;
        if (mUseBias)
        {
            // Allow space for the slice of bias1 in the scratch
            mTpExpertScratchSize += experts_per_node * gated_inter / parallelism_config.tp_size;
            mExpertBias1 = allocBuffer<DataType>(mNumExperts * gated_inter);
            mExpertBias2 = allocBuffer<DataType>(mNumExperts * mHiddenSize);

            check_cuda_error(cudaMemsetAsync(mExpertBias1, 0x0, mNumExperts * gated_inter * sizeof(DataType), stream));
            check_cuda_error(cudaMemsetAsync(mExpertBias2, 0x0, mNumExperts * mHiddenSize * sizeof(DataType), stream));
        }

        if constexpr (INT_QUANT)
        {
            mExpertWeight1 = allocBuffer<WeightStorage>(expert_matrix_size * mGatedMultiplier / WEIGHT_ELEM_PER_BYTE);
            mExpertWeight2 = allocBuffer<WeightStorage>(expert_matrix_size / WEIGHT_ELEM_PER_BYTE);

            mTpExpertScratchSize += experts_per_node * gated_inter / parallelism_config.tp_size;
            mExpertIntScale1 = allocBuffer<DataType>(mNumExperts * gated_inter);
            mExpertIntScale2 = allocBuffer<DataType>(mNumExperts * mHiddenSize);
        }
        else if constexpr (FP4)
        {
            mExpertWeight1 = allocBuffer<WeightStorage>(expert_matrix_size * mGatedMultiplier / WEIGHT_ELEM_PER_BYTE);
            mExpertWeight2 = allocBuffer<WeightStorage>(expert_matrix_size / WEIGHT_ELEM_PER_BYTE);

            size_t const padded_fc1_size = mNumExperts * mHiddenSize
                * cute::ceil_div(
                    mInterSize * mGatedMultiplier, TmaWarpSpecializedGroupedGemmInput::MinNumRowsAlignmentFP4)
                * TmaWarpSpecializedGroupedGemmInput::MinNumRowsAlignmentFP4;
            size_t const padded_fc2_size = mNumExperts * mInterSize
                * cute::ceil_div(mHiddenSize, TmaWarpSpecializedGroupedGemmInput::MinNumRowsAlignmentFP4)
                * TmaWarpSpecializedGroupedGemmInput::MinNumRowsAlignmentFP4;
            mFP4ScalingFactorsW1 = allocBuffer<ElementSF>(padded_fc1_size / FP4VecSize);
            mFP4ScalingFactorsW2 = allocBuffer<ElementSF>(padded_fc2_size / FP4VecSize);
        }
        else
        {
            mExpertWeight1 = mRawExpertWeight1;
            mExpertWeight2 = mRawExpertWeight2;
        }

        if constexpr (FP8 || FP4)
        {
            // FP4 uses the same logic as FP8 to generate the global scales
            mExpertFPXScale1 = allocBuffer<float>(mNumExperts);
            mExpertFPXScale2 = allocBuffer<float>(1);
            mExpertFPXScale3 = allocBuffer<float>(mNumExperts);

            if (FP4)
            {
                mExpertFP4ActGlobalScale1 = allocBuffer<float>(1);
                mExpertFP4WeightGlobalScale1 = allocBuffer<float>(mNumExperts);
                mExpertFP4WeightGlobalScale2 = allocBuffer<float>(mNumExperts);
            }

            EXPECT_NE(mMaxInput, 0.0f);
            initFPQuantScales(mMaxInput);
        }

        if (parallelism_config.tp_size > 1 || parallelism_config.ep_size > 1)
        {
            mTpExpertScratch = allocBuffer<DataType>(mTpExpertScratchSize);
        }

        mTokenFinalScales = allocBuffer<float>(mTotalTokens * mK);
        mSelectedExpert = allocBuffer<int>(mTotalTokens * mK);

        mInputTensor = allocBuffer<DataType>(mTotalTokens * mHiddenSize);
        mFinalOutput = allocBuffer<OutputType>(mTotalTokens * mHiddenSize);

        mSourceToExpandedMap = allocBuffer<int>(mTotalTokens * mK);

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

    void doFP4Quant(DataType const* raw_weights, WeightStorage* quant_weights, float const* global_scales,
        ElementSF* scaling_factors, int in_shape, int out_shape, int num_experts)
    {
        int const mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
        int padded_stride = cute::ceil_div(out_shape, TmaWarpSpecializedGroupedGemmInput::MinNumRowsAlignmentFP4)
            * TmaWarpSpecializedGroupedGemmInput::MinNumRowsAlignmentFP4;
        check_cuda_error(cudaMemsetAsync(scaling_factors, 0x0,
            num_experts * padded_stride * cutlass::ceil_div(in_shape, FP4VecSize) * sizeof(ElementSF), mStream->get()));
        for (int i = 0; i < num_experts; i++)
        {
            auto* weight_start = raw_weights + i * in_shape * out_shape;
            auto* quant_weight_start = quant_weights + i * in_shape * out_shape / WEIGHT_ELEM_PER_BYTE;
            auto* scaling_factor_start
                = scaling_factors + i * (int64_t) padded_stride * cutlass::ceil_div(in_shape, FP4VecSize);

            tensorrt_llm::kernels::invokeFP4Quantization(out_shape, in_shape, weight_start, global_scales + i,
                reinterpret_cast<int64_t*>(quant_weight_start), reinterpret_cast<int32_t*>(scaling_factor_start), false,
                tensorrt_llm::FP4QuantizationSFLayout::SWIZZLED, mMultiProcessorCount, mStream->get());
        }
    }

    constexpr static float getFP8Scalar(float in)
    {
        if (FP8)
            return FP8_MAX / in;
        if (FP4)
            // We need to represent the block SF using FP8, so the largest value should be at most FP4_MAX * FP8_MAX
            // return FP8_MAX * FP4_MAX / in;
            // We carefully control precision in FP4. We want to avoid introducing any non-powers of two
            return 2.0f;
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
        float scaleW1 = getFP8Scalar(maxW1);
        float scaleW2 = getFP8Scalar(maxW2);
        float scaleAct1 = getFP8Scalar(max_input);

        float maxFC1Output = calcMLPVal(max_input, maxIndex) / maxW2;
        float scaleAct2 = getFP8Scalar(maxFC1Output);

        ASSERT_NE(mExpertFPXScale1, nullptr);
        ASSERT_NE(mExpertFPXScale2, nullptr);
        ASSERT_NE(mExpertFPXScale3, nullptr);

        std::vector<float> scales_1;
        std::vector<float> scales_2;
        std::vector<float> scales_3;
        if (FP4)
        {
            std::vector<float> scale_global_w1(mNumExperts);
            std::vector<float> scale_global_w2(mNumExperts);

            std::vector<float> scales_0(1, scaleAct1);
            scales_1 = std::vector<float>(mNumExperts);
            scales_2 = std::vector<float>(1, scaleAct2);
            scales_3 = std::vector<float>(mNumExperts);

            for (int i = 0; i < mNumExperts; i++)
            {
                float maxW1 = applyExpertShift(maxW1GatedVal, i);
                float maxW2 = applyExpertShift(mExpertWDiag2, i);
                float scaleW1 = getFP8Scalar(maxW1);
                float scaleW2 = getFP8Scalar(maxW2);
                scale_global_w1[i] = scaleW1;
                scale_global_w2[i] = scaleW2;

                // TODO Per expert scaling factors
                scales_1[i] = 1.f / (scaleAct1 * scaleW1);
                scales_3[i] = 1.f / (scaleAct2 * scaleW2);
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
            scales_2 = std::vector<float>(1, scaleAct2);
            scales_3 = std::vector<float>(mNumExperts, 1.f / (scaleW2 * scaleAct2));
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
        // Can't use FP8 param because we recurse with a different type
        if constexpr (std::is_same_v<T, SafeFP8>)
        {
            // Call the standard setup and then perform the quantization manually
            std::vector<OutputType> internal_states(hidden_states.size());
            populateTokens(internal_states);

            mMaxInput = *std::max_element(internal_states.begin(), internal_states.end());
            float scalar = getFP8Scalar(mMaxInput);
            std::transform(internal_states.begin(), internal_states.end(), hidden_states.begin(),
                [scalar](OutputType in) -> T { return static_cast<T>((float) in * scalar); });
            // Do the reverse transformation since we only have so much precision and this is a pretty broad range
            std::transform(hidden_states.begin(), hidden_states.end(), internal_states.begin(),
                [scalar](T in) -> OutputType { return static_cast<OutputType>(((float) in) / scalar); });
            return internal_states;
        }
        else if constexpr (FP4)
        {
            float const max_scale = 1.0f;
            mMaxInput = FP4_MAX * max_scale;
            // Excludes 0.75 as this causes increased quantization error
            std::array allowed_values{-6.f, -4.f, -3.f, -2.f, -1.5f, -1.f, 0.0f, 1.f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
            float scale = 1.f / 32.f;
            int stride = TmaWarpSpecializedGroupedGemmInput::BlockScaleVectorSize;
            for (int i = 0; i < hidden_states.size(); i += stride)
            {
                for (int j = 0; j < stride; j++)
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
        else
        {
            // Generates numbers in increments of 1/max_order_of_magnitude in the range [0, 1)
            constexpr int max_order_of_magnitude = 256;
            std::vector<int> base(hidden_states.size());
            std::iota(base.begin(), base.end(), 0);
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
        MOEParallelismConfig parallelism_config = {})
    {
        initBuffersPermute(std::move(h_hidden_states), std::move(h_token_selected_experts),
            std::move(h_token_final_scales), hidden_size, num_experts, k, parallelism_config);
        runMoEPermute(parallelism_config);
    }

    auto getWeights(MOEParallelismConfig parallelism_config)
    {
        void* ep_scale_1 = (FP8 || FP4) ? (void*) mExpertFPXScale1 : (void*) mExpertIntScale1;
        void* ep_scale_2 = (FP8 || FP4) ? (void*) mExpertFPXScale2 : (void*) mExpertIntScale2;
        void* ep_scale_3 = (FP8 || FP4) ? mExpertFPXScale3 : nullptr;

        using SliceWeightType = std::conditional_t<FP4, DataType, WeightStorage>;
        // FP4 accesses the unquantized weight
        constexpr int SLICED_WEIGHT_ELEM_PER_BYTE = FP4 ? 1 : WEIGHT_ELEM_PER_BYTE;
        SliceWeightType* slice_weight_1{};
        SliceWeightType* slice_weight_2{};
        if constexpr (FP4)
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
        if constexpr (FP8 || FP4)
        {
            ep_scale_1 = mExpertFPXScale1 + experts_per_node * parallelism_config.ep_rank;
            ep_scale_3 = mExpertFPXScale3 + experts_per_node * parallelism_config.ep_rank;
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
            = reinterpret_cast<DataType*>(weight_2 + experts_per_node * matrix_size / SLICED_WEIGHT_ELEM_PER_BYTE);

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
            size_t const row_size_bias = row_size_inter * sizeof(DataType);
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

    auto getFilteredConfigs(int sm)
    {
        auto tactics = mMoERunner.getTactics();
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

    auto selectTacticsForArch(int sm)
    {
        bool is_tma_warp_specialized = sm >= 90 && !INT_QUANT;
        auto tactics = getFilteredConfigs(sm);
        auto it = std::find_if(tactics.begin(), tactics.end(),
            [is_tma_warp_specialized](auto& c) { return c.is_tma_warp_specialized == is_tma_warp_specialized; });
        if (it == tactics.end())
        {
            // Fall back to any tactic
            std::cout << "WARNING: Could not find config for sm version " << sm << std::endl;
            return std::pair{tactics[0], tactics[0]};
        }

        return std::pair(*it, *it);
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
        ConfigsToTestVec tactics = {selectTacticsForArch(sm)};
        if (sm >= 90 && !FP8 && !FP4)
        {
            // SM90+ should also grab some configs for SM80 to test them
            tactics.push_back(selectTacticsForArch(80));
        }
        return tactics;
    }

    void runMoEPermute(MOEParallelismConfig parallelism_config)
    {
        // Clear the buffers to blank so we can assume zero if not written
        resetOutBuffers();

        auto [weight1_ptr, weight2_ptr, bias1_ptr, bias2_ptr, scale1_ptr, scale2_ptr, scale3_ptr]
            = getWeights(parallelism_config);

        auto stream = mStream->get();
        auto tactic1 = mInternalSelectedConfig1;
        auto tactic2 = mInternalSelectedConfig2;
        if (!tactic1)
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
            quant_params = QuantParams::FP8(static_cast<float const*>(scale1_ptr),
                static_cast<float const*>(scale2_ptr), static_cast<float const*>(scale3_ptr));
        }
        else if (FP4)
        {
            ASSERT_TRUE(mExpertFP4ActGlobalScale1);
            ASSERT_TRUE(mFP4ScalingFactorsW1 && mFP4ScalingFactorsW2);
            ASSERT_TRUE(scale1_ptr && scale2_ptr && scale3_ptr);
            quant_params = QuantParams::FP4(mExpertFP4ActGlobalScale1, mFP4ScalingFactorsW1,
                static_cast<float const*>(scale1_ptr), static_cast<float const*>(scale2_ptr), mFP4ScalingFactorsW2,
                static_cast<float const*>(scale3_ptr));
        }

        if constexpr (FP4)
        {
            // Dynamically quantize using the proper tp slice
            doFP4Quant(static_cast<DataType const*>(weight1_ptr), mExpertWeight1, mExpertFP4WeightGlobalScale1,
                mFP4ScalingFactorsW1, mHiddenSize, mGatedMultiplier * mInterSize / parallelism_config.tp_size,
                mNumExperts / parallelism_config.ep_size);
            doFP4Quant(static_cast<DataType const*>(weight2_ptr), mExpertWeight2, mExpertFP4WeightGlobalScale2,
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
        mMoERunner.runMoe(mInputTensor, nullptr, mSelectedExpert, mTokenFinalScales, weight1_ptr, bias1_ptr, mActType,
            weight2_ptr, bias2_ptr, quant_params, mTotalTokens, mHiddenSize, mInterSize / parallelism_config.tp_size,
            mNumExperts, mK, mWorkspace, mFinalOutput, mSourceToExpandedMap, parallelism_config, mUseLora, lora_params,
            useFp8BlockScales, minLatencyMode, min_latency_params, stream);

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
    T actfn(T in)
    {
        if (mActType == ActivationType::Identity)
            return in;
        if (mActType == ActivationType::Relu)
            return std::max(in, T(0.0f));
        if (mActType == ActivationType::Gelu || mActType == ActivationType::Geglu)
            return (std::erf(float(in) * float(sqrt(0.5))) + 1) * 0.5f * float(in);
        if (mActType == ActivationType::Silu || mActType == ActivationType::Swiglu)
        {
            return (float(in) / (1.f + std::exp(-(in))));
        }
        assert(false);
        return in;
    }

    float calcMLPVal(float input, int expert_id, bool final_bias = false)
    {
        if (expert_id >= mNumExperts)
            return 0;

        float w1_bias = mUseBias ? expert_id : 0.f;
        float activated = 0;
        if (mIsGated)
        {
            float scalar = applyExpertShift(mExpertWDiag1, expert_id);
            float fc1 = input * scalar + w1_bias;

            float gated_scalar = applyExpertShift(mExpertWDiagGated, expert_id);
            float gated_bias = mUseBias ? w1_bias + 1.f : 0.f;
            float gate = input * gated_scalar + gated_bias;

            activated = fc1 * actfn(gate);
        }
        else
        {
            float scalar = applyExpertShift(mExpertWDiag1, expert_id);
            float fc1 = input * scalar + w1_bias;
            activated = actfn(fc1);
        }

        EXPECT_TRUE(mUseBias || !final_bias);
        float result = activated * applyExpertShift(mExpertWDiag2, expert_id) + (float) (final_bias ? expert_id : 0);
        return result;
    }

    float calcMLPValWithFinalBias(float input, int expert_id)
    {
        return calcMLPVal(input, expert_id, mUseBias);
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

    void compareFinal(std::vector<int> const& expected_experts, std::vector<float> const& token_final_scales,
        std::vector<OutputType> const& input_data, std::vector<OutputType> final_results = {})
    {
        ASSERT_EQ(expected_experts.size(), token_final_scales.size());
        ASSERT_EQ(expected_experts.size() / mK, input_data.size() / mHiddenSize);
        if (final_results.empty())
            final_results = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);

        for (int64_t token_id = 0; token_id < mTotalTokens; token_id++)
        {
            // NOTE: When mInterSize < mHiddenSize, those values get zeroed out by fc1 and lost
            for (int64_t hidden_id = 0; hidden_id < std::min(mHiddenSize, mInterSize); hidden_id++)
            {
                float sum = 0.0f;
                // Loop for the number of times each token is duplicated
                for (int k_idx = 0; k_idx < mK; k_idx++)
                {
                    int selected_expert = expected_experts[token_id * mK + k_idx];
                    float final_scale_value = token_final_scales[token_id * mK + k_idx];

                    float final_value = float(calcMLPValWithFinalBias(
                        static_cast<float>(input_data[token_id * mHiddenSize + hidden_id]), selected_expert));
                    sum += final_value * final_scale_value;
                }

                ASSERT_NEAR(OutputType{sum}, final_results[token_id * mHiddenSize + hidden_id], getTolerance(sum))
                    << "Incorrect final value at for token: " << token_id << " offset: " << hidden_id
                    << " hidden_size: " << mHiddenSize << " inter_size: " << mInterSize;
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
    }

    // Tensor parallel tests default to inter_size_fraction = 1.0f so that all ranks have interesting values
    void TensorParallelTest(int k = 1, int64_t hidden_size = DEFAULT_HIDDEN_SIZE, int64_t num_experts = 4,
        int64_t num_tokens = 3, float inter_size_fraction = 1.0f)
    {
        mInterSizeFraction = inter_size_fraction;
        ParallelismTest(k, 2, 1, hidden_size, num_experts, num_tokens);
        ParallelismTest(k, 4, 1, hidden_size, num_experts, num_tokens);
        ParallelismTest(k, 8, 1, hidden_size, num_experts, num_tokens);
    }

    void MixedParallelTest(int k = 1, int64_t hidden_size = DEFAULT_HIDDEN_SIZE, int64_t num_experts = 4,
        int64_t num_tokens = 3, float inter_size_fraction = 1.0f)
    {
        mInterSizeFraction = inter_size_fraction;

        // 2 experts per rank
        ParallelismTest(k, 2, num_experts / 2, hidden_size, num_experts, num_tokens);
        ParallelismTest(k, 8, num_experts / 2, hidden_size, num_experts, num_tokens);

        // 1 expert per rank
        ParallelismTest(k, 2, num_experts, hidden_size, num_experts, num_tokens);
        ParallelismTest(k, 8, num_experts, hidden_size, num_experts, num_tokens);
    }

    void ParallelismTest(int k = 1, int tp_size = 4, int ep_size = 2, int64_t hidden_size = DEFAULT_HIDDEN_SIZE,
        int64_t num_experts = 4, int64_t num_tokens = 3);
};

template <class WeightParams>
using LargeMixtureOfExpertsTest = MixtureOfExpertsTest<WeightParams>;

template <class DataType_, class WeightType_ = DataType_, class OutputType_ = DataType_>
struct WeightParams
{
    using DataType = DataType_;
    using WeightType = WeightType_;
    using OutputType = OutputType_;
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
    WeightParams<SafeFP4, SafeFP4, half>,
#endif
    WeightParams<half>, WeightParams<float>

    //, WeightParams<half, uint8_t>, WeightParams<half, cutlass::uint4b_t>

    >;
TYPED_TEST_SUITE(MixtureOfExpertsTest, Types);

// Have a separate test with only FP8 and half data type because this test is long
using LargeTestTypes = ::testing::Types<
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
    if constexpr (FP8 || FP4)
    {
        // TODO Remove this when bias + FP8 is supported
        mUseBias = false;
    }

    if (FP4)
    {
        if (mActType != ActivationType::Relu)
        {
            // FP4 has far too little precision to get any sort of consistency with non-relu actfn
            GTEST_SKIP();
            return;
        }
    }

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
        bool should_be_deterministic
            = mUseDeterminsiticHopperReduce || mK < 3 || getSMVersion() < 90 || getSMVersion() >= 120;
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
        ASSERT_EQ(permute_map, proj_map);
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

TYPED_TEST(MixtureOfExpertsTest, PermuteNonDeterministic)
{
    this->mUseDeterminsiticHopperReduce = false;
    // Just test case 3, cases 1&2 always use the fused paths
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteVerySmall)
{
    for (int i = 1; i <= 3; i++)
    {
        this->BasicPermuteTest(1, this->MINIMUM_ALIGNMENT * i);
        this->BasicPermuteTest(2, this->MINIMUM_ALIGNMENT * i);
        this->BasicPermuteTest(3, this->MINIMUM_ALIGNMENT * i);
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
    this->BasicPermuteTest(2, this->MINIMUM_ALIGNMENT, 512);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteSwigluVerySmall)
{
    this->mActType = ActivationType::Swiglu;
    for (int i = 1; i <= 3; i++)
    {
        this->BasicPermuteTest(1, this->MINIMUM_ALIGNMENT * i);
        this->BasicPermuteTest(2, this->MINIMUM_ALIGNMENT * i);
        this->BasicPermuteTest(3, this->MINIMUM_ALIGNMENT * i);
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

template <class TypeParam_>
std::vector<int> MixtureOfExpertsTest<TypeParam_>::calcPermuteMapExpertParallel(
    std::vector<int> const& expected_experts)
{
    std::vector<int> map(expected_experts.size());
    auto getInterleavedIndex = [this](int i) { return (i % mK) * mTotalTokens + i / mK; };
    int map_idx = 0;
    for (int expert = 0; expert < mNumExperts * 2; expert++)
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
    int k, int tp_size, int ep_size, int64_t hidden_size, int64_t num_experts, int64_t num_tokens)
{
    if (FP8 || FP4)
    {
        // TODO Remove this when bias + FP8 is supported
        mUseBias = false;
    }

    if (FP4)
    {
        if (mActType != ActivationType::Relu)
        {
            // FP4 has far too little precision to get any sort of consistency with non-relu actfn
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
                        MOEParallelismConfig{tp_size, i, ep_size, j});
                    bool should_be_deterministic
                        = mUseDeterminsiticHopperReduce || mK < 3 || getSMVersion() < 90 || getSMVersion() >= 120;
                    if (should_be_deterministic && !mIsLongTest)
                    {
                        auto first_iter = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);
                        mMemsetValue = ~mMemsetValue; // Also check it doesn't depend on uninitialised memory
                        runMoEPermute(hidden_input, expected_experts, token_final_scales, hidden_size, num_experts, k,
                            MOEParallelismConfig{tp_size, i, ep_size, j});
                        auto second_iter = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);
                        ASSERT_TRUE(std::equal(first_iter.begin(), first_iter.end(), second_iter.begin()))
                            << "Running permute a second time does not generate the same results";
                    }
                }
                else
                {
                    runMoEPermute(MOEParallelismConfig{tp_size, i, ep_size, j});
                    bool should_be_deterministic
                        = mUseDeterminsiticHopperReduce || mK < 3 || getSMVersion() < 90 || getSMVersion() >= 120;
                    if (should_be_deterministic && !mIsLongTest)
                    {
                        auto first_iter = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);
                        runMoEPermute(MOEParallelismConfig{tp_size, i, ep_size, j});
                        auto second_iter = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);
                        ASSERT_TRUE(std::equal(first_iter.begin(), first_iter.end(), second_iter.begin()))
                            << "Running permute a second time does not generate the same results";
                    }
                }

                auto masked_expected_experts = maskSelectedExpertsForTP(expected_experts, ep_size, j);
                auto proj_map = getDataFromDevice(mSourceToExpandedMap, mTotalTokens * k);
                auto permute_map = calcPermuteMapExpertParallel(masked_expected_experts);
                ASSERT_EQ(permute_map, proj_map) << "Iteration " << i << " " << j << " seq len " << num_tokens;

                // Do the final reduce
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
        this->mInterSizeFraction = float(inter_size) / hidden_size;                                                    \
                                                                                                                       \
        if (!this->checkSufficientTestMemory(75, hidden_size, 256, 8, true))                                           \
        {                                                                                                              \
            GTEST_SKIP() << "Insufficient free memory for test";                                                       \
        }                                                                                                              \
                                                                                                                       \
        this->ParallelismType##Test(8, hidden_size, 256, 75, this->mInterSizeFraction);                                \
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
        this->ParallelismType##Test(2, this->MINIMUM_ALIGNMENT, 512, 3, this->FP4 ? 8.0f : 4.0f);                      \
    }

PARALLEL_TEST_SUITE(ExpertParallel)
PARALLEL_TEST_SUITE(TensorParallel)
PARALLEL_TEST_SUITE(MixedParallel)

TYPED_TEST(MixtureOfExpertsTest, ConfigSweep)
{
    this->mIsLongTest = true;
    auto genConfigName = [](auto conf) -> std::string
    {
        using namespace tensorrt_llm::cutlass_extensions;
        std::stringstream tactic;
        tactic << "sm" << conf.sm_version << " tactic with tile shape ";
        if (conf.is_tma_warp_specialized)
        {
            tactic << conf.getTileConfigAsInt() << " and cluster shape " << (int) conf.cluster_shape
                   << " mainloop sched " << (int) conf.mainloop_schedule << " epi sched "
                   << (int) conf.epilogue_schedule;
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

    auto activation_pool = std::vector{ActivationType::Relu, ActivationType::Swiglu, ActivationType::Geglu};
    if (this->FP4)
        activation_pool = {ActivationType::Relu};
    auto configs = this->getFilteredConfigs(getSMVersion());
    for (auto const activation_type : activation_pool)
    {
        for (auto conf1 : configs)
        {
            for (auto conf2 : configs)
            {
                auto name1 = genConfigName(conf1);
                auto name2 = genConfigName(conf2);
                if (name1.empty() || name2.empty())
                {
                    FAIL() << "Uninitialised tactic encountered";
                }
                ASSERT_NO_THROW({
                    this->mActType = activation_type;
                    for (int k = 1; k <= 3; k++)
                    {

                        this->mOverrideSelectedConfig1 = conf1;
                        this->mOverrideSelectedConfig2 = conf2;
                        this->BasicPermuteTest(k);
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
    this->mUseBias = !this->FP8;

    using DataType = typename TypeParam::DataType;
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

TYPED_TEST(LargeMixtureOfExpertsTest, RunProfiler)
{
    constexpr bool is_half = std::is_same<typename TypeParam::DataType, half>::value;
    ASSERT_TRUE(this->FP8 || is_half) << "Unimplemented data type for profiler test";
    auto test_func = [this](GemmProfilerBackend::GemmToProfile gemm_to_profile)
    {
        int64_t num_experts = 4;
        int64_t k = 2;

        GemmProfilerBackend backend;
        backend.init(this->mMoERunner, gemm_to_profile,
            this->FP8 ? nvinfer1::DataType::kFP8 : nvinfer1::DataType::kHALF,
            this->FP8 ? nvinfer1::DataType::kFP8 : nvinfer1::DataType::kHALF, nvinfer1::DataType::kHALF, num_experts, k,
            this->DEFAULT_HIDDEN_SIZE, this->DEFAULT_HIDDEN_SIZE * 4, this->mGroupSize, ActivationType::Geglu, false,
            this->mUseLora, /*min_latency_mode=*/false,
            /*need_weights=*/true, MOEParallelismConfig{});

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
            backend.init(this->mMoERunner, GemmProfilerBackend::GemmToProfile::GEMM_1, nvinfer1::DataType::kHALF,
                nvinfer1::DataType::kHALF, nvinfer1::DataType::kHALF, num_experts, k, 1024, 4096, mGroupSize, {}, false,
                mUseLora, /*min_latency_mode=*/false, /*need_weights=*/true, MOEParallelismConfig{1, 0, ep, ep - 1});

            auto ws_size = backend.getWorkspaceSize(num_tokens);
            auto workspace = this->allocBuffer<char>(ws_size);
            int64_t num_experts_per_node = num_experts / ep;

            backend.prepare(num_tokens, workspace, /*expert_weights=*/nullptr, mStream->get());

            auto workspaces = backend.getProfilerWorkspaces(num_tokens, getSMVersion() >= 90 && getSMVersion() < 120);
#define GET_WS_PTR(type, name) auto* name = reinterpret_cast<type>(workspace + workspaces.at(#name).second)

            GET_WS_PTR(int64_t*, expert_first_token_offset);
            GET_WS_PTR(int*, source_to_dest);
            GET_WS_PTR(int*, dest_to_source);
            GET_WS_PTR(int*, unpermuted_selected_experts);

#undef GET_WS_PTR

            for (int sample = 0; sample < backend.NUM_ROUTING_SAMPLES; sample++)
            {
                auto host_expert_first_token_offset_size = getDataFromDevice(
                    expert_first_token_offset + sample * (num_experts_per_node + 1), num_experts_per_node + 1);
                auto host_source_to_dest_map
                    = getDataFromDevice(source_to_dest + sample * expanded_num_tokens, expanded_num_tokens);
                auto host_dest_to_source_map
                    = getDataFromDevice(dest_to_source + sample * expanded_num_tokens, expanded_num_tokens);
                auto host_token_selected_experts = getDataFromDevice(
                    unpermuted_selected_experts + sample * expanded_num_tokens, expanded_num_tokens);

                std::vector<int64_t> calculated_routing_values(num_experts_per_node + 1, 0);
                int skipped = 0;
                for (auto v : host_token_selected_experts)
                {
                    ASSERT_TRUE(v < num_experts_per_node || (v == num_experts_per_node && ep > 1))
                        << "v " << v << " num_experts_per_node " << num_experts_per_node << " ep " << ep;
                    skipped += (v == num_experts_per_node);
                    if (v < num_experts_per_node)
                    {
                        calculated_routing_values[v]++;
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

                        if (expert_idx < num_experts)
                        {
                            int64_t source_location = k_idx * num_tokens + token_idx;
                            int64_t dest_location = host_expert_first_token_offset_size[expert_idx]
                                + calculated_routing_values[expert_idx];

                            ASSERT_EQ(host_source_to_dest_map[source_location], dest_location);
                            ASSERT_EQ(host_dest_to_source_map[dest_location], source_location);

                            calculated_routing_values[expert_idx]++;
                        }
                    }
                }
            }
        }
    }
}
