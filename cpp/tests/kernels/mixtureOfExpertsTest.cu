#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

constexpr static float FP8_MAX = 440; // FP8_E4M3_MAX;

__host__ __device__ constexpr float expertShift(int expert, int num_experts)
{
    return float(expert) / num_experts;
}

template <class T>
__global__ void initWeightsKernel(T* data, int64_t w, int64_t h, float base, float scale)
{
    size_t expert_id = blockIdx.z;
    T* start_offset = data + expert_id * w * h;
    float expert_shift = scale * expertShift(expert_id, gridDim.z);

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h)
    {
        start_offset[y * w + x] = (x == y) ? T(base * scale + expert_shift) : T(0.f);
    }
}

template <class T>
__global__ void initWeightsGatedKernel(T* data, int64_t w, int64_t h, float base_1, float base_2, float scale)
{
    size_t expert_id = blockIdx.z;
    T* start_offset = data + expert_id * w * h * 2;

    float expert_shift = scale * expertShift(expert_id, gridDim.z);

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h)
    {
        start_offset[y * w + x] = (x == y) ? T(base_1 * scale + expert_shift) : T(0.f);
        start_offset[(y + h) * w + x] = (x == y) ? T(base_2 * scale + expert_shift) : T(0.f);
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

#ifdef ENABLE_FP8
using SafeFP8 = __nv_fp8_e4m3;
#else
using SafeFP8 = void;
#endif

template <class TypeTuple_>
class MixtureOfExpertsTest : public ::testing::Test
{
protected:
    using DataType = typename TypeTuple_::DataType;
    using WeightType = typename TypeTuple_::WeightType;
    using OutputType = typename TypeTuple_::OutputType;
    constexpr static bool INT4 = std::is_same_v<WeightType, cutlass::uint4b_t>;
    constexpr static bool FP8 = std::is_same_v<DataType, SafeFP8>;
    constexpr static bool INT_QUANT = !std::is_same_v<DataType, WeightType>;
    using WeightStorage = std::conditional_t<INT_QUANT, uint8_t, WeightType>;
    constexpr static int WEIGHT_ELEM_PER_BYTE = INT4 ? 2 : 1;
    constexpr static int64_t HIDDEN_SIZE_MULTIPLIER = 8;
    constexpr static int64_t DEFAULT_HIDDEN_SIZE
        = HIDDEN_SIZE_MULTIPLIER * 64 / sizeof(WeightType) * WEIGHT_ELEM_PER_BYTE;

    static BufferManager::CudaStreamPtr mStream;
    static std::unique_ptr<BufferManager> mBufferManager;
    static int mDeviceCount;

    std::vector<BufferManager::IBufferPtr> managed_buffers;
    float* mInputProbabilities{};
    DataType* mInputTensor{};

    int64_t mHiddenSize{};
    int64_t mNumExperts{};
    int64_t mK{};

    float getTolerance(float scale = 1.f)
    {
        // These FP8 tolerances are tuned quite tightly so should pick up any regressions
        // Whether the current results are as tight as they should be requires further investigation
        // They can be much tighter if we use the same value for all experts (so the scaling factors are trivial)
        // But that is hardly representative
        bool loose_fp8 = mIsGated || mNormMode == MOEExpertScaleNormalizationMode::RENORMALIZE;
        float tol = std::is_same_v<DataType, float> ? 0.001
            : std::is_same_v<DataType, half>        ? 0.01
            : std::is_same_v<DataType, SafeFP8>     ? (loose_fp8 ? 0.1 : 0.07)
                                                    : 0.1;

        // Keep the scale in a sane range
        scale = std::clamp(scale, 1.f, 30.f);
        return scale * tol;
    }

    static bool shouldSkip()
    {
#ifndef ENABLE_FP8
        static_assert(!FP8, "FP8 Tests enabled on unsupported CUDA version");
#endif
        bool should_skip_no_device = mDeviceCount <= 0;
        bool should_skip_unsupported_fp8 = getSMVersion() < 90 && FP8;
        return should_skip_no_device || should_skip_unsupported_fp8;
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

    void TearDown()
    {
        managed_buffers.clear();
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

    CutlassMoeFCRunner<DataType, WeightType, OutputType> mMoERunner{};
    char* mWorkspace{};
    float* mScaleProbs{};
    DataType* mRawExpertWeight1{};
    DataType* mRawExpertWeight2{};
    WeightStorage* mExpertWeight1{};
    WeightStorage* mExpertWeight2{};
    DataType* mExpertIntScale1{};
    DataType* mExpertIntScale2{};

    float mFP8WeightScalar1{1.f};
    float mFP8WeightScalar2{1.f};
    float* mExpertFP8Scale1{};
    float* mExpertFP8Scale2{};
    float* mExpertFP8Scale3{};

    DataType* mExpertBias1{};
    DataType* mExpertBias2{};

    void* mTpExpertScratch{}; // Copy the experts here when slicing up inputs
    size_t mTpExpertScratchSize{};

    OutputType* mFinalOutput{};
    int* mSourceToExpandedMap;
    int* mSelectedExpert;
    bool* mFinished{};
    int64_t mInterSize{};
    int64_t mTotalTokens{};
    int64_t mActiveRows{};

    bool mUseBias = true;

    bool mIsGated = false;
    int64_t mGatedMultiplier = 1;

    tensorrt_llm::ActivationType mActType = tensorrt_llm::ActivationType::Relu;
    MOEExpertScaleNormalizationMode mNormMode = MOEExpertScaleNormalizationMode::NONE;

    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> mSelectedConfig = std::nullopt;

    // Keep to simple power of two so we can have tight bounds on precision for quantized modes
    float const mExpertWDiag1{0.5};
    float const mExpertWDiagGated{1};
    float const mExpertWDiag2{2};

    float mMaxInput{};

    template <class T>
    T* allocBuffer(size_t size)
    {
        managed_buffers.emplace_back(mBufferManager->gpu(size * sizeof(T)));
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "Error allocating buffer of size: " << size;
        T* ptr = static_cast<T*>(managed_buffers.back()->data());
        return ptr;
    }

    bool checkSufficientTestMemory(int64_t num_tokens, int64_t hidden_size, int64_t num_experts, int64_t k)
    {
        this->managed_buffers.clear();             // Make sure all the previous buffers are freed
        check_cuda_error(cudaDeviceSynchronize()); // Sync to make sure all previous operations are resolved

        // Calculate the size contributions for all the large buffers to check if the GPU has enough space
        bool const is_gated = tensorrt_llm::isGatedActivation(mActType);
        size_t const num_gemms = 2 + is_gated;
        // Expert weights
        size_t const weight_size = hidden_size * (hidden_size * 4) * num_experts * sizeof(WeightStorage) * num_gemms;
        // Workspace size
        size_t const workspace_size = this->mMoERunner.getWorkspaceSize(
            num_tokens, hidden_size, hidden_size * 4, num_experts, k, this->mActType, {});
        // The input/output buffers
        size_t const in_out_size = 2 * num_tokens * hidden_size * sizeof(DataType);

        // This should be correct to within 100MiB (on tests with 30GiB total)
        size_t const total_size = workspace_size + weight_size + in_out_size;

        size_t const memory_pool_free_mem_size = mBufferManager->memoryPoolFree();
        auto const [freeMem, totalMem] = tensorrt_llm::common::getDeviceMemoryInfo(false);
        float const freeMemBuffer = 0.9f; // Add some buffer so we aren't completely pushing the limits
        std::cout << "Free memory is: " << freeMem << ", memory pool size is: " << memory_pool_free_mem_size
                  << ", required memory is: " << total_size << ", device total memory capacity: " << totalMem
                  << std::endl;
        return (freeMem + memory_pool_free_mem_size) * freeMemBuffer >= total_size;
    }

    void initBuffersPermute(std::vector<std::vector<DataType>> h_hidden_states,
        std::vector<std::vector<float>> h_router_results, int64_t hidden_size, int64_t num_experts, int64_t k,
        std::vector<uint8_t> finished, MOEParallelismConfig parallelism_config)
    {
        managed_buffers.clear();

        mHiddenSize = hidden_size;
        mInterSize = hidden_size * 4;
        mNumExperts = num_experts;
        mK = k;
        mIsGated = tensorrt_llm::isGatedActivation(mActType);
        mGatedMultiplier = mIsGated ? 2 : 1;
        auto const gated_inter = mInterSize * mGatedMultiplier;

        mTotalTokens = 0;

        std::vector<int64_t> h_seq_lens;
        h_seq_lens.push_back(0);
        for (auto& sequence : h_hidden_states)
        {
            assert(sequence.size() % hidden_size == 0);
            int64_t num_tokens = sequence.size() / hidden_size;
            h_seq_lens.emplace_back(h_seq_lens.back() + num_tokens);
            mTotalTokens += num_tokens;
        }

        size_t workspace_size = mMoERunner.getWorkspaceSize(
            mTotalTokens, mHiddenSize, mInterSize, mNumExperts, mK, mActType, parallelism_config);

        auto const stream = mStream->get();

        mWorkspace = allocBuffer<char>(workspace_size);
        // Memset to an obviously incorrect value, so we detect any issues with uninitialised fields
        check_cuda_error(cudaMemsetAsync(mWorkspace, 0xD5, workspace_size, stream));
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
        else
        {
            mExpertWeight1 = mRawExpertWeight1;
            mExpertWeight2 = mRawExpertWeight2;
        }

        if constexpr (FP8)
        {
            mExpertFP8Scale1 = allocBuffer<float>(mNumExperts);
            mExpertFP8Scale2 = allocBuffer<float>(1);
            mExpertFP8Scale3 = allocBuffer<float>(mNumExperts);

            EXPECT_NE(mMaxInput, 0.0f);
            initFP8Scales(mMaxInput);
        }

        if (parallelism_config.tp_size > 1 || parallelism_config.ep_size > 1)
        {
            mTpExpertScratch = allocBuffer<DataType>(mTpExpertScratchSize);
        }

        mActiveRows = mTotalTokens;
        mFinished = nullptr;
        if (!finished.empty())
        {
            mFinished = allocBuffer<bool>(mTotalTokens);
            check_cuda_error(cudaMemcpyAsync(
                mFinished, finished.data(), mTotalTokens * sizeof(bool), cudaMemcpyHostToDevice, stream));
            static_assert(sizeof(bool) == sizeof(uint8_t), "Test assumes bool is interchangeable with uint8_t");
            mActiveRows = std::count(finished.begin(), finished.end(), 0);
        }

        mInputProbabilities = allocBuffer<float>(mTotalTokens * mNumExperts);
        mScaleProbs = allocBuffer<float>(mTotalTokens * mK);
        mInputTensor = allocBuffer<DataType>(mTotalTokens * mHiddenSize);
        mFinalOutput = allocBuffer<OutputType>(mTotalTokens * mHiddenSize);

        mSourceToExpandedMap = allocBuffer<int>(mTotalTokens * mK);
        mSelectedExpert = allocBuffer<int>(mTotalTokens * mK);

        auto* input_probs_ptr = mInputProbabilities;
        for (auto& sequence : h_router_results)
        {
            check_cuda_error(cudaMemcpyAsync(
                input_probs_ptr, sequence.data(), sequence.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            input_probs_ptr += sequence.size();
        }

        auto* hidden_states_ptr = mInputTensor;
        for (auto& sequence : h_hidden_states)
        {
            check_cuda_error(cudaMemcpyAsync(hidden_states_ptr, sequence.data(), sequence.size() * sizeof(DataType),
                cudaMemcpyHostToDevice, stream));
            hidden_states_ptr += sequence.size();
        }

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

    constexpr static float getFP8Scalar(float in)
    {
        return FP8_MAX / in;
    }

    void initFP8Scales(float max_input)
    {
        check_cuda_error(cudaStreamSynchronize(mStream->get()));

        // Add shift to the max because we add an adjustment for each expert so they get different results.
        float max_shift = expertShift(mNumExperts - 1, mNumExperts);
        float maxW1 = max_shift + (mIsGated ? std::max(mExpertWDiag1, mExpertWDiagGated) : mExpertWDiag1);
        float maxW2 = max_shift + mExpertWDiag2;
        float scaleW1 = getFP8Scalar(maxW1);
        float scaleW2 = getFP8Scalar(maxW2);
        mFP8WeightScalar1 = scaleW1;
        mFP8WeightScalar2 = scaleW2;

        float scaleAct1 = getFP8Scalar(max_input);

        float maxFC1Output = calcMLPVal(max_input, mNumExperts - 1) / maxW2;
        float scaleAct2 = getFP8Scalar(maxFC1Output);

        ASSERT_NE(mExpertFP8Scale1, nullptr);
        ASSERT_NE(mExpertFP8Scale2, nullptr);
        ASSERT_NE(mExpertFP8Scale3, nullptr);

        // Dequant values for each expert are 1/(w_i*a_i) calculated above
        std::vector<float> scales_1(mNumExperts, 1.f / (scaleW1 * scaleAct1));
        std::vector<float> scales_2(1, scaleAct2);
        std::vector<float> scales_3(mNumExperts, 1.f / (scaleW2 * scaleAct2));

        check_cuda_error(cudaMemcpyAsync(mExpertFP8Scale1, scales_1.data(), scales_1.size() * sizeof(float),
            cudaMemcpyHostToDevice, mStream->get()));
        check_cuda_error(cudaMemcpyAsync(mExpertFP8Scale2, scales_2.data(), scales_2.size() * sizeof(float),
            cudaMemcpyHostToDevice, mStream->get()));
        check_cuda_error(cudaMemcpyAsync(mExpertFP8Scale3, scales_3.data(), scales_3.size() * sizeof(float),
            cudaMemcpyHostToDevice, mStream->get()));

        check_cuda_error(cudaStreamSynchronize(mStream->get()));
    }

    void resetOutBuffers()
    {
        auto stream = mStream->get();

        if (mTpExpertScratch)
            check_cuda_error(cudaMemsetAsync(mTpExpertScratch, 0x0, mTpExpertScratchSize, stream));
        check_cuda_error(cudaMemsetAsync(mFinalOutput, 0x0, mTotalTokens * mHiddenSize * sizeof(DataType), stream));
        check_cuda_error(cudaMemsetAsync(mSourceToExpandedMap, 0x0, sizeof(int) * mTotalTokens * mK, stream));
        check_cuda_error(cudaMemsetAsync(mSelectedExpert, 0x0, sizeof(int) * mTotalTokens * mK, stream));
        check_cuda_error(cudaMemsetAsync(mScaleProbs, 0x0, sizeof(float) * mTotalTokens * mK, stream));

        check_cuda_error(cudaStreamSynchronize(stream));
    }

    void resizeRouterInputs(
        std::vector<std::vector<float>>& h_router_results, int64_t num_experts, int64_t num_tokens_per_seq)
    {
        for (int64_t i = 0; i < h_router_results.size(); i++)
        {
            auto& seq_routing = h_router_results[i];
            int64_t num_tokens = num_tokens_per_seq;
            auto hardcoded_experts = seq_routing.size() / num_tokens;
            ASSERT_EQ(seq_routing.size(), hardcoded_experts * num_tokens);
            if (num_experts > hardcoded_experts)
            {
                auto pos = seq_routing.begin() + hardcoded_experts;
                for (int64_t i = 0; i < num_tokens; i++, pos += num_experts)
                {
                    pos = seq_routing.insert(pos, num_experts - hardcoded_experts, 0);
                }
            }
            ASSERT_EQ(seq_routing.size(), num_experts * num_tokens);
        }
    }

    template <class T>
    auto populateTokens(std::vector<T>& hidden_states)
    {
        if constexpr (std::is_same_v<T, SafeFP8>)
        {
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
        else
        {
            std::vector<float> base(hidden_states.size());
            std::iota(base.begin(), base.end(), 0.0f);
            // Lambda subtracts a small value so we have some < 0 to test the activation for negatives
            std::transform(base.begin(), base.end(), hidden_states.begin(),
                [l = hidden_states.size()](auto a) { return T(a / l) - T(0.01f); });
            return hidden_states;
        }
    }

    void runMoEPermute(std::vector<std::vector<DataType>> h_hidden_states,
        std::vector<std::vector<float>> h_router_results, int64_t hidden_size, int64_t num_experts, int64_t k,
        std::vector<uint8_t> finished = {}, MOEParallelismConfig parallelism_config = {})
    {
        initBuffersPermute(std::move(h_hidden_states), std::move(h_router_results), hidden_size, num_experts, k,
            finished, parallelism_config);
        runMoEPermute(parallelism_config);
    }

    auto getWeights(MOEParallelismConfig parallelism_config)
    {
        void* ep_scale_1 = FP8 ? (void*) mExpertFP8Scale1 : (void*) mExpertIntScale1;
        void* ep_scale_2 = FP8 ? (void*) mExpertFP8Scale2 : (void*) mExpertIntScale2;
        void* ep_scale_3 = FP8 ? mExpertFP8Scale3 : nullptr;

        // Handle the case with no parallelism to not require the extra alloc
        if (parallelism_config.tp_size == 1 && parallelism_config.ep_size == 1)
        {
            return std::tuple{
                mExpertWeight1, mExpertWeight2, mExpertBias1, mExpertBias2, ep_scale_1, ep_scale_2, ep_scale_3};
        }

        // Slice weights for EP
        size_t const gated_inter = mInterSize * mGatedMultiplier;
        size_t const experts_per_node = mNumExperts / parallelism_config.ep_size;
        size_t const weight_matrix_size = mHiddenSize * mInterSize * experts_per_node / WEIGHT_ELEM_PER_BYTE;
        size_t const bias_fc1_size = gated_inter * experts_per_node;
        size_t const bias_fc2_size = mHiddenSize * experts_per_node;
        size_t const scale1_size = gated_inter * experts_per_node;
        size_t const scale2_size = mHiddenSize * experts_per_node;
        auto* weight1_ptr = mExpertWeight1 + weight_matrix_size * mGatedMultiplier * parallelism_config.ep_rank;
        auto* weight2_ptr = mExpertWeight2 + weight_matrix_size * parallelism_config.ep_rank;
        auto* bias1_ptr = mUseBias ? mExpertBias1 + bias_fc1_size * parallelism_config.ep_rank : nullptr;
        auto* bias2_ptr = mUseBias ? mExpertBias2 + bias_fc2_size * parallelism_config.ep_rank : nullptr;

        if (INT_QUANT)
        {
            ep_scale_1 = mExpertIntScale1 + scale1_size * parallelism_config.ep_rank;
            ep_scale_2 = mExpertIntScale2 + scale2_size * parallelism_config.ep_rank;
        }
        if constexpr (FP8)
        {
            ep_scale_1 = mExpertFP8Scale1 + experts_per_node * parallelism_config.ep_rank;
            ep_scale_3 = mExpertFP8Scale3 + experts_per_node * parallelism_config.ep_rank;
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

        auto* weight_1 = reinterpret_cast<WeightStorage*>(mTpExpertScratch);
        auto* weight_2 = weight_1 + experts_per_node * gated_matrix_size;
        auto* bias_1 = reinterpret_cast<DataType*>(weight_2 + experts_per_node * matrix_size);

        // 2D memcpy just the slices we care about
        // TODO Re-quantize here with matrices divided
        size_t const row_size_1 = matrix_size * sizeof(WeightStorage) / WEIGHT_ELEM_PER_BYTE;
        check_cuda_error(
            cudaMemcpy2DAsync(weight_1, row_size_1, (uint8_t*) weight1_ptr + row_size_1 * tp_rank, row_size_1 * tp_size,
                row_size_1, experts_per_node * mGatedMultiplier, cudaMemcpyDeviceToDevice, mStream->get()));

        size_t const row_size_2 = row_size_inter * sizeof(WeightStorage) / WEIGHT_ELEM_PER_BYTE;
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

        return std::tuple{weight_1, weight_2, bias_1, bias2_ptr, scale_1, scale_2, scale_3};
    }

    void runMoEPermute(MOEParallelismConfig parallelism_config)
    {
        // Clear the buffers to blank so we can assume zero if not written
        resetOutBuffers();

        auto const [weight1_ptr, weight2_ptr, bias1_ptr, bias2_ptr, scale1_ptr, scale2_ptr, scale3_ptr]
            = getWeights(parallelism_config);

        auto stream = mStream->get();
        auto tactic = mSelectedConfig;
        if (!tactic)
        {
            int sm = getSMVersion();
            bool is_sm90 = sm >= 90 && !INT_QUANT;
            auto tactics = mMoERunner.getTactics();
            auto it = std::find_if(tactics.begin(), tactics.end(), [is_sm90](auto& c) { return c.is_sm90 == is_sm90; });
            if (it == tactics.end())
            {
                // Fall back to any tactic
                std::cout << "WARNING: Could not find config for sm version " << sm << std::endl;
                tactic = tactics[0];
            }
            else
            {
                tactic = *it;
            }
        }

        QuantParams quant_params;
        if constexpr (INT_QUANT)
        {
            quant_params = QuantParams::Int(scale1_ptr, scale2_ptr);
        }
        else
        {
            quant_params = QuantParams::FP8(static_cast<float const*>(scale1_ptr),
                static_cast<float const*>(scale2_ptr), static_cast<float const*>(scale3_ptr));
        }

        mMoERunner.setTactic(tactic);
        mMoERunner.runMoe(mInputTensor, mInputProbabilities, weight1_ptr, bias1_ptr, mActType, weight2_ptr, bias2_ptr,
            quant_params, mTotalTokens, mHiddenSize, mInterSize / parallelism_config.tp_size, mNumExperts, mK,
            mWorkspace, mFinalOutput, mFinished, mActiveRows, mScaleProbs, mSourceToExpandedMap, mSelectedExpert,
            parallelism_config, mNormMode, stream);

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
        PRINT_CAST((uint8_t*) mFinished, mTotalTokens, (int) );
        PRINT(mInputProbabilities, mTotalTokens * mNumExperts);
        PRINT(mScaleProbs, mTotalTokens * mK);
        PRINT(mInputProbabilities, mTotalTokens * mNumExperts);
        PRINT_CAST(mInputTensor, mTotalTokens * mHiddenSize, float);
        PRINT(mSourceToExpandedMap, mTotalTokens * mK);
        PRINT(mSelectedExpert, mTotalTokens * mK);

#undef PRINT_CAST
#undef PRINT
    }

    template <class T>
    T actfn(T in)
    {
        if (mActType == tensorrt_llm::ActivationType::Identity)
            return in;
        if (mActType == tensorrt_llm::ActivationType::Relu)
            return std::max(in, T(0.0f));
        if (mActType == tensorrt_llm::ActivationType::Gelu || mActType == tensorrt_llm::ActivationType::Geglu)
            return (std::erf(float(in) * float(sqrt(0.5))) + 1) * 0.5f * float(in);
        if (mActType == tensorrt_llm::ActivationType::Silu || mActType == tensorrt_llm::ActivationType::Swiglu)
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

        float expert_shift = expertShift(expert_id, mNumExperts);
        float w1_bias = mUseBias ? expert_id : 0.f;
        float activated = 0;
        if (mIsGated)
        {
            float scalar = mExpertWDiag1 + expert_shift;
            float fc1 = (float) input * scalar + (float) w1_bias;

            float gated_scalar = mExpertWDiagGated + expert_shift;
            float gated_bias = mUseBias ? (float) w1_bias + 1.f : 0.f;
            float gate = (float) input * gated_scalar + gated_bias;

            activated = fc1 * actfn(gate);
        }
        else
        {
            float scalar = mExpertWDiag1 + expert_shift;
            float fc1 = input * scalar + w1_bias;
            activated = actfn(fc1);
        }

        EXPECT_TRUE(mUseBias || !final_bias);
        float result = activated * (mExpertWDiag2 + expert_shift) + (float) (final_bias ? expert_id : 0);
        return result;
    }

    float calcMLPValWithFinalBias(float input, int expert_id)
    {
        return calcMLPVal(input, expert_id, mUseBias);
    }

    // NOTE This is a useful function for debugging routing failures. But you need to know the exact offset of
    //   this info in the workspace so having a test depend on something so internal is suboptimal
    //
    // void comparePermuted(const std::vector<int>& expected_experts, const std::vector<int>& expected_permutation,
    //     const std::vector<DataType>& input_data)
    //{
    //     auto states = getDataFromDevice(magic incantation into workspace, mTotalTokens * mK * mHiddenSize);
    //
    //    // Loop for the number of times each token is duplicated
    //    for (int k_idx = 0; k_idx < mK; k_idx++)
    //    {
    //        for (int64_t token_id = 0; token_id < mTotalTokens; token_id++)
    //        {
    //            // Permutation has the position of the first copy of all token,
    //            // followed by the position of the second copy of all tokens etc.
    //            const int64_t permuted_position = expected_permutation[k_idx * mTotalTokens + token_id];
    //
    //            // Expected experts has all the selected experts for token one,
    //            // followed by all the selected experts for token two etc.
    //            const int64_t expert_id = expected_experts[token_id * mK + k_idx];
    //
    //            // Compare the copied tokens with the projection applied
    //            for (int64_t hidden_id = 0; hidden_id < mHiddenSize; hidden_id++)
    //            {
    //                auto ref = calcMLPVal(input_data[token_id * mHiddenSize + hidden_id], expert_id);
    //                auto actual = states[permuted_position * mHiddenSize + hidden_id];
    //                ASSERT_NEAR(ref, actual, getTolerance(ref))
    //                    << "Incorrect value at position: mK: " << k_idx << ", token: " << token_id
    //                    << ", permuted dest: " << permuted_position << ", expert id: " << expert_id
    //                    << ", hidden id: " << hidden_id;
    //            }
    //        }
    //    }
    //}

    std::vector<float> softmax(std::vector<float> const& expected_probs)
    {
        std::vector<float> softmax;
        // All values we test are 0-1 so we can skip the normalization step
        std::transform(expected_probs.begin(), expected_probs.end(), std::back_inserter(softmax),
            [&](float const in) -> float
            {
                auto res = exp(in);
                return res;
            });

        for (int64_t token = 0; token < mTotalTokens; token++)
        {
            auto start = softmax.begin() + token * mNumExperts;
            auto end = start + mNumExperts;
            auto sum = std::accumulate(start, end, 0.f);
            std::transform(start, end, start, [=](auto in) { return in / sum; });
        }

        return softmax;
    }

    void compareSoftmax(std::vector<int> const& expected_experts, std::vector<float> const& expected_probs,
        std::vector<float> scale_probs = {})
    {
        if (scale_probs.empty())
            scale_probs = getDataFromDevice(mScaleProbs, mTotalTokens * mK);
        auto softmax_probs = softmax(expected_probs);
        for (int64_t token_id = 0; token_id < mTotalTokens; token_id++)
        {
            for (int k_idx = 0; k_idx < mK; k_idx++)
            {
                int selected_expert = expected_experts[token_id * mK + k_idx];
                if (selected_expert < mNumExperts) // Ignore 'finished' values
                {
                    ASSERT_NEAR(softmax_probs[token_id * mNumExperts + selected_expert],
                        scale_probs[token_id * mK + k_idx], getTolerance())
                        << "Scales mismatched for token: " << token_id << " k: " << k_idx
                        << " selected_expert: " << selected_expert;
                }
            }
        }
    }

    void renormScales(float* probs, int const* experts)
    {
        if (mNormMode == MOEExpertScaleNormalizationMode::NONE)
            return;
        float sum = 0;
        for (int k_idx = 0; k_idx < mK; k_idx++)
        {
            sum += probs[experts[k_idx]];
        }
        float norm_factor = 1.0f / sum;
        for (int k_idx = 0; k_idx < mK; k_idx++)
        {
            probs[experts[k_idx]] *= norm_factor;
        }
    }

    void compareFinal(std::vector<int> const& expected_experts, std::vector<float> const& expected_probs,
        std::vector<OutputType> const& input_data, std::vector<OutputType> final_results = {})
    {
        if (final_results.empty())
            final_results = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);

        auto softmax_probs = softmax(expected_probs);
        for (int64_t token_id = 0; token_id < mTotalTokens; token_id++)
        {
            renormScales(&softmax_probs[token_id * mNumExperts], &expected_experts[token_id * mK]);

            for (int64_t hidden_id = 0; hidden_id < mHiddenSize; hidden_id++)
            {
                float sum = 0.0f;
                // Loop for the number of times each token is duplicated
                for (int k_idx = 0; k_idx < mK; k_idx++)
                {
                    int selected_expert = expected_experts[token_id * mK + k_idx];
                    sum += float(calcMLPValWithFinalBias(
                               static_cast<float>(input_data[token_id * mHiddenSize + hidden_id]), selected_expert))
                        * softmax_probs[token_id * mNumExperts + selected_expert];
                }

                ASSERT_NEAR(OutputType{sum}, final_results[token_id * mHiddenSize + hidden_id], getTolerance(sum))
                    << "Incorrect final value at position: " << token_id * mHiddenSize + hidden_id;
            }
        }
    }

    void BasicPermuteTest(int k = 1, int64_t hidden_size = DEFAULT_HIDDEN_SIZE);

    std::vector<int> calcPermuteMapExpertParallel(std::vector<int> const& expected_experts);
    void ExpertParallelTest(int k = 1);

    void TensorParallelTest(int k = 1);

    void MixedParallelTest(int k = 1);
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
void MixtureOfExpertsTest<TypeParam_>::BasicPermuteTest(int k, int64_t hidden_size)
{
    if constexpr (FP8)
    {
        // TODO Remove this when bias + FP8 is supported
        mUseBias = false;
    }

    int64_t num_experts = 4;
    int64_t num_tokens = 3;

    std::vector<DataType> hidden_states(hidden_size * num_tokens);
    auto raw_unquant_input = populateTokens(hidden_states);

    std::vector<float> probs = {
        0.5, 0.1, 0.25, 0.15,   //
        0.03, 0.2, 0.07, 0.7,   //
        0.25, 0.21, 0.35, 0.19, //
    };

    std::vector<std::vector<DataType>> hidden_input = {hidden_states};
    std::vector<std::vector<float>> router_input = {probs};
    resizeRouterInputs(router_input, num_experts, num_tokens);

    runMoEPermute(hidden_input, router_input, hidden_size, num_experts, k);

    std::vector<int> expected_experts{0, 3, 2};
    if (k == 2)
        expected_experts = {0, 2, 3, 1, 2, 0};
    else if (k == 3)
        expected_experts = {0, 2, 3, 3, 1, 2, 2, 0, 1};

    auto selected_expert = getDataFromDevice(mSelectedExpert, num_tokens * k);
    EXPECT_EQ(selected_expert, expected_experts);

    auto proj_map = getDataFromDevice(mSourceToExpandedMap, num_tokens * k);
    // This is the final position of:
    // Token 1 Expert 1, T2E1, T3E1, T1E2, T2E2, T3E2
    std::vector<int> permute_map{0, 2, 1};
    if (k == 2)
        permute_map = {0, 5, 4, 3, 2, 1};
    if (k == 3)
        permute_map = {0, 8, 6, 4, 2, 1, 7, 5, 3};
    ASSERT_EQ(permute_map, proj_map);
    compareSoftmax(selected_expert, router_input[0]);
    compareFinal(selected_expert, router_input[0], raw_unquant_input);
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

TYPED_TEST(MixtureOfExpertsTest, PermuteNoBias)
{
    this->mUseBias = false;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteRenormalization)
{
    this->mNormMode = tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteGeglu)
{
    this->mActType = tensorrt_llm::ActivationType::Geglu;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, PermuteSwiglu)
{
    this->mActType = tensorrt_llm::ActivationType::Swiglu;
    this->BasicPermuteTest();
    this->BasicPermuteTest(2);
    this->BasicPermuteTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, Finished)
{
    if (this->FP8)
    {
        // TODO Remove this when bias + FP8 is supported
        this->mUseBias = false;
    }

    using DataType = typename TypeParam::DataType;
    int64_t hidden_size = this->DEFAULT_HIDDEN_SIZE;
    int64_t num_experts = 4;
    int64_t num_tokens = 3;
    int64_t k = 2;

    std::vector<DataType> hidden_states(hidden_size * num_tokens);
    auto raw_unquant_input = this->populateTokens(hidden_states);

    std::vector<float> probs = {
        0.5, 0.1, 0.25, 0.15, //
        0.05, 0.2, 0.05, 0.7, //
        0.25, 0.2, 0.35, 0.2, //
    };

    this->runMoEPermute({hidden_states}, {probs}, hidden_size, num_experts, k, {0, 0, 1});

    auto selected_expert = this->getDataFromDevice(this->mSelectedExpert, num_tokens * k);
    // Token 1
    EXPECT_EQ(selected_expert[0], 0);
    EXPECT_EQ(selected_expert[1], 2);
    // Token 2
    EXPECT_EQ(selected_expert[2], 3);
    EXPECT_EQ(selected_expert[3], 1);
    // Token 3
    EXPECT_EQ(selected_expert[4], num_experts); // One past the end
    EXPECT_EQ(selected_expert[5], num_experts);

    auto proj_map = this->getDataFromDevice(this->mSourceToExpandedMap, num_tokens * k);
    // This is the final position of:
    // Token 1 Expert 1, T2E1, T3E1, T1E2, T2E2, T3E3
    std::vector<int> permute_map{0, 3, 4, 2, 1, 5};
    ASSERT_EQ(permute_map, proj_map);
    this->compareSoftmax(selected_expert, probs);
    this->compareFinal(selected_expert, probs, raw_unquant_input);
}

template <class TypeParam_>
std::vector<int> MixtureOfExpertsTest<TypeParam_>::calcPermuteMapExpertParallel(
    std::vector<int> const& expected_experts)
{
    std::vector<int> map(expected_experts.size());
    auto getInterleavedIndex = [this](int i) { return (i % mK) * mTotalTokens + i / mK; };
    int map_idx = 0;
    for (int expert = 0; expert <= mNumExperts; expert++)
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
void MixtureOfExpertsTest<TypeParam_>::ExpertParallelTest(int k)
{
    if (FP8)
    {
        // TODO Remove this when bias + FP8 is supported
        mUseBias = false;
    }

    int64_t hidden_size = DEFAULT_HIDDEN_SIZE;
    int parallelism = 2;
    int64_t num_experts = 4;
    int64_t num_tokens = 3;

    std::vector<DataType> hidden_states(hidden_size * num_tokens);
    auto raw_unquant_input = populateTokens(hidden_states);

    std::vector<float> probs = {
        0.5, 0.1, 0.25, 0.15,   //
        0.03, 0.2, 0.07, 0.7,   //
        0.25, 0.21, 0.35, 0.19, //
    };

    std::vector<int> expected_experts{0, 3, 2};
    if (k == 2)
        expected_experts = {0, 2, 3, 1, 2, 0};
    else if (k == 3)
        expected_experts = {0, 2, 3, 3, 1, 2, 2, 0, 1};
    std::vector<OutputType> results(hidden_states.size(), 0);
    for (int i = 0; i < parallelism; i++)
    {
        if (i == 0)
        {
            // Only need to init the inputs on the first iteration
            runMoEPermute(
                {hidden_states}, {probs}, hidden_size, num_experts, k, {}, MOEParallelismConfig{1, 0, parallelism, i});
        }
        else
        {
            runMoEPermute(MOEParallelismConfig{1, 0, parallelism, i});
        }

        auto selected_expert = getDataFromDevice(mSelectedExpert, num_tokens * k);
        // Experts should only be selected when we are on the right node
        // Note the index is [0,num_experts_per_node), so we offset the experts by the start for this node
        int const start_expert = i * (mNumExperts / parallelism);
        std::transform(selected_expert.begin(), selected_expert.end(), selected_expert.begin(),
            [&](int val) { return val == mNumExperts ? mNumExperts : val + start_expert; });
        auto masked_expected_experts = maskSelectedExpertsForTP(expected_experts, parallelism, i);
        ASSERT_EQ(selected_expert, masked_expected_experts);

        auto proj_map = getDataFromDevice(mSourceToExpandedMap, num_tokens * k);
        auto permute_map = calcPermuteMapExpertParallel(masked_expected_experts);
        ASSERT_EQ(permute_map, proj_map) << "Iteration " << i;
        compareSoftmax(expected_experts, probs);

        // Do the final reduce
        auto iter_results = getDataFromDevice(mFinalOutput, num_tokens * hidden_size);
        std::transform(iter_results.cbegin(), iter_results.cend(), results.cbegin(), results.begin(), std::plus<>{});
    }

    compareFinal(expected_experts, probs, raw_unquant_input, results);
}

TYPED_TEST(MixtureOfExpertsTest, ExpertParallel)
{
    this->ExpertParallelTest();
}

TYPED_TEST(MixtureOfExpertsTest, ExpertParallelK2)
{
    this->ExpertParallelTest(2);
}

TYPED_TEST(MixtureOfExpertsTest, ExpertParallelNoBias)
{
    this->mUseBias = false;
    this->ExpertParallelTest();
    this->ExpertParallelTest(2);
}

TYPED_TEST(MixtureOfExpertsTest, ExpertParallelRenorm)
{
    this->mNormMode = MOEExpertScaleNormalizationMode::RENORMALIZE;
    this->ExpertParallelTest();
    this->ExpertParallelTest(2);
}

TYPED_TEST(MixtureOfExpertsTest, ExpertParallelGeglu)
{
    this->mActType = tensorrt_llm::ActivationType::Geglu;
    this->ExpertParallelTest();
    this->ExpertParallelTest(2);
}

TYPED_TEST(MixtureOfExpertsTest, ExpertParallelSwiglu)
{
    this->mActType = tensorrt_llm::ActivationType::Swiglu;
    this->ExpertParallelTest();
    this->ExpertParallelTest(2);
}

template <class TypeParam_>
void MixtureOfExpertsTest<TypeParam_>::TensorParallelTest(int k)
{
    if (FP8)
    {
        // TODO Remove this when bias + FP8 is supported
        mUseBias = false;
    }

    int64_t hidden_size = DEFAULT_HIDDEN_SIZE;
    int parallelism = 8;
    int64_t num_experts = 4;
    int64_t num_tokens = 3;

    std::vector<DataType> hidden_states(hidden_size * num_tokens);
    auto raw_unquant_input = populateTokens(hidden_states);

    std::vector<float> probs = {
        0.5, 0.1, 0.25, 0.15,   //
        0.03, 0.2, 0.07, 0.7,   //
        0.25, 0.21, 0.35, 0.19, //
    };

    std::vector<int> expected_experts{0, 3, 2};
    if (k == 2)
        expected_experts = {0, 2, 3, 1, 2, 0};
    else if (k == 3)
        expected_experts = {0, 2, 3, 3, 1, 2, 2, 0, 1};
    std::vector<OutputType> results(hidden_states.size(), 0);
    for (int i = 0; i < parallelism; i++)
    {
        if (i == 0)
        {
            // Only need to init the inputs on the first iteration
            runMoEPermute(
                {hidden_states}, {probs}, hidden_size, num_experts, k, {}, MOEParallelismConfig{parallelism, i, 1, 0});
        }
        else
        {
            runMoEPermute(MOEParallelismConfig{parallelism, i, 1, 0});
        }

        auto selected_expert = getDataFromDevice(mSelectedExpert, num_tokens * k);
        EXPECT_EQ(selected_expert, expected_experts);

        auto proj_map = getDataFromDevice(mSourceToExpandedMap, num_tokens * k);
        std::vector<int> permute_map{0, 2, 1};
        if (k == 2)
            permute_map = {0, 5, 4, 3, 2, 1};
        if (k == 3)
            permute_map = {0, 8, 6, 4, 2, 1, 7, 5, 3};

        ASSERT_EQ(permute_map, proj_map) << "Iteration " << i;

        // Do the final reduce
        auto iter_results = getDataFromDevice(mFinalOutput, num_tokens * hidden_size);
        std::transform(iter_results.cbegin(), iter_results.cend(), results.cbegin(), results.begin(), std::plus<>{});
    }

    compareFinal(expected_experts, probs, raw_unquant_input, results);
}

TYPED_TEST(MixtureOfExpertsTest, TensorParallel)
{
    this->TensorParallelTest();
}

TYPED_TEST(MixtureOfExpertsTest, TensorParallelK2)
{
    this->TensorParallelTest(2);
}

TYPED_TEST(MixtureOfExpertsTest, TensorParallelK3)
{
    this->TensorParallelTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, TensorParallelNoBias)
{
    this->mUseBias = false;
    this->TensorParallelTest();
    this->TensorParallelTest(2);
    this->TensorParallelTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, TensorParallelRenorm)
{
    this->mNormMode = MOEExpertScaleNormalizationMode::RENORMALIZE;
    this->TensorParallelTest();
    this->TensorParallelTest(2);
    this->TensorParallelTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, TensorParallelGeglu)
{
    this->mActType = tensorrt_llm::ActivationType::Geglu;
    this->TensorParallelTest();
    this->TensorParallelTest(2);
    this->TensorParallelTest(3);
}

TYPED_TEST(MixtureOfExpertsTest, TensorParallelSwiglu)
{
    this->mActType = tensorrt_llm::ActivationType::Swiglu;
    this->TensorParallelTest();
    this->TensorParallelTest(2);
    this->TensorParallelTest(3);
}

template <class TypeParam_>
void MixtureOfExpertsTest<TypeParam_>::MixedParallelTest(int k)
{
    if (FP8)
    {
        // TODO Remove this when bias + FP8 is supported
        mUseBias = false;
    }

    int64_t hidden_size = DEFAULT_HIDDEN_SIZE;
    int tp_parallelism = 2;
    int ep_parallelism = 2;
    int64_t num_experts = 4;
    int64_t num_tokens = 3;

    std::vector<DataType> hidden_states(hidden_size * num_tokens);
    auto raw_unquant_input = populateTokens(hidden_states);

    std::vector<float> probs = {
        0.5, 0.1, 0.25, 0.15,   //
        0.03, 0.2, 0.07, 0.7,   //
        0.25, 0.21, 0.35, 0.19, //
    };

    std::vector<int> expected_experts{0, 3, 2};
    if (k == 2)
        expected_experts = {0, 2, 3, 1, 2, 0};
    else if (k == 3)
        expected_experts = {0, 2, 3, 3, 1, 2, 2, 0, 1};
    std::vector<OutputType> results(hidden_states.size(), 0);
    for (int i = 0; i < tp_parallelism; i++)
    {
        for (int j = 0; j < ep_parallelism; j++)
        {
            if (i == 0 && j == 0)
            {
                // Only need to init the inputs on the first iteration
                runMoEPermute({hidden_states}, {probs}, hidden_size, num_experts, k, {},
                    MOEParallelismConfig{tp_parallelism, i, ep_parallelism, j});
            }
            else
            {
                runMoEPermute(MOEParallelismConfig{tp_parallelism, i, ep_parallelism, j});
            }

            auto selected_expert = getDataFromDevice(mSelectedExpert, num_tokens * k);
            // Experts should only be selected when we are on the right node
            // Note the index is [0,num_experts_per_node), so we offset the experts by the start for this node
            int const start_expert = j * (mNumExperts / ep_parallelism);
            std::transform(selected_expert.begin(), selected_expert.end(), selected_expert.begin(),
                [&](int val) { return val == mNumExperts ? mNumExperts : val + start_expert; });
            auto masked_expected_experts = maskSelectedExpertsForTP(expected_experts, ep_parallelism, j);
            ASSERT_EQ(selected_expert, masked_expected_experts);

            auto proj_map = getDataFromDevice(mSourceToExpandedMap, num_tokens * k);
            auto permute_map = calcPermuteMapExpertParallel(masked_expected_experts);
            ASSERT_EQ(permute_map, proj_map) << "Iteration " << i << " " << j;
            compareSoftmax(expected_experts, probs);

            // Do the final reduce
            auto iter_results = getDataFromDevice(mFinalOutput, num_tokens * hidden_size);
            std::transform(
                iter_results.cbegin(), iter_results.cend(), results.cbegin(), results.begin(), std::plus<>{});
        }
    }

    compareFinal(expected_experts, probs, raw_unquant_input, results);
}

TYPED_TEST(MixtureOfExpertsTest, MixedParallel)
{
    this->MixedParallelTest();
}

TYPED_TEST(MixtureOfExpertsTest, MixedParallelK2)
{
    this->MixedParallelTest(2);
}

TYPED_TEST(MixtureOfExpertsTest, MixedParallelNoBias)
{
    this->mUseBias = false;
    this->MixedParallelTest();
    this->MixedParallelTest(2);
}

TYPED_TEST(MixtureOfExpertsTest, MixedParallelRenorm)
{
    this->mNormMode = MOEExpertScaleNormalizationMode::RENORMALIZE;
    this->MixedParallelTest();
    this->MixedParallelTest(2);
}

TYPED_TEST(MixtureOfExpertsTest, MixedParallelGeglu)
{
    this->mActType = tensorrt_llm::ActivationType::Geglu;
    this->MixedParallelTest();
    this->MixedParallelTest(2);
}

TYPED_TEST(MixtureOfExpertsTest, MixedParallelSwiglu)
{
    this->mActType = tensorrt_llm::ActivationType::Swiglu;
    this->MixedParallelTest();
    this->MixedParallelTest(2);
}

TYPED_TEST(MixtureOfExpertsTest, ConfigSweep)
{
    std::vector<tensorrt_llm::ActivationType> actiavtion_pool = {
        tensorrt_llm::ActivationType::Relu, tensorrt_llm::ActivationType::Swiglu, tensorrt_llm::ActivationType::Geglu};
    auto configs = this->mMoERunner.getTactics();
    for (auto const activation_type : actiavtion_pool)
    {
        for (auto conf : configs)
        {
            using namespace tensorrt_llm::cutlass_extensions;
            std::stringstream tactic;
            tactic << "Failed " << (conf.is_sm90 ? "SM90+" : "<SM90") << " tactic with tile shape ";
            if (conf.tile_config_sm90 != CutlassTileConfigSM90::ChooseWithHeuristic)
            {
                tactic << (int) conf.tile_config_sm90 << " and cluster shape " << (int) conf.cluster_shape
                       << " mainloop sched " << (int) conf.mainloop_schedule << " epi sched "
                       << (int) conf.epilogue_schedule;
            }
            else if (conf.tile_config != CutlassTileConfig::ChooseWithHeuristic)
            {
                tactic << (int) conf.tile_config << " and stages " << (int) conf.stages << " split k "
                       << (int) conf.split_k_factor;
            }
            else
            {
                FAIL() << "Uninitialised tactic encountered";
            }
            tactic << " and activation type: " << static_cast<int>(activation_type);

            EXPECT_NO_THROW({
                this->mActType = activation_type;
                this->mSelectedConfig = conf;
                this->BasicPermuteTest();
                if (::testing::Test::HasFailure())
                    throw std::runtime_error("Test Failed");
            }) << tactic.str();
        }
    }
}

TYPED_TEST(LargeMixtureOfExpertsTest, PermuteVeryLargeExperts)
{
    // Chosen so that hidden_size * inter_size * num_experts >> 2^32, but we can still fit in 80GB for `half`
    // Uses a non-power of two so any integer overflow will have bad alignment
    int64_t hidden_size = 31 * 1024;
    ASSERT_GT(hidden_size * hidden_size * 4, (int64_t) std::numeric_limits<int>::max() + 1ull);

    int64_t k = 2; // Use k=2 so all experts get a value
    // 3 tokens 4 experts are the defaults for BasicPermuteTest
    if (!this->checkSufficientTestMemory(3, hidden_size, 4, k))
    {
        GTEST_SKIP() << "Insufficient free memory for test";
    }

    this->BasicPermuteTest(k, hidden_size); // 4 x 32k x 128K experts
}

TYPED_TEST(LargeMixtureOfExpertsTest, PermuteVeryLongSequence)
{
    this->mUseBias = !this->FP8;

    using DataType = typename TypeParam::DataType;
    // Sequence * hidden size > INT32_MAX
    int64_t hidden_size = 2048ll;
    int64_t num_experts = 4;
    int64_t k = 1;
    int64_t num_tokens = 1024ll * 1024ll + 1ll;
    int64_t tokens_to_test = 10;
    ASSERT_GT(hidden_size * num_tokens, (uint64_t) std::numeric_limits<int>::max() + 1ull);

    if (!this->checkSufficientTestMemory(num_tokens, hidden_size, num_experts, k))
    {
        GTEST_SKIP() << "Insufficient free memory for test";
    }

    std::vector<DataType> hidden_states(hidden_size * num_tokens);
    this->mMaxInput = 1.f; // Any arbitrary non-zero value

    // All tokens to expert 0, so we catch the case where an expert has more than 2^32 tokens
    float const token_probs[] = {1.f, 0.5f, 0.f, 0.f};
    std::vector<float> probs;
    probs.reserve(num_tokens * num_experts);
    for (size_t i = 0; i < num_tokens; i++)
    {
        probs.insert(probs.cend(), std::begin(token_probs), std::end(token_probs));
    }
    // Override the first few tokens to go to different experts.
    // This covers the regression case where an overflow only impacts one of the last experts
    // In particular the case when there are more than 2^32 elements before the last expert
    for (int i = 1; i < tokens_to_test; i++)
    {
        probs[i * num_experts + i % num_experts] = 2.f;
    }

    this->runMoEPermute({hidden_states}, {probs}, hidden_size, num_experts, k);

    // Just look at the first few tokens
    this->mTotalTokens = tokens_to_test;

    probs.resize(num_experts * this->mTotalTokens);
    hidden_states.resize(hidden_size * this->mTotalTokens);

    auto selected_expert = this->getDataFromDevice(this->mSelectedExpert, k * this->mTotalTokens);
    // We set the first few tokens to go to the corresponding i-th expert
    for (int i = 0; i < tokens_to_test; i++)
    {
        ASSERT_EQ(selected_expert[i], i % num_experts);
    }

    this->compareSoftmax(selected_expert, probs);
    // Create a default vector for the reference outputs of the correct type for FP8
    std::vector<typename TypeParam::OutputType> unquant_states(this->mTotalTokens * hidden_size);
    this->compareFinal(selected_expert, probs, unquant_states);
}
