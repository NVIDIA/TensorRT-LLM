#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

template <class T>
__global__ void initWeightsKernel(T* data, int w, int h, T scalar)
{
    size_t expert_id = blockIdx.z;
    T* start_offset = data + expert_id * w * h;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h)
        start_offset[y * w + x] = (x == y) ? scalar : 0;
}

template <class T>
__global__ void initBiasToExpertIdKernel(T* data, int w)
{
    size_t expert_id = blockIdx.y;
    T* start_offset = data + expert_id * w;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < w)
        start_offset[x] = expert_id;
}

class MixtureOfExpertsTest : public ::testing::Test
{
protected:
    using DataType = float;

    static BufferManager::CudaStreamPtr mStream;
    static std::unique_ptr<BufferManager> mBufferManager;
    static int mDeviceCount;

    std::vector<BufferManager::IBufferPtr> managed_buffers;
    float* mInputProbabilities{};
    DataType* mInputTensor{};

    int mMaxSeqLen = 0;

    int mHiddenSize{};
    int mNumExperts{};
    int mK{};

    static void SetUpTestCase()
    {
        mDeviceCount = getDeviceCount();
        if (mDeviceCount > 0)
        {
            mStream = std::make_shared<CudaStream>();
            mBufferManager = std::make_unique<BufferManager>(mStream);
        }
        else
        {
            GTEST_SKIP();
        }
    }

    static void TearDownTestCase()
    {
        mBufferManager.reset();
        mStream.reset();
    }

    void SetUp() override
    {
        assert(mBufferManager);
        if (mDeviceCount == 0)
        {
            GTEST_SKIP();
        }
    }

    void TearDown()
    {
        managed_buffers.clear();
    }

    void initWeights(DataType* buffer, int w, int h, DataType scalar)
    {
        dim3 block(16, 16, 1);
        dim3 grid(divUp(w, block.x), divUp(h, block.y), mNumExperts);
        initWeightsKernel<DataType><<<grid, block, 0, mStream->get()>>>(buffer, w, h, scalar);
    }

    void initBias(DataType* buffer, int w)
    {
        dim3 block(256, 1, 1);
        dim3 grid(divUp(w, block.x), mNumExperts);
        initBiasToExpertIdKernel<DataType><<<grid, block, 0, mStream->get()>>>(buffer, w);
    }

    CutlassMoeFCRunner<DataType, DataType> mMoERunner{};
    char* mWorkspace{};
    DataType* mScaleProbs{};
    DataType* mExpertWeight1{};
    DataType* mExpertWeight2{};
    DataType* mExpertBias1{};
    DataType* mExpertBias2{};

    DataType* mTpExpertScratch{}; // Copy the experts here when slicing up inputs
    size_t mTpExpertScratchSize{};

    DataType* mExpertOutput{};
    DataType* mFinalOutput{};
    int* mSourceToExpandedMap;
    int* mSelectedExpert;
    bool* mFinished{};
    int mInterSize{};
    int mTotalTokens{};
    int mActiveRows{};

    bool mUseBias = true;

    tensorrt_llm::ActivationType mActType = tensorrt_llm::ActivationType::Relu;
    MOEExpertScaleNormalizationMode mNormMode = MOEExpertScaleNormalizationMode::NONE;

    int mExpertWDiag1 = 1;
    int mExpertWDiag2 = 2;

    template <class T>
    T* allocBuffer(size_t size)
    {
        managed_buffers.emplace_back(mBufferManager->gpu(size * sizeof(T)));
        T* ptr = static_cast<T*>(managed_buffers.back()->data());
        return ptr;
    }

    void initBuffersPermute(std::vector<std::vector<DataType>> h_hidden_states,
        std::vector<std::vector<float>> h_router_results, int hidden_size, int num_experts, int k,
        std::vector<uint8_t> finished, MOEParallelismConfig parallelism_config)
    {
        managed_buffers.clear();

        mHiddenSize = hidden_size;
        mInterSize = hidden_size * 4;
        mNumExperts = num_experts;
        mK = k;

        mTotalTokens = 0;
        std::vector<int> h_seq_lens;
        h_seq_lens.push_back(0);
        for (auto& sequence : h_hidden_states)
        {
            assert(sequence.size() % hidden_size == 0);
            int num_tokens = sequence.size() / hidden_size;
            h_seq_lens.emplace_back(h_seq_lens.back() + num_tokens);
            mTotalTokens += num_tokens;
            mMaxSeqLen = std::max(mMaxSeqLen, num_tokens);
        }

        size_t workspace_size = mMoERunner.getWorkspaceSize(
            mTotalTokens, mHiddenSize, mInterSize, mNumExperts, mK, mActType, parallelism_config);

        const auto stream = mStream->get();

        mWorkspace = allocBuffer<char>(workspace_size);
        check_cuda_error(cudaMemsetAsync(mWorkspace, 0xD5, workspace_size, stream));
        const size_t expert_matrix_size = mNumExperts * mHiddenSize * mInterSize;

        mExpertWeight1 = allocBuffer<DataType>(expert_matrix_size);
        mExpertWeight2 = allocBuffer<DataType>(expert_matrix_size);

        mTpExpertScratchSize = 2 * expert_matrix_size / parallelism_config.tp_size;

        mExpertBias1 = nullptr;
        mExpertBias2 = nullptr;
        if (mUseBias)
        {
            // Allow space for the slice of bias1 in the scratch
            mTpExpertScratchSize += mNumExperts * mInterSize / parallelism_config.tp_size;
            mExpertBias1 = allocBuffer<DataType>(mNumExperts * mInterSize);
            mExpertBias2 = allocBuffer<DataType>(mNumExperts * mHiddenSize);

            check_cuda_error(cudaMemsetAsync(mExpertBias1, 0x0, mNumExperts * mInterSize * sizeof(DataType), stream));
            check_cuda_error(cudaMemsetAsync(mExpertBias2, 0x0, mNumExperts * mHiddenSize * sizeof(DataType), stream));
        }

        mExpertOutput = allocBuffer<DataType>(mTotalTokens * mHiddenSize * mK);

        mTpExpertScratch = nullptr;
        if (parallelism_config.tp_size > 1)
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
        mScaleProbs = allocBuffer<DataType>(mTotalTokens * mK);
        mInputTensor = allocBuffer<DataType>(mTotalTokens * mHiddenSize);
        mFinalOutput = allocBuffer<DataType>(mTotalTokens * mHiddenSize);

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

        // Init the diagonals of our matrix, this will set to the scalar value * expert_id
        initWeights(mExpertWeight1, mHiddenSize, mInterSize, mExpertWDiag1);
        initWeights(mExpertWeight2, mInterSize, mHiddenSize, mExpertWDiag2);

        if (mUseBias)
        {
            initBias(mExpertBias1, mInterSize);
            initBias(mExpertBias2, mHiddenSize);
        }

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
        check_cuda_error(cudaMemsetAsync(mScaleProbs, 0x0, sizeof(DataType) * mTotalTokens * mK, stream));
        check_cuda_error(
            cudaMemsetAsync(mExpertOutput, 0x0, mTotalTokens * mHiddenSize * mK * sizeof(DataType), stream));

        check_cuda_error(cudaStreamSynchronize(mStream->get()));
    }

    void runMoEPermute(std::vector<std::vector<DataType>> h_hidden_states,
        std::vector<std::vector<float>> h_router_results, int hidden_size, int num_experts, int k,
        std::vector<uint8_t> finished = {}, MOEParallelismConfig parallelism_config = {})
    {
        initBuffersPermute(std::move(h_hidden_states), std::move(h_router_results), hidden_size, num_experts, k,
            finished, parallelism_config);
        runMoEPermute(parallelism_config);
    }

    auto getWeights(MOEParallelismConfig parallelism_config)
    {
        if (parallelism_config.tp_size > 1)
        {
            const int tp_size = parallelism_config.tp_size;
            const int tp_rank = parallelism_config.tp_rank;

            const size_t matrix_size = mHiddenSize * mInterSize / tp_size;

            auto* weight_1 = mTpExpertScratch;
            auto* weight_2 = weight_1 + mNumExperts * matrix_size;
            auto* bias_1 = mUseBias ? weight_2 + mNumExperts * matrix_size : nullptr;

            // 2D memcpy just the slices we care about
            const size_t row_size_1 = matrix_size * sizeof(DataType);
            check_cuda_error(cudaMemcpy2DAsync(weight_1, row_size_1, (uint8_t*) mExpertWeight1 + row_size_1 * tp_rank,
                row_size_1 * tp_size, row_size_1, mNumExperts, cudaMemcpyDeviceToDevice, mStream->get()));

            const size_t row_size_2 = mInterSize / tp_size * sizeof(DataType);
            check_cuda_error(cudaMemcpy2DAsync(weight_2, row_size_2, (uint8_t*) mExpertWeight2 + row_size_2 * tp_rank,
                row_size_2 * tp_size, row_size_2, mNumExperts * mHiddenSize, cudaMemcpyDeviceToDevice, mStream->get()));

            if (mUseBias)
            {
                const size_t row_size_bias = mInterSize / tp_size * sizeof(DataType);
                check_cuda_error(
                    cudaMemcpy2DAsync(bias_1, row_size_bias, (uint8_t*) mExpertBias1 + row_size_bias * tp_rank,
                        row_size_bias * tp_size, row_size_bias, mNumExperts, cudaMemcpyDeviceToDevice, mStream->get()));
            }

            return std::tuple{weight_1, weight_2, bias_1, mExpertBias2};
        }
        else if (parallelism_config.ep_size > 1)
        {
            const size_t experts_per_node = mNumExperts / parallelism_config.ep_size;
            const size_t weight_matrix_size = mHiddenSize * mInterSize * experts_per_node;
            const size_t bias_fc1_size = mInterSize * experts_per_node;
            const size_t bias_fc2_size = mHiddenSize * experts_per_node;
            auto* weight1_ptr = mExpertWeight1 + weight_matrix_size * parallelism_config.ep_rank;
            auto* weight2_ptr = mExpertWeight2 + weight_matrix_size * parallelism_config.ep_rank;
            auto* bias1_ptr = mUseBias ? mExpertBias1 + bias_fc1_size * parallelism_config.ep_rank : nullptr;
            auto* bias2_ptr = mUseBias ? mExpertBias2 + bias_fc2_size * parallelism_config.ep_rank : nullptr;
            return std::tuple{weight1_ptr, weight2_ptr, bias1_ptr, bias2_ptr};
        }
        return std::tuple{mExpertWeight1, mExpertWeight2, mExpertBias1, mExpertBias2};
    }

    void runMoEPermute(MOEParallelismConfig parallelism_config)
    {
        // Clear the buffers to blank so we can assume zero if not written
        resetOutBuffers();

        const auto [weight1_ptr, weight2_ptr, bias1_ptr, bias2_ptr] = getWeights(parallelism_config);

        auto stream = mStream->get();
        mMoERunner.setTactic(std::nullopt);
        mMoERunner.runMoe(mInputTensor, mInputProbabilities, weight1_ptr, nullptr, bias1_ptr, mActType, weight2_ptr,
            nullptr, bias2_ptr, mTotalTokens, mHiddenSize, mInterSize / parallelism_config.tp_size, mNumExperts, mK,
            mWorkspace, mFinalOutput, mExpertOutput, mFinished, mActiveRows, mScaleProbs, mSourceToExpandedMap,
            mSelectedExpert, parallelism_config, mNormMode, stream);
        check_cuda_error(cudaStreamSynchronize(mStream->get()));
    }

    template <class T>
    std::vector<T> getDataFromDevice(const T* in, size_t length)
    {
        std::vector<T> data(length);

        const auto stream = mStream->get();
        check_cuda_error(cudaMemcpyAsync(data.data(), in, length * sizeof(T), cudaMemcpyDeviceToHost, stream));
        check_cuda_error(cudaStreamSynchronize(mStream->get()));

        return data;
    }

    auto maskSelectedExpertsForTP(const std::vector<int>& vector, int tp_size, int tp_rank)
    {
        std::vector<int> result;
        int num_experts_per_node = mNumExperts / tp_size;
        std::transform(vector.begin(), vector.end(), std::back_inserter(result),
            [=](int entry)
            {
                if (entry >= num_experts_per_node * tp_rank && entry < num_experts_per_node * (tp_rank + 1))
                    return entry;
                return mNumExperts;
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
                std::cout << cast(v) << ", ";                                                                          \
            std::cout << std::endl;                                                                                    \
        }                                                                                                              \
    while (0)
#define PRINT(array, size) PRINT_CAST(array, size, )

        PRINT(mExpertWeight1, mNumExperts * mHiddenSize * mInterSize);
        PRINT(mExpertWeight2, mNumExperts * mHiddenSize * mInterSize);
        PRINT(mExpertBias1, mNumExperts * mInterSize);
        PRINT(mExpertBias2, mNumExperts * mHiddenSize);
        PRINT(mExpertOutput, mTotalTokens * mK * mHiddenSize);
        PRINT(mFinalOutput, mTotalTokens * mK * mHiddenSize);
        PRINT_CAST((uint8_t*) mFinished, mTotalTokens, (int) );
        PRINT(mInputProbabilities, mTotalTokens * mNumExperts);
        PRINT(mScaleProbs, mTotalTokens * mK);
        PRINT(mInputProbabilities, mTotalTokens * mNumExperts);
        PRINT(mInputTensor, mTotalTokens * mHiddenSize);
        PRINT(mSourceToExpandedMap, mTotalTokens * mK);
        PRINT(mSelectedExpert, mTotalTokens * mK);

#undef PRINT_CAST
#undef PRINT
    }

    DataType actfn(DataType in)
    {
        if (mActType == tensorrt_llm::ActivationType::Identity)
            return in;
        if (mActType == tensorrt_llm::ActivationType::Relu)
            return std::max(in, 0.0f);
        assert(false);
        return in;
    }

    DataType calcMLPVal(DataType input, int expert_id, bool final_bias = false)
    {
        if (expert_id >= mNumExperts)
            return 0;
        auto fc1 = input * mExpertWDiag1 + (DataType) (mUseBias ? expert_id : 0);
        auto activated = actfn(fc1) * mExpertWDiag2;
        return activated + (DataType) (final_bias ? expert_id : 0);
    }

    DataType calcMLPValWithFinalBias(DataType input, int expert_id)
    {
        return calcMLPVal(input, expert_id, mUseBias);
    }

    void comparePermuted(const std::vector<int>& expected_experts, const std::vector<int>& expected_permutation,
        const std::vector<DataType>& input_data)
    {
        auto states = getDataFromDevice(mExpertOutput, mTotalTokens * mK * mHiddenSize);

        // Loop for the number of times each token is duplicated
        for (int k_idx = 0; k_idx < mK; k_idx++)
        {
            for (int token_id = 0; token_id < mTotalTokens; token_id++)
            {
                // Permutation has the position of the first copy of all token,
                // followed by the position of the second copy of all tokens etc.
                const int permuted_position = expected_permutation[k_idx * mTotalTokens + token_id];

                // Expected experts has all the selected experts for token one,
                // followed by all the selected experts for token two etc.
                const int expert_id = expected_experts[token_id * mK + k_idx];

                // Compare the copied tokens with the projection applied
                for (int hidden_id = 0; hidden_id < mHiddenSize; hidden_id++)
                {
                    EXPECT_FLOAT_EQ(calcMLPVal(input_data[token_id * mHiddenSize + hidden_id], expert_id),
                        states[permuted_position * mHiddenSize + hidden_id])
                        << "Incorrect value at position: mK: " << k_idx << ", token: " << token_id
                        << ", permuted dest: " << permuted_position << ", expert id: " << expert_id;
                }
            }
        }
    }

    std::vector<DataType> softmax(const std::vector<DataType>& expected_probs)
    {
        std::vector<DataType> result;
        // All values we test are 0-1 so we can skip the normalization step
        std::transform(expected_probs.begin(), expected_probs.end(), std::back_inserter(result),
            [&](const DataType in)
            {
                auto res = exp(in);
                return res;
            });

        for (int token = 0; token < mTotalTokens; token++)
        {
            auto start = result.begin() + token * mNumExperts;
            auto end = start + mNumExperts;
            auto sum = std::accumulate(start, end, (DataType) 0);
            std::transform(start, end, start, [=](auto in) { return in / sum; });
        }

        return result;
    }

    void compareSoftmax(const std::vector<int>& expected_experts, const std::vector<DataType>& expected_probs,
        std::vector<DataType> scale_probs = {})
    {
        if (scale_probs.empty())
            scale_probs = getDataFromDevice(mScaleProbs, mTotalTokens * mK);
        auto softmax_probs = softmax(expected_probs);
        for (int token_id = 0; token_id < mTotalTokens; token_id++)
        {
            for (int k_idx = 0; k_idx < mK; k_idx++)
            {
                int selected_expert = expected_experts[token_id * mK + k_idx];
                if (selected_expert < mNumExperts) // Ignore 'finished' values
                {
                    ASSERT_FLOAT_EQ(
                        softmax_probs[token_id * mNumExperts + selected_expert], scale_probs[token_id * mK + k_idx])
                        << "Scales mismatched for token: " << token_id << " k: " << k_idx
                        << " selected_expert: " << selected_expert;
                }
            }
        }
    }

    void renormScales(DataType* probs, const int* experts)
    {
        if (mNormMode == MOEExpertScaleNormalizationMode::NONE)
            return;
        DataType sum = 0;
        for (int k_idx = 0; k_idx < mK; k_idx++)
        {
            sum += probs[experts[k_idx]];
        }
        DataType norm_factor = 1.0 / sum;
        for (int k_idx = 0; k_idx < mK; k_idx++)
        {
            probs[experts[k_idx]] *= norm_factor;
        }
    }

    void compareFinal(const std::vector<int>& expected_experts, const std::vector<DataType>& expected_probs,
        const std::vector<DataType>& input_data, std::vector<DataType> final_results = {})
    {
        if (final_results.empty())
            final_results = getDataFromDevice(mFinalOutput, mTotalTokens * mHiddenSize);

        auto softmax_probs = softmax(expected_probs);
        for (int token_id = 0; token_id < mTotalTokens; token_id++)
        {
            // Compare the copied tokens with the projection applied
            for (int hidden_id = 0; hidden_id < mHiddenSize; hidden_id++)
            {
                renormScales(&softmax_probs[token_id * mNumExperts], &expected_experts[token_id * mK]);

                DataType sum = 0.0f;
                // Loop for the number of times each token is duplicated
                for (int k_idx = 0; k_idx < mK; k_idx++)
                {
                    int selected_expert = expected_experts[token_id * mK + k_idx];
                    sum += calcMLPValWithFinalBias(input_data[token_id * mHiddenSize + hidden_id], selected_expert)
                        * softmax_probs[token_id * mNumExperts + selected_expert];
                }

                EXPECT_FLOAT_EQ(sum, final_results[token_id * mHiddenSize + hidden_id])
                    << "Incorrect final value at position: " << token_id * mHiddenSize + hidden_id;
            }
        }
    }

    void BasicPermuteTest(int k = 1);

    std::vector<int> calcPermuteMapExpertParallel(const std::vector<int>& expected_experts);
    void ExpertParallelTest(int k = 1);

    void TensorParallelTest(int k = 1);
};

BufferManager::CudaStreamPtr MixtureOfExpertsTest::mStream{};
std::unique_ptr<BufferManager> MixtureOfExpertsTest::mBufferManager{};
int MixtureOfExpertsTest::mDeviceCount{};

const int DEFAULT_HIDDEN_SIZE = 4;

void MixtureOfExpertsTest::BasicPermuteTest(int k)
{
    int hidden_size = DEFAULT_HIDDEN_SIZE;
    int num_experts = 4;
    int num_tokens = 3;

    std::vector<DataType> hidden_states(hidden_size * num_tokens, 0);
    std::iota(hidden_states.begin(), hidden_states.end(), 0.0f);

    std::vector<float> probs = {
        0.5, 0.1, 0.25, 0.15,   //
        0.03, 0.2, 0.07, 0.7,   //
        0.25, 0.21, 0.35, 0.19, //
    };

    runMoEPermute({hidden_states}, {probs}, hidden_size, num_experts, k);

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
    comparePermuted(selected_expert, permute_map, hidden_states);
    compareSoftmax(selected_expert, probs);
    compareFinal(selected_expert, probs, hidden_states);
}

TEST_F(MixtureOfExpertsTest, Permute)
{
    BasicPermuteTest();
}

TEST_F(MixtureOfExpertsTest, PermuteK2)
{
    BasicPermuteTest(2);
}

TEST_F(MixtureOfExpertsTest, PermuteK3)
{
    BasicPermuteTest(3);
}

TEST_F(MixtureOfExpertsTest, PermuteNoBias)
{
    mUseBias = false;
    BasicPermuteTest();
    BasicPermuteTest(2);
    BasicPermuteTest(3);
}

TEST_F(MixtureOfExpertsTest, PermuteRenormalization)
{
    mNormMode = tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE;
    BasicPermuteTest();
    BasicPermuteTest(2);
    BasicPermuteTest(3);
}

TEST_F(MixtureOfExpertsTest, Finished)
{
    int hidden_size = DEFAULT_HIDDEN_SIZE;
    int num_experts = 4;
    int num_tokens = 3;
    int k = 2;

    std::vector<DataType> hidden_states(hidden_size * num_tokens, 0);
    std::iota(hidden_states.begin(), hidden_states.end(), 0.0f);

    std::vector<float> probs = {
        0.5, 0.1, 0.25, 0.15, //
        0.05, 0.2, 0.05, 0.7, //
        0.25, 0.2, 0.35, 0.2, //
    };

    runMoEPermute({hidden_states}, {probs}, hidden_size, num_experts, k, {0, 0, 1});

    auto selected_expert = getDataFromDevice(mSelectedExpert, num_tokens * k);
    // Token 1
    EXPECT_EQ(selected_expert[0], 0);
    EXPECT_EQ(selected_expert[1], 2);
    // Token 2
    EXPECT_EQ(selected_expert[2], 3);
    EXPECT_EQ(selected_expert[3], 1);
    // Token 3
    EXPECT_EQ(selected_expert[4], num_experts); // One past the end
    EXPECT_EQ(selected_expert[5], num_experts);

    auto proj_map = getDataFromDevice(mSourceToExpandedMap, num_tokens * k);
    // This is the final position of:
    // Token 1 Expert 1, T2E1, T3E1, T1E2, T2E2, T3E3
    std::vector<int> permute_map{0, 3, 4, 2, 1, 5};
    ASSERT_EQ(permute_map, proj_map);
    comparePermuted(selected_expert, permute_map, hidden_states);
    compareSoftmax(selected_expert, probs);
    compareFinal(selected_expert, probs, hidden_states);
}

std::vector<int> MixtureOfExpertsTest::calcPermuteMapExpertParallel(const std::vector<int>& expected_experts)
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

void MixtureOfExpertsTest::ExpertParallelTest(int k)
{
    int hidden_size = DEFAULT_HIDDEN_SIZE;
    int parallelism = 2;
    int num_experts = 4;
    int num_tokens = 3;

    std::vector<DataType> hidden_states(hidden_size * num_tokens, 0);
    std::iota(hidden_states.begin(), hidden_states.end(), 0.0f);

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
    std::vector<DataType> results(hidden_states.size(), 0);
    for (int i = 0; i < parallelism; i++)
    {
        if (i == 0)
        {
            // Only need to init the inputs on the first iteration
            runMoEPermute({hidden_states}, {probs}, hidden_size, num_experts, k, {},
                MOEParallelismConfig::ExpertParallelism(parallelism, i));
        }
        else
        {
            runMoEPermute(MOEParallelismConfig::ExpertParallelism(parallelism, i));
        }

        auto selected_expert = getDataFromDevice(mSelectedExpert, num_tokens * k);
        // Experts should only be selected when we are on the right node
        // Note the index is [0,num_experts_per_node), so we offset the experts by the start for this node
        const int start_expert = i * (mNumExperts / parallelism);
        std::transform(selected_expert.begin(), selected_expert.end(), selected_expert.begin(),
            [&](int val) { return val == mNumExperts ? mNumExperts : val + start_expert; });
        auto masked_expected_experts = maskSelectedExpertsForTP(expected_experts, parallelism, i);
        ASSERT_EQ(selected_expert, masked_expected_experts);

        auto proj_map = getDataFromDevice(mSourceToExpandedMap, num_tokens * k);
        auto permute_map = calcPermuteMapExpertParallel(masked_expected_experts);
        ASSERT_EQ(permute_map, proj_map) << "Iteration " << i;
        comparePermuted(masked_expected_experts, permute_map, hidden_states);
        compareSoftmax(expected_experts, probs);

        // Do the final reduce
        auto iter_results = getDataFromDevice(mFinalOutput, num_tokens * hidden_size);
        std::transform(iter_results.cbegin(), iter_results.cend(), results.cbegin(), results.begin(), std::plus<>{});
    }

    compareFinal(expected_experts, probs, hidden_states, results);
}

TEST_F(MixtureOfExpertsTest, ExpertParallel)
{
    ExpertParallelTest();
}

TEST_F(MixtureOfExpertsTest, ExpertParallelK2)
{
    ExpertParallelTest(2);
}

TEST_F(MixtureOfExpertsTest, ExpertParallelNoBias)
{
    mUseBias = false;
    ExpertParallelTest();
    ExpertParallelTest(2);
}

TEST_F(MixtureOfExpertsTest, ExpertParallelRenorm)
{
    mNormMode = MOEExpertScaleNormalizationMode::RENORMALIZE;
    ExpertParallelTest();
    ExpertParallelTest(2);
}

void MixtureOfExpertsTest::TensorParallelTest(int k)
{
    int hidden_size = DEFAULT_HIDDEN_SIZE;
    int parallelism = 8;
    int num_experts = 4;
    int num_tokens = 3;

    std::vector<DataType> hidden_states(hidden_size * num_tokens, 0);
    std::iota(hidden_states.begin(), hidden_states.end(), 0.0f);

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
    std::vector<DataType> results(hidden_states.size(), 0);
    for (int i = 0; i < parallelism; i++)
    {
        if (i == 0)
        {
            // Only need to init the inputs on the first iteration
            runMoEPermute({hidden_states}, {probs}, hidden_size, num_experts, k, {},
                MOEParallelismConfig::TensorParallelism(parallelism, i));
        }
        else
        {
            runMoEPermute(MOEParallelismConfig::TensorParallelism(parallelism, i));
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

    compareFinal(expected_experts, probs, hidden_states, results);
}

TEST_F(MixtureOfExpertsTest, TensorParallel)
{
    TensorParallelTest();
}

TEST_F(MixtureOfExpertsTest, TensorParallelK2)
{
    TensorParallelTest(2);
}

TEST_F(MixtureOfExpertsTest, TensorParallelK3)
{
    TensorParallelTest(3);
}

TEST_F(MixtureOfExpertsTest, TensorParallelNoBias)
{
    mUseBias = false;
    TensorParallelTest();
    TensorParallelTest(2);
    TensorParallelTest(3);
}

TEST_F(MixtureOfExpertsTest, TensorParallelRenorm)
{
    mNormMode = MOEExpertScaleNormalizationMode::RENORMALIZE;
    TensorParallelTest();
    TensorParallelTest(2);
    TensorParallelTest(3);
}
