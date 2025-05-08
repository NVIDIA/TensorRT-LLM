#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include "tensorrt_llm/kernels/mlaChunkedPrefill.cuh"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <random>

namespace
{
// Q {B, S_Q, H, D}
// KV {B, 2, H, S_KV, D}
// softmax_sum {2, B, S_Q, H}
// output {B, S_Q, H, D}
// S_Q <= S_KV
template <typename T>
void selfAttentionRef(T* output, T* const Q, T* const KV, int batch_size, int num_heads, int seq_len_q, int seq_len_kv,
    int head_size, bool return_softmax, float* softmax_sum, bool causal_mask)
{
    std::vector<float> P(seq_len_q * seq_len_kv);

    for (int b = 0; b < batch_size; b++)
    {
        for (int h = 0; h < num_heads; h++)
        {
            // BMM1
            std::fill(P.begin(), P.end(), std::numeric_limits<double>::lowest());
            T* const q_ptr = Q + b * seq_len_q * num_heads * head_size;
            T* const k_ptr = KV + b * 2 * num_heads * seq_len_kv * head_size + h * seq_len_kv * head_size;
            T* const v_ptr = k_ptr + num_heads * seq_len_kv * head_size;
            T* output_ptr = output + b * seq_len_q * num_heads * head_size;
            for (int s_q = 0; s_q < seq_len_q; s_q++)
            {
                float softmax_max = std::numeric_limits<double>::lowest();
                for (int s_kv = 0; s_kv < seq_len_kv; s_kv++)
                {
                    // lower right mask
                    if (causal_mask && s_kv > seq_len_kv - seq_len_q + s_q)
                    {
                        break;
                    }
                    P[s_q * seq_len_kv + s_kv] = 0;
                    for (int d = 0; d < head_size; d++)
                    {
                        P[s_q * seq_len_kv + s_kv] += static_cast<float>(
                            q_ptr[s_q * num_heads * head_size + h * head_size + d] * k_ptr[s_kv * head_size + d]);
                    }
                    if (softmax_max < P[s_q * seq_len_kv + s_kv])
                    {
                        softmax_max = P[s_q * seq_len_kv + s_kv];
                    }
                }
                for (int s_kv = 0; s_kv < seq_len_kv; s_kv++)
                {
                    // lower right mask
                    if (causal_mask && s_kv > seq_len_kv - seq_len_q + s_q)
                    {
                        break;
                    }
                    P[s_q * seq_len_kv + s_kv] -= softmax_max;
                }
                if (return_softmax)
                {
                    softmax_sum[batch_size * seq_len_q * num_heads + b * seq_len_q * num_heads + s_q * num_heads + h]
                        = softmax_max;
                }
            }
            // softmax
            for (int s_q = 0; s_q < seq_len_q; s_q++)
            {
                float sum = 0;
                for (int s_kv = 0; s_kv < seq_len_kv; s_kv++)
                {
                    P[s_q * seq_len_kv + s_kv] = std::exp(P[s_q * seq_len_kv + s_kv]);
                    sum += P[s_q * seq_len_kv + s_kv];
                }
                for (int s_kv = 0; s_kv < seq_len_kv; s_kv++)
                {
                    P[s_q * seq_len_kv + s_kv] /= sum;
                }
                if (return_softmax)
                {
                    softmax_sum[b * seq_len_q * num_heads + s_q * num_heads + h] = sum;
                }
            }
            // BMM2
            for (int s_q = 0; s_q < seq_len_q; s_q++)
            {
                for (int d = 0; d < head_size; d++)
                {
                    output_ptr[s_q * num_heads * head_size + h * head_size + d] = 0;
                    for (int s_kv = 0; s_kv < seq_len_kv; s_kv++)
                    {
                        output_ptr[s_q * num_heads * head_size + h * head_size + d] += static_cast<T>(
                            P[s_q * seq_len_kv + s_kv] * static_cast<float>(v_ptr[s_kv * head_size + d]));
                    }
                }
            }
        }
    }
}

// chunked_KV {B, 2, H, S=chunked_size, D}
// KV {B, 2, H, S_KV, D}
template <typename T>
void copyRelatedChunkedKV(T* chunked_kv, T* const kv, int chunk_idx, int batch_size, int num_heads, int chunked_size,
    int seq_len_kv, int head_size)
{
    for (int b = 0; b < batch_size; b++)
    {
        for (int kv_type = 0; kv_type < 2; kv_type++)
        {
            for (int h = 0; h < num_heads; h++)
            {
                T* chunked_kv_ptr = chunked_kv + b * 2 * num_heads * chunked_size * head_size
                    + kv_type * num_heads * chunked_size * head_size + h * chunked_size * head_size;
                T* const kv_ptr = kv + b * 2 * num_heads * seq_len_kv * head_size
                    + kv_type * num_heads * seq_len_kv * head_size + h * seq_len_kv * head_size
                    + chunk_idx * chunked_size * head_size;
                std::memcpy(chunked_kv_ptr, kv_ptr, chunked_size * head_size * sizeof(T));
            }
        }
    }
}

template <typename WeightType>
float getTolerance(float scale = 1.f)
{
    float tol = 0.0;
    if constexpr (std::is_same_v<WeightType, uint8_t>)
    {
        tol = 0.1;
    }
    else if constexpr (std::is_same_v<WeightType, float>)
    {
        tol = 0.001;
    }
    else if constexpr (std::is_same_v<WeightType, half>)
    {
        tol = 0.005;
    }
    else if constexpr (std::is_same_v<WeightType, __nv_bfloat16>)
    {
        tol = 0.05;
    }
    // Keep the scale in a sane range
    return std::max(tol, scale * tol);
}
}; // namespace

template <typename _DataType>
class MlaChunkedPrefillTest : public ::testing::Test
{
protected:
    using DataType = _DataType;

    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;

    tensorrt_llm::runtime::BufferManager::ITensorPtr h_q_tensor{nullptr}, h_kv_tensor{nullptr},
        h_chunked_kv_tensor{nullptr}, h_output_tensor{nullptr}, h_softmax_sum_tensor{nullptr},
        h_softmax_sum_accum_tensor{nullptr}, h_output_tensor_ref{nullptr}, h_output_tensor_accum_ref{nullptr},
        d_q_tensor{nullptr}, d_kv_tensor{nullptr}, d_chunked_kv_tensor{nullptr}, d_output_tensor{nullptr},
        d_softmax_sum_tensor{nullptr}, d_softmax_sum_accum_tensor{nullptr}, d_output_tensor_ref{nullptr},
        d_output_tensor_accum_ref{nullptr};

    int mBatchSize{};
    int mSeqLen{};    // seq_len_kv
    int mChunkSize{}; // seq_len_q
    int mNumHeads{};
    int mHeadSize{};
    bool mIsCausalMask{};

    std::mt19937 gen;

    void SetUp() override
    {
        if (shouldSkip())
        {
            GTEST_SKIP() << "Skipping mla chunked prefill test";
        }
        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        gen.seed(42U);
    }

    static bool shouldSkip()
    {
        return false;
    }

    void setDefaultParams()
    {
        mBatchSize = 2;
        mSeqLen = 256;
        mChunkSize = 32;
        mNumHeads = 4;
        mHeadSize = 32;
        mIsCausalMask = false;
    }

    void memsetZeroHost(tensorrt_llm::runtime::BufferManager::ITensorPtr& tensor)
    {
        void* ptr = tensor->data();
        std::memset(ptr, 0, tensor->getSizeInBytes());
    }

    template <typename T>
    void showHostTensor(tensorrt_llm::runtime::BufferManager::ITensorPtr& tensor)
    {
        auto* const ptr = reinterpret_cast<T*>(tensor->data());
        for (int _ = 0; _ < tensor->getSize(); _++)
        {
            std::cout << static_cast<float>(ptr[_]) << " ";
        }
        std::cout << std::endl;
    }

    float generateRandomFloat(float min, float max)
    {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(gen);
    }

    template <typename T>
    void generateRandomData(T* data, int size)
    {
        for (int i = 0; i < size; i++)
        {
            data[i] = static_cast<T>(generateRandomFloat(-1.0f, 1.0f));
        }
    }

    bool allocateBuffers()
    {
        using tensorrt_llm::runtime::BufferManager;
        using tensorrt_llm::runtime::CudaStream;
        using tensorrt_llm::runtime::ITensor;
        using tensorrt_llm::runtime::bufferCast;

        auto dtype = nvinfer1::DataType::kHALF;
        if constexpr (std::is_same_v<DataType, float>)
        {
            dtype = nvinfer1::DataType::kFLOAT;
        }
        else if constexpr (std::is_same_v<DataType, half>)
        {
            dtype = nvinfer1::DataType::kHALF;
        }
        else if constexpr (std::is_same_v<DataType, __nv_bfloat16>)
        {
            dtype = nvinfer1::DataType::kBF16;
        }
        else
        {
            return false;
        }
        this->h_q_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize, this->mChunkSize, this->mNumHeads, this->mHeadSize}), dtype);
        this->h_kv_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize, 2, this->mNumHeads, this->mSeqLen, this->mHeadSize}), dtype);
        this->h_chunked_kv_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize, 2, this->mNumHeads, this->mChunkSize, this->mHeadSize}), dtype);
        this->h_output_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize, this->mChunkSize, this->mNumHeads, this->mHeadSize}), dtype);
        this->h_softmax_sum_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({2, this->mBatchSize, this->mChunkSize, this->mNumHeads}), nvinfer1::DataType::kFLOAT);
        this->h_softmax_sum_accum_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({2, this->mBatchSize, this->mChunkSize, this->mNumHeads}), nvinfer1::DataType::kFLOAT);
        this->h_output_tensor_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize, this->mChunkSize, this->mNumHeads, this->mHeadSize}), dtype);
        this->h_output_tensor_accum_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize, this->mChunkSize, this->mNumHeads, this->mHeadSize}), dtype);
        this->d_q_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_q_tensor->getShape(), dtype);
        this->d_kv_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_kv_tensor->getShape(), dtype);
        this->d_chunked_kv_tensor
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_chunked_kv_tensor->getShape(), dtype);
        this->d_output_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_output_tensor->getShape(), dtype);
        this->d_softmax_sum_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->h_softmax_sum_tensor->getShape(), nvinfer1::DataType::kFLOAT);
        this->d_softmax_sum_accum_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->h_softmax_sum_accum_tensor->getShape(), nvinfer1::DataType::kFLOAT);
        this->d_output_tensor_ref
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_output_tensor_ref->getShape(), dtype);
        this->d_output_tensor_accum_ref
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_output_tensor_accum_ref->getShape(), dtype);

        {
            auto* q_ptr = bufferCast<DataType>(*(this->h_q_tensor));
            auto* kv_ptr = bufferCast<DataType>(*(this->h_kv_tensor));

            generateRandomData(q_ptr, h_q_tensor->getSize());
            generateRandomData(kv_ptr, h_kv_tensor->getSize());
            this->memsetZeroHost(h_chunked_kv_tensor);
            this->memsetZeroHost(h_output_tensor);
            this->memsetZeroHost(h_softmax_sum_tensor);
            this->memsetZeroHost(h_softmax_sum_accum_tensor);
            this->memsetZeroHost(h_output_tensor_ref);
            this->memsetZeroHost(h_output_tensor_accum_ref);

            // Copy data to device
            cudaMemcpyAsync(d_q_tensor->data(), h_q_tensor->data(), h_q_tensor->getSizeInBytes(),
                cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(d_kv_tensor->data(), h_kv_tensor->data(), h_kv_tensor->getSizeInBytes(),
                cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(d_chunked_kv_tensor->data(), h_chunked_kv_tensor->data(),
                h_chunked_kv_tensor->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(d_output_tensor->data(), h_output_tensor->data(), h_output_tensor->getSizeInBytes(),
                cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(d_softmax_sum_tensor->data(), h_softmax_sum_tensor->data(),
                h_softmax_sum_tensor->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(d_softmax_sum_accum_tensor->data(), h_softmax_sum_accum_tensor->data(),
                h_softmax_sum_accum_tensor->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(d_output_tensor_ref->data(), h_output_tensor_ref->data(),
                h_output_tensor_ref->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(d_output_tensor_accum_ref->data(), h_output_tensor_accum_ref->data(),
                h_output_tensor_accum_ref->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaStreamSynchronize(mStream->get());
        }
        return true;
    }

    void PerformNormalAttention()
    {
        using tensorrt_llm::runtime::bufferCast;

        auto* q_ptr = bufferCast<DataType>(*(this->h_q_tensor));
        auto* kv_ptr = bufferCast<DataType>(*(this->h_kv_tensor));
        auto* output_ptr = bufferCast<DataType>(*(this->h_output_tensor));
        selfAttentionRef(output_ptr, q_ptr, kv_ptr, this->mBatchSize, this->mNumHeads, this->mChunkSize, this->mSeqLen,
            this->mHeadSize, false, nullptr, this->mIsCausalMask);
    }

    void PerformMergedAttention()
    {
        using tensorrt_llm::runtime::bufferCast;

        auto* h_q_ptr = bufferCast<DataType>(*(this->h_q_tensor));
        auto* h_kv_ptr = bufferCast<DataType>(*(this->h_kv_tensor));
        auto* h_chunked_kv_ptr = bufferCast<DataType>(*(this->h_chunked_kv_tensor));
        auto* h_output_ptr = bufferCast<DataType>(*(this->h_output_tensor_ref));
        auto* h_output_accum_ptr = bufferCast<DataType>(*(this->h_output_tensor_accum_ref));
        auto* h_softmax_sum_ptr = bufferCast<float>(*(this->h_softmax_sum_tensor));
        auto* h_softmax_sum_accum_ptr = bufferCast<float>(*(this->h_softmax_sum_accum_tensor));
        auto* d_kv_ptr = bufferCast<DataType>(*(this->d_kv_tensor));
        auto* d_chunked_kv_ptr = bufferCast<DataType>(*(this->d_chunked_kv_tensor));
        auto* d_softmax_sum_ptr = bufferCast<float>(*(this->d_softmax_sum_tensor));
        auto* d_softmax_sum_accum_ptr = bufferCast<float>(*(this->d_softmax_sum_accum_tensor));
        auto* d_output_ptr = bufferCast<DataType>(*(this->d_output_tensor_ref));
        auto* d_output_accum_ptr = bufferCast<DataType>(*(this->d_output_tensor_accum_ref));

        int const loop_count = this->mSeqLen / this->mChunkSize;
        for (int _ = 0; _ < loop_count; _++)
        {
            // copy related kv chunk data
            copyRelatedChunkedKV(h_chunked_kv_ptr, h_kv_ptr, _, this->mBatchSize, this->mNumHeads, this->mChunkSize,
                this->mSeqLen, this->mHeadSize);
            // attention
            selfAttentionRef(h_output_ptr, h_q_ptr, h_chunked_kv_ptr, this->mBatchSize, this->mNumHeads,
                this->mChunkSize, this->mChunkSize, this->mHeadSize, true, h_softmax_sum_ptr,
                this->mIsCausalMask && (_ == loop_count - 1));
            // merge attention
            if (_ == 0)
            {
                std::memcpy(h_softmax_sum_accum_ptr, h_softmax_sum_ptr, this->h_softmax_sum_tensor->getSizeInBytes());
                std::memcpy(h_output_accum_ptr, h_output_ptr, this->h_output_tensor->getSizeInBytes());
            }
            else
            {
                // copy curr_attn and softmax_sum to device
                cudaMemcpyAsync(d_softmax_sum_accum_ptr, h_softmax_sum_accum_ptr,
                    this->h_softmax_sum_accum_tensor->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
                cudaMemcpyAsync(d_softmax_sum_ptr, h_softmax_sum_ptr, this->h_softmax_sum_tensor->getSizeInBytes(),
                    cudaMemcpyHostToDevice, mStream->get());
                cudaMemcpyAsync(d_output_accum_ptr, h_output_accum_ptr,
                    this->h_output_tensor_accum_ref->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
                cudaMemcpyAsync(d_output_ptr, h_output_ptr, this->h_output_tensor->getSizeInBytes(),
                    cudaMemcpyHostToDevice, mStream->get());
                sync_check_cuda_error(mStream->get());
                // merge softmax
                invokeMergeAttnWithSoftmax<DataType>(d_output_accum_ptr, d_softmax_sum_accum_ptr, d_output_accum_ptr,
                    d_softmax_sum_accum_ptr, d_output_ptr, d_softmax_sum_ptr, this->mBatchSize, this->mChunkSize,
                    this->mNumHeads, this->mHeadSize, mStream->get());
                sync_check_cuda_error(mStream->get());
                // copy merged softmax sum back to host
                cudaMemcpyAsync(h_softmax_sum_accum_ptr, d_softmax_sum_accum_ptr,
                    this->h_softmax_sum_tensor->getSizeInBytes(), cudaMemcpyDeviceToHost, mStream->get());
                cudaMemcpyAsync(h_output_accum_ptr, d_output_accum_ptr, this->h_output_tensor->getSizeInBytes(),
                    cudaMemcpyDeviceToHost, mStream->get());
                sync_check_cuda_error(mStream->get());
            }
        }
    }
};

using MLATypes = ::testing::Types<half, __nv_bfloat16, float>;
TYPED_TEST_SUITE(MlaChunkedPrefillTest, MLATypes);

TYPED_TEST(MlaChunkedPrefillTest, MlaChunkedPrefillDefault)
{
    using tensorrt_llm::runtime::bufferCast;
    using DataType = typename TestFixture::DataType;
    this->setDefaultParams();
    this->allocateBuffers();

    sync_check_cuda_error(this->mStream->get());
    bool allEqual{true};

    this->PerformNormalAttention();
    sync_check_cuda_error(this->mStream->get());

    this->PerformMergedAttention();
    sync_check_cuda_error(this->mStream->get());

    // check result
    auto* output_ptr = bufferCast<DataType>(*(this->h_output_tensor));
    auto* output_ref_ptr = bufferCast<DataType>(*(this->h_output_tensor_accum_ref));
    for (int i = 0; i < this->h_output_tensor->getSize(); i++)
    {
        if (std::abs(static_cast<float>(output_ptr[i]) - static_cast<float>(output_ref_ptr[i]))
            > getTolerance<DataType>(output_ptr[i]))
        {
            std::cout << "Output mismatch at index " << i << ": "
                      << "expected " << static_cast<float>(output_ref_ptr[i]) << ", got "
                      << static_cast<float>(output_ptr[i]) << std::endl;
            allEqual = false;
            break;
        }
    }
    ASSERT_TRUE(allEqual);
}

TYPED_TEST(MlaChunkedPrefillTest, MlaChunkedPrefillCausalMask)
{
    using tensorrt_llm::runtime::bufferCast;
    using DataType = typename TestFixture::DataType;
    this->setDefaultParams();
    this->mIsCausalMask = true;
    this->allocateBuffers();

    sync_check_cuda_error(this->mStream->get());
    bool allEqual{true};

    this->PerformNormalAttention();
    sync_check_cuda_error(this->mStream->get());

    this->PerformMergedAttention();
    sync_check_cuda_error(this->mStream->get());

    // check result
    auto* output_ptr = bufferCast<DataType>(*(this->h_output_tensor));
    auto* output_ref_ptr = bufferCast<DataType>(*(this->h_output_tensor_accum_ref));
    for (int i = 0; i < this->h_output_tensor->getSize(); i++)
    {
        std::cout << "diff: " << std::abs(static_cast<float>(output_ptr[i]) - static_cast<float>(output_ref_ptr[i]))
                  << std::endl;
        if (std::abs(static_cast<float>(output_ptr[i]) - static_cast<float>(output_ref_ptr[i]))
            > getTolerance<DataType>(output_ptr[i]))
        {
            std::cout << "Output mismatch at index " << i << ": "
                      << "expected " << static_cast<float>(output_ref_ptr[i]) << ", got "
                      << static_cast<float>(output_ptr[i]) << std::endl;
            allEqual = false;
            break;
        }
    }
    ASSERT_TRUE(allEqual);
}
