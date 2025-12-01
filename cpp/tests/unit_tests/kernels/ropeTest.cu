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
#include <gtest/gtest.h>

#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include <random>
#include <tensorrt_llm/kernels/quantization.h>

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::kernels;

#ifdef ENABLE_FP4
template <typename fpType>
void computeReferenceFP4Scales(__nv_fp8_e4m3* blockScalesOut, fpType const* kvVec, SizeType32 size, float globalScale)
{
    constexpr auto quantBlockSize = 16;
    TLLM_CHECK(size % quantBlockSize == 0);

    for (SizeType32 i = 0; i < size / quantBlockSize; ++i)
    {
        auto amax = 0.f;
        for (SizeType32 j = 0; j < quantBlockSize; ++j)
        {
            amax = std::max(amax, std::abs(static_cast<float>(kvVec[i * 16 + j])));
        }
        auto sfValue = globalScale * amax / 6.f;
        blockScalesOut[i] = __nv_fp8_e4m3(sfValue);
    }
}

template <typename fpType>
uint8_t quantizeFP4(fpType val, __nv_fp8_e4m3 blockScale, float globalScale)
{
    float blockScaleFP32 = float(blockScale);

    float outputScale = 0.f;
    if (blockScaleFP32 != 0.f)
    {
        outputScale = 1.f / (blockScaleFP32 / globalScale);
    }

    float quantizedValFP32 = outputScale * static_cast<float>(val);
    auto result = __nv_fp4_e2m1(quantizedValFP32);
    return *reinterpret_cast<uint8_t*>(&result);
}

template <typename fpType>
void quantizeAndCacheFP4(
    __nv_fp4_e2m1* cachePtr, __nv_fp8_e4m3* blockScalesPtr, fpType const* kvVec, SizeType32 size, float globalScale)
{
    computeReferenceFP4Scales<fpType>(blockScalesPtr, kvVec, size, globalScale);

    for (SizeType32 i = 0; i < size / 2; ++i)
    {
        constexpr auto quantBlockSize = 16;
        auto blockScale = blockScalesPtr[(2 * i) / quantBlockSize];

        uint8_t val0 = quantizeFP4<fpType>(kvVec[2 * i], blockScale, globalScale);
        uint8_t val1 = quantizeFP4<fpType>(kvVec[2 * i + 1], blockScale, globalScale);

        // Even elements go in low bits, odd go in high
        uint8_t packedVals = (val1 << 4) | val0;
        cachePtr[i] = *reinterpret_cast<__nv_fp4_e2m1*>(&packedVals);
    }
}
#endif

template <typename fpType>
void calculateMeanAndStdDev(fpType const* data, SizeType32 size, float& mean, float& stdDev)
{
    if (size == 0)
    {
        mean = 0.0f;
        stdDev = 0.0f;
        return;
    }

    float sum = 0.0f;
    float sumSq = 0.0f;
    for (SizeType32 i = 0; i < size; ++i)
    {
        float val = static_cast<float>(data[i]);
        sum += val;
        sumSq += val * val;
    }
    mean = sum / size;

    // Variance = E[X^2] - (E[X])^2
    float variance = (sumSq / size) - (mean * mean);
    stdDev = std::sqrt(variance);
}

template <typename fpType>
void calculateMeanAndStdDevOfDifferences(
    fpType const* data1, fpType const* data2, SizeType32 size, float& mean, float& stdDev)
{
    if (size == 0)
    {
        mean = 0.0f;
        stdDev = 0.0f;
        return;
    }

    float sum = 0.0f;
    float sumSq = 0.0f;
    for (SizeType32 i = 0; i < size; ++i)
    {
        float diff = std::abs(static_cast<float>(data1[i]) - static_cast<float>(data2[i]));
        sum += diff;
        sumSq += diff * diff;
    }
    mean = sum / size;

    // Variance = E[X^2] - (E[X])^2
    float variance = (sumSq / size) - (mean * mean);
    stdDev = std::sqrt(variance);
}

inline bool almostEqual(float a, float b, float atol = 1e-2, float rtol = 1e-3)
{
    // Params: a = value to compare and b = reference
    // This function follows implementation of numpy.isclose(), which checks
    //   abs(a - b) <= (atol + rtol * abs(b)).
    // Note that the inequality above is asymmetric where b is considered as
    // a reference value. To account into both absolute/relative errors, it
    // uses absolute tolerance and relative tolerance at the same time. The
    // default values of atol and rtol borrowed from numpy.isclose(). For the
    // case of nan value, the result will be true.
    if (isnan(a) && isnan(b))
    {
        return true;
    }
    return fabs(a - b) <= (atol + rtol * fabs(b));
}

void createCosSinBuf(float* finalValues, SizeType32 numPos, SizeType32 dim, float theta = 10000.0f)
{
    // TODO(dblanaru) fix this function - it generates wrong values
    TLLM_THROW("The createCosSinBuf function contains an error at the moment. Use random initialization instead.");
    // Calculate the inverse frequencies
    std::vector<float> invFreq(dim / 2);
    for (SizeType32 ii = 0; ii < dim / 2; ++ii)
    {
        invFreq[ii] = 1.0f / std::pow(theta, (2.f * ii) / dim);
    }

    // Calculate the sinusoidal inputs and immediately calculate cos and sin values
    SizeType32 index = 0;
    for (SizeType32 pos = 0; pos < numPos; ++pos)
    {
        for (SizeType32 ii = 0; ii < dim / 2; ++ii)
        {
            auto const value = pos * invFreq[ii];
            // Append cos and sin values for each frequency, interleaving them
            finalValues[index++] = std::cos(value);
            finalValues[index++] = std::sin(value);
        }
    }
}

template <typename fpType>
fpType rotateHalfIndex(SizeType32 ii, SizeType32 size, fpType* vec)
{
    auto const halfSize = size / 2;
    auto const sign = (ii < size / 2) ? fpType{-1} : fpType{1};
    return vec[(ii + halfSize) % size] * sign;
}

template <typename fpType>
fpType rotateEveryTwo(SizeType32 ii, SizeType32 size, fpType* vec)
{
    auto sign = (ii % 2) ? fpType{1} : fpType{-1};
    auto offset = (ii % 2) ? SizeType32{-1} : SizeType32{1};
    return vec[ii + offset] * sign;
}

template <typename fpType>
void fillWithOnes(fpType* ptr, SizeType32 sz)
{
    for (SizeType32 ii = 0; ii < sz; ++ii)
    {
        ptr[ii] = static_cast<fpType>(1.0);
    }
}

template <typename fpType>
void fillWithOnesAndZerosInterleaved(fpType* ptr, SizeType32 sz)
{
    for (SizeType32 ii = 0; ii < sz; ii += 2)
    {
        ptr[ii] = static_cast<fpType>(1.0);
        ptr[ii + 1] = static_cast<fpType>(0.0);
    }
}

template <typename fpType>
void applyRopeToBuffer(fpType* srcBuffer, fpType* resBuffer, float2 const* cosSinBuffer, SizeType32 rotaryEmbeddingDim)
{

    for (SizeType32 ii = 0; ii <= rotaryEmbeddingDim; ++ii)
    {
        auto curr = static_cast<float>(srcBuffer[ii]);
        auto currReversed = static_cast<float>(rotateHalfIndex(ii, rotaryEmbeddingDim, srcBuffer));
        auto currFactors = cosSinBuffer[ii % (rotaryEmbeddingDim / 2)];
        auto tmp = currFactors.x * curr + currFactors.y * currReversed;

        resBuffer[ii] = static_cast<float>(tmp);
    }
}

// TODO(dblanaru) add (batch_size, seq_len, 3, head_num, size_per_head) as supported source
// TODO(dblanaru) support different number of kv heads (other than # q heads)
template <typename fpType, typename KVCacheType, typename KVCacheBuffer>
void computeReferenceBiasRope(QKVPreprocessingParams<fpType, KVCacheBuffer> params)
{
    // QKV shape (num_tokens, 3, head_num, size_per_head) in case of non-padded inputs
    // rotary_cos_sin has shape (rotary_embedding_max_positions, rotary_embedding_dim)
    // eg (2048, 128)

    params.setCommonParameters();
    auto const& batchSize = params.batch_size;
    auto const& numTokens = params.token_num;

    // number of query heads
    auto const& qHeadNum = params.head_num;
    // total dimensions of embedding for query heads = qHeadNum*sizePerHead
    auto const& qHiddenSize = params.q_hidden_size;

    // number of key/value heads
    auto const& kvHeadNum = params.kv_head_num;
    // total dimensions of embedding for key/value heads = kvHeadNum*sizePerHead
    auto const& kvHiddenSize = params.kv_hidden_size;

    // dim per head needs to be the same for kv and q for self-attn to work
    auto const& sizePerHead = params.size_per_head;
    // rotaryEmbeddingDim is different for GPT-J and GPT-neox flavors of RoPE
    // for GPT-neox, its just half of the sizePerHead
    // for GPT-J, its an adjustable parameter
    auto const& rotaryEmbeddingDim = params.rotary_embedding_dim;

    // the total size needed for a token (all q+k+v heads)
    auto const& tokenSize = params.hidden_size;

    auto const& kvCache = params.kv_cache_buffer;

    auto tmpResultTensor
        = BufferManager::pinned(ITensor::makeShape({SizeType32(rotaryEmbeddingDim)}), TRTDataType<fpType>::value);
    auto tmpResultPtr = bufferCast<fpType>(*tmpResultTensor);
    // keeps the current token we are looking at (both padded and non-padded version). Needed due to lack of padding
    SizeType32 tokenIt{};

    float kGlobalScale = 0.f;
    float vGlobalScale = 0.f;

    // the size of a (Q)/K/V matrix TODO(dblanaru) separate this into q and kv sizes

    for (SizeType32 batchIt = 0; batchIt < batchSize; ++batchIt)
    {
        auto const& currContextSize = params.seq_lens[batchIt]; // the context size of the current batch
        for (SizeType32 contextIt = 0; contextIt < currContextSize; ++contextIt)
        {
            // contextIt acts as iterator through the tokens that make up one request

            // currently looking at the beginning of (3, head_num, size_per_head) for a particular token
            // execute this on q (head_num, size_per_head)
            auto const currentCosSinPtr = params.rotary_coef_cache_buffer + contextIt * (rotaryEmbeddingDim / 2);
            for (SizeType32 headIt = 0; headIt < qHeadNum; ++headIt)
            {
                auto const currOffset = tokenIt * tokenSize + headIt * sizePerHead;
                auto const currPtr = params.qkv_input + currOffset;

                applyRopeToBuffer<fpType>(currPtr, tmpResultPtr, currentCosSinPtr, rotaryEmbeddingDim);
                memcpy(currPtr, tmpResultPtr, rotaryEmbeddingDim * sizeof(fpType));
            }

            // do the same for k
            for (SizeType32 headIt = 0; headIt < kvHeadNum; ++headIt)
            {
                auto const currOffset = tokenIt * tokenSize + headIt * sizePerHead + qHiddenSize;
                auto const currPtr = params.qkv_input + currOffset;

                applyRopeToBuffer<fpType>(currPtr, tmpResultPtr, currentCosSinPtr, rotaryEmbeddingDim);
                memcpy(currPtr, tmpResultPtr, rotaryEmbeddingDim * sizeof(fpType));

                auto token_kv_idx = kvCache.getKVTokenIdx(contextIt);
                auto kCachePtr = reinterpret_cast<KVCacheType*>(kvCache.getKBlockPtr(batchIt, token_kv_idx));
                auto offset = kvCache.getKVLocalIdx(contextIt, headIt, sizePerHead, 0);

                if constexpr (std::is_same_v<fpType, KVCacheType>)
                {
                    memcpy(kCachePtr + offset, currPtr, sizePerHead * sizeof(fpType));
                }
                else
                {
#ifdef ENABLE_FP4
                    static_assert(std::is_same_v<KVCacheType, __nv_fp4_e2m1>);

                    auto kBlockScalesCachePtr = reinterpret_cast<__nv_fp8_e4m3*>(
                        params.kv_cache_block_scales_buffer.getKBlockPtr(batchIt, token_kv_idx));
                    TLLM_CHECK(sizePerHead % 16 == 0);
                    auto const sizePerHeadBlockScales = sizePerHead / 16;
                    auto const blockScalesOffset = params.kv_cache_block_scales_buffer.getKVLocalIdx(
                        contextIt, headIt, sizePerHeadBlockScales, 0);

                    quantizeAndCacheFP4<fpType>(
                        // We divide the offset by 2 because there are 2 packed FP4
                        // values per byte.
                        reinterpret_cast<__nv_fp4_e2m1*>(kCachePtr + offset / 2),
                        kBlockScalesCachePtr + blockScalesOffset, currPtr, sizePerHead, kGlobalScale);
#else
                    TLLM_THROW("Enable FP4 to run this test.");
#endif
                }
                // dont use tmpResultPtr, but currptr
                // tmpResultPtr will only have {rotaryEmbeddingDim}, but we need {sizePerHead} to also
                // pass the unmodified part of the head to the kv cache
            }

            for (SizeType32 headIt = 0; headIt < kvHeadNum; ++headIt)
            {
                auto const currOffset = tokenIt * tokenSize + headIt * sizePerHead + qHiddenSize + kvHiddenSize;
                auto const currPtr = params.qkv_input + currOffset;

                auto token_kv_idx = kvCache.getKVTokenIdx(contextIt);
                auto vCachePtr = reinterpret_cast<KVCacheType*>(kvCache.getVBlockPtr(batchIt, token_kv_idx));
                auto offset = kvCache.getKVLocalIdx(contextIt, headIt, sizePerHead, 0);
                if constexpr (std::is_same_v<fpType, KVCacheType>)
                {
                    memcpy(vCachePtr + offset, currPtr, sizePerHead * sizeof(fpType));
                }
                else
                {
#ifdef ENABLE_FP4
                    auto vBlockScalesCachePtr = reinterpret_cast<__nv_fp8_e4m3*>(
                        params.kv_cache_block_scales_buffer.getVBlockPtr(batchIt, token_kv_idx));
                    TLLM_CHECK(sizePerHead % 16 == 0);
                    auto const sizePerHeadBlockScales = sizePerHead / 16;
                    auto const blockScalesOffset = params.kv_cache_block_scales_buffer.getKVLocalIdx(
                        contextIt, headIt, sizePerHeadBlockScales, 0);

                    quantizeAndCacheFP4<fpType>(reinterpret_cast<__nv_fp4_e2m1*>(vCachePtr + offset / 2),
                        vBlockScalesCachePtr + blockScalesOffset, currPtr, sizePerHead, vGlobalScale);
#else
                    TLLM_THROW("Enable FP4 to run this test.");
#endif
                }
            }
            ++tokenIt;
        }
    }
}

template <typename Pair>
class RopeTest : public testing::Test
{
protected:
    // internal variables
    using FPTypeAndKVCacheType = typename Pair::first_type;
    using KVCacheBuffer = typename Pair::second_type;

    using fpType = typename FPTypeAndKVCacheType::first_type;
    using KVCacheType = typename FPTypeAndKVCacheType::second_type;

    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    BufferManager::ITensorPtr cu_q_seqlens_tensor{nullptr}, cu_kv_seqlens_tensor{nullptr},
        padding_offset_tensor{nullptr}, encoder_padding_offset_tensor{nullptr}, fmha_tile_counter_ptr_tensor{nullptr},
        rotary_inv_freq_buf_tensor{nullptr}, tokens_info_tensor{nullptr}, global_scale_tensor{nullptr};

    std::mt19937 gen;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // initialize params coming from GPTAttentionPluginCommon
    int mNumHeads{0};
    int mNumKVHeads{0};
    int mHeadSize{0};
    int mRotaryEmbeddingDim{0};
    float mRotaryEmbeddingBase{0.0};
    RotaryScalingType mRotaryEmbeddingScaleType{RotaryScalingType::kNONE};
    float mRotaryEmbeddingScale{0.0};
    int mRotaryEmbeddingMaxPositions{0};
    PositionEmbeddingType mPositionEmbeddingType{PositionEmbeddingType::kROPE_GPT_NEOX};
    bool mRemovePadding{false};
    AttentionMaskType mMaskType{AttentionMaskType::CAUSAL};
    // NOTE: default values for paged kv cache.
    bool mPagedKVCache{false};
    int mTokensPerBlock{0};
    QuantMode mKVCacheQuantMode{};

    bool mCrossAttention{false};

    bool mPosShiftEnabled{false};
    bool mPagedContextFMHA{false};
    bool mFP8ContextFMHA{false};

    // fmha runner (disable by default)
    // flag: disabled = 0, enabled = 1, enabled with fp32 accumulation = 2
    bool mEnableContextFMHA{false};
    int mMultiProcessorCount{0};
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // initialize params coming from runtime - usually packed in EnqueueContextParams
    fpType* attention_input{nullptr};
    fpType const* qkv_bias{nullptr};
    // // Rotary cos sin cache buffer to avoid re-computing.
    BufferManager::ITensorPtr rotary_cos_sin_tensor{nullptr};
    float* rotary_fill_help{nullptr};
    float2 const* rotary_cos_sin{nullptr};

    // TODO(dblanaru) change these to SizeType32
    int32_t input_seq_length{0};
    int32_t max_past_kv_len{0};
    // // By default, max_attention_window == cyclic_attention_window_size
    // // unless each layer has different cyclic kv cache length.
    // // Max cache capacity (used to allocate KV cache)
    int32_t max_attention_window{0};
    // // Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
    int32_t cyclic_attention_window_size{0};
    int32_t sink_token_length{0};
    // these two are actually the same in LLama
    BufferManager::ITensorPtr q_seq_lengths_tensor{nullptr};
    int32_t* q_seq_lengths{nullptr};
    int32_t* kv_seq_lengths{nullptr};
    float* kv_scale_orig_quant{nullptr};

    KVBlockArray::DataType* block_offsets{nullptr};
    void* host_primary_pool_pointer{nullptr};
    void* host_secondary_pool_pointer{nullptr};
    int32_t batch_size{0};
    int32_t num_tokens{0}; // sum of q_seq_lengths
    int32_t max_blocks_per_sequence{0};

    int32_t cross_qkv_length{0};
    int32_t const* encoder_input_lengths{nullptr};

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Other used in `enqueueContext` from `GPTAttentionPluginCommon`
    SizeType32 qkv_size{0};
    QKVPreprocessingParams<fpType, KVCacheBuffer> preprocessingParams;
    float* rotary_inv_freq_buf{nullptr};
    int* tokens_info{nullptr};
    int* cu_q_seqlens{nullptr};
    BufferManager::ITensorPtr attention_input_buf{nullptr};
    KVCacheBuffer keyValueCache, keyValueCacheReference;
    KVCacheBuffer blockScalesCache, blockScalesCacheReference;
    BufferManager::IBufferPtr keyValueCacheBuffer{nullptr}, keyValueCacheBufferReference{nullptr};
    BufferManager::IBufferPtr blockScalesCacheBuffer{nullptr}, blockScalesCacheBufferReference{nullptr};

    void SetUp() override
    {
        if (shouldSkip())
        {
            GTEST_SKIP() << "Skipping due to no/unsupported GPU";
        }

        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);

#ifdef ENABLE_FP4
        if constexpr (std::is_same_v<KVCacheType, __nv_fp4_e2m1>)
        {
            mKVCacheQuantMode += QuantMode::fp4KvCache();
        }
#endif
        gen.seed(42U);
    }

    static bool shouldSkip()
    {
#ifdef ENABLE_FP4
        if constexpr (std::is_same_v<KVCacheType, __nv_fp4_e2m1>)
        {
            // Quant helper functions will not work on lower SM versions.
            return getSMVersion() < 100 || getSMVersion() >= 120;
        }
#endif
        return false;
    }

    void allocateBuffers()
    {
        auto const cu_seqlens_size = batch_size + 1;

        cu_q_seqlens_tensor = mBufferManager->pinned(ITensor::makeShape({cu_seqlens_size}), nvinfer1::DataType::kINT32);
        cu_kv_seqlens_tensor
            = mBufferManager->pinned(ITensor::makeShape({cu_seqlens_size}), nvinfer1::DataType::kINT32);
        padding_offset_tensor
            = mBufferManager->pinned(ITensor::makeShape({batch_size, input_seq_length}), nvinfer1::DataType::kINT32);
        encoder_padding_offset_tensor
            = mBufferManager->pinned(ITensor::makeShape({batch_size, cross_qkv_length}), nvinfer1::DataType::kINT32);
        fmha_tile_counter_ptr_tensor
            = mBufferManager->pinned(ITensor::makeShape({mEnableContextFMHA ? 1 : 0}), nvinfer1::DataType::kINT32);
        rotary_inv_freq_buf_tensor = mBufferManager->pinned(
            ITensor::makeShape({batch_size, mRotaryEmbeddingDim / 2}), nvinfer1::DataType::kFLOAT);

        int const max_num_tokens = batch_size * input_seq_length;
        tokens_info_tensor
            = mBufferManager->pinned(ITensor::makeShape({max_num_tokens, 2}), nvinfer1::DataType::kINT32);

#ifdef ENABLE_FP4
        if constexpr (std::is_same_v<KVCacheType, __nv_fp4_e2m1>)
        {
            global_scale_tensor = mBufferManager->pinned(ITensor::makeShape({2}), nvinfer1::DataType::kFLOAT);
        }
#endif
    }

    SizeType32 generateRandomSizeSmallerThan(SizeType32 a)
    {
        // Check if 'a' is less than or equal to 0 to avoid invalid ranges
        if (a <= 0)
        {
            TLLM_CHECK_WITH_INFO(a > 0, "Upped bound of random value must be greater than 0.");
            return 0; // Return an error code or handle as appropriate
        }

        // Define a distribution in the range [0, a-1]
        std::uniform_int_distribution<> distrib(0, a - 1);

        // Generate and return the random number
        return SizeType32{distrib(gen)};
    }

    template <typename fpType>
    void fillRandomNormal(fpType* ptr, SizeType32 sz, float mean = 0.0f, float stdDev = 1.0f)
    {
        std::normal_distribution<float> distr(mean, stdDev);

        for (SizeType32 ii = 0; ii < sz; ++ii)
        {
            ptr[ii] = static_cast<fpType>(distr(gen));
        }
    }

    void setMembersLLama7b()
    {
        mNumHeads = 32;
        mNumKVHeads = 32;
        mHeadSize = 128;
        mRotaryEmbeddingDim = 128;
        mRotaryEmbeddingBase = 10000.0f;
        mRotaryEmbeddingScaleType = RotaryScalingType::kNONE;
        mRotaryEmbeddingScale = 1.0f;
        mRotaryEmbeddingMaxPositions = 2048;
        mPositionEmbeddingType = PositionEmbeddingType::kROPE_GPT_NEOX;
        mRemovePadding = true;
        mMaskType = AttentionMaskType::CAUSAL;

        mPagedKVCache = false;
        mTokensPerBlock = 128;
        mCrossAttention = false;

        mPosShiftEnabled = false;
        mPagedContextFMHA = false;
        mFP8ContextFMHA = false;

        mEnableContextFMHA = true;
        mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    }

    void setEnqueueParamsLLama7()
    {
        // // Rotary cos sin cache buffer to avoid re-computing.
        SizeType32 maxOutputSize{generateRandomSizeSmallerThan(1024)};
        rotary_cos_sin_tensor = this->mBufferManager->pinned(
            ITensor::makeShape({mRotaryEmbeddingMaxPositions, mRotaryEmbeddingDim}), nvinfer1::DataType::kFLOAT);
        rotary_fill_help = bufferCast<float>(*(rotary_cos_sin_tensor));
        // createCosSinBuf(rotary_fill_help, mRotaryEmbeddingMaxPositions, mRotaryEmbeddingDim); //currently broken
        // fillWithOnesAndZerosInterleaved(rotary_fill_help, mRotaryEmbeddingMaxPositions*
        // mRotaryEmbeddingDim); //maxes the cos to 1 so it's an identity op
        fillRandomNormal(rotary_fill_help, mRotaryEmbeddingMaxPositions * mRotaryEmbeddingDim);
        rotary_cos_sin = (float2*) (rotary_fill_help);

        batch_size = generateRandomSizeSmallerThan(12);

        q_seq_lengths_tensor = mBufferManager->pinned(ITensor::makeShape({batch_size}), nvinfer1::DataType::kINT32);
        q_seq_lengths = bufferCast<int32_t>(*(q_seq_lengths_tensor));

        for (SizeType32 ii = 0; ii < batch_size; ++ii)
        {
            q_seq_lengths[ii] = generateRandomSizeSmallerThan(1024);
            input_seq_length = std::max(input_seq_length, q_seq_lengths[ii]);
            num_tokens += q_seq_lengths[ii];
        }

        max_past_kv_len = input_seq_length;
        // // By default, max_attention_window == cyclic_attention_window_size
        // // unless each layer has different cyclic kv cache length.
        // // Max cache capacity (used to allocate KV cache)
        max_attention_window = input_seq_length + maxOutputSize;
        // // Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
        cyclic_attention_window_size = input_seq_length + maxOutputSize;
        kv_seq_lengths = q_seq_lengths;

        max_blocks_per_sequence = 0;
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }

    void buildDecoderParams()
    {
        fpType* attention_mask{};
        cu_q_seqlens = bufferCast<int32_t>(*(this->cu_q_seqlens_tensor));
        int* cu_kv_seqlens = bufferCast<int32_t>(*(this->cu_kv_seqlens_tensor));
        int* padding_offset = bufferCast<int32_t>(*(this->padding_offset_tensor));
        int* encoder_padding_offset = bufferCast<int32_t>(*(this->encoder_padding_offset_tensor));
        uint32_t* fmha_tile_counter_ptr = bufferCast<uint32_t>(*(this->fmha_tile_counter_ptr_tensor));
        rotary_inv_freq_buf = bufferCast<float>(*(this->rotary_inv_freq_buf_tensor));
        tokens_info = bufferCast<int>(*(this->tokens_info_tensor));

        BuildDecoderInfoParams<fpType> decoderParams;
        memset(&decoderParams, 0, sizeof(decoderParams));
        decoderParams.seqQOffsets = cu_q_seqlens;
        decoderParams.seqKVOffsets = cu_kv_seqlens;
        decoderParams.paddingOffsets = padding_offset;
        decoderParams.tokensInfo = reinterpret_cast<int2*>(tokens_info);
        decoderParams.encoderPaddingOffsets = mCrossAttention ? encoder_padding_offset : nullptr;
        decoderParams.attentionMask = mCrossAttention ? nullptr : attention_mask; // manually set for cross attn
        // Fixed sequence length offset if not removing the padding (cu_q_seqlens[ii] = ii * seq_length).
        decoderParams.seqQLengths = q_seq_lengths;
        decoderParams.seqKVLengths = mCrossAttention ? encoder_input_lengths : kv_seq_lengths;
        decoderParams.batchSize = batch_size;
        decoderParams.maxQSeqLength = input_seq_length;
        decoderParams.maxEncoderQSeqLength = mCrossAttention ? cross_qkv_length : 0;
        decoderParams.attentionWindowSize = cyclic_attention_window_size;
        decoderParams.sinkTokenLength = sink_token_length;
        decoderParams.numTokens = num_tokens;
        decoderParams.removePadding = mRemovePadding;
        decoderParams.attentionMaskType = mMaskType;
        decoderParams.fmhaTileCounter = fmha_tile_counter_ptr;
        // Rotary embedding inv_freq buffer.
        decoderParams.rotaryEmbeddingScale = mRotaryEmbeddingScale;
        decoderParams.rotaryEmbeddingBase = mRotaryEmbeddingBase;
        decoderParams.rotaryEmbeddingDim = mRotaryEmbeddingDim;
        decoderParams.rotaryScalingType = mRotaryEmbeddingScaleType;
        decoderParams.rotaryEmbeddingInvFreq = rotary_inv_freq_buf;
        decoderParams.rotaryEmbeddingMaxPositions = mRotaryEmbeddingMaxPositions;

        TLLM_LOG_DEBUG(decoderParams.toString());
        invokeBuildDecoderInfo(decoderParams, this->mStream->get());
        sync_check_cuda_error(this->mStream->get());
    }

    void buildPreprocessingParams()
    {
        bool const enablePagedKVContextFMHA = mPagedKVCache && mPagedContextFMHA;

        KvCacheDataType cache_type = KvCacheDataType::BASE;
        if (mKVCacheQuantMode.hasInt8KvCache())
        {
            cache_type = KvCacheDataType::INT8;
        }
        else if (mKVCacheQuantMode.hasFp8KvCache())
        {
            cache_type = KvCacheDataType::FP8;
        }
        else if (mKVCacheQuantMode.hasFp4KvCache())
        {
            cache_type = KvCacheDataType::NVFP4;
        }

        if (global_scale_tensor)
        {
            kv_scale_orig_quant = bufferCast<float>(*(global_scale_tensor));
            kv_scale_orig_quant[0] = 5.1f;
            kv_scale_orig_quant[1] = 0.25f;
        }

        qkv_size = num_tokens * 3 * mNumHeads * mHeadSize;

        attention_input_buf = this->mBufferManager->pinned(
            ITensor::makeShape({num_tokens, 3, mNumHeads, mHeadSize}), TRTDataType<fpType>::value);
        // ITensor::volume(attention_input_buf->getShape())
        attention_input = bufferCast<fpType>(*attention_input_buf);
        this->fillRandomNormal(attention_input, qkv_size);

        preprocessingParams.qkv_input = const_cast<fpType*>(attention_input);
        preprocessingParams.cross_kv_input = nullptr;
        preprocessingParams.quantized_qkv_output = nullptr; // Assuming this is the correct member for 'O'
        preprocessingParams.q_output = nullptr;
        preprocessingParams.kv_cache_buffer = keyValueCache;
        preprocessingParams.kv_cache_block_scales_buffer = blockScalesCache;
        preprocessingParams.qkv_bias = qkv_bias;
        preprocessingParams.tokens_info = reinterpret_cast<int2*>(tokens_info);
        preprocessingParams.seq_lens = q_seq_lengths;
        preprocessingParams.cache_seq_lens = kv_seq_lengths;
        preprocessingParams.encoder_seq_lens = nullptr;
        preprocessingParams.cu_seq_lens = cu_q_seqlens;
        preprocessingParams.cu_kv_seq_lens = nullptr; // Only used by cross attention.
        preprocessingParams.rotary_embedding_inv_freq = rotary_inv_freq_buf;
        preprocessingParams.rotary_coef_cache_buffer = rotary_cos_sin;
        preprocessingParams.qkv_scale_orig_quant = kv_scale_orig_quant;
        preprocessingParams.spec_decoding_position_offsets = nullptr; // Cast to int* if necessary
        preprocessingParams.batch_size = batch_size;
        preprocessingParams.max_input_seq_len = input_seq_length;
        preprocessingParams.max_kv_seq_len = max_past_kv_len;
        preprocessingParams.cyclic_kv_cache_len = cyclic_attention_window_size;
        preprocessingParams.sink_token_len = sink_token_length;
        preprocessingParams.token_num = num_tokens;
        preprocessingParams.remove_padding = mRemovePadding;
        preprocessingParams.head_num = mNumHeads;
        preprocessingParams.kv_head_num = mNumKVHeads;
        preprocessingParams.qheads_per_kv_head = mNumHeads / mNumKVHeads;
        preprocessingParams.size_per_head = mHeadSize;
        preprocessingParams.rotary_embedding_dim = mRotaryEmbeddingDim;
        preprocessingParams.rotary_embedding_base = mRotaryEmbeddingBase;
        preprocessingParams.rotary_scale_type = mRotaryEmbeddingScaleType;
        preprocessingParams.rotary_embedding_scale = mRotaryEmbeddingScale;
        preprocessingParams.rotary_embedding_max_positions = mRotaryEmbeddingMaxPositions;
        preprocessingParams.position_embedding_type = mPositionEmbeddingType;
        preprocessingParams.position_shift_enabled = mPosShiftEnabled;
        preprocessingParams.cache_type = cache_type;
        preprocessingParams.separate_q_kv_output = enablePagedKVContextFMHA || mCrossAttention;
        preprocessingParams.quantized_fp8_output = mFP8ContextFMHA;
        preprocessingParams.generation_phase = false;
        preprocessingParams.multi_processor_count = mMultiProcessorCount;
        TLLM_CHECK_WITH_INFO(sink_token_length == 0, "sink_token_length != 0 is not supported in the RoPE test.");
    }

    void buildKVCaches()
    {
        size_t elemSizeBits;
        if (mKVCacheQuantMode.hasInt8KvCache() || mKVCacheQuantMode.hasFp8KvCache())
        {
            elemSizeBits = 8;
        }
        else if (mKVCacheQuantMode.hasFp4KvCache())
        {
            elemSizeBits = 4;
        }
        else
        {
            elemSizeBits = sizeof(fpType) * 8;
        }

        auto const sizePerToken = (this->mNumKVHeads * this->mHeadSize * elemSizeBits) / 8;
        auto const totalSize
            = (this->batch_size * 2 * (this->mCrossAttention ? this->cross_qkv_length : this->max_attention_window)
                  * this->mNumKVHeads * this->mHeadSize * elemSizeBits)
            / 8;
        keyValueCacheBuffer = BufferManager::pinned(totalSize);
        void* key_value_cache = static_cast<void*>(keyValueCacheBuffer->data());

        keyValueCacheBufferReference = BufferManager::pinned(totalSize);
        void* key_value_cache_reference = static_cast<void*>(keyValueCacheBufferReference->data());

        void* block_scales_cache = nullptr;
        void* block_scales_cache_reference = nullptr;

        if (mKVCacheQuantMode.hasFp4KvCache())
        {
            TLLM_CHECK_WITH_INFO(
                this->mHeadSize % 16 == 0, "head dim must be divisible by quantization block size (16) for FP4");
            // Dtype of block scales is FP8
            auto const blockScalesTotalSize = this->batch_size * 2
                * (this->mCrossAttention ? this->cross_qkv_length : this->max_attention_window) * this->mNumKVHeads
                * this->mHeadSize / 16;

            blockScalesCacheBuffer = BufferManager::pinned(blockScalesTotalSize);
            block_scales_cache = static_cast<void*>(blockScalesCacheBuffer->data());

            blockScalesCacheBufferReference = BufferManager::pinned(blockScalesTotalSize);
            block_scales_cache_reference = static_cast<void*>(blockScalesCacheBufferReference->data());
        }

        // KVBlockArray::DataType* hostKvCacheBlockOffsets;
        if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
        {
            TLLM_THROW("Paged KV Cache currently not supported in ropeTest");
        }
        else if constexpr (std::is_same_v<KVCacheBuffer, KVLinearBuffer>)
        {
            // Pointer to the of K/V cache data
            // Shape [B, 2, S*H*D], where 2 is for K and V,
            // B is current number of sequences and
            // H is number of heads
            // S is maximum sequence length
            // D is dimension per head
            // K shape is [B, 1, H, S, D]
            // V shape is [B, 1, H, S, D]
            // NOTE: we have remapped K layout as the same of V.
            using BufferDataType = typename KVCacheBuffer::DataType;
            keyValueCache = KVLinearBuffer(this->batch_size,
                this->mCrossAttention ? this->cross_qkv_length : this->max_attention_window, sizePerToken,
                this->cyclic_attention_window_size, this->sink_token_length, false,
                reinterpret_cast<BufferDataType*>(key_value_cache));

            using BufferDataType = typename KVCacheBuffer::DataType;
            keyValueCacheReference = KVLinearBuffer(this->batch_size,
                this->mCrossAttention ? this->cross_qkv_length : this->max_attention_window, sizePerToken,
                this->cyclic_attention_window_size, this->sink_token_length, false,
                reinterpret_cast<BufferDataType*>(key_value_cache_reference));

            // Block scale buffers are for FP4 only. The shape matches the KV cache pools,
            // except that the block scale is divided by the quantization block size (16
            // for NVFP4).

            if (mKVCacheQuantMode.hasFp4KvCache())
            {
                auto sizePerTokenBlockScales = this->mNumKVHeads * this->mHeadSize / 16;
                blockScalesCache = KVLinearBuffer(this->batch_size,
                    this->mCrossAttention ? this->cross_qkv_length : this->max_attention_window,
                    sizePerTokenBlockScales, this->cyclic_attention_window_size, this->sink_token_length, false,
                    reinterpret_cast<BufferDataType*>(block_scales_cache));

                blockScalesCacheReference = KVLinearBuffer(this->batch_size,
                    this->mCrossAttention ? this->cross_qkv_length : this->max_attention_window,
                    sizePerTokenBlockScales, this->cyclic_attention_window_size, this->sink_token_length, false,
                    reinterpret_cast<BufferDataType*>(block_scales_cache_reference));
            }
        }
    }
};

using RopeTypes = ::testing::Types<
#ifdef ENABLE_FP4
    std::pair<std::pair<half, __nv_fp4_e2m1>, KVLinearBuffer>,
    std::pair<std::pair<nv_bfloat16, __nv_fp4_e2m1>, KVLinearBuffer>,
#endif
    std::pair<std::pair<half, half>, KVLinearBuffer>, std::pair<std::pair<nv_bfloat16, nv_bfloat16>, KVLinearBuffer>,
    std::pair<std::pair<float, float>, KVLinearBuffer>>;

TYPED_TEST_SUITE(RopeTest, RopeTypes);

TYPED_TEST(RopeTest, RopeTestLLamaLinearCache)
{
    using fpType = typename TestFixture::fpType;
    using KVCacheType = typename TestFixture::KVCacheType;
    using KVCacheBuffer = typename TestFixture::KVCacheBuffer;

    this->setMembersLLama7b();
    this->setEnqueueParamsLLama7();

    this->allocateBuffers();

    sync_check_cuda_error(this->mStream->get());

    this->buildDecoderParams();
    this->buildKVCaches();
    this->buildPreprocessingParams();

    TLLM_LOG_DEBUG(this->preprocessingParams.toString());

    bool allEqual{true};
    BufferManager::ITensorPtr reference_qkv_buf
        = this->mBufferManager->copyFrom(*(this->attention_input_buf), tensorrt_llm::runtime::MemoryType::kPINNEDPOOL);
    fpType* reference_qkv = bufferCast<fpType>(*reference_qkv_buf);

    for (SizeType32 iAssert = 0; iAssert < this->qkv_size; iAssert++)
    {
        if (!almostEqual(static_cast<float>(this->attention_input[iAssert]), static_cast<float>(reference_qkv[iAssert]),
                1e-3, 1e-3))
        {
            TLLM_LOG_ERROR("Mismatch input value. Position of inputs: %d, expected value: %f, output value: %f",
                iAssert, static_cast<float>(this->attention_input[iAssert]),
                static_cast<float>(reference_qkv[iAssert]));
            allEqual = false;
        }
    }
    EXPECT_TRUE(allEqual);

    TLLM_LOG_DEBUG("Parameters generated, random inputs copied. Calling kernel");

    invokeQKVPreprocessing(this->preprocessingParams, this->mStream->get());
    cudaDeviceSynchronize();

    this->preprocessingParams.qkv_input = const_cast<fpType*>(reference_qkv);
    this->preprocessingParams.kv_cache_buffer = this->keyValueCacheReference;
    this->preprocessingParams.kv_cache_block_scales_buffer = this->blockScalesCacheReference;
    TLLM_LOG_DEBUG("Kernel finished, calling reference");

    computeReferenceBiasRope<fpType, KVCacheType, KVCacheBuffer>(this->preprocessingParams);
    TLLM_LOG_DEBUG("Reference finished, comparing results");

    cudaDeviceSynchronize();
    float mean, stdDev;

    calculateMeanAndStdDev(this->attention_input, this->qkv_size, mean, stdDev);

    TLLM_LOG_DEBUG("Output Mean: %e, Standard Deviation: %e", mean, stdDev);

    calculateMeanAndStdDevOfDifferences(this->attention_input, reference_qkv, this->qkv_size, mean, stdDev);

    TLLM_LOG_DEBUG("Output Abs difference Mean: %e, Standard Deviation: %e", mean, stdDev);
    bool resultsEqual{true};
    for (SizeType32 iAssert = 0; iAssert < this->qkv_size; iAssert++)
    {
        if (!almostEqual(static_cast<float>(this->attention_input[iAssert]), static_cast<float>(reference_qkv[iAssert]),
                1e-5, 0.01f))
        {
            TLLM_LOG_ERROR("Mismatch output value. Position of outputs: %d, expected value: %e, output value: %e",
                iAssert, static_cast<float>(this->attention_input[iAssert]),
                static_cast<float>(reference_qkv[iAssert]));
            resultsEqual = false;
            break;
        }
    }

    EXPECT_TRUE(resultsEqual);

    auto floatKernelKV = reinterpret_cast<KVCacheType*>(this->keyValueCache.data);
    auto floatReferenceKV = reinterpret_cast<KVCacheType*>(this->keyValueCacheReference.data);
    auto const totalSize = this->batch_size * 2
        * (this->mCrossAttention ? this->cross_qkv_length : this->max_attention_window) * this->mNumKVHeads
        * this->mHeadSize;

    auto kernelBlockScales = reinterpret_cast<__nv_fp8_e4m3*>(this->blockScalesCache.data);
    auto referenceBlockScales = reinterpret_cast<__nv_fp8_e4m3*>(this->blockScalesCacheReference.data);

#ifdef ENABLE_FP4
    constexpr auto isFP4 = std::is_same_v<KVCacheType, __nv_fp4_e2m1>;
#else
    constexpr auto isFP4 = false;
#endif

    auto getKVCacheVal = [&](SizeType32 iAssert, auto* data)
    {
        if constexpr (!isFP4)
        {
            return static_cast<float>(data[iAssert]);
        }
        else
        {
#ifdef ENABLE_FP4
            auto fp4Idx = iAssert / 2;
            auto dataPacked = *reinterpret_cast<uint8_t*>(data + fp4Idx);

            uint8_t storage;
            if (iAssert % 2 == 0)
            {
                storage = dataPacked & 0x0f;
            }
            else
            {
                storage = dataPacked >> 4;
            }
            __nv_fp4_e2m1 val = *reinterpret_cast<__nv_fp4_e2m1*>(&storage);
            return static_cast<float>(val);
#else
            return 0.f;
#endif
        }
    };

    // We allow a very small amount of elements to mismatch for FP4 cases.
    // The computation isn't bit exact compared to the reference, likely due
    // to the fast reciprocal approximation. In our BF16 case, only 2 out of
    // over 100,000,000 elements are mismatched. We leave the tolerance a bit
    // higher than 2 for different random seeds.
    //
    // Note that this approach leads to a more accurate test compared to just
    // increasing the atol to ~0.5 for FP4. With this approach, we make sure that
    // the vast majority of elements match the reference _exactly_ when the dtype
    // is FP4.
    int maxAllowedMismatch = isFP4 ? 10 : 0;
    int numMismatched = 0;
    for (SizeType32 iAssert = 0; iAssert < totalSize; iAssert++)
    {
        auto const ref = getKVCacheVal(iAssert, floatReferenceKV);
        auto const actual = getKVCacheVal(iAssert, floatKernelKV);
        if (!almostEqual(actual, ref, 1e-5, 0.01f))
        {
            TLLM_LOG_ERROR("Mismatch kv cache value. Position in kv cache: %d, expected value: %e, output value: %e",
                iAssert, actual, ref);
            if (++numMismatched > maxAllowedMismatch)
            {
                break;
            }
        }
    }
    EXPECT_TRUE(numMismatched <= maxAllowedMismatch);

    if (isFP4)
    {
        bool kvCacheBlockScalesEqual = true;
        constexpr auto quantBlockSize = 16;
        auto totalSizeBlockScales = totalSize / quantBlockSize;

        for (SizeType32 iAssert = 0; iAssert < totalSizeBlockScales; ++iAssert)
        {
            auto const ref = static_cast<float>(kernelBlockScales[iAssert]);
            auto const actual = static_cast<float>(referenceBlockScales[iAssert]);
            if (!almostEqual(actual, ref, 1e-5, 0.01f))
            {
                TLLM_LOG_ERROR(
                    "Mismatch kv cache block scale value. Position in kv cache: %d, expected value: %e, output value: "
                    "%e",
                    iAssert, actual, ref);
                kvCacheBlockScalesEqual = false;
                break;
            }
        }
        EXPECT_TRUE(kvCacheBlockScalesEqual);
    }
}
