#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include <numeric>
#include <random>
#include <type_traits>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::kernels;

namespace tc = tensorrt_llm::common;
namespace trk = tensorrt_llm::runtime::kernels;

namespace
{

template <typename T>
void initRandom(T* ptr, size_t size, float minval, float maxval)
{
    for (size_t i = 0; i < size; ++i)
    {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        val *= (maxval - minval);
        ptr[i] = static_cast<T>(minval + val);
    }
}

template <typename T>
struct SATypeConverter
{
    using Type = T;
};

template <>
struct SATypeConverter<half>
{
    using Type = uint16_t;
};

template <typename T, typename KVCacheBuffer>
__global__ void applyRoPE(KVCacheBuffer kCacheRead, KVLinearBuffer kCacheWrite, int const sizePerHead,
    int const beam_width, int const* token_read_idxs, int const* token_write_idxs, int const* token_pos_idxs,
    int const* token_seq_idxs, int const* sequence_lengths, int const* input_lengths, int const rotary_embedding_dim,
    float rotary_embedding_base, RotaryScalingType const rotary_scale_type, float rotary_embedding_scale,
    int const rotary_embedding_max_positions, PositionEmbeddingType const position_embedding_type)
{
    // We allow only fp32/fp16/bf16 as the data types to apply rotary
    static_assert(sizeof(T) == 4 || sizeof(T) == 2, "");

    extern __shared__ __align__(sizeof(float2)) char smem_[]; // align on largest vector type
    // Each thread will handle 16 bytes.
    constexpr int vec_size = 16u / sizeof(T);
    using Vec_k = typename mmha::packed_type<T, vec_size>::type;
    int const sizePerHeadDivX = sizePerHead / vec_size;

    // The position idx
    int const token_idx = token_seq_idxs[blockIdx.x];
    int const token_read_idx = token_read_idxs[blockIdx.x];
    int const token_write_idx = token_write_idxs[blockIdx.x];
    int const token_pos_idx = token_pos_idxs[blockIdx.x];
    // Head
    int const head_idx = blockIdx.y;
    // The batch beam idx
    int const batch_beam_idx = blockIdx.z;
    // The beam idx
    int const beam_idx = batch_beam_idx % beam_width;
    // Thread idx
    int const tidx = threadIdx.x;

    // The actual sequence length excluding the paddings.
    int const tlength = sequence_lengths[batch_beam_idx] - 1;
    // The context length
    int const inlength = input_lengths[batch_beam_idx];
    // Mask out the tokens exceed the real total length and tokens in the context phase with beam_idx>0
    bool const valid_seq = token_idx < tlength && !(token_idx < inlength && beam_idx > 0);
    bool const is_head_size_masked = tidx * vec_size >= sizePerHead;

    if (!valid_seq || is_head_size_masked)
    {
        return;
    }

    // Read k
    Vec_k k;
    T* k_cache = reinterpret_cast<T*>(kCacheRead.getKBlockPtr(batch_beam_idx, token_read_idx));
    int inBlockIdx_r = kCacheRead.getKVLocalIdx(token_read_idx, head_idx, sizePerHead, tidx * vec_size);
    k = *reinterpret_cast<Vec_k const*>(&k_cache[inBlockIdx_r]);

    // Apply position embedding
    switch (position_embedding_type)
    {
    case PositionEmbeddingType::kROPE_GPTJ:
    {
        mmha::apply_rotary_embedding(
            k, tidx, rotary_embedding_dim, rotary_embedding_base, rotary_embedding_scale, token_pos_idx);
        break;
    }
    case PositionEmbeddingType::kROPE_M: [[fallthrough]];
    case PositionEmbeddingType::kROPE_GPT_NEOX:
    {
        bool const do_rotary = vec_size * tidx < rotary_embedding_dim;

        T* k_smem = reinterpret_cast<T*>(smem_);

        int const half_rotary_dim = rotary_embedding_dim / 2;
        int const half_idx = (tidx * vec_size) / half_rotary_dim;
        int const intra_half_idx = (tidx * vec_size) % half_rotary_dim;
        int const smem_pitch = half_rotary_dim; // TODO: adjust for bank conflicts?

        if (do_rotary)
        {
            *reinterpret_cast<Vec_k*>(k_smem + half_idx * smem_pitch + intra_half_idx) = k;
        }

        __syncthreads();

        int const transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
        constexpr int tidx_factor = vec_size / 2;
        if (do_rotary)
        {
            mmha::vec_from_smem_transpose(k, k_smem, transpose_idx, smem_pitch);
            mmha::apply_rotary_embedding(k, transpose_idx / tidx_factor, rotary_embedding_dim, rotary_embedding_base,
                rotary_embedding_scale, token_pos_idx);
            mmha::write_smem_transpose(k, k_smem, transpose_idx, smem_pitch);
        }

        __syncthreads();

        if (do_rotary)
        {
            k = *reinterpret_cast<Vec_k*>(k_smem + half_idx * smem_pitch + intra_half_idx);
        }
        break;
    }
    }

    // write back to cache
    T* kDst = reinterpret_cast<T*>(kCacheWrite.getKBlockPtr(batch_beam_idx, token_write_idx));
    int inBlockIdx_w = kCacheWrite.getKVLocalIdx(token_write_idx, head_idx, sizePerHeadDivX, tidx);
    reinterpret_cast<Vec_k*>(kDst)[inBlockIdx_w] = k;
}

template <typename T, typename KVCacheBuffer>
void invokeApplyRoPE(KVCacheBuffer const& kCacheRead, KVLinearBuffer const& kCacheWrite, int const sizePerHead,
    int const batch_beam, int const kv_head_num, int const beam_width, int const* token_read_idxs,
    int const* token_write_idxs, int const* token_pos_idxs, int const* token_seq_idxs, int const token_num,
    int const* sequence_lengths, int const* input_lengths, int const rotary_embedding_dim, float rotary_embedding_base,
    RotaryScalingType const rotary_scale_type, float rotary_embedding_scale, int const rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, cudaStream_t stream)
{
    // Block handles K tile.
    int const vec_size = 16u / sizeof(T);
    dim3 block((sizePerHead / vec_size + 31) / 32 * 32);
    dim3 grid(token_num, kv_head_num, batch_beam);
    size_t smem_size = ((position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX
                            || position_embedding_type == PositionEmbeddingType::kROPE_M)
            ? 2 * rotary_embedding_dim * sizeof(T)
            : 0);

    applyRoPE<T, KVCacheBuffer><<<grid, block, smem_size, stream>>>(kCacheRead, kCacheWrite, sizePerHead, beam_width,
        token_read_idxs, token_write_idxs, token_pos_idxs, token_seq_idxs, sequence_lengths, input_lengths,
        rotary_embedding_dim, rotary_embedding_base, rotary_scale_type, rotary_embedding_scale,
        rotary_embedding_max_positions, position_embedding_type);
}

template <typename T>
class ShiftKCacheKernelTest : public ::testing::Test
{
public:
    using TensorPtr = ITensor::SharedPtr;

    void SetUp() override
    {
        mStream = std::make_shared<CudaStream>();
        mBufferManager = std::make_shared<BufferManager>(mStream);
        auto const deviceCount = tc::getDeviceCount();
        if (deviceCount == 0)
        {
            GTEST_SKIP();
        }
    }

    void TearDown() override {}

    void initData(int32_t batchSize, int32_t beamWidth, int32_t numHeads, int32_t maxAttentionWindow, int32_t headSize,
        bool pagedKvCache, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, std::vector<int32_t> const& seqLengths,
        std::vector<int32_t> const& inputLengths, std::vector<int32_t> const& tokenReadIdxs,
        std::vector<int32_t> const& tokenWriteIdxs, std::vector<int32_t> const& tokenPosIdxs,
        std::vector<int32_t> const& tokenSeqIdxs)
    {
        // allocate buffer
        mSeqLengthsHost = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
        mSeqLengthsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);

        mInputLengthsHost = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
        mInputLengthsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);

        mKScaleQuantOrigDevice = mBufferManager->gpu(ITensor::makeShape({1}), nvinfer1::DataType::kFLOAT);

        mTokenReadIdxsHost = mBufferManager->pinned(
            ITensor::makeShape({static_cast<int>(tokenReadIdxs.size())}), nvinfer1::DataType::kINT32);
        mTokenReadIdxsDevice = mBufferManager->gpu(
            ITensor::makeShape({static_cast<int>(tokenReadIdxs.size())}), nvinfer1::DataType::kINT32);

        mTokenWriteIdxsHost = mBufferManager->pinned(
            ITensor::makeShape({static_cast<int>(tokenWriteIdxs.size())}), nvinfer1::DataType::kINT32);
        mTokenWriteIdxsDevice = mBufferManager->gpu(
            ITensor::makeShape({static_cast<int>(tokenWriteIdxs.size())}), nvinfer1::DataType::kINT32);

        mTokenPosIdxsHost = mBufferManager->pinned(
            ITensor::makeShape({static_cast<int>(tokenPosIdxs.size())}), nvinfer1::DataType::kINT32);
        mTokenPosIdxsDevice = mBufferManager->gpu(
            ITensor::makeShape({static_cast<int>(tokenPosIdxs.size())}), nvinfer1::DataType::kINT32);

        mTokenSeqIdxsHost = mBufferManager->pinned(
            ITensor::makeShape({static_cast<int>(tokenSeqIdxs.size())}), nvinfer1::DataType::kINT32);
        mTokenSeqIdxsDevice = mBufferManager->gpu(
            ITensor::makeShape({static_cast<int>(tokenSeqIdxs.size())}), nvinfer1::DataType::kINT32);

        // nvinfer1::DataType dataType = nvinfer1::DataType::kHALF
        // nvinfer1::DataType::kHALF
        // nvinfer1::DataType::kBF16
        int32_t batchBeam = batchSize * beamWidth;
        if (pagedKvCache)
        {
            mInputDataHost = mBufferManager->pinned(
                ITensor::makeShape({batchSize, beamWidth, 2, maxBlocksPerSeq, numHeads * tokensPerBlock * headSize}),
                TRTDataType<T>::value);
            mInputDataDevice = mBufferManager->gpu(
                ITensor::makeShape({batchSize, beamWidth, 2, maxBlocksPerSeq, numHeads * tokensPerBlock * headSize}),
                TRTDataType<T>::value);
            mInputBlockOffsetsHost
                = mBufferManager->pinned(ITensor::makeShape({batchSize, beamWidth, 2, maxBlocksPerSeq}),
                    TRTDataType<std::remove_const_t<KVBlockArray::DataType>>::value);
            mInputBlockOffsetsDevice
                = mBufferManager->gpu(ITensor::makeShape({batchSize, beamWidth, 2, maxBlocksPerSeq}),
                    TRTDataType<std::remove_const_t<KVBlockArray::DataType>>::value);
        }
        else
        {
            mInputDataHost = mBufferManager->pinned(
                ITensor::makeShape({batchBeam, 2, numHeads, maxAttentionWindow, headSize}), TRTDataType<T>::value);
            mInputDataDevice = mBufferManager->gpu(
                ITensor::makeShape({batchBeam, 2, numHeads, maxAttentionWindow, headSize}), TRTDataType<T>::value);
        }

        mOutputDataHost = mBufferManager->pinned(
            ITensor::makeShape({batchBeam, 1, numHeads, maxAttentionWindow, headSize}), TRTDataType<T>::value);
        mOutputDataDevice = mBufferManager->gpu(
            ITensor::makeShape({batchBeam, 1, numHeads, maxAttentionWindow, headSize}), TRTDataType<T>::value);

        mRefOutputDataHost = mBufferManager->pinned(
            ITensor::makeShape({batchBeam, 1, numHeads, maxAttentionWindow, headSize}), TRTDataType<T>::value);
        mRefOutputDataDevice = mBufferManager->gpu(
            ITensor::makeShape({batchBeam, 1, numHeads, maxAttentionWindow, headSize}), TRTDataType<T>::value);

        // init data
        auto inputDataHostPtr = bufferCast<T>(*mInputDataHost);
        initRandom(inputDataHostPtr, batchSize * beamWidth * 2 * numHeads * maxAttentionWindow * headSize, -3.0f, 3.0f);
        trk::invokeFill(*mKScaleQuantOrigDevice, float{1.0f}, *mStream);

        auto seqLengthsHostPtr = bufferCast<int32_t>(*mSeqLengthsHost);
        auto inputLengthsHostPtr = bufferCast<int32_t>(*mInputLengthsHost);
        auto tokenReadIdxsHostPtr = bufferCast<int32_t>(*mTokenReadIdxsHost);
        auto tokenWriteIdxsHostPtr = bufferCast<int32_t>(*mTokenWriteIdxsHost);
        auto tokenPosIdxsHostPtr = bufferCast<int32_t>(*mTokenPosIdxsHost);
        auto tokenSeqIdxsHostPtr = bufferCast<int32_t>(*mTokenSeqIdxsHost);
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            seqLengthsHostPtr[bi] = seqLengths[bi];
            inputLengthsHostPtr[bi] = inputLengths[bi];
        }

        for (SizeType32 idx = 0; idx < tokenReadIdxs.size(); ++idx)
        {
            tokenReadIdxsHostPtr[idx] = tokenReadIdxs[idx];
            tokenWriteIdxsHostPtr[idx] = tokenWriteIdxs[idx];
            tokenPosIdxsHostPtr[idx] = tokenPosIdxs[idx];
            tokenSeqIdxsHostPtr[idx] = tokenSeqIdxs[idx];
        }

        if (pagedKvCache)
        {
            auto inputBlockOffsetsHostRange
                = BufferRange<std::remove_const_t<KVBlockArray::DataType::UnderlyingType>>(*mInputBlockOffsetsHost);
            std::iota(inputBlockOffsetsHostRange.begin(), inputBlockOffsetsHostRange.end(), 0);
            mBufferManager->copy(*mInputBlockOffsetsHost, *mInputBlockOffsetsDevice);
        }

        mBufferManager->copy(*mInputDataHost, *mInputDataDevice);
        mBufferManager->copy(*mSeqLengthsHost, *mSeqLengthsDevice);
        mBufferManager->copy(*mInputLengthsHost, *mInputLengthsDevice);
        mBufferManager->copy(*mTokenReadIdxsHost, *mTokenReadIdxsDevice);
        mBufferManager->copy(*mTokenWriteIdxsHost, *mTokenWriteIdxsDevice);
        mBufferManager->copy(*mTokenPosIdxsHost, *mTokenPosIdxsDevice);
        mBufferManager->copy(*mTokenSeqIdxsHost, *mTokenSeqIdxsDevice);
    }

    float compareResults(KVLinearBuffer const& kCacheOut, KVLinearBuffer const& kCacheRef, int32_t batchBeam,
        int32_t beamWidth, int32_t numHeads, int32_t headSize, int32_t validTokenNum, int32_t const* seqLengths,
        int32_t const* inputLengths, int32_t const* tokenWriteIdxs, int32_t const* tokenSeqIdxs)
    {
        mBufferManager->copy(*mOutputDataDevice, *mOutputDataHost);
        mBufferManager->copy(*mRefOutputDataDevice, *mRefOutputDataHost);

        // Synchronize
        mStream->synchronize();

        // Compare the results
        float tot_diff = 0.f;
        for (SizeType32 bi = 0; bi < batchBeam; ++bi)
        {
            int const tlength = seqLengths[bi] - 1;
            int const inlength = inputLengths[bi];
            int const beam_idx = bi % beamWidth;
            for (SizeType32 hi = 0; hi < numHeads; ++hi)
            {
                for (SizeType32 ti = 0; ti < validTokenNum; ++ti)
                {
                    int const token_seq_idx = tokenSeqIdxs[ti];
                    int const token_write_idx = tokenWriteIdxs[ti];
                    bool const valid_seq = token_seq_idx < tlength && !(token_seq_idx < inlength && beam_idx > 0);
                    if (!valid_seq)
                    {
                        continue;
                    }
                    for (SizeType32 ci = 0; ci < headSize; ++ci)
                    {
                        T* kRes = reinterpret_cast<T*>(kCacheOut.getKBlockPtr(bi, token_write_idx));
                        int resIdx = kCacheOut.getKVLocalIdx(token_write_idx, hi, headSize, ci);
                        T* kRef = reinterpret_cast<T*>(kCacheRef.getKBlockPtr(bi, token_write_idx));
                        int refIdx = kCacheRef.getKVLocalIdx(token_write_idx, hi, headSize, ci);
                        float res = static_cast<float>(kRes[resIdx]);
                        float ref = static_cast<float>(kRef[refIdx]);
                        float diff = std::abs(res - ref);
                        tot_diff += diff;
                    }
                }
            }
        }
        return tot_diff;
    }

    void runTest(int32_t batchSize, int32_t beamWidth, int32_t numHeads, int32_t headSize, int32_t maxAttentionWindow,
        int32_t sinkTokenLength, int32_t pastKCacheLength, int32_t validTokenNum, bool pagedKvCache,
        int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int rotaryEmbeddingDim, float rotaryEmbeddingBase,
        RotaryScalingType const rotaryScaleType, float rotaryEmbeddingScale, int const rotaryEmbeddingMaxPositions,
        PositionEmbeddingType const positionEmbeddingType)
    {
        // Synchronize
        mStream->synchronize();

        // get kv cache
        int32_t const batchBeam = batchSize * beamWidth;
        auto const elemSize = sizeof(T);
        auto const sizePerToken = numHeads * headSize * elemSize;
        bool const canUseOneMoreBlock = beamWidth > 1;

        auto shiftKCacheBuffer = KVLinearBuffer(batchBeam, maxAttentionWindow, sizePerToken, maxAttentionWindow,
            sinkTokenLength, true, reinterpret_cast<int8_t*>(bufferCast<T>(*mOutputDataDevice)));

        auto refShiftKCacheBuffer = KVLinearBuffer(batchBeam, maxAttentionWindow, sizePerToken, maxAttentionWindow,
            sinkTokenLength, true, reinterpret_cast<int8_t*>(bufferCast<T>(*mRefOutputDataDevice)));

        // run shift k cache
        const KvCacheDataType kv_cache_type = KvCacheDataType::BASE;
        using DataType = typename SATypeConverter<T>::Type;

        if (pagedKvCache)
        {
            auto inputDataDevicePtr = bufferCast<T>(*mInputDataDevice);

            auto const kvCacheBuffer = KVBlockArray(batchBeam, maxBlocksPerSeq, tokensPerBlock, sizePerToken,
                maxAttentionWindow, maxAttentionWindow, sinkTokenLength, canUseOneMoreBlock, inputDataDevicePtr,
                nullptr, bufferCast<KVBlockArray::DataType>(*mInputBlockOffsetsDevice));
            invokeShiftKCache<DataType, KVBlockArray>(kvCacheBuffer, shiftKCacheBuffer, kv_cache_type, headSize,
                pastKCacheLength, batchBeam, numHeads, beamWidth, maxAttentionWindow, sinkTokenLength,
                bufferCast<float>(*mKScaleQuantOrigDevice), bufferCast<int32_t>(*mSeqLengthsDevice),
                bufferCast<int32_t>(*mInputLengthsDevice), rotaryEmbeddingDim, rotaryEmbeddingBase, rotaryScaleType,
                rotaryEmbeddingScale, rotaryEmbeddingMaxPositions, positionEmbeddingType, mStream->get());
            // run ref
            invokeApplyRoPE<DataType, KVBlockArray>(kvCacheBuffer, refShiftKCacheBuffer, headSize, batchBeam, numHeads,
                beamWidth, bufferCast<int32_t>(*mTokenReadIdxsDevice), bufferCast<int32_t>(*mTokenWriteIdxsDevice),
                bufferCast<int32_t>(*mTokenPosIdxsDevice), bufferCast<int32_t>(*mTokenSeqIdxsDevice), validTokenNum,
                bufferCast<int32_t>(*mSeqLengthsDevice), bufferCast<int32_t>(*mInputLengthsDevice), rotaryEmbeddingDim,
                rotaryEmbeddingBase, rotaryScaleType, rotaryEmbeddingScale, rotaryEmbeddingMaxPositions,
                positionEmbeddingType, mStream->get());
        }
        else
        {
            auto const kvCacheBuffer
                = KVLinearBuffer(batchBeam, maxAttentionWindow, numHeads * headSize * elemSize, maxAttentionWindow,
                    sinkTokenLength, false, reinterpret_cast<int8_t*>(bufferCast<T>(*mInputDataDevice)));
            // run shift k cache
            invokeShiftKCache<DataType, KVLinearBuffer>(kvCacheBuffer, shiftKCacheBuffer, kv_cache_type, headSize,
                pastKCacheLength, batchBeam, numHeads, beamWidth, maxAttentionWindow, sinkTokenLength,
                bufferCast<float>(*mKScaleQuantOrigDevice), bufferCast<int32_t>(*mSeqLengthsDevice),
                bufferCast<int32_t>(*mInputLengthsDevice), rotaryEmbeddingDim, rotaryEmbeddingBase, rotaryScaleType,
                rotaryEmbeddingScale, rotaryEmbeddingMaxPositions, positionEmbeddingType, mStream->get());
            // run ref
            invokeApplyRoPE<DataType, KVLinearBuffer>(kvCacheBuffer, refShiftKCacheBuffer, headSize, batchBeam,
                numHeads, beamWidth, bufferCast<int32_t>(*mTokenReadIdxsDevice),
                bufferCast<int32_t>(*mTokenWriteIdxsDevice), bufferCast<int32_t>(*mTokenPosIdxsDevice),
                bufferCast<int32_t>(*mTokenSeqIdxsDevice), validTokenNum, bufferCast<int32_t>(*mSeqLengthsDevice),
                bufferCast<int32_t>(*mInputLengthsDevice), rotaryEmbeddingDim, rotaryEmbeddingBase, rotaryScaleType,
                rotaryEmbeddingScale, rotaryEmbeddingMaxPositions, positionEmbeddingType, mStream->get());
        }

        // Synchronize
        mStream->synchronize();
        shiftKCacheBuffer.data = reinterpret_cast<int8_t*>(bufferCast<T>(*mOutputDataHost));
        refShiftKCacheBuffer.data = reinterpret_cast<int8_t*>(bufferCast<T>(*mRefOutputDataHost));

        float diff = compareResults(shiftKCacheBuffer, refShiftKCacheBuffer, batchBeam, beamWidth, numHeads, headSize,
            validTokenNum, bufferCast<int32_t>(*mSeqLengthsHost), bufferCast<int32_t>(*mInputLengthsHost),
            bufferCast<int32_t>(*mTokenWriteIdxsHost), bufferCast<int32_t>(*mTokenSeqIdxsHost));
        EXPECT_EQ(diff, 0);
    }

protected:
    std::shared_ptr<BufferManager> mBufferManager;
    std::shared_ptr<CudaStream> mStream;

    TensorPtr mSeqLengthsHost;
    TensorPtr mSeqLengthsDevice;

    TensorPtr mInputLengthsHost;
    TensorPtr mInputLengthsDevice;

    TensorPtr mInputBlockOffsetsHost;
    TensorPtr mInputBlockOffsetsDevice;

    TensorPtr mKScaleQuantOrigDevice;

    TensorPtr mTokenReadIdxsHost;
    TensorPtr mTokenReadIdxsDevice;

    TensorPtr mTokenWriteIdxsHost;
    TensorPtr mTokenWriteIdxsDevice;

    TensorPtr mTokenPosIdxsHost;
    TensorPtr mTokenPosIdxsDevice;

    TensorPtr mTokenSeqIdxsHost;
    TensorPtr mTokenSeqIdxsDevice;

    TensorPtr mInputDataHost;
    TensorPtr mInputDataDevice;

    TensorPtr mOutputDataHost;
    TensorPtr mOutputDataDevice;

    TensorPtr mRefOutputDataHost;
    TensorPtr mRefOutputDataDevice;
};

typedef testing::Types<float, half> FloatAndHalfTypes;

TYPED_TEST_SUITE(ShiftKCacheKernelTest, FloatAndHalfTypes);

TYPED_TEST(ShiftKCacheKernelTest, UncyclicShiftKCache)
{
    auto constexpr batchSize = 1;
    auto constexpr beamWidth = 1;
    auto constexpr numHeads = 2;
    auto constexpr headSize = 64;
    auto constexpr maxAttentionWindow = 32;
    auto constexpr sinkTokenLength = 0;
    auto constexpr timestep = 17;
    auto constexpr pastKCacheLength = timestep - 1;
    auto constexpr validTokenNum = std::min(pastKCacheLength, maxAttentionWindow);
    auto constexpr rotaryEmbeddingDim = headSize;
    auto constexpr rotaryEmbeddingBase = 10000.0;
    auto constexpr rotaryScaleType = RotaryScalingType::kNONE;
    auto constexpr rotaryEmbeddingScale = 1.0;
    auto constexpr rotaryEmbeddingMaxPositions = 4096;
    auto constexpr positionEmbeddingType = PositionEmbeddingType::kROPE_GPT_NEOX;

    auto pagedKvCaches = std::vector<bool>{false, true};
    for (auto pagedKvCache : pagedKvCaches)
    {
        const SizeType32 maxBlocksPerSeq = (pagedKvCache) ? 2 : 0;
        const SizeType32 tokensPerBlock = (pagedKvCache) ? 16 : 0;
        // include one more token for the current time step in seqLengths.
        std::vector<int32_t> seqLengths = {timestep};
        std::vector<int32_t> inputLengths = {8};
        std::vector<int32_t> tokenReadIdxs;
        std::vector<int32_t> tokenWriteIdxs;
        std::vector<int32_t> tokenPosIdxs;
        std::vector<int32_t> tokenSeqIdxs;
        for (SizeType32 idx = 0; idx < timestep - 1; ++idx)
        {
            tokenReadIdxs.push_back(idx);
            tokenWriteIdxs.push_back(idx);
            tokenPosIdxs.push_back(idx);
            tokenSeqIdxs.push_back(idx);
        }

        this->initData(batchSize, beamWidth, numHeads, maxAttentionWindow, headSize, pagedKvCache, maxBlocksPerSeq,
            tokensPerBlock, seqLengths, inputLengths, tokenReadIdxs, tokenWriteIdxs, tokenPosIdxs, tokenSeqIdxs);
        this->runTest(batchSize, beamWidth, numHeads, headSize, maxAttentionWindow, sinkTokenLength, pastKCacheLength,
            validTokenNum, pagedKvCache, maxBlocksPerSeq, tokensPerBlock, rotaryEmbeddingDim, rotaryEmbeddingBase,
            rotaryScaleType, rotaryEmbeddingScale, rotaryEmbeddingMaxPositions, positionEmbeddingType);
    }
};

TYPED_TEST(ShiftKCacheKernelTest, CyclicShiftKCacheSimple)
{
    auto constexpr batchSize = 1;
    auto constexpr beamWidth = 1;
    auto constexpr numHeads = 2;
    auto constexpr headSize = 64;
    auto constexpr maxAttentionWindow = 32;
    auto constexpr sinkTokenLength = 0;
    auto constexpr timestep = 45;
    auto constexpr pastKCacheLength = timestep - 1;
    auto constexpr validTokenNum = std::min(pastKCacheLength, maxAttentionWindow);
    auto constexpr rotaryEmbeddingDim = headSize;
    auto constexpr rotaryEmbeddingBase = 10000.0;
    auto constexpr rotaryScaleType = RotaryScalingType::kNONE;
    auto constexpr rotaryEmbeddingScale = 1.0;
    auto constexpr rotaryEmbeddingMaxPositions = 4096;
    auto constexpr positionEmbeddingType = PositionEmbeddingType::kROPE_GPT_NEOX;

    auto pagedKvCaches = std::vector<bool>{false, true};
    for (auto pagedKvCache : pagedKvCaches)
    {
        const SizeType32 maxBlocksPerSeq = (pagedKvCache) ? 2 : 0;
        const SizeType32 tokensPerBlock = (pagedKvCache) ? 16 : 0;
        // include one more token for the current time step in seqLengths.
        std::vector<int32_t> seqLengths = {timestep};
        std::vector<int32_t> inputLengths = {8};
        std::vector<int32_t> tokenReadIdxs;
        std::vector<int32_t> tokenWriteIdxs;
        std::vector<int32_t> tokenPosIdxs;
        std::vector<int32_t> tokenSeqIdxs;
        for (SizeType32 idx = pastKCacheLength - maxAttentionWindow; idx < pastKCacheLength; ++idx)
        {
            tokenReadIdxs.push_back(idx % maxAttentionWindow);
            tokenWriteIdxs.push_back(idx % maxAttentionWindow);
            tokenPosIdxs.push_back(idx - pastKCacheLength + maxAttentionWindow);
            tokenSeqIdxs.push_back(idx);
        }

        this->initData(batchSize, beamWidth, numHeads, maxAttentionWindow, headSize, pagedKvCache, maxBlocksPerSeq,
            tokensPerBlock, seqLengths, inputLengths, tokenReadIdxs, tokenWriteIdxs, tokenPosIdxs, tokenSeqIdxs);
        this->runTest(batchSize, beamWidth, numHeads, headSize, maxAttentionWindow, sinkTokenLength, pastKCacheLength,
            validTokenNum, pagedKvCache, maxBlocksPerSeq, tokensPerBlock, rotaryEmbeddingDim, rotaryEmbeddingBase,
            rotaryScaleType, rotaryEmbeddingScale, rotaryEmbeddingMaxPositions, positionEmbeddingType);
    }
};

TYPED_TEST(ShiftKCacheKernelTest, CyclicShiftKCacheSink)
{
    auto constexpr batchSize = 1;
    auto constexpr beamWidth = 1;
    auto constexpr numHeads = 2;
    auto constexpr headSize = 64;
    auto constexpr maxAttentionWindow = 32;
    auto constexpr sinkTokenLength = 4;
    auto constexpr timestep = 67;
    auto constexpr pastKCacheLength = timestep - 1;
    auto constexpr validTokenNum = std::min(pastKCacheLength, maxAttentionWindow);
    auto constexpr rotaryEmbeddingDim = headSize;
    auto constexpr rotaryEmbeddingBase = 10000.0;
    auto constexpr rotaryScaleType = RotaryScalingType::kNONE;
    auto constexpr rotaryEmbeddingScale = 1.0;
    auto constexpr rotaryEmbeddingMaxPositions = 4096;
    auto constexpr positionEmbeddingType = PositionEmbeddingType::kROPE_GPT_NEOX;

    auto pagedKvCaches = std::vector<bool>{false, true};
    for (auto pagedKvCache : pagedKvCaches)
    {
        const SizeType32 maxBlocksPerSeq = (pagedKvCache) ? 3 : 0;
        const SizeType32 tokensPerBlock = (pagedKvCache) ? 16 : 1;
        const SizeType32 sinkTokensInLastBlock = sinkTokenLength % tokensPerBlock;
        const SizeType32 bubbleLength = sinkTokensInLastBlock == 0 ? 0 : tokensPerBlock - sinkTokensInLastBlock;
        // include one more token for the current time step in seqLengths.
        std::vector<int32_t> seqLengths = {timestep};
        std::vector<int32_t> inputLengths = {8};
        std::vector<int32_t> tokenReadIdxs = {0, 1, 2, 3};
        std::vector<int32_t> tokenWriteIdxs = {0, 1, 2, 3};
        std::vector<int32_t> tokenPosIdxs = {0, 1, 2, 3};
        std::vector<int32_t> tokenSeqIdxs = {0, 1, 2, 3};

        int const cyclicLength = maxAttentionWindow - sinkTokenLength;
        for (SizeType32 idx = pastKCacheLength - cyclicLength; idx < pastKCacheLength; ++idx)
        {
            tokenReadIdxs.push_back(sinkTokenLength + bubbleLength + (idx - sinkTokenLength) % cyclicLength);
            tokenWriteIdxs.push_back(sinkTokenLength + (idx - sinkTokenLength) % cyclicLength);
            tokenPosIdxs.push_back(sinkTokenLength + idx - pastKCacheLength + cyclicLength);
            tokenSeqIdxs.push_back(idx);
        }

        this->initData(batchSize, beamWidth, numHeads, maxAttentionWindow, headSize, pagedKvCache, maxBlocksPerSeq,
            tokensPerBlock, seqLengths, inputLengths, tokenReadIdxs, tokenWriteIdxs, tokenPosIdxs, tokenSeqIdxs);
        this->runTest(batchSize, beamWidth, numHeads, headSize, maxAttentionWindow, sinkTokenLength, pastKCacheLength,
            validTokenNum, pagedKvCache, maxBlocksPerSeq, tokensPerBlock, rotaryEmbeddingDim, rotaryEmbeddingBase,
            rotaryScaleType, rotaryEmbeddingScale, rotaryEmbeddingMaxPositions, positionEmbeddingType);
    }
};

TYPED_TEST(ShiftKCacheKernelTest, CyclicShiftKCacheSinkOneMoreBlock)
{
    auto constexpr batchSize = 1;
    auto constexpr beamWidth = 2;
    auto constexpr numHeads = 2;
    auto constexpr headSize = 64;
    auto constexpr maxAttentionWindow = 32;
    auto constexpr sinkTokenLength = 4;
    auto constexpr timestep = 67;
    auto constexpr pastKCacheLength = timestep - 1;
    auto constexpr validTokenNum = std::min(pastKCacheLength, maxAttentionWindow);
    auto constexpr rotaryEmbeddingDim = headSize;
    auto constexpr rotaryEmbeddingBase = 10000.0;
    auto constexpr rotaryScaleType = RotaryScalingType::kNONE;
    auto constexpr rotaryEmbeddingScale = 1.0;
    auto constexpr rotaryEmbeddingMaxPositions = 4096;
    auto constexpr positionEmbeddingType = PositionEmbeddingType::kROPE_GPT_NEOX;

    auto constexpr pagedKvCache = true;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr tokensPerBlock = 16;
    auto constexpr sinkTokensInLastBlock = sinkTokenLength % tokensPerBlock;
    auto constexpr bubbleLength = sinkTokensInLastBlock == 0 ? 0 : tokensPerBlock - sinkTokensInLastBlock;
    // include one more token for the current time step in seqLengths.
    std::vector<int32_t> seqLengths = {timestep};
    std::vector<int32_t> inputLengths = {8};
    std::vector<int32_t> tokenReadIdxs = {0, 1, 2, 3};
    std::vector<int32_t> tokenWriteIdxs = {0, 1, 2, 3};
    std::vector<int32_t> tokenPosIdxs = {0, 1, 2, 3};
    std::vector<int32_t> tokenSeqIdxs = {0, 1, 2, 3};

    auto constexpr cyclicLength = maxAttentionWindow - sinkTokenLength;
    auto constexpr rCyclicLength = maxAttentionWindow - sinkTokenLength + tokensPerBlock;
    auto constexpr wCyclicLength = cyclicLength;
    for (SizeType32 idx = pastKCacheLength - cyclicLength; idx < pastKCacheLength; ++idx)
    {
        tokenReadIdxs.push_back(sinkTokenLength + bubbleLength + (idx - sinkTokenLength) % rCyclicLength);
        tokenWriteIdxs.push_back(sinkTokenLength + (idx - sinkTokenLength) % wCyclicLength);
        tokenPosIdxs.push_back(sinkTokenLength + idx - pastKCacheLength + cyclicLength);
        tokenSeqIdxs.push_back(idx);
    }

    this->initData(batchSize, beamWidth, numHeads, maxAttentionWindow, headSize, pagedKvCache, maxBlocksPerSeq,
        tokensPerBlock, seqLengths, inputLengths, tokenReadIdxs, tokenWriteIdxs, tokenPosIdxs, tokenSeqIdxs);
    this->runTest(batchSize, beamWidth, numHeads, headSize, maxAttentionWindow, sinkTokenLength, pastKCacheLength,
        validTokenNum, pagedKvCache, maxBlocksPerSeq, tokensPerBlock, rotaryEmbeddingDim, rotaryEmbeddingBase,
        rotaryScaleType, rotaryEmbeddingScale, rotaryEmbeddingMaxPositions, positionEmbeddingType);
};

} // end of namespace
