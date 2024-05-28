#pragma once

#include "lookaheadPoolManager.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"
#include <curand_kernel.h>

namespace tensorrt_llm::layers
{

//! @brief An CPU implementation of Lookahead with ITensor.
class LookaheadAlgorithm
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;

    //! @brief Currently the resource management is to be aligned with batch manager.
    //! @param w, n, g is the Jacobi window, n-gram level and guess set size respectively.
    LookaheadAlgorithm(runtime::SizeType32 w, runtime::SizeType32 n, runtime::SizeType32 g,
        std::shared_ptr<runtime::BufferManager> bufferManager)
        : mW(w)
        , mN(n)
        , mG(g)
        , mFilling(0)
        , mBufferManager(bufferManager)
        , mPoolManager(g, bufferManager)
        , mGoldenTokens(mBufferManager->cpu(runtime::ITensor::makeShape({n * 2 - 1}), nvinfer1::DataType::kINT32))
        , mPrefills(mBufferManager->cpu(runtime::ITensor::makeShape({n - 2}), nvinfer1::DataType::kINT32))
        , mKeyTokens(mBufferManager->cpu(runtime::ITensor::makeShape({w}), nvinfer1::DataType::kINT32))
        , mPastTokens(mBufferManager->cpu(runtime::ITensor::makeShape({w, n - 1}), nvinfer1::DataType::kINT32))
        , mGuessTokensMax(mBufferManager->cpu(runtime::ITensor::makeShape({g * (n - 1)}), nvinfer1::DataType::kINT32))
    {
    }

    //! @brief setup per request, fill internal states from @param prompt.
    void setup(TensorPtr prompt);

    //! @brief accept the new generated tokens.
    //! LookaheadDecodingLayer need call once for the first token in generation phase.
    void accept(TensorPtr generatedTokens);

    //! @brief combine lookahead and guess to prepare the tensors.
    //! input @param offsetPtr is position id of the last golden token, in a TensorPtr.
    //! input @param lastTokenPtr the last golden token for searching in the pool, in a TensorPtr.
    //! output @param draftTokens, positionIds, samplingMask; including the golden token, the lookahead
    //! and the verification branch information. @param length holds the draft tokens length.
    void prepare(TensorPtr draftTokens, TensorPtr positionIds, TensorPtr samplingMask, TensorPtr length,
        TensorPtr offsetPtr, TensorPtr lastTokenPtr);

    //! @brief update the internal states and generate accepted tokens from @param outputTokens.
    //! input @param sampledTokens is the all the tokens from the language model. The position at samplingMask=1 is
    //! valid. input @param endToken is the end token for `verify` early quit.
    //! output @param acceptedTokens, acceptedOffsets ind @param acceptedLength.
    void update(TensorPtr acceptedTokens, TensorPtr acceptedOffsets, TensorPtr acceptedLength, TensorPtr sampledTokens,
        TensorPtr endToken);

private:
    //! @brief generate lookahead branch information.
    //! input @param offset the position id of the last golden token.
    //! output @param draftTokens, positionIds, samplingMask of the lookahead branch.
    //! @return the actual filled lookahead length.
    runtime::SizeType32 lookahead(
        TensorPtr draftTokens, TensorPtr positionIds, TensorPtr samplingMask, runtime::SizeType32 offset);

    //! @brief generate verification branch information. Also save the guessed tokens for future verification.
    //! input @param offset the position id of the last golden token.
    //! input @param lastToken the last golden token for searching in the pool.
    //! output @param guessTokens, guessIds, samplingMask of the verification branch.
    //! @return the actual filled guess length.
    runtime::SizeType32 guess(TensorPtr guessTokens, TensorPtr guessIds, TensorPtr samplingMask,
        runtime::SizeType32 offset, runtime::TokenIdType lastToken);

    //! @brief verify the guessed tokens results and generate the longest accepted tokens.
    //! input @param newLastToken is the new-generated last golden token.
    //! input @param goldenTokens is the guessed token results from the language model.
    //! input @param endToken is the end token for early quit detection.
    //! output @param accepted, acceptedOffsets in @param acceptedLength, .
    void verify(TensorPtr accepted, TensorPtr acceptedOffsets, TensorPtr acceptedLength,
        runtime::TokenIdType newLastToken, TensorPtr goldenTokens, TensorPtr endToken);

private:
    std::shared_ptr<runtime::BufferManager> mBufferManager;
    LookaheadPoolManager mPoolManager;
    //! the random prefill tokens, shape [(mN-2)]
    TensorPtr mPrefills;
    //! shape [mW, (mN-1)], the look ahead branch window
    TensorPtr mPastTokens;
    //! shape [mW], the shifted mPastTokens as key tokens;
    TensorPtr mKeyTokens;
    //! all the moving tail golden tokens, shape[mN*2-1]
    TensorPtr mGoldenTokens;
    //! the same guess tokens from `guess` and used in `verify`
    TensorPtr mGuessTokens;
    //! shape [mG*(mN-1)]. the pre-allocated space for mGuessTokens.
    TensorPtr mGuessTokensMax;
    //! look ahead algorithm parameters, Window size, Level and Guess set size.
    runtime::SizeType32 mW, mN, mG;
    //! in prefilling mode when mFilling < mN-1.
    runtime::SizeType32 mFilling;

    //! @brief record the current golden token for debugging.
    runtime::TokenIdType mCurrentToken;
};

} // namespace tensorrt_llm::layers
