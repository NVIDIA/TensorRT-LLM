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
        runtime::TokenIdType endToken, std::shared_ptr<runtime::BufferManager> bufferManager)
        : mW(w)
        , mN(n)
        , mG(g)
        , mEndToken(endToken)
        , mFilling(0)
        , mBufferManager(bufferManager)
        , mPoolManager(g, bufferManager)
        , mGoldenTokens(mBufferManager->cpu(runtime::ITensor::makeShape({n * 2 - 2}), nvinfer1::DataType::kINT32))
        , mPastTokens(mBufferManager->cpu(runtime::ITensor::makeShape({w, n - 1}), nvinfer1::DataType::kINT32))
        , mPrefills(mBufferManager->cpu(runtime::ITensor::makeShape({n - 2}), nvinfer1::DataType::kINT32))
    {
    }

    //! @brief setup per request, fill internal states from @param prompt.
    void setup(TensorPtr prompt);

    //! @brief combine lookahead and guess to prepare the tensors.
    //! @param offset is position id of the last golden token.
    //! @param lastToken the last golden token for searching in the pool.
    //! @return a tuple of <lookahead tokens, position ids, sampling mask>, including the golden token, the lookahead
    //! and the verification branch information.
    std::tuple<TensorPtr, TensorPtr, TensorPtr> prepare(runtime::SizeType32 offset, runtime::TokenIdType lastToken);

    //! @brief update the internal states and generate accepted tokens from @param outputTokens.
    //! @param outputTokens is the all the tokens from the language model. The position at samplingMask=1 is valid.
    //! @return the longest accepted token tensor, note, at least one.
    TensorPtr update(TensorPtr outputTokens);

private:
    //! @brief generate lookahead branch information.
    //! @param offset the position id of the last golden token.
    //! @return a tuple of <lookahead tokens, position ids, sampling mask>.
    std::tuple<TensorPtr, TensorPtr, TensorPtr> lookahead(runtime::SizeType32 offset);

    //! @brief generate verification branch information. Also save the guessed tokens for future verification.
    //! @param offset the position id of the last golden token.
    //! @param lastToken the last golden token for searching in the pool.
    //! @return a tuple of <lookahead tokens, position ids>.
    std::tuple<TensorPtr, TensorPtr> guess(runtime::SizeType32 offset, runtime::TokenIdType lastToken);

    //! @brief verify the guessed tokens results and generate the longest accepted tokens.
    //! @param newLastToken is the new-generated last golden token.
    //! @param goldenTokens is the guessed token results from the language model.
    //! @return the longest accepted token tensor, note, at least one.
    TensorPtr verify(runtime::TokenIdType newLastToken, TensorPtr goldenTokens);

private:
    std::shared_ptr<runtime::BufferManager> mBufferManager;
    LookaheadPoolManager mPoolManager;
    //! the random prefill tokens, shape [(mN-2)]
    TensorPtr mPrefills;
    //! shape [mW, (mN-1)], the look ahead branch window
    TensorPtr mPastTokens;
    //! all the moving tail golden tokens, shape[mN*2-2]
    TensorPtr mGoldenTokens;
    //! the same guess tokens from `guess` and used in `verify`
    TensorPtr mGuessTokens;
    //! look ahead algorithm parameters, Window size, Level and Guess set size.
    runtime::SizeType32 mW, mN, mG;
    //! in prefilling mode when mFilling < mN-1.
    runtime::SizeType32 mFilling;
    //! the end token for verification early quit.
    runtime::TokenIdType mEndToken;

    //! @brief record the current golden token for debugging.
    runtime::TokenIdType mCurrentToken;
};

} // namespace tensorrt_llm::layers
