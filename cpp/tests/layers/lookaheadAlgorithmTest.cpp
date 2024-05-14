#include <gtest/gtest.h>

#include "tensorrt_llm/layers/lookaheadAlgorithm.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tests/layers/randomLlm.h"

namespace tensorrt_llm::tests::layers
{
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::layers;
using TensorPtr = runtime::ITensor::SharedPtr;

TEST(LookaheadRandomllm, forward)
{
    // '&' is the end token ID, and '#' is the invalid token ID.
    std::string asciiVocab("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -+_,:;.!?()[]{}'\"&#");
    auto ascii = std::make_shared<RandomTokenLogits>(asciiVocab);
    EXPECT_EQ(ascii->getVocabSize(), asciiVocab.size());
    {
        auto tensor = ascii->tokenToLogits(static_cast<TokenIdType>('a'));
        auto token = ascii->logitsToToken(tensor);
        EXPECT_EQ(static_cast<char>(token), 'a');
    }
    {
        auto tensor = ascii->tokenToLogits(static_cast<TokenIdType>('W'));
        auto token = ascii->logitsToToken(tensor);
        EXPECT_EQ(static_cast<char>(token), 'W');
    }
    {
        auto tensor = ascii->tokenToLogits(-1);
        auto token = ascii->logitsToToken(tensor);
        EXPECT_EQ(static_cast<char>(token), '#');
    }
    {
        std::string str("hello world!");
        TensorPtr logits
            = BufferManager::cpu(ITensor::makeShape({static_cast<SizeType32>(str.size()), ascii->getVocabSize()}),
                nvinfer1::DataType::kFLOAT);
        ascii->stringToLogits(logits, str);
        auto result = ascii->logitsToString(logits);
        EXPECT_EQ(result, str);
    }

    std::string oracle(
        "The following example uses a lambda-expression to increment all of the elements of a vector and "
        "then uses an overloaded operator() in a function object (a.k.a., \"functor\") to compute their sum. Note that "
        "to compute the sum, it is recommended to use the dedicated algorithm std::accumulate.");
    LookaheadRandomLlm llm(ascii, oracle);
    {
        TLLM_LOG_DEBUG("oracle[22]='%c'", oracle[22]);
        std::string input("ubcs23eess a la");
        auto len = static_cast<SizeType32>(input.size());
        TensorPtr inputTokens = initTensor(input);
        std::vector<TokenIdType> positionIdVec({22, 23, 24, 23, 24, 25, 24, 25, 26, 25, 26, 27, 26, 27, 28});
        TensorPtr positionIds = ITensor::wrap(positionIdVec, ITensor::makeShape({len}));
        TensorPtr outputLogits
            = BufferManager::cpu(ITensor::makeShape({len, ascii->getVocabSize()}), nvinfer1::DataType::kFLOAT);

        llm.forward(outputLogits, inputTokens, positionIds);

        auto result = ascii->logitsToString(outputLogits);
        auto invalid = ascii->getInvalidToken();
        TLLM_LOG_DEBUG("result=%s", result.c_str());
        for (SizeType32 i = 0; i < len; i++)
        {
            if (result[i] != invalid)
            {
                EXPECT_EQ(result[i], oracle[positionIdVec[i] + 1]);
            }
        }
    }
}

class LookaheadAlgorithmTest : public ::testing::TestWithParam<std::tuple<int, int, int>>
{
};

TEST_P(LookaheadAlgorithmTest, predict)
{
    auto [W, N, G] = GetParam();
    auto mStream = std::make_shared<CudaStream>();
    auto mBufferManager = std::make_shared<BufferManager>(mStream);

    // '&' is the end token ID, and '#' is the invalid token ID.
    std::string asciiVocab("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -+_,:;.!?()[]{}'\"&#");
    auto ascii = std::make_shared<RandomTokenLogits>(asciiVocab);

    std::string oracle(
        "The following example uses the following lambda-expression to increment all of the elements of a vector and "
        "then uses an overloaded operator() in a function object (a.k.a., \"functor\") to compute their sum. Note that "
        "to compute the sum, it is recommended to use the dedicated algorithm std::accumulate.&");
    LookaheadRandomLlm llm(ascii, oracle);

    auto prompt = initTensor(std::string(oracle.substr(0, 20)));
    auto promptLen = ITensor::volume(prompt->getShape());
    std::string result(oracle.substr(0, 20));

    SizeType32 lastPosid = promptLen;
    TokenIdType lastToken = oracle[lastPosid]; // from context phase.
    result.push_back(static_cast<char>(lastToken));

    tensorrt_llm::layers::LookaheadAlgorithm algo(W, N, G, ascii->getEndToken(), mBufferManager);
    algo.setup(prompt);

    SizeType32 seqLen = oracle.size();
    std::vector<int> histogram(N - 1, 0);
    for (; lastPosid < seqLen - 1;)
    {
        TLLM_LOG_DEBUG("oracle[%d] = '%c'", lastPosid, static_cast<char>(lastToken));
        auto [input, posid, smask] = algo.prepare(lastPosid, lastToken);

        PRINT_TOKENS(input);
        PRINT_TENSOR(posid);
        PRINT_TENSOR(smask);
        TensorPtr output = BufferManager::cpu(input->getShape(), nvinfer1::DataType::kINT32);
        llm.foretell(output, input, posid);
        llm.sampleByMask(output, smask);
        PRINT_TOKENS(output);

        auto accepted = algo.update(output);
        EXPECT_TRUE(llm.verify(lastPosid + 1, accepted));

        auto acceptedLen = ITensor::volume(accepted->getShape());
        auto acceptedRange = BufferRange<TokenIdType>(*accepted);
        histogram[acceptedLen] += 1;
        lastPosid += acceptedLen;
        lastToken = acceptedRange[acceptedLen - 1];
        std::for_each(
            acceptedRange.begin(), acceptedRange.end(), [&result](auto t) { result.push_back(static_cast<char>(t)); });
        TLLM_LOG_DEBUG("seqLen=%d, lastPosid=%d, RESULT: %s", seqLen, lastPosid, result.c_str());
    }
    EXPECT_EQ(lastPosid, seqLen - 1);

    std::ostringstream buf;
    buf << "Lookahead acceptance histogram: ";
    std::for_each(histogram.begin(), histogram.end(), [&buf](auto& v) { buf << v << ", "; });
    TLLM_LOG_DEBUG(buf.str());
}

INSTANTIATE_TEST_CASE_P(CombineLookaheadAlgorithmTest, LookaheadAlgorithmTest,
    testing::Combine(testing::Values(1, 3, 5), testing::Values(3, 5), testing::Values(3, 5)));

} // namespace tensorrt_llm::tests::layers
