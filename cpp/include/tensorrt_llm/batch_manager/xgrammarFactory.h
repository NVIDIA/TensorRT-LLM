#include "tensorrt_llm/batch_manager/guidedDecoder.h"

namespace xgrammar
{
class GrammarMatcher;
class GrammarCompiler;
} // namespace xgrammar

namespace tensorrt_llm::batch_manager
{
class XGrammarMatcher final : public IGrammarMatcher
{
public:
    explicit XGrammarMatcher(std::shared_ptr<xgrammar::GrammarMatcher> grammarMatcher);

    bool AcceptToken(int32_t tokenId) override;
    void FillNextTokenBitmask(DLTensor* nextTokenBitmask) override;

private:
    std::shared_ptr<xgrammar::GrammarMatcher> mGrammarMatcher;
};

class XGrammarMatcherFactory final : public IGrammarMatcherFactory
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    XGrammarMatcherFactory(
        tensorrt_llm::executor::GuidedDecodingConfig const& guidedDecodingConfig, SizeType32 vocabSizePadded);

    std::shared_ptr<IGrammarMatcher> Create(tensorrt_llm::executor::GuidedDecodingParams const& params) override;

private:
    std::shared_ptr<xgrammar::GrammarCompiler> mXGrammarCompiler;
};
} // namespace tensorrt_llm::batch_manager
