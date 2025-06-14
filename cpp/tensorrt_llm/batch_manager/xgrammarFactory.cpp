#include "tensorrt_llm/batch_manager/xgrammarFactory.h"

#include <picojson.h>
#include <xgrammar/xgrammar.h>

namespace tle = tensorrt_llm::executor;

namespace tensorrt_llm::batch_manager
{
XGrammarMatcher::XGrammarMatcher(std::shared_ptr<xgrammar::GrammarMatcher> grammarMatcher)
    : mGrammarMatcher(grammarMatcher)
{
}

bool XGrammarMatcher::AcceptToken(int32_t tokenId)
{
    return mGrammarMatcher->AcceptToken(tokenId);
}

void XGrammarMatcher::FillNextTokenBitmask(DLTensor* nextTokenBitmask)
{
    mGrammarMatcher->FillNextTokenBitmask(nextTokenBitmask);
}

XGrammarMatcherFactory::XGrammarMatcherFactory(
    tle::GuidedDecodingConfig const& guidedDecodingConfig, const SizeType32 vocabSizePadded)
{
    auto const& tokenizerStr = guidedDecodingConfig.getTokenizerStr();
    if (tokenizerStr)
    {
        auto const& tokenizerInfo = xgrammar::TokenizerInfo::FromHuggingFace(
            guidedDecodingConfig.getEncodedVocab().value(), guidedDecodingConfig.getTokenizerStr().value(),
            vocabSizePadded, guidedDecodingConfig.getStopTokenIds());
        mXGrammarCompiler = std::make_shared<xgrammar::GrammarCompiler>(tokenizerInfo);
    }
    else
    {
        auto const& tokenizerInfo = xgrammar::TokenizerInfo(guidedDecodingConfig.getEncodedVocab().value(),
            xgrammar::VocabType::RAW, vocabSizePadded, guidedDecodingConfig.getStopTokenIds());
        mXGrammarCompiler = std::make_shared<xgrammar::GrammarCompiler>(tokenizerInfo);
    }
}

std::shared_ptr<IGrammarMatcher> XGrammarMatcherFactory::Create(tle::GuidedDecodingParams const& guidedDecodingParams)
{
    auto const& guideType = guidedDecodingParams.getGuideType();
    auto const& guide = guidedDecodingParams.getGuide();
    std::shared_ptr<xgrammar::GrammarMatcher> grammarMatcher;

    switch (guideType)
    {
    case tle::GuidedDecodingParams::GuideType::kJSON:
    {
        grammarMatcher = std::make_shared<xgrammar::GrammarMatcher>(mXGrammarCompiler->CompileBuiltinJSONGrammar());
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kJSON_SCHEMA:
    {
        grammarMatcher
            = std::make_shared<xgrammar::GrammarMatcher>(mXGrammarCompiler->CompileJSONSchema(guide.value()));
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kREGEX:
    {
        auto const& grammar = xgrammar::Grammar::FromRegex(guide.value());
        grammarMatcher = std::make_shared<xgrammar::GrammarMatcher>(mXGrammarCompiler->CompileGrammar(grammar));
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kEBNF_GRAMMAR:
    {
        auto const& grammar = xgrammar::Grammar::FromEBNF(guide.value());
        grammarMatcher = std::make_shared<xgrammar::GrammarMatcher>(mXGrammarCompiler->CompileGrammar(grammar));
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kSTRUCTURAL_TAG:
    {
        TLLM_CHECK_WITH_INFO(false, "kSTRUCTURAL_TAG is not supported by the xgrammar backend");
    }
    case tle::GuidedDecodingParams::GuideType::kLARK_GRAMMAR:
    {
        TLLM_CHECK_WITH_INFO(false, "kLARK_GRAMMAR is not supported by the xgrammar backend");
    }
    }

    return std::make_shared<XGrammarMatcher>(grammarMatcher);
}
} // namespace tensorrt_llm::batch_manager
