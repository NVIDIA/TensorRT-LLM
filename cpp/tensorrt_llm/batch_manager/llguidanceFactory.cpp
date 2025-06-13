#include "tensorrt_llm/batch_manager/llguidanceFactory.h"

namespace tle = tensorrt_llm::executor;

namespace
{

void checkMatcher(LlgMatcher* matcher)
{
    char const* matcherError = llg_matcher_get_error(matcher);
    TLLM_CHECK_WITH_INFO(matcherError == nullptr, "llguidance matcher error: %s", matcherError);
}

} // namespace

namespace tensorrt_llm::batch_manager
{
LLGuidanceMatcher::LLGuidanceMatcher(LlgMatcherPtr matcher)
    : mMatcher(matcher)
{
}

bool LLGuidanceMatcher::AcceptToken(int32_t tokenId)
{
    if (llg_matcher_consume_token(mMatcher.get(), tokenId) != 0)
    {
        checkMatcher(mMatcher.get());
        return false;
    }
    return true;
}

void LLGuidanceMatcher::FillNextTokenBitmask(DLTensor* nextTokenBitmask)
{
    uint32_t* mask = static_cast<uint32_t*>(nextTokenBitmask->data);
    TLLM_CHECK_WITH_INFO(nextTokenBitmask->ndim == 1, "expected nextTokenBitmask ndim to be 1");
    size_t mask_byte_len = nextTokenBitmask->shape[0] * sizeof(int32_t);

    if (llg_matcher_compute_mask_into(mMatcher.get(), mask, mask_byte_len) != 0)
    {
        checkMatcher(mMatcher.get());
    }
}

LLGuidanceMatcherFactory::LLGuidanceMatcherFactory(
    tle::GuidedDecodingConfig const& guidedDecodingConfig, const SizeType32 vocabSize)
{
    auto const& tokenizerStr = guidedDecodingConfig.getTokenizerStr();
    TLLM_CHECK_WITH_INFO(tokenizerStr, "missing tokenizerStr");

    TLLM_CHECK_WITH_INFO(
        guidedDecodingConfig.getStopTokenIds().value().size() == 1, "expected stopTokenIds size to be 1");
    const int32_t eosId = guidedDecodingConfig.getStopTokenIds().value().back();

    LlgTokenizerInit llgTokenizerInit{.vocab_size = static_cast<uint32_t>(vocabSize),
        .tok_eos = static_cast<uint32_t>(eosId),
        .tokenizer_json = tokenizerStr.value().c_str(),
        .use_approximate_greedy_tokenize_fn = true};

    char errorBuf[1024];
    mTokenizer = std::shared_ptr<LlgTokenizer>(
        llg_new_tokenizer(&llgTokenizerInit, errorBuf, sizeof(errorBuf)), llg_free_tokenizer);
    TLLM_CHECK_WITH_INFO(mTokenizer, "error creating LLGuidance tokenizer: %s", errorBuf);
}

std::shared_ptr<IGrammarMatcher> LLGuidanceMatcherFactory::Create(tle::GuidedDecodingParams const& guidedDecodingParams)
{
    auto const& guideType = guidedDecodingParams.getGuideType();
    auto const& guide = guidedDecodingParams.getGuide();

    LlgMatcherPtr matcher;
    LlgConstraintInit init;
    llg_constraint_init_set_defaults(&init, mTokenizer.get());

    switch (guideType)
    {
    case tle::GuidedDecodingParams::GuideType::kJSON:
    {
        matcher = std::shared_ptr<LlgMatcher>(
            llg_new_matcher(&init, "json_schema", "{\"type\": \"object\"}"), llg_free_matcher);
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kJSON_SCHEMA:
    {
        matcher = std::shared_ptr<LlgMatcher>(
            llg_new_matcher(&init, "json_schema", guide.value().c_str()), llg_free_matcher);
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kREGEX:
    {
        matcher = std::shared_ptr<LlgMatcher>(llg_new_matcher(&init, "regex", guide.value().c_str()), llg_free_matcher);
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kLARK_GRAMMAR:
    {
        matcher = std::shared_ptr<LlgMatcher>(llg_new_matcher(&init, "lark", guide.value().c_str()), llg_free_matcher);
        break;
    }
    case tle::GuidedDecodingParams::GuideType::kEBNF_GRAMMAR:
    {
        TLLM_CHECK_WITH_INFO(false, "kEBNF_GRAMMAR is not supported by the llguidance backend");
    }
    case tle::GuidedDecodingParams::GuideType::kSTRUCTURAL_TAG:
    {
        TLLM_CHECK_WITH_INFO(false, "kSTRUCTURAL_TAG is not supported by the llguidance backend");
    }
    }

    checkMatcher(matcher.get());

    return std::make_shared<LLGuidanceMatcher>(matcher);
}

} // namespace tensorrt_llm::batch_manager
