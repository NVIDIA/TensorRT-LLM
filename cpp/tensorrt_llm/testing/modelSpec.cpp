/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "modelSpec.h"
#include "tensorrt_llm/common/dataType.h"

#include <numeric>

namespace tensorrt_llm::testing
{

std::string ModelSpec::getQuantMethodString() const
{
    switch (mQuantMethod)
    {
    case QuantMethod::kNONE:
        // Bypass here.
        break;
    case QuantMethod::kSMOOTH_QUANT: return "sq"; break;
    default: throw std::runtime_error("Unsupported quant method"); break;
    }

    return "";
}

std::string ModelSpec::getKVCacheTypeString() const
{
    switch (mKVCacheType)
    {
    case KVCacheType::kDISABLED: return "no-cache"; break;
    case KVCacheType::kPAGED: return "paged"; break;
    case KVCacheType::kCONTINUOUS: return "continuous"; break;
    default: throw std::runtime_error("Unsupported KV cache type"); break;
    }

    return "";
}

std::string ModelSpec::getSpeculativeDecodingModeString() const
{
    if (mSpecDecodingMode.isLookaheadDecoding())
    {
        return "la-decoding";
    }
    else if (mSpecDecodingMode.isDraftTokensExternal())
    {
        return "draft-tokens";
    }
    else if (mSpecDecodingMode.isNone())
    {
        // Bypass here.
    }
    else if (mSpecDecodingMode.isExplicitDraftTokens())
    {
        return "explicit-draft-tokens";
    }
    else if (mSpecDecodingMode.isMedusa())
    {
        return "medusa";
    }
    else if (mSpecDecodingMode.isEagle())
    {
        return "eagle";
    }
    else
    {
        throw std::runtime_error("Unsupported decoding mode");
    }

    return "";
}

std::string ModelSpec::getCapacitySchedulerString() const
{
    if (mCapacitySchedulerPolicy)
    {
        if (mCapacitySchedulerPolicy.value() == tensorrt_llm::executor::CapacitySchedulerPolicy::kMAX_UTILIZATION)
        {
            return "MaxUtilization";
        }
        else if (mCapacitySchedulerPolicy.value()
            == tensorrt_llm::executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
        {
            return "GuaranteedNoEvict";
        }
        else if (mCapacitySchedulerPolicy.value() == tensorrt_llm::executor::CapacitySchedulerPolicy::kSTATIC_BATCH)
        {
            return "StaticBatch";
        }
        else
        {
            throw std::runtime_error("Unsupported capacity scheduler");
        }
    }
    return "";
}

std::string ModelSpec::getInputFile() const
{
    return mInputFile;
}

std::string ModelSpec::getModelPath() const
{
    std::vector<std::string> ret;

    ret.emplace_back(getDtypeString());

    if (mUseGptAttentionPlugin || mUseMambaPlugin)
    {
        if (mUseGptAttentionPlugin && mUseMambaPlugin)
        {
            throw std::runtime_error("Cannot use both GPT attention plugin and MAMBA plugin");
        }

        ret.emplace_back("plugin");
    }
    else
    {
        ret.emplace_back("default");
    }

    if (mUsePackedInput)
    {
        ret.emplace_back("packed");
    }

    ret.emplace_back(getKVCacheTypeString());

    if (mMaxInputLength)
    {
        ret.emplace_back("in" + std::to_string(mMaxInputLength));
    }

    ret.emplace_back(getSpeculativeDecodingModeString());

    if (mUseLoraPlugin)
    {
        ret.emplace_back("lora");
    }

    ret.emplace_back(getQuantMethodString());

    if (mUseMultipleProfiles)
    {
        ret.emplace_back("nprofiles");
    }

    if (mGatherLogits)
    {
        ret.emplace_back("gather");
    }

    auto finalRet = std::accumulate(ret.begin(), ret.end(), std::string(),
        [](std::string& a, std::string& b)
        {
            if (a.empty())
            {
                return b;
            }
            else
            {
                return b.empty() ? a : a + "_" + b;
            }
        });

    return finalRet;
}

std::string ModelSpec::getResultsFileInternal(OutputContentType outputContentType) const
{
    std::vector<std::string> ret;

    if (mInputFile == "input_tokens_long.npy")
    {
        ret.emplace_back("output_tokens_long");
    }
    else
    {
        ret.emplace_back("output_tokens");
    }

    if (mMaxOutputLength)
    {
        ret.emplace_back("out" + std::to_string(mMaxOutputLength));
    }

    ret.emplace_back(getDtypeString());

    if (mUseGptAttentionPlugin || mUseMambaPlugin)
    {
        if (mUseGptAttentionPlugin && mUseMambaPlugin)
        {
            throw std::runtime_error("Cannot use both GPT attention plugin and MAMBA plugin");
        }
        ret.emplace_back("plugin");
    }

    if (mUsePackedInput)
    {
        ret.emplace_back("packed");
    }

    ret.emplace_back(getKVCacheTypeString());

    ret.emplace_back(getQuantMethodString());

    if (mGatherLogits)
    {
        ret.emplace_back("gather");
    }

    ret.emplace_back("tp" + std::to_string(mTPSize));

    ret.emplace_back("pp" + std::to_string(mPPSize));

    ret.emplace_back("cp" + std::to_string(mCPSize));

    if (mEnableContextFMHAFp32Acc)
    {
        ret.emplace_back("fmhafp32acc");
    }

    switch (outputContentType)
    {
    case OutputContentType::kNONE:
        // Bypass here.
        break;
    case OutputContentType::kCONTEXT_LOGITS: ret.emplace_back("logits_context"); break;
    case OutputContentType::kGENERATION_LOGITS: ret.emplace_back("logits_generation"); break;
    case OutputContentType::kLOG_PROBS: ret.emplace_back("log_probs"); break;
    case OutputContentType::kCUM_LOG_PROBS: ret.emplace_back("cum_log_probs"); break;
    default: throw std::runtime_error("Unsupported output content type"); break;
    }

    auto finalRet = std::accumulate(ret.begin(), ret.end(), std::string(),
        [](std::string& a, std::string& b)
        {
            if (a.empty())
            {
                return b;
            }
            else
            {
                return b.empty() ? a : a + "_" + b;
            }
        });
    return finalRet + ".npy";
}

std::string ModelSpec::getResultsFile() const
{
    return mOtherModelSpecToCompare ? mOtherModelSpecToCompare->getResultsFileInternal(OutputContentType::kNONE)
                                    : getResultsFileInternal(OutputContentType::kNONE);
}

std::string ModelSpec::getGenerationLogitsFile() const
{
    return mOtherModelSpecToCompare
        ? mOtherModelSpecToCompare->getResultsFileInternal(OutputContentType::kGENERATION_LOGITS)
        : getResultsFileInternal(OutputContentType::kGENERATION_LOGITS);
}

std::string ModelSpec::getContextLogitsFile() const
{
    return mOtherModelSpecToCompare
        ? mOtherModelSpecToCompare->getResultsFileInternal(OutputContentType::kCONTEXT_LOGITS)
        : getResultsFileInternal(OutputContentType::kCONTEXT_LOGITS);
}

std::string ModelSpec::getCumLogProbsFile() const
{
    return mOtherModelSpecToCompare
        ? mOtherModelSpecToCompare->getResultsFileInternal(OutputContentType::kCUM_LOG_PROBS)
        : getResultsFileInternal(OutputContentType::kCUM_LOG_PROBS);
}

std::string ModelSpec::getLogProbsFile() const
{
    return mOtherModelSpecToCompare ? mOtherModelSpecToCompare->getResultsFileInternal(OutputContentType::kLOG_PROBS)
                                    : getResultsFileInternal(OutputContentType::kLOG_PROBS);
}

std::string ModelSpec::getDtypeString() const
{
    return tensorrt_llm::common::getDtypeString(mDataType);
}

} // namespace tensorrt_llm::testing
