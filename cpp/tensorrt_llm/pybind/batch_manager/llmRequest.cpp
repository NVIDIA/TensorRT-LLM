/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "llmRequest.h"

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <memory>

namespace tb = tensorrt_llm::batch_manager;
namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::pybind::batch_manager;

namespace
{

std::optional<tb::LlmRequest::TensorPtr> from_torch(std::optional<LlmRequest::TensorPtr> torchPtr)
{
    if (torchPtr)
    {
        return tr::TorchView::of(torchPtr.value());
    }
    return std::nullopt;
}

} // namespace

std::optional<tb::LlmRequest::LogitsPostProcessor> LlmRequest::callbackAdapter(
    std::optional<LlmRequest::LogitsPostProcessor> callback)
{
    if (!callback)
    {
        return std::nullopt;
    }

    return [callback](RequestIdType reqId, tensorrt_llm::runtime::ITensor::SharedPtr& tensor,
               tensorrt_llm::batch_manager::LlmRequest::BeamTokens const& tokens,
               tensorrt_llm::runtime::BufferManager::CudaStreamPtr stream, std::optional<RequestIdType> clientId)
    {
        at::Tensor atTensor = tr::Torch::tensor(tensor);
        callback.value()(reqId, atTensor, tokens, runtime::TorchUtils::stream(*stream).unwrap(), clientId);
    };
}

std::shared_ptr<tb::LlmRequest> LlmRequest::toTrtLlm() const
{
    auto embeddingBias = from_torch(mEmbeddingBias);
    auto badWordsList = from_torch(mBadWordsList);
    auto stopWordsList = from_torch(mStopWordsList);
    auto promptEmbeddingTable = from_torch(mPromptEmbeddingTable);
    auto loraWeights = from_torch(mLoraWeights);
    auto loraConfig = from_torch(mLoraConfig);
    auto draftLogits = from_torch(mDraftLogits);
    auto encoderInputFeatures = from_torch(mEncoderInputFeatures);

    return std::make_shared<tb::LlmRequest>(mRequestId, mMaxNewTokens,
        std::make_shared<std::vector<TokenIdType>>(mTokens.at(0)), mSamplingConfig, mIsStreaming, mEndId, mPadId,
        embeddingBias, badWordsList, stopWordsList, mPositionIds, promptEmbeddingTable, mPromptVocabSize, mLoraTaskId,
        loraWeights, loraConfig, mLookaheadConfig, returnLogProbs(), mReturnContextLogits, mReturnGenerationLogits,
        mDraftTokens, draftLogits, mExcludeInputFromOutput, callbackAdapter(mLogitsPostProcessor),
        mApplyLogitsPostProcessorBatched, mEncoderTokens, mReturnEncoderOutput, mClientId, mPriority,
        encoderInputFeatures, mEncoderOutputLength, tb::LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        mInputTokenExtraIds, mNumReturnSequences);
}

void LlmRequest::initBindings(py::module_& m)
{
    py::class_<LlmRequest>(m, "LlmRequest")
        .def(py::init<LlmRequest::RequestIdType, LlmRequest::SizeType32, LlmRequest::VecTokens, tr::SamplingConfig,
                 bool, std::optional<LlmRequest::SizeType32>, std::optional<LlmRequest::SizeType32>,
                 std::optional<LlmRequest::TensorPtr>, std::optional<LlmRequest::TensorPtr>,
                 std::optional<LlmRequest::TensorPtr>, std::optional<std::vector<LlmRequest::SizeType32>>,
                 std::optional<LlmRequest::TensorPtr>, std::optional<LlmRequest::SizeType32>, std::optional<uint64_t>,
                 std::optional<LlmRequest::TensorPtr>, std::optional<LlmRequest::TensorPtr>,
                 std::optional<executor::LookaheadDecodingConfig>, bool, bool, bool,
                 std::optional<LlmRequest::VecTokens>, std::optional<LlmRequest::TensorPtr>, bool,
                 std::optional<LlmRequest::LogitsPostProcessor>, bool, std::optional<LlmRequest::VecTokens>, bool,
                 std::optional<RequestIdType>, executor::PriorityType, std::optional<LlmRequest::TensorPtr>,
                 std::optional<LlmRequest::SizeType32>, std::optional<LlmRequest::VecTokenExtraIds>,
                 LlmRequest::SizeType32>(),
            py::arg("request_id"), py::arg("max_new_tokens"), py::arg("input_tokens"), py::arg("sampling_config"),
            py::arg("is_streaming"), py::arg("end_id") = std::nullopt, py::arg("pad_id") = std::nullopt,
            py::arg("embedding_bias") = std::nullopt, py::arg("bad_words_list") = std::nullopt,
            py::arg("stop_words_list") = std::nullopt, py::arg("position_ids") = std::nullopt,
            py::arg("prompt_embedding_table") = std::nullopt, py::arg("prompt_vocab_size") = std::nullopt,
            py::arg("lora_task_id") = std::nullopt, py::arg("lora_weights") = std::nullopt,
            py::arg("lora_config") = std::nullopt, py::arg("lookahead_config") = std::nullopt,
            py::arg("return_log_probs") = false, py::arg("return_context_logits") = false,
            py::arg("return_generation_logits") = false, py::arg("draft_tokens") = std::nullopt,
            py::arg("draft_logits") = std::nullopt, py::arg("exclude_input_from_output") = false,
            py::arg("logits_post_processor") = std::nullopt, py::arg("apply_logits_post_processor_batched") = false,
            py::arg("encoder_input_tokens") = std::nullopt, py::arg("return_encoder_output") = false,
            py::arg("client_id") = std::nullopt, py::arg("priority") = executor::Request::kDefaultPriority,
            py::arg("encoder_input_features") = std::nullopt, py::arg("encoder_output_length") = std::nullopt,
            py::arg("input_token_extra_ids") = std::nullopt, py::arg("num_return_sequences") = 1)
        .def("get_num_tokens", &LlmRequest::getNumTokens, py::arg("beam"))
        .def_property_readonly("max_beam_num_tokens", &LlmRequest::getMaxBeamNumTokens)
        .def("get_token", &LlmRequest::getToken, py::arg("beam"), py::arg("pos"))
        .def("get_tokens", py::overload_cast<LlmRequest::SizeType32>(&LlmRequest::getTokens, py::const_),
            py::arg("beam"))
        .def("get_tokens", py::overload_cast<>(&LlmRequest::getTokens, py::const_))
        .def_property_readonly("max_num_generated_tokens", &LlmRequest::getMaxNumGeneratedTokens)
        .def("add_new_token", &LlmRequest::addNewToken, py::arg("token"), py::arg("beam"))
        .def("add_new_tokens", &LlmRequest::addNewTokens, py::arg("beam_tokens"))
        .def("set_generated_tokens", &LlmRequest::setGeneratedTokens, py::arg("generated_beam_tokens"))
        .def("pause", &LlmRequest::pause, py::arg("max_input_len"))
        .def_property("max_sent_token_len", &LlmRequest::getMaxSentTokenLen, &LlmRequest::setMaxSentTokenLen)
        .def_property_readonly("prompt_embedding_table", &LlmRequest::getPromptEmbeddingTable)
        .def_property_readonly("prompt_vocab_size", &LlmRequest::getPromptVocabSize)
        .def_property_readonly("lora_task_id", &LlmRequest::getLoraTaskId)
        .def_property_readonly("lora_weights", &LlmRequest::getLoraWeights)
        .def_property_readonly("lora_config", &LlmRequest::getLoraConfig)
        .def_property_readonly("lookahead_config", &LlmRequest::getLookaheadConfig)
        .def_property_readonly("embedding_bias", &LlmRequest::getEmbeddingBias)
        .def_property_readonly("bad_words_list", &LlmRequest::getBadWordsList)
        .def_property_readonly("stop_words_list", &LlmRequest::getStopWordsList)
        .def_property_readonly("position_ids",
            [](LlmRequest& self)
            { return *self.getPositionIds().value_or(std::make_shared<std::vector<SizeType32>>()); })
        .def_property_readonly(
            "context_current_position", py::overload_cast<>(&LlmRequest::getContextCurrentPosition, py::const_))
        .def_property("context_chunk_size", &LlmRequest::getContextChunkSize, &LlmRequest::setContextChunkSize)
        .def_readwrite("request_id", &LlmRequest::mRequestId)
        .def_readwrite("prompt_len", &LlmRequest::mPromptLen)
        .def_readwrite("max_new_tokens", &LlmRequest::mMaxNewTokens)
        .def_readwrite("sampling_config", &LlmRequest::mSamplingConfig)
        .def_readwrite("state", &LlmRequest::mState)
        .def_readwrite("is_streaming", &LlmRequest::mIsStreaming)
        .def_readwrite("end_id", &LlmRequest::mEndId)
        .def_readwrite("pad_id", &LlmRequest::mPadId)
        .def_readwrite("seq_slot", &LlmRequest::mSeqSlot)
        .def_property_readonly("return_log_probs", &LlmRequest::returnLogProbs)
        .def_property_readonly("return_context_logits", &LlmRequest::setReturnContextLogits)
        .def_property_readonly("return_generation_logits", &LlmRequest::setReturnGenerationLogits)
        .def_property_readonly("log_probs", py::overload_cast<>(&LlmRequest::getLogProbs, py::const_))
        .def("get_log_probs", py::overload_cast<SizeType32>(&LlmRequest::getLogProbs, py::const_))
        .def("set_log_probs", &LlmRequest::setLogProbs, py::arg("log_probs"), py::arg("beam"))
        .def("set_return_encoder_output", &LlmRequest::setReturnEncoderOutput, py::arg("return_encoder_output"))
        .def("get_return_encoder_output", &LlmRequest::getReturnEncoderOutput)
        .def("priority", py::overload_cast<>(&LlmRequest::priority, py::const_))
        .def("set_priority", py::overload_cast<executor::PriorityType>(&LlmRequest::setPriority))
        .def_property_readonly("cum_log_probs", &LlmRequest::getCumLogProbs)
        .def("set_cum_log_prob", &LlmRequest::setCumLogProb, py::arg("cum_log_prob"), py::arg("beam"))
        .def_property_readonly("orig_prompt_len", &LlmRequest::getOrigPromptLen)
        .def("has_draft_tokens", &LlmRequest::hasDraftTokens)
        .def("move_to_next_context_chunk", &LlmRequest::moveToNextContextChunk)
        .def("is_full_context_request", py::overload_cast<>(&LlmRequest::isFullContextRequest, py::const_))
        .def("is_last_context_chunk", py::overload_cast<>(&LlmRequest::isLastContextChunk, py::const_))
        .def("is_first_context_chunk", py::overload_cast<>(&LlmRequest::isFirstContextChunk, py::const_))
        .def("get_context_remaining_length", py::overload_cast<>(&LlmRequest::getContextRemainingLength, py::const_))
        .def_property(
            "draft_tokens", [](LlmRequest& self) { return *self.getDraftTokens(); },
            [](LlmRequest& self, LlmRequest::VecTokens& draftTokens)
            { self.setDraftTokens(std::make_shared<LlmRequest::VecTokens>(std::move(draftTokens))); })
        .def_property(
            "draft_logits", [](LlmRequest& self) { return self.getDraftLogits(); },
            [](LlmRequest& self, LlmRequest::TensorPtr& logits)
            { self.setDraftLogits(std::make_optional<LlmRequest::TensorPtr>(logits)); })
        .def_property("num_return_sequences", &LlmRequest::getNumReturnSequences, &LlmRequest::setNumReturnSequences);
}
