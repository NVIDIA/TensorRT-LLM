/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "bindings.h"

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/kvCacheConnector.h"
#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/pybind/common/bindTypes.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <ATen/ATen.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>
#include <tuple>

namespace py = pybind11;
namespace tb = tensorrt_llm::batch_manager;
namespace tle = tensorrt_llm::executor;
namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::pybind::batch_manager
{

void initBindings(pybind11::module_& m)
{
    using GenLlmReq = tb::GenericLlmRequest<runtime::ITensor::SharedPtr>;

    // Create and register exceptions in module scope
    static PyObject* peft_exc = PyErr_NewException(
        "tensorrt_llm.bindings.internal.batch_manager.PeftTaskNotCachedException", nullptr, nullptr);
    static PyObject* lora_exc
        = PyErr_NewException("tensorrt_llm.bindings.internal.batch_manager.LoraCacheFullException", nullptr, nullptr);

    m.add_object("PeftTaskNotCachedException", py::handle(peft_exc));
    m.add_object("LoraCacheFullException", py::handle(lora_exc));

    // Register with no captures
    py::register_exception_translator(
        [](std::exception_ptr p)
        {
            try
            {
                if (p)
                    std::rethrow_exception(p);
            }
            catch (const tb::PeftTaskNotCachedException& e)
            {
                PyErr_SetString(peft_exc, e.what());
            }
            catch (const tr::LoraCacheFullException& e)
            {
                PyErr_SetString(lora_exc, e.what());
            }
        });

    PybindUtils::bindSet<tb::ReqIdsSet>(m, "ReqIdsSet");

    py::enum_<tb::LlmRequestType>(m, "LlmRequestType")
        .value("LLMREQUEST_TYPE_CONTEXT_AND_GENERATION", tb::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION)
        .value("LLMREQUEST_TYPE_CONTEXT_ONLY", tb::LLMREQUEST_TYPE_CONTEXT_ONLY)
        .value("LLMREQUEST_TYPE_GENERATION_ONLY", tb::LLMREQUEST_TYPE_GENERATION_ONLY)
        .export_values();

    py::class_<tb::batch_scheduler::ContextChunkingConfig>(m, "ContextChunkingConfig")
        .def(py::init<tle::ContextChunkingPolicy, tensorrt_llm::runtime::SizeType32>(), py::arg("chunking_policy"),
            py::arg("chunk_unit_size"))
        .def_readwrite("chunking_policy", &tb::batch_scheduler::ContextChunkingConfig::chunkingPolicy)
        .def_readwrite("chunk_unit_size", &tb::batch_scheduler::ContextChunkingConfig::chunkUnitSize);

    py::classh<GenLlmReq>(m, "GenericLlmRequest")
        .def("set_exclude_input_from_output", &GenLlmReq::setExcludeInputFromOutput, py::arg("exclude"))
        .def("get_num_tokens", &GenLlmReq::getNumTokens, py::arg("beam"))
        .def_property_readonly("max_beam_num_tokens", &GenLlmReq::getMaxBeamNumTokens)
        .def("get_token", &GenLlmReq::getToken, py::arg("beam"), py::arg("pos"))
        .def("get_tokens", py::overload_cast<GenLlmReq::SizeType32>(&GenLlmReq::getTokens, py::const_), py::arg("beam"))
        .def("get_tokens", py::overload_cast<>(&GenLlmReq::getTokens, py::const_))
        .def("get_last_tokens", py::overload_cast<GenLlmReq::SizeType32>(&GenLlmReq::getLastTokens), py::arg("beam"))
        .def("get_last_tokens", py::overload_cast<>(&GenLlmReq::getLastTokens))
        .def("get_beam_width_by_iter", &GenLlmReq::getBeamWidthByIter, py::arg("for_next_iteration") = false)
        .def_property_readonly("max_num_generated_tokens", &GenLlmReq::getMaxNumGeneratedTokens)
        .def("add_new_token", &GenLlmReq::addNewToken, py::arg("token"), py::arg("beam"))
        .def("add_new_tokens", &GenLlmReq::addNewTokens, py::arg("beam_tokens"))
        .def_property_readonly("num_draft_tokens", &GenLlmReq::getNumDraftTokens)
        .def("set_generated_tokens", &GenLlmReq::setGeneratedTokens, py::arg("generated_beam_tokens"))
        .def("pause", &GenLlmReq::pause, py::arg("max_input_len"))
        .def_property("max_sent_token_len", &GenLlmReq::getMaxSentTokenLen, &GenLlmReq::setMaxSentTokenLen)
        .def_property_readonly("prompt_embedding_table", &GenLlmReq::getPromptEmbeddingTable)
        .def_property_readonly("multimodal_embedding", &GenLlmReq::getMultimodalEmbedding)
        .def_property_readonly("mrope_rotary_cos_sin", &GenLlmReq::getMropeRotaryCosSin)
        .def_property_readonly("bad_words_list", &GenLlmReq::getBadWordsList)
        .def_property("draft_logits", &GenLlmReq::getDraftLogits, &GenLlmReq::setDraftLogits)
        .def_property_readonly("embedding_bias", &GenLlmReq::getEmbeddingBias)
        .def_property("lora_config", &GenLlmReq::getLoraConfig, &GenLlmReq::setLoraConfig)
        .def_property("lora_weights", &GenLlmReq::getLoraWeights, &GenLlmReq::setLoraWeights)
        .def_property_readonly("stop_words_list", &GenLlmReq::getStopWordsList)
        .def_property_readonly("context_logits", &GenLlmReq::getContextLogitsHost)
        .def_property_readonly("generation_logits", &GenLlmReq::getGenerationLogitsHost)
        .def_property_readonly("prompt_vocab_size", &GenLlmReq::getPromptVocabSize)
        .def_property_readonly("mrope_position_deltas", &GenLlmReq::getMropePositionDeltas)
        .def_property_readonly("lora_task_id", &GenLlmReq::getLoraTaskId)
        .def_property_readonly("lookahead_config", &GenLlmReq::getLookaheadConfig)
        .def_property("context_chunk_size", &GenLlmReq::getContextChunkSize, &GenLlmReq::setContextChunkSize)
        .def_property("decoding_iter", &GenLlmReq::getDecodingIter, &GenLlmReq::setDecodingIter)
        .def_readwrite("request_id", &GenLlmReq::mRequestId)
        .def_readwrite("prompt_len", &GenLlmReq::mPromptLen)
        .def_readwrite("max_new_tokens", &GenLlmReq::mMaxNewTokens)
        .def_readwrite("sampling_config", &GenLlmReq::mSamplingConfig)
        .def_property("state", &GenLlmReq::getState, &GenLlmReq::setState)
        .def_property("streaming", &GenLlmReq::isStreaming, &GenLlmReq::setStreaming)
        .def_readwrite("end_id", &GenLlmReq::mEndId)
        .def_readwrite("pad_id", &GenLlmReq::mPadId)
        .def_readwrite("seq_slot", &GenLlmReq::mSeqSlot)
        .def_property_readonly("return_log_probs", &GenLlmReq::returnLogProbs)
        .def_property_readonly("return_context_logits", &GenLlmReq::getReturnContextLogits)
        .def_property_readonly("return_generation_logits", &GenLlmReq::getReturnGenerationLogits)
        .def_property_readonly("log_probs", py::overload_cast<>(&GenLlmReq::getLogProbs, py::const_))
        .def("get_log_probs", py::overload_cast<GenLlmReq::SizeType32>(&GenLlmReq::getLogProbs, py::const_))
        .def("set_log_probs", &GenLlmReq::setLogProbs, py::arg("log_probs"), py::arg("beam"))
        .def("set_return_encoder_output", &GenLlmReq::setReturnEncoderOutput, py::arg("return_encoder_output"))
        .def("get_return_encoder_output", &GenLlmReq::getReturnEncoderOutput)
        .def("priority", py::overload_cast<>(&GenLlmReq::priority, py::const_))
        .def("set_priority", py::overload_cast<tle::PriorityType>(&GenLlmReq::setPriority))
        .def_property_readonly("cum_log_probs", &GenLlmReq::getCumLogProbs)
        .def("set_cum_log_prob", &GenLlmReq::setCumLogProb, py::arg("cum_log_prob"), py::arg("beam"))
        .def("update_num_tokens_per_iteration", &GenLlmReq::updateNumTokensPerIteration,
            py::arg("num_tokens_per_iteration"), py::arg("model_config"))
        .def_property_readonly("orig_prompt_len", &GenLlmReq::getOrigPromptLen)
        .def("has_draft_tokens", &GenLlmReq::hasDraftTokens)
        .def("move_to_next_context_chunk", &GenLlmReq::moveToNextContextChunk)
        .def_property_readonly("is_last_context_chunk", &GenLlmReq::isLastContextChunk)
        .def_property_readonly("is_first_context_chunk", &GenLlmReq::isFirstContextChunk)
        .def_property_readonly("context_remaining_length", &GenLlmReq::getContextRemainingLength)
        .def_property_readonly("context_logits", &GenLlmReq::getContextLogitsHost)
        .def_property_readonly("num_draft_tokens", &GenLlmReq::getNumDraftTokens)
        .def("set_finished_reason", &GenLlmReq::setFinishedReason, py::arg("finish_reason"), py::arg("beam"))
        .def_property_readonly("is_finished", &GenLlmReq::isFinished)
        .def_property_readonly("is_finished_due_to_length", &GenLlmReq::isFinishedDueToLength)
        .def_property(
            "context_current_position", &GenLlmReq::getContextCurrentPosition, &GenLlmReq::setContextCurrentPosition)
        .def_property_readonly("prepopulated_prompt_len", &GenLlmReq::getPrepopulatedPromptLen)
        .def_property(
            "guided_decoding_params", &GenLlmReq::getGuidedDecodingParams, &GenLlmReq::setGuidedDecodingParams)
        .def_property_readonly("context_phase_params", &GenLlmReq::getContextPhaseParams)
        .def_property_readonly("is_context_only_request", &GenLlmReq::isContextOnlyRequest)
        .def_property_readonly("is_generation_only_request", &GenLlmReq::isGenerationOnlyRequest)
        .def_property_readonly("is_generation_complete_state", &GenLlmReq::isGenerationCompleteState)
        .def_property_readonly("is_context_finished", &GenLlmReq::isContextFinished)
        .def_property_readonly("is_disagg_generation_init_state", &GenLlmReq::isDisaggGenerationInitState)
        .def_property_readonly(
            "is_disagg_generation_transmission_complete", &GenLlmReq::isDisaggGenerationTransmissionComplete)
        .def_property_readonly(
            "is_disagg_generation_transmission_in_progress", &GenLlmReq::isDisaggGenerationTransmissionInProgress)
        .def_property_readonly("is_context_init_state", &GenLlmReq::isContextInitState)
        .def_property_readonly("is_generation_in_progress_state", &GenLlmReq::isGenerationInProgressState)
        .def_property_readonly("is_disagg_context_transmission_state", &GenLlmReq::isDisaggContextTransmissionState)
        .def_property_readonly("is_disagg_context_complete_state", &GenLlmReq::isDisaggContextCompleteState)
        .def_property_readonly("stage", &GenLlmReq::getRequestStage)
        .def_property_readonly("kv_cache_transfer_time_ms", &GenLlmReq::getKvCacheTransferTimeMS)
        .def_property_readonly("kv_cache_size", &GenLlmReq::getKvCacheSize)
        .def_property_readonly("avg_decoded_tokens_per_iter", &GenLlmReq::getAvgDecodedTokensPerIter)
        .def_property_readonly("alloc_total_blocks", &GenLlmReq::getAllocTotalBlocksPerRequest)
        .def_property_readonly("alloc_new_blocks", &GenLlmReq::getAllocNewBlocksPerRequest)
        .def("alloc_context_logits", &GenLlmReq::allocContextLogitsHost, py::arg("vocab_size"), py::arg("logit_dtype"))
        .def_property_readonly("reused_blocks", &GenLlmReq::getReusedBlocksPerRequest)
        .def_property_readonly("missed_blocks", &GenLlmReq::getMissedBlocksPerRequest)
        .def_property_readonly("kv_cache_hit_rate", &GenLlmReq::getKVCacheHitRatePerRequest)
        .def_property_readonly("llm_request_type", &GenLlmReq::getLlmRequestType)
        .def_property_readonly("parent_request_id", &GenLlmReq::getParentRequestId)
        .def_property_readonly("is_child", &GenLlmReq::isChild)
        .def_property_readonly("cache_salt_id", &GenLlmReq::getCacheSaltID)
        .def_property_readonly("multimodal_hashes",
            [](GenLlmReq& self)
            {
                std::optional<std::vector<std::vector<GenLlmReq::SizeType32>>> hashes = std::nullopt;
                if (self.getMultimodalHashes())
                {
                    hashes = *self.getMultimodalHashes().value();
                }
                return hashes;
            })
        .def_property_readonly("multimodal_positions",
            [](GenLlmReq& self)
            {
                std::optional<std::vector<GenLlmReq::SizeType32>> positions = std::nullopt;
                if (self.getMultimodalPositions())
                {
                    positions = *self.getMultimodalPositions().value();
                }
                return positions;
            })
        .def_property_readonly("multimodal_lengths",
            [](GenLlmReq& self)
            {
                std::optional<std::vector<GenLlmReq::SizeType32>> lengths = std::nullopt;
                if (self.getMultimodalLengths())
                {
                    lengths = *self.getMultimodalLengths().value();
                }
                return lengths;
            })
        .def_property_readonly("position_ids",
            [](GenLlmReq& self)
            {
                std::optional<std::vector<GenLlmReq::SizeType32>> positionIds = std::nullopt;
                if (self.getPositionIds())
                {
                    positionIds = *self.getPositionIds().value();
                }
                return positionIds;
            })
        .def_property(
            "draft_tokens",
            [](GenLlmReq& self)
            {
                std::optional<GenLlmReq::VecTokens> draftTokens = std::nullopt;
                if (self.hasDraftTokens())
                {
                    draftTokens = *self.getDraftTokens();
                }
                return draftTokens;
            },
            [](GenLlmReq& self, std::optional<GenLlmReq::VecTokens> const& draftTokens)
            {
                if (draftTokens)
                {
                    self.setDraftTokens(std::make_shared<GenLlmReq::VecTokens>(draftTokens.value()));
                }
            })
        .def_property("is_dummy_request", &GenLlmReq::isDummyRequest, &GenLlmReq::setIsDummyRequest)
        .def_property_readonly("return_perf_metrics", &GenLlmReq::getReturnPerfMetrics)
        .def_property("use_draft_model", &GenLlmReq::useDraftModel, &GenLlmReq::setUseDraftModel);

    py::classh<tb::LlmRequest, GenLlmReq>(m, "LlmRequest", pybind11::dynamic_attr())
        .def(py::init<>(
                 [](tb::LlmRequest::RequestIdType request_id, tb::LlmRequest::SizeType32 max_new_tokens,
                     std::vector<tb::LlmRequest::TokenIdType> input_tokens, runtime::SamplingConfig sampling_config,
                     bool is_streaming, std::optional<tb::LlmRequest::SizeType32> end_id,
                     std::optional<tb::LlmRequest::SizeType32> pad_id, std::optional<at::Tensor> embedding_bias,
                     std::optional<at::Tensor> bad_words_list, std::optional<at::Tensor> stop_words_list,
                     std::optional<std::vector<tb::LlmRequest::SizeType32>> position_ids,
                     std::optional<at::Tensor> prompt_embedding_table,
                     std::optional<tb::LlmRequest::SizeType32> prompt_vocab_size,
                     std::optional<std::vector<std::vector<tb::LlmRequest::SizeType32>>> multimodal_hashes,
                     std::optional<std::vector<tb::LlmRequest::SizeType32>> multimodal_positions,
                     std::optional<std::vector<tb::LlmRequest::SizeType32>> multimodal_lengths,
                     std::optional<at::Tensor> multimodal_embedding, std::optional<at::Tensor> mrope_rotary_cos_sin,
                     std::optional<tb::LlmRequest::SizeType32> mrope_position_deltas,
                     std::optional<LoraTaskIdType> lora_task_id, std::optional<at::Tensor> lora_weights,
                     std::optional<at::Tensor> lora_config,
                     std::optional<executor::LookaheadDecodingConfig> lookahead_config,
                     std::optional<executor::KvCacheRetentionConfig> kv_cache_retention_config, bool return_log_probs,
                     bool return_context_logits, bool return_generation_logits,
                     std::optional<tb::LlmRequest::VecTokens> draft_tokens, std::optional<at::Tensor> draft_logits,
                     bool exclude_input_from_output,
                     std::optional<tb::LlmRequest::LogitsPostProcessor> logits_post_processor,
                     bool apply_logits_post_processor_batched,
                     std::optional<tb::LlmRequest::VecTokens> encoder_input_tokens, bool return_encoder_output,
                     std::optional<tb::LlmRequest::RequestIdType> client_id, executor::PriorityType priority,
                     std::optional<at::Tensor> encoder_input_features,
                     std::optional<tb::LlmRequest::SizeType32> encoder_output_length,
                     std::optional<at::Tensor> cross_attention_mask, tb::LlmRequestType llm_request_type,
                     std::optional<tb::LlmRequest::VecTokenExtraIds> input_token_extra_ids,
                     tb::LlmRequest::SizeType32 num_return_sequences, std::optional<executor::EagleConfig> eagle_config,
                     std::optional<at::Tensor> skip_cross_attn_blocks, bool return_perf_metrics,
                     std::optional<executor::GuidedDecodingParams> guided_decoding_params,
                     std::optional<tb::LlmRequest::SizeType32> language_adapter_uid,
                     std::optional<tb::LlmRequest::MillisecondsType> allotted_time_ms,
                     std::optional<executor::ContextPhaseParams> context_phase_params,
                     std::optional<tb::LlmRequest::CacheSaltIDType> cache_salt_id,
                     std::optional<tb::LlmRequest::TimePoint> arrival_time)
                 {
                     auto makeOptionalTensor = [](std::optional<at::Tensor> const& atTensor, bool unsqueeze = false)
                     {
                         std::optional<tb::LlmRequest::TensorPtr> tensorPtr = std::nullopt;
                         if (atTensor)
                         {
                             tensorPtr = tr::TorchView::of(atTensor.value());
                             if (unsqueeze)
                             {
                                 (*tensorPtr)->unsqueeze(0);
                             }
                         }
                         return tensorPtr;
                     };

                     auto embedding_bias_tensor_ptr = makeOptionalTensor(embedding_bias, true);
                     auto bad_words_list_tensor_ptr = makeOptionalTensor(bad_words_list, true);
                     auto stop_words_list_tensor_ptr = makeOptionalTensor(stop_words_list, true);
                     auto prompt_embedding_table_tensor_ptr = makeOptionalTensor(prompt_embedding_table);
                     auto multimodal_embedding_tensor_ptr = makeOptionalTensor(multimodal_embedding);
                     auto lora_weights_tensor_ptr = makeOptionalTensor(lora_weights);
                     auto mrope_rotary_cos_sin_tensor_ptr = makeOptionalTensor(mrope_rotary_cos_sin);
                     auto lora_config_tensor_ptr = makeOptionalTensor(lora_config);
                     auto draft_logits_tensor_ptr = makeOptionalTensor(draft_logits);
                     auto encoder_input_features_tensor_ptr = makeOptionalTensor(encoder_input_features);
                     auto cross_attention_mask_tensor_ptr = makeOptionalTensor(cross_attention_mask);
                     auto skip_cross_attn_blocks_tensor_ptr = makeOptionalTensor(skip_cross_attn_blocks);

                     return tb::LlmRequest{request_id, max_new_tokens, input_tokens, sampling_config, is_streaming,
                         end_id, pad_id, embedding_bias_tensor_ptr, bad_words_list_tensor_ptr,
                         stop_words_list_tensor_ptr, position_ids, prompt_embedding_table_tensor_ptr, prompt_vocab_size,
                         multimodal_hashes, multimodal_positions, multimodal_lengths, multimodal_embedding_tensor_ptr,
                         mrope_rotary_cos_sin_tensor_ptr, mrope_position_deltas, lora_task_id, lora_weights_tensor_ptr,
                         lora_config_tensor_ptr, lookahead_config, kv_cache_retention_config, return_log_probs,
                         return_context_logits, return_generation_logits, draft_tokens, draft_logits_tensor_ptr,
                         exclude_input_from_output, logits_post_processor, apply_logits_post_processor_batched,
                         encoder_input_tokens, return_encoder_output, client_id, priority,
                         encoder_input_features_tensor_ptr, encoder_output_length, cross_attention_mask_tensor_ptr,
                         llm_request_type, input_token_extra_ids, num_return_sequences, eagle_config,
                         skip_cross_attn_blocks_tensor_ptr, return_perf_metrics, guided_decoding_params,
                         language_adapter_uid, allotted_time_ms, context_phase_params, cache_salt_id, arrival_time};
                 }),
            py::arg("request_id"), py::arg("max_new_tokens"), py::arg("input_tokens"), py::arg("sampling_config"),
            py::arg("is_streaming"), py::arg("end_id") = std::nullopt, py::arg("pad_id") = std::nullopt,
            py::arg("embedding_bias") = std::nullopt, py::arg("bad_words_list") = std::nullopt,
            py::arg("stop_words_list") = std::nullopt, py::arg("position_ids") = std::nullopt,
            py::arg("prompt_embedding_table") = std::nullopt, py::arg("prompt_vocab_size") = std::nullopt,
            py::arg("multimodal_hashes") = std::nullopt, py::arg("multimodal_positions") = std::nullopt,
            py::arg("multimodal_lengths") = std::nullopt, py::arg("multimodal_embedding") = std::nullopt,
            py::arg("mrope_rotary_cos_sin") = std::nullopt, py::arg("mrope_position_deltas") = std::nullopt,
            py::arg("lora_task_id") = std::nullopt, py::arg("lora_weights") = std::nullopt,
            py::arg("lora_config") = std::nullopt, py::arg("lookahead_config") = std::nullopt,
            py::arg("kv_cache_retention_config") = std::nullopt, py::arg("return_log_probs") = false,
            py::arg("return_context_logits") = false, py::arg("return_generation_logits") = false,
            py::arg("draft_tokens") = std::nullopt, py::arg("draft_logits") = std::nullopt,
            py::arg("exclude_input_from_output") = false, py::arg("logits_post_processor") = std::nullopt,
            py::arg("apply_logits_post_processor_batched") = false, py::arg("encoder_input_tokens") = std::nullopt,
            py::arg("return_encoder_output") = false, py::arg("client_id") = std::nullopt,
            py::arg("priority") = executor::Request::kDefaultPriority, py::arg("encoder_input_features") = std::nullopt,
            py::arg("encoder_output_len") = std::nullopt, py::arg("cross_attention_mask") = std::nullopt,
            py::arg_v("llm_request_type", tb::LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
                "LlmRequestType.LLMREQUEST_TYPE_CONTEXT_AND_GENERATION"),
            py::arg("input_token_extra_ids") = std::nullopt, py::arg("num_return_sequences") = 1,
            py::arg("eagle_config") = std::nullopt, py::arg("skip_cross_attn_blocks") = std::nullopt,
            py::arg("return_perf_metrics") = false, py::arg("guided_decoding_params") = std::nullopt,
            py::arg("language_adapter_uid") = std::nullopt, py::arg("allotted_time_ms") = std::nullopt,
            py::arg("context_phase_params") = std::nullopt, py::arg("cache_salt_id") = std::nullopt,
            py::arg("arrival_time") = std::nullopt)
        .def("check_token_id_range", &tb::LlmRequest::checkTokenIdRange, py::arg("vocab_size"))
        .def(py::init<tb::LlmRequest const&>())
        .def("validate", &tb::LlmRequest::validate, py::arg("max_input_len"), py::arg("max_seq_len"),
            py::arg("max_draft_len"), py::arg("vocab_size_padded"), py::arg("max_endocer_input_len") = std::nullopt,
            py::arg("enable_kv_cache_reuse") = false)
        .def("create_response", &tb::LlmRequest::createResponse, py::arg("use_fast_logits") = false,
            py::arg("mpi_world_rank") = 0)
        .def("create_child_request", &tb::LlmRequest::createChildRequest, py::arg("child_id"))
        .def("create_result", &tb::LlmRequest::createResult, py::arg("use_fast_logits") = false,
            py::arg("mpi_world_rank") = 0)
        .def("create_serialized_result",
            [](tb::LlmRequest& self, bool use_fast_logits = false, int mpi_world_rank = 0)
            {
                std::vector<char> serialized_result;
                bool is_final = false;
                self.createSerializedResult(serialized_result, is_final, use_fast_logits, mpi_world_rank);
                return std::make_tuple(py::bytes(serialized_result.data(), serialized_result.size()), is_final);
            })
        .def("move_prompt_embedding_table_to_gpu", &tb::LlmRequest::movePromptEmbeddingTableToGpu, py::arg("manager"))
        .def("move_lora_weights_to_gpu", &tb::LlmRequest::moveLoraWeightsToGpu, py::arg("manager"))
        .def("finish_by_reason", &tb::LlmRequest::finishByReason, py::arg("finish_reason"))
        .def("set_first_scheduled_time", &tb::LlmRequest::setFirstScheduledTime)
        .def("update_perf_metrics", &tb::LlmRequest::updatePerfMetrics, py::arg("iter_counter"))
        .def("remove_lora_tensors", &tb::LlmRequest::removeLoraTensors)
        .def_readwrite_static("global_steady_clock_offset", &tb::LlmRequest::mGlobalSteadyClockOffset);

    py::classh<tb::SequenceSlotManager>(m, "SequenceSlotManager")
        .def(py::init<tb::SequenceSlotManager::SlotIdType, uint64_t>(), py::arg("max_num_slots"),
            py::arg("max_sequence_idle_microseconds"))
        .def("get_sequence_slot", &tb::SequenceSlotManager::getSequenceSlot, py::arg("start_flag"),
            py::arg("sequence_id"))
        .def("free_sequence_slot", &tb::SequenceSlotManager::freeSequenceSlot, py::arg("sequence_id"))
        .def("free_idle_sequence_slots", &tb::SequenceSlotManager::freeIdleSequenceSlots);

    py::classh<tb::rnn_state_manager::RnnStateManager>(m, "RnnStateManager")
        .def(py::init<tr::SizeType32, tr::ModelConfig, tr::WorldConfig, tr::BufferManager>(),
            py::arg("max_num_sequences"), py::arg("model_config"), py::arg("world_config"), py::arg("buffer_manager"));

    py::class_<tb::DecoderInputBuffers>(m, "DecoderInputBuffers")
        .def(py::init<runtime::SizeType32, runtime::SizeType32, tr::BufferManager>(), py::arg("max_batch_size"),
            py::arg("max_tokens_per_engine_step"), py::arg("manager"))
        .def_readwrite("setup_batch_slots", &tb::DecoderInputBuffers::setupBatchSlots)
        .def_readwrite("setup_batch_slots_device", &tb::DecoderInputBuffers::setupBatchSlotsDevice)
        .def_readwrite("fill_values", &tb::DecoderInputBuffers::fillValues)
        .def_readwrite("fill_values_device", &tb::DecoderInputBuffers::fillValuesDevice)
        .def_readwrite("inputs_ids", &tb::DecoderInputBuffers::inputsIds)
        .def_readwrite("forward_batch_slots", &tb::DecoderInputBuffers::forwardBatchSlots)
        .def_readwrite("logits", &tb::DecoderInputBuffers::logits)
        .def_readwrite("decoder_requests", &tb::DecoderInputBuffers::decoderRequests);

    py::class_<tb::DecoderOutputBuffers>(m, "DecoderOutputBuffers")
        .def_readwrite("sequence_lengths_host", &tb::DecoderOutputBuffers::sequenceLengthsHost)
        .def_readwrite("finished_sum_host", &tb::DecoderOutputBuffers::finishedSumHost)
        .def_property_readonly("new_output_tokens_host",
            [](tb::DecoderOutputBuffers& self) { return tr::Torch::tensor(self.newOutputTokensHost); })
        .def_readwrite("cum_log_probs_host", &tb::DecoderOutputBuffers::cumLogProbsHost)
        .def_readwrite("log_probs_host", &tb::DecoderOutputBuffers::logProbsHost)
        .def_readwrite("finish_reasons_host", &tb::DecoderOutputBuffers::finishReasonsHost);

    py::class_<tb::SlotDecoderBuffers>(m, "SlotDecoderBuffers")
        .def(py::init<runtime::SizeType32, runtime::SizeType32, runtime::BufferManager const&>(),
            py::arg("max_beam_width"), py::arg("max_seq_len"), py::arg("buffer_manager"))
        .def_readwrite("output_ids", &tb::SlotDecoderBuffers::outputIds)
        .def_readwrite("output_ids_host", &tb::SlotDecoderBuffers::outputIdsHost)
        .def_readwrite("sequence_lengths_host", &tb::SlotDecoderBuffers::sequenceLengthsHost)
        .def_readwrite("cum_log_probs", &tb::SlotDecoderBuffers::cumLogProbs)
        .def_readwrite("cum_log_probs_host", &tb::SlotDecoderBuffers::cumLogProbsHost)
        .def_readwrite("log_probs", &tb::SlotDecoderBuffers::logProbs)
        .def_readwrite("log_probs_host", &tb::SlotDecoderBuffers::logProbsHost)
        .def_readwrite("finish_reasons_host", &tb::SlotDecoderBuffers::finishReasonsHost);

    m.def(
        "add_new_tokens_to_requests",
        [](std::vector<std::shared_ptr<tb::LlmRequest>>& requests,
            std::vector<tb::LlmRequest::TokenIdType> const& tokens, int beam_idx)
        {
            TLLM_CHECK_WITH_INFO(requests.size() == tokens.size(), "Expected the same number of requests and tokens.");

            for (int i = 0; i < requests.size(); ++i)
            {
                requests[i]->addNewToken(tokens[i], beam_idx);
            }
        },
        py::arg("requests"), py::arg("tokens"), py::arg("beam_idx"),
        "Add new tokens to multiple LLM requests. The tokens vector should contain tokens for beam beam_idx of all "
        "requests in order.");

    m.def(
        "make_decoding_batch_input",
        [](std::vector<std::shared_ptr<tb::LlmRequest>>& contextRequests,
            std::vector<std::shared_ptr<tb::LlmRequest>>& genRequests, tr::ITensor::SharedPtr logits, int beamWidth,
            std::vector<int> const& numContextLogitsPrefixSum, tb::DecoderInputBuffers const& decoderInputBuffers,
            runtime::decoder::DecoderState& decoderState, tr::BufferManager const& manager)
        {
            std::vector<int> activeSlots;
            std::vector<int> generationSteps;
            std::vector<std::vector<tr::ITensor::SharedConstPtr>> logitsVec = {{}};

            for (int i = 0; i < contextRequests.size(); ++i)
            {
                if (contextRequests[i]->isLastContextChunk())
                {
                    activeSlots.push_back(*contextRequests[i]->mSeqSlot);
                    generationSteps.push_back(contextRequests[i]->getDecodingIter());
                    auto contextLogitsOffset = numContextLogitsPrefixSum[i + 1] - 1;
                    tr::ITensor::SharedPtr logitsView = ITensor::slice(logits, contextLogitsOffset, 1);

                    if (beamWidth > 1)
                    {
                        // Tile logits of context requests
                        auto const logitsShape = logitsView->getShape();
                        auto const logitsType = logitsView->getDataType();
                        auto decoderLogits = manager.gpu(ITensor::makeShape({beamWidth, logitsShape.d[1]}), logitsType);
                        tensorrt_llm::runtime::kernels::tileTensor(
                            *decoderLogits, *logitsView, beamWidth, manager.getStream());
                        decoderLogits->unsqueeze(0);
                        logitsVec[0].push_back(std::move(decoderLogits));
                    }
                    else
                    {
                        logitsView->unsqueeze(1);
                        logitsVec[0].push_back(std::move(logitsView));
                    }
                }
            }

            auto genLogitsOffset = numContextLogitsPrefixSum.back();
            for (int i = 0; i < genRequests.size(); ++i)
            {
                if (genRequests[i]->isGenerationInProgressState())
                {
                    activeSlots.push_back(*genRequests[i]->mSeqSlot);
                    generationSteps.push_back(genRequests[i]->getDecodingIter());

                    auto logitsOffset = genLogitsOffset + i * beamWidth;
                    auto numberOfLogits = beamWidth;
                    tr::ITensor::SharedPtr logitsView = ITensor::slice(logits, logitsOffset, numberOfLogits);
                    logitsView->unsqueeze(0);
                    logitsVec[0].push_back(std::move(logitsView));
                }
            }

            auto& batchSlots = decoderInputBuffers.forwardBatchSlots;
            batchSlots[0]->resize(activeSlots.size());
            auto batchSlotsRange = tr::BufferRange<SizeType32>(*batchSlots[0]);
            for (int i = 0; i < activeSlots.size(); ++i)
            {
                batchSlotsRange[i] = activeSlots[i];
            }

            auto decodingInput = std::make_unique<tr::decoder_batch::Input>(logitsVec, 1);
            decodingInput->batchSlots = batchSlots;

            auto const maxBeamWidth = decoderState.getMaxBeamWidth();
            if (maxBeamWidth > 1)
            {
                // For Variable-Beam-Width-Search
                decoderState.getJointDecodingInput().generationSteps = generationSteps;
            }

            return decodingInput;
        },
        py::arg("context_requests"), py::arg("generation_requests"), py::arg("logits"), py::arg("beam_width"),
        py::arg("num_context_logits_prefix_sum"), py::arg("decoder_input_buffers"), py::arg("decoder_state"),
        py::arg("buffer_manager"), "Make decoding batch input.");
}

} // namespace tensorrt_llm::pybind::batch_manager
