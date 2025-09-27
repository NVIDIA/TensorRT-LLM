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
#include "tensorrt_llm/nanobind/common/customCasters.h"

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/nanobind/common/bindTypes.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <ATen/ATen.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <torch/extension.h>
#include <tuple>

namespace nb = nanobind;
namespace tb = tensorrt_llm::batch_manager;
namespace tle = tensorrt_llm::executor;
namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::nanobind::batch_manager
{

void initBindings(nb::module_& m)
{
    using GenLlmReq = tb::GenericLlmRequest<runtime::ITensor::SharedPtr>;

    // Create and register exceptions in module scope
    static nb::object peft_exc = nb::exception<tb::PeftTaskNotCachedException>(m, "PeftTaskNotCachedException");
    static nb::object lora_exc = nb::exception<tr::LoraCacheFullException>(m, "LoraCacheFullException");

    // Register with no captures
    nb::register_exception_translator(
        [](std::exception_ptr const& p, void*)
        {
            try
            {
                if (p)
                    std::rethrow_exception(p);
            }
            catch (const tb::PeftTaskNotCachedException& e)
            {
                PyErr_SetString(peft_exc.ptr(), e.what());
            }
            catch (const tr::LoraCacheFullException& e)
            {
                PyErr_SetString(lora_exc.ptr(), e.what());
            }
        });

    NanobindUtils::bindSet<tb::ReqIdsSet>(m, "ReqIdsSet");

    nb::enum_<tb::LlmRequestType>(m, "LlmRequestType")
        .value("LLMREQUEST_TYPE_CONTEXT_AND_GENERATION", tb::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION)
        .value("LLMREQUEST_TYPE_CONTEXT_ONLY", tb::LLMREQUEST_TYPE_CONTEXT_ONLY)
        .value("LLMREQUEST_TYPE_GENERATION_ONLY", tb::LLMREQUEST_TYPE_GENERATION_ONLY)
        .export_values();

    nb::class_<tb::batch_scheduler::ContextChunkingConfig>(m, "ContextChunkingConfig")
        .def(nb::init<tle::ContextChunkingPolicy, tensorrt_llm::runtime::SizeType32>(), nb::arg("chunking_policy"),
            nb::arg("chunk_unit_size"))
        .def_rw("chunking_policy", &tb::batch_scheduler::ContextChunkingConfig::chunkingPolicy)
        .def_rw("chunk_unit_size", &tb::batch_scheduler::ContextChunkingConfig::chunkUnitSize);

    nb::class_<GenLlmReq>(m, "GenericLlmRequest")
        .def("set_exclude_input_from_output", &GenLlmReq::setExcludeInputFromOutput, nb::arg("exclude"))
        .def("get_num_tokens", &GenLlmReq::getNumTokens, nb::arg("beam"))
        .def_prop_ro("max_beam_num_tokens", &GenLlmReq::getMaxBeamNumTokens)
        .def("get_token", &GenLlmReq::getToken, nb::arg("beam"), nb::arg("pos"))
        .def("get_tokens", nb::overload_cast<GenLlmReq::SizeType32>(&GenLlmReq::getTokens, nb::const_), nb::arg("beam"))
        .def("get_tokens", nb::overload_cast<>(&GenLlmReq::getTokens, nb::const_))
        .def("get_last_tokens", nb::overload_cast<GenLlmReq::SizeType32>(&GenLlmReq::getLastTokens), nb::arg("beam"))
        .def("get_last_tokens", nb::overload_cast<>(&GenLlmReq::getLastTokens))
        .def("get_beam_width_by_iter", &GenLlmReq::getBeamWidthByIter, nb::arg("for_next_iteration") = false)
        .def_prop_ro("max_num_generated_tokens", &GenLlmReq::getMaxNumGeneratedTokens)
        .def("add_new_token", &GenLlmReq::addNewToken, nb::arg("token"), nb::arg("beam"))
        .def("add_new_tokens", &GenLlmReq::addNewTokens, nb::arg("beam_tokens"))
        .def_prop_ro("num_draft_tokens", &GenLlmReq::getNumDraftTokens)
        .def("set_generated_tokens", &GenLlmReq::setGeneratedTokens, nb::arg("generated_beam_tokens"))
        .def("pause", &GenLlmReq::pause, nb::arg("max_input_len"))
        .def_prop_rw("max_sent_token_len", &GenLlmReq::getMaxSentTokenLen, &GenLlmReq::setMaxSentTokenLen)
        .def_prop_ro("prompt_embedding_table", &GenLlmReq::getPromptEmbeddingTable)
        .def_prop_ro("multimodal_embedding", &GenLlmReq::getMultimodalEmbedding)
        .def_prop_ro("mrope_rotary_cos_sin", &GenLlmReq::getMropeRotaryCosSin)
        .def_prop_ro("bad_words_list", &GenLlmReq::getBadWordsList)
        .def_prop_rw("draft_logits", &GenLlmReq::getDraftLogits, &GenLlmReq::setDraftLogits)
        .def_prop_ro("embedding_bias", &GenLlmReq::getEmbeddingBias)
        .def_prop_rw("lora_config", &GenLlmReq::getLoraConfig, &GenLlmReq::setLoraConfig)
        .def_prop_rw("lora_weights", &GenLlmReq::getLoraWeights, &GenLlmReq::setLoraWeights)
        .def_prop_ro("stop_words_list", &GenLlmReq::getStopWordsList)
        .def_prop_ro("context_logits", &GenLlmReq::getContextLogitsHost)
        .def_prop_ro("generation_logits", &GenLlmReq::getGenerationLogitsHost)
        .def_prop_ro("prompt_vocab_size", &GenLlmReq::getPromptVocabSize)
        .def_prop_ro("mrope_position_deltas", &GenLlmReq::getMropePositionDeltas)
        .def_prop_ro("lora_task_id", &GenLlmReq::getLoraTaskId)
        .def_prop_ro("lookahead_config", &GenLlmReq::getLookaheadConfig)
        .def_prop_rw("context_chunk_size", &GenLlmReq::getContextChunkSize, &GenLlmReq::setContextChunkSize)
        .def_prop_rw("decoding_iter", &GenLlmReq::getDecodingIter, &GenLlmReq::setDecodingIter)
        .def_rw("request_id", &GenLlmReq::mRequestId)
        .def_rw("prompt_len", &GenLlmReq::mPromptLen)
        .def_rw("max_new_tokens", &GenLlmReq::mMaxNewTokens)
        .def_rw("sampling_config", &GenLlmReq::mSamplingConfig)
        .def_prop_rw("state", &GenLlmReq::getState, &GenLlmReq::setState)
        .def_prop_rw("streaming", &GenLlmReq::isStreaming, &GenLlmReq::setStreaming)
        .def_rw("end_id", &GenLlmReq::mEndId)
        .def_rw("pad_id", &GenLlmReq::mPadId)
        .def_rw("seq_slot", &GenLlmReq::mSeqSlot)
        .def_prop_ro("return_log_probs", &GenLlmReq::returnLogProbs)
        .def_prop_ro("return_context_logits", &GenLlmReq::getReturnContextLogits)
        .def_prop_ro("return_generation_logits", &GenLlmReq::getReturnGenerationLogits)
        .def_prop_ro("log_probs", nb::overload_cast<>(&GenLlmReq::getLogProbs, nb::const_))
        .def("get_log_probs", nb::overload_cast<GenLlmReq::SizeType32>(&GenLlmReq::getLogProbs, nb::const_))
        .def("set_log_probs", &GenLlmReq::setLogProbs, nb::arg("log_probs"), nb::arg("beam"))
        .def("set_return_encoder_output", &GenLlmReq::setReturnEncoderOutput, nb::arg("return_encoder_output"))
        .def("get_return_encoder_output", &GenLlmReq::getReturnEncoderOutput)
        .def("priority", nb::overload_cast<>(&GenLlmReq::priority, nb::const_))
        .def("set_priority", nb::overload_cast<tle::PriorityType>(&GenLlmReq::setPriority))
        .def_prop_ro("cum_log_probs", &GenLlmReq::getCumLogProbs)
        .def("set_cum_log_prob", &GenLlmReq::setCumLogProb, nb::arg("cum_log_prob"), nb::arg("beam"))
        .def("update_num_tokens_per_iteration", &GenLlmReq::updateNumTokensPerIteration,
            nb::arg("num_tokens_per_iteration"), nb::arg("model_config"))
        .def_prop_ro("orig_prompt_len", &GenLlmReq::getOrigPromptLen)
        .def("has_draft_tokens", &GenLlmReq::hasDraftTokens)
        .def("move_to_next_context_chunk", &GenLlmReq::moveToNextContextChunk)
        .def_prop_ro("is_last_context_chunk", &GenLlmReq::isLastContextChunk)
        .def_prop_ro("is_first_context_chunk", &GenLlmReq::isFirstContextChunk)
        .def_prop_ro("context_remaining_length", &GenLlmReq::getContextRemainingLength)
        .def_prop_ro("context_logits", &GenLlmReq::getContextLogitsHost)
        .def_prop_ro("num_draft_tokens", &GenLlmReq::getNumDraftTokens)
        .def("set_finished_reason", &GenLlmReq::setFinishedReason, nb::arg("finish_reason"), nb::arg("beam"))
        .def_prop_ro("is_finished", &GenLlmReq::isFinished)
        .def_prop_ro("is_finished_due_to_length", &GenLlmReq::isFinishedDueToLength)
        .def_prop_rw(
            "context_current_position", &GenLlmReq::getContextCurrentPosition, &GenLlmReq::setContextCurrentPosition)
        .def_prop_ro("prepopulated_prompt_len", &GenLlmReq::getPrepopulatedPromptLen)
        .def_prop_rw("guided_decoding_params", &GenLlmReq::getGuidedDecodingParams, &GenLlmReq::setGuidedDecodingParams)
        .def_prop_ro("context_phase_params", &GenLlmReq::getContextPhaseParams)
        .def_prop_ro("is_context_only_request", &GenLlmReq::isContextOnlyRequest)
        .def_prop_ro("is_generation_only_request", &GenLlmReq::isGenerationOnlyRequest)
        .def_prop_ro("is_generation_complete_state", &GenLlmReq::isGenerationCompleteState)
        .def_prop_ro("is_context_finished", &GenLlmReq::isContextFinished)
        .def_prop_ro("is_disagg_generation_init_state", &GenLlmReq::isDisaggGenerationInitState)
        .def_prop_ro("is_disagg_generation_transmission_complete", &GenLlmReq::isDisaggGenerationTransmissionComplete)
        .def_prop_ro(
            "is_disagg_generation_transmission_in_progress", &GenLlmReq::isDisaggGenerationTransmissionInProgress)
        .def_prop_ro("is_context_init_state", &GenLlmReq::isContextInitState)
        .def_prop_ro("is_generation_in_progress_state", &GenLlmReq::isGenerationInProgressState)
        .def_prop_ro("is_disagg_context_transmission_state", &GenLlmReq::isDisaggContextTransmissionState)
        .def_prop_ro("is_disagg_context_complete_state", &GenLlmReq::isDisaggContextCompleteState)
        .def_prop_ro("stage", &GenLlmReq::getRequestStage)
        .def_prop_ro("kv_cache_transfer_time_ms", &GenLlmReq::getKvCacheTransferTimeMS)
        .def_prop_ro("kv_cache_size", &GenLlmReq::getKvCacheSize)
        .def_prop_ro("avg_decoded_tokens_per_iter", &GenLlmReq::getAvgDecodedTokensPerIter)
        .def_prop_ro("alloc_total_blocks", &GenLlmReq::getAllocTotalBlocksPerRequest)
        .def_prop_ro("alloc_new_blocks", &GenLlmReq::getAllocNewBlocksPerRequest)
        .def("alloc_context_logits", &GenLlmReq::allocContextLogitsHost, nb::arg("vocab_size"), nb::arg("logit_dtype"))
        .def_prop_ro("reused_blocks", &GenLlmReq::getReusedBlocksPerRequest)
        .def_prop_ro("missed_blocks", &GenLlmReq::getMissedBlocksPerRequest)
        .def_prop_ro("kv_cache_hit_rate", &GenLlmReq::getKVCacheHitRatePerRequest)
        .def_prop_ro("llm_request_type", &GenLlmReq::getLlmRequestType)
        .def_prop_ro("parent_request_id", &GenLlmReq::getParentRequestId)
        .def_prop_ro("is_child", &GenLlmReq::isChild)
        .def_prop_ro("cache_salt_id", &GenLlmReq::getCacheSaltID)
        .def_prop_ro("multimodal_hashes",
            [](GenLlmReq& self)
            {
                std::optional<std::vector<std::vector<GenLlmReq::SizeType32>>> hashes = std::nullopt;
                if (self.getMultimodalHashes())
                {
                    hashes = *self.getMultimodalHashes().value();
                }
                return hashes;
            })
        .def_prop_ro("multimodal_positions",
            [](GenLlmReq& self)
            {
                std::optional<std::vector<GenLlmReq::SizeType32>> positions = std::nullopt;
                if (self.getMultimodalPositions())
                {
                    positions = *self.getMultimodalPositions().value();
                }
                return positions;
            })
        .def_prop_ro("multimodal_lengths",
            [](GenLlmReq& self)
            {
                std::optional<std::vector<GenLlmReq::SizeType32>> lengths = std::nullopt;
                if (self.getMultimodalLengths())
                {
                    lengths = *self.getMultimodalLengths().value();
                }
                return lengths;
            })
        .def_prop_ro("position_ids",
            [](GenLlmReq& self)
            {
                std::optional<std::vector<GenLlmReq::SizeType32>> positionIds = std::nullopt;
                if (self.getPositionIds())
                {
                    positionIds = *self.getPositionIds().value();
                }
                return positionIds;
            })
        .def_prop_rw(
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
        .def_prop_rw("is_dummy_request", &GenLlmReq::isDummyRequest, &GenLlmReq::setIsDummyRequest)
        .def_prop_ro("return_perf_metrics", &GenLlmReq::getReturnPerfMetrics)
        .def_prop_rw("use_draft_model", &GenLlmReq::useDraftModel, &GenLlmReq::setUseDraftModel);

    nb::class_<tb::LlmRequest, GenLlmReq>(m, "LlmRequest", nb::dynamic_attr())
        .def(
            "__init__",
            [](tb::LlmRequest* self, tb::LlmRequest::RequestIdType request_id,
                tb::LlmRequest::SizeType32 max_new_tokens, std::vector<tb::LlmRequest::TokenIdType> input_tokens,
                runtime::SamplingConfig sampling_config, bool is_streaming,
                std::optional<tb::LlmRequest::SizeType32> end_id, std::optional<tb::LlmRequest::SizeType32> pad_id,
                std::optional<at::Tensor> embedding_bias, std::optional<at::Tensor> bad_words_list,
                std::optional<at::Tensor> stop_words_list,
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
                bool apply_logits_post_processor_batched, std::optional<tb::LlmRequest::VecTokens> encoder_input_tokens,
                bool return_encoder_output, std::optional<tb::LlmRequest::RequestIdType> client_id,
                executor::PriorityType priority, std::optional<at::Tensor> encoder_input_features,
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

                new (self) tb::LlmRequest{request_id, max_new_tokens, input_tokens, sampling_config, is_streaming,
                    end_id, pad_id, embedding_bias_tensor_ptr, bad_words_list_tensor_ptr, stop_words_list_tensor_ptr,
                    position_ids, prompt_embedding_table_tensor_ptr, prompt_vocab_size, multimodal_hashes,
                    multimodal_positions, multimodal_lengths, multimodal_embedding_tensor_ptr,
                    mrope_rotary_cos_sin_tensor_ptr, mrope_position_deltas, lora_task_id, lora_weights_tensor_ptr,
                    lora_config_tensor_ptr, lookahead_config, kv_cache_retention_config, return_log_probs,
                    return_context_logits, return_generation_logits, draft_tokens, draft_logits_tensor_ptr,
                    exclude_input_from_output, logits_post_processor, apply_logits_post_processor_batched,
                    encoder_input_tokens, return_encoder_output, client_id, priority, encoder_input_features_tensor_ptr,
                    encoder_output_length, cross_attention_mask_tensor_ptr, llm_request_type, input_token_extra_ids,
                    num_return_sequences, eagle_config, skip_cross_attn_blocks_tensor_ptr, return_perf_metrics,
                    guided_decoding_params, language_adapter_uid, allotted_time_ms, context_phase_params, cache_salt_id,
                    arrival_time};
            },
            nb::arg("request_id"), nb::arg("max_new_tokens"), nb::arg("input_tokens"), nb::arg("sampling_config"),
            nb::arg("is_streaming"), nb::arg("end_id") = std::nullopt, nb::arg("pad_id") = std::nullopt,
            nb::arg("embedding_bias") = std::nullopt, nb::arg("bad_words_list") = std::nullopt,
            nb::arg("stop_words_list") = std::nullopt, nb::arg("position_ids") = std::nullopt,
            nb::arg("prompt_embedding_table") = std::nullopt, nb::arg("prompt_vocab_size") = std::nullopt,
            nb::arg("multimodal_hashes") = std::nullopt, nb::arg("multimodal_positions") = std::nullopt,
            nb::arg("multimodal_lengths") = std::nullopt, nb::arg("multimodal_embedding") = std::nullopt,
            nb::arg("mrope_rotary_cos_sin") = std::nullopt, nb::arg("mrope_position_deltas") = std::nullopt,
            nb::arg("lora_task_id") = std::nullopt, nb::arg("lora_weights") = std::nullopt,
            nb::arg("lora_config") = std::nullopt, nb::arg("lookahead_config") = std::nullopt,
            nb::arg("kv_cache_retention_config") = std::nullopt, nb::arg("return_log_probs") = false,
            nb::arg("return_context_logits") = false, nb::arg("return_generation_logits") = false,
            nb::arg("draft_tokens") = std::nullopt, nb::arg("draft_logits") = std::nullopt,
            nb::arg("exclude_input_from_output") = false, nb::arg("logits_post_processor") = std::nullopt,
            nb::arg("apply_logits_post_processor_batched") = false, nb::arg("encoder_input_tokens") = std::nullopt,
            nb::arg("return_encoder_output") = false, nb::arg("client_id") = std::nullopt,
            nb::arg("priority") = executor::Request::kDefaultPriority, nb::arg("encoder_input_features") = std::nullopt,
            nb::arg("encoder_output_len") = std::nullopt, nb::arg("cross_attention_mask") = std::nullopt,
            nb::arg("llm_request_type") = tb::LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
            nb::arg("input_token_extra_ids") = std::nullopt, nb::arg("num_return_sequences") = 1,
            nb::arg("eagle_config") = std::nullopt, nb::arg("skip_cross_attn_blocks") = std::nullopt,
            nb::arg("return_perf_metrics") = false, nb::arg("guided_decoding_params") = std::nullopt,
            nb::arg("language_adapter_uid") = std::nullopt, nb::arg("allotted_time_ms") = std::nullopt,
            nb::arg("context_phase_params") = std::nullopt, nb::arg("cache_salt_id") = std::nullopt,
            nb::arg("arrival_time") = std::nullopt)
        .def("check_token_id_range", &tb::LlmRequest::checkTokenIdRange, nb::arg("vocab_size"))
        .def(nb::init<tb::LlmRequest const&>())
        .def("validate", &tb::LlmRequest::validate, nb::arg("max_input_len"), nb::arg("max_seq_len"),
            nb::arg("max_draft_len"), nb::arg("vocab_size_padded"), nb::arg("max_endocer_input_len") = std::nullopt,
            nb::arg("enable_kv_cache_reuse") = false)
        .def("create_response", &tb::LlmRequest::createResponse, nb::arg("use_fast_logits") = false,
            nb::arg("mpi_world_rank") = 0)
        .def("create_child_request", &tb::LlmRequest::createChildRequest, nb::arg("child_id"))
        .def("create_result", &tb::LlmRequest::createResult, nb::arg("use_fast_logits") = false,
            nb::arg("mpi_world_rank") = 0)
        .def("create_serialized_result",
            [](tb::LlmRequest& self, bool use_fast_logits = false, int mpi_world_rank = 0)
            {
                std::vector<char> serialized_result;
                bool is_final = false;
                self.createSerializedResult(serialized_result, is_final, use_fast_logits, mpi_world_rank);
                return std::make_tuple(nb::bytes(serialized_result.data(), serialized_result.size()), is_final);
            })
        .def("move_prompt_embedding_table_to_gpu", &tb::LlmRequest::movePromptEmbeddingTableToGpu, nb::arg("manager"))
        .def("move_lora_weights_to_gpu", &tb::LlmRequest::moveLoraWeightsToGpu, nb::arg("manager"))
        .def("finish_by_reason", &tb::LlmRequest::finishByReason, nb::arg("finish_reason"))
        .def("set_first_scheduled_time", &tb::LlmRequest::setFirstScheduledTime)
        .def("update_perf_metrics", &tb::LlmRequest::updatePerfMetrics, nb::arg("iter_counter"))
        .def("remove_lora_tensors", &tb::LlmRequest::removeLoraTensors)
        .def_rw_static("global_steady_clock_offset", &tb::LlmRequest::mGlobalSteadyClockOffset);

    nb::class_<tb::SequenceSlotManager>(m, "SequenceSlotManager")
        .def(nb::init<tb::SequenceSlotManager::SlotIdType, uint64_t>(), nb::arg("max_num_slots"),
            nb::arg("max_sequence_idle_microseconds"))
        .def("get_sequence_slot", &tb::SequenceSlotManager::getSequenceSlot, nb::arg("start_flag"),
            nb::arg("sequence_id"))
        .def("free_sequence_slot", &tb::SequenceSlotManager::freeSequenceSlot, nb::arg("sequence_id"))
        .def("free_idle_sequence_slots", &tb::SequenceSlotManager::freeIdleSequenceSlots);

    nb::class_<tb::rnn_state_manager::RnnStateManager>(m, "RnnStateManager")
        .def(nb::init<tr::SizeType32, tr::ModelConfig, tr::WorldConfig, tr::BufferManager>(),
            nb::arg("max_num_sequences"), nb::arg("model_config"), nb::arg("world_config"), nb::arg("buffer_manager"));

    nb::class_<tb::DecoderInputBuffers>(m, "DecoderInputBuffers")
        .def(nb::init<runtime::SizeType32, runtime::SizeType32, tr::BufferManager>(), nb::arg("max_batch_size"),
            nb::arg("max_tokens_per_engine_step"), nb::arg("manager"))
        .def_rw("setup_batch_slots", &tb::DecoderInputBuffers::setupBatchSlots)
        .def_rw("setup_batch_slots_device", &tb::DecoderInputBuffers::setupBatchSlotsDevice)
        .def_rw("fill_values", &tb::DecoderInputBuffers::fillValues)
        .def_rw("fill_values_device", &tb::DecoderInputBuffers::fillValuesDevice)
        .def_rw("inputs_ids", &tb::DecoderInputBuffers::inputsIds)
        .def_rw("forward_batch_slots", &tb::DecoderInputBuffers::forwardBatchSlots)
        .def_rw("logits", &tb::DecoderInputBuffers::logits)
        .def_rw("decoder_requests", &tb::DecoderInputBuffers::decoderRequests);

    nb::class_<tb::DecoderOutputBuffers>(m, "DecoderOutputBuffers")
        .def_rw("sequence_lengths_host", &tb::DecoderOutputBuffers::sequenceLengthsHost)
        .def_rw("finished_sum_host", &tb::DecoderOutputBuffers::finishedSumHost)
        .def_prop_ro("new_output_tokens_host",
            [](tb::DecoderOutputBuffers& self) { return tr::Torch::tensor(self.newOutputTokensHost); })
        .def_rw("cum_log_probs_host", &tb::DecoderOutputBuffers::cumLogProbsHost)
        .def_rw("log_probs_host", &tb::DecoderOutputBuffers::logProbsHost)
        .def_rw("finish_reasons_host", &tb::DecoderOutputBuffers::finishReasonsHost);

    nb::class_<tb::SlotDecoderBuffers>(m, "SlotDecoderBuffers")
        .def(nb::init<runtime::SizeType32, runtime::SizeType32, runtime::BufferManager const&>(),
            nb::arg("max_beam_width"), nb::arg("max_seq_len"), nb::arg("buffer_manager"))
        .def_rw("output_ids", &tb::SlotDecoderBuffers::outputIds)
        .def_rw("output_ids_host", &tb::SlotDecoderBuffers::outputIdsHost)
        .def_rw("sequence_lengths_host", &tb::SlotDecoderBuffers::sequenceLengthsHost)
        .def_rw("cum_log_probs", &tb::SlotDecoderBuffers::cumLogProbs)
        .def_rw("cum_log_probs_host", &tb::SlotDecoderBuffers::cumLogProbsHost)
        .def_rw("log_probs", &tb::SlotDecoderBuffers::logProbs)
        .def_rw("log_probs_host", &tb::SlotDecoderBuffers::logProbsHost)
        .def_rw("finish_reasons_host", &tb::SlotDecoderBuffers::finishReasonsHost);

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
        nb::arg("requests"), nb::arg("tokens"), nb::arg("beam_idx"),
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
        nb::arg("context_requests"), nb::arg("generation_requests"), nb::arg("logits"), nb::arg("beam_width"),
        nb::arg("num_context_logits_prefix_sum"), nb::arg("decoder_input_buffers"), nb::arg("decoder_state"),
        nb::arg("buffer_manager"), "Make decoding batch input.");
}

} // namespace tensorrt_llm::nanobind::batch_manager
