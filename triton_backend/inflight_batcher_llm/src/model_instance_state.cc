// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "model_instance_state.h"
#include "utils.h"

#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <nlohmann/json.hpp>

#include <fstream>

namespace tle = tensorrt_llm::executor;
using executor::SizeType32;

namespace triton::backend::inflight_batcher_llm
{

TRITONSERVER_Error* ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance, ModelInstanceState** state)
{
    try
    {
        *state = new ModelInstanceState(model_state, triton_model_instance);
    }
    catch (std::exception const& ex)
    {
        std::string errStr = std::string("unexpected error when creating modelInstanceState: ") + ex.what();
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errStr.c_str());
    }

    return nullptr; // success
}

executor::BatchingType ModelInstanceState::getBatchingTypeFromParams()
{
    executor::BatchingType batchingType;
    auto gpt_model_type = model_state_->GetParameter<std::string>("gpt_model_type");

    if (gpt_model_type == "V1" || gpt_model_type == "v1")
    {
        batchingType = executor::BatchingType::kSTATIC;
    }
    else if (gpt_model_type == "inflight_batching" || gpt_model_type == "inflight_fused_batching")
    {
        batchingType = executor::BatchingType::kINFLIGHT;
    }
    else
    {
        throw std::runtime_error(
            "Invalid gpt_model_type. Must be "
            "v1/inflight_batching/inflight_fused_batching.");
    }
    return batchingType;
}

executor::KvCacheConfig ModelInstanceState::getKvCacheConfigFromParams()
{
    std::optional<int32_t> maxTokensInPagedKvCache = std::nullopt;
    try
    {
        maxTokensInPagedKvCache = model_state_->GetParameter<int32_t>("max_tokens_in_paged_kv_cache");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING(
            "max_tokens_in_paged_kv_cache is not specified, will "
            "use default value");
    }

    std::optional<float> kvCacheFreeGpuMemFraction = std::nullopt;
    try
    {
        kvCacheFreeGpuMemFraction = model_state_->GetParameter<float>("kv_cache_free_gpu_mem_fraction");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING(
            "kv_cache_free_gpu_mem_fraction is not specified, will use default value of 0.9 or "
            "max_tokens_in_paged_kv_cache");
    }

    std::optional<float> crossKvCacheFraction = std::nullopt;
    try
    {
        crossKvCacheFraction = model_state_->GetParameter<float>("cross_kv_cache_fraction");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("cross_kv_cache_fraction is not specified, error if it's encoder-decoder model, otherwise ok");
    }

    std::optional<size_t> kvCacheHostCacheSize = std::nullopt;
    try
    {
        kvCacheHostCacheSize = model_state_->GetParameter<size_t>("kv_cache_host_memory_bytes");
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("kv_cache_host_memory_bytes not set, defaulting to 0");
    }

    bool kvCacheOnboardBlocks = true;
    try
    {
        kvCacheOnboardBlocks = model_state_->GetParameter<bool>("kv_cache_onboard_blocks");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("kv_cache_onboard_blocks not set, defaulting to true");
    }

    std::optional<std::vector<int32_t>> maxAttentionWindow = std::nullopt;
    try
    {
        maxAttentionWindow = model_state_->GetParameter<std::vector<int32_t>>("max_attention_window_size");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING(
            "max_attention_window_size is not specified, will "
            "use default value (i.e. max_sequence_length)");
    }

    std::optional<int32_t> sinkTokenLength = std::nullopt;
    try
    {
        sinkTokenLength = model_state_->GetParameter<int32_t>("sink_token_length");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING(
            "sink_token_length is not specified, will "
            "use default value");
    }

    bool enableKVCacheReuse = true;
    try
    {
        enableKVCacheReuse = model_state_->GetParameter<bool>("enable_kv_cache_reuse");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("enable_kv_cache_reuse is not specified, will be set to true");
    }

    std::optional<std::vector<SizeType32>> maxAttentionWindowVec = std::nullopt;
    if (maxAttentionWindow.has_value())
    {
        maxAttentionWindowVec
            = std::vector<SizeType32>(maxAttentionWindow.value().begin(), maxAttentionWindow.value().end());
    }

    return executor::KvCacheConfig(enableKVCacheReuse, maxTokensInPagedKvCache, maxAttentionWindowVec, sinkTokenLength,
        kvCacheFreeGpuMemFraction, kvCacheHostCacheSize, kvCacheOnboardBlocks, crossKvCacheFraction);
}

executor::ExtendedRuntimePerfKnobConfig ModelInstanceState::getExtendedRuntimePerfKnobConfigFromParams()
{
    bool multiBlockMode = true;
    try
    {
        multiBlockMode = model_state_->GetParameter<bool>("multi_block_mode");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("multi_block_mode is not specified, will be set to true");
    }

    bool enableContextFMHAFP32Acc = false;
    try
    {
        enableContextFMHAFP32Acc = model_state_->GetParameter<bool>("enable_context_fmha_fp32_acc");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("enable_context_fmha_fp32_acc is not specified, will be set to false");
    }

    bool cudaGraphMode = false;
    try
    {
        cudaGraphMode = model_state_->GetParameter<bool>("cuda_graph_mode");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("cuda_graph_mode is not specified, will be set to false");
    }

    SizeType32 cudaGraphCacheSize = 0;
    try
    {
        cudaGraphCacheSize = model_state_->GetParameter<SizeType32>("cuda_graph_cache_size");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("cuda_graph_cache_size is not specified, will be set to 0");
    }

    return executor::ExtendedRuntimePerfKnobConfig(
        multiBlockMode, enableContextFMHAFP32Acc, cudaGraphMode, cudaGraphCacheSize);
}

executor::ParallelConfig ModelInstanceState::getParallelConfigFromParams()
{
    executor::ParallelConfig parallelConfig;
    if (mGpuDeviceIds)
    {
        parallelConfig.setDeviceIds(mGpuDeviceIds.value());
    }

    char const* useOrchestratorMode = std::getenv("TRTLLM_ORCHESTRATOR");
    auto const spawnProcessesEnvVar = std::getenv("TRTLLM_ORCHESTRATOR_SPAWN_PROCESSES");
    auto const spawnProcesses = !spawnProcessesEnvVar || std::atoi(spawnProcessesEnvVar);
    if (useOrchestratorMode && std::atoi(useOrchestratorMode) != 0)
    {
        parallelConfig.setCommunicationMode(executor::CommunicationMode::kORCHESTRATOR);
        mIsOrchestratorMode = true;

        tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);

        auto const workerExecutablePath = model_state_->GetExecutorWorkerPath();
        auto const isOrchestrator = spawnProcesses || (tensorrt_llm::mpi::MpiComm::world().getRank() == 0);
        auto orchestratorConfig
            = executor::OrchestratorConfig(isOrchestrator, workerExecutablePath, nullptr, spawnProcesses);
        parallelConfig.setOrchestratorConfig(orchestratorConfig);
    }

    if (mParticipantIds)
    {
        parallelConfig.setParticipantIds(mParticipantIds.value());
    }
    else if (!spawnProcesses && mIsOrchestratorMode)
    {
        TLLM_THROW("Spawning of processes was disabled in orchestrator mode, but participant IDs is missing.");
    }

    executor::SizeType32 numNodes = 1;
    try
    {
        numNodes = model_state_->GetParameter<int32_t>("num_nodes");
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_INFO("num_nodes is not specified, will be set to 1");
    }
    parallelConfig.setNumNodes(numNodes);

    return parallelConfig;
}

executor::PeftCacheConfig ModelInstanceState::getPeftCacheConfigFromParams()
{
    // parse LoRA / Peft cache parameters
    // lora_cache_max_adapter_size
    // lora_cache_optimal_adapter_size
    // lora_cache_gpu_memory_fraction
    // lora_cache_host_memory_bytes
    // lora_prefetch_dir

    SizeType32 maxAdapterSize = 64;
    SizeType32 optimalAdapterSize = 8;
    std::optional<size_t> hostCacheSize = std::nullopt;
    std::optional<float> deviceCachePercent = std::nullopt;
    std::optional<std::string> loraPrefetchDir = std::nullopt;

    std::string fieldName = "lora_cache_max_adapter_size";
    try
    {
        maxAdapterSize = model_state_->GetParameter<SizeType32>(fieldName);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(fieldName + " not set, defaulting to 64");
    }

    fieldName = "lora_cache_optimal_adapter_size";
    try
    {
        optimalAdapterSize = model_state_->GetParameter<SizeType32>(fieldName);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(fieldName + " not set, defaulting to 8");
    }
    fieldName = "lora_cache_gpu_memory_fraction";
    try
    {
        deviceCachePercent = model_state_->GetParameter<float>(fieldName);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(fieldName + " not set, defaulting to 0.05");
    }
    fieldName = "lora_cache_host_memory_bytes";
    try
    {
        hostCacheSize = model_state_->GetParameter<size_t>(fieldName);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(fieldName + " not set, defaulting to 1GB");
    }
    fieldName = "lora_prefetch_dir";
    try
    {
        loraPrefetchDir = model_state_->GetParameter<std::string>(fieldName);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(fieldName + " not set, defaulting to 1GB");
    }

    return executor::PeftCacheConfig(0, 0, optimalAdapterSize, maxAdapterSize,
        ModelInstanceState::kPeftCacheNumPutWorkers, ModelInstanceState::kPeftCacheNumEnsureWorkers,
        ModelInstanceState::kPeftCacheNumCopyStreams, 24, 8, deviceCachePercent, hostCacheSize, loraPrefetchDir);
}

executor::SchedulerConfig ModelInstanceState::getSchedulerConfigFromParams(bool enableChunkedContext)
{
    using executor::CapacitySchedulerPolicy;
    auto schedulerPolicy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
    try
    {
        std::string schedulerPolicyStr = model_state_->GetParameter<std::string>("batch_scheduler_policy");
        if (schedulerPolicyStr == "max_utilization")
        {
            schedulerPolicy = CapacitySchedulerPolicy::kMAX_UTILIZATION;
        }
        else if (schedulerPolicyStr == "guaranteed_no_evict")
        {
            schedulerPolicy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
        }
        else
        {
            throw std::runtime_error(
                "batch_scheduler_policy parameter was not found or is invalid "
                "(must be max_utilization or guaranteed_no_evict)");
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(e.what());
    }

    return executor::SchedulerConfig(schedulerPolicy);
}

executor::SpeculativeDecodingConfig ModelInstanceState::getSpeculativeDecodingConfigFromParams(
    std::optional<executor::OrchestratorConfig> orchConfig)
{
    bool fastLogits = false;
    try
    {
        fastLogits = model_state_->GetParameter<bool>("speculative_decoding_fast_logits");
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_INFO("speculative_decoding_fast_logits is not specified, will be set to false");
    }

    if (fastLogits && (!orchConfig.has_value() || orchConfig.value().getSpawnProcesses()))
    {
        TLLM_LOG_WARNING(
            "speculative_decoding_fast_logits is set, but requires orchestrator with spawn_processes disabled."
            "Disabling fast logits.");
        fastLogits = false;
    }

    mSpeculativeDecodingFastLogits = fastLogits;

    return executor::SpeculativeDecodingConfig(fastLogits);
}

std::optional<executor::GuidedDecodingConfig> ModelInstanceState::getGuidedDecodingConfigFromParams()
{
    std::optional<executor::GuidedDecodingConfig> guidedDecodingConfig = std::nullopt;
    std::string tokenizerDir = model_state_->GetParameter<std::string>("tokenizer_dir");
    std::string tokenizerInfoPath = model_state_->GetParameter<std::string>("xgrammar_tokenizer_info_path");
    std::string guidedDecodingBackendStr = model_state_->GetParameter<std::string>("guided_decoding_backend");

    if (!tokenizerDir.empty() && tokenizerDir != "${tokenizer_dir}")
    {
        TLLM_LOG_INFO(
            "Guided decoding C++ workflow does not use tokenizer_dir, this parameter will "
            "be ignored.");
    }

    if (guidedDecodingBackendStr.empty() || guidedDecodingBackendStr == "${guided_decoding_backend}"
        || tokenizerInfoPath.empty() || tokenizerInfoPath == "${xgrammar_tokenizer_info_path}")
    {
        return guidedDecodingConfig;
    }

    TLLM_CHECK_WITH_INFO(std::filesystem::exists(tokenizerInfoPath),
        "Xgrammar's tokenizer info path at %s does not exist.", tokenizerInfoPath.c_str());

    auto const tokenizerInfo = nlohmann::json::parse(std::ifstream{std::filesystem::path(tokenizerInfoPath)});
    auto const encodedVocab = tokenizerInfo["encoded_vocab"].template get<std::vector<std::string>>();
    auto const tokenizerStr = tokenizerInfo["tokenizer_str"].template get<std::string>();
    auto const stopTokenIds
        = tokenizerInfo["stop_token_ids"].template get<std::vector<tensorrt_llm::runtime::TokenIdType>>();

    executor::GuidedDecodingConfig::GuidedDecodingBackend guidedDecodingBackend;
    if (guidedDecodingBackendStr == "xgrammar")
    {
        guidedDecodingBackend = executor::GuidedDecodingConfig::GuidedDecodingBackend::kXGRAMMAR;
    }
    else
    {
        TLLM_THROW(
            "Guided decoding is currently supported with 'xgrammar' backend. Invalid guided_decoding_backend parameter "
            "provided.");
    }
    guidedDecodingConfig
        = executor::GuidedDecodingConfig(guidedDecodingBackend, encodedVocab, tokenizerStr, stopTokenIds);
    return guidedDecodingConfig;
}

executor::ExecutorConfig ModelInstanceState::getExecutorConfigFromParams()
{
    auto batchingType = getBatchingTypeFromParams();

    int32_t maxBeamWidth = 1;
    try
    {
        maxBeamWidth = model_state_->GetParameter<int32_t>("max_beam_width");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("max_beam_width is not specified, will use default value of 1");
    }

    int32_t iterStatsMaxIterations = tle::ExecutorConfig::kDefaultIterStatsMaxIterations;
    try
    {
        iterStatsMaxIterations = model_state_->GetParameter<int32_t>("iter_stats_max_iterations");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("iter_stats_max_iterations is not specified, will use default value of "
            + std::to_string(iterStatsMaxIterations));
    }

    int32_t requestStatsMaxIterations = tle::ExecutorConfig::kDefaultRequestStatsMaxIterations;
    try
    {
        requestStatsMaxIterations = model_state_->GetParameter<int32_t>("request_stats_max_iterations");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("request_stats_max_iterations is not specified, will use default value of "
            + std::to_string(requestStatsMaxIterations));
    }

    try
    {
        model_state_->GetParameter<bool>("enable_trt_overlap");
        TLLM_LOG_WARNING("enable_trt_overlap is deprecated and will be ignored");
    }
    catch (std::exception const& e)
    {
    }

    bool normalizeLogProbs = true;
    try
    {
        normalizeLogProbs = model_state_->GetParameter<bool>("normalize_log_probs");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("normalize_log_probs is not specified, will be set to true");
    }

    executor::ExecutorConfig executorConfig;

    auto kvCacheConfig = getKvCacheConfigFromParams();

    bool enableChunkedContext = false;
    try
    {
        enableChunkedContext = model_state_->GetParameter<bool>("enable_chunked_context");
        if (enableChunkedContext)
        {
            TLLM_LOG_WARNING(
                "enable_chunked_context is set to true, will use context chunking "
                "(requires building the model with use_paged_context_fmha).");
        }
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("enable_chunked_context is not specified, will be set to false.");
    }

    auto schedulerConfig = getSchedulerConfigFromParams(enableChunkedContext);

    auto peftCacheConfig = getPeftCacheConfigFromParams();

    auto parallelConfig = getParallelConfigFromParams();

    auto extendedRuntimePerfKnobConfig = getExtendedRuntimePerfKnobConfigFromParams();

    auto specDecConfig = getSpeculativeDecodingConfigFromParams(parallelConfig.getOrchestratorConfig());

    std::optional<executor::DecodingMode> decodingMode = std::nullopt;
    try
    {
        std::string decodingModeStr = model_state_->GetParameter<std::string>("decoding_mode");
        if (decodingModeStr == "top_k")
        {
            decodingMode = executor::DecodingMode::TopK();
        }
        else if (decodingModeStr == "top_p")
        {
            decodingMode = executor::DecodingMode::TopP();
        }
        else if (decodingModeStr == "top_k_top_p")
        {
            decodingMode = executor::DecodingMode::TopKTopP();
        }
        else if (decodingModeStr == "beam_search")
        {
            decodingMode = executor::DecodingMode::BeamSearch();
        }
        else if (decodingModeStr == "medusa")
        {
            decodingMode = executor::DecodingMode::Medusa();
        }
        else if (decodingModeStr == "redrafter")
        {
            decodingMode = executor::DecodingMode::ExplicitDraftTokens();
        }
        else if (decodingModeStr == "lookahead")
        {
            decodingMode = executor::DecodingMode::Lookahead();
        }
        else if (decodingModeStr == "eagle")
        {
            decodingMode = executor::DecodingMode::Eagle();
        }
        else
        {
            throw std::runtime_error("");
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(
            "decoding_mode parameter is invalid or not specified"
            "(must be one of the {top_k, top_p, top_k_top_p, beam_search, medusa, redrafter, lookahead, eagle})."
            "Using default: top_k_top_p if max_beam_width == 1, beam_search otherwise");
    }

    executor::DecodingConfig decodingConfig(decodingMode);

    try
    {
        auto medusaChoices = model_state_->GetParameter<executor::MedusaChoices>("medusa_choices");
        decodingConfig.setMedusaChoices(medusaChoices);
    }
    catch (std::exception const& e)
    {
        if (decodingMode && decodingMode->isMedusa())
        {
            TLLM_LOG_WARNING(
                "medusa_choices parameter is not specified. "
                "Will be using default mc_sim_7b_63 choices instead.");
        }
    }

    try
    {
        auto eagleChoices = model_state_->GetParameter<executor::EagleChoices>("eagle_choices");
        executor::EagleConfig eagleConfig(eagleChoices);
        decodingConfig.setEagleConfig(eagleConfig);
    }
    catch (std::exception const& e)
    {
        if (decodingMode && decodingMode->isEagle())
        {
            TLLM_LOG_WARNING(
                "eagle_choices parameter is not specified. "
                "Will be using default mc_sim_7b_63 choices instead or choices specified per-request.");
        }
    }

    if (decodingMode && decodingMode->isLookahead())
    {
        try
        {
            executor::SizeType32 windowSize = 0, ngramSize = 0, verificationSetSize = 0;
            windowSize = model_state_->GetParameter<uint32_t>("lookahead_window_size");
            ngramSize = model_state_->GetParameter<uint32_t>("lookahead_ngram_size");
            verificationSetSize = model_state_->GetParameter<uint32_t>("lookahead_verification_set_size");

            mExecutorLookaheadDecodingConfig
                = executor::LookaheadDecodingConfig{windowSize, ngramSize, verificationSetSize};
            decodingConfig.setLookaheadDecodingConfig(mExecutorLookaheadDecodingConfig.value());
        }
        catch (std::exception const& e)
        {
            TLLM_THROW(
                "Decoding mode is set to lookahead but lookahead parameters are not specified. "
                "Please set parameters lookahead_window_size, lookahead_ngram_size, and "
                "lookahead_verification_set_size.");
        }
    }

    float gpuWeightsPercent = 1.0f;
    try
    {
        gpuWeightsPercent = model_state_->GetParameter<float>("gpu_weights_percent");
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("gpu_weights_percent parameter is not specified, will use default value of 1.0");
    }

    std::optional<SizeType32> maxQueueSize = std::nullopt;
    try
    {
        triton::common::TritonJson::Value dynamic_batching;
        if (model_state_->GetModelConfig().Find("dynamic_batching", &dynamic_batching))
        {
            triton::common::TritonJson::Value default_queue_policy;
            if (dynamic_batching.Find("default_queue_policy", &default_queue_policy))
            {
                int64_t max_queue_size = 0;
                auto err = default_queue_policy.MemberAsInt("max_queue_size", &max_queue_size);
                if (err == nullptr)
                {
                    maxQueueSize = static_cast<SizeType32>(max_queue_size);
                }
            }
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(e.what());
    }

    SizeType32 recvPollPeriodMs = 0;
    try
    {
        recvPollPeriodMs = model_state_->GetParameter<int>("recv_poll_period_ms");
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_INFO("recv_poll_period_ms is not set, will use busy loop");
    }

    auto guidedConfig = getGuidedDecodingConfigFromParams();

    auto execConfig = executor::ExecutorConfig{maxBeamWidth, schedulerConfig, kvCacheConfig, enableChunkedContext,
        normalizeLogProbs, iterStatsMaxIterations, requestStatsMaxIterations, batchingType,
        /*maxBatchSize*/ std::nullopt, /*maxNumTokens*/ std::nullopt, parallelConfig, peftCacheConfig,
        /*LogitsPostProcessorConfig*/ std::nullopt, decodingConfig, /*useGpuDirectStorage*/ false, gpuWeightsPercent,
        maxQueueSize, extendedRuntimePerfKnobConfig,
        /*DebugConfig*/ std::nullopt, recvPollPeriodMs};
    execConfig.setSpecDecConfig(specDecConfig);
    execConfig.setCacheTransceiverConfig(tle::CacheTransceiverConfig(tle::CacheTransceiverConfig::BackendType::MPI));
    if (guidedConfig.has_value())
    {
        execConfig.setGuidedDecodingConfig(guidedConfig.value());
        TLLM_LOG_INFO("Guided decoding config has been provided and set as guided decoder.");
    }
    return execConfig;
}

ModelInstanceState::ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : model_state_(model_state)
    , modelInstance_(triton_model_instance)
{

    mInstanceIndex = model_state->getAndIncrementInstanceIndex();
    if (model_state_->getDeviceIds() && model_state_->getDeviceIds().value().size())
    {
        mGpuDeviceIds
            = model_state_->getDeviceIds().value()[mInstanceIndex % model_state_->getDeviceIds().value().size()];
    }
    else
    {
        mGpuDeviceIds = std::nullopt;
    }
    mIsOrchestratorMode = false;

    auto participantIds = model_state_->getParticipantIds();
    if (participantIds && !participantIds.value().empty())
    {
        mParticipantIds = participantIds.value()[mInstanceIndex % participantIds.value().size()];
    }
    else
    {
        mParticipantIds = std::nullopt;
    }
    auto executorConfig = getExecutorConfigFromParams();

#ifdef TRITON_ENABLE_METRICS
    custom_metrics_reporter_ = std::make_unique<custom_metrics_reporter::CustomMetricsReporter>();
    custom_metrics_reporter_->InitializeReporter(model_state->GetModelName(), model_state->GetModelVersion(),
        (executorConfig.getBatchingType() == executor::BatchingType::kSTATIC));
#endif

    std::string decoderModelPath;
    try
    {
        decoderModelPath = model_state_->GetParameter<std::string>("gpt_model_path");
        TLLM_CHECK_WITH_INFO(std::filesystem::exists(decoderModelPath),
            "Decoder (GPT) model path at %s does not exist.", decoderModelPath.c_str());
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("gpt_model_path is not specified, will be left empty");
        decoderModelPath = "";
    }

    std::string encoderModelPath;
    try
    {
        encoderModelPath = model_state_->GetParameter<std::string>("encoder_model_path");
        TLLM_CHECK_WITH_INFO(std::filesystem::exists(encoderModelPath), "Encoder model path at %s does not exist.",
            encoderModelPath.c_str());
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("encoder_model_path is not specified, will be left empty");
        encoderModelPath = "";
    }

    TLLM_CHECK_WITH_INFO(
        !decoderModelPath.empty() || !encoderModelPath.empty(), "Both encoder and decoder model paths are empty");

    if (!decoderModelPath.empty())
    {
        // Encoder-decoder model
        if (!encoderModelPath.empty())
        {
            mModelType = executor::ModelType::kENCODER_DECODER;
            mExecutor
                = std::make_unique<executor::Executor>(encoderModelPath, decoderModelPath, mModelType, executorConfig);
        }
        // Decoder only model
        else
        {
            mModelType = executor::ModelType::kDECODER_ONLY;
            mExecutor = std::make_unique<executor::Executor>(decoderModelPath, mModelType, executorConfig);
        }
    }
    // Encoder only
    else
    {
        mModelType = executor::ModelType::kENCODER_ONLY;
        mExecutor = std::make_unique<executor::Executor>(encoderModelPath, mModelType, executorConfig);
    }

    bool excludeInputInOutput = false;
    try
    {
        excludeInputInOutput = model_state_->GetParameter<bool>("exclude_input_in_output");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("exclude_input_in_output is not specified, will be set to false");
    }
    mInstanceSpecificConfig.excludeInputFromOutput = excludeInputInOutput;

    int cancellationCheckPeriodMs = 100;
    try
    {
        cancellationCheckPeriodMs = model_state_->GetParameter<int>("cancellation_check_period_ms");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("cancellation_check_period_ms is not specified, will be set to 100 (ms)");
    }
    mInstanceSpecificConfig.cancellationCheckPeriodMs = cancellationCheckPeriodMs;

    int statsCheckPeriodMs = 100;
    try
    {
        statsCheckPeriodMs = model_state_->GetParameter<int>("stats_check_period_ms");
    }
    catch (std::exception const& e)
    {
        // If parameter is not specified, just ignore
        TLLM_LOG_WARNING("stats_check_period_ms is not specified, will be set to 100 (ms)");
    }
    mInstanceSpecificConfig.statsCheckPeriodMs = statsCheckPeriodMs;

    if (mExecutor->canEnqueueRequests())
    {
        mStopWaitForResponse = false;
        mWaitForResponseThread = std::thread(&ModelInstanceState::WaitForResponse, this);

        mStopWaitForStats = false;
        mWaitForStatsThread = std::thread(&ModelInstanceState::WaitForStats, this);

        mStopWaitForCancel = false;
        mWaitForCancelThread = std::thread(&ModelInstanceState::WaitForCancel, this);
    }
    else
    {
        // Shutdown the worker ranks which will cause them to wait for leader/orchestrator to terminate
        mExecutor->shutdown();

        if (mExecutor->isParticipant())
        {
            // Since leader/orchestrator can terminate if there are issues loading other models like pre/post processing
            // we still don't want to return from initialize since Triton server would appear as ready
            // So exit
            TLLM_LOG_INFO("Terminating worker process since shutdown signal was received from leader or orchestrator");
            exit(0);
        }
    }
}

void ModelInstanceState::sendEnqueueResponse(TRITONBACKEND_Request* request, TRITONSERVER_Error* error)
{
    TRITONBACKEND_ResponseFactory* factory;
    LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory, request), "failed to create triton response factory");
    TRITONBACKEND_Response* tritonResponse;
    LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");
    LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error),
        "Cannot send response");
    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL), "Cannot release request");
}

bool ModelInstanceState::handleStopRequest(TRITONBACKEND_Request* request, std::string const& tritonRequestId)
{
    bool stopRequest = utils::getRequestBooleanInputTensor(request, kStopInputTensorName);
    if (!stopRequest)
    {
        return false;
    }

    TRITONSERVER_Error* error = nullptr;

    try
    {
        if (tritonRequestId == "")
        {
            throw std::runtime_error("Trying to stop a request but request ID is not provided");
        }
        std::lock_guard<std::mutex> lock(mRequestIdToRequestDataMutex);
        if (mTritonRequestIdToRequestIds.count(tritonRequestId))
        {
            auto requestIds = mTritonRequestIdToRequestIds[tritonRequestId];
            for (auto const& requestId : requestIds)
            {
                mExecutor->cancelRequest(requestId);
            }
        }
    }
    catch (std::exception const& e)
    {
        error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
    }
    // mTritonRequestIdToRequestIds.count(tritonRequestId) == false doesn't necessary mean an error since the
    // request to cancel may already be completed.
    // Send an empty response to indicate the request has been successfully cancelled
    sendEnqueueResponse(request, error);
    return true;
}

// Split batched TRITONBACKEND_Request into one executor:Request object per sample.
std::vector<executor::Request> ModelInstanceState::createExecutorRequests(TRITONBACKEND_Request* request,
    bool excludeInputFromOutput, bool isDecoupled, executor::ModelType modelType, bool isOrchestrator,
    bool specDecFastLogits, std::optional<executor::LookaheadDecodingConfig> const& lookaheadDecodingConfig)
{
    auto inputsTensors = utils::readInputsTensors(request);
    bool streaming = utils::getRequestBooleanInputTensor(request, kStreamingInputTensorName);
    executor::RequestType requestType = utils::getRequestType(request);

    return utils::createRequestsFromInputTensors(inputsTensors, excludeInputFromOutput, isDecoupled, streaming,
        modelType, requestType, isOrchestrator, specDecFastLogits, lookaheadDecodingConfig);
}

void ModelInstanceState::enqueue(TRITONBACKEND_Request** requests, uint32_t const request_count)
{

    uint64_t exec_start_ns{0};
    SET_TIMESTAMP(exec_start_ns);

    for (uint32_t i = 0; i < request_count; ++i)
    {
        TRITONBACKEND_Request* request = requests[i];

        try
        {
            char const* charRequestId = nullptr;
            TRITONBACKEND_RequestId(request, &charRequestId);
            std::string tritonRequestId;
            if (charRequestId != nullptr)
            {
                tritonRequestId = charRequestId;
            }

            if (handleStopRequest(request, tritonRequestId))
            {
                continue;
            }

            auto executorRequests
                = createExecutorRequests(request, mInstanceSpecificConfig.excludeInputFromOutput, isDecoupled(),
                    mModelType, mIsOrchestratorMode, mSpeculativeDecodingFastLogits, mExecutorLookaheadDecodingConfig);

            std::lock_guard<std::mutex> lock(mRequestIdToRequestDataMutex);
            TRITONBACKEND_ResponseFactory* factory;
            LOG_IF_ERROR(
                TRITONBACKEND_ResponseFactoryNew(&factory, request), "failed to create triton response factory");

            uint64_t compute_start_ns{0};
            SET_TIMESTAMP(compute_start_ns);

            auto requestIds = mExecutor->enqueueRequests(executorRequests);
            auto requestIdsSet = std::make_shared<std::set<executor::IdType>>(requestIds.begin(), requestIds.end());

            bool returnKvCacheReuseStats
                = utils::getRequestBooleanInputTensor(request, InputFieldsNames::returnKvCacheReuseStats);
            if (returnKvCacheReuseStats)
            {
                TLLM_LOG_WARNING("return_kv_cache_reuse_stats is deprecated, please use return_perf_metrics instead");
            }

            bool returnPerfMetrics = utils::getRequestBooleanInputTensor(request, InputFieldsNames::returnPerfMetrics);
            bool returnNumInputTokens
                = utils::getRequestBooleanInputTensor(request, InputFieldsNames::returnNumInputTokens);
            bool returnNumOutputTokens
                = utils::getRequestBooleanInputTensor(request, InputFieldsNames::returnNumOutputTokens);

            // Note:
            // A single TRITONBACKEND_Request will produce multiple executor requests when bs > 1.
            // They are treated as individual executor requests until they come back to triton server,
            // which generates a single response combining responses for all requests in the batch.
            for (int32_t batchIndex = 0; batchIndex < static_cast<int32_t>(requestIds.size()); ++batchIndex)
            {
                auto const& requestId = requestIds.at(batchIndex);
                auto const& executorRequest = executorRequests.at(batchIndex);
                int64_t inputTokensSize = executorRequest.getInputTokenIds().size();
                bool streaming = executorRequest.getStreaming();
                executor::SizeType32 beamWidthCopy = executorRequest.getSamplingConfig().getBeamWidth();
                bool excludeInputFromOutput = executorRequest.getOutputConfig().excludeInputFromOutput;
                if (mRequestIdToRequestData.count(requestId))
                {
                    TLLM_LOG_ERROR(
                        "Executor returns a request ID that already exists. This shouldn't happen unless there is "
                        "something "
                        "wrong in TRT-LLM runtime.");
                }
                auto requestOutputNames = utils::getRequestOutputNames(request);
                int32_t const numReturnSequences
                    = executorRequest.getSamplingConfig().getNumReturnSequences().value_or(1);
                mRequestIdToRequestData.emplace(requestId,
                    RequestData{factory, request, tritonRequestId, inputTokensSize, 0, streaming,
                        excludeInputFromOutput, beamWidthCopy, std::move(requestOutputNames),
                        {exec_start_ns, compute_start_ns, 0, 0}, batchIndex, static_cast<int32_t>(requestIds.size()),
                        numReturnSequences, requestIdsSet, executorRequest.getRequestType(), returnPerfMetrics,
                        returnNumInputTokens, returnNumOutputTokens});
            }
            if (tritonRequestId != "")
            {
                mTritonRequestIdToRequestIds[tritonRequestId] = *requestIdsSet;
            }
        }
        catch (std::exception const& e)
        {
            sendEnqueueResponse(request, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what()));
        }
    }
    return;
}

TRITONSERVER_Error* ModelInstanceState::reportBaseMetrics(RequestData& requestData, TRITONSERVER_Error* error)
{
    auto& timestamps = requestData.timestamps;
    SET_TIMESTAMP(timestamps.exec_end_ns);

    RETURN_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(modelInstance_, requestData.tritonRequest, (error == nullptr),
            timestamps.exec_start_ns, timestamps.compute_start_ns, timestamps.compute_end_ns, timestamps.exec_end_ns));

    // For now we will assume a batch size of 1 for each request. This may change in the future but for
    // now it seems that even when requests are dynamically batched together each workItem is associated
    // with its own request object and is handled independently due to the nature of IFB.
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(modelInstance_, 1 /* batch size */,
        timestamps.exec_start_ns, timestamps.compute_start_ns, timestamps.compute_end_ns, timestamps.exec_end_ns));

    return nullptr; // success
}

TRITONSERVER_Error* ModelInstanceState::reportCustomMetrics(
    int64_t inputTokensSize, int64_t outputTokensSize, TRITONSERVER_Error* error)
{
    std::string statJson = "{";
    statJson.append("\"Total Output Tokens\":" + std::to_string(outputTokensSize) + ",");
    statJson.append("\"Total Input Tokens\":" + std::to_string(inputTokensSize) + ",");
    statJson.back() = '}';
#ifdef TRITON_ENABLE_METRICS
    LOG_IF_ERROR(custom_metrics_reporter_->UpdateCustomMetrics(statJson), "Failed updating TRT LLM statistics");
#endif
    return nullptr; // success
}

std::tuple<TRITONBACKEND_Response*, bool, TRITONSERVER_Error*, int64_t> ModelInstanceState::fillTritonResponse(
    TRITONBACKEND_ResponseFactory* factory, executor::Response const& response, RequestData const& requestData)
{
    TRITONBACKEND_Response* tritonResponse;
    LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");

    TRITONSERVER_Error* error = nullptr;
    bool isFinal = false;
    int64_t outputTokensSize = 0;
    try
    {
        if (!response.hasError())
        {
            auto result = response.getResult();
            isFinal = result.isFinal;
            error = nullptr;
            auto sequenceIndex = result.sequenceIndex;
            auto& outputIds = result.outputTokenIds;
            std::vector<int32_t> beamLength(outputIds.size());
            int32_t maxBeamLength = -1;
            for (size_t i = 0; i < outputIds.size(); ++i)
            {
                // We want to capture ALL output tokens for ALL beams
                outputTokensSize += outputIds[i].size();
                beamLength[i] = outputIds[i].size();
                maxBeamLength = std::max(beamLength[i], maxBeamLength);
            }
            if (maxBeamLength == -1)
            {
                TLLM_LOG_ERROR("Output ids is empty");
                maxBeamLength = 0;
            }
            for (auto& vec : outputIds)
            {
                vec.resize(maxBeamLength, -1);
            }

            if (requestData.outputNames.count(OutputFieldsNames::outputIds) > 0)
            {
                std::vector<int64_t> outputIdsShape{1, static_cast<int64_t>(outputIds.size()), maxBeamLength};
                auto outputIdsType = TRITONSERVER_TYPE_INT32;
                auto outputIdsBuffer = utils::getResponseBuffer<int32_t>(
                    tritonResponse, outputIdsShape, outputIdsType, OutputFieldsNames::outputIds);
                utils::flatten<int32_t>(outputIds, outputIdsBuffer, outputIdsShape);
            }
            else
            {
                TLLM_THROW("%s tensor must be present in list of output tensors", OutputFieldsNames::outputIds);
            }

            if (requestData.outputNames.count(OutputFieldsNames::sequenceLength) > 0)
            {
                std::vector<int64_t> sequenceLengthShape{1, static_cast<int64_t>(outputIds.size())};
                auto sequenceLengthType = TRITONSERVER_TYPE_INT32;
                auto sequenceLengthBuffer = utils::getResponseBuffer<int32_t>(
                    tritonResponse, sequenceLengthShape, sequenceLengthType, OutputFieldsNames::sequenceLength);
                utils::flatten<int32_t>(beamLength, sequenceLengthBuffer, sequenceLengthShape);
            }
            else
            {
                TLLM_THROW("%s tensor must be present in list of output tensors", OutputFieldsNames::sequenceLength);
            }

            if (requestData.outputNames.count(OutputFieldsNames::contextLogits) > 0)
            {
                if (result.contextLogits.has_value())
                {
                    auto contextLogitsShapeOriginal = result.contextLogits.value().getShape();
                    std::vector<int64_t> contextLogitsShape{
                        1, contextLogitsShapeOriginal[0], contextLogitsShapeOriginal[1]};
                    auto contextLogitsType = utils::to_triton_datatype(result.contextLogits.value().getDataType());
                    TLLM_CHECK(contextLogitsType == model_state_->getLogitsDataType());
                    if (contextLogitsType == TRITONSERVER_TYPE_FP32)
                    {
                        auto contextLogitsBuffer = utils::getResponseBuffer<float>(
                            tritonResponse, contextLogitsShape, contextLogitsType, OutputFieldsNames::contextLogits);
                        utils::flatten<float>(result.contextLogits.value(), contextLogitsBuffer, contextLogitsShape);
                    }
                    else if (contextLogitsType == TRITONSERVER_TYPE_FP16)
                    {
                        auto contextLogitsBuffer = utils::getResponseBuffer<half>(
                            tritonResponse, contextLogitsShape, contextLogitsType, OutputFieldsNames::contextLogits);
                        utils::flatten<half>(result.contextLogits.value(), contextLogitsBuffer, contextLogitsShape);
                    }
                    else
                    {
                        TLLM_THROW("Logits type is not supported");
                    }
                }
            }

            if (requestData.outputNames.count(OutputFieldsNames::generationLogits) > 0)
            {
                if (result.generationLogits.has_value())
                {
                    auto generationLogitsShapeOriginal = result.generationLogits.value().getShape();
                    std::vector<int64_t> generationLogitsShape{1, generationLogitsShapeOriginal[0],
                        generationLogitsShapeOriginal[1], generationLogitsShapeOriginal[2]};
                    auto generationLogitsType
                        = utils::to_triton_datatype(result.generationLogits.value().getDataType());
                    TLLM_CHECK(generationLogitsType == model_state_->getLogitsDataType());
                    if (generationLogitsType == TRITONSERVER_TYPE_FP32)
                    {
                        auto generationLogitsBuffer = utils::getResponseBuffer<float>(tritonResponse,
                            generationLogitsShape, generationLogitsType, OutputFieldsNames::generationLogits);
                        utils::flatten<float>(
                            result.generationLogits.value(), generationLogitsBuffer, generationLogitsShape);
                    }
                    else if (generationLogitsType == TRITONSERVER_TYPE_FP16)
                    {
                        auto generationLogitsBuffer = utils::getResponseBuffer<half>(tritonResponse,
                            generationLogitsShape, generationLogitsType, OutputFieldsNames::generationLogits);
                        utils::flatten<half>(
                            result.generationLogits.value(), generationLogitsBuffer, generationLogitsShape);
                    }
                    else
                    {
                        TLLM_THROW("Logits type is not supported");
                    }
                }
                else if (result.specDecFastLogitsInfo.has_value())
                {
                    auto const& logitsInfo = result.specDecFastLogitsInfo.value();
                    size_t const numLogitsNeeded = (sizeof(logitsInfo) + 1) / sizeof(float);
                    std::vector<int64_t> generationLogitsShape{1, 1, 1, numLogitsNeeded};
                    auto generationLogitsType = TRITONSERVER_TYPE_FP32;
                    std::vector<float> data(numLogitsNeeded);
                    std::memcpy(data.data(), &logitsInfo, sizeof(logitsInfo));
                    auto generationLogitsBuffer = utils::getResponseBuffer<float>(tritonResponse, generationLogitsShape,
                        generationLogitsType, OutputFieldsNames::generationLogits);
                    utils::flatten<float>(data, generationLogitsBuffer, generationLogitsShape);
                }
            }

            if (requestData.outputNames.count(OutputFieldsNames::outputLogProbs) > 0)
            {
                if (result.logProbs.has_value())
                {
                    auto& logProbs = result.logProbs.value();
                    size_t maxLogProbs = 0;
                    for (auto const& vec : logProbs)
                    {
                        maxLogProbs = std::max(maxLogProbs, vec.size());
                    }
                    for (auto& vec : logProbs)
                    {
                        vec.resize(maxLogProbs, -1);
                    }
                    std::vector<int64_t> outputLogProbsShape{
                        1, static_cast<int64_t>(logProbs.size()), static_cast<int64_t>(logProbs[0].size())};
                    auto outputLogProbsType = TRITONSERVER_TYPE_FP32;
                    auto outputLogProbsBuffer = utils::getResponseBuffer<float>(
                        tritonResponse, outputLogProbsShape, outputLogProbsType, OutputFieldsNames::outputLogProbs);
                    utils::flatten<float>(logProbs, outputLogProbsBuffer, outputLogProbsShape);
                }
            }

            if (requestData.outputNames.count(OutputFieldsNames::cumLogProbs) > 0)
            {
                if (result.cumLogProbs.has_value())
                {
                    std::vector<int64_t> cumLogProbsShape{1, static_cast<int64_t>(result.cumLogProbs.value().size())};
                    auto cumLogProbsType = TRITONSERVER_TYPE_FP32;
                    auto cumLogProbsBuffer = utils::getResponseBuffer<float>(
                        tritonResponse, cumLogProbsShape, cumLogProbsType, OutputFieldsNames::cumLogProbs);
                    utils::flatten<float>(result.cumLogProbs.value(), cumLogProbsBuffer, cumLogProbsShape);
                }
            }

            if (requestData.outputNames.count(OutputFieldsNames::batchIndex) > 0 && requestData.batchSize > 1)
            {
                std::vector<int64_t> batchIndexShape{1, 1};
                auto batchIndexType = TRITONSERVER_TYPE_INT32;
                auto batchIndexBuffer = utils::getResponseBuffer<int32_t>(
                    tritonResponse, batchIndexShape, batchIndexType, OutputFieldsNames::batchIndex);
                std::vector<int32_t> batchIndexVec = {requestData.batchIndex};
                utils::flatten<int32_t>(batchIndexVec, batchIndexBuffer, batchIndexShape);
            }

            if (requestData.outputNames.count(OutputFieldsNames::sequenceIndex) > 0
                && requestData.numReturnSequences > 1)
            {
                std::vector<int64_t> sequenceIndexShape{1, 1};
                auto sequenceIndexType = TRITONSERVER_TYPE_INT32;
                auto sequenceIndexBuffer = utils::getResponseBuffer<int32_t>(
                    tritonResponse, sequenceIndexShape, sequenceIndexType, OutputFieldsNames::sequenceIndex);
                std::vector<int32_t> sequenceIndexVec = {sequenceIndex};
                utils::flatten<int32_t>(sequenceIndexVec, sequenceIndexBuffer, sequenceIndexShape);
            }

            if (requestData.requestType == executor::RequestType::REQUEST_TYPE_CONTEXT_ONLY)
            {
                if (response.getResult().contextPhaseParams.has_value())
                {
                    size_t contextPhaseParamsSize
                        = executor::Serialization::serializedSize(response.getResult().contextPhaseParams.value());
                    std::vector<int64_t> contextPhaseParamsShape{1, static_cast<int64_t>(contextPhaseParamsSize)};
                    TRITONSERVER_DataType contextPhaseParamsType = TRITONSERVER_TYPE_UINT8;
                    auto contextPhaseParamsBuffer = utils::getResponseBuffer<uint8_t>(tritonResponse,
                        contextPhaseParamsShape, contextPhaseParamsType, OutputFieldsNames::contextPhaseParams);

                    std::stringbuf contextPhaseSerializationBuffer(std::ios_base::out | std::ios_base::in);
                    contextPhaseSerializationBuffer.pubsetbuf(
                        reinterpret_cast<char*>(contextPhaseParamsBuffer), contextPhaseParamsSize);
                    std::ostream os(&contextPhaseSerializationBuffer);
                    executor::Serialization::serialize(response.getResult().contextPhaseParams.value(), os);
                }
                else
                {
                    TLLM_THROW("contextParams must be present in the response");
                }
            }

            // Add token count outputs if requested
            if (requestData.returnNumInputTokens
                && requestData.outputNames.count(OutputFieldsNames::numInputTokens) > 0)
            {
                std::vector<int64_t> inputTokenCountShape{1, 1};
                auto inputTokenCountType = TRITONSERVER_TYPE_INT32;
                auto inputTokenCountBuffer = utils::getResponseBuffer<int32_t>(
                    tritonResponse, inputTokenCountShape, inputTokenCountType, OutputFieldsNames::numInputTokens);
                std::vector<int32_t> inputTokenCountVec = {static_cast<int32_t>(requestData.inputTokensSize)};
                utils::flatten<int32_t>(inputTokenCountVec, inputTokenCountBuffer, inputTokenCountShape);
            }

            if (requestData.returnNumOutputTokens
                && requestData.outputNames.count(OutputFieldsNames::numOutputTokens) > 0)
            {
                std::vector<int64_t> outputTokenCountShape{1, 1};
                auto outputTokenCountType = TRITONSERVER_TYPE_INT32;
                auto outputTokenCountBuffer = utils::getResponseBuffer<int32_t>(
                    tritonResponse, outputTokenCountShape, outputTokenCountType, OutputFieldsNames::numOutputTokens);
                std::vector<int32_t> outputTokenCountVec = {static_cast<int32_t>(outputTokensSize)};
                utils::flatten<int32_t>(outputTokenCountVec, outputTokenCountBuffer, outputTokenCountShape);
            }

            if (requestData.returnPerfMetrics)
            {
                auto processStats = [&](std::string const& fieldName, auto const& value)
                {
                    std::vector<int64_t> shape{1, 1};

                    if constexpr (std::is_same_v<decltype(value), int32_t const&>)
                    {
                        auto type = TRITONSERVER_TYPE_INT32;
                        auto buffer = utils::getResponseBuffer<int32_t>(tritonResponse, shape, type, fieldName);
                        std::vector<int32_t> vec = {value};
                        utils::flatten<int32_t>(vec, buffer, shape);
                    }
                    else if constexpr (std::is_same_v<decltype(value), float const&>)
                    {
                        auto type = TRITONSERVER_TYPE_FP32;
                        auto buffer = utils::getResponseBuffer<float>(tritonResponse, shape, type, fieldName);
                        std::vector<float> vec = {value};
                        utils::flatten<float>(vec, buffer, shape);
                    }
                    else if constexpr (std::is_same_v<decltype(value), executor::RequestPerfMetrics::TimePoint const&>)
                    {
                        auto type = TRITONSERVER_TYPE_INT64;
                        auto buffer = utils::getResponseBuffer<int64_t>(tritonResponse, shape, type, fieldName);
                        auto duration = value.time_since_epoch();
                        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
                        std::vector<int64_t> vec = {nanoseconds.count()};
                        utils::flatten<int64_t>(vec, buffer, shape);
                    }
                    else
                    {
                        TLLM_THROW("Unexpected type for field %s", fieldName.c_str());
                    }
                };
                if (result.requestPerfMetrics.has_value())
                {
                    auto const& kvStats = result.requestPerfMetrics.value().kvCacheMetrics;
                    processStats(OutputFieldsNames::kvCacheAllocNewBlocks, kvStats.numNewAllocatedBlocks);
                    processStats(OutputFieldsNames::kvCacheReusedBlocks, kvStats.numReusedBlocks);
                    processStats(OutputFieldsNames::kvCacheAllocTotalBlocks, kvStats.numTotalAllocatedBlocks);

                    auto const& timingStats = result.requestPerfMetrics.value().timingMetrics;
                    processStats(OutputFieldsNames::arrivalTime, timingStats.arrivalTime);
                    processStats(OutputFieldsNames::firstScheduledTime, timingStats.firstScheduledTime);
                    processStats(OutputFieldsNames::firstTokenTime, timingStats.firstTokenTime);
                    processStats(OutputFieldsNames::lastTokenTime, timingStats.lastTokenTime);

                    auto const& specDecodingStats = result.requestPerfMetrics.value().speculativeDecoding;
                    processStats(OutputFieldsNames::acceptanceRate, specDecodingStats.acceptanceRate);
                    processStats(
                        OutputFieldsNames::totalAcceptedDraftTokens, specDecodingStats.totalAcceptedDraftTokens);
                    processStats(OutputFieldsNames::totalDraftTokens, specDecodingStats.totalDraftTokens);
                }
            }
        }
        else
        {
            isFinal = true;
            std::string errMsg = "Executor failed process requestId " + std::to_string(response.getRequestId())
                + " due to the following error: " + response.getErrorMsg();
            error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errMsg.c_str());
        }
    }
    catch (std::exception const& e)
    {
        // In case of error while processing response, return response with error
        isFinal = true;
        std::string errMsg = "Error encountered while populating response: " + std::string(e.what());
        error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errMsg.c_str());
    }

    return {tritonResponse, isFinal, error, outputTokensSize};
}

void ModelInstanceState::WaitForResponse()
{
    while (!mStopWaitForResponse)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = mExecutor->awaitResponses(waitTime);
        uint64_t compute_end_ns{0};
        SET_TIMESTAMP(compute_end_ns);

        for (auto const& response : responses)
        {
            auto requestId = response.getRequestId();
            RequestData requestData;
            {
                std::lock_guard<std::mutex> lock(mRequestIdToRequestDataMutex);
                if (!mRequestIdToRequestData.count(requestId))
                {
                    TLLM_LOG_ERROR("Unexpected response for a request ID that is not active");
                    continue;
                }
                requestData = mRequestIdToRequestData[requestId];
            }

            auto factory = requestData.factory;

            auto [tritonResponse, isFinal, error, outputTokensSize]
                = fillTritonResponse(factory, response, requestData);
            {
                std::lock_guard<std::mutex> lock(mRequestIdToRequestDataMutex);
                if (!mRequestIdToRequestData.count(requestId))
                {
                    TLLM_LOG_ERROR("Unexpected response for a request ID that is not active");
                    continue;
                }
                mRequestIdToRequestData[requestId].outputTokensSize += outputTokensSize;
                requestData = mRequestIdToRequestData[requestId];
            }

            if (isFinal)
            {
                std::lock_guard<std::mutex> lock(mRequestIdToRequestDataMutex);
                bool signalFinal = requestData.pendingBatchedRequestIds->size() == 1;
                LOG_IF_ERROR(TRITONBACKEND_ResponseSend(
                                 tritonResponse, signalFinal ? TRITONSERVER_RESPONSE_COMPLETE_FINAL : 0, error),
                    "Cannot send response");
                if (requestData.tritonRequestId != "")
                {
                    auto itr = mTritonRequestIdToRequestIds.find(requestData.tritonRequestId);
                    if (itr != mTritonRequestIdToRequestIds.end())
                    {
                        auto& pendingBatchedRequestIds = itr->second;
                        pendingBatchedRequestIds.erase(requestId);
                        if (pendingBatchedRequestIds.size() == 0)
                        {
                            mTritonRequestIdToRequestIds.erase(requestData.tritonRequestId);
                        }
                    }
                }
                auto& pendingBatchedRequestIds = *requestData.pendingBatchedRequestIds;
                pendingBatchedRequestIds.erase(requestId);
                if (pendingBatchedRequestIds.size() == 0)
                {
                    if (!requestData.excludeInputInOutput && !requestData.streaming)
                    {
                        // Need to do this as initial tokens sent by the executor are the input tokens IF NOT Streaming
                        requestData.outputTokensSize -= (requestData.inputTokensSize * requestData.beamWidth);
                    }
                    requestData.timestamps.compute_end_ns = compute_end_ns;
                    LOG_IF_ERROR(reportBaseMetrics(requestData, error), "Error reporting metrics");
                    LOG_IF_ERROR(reportCustomMetrics(requestData.inputTokensSize, requestData.outputTokensSize, error),
                        "Error reporting custom metrics per request");
                    LOG_IF_ERROR(
                        TRITONBACKEND_RequestRelease(requestData.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL),
                        "Cannot release request");

                    LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory), "Cannot delete response factory");
                }
                mRequestIdToRequestData.erase(requestId);
            }
            else
            {
                LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, 0, error), "Cannot send response");
            }
        }
    }
}

void ModelInstanceState::WaitForStats()
{
    while (!mStopWaitForStats)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(mInstanceSpecificConfig.statsCheckPeriodMs));
        auto stats = mExecutor->getLatestIterationStats();
        for (auto const& stat : stats)
        {
            std::string statJson = "{";
            statJson.append("\"Active Request Count\":" + std::to_string(stat.numActiveRequests) + ",");
            statJson.append("\"Iteration Counter\":" + std::to_string(stat.iter) + ",");
            statJson.append("\"Max Request Count\":" + std::to_string(stat.maxNumActiveRequests) + ",");
            statJson.append("\"Runtime CPU Memory Usage\":" + std::to_string(stat.cpuMemUsage) + ",");
            statJson.append("\"Runtime GPU Memory Usage\":" + std::to_string(stat.gpuMemUsage) + ",");
            statJson.append("\"Runtime Pinned Memory Usage\":" + std::to_string(stat.pinnedMemUsage) + ",");
            statJson.append("\"Timestamp\":" + ("\"" + stat.timestamp + "\"") + ",");

            if (stat.inflightBatchingStats.has_value())
            {
                auto const& modelStats = stat.inflightBatchingStats.value();
                statJson.append("\"Context Requests\":" + std::to_string(modelStats.numContextRequests) + ",");
                statJson.append("\"Generation Requests\":" + std::to_string(modelStats.numGenRequests) + ",");
                statJson.append("\"MicroBatch ID\":" + std::to_string(modelStats.microBatchId) + ",");
                statJson.append("\"Paused Requests\":" + std::to_string(modelStats.numPausedRequests) + ",");
                statJson.append("\"Scheduled Requests\":" + std::to_string(modelStats.numScheduledRequests) + ",");
                statJson.append("\"Total Context Tokens\":" + std::to_string(modelStats.numCtxTokens) + ",");
                statJson.append("\"Waiting Requests\":"
                    + std::to_string(stat.numActiveRequests - modelStats.numScheduledRequests) + ",");
            }
            else if (stat.staticBatchingStats.has_value())
            {
                auto const& modelStats = stat.staticBatchingStats.value();
                statJson.append("\"Context Requests\":" + std::to_string(modelStats.numContextRequests) + ",");
                statJson.append("\"Scheduled Requests\":" + std::to_string(modelStats.numScheduledRequests) + ",");
                statJson.append("\"Total Context Tokens\":" + std::to_string(modelStats.numCtxTokens) + ",");
                statJson.append("\"Total Generation Tokens\":" + std::to_string(modelStats.numGenTokens) + ",");
                statJson.append("\"Empty Generation Slots\":" + std::to_string(modelStats.emptyGenSlots) + ",");
                statJson.append("\"Waiting Requests\":"
                    + std::to_string(stat.numActiveRequests - modelStats.numScheduledRequests) + ",");
            }
            else
            {
                TLLM_LOG_ERROR("Missing stats");
                continue;
            }

            if (stat.kvCacheStats.has_value())
            {
                auto const& kvStats = stat.kvCacheStats.value();
                statJson.append("\"Free KV cache blocks\":" + std::to_string(kvStats.freeNumBlocks) + ",");
                statJson.append("\"Max KV cache blocks\":" + std::to_string(kvStats.maxNumBlocks) + ",");
                statJson.append("\"Tokens per KV cache block\":" + std::to_string(kvStats.tokensPerBlock) + ",");
                statJson.append("\"Used KV cache blocks\":" + std::to_string(kvStats.usedNumBlocks) + ",");
                statJson.append("\"Reused KV cache blocks\":" + std::to_string(kvStats.reusedBlocks) + ",");
                // Calculate and append the used KV cache block fraction.
                double fraction = 0.0;
                if (static_cast<double>(kvStats.maxNumBlocks) > 0.0)
                {
                    fraction = static_cast<double>(kvStats.usedNumBlocks) / static_cast<double>(kvStats.maxNumBlocks);
                }
                statJson.append("\"Fraction used KV cache blocks\":" + std::to_string(fraction) + ",");
            }

            // requestStats is a list where each item is associated with an iteration,
            // currently the metrics related to request stats only concern with aggregated
            // results so that we can retrieve request stats and process all of them
            // whenever metrics is to be reported.
            double totalKvCacheTransferMS = 0;
            size_t requestCount = 0;
            if (!mIsOrchestratorMode)
            {
                // TODO: implement orchestrator mode support: https://jirasw.nvidia.com/browse/TRTLLM-1581
                auto requestStats = mExecutor->getLatestRequestStats();
                for (auto const& iteration : requestStats)
                {
                    for (auto const& request : iteration.requestStats)
                    {
                        // only check and aggregate results when request is completed
                        if (request.stage == executor::RequestStage::kGENERATION_COMPLETE)
                        {
                            if (request.disServingStats.has_value())
                            {
                                auto const& disServingStats = request.disServingStats.value();
                                totalKvCacheTransferMS += disServingStats.kvCacheTransferMS;
                                requestCount++;
                            }
                        }
                    }
                }
            }
            statJson.append("\"KV cache transfer time\":" + std::to_string(totalKvCacheTransferMS) + ",");
            statJson.append("\"Request count\":" + std::to_string(requestCount) + ",");

            statJson.back() = '}';

            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, statJson.c_str());
#ifdef TRITON_ENABLE_METRICS
            LOG_IF_ERROR(custom_metrics_reporter_->UpdateCustomMetrics(statJson), "Failed updating TRT LLM statistics");
#endif
        }
    }
}

void ModelInstanceState::WaitForCancel()
{
    while (!mStopWaitForCancel)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(mInstanceSpecificConfig.cancellationCheckPeriodMs));
        std::lock_guard<std::mutex> lock(mRequestIdToRequestDataMutex);
        for (auto const& pair : mRequestIdToRequestData)
        {
            auto const& requestId = pair.first;
            auto const& requestData = pair.second;
            bool isCancelled = false;
            LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryIsCancelled(requestData.factory, &isCancelled),
                "Failed to query factory status");
            if (isCancelled)
            {
                mExecutor->cancelRequest(requestId);
            }
        }
    }
}

} // namespace triton::backend::inflight_batcher_llm
