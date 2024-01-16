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
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <NvInfer.h>
#include <chrono>
#include <cxxopts.hpp>
#include <sstream>
#include <string>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace trt = nvinfer1;

namespace
{
void benchmarkGptSession(std::string const& modelName, std::filesystem::path const& dataPath,
    std::vector<int> const& batchSizes, int beamWidth, std::vector<std::vector<int>> const& inOutLen,
    std::shared_ptr<nvinfer1::ILogger> const& logger, int warmUp, int numRuns, int duration,
    GptSession::Config& sessionConfig, bool cudaGraphMode, bool printAllLogits, bool disableForceMaxTokens)
{
    std::string modelNameHyphen = modelName;
    std::filesystem::path jsonFileName = dataPath / "config.json";
    auto const json = GptJsonConfig::parse(jsonFileName);
    auto const modelConfig = json.getModelConfig();
    auto const inputPacked = modelConfig.usePackedInput();
    SizeType deviceCount{0};
    TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    auto const worldConfig = WorldConfig::mpi(deviceCount, json.getTensorParallelism(), json.getPipelineParallelism());
    auto const enginePath = dataPath / json.engineFilename(worldConfig, modelNameHyphen);
    auto const dtype = modelConfig.getDataType();
    auto const useHalf = (dtype == nvinfer1::DataType::kHALF);

    SamplingConfig samplingConfig{beamWidth};
    samplingConfig.temperature = std::vector{1.0f};
    samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
    samplingConfig.topK = std::vector{1};
    samplingConfig.topP = std::vector{0.0f};

    auto const maxBatchSize = *std::max_element(batchSizes.begin(), batchSizes.end());
    sessionConfig.maxBatchSize = maxBatchSize;
    sessionConfig.maxBeamWidth = beamWidth;
    sessionConfig.decoderPerRequest = false;
    sessionConfig.cudaGraphMode = cudaGraphMode;

    for (auto inOut : inOutLen)
    {
        auto const maxInputLength = inOut[0];
        auto const maxNewTokens = inOut[1];

        sessionConfig.maxSequenceLength = maxInputLength + maxNewTokens;
        samplingConfig.minLength = std::vector{disableForceMaxTokens ? 1 : maxNewTokens};

        GptSession session{sessionConfig, modelConfig, worldConfig, enginePath.string(), logger};

        // Use bufferManager for copying data to and from the GPU
        auto& bufferManager = session.getBufferManager();

        auto constexpr endId = 50256;
        auto constexpr padId = 50256;

        auto& memoryCounter = MemoryCounters::getInstance();
        TLLM_LOG_INFO(memoryCounter.toString());

        for (auto const batchSize : batchSizes)
        {
            try
            {
                TLLM_LOG_INFO(memoryCounter.toString());

                std::vector<SizeType> inputLenghtsHost(batchSize, maxInputLength);
                auto inputLenghts
                    = bufferManager.copyFrom(inputLenghtsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

                // copy inputs and wrap into shared_ptr
                GenerationInput::TensorPtr inputIds;
                std::vector<int32_t> inputsHost(batchSize * maxInputLength);
                srand(time(0));
                for (int i = 0; i < inputsHost.size(); i++)
                {
                    inputsHost[i] = rand() % modelConfig.getVocabSizePadded(worldConfig.getSize());
                }

                if (inputPacked)
                {
                    inputIds = bufferManager.copyFrom(
                        inputsHost, ITensor::makeShape({batchSize * maxInputLength}), MemoryType::kGPU);
                }
                else
                {
                    inputIds = bufferManager.copyFrom(
                        inputsHost, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);
                }

                TLLM_LOG_INFO(memoryCounter.toString());

                GenerationInput generationInput{
                    endId, padId, std::move(inputIds), std::move(inputLenghts), inputPacked};

                // runtime will allocate memory for output if this tensor is empty
                GenerationOutput generationOutput{
                    bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
                    bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};

                if (session.getModelConfig().computeContextLogits())
                {
                    generationOutput.contextLogits
                        = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
                }
                if (session.getModelConfig().computeGenerationLogits())
                {
                    generationOutput.generationLogits
                        = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
                }

                TLLM_LOG_INFO(memoryCounter.toString());

                for (auto r = 0; r < warmUp; ++r)
                {
                    SizeType numSteps = 0;
                    generationOutput.onTokenGenerated
                        = [&numSteps, maxNewTokens](GenerationOutput::TensorPtr const& outputIds, SizeType step,
                              bool finished) { ++numSteps; };
                    session.generate(generationOutput, generationInput, samplingConfig);
                    bufferManager.getStream().synchronize();
                }
                cudaDeviceSynchronize();

                TLLM_LOG_INFO(memoryCounter.toString());

                int iterIdx = 0;
                float curDuration = 0;
                while (iterIdx < numRuns || curDuration / 1000 < duration)
                {
                    auto const start = std::chrono::steady_clock::now();
                    SizeType numSteps = 0;
                    generationOutput.onTokenGenerated
                        = [&numSteps, maxNewTokens](GenerationOutput::TensorPtr const& outputIds, SizeType step,
                              bool finished) { ++numSteps; };
                    session.generate(generationOutput, generationInput, samplingConfig);
                    bufferManager.getStream().synchronize();
                    auto const end = std::chrono::steady_clock::now();

                    iterIdx += 1;
                    curDuration += std::chrono::duration<float, std::milli>(end - start).count();
                }

                TLLM_LOG_INFO(memoryCounter.toString());

                printf("Benchmarking done. Iteration: %d, duration: %.2f sec.\n", iterIdx, curDuration / 1000);

                if (worldConfig.getRank() == 0)
                {
                    auto const averageLatency = curDuration / iterIdx;
                    float const tokensPerSec = batchSize * maxNewTokens / (averageLatency / 1000);
                    printf(
                        "[BENCHMARK] batch_size %d input_length %d output_length %d latency(ms) %.2f tokensPerSec "
                        "%.2f\n",
                        batchSize, maxInputLength, maxNewTokens, averageLatency, tokensPerSec);
                }

                // logits are store in last rank
                if (worldConfig.getRank() == worldConfig.getSize() - 1)
                {
                    if (session.getModelConfig().computeContextLogits() && printAllLogits)
                    {
                        std::cout << "generationOutput.contextLogits.shape: "
                                  << generationOutput.contextLogits->getShape()
                                  << std::endl; // (batchsize, prompt_len, vocabsize)
                        std::cout << "generationOutput.contextLogits: " << *generationOutput.contextLogits << std::endl;
                    }

                    if (session.getModelConfig().computeGenerationLogits() && printAllLogits)
                    {
                        std::cout << "generationOutput.generationLogits.shape: "
                                  << generationOutput.generationLogits->getShape()
                                  << std::endl; // (batchsize, beamwidth, maxNewTokens, vocabsize)
                        generationOutput.generationLogits->reshape(ITensor::makeShape({batchSize * beamWidth,
                            maxNewTokens, modelConfig.getVocabSizePadded(worldConfig.getSize())}));

                        std::cout << "generationOutput.generationLogits: " << *generationOutput.generationLogits
                                  << std::endl;
                    }
                }
            }
            catch (std::runtime_error& e)
            {
                std::size_t found = std::string(e.what()).find("out of memory");

                // Unexpected error; rethrow
                if (found == std::string::npos)
                {
                    throw;
                }

                // We can ignore the OOM exception and continue the rest of the benchmark
                if (worldConfig.getRank() == 0)
                {
                    TLLM_LOG_EXCEPTION(e);
                    printf(
                        "[BENCHMARK] batch_size %d input_length %d output_length %d latency(ms) N/A tokensPerSec N/A\n",
                        batchSize, maxInputLength, maxNewTokens);
                }
                continue;
            }
        }
        TLLM_LOG_INFO(memoryCounter.toString());
    }
}

} // namespace

int main(int argc, char* argv[])
{
    cxxopts::Options options(
        "TensorRT-LLM C++ Runtime Benchmark", "TensorRT-LLM C++ Runtime Benchmark for GPT and GPT-like models.");
    options.add_options()("h,help", "Print usage");
    options.add_options()(
        "m,model", "Model name specified for engines.", cxxopts::value<std::string>()->default_value("gpt_350m"));
    options.add_options()("engine_dir", "Directory that store the engines.", cxxopts::value<std::string>());
    options.add_options()("batch_size",
        "Specify batch size(s) you want to benchmark. Multiple batch sizes can be separated by \";\", example: "
        "\"1;8;64\".",
        cxxopts::value<std::string>()->default_value("8"));
    options.add_options()(
        "beam_width", "Specify beam width you want to benchmark.", cxxopts::value<int>()->default_value("1"));
    options.add_options()("input_output_len",
        "Specify input-output length(s) you want to benchmark. Multiple input lengths can be separated by \";\", "
        "example: \"60,20;128,20\".",
        cxxopts::value<std::string>()->default_value("128,20"));

    options.add_options()("log_level", "Choose log level between verbose/info/warning/error/internal_error.",
        cxxopts::value<std::string>()->default_value("error"));
    options.add_options()(
        "warm_up", "Specify warm up iterations before benchmark starts.", cxxopts::value<int>()->default_value("2"));
    options.add_options()("num_runs", "Minimal number of iterations to run during benchmarking.",
        cxxopts::value<int>()->default_value("10"));
    options.add_options()("duration", "Minimal duration of iterations to measure in seconds.",
        cxxopts::value<int>()->default_value("60"));

    options.add_options()("ctx_micro_batch_size", "Batch size for context phase.", cxxopts::value<int>());
    options.add_options()("gen_micro_batch_size", "Batch size for generation phase.", cxxopts::value<int>());
    options.add_options()("max_attention_window", "Max kv cache length per sequence.", cxxopts::value<int>());
    options.add_options()("max_tokens_in_paged_kvcache", "Max tokens in paged K-V Cache.", cxxopts::value<int>());
    options.add_options()("sink_token_len", "Sink token length in kv cache per sequence.", cxxopts::value<int>());
    options.add_options()(
        "kv_cache_free_gpu_mem_fraction", "K-V Cache Free Gpu Mem Fraction.", cxxopts::value<float>());

    options.add_options()("enable_cuda_graph", "Execute GPT session with CUDA graph.");
    options.add_options()("print_all_logits", "Print all context and generation logits.");
    options.add_options()("disable_force_max_tokens", "Disable force the engine generating new max_tokens.");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    // Argument: Engine directory
    if (!result.count("engine_dir"))
    {
        std::cout << options.help() << std::endl;
        TLLM_LOG_ERROR("Please specify engine directory.");
        return 1;
    }

    // Argument: Batch sizes
    std::istringstream ssBatchSizesArg;
    ssBatchSizesArg.str(result["batch_size"].as<std::string>());
    std::vector<int> batchSizes;
    for (std::string token; std::getline(ssBatchSizesArg, token, ';');)
    {
        batchSizes.push_back(std::stoi(token));
    }

    // Argument: beam width
    auto const beamWidth = result["beam_width"].as<int>();

    // Argument: Input-output lengths
    std::istringstream ssInOutLenArg;
    ssInOutLenArg.str(result["input_output_len"].as<std::string>());
    std::vector<std::vector<int>> inOutLen;
    for (std::string token; std::getline(ssInOutLenArg, token, ';');)
    {
        std::istringstream ssTmp(token);
        std::vector<int> inOut;
        for (std::string t; std::getline(ssTmp, t, ',');)
        {
            inOut.push_back(std::stoi(t));
        }
        inOutLen.push_back(inOut);
    }

    // Argument: Log level
    auto logger = std::make_shared<TllmLogger>();
    auto const logLevel = result["log_level"].as<std::string>();
    if (logLevel == "verbose")
    {
        logger->setLevel(trt::ILogger::Severity::kVERBOSE);
    }
    else if (logLevel == "info")
    {
        logger->setLevel(trt::ILogger::Severity::kINFO);
    }
    else if (logLevel == "warning")
    {
        logger->setLevel(trt::ILogger::Severity::kWARNING);
    }
    else if (logLevel == "error")
    {
        logger->setLevel(trt::ILogger::Severity::kERROR);
    }
    else if (logLevel == "internal_error")
    {
        logger->setLevel(trt::ILogger::Severity::kINTERNAL_ERROR);
    }
    else
    {
        TLLM_LOG_ERROR("Unexpected log level: " + logLevel);
        return 1;
    }

    GptSession::Config sessionConfig{0, 0, 0};
    // Argument: Batch size for context phase
    if (result.count("ctx_micro_batch_size"))
    {
        sessionConfig.ctxMicroBatchSize = result["ctx_micro_batch_size"].as<int>();
    }
    // Argument: Batch size for generation phase
    if (result.count("gen_micro_batch_size"))
    {
        sessionConfig.genMicroBatchSize = result["gen_micro_batch_size"].as<int>();
    }
    // Argument: Max tokens in paged K-V Cache
    if (result.count("max_tokens_in_paged_kvcache"))
    {
        sessionConfig.kvCacheConfig.maxTokens = result["max_tokens_in_paged_kvcache"].as<int>();
    }
    // Argument: Max KV Cache Length
    if (result.count("max_attention_window"))
    {
        sessionConfig.kvCacheConfig.maxAttentionWindow = result["max_attention_window"].as<int>();
    }
    // Argument: Sink token length
    if (result.count("sink_token_len"))
    {
        sessionConfig.kvCacheConfig.sinkTokenLength = result["sink_token_len"].as<int>();
    }
    // Argument: K-V Cache Free Gpu Mem Fraction
    if (result.count("kv_cache_free_gpu_mem_fraction"))
    {
        sessionConfig.kvCacheConfig.freeGpuMemoryFraction = result["kv_cache_free_gpu_mem_fraction"].as<float>();
    }

    // Argument: Enable CUDA graph
    auto enableCudaGraph = result.count("enable_cuda_graph") > 0;
    auto printAllLogits = result.count("print_all_logits") > 0;
    auto disableForceMaxTokens = result.count("disable_force_max_tokens") > 0;

    initTrtLlmPlugins(logger.get());

    try
    {
        benchmarkGptSession(result["model"].as<std::string>(), result["engine_dir"].as<std::string>(), batchSizes,
            beamWidth, inOutLen, logger, result["warm_up"].as<int>(), result["num_runs"].as<int>(),
            result["duration"].as<int>(), sessionConfig, enableCudaGraph, printAllLogits, disableForceMaxTokens);
    }
    catch (const std::exception& e)
    {
        TLLM_LOG_ERROR(e.what());
        return 1;
    }
    return 0;
}
