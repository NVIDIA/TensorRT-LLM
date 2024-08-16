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

/*****************************************************************************
 *
 * GptSession is going to be deprecated soon.
 * Please do not add new functionality in this file!
 *
 *****************************************************************************/

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <NvInfer.h>
#include <atomic>
#include <chrono>
#include <cuda_profiler_api.h>
#include <cxxopts.hpp>
#include <future>
#include <sstream>
#include <string>
#include <thread>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace tmpi = tensorrt_llm::mpi;
namespace trt = nvinfer1;

namespace
{
size_t monitorMemory(std::atomic_bool& done)
{
    // A simple memory monitor function that monitors peak GPU memory usage
    size_t peakMem = 0;
    while (!done)
    {
        auto const [freeMem, totalMem] = tc::getDeviceMemoryInfo(false);
        if (totalMem - freeMem > peakMem)
        {
            peakMem = totalMem - freeMem;
        }
        // Sleep for 50 ms to avoid spamming getDeviceMemoryInfo to reduce overhead
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return peakMem;
}

void benchmarkGptSession(std::filesystem::path const& dataPath, std::vector<int> const& batchSizes, int beamWidth,
    std::vector<std::vector<int>> const& inOutLen, std::shared_ptr<nvinfer1::ILogger> const& logger, int warmUp,
    int numRuns, int duration, GptSession::Config& sessionConfig, bool cudaGraphMode, bool printAllLogits,
    bool disableForceMaxTokens, bool dumpLayerInfo, bool dumpProfile, std::vector<float> const& gpuWeightsPercents)
{
    std::filesystem::path jsonFileName = dataPath / "config.json";
    auto const json = GptJsonConfig::parse(jsonFileName);
    auto const modelConfig = json.getModelConfig();
    auto const inputPacked = modelConfig.usePackedInput();
    auto const worldConfig
        = WorldConfig::mpi(json.getGpusPerNode(), json.getTensorParallelism(), json.getPipelineParallelism());
    auto& comm = COMM_SESSION;
    auto const enginePath = dataPath / json.engineFilename(worldConfig);
    auto const dtype = modelConfig.getDataType();
    auto const maxNumTokens = modelConfig.getMaxNumTokens();
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

    struct RuntimeConfig
    {
        int inLen;
        int maxNewTokens;
        float gpuWeightsPercent;
    };

    std::vector<RuntimeConfig> benchmarkConfigs;
    for (auto inOut : inOutLen)
    {
        for (auto gpuWeightsPercent : gpuWeightsPercents)
        {
            benchmarkConfigs.push_back({inOut[0], inOut[1], gpuWeightsPercent});
        }
    }

    for (auto const& bc : benchmarkConfigs)
    {
        auto const maxInputLength = bc.inLen;
        auto const maxNewTokens = bc.maxNewTokens;

        sessionConfig.maxSequenceLength = maxInputLength + maxNewTokens;
        sessionConfig.gpuWeightsPercent = bc.gpuWeightsPercent;
        samplingConfig.minLength = std::vector{disableForceMaxTokens ? 1 : maxNewTokens};

        GptSession session{sessionConfig, modelConfig, worldConfig, enginePath.string(), logger};

        // Use bufferManager for copying data to and from the GPU
        auto& bufferManager = session.getBufferManager();

        auto constexpr endId = 50256;
        auto constexpr padId = 50256;

        auto& memoryCounter = MemoryCounters::getInstance();
        TLLM_LOG_INFO(memoryCounter.toString());
        std::atomic_bool done;
        for (auto const batchSize : batchSizes)
        {

            if (inputPacked && maxNumTokens != std::nullopt)
            {
                TLLM_CHECK_WITH_INFO(maxBatchSize * maxInputLength <= maxNumTokens.value(),
                    "The engine is built with remove_input_padding=True and max_num_tokens=%d, while trying to "
                    "benchmark on %d tokens",
                    maxNumTokens.value(), maxBatchSize * maxInputLength);
            }
            done = false;
            auto peakMemFuture = std::async(&monitorMemory, std::ref(done));
            size_t peakMem;
            try
            {
                TLLM_LOG_INFO(memoryCounter.toString());

                std::vector<SizeType32> inputLengthsHost(batchSize, maxInputLength);
                auto inputLengths
                    = bufferManager.copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

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
                    endId, padId, std::move(inputIds), std::move(inputLengths), inputPacked};

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
                    SizeType32 numSteps = 0;
                    generationOutput.onTokenGenerated
                        = [&numSteps, maxNewTokens](GenerationOutput::TensorPtr const& outputIds, SizeType32 step,
                              bool finished) { ++numSteps; };
                    session.generate(generationOutput, generationInput, samplingConfig);
                    bufferManager.getStream().synchronize();
                }
                cudaDeviceSynchronize();

                TLLM_LOG_INFO(memoryCounter.toString());

                int iterIdx = 0;
                float curDuration = 0;
                std::vector<float> latencies;
                std::vector<float> generationTimes;
                auto generationProfiler = std::make_shared<GptSession::GenerationProfiler>();
                cudaProfilerStart();
                while (iterIdx < numRuns)
                {
                    auto const start = std::chrono::steady_clock::now();
                    SizeType32 numSteps = 0;
                    generationOutput.onTokenGenerated
                        = [&numSteps, maxNewTokens](GenerationOutput::TensorPtr const& outputIds, SizeType32 step,
                              bool finished) { ++numSteps; };
                    session.generate(generationOutput, generationInput, samplingConfig, generationProfiler);
                    bufferManager.getStream().synchronize();
                    auto const end = std::chrono::steady_clock::now();

                    iterIdx += 1;
                    float latency = std::chrono::duration<float, std::milli>(end - start).count();
                    curDuration += latency;
                    latencies.emplace_back(latency);
                    generationTimes.emplace_back(generationProfiler->getElapsedTimeMs());

                    bool durationLimitReached{curDuration / 1000 >= duration};
                    if (worldConfig.getSize() > 1)
                    {
                        bool result{false};
                        comm.allreduce(&durationLimitReached, &result, 1, tmpi::MpiType::kBOOL, tmpi::MpiOp::LOR);
                        durationLimitReached = result;
                    }
                    if (durationLimitReached)
                    {
                        break;
                    }
                }
                cudaProfilerStop();

                TLLM_LOG_INFO(memoryCounter.toString());
                done = true;
                peakMemFuture.wait();
                peakMem = peakMemFuture.get();
                if (dumpLayerInfo)
                {
                    printf("Dump layer information:\n");
                    printf("%s\n",
                        session.getEngineInspector().getEngineInformation(nvinfer1::LayerInformationFormat::kONELINE));
                }

                printf("Benchmarking done. Iteration: %d, duration: %.2f sec.\n", iterIdx, curDuration / 1000);

                // Print latencies to make it easier to identify perf stability issue.
                printf("Latencies: [");
                constexpr int maxPrintedLatencies{20};
                for (int i = 0; i < latencies.size(); ++i)
                {
                    printf("%.2f", latencies[i]);
                    if (i == latencies.size() - 1)
                    {
                        printf("]\n");
                    }
                    else if (latencies.size() > maxPrintedLatencies && i == (maxPrintedLatencies / 2 - 1))
                    {
                        printf(" ... ");
                        i = latencies.size() - maxPrintedLatencies / 2;
                    }
                    else
                    {
                        printf(", ");
                    }
                }

                if (worldConfig.getRank() == 0)
                {
                    auto const averageLatency = curDuration / iterIdx;
                    float const tokensPerSec = batchSize * maxNewTokens / (averageLatency / 1000);
                    auto const avgGenerationTime
                        = std::reduce(generationTimes.begin(), generationTimes.end(), 0.0f) / generationTimes.size();
                    float const generationTokensPerSec = batchSize * maxNewTokens / (avgGenerationTime / 1000);
                    // convert to GB
                    float const peakMemGB = peakMem / 1e9;
                    printf(
                        "[BENCHMARK] batch_size %d input_length %d output_length %d latency(ms) %.2f tokensPerSec "
                        "%.2f generation_time(ms) %.2f generationTokensPerSec %.2f gpu_peak_mem(gb) %.2f\n",
                        batchSize, maxInputLength, maxNewTokens, averageLatency, tokensPerSec, avgGenerationTime,
                        generationTokensPerSec, peakMemGB);
                }

                // logits are store in last rank
                if (worldConfig.getRank() == worldConfig.getSize() - 1)
                {
                    if (session.getModelConfig().computeContextLogits() && printAllLogits)
                    {
                        std::cout << "generationOutput.contextLogits.shape: "
                                  << generationOutput.contextLogits->getShape()
                                  << std::endl; // (batch_size, prompt_len, vocab_size)
                        std::cout << "generationOutput.contextLogits: " << *generationOutput.contextLogits << std::endl;
                    }

                    if (session.getModelConfig().computeGenerationLogits() && printAllLogits)
                    {
                        std::cout << "generationOutput.generationLogits.shape: "
                                  << generationOutput.generationLogits->getShape()
                                  << std::endl; // (batch_size, beam_width, maxNewTokens, vocab_size)
                        generationOutput.generationLogits->reshape(ITensor::makeShape({batchSize * beamWidth,
                            maxNewTokens, modelConfig.getVocabSizePadded(worldConfig.getSize())}));

                        std::cout << "generationOutput.generationLogits: " << *generationOutput.generationLogits
                                  << std::endl;
                    }
                }
                // Do per-layer profiling after normal benchmarking to avoid introducing perf overhead.
                if (dumpProfile)
                {
                    session.setLayerProfiler();
                    iterIdx = 0;

                    while (iterIdx < numRuns)
                    {
                        auto const start = std::chrono::steady_clock::now();
                        SizeType32 numSteps = 0;
                        generationOutput.onTokenGenerated
                            = [&numSteps, maxNewTokens](GenerationOutput::TensorPtr const& outputIds, SizeType32 step,
                                  bool finished) { ++numSteps; };
                        session.generate(generationOutput, generationInput, samplingConfig, generationProfiler);
                        bufferManager.getStream().synchronize();
                        auto const end = std::chrono::steady_clock::now();

                        iterIdx += 1;
                        float latency = std::chrono::duration<float, std::milli>(end - start).count();
                        curDuration += latency;
                        latencies.emplace_back(latency);
                        generationTimes.emplace_back(generationProfiler->getElapsedTimeMs());

                        bool durationLimitReached{curDuration / 1000 >= duration};
                        if (worldConfig.getSize() > 1)
                        {
                            bool result{false};
                            comm.allreduce(&durationLimitReached, &result, 1, tmpi::MpiType::kBOOL, tmpi::MpiOp::LOR);
                            durationLimitReached = result;
                        }
                        if (durationLimitReached)
                        {
                            break;
                        }
                    }
                    if (worldConfig.getRank() == 0)
                    {
                        printf("%s\n", session.getLayerProfileInfo().c_str());
                    }
                }
            }
            catch (std::runtime_error& e)
            {
                std::size_t found = std::string(e.what()).find("out of memory");
                // We need to kill the memory monitor when OOM.
                done = true;
                peakMemFuture.wait();
                peakMem = peakMemFuture.get();

                // Unexpected error; rethrow
                if (found == std::string::npos)
                {
                    TLLM_LOG_ERROR(e.what());
                    throw e;
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
            catch (...)
            {
                // We need to kill memory monitor when any other issue occurs
                done = true;
                peakMemFuture.wait();
                peakMem = peakMemFuture.get();
                throw;
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
    options.add_options()(
        "max_attention_window", "Max kv cache length per sequence.", cxxopts::value<std::vector<int>>());
    options.add_options()("max_tokens_in_paged_kvcache", "Max tokens in paged K-V Cache.", cxxopts::value<int>());
    options.add_options()("sink_token_len", "Sink token length in kv cache per sequence.", cxxopts::value<int>());
    options.add_options()(
        "kv_cache_free_gpu_mem_fraction", "K-V Cache Free Gpu Mem Fraction.", cxxopts::value<float>());

    options.add_options()("enable_cuda_graph", "Execute GPT session with CUDA graph.");
    options.add_options()("print_all_logits", "Print all context and generation logits.");
    options.add_options()("disable_force_max_tokens", "Disable force the engine generating new max_tokens.");
    options.add_options()("dump_layer_info", "Print layer information of the engine to console.");
    options.add_options()("dump_profile", "Print profile information per layer.");
    options.add_options()("gpu_weights_percent",
        "Specify the percentage of weights that reside on GPU (from 0.0 to 1.0). Multiple percentages can be separated "
        "by \";\", "
        "example: \"0.0;0.5;1.0\".",
        cxxopts::value<std::string>()->default_value("1.0"));

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
        sessionConfig.kvCacheConfig.maxAttentionWindowVec = result["max_attention_window"].as<std::vector<int>>();
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
    auto dumpLayerInfo = result.count("dump_layer_info") > 0;
    auto dumpProfile = result.count("dump_profile") > 0;

    // Argument: GPU weights percentage
    std::istringstream ssGpuPercentArg;
    ssGpuPercentArg.str(result["gpu_weights_percent"].as<std::string>());
    std::vector<float> gpuWeightsPercents;
    for (std::string token; std::getline(ssGpuPercentArg, token, ';');)
    {
        auto gpuWeightsPercent = std::stof(token);
        if (gpuWeightsPercent < 0 || gpuWeightsPercent > 1)
        {
            TLLM_LOG_ERROR(
                "--gpu_weights_percent must have percents between 0.0 and 1.0 but got: %f", gpuWeightsPercent);
            return 1;
        }
        gpuWeightsPercents.push_back(gpuWeightsPercent);
    }

    initTrtLlmPlugins(logger.get());

    try
    {
        benchmarkGptSession(result["engine_dir"].as<std::string>(), batchSizes, beamWidth, inOutLen, logger,
            result["warm_up"].as<int>(), result["num_runs"].as<int>(), result["duration"].as<int>(), sessionConfig,
            enableCudaGraph, printAllLogits, disableForceMaxTokens, dumpLayerInfo, dumpProfile, gpuWeightsPercents);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_ERROR(e.what());
        return 1;
    }
    return 0;
}
