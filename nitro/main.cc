#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <NvInfer.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace trt = nvinfer1;

namespace
{
void runBenchmark()
{
    // Fixed settings
    const std::string modelName = "mistral";
    const std::filesystem::path engineDir = "/app/mistral_engine_2/";
    const int batchSize = 1;
    const std::vector<int> inOutLen = {60, 100}; // input_length, output_length

    // Logger setup
    auto logger = std::make_shared<TllmLogger>();
    logger->setLevel(nvinfer1::ILogger::Severity::kINFO);

    initTrtLlmPlugins(logger.get());

    // Load model configuration
    std::filesystem::path jsonFileName = engineDir / "config.json";
    auto const json = GptJsonConfig::parse(jsonFileName);
    auto const modelConfig = json.getModelConfig();
    auto const worldConfig = WorldConfig::mpi(1, json.getTensorParallelism(), json.getPipelineParallelism());
    auto const enginePath = engineDir / json.engineFilename(worldConfig, modelName);
    auto const dtype = modelConfig.getDataType();

    GptSession::Config sessionConfig{1, 1, 1};
    sessionConfig.maxBatchSize = batchSize;
    sessionConfig.maxBeamWidth = 1; // Fixed for simplicity
    sessionConfig.maxSequenceLength = inOutLen[0] + inOutLen[1];
    sessionConfig.cudaGraphMode = false; // Fixed for simplicity

    SamplingConfig samplingConfig{1}; // Fixed for simplicity
    samplingConfig.temperature = std::vector{1.0f};
    samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
    samplingConfig.topK = std::vector{1};
    samplingConfig.topP = std::vector{0.0f};
    samplingConfig.minLength = std::vector{inOutLen[1]};

    // Initialize session
    GptSession session{sessionConfig, modelConfig, worldConfig, enginePath.string(), logger};
    // Generate random input IDs within the model's vocabulary range
    const int vocabSize = modelConfig.getVocabSize();
    std::vector<int32_t> inputIdsHost(batchSize * inOutLen[0]);
    for (auto& id : inputIdsHost)
    {
        id = rand() % vocabSize; // Random token ID within vocabulary range
        std::cout << id << std::endl;
    }
    // Simplified benchmarking process for a single run
    // Note: This example does not include input data preparation or output handling for brevity

    // Input preparation
    auto& bufferManager = session.getBufferManager();
    GenerationInput::TensorPtr inputIds
        = bufferManager.copyFrom(inputIdsHost, ITensor::makeShape({batchSize, inOutLen[0]}), MemoryType::kGPU);

    std::vector<int32_t> inputLengthsHost(batchSize, inOutLen[0]);
    GenerationInput::TensorPtr inputLengths
        = bufferManager.copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    bool inputPacked = modelConfig.usePackedInput();

    GenerationInput generationInput{50256, 50256, inputIds, inputLengths, inputPacked};

    GenerationOutput generationOutput{bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
        bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};

    // Define the callback to stream each generated token
    generationOutput.onTokenGenerated
        = [&bufferManager](GenerationOutput::TensorPtr const& outputIds, SizeType step, bool finished)
    {
        if (!finished)
        {
            // Assuming the shape of outputIds tensor is (1, 1, 160), where 160 is the number of tokens
            int outputLength = outputIds->getShape().d[2]; // Get the length of output IDs based on the tensor shape

            // Copy output IDs from GPU to host for printing
            std::vector<int32_t> outputIdsHost(outputLength);
            bufferManager.copy(*outputIds, outputIdsHost.data(), MemoryType::kCPU);

            // Print the entire output IDs array
            std::cout << "Output IDs at step " << step << ": ";
            for (int i = 0; i < outputLength; ++i)
            {
                std::cout << outputIdsHost[i] << " ";
            }
            std::cout << "\n";
            // Copy output IDs from GPU to host for printing
            // std::vector<int32_t> outputIdsHost(outputIds->size());
            // bufferManager.copy(outputIdsHost, outputIds);

            // Print the entire output IDs array
            // std::cout << "Output IDs at step " << outputIds->getShape().d[2] << ": ";
        }
    };

    session.generate(generationOutput, generationInput, samplingConfig);
    bufferManager.getStream().synchronize();
}

} // namespace

int main()
{
    try
    {
        runBenchmark();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
