#include "tensorrtllm.h"
#include "models/chat_completion_request.h"
#include "nlohmann/json.hpp"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "utils/nitro_utils.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <queue>
#include <string>
#include <trantor/utils/Logger.h>
#include <vector>

using json = nlohmann::json;
using namespace inferences;

void removeId(std::vector<int>& vec, int id)
{
    vec.erase(std::remove(vec.begin(), vec.end(), id), vec.end());
}

struct inferenceState
{
    int prevPos{0};
    std::string prevText;
    bool isFinished;
    std::queue<std::string> textsToStream;
    std::mutex queueMutex; // Mutex to protect access to textsToStream

    size_t stopWordMatchLen = 0;
    std::vector<std::string> sequence{"<", "|", "im", "_", "end", "|", ">"};

    void reset()
    {
        stopWordMatchLen = 0;
        prevText = "";
    }

    bool isComplete() const
    {
        return stopWordMatchLen >= sequence.size();
    }
};

bool handleMatch(const std::string& rawText, std::shared_ptr<inferenceState> inferState)
{
    if (inferState->isComplete())
    {
        return true;
    }

    if (rawText == inferState->sequence[inferState->stopWordMatchLen])
    {
        inferState->stopWordMatchLen++; // Move to next state
        inferState->prevText = rawText;
        return true;
    }
    else if (inferState->stopWordMatchLen > 0 && rawText == inferState->sequence[0])
    {
        inferState->stopWordMatchLen = 1; // Restart from first match if sequence breaks but matches start
        inferState->prevText = rawText;
        return true;
    }
    else
    {
        inferState->reset();
        return false; // Reset to start if sequence breaks
    }
}

// Only support single token stopping point now
std::string create_return_json(const std::string& id, const std::string& model, const std::string& content,
    Json::Value finish_reason = Json::Value())
{
    Json::Value root;

    root["id"] = id;
    root["model"] = model;
    root["created"] = static_cast<int>(std::time(nullptr));
    root["object"] = "chat.completion.chunk";

    Json::Value choicesArray(Json::arrayValue);
    Json::Value choice;

    choice["index"] = 0;
    Json::Value delta;
    delta["content"] = content;
    choice["delta"] = delta;
    choice["finish_reason"] = finish_reason;

    choicesArray.append(choice);
    root["choices"] = choicesArray;

    Json::StreamWriterBuilder writer;
    writer["indentation"] = ""; // This sets the indentation to an empty string,
                                // producing compact output.
    return Json::writeString(writer, root);
}

GenerationInput::TensorPtr tensorrtllm::getTensorSingleStopWordList(int stopToken)
{

    std::vector<int32_t> stopWordsTokens = {stopToken, -1, 1, -1}; // Extend with -1 for increased length
    return gptSession->getBufferManager().copyFrom(stopWordsTokens, ITensor::makeShape({1, 2, 2}), MemoryType::kGPU);
}

GenerationInput::TensorPtr tensorrtllm::getTensorChatMLStopWordList()
{
    std::vector<int32_t> stopWordsTokens = {28789, 28766, 321, 28730, 416, 28766, 28767, 32000, 6, 8, -1, -1, -1, -1,
        -1, -1}; // Extend with -1 for increased length
    return gptSession->getBufferManager().copyFrom(stopWordsTokens, ITensor::makeShape({1, 2, 8}), MemoryType::kGPU);
}

GenerationInput tensorrtllm::createGenerationInput(std::vector<int32_t> inputIdsHost)
{
    int inputLen = inputIdsHost.size();
    std::vector<int32_t> inputLengthsHost(batchSize, inputLen);
    GenerationInput::TensorPtr inputLengths
        = gptSession->getBufferManager().copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);
    GenerationInput::TensorPtr inputIds = gptSession->getBufferManager().copyFrom(
        inputIdsHost, ITensor::makeShape({batchSize, inputLen}), MemoryType::kGPU);

    GenerationInput generationInput{0, 0, inputIds, inputLengths, modelConfig->usePackedInput()};

    generationInput.stopWordsList = getTensorChatMLStopWordList();
    return generationInput;
}

GenerationOutput tensorrtllm::createGenerationOutput()
{
    GenerationOutput generationOutput{
        gptSession->getBufferManager().emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
        gptSession->getBufferManager().emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};
    return generationOutput;
}

void inferenceThread(std::shared_ptr<inferenceState> inferState, std::vector<int32_t> inputIdsHost,
    std::function<void(const HttpResponsePtr&)> callback, tensorrtllm* self)
{
    const int inputLen = inputIdsHost.size();
    const int outputLen = 2048 - inputLen;

    // Create sampling config
    SamplingConfig samplingConfig{1};
    samplingConfig.temperature = std::vector{0.0f};
    samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
    samplingConfig.topK = std::vector{40};
    samplingConfig.topP = std::vector{0.0f};
    samplingConfig.minLength = std::vector{outputLen};
    samplingConfig.repetitionPenalty = std::vector{1.3f};

    std::cout << "Start Nitro testing session: " << std::endl;

    // Input preparation

    GenerationInput generationInput = self->createGenerationInput(inputIdsHost);

    GenerationOutput generationOutput = self->createGenerationOutput();

    // Define the callback to stream each generated token
    generationOutput.onTokenGenerated = [&inferState, inputLen, outputLen, self, &generationOutput](
                                            GenerationOutput::TensorPtr const& outputIds, SizeType step, bool finished)
    {
        // Assuming the shape of outputIds tensor is (1, 1, 160), where 160 is the number of tokens
        int outputLength = outputIds->getShape().d[2]; // Get the length of output IDs based on the tensor shape
        // Copy output IDs from GPU to host for printing
        std::vector<int32_t> outputIdsHost(outputLength);
        self->gptSession->getBufferManager().copy(*outputIds, outputIdsHost.data(), MemoryType::kCPU);
        // Find the last non-zero value in the output IDs starting from the end of the input sequence
        std::vector<int> outputIdsHostDecode(outputIdsHost.begin() + inputLen, outputIdsHost.end());
        removeId(outputIdsHostDecode, 0);
        std::string text = self->nitro_tokenizer->decode(outputIdsHostDecode);

        if (inferState->prevPos > 0 && inferState->prevPos < text.size())
        {
            // Valid prevPos, proceed with slicing the string from prevPos to the end
            std::string stringTok(text.begin() + inferState->prevPos, text.end());
            std::lock_guard<std::mutex> guard(inferState->queueMutex); // Protect access with a lock
            inferState->textsToStream.push(stringTok);
        }
        else if (inferState->prevPos >= text.size())
        {
            inferState->prevPos = text.size();
        }
        inferState->prevPos = text.size();
        if (finished)
        {

            std::lock_guard<std::mutex> guard(inferState->queueMutex); // Protect access with a lock
            inferState->textsToStream.push("[DONE]");
            return;
        }
    };
    // The rest of the logic inside the `chat_completion` remains unchanged...
    // After finishing the setup, call the inference logic
    self->gptSession->generate(generationOutput, generationInput, samplingConfig);
}

void tensorrtllm::chat_completion(
    inferences::ChatCompletionRequest&& completion, std::function<void(const HttpResponsePtr&)>&& callback)
{

    std::string formatted_input = pre_prompt;

    nlohmann::json data;

    data["stream"] = completion.stream;
    data["n_predict"] = completion.max_tokens;
    data["top_p"] = completion.top_p;
    data["temperature"] = completion.temperature;
    data["frequency_penalty"] = completion.frequency_penalty;
    data["presence_penalty"] = completion.presence_penalty;
    const Json::Value& messages = completion.messages;

    // Format the input from user
    for (const auto& message : messages)
    {
        std::string input_role = message["role"].asString();
        std::string role;
        if (input_role == "user")
        {
            role = user_prompt;
            std::string content = message["content"].asString();
            formatted_input += role + content;
        }
        else if (input_role == "assistant")
        {
            role = ai_prompt;
            std::string content = message["content"].asString();
            formatted_input += role + content;
        }
        else if (input_role == "system")
        {
            role = system_prompt;
            std::string content = message["content"].asString();
            formatted_input = role + content + formatted_input;
        }
        else
        {
            role = input_role;
            std::string content = message["content"].asString();
            formatted_input += role + content;
        }
    }
    formatted_input += ai_prompt;
    // Format the input from user

    std::shared_ptr<inferenceState> inferState = std::make_shared<inferenceState>();

    std::vector<int32_t> inputIdsHost = nitro_tokenizer->encode(formatted_input);
    const int inputLen = inputIdsHost.size();
    const int outputLen = 2048 - inputLen;

    // Create sampling config
    SamplingConfig samplingConfig{1};
    samplingConfig.temperature = std::vector{0.0f};
    samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
    samplingConfig.topK = std::vector{40};
    samplingConfig.topP = std::vector{0.0f};
    samplingConfig.minLength = std::vector{outputLen};
    samplingConfig.repetitionPenalty = std::vector{1.3f};

    std::cout << "Start Nitro testing session: " << std::endl;

    // Input preparation

    std::thread infThread(inferenceThread, inferState, inputIdsHost, callback, this);
    infThread.detach(); // Detach the thread to allow it to run independently

    auto chunked_content_provider = [inferState](char* pBuffer, std::size_t nBuffSize) -> std::size_t
    {
        if (!pBuffer)
        {
            LOG_INFO << "Connection closed or buffer is null. Reset context";
            return 0; // Indicate no more data to send
        }

        if (inferState->isFinished)
        {
            return 0;
        }

        while (true) // Continuously check if the queue is not empty
        {
            std::unique_lock<std::mutex> lock(inferState->queueMutex); // Lock the queue for exclusive access
            if (!inferState->textsToStream.empty())
            {

                std::string rawText = inferState->textsToStream.front();
                inferState->textsToStream.pop();
                if (handleMatch(rawText, inferState))
                {
                    continue;
                };

                if (rawText == "[DONE]")
                {
                    LOG_INFO << "End of result";
                    const std::string str
                        = "data: " + create_return_json(nitro_utils::generate_random_string(20), "_", "", "stop")
                        + "\n\n" + "data: [DONE]" + "\n\n";

                    std::size_t nRead = std::min(str.size(), nBuffSize);
                    memcpy(pBuffer, str.data(), nRead);
                    inferState->isFinished = true;
                    return nRead;
                }
                const std::string textToStream
                    = "data: " + create_return_json(nitro_utils::generate_random_string(20), "_", rawText) + "\n\n";
                lock.unlock(); // Unlock as soon as possible

                // Ensure we do not exceed the buffer size. Truncate if necessary.
                std::size_t bytesToWrite = std::min(nBuffSize, textToStream.size());

                // Copy the text to the provided buffer
                std::memcpy(pBuffer, textToStream.data(), bytesToWrite);
                inferState->prevText = rawText;
                return bytesToWrite; // Return the number of bytes written to the buffer
            }
            else
            {
                // If the queue is empty, release the lock and wait before trying again
                lock.unlock();
            }
        }
    };

    auto streamResponse = nitro_utils::nitroStreamResponse(chunked_content_provider);
    callback(streamResponse);
    return;
};

void tensorrtllm::loadModel(const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& callback)
{
    const auto& jsonBody = req->getJsonObject();

    if (!jsonBody)
    {
        Json::Value jsonResp;
        jsonResp["message"] = "Require params!";
        auto resp = nitro_utils::nitroHttpJsonResponse(jsonResp);
        callback(resp);
        return;
    }

    const std::filesystem::path engineDir = jsonBody->operator[]("engine_path").asString();
    int ctx_len = jsonBody->get("ctx_len", 2048).asInt();

    logger = std::make_shared<TllmLogger>();
    logger->setLevel(nvinfer1::ILogger::Severity::kINFO);
    // Fixed settings
    const std::string modelName = "mistral";
    initTrtLlmPlugins(logger.get());
    // Load model configuration
    std::filesystem::path jsonFileName = engineDir / "config.json";
    std::filesystem::path tokenizerModelName = engineDir / "tokenizer.model";

    nitro_tokenizer = std::make_unique<Tokenizer>(tokenizerModelName.string());
    LOG_INFO << "Loaded tokenizer";

    auto const json = GptJsonConfig::parse(jsonFileName);
    auto config = json.getModelConfig();
    modelConfig = std::make_unique<GptModelConfig>(config);
    auto const worldConfig = WorldConfig::mpi(1, json.getTensorParallelism(), json.getPipelineParallelism());
    auto const enginePath = engineDir / json.engineFilename(worldConfig, modelName);
    LOG_INFO << "Engine Path : " << enginePath.string();
    auto const dtype = modelConfig->getDataType();

    // Currently doing fixed session config
    sessionConfig.maxBatchSize = batchSize;
    sessionConfig.maxBeamWidth = 1; // Fixed for simplicity
    sessionConfig.maxSequenceLength = ctx_len;
    sessionConfig.cudaGraphMode = true; // Fixed for simplicity

    // Init gptSession
    gptSession = std::make_unique<GptSession>(sessionConfig, *modelConfig, worldConfig, enginePath.string(), logger);
    // Model loaded successfully
    Json::Value jsonResp;
    jsonResp["message"] = "Model loaded successfully";
    auto resp = nitro_utils::nitroHttpJsonResponse(jsonResp);
    callback(resp);
    return;
};

// Add definition of your processing function here
