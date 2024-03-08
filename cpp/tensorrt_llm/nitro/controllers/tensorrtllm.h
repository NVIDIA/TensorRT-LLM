#pragma once

#include "drogon/HttpTypes.h"
#include "sentencepiece_processor.h"
#include <cstdint>
#include <drogon/HttpController.h>

#include "sentencepiece_processor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include <NvInfer.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include "models/chat_completion_request.h"

using namespace drogon;

using namespace tensorrt_llm::runtime;

class Tokenizer
{
private:
    sentencepiece::SentencePieceProcessor processor;

    void replaceSubstring(std::string& base, const std::string& from, const std::string& to)
    {
        size_t start_pos = 0;
        while ((start_pos = base.find(from, start_pos)) != std::string::npos)
        {
            base.replace(start_pos, from.length(), to);
            start_pos += to.length();
        }
    }

public:
    Tokenizer(const std::string& modelPath)
    {
        auto status = processor.Load(modelPath);
        if (!status.ok())
        {
            std::cerr << status.ToString() << std::endl;
        }
        LOG_INFO << "Successully loaded the tokenizer";
    }

    std::string decodeWithSpace(const int id)
    {
        std::string text = processor.IdToPiece(id);
        replaceSubstring(text, "▁", " ");
        return text;
    }

    std::string decode(const std::vector<int32_t> ids)
    {
        std::string text = processor.DecodeIds(ids);
        return text;
    }

    std::vector<int> encode(const std::string& input)
    {
        std::vector<int> ids;
        processor.Encode(input, &ids);
        return ids;
    }
};

namespace inferences
{

class tensorrtllm : public drogon::HttpController<tensorrtllm>
{
public:
    tensorrtllm(){};

    METHOD_LIST_BEGIN
    // use METHOD_ADD to add your custom processing function here;
    ADD_METHOD_TO(tensorrtllm::chat_completion, "/v1/chat/completions", Post); // path is
    METHOD_ADD(tensorrtllm::loadModel, "loadmodel", Post);

    METHOD_LIST_END
    // your declaration of processing function maybe like this:
    // void get(const HttpRequestPtr& req, std::function<void (const HttpResponsePtr &)> &&callback, int p1, std::string
    // p2);
    void chat_completion(
        inferences::ChatCompletionRequest&& completion, std::function<void(const HttpResponsePtr&)>&& callback);

    void loadModel(const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& callback);
    std::unique_ptr<GptSession> gptSession;
    GenerationInput::TensorPtr getTensorSingleStopWordList(int stopToken);
    GenerationInput createGenerationInput(std::vector<int32_t> inputIds);
    GenerationOutput createGenerationOutput();
    std::unique_ptr<Tokenizer> nitro_tokenizer;

private:
    GptSession::Config sessionConfig{1, 1, 1};
    SamplingConfig samplingConfig{1};
    std::unique_ptr<GptModelConfig> modelConfig;
    std::shared_ptr<TllmLogger> logger;
    std::string example_string{
        "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nPlease write a long and sad "
        "story<|im_end|>\n<|im_start|>assistant"};
    std::string user_prompt{"<|im_end|>\n<|im_start|>user\n"};
    std::string ai_prompt{"<|im_end|>\n<|im_start|>assistant\n"};
    std::string system_prompt{"<|im_start|>system\n"};
    std::string pre_prompt;
    int batchSize = 1;
};

} // namespace inferences
