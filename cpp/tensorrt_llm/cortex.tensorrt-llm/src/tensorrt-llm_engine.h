#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <queue>
#include <string>

#include "NvInfer.h"
#include "base/cortex-common/cortextensorrtllmi.h"
#include "models/chat_completion_request.h"
#include "models/load_model_request.h"
#include "sentencepiece_processor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "trantor/utils/ConcurrentTaskQueue.h"
#include "trantor/utils/Logger.h"
#include <nlohmann/json.hpp>


using namespace tensorrt_llm::runtime;

// class Tokenizer {
//  private:
//   sentencepiece::SentencePieceProcessor processor;

//   void ReplaceSubstring(std::string& base, const std::string& from, const std::string& to) {
//     size_t start_pos = 0;
//     while ((start_pos = base.find(from, start_pos)) != std::string::npos) {
//         base.replace(start_pos, from.length(), to);
//         start_pos += to.length();
//     }
//   }

//  public:
//   Tokenizer(const std::string& model_path) {
//     auto status = processor.Load(model_path);
//     if (!status.ok()) {
//       std::cerr << status.ToString() << std::endl;
//     }
//     LOG_INFO << "Successully loaded the tokenizer";
//   }

//   std::string DecodeWithSpace(const int id) {
//     std::string text = processor.IdToPiece(id);
//     ReplaceSubstring(text, "▁", " ");
//     return text;
//   }

//   std::string Decode(const std::vector<int32_t> ids) {
//     std::string text = processor.DecodeIds(ids);
//     return text;
//   }

//   std::vector<int> Encode(const std::string& input) {
//     std::vector<int> ids;
//     processor.Encode(input, &ids);
//     return ids;
//   }
// };

class Tokenizer {
 private:
  std::unordered_map<std::string, int> vocab;
  std::unordered_map<int, std::string> id_to_token;

  void ReplaceSubstring(std::string& base, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = base.find(from, start_pos)) != std::string::npos) {
        base.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
  }

 public:
  Tokenizer(const std::string& json_path) {
    // Load tokenizer.json
    std::ifstream file(json_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open tokenizer JSON file");
    }

    nlohmann::json tokenizer_json;
    file >> tokenizer_json;

    // Parse vocabulary
    vocab = tokenizer_json["model"]["vocab"].get<std::unordered_map<std::string, int>>();
    for (const auto& [key, value] : vocab) {
      id_to_token[value] = key;
    }

    LOG_INFO << "Successfully loaded the tokenizer from JSON";
  }

  std::string DecodeWithSpace(const int id) {
    std::string text = id_to_token[id];
    ReplaceSubstring(text, "▁", " ");
    return text;
  }

  std::string Decode(const std::vector<int32_t>& ids) {
    std::string text;
    for (int id : ids) {
      text += id_to_token[id];
    }
    ReplaceSubstring(text, "▁", " ");
    return text;
  }

  std::vector<int> Encode(const std::string& input) {
    std::vector<int> ids;
    std::string word;
    for (char ch : input) {
      word += ch;
      if (vocab.find(word) != vocab.end()) {
        ids.push_back(vocab[word]);
        word.clear();
      }
    }
    return ids;
  }
};

struct InferenceState {
    int prev_pos{0};
    std::string prev_text;
    bool is_finished;
    std::queue<std::string> texts_to_stream;
    std::mutex queue_mutex; // Mutex to protect access to textsToStream
    size_t stop_word_match_len = 0;
    std::vector<std::string> sequence{"<", "|", "im", "_", "end", "|", ">"};
    int token_gen_count = 0;

    void Reset() {
        stop_word_match_len = 0;
        prev_text = "";
    }

    bool IsComplete() const {
        return stop_word_match_len >= sequence.size();
    }
};

namespace tensorrtllm {

class TensorrtllmEngine : public CortexTensorrtLlmEngineI {
 public:
  ~TensorrtllmEngine() final;
  // ### Interface ###
  void LoadModel(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void HandleChatCompletion(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void Destroy(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final; 

  GenerationInput::TensorPtr GetTensorSingleStopWordList(int stopToken);
  GenerationInput CreateGenerationInput(std::vector<int32_t> inputIds);
  GenerationOutput CreateGenerationOutput();
  GenerationInput::TensorPtr GetTensorChatMLStopWordList();

  std::unique_ptr<GptSession> gpt_session;
  std::unique_ptr<Tokenizer> cortex_tokenizer;

  void LoadModelImpl(model::LoadModelRequest&& request, std::function<void(Json::Value&&, Json::Value&&)>&& callback);
  void HandleChatCompletionImpl(inferences::ChatCompletionRequest&& request, std::function<void(Json::Value&&, Json::Value&&)>&& callback);  
 private:
  std::unique_ptr<trantor::ConcurrentTaskQueue> q;
  GptSession::Config session_config{1, 1, 1};
  SamplingConfig sampling_config{1};
  std::unique_ptr<GptModelConfig> model_config;
  std::shared_ptr<TllmLogger> logger;
  std::string user_prompt;
  std::string ai_prompt;
  std::string system_prompt;
  std::string pre_prompt;
  int batchSize = 1;
};

} // namespace inferences
