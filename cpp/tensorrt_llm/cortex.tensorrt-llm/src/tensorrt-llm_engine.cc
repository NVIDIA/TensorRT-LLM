#include "tensorrt-llm_engine.h"
#include "models/chat_completion_request.h"
#include "nlohmann/json.hpp"

#include "src/models/load_model_request.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "utils/tensorrt-llm_utils.h"
#include "json/writer.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <trantor/utils/Logger.h>
#include <vector>

using json = nlohmann::json;
using namespace tensorrtllm;


constexpr const int k200OK = 200;
constexpr const int k400BadRequest = 400;
constexpr const int k409Conflict = 409;
constexpr const int k500InternalServerError = 500;

TensorrtllmEngine::~TensorrtllmEngine() {}

void RemoveId(std::vector<int>& vec, int id) {
  vec.erase(std::remove(vec.begin(), vec.end(), id), vec.end());
}

bool HandleMatch(std::string const& rew_text, std::shared_ptr<InferenceState> infer_state) {
  if (infer_state->IsComplete()) {
    return false;
  }
  if (infer_state->stop_word_match_len == 0) {
    if (rew_text.find('<') != std::string::npos) { // Found "<" anywhere in the text
      infer_state->stop_word_match_len++; // Move to next state
      infer_state->prev_text = rew_text;
      return true;
    }
  }
  else if (rew_text == infer_state->sequence[infer_state->stop_word_match_len]) {
    infer_state->stop_word_match_len++; // Move to next state
    infer_state->prev_text = rew_text;
    return true;
  }
  else if (infer_state->stop_word_match_len > 0 && rew_text == infer_state->sequence[0]) {
    infer_state->stop_word_match_len = 1; // Restart from first match if sequence breaks but matches start
    infer_state->prev_text = rew_text;
    return true;
  }
  else {
    infer_state->Reset();
    return false; // Reset to start if sequence breaks
  }
  return false;
}

GenerationInput::TensorPtr TensorrtllmEngine::GetTensorSingleStopWordList(int stopToken) {
  std::vector<int32_t> stop_words_tokens = {stopToken, -1, 1, -1}; // Extend with -1 for increased length
  return gpt_session->getBufferManager().copyFrom(stop_words_tokens, ITensor::makeShape({1, 2, 2}), MemoryType::kGPU);
}

GenerationInput::TensorPtr TensorrtllmEngine::GetTensorChatMLStopWordList() {
  std::vector<int32_t> stop_words_tokens
    = {321, 28730, 416, 2, 32000, 3, 4, 5, -1, -1}; // Extend with -1 for increased length
  return gpt_session->getBufferManager().copyFrom(stop_words_tokens, ITensor::makeShape({1, 2, 5}), MemoryType::kGPU);
}

GenerationInput TensorrtllmEngine::CreateGenerationInput(std::vector<int32_t> input_ids_host) {
  int input_len = input_ids_host.size();
  std::vector<int32_t> input_lengths_host(batchSize, input_len);
  GenerationInput::TensorPtr input_lengths
      = gpt_session->getBufferManager().copyFrom(input_lengths_host, ITensor::makeShape({batchSize}), MemoryType::kGPU);
  GenerationInput::TensorPtr input_ids = gpt_session->getBufferManager().copyFrom(
      input_ids_host, ITensor::makeShape({batchSize, input_len}), MemoryType::kGPU);
  GenerationInput generation_input{0, 0, input_ids, input_lengths, model_config->usePackedInput()};
  generation_input.stopWordsList = GetTensorChatMLStopWordList();

  LOG_INFO << "Create generation input successfully";
  return generation_input;
}

GenerationOutput TensorrtllmEngine::CreateGenerationOutput() {
  GenerationOutput generation_output {
    gpt_session->getBufferManager().emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
    gpt_session->getBufferManager().emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)
  };
  LOG_INFO << "Create generation input successfully";
  return generation_output;
}

void InferenceThread(
    std::shared_ptr<InferenceState> infer_state,
    std::vector<int32_t> input_ids_host,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback,
    TensorrtllmEngine* self,
    SamplingConfig sampling_config,
    int input_len,
    int outputLen) {

  // Input preparation
  LOG_INFO << "Inference thread started";
  GenerationInput generation_input = self->CreateGenerationInput(input_ids_host);
  GenerationOutput generation_output = self->CreateGenerationOutput();

  // Define the callback to stream each generated token
  generation_output.onTokenGenerated = [&infer_state, input_len, outputLen, self, &generation_output](
                                          GenerationOutput::TensorPtr const& output_ids, SizeType step, bool finished) {
    LOG_INFO << "Generating tokenizer in thread";                                            
    // Assuming the shape of output_ids tensor is (1, 1, 160), where 160 is the number of tokens
    int output_length = output_ids->getShape().d[2]; // Get the length of output IDs based on the tensor shape
    // Copy output IDs from GPU to host for printing
    std::vector<int32_t> output_idsHost(output_length);
    self->gpt_session->getBufferManager().copy(*output_ids, output_idsHost.data(), MemoryType::kCPU);
    // Find the last non-zero value in the output IDs starting from the end of the input sequence
    std::vector<int> output_idsHostDecode(output_idsHost.begin() + input_len, output_idsHost.end());
    RemoveId(output_idsHostDecode, 0);
    RemoveId(output_idsHostDecode, 32000);
    RemoveId(output_idsHostDecode, 32001);
    std::string text = self->cortex_tokenizer->Decode(output_idsHostDecode);

    if (infer_state->prev_pos >= 0 && infer_state->prev_pos < text.size()) {
      // Valid prev_pos, proceed with slicing the string from prev_pos to the end
      std::string string_tok(text.begin() + infer_state->prev_pos, text.end());
      std::lock_guard<std::mutex> guard(infer_state->queue_mutex); // Protect access with a lock
      infer_state->texts_to_stream.push(string_tok);
      ++infer_state->token_gen_count;
    }
    else if (infer_state->prev_pos >= text.size()) {
      infer_state->prev_pos = text.size();
    }
    infer_state->prev_pos = text.size();
    if (finished) {
      std::lock_guard<std::mutex> guard(infer_state->queue_mutex); // Protect access with a lock
      infer_state->texts_to_stream.push("[DONE]");
      LOG_INFO << "Cortex.tensorrtllm generated " << infer_state->token_gen_count << " tokens";
      return;
    }
    return;
  };
  // The rest of the logic inside the `chat_completion` remains unchanged...
  // After finishing the setup, call the inference logic
  self->gpt_session->generate(generation_output, generation_input, sampling_config);
}

inline std::string GetModelId(const Json::Value& json_body) {
  // First check if model exists in request
  if (!json_body["model"].isNull()) {
    return json_body["model"].asString();
  } else if (!json_body["model_alias"].isNull()) {
    return json_body["model_alias"].asString();
  }

  // We check model_path for loadmodel request
  auto input = json_body["model_path"];
  if (!input.isNull()) {
    auto s = input.asString();
    std::replace(s.begin(), s.end(), '\\', '/');
    auto const pos = s.find_last_of('/');
    return s.substr(pos + 1);
  }
  return {};
}

bool TensorrtllmEngine::CheckModelLoaded(std::function<void(Json::Value&&, Json::Value&&)>& callback) {
  if (!model_loaded_) {
    LOG_WARN << "Model is not loaded yet";
    Json::Value json_resp;
    json_resp["message"] =
        "Model has not been loaded, please load model into cortex.tensorrt-llm";
    Json::Value status;
    status["is_done"] = false;
    status["has_error"] = true;
    status["is_stream"] = false;
    status["status_code"] = k409Conflict;
    callback(std::move(status), std::move(json_resp));
    return false;
  }
  return true;
}

//#########################
//### ENGINE END POINTS ###
//#########################


void TensorrtllmEngine::HandleChatCompletion(std::shared_ptr<Json::Value> json_body, std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  inferences::ChatCompletionRequest request = inferences::fromJson(json_body);
  std::string formatted_input = pre_prompt;
  nlohmann::json data;
  // data["stream"] = completion.stream;
  // data["n_predict"] = completion.max_tokens;
  data["presence_penalty"] = request.presence_penalty;
  Json::Value const& messages = request.messages;

  // Format the input from user
  for (auto const& message : messages) {
    std::string input_role = message["role"].asString();
    std::string role;
    if (input_role == "user") {
        role = user_prompt;
        std::string content = message["content"].asString();
        formatted_input += role + content;
    }
    else if (input_role == "assistant") {
        role = ai_prompt;
        std::string content = message["content"].asString();
        formatted_input += role + content;
    }
    else if (input_role == "system") {
        role = system_prompt;
        std::string content = message["content"].asString();
        formatted_input = role + content + formatted_input;
    }
    else {
        role = input_role;
        std::string content = message["content"].asString();
        formatted_input += role + content;
    }
  }
  formatted_input += ai_prompt;
  // Format the input from user

  std::shared_ptr<InferenceState> infer_state = std::make_shared<InferenceState>();

  std::vector<int32_t> input_ids_host = cortex_tokenizer->Encode(formatted_input);
  int const input_len = input_ids_host.size();
  int const outputLen = request.max_tokens - input_len;

  // Create sampling config
  SamplingConfig sampling_config{1};
  sampling_config.temperature = std::vector{request.temperature};
  sampling_config.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
  sampling_config.topK = std::vector{40};
  sampling_config.topP = std::vector{request.top_p};
  sampling_config.minLength = std::vector{outputLen};
  sampling_config.repetitionPenalty = std::vector{request.frequency_penalty};
  // Input preparation

  std::thread inference_thread(InferenceThread, infer_state, input_ids_host, callback, this, sampling_config, input_len, outputLen);
  inference_thread.detach(); // Detach the thread to allow it to run independently

  q_->runTaskInQueue([cb = std::move(callback), infer_state]() {
    LOG_INFO << "Preparing to run inference task queue...";
    while (true) { // Continuously check if the queue is not empty
      std::unique_lock<std::mutex> lock(infer_state->queue_mutex); // Lock the queue for exclusive access
      if (!infer_state->texts_to_stream.empty()) {
        std::string rew_text = infer_state->texts_to_stream.front();
        infer_state->texts_to_stream.pop();
        if (HandleMatch(rew_text, infer_state) && rew_text != "[DONE]") {
            continue;
        };

        if (rew_text == "[DONE]") {
          const std::string str
              = "data: " + tensorrtllm_utils::CreateReturnJson(tensorrtllm_utils::GenerateRandomString(20), "_", "", "stop")
              + "\n\n" + "data: [DONE]" + "\n\n";

          infer_state->is_finished = true;

          Json::Value resp_data;
          resp_data["data"] = str;
          Json::Value status;
          status["is_done"] = true;
          status["has_error"] = false;
          status["is_stream"] = true;
          status["status_code"] = k200OK;
          cb(std::move(status), std::move(resp_data));
          break;
        }
        const std::string text_to_stream
            = "data: " + tensorrtllm_utils::CreateReturnJson(tensorrtllm_utils::GenerateRandomString(20), "_", rew_text) + "\n\n";

        lock.unlock(); // Unlock as soon as possible
        infer_state->prev_text = rew_text;
        
        Json::Value resp_data;
        resp_data["data"] = text_to_stream;
        Json::Value status;
        status["is_done"] = false;
        status["has_error"] = false;
        status["is_stream"] = true;
        status["status_code"] = k200OK;
        cb(std::move(status), std::move(resp_data));
      } else {
        // If the queue is empty, release the lock and wait before trying again
        lock.unlock();
      }
    }
  });

  LOG_INFO << "Inference completed";
  return;
};

void TensorrtllmEngine::LoadModel(std::shared_ptr<Json::Value> json_body, std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
    model::LoadModelRequest request = model::fromJson(json_body);
    std::filesystem::path model_dir = request.model_path;

    int ctx_len = request.ctx_len;
    this->user_prompt = request.user_prompt;
    this->ai_prompt = request.ai_prompt;
    this->system_prompt = request.system_prompt;
    this->model_id_ = GetModelId(*json_body);

    logger = std::make_shared<TllmLogger>();
    logger->setLevel(nvinfer1::ILogger::Severity::kINFO);
    initTrtLlmPlugins(logger.get());

    std::filesystem::path tokenizer_model_name = model_dir / "tokenizer.model";
    cortex_tokenizer = std::make_unique<Tokenizer>(tokenizer_model_name.string());
    LOG_INFO << "Loaded tokenizer from " << tokenizer_model_name.string();

    std::filesystem::path json_file_name = model_dir / "config.json";
    auto json = GptJsonConfig::parse(json_file_name);
    auto config = json.getModelConfig();
    model_config = std::make_unique<GptModelConfig>(config);
    auto world_config = WorldConfig::mpi(1, json.getTensorParallelism(), json.getPipelineParallelism());
    LOG_INFO << "Loaded config from " << json_file_name.string();
    // auto dtype = model_config->getDataType();

    // Currently doing fixed session config
    session_config.maxBatchSize = batchSize;
    session_config.maxBeamWidth = 1; // Fixed for simplicity
    session_config.maxSequenceLength = ctx_len;
    session_config.cudaGraphMode = true; // Fixed for simplicity

    // Init gpt_session
    auto model_path = model_dir / json.engineFilename(world_config, model_id_);
    gpt_session = std::make_unique<GptSession>(session_config, *model_config, world_config, model_path.string(), logger);

    model_loaded_ = true;
    if (q_ == nullptr) {
     q_ = std::make_unique<trantor::ConcurrentTaskQueue>(1, model_id_);
    }

    // Model loaded successfully
    LOG_INFO << "Model " << model_id_ << " loaded successfully from path " << model_path.string();
    Json::Value json_resp;
    json_resp["message"] = "Model loaded successfully";
    Json::Value status_resp;
    status_resp["status_code"] = k200OK;
    callback(std::move(status_resp), std::move(json_resp));
    return;
};

void TensorrtllmEngine::UnloadModel(std::shared_ptr<Json::Value> json_body, std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  if (!CheckModelLoaded(callback)) {
    LOG_WARN << "Model was not loaded";
    Json::Value json_resp;
    json_resp["message"] = "Model was not loaded";
    Json::Value status;
    status["status_code"] = k400BadRequest;
    callback(std::move(status), std::move(json_resp));
    return;
  }
    
  gpt_session.reset();
  cortex_tokenizer.reset();
  q_.reset();
  model_config.reset();
  logger.reset();
  model_loaded_ = false;

  Json::Value json_resp;
  json_resp["message"] = "Model unloaded successfully";
  Json::Value status;
  status["is_done"] = true;
  status["has_error"] = false;
  status["is_stream"] = false;
  status["status_code"] = k200OK;
  callback(std::move(status), std::move(json_resp));
  LOG_INFO << "Model unloaded sucessfully";
}

void TensorrtllmEngine::HandleEmbedding( std::shared_ptr<Json::Value> json_body, std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  LOG_WARN << "Engine does not support embedding yet";
  Json::Value json_resp;
  json_resp["message"] = "Engine does not support embedding yet";
  Json::Value status;
  status["status_code"] = k409Conflict;
  callback(std::move(status), std::move(json_resp));
}

void TensorrtllmEngine::GetModelStatus(std::shared_ptr<Json::Value> json_body, std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  LOG_WARN << "Engine does not support get model status method yet";
  Json::Value json_resp;
  json_resp["message"] = "Engine does not support get model status method yet";
  Json::Value status;
  status["status_code"] = k409Conflict;
  callback(std::move(status), std::move(json_resp));
}

void TensorrtllmEngine::GetModels(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  Json::Value json_resp;
  Json::Value model_array = Json::arrayValue;

  if (model_loaded_) {
    Json::Value val;
    val["id"] = model_id_;
    val["engine"] = "cortex.tensorrt-llm";
    val["start_time"] = start_time_;
    val["vram"] = "-";
    val["ram"] = "-";
    val["object"] = "model";
    model_array.append(val);
  }

  json_resp["object"] = "list";
  json_resp["data"] = model_array;

  Json::Value status;
  status["is_done"] = true;
  status["has_error"] = false;
  status["is_stream"] = false;
  status["status_code"] = k200OK;
  callback(std::move(status), std::move(json_resp));
  LOG_INFO << "Running models responded";
}

extern "C" {
EngineI* get_engine() {
  return new TensorrtllmEngine();
}
}
