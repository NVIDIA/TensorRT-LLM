#pragma once
#include "json/value.h"

namespace tensorrtllm::inferences {
struct ChatCompletionRequest {
  int max_tokens = 2048;
  bool stream = false;
  float top_p = 0.95;
  float temperature = 0.00001f;
  float frequency_penalty = 1.3;
  float presence_penalty = 0;
  Json::Value messages = Json::Value(Json::arrayValue);
  Json::Value stop = Json::Value(Json::arrayValue);
};

inline ChatCompletionRequest fromJson(std::shared_ptr<Json::Value> json_body) {
  ChatCompletionRequest request;
  if (json_body) {
    request.max_tokens        = json_body->get("max_tokens", 2048).asInt();
    request.stream            = json_body->get("stream", false).asBool();
    request.top_p             = json_body->get("top_p", 0.95).asFloat();
    request.temperature       = json_body->get("temperature", 0.00001f).asFloat();
    request.frequency_penalty = json_body->get("frequency_penalty", 1.3).asFloat();
    request.presence_penalty  = json_body->get("presence_penalty", 0).asFloat();
    request.messages          = json_body->operator[]("messages");
    request.stop              = json_body->operator[]("stop");
  }
  return request;
}
} // namespace tensorrtllm::inferences
