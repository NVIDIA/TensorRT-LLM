#pragma once
#include <drogon/HttpController.h>

namespace inferences {
struct ChatCompletionRequest {
  bool stream = false;
  int max_tokens = 500;
  float top_p = 0.95;
  float temperature = 0.8;
  float frequency_penalty = 0;
  float presence_penalty = 0;
  Json::Value stop = Json::Value(Json::arrayValue);
  Json::Value messages = Json::Value(Json::arrayValue);
};
}  // namespace inferences

namespace drogon {
template <>
inline inferences::ChatCompletionRequest fromRequest(const HttpRequest& req) {
  auto jsonBody = req.getJsonObject();
  inferences::ChatCompletionRequest completion;
  if (jsonBody) {
    completion.stream = (*jsonBody).get("stream", false).asBool();
    completion.max_tokens = (*jsonBody).get("max_tokens", 500).asInt();
    completion.top_p = (*jsonBody).get("top_p", 0.95).asFloat();
    completion.temperature = (*jsonBody).get("temperature", 0.8).asFloat();
    completion.frequency_penalty =
        (*jsonBody).get("frequency_penalty", 0).asFloat();
    completion.presence_penalty =
        (*jsonBody).get("presence_penalty", 0).asFloat();
    completion.messages = (*jsonBody)["messages"];
    completion.stop = (*jsonBody)["stop"];
  }
  return completion;
}
}  // namespace inferences
