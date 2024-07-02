#pragma once
#include <string>

#include "json/value.h"

namespace tensorrtllm::model {
struct LoadModelRequest {
    int ctx_len = 2048;
    int n_parallel = 1;
    std::string model_path;
    std::string user_prompt = "";
    std::string ai_prompt = "";
    std::string system_prompt = "";
};

inline LoadModelRequest fromJson(std::shared_ptr<Json::Value> json_body) {
  LoadModelRequest request;
  if (json_body) {
    request.ctx_len       = json_body->get("ctx_len", 2048).asInt();
    request.n_parallel    = json_body->get("n_parallel", 1).asInt();
    request.model_path   = json_body->get("model_path", "").asString();
    request.user_prompt   = json_body->get("user_prompt", "").asString();
    request.ai_prompt     = json_body->get("ai_prompt", "").asString();
    request.system_prompt = json_body->get("system_prompt", "").asString();
  } 
  return request;
}
} // namespace tensorrtllm::inferences
