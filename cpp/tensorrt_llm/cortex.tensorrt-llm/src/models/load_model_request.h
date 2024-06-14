#pragma once
#include <string>

#include "json/value.h"

namespace tensorrtllm::model {
struct LoadModelRequest {
    int ctx_len = 2048;
    int n_parallel = 1;
    std::string model_id = "default";
    std::string engine_path;
    std::string user_prompt = "<|im_end|>\n<|im_start|>user\n";
    std::string ai_prompt = "<|im_end|>\n<|im_start|>user\n";
    std::string system_prompt = "<|im_end|>\n<|im_start|>user\n";
};

inline LoadModelRequest fromJson(std::shared_ptr<Json::Value> json_body) {
  LoadModelRequest request;
  if (json_body) {
    request.ctx_len       = json_body->get("ctx_len", 2048).asInt();
    request.n_parallel    = json_body->get("n_parallel", 1).asInt();
    request.model_id      = json_body->get("model_id", "default").asString();
    request.engine_path   = json_body->get("engine_path", "").asString();
    request.user_prompt   = json_body->get("user_prompt", "<|im_end|>\n<|im_start|>user\n").asString();
    request.ai_prompt     = json_body->get("ai_prompt", "<|im_end|>\n<|im_start|>assistant\n").asString();
    request.system_prompt = json_body->get("system_prompt", "<|im_start|>system\n").asString();
  } 
  return request;
}
} // namespace tensorrtllm::inferences
