#pragma once

#include <functional>
#include <memory>

#include "json/value.h"

class CortexTensorrtLlmEngineI {
 public: 
  virtual ~CortexTensorrtLlmEngineI() {}

  virtual void HandleChatCompletion(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
  virtual void LoadModel(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
  virtual void Destroy(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;  
};
