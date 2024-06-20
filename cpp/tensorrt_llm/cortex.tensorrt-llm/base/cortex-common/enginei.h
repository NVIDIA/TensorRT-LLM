#pragma once

#include <functional>
#include <memory>

#include "json/value.h"

// Interface for inference engine.
// Note: only append new function to keep the compatibility.
class EngineI {
 public:
  virtual ~EngineI() {}

  virtual void HandleChatCompletion(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
  virtual void HandleEmbedding(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
  virtual void LoadModel(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
  virtual void UnloadModel(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
  virtual void GetModelStatus(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;

  // For backward compatible checking, add to list when we add more APIs
  virtual bool IsSupported(const std::string& f) {
    if (f == "HandleChatCompletion" || f == "HandleEmbedding" ||
        f == "UnloadModel" || f == "GetModelStatus" ||
        f == "GetModels") {
      return true;
    }
    return false;
  }

  // API to get running models.
  virtual void GetModels(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
};
