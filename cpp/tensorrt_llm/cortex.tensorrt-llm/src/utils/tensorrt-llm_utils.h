#pragma once
#include "cstdio"
#include "random"
#include "string"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <json/value.h>
#include <json/writer.h>
#include <ostream>
#include <regex>
#include <vector>
// Include platform-specific headers
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

namespace tensorrtllm_utils {

inline std::string GenerateRandomString(std::size_t length) {
  const std::string characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<> distribution(0, characters.size() - 1);
  std::string random_string(length, '\0');
  std::generate_n(random_string.begin(), length, [&]() { return characters[distribution(generator)]; });

  return random_string;
}

inline std::string CreateReturnJson(std::string const& id, std::string const& model, std::string const& content, 
                                    Json::Value finish_reason = Json::Value()) {
    Json::Value root;
    root["id"] = id;
    root["model"] = model;
    root["created"] = static_cast<int>(std::time(nullptr));
    root["object"] = "chat.completion.chunk";

    Json::Value choices_array(Json::arrayValue);
    Json::Value choice;

    choice["index"] = 0;
    Json::Value delta;
    delta["content"] = content;
    choice["delta"] = delta;
    choice["finish_reason"] = finish_reason;

    choices_array.append(choice);
    root["choices"] = choices_array;

    Json::StreamWriterBuilder writer;
    writer["indentation"] = ""; // This sets the indentation to an empty string,
                                // producing compact output.
    return Json::writeString(writer, root);
}

} // namespace tensorrtllm_utils
