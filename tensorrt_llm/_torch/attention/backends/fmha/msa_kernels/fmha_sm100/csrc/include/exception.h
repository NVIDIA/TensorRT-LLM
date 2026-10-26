/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_EXCEPTION_H_
#define FLASHINFER_EXCEPTION_H_

#include <exception>
#include <iostream>
#include <sstream>

#define FLASHINFER_ERROR(message) throw flashinfer::Error(__FUNCTION__, __FILE__, __LINE__, message)

// Base case for empty arguments
inline void write_to_stream(std::ostringstream& oss) {
  // No-op for empty arguments
}

template <typename T>
void write_to_stream(std::ostringstream& oss, T&& val) {
  oss << std::forward<T>(val);
}

template <typename T, typename... Args>
void write_to_stream(std::ostringstream& oss, T&& val, Args&&... args) {
  oss << std::forward<T>(val) << " ";
  write_to_stream(oss, std::forward<Args>(args)...);
}

// Helper macro to handle empty __VA_ARGS__
#define FLASHINFER_CHECK_IMPL(condition, message) \
  if (!(condition)) {                             \
    FLASHINFER_ERROR(message);                    \
  }

// Main macro that handles both cases
#define FLASHINFER_CHECK(condition, ...)   \
  do {                                     \
    if (!(condition)) {                    \
      std::ostringstream oss;              \
      write_to_stream(oss, ##__VA_ARGS__); \
      std::string msg = oss.str();         \
      if (msg.empty()) {                   \
        msg = "Check failed: " #condition; \
      }                                    \
      FLASHINFER_ERROR(msg);               \
    }                                      \
  } while (0)

// Warning macro
#define FLASHINFER_WARN(...)                                           \
  do {                                                                 \
    std::ostringstream oss;                                            \
    write_to_stream(oss, ##__VA_ARGS__);                               \
    std::string msg = oss.str();                                       \
    if (msg.empty()) {                                                 \
      msg = "Warning triggered";                                       \
    }                                                                  \
    flashinfer::Warning(__FUNCTION__, __FILE__, __LINE__, msg).emit(); \
  } while (0)

namespace flashinfer {
class Error : public std::exception {
 private:
  std::string message_;

 public:
  Error(const std::string& func, const std::string& file, int line, const std::string& message) {
    std::ostringstream oss;
    oss << "Error in function '" << func << "' "
        << "at " << file << ":" << line << ": " << message;
    message_ = oss.str();
  }

  virtual const char* what() const noexcept override { return message_.c_str(); }
};

class Warning {
 private:
  std::string message_;

 public:
  Warning(const std::string& func, const std::string& file, int line, const std::string& message) {
    std::ostringstream oss;
    oss << "Warning in function '" << func << "' "
        << "at " << file << ":" << line << ": " << message;
    message_ = oss.str();
  }

  void emit() const { std::cerr << message_ << std::endl; }
};

}  // namespace flashinfer

#endif  // FLASHINFER_EXCEPTION_H_
