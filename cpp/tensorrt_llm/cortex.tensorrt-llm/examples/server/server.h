#pragma once

#include "cortex-common/cortextensorrtllmi.h"
#include "dylib.h"

#include <condition_variable>
#include <mutex>
#include <queue>

class Server {
 public:
  Server() {
    dylib_ = std::make_unique<dylib>("./engines/cortex.tensorrt-llm", "engine");
    auto func = dylib_->get_function<CortexTensorrtLlmEngineI*()>("get_engine");
    engine_ = func();
  }

  ~Server() {
    if (engine_) {
      delete engine_;
    }
  }

 public:
  std::unique_ptr<dylib> dylib_;
  CortexTensorrtLlmEngineI* engine_;

  struct SyncQueue {
    void push(std::pair<Json::Value, Json::Value>&& p) {
      std::unique_lock<std::mutex> l(mtx);
      q.push(p);
      cond.notify_one();
    }

    std::pair<Json::Value, Json::Value> wait_and_pop() {
      std::unique_lock<std::mutex> l(mtx);
      cond.wait(l, [this] { return !q.empty(); });
      auto res = q.front();
      q.pop();
      return res;
    }

    std::mutex mtx;
    std::condition_variable cond;
    // Status and result
    std::queue<std::pair<Json::Value, Json::Value>> q;
  };
};

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
  if (is_terminating.test_and_set()) {
    // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
    // this is for better developer experience, we can remove when the server is stable enough
    fprintf(stderr, "Received second interrupt, terminating immediately.\n");
    exit(1);
  }

  shutdown_handler(signal);
}
