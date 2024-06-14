#include "server.h"

#include "json/reader.h"
#include "httplib.h"
#include "trantor/utils/Logger.h"

using SyncQueue = Server::SyncQueue;

int main(int argc, char** argv) {
  std::string hostname = "127.0.0.1";
  int port = 3928;
  if (argc > 1) {
    hostname = argv[1];
  }

  // Check for port argument
  if (argc > 2) {
    port = std::atoi(argv[2]);  // Convert string argument to int
  }

  Server server;
  Json::Reader r;
  auto svr = std::make_unique<httplib::Server>();

  if (!svr->bind_to_port(hostname, port)) {
    fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n",
            hostname.c_str(), port);
    return 1;
  }

  auto process_stream_res = [&server](httplib::Response& resp,
                                      std::shared_ptr<SyncQueue> q) {
    const auto chunked_content_provider =
        [&server, q](size_t size, httplib::DataSink& sink) {
          while (true) {
            auto [status, res] = q->wait_and_pop();
            auto str = res["data"].asString();
            LOG_TRACE << "data: " << str;

            if (!sink.write(str.c_str(), str.size())) {
              LOG_WARN << "Failed to write";
              //   return false;
            }
            if (status["has_error"].asBool() || status["is_done"].asBool()) {
              LOG_INFO << "Done";
              sink.done();
              break;
            }
          }

          return true;
        };
    resp.set_chunked_content_provider("text/event-stream",
                                      chunked_content_provider,
                                      [](bool) { LOG_INFO << "Done"; });
  };

  const auto handle_load_model = [&](const httplib::Request& req,
                                     httplib::Response& resp) {
    resp.set_header("Access-Control-Allow-Origin",
                    req.get_header_value("Origin"));
    auto req_body = std::make_shared<Json::Value>();
    r.parse(req.body, *req_body);
    server.engine_->LoadModel(
        req_body, [&server, &resp](Json::Value status, Json::Value res) {
          resp.set_content(res.toStyledString().c_str(),
                           "application/json; charset=utf-8");
          resp.status = status["status_code"].asInt();
        });
  };

  const auto handle_completions = [&](const httplib::Request& req,
                                      httplib::Response& resp) {
    resp.set_header("Access-Control-Allow-Origin",
                    req.get_header_value("Origin"));
    auto req_body = std::make_shared<Json::Value>();
    r.parse(req.body, *req_body);
    bool is_stream = (*req_body).get("stream", false).asBool();
    // This is an async call, need to use queue
    auto q = std::make_shared<SyncQueue>();
    server.engine_->HandleChatCompletion(
        req_body, [&server, q](Json::Value status, Json::Value res) {
          q->push(std::make_pair(status, res));
        });
    process_stream_res(resp, q);
  };

  // Use POST since httplib does not read request body for GET method
  svr->Post("/inferences/tensorrt-llm/loadmodel", handle_load_model);
  svr->Post("/v1/chat/completions", handle_completions);

  LOG_INFO << "HTTP server listening: " << hostname << ":" << port;
  svr->new_task_queue = [] {
    return new httplib::ThreadPool(5);
  };
  // run the HTTP server in a thread - see comment below
  std::thread t([&]() {
    if (!svr->listen_after_bind()) {
      return 1;
    }

    return 0;
  });
  std::atomic<bool> running = true;

  shutdown_handler = [&](int) {
    running = false;
  };
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
  struct sigaction sigint_action;
  sigint_action.sa_handler = signal_handler;
  sigemptyset(&sigint_action.sa_mask);
  sigint_action.sa_flags = 0;
  sigaction(SIGINT, &sigint_action, NULL);
  sigaction(SIGTERM, &sigint_action, NULL);
#elif defined(_WIN32)
  auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
    return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
  };
  SetConsoleCtrlHandler(
      reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

  while (running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  svr->stop();
  t.join();
  LOG_DEBUG << "Server shutdown";
  return 0;
}
