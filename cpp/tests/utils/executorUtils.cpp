#include "executorUtils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include <exception>
#include <future>

std::unordered_map<tensorrt_llm::batch_manager::RequestIdType, std::vector<tensorrt_llm::executor::Response>>
tensorrt_llm::testing::runThroughRequests(executor::Executor& executor, std::vector<executor::Request> const& requests,
    std::chrono::duration<float, std::milli> timeout)
{
    std::unordered_map<batch_manager::RequestIdType, std::vector<executor::Response>> accumulatedResponses;
    auto responseReadFuture = std::async(std::launch::async,
        [&]() -> std::optional<std::exception>
        {
            auto remainingRequests = requests.size();
            try
            {
                while (remainingRequests > 0)
                {
                    auto const responses = executor.awaitResponses();
                    for (auto const& response : responses)
                    {
                        auto const requestId = response.getRequestId();
                        if (response.hasError())
                        {
                            TLLM_LOG_ERROR("Error response received for request: %lu", requestId);
                            TLLM_THROW(response.getErrorMsg());
                        }
                        auto const isFinal = response.hasError() || response.getResult().isFinal;
                        accumulatedResponses[requestId].emplace_back(response);
                        if (isFinal)
                        {
                            TLLM_LOG_DEBUG("Final response received for request: %lu", requestId);
                            --remainingRequests;
                        }
                    }
                }

                return std::nullopt;
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_EXCEPTION(e);
                return e;
            }
        });
    auto const requestIds = executor.enqueueRequests(requests);
    responseReadFuture.wait_for(timeout);
    auto const readResult = responseReadFuture.get();
    if (readResult.has_value())
    {
        throw std::exception(readResult.value());
    }
    return accumulatedResponses;
}
