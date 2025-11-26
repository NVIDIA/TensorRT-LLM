#ifndef A073F2DA_315E_434B_B811_D420F0A59DF3
#define A073F2DA_315E_434B_B811_D420F0A59DF3

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/executor/executor.h"
#include <ratio>
#include <unordered_map>

TRTLLM_NAMESPACE_BEGIN
namespace testing
{

std::unordered_map<batch_manager::RequestIdType, std::vector<executor::Response>> runThroughRequests(
    executor::Executor& executor, std::vector<executor::Request> const& requests,
    std::chrono::duration<float, std::milli> timeout);

} // namespace testing
TRTLLM_NAMESPACE_END

#endif /* A073F2DA_315E_434B_B811_D420F0A59DF3 */
