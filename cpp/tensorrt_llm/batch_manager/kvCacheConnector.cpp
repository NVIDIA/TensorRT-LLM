#include "tensorrt_llm/batch_manager/kvCacheConnector.h"

namespace tensorrt_llm::batch_manager::kv_connector
{

void KvCacheConnectorWorker::registerKvCaches(KvCacheConnectorPoolsData const& kvCacheConnectorPoolsData) {}

std::tuple<std::vector<RequestIdType>, std::vector<RequestIdType>> KvCacheConnectorWorker::getFinished(
    std::vector<RequestIdType> const& finishedReqIds)
{
    return std::make_tuple(std::vector<RequestIdType>(), std::vector<RequestIdType>());
}

void KvCacheConnectorScheduler::updateStateAfterAlloc() {}

bool KvCacheConnectorScheduler::requestFinished(LlmRequest const& request)
{
    return false;
}

} // namespace tensorrt_llm::batch_manager::kv_connector
