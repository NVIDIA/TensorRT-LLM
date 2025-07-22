#include "tensorrt_llm/batch_manager/kvCacheConnector.h"

namespace tensorrt_llm::batch_manager::kv_connector
{

KvCacheConnector::KvCacheConnector(KvCacheConnectorRole role)
    : mRole(role)
{
}

KvCacheConnectorRole KvCacheConnector::role() const
{
    return mRole;
}

void KvCacheConnector::registerKvCaches() {}

std::tuple<std::vector<RequestIdType>, std::vector<RequestIdType>> KvCacheConnector::getFinished(
    std::vector<RequestIdType> const& finishedReqIds)
{
    return std::make_tuple(std::vector<RequestIdType>(), std::vector<RequestIdType>());
}

void KvCacheConnector::updateStateAfterAlloc() {}

bool KvCacheConnector::requestFinished(LlmRequest const& request)
{
    return false;
}

} // namespace tensorrt_llm::batch_manager::kv_connector
