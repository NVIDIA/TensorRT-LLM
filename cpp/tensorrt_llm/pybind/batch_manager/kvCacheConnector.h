#pragma once

#include "tensorrt_llm/batch_manager/kvCacheConnector.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace tensorrt_llm::pybind::batch_manager::kv_connector
{

using namespace tensorrt_llm::batch_manager::kv_connector;

class PyKvCacheConnector : public KvCacheConnector
{
public:
    using KvCacheConnector::KvCacheConnector;

    //
    // WORKER SIDE METHODS
    //

    void registerKvCaches() override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, registerKvCaches);
    }

    void startLoadKv() override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, startLoadKv);
    }

    void waitForLayerLoad(SizeType32 layer_idx) override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, waitForLayerLoad, layer_idx);
    }

    void saveKvLayer(SizeType32 layer_idx) override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, saveKvLayer, layer_idx);
    }

    void waitForSave() override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, waitForSave);
    }

    using FinishedReqs = std::tuple<std::vector<RequestIdType>, std::vector<RequestIdType>>;

    FinishedReqs getFinished(std::vector<RequestIdType> const& finishedReqIds) override
    {
        PYBIND11_OVERRIDE_PURE(FinishedReqs, KvCacheConnector, getFinished, finishedReqIds);
    }

    //
    // SCHEDULER SIDE METHODS
    //

    using NumNewMatchedTokens = std::tuple<SizeType32, bool>;

    NumNewMatchedTokens getNumNewMatchedTokens(LlmRequest const& request, SizeType32 numComputedTokens) override
    {
        PYBIND11_OVERRIDE_PURE(
            NumNewMatchedTokens, KvCacheConnector, getNumNewMatchedTokens, request, numComputedTokens);
    }

    void updateStateAfterAlloc() override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, updateStateAfterAlloc);
    }

    bool requestFinished(LlmRequest const& request) override
    {
        PYBIND11_OVERRIDE_PURE(bool, KvCacheConnector, requestFinished, request);
    }
};

} // namespace tensorrt_llm::pybind::batch_manager::kv_connector
