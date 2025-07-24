#pragma once

#include "tensorrt_llm/batch_manager/kvCacheConnector.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class KVCacheManagerConnectorBindings
{
public:
    static void initBindings(pybind11::module_& m);
};
} // namespace tensorrt_llm::batch_manager::kv_cache_manager

namespace tensorrt_llm::pybind::batch_manager::kv_connector
{

using namespace tensorrt_llm::batch_manager::kv_connector;

} // namespace tensorrt_llm::pybind::batch_manager::kv_connector
