#pragma once

#include "tensorrt_llm/batch_manager/kvCacheConnector.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class KVCacheManagerConnectorBindings
{
public:
    static void initBindings(nb::module_& m);
};
} // namespace tensorrt_llm::batch_manager::kv_cache_manager

namespace tensorrt_llm::pybind::batch_manager::kv_connector
{

using namespace tensorrt_llm::batch_manager::kv_connector;

} // namespace tensorrt_llm::pybind::batch_manager::kv_connector
