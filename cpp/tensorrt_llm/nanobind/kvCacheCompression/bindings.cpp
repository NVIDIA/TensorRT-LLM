/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "bindings.h"
#include "tensorrt_llm/kv_cache_compression/nativeColdPageCodec.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace nb = nanobind;
namespace compression = tensorrt_llm::kv_cache_compression;
namespace kv = tensorrt_llm::batch_manager::kv_cache_manager_v2;

namespace tensorrt_llm::nanobind::kv_cache_compression
{
namespace
{

static_assert(sizeof(kv::PageIndexPair) == 8);
static_assert(alignof(kv::PageIndexPair) == 8);
static_assert(offsetof(kv::PageIndexPair, dst) == 0);
static_assert(offsetof(kv::PageIndexPair, src) == 4);
static_assert(std::is_trivially_copyable_v<kv::PageIndexPair>);

//! Algorithm-neutral adapter from KVCM's native codec calls to one Python policy.
class PythonColdPageCodec final : public compression::NativeColdPageCodec
{
public:
    explicit PythonColdPageCodec(nb::handle policy)
        : NativeColdPageCodec(readLayerIds(policy))
        , mPolicy(policy.ptr())
    {
        Py_INCREF(mPolicy);
    }

    ~PythonColdPageCodec() override
    {
        if (Py_IsInitialized())
        {
            nb::gil_scoped_acquire acquire;
            Py_DECREF(mPolicy);
        }
    }

private:
    static std::set<kv::LayerId> readLayerIds(nb::handle policy)
    {
        if (policy.is_none())
        {
            throw std::invalid_argument("Cold-page policy must not be None");
        }
        auto const layerIds = nb::cast<std::vector<kv::LayerId>>(policy.attr("layer_ids"));
        std::set<kv::LayerId> result(layerIds.begin(), layerIds.end());
        if (result.size() != layerIds.size())
        {
            throw std::invalid_argument("Cold-page policy layer IDs must be unique");
        }
        return result;
    }

    std::vector<compression::ColdPageLifecycleProperties> configurePolicy(
        std::vector<compression::ResolvedHotLifecycle> const& lifecycles) override
    {
        nb::gil_scoped_acquire acquire;
        try
        {
            return nb::cast<std::vector<compression::ColdPageLifecycleProperties>>(
                nb::borrow<nb::object>(mPolicy).attr("configure")(lifecycles));
        }
        catch (nb::python_error const& error)
        {
            throw std::runtime_error(error.what());
        }
    }

    void encodePolicy(std::size_t lifecycleIndex, void* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream) override
    {
        invoke("encode", lifecycleIndex, coldBase, pageIndices, numPages, stream);
    }

    void decodePolicy(std::size_t lifecycleIndex, void const* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream) override
    {
        invoke("decode", lifecycleIndex, coldBase, pageIndices, numPages, stream);
    }

    template <typename ColdPointer>
    void invoke(char const* method, std::size_t lifecycleIndex, ColdPointer coldBase,
        kv::PageIndexPair const* pageIndices, std::size_t numPages, cudaStream_t stream)
    {
        // Forward the complete KVCM batch once. The method custom op owns any launch chunking.
        nb::gil_scoped_acquire acquire;
        try
        {
            nb::borrow<nb::object>(mPolicy).attr(method)(lifecycleIndex, reinterpret_cast<std::uintptr_t>(coldBase),
                reinterpret_cast<std::uintptr_t>(pageIndices), numPages, reinterpret_cast<std::uintptr_t>(stream));
        }
        catch (nb::python_error const& error)
        {
            throw std::runtime_error(error.what());
        }
    }

    PyObject* mPolicy;
};

std::unique_ptr<kv::IKvCacheColdPageCodec> createPythonColdPageCodec(nb::handle policy)
{
    return std::make_unique<PythonColdPageCodec>(policy);
}

} // namespace

void initBindings(nb::module_& module)
{
    nb::enum_<kv::PageIndexLocation>(module, "ColdPageIndexLocation")
        .value("BAD_LOCATION", kv::PageIndexLocation::kBadLocation)
        .value("HOST", kv::PageIndexLocation::kHost)
        .value("DEVICE", kv::PageIndexLocation::kDevice);

    nb::class_<compression::ResolvedHotBuffer>(module, "ResolvedHotBuffer")
        .def_ro("raw_base", &compression::ResolvedHotBuffer::rawBase)
        .def_ro("raw_slot_bytes", &compression::ResolvedHotBuffer::rawSlotBytes)
        .def_ro("raw_bytes", &compression::ResolvedHotBuffer::rawBytes);

    nb::class_<compression::ResolvedHotLifecycle>(module, "ResolvedHotLifecycle")
        .def_prop_ro("life_cycle_id",
            [](compression::ResolvedHotLifecycle const& lifecycle) { return lifecycle.lifeCycleId.value(); })
        .def_ro("layers", &compression::ResolvedHotLifecycle::layers);

    nb::class_<compression::ColdPageLifecycleProperties>(module, "ColdPageLifecycleProperties")
        .def(nb::init<>())
        .def_rw("cold_page_bytes", &compression::ColdPageLifecycleProperties::coldPageBytes)
        .def_rw("page_index_location", &compression::ColdPageLifecycleProperties::pageIndexLocation);

    module.def("create_python_cold_page_codec", &createPythonColdPageCodec, nb::arg("policy"));
}

} // namespace tensorrt_llm::nanobind::kv_cache_compression
