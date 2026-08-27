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

//! Algorithm-neutral adapter from KVCM migration calls to a Python provider.
class PythonColdPageCodec final : public compression::NativeColdPageCodec
{
public:
    PythonColdPageCodec(nb::handle provider, nb::handle codecState)
        : NativeColdPageCodec(readLayerIds(codecState))
        , mProvider(provider.ptr())
        , mCodecState(codecState.ptr())
    {
        Py_INCREF(mProvider);
        Py_INCREF(mCodecState);
    }

    ~PythonColdPageCodec() override
    {
        if (Py_IsInitialized())
        {
            nb::gil_scoped_acquire acquire;
            Py_DECREF(mCodecState);
            Py_DECREF(mProvider);
        }
    }

private:
    static std::set<kv::LayerId> readLayerIds(nb::handle codecState)
    {
        if (codecState.is_none())
        {
            throw std::invalid_argument("Cold-page codec state must not be None");
        }
        auto const layerIds = nb::cast<std::vector<kv::LayerId>>(codecState.attr("layer_ids"));
        std::set<kv::LayerId> result(layerIds.begin(), layerIds.end());
        if (result.size() != layerIds.size())
        {
            throw std::invalid_argument("Cold-page codec state layer IDs must be unique");
        }
        return result;
    }

    std::vector<compression::ColdPageLifecycleProperties> configureProvider(
        std::vector<compression::ResolvedHotLifecycle> const& lifecycles) override
    {
        nb::gil_scoped_acquire acquire;
        try
        {
            return nb::cast<std::vector<compression::ColdPageLifecycleProperties>>(
                nb::borrow<nb::object>(mProvider).attr("configure")(nb::borrow<nb::object>(mCodecState), lifecycles));
        }
        catch (nb::python_error const& error)
        {
            throw std::runtime_error(error.what());
        }
    }

    void encodeProvider(std::size_t lifecycleIndex, void* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream) override
    {
        invoke("encode_cold_pages", lifecycleIndex, coldBase, pageIndices, numPages, stream);
    }

    void decodeProvider(std::size_t lifecycleIndex, void const* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream) override
    {
        invoke("decode_cold_pages", lifecycleIndex, coldBase, pageIndices, numPages, stream);
    }

    template <typename ColdPointer>
    void invoke(char const* method, std::size_t lifecycleIndex, ColdPointer coldBase,
        kv::PageIndexPair const* pageIndices, std::size_t numPages, cudaStream_t stream)
    {
        // Forward the complete KVCM batch once. The native launcher owns any chunking.
        nb::gil_scoped_acquire acquire;
        try
        {
            nb::borrow<nb::object>(mProvider).attr(method)(nb::borrow<nb::object>(mCodecState), lifecycleIndex,
                reinterpret_cast<std::uintptr_t>(coldBase), reinterpret_cast<std::uintptr_t>(pageIndices), numPages,
                reinterpret_cast<std::uintptr_t>(stream));
        }
        catch (nb::python_error const& error)
        {
            throw std::runtime_error(error.what());
        }
    }

    PyObject* mProvider;
    PyObject* mCodecState;
};

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

    module.def(
        "create_python_cold_page_codec",
        [](nb::handle provider, nb::handle codecState) -> std::unique_ptr<kv::IKvCacheColdPageCodec>
        { return std::make_unique<PythonColdPageCodec>(provider, codecState); },
        nb::arg("provider"), nb::arg("codec_state"));
}

} // namespace tensorrt_llm::nanobind::kv_cache_compression
