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
#include "tensorrt_llm/kv_cache_compression/coldPageCodec.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <utility>
#include <vector>

namespace nb = nanobind;
namespace compression = tensorrt_llm::kv_cache_compression;
namespace kernels = tensorrt_llm::kernels;
namespace kv = tensorrt_llm::batch_manager::kv_cache_manager_v2;

namespace tensorrt_llm::nanobind::kv_cache_compression
{

void initBindings(nb::module_& module)
{
    nb::enum_<compression::ColdPageTransformKind>(module, "ColdPageTransformKind")
        .value("LOSSLESS_COPY", compression::ColdPageTransformKind::kLosslessCopy)
        .value("NVFP4", compression::ColdPageTransformKind::kNvfp4);

    nb::enum_<kernels::Nvfp4ColdPageRuntimeType>(module, "Nvfp4ColdPageRuntimeType")
        .value("FLOAT16", kernels::Nvfp4ColdPageRuntimeType::kFloat16)
        .value("BFLOAT16", kernels::Nvfp4ColdPageRuntimeType::kBfloat16)
        .value("FP8_E4M3", kernels::Nvfp4ColdPageRuntimeType::kFp8E4m3);

    nb::class_<compression::Nvfp4ColdPageParams>(module, "Nvfp4ColdPageParams")
        .def(nb::init<>())
        .def_rw("runtime_type", &compression::Nvfp4ColdPageParams::runtimeType)
        .def_rw("num_kv_heads", &compression::Nvfp4ColdPageParams::numKvHeads)
        .def_rw("tokens_per_page", &compression::Nvfp4ColdPageParams::tokensPerPage)
        .def_rw("head_dim", &compression::Nvfp4ColdPageParams::headDim)
        .def_rw("nvfp4_scale_orig_quant", &compression::Nvfp4ColdPageParams::nvfp4ScaleOrigQuant)
        .def_rw("nvfp4_scale_quant_orig", &compression::Nvfp4ColdPageParams::nvfp4ScaleQuantOrig)
        .def_rw("fp8_scale_orig_quant", &compression::Nvfp4ColdPageParams::fp8ScaleOrigQuant)
        .def_rw("fp8_scale_quant_orig", &compression::Nvfp4ColdPageParams::fp8ScaleQuantOrig);

    nb::class_<compression::ColdPageBufferPlan>(module, "ColdPageBufferPlan")
        .def(nb::init<>())
        .def_rw("role", &compression::ColdPageBufferPlan::role)
        .def_rw("transform", &compression::ColdPageBufferPlan::transform)
        .def_rw("raw_bytes", &compression::ColdPageBufferPlan::rawBytes)
        .def_rw("cold_data_offset", &compression::ColdPageBufferPlan::coldDataOffset)
        .def_rw("cold_scale_offset", &compression::ColdPageBufferPlan::coldScaleOffset)
        .def_rw("nvfp4_params", &compression::ColdPageBufferPlan::nvfp4Params);

    nb::class_<compression::ColdPageLayerPlan>(module, "ColdPageLayerPlan")
        .def(nb::init<>())
        .def_rw("layer_id", &compression::ColdPageLayerPlan::layerId)
        .def_rw("cold_page_bytes", &compression::ColdPageLayerPlan::coldPageBytes)
        .def_rw("cold_padding_offset", &compression::ColdPageLayerPlan::coldPaddingOffset)
        .def_rw("cold_padding_bytes", &compression::ColdPageLayerPlan::coldPaddingBytes)
        .def_rw("buffers", &compression::ColdPageLayerPlan::buffers);

    // Construct in C++ so ownership can transfer to KVCM as a unique_ptr codec.
    module.def(
        "create_cold_page_codec",
        [](std::vector<compression::ColdPageLayerPlan> layerPlans) -> std::unique_ptr<kv::IKvCacheColdPageCodec>
        { return std::make_unique<compression::PlannedColdPageCodec>(std::move(layerPlans)); },
        nb::arg("layer_plans"), "Create an owning planned cold-page codec for transfer into KVCacheManager.");
}

} // namespace tensorrt_llm::nanobind::kv_cache_compression
