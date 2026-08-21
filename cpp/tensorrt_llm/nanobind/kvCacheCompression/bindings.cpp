/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/kv_cache_compression/nvfp4ColdPageCodec.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
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
    nb::enum_<kernels::Nvfp4BoundaryRuntimeType>(module, "Nvfp4BoundaryRuntimeType")
        .value("FLOAT16", kernels::Nvfp4BoundaryRuntimeType::kFloat16)
        .value("BFLOAT16", kernels::Nvfp4BoundaryRuntimeType::kBfloat16)
        .value("FP8_E4M3", kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3);

    nb::class_<compression::Nvfp4ColdPageLayerConfig>(module, "Nvfp4ColdPageLayerConfig")
        .def(nb::init<>())
        .def_rw("layer_id", &compression::Nvfp4ColdPageLayerConfig::layerId)
        .def_rw("runtime_type", &compression::Nvfp4ColdPageLayerConfig::runtimeType)
        .def_rw("num_kv_heads", &compression::Nvfp4ColdPageLayerConfig::numKvHeads)
        .def_rw("tokens_per_page", &compression::Nvfp4ColdPageLayerConfig::tokensPerPage)
        .def_rw("head_dim", &compression::Nvfp4ColdPageLayerConfig::headDim)
        .def_rw("nvfp4_scale_orig_quant", &compression::Nvfp4ColdPageLayerConfig::nvfp4ScaleOrigQuant)
        .def_rw("nvfp4_scale_quant_orig", &compression::Nvfp4ColdPageLayerConfig::nvfp4ScaleQuantOrig)
        .def_rw("fp8_scale_orig_quant", &compression::Nvfp4ColdPageLayerConfig::fp8ScaleOrigQuant)
        .def_rw("fp8_scale_quant_orig", &compression::Nvfp4ColdPageLayerConfig::fp8ScaleQuantOrig);

    // Construct in C++ so ownership can transfer to KVCM as a unique_ptr codec.
    module.def(
        "create_nvfp4_cold_page_codec",
        [](std::vector<compression::Nvfp4ColdPageLayerConfig> layerConfigs)
            -> std::unique_ptr<kv::IKvCacheColdPageCodec>
        { return std::make_unique<compression::Nvfp4ColdPageCodec>(std::move(layerConfigs)); },
        nb::arg("layer_configs"), "Create an owning NVFP4 cold-page codec for one-time transfer into KVCacheManager.");
}

} // namespace tensorrt_llm::nanobind::kv_cache_compression
