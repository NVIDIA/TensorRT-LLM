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
#include "tensorrt_llm/kv_cache_compression/nvfp4ColdPageCodecBackend.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
namespace compression = tensorrt_llm::kv_cache_compression;
namespace kernels = tensorrt_llm::kernels;

namespace tensorrt_llm::nanobind::kv_cache_compression
{

void initBindings(nb::module_& module)
{
    nb::enum_<kernels::Nvfp4ColdPageRuntimeType>(module, "Nvfp4ColdPageRuntimeType")
        .value("FLOAT16", kernels::Nvfp4ColdPageRuntimeType::kFloat16)
        .value("BFLOAT16", kernels::Nvfp4ColdPageRuntimeType::kBfloat16)
        .value("FP8_E4M3", kernels::Nvfp4ColdPageRuntimeType::kFp8E4m3);

    nb::class_<compression::Nvfp4ColdPageScales>(module, "Nvfp4ColdPageScales")
        .def(nb::init<>())
        .def_rw("nvfp4_scale_orig_quant", &compression::Nvfp4ColdPageScales::nvfp4ScaleOrigQuant)
        .def_rw("nvfp4_scale_quant_orig", &compression::Nvfp4ColdPageScales::nvfp4ScaleQuantOrig)
        .def_rw("fp8_scale_orig_quant", &compression::Nvfp4ColdPageScales::fp8ScaleOrigQuant)
        .def_rw("fp8_scale_quant_orig", &compression::Nvfp4ColdPageScales::fp8ScaleQuantOrig);

    nb::class_<compression::Nvfp4ColdPageBufferLayout>(module, "Nvfp4ColdPageBufferLayout")
        .def(nb::init<>())
        .def_rw("role", &compression::Nvfp4ColdPageBufferLayout::role)
        .def_rw("cold_data_offset", &compression::Nvfp4ColdPageBufferLayout::coldDataOffset)
        .def_rw("cold_scale_offset", &compression::Nvfp4ColdPageBufferLayout::coldScaleOffset)
        .def_rw("scales", &compression::Nvfp4ColdPageBufferLayout::scales);

    nb::class_<compression::Nvfp4ColdPageLayerLayout>(module, "Nvfp4ColdPageLayerLayout")
        .def(nb::init<>())
        .def_rw("layer_id", &compression::Nvfp4ColdPageLayerLayout::layerId)
        .def_rw("runtime_type", &compression::Nvfp4ColdPageLayerLayout::runtimeType)
        .def_rw("num_kv_heads", &compression::Nvfp4ColdPageLayerLayout::numKvHeads)
        .def_rw("tokens_per_page", &compression::Nvfp4ColdPageLayerLayout::tokensPerPage)
        .def_rw("head_dim", &compression::Nvfp4ColdPageLayerLayout::headDim)
        .def_rw("cold_page_bytes", &compression::Nvfp4ColdPageLayerLayout::coldPageBytes)
        .def_rw("cold_padding_offset", &compression::Nvfp4ColdPageLayerLayout::coldPaddingOffset)
        .def_rw("buffers", &compression::Nvfp4ColdPageLayerLayout::buffers);

    module.def("create_nvfp4_cold_page_codec", &compression::createNvfp4ColdPageCodec, nb::arg("layer_layouts"));
}

} // namespace tensorrt_llm::nanobind::kv_cache_compression
