/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "kvCacheManager.h"
#include "tensorrt_llm/pybind/utils/bindTypes.h"

namespace tb = tensorrt_llm::batch_manager;
namespace py = pybind11;

void tb::kv_cache_manager::KVCacheManagerBindings::initBindings(py::module_& m)
{
    // TODO: Provide proper bindings
    py::classh<tb::kv_cache_manager::KVCacheManager>(m, "KVCacheManager");
}

void tb::BasePeftCacheManagerBindings::initBindings(py::module_& m)
{
    // TODO: Provide proper bindings
    py::classh<tb::BasePeftCacheManager>(m, "BasePeftCacheManager");
}
