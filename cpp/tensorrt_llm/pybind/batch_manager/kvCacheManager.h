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

#pragma once

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/pybind/common/opaqueBindings.h"
#include <pybind11/pybind11.h>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class KVCacheManagerBindings
{
public:
    static void initBindings(pybind11::module_& m);
};
} // namespace tensorrt_llm::batch_manager::kv_cache_manager

namespace tensorrt_llm::batch_manager
{
class BasePeftCacheManagerBindings
{
public:
    static void initBindings(pybind11::module_& m);
};
} // namespace tensorrt_llm::batch_manager
