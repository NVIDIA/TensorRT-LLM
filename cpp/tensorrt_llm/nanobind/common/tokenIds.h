/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace NanobindUtils
{

namespace nb = nanobind;

//! Token ids from a Python list[int] or a 1-D contiguous int32 CPU ndarray. The ndarray is copied with
//! one memcpy instead of O(len) PyLong casts under the GIL.
inline std::vector<int32_t> toVecTokens(nb::handle ids)
{
    nb::ndarray<int32_t const, nb::ndim<1>, nb::c_contig, nb::device::cpu> arr;
    if (nb::try_cast(ids, arr, /*convert=*/false))
    {
        std::vector<int32_t> out(arr.shape(0));
        if (arr.shape(0) > 0)
        {
            std::memcpy(out.data(), arr.data(), arr.shape(0) * sizeof(int32_t));
        }
        return out;
    }
    return nb::cast<std::vector<int32_t>>(ids);
}

} // namespace NanobindUtils
