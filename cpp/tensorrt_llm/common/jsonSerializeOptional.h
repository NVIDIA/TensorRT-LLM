/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once
#include <nlohmann/json.hpp>
#include <optional>

namespace nlohmann
{

template <typename T>
struct adl_serializer<std::optional<T>>
{
    static void to_json(nlohmann::json& j, std::optional<T> const& opt)
    {
        if (opt == std::nullopt)
        {
            j = nullptr;
        }
        else
        {
            j = opt.value(); // this will call adl_serializer<T>::to_json which will
                             // find the free function to_json in T's namespace!
        }
    }

    static void from_json(nlohmann::json const& j, std::optional<T>& opt)
    {
        if (j.is_null())
        {
            opt = std::nullopt;
        }
        else
        {
            opt = j.template get<T>(); // same as above, but with
                                       // adl_serializer<T>::from_json
        }
    }
};
} // namespace nlohmann
