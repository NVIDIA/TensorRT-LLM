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

#include "kv_cache_manager_v2/localityDomain.h"
#include "kv_cache_manager_v2/exceptions.h"

#include "tensorrt_llm/runtime/locality_domain/localityDomainResourceConfig.h"
#include "tensorrt_llm/runtime/locality_domain/locality_domain_utils.h"

#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

LocalizationMode localizationMode()
{
    char const* value = std::getenv("TRT_LLM_MOCK_LOCALIZATION_SUPPORT");
    if (value == nullptr)
    {
        return LocalizationMode::AUTO;
    }
    std::string_view const mode{value};
    if (mode == "0")
    {
        return LocalizationMode::OFF;
    }
    if (mode == "1")
    {
        return LocalizationMode::MOCK;
    }
    throw std::invalid_argument("TRT_LLM_MOCK_LOCALIZATION_SUPPORT must be '0' or '1' when set");
}

int resolveNumLocalityDomains(bool enableLocalityDomains)
{
    if (!enableLocalityDomains)
    {
        return 1;
    }
    switch (localizationMode())
    {
    case LocalizationMode::OFF: return 1;
    case LocalizationMode::MOCK: return locality_domain::detail::kLocalityDomainCount;
    case LocalizationMode::AUTO:
    {
        CUdevice device{};
        cuCheck(cuCtxGetDevice(&device));
        return locality_domain::deviceSupportsLocalization(device) ? locality_domain::detail::kLocalityDomainCount : 1;
    }
    }
    return 1;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
