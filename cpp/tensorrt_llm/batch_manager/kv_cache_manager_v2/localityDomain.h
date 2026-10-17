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

#pragma once

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

enum class LocalizationMode
{
    AUTO,
    OFF,
    MOCK,
};

// TRT_LLM_MOCK_LOCALIZATION_SUPPORT: unset probes the device, "0" disables locality domains,
// "1" reports the fixed Rubin domain count on any device while backing every domain with
// ordinary device memory.
LocalizationMode localizationMode();

// 1 when locality domains are disabled or the current device has none, otherwise the domain count.
int resolveNumLocalityDomains(bool enableLocalityDomains);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
