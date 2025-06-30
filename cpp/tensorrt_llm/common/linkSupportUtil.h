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
 
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm
{
namespace common
{

#if ENABLE_MULTI_DEVICE

std::set<int> getLocalGroup(std::set<int> const& group);
std::tuple<bool, bool> setGroupTopology(std::set<int> group);
#endif // ENABLE_MULTI_DEVICE

std::vector<bool> initGroupTopology(std::set<int> group);

} // namespace common
} // namespace tensorrt_llm
