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

#include "transferAgent.h"

#include <nanobind/nanobind.h>

namespace tensorrt_llm::executor::kv_cache::bounce_v2
{

/// Register the bounce v2 mechanism-layer bindings (CompletionPoller / BatchedCopyPool /
/// FabricArena) on `m` and the below-the-splitter agent primitives (register_region /
/// deregister_region / post_transfer_1to1) on the already-registered NixlTransferAgent class.
/// Compiled only when NIXL is enabled (the bounce_v2 sources live in the NIXL wrapper).
void initBounceV2Bindings(nanobind::module_& m,
    nanobind::class_<tensorrt_llm::executor::kv_cache::NixlTransferAgent,
        tensorrt_llm::executor::kv_cache::BaseTransferAgent>& agentCls);

} // namespace tensorrt_llm::executor::kv_cache::bounce_v2
