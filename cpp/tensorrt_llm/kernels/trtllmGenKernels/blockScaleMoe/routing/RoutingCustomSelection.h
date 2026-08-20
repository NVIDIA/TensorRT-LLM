/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "RoutingKernel.h"

#include <cstdint>

namespace moe::dev::routing::routingCustom
{

//! Whether the cooperative block kernel is the preferred launcher for this shape.
//!
//! Split out of run() so the selection table can be unit tested without launching a
//! kernel. This is host-only and holds no state: both escape hatches,
//! TLLM_ROUTING_DISABLE_COOP_BLOCK and TLLM_ROUTING_COOP_BLOCK_MIN_EXPERTS, are read by
//! the caller and applied here only through arguments.
//!
//! \param preprocessType routing preprocess applied before top-k.
//! \param postprocessType routing postprocess applied to the top-k scores. Paired with
//!        preprocessType it identifies the policy, exactly as dispatchRoutingPolicy() does.
//! \param numTokens number of routing tokens in this launch.
//! \param dispatchedMaxExperts compile-time tier from queryDispatchedMaxExperts(), which
//!        is not the model's raw expert count.
//! \param minNumExpertsForCoopOverride replaces the built-in Renormalize lower tier bound
//!        when non-negative. 0 restores the parent behaviour of always preferring the
//!        cooperative kernel; a value above every tier forces the classic kernel. It has
//!        no effect on any other policy, which is not subject to the bound.
bool prefersCoopBlockKernel(RoutingPreprocessType preprocessType, RoutingPostprocessType postprocessType,
    int32_t numTokens, int32_t dispatchedMaxExperts, int32_t minNumExpertsForCoopOverride = -1);

} // namespace moe::dev::routing::routingCustom
