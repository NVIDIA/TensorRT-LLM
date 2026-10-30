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

#include "tensorrt_llm/common/config.h"

// add_modeling_v2_target() and add_modeling_v2_shared() define TRTLLM_MV2_SCOPE
// as `<family>::<target>` or `<family>::shared` (cpp/cmake/modules/modeling_v2.cmake).
#ifndef TRTLLM_MV2_SCOPE
#error "modeling_v2 native code is compiled through add_modeling_v2_target() or add_modeling_v2_shared()"
#endif

//! Opens tensorrt_llm::modeling_v2::<family>::<target> (or ::shared) for the unit
//! being compiled. Every target has its own C++ namespace, so sibling targets can
//! carry identically named code in one family library.
#define TRTLLM_MV2_NAMESPACE_BEGIN                                                                                     \
    TRTLLM_NAMESPACE_BEGIN                                                                                             \
    namespace modeling_v2::TRTLLM_MV2_SCOPE                                                                            \
    {

#define TRTLLM_MV2_NAMESPACE_END                                                                                       \
    }                                                                                                                  \
    TRTLLM_NAMESPACE_END

//! The fully qualified namespace TRTLLM_MV2_NAMESPACE_BEGIN opens.
#define TRTLLM_MV2_NS ::tensorrt_llm::modeling_v2::TRTLLM_MV2_SCOPE
