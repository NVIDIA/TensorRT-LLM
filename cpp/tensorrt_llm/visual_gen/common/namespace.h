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

#ifndef TRTLLM_VG_SCOPE
#error "VisualGen native code must be compiled with add_visual_gen_native"
#endif

#define TRTLLM_VG_NAMESPACE_BEGIN                                                                                      \
    TRTLLM_NAMESPACE_BEGIN                                                                                             \
    namespace visual_gen::TRTLLM_VG_SCOPE                                                                              \
    {

#define TRTLLM_VG_NAMESPACE_END                                                                                        \
    }                                                                                                                  \
    TRTLLM_NAMESPACE_END

#define TRTLLM_VG_NS ::tensorrt_llm::visual_gen::TRTLLM_VG_SCOPE
