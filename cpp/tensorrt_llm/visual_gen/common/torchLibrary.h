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

#include <torch/library.h>

#ifndef TRTLLM_VG_TORCH_NS
#error "VisualGen Torch bindings must be compiled with add_visual_gen_native"
#endif

#define TRTLLM_VG_DETAIL_FRAGMENT(ns, m) TORCH_LIBRARY_FRAGMENT(ns, m)
#define TRTLLM_VG_DETAIL_IMPL(ns, key, m) TORCH_LIBRARY_IMPL(ns, key, m)

#define TRTLLM_VG_LIBRARY_FRAGMENT(m) TRTLLM_VG_DETAIL_FRAGMENT(TRTLLM_VG_TORCH_NS, m)
#define TRTLLM_VG_LIBRARY_IMPL(key, m) TRTLLM_VG_DETAIL_IMPL(TRTLLM_VG_TORCH_NS, key, m)
