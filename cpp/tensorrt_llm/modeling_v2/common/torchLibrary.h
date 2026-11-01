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

// add_modeling_v2_target() defines TRTLLM_MV2_TORCH_NS as mv2_<family>_<target>
// and add_modeling_v2_shared() as mv2_<family> (cpp/cmake/modules/modeling_v2.cmake). The
// family library claims each namespace once with TORCH_LIBRARY, so op files only
// ever add fragments to it.
#ifndef TRTLLM_MV2_TORCH_NS
#error "modeling_v2 torch bindings are compiled through add_modeling_v2_target() or add_modeling_v2_shared()"
#endif

// One level of indirection so that TORCH_LIBRARY_*'s own # and ## operators see
// the expanded namespace rather than the macro name.
#define TRTLLM_MV2_DETAIL_LIBRARY_FRAGMENT(ns, m) TORCH_LIBRARY_FRAGMENT(ns, m)
#define TRTLLM_MV2_DETAIL_LIBRARY_IMPL(ns, k, m) TORCH_LIBRARY_IMPL(ns, k, m)

//! TORCH_LIBRARY_FRAGMENT on this unit's own namespace: declare op schemas.
#define TRTLLM_MV2_LIBRARY_FRAGMENT(m) TRTLLM_MV2_DETAIL_LIBRARY_FRAGMENT(TRTLLM_MV2_TORCH_NS, m)

//! TORCH_LIBRARY_IMPL on this unit's own namespace: register kernels for dispatch key `k`.
#define TRTLLM_MV2_LIBRARY_IMPL(k, m) TRTLLM_MV2_DETAIL_LIBRARY_IMPL(TRTLLM_MV2_TORCH_NS, k, m)
