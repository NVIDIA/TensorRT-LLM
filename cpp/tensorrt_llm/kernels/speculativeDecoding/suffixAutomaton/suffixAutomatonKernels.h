/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * Adapted from Baseten's sa_spec library (Apache-2.0)
 * https://github.com/basetenlabs/sa_spec
 */

#pragma once

// Full SA class definition — needed by .cu files that operate on SuffixAutomaton
// objects. This header redefines cudaStream_t when __CUDACC__ is not defined
// (via saCudaCallable.h), so only include this header from CUDA translation units.
#include "suffixAutomaton.h"

// Param structs and function declarations — shared with nanobind bindings.
#include "suffixAutomatonParams.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include <cuda_fp16.h>
