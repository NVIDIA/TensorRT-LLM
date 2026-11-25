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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/ropeKvAppend.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace indexer_layer_front
{

//! Fork of the generation rope of ropeKvAppend.h (ropeAppendScatterKernel): the same bytes for the same parameters
//! (rope_kv_append::RopeAppendParams), with the programmatic-launch trigger moved.
//!
//! Memory order: like ropeAppendScatterKernel, only the quantization scales are read before griddepcontrol.wait; every
//! producer output and every per-step metadata word is read after it. With earlyTrigger every thread executes
//! griddepcontrol.launch_dependents right after the global reads of its work item (source item, cache length, rotary
//! coefficients, KV block offset and indexer slot mapping), before its stores. A block counts as triggered at its first
//! such invocation, so dependents may become resident before any store of the kernel, including block (0, 0)'s
//! scheduler buffers (cuQSeqlens, cuKvSeqlens, the tile counter, the bmm scales) and the cache-length reads of its
//! prefix sum. Without earlyTrigger the trigger is at the end of the kernel, as there. A dependent must not read
//! anything this kernel writes, nor write anything it reads, before its own griddepcontrol.wait; the rope's dependents
//! already require this, since a trigger at the end of the code does not make the stores visible either.
void invokeIndexerLayerRopeKvAppend(
    rope_kv_append::RopeAppendParams const& params, bool enablePdl, bool earlyTrigger, cudaStream_t stream);

} // namespace indexer_layer_front

TRTLLM_MV2_NAMESPACE_END
