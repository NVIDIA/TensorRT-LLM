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

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! TP2CTX_AR1 / TP2CTX_SP (MiniMax-M3.1 tp2ctx prefill boundary, 2 ranks): out[i] = bf16(float(a[i]) + float(b[i])) for
//! n bf16 values, where `b` may be the peer rank's buffer (a symmetric-memory pointer; read over NVLink). One rounding
//! of a commutative 2-operand sum, so both ranks get the bits NCCL's 2-rank all-reduce produces. 16-byte vectors; n % 8
//! == 0 and 16-byte aligned pointers (checked by the op).
void launchTp2ctxPeerAdd(void const* a, void const* b, void* out, int64_t n, int smCount, cudaStream_t stream);

//! dst[i] = src[i] for nBytes bytes, 16-byte vectors, where `dst` may be the peer rank's buffer (NVLink stores). The
//! TP2CTX_SP all-gather half / the two-shot push. nBytes % 16 == 0.
void launchTp2ctxPeerPush(void const* src, void* dst, int64_t nBytes, int smCount, cudaStream_t stream);

//! TP2CTX_SP / AR1 cross-rank barrier for two ranks on peer buffers mapped by MPI IPC (no torch.distributed): one CTA
//! bumps the local epoch (state[0]), publishes it to the peer's flag slot `peerFlags[rank]` (release, system scope) and
//! spins on `localFlags[1 - rank]` (acquire) until the peer reaches the same epoch. Stream order makes every earlier
//! kernel's writes, local and pushed, visible to the peer once it passes. The epoch lives on the device, so the barrier
//! is graph-safe.
void launchTp2ctxIpcBarrier(int* localFlags, int* peerFlags, int* state, int rank, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
