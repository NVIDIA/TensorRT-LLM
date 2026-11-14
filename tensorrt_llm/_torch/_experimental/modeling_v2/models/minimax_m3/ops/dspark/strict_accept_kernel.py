# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# Strict draft acceptance and the next step's input tokens -- CuTe DSL kernels (one CTA, one thread per request)
# =============================================================================
#
# strict_accept, for request b (draft length D, K1 = D + 1 verify rows per request):
#   accepted[b, j]   = target[b K1 + j]                                    j = 0 .. D   (int32)
#   num_accepted[b]  = 1 + the length of the prefix k = 0, 1, .. where draft[b, k] == target[b K1 + k]
# which is the strict path's `1 + cumprod(draft == target[:, :D]).sum(1)`. Tokens are compared as int64, so an
# int32 / int64 mix compares the values torch's promotion compares.
#
# next_new_tokens, for request b (D next draft tokens):
#   out[b, 0]        = accepted[batch_idx[b], num_accepted[b] - 1]
#   out[b, 1 + j]    = next_draft[b, j]                                     j = 0 .. D - 1
# which is `concat([accepted[batch_idx, num_accepted - 1].unsqueeze(1), next_draft], 1)`.
#
# kv_rewind, for generation request b (rows nc .. nc + n of the batch), after the forced-acceptance override:
#   rewind[b]           = 1 - num_accepted[nc + b]
#   kv_lens[nc + b]    += 1
# which is DFlash's `_kv_rewind_amount = 1 - num_accepted[nc:bs]` and `kv_lens_cuda[nc:bs] += 1`.
#
# Integer copies and compares only: bitwise to the torch path by construction. Thread b handles request b; the
# grid is one CTA of MAX_REQUESTS threads, so a launch serves up to MAX_REQUESTS requests.
# =============================================================================
from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

MAX_REQUESTS = 64  # capacity: fixed: one CTA, one thread per request


@cute.kernel
def strict_accept_kernel(
    target: cute.Tensor,  # int32 or int64 [n_req * (D + 1)]
    draft: cute.Tensor,  # int32 or int64 [n_req * D]
    accepted: cute.Tensor,  # int32 [n_req * (D + 1)], out
    num_accepted: cute.Tensor,  # int32 [n_req], out
    n_req: cutlass.Int32,
    draft_len: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    prims.griddepcontrol(prims.GridDepAction.WAIT)
    if tidx < n_req:
        k1 = draft_len + 1
        run = cutlass.Int32(1)
        count = cutlass.Int32(0)
        for k in cutlass.range_constexpr(draft_len):
            tok = cutlass.Int64(target[tidx * k1 + k])
            drf = cutlass.Int64(draft[tidx * draft_len + k])
            accepted[tidx * k1 + k] = cutlass.Int32(tok)
            run = cutlass.select_(tok == drf, run, cutlass.Int32(0))
            count = count + run
        accepted[tidx * k1 + draft_len] = cutlass.Int32(
            cutlass.Int64(target[tidx * k1 + draft_len])
        )
        num_accepted[tidx] = count + cutlass.Int32(1)


@cute.jit
def strict_accept(
    target: cute.Tensor,
    draft: cute.Tensor,
    accepted: cute.Tensor,
    num_accepted: cute.Tensor,
    n_req: cutlass.Int32,
    draft_len: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    strict_accept_kernel(target, draft, accepted, num_accepted, n_req, draft_len).launch(
        grid=[1, 1, 1], block=[MAX_REQUESTS, 1, 1], stream=stream, use_pdl=use_pdl
    )


@cute.kernel
def next_new_tokens_kernel(
    accepted: cute.Tensor,  # [rows, accepted_stride] (flat), any int dtype
    num_accepted: cute.Tensor,  # [n_req], any int dtype
    batch_idx: cute.Tensor,  # [n_req], any int dtype
    next_draft: cute.Tensor,  # [n_req, draft_stride] (flat), any int dtype
    out: cute.Tensor,  # [n_req * (D + 1)], out
    accepted_stride: cutlass.Int64,
    draft_stride: cutlass.Int64,
    n_req: cutlass.Int32,
    draft_len: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    prims.griddepcontrol(prims.GridDepAction.WAIT)
    if tidx < n_req:
        k1 = draft_len + 1
        row = cutlass.Int64(batch_idx[tidx])
        col = cutlass.Int64(num_accepted[tidx]) - cutlass.Int64(1)
        out[tidx * k1] = out.element_type(accepted[row * accepted_stride + col])
        for j in cutlass.range_constexpr(draft_len):
            out[tidx * k1 + 1 + j] = out.element_type(
                next_draft[cutlass.Int64(tidx) * draft_stride + j]
            )


@cute.jit
def next_new_tokens(
    accepted: cute.Tensor,
    num_accepted: cute.Tensor,
    batch_idx: cute.Tensor,
    next_draft: cute.Tensor,
    out: cute.Tensor,
    accepted_stride: cutlass.Int64,
    draft_stride: cutlass.Int64,
    n_req: cutlass.Int32,
    draft_len: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    next_new_tokens_kernel(
        accepted,
        num_accepted,
        batch_idx,
        next_draft,
        out,
        accepted_stride,
        draft_stride,
        n_req,
        draft_len,
    ).launch(grid=[1, 1, 1], block=[MAX_REQUESTS, 1, 1], stream=stream, use_pdl=use_pdl)


@cute.kernel
def kv_rewind_kernel(
    num_accepted: cute.Tensor,  # [>= nc + n], any int dtype
    kv_lens: cute.Tensor,  # [>= nc + n], any int dtype, updated in place
    rewind: cute.Tensor,  # [n], num_accepted's dtype, out
    nc: cutlass.Int32,
    n: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    prims.griddepcontrol(prims.GridDepAction.WAIT)
    if tidx < n:
        row = nc + tidx
        rewind[tidx] = rewind.element_type(cutlass.Int64(1) - cutlass.Int64(num_accepted[row]))
        kv_lens[row] = kv_lens.element_type(cutlass.Int64(kv_lens[row]) + cutlass.Int64(1))


@cute.jit
def kv_rewind(
    num_accepted: cute.Tensor,
    kv_lens: cute.Tensor,
    rewind: cute.Tensor,
    nc: cutlass.Int32,
    n: cutlass.Int32,
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    kv_rewind_kernel(num_accepted, kv_lens, rewind, nc, n).launch(
        grid=[1, 1, 1], block=[MAX_REQUESTS, 1, 1], stream=stream, use_pdl=use_pdl
    )
