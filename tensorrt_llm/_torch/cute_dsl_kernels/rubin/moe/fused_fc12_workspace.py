# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Reset the MXFP8 fused FC12 synchronization workspace in one launch.

The fused FC12 kernel gates FC2 A-loads on per-M-tile readiness flags and
draws work IDs from two L2-atomic counters. All of them must be zero on
entry. Zeroing them with three ``torch.zeros`` calls costs three allocations
and three fill launches per forward; this kernel clears the cached buffers
with a single launch on the GEMM stream.

Optional PDL (``use_pdl``) lets this launch overlap the tail of the preceding
kernel and lets the following fused GEMM's prologue overlap this kernel. The
GEMM's ``griddepcontrol_wait`` precedes its first workspace access, so the
ordering stays correct. The output buffer is not touched here: the backend
zeroes it separately (optionally on an auxiliary stream).
"""

import cutlass
import cutlass.cute as cute

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


@cute.kernel
def _reset_fc12_sync_workspace_kernel(
    ready: cute.Tensor,
    fc1_counter: cute.Tensor,
    fc2_counter: cute.Tensor,
    use_pdl: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    idx = bidx * 256 + tidx
    if cutlass.const_expr(use_pdl):
        # The following GEMM may initialize private state before these stores
        # finish. Its griddepcontrol_wait protects every workspace access.
        cute.arch.griddepcontrol_launch_dependents()
        # A replay may follow another GEMM using this workspace. Its atomics
        # must finish before this invocation clears the same memory.
        cute.arch.griddepcontrol_wait()
    if idx < cute.size(ready):
        ready[idx] = cutlass.Int32(0)
    if idx == 0:
        fc1_counter[0] = cutlass.Int32(0)
        fc2_counter[0] = cutlass.Int32(0)


@cute.jit
def reset_fc12_sync_workspace(
    ready_ptr: cute.Pointer,
    num_ready: cutlass.Int32,
    fc1_counter_ptr: cute.Pointer,
    fc2_counter_ptr: cute.Pointer,
    stream: cuda.CUstream,
    use_pdl: cutlass.Constexpr = False,
):
    """Clear ``num_ready`` Int32 ready flags and two one-element Int32 counters.

    Raw pointers plus a dynamic length let one compile serve every padded
    route count; the fused GEMM wrapper follows the same convention.
    """
    ready = cute.make_tensor(ready_ptr, layout=cute.make_layout((num_ready,)))
    fc1_counter = cute.make_tensor(fc1_counter_ptr, layout=cute.make_layout((1,)))
    fc2_counter = cute.make_tensor(fc2_counter_ptr, layout=cute.make_layout((1,)))
    grid = cutlass.max(1, cute.ceil_div(num_ready, 256))
    _reset_fc12_sync_workspace_kernel(ready, fc1_counter, fc2_counter, use_pdl).launch(
        grid=(grid, 1, 1),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )
