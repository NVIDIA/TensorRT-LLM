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
ordering stays correct.

With ``zero_output`` the same launch also clears the BF16 output the fused
finalize scatter-adds into, replacing the backend's separate memset launch
(and its aux-stream event pair) for the dense case. Vector stores require
compact BF16 storage, 16-byte alignment and an element count divisible by
eight, which the runner asserts.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


@dsl_user_op
def _store_zero_128(ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Store eight BF16 zeros with one aligned 128-bit global instruction."""
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(0).ir_value()],
        "st.global.v4.b32 [$0], {$1, $1, $1, $1};",
        "l,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@cute.kernel
def _reset_fc12_sync_workspace_kernel(
    ready: cute.Tensor,
    fc1_counter: cute.Tensor,
    fc2_counter: cute.Tensor,
    output: cute.Tensor,
    use_pdl: cutlass.Constexpr,
    zero_output: cutlass.Constexpr,
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
    if cutlass.const_expr(zero_output):
        if idx * 8 < cute.size(output):
            # Explicit vector width avoids the eight scalar stores an
            # automatic copy emits for this layout.
            _store_zero_128(output.iterator + idx * 8)


@cute.jit
def reset_fc12_sync_workspace(
    ready_ptr: cute.Pointer,
    num_ready: cutlass.Int32,
    fc1_counter_ptr: cute.Pointer,
    fc2_counter_ptr: cute.Pointer,
    output_ptr: cute.Pointer,
    output_numel: cutlass.Int32,
    stream: cuda.CUstream,
    use_pdl: cutlass.Constexpr = False,
    zero_output: cutlass.Constexpr = False,
):
    """Clear ``num_ready`` Int32 ready flags, two one-element Int32 counters
    and, with ``zero_output``, ``output_numel`` BF16 output elements.

    Raw pointers plus dynamic lengths let one compile serve every padded
    route count and token count; the fused GEMM wrapper follows the same
    convention. ``output_ptr`` / ``output_numel`` are ignored when
    ``zero_output`` is False.
    """
    ready = cute.make_tensor(ready_ptr, layout=cute.make_layout((num_ready,)))
    fc1_counter = cute.make_tensor(fc1_counter_ptr, layout=cute.make_layout((1,)))
    fc2_counter = cute.make_tensor(fc2_counter_ptr, layout=cute.make_layout((1,)))
    output = cute.make_tensor(output_ptr, layout=cute.make_layout((output_numel,)))
    grid = cutlass.max(1, cute.ceil_div(num_ready, 256))
    if cutlass.const_expr(zero_output):
        grid = cutlass.max(grid, cute.ceil_div(output_numel, 256 * 8))
    _reset_fc12_sync_workspace_kernel(
        ready, fc1_counter, fc2_counter, output, use_pdl, zero_output
    ).launch(
        grid=(grid, 1, 1),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )
