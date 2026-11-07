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
entry. The op allocates them per call, uninitialized, and this kernel clears
them with a single launch on the GEMM stream.

Optional PDL (``use_pdl``) lets this launch overlap the tail of the preceding
kernel and lets the following fused GEMM's prologue overlap this kernel. The
GEMM's ``griddepcontrol_wait`` precedes its first workspace access, so the
ordering stays correct.

With ``zero_output`` the same launch also clears the BF16 output the fused
finalize scatter-adds into, replacing the backend's separate memset launch
(and its aux-stream event pair) for the dense case. Vector stores require
compact BF16 storage, 16-byte alignment and an element count divisible by
eight, which the runner asserts.

For local BF16/FP16 activations, this launch also performs native-compatible
MXFP8 quantization (E4M3 values and linear UE8M0 scales per 32 elements).
This removes the separate input-quantization launch without changing GEMM
register pressure, gather scheduling, or FC1 scale reuse. Quantized inputs
and scales remain global scratch buffers, consumed after the GEMM's PDL wait.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .mxfp8_input_quant import MXFP8_SCALE_GROUP_SIZE, quantize_mxfp8_chunk

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

#: Values each lane quantizes; four lanes share one MXFP8 scale group.
_QUANT_VALUES_PER_LANE = 8
#: Threads per block when the launch also quantizes input, and when it only resets.
_QUANT_BLOCK_THREADS = 128
_RESET_BLOCK_THREADS = 256


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
    input_raw: Optional[cute.Tensor],
    input_quant: Optional[cute.Tensor],
    input_scale: Optional[cute.Tensor],
    block_threads: cutlass.Constexpr,
) -> None:
    values_per_lane = _QUANT_VALUES_PER_LANE
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    idx = bidx * block_threads + tidx
    if cutlass.const_expr(use_pdl):
        # The following GEMM may initialize private state before these stores
        # finish. Its griddepcontrol_wait protects every workspace access.
        cute.arch.griddepcontrol_launch_dependents()
        # A replay may follow another GEMM using this workspace. Its atomics
        # must finish before this invocation clears the same memory.
        cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(input_raw is not None):
        # Every lane of every warp runs the quantizer, so partial warps still
        # call each scale-group shuffle with all lanes; lanes past the input
        # clamp their load address and store nothing.
        chunk = idx * values_per_lane
        valid = chunk < cute.size(input_raw)
        chunk = cutlass.min(chunk, cutlass.max(cute.size(input_raw) - values_per_lane, 0))
        layout = cute.make_layout((values_per_lane,))
        src = cute.make_tensor(input_raw.iterator + chunk, layout)
        dst = cute.make_tensor(input_quant.iterator + chunk, layout)
        sf = quantize_mxfp8_chunk(src, dst, valid, values_per_lane)
        lanes_per_scale = MXFP8_SCALE_GROUP_SIZE // values_per_lane
        if valid:
            if idx % lanes_per_scale == 0:
                input_scale[idx // lanes_per_scale] = sf
    if idx < cute.size(ready):
        ready[idx] = cutlass.Int32(0)
    if idx == 0:
        fc1_counter[0] = cutlass.Int32(0)
        fc2_counter[0] = cutlass.Int32(0)
    if cutlass.const_expr(zero_output):
        if idx * (128 // cutlass.BFloat16.width) < cute.size(output):
            # Explicit vector width avoids the eight scalar stores an
            # automatic copy emits for this layout.
            _store_zero_128(output.iterator + idx * (128 // cutlass.BFloat16.width))


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
    input_raw_ptr: Optional[cute.Pointer] = None,
    input_quant_ptr: Optional[cute.Pointer] = None,
    input_scale_ptr: Optional[cute.Pointer] = None,
    input_numel: cutlass.Int32 = 0,
) -> None:
    """Clear the sync workspace and optional BF16 output, and quantize raw input.

    Clear ``num_ready`` Int32 ready flags, two one-element Int32 counters
    and, with ``zero_output``, ``output_numel`` BF16 output elements. With
    ``input_raw_ptr``, the same launch also quantizes the input to MXFP8.

    Raw pointers plus dynamic lengths let one compile serve every padded
    route count and token count; the fused GEMM wrapper follows the same
    convention. ``output_ptr`` / ``output_numel`` are ignored when
    ``zero_output`` is False. Optional ``input_raw_ptr`` accepts compact BF16
    or FP16 input; the other two input pointers receive E4M3 values and E8M0
    scales in linear order. ``input_numel`` must be divisible by 32, all three
    pointers must be 16-byte aligned, and output buffers must not alias input.
    A zero element count performs only the reset.

    Each quantizing lane handles ``_QUANT_VALUES_PER_LANE`` values, so four
    lanes share one scale; a quantizing launch uses ``_QUANT_BLOCK_THREADS``
    threads per block and a reset-only launch ``_RESET_BLOCK_THREADS``.
    """
    block_threads = (
        _QUANT_BLOCK_THREADS
        if cutlass.const_expr(input_raw_ptr is not None)
        else _RESET_BLOCK_THREADS
    )
    ready = cute.make_tensor(ready_ptr, layout=cute.make_layout((num_ready,)))
    fc1_counter = cute.make_tensor(fc1_counter_ptr, layout=cute.make_layout((1,)))
    fc2_counter = cute.make_tensor(fc2_counter_ptr, layout=cute.make_layout((1,)))
    output = cute.make_tensor(output_ptr, layout=cute.make_layout((output_numel,)))
    grid = cutlass.max(1, cute.ceil_div(num_ready, block_threads))
    if cutlass.const_expr(zero_output):
        grid = cutlass.max(
            grid,
            cute.ceil_div(output_numel, block_threads * (128 // cutlass.BFloat16.width)),
        )
    input_raw = None
    input_quant = None
    input_scale = None
    if cutlass.const_expr(input_raw_ptr is not None):
        input_raw = cute.make_tensor(input_raw_ptr, cute.make_layout((input_numel,)))
        input_quant = cute.make_tensor(input_quant_ptr, cute.make_layout((input_numel,)))
        input_scale = cute.make_tensor(
            input_scale_ptr, cute.make_layout((input_numel // MXFP8_SCALE_GROUP_SIZE,))
        )
        grid = cutlass.max(grid, cute.ceil_div(input_numel, block_threads * _QUANT_VALUES_PER_LANE))
    _reset_fc12_sync_workspace_kernel(
        ready,
        fc1_counter,
        fc2_counter,
        output,
        use_pdl,
        zero_output,
        input_raw,
        input_quant,
        input_scale,
        block_threads,
    ).launch(
        grid=(grid, 1, 1),
        block=(block_threads, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )
