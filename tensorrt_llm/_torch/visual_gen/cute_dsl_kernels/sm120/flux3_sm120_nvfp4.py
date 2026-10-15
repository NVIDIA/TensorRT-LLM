# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Torch launcher for the experimental SM120 Flux.3 NVFP4 fused kernel."""

from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
import torch.nn.functional as F
from cutlass.cute.runtime import make_ptr

from .flux3_sm120_nvfp4_kernel import LINEAR, PACKED_SWIGLU, QKV, SWIGLU, Flux3Sm120Nvfp4Kernel

_COMPILED = {}


def _scale_size(rows, columns):
    return ((rows + 127) // 128) * 128 * ((columns // 16 + 3) // 4) * 4


# Unbounded: one entry per distinct per-layer scale. Eviction would free tensors
# that captured CUDA graphs still reference.
@lru_cache(maxsize=None)
def _scalar(device, value, count=1):
    return torch.full((count,), value, device=device, dtype=torch.float32)


@lru_cache(maxsize=64)
def _dummy_tensors(device, rows):
    return (
        torch.ones(128, device=device, dtype=torch.bfloat16),
        torch.zeros((rows, 128), device=device, dtype=torch.float32),
        torch.zeros(rows, device=device, dtype=torch.int64),
        torch.zeros(1, device=device, dtype=torch.uint8),
    )


@cute.jit
def _launch(
    gemm: cutlass.Constexpr,
    activation_ptr: cute.Pointer,
    weight_ptr: cute.Pointer,
    activation_scale_ptr: cute.Pointer,
    weight_scale_ptr: cute.Pointer,
    logical_output_ptr: cute.Pointer,
    output_ptr: cute.Pointer,
    output_scale_ptr: cute.Pointer,
    inverse_output_scale: cute.Pointer,
    global_scale: cute.Pointer,
    row_scale_ptr: cute.Pointer,
    column_scale_ptr: cute.Pointer,
    q_norm: cute.Pointer,
    k_norm: cute.Pointer,
    cos_sin: cute.Pointer,
    positions: cute.Pointer,
    qkv_scale: cute.Pointer,
    rows: cutlass.Constexpr,
    columns: cutlass.Constexpr,
    reduction: cutlass.Constexpr,
    output_columns: cutlass.Constexpr,
    max_clusters: cutlass.Constexpr,
    stream,
):
    activation = cute.make_tensor(
        activation_ptr,
        cute.make_layout((rows, reduction, 1), stride=(reduction, 1, rows * reduction)),
    )
    weight = cute.make_tensor(
        weight_ptr,
        cute.make_layout((columns, reduction, 1), stride=(reduction, 1, columns * reduction)),
    )
    activation_scale = cute.make_tensor(
        activation_scale_ptr, blockscaled_utils.tile_atom_to_shape_SF((rows, reduction, 1), 16)
    )
    weight_scale = cute.make_tensor(
        weight_scale_ptr, blockscaled_utils.tile_atom_to_shape_SF((columns, reduction, 1), 16)
    )
    logical_output = cute.make_tensor(
        logical_output_ptr,
        cute.make_layout((rows, columns, 1), stride=(columns, 1, rows * columns)),
    )
    output = cute.make_tensor(
        output_ptr,
        cute.make_layout(
            (rows, output_columns, 1), stride=(output_columns, 1, rows * output_columns)
        ),
    )
    output_scale = cute.make_tensor(output_scale_ptr, cute.make_layout((rows * columns // 16,)))
    gemm(
        activation,
        weight,
        activation_scale,
        weight_scale,
        logical_output,
        output,
        output_scale,
        inverse_output_scale,
        global_scale,
        row_scale_ptr,
        column_scale_ptr,
        q_norm,
        k_norm,
        cos_sin,
        positions,
        qkv_scale,
        max_clusters,
        stream,
    )


def run(
    mode,
    activation,
    weight,
    activation_scale,
    weight_scale,
    input_global_scale,
    weight_global_scale,
    *,
    output_inverse_scale=None,
    q_norm=None,
    k_norm=None,
    cos_sin=None,
    positions=None,
    qkv_scale=None,
):
    capability = torch.cuda.get_device_capability(activation.device) if activation.is_cuda else None
    if capability not in ((12, 0), (12, 1)):
        raise RuntimeError("Flux.3 NVFP4 kernel requires an SM120 or SM121 CUDA device")
    rows, packed_k = activation.shape
    columns = weight.shape[0]
    reduction = packed_k * 2
    if columns % 128 or reduction % 128:
        raise ValueError(f"unaligned trial shape: {(rows, columns, reduction)}")
    padded_rows = ((rows + 127) // 128) * 128
    if padded_rows != rows:
        activation = F.pad(activation, (0, 0, 0, padded_rows - rows))
    if activation_scale.numel() != _scale_size(rows, reduction):
        raise ValueError("activation scale layout does not match 128x4")
    if weight_scale.numel() != _scale_size(columns, reduction):
        raise ValueError("weight scale layout does not match 128x4")
    device = activation.device
    global_scale = _scalar(device, input_global_scale * weight_global_scale)
    inverse_output_scale = (
        output_inverse_scale if output_inverse_scale is not None else _scalar(device, 1.0)
    )
    dummy_norm, dummy_cos_sin, dummy_positions, dummy_output_scale = _dummy_tensors(
        device, padded_rows
    )
    q_norm = q_norm if q_norm is not None else dummy_norm
    k_norm = k_norm if k_norm is not None else q_norm
    cos_sin = cos_sin if cos_sin is not None else dummy_cos_sin
    if positions is None:
        positions = dummy_positions
    elif padded_rows != rows:
        positions = F.pad(positions, (0, padded_rows - rows))
    qkv_scale = (
        qkv_scale
        if qkv_scale is not None
        else _scalar(device, input_global_scale * weight_global_scale, 3)
    )
    if mode == "linear":
        epilogue = LINEAR
        output = torch.empty((padded_rows, columns), device=device, dtype=torch.bfloat16)
        output_dtype = cutlass.BFloat16
    elif mode == "swiglu":
        epilogue = SWIGLU
        output = torch.empty((padded_rows, columns // 2), device=device, dtype=torch.bfloat16)
        output_dtype = cutlass.BFloat16
    elif mode == "qkv":
        epilogue = QKV
        output = torch.empty((padded_rows, columns), device=device, dtype=torch.bfloat16)
        output_dtype = cutlass.BFloat16
    elif mode == "packed":
        epilogue = PACKED_SWIGLU
        output = torch.empty((padded_rows, columns // 4), device=device, dtype=torch.uint8)
        output_dtype = cutlass.Int8
    else:
        raise ValueError(mode)
    output_scale = (
        torch.empty((_scale_size(padded_rows, columns // 2),), device=device, dtype=torch.uint8)
        if mode == "packed"
        else dummy_output_scale
    )
    out_columns = output.shape[1]
    pointers = [
        make_ptr(dtype, tensor.data_ptr(), assumed_align=16)
        for dtype, tensor in (
            (cutlass.Float4E2M1FN, activation),
            (cutlass.Float4E2M1FN, weight),
            (cutlass.Float8E4M3FN, activation_scale),
            (cutlass.Float8E4M3FN, weight_scale),
            (cutlass.BFloat16, output),
            (output_dtype, output),
            (cutlass.Int8, output_scale),
            (cutlass.Float32, inverse_output_scale),
            (cutlass.Float32, global_scale),
            (cutlass.Float32, global_scale),
            (cutlass.Float32, global_scale),
            (cutlass.BFloat16, q_norm),
            (cutlass.BFloat16, k_norm),
            (cutlass.Float32, cos_sin),
            (cutlass.Int64, positions),
            (cutlass.Float32, qkv_scale),
        )
    ]
    key = (device.index, capability, mode, padded_rows, columns, reduction)
    stream = cutlass_torch.current_stream()
    compiled = _COMPILED.get(key)
    if compiled is None:
        gemm = Flux3Sm120Nvfp4Kernel(
            cutlass.Float32,
            16,
            (128, 128, 128),
            (64, 32),
            epilogue,
            columns,
            architecture=f"sm_{capability[0]}{capability[1]}",
        )
        max_clusters = cutlass.utils.HardwareInfo(device.index).get_max_active_clusters(1)
        compiled = cute.compile(
            _launch,
            gemm,
            *pointers,
            padded_rows,
            columns,
            reduction,
            out_columns,
            max_clusters,
            stream,
        )
        _COMPILED[key] = compiled
    compiled(*pointers, stream)
    if mode == "packed":
        return output[:rows], output_scale
    return output[:rows]
