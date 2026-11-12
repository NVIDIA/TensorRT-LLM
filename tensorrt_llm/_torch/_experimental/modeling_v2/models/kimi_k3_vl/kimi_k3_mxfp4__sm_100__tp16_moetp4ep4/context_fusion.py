# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inferact context preparation; same norm/YaRN math, direct output.

No page/cache/acceptance handling. Input and output are disjoint BF16 [N,5,576].
The per-layer route reuses the installed stock arithmetic with strided operands.
The all-layer route changes indexing and launch geometry, preserving per-row math.
"""
import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from tensorrt_llm._torch.attention.kernels.blackwell.dspark_rmsnorm_rope import DSparkRMSNormRoPEKernel
try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

LAYERS, WIDTH, NORM, PAIRS = 5, 576, 512, 32


class AllLayers:
    def __init__(self, eps):
        self.eps = eps
        self.norm_dim = NORM

    @cute.jit
    def __call__(self, x: cute.Tensor, weight: cute.Tensor, freqs: cute.Tensor,
                 output: cute.Tensor, stream: cuda.CUstream):
        self.kernel(x, weight, freqs, output).launch(
            grid=[x.shape[0], LAYERS, 1], block=[32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor, weight: cute.Tensor, freqs: cute.Tensor,
               output: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        row, layer, _ = cute.arch.block_idx()
        sum_sq = cutlass.Float32(0.0)
        for item in cutlass.range_constexpr(16):
            dim = tidx + item * 32
            value = cutlass.Float32(x[row, layer, dim])
            sum_sq += value * value
        sum_sq = cute.arch.warp_reduction_sum(sum_sq)
        inverse_rms = cute.math.rsqrt(sum_sq / self.norm_dim + self.eps)
        for item in cutlass.range_constexpr(16):
            dim = tidx + item * 32
            value = cutlass.Float32(x[row, layer, dim]) * inverse_rms
            value *= cutlass.Float32(weight[layer, dim])
            output[row, layer, dim] = value.to(output.element_type)
        for item in cutlass.range_constexpr(1):
            pair = tidx + item * 32
            real_dim = NORM + pair * 2
            imag_dim = real_dim + 1
            real = cutlass.Float32(x[row, layer, real_dim])
            imag = cutlass.Float32(x[row, layer, imag_dim])
            cos = cutlass.Float32(freqs[row, pair, 0])
            sin = cutlass.Float32(freqs[row, pair, 1])
            output[row, layer, real_dim] = (real * cos - imag * sin).to(output.element_type)
            output[row, layer, imag_dim] = (imag * cos + real * sin).to(output.element_type)


def supported(x, weights, freqs, out, eps):
    tensors = (x, weights, freqs, out)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    if not isinstance(eps, (float, int)) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
        return False
    if x.ndim != 3 or not 1 <= x.shape[0] <= 512 or tuple(x.shape[1:]) != (LAYERS, WIDTH):
        return False
    if tuple(weights.shape) != (LAYERS, NORM) or tuple(freqs.shape) != (x.shape[0], PAIRS, 2) or out.shape != x.shape:
        return False
    if x.dtype != torch.bfloat16 or weights.dtype != x.dtype or out.dtype != x.dtype or freqs.dtype != torch.float32:
        return False
    if not all(t.is_cuda and t.device == x.device and t.layout == torch.strided
               and t.is_contiguous() and not t.is_conj() and not t.is_neg()
               and not t.requires_grad and t.data_ptr() % 16 == 0 for t in tensors):
        return False
    start, end = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
    for t in (x, weights, freqs):
        if t.data_ptr() < end and start < t.data_ptr() + t.numel() * t.element_size():
            return False
    return True


@functools.cache
def compiled(eps, device_index, all_layers):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Warm the context kernel before CUDA Graph capture')
    if torch.cuda.get_device_capability(device_index) != (10, 0):
        raise ValueError('K3 context fusion requires SM100')
    n = cute.sym_int()
    freqs = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (n, PAIRS, 2), stride_order=(2, 1, 0))
    if all_layers:
        x = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (n, LAYERS, WIDTH), stride_order=(2, 1, 0))
        out = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (n, LAYERS, WIDTH), stride_order=(2, 1, 0))
        weight = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (LAYERS, NORM), stride_order=(1, 0))
        kernel = AllLayers(eps)
    else:
        x = cute.runtime.make_fake_tensor(cutlass.BFloat16, (n, WIDTH), stride=(LAYERS * WIDTH, 1))
        out = cute.runtime.make_fake_tensor(cutlass.BFloat16, (n, WIDTH), stride=(LAYERS * WIDTH, 1))
        weight = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (NORM,), stride_order=(0,))
        kernel = DSparkRMSNormRoPEKernel(WIDTH, 64, 1, eps, True, True, False, norm_dim=NORM)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, x, weight, freqs, out, stream, options='--opt-level 2 --enable-tvm-ffi')


@torch.library.custom_op('trtllm::k3_inferact_context_into', mutates_args=('out',), device_types='cuda')
def context_into(x: torch.Tensor, weights: torch.Tensor, freqs: torch.Tensor,
                 out: torch.Tensor, eps: float, all_layers: bool) -> None:
    if not supported(x, weights, freqs, out, eps):
        raise ValueError('Unsupported or aliased context-fusion operands')
    with torch.cuda.device(x.device):
        fn = compiled(float(eps), x.device.index, all_layers)
        if all_layers:
            fn(x, weights, freqs, out)
        else:
            for layer in range(LAYERS):
                fn(x[:, layer], weights[layer], freqs, out[:, layer])


@torch.library.register_fake('trtllm::k3_inferact_context_into')
def fake_context_into(x, weights, freqs, out, eps, all_layers):
    return None
