# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource-bounded BF16 router GEMM candidates for GLM-5.2 decode.

The decode SIMT path defaults to one expert per CTA.  Its 256-CTA grid matches the
production router's parallelism and wins at very small T even in the concurrent
shared-expert module schedule.  The two-expert/CTA arm remains available for profiling,
and a compact tensor-core path takes over once T is large enough to amortize it.
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import primitives as prims
from cutlass.memory import SmemAllocator

_THREADS = 128
_WARP = 32
_WARPS = _THREADS // _WARP
_EXPERTS = 256
_HIDDEN = 6144
_VECTOR = 8
_K_CHUNKS = _HIDDEN // (_THREADS * _VECTOR)
_TOKENS_PER_CTA = 16
_MAX_VEC2_TOKENS = 64
_SIMT_EXPERTS_PER_CTA = 1
_PDL_PRELOAD_CHUNKS = 1
if _SIMT_EXPERTS_PER_CTA not in (1, 2):
    raise ValueError("_SIMT_EXPERTS_PER_CTA must be 1 or 2")
if not 0 <= _PDL_PRELOAD_CHUNKS <= _K_CHUNKS:
    raise ValueError(f"_PDL_PRELOAD_CHUNKS must be in [0,{_K_CHUNKS}]")

RouterFn = Callable[..., None]
_CACHE: dict[tuple[int, int, bool, int, bool], RouterFn] = {}


@cute.jit
def _warp_sum(value: cutlass.Float32) -> cutlass.Float32:
    """Butterfly FP32 reduction, returned to every lane."""
    value = value + cute.arch.shuffle_sync_bfly(value, 16)
    value = value + cute.arch.shuffle_sync_bfly(value, 8)
    value = value + cute.arch.shuffle_sync_bfly(value, 4)
    value = value + cute.arch.shuffle_sync_bfly(value, 2)
    value = value + cute.arch.shuffle_sync_bfly(value, 1)
    return value


@cute.kernel
def router_vec2_kernel(
    input_tensor: cute.Tensor,
    weight_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    zero_output: cute.Tensor,
    token_count: cutlass.Constexpr,
    tokens_per_cta: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
    experts_per_cta: cutlass.Constexpr,
    fuse_zero: cutlass.Constexpr,
) -> None:
    """Compute one or two expert columns per CTA with 128-bit loads."""
    tidx, _, _ = cute.arch.thread_idx()
    expert_cta, token_tile, _ = cute.arch.block_idx()
    lane = tidx & cutlass.Int32(_WARP - 1)
    warp = tidx >> cutlass.Int32(5)
    expert0 = expert_cta * cutlass.Int32(experts_per_cta)
    first_token = token_tile * cutlass.Int32(tokens_per_cta)

    allocator = SmemAllocator()
    partials = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_ordered_layout((tokens_per_cta, experts_per_cta, _WARPS), order=(2, 1, 0)),
        byte_alignment=16,
    )

    acc0 = cutlass.Array(cutlass.Float32, tokens_per_cta)
    acc1 = cutlass.Array(cutlass.Float32, tokens_per_cta)
    for token in cutlass.range_constexpr(tokens_per_cta):
        acc0[token] = cutlass.Float32(0.0)
        acc1[token] = cutlass.Float32(0.0)

    weight_ptr = weight_tensor.iterator.raw_ptr()
    zero_ptr = zero_output.iterator.raw_ptr()
    zero_128b = cutlass.Vector.from_elements(
        (cutlass.Int64(0), cutlass.Int64(0)), dtype=cutlass.Int64
    )
    weight0_cache = cutlass.Array(cutlass.Int16, max(_PDL_PRELOAD_CHUNKS, 1) * _VECTOR)
    weight1_cache = cutlass.Array(cutlass.Int16, max(_PDL_PRELOAD_CHUNKS, 1) * _VECTOR)

    # Router weights and the routed-output clear are independent of the prior
    # all-reduce.  Preload a bounded prefix before the PDL wait so the secondary
    # grid can do useful work without extending every weight value's register
    # live range across the synchronization point.  The dependent input load
    # remains below the wait and retains producer-consumer ordering.
    for ki in cutlass.range_constexpr(_K_CHUNKS):
        k_base = cutlass.Int32(ki * _THREADS * _VECTOR) + tidx * cutlass.Int32(_VECTOR)
        if cutlass.const_expr(fuse_zero):
            # One of the already-resident expert CTAs owns each output row.
            # Clear it in this K-chunk loop so output initialization does not
            # introduce another grid or serial phase.  Only the bounded prefix
            # selected below also preloads router weights before the PDL wait.
            if expert_cta < cutlass.Int32(token_count):
                prims.store_ext(
                    zero_128b,
                    zero_ptr + expert_cta * cutlass.Int32(_HIDDEN) + k_base,
                )
        if cutlass.const_expr(use_pdl and ki < _PDL_PRELOAD_CHUNKS):
            weight0_values = prims.load_ext(
                weight_ptr + expert0 * cutlass.Int32(_HIDDEN) + k_base,
                dtype=cutlass.Int16,
                count=_VECTOR,
                cache_modifier=prims.LoadCacheModifier.CG,
            )
            if cutlass.const_expr(experts_per_cta == 2):
                weight1_values = prims.load_ext(
                    weight_ptr + (expert0 + cutlass.Int32(1)) * cutlass.Int32(_HIDDEN) + k_base,
                    dtype=cutlass.Int16,
                    count=_VECTOR,
                    cache_modifier=prims.LoadCacheModifier.CG,
                )
            for value_idx in cutlass.range_constexpr(_VECTOR):
                cache_idx = ki * _VECTOR + value_idx
                weight0_cache[cache_idx] = weight0_values[value_idx]
                if cutlass.const_expr(experts_per_cta == 2):
                    weight1_cache[cache_idx] = weight1_values[value_idx]

    if cutlass.const_expr(use_pdl):
        prims.griddepcontrol(prims.GridDepAction.WAIT)

    input_ptr = input_tensor.iterator.raw_ptr()
    for ki in cutlass.range_constexpr(_K_CHUNKS):
        k_base = cutlass.Int32(ki * _THREADS * _VECTOR) + tidx * cutlass.Int32(_VECTOR)
        if cutlass.const_expr(not use_pdl or ki >= _PDL_PRELOAD_CHUNKS):
            weight0_values = prims.load_ext(
                weight_ptr + expert0 * cutlass.Int32(_HIDDEN) + k_base,
                dtype=cutlass.Int16,
                count=_VECTOR,
                cache_modifier=prims.LoadCacheModifier.CG,
            )
            if cutlass.const_expr(experts_per_cta == 2):
                weight1_values = prims.load_ext(
                    weight_ptr + (expert0 + cutlass.Int32(1)) * cutlass.Int32(_HIDDEN) + k_base,
                    dtype=cutlass.Int16,
                    count=_VECTOR,
                    cache_modifier=prims.LoadCacheModifier.CG,
                )
        for token in cutlass.range_constexpr(tokens_per_cta):
            global_token = first_token + cutlass.Int32(token)
            if global_token < cutlass.Int32(token_count):
                values = prims.load_ext(
                    input_ptr + global_token * cutlass.Int32(_HIDDEN) + k_base,
                    dtype=cutlass.Int16,
                    count=_VECTOR,
                )
                for v in cutlass.range_constexpr(_VECTOR):
                    x = values[v].bitcast(cutlass.BFloat16).to(cutlass.Float32)
                    if cutlass.const_expr(use_pdl and ki < _PDL_PRELOAD_CHUNKS):
                        weight0_value_i16 = weight0_cache[ki * _VECTOR + v]
                    else:
                        weight0_value_i16 = weight0_values[v]
                    weight0_value = weight0_value_i16.bitcast(cutlass.BFloat16).to(cutlass.Float32)
                    acc0[token] = x * weight0_value + acc0[token]
                    if cutlass.const_expr(experts_per_cta == 2):
                        if cutlass.const_expr(use_pdl and ki < _PDL_PRELOAD_CHUNKS):
                            weight1_value_i16 = weight1_cache[ki * _VECTOR + v]
                        else:
                            weight1_value_i16 = weight1_values[v]
                        weight1_value = weight1_value_i16.bitcast(cutlass.BFloat16).to(
                            cutlass.Float32
                        )
                        acc1[token] = x * weight1_value + acc1[token]

    for token in cutlass.range_constexpr(tokens_per_cta):
        sum0 = _warp_sum(acc0[token])
        if cutlass.const_expr(experts_per_cta == 2):
            sum1 = _warp_sum(acc1[token])
        if lane == cutlass.Int32(0):
            partials[token, 0, warp] = sum0
            if cutlass.const_expr(experts_per_cta == 2):
                partials[token, 1, warp] = sum1
    prims.barrier_cta_sync(0)

    if tidx == cutlass.Int32(0):
        output_ptr = output_tensor.iterator.raw_ptr()
        for token in cutlass.range_constexpr(tokens_per_cta):
            global_token = first_token + cutlass.Int32(token)
            if global_token < cutlass.Int32(token_count):
                total0 = cutlass.Float32(0.0)
                for source_warp in cutlass.range_constexpr(_WARPS):
                    total0 = total0 + partials[token, 0, source_warp]
                base = global_token * cutlass.Int32(_EXPERTS) + expert0
                prims.store_ext(total0, output_ptr + base)
                if cutlass.const_expr(experts_per_cta == 2):
                    total1 = cutlass.Float32(0.0)
                    for source_warp in cutlass.range_constexpr(_WARPS):
                        total1 = total1 + partials[token, 1, source_warp]
                    prims.store_ext(total1, output_ptr + base + cutlass.Int32(1))

    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(1)
        if tidx == cutlass.Int32(0):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


router_vec2_kernel.set_name_prefix("glm52_moe_router_simt", remove_cutlass_symbol=True)


@cute.jit
def launch_router_vec2(
    input_tensor: cute.Tensor,
    weight_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    zero_output: cute.Tensor,
    token_count: cutlass.Constexpr,
    tokens_per_cta: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
    experts_per_cta: cutlass.Constexpr,
    fuse_zero: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
) -> None:
    router_vec2_kernel(
        input_tensor,
        weight_tensor,
        output_tensor,
        zero_output,
        token_count,
        tokens_per_cta,
        use_pdl,
        experts_per_cta,
        fuse_zero,
    ).launch(
        grid=(
            _EXPERTS // experts_per_cta,
            cute.ceil_div(token_count, tokens_per_cta),
            1,
        ),
        block=(_THREADS, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def _device_index(tensor: torch.Tensor) -> int:
    return tensor.device.index if tensor.device.index is not None else torch.cuda.current_device()


def _as_tensor(tensor: torch.Tensor, leading_dim: int) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=leading_dim
    )


def compile_router_vec2(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    *,
    use_pdl: bool,
    zero_output: torch.Tensor | None = None,
) -> RouterFn:
    """Compile or retrieve the exact-T decode variant."""
    token_count, hidden = input_tensor.shape
    if not 1 <= token_count <= _MAX_VEC2_TOKENS:
        raise ValueError(f"vec2 router supports T in [1,{_MAX_VEC2_TOKENS}], got {token_count}")
    if weight_tensor.shape != (_EXPERTS, _HIDDEN) or hidden != _HIDDEN:
        raise ValueError(
            "GLM router requires input [T,6144] and weight [256,6144], got "
            f"{tuple(input_tensor.shape)} and {tuple(weight_tensor.shape)}"
        )
    if output_tensor.shape != (token_count, _EXPERTS):
        raise ValueError(
            f"router output must be [{token_count},{_EXPERTS}], got {tuple(output_tensor.shape)}"
        )
    if input_tensor.dtype != torch.bfloat16 or weight_tensor.dtype != torch.bfloat16:
        raise TypeError("router input and weight must be BF16")
    if output_tensor.dtype != torch.float32:
        raise TypeError("router output must be FP32")
    if (
        not input_tensor.is_contiguous()
        or not weight_tensor.is_contiguous()
        or not output_tensor.is_contiguous()
    ):
        raise ValueError("router tensors must be contiguous")
    if zero_output is not None:
        if zero_output.shape != (token_count, _HIDDEN):
            raise ValueError(
                f"zero output must be [{token_count},{_HIDDEN}], got {tuple(zero_output.shape)}"
            )
        if zero_output.dtype != torch.bfloat16 or not zero_output.is_contiguous():
            raise TypeError("zero output must be contiguous BF16")

    device = _device_index(input_tensor)
    if _device_index(weight_tensor) != device or _device_index(output_tensor) != device:
        raise ValueError("router tensors must be on the same CUDA device")
    if zero_output is not None and _device_index(zero_output) != device:
        raise ValueError("zero output must share the router CUDA device")
    fuse_zero = zero_output is not None
    zero_arg = output_tensor if zero_output is None else zero_output
    key = (
        device,
        token_count,
        use_pdl,
        _SIMT_EXPERTS_PER_CTA,
        fuse_zero,
    )
    if key not in _CACHE:
        cutlass.cuda.initialize_cuda_context(device)
        input_arg = _as_tensor(input_tensor, 1)
        weight_arg = _as_tensor(weight_tensor, 1)
        output_arg = _as_tensor(output_tensor, 1)
        _CACHE[key] = cute.compile(
            launch_router_vec2,
            input_arg,
            weight_arg,
            output_arg,
            _as_tensor(zero_arg, 1),
            token_count,
            min(token_count, _TOKENS_PER_CTA),
            use_pdl,
            _SIMT_EXPERTS_PER_CTA,
            fuse_zero,
            cutlass_torch.current_stream(),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    return _CACHE[key]


def run_router_vec2(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    *,
    use_pdl: bool = False,
    zero_output: torch.Tensor | None = None,
) -> None:
    """Run the resource-bounded decode router on the current PyTorch stream."""
    compiled = compile_router_vec2(
        input_tensor,
        weight_tensor,
        output_tensor,
        use_pdl=use_pdl,
        zero_output=zero_output,
    )
    zero_arg = output_tensor if zero_output is None else zero_output
    compiled(
        _as_tensor(input_tensor, 1),
        _as_tensor(weight_tensor, 1),
        _as_tensor(output_tensor, 1),
        _as_tensor(zero_arg, 1),
        cutlass_torch.current_stream(),
    )
