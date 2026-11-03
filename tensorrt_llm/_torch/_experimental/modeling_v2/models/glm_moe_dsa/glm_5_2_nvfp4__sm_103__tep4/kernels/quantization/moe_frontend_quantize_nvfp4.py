# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 BF16-to-NVFP4 quantization, with an optional output-zero plane.

This is the linear-scale layout used immediately before the NVFP4 routed-expert FC1.
The launch can group one or two token rows in a CTA.  Grouping preserves the work per
row while reducing the number of SMs that this auxiliary-stream kernel can occupy;
the two-stream MoE timeline, not isolated kernel latency, selects the production arm.
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import primitives as prims

_DEFAULT_THREADS_PER_ROW = 512
_DEFAULT_ROWS_PER_CTA = 1
_HIDDEN = 6144
_VALUES_PER_THREAD = 8
_SCALE_BLOCK = 16
_SCALES_PER_ROW = _HIDDEN // _SCALE_BLOCK
_FP4_BYTES_PER_ROW = _HIDDEN // 2
_BF16_PER_ZERO_STORE = 8

QuantizeFn = Callable[..., None]
_CACHE: dict[tuple[int, int, bool, bool, bool, int, int], QuantizeFn] = {}


@cute.jit
def _decode_positive_e4m3(code: cutlass.Int32) -> cutlass.Float32:
    """Decode a finite non-negative E4M3 byte to FP32."""
    exponent = (code >> cutlass.Int32(3)) & cutlass.Int32(0xF)
    mantissa = code & cutlass.Int32(0x7)
    value = cutlass.Float32(mantissa) * cutlass.Float32(1.0 / 512.0)
    if exponent != cutlass.Int32(0):
        power = cutlass.Int32(exponent + cutlass.Int32(120)) << cutlass.Int32(23)
        value = power.bitcast(cutlass.Float32) * (
            cutlass.Float32(1.0) + cutlass.Float32(mantissa) * cutlass.Float32(0.125)
        )
    return value


@cute.kernel
def quantize_linear_kernel(
    input_tensor: cute.Tensor,
    global_scale: cute.Tensor,
    fp4_output: cute.Tensor,
    scale_output: cute.Tensor,
    zero_output: cute.Tensor,
    token_count: cutlass.Constexpr,
    wait_for_prior_pdl: cutlass.Constexpr,
    signal_gg1_pdl: cutlass.Constexpr,
    fuse_zero: cutlass.Constexpr,
    threads_per_row: cutlass.Constexpr,
    rows_per_cta: cutlass.Constexpr,
) -> None:
    """Quantize independent BF16 rows and optionally clear their output rows."""
    tidx, _, _ = cute.arch.thread_idx()
    row_group, _, _ = cute.arch.block_idx()
    row_in_cta = tidx // cutlass.Int32(threads_per_row)
    row_thread = tidx % cutlass.Int32(threads_per_row)
    row = row_group * cutlass.Int32(rows_per_cta) + row_in_cta
    row_is_active = row < cutlass.Int32(token_count)

    if cutlass.const_expr(wait_for_prior_pdl):
        prims.griddepcontrol(prims.GridDepAction.WAIT)

    input_ptr = input_tensor.iterator.raw_ptr()
    scale_ptr = global_scale.iterator.raw_ptr()
    fp4_ptr = fp4_output.iterator.raw_ptr()
    sf_ptr = scale_output.iterator.raw_ptr()

    scale_value = prims.load_ext(scale_ptr, dtype=cutlass.Float32)
    scale_blocks_per_round: cutlass.Constexpr = threads_per_row // 2
    scale_rounds: cutlass.Constexpr = cute.ceil_div(_SCALES_PER_ROW, scale_blocks_per_round)
    for scale_round in cutlass.range_constexpr(scale_rounds):
        scale_block = cutlass.Int32(scale_round * scale_blocks_per_round) + (
            row_thread >> cutlass.Int32(1)
        )
        if row_is_active and scale_block < cutlass.Int32(_SCALES_PER_ROW):
            half_block = row_thread & cutlass.Int32(1)
            values_bits = prims.load_ext(
                input_ptr
                + row * cutlass.Int32(_HIDDEN)
                + scale_block * cutlass.Int32(_SCALE_BLOCK)
                + half_block * cutlass.Int32(_VALUES_PER_THREAD),
                dtype=cutlass.Int16,
                count=_VALUES_PER_THREAD,
            )
            values = cutlass.Array(cutlass.Float32, _VALUES_PER_THREAD)
            local_max = cutlass.Float32(0.0)
            for item in cutlass.range_constexpr(_VALUES_PER_THREAD):
                value = values_bits[item].bitcast(cutlass.BFloat16).to(cutlass.Float32)
                values[item] = value
                magnitude = cute.math.abs(value)
                if magnitude > local_max:
                    local_max = magnitude

            block_max = local_max
            peer_max = cute.arch.shuffle_sync_bfly(block_max, 1)
            if peer_max > block_max:
                block_max = peer_max

            sf_unquantized = scale_value * block_max * cutlass.Float32(1.0 / 6.0)
            sf_packed = prims.cvt_f32x2_to_f8x2(
                sf_unquantized,
                sf_unquantized,
                cutlass.Float8E4M3FN,
                rnd=prims.FPRoundingMode.RN,
                sat=prims.SaturationMode.SATFINITE,
            )
            sf_code = cutlass.Int32(sf_packed) & cutlass.Int32(0xFF)
            if half_block == cutlass.Int32(0):
                sf_index = row * cutlass.Int32(_SCALES_PER_ROW) + scale_block
                prims.store_ext(cutlass.Int8(sf_code), sf_ptr + sf_index)

            narrowed_sf = _decode_positive_e4m3(sf_code)
            output_scale = cutlass.Float32(0.0)
            if block_max != cutlass.Float32(0.0):
                output_scale = cute.arch.rcp_approx(narrowed_sf * cute.arch.rcp_approx(scale_value))

            packed_word = cutlass.Int32(0)
            for pair in cutlass.range_constexpr(_VALUES_PER_THREAD // 2):
                low = values[2 * pair] * output_scale
                high = values[2 * pair + 1] * output_scale
                packed_byte = prims.cvt_f32x2_to_f4x2(
                    high,
                    low,
                    cutlass.Float4E2M1FN,
                    rnd=prims.FPRoundingMode.RN,
                ) & cutlass.Int32(0xFF)
                packed_word = packed_word | (packed_byte << cutlass.Int32(pair * 8))
            fp4_index = (
                row * cutlass.Int32(_FP4_BYTES_PER_ROW)
                + scale_block * cutlass.Int32(_SCALE_BLOCK // 2)
                + half_block * cutlass.Int32(_VALUES_PER_THREAD // 2)
            )
            prims.store_ext(packed_word, fp4_ptr + fp4_index)

    if cutlass.const_expr(fuse_zero):
        zero_ptr = zero_output.iterator.raw_ptr()
        zero_128b = cutlass.Vector.from_elements(
            (cutlass.Int64(0), cutlass.Int64(0)), dtype=cutlass.Int64
        )
        if row_is_active:
            vector = row_thread
            while vector < cutlass.Int32(_HIDDEN // _BF16_PER_ZERO_STORE):
                zero_index = row * cutlass.Int32(_HIDDEN) + vector * cutlass.Int32(
                    _BF16_PER_ZERO_STORE
                )
                prims.store_ext(zero_128b, zero_ptr + zero_index)
                vector += cutlass.Int32(threads_per_row)

    if cutlass.const_expr(signal_gg1_pdl):
        prims.barrier_cta_sync(0)
        if tidx == cutlass.Int32(0):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


quantize_linear_kernel.set_name_prefix("glm52_moe_quantize_nvfp4", remove_cutlass_symbol=True)


@cute.jit
def launch_quantize_linear(
    input_tensor: cute.Tensor,
    global_scale: cute.Tensor,
    fp4_output: cute.Tensor,
    scale_output: cute.Tensor,
    zero_output: cute.Tensor,
    token_count: cutlass.Constexpr,
    wait_for_prior_pdl: cutlass.Constexpr,
    signal_gg1_pdl: cutlass.Constexpr,
    fuse_zero: cutlass.Constexpr,
    threads_per_row: cutlass.Constexpr,
    rows_per_cta: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
) -> None:
    grid_rows = token_count
    if cutlass.const_expr(rows_per_cta > 1):
        grid_rows = (token_count + rows_per_cta - 1) // rows_per_cta
    quantize_linear_kernel(
        input_tensor,
        global_scale,
        fp4_output,
        scale_output,
        zero_output,
        token_count,
        wait_for_prior_pdl,
        signal_gg1_pdl,
        fuse_zero,
        threads_per_row,
        rows_per_cta,
    ).launch(
        grid=(grid_rows, 1, 1),
        block=(threads_per_row * rows_per_cta, 1, 1),
        use_pdl=wait_for_prior_pdl,
        stream=stream,
    )


def _schedule(
    token_count: int,
    threads_per_row: int | None,
    rows_per_cta: int | None,
) -> tuple[int, int]:
    """Validate and canonicalize one resource-footprint schedule."""
    threads = _DEFAULT_THREADS_PER_ROW if threads_per_row is None else threads_per_row
    rows = _DEFAULT_ROWS_PER_CTA if rows_per_cta is None else rows_per_cta
    if threads not in (256, 512):
        raise ValueError(f"threads_per_row must be 256 or 512, got {threads}")
    if rows not in (1, 2, 4):
        raise ValueError(f"rows_per_cta must be 1, 2, or 4, got {rows}")
    if threads * rows > 1024:
        raise ValueError(
            f"threads_per_row * rows_per_cta must not exceed 1024, got {threads * rows}"
        )
    # Avoid launching inactive warps for the single-token decode case while
    # retaining the requested grouping for every larger exact-T variant.
    rows = min(rows, token_count)
    return threads, rows


def _as_tensor(tensor: torch.Tensor, leading_dim: int = 0) -> cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True)
    # Production stores the per-layer global scale as a true 0-D tensor.  A
    # scalar has no stride that can be marked dynamic; attempting to mark dim 0
    # is rejected by the DLPack wrapper before kernel compilation.
    if tensor.ndim == 0:
        return cute_tensor
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def _validate(
    input_tensor: torch.Tensor,
    global_scale: torch.Tensor,
    fp4_output: torch.Tensor,
    scale_output: torch.Tensor,
    zero_output: torch.Tensor,
    fuse_zero: bool,
) -> tuple[int, int]:
    if input_tensor.ndim != 2 or input_tensor.shape[1] != _HIDDEN:
        raise ValueError(f"input must have shape [T,{_HIDDEN}], got {tuple(input_tensor.shape)}")
    token_count = int(input_tensor.shape[0])
    if not 1 <= token_count <= 64:
        raise ValueError(f"input T must be in [1,64], got {token_count}")
    if input_tensor.dtype != torch.bfloat16 or not input_tensor.is_contiguous():
        raise TypeError("input must be contiguous BF16")
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise TypeError("global_scale must contain one FP32 value")
    if fp4_output.dtype != torch.uint8 or fp4_output.shape != (
        token_count,
        _FP4_BYTES_PER_ROW,
    ):
        raise ValueError(f"fp4_output must be uint8 [{token_count},{_FP4_BYTES_PER_ROW}]")
    if scale_output.dtype != torch.uint8 or scale_output.shape != (
        token_count,
        _SCALES_PER_ROW,
    ):
        raise ValueError(f"scale_output must be uint8 [{token_count},{_SCALES_PER_ROW}]")
    if fuse_zero and (
        zero_output.dtype != torch.bfloat16
        or zero_output.shape != (token_count, _HIDDEN)
        or not zero_output.is_contiguous()
    ):
        raise ValueError(f"zero_output must be contiguous BF16 [{token_count},{_HIDDEN}]")
    device = input_tensor.device.index
    if device is None:
        device = torch.cuda.current_device()
    tensors = (global_scale, fp4_output, scale_output, zero_output)
    if any(tensor.device.index != device for tensor in tensors):
        raise ValueError("quantize tensors must be on the same CUDA device")
    return device, token_count


def compile_quantize_linear(
    input_tensor: torch.Tensor,
    global_scale: torch.Tensor,
    fp4_output: torch.Tensor,
    scale_output: torch.Tensor,
    zero_output: torch.Tensor,
    *,
    wait_for_prior_pdl: bool = False,
    signal_gg1_pdl: bool = False,
    fuse_zero: bool = False,
    threads_per_row: int | None = None,
    rows_per_cta: int | None = None,
) -> QuantizeFn:
    """Compile or retrieve one exact-T quantization variant."""
    device, token_count = _validate(
        input_tensor,
        global_scale,
        fp4_output,
        scale_output,
        zero_output,
        fuse_zero,
    )
    threads, rows = _schedule(token_count, threads_per_row, rows_per_cta)
    key = (
        device,
        token_count,
        wait_for_prior_pdl,
        signal_gg1_pdl,
        fuse_zero,
        threads,
        rows,
    )
    if key not in _CACHE:
        cutlass.cuda.initialize_cuda_context(device)
        args = (
            _as_tensor(input_tensor, 1),
            _as_tensor(global_scale),
            _as_tensor(fp4_output, 1),
            _as_tensor(scale_output, 1),
            _as_tensor(zero_output, 1),
        )
        _CACHE[key] = cute.compile(
            launch_quantize_linear,
            *args,
            token_count,
            wait_for_prior_pdl,
            signal_gg1_pdl,
            fuse_zero,
            threads,
            rows,
            cutlass_torch.current_stream(),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    return _CACHE[key]


def run_quantize_linear(
    input_tensor: torch.Tensor,
    global_scale: torch.Tensor,
    fp4_output: torch.Tensor,
    scale_output: torch.Tensor,
    zero_output: torch.Tensor,
    *,
    wait_for_prior_pdl: bool = False,
    signal_gg1_pdl: bool = False,
    fuse_zero: bool = False,
    threads_per_row: int | None = None,
    rows_per_cta: int | None = None,
) -> None:
    """Quantize on the current PyTorch stream without allocating."""
    compiled = compile_quantize_linear(
        input_tensor,
        global_scale,
        fp4_output,
        scale_output,
        zero_output,
        wait_for_prior_pdl=wait_for_prior_pdl,
        signal_gg1_pdl=signal_gg1_pdl,
        fuse_zero=fuse_zero,
        threads_per_row=threads_per_row,
        rows_per_cta=rows_per_cta,
    )
    compiled(
        _as_tensor(input_tensor, 1),
        _as_tensor(global_scale),
        _as_tensor(fp4_output, 1),
        _as_tensor(scale_output, 1),
        _as_tensor(zero_output, 1),
        cutlass_torch.current_stream(),
    )
