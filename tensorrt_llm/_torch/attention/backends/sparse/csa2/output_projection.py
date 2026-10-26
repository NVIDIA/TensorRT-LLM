# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped CSA2 O-A projection as batched MXFP8 GEMMs.

Applies when the grouped weight's 128x128 block scales are powers of two: the same
E4M3 weight then has exact UE8M0 1x32 scales, and the attention output is quantized
to MXFP8 instead of 1x128 FP8 blocks.
"""

from functools import cache

import torch
import triton
import triton.language as tl


def mxfp8_weight_scale(scale: torch.Tensor | None) -> torch.Tensor | None:
    """CUTLASS-swizzled UE8M0 ``[groups, R * D / 32]`` scales equal to power-of-two 128-block ones.

    ``scale`` is the FP32 ``[groups, R / 128, D / 128]`` block scale. Returns None when
    any block scale is not an exact power of two in the UE8M0 range.
    """
    if scale is None or scale.dtype != torch.float32 or scale.ndim != 3 or scale.numel() == 0:
        return None
    mantissa, exponent = torch.frexp(scale)
    if not bool(((mantissa == 0.5) & (exponent >= -126) & (exponent <= 128)).all()):
        return None
    # 2**k = 0.5 * 2**(k + 1), so the UE8M0 byte k + 127 is exponent + 126.
    rows = (exponent + 126).to(torch.uint8).repeat_interleave(128, 1).repeat_interleave(4, 2)
    return torch.stack(
        [torch.ops.trtllm.block_scale_interleave(group.contiguous()).reshape(-1) for group in rows]
    )


@triton.jit
def _mxfp8_blocks(values):
    """E4M3 values and UE8M0 exponents of rows of 32 FP32 values, as ``mxfp8_quantize``."""
    amax = tl.max(tl.abs(values), 1)
    scale = amax * (1.0 / 448.0)
    bits = scale.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.where(scale <= 2.0**-127, 0, tl.minimum(exponent, 254))
    reciprocal = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    reciprocal = tl.where(amax == 0, 0.0, reciprocal)
    return (values * reciprocal[:, None]).to(tl.float8e4nv), exponent


@triton.jit
def _inverse_rope_mxfp8_kernel(
    X,
    POS,
    CS,
    Q,
    S,
    M,
    stride_xm,
    D: tl.constexpr,
    SF_ROWS: tl.constexpr,
    HEADS: tl.constexpr,
):
    """One token row x HEADS heads of one group -> E4M3 ``[groups, M, D]`` + swizzled UE8M0.

    Channels 0..447 of each head quantize as loaded. The last 64 carry GPT-J RoPE:
    they load again as pairs, rotate back in FP32 and round to BF16 before their two
    1x32 blocks quantize. Padding rows up to SF_ROWS write zero scales.
    """
    row = tl.program_id(0)
    group = tl.program_id(1)
    part = tl.program_id(2)
    valid = row < M
    heads = part * HEADS + tl.arange(0, HEADS)
    channel = tl.arange(0, 512)
    column = heads[:, None] * 512 + channel[None, :]
    source = X + row * stride_xm + group * D
    target = Q + (group * M + row) * D
    # CUTLASS 128x4 scale layout per group.
    scales = (
        S
        + group * SF_ROWS * (D // 32)
        + (row // 128) * (D // 32) * 128
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
    )
    plain = (channel < 448)[None, :]
    x = tl.load(source + column, valid & plain, 0.0).to(tl.float32)
    quantized, exponent = _mxfp8_blocks(tl.reshape(x, (HEADS * 16, 32)))
    tl.store(target + column, tl.reshape(quantized, (HEADS, 512)), valid & plain)
    blocks = part * HEADS * 16 + tl.arange(0, HEADS * 16)
    tl.store(
        scales + (blocks // 4) * 512 + blocks % 4,
        tl.where(valid, exponent, 0).to(tl.uint8),
        blocks % 16 < 14,
    )
    tail = heads[:, None] * 512 + 448 + tl.arange(0, 64)[None, :]
    trig = tl.load(POS + row, valid, 0) * 64 + tl.arange(0, 32)
    cosine = tl.load(CS + trig, valid, 1.0)[None, :]
    sine = tl.load(CS + trig + 32, valid, 0.0)[None, :]
    even, odd = tl.split(
        tl.reshape(tl.load(source + tail, valid, 0.0).to(tl.float32), (HEADS, 32, 2))
    )
    rotated = tl.join(even * cosine + odd * sine, odd * cosine - even * sine)
    quantized, exponent = _mxfp8_blocks(
        tl.reshape(rotated.to(tl.bfloat16).to(tl.float32), (HEADS * 2, 32))
    )
    tl.store(target + tail, tl.reshape(quantized, (HEADS, 64)), valid)
    blocks = (heads * 16 + 14)[:, None] + tl.arange(0, 2)[None, :]
    tl.store(
        scales + (blocks // 4) * 512 + blocks % 4,
        tl.reshape(tl.where(valid, exponent, 0).to(tl.uint8), (HEADS, 2)),
    )


def inverse_rope_mxfp8(
    output: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse GPT-J RoPE and MXFP8 quantization of ``[M, groups * D]`` BF16 attention output.

    Returns E4M3 ``[groups, M, D]`` and per-group CUTLASS-swizzled UE8M0 scales with
    rows padded to 128.
    """
    rows = output.shape[0]
    depth = output.shape[1] // groups
    sf_rows = triton.cdiv(rows, 128) * 128
    quantized = torch.empty((groups, rows, depth), device=output.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(groups * sf_rows * (depth // 32), device=output.device, dtype=torch.uint8)
    heads = next(count for count in (8, 4, 2, 1) if (depth // 512) % count == 0)
    _inverse_rope_mxfp8_kernel[(sf_rows, groups, depth // 512 // heads)](
        output,
        positions,
        cos_sin,
        quantized,
        scales,
        rows,
        output.stride(0),
        depth,
        sf_rows,
        heads,
        num_warps=2,
    )
    return quantized, scales


@cache
def _kernel_type() -> type:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    class GroupedOutputGemm(Sm100BlockScaledPersistentDenseGemmKernel):
        @cute.jit
        def wrapper_grouped(
            self,
            m: cutlass.Int64,
            sf_m: cutlass.Int64,
            n: cutlass.Constexpr,
            k: cutlass.Constexpr,
            groups: cutlass.Constexpr,
            a: cute.Pointer,
            b: cute.Pointer,
            sa: cute.Pointer,
            sb: cute.Pointer,
            c: cute.Pointer,
            c_scale: cute.Pointer,
            alpha: cute.Tensor,
            max_active_clusters: cutlass.Constexpr,
            stream: cuda.CUstream,
        ):
            a_tensor = cute.make_tensor(
                a, cute.make_ordered_layout((m, k, groups), order=(1, 0, 2))
            )
            b_tensor = cute.make_tensor(
                b, cute.make_ordered_layout((n, k, groups), order=(1, 0, 2))
            )
            # Rows are tokens holding every group's ranks side by side: [M, groups * N].
            c_layout = cute.make_layout((m, n, groups), stride=(groups * n, 1, n))
            sa_tensor = cute.make_tensor(
                sa,
                cute.make_ordered_layout(
                    (32, 4, sf_m, 4, k // 128, groups), order=(2, 1, 4, 0, 3, 5)
                ),
            )
            sb_tensor = cute.make_tensor(
                sb,
                cute.make_ordered_layout(
                    (32, 4, n // 128, 4, k // 128, groups), order=(2, 1, 4, 0, 3, 5)
                ),
            )
            if cutlass.const_expr(self.mxfp8_out):
                # The BF16 tile is never stored; its descriptor only needs a valid address.
                quantized = cute.make_tensor(c, c_layout)
                self(
                    a_tensor,
                    b_tensor,
                    sa_tensor,
                    sb_tensor,
                    cute.make_tensor(cute.recast_ptr(c, dtype=cutlass.BFloat16), c_layout),
                    alpha,
                    max_active_clusters,
                    stream,
                    q_fp8_tensor=quantized,
                    out_scale_tensor=cute.make_tensor(
                        c_scale, cute.make_layout((sf_m * 128 * groups * n // 32,))
                    ),
                )
            else:
                self(
                    a_tensor,
                    b_tensor,
                    sa_tensor,
                    sb_tensor,
                    cute.make_tensor(c, c_layout),
                    alpha,
                    max_active_clusters,
                    stream,
                )

    return GroupedOutputGemm


_COMPILED = {}


def grouped_output_mxfp8(
    output: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    *,
    quantize_output: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Grouped O-A of BF16 attention output with inverse RoPE, as MXFP8 GEMMs.

    ``output`` is ``[M, groups * D]`` BF16 with GPT-J rotary tails on each 512-channel
    head, ``weight`` ``[groups, R, D]`` E4M3, ``weight_scale`` from ``mxfp8_weight_scale``
    and ``alpha`` a unit FP32 scalar. Returns BF16 ``[M, groups * R]``, or with
    ``quantize_output`` its MXFP8 E4M3 values and CUTLASS-swizzled UE8M0 scales as
    ``mxfp8_quantize`` lays them out. Returns None while a graph capture would have to
    compile the GEMM.
    """
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

    rows = output.shape[0]
    groups, rank, depth = weight.shape
    if (
        rows == 0
        or output.dtype != torch.bfloat16
        or output.stride(1) != 1
        or output.stride(0) % 8
        or output.shape[1] != groups * depth
        or positions.numel() != rows
        or positions.dtype not in (torch.int32, torch.int64)
        or not positions.is_contiguous()
        or cos_sin.dtype != torch.float32
        or not cos_sin.is_contiguous()
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
        or weight_scale.dtype != torch.uint8
        or weight_scale.numel() != groups * rank * depth // 32
        or not weight_scale.is_contiguous()
        or alpha.dtype != torch.float32
        or alpha.numel() != 1
        or depth % 512
        or rank % 128
    ):
        raise ValueError("CSA2 grouped MXFP8 O-A projection got an unsupported input")
    # Measured on GB300: one-CTA 128x128 tiles up to 128 rows, 2-CTA 256x128 tiles beyond.
    if rows <= 128:
        tile, cluster = (128, 128), (1, 1)
    else:
        tile, cluster = (256, 128), (2, 1)
    key = (output.device.index, tile, rank, depth, groups, quantize_output)
    if key not in _COMPILED and torch.cuda.is_current_stream_capturing():
        return None
    sf_rows = triton.cdiv(rows, 128) * 128
    quantized, scales = inverse_rope_mxfp8(output, positions, cos_sin, groups)
    if quantize_output:
        projected = torch.empty(
            (rows, groups * rank), device=output.device, dtype=torch.float8_e4m3fn
        )
        projected_scale = torch.empty(
            sf_rows * groups * rank // 32, device=output.device, dtype=torch.uint8
        )
    else:
        projected = torch.empty((rows, groups * rank), device=output.device, dtype=torch.bfloat16)
        projected_scale = projected
    tensors = (quantized, weight, scales, weight_scale, projected, projected_scale)
    if key not in _COMPILED:
        dtypes = (
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            cutlass.Float8E8M0FNU,
            cutlass.Float8E4M3FN if quantize_output else cutlass.BFloat16,
            cutlass.Uint8,
        )
        pointers = [
            make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
            for tensor, dtype in zip(tensors, dtypes)
        ]
        kernel = _kernel_type()(32, tile, cluster, mxfp8_out=quantize_output)
        active = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
        _COMPILED[key] = cute.compile(
            kernel.wrapper_grouped,
            rows,
            sf_rows // 128,
            rank,
            depth,
            groups,
            *pointers,
            cute.runtime.from_dlpack(alpha),
            active,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    _COMPILED[key](rows, sf_rows // 128, *(tensor.data_ptr() for tensor in tensors), alpha)
    return (projected, projected_scale) if quantize_output else projected


def dense_mxfp8_gemm(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """BF16 ``activation @ weight.T`` of MXFP8 operands on the persistent CuTe DSL GEMM.

    ``activation`` is E4M3 ``[M, K]`` and ``weight`` E4M3 ``[N, K]``, each with
    CUTLASS-swizzled UE8M0 scales (activation rows padded to 128), and ``alpha`` a unit
    FP32 scalar. The result equals ``mxfp8_mxfp8_gemm`` byte for byte; measured on GB300
    for the O-B shape (N 5120, K 8192), 2-CTA 256x256 tiles on 2x2 clusters beat the
    autotuned CUTLASS GEMM from 1024 rows on.
    """
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

    rows, depth = activation.shape
    columns = weight.shape[0]
    if (
        rows == 0
        or activation.dtype != torch.float8_e4m3fn
        or weight.dtype != torch.float8_e4m3fn
        or weight.shape[1] != depth
        or depth % 128
        or columns % 256
        or activation_scale.dtype != torch.uint8
        or activation_scale.numel() != triton.cdiv(rows, 128) * 128 * depth // 32
        or weight_scale.dtype != torch.uint8
        or weight_scale.numel() != columns * depth // 32
        or alpha.dtype != torch.float32
        or alpha.numel() != 1
        or not all(
            tensor.is_contiguous()
            for tensor in (activation, activation_scale, weight, weight_scale, alpha)
        )
    ):
        raise ValueError("CSA2 dense MXFP8 GEMM got an unsupported input")
    tile, cluster = (256, 256), (2, 2)
    key = ("dense", activation.device.index, tile, cluster, columns, depth)
    sf_rows = triton.cdiv(rows, 128) * 128
    output = torch.empty((rows, columns), device=activation.device, dtype=torch.bfloat16)
    tensors = (activation, weight, activation_scale, weight_scale, output, output)
    if key not in _COMPILED:
        dtypes = (
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            cutlass.Float8E8M0FNU,
            cutlass.BFloat16,
            cutlass.Uint8,
        )
        pointers = [
            make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
            for tensor, dtype in zip(tensors, dtypes)
        ]
        # One group of the grouped O-A kernel is a plain [M, N] GEMM.
        kernel = _kernel_type()(32, tile, cluster)
        active = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
        _COMPILED[key] = cute.compile(
            kernel.wrapper_grouped,
            rows,
            sf_rows // 128,
            columns,
            depth,
            1,
            *pointers,
            cute.runtime.from_dlpack(alpha),
            active,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    _COMPILED[key](rows, sf_rows // 128, *(tensor.data_ptr() for tensor in tensors), alpha)
    return output


@cache
def _block128_kernel_type() -> type:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockwise_gemm.blockwise_gemm import (
        Sm100BlockwiseGemmKernel,
    )

    class GroupedBlock128OutputGemm(Sm100BlockwiseGemmKernel):
        @cute.jit
        def wrapper_mxfp8(
            self,
            m: cutlass.Int32,
            sf_m: cutlass.Int32,
            n: cutlass.Constexpr,
            k: cutlass.Constexpr,
            groups: cutlass.Constexpr,
            a: cute.Pointer,
            b: cute.Pointer,
            sa: cute.Pointer,
            c: cute.Pointer,
            c_scale: cute.Pointer,
            sb: cute.Tensor,
            max_active_clusters: cutlass.Constexpr,
            stream: cuda.CUstream,
        ):
            # ``sb`` is a tensor so that TVM FFI finds the device stream.
            # Rows are tokens holding every group's ranks side by side: [M, groups * N].
            c_layout = cute.make_layout((m, n, groups), stride=(groups * n, 1, n))
            self(
                cute.make_tensor(a, cute.make_ordered_layout((m, k, groups), order=(1, 0, 2))),
                cute.make_tensor(b, cute.make_ordered_layout((n, k, groups), order=(1, 0, 2))),
                # The BF16 tile is never stored; its descriptor only sizes the output.
                cute.make_tensor(cute.recast_ptr(c, dtype=cutlass.BFloat16), c_layout),
                cute.make_tensor(
                    sa, cute.make_ordered_layout((sf_m, k // 128, groups), order=(0, 1, 2))
                ),
                cute.make_tensor(
                    sb.iterator,
                    cute.make_ordered_layout((n // 128, k // 128, groups), order=(1, 0, 2)),
                ),
                max_active_clusters,
                stream,
                c_fp8=cute.make_tensor(c, c_layout),
                c_scale=cute.make_tensor(
                    c_scale, cute.make_layout(((m + 127) // 128 * 128 * groups * n // 32,))
                ),
            )

    return GroupedBlock128OutputGemm


_BLOCK128_COMPILED = {}


def grouped_output_block128_mxfp8(
    data: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    cluster: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped O-A of 1x128 block-scaled E4M3 attention output, quantized to MXFP8 for O-B.

    ``data`` is E4M3 ``[groups, M, D]`` with FP32 ``scale`` ``[groups, D / 128, pad4(M)]``
    (one scale per token and 128 channels), ``weight`` E4M3 ``[groups, R, D]`` with FP32
    128x128 block scales ``[groups, R / 128, D / 128]``. Returns the projection's MXFP8
    E4M3 values ``[M, groups * R]`` and CUTLASS-swizzled UE8M0 scales, as ``mxfp8_quantize``
    of the BF16 projection lays them out, without materializing that BF16 tensor.
    ``cluster`` overrides the measured cluster shape of the 2-CTA 256x128 tiles.
    """
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

    groups, rows, depth = data.shape
    rank = weight.shape[1]
    if (
        rows == 0
        or data.dtype != torch.float8_e4m3fn
        or not data.is_contiguous()
        or weight.dtype != torch.float8_e4m3fn
        or weight.shape != (groups, rank, depth)
        or not weight.is_contiguous()
        or scale.dtype != torch.float32
        or scale.shape != (groups, depth // 128, (rows + 3) // 4 * 4)
        or not scale.is_contiguous()
        or weight_scale.dtype != torch.float32
        or weight_scale.shape != (groups, rank // 128, depth // 128)
        or not weight_scale.is_contiguous()
        or depth % 128
        or rank % 128
    ):
        raise ValueError("CSA2 grouped block-128 O-A projection got an unsupported input")
    # Measured on GB300 at 16K rows: 2-CTA 256x128 tiles on 2x2 clusters. A cluster cannot
    # span groups, so it pairs N tiles only when every group has an even number of them
    # (a pair would otherwise idle half its CTAs).
    tactic = (True, (256, 128), cluster or (2, 2 if rank % 256 == 0 else 1))
    key = (data.device.index, tactic, rank, depth, groups)
    sf_rows = triton.cdiv(rows, 128) * 128
    projected = torch.empty((rows, groups * rank), device=data.device, dtype=torch.float8_e4m3fn)
    projected_scale = torch.empty(
        sf_rows * groups * rank // 32, device=data.device, dtype=torch.uint8
    )
    tensors = (data, weight, scale, projected, projected_scale)
    if key not in _BLOCK128_COMPILED:
        dtypes = (
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.Float32,
            cutlass.Float8E4M3FN,
            cutlass.Uint8,
        )
        pointers = [
            make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
            for tensor, dtype in zip(tensors, dtypes)
        ]
        use_2cta, tile, cluster = tactic
        kernel = _block128_kernel_type()(cutlass.Float32, use_2cta, tile, cluster, mxfp8_out=True)
        active = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
        _BLOCK128_COMPILED[key] = cute.compile(
            kernel.wrapper_mxfp8,
            rows,
            (rows + 3) // 4 * 4,
            rank,
            depth,
            groups,
            *pointers,
            cute.runtime.from_dlpack(weight_scale),
            active,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    _BLOCK128_COMPILED[key](
        rows, (rows + 3) // 4 * 4, *(tensor.data_ptr() for tensor in tensors), weight_scale
    )
    return projected, projected_scale
