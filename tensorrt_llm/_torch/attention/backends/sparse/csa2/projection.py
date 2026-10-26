# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 query normalization and native MXFP8 projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.custom_ops.torch_custom_ops import mxfp8_gemm_autotuned
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._utils import get_sm_version

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.linear import Linear


def native_mxfp8_gemm(
    linear: Linear,
    activation: torch.Tensor,
    scales: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Consume prequantized input with the native projection's startup tactic cache."""
    gemm = (
        torch.ops.trtllm.mxfp8_mxfp8_gemm_autotuned
        if linear.quant_method.needs_native_autotune
        else torch.ops.trtllm.mxfp8_mxfp8_gemm
    )
    return gemm(
        activation,
        scales,
        linear.weight if weight is None else weight,
        linear.weight_scale if weight_scale is None else weight_scale,
        linear._mxfp8_alpha,
        linear.dtype,
    )


@triton.jit
def _prefetch_share(BASE, BYTES, LINES: tl.constexpr):
    """Prefetch this program's share of the ``BYTES`` bytes at ``BASE`` into L2.

    Program ``p`` of ``P`` takes the 128-byte lines ``p, p + P, ...``; ``LINES`` covers
    every line once over all programs.
    """
    programs = tl.num_programs(0)
    offsets = (tl.program_id(0) + tl.arange(0, LINES) * programs).to(tl.int64) * 128
    tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p prefetch.global.L2 [$1]; mov.u32 $0, 0; }",
        "=r,l,r",
        [BASE + offsets, (offsets < BYTES).to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def _prefetch_args(tensors, programs: int, enabled: bool, slots: int = 2) -> list:
    """``(base, bytes, lines)`` kernel arguments that spread each tensor over ``programs``.

    Unused slots, and every slot unless ``enabled``, get zero lines, which the kernels
    skip at compile time.
    """
    if len(tensors) > slots or not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("L2 prefetch takes at most two contiguous tensors")
    args = []
    for tensor in tensors if enabled else ():
        size = tensor.numel() * tensor.element_size()
        lines = triton.cdiv(size, 128)
        args += [
            tensor.view(torch.uint8),
            size,
            triton.next_power_of_2(triton.cdiv(lines, programs)),
        ]
    while len(args) < 3 * slots:
        args += [None, 0, 0]
    return args


@triton.jit
def _touch_range(BASE, WORDS, acc, BLOCK: tl.constexpr):
    for start in range(tl.program_id(0) * BLOCK, WORDS, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        acc ^= tl.load(BASE + offsets, offsets < WORDS, 0)
    return acc


@triton.jit
def _l2_touch_kernel(A, A_WORDS, B, B_WORDS, C, C_WORDS, SINK, BLOCK: tl.constexpr):
    """Read every int32 word of three ranges, spread over the programs, so they land in L2."""
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    acc = _touch_range(A, A_WORDS, acc, BLOCK)
    acc = _touch_range(B, B_WORDS, acc, BLOCK)
    acc = _touch_range(C, C_WORDS, acc, BLOCK)
    # The per-program sink keeps the loads; nothing reads it.
    tl.store(SINK + tl.program_id(0), tl.sum(acc, 0))


def l2_touch(tensors: tuple[torch.Tensor, ...], sink: torch.Tensor, block: int = 2048) -> None:
    """Load up to three contiguous tensors on the current stream so later reads hit L2.

    Unlike ``prefetch.global.L2``, the loads cannot be dropped; ``sink`` (int32, one word per
    program) sizes the grid.
    """
    if len(tensors) > 3 or sink.dtype != torch.int32:
        raise ValueError("L2 touch takes at most three tensors and an int32 sink")
    args = []
    for tensor in tensors:
        size = tensor.numel() * tensor.element_size()
        if not tensor.is_contiguous() or size % 4:
            raise ValueError("L2 touch ranges must be contiguous whole int32 words")
        args += [tensor.view(torch.uint8).view(torch.int32), size // 4]
    while len(args) < 6:
        args += [sink, 0]
    _l2_touch_kernel[(sink.numel(),)](*args, sink, block, num_warps=8)


# QUANTIZE is a runtime value, so both modes run one compiled kernel and write the same BF16
# rows; it only masks the MXFP8 stores.
@triton.jit(do_not_specialize=["M", "QUANTIZE"])
def _rms_norm_mxfp8_kernel(
    X,
    W,
    Y,
    Q,
    S,
    M,
    QUANTIZE,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,
    NORM: tl.constexpr,
    STORE_NORM: tl.constexpr,
    PA,
    PA_BYTES,
    PA_LINES: tl.constexpr,
    PB,
    PB_BYTES,
    PB_LINES: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    # Under PDL, the next projection's weights head for L2 while earlier kernels finish.
    if PA_LINES > 0:
        _prefetch_share(PA, PA_BYTES, PA_LINES)
    if PB_LINES > 0:
        _prefetch_share(PB, PB_BYTES, PB_LINES)
    if NORM:
        weight = tl.load(W + columns, columns < N, 0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(X + row * STRIDE + columns, (row < M) & (columns < N), 0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    if NORM:
        inv = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
        normalized = ((x * inv) * weight).to(tl.bfloat16)
        if STORE_NORM:
            tl.store(Y + row * N + columns, normalized, (row < M) & (columns < N))
        # Quantization consumes the same rounded BF16 value as the separate norm.
        values = tl.reshape(normalized.to(tl.float32), (BLOCK // 32, 32))
    else:
        values = tl.reshape(x, (BLOCK // 32, 32))
    amax = tl.max(tl.abs(values), 1)
    scale = amax * (1.0 / 448.0)
    bits = scale.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.where(scale <= 2.0**-127, 0, tl.minimum(exponent, 254))
    reciprocal = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    reciprocal = tl.where(amax == 0, 0.0, reciprocal)
    quantized = (values * reciprocal[:, None]).to(tl.float8e4nv)
    quantize = QUANTIZE != 0
    tl.store(
        Q + row * N + columns,
        tl.reshape(quantized, (BLOCK,)),
        (row < M) & (columns < N) & quantize,
    )
    blocks = tl.arange(0, BLOCK // 32)
    # CUTLASS 128x4 scale layout, including initialized padding rows.
    offset = (
        (row // 128) * (N // 32) * 128
        + (blocks // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + blocks % 4
    )
    tl.store(S + offset, tl.where(row < M, exponent, 0).to(tl.uint8), (blocks < N // 32) & quantize)


def rms_norm_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    prefetch: tuple[torch.Tensor, ...] = (),
    normalized: bool = True,
    quantize: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Return normalized BF16, E4M3 values and swizzled UE8M0 scales.

    The BF16 output preserves the indexer's independent input; without ``normalized``
    it is not stored and None is returned in its place, and without ``quantize`` neither
    are the E4M3 values and scales. Column slices of fused projections are supported
    without copying their row padding. Under PDL, the kernel pulls the (at most two)
    contiguous ``prefetch`` tensors into L2 before waiting on earlier kernels.
    """
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or x.stride(1) != 1
        or x.shape[1] % 128
        or x.shape[1] > 8192
        or weight.shape != (x.shape[1],)
        or not weight.is_contiguous()
        or weight.device != x.device
        or weight.dtype != x.dtype
    ):
        raise ValueError("CSA2 fused norm requires CUDA BF16 rows with 128-aligned width <= 8192")
    if not (normalized or quantize):
        raise ValueError("CSA2 fused norm must store its BF16 rows or their MXFP8 copy")
    rows, width = x.shape
    output = torch.empty((rows, width), device=x.device, dtype=x.dtype) if normalized else None
    quantized = scales = None
    programs = rows
    if quantize:
        # The scale layout's padding rows are written too.
        programs = triton.cdiv(rows, 128) * 128
        quantized = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
        scales = torch.empty(programs * (width // 32), device=x.device, dtype=torch.uint8)
    if rows:
        pdl = _pdl()
        _rms_norm_mxfp8_kernel[(programs,)](
            x,
            weight,
            quantized if output is None else output,
            # Unwritten outputs take same-typed views of the BF16 rows: one compiled kernel.
            quantized if quantized is not None else output.view(torch.float8_e4m3fn),
            scales if scales is not None else output.view(torch.uint8),
            rows,
            int(quantize),
            width,
            x.stride(0),
            eps,
            triton.next_power_of_2(width),
            pdl,
            True,
            normalized,
            *_prefetch_args(prefetch, programs, pdl),
            num_warps=4,
            enable_fp_fusion=False,
            launch_pdl=pdl,
        )
    return output, quantized, scales


def mxfp8_quantize_rows(
    x: torch.Tensor, *, prefetch: tuple[torch.Tensor, ...] = ()
) -> tuple[torch.Tensor, torch.Tensor]:
    """``mxfp8_quantize(x, True)`` of BF16 rows: E4M3 values and swizzled UE8M0 scales.

    Scale rows past ``x`` up to the next multiple of 128 are written as zero. Under
    PDL, the kernel pulls the (at most two) contiguous ``prefetch`` tensors into L2
    before waiting on earlier kernels.
    """
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or x.stride(1) != 1
        or x.shape[1] % 128
        or x.shape[1] > 8192
    ):
        raise ValueError(
            "CSA2 row quantization requires CUDA BF16 rows with 128-aligned width <= 8192"
        )
    rows, width = x.shape
    padded_rows = triton.cdiv(rows, 128) * 128
    quantized = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(padded_rows * (width // 32), device=x.device, dtype=torch.uint8)
    if rows:
        pdl = _pdl()
        _rms_norm_mxfp8_kernel[(padded_rows,)](
            x,
            x,
            x,
            quantized,
            scales,
            rows,
            1,
            width,
            x.stride(0),
            0.0,
            triton.next_power_of_2(width),
            pdl,
            False,
            False,
            *_prefetch_args(prefetch, padded_rows, pdl),
            num_warps=4,
            enable_fp_fusion=False,
            launch_pdl=pdl,
        )
    return quantized, scales


def mxfp8_query_gemm(
    activation: torch.Tensor,
    scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    allowed_backends: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Consume fused activation quantization using the existing GEMM tactic cache."""
    from tensorrt_llm._torch.custom_ops.mxfp8_gemm import MXFP8_DEFAULT_BACKENDS

    return mxfp8_gemm_autotuned(
        activation,
        scales,
        weight,
        weight_scale,
        torch.bfloat16,
        MXFP8_DEFAULT_BACKENDS if allowed_backends is None else allowed_backends,
    )


def mxfp8_linear_quantized(
    linear: Linear,
    activation: torch.Tensor,
    scales: torch.Tensor,
    *,
    cute_min_rows: int | None = None,
) -> torch.Tensor:
    """``linear`` applied to an MXFP8-quantized activation, dispatched as its ``apply`` would.

    ``scales`` are the activation's CUTLASS-swizzled UE8M0 scales, as from
    ``mxfp8_quantize``; ``linear`` must use the CUTLASS MXFP8 method without bias. From
    ``cute_min_rows`` rows on, the native GEMM runs as the byte-identical persistent CuTe
    DSL GEMM (``dense_mxfp8_gemm``).
    """
    from tensorrt_llm._torch.modules.linear import (
        _FLASHINFER_MXFP8_AUTOTUNE_ACTIVE,
        _FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE,
    )

    method = linear.quant_method
    capture = _FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
    tune = _FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get()
    if method.uses_mxfp8_backend_autotune:
        return mxfp8_query_gemm(
            activation,
            scales,
            linear.weight,
            linear.weight_scale,
            allowed_backends=linear.mxfp8_allowed_backends,
        )
    if method.backend == "flashinfer" or (
        method.backend == "auto" and (tune or (method._flashinfer_autotuned and capture))
    ):
        return method._flashinfer_mxfp8(
            activation, scales, linear.weight, linear.weight_scale, linear.dtype
        )
    if (
        cute_min_rows is not None
        and activation.shape[0] >= cute_min_rows
        and linear.dtype == torch.bfloat16
        and linear.weight.shape[0] % 256 == 0
        and linear.weight.is_contiguous()
        and not torch.cuda.is_current_stream_capturing()
    ):
        from .output_projection import dense_mxfp8_gemm

        return dense_mxfp8_gemm(
            activation, scales, linear.weight, linear.weight_scale, linear._mxfp8_alpha
        )
    return native_mxfp8_gemm(linear, activation, scales)


# Rows a small-M projection serves: decode with a handful of tokens per step.
SMALL_M_ROWS = 16
# Past this FP8 weight size the warp-MMA kernel's weight stream is slower than the tensor-core GEMM.
LOWM_MAX_WEIGHT_BYTES = 8 << 20  # 8 MiB

# FP32 partial sums and self-clearing tile counters per (device, call site, shape).
_SPLIT_K_WORKSPACES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _split_k_workspace(
    device: torch.device, name: str, columns: int, tiles: int, split_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Workspace of one call site, kept for the lifetime of the graphs that captured it."""
    key = (device, name, columns, tiles, split_k)
    workspace = _SPLIT_K_WORKSPACES.get(key)
    if workspace is None:
        workspace = (
            torch.empty(split_k * SMALL_M_ROWS * columns, device=device, dtype=torch.float32),
            torch.zeros(max(tiles, 1), device=device, dtype=torch.int32),
        )
        _SPLIT_K_WORKSPACES[key] = workspace
    return workspace


# Zero-padded UE8M0 activation scales of quantized small-M outputs, per call site.
_SMALL_M_SCALES: dict[tuple, torch.Tensor] = {}


def _small_m_scales(device: torch.device, name: str, columns: int) -> torch.Tensor:
    """CUTLASS 128x4 scale buffer of one 128-row tile; rows past the step stay zero."""
    key = (device, name, columns)
    scales = _SMALL_M_SCALES.get(key)
    if scales is None:
        scales = torch.zeros(128 * columns // 32, device=device, dtype=torch.uint8)
        _SMALL_M_SCALES[key] = scales
    return scales


def _pdl() -> bool:
    return get_env_enable_pdl() and get_sm_version() >= 90


@triton.jit
def _wait_prefetching(rows, span: tl.constexpr, LINES: tl.constexpr):
    """Prefetch ``span`` bytes from each row pointer into L2, then wait on earlier kernels.

    Weights do not depend on earlier kernels, so under PDL their HBM reads overlap them.
    """
    lines = tl.minimum(tl.arange(0, LINES), (span + 127) // 128 - 1)
    tl.inline_asm_elementwise(
        "mov.u32 $0, 0; prefetch.global.L2 [$1];",
        "=r,l",
        [rows[:, None] + lines[None, :] * 128],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.extra.cuda.gdc_wait()


@triton.jit
def _store_small_m_output(
    total,
    Y,
    SF,
    M,
    stride_ym,
    columns,
    COLUMNS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    QUANTIZE: tl.constexpr,
):
    """Store BF16 rows, or their MXFP8 quantization when the columns are one 1x32 block.

    The quantization matches ``mxfp8_quantize`` of the BF16 rows: E4M3 values in ``Y``
    and UE8M0 bytes in ``SF`` (CUTLASS 128x4 layout of a single 128-row tile).
    """
    rows = tl.arange(0, BLOCK_M)
    mask = rows[:, None] < M
    output = Y + rows[:, None] * stride_ym + columns[None, :]
    if QUANTIZE:
        values = total.to(tl.bfloat16).to(tl.float32)
        amax = tl.max(tl.abs(values), 1)
        scale = amax * (1.0 / 448.0)
        bits = scale.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        exponent = tl.where(scale <= 2.0**-127, 0, tl.minimum(exponent, 254))
        reciprocal = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
        reciprocal = tl.where(amax == 0, 0.0, reciprocal)
        tl.store(output, (values * reciprocal[:, None]).to(tl.float8e4nv), mask)
        block = tl.min(columns, 0) // 32
        offset = (block // 4) * 512 + (rows % 32) * 16 + (rows // 32) * 4 + block % 4
        tl.store(SF + offset, tl.where(rows < M, exponent, 0).to(tl.uint8))
    else:
        tl.store(output, total.to(tl.bfloat16), mask)


@triton.jit
def _split_k_epilogue(
    accumulator,
    Y,
    SF,
    P,
    C,
    M,
    stride_ym,
    columns,
    tile,
    part,
    COLUMNS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    QUANTIZE: tl.constexpr,
):
    """Store one K part; the last part of a tile adds all parts in a fixed order."""
    rows = tl.arange(0, BLOCK_M)
    mask = rows[:, None] < M
    if SPLIT_K == 1:
        _store_small_m_output(accumulator, Y, SF, M, stride_ym, columns, COLUMNS, BLOCK_M, QUANTIZE)
    else:
        partials = P + rows[:, None] * COLUMNS + columns[None, :]
        tl.store(partials + part * BLOCK_M * COLUMNS, accumulator, mask)
        tl.debug_barrier()
        if tl.atomic_add(C + tile, 1, sem="acq_rel", scope="gpu") == SPLIT_K - 1:
            total = tl.load(partials, mask, 0.0, cache_modifier=".cg")
            for index in tl.static_range(1, SPLIT_K):
                total += tl.load(
                    partials + index * BLOCK_M * COLUMNS, mask, 0.0, cache_modifier=".cg"
                )
            _store_small_m_output(total, Y, SF, M, stride_ym, columns, COLUMNS, BLOCK_M, QUANTIZE)
            tl.store(C + tile, 0)


@triton.jit
def _inverse_rope_block128_small_m_kernel(
    X,
    POS,
    CS,
    W,
    S,
    Y,
    SF,
    P,
    C,
    M,
    stride_xm,
    stride_ym,
    GROUPS: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PDL: tl.constexpr,
    PREFETCH_LINES: tl.constexpr,
    QUANTIZE: tl.constexpr,
):
    """Y[M, g, :] = inverse_rope(X[M, g]) @ dequant(W[g])^T with block-128 FP8 weights.

    X holds BF16 attention output, heads of 512 channels whose last 64 carry GPT-J
    RoPE; those are rotated back in FP32 on load. Programs own BLOCK_N columns of one
    group and one of SPLIT_K slices of its heads, prefetched into L2 before waiting on
    earlier kernels under PDL. Each 128-wide K chunk is one weight scale block: its
    FP32 dot product is scaled once, as the block-128 BMM does, but the activation
    stays BF16 instead of being quantized to E4M3.
    """
    tile = tl.program_id(0)
    part = tl.program_id(1)
    group = tile // (R // BLOCK_N)
    local = (tile % (R // BLOCK_N)) * BLOCK_N + tl.arange(0, BLOCK_N)
    weights = W + group * R * D + local[:, None] * D
    heads: tl.constexpr = D // 512 // SPLIT_K
    if PDL:
        _wait_prefetching(
            W + group * R * D + local * D + part * heads * 512, heads * 512, PREFETCH_LINES
        )
    rows = tl.arange(0, BLOCK_M)
    positions = tl.load(POS + rows, rows < M, 0)
    pairs = tl.arange(0, 64)
    rotary = (rows[:, None] < M) & (pairs >= 32)[None, :]
    trig = positions[:, None] * 64 + tl.where(pairs >= 32, pairs - 32, 0)[None, :]
    # Pairs 32..63 of a head's last chunk are rotary; the rest keep cos 1, sin 0.
    cosine = tl.load(CS + trig, rotary, 1.0)
    sine = tl.load(CS + trig + 32, rotary, 0.0)
    scales = S + (group * (R // 128) + local // 128) * (D // 128)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for head in range(part * heads, part * heads + heads):
        for chunk in tl.static_range(4):
            depth = head * 512 + chunk * 128 + tl.arange(0, 128)
            x = tl.load(
                X + rows[:, None] * stride_xm + group * D + depth[None, :],
                rows[:, None] < M,
                0.0,
            ).to(tl.float32)
            if chunk == 3:
                even, odd = tl.split(tl.reshape(x, (BLOCK_M, 64, 2)))
                x = tl.reshape(
                    tl.join(even * cosine + odd * sine, odd * cosine - even * sine),
                    (BLOCK_M, 128),
                )
            w = tl.load(weights + depth[None, :])
            partial = tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
            accumulator += partial * tl.load(scales + head * 4 + chunk)[None, :]
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    _split_k_epilogue(
        accumulator,
        Y,
        SF,
        P,
        C,
        M,
        stride_ym,
        group * R + local,
        tile,
        part,
        GROUPS * R,
        BLOCK_M,
        SPLIT_K,
        QUANTIZE,
    )


def inverse_rope_block128_small_m(
    output: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    name: str,
    block_n: int = 32,
    split_k: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
    quantize_output: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Grouped O-A projection of at most ``SMALL_M_ROWS`` decode rows with inverse RoPE fused.

    ``output`` is ``[M, groups * D]`` BF16 attention output (GPT-J rotary tails),
    ``weight`` ``[groups, R, D]`` E4M3 and ``weight_scale`` ``[groups, R / 128, D / 128]``
    FP32. ``split_k`` must divide the ``D / 512`` heads of a group; unset, every head
    up to eight is its own K slice. ``name`` identifies the call site's split-K
    workspace. Returns BF16 ``[M, groups * R]``, or with ``quantize_output`` (32-column
    tiles) its MXFP8 E4M3 values and CUTLASS-swizzled UE8M0 scales as ``mxfp8_quantize``
    lays them out; the scales belong to the call site and are rewritten by its next call.
    """
    rows = output.shape[0]
    groups, rank, depth = weight.shape
    if split_k is None:
        split_k = next((split for split in (8, 4, 2) if depth % (512 * split) == 0), 1)
    if (
        output.dtype != torch.bfloat16
        or output.stride(1) != 1
        or output.shape[1] != groups * depth
        or not 0 < rows <= SMALL_M_ROWS
        or positions.dtype != torch.int32
        or positions.numel() != rows
        or not positions.is_contiguous()
        or cos_sin.dtype != torch.float32
        or not cos_sin.is_contiguous()
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
        or weight_scale.dtype != torch.float32
        or weight_scale.shape != (groups, rank // 128, depth // 128)
        or not weight_scale.is_contiguous()
        or depth % 512
        or (depth // 512) % split_k
        or rank % 128
        or 128 % block_n
        or (quantize_output and block_n != 32)
    ):
        raise ValueError("CSA2 small-M O-A projection got an unsupported input")
    tiles = groups * rank // block_n
    partials, counters = _split_k_workspace(output.device, name, groups * rank, tiles, split_k)
    out = torch.empty(
        (rows, groups * rank),
        device=output.device,
        dtype=torch.float8_e4m3fn if quantize_output else torch.bfloat16,
    )
    scales = _small_m_scales(output.device, name, groups * rank) if quantize_output else out
    pdl = _pdl()
    _inverse_rope_block128_small_m_kernel[(tiles, split_k)](
        output,
        positions,
        cos_sin,
        weight,
        weight_scale,
        out,
        scales,
        partials,
        counters,
        rows,
        output.stride(0),
        out.stride(0),
        groups,
        rank,
        depth,
        SMALL_M_ROWS,
        block_n,
        split_k,
        pdl,
        triton.next_power_of_2(depth // split_k // 128),
        quantize_output,
        num_warps=num_warps,
        num_stages=num_stages,
        launch_pdl=pdl,
    )
    return (out, scales) if quantize_output else out


@triton.jit
def _gptj_rope_small_m_kernel(
    Q,
    POS,
    CS,
    AMAX,
    HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    HAS_AMAX: tl.constexpr,
    PDL: tl.constexpr,
):
    """Rotate the last 64 channels of every head of one token in place, as ``mla_rope_inplace``.

    With ``HAS_AMAX`` the token's rotated ``amax(|Q|)`` also raises the int32 at ``AMAX`` to
    its FP32 bits: a BF16 with its sign bit cleared orders as an integer, NaN above infinity.
    """
    token = tl.program_id(0)
    heads = tl.arange(0, BLOCK_H)
    valid = (heads < HEADS)[:, None]
    trig = tl.load(POS + token).to(tl.int64) * 64 + tl.arange(0, 32)
    cosine = tl.load(CS + trig)[None, :]
    sine = tl.load(CS + trig + 32)[None, :]
    base = (token * HEADS + heads[:, None]).to(tl.int64) * HEAD_DIM
    offset = base + (HEAD_DIM - 64) + tl.arange(0, 64)[None, :]
    if PDL:
        tl.extra.cuda.gdc_wait()
    x, y = tl.split(tl.reshape(tl.load(Q + offset, valid, 0.0).to(tl.float32), (BLOCK_H, 32, 2)))
    if HAS_AMAX:
        channels = tl.arange(0, NOPE_BLOCK)[None, :]
        nope = tl.load(Q + base + channels, valid & (channels < HEAD_DIM - 64), 0.0)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    # A rounded product, then one FMA, then a BF16 round, as the CUDA RoPE kernel.
    rotated = tl.join(tl.fma(cosine, x, -(sine * y)), tl.fma(sine, x, cosine * y))
    rotated = tl.reshape(rotated, (BLOCK_H, 64)).to(tl.bfloat16)
    tl.store(Q + offset, rotated, valid)
    if HAS_AMAX:
        largest = tl.maximum(
            tl.max(tl.max(nope.to(tl.int16, bitcast=True).to(tl.int32) & 0x7FFF, 1), 0),
            tl.max(tl.max(rotated.to(tl.int16, bitcast=True).to(tl.int32) & 0x7FFF, 1), 0),
        )
        if largest > 0:
            tl.atomic_max(AMAX, largest << 16)


def gptj_rope_small_m(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    amax: torch.Tensor | None = None,
) -> None:
    """GPT-J RoPE of the last 64 channels of each head of ``q`` (``[M, heads, D]`` BF16), in place.

    For at most ``SMALL_M_ROWS`` tokens; bit-identical to ``mla_rope_inplace`` and launched
    with PDL so it overlaps the end of the projection that produced ``q``. ``amax`` (int32),
    when given, is raised to the FP32 bits of the rotated ``amax(|q|)``, exactly as the CSA2
    staging scale reduction reads it from ``q``.
    """
    rows, heads, depth = q.shape
    if (
        q.dtype != torch.bfloat16
        or not q.is_contiguous()
        or not 0 < rows <= SMALL_M_ROWS
        or depth < 64
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.numel() != rows
        or not positions.is_contiguous()
        or cos_sin.dtype != torch.float32
        or not cos_sin.is_contiguous()
        or (
            amax is not None
            and (amax.dtype != torch.int32 or amax.numel() < 1 or amax.device != q.device)
        )
    ):
        raise ValueError("CSA2 small-M RoPE got an unsupported input")
    pdl = _pdl()
    _gptj_rope_small_m_kernel[(rows,)](
        q,
        positions,
        cos_sin,
        positions if amax is None else amax,
        heads,
        triton.next_power_of_2(heads),
        depth,
        triton.next_power_of_2(max(depth - 64, 1)),
        amax is not None,
        pdl,
        num_warps=2 if amax is None else 4,
        enable_fp_fusion=False,
        launch_pdl=pdl,
    )
