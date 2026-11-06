# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 query normalization, native MXFP8 projections and the grouped O-A projection.

The grouped O-A projection runs as batched MXFP8 GEMMs when the grouped weight's 128x128 block scales are
powers of two: the same E4M3 weight then has exact UE8M0 1x32 scales, and the attention output is quantized
to MXFP8 instead of 1x128 FP8 blocks.
"""

from functools import cache
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
    linear: "Linear",
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


# Decode steps of up to this many rows run the attention projections (QKV-A, Q-B, O-B) without the
# FlashInfer CUTLASS GEMM. Their weights then stream from HBM, where it loses to the CuTe DSL GEMMs
# on GB300, while the autotuner compares the candidates with the weight resident in L2.
PROJECTION_CUTE_DSL_ROWS = 256


def projection_backends(rows: int, allowed_backends: tuple[str, ...]) -> tuple[str, ...]:
    """The MXFP8 GEMM candidates of a ``rows``-row projection (``PROJECTION_CUTE_DSL_ROWS``)."""
    if rows <= PROJECTION_CUTE_DSL_ROWS:
        kept = tuple(backend for backend in allowed_backends if backend != "flashinfer_cutlass")
        if kept:
            return kept
    return allowed_backends


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
    linear: "Linear",
    activation: torch.Tensor,
    scales: torch.Tensor,
    *,
    cute_min_rows: int | None = None,
    allowed_backends: tuple[str, ...] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``linear`` applied to an MXFP8-quantized activation, dispatched as its ``apply`` would.

    ``scales`` are the activation's CUTLASS-swizzled UE8M0 scales, as from
    ``mxfp8_quantize``; ``linear`` must use the CUTLASS MXFP8 method without bias. From
    ``cute_min_rows`` rows on, the native GEMM runs as the byte-identical persistent CuTe
    DSL GEMM (``dense_mxfp8_gemm``), which writes into ``out`` when given. The autotuned GEMM
    chooses among ``allowed_backends``, by default the linear's.
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
            allowed_backends=(
                linear.mxfp8_allowed_backends if allowed_backends is None else allowed_backends
            ),
        )
    if method.backend == "flashinfer" or (
        method.backend == "auto" and (tune or (method._flashinfer_autotuned and capture))
    ):
        return method._flashinfer_mxfp8(
            activation, scales, linear.weight, linear.weight_scale, linear.dtype
        )
    if cute_min_rows is not None:
        return dense_or_native_mxfp8_gemm(
            linear, activation, scales, min_rows=cute_min_rows, out=out
        )
    return native_mxfp8_gemm(linear, activation, scales)


def dense_or_native_mxfp8_gemm(
    linear: "Linear",
    activation: torch.Tensor,
    scales: torch.Tensor,
    *,
    min_rows: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``native_mxfp8_gemm`` of ``linear`` on the byte-identical persistent CuTe DSL GEMM
    (``dense_mxfp8_gemm``, which writes into ``out`` when given) from ``min_rows`` rows where its
    weight shape allows; a graph capture takes that GEMM only once an eager call compiled it."""
    columns, depth = linear.weight.shape
    if (
        activation.shape[0] >= max(min_rows, 1)
        and linear.dtype == torch.bfloat16
        and columns % 256 == 0
        and depth % 128 == 0
        and linear.weight.is_contiguous()
        and linear.weight_scale.numel() == columns * depth // 32
        and (
            not torch.cuda.is_current_stream_capturing()
            or dense_mxfp8_gemm_compiled(activation.device, columns, depth)
        )
    ):
        return dense_mxfp8_gemm(
            activation,
            scales,
            linear.weight,
            linear.weight_scale,
            linear._mxfp8_alpha,
            out=out,
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


# The row counts are runtime values, so every token count runs one compiled kernel.
@triton.jit(do_not_specialize=["M", "SF_ROWS"])
def _inverse_rope_mxfp8_kernel(
    X,
    POS,
    CS,
    Q,
    S,
    M,
    SF_ROWS,
    stride_xm,
    D: tl.constexpr,
    HEADS: tl.constexpr,
):
    """One token row x HEADS heads of one group -> E4M3 ``[groups, M, D]`` + swizzled UE8M0.

    Channels 0..447 of each head quantize as loaded. The last 64 carry GPT-J RoPE:
    they load again as pairs, rotate back in FP32 and round to BF16 before their two
    1x32 blocks quantize. Padding rows up to SF_ROWS write zero scales.
    """
    # Element offsets reach groups * M * D, which passes int32 for long token batches.
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
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
        sf_rows,
        output.stride(0),
        depth,
        heads,
        num_warps=2,
    )
    return quantized, scales


@cache
def _kernel_type() -> type:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.kernels.blackwell.dense_blockscaled_gemm_persistent import (
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
    prequantized: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Grouped O-A of BF16 attention output with inverse RoPE, as MXFP8 GEMMs.

    ``output`` is ``[M, groups * D]`` BF16 with GPT-J rotary tails on each 512-channel
    head, ``weight`` ``[groups, R, D]`` E4M3, ``weight_scale`` from ``mxfp8_weight_scale``
    and ``alpha`` a unit FP32 scalar. ``prequantized``, when given, is what
    ``inverse_rope_mxfp8`` makes of ``output``, which is then not read. Returns BF16
    ``[M, groups * R]``, or with ``quantize_output`` its MXFP8 E4M3 values and
    CUTLASS-swizzled UE8M0 scales as ``mxfp8_quantize`` lays them out. Returns None while
    a graph capture would have to compile the GEMM.
    """
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

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
    # Measured on GB300: up to 128 rows the GEMM streams the weight, which one-CTA 128x64
    # tiles spread over twice the SMs of 128x128 ones; 2-CTA 256x128 tiles beyond.
    if rows <= 128:
        tile, cluster = (128, 64), (1, 1)
    else:
        tile, cluster = (256, 128), (2, 1)
    key = (output.device.index, tile, rank, depth, groups, quantize_output)
    if key not in _COMPILED and torch.cuda.is_current_stream_capturing():
        return None
    sf_rows = triton.cdiv(rows, 128) * 128
    quantized, scales = (
        prequantized
        if prequantized is not None
        else inverse_rope_mxfp8(output, positions, cos_sin, groups)
    )
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


_DENSE_TILE, _DENSE_CLUSTER = (256, 256), (2, 2)


def _dense_key(device: torch.device, columns: int, depth: int) -> tuple:
    return ("dense", device.index, _DENSE_TILE, _DENSE_CLUSTER, columns, depth)


def dense_mxfp8_gemm_compiled(device: torch.device, columns: int, depth: int) -> bool:
    """Whether ``dense_mxfp8_gemm`` is compiled for an ``[N, K]`` weight, so a graph capture can
    run it."""
    return _dense_key(device, columns, depth) in _COMPILED


def dense_mxfp8_gemm(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """BF16 ``activation @ weight.T`` of MXFP8 operands on the persistent CuTe DSL GEMM.

    ``activation`` is E4M3 ``[M, K]`` and ``weight`` E4M3 ``[N, K]``, each with
    CUTLASS-swizzled UE8M0 scales (activation rows padded to 128), and ``alpha`` a unit
    FP32 scalar. The result equals ``mxfp8_mxfp8_gemm`` byte for byte; measured on GB300
    for the O-B shape (N 5120, K 8192), 2-CTA 256x256 tiles on 2x2 clusters beat the
    autotuned CUTLASS GEMM from 1024 rows on. A contiguous BF16 ``[M, N]`` ``out`` receives it.
    """
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

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
    tile, cluster = _DENSE_TILE, _DENSE_CLUSTER
    key = _dense_key(activation.device, columns, depth)
    sf_rows = triton.cdiv(rows, 128) * 128
    if (
        out is not None
        and out.shape == (rows, columns)
        and out.dtype == torch.bfloat16
        and out.device == activation.device
        and out.is_contiguous()
        and out.data_ptr() % 16 == 0
    ):
        output = out
    else:
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

    from tensorrt_llm._torch.kernels.blackwell.blockwise_gemm.blockwise_gemm import (
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


@cache
def _n256_kernel_type() -> type:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.kernels.blackwell.blockwise_gemm.blockwise_gemm_n256 import (
        Sm100BlockwiseGemmMxfp8N256Kernel,
    )

    class GroupedN256OutputGemm(Sm100BlockwiseGemmMxfp8N256Kernel):
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
            # Same operands and token-major output as the 256x128 kernel's ``wrapper_mxfp8``.
            self(
                cute.make_tensor(a, cute.make_ordered_layout((m, k, groups), order=(1, 0, 2))),
                cute.make_tensor(b, cute.make_ordered_layout((n, k, groups), order=(1, 0, 2))),
                cute.make_tensor(c, cute.make_layout((m, n, groups), stride=(groups * n, 1, n))),
                cute.make_tensor(
                    sa, cute.make_ordered_layout((sf_m, k // 128, groups), order=(0, 1, 2))
                ),
                cute.make_tensor(
                    sb.iterator,
                    cute.make_ordered_layout((n // 128, k // 128, groups), order=(1, 0, 2)),
                ),
                max_active_clusters,
                stream,
                out_sf=cute.make_tensor(
                    c_scale, cute.make_layout(((m + 127) // 128 * 128 * groups * n // 32,))
                ),
            )

    return GroupedN256OutputGemm


# Rows from which the 2-CTA 256x256 tiles beat the 256x128 ones on GB300 when a context pass
# brings each layer's weights in cold.
_N256_MIN_ROWS = 4096  # 4K
_BLOCK128_COMPILED = {}


def grouped_output_block128_mxfp8(
    data: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    cluster: tuple[int, int] | None = None,
    raster_along_m: bool = False,
    n256: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped O-A of 1x128 block-scaled E4M3 attention output, quantized to MXFP8 for O-B.

    ``data`` is E4M3 ``[groups, M, D]`` with FP32 ``scale`` ``[groups, D / 128, pad4(M)]``
    (one scale per token and 128 channels), ``weight`` E4M3 ``[groups, R, D]`` with FP32
    128x128 block scales ``[groups, R / 128, D / 128]``. Returns the projection's MXFP8
    E4M3 values ``[M, groups * R]`` and CUTLASS-swizzled UE8M0 scales, as ``mxfp8_quantize``
    of the BF16 projection lays them out, without materializing that BF16 tensor.
    ``cluster`` overrides the measured cluster shape of the 2-CTA 256x128 tiles;
    ``raster_along_m`` sets the persistent tile order (same bits); N-major order keeps a
    group's 128-row input block in L2 while its output columns are computed.
    ``n256`` runs the N-major 2-CTA 256x256-tile kernel instead (same bits; default from
    ``_N256_MIN_ROWS`` rows when the rank is a multiple of 256).
    """
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

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
    if n256 is None:
        n256 = _N256_MIN_ROWS <= rows and rank % 256 == 0
    elif n256 and rank % 256:
        raise ValueError("The 256x256-tile O-A needs a rank that is a multiple of 256")
    # Measured on GB300 at 16K rows: 2-CTA 256x128 tiles on 2x2 clusters. A cluster cannot
    # span groups, so it pairs N tiles only when every group has an even number of them
    # (a pair would otherwise idle half its CTAs).
    tactic = (True, (256, 128), cluster or (2, 2 if rank % 256 == 0 else 1))
    key = (
        (data.device.index, "n256", rank, depth, groups)
        if n256
        else (data.device.index, tactic, rank, depth, groups, raster_along_m)
    )
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
        if n256:
            cluster = (2, 1)  # the 256x256-tile kernel's fixed cluster shape
            kernel = _n256_kernel_type()()
        else:
            kernel = _block128_kernel_type()(
                cutlass.Float32,
                use_2cta,
                tile,
                cluster,
                mxfp8_out=True,
                raster_along_m=raster_along_m,
            )
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
