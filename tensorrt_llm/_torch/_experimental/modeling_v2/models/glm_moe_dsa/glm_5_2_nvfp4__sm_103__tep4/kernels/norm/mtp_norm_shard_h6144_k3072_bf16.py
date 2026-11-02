# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 MTP RMSNorm plus TP4 shard materialization for B300.

TRT-LLM contract
----------------
``Glm4MTP.forward`` RMS-normalizes the embedding and target hidden state,
concatenates their two ``[M,6144]`` outputs, and selects one TP4 chunk before
``eh_proj``.  Each rank consumes only one normalized source and one contiguous
half, so the rank mapping is::

    rank 0: embedding, columns [0, 3072)
    rank 1: embedding, columns [3072, 6144)
    rank 2: hidden,    columns [0, 3072)
    rank 3: hidden,    columns [3072, 6144)

This kernel replaces two FlashInfer RMSNorm launches plus concat with one
three-pointer launch::

    selected_source: BF16 [M, 6144], contiguous
    selected_weight: BF16 [6144], contiguous
    output:          BF16 [M, 3072], contiguous and caller-owned

Integration boundary and ABI
----------------------------
This kernel is not a drop-in replacement for either individual TRT-LLM
``RMSNorm.forward(hidden_states, residual=...)`` module.  Its tensor result is
instead equivalent to the complete fused prefix in ``Glm4MTP.forward``::

    embedding_norm = enorm(embedding)
    hidden_norm = hnorm(hidden_states)
    local = chunk(concat(embedding_norm, hidden_norm), tp_size)[tp_rank]

Integration must replace that full ``enorm + hnorm + concat + chunk`` callsite
with ``norm_shard(embedding, hidden_states)``.  The replacement returns the
same contiguous local ``[M,3072]`` tensor consumed by ``eh_proj``.  It does not
preserve the single-input RMSNorm module signature or its full-width
``[M,6144]`` output, so assigning this adapter directly to ``self.enorm`` or
``self.hnorm`` is invalid.

``H=6144``, local ``K=3072``, TP=4, BF16, and epsilon ``1e-5`` are model
constants.  The shard offset is specialized from the model's TP rank.  ``M``
is read from tensor metadata at runtime; the production adapter admits the
observed decode and context/prefill bucket set ``{1,2,4,8,16,24,32,48,64,128,
256,512,1024,2048,4096,8192}`` without calling ``.item()`` or synchronizing the
host.

Implementation
--------------
One 256-thread CTA owns one row.  Each lane performs three aligned 16-byte
BF16 loads, converts eight elements at a time to FP32, and applies the vector
``reduce("add")`` pattern.  The CTA computes one FP32 sum of
squares and then rereads only the selected 3072 elements for the weighted BF16
epilogue.  Compile-time ``1/6144`` replaces a division.

Cluster DSMEM row reduction, several rows per CTA, register-resident selected
vectors, vector-FMA accumulation, and a warp-parallel subtotal reduction all
measured slower or equal, so the kernel keeps one CTA per row.

LayerNorm's ``E[x^2] - E[x]^2`` variance identity is not relevant: RMSNorm
needs only ``E[x^2]``, so this statistic is already a numerically stable
single pass.

Performance and PDL
-------------------
On B300 the fused kernel is about 2-6x faster than the unfused TRT-LLM prefix
(two FlashInfer RMSNorm launches plus concat and chunk); the gain grows with M.

PDL defaults on.  The consumer wait is immediately before the first source
read, and each CTA signals dependents only after its output stores.  A
following PDL-aware ``eh_proj`` may prefetch its invariant weight before
waiting to read this output.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims

HIDDEN_SIZE = 6_144
LOCAL_SIZE = 3_072
EPSILON = 1.0e-5
SHARD_OFFSETS = (0, LOCAL_SIZE)

_THREADS = 256
_WARP_SIZE = 32
_WARPS = _THREADS // _WARP_SIZE
_VECTOR_ELEMENTS = 8
_VECTOR_BYTES = 16
_INV_HIDDEN_SIZE = 1.0 / HIDDEN_SIZE

CompiledKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]


@cute.kernel
def _kernel(
    source: cute.Tensor,
    norm_weight: cute.Tensor,
    output: cute.Tensor,
    SHARD_OFFSET: cutlass.Constexpr[int],
    USE_PDL: cutlass.Constexpr[bool],
) -> None:
    """Normalize one selected source row and store only its TP4 shard."""
    warp_sums = cutlass.Array(
        cutlass.Float32,
        _WARPS,
        space=cutlass.AddressSpace.smem,
    )

    _, row, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    warp_id = tid // _WARP_SIZE
    lane_id = tid % _WARP_SIZE

    if cutlass.const_expr(USE_PDL):
        cute.arch.griddepcontrol_wait()

    source_row = source.iterator.raw_ptr() + row * HIDDEN_SIZE
    local_sumsq = cutlass.Float32(0.0)
    for base in cutlass.range_constexpr(
        0,
        HIDDEN_SIZE,
        _THREADS * _VECTOR_ELEMENTS,
    ):
        column = base + tid * _VECTOR_ELEMENTS
        values = (source_row + column).load(
            count=_VECTOR_ELEMENTS,
            alignment=_VECTOR_BYTES,
        )
        values_f32 = values.to(cutlass.Float32)
        local_sumsq = local_sumsq + (values_f32 * values_f32).reduce("add")

    for offset in [16, 8, 4, 2, 1]:
        local_sumsq = local_sumsq + cute.arch.shuffle_sync_down(
            local_sumsq,
            offset,
        )
    if lane_id == 0:
        warp_sums[warp_id] = local_sumsq

    prims.barrier_cta_sync(0)

    if tid == 0:
        row_sumsq = cutlass.Float32(0.0)
        for warp in cutlass.range_constexpr(_WARPS):
            row_sumsq = row_sumsq + warp_sums[warp]
        warp_sums[0] = cute.math.rsqrt(
            row_sumsq * cutlass.Float32(_INV_HIDDEN_SIZE) + cutlass.Float32(EPSILON)
        )

    prims.barrier_cta_sync(0)

    inv_rms = warp_sums[0]
    weight_ptr = norm_weight.iterator.raw_ptr()
    output_row = output.iterator.raw_ptr() + row * LOCAL_SIZE
    for base in cutlass.range_constexpr(
        0,
        LOCAL_SIZE,
        _THREADS * _VECTOR_ELEMENTS,
    ):
        output_column = base + tid * _VECTOR_ELEMENTS
        if output_column + _VECTOR_ELEMENTS <= LOCAL_SIZE:
            source_column = SHARD_OFFSET + output_column
            selected_values = (source_row + source_column).load(
                count=_VECTOR_ELEMENTS,
                alignment=_VECTOR_BYTES,
            )
            selected_weights = (weight_ptr + source_column).load(
                count=_VECTOR_ELEMENTS,
                alignment=_VECTOR_BYTES,
            )
            result = (
                selected_values.to(cutlass.Float32) * selected_weights.to(cutlass.Float32) * inv_rms
            )
            (output_row + output_column).store(
                result.to(cutlass.BFloat16),
                alignment=_VECTOR_BYTES,
            )

    if cutlass.const_expr(USE_PDL):
        # Every CTA signals after its own stores.  The runtime releases a
        # dependent grid only after all CTAs have reached this instruction.
        prims.barrier_cta_sync(0)
        if tid == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch(
    source: cute.Tensor,
    norm_weight: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
    SHARD_OFFSET: cutlass.Constexpr[int],
    USE_PDL: cutlass.Constexpr[bool],
) -> None:
    _kernel(source, norm_weight, output, SHARD_OFFSET, USE_PDL).launch(
        grid=(1, source.shape[0], 1),
        block=(_THREADS, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


def _validate_shard_offset(shard_offset: int) -> None:
    if shard_offset not in SHARD_OFFSETS:
        raise ValueError(f"shard_offset must be one of {SHARD_OFFSETS}, got {shard_offset}")


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    shard_offset: int = 0,
    *,
    use_pdl: bool = True,
) -> CompiledKernel:
    """Compile one runtime-M, fixed-shard kernel instance.

    The cache key contains only immutable model/schedule values; tensor
    pointers and tensor identities remain runtime arguments.

    :param shard_offset: First source column in the local TP4 shard, defaults
        to 0.
    :type shard_offset: int, optional
    :param use_pdl: Enable the consumer wait, producer signal, and dependent
        launch attribute, defaults to True.
    :type use_pdl: bool, optional
    :raises ValueError: If ``shard_offset`` is not 0 or 3072.
    :return: Runtime callable with the three-tensor device ABI.
    :rtype: CompiledKernel
    """
    _validate_shard_offset(shard_offset)
    rows = cute.sym_int64()
    source = make_fake_compact_tensor(
        cutlass.BFloat16,
        (rows, HIDDEN_SIZE),
        stride_order=(1, 0),
        assumed_align=_VECTOR_BYTES,
    )
    norm_weight = make_fake_compact_tensor(
        cutlass.BFloat16,
        (HIDDEN_SIZE,),
        stride_order=(0,),
        assumed_align=_VECTOR_BYTES,
    )
    output = make_fake_compact_tensor(
        cutlass.BFloat16,
        (rows, LOCAL_SIZE),
        stride_order=(1, 0),
        assumed_align=_VECTOR_BYTES,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _launch,
        source,
        norm_weight,
        output,
        stream,
        shard_offset,
        bool(use_pdl),
        options="--enable-tvm-ffi",
    )


def _validate_tensor_contract(
    source: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if source.ndim != 2 or source.shape[1] != HIDDEN_SIZE:
        raise ValueError(f"source must have shape [M,{HIDDEN_SIZE}], got {source.shape}")
    if source.shape[0] <= 0:
        raise ValueError("source M must be positive")
    if norm_weight.shape != (HIDDEN_SIZE,):
        raise ValueError(f"norm_weight must have shape [{HIDDEN_SIZE}], got {norm_weight.shape}")
    expected_output = (source.shape[0], LOCAL_SIZE)
    if output.shape != expected_output:
        raise ValueError(f"output must have shape {expected_output}, got {output.shape}")
    for name, tensor in (
        ("source", source),
        ("norm_weight", norm_weight),
        ("output", output),
    ):
        if tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA BF16 tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride={tensor.stride()}")
    if source.device != norm_weight.device or source.device != output.device:
        raise ValueError("source, norm_weight, and output must share a CUDA device")
