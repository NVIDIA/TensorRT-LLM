# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone CuTeDSL topk reduce kernel.

Form A writes one BF16 fc2 output row per ``(token, topk)`` cell into
``combine_output`` with logical shape ``(T, K, H)``.  This module provides the
device-side final reduce used by the default form-A path:

    BF16 (T, K, H) -> FP32 accumulate over K -> FP32/BF16 (T, H)

It also supports an explicit MXFP8 input mode:

    FP8_E4M3 (T, K, H) + UE8M0 scale -> FP32 dequant/reduce -> BF16 (T, H)

and an explicit NVFP4 input mode:

    FP4_E2M1 (T, K, H) + per-16 FP8 scale + per-128 FP32 scale
        -> FP32 dequant/reduce -> FP32/BF16 (T, H)

It intentionally does not touch dispatch metadata, peer pointer mapping, or
the fc2 epilogue STG path.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Float32, Int32

DEFAULT_THREADS = 256

BF16_VECTOR_THREADS = 512
BF16_HIDDEN_PER_THREAD = 8
BF16_STORE_ELEMENTS_PER_256B = 16

MXFP8_VECTOR_THREADS = 128
MXFP8_HIDDEN_PER_THREAD = 16
MXFP8_SCALE_BLOCK_SIZE = 32

NVFP4_VECTOR_THREADS = 128
NVFP4_HIDDEN_PER_THREAD = 32
NVFP4_SFC_SCALE_BLOCK_SIZE = 16
NVFP4_SFC_PACKED_BYTES = NVFP4_SFC_SCALE_BLOCK_SIZE // 2
NVFP4_SFC_INPUT_BITS_PER_COPY = NVFP4_SFC_PACKED_BYTES * 8
NVFP4_GLOBAL_SCALE_BLOCK_SIZE = 128

NVFP4_E2M1_MAX = 6.0
FP8_E4M3FN_MAX = 448.0

_Fp4DecodeTable: torch.Tensor = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

_Fp4ValuesEvenFirst: torch.Tensor = torch.tensor(
    [
        0.0,
        1.0,
        2.0,
        4.0,
        -0.0,
        -1.0,
        -2.0,
        -4.0,
        0.5,
        1.5,
        3.0,
        6.0,
        -0.5,
        -1.5,
        -3.0,
        -6.0,
    ],
    dtype=torch.float32,
)

_ReorderToNibble: torch.Tensor = torch.tensor(
    [
        0x0,
        0x2,
        0x4,
        0x6,
        0x8,
        0xA,
        0xC,
        0xE,
        0x1,
        0x3,
        0x5,
        0x7,
        0x9,
        0xB,
        0xD,
        0xF,
    ],
    dtype=torch.uint8,
)


def logical_io_bytes(
    combine_output: torch.Tensor,
    reduced_output: torch.Tensor,
    topk_score: Optional[torch.Tensor] = None,
    mxfp8_scale: Optional[torch.Tensor] = None,
    nvfp4_sfc_scale: Optional[torch.Tensor] = None,
    nvfp4_global_scale: Optional[torch.Tensor] = None,
) -> Tuple[int, int, int]:
    """Return logical read, write and total bytes for one topk reduce pass."""
    read_bytes = combine_output.numel() * combine_output.element_size()
    if topk_score is not None:
        read_bytes += topk_score.numel() * topk_score.element_size()
    if mxfp8_scale is not None:
        read_bytes += mxfp8_scale.numel() * mxfp8_scale.element_size()
    if nvfp4_sfc_scale is not None:
        read_bytes += nvfp4_sfc_scale.numel() * nvfp4_sfc_scale.element_size()
    if nvfp4_global_scale is not None:
        read_bytes += nvfp4_global_scale.numel(
        ) * nvfp4_global_scale.element_size()
    write_bytes = reduced_output.numel() * reduced_output.element_size()
    return int(read_bytes), int(write_bytes), int(read_bytes + write_bytes)


def bandwidth_gbps(num_bytes: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0.0:
        return float("inf")
    return float(num_bytes) / (elapsed_ms * 1.0e6)


def make_mxfp8_input(
    src: torch.Tensor,
    *,
    scale_rank: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 ``src`` to MXFP8 data plus UE8M0 dequant scale."""
    if src.dim() != 3:
        raise ValueError(
            f"src must have shape (T, K, H), got {tuple(src.shape)}.")
    if src.dtype != torch.float32:
        raise TypeError(f"src must be torch.float32, got {src.dtype}.")
    if not src.is_cuda:
        raise ValueError("src must be a CUDA tensor.")

    T, K, H = src.shape
    block = MXFP8_SCALE_BLOCK_SIZE
    scale_cols = (H + block - 1) // block
    padded_abs = torch.zeros(
        (T, K, scale_cols * block),
        device=src.device,
        dtype=torch.float32,
    )
    padded_abs[:, :, :H] = src.abs()
    amax = padded_abs.reshape(T, K, scale_cols, block).amax(dim=-1)
    if scale_rank == 2:
        scale_f32 = amax.amax(dim=1) / 448.0
        scale_for_q = scale_f32[:, None, :]
    elif scale_rank == 3:
        scale_f32 = amax / 448.0
        scale_for_q = scale_f32
    else:
        raise ValueError(f"scale_rank must be 2 or 3, got {scale_rank}.")

    def _round_up_to_power_of_two(scale: torch.Tensor) -> torch.Tensor:
        return torch.pow(
            torch.full_like(scale, 2.0),
            torch.ceil(torch.log2(torch.clamp(scale, min=2.0**-30))),
        )

    scale_f32 = _round_up_to_power_of_two(scale_f32)
    scale_for_q = _round_up_to_power_of_two(scale_for_q)
    expanded_scale = scale_for_q.repeat_interleave(block, dim=-1)[:, :, :H]
    q = (src / expanded_scale).to(torch.float8_e4m3fn)
    return q, scale_f32.to(torch.float8_e8m0fnu)


def _pack_f32_to_fp4(fp32: torch.Tensor) -> torch.Tensor:
    """Round FP32 to FP4 E2M1 and pack pairs along the last dimension."""
    if fp32.dim() == 0 or fp32.shape[-1] % 2 != 0:
        raise ValueError(
            f"FP4 packing requires an even non-empty last dim, got "
            f"{tuple(fp32.shape)}.")
    device = fp32.device
    boundaries = torch.tensor(
        [
            -5.0,
            -3.5,
            -2.5,
            -1.75,
            -1.25,
            -0.75,
            -0.25,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
        ],
        device=device,
        dtype=fp32.dtype,
    )
    bucket_to_nibble = torch.tensor(
        [
            0xF,
            0xE,
            0xD,
            0xC,
            0xB,
            0xA,
            0x9,
            0x0,
            0x1,
            0x2,
            0x3,
            0x4,
            0x5,
            0x6,
            0x7,
        ],
        device=device,
        dtype=torch.uint8,
    )
    bucket = torch.bucketize(fp32.contiguous(), boundaries)
    indices = bucket_to_nibble[bucket]
    lo = indices[..., 0::2]
    hi = indices[..., 1::2]
    return ((hi << 4) | lo).contiguous()


def unpack_fp4_to_f32(packed: torch.Tensor) -> torch.Tensor:
    """Unpack a last-dim-packed FP4 tensor or uint8 byte tensor to FP32."""
    if packed.dtype == torch.uint8:
        raw = packed
    elif hasattr(torch,
                 "float4_e2m1fn_x2") and packed.dtype == torch.float4_e2m1fn_x2:
        raw = packed.view(torch.uint8)
    else:
        raise TypeError(
            "packed must be torch.uint8 or torch.float4_e2m1fn_x2, got "
            f"{packed.dtype}.")
    lo = (raw & 0x0F).to(torch.int64)
    hi = (raw >> 4).to(torch.int64)
    lut = _Fp4DecodeTable.to(raw.device)
    unpacked_shape = list(raw.shape)
    unpacked_shape[-1] *= 2
    unpacked = torch.empty(unpacked_shape,
                           dtype=torch.float32,
                           device=raw.device)
    unpacked[..., 0::2] = lut[lo]
    unpacked[..., 1::2] = lut[hi]
    return unpacked


def make_nvfp4_input(
    src: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize FP32 ``src`` to NVFP4 plus per-16 FP8 and per-128 FP32 scales.

    The returned scales are dequant scales along hidden:
    ``x_hat = fp4 * sfc_fp8 * global_fp32``.
    """
    if not hasattr(torch, "float8_e4m3fn"):
        raise TypeError("NVFP4 mode requires torch float8_e4m3fn.")
    if src.dim() != 3:
        raise ValueError(
            f"src must have shape (T, K, H), got {tuple(src.shape)}.")
    if src.dtype != torch.float32:
        raise TypeError(f"src must be torch.float32, got {src.dtype}.")
    if not src.is_cuda:
        raise ValueError("src must be a CUDA tensor.")
    if src.shape[-1] % 2 != 0:
        raise ValueError(
            f"NVFP4 input hidden must be even for fp4x2 packing, got {src.shape[-1]}."
        )

    T, K, H = src.shape
    sfc_block = NVFP4_SFC_SCALE_BLOCK_SIZE
    global_block = NVFP4_GLOBAL_SCALE_BLOCK_SIZE
    sfc_cols = (H + sfc_block - 1) // sfc_block
    global_cols = (H + global_block - 1) // global_block

    padded_abs_sfc = torch.zeros(
        (T, K, sfc_cols * sfc_block),
        device=src.device,
        dtype=torch.float32,
    )
    padded_abs_sfc[:, :, :H] = src.abs()
    amax16 = padded_abs_sfc.reshape(T, K, sfc_cols, sfc_block).amax(dim=-1)

    padded_abs_global = torch.zeros(
        (T, K, global_cols * global_block),
        device=src.device,
        dtype=torch.float32,
    )
    padded_abs_global[:, :, :H] = src.abs()
    amax128 = padded_abs_global.reshape(T, K, global_cols,
                                        global_block).amax(dim=-1)

    global_scale = torch.clamp(
        amax128 / (NVFP4_E2M1_MAX * FP8_E4M3FN_MAX),
        min=2.0**-16,
    )
    global_for_sfc = global_scale.repeat_interleave(
        global_block // sfc_block,
        dim=-1,
    )[:, :, :sfc_cols]
    sfc_fp32 = amax16 / (NVFP4_E2M1_MAX * global_for_sfc)
    sfc_fp32 = torch.clamp(sfc_fp32, min=2.0**-16, max=FP8_E4M3FN_MAX)
    sfc_fp8 = sfc_fp32.to(torch.float8_e4m3fn)
    sfc_rt = sfc_fp8.to(torch.float32)

    expanded_sfc = sfc_rt.repeat_interleave(sfc_block, dim=-1)[:, :, :H]
    expanded_global = global_scale.repeat_interleave(global_block,
                                                     dim=-1)[:, :, :H]
    q = _pack_f32_to_fp4(src / (expanded_sfc * expanded_global))
    return q, sfc_fp8, global_scale


def mxfp8_reference_sum(
    q: torch.Tensor,
    scale: torch.Tensor,
    topk_score: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return K-ordered FP32 reduce of MXFP8 input after dequantization."""
    T, K, H = q.shape
    block = MXFP8_SCALE_BLOCK_SIZE
    if scale.dim() == 2:
        scale_for_q = scale.to(torch.float32)[:, None, :]
    else:
        scale_for_q = scale.to(torch.float32)
    expanded_scale = scale_for_q.repeat_interleave(block, dim=-1)[:, :, :H]
    dequant = q.to(torch.float32) * expanded_scale
    acc = torch.zeros((T, H), device=q.device, dtype=torch.float32)
    for k in range(K):
        contrib = dequant[:, k, :]
        if topk_score is not None:
            acc = torch.addcmul(acc, contrib, topk_score[:, k, None])
        else:
            acc = acc + contrib
    return acc


def nvfp4_reference_sum(
    q: torch.Tensor,
    sfc_scale: torch.Tensor,
    global_scale: torch.Tensor,
    topk_score: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return K-ordered FP32 reduce of hierarchical NVFP4 input."""
    unpacked = unpack_fp4_to_f32(q)
    T, K, H = unpacked.shape
    expanded_sfc = sfc_scale.to(torch.float32).repeat_interleave(
        NVFP4_SFC_SCALE_BLOCK_SIZE,
        dim=-1,
    )[:, :, :H]
    expanded_global = global_scale.to(torch.float32).repeat_interleave(
        NVFP4_GLOBAL_SCALE_BLOCK_SIZE,
        dim=-1,
    )[:, :, :H]
    acc = torch.zeros((T, H), device=q.device, dtype=torch.float32)
    for k in range(K):
        contrib = unpacked[:, k, :] * expanded_sfc[:, k, :]
        if topk_score is not None:
            contrib = contrib * expanded_global[:, k, :]
            acc = torch.addcmul(acc, contrib, topk_score[:, k, None])
        else:
            acc = torch.addcmul(acc, contrib, expanded_global[:, k, :])
    return acc


def weighted_reference_sum(
    src: torch.Tensor,
    topk_score: torch.Tensor,
) -> torch.Tensor:
    """Return K-ordered FP32 weighted reduce using FMA/addcmul semantics."""
    src_f32 = src.to(torch.float32)
    acc = torch.zeros(
        (src.shape[0], src.shape[2]),
        device=src.device,
        dtype=torch.float32,
    )
    for k in range(src.shape[1]):
        acc = torch.addcmul(acc, src_f32[:, k, :], topk_score[:, k, None])
    return acc


def ordered_reference_sum(src: torch.Tensor) -> torch.Tensor:
    """Return K-ordered FP32 reduce of BF16 input."""
    src_f32 = src.to(torch.float32)
    acc = torch.zeros(
        (src.shape[0], src.shape[2]),
        device=src.device,
        dtype=torch.float32,
    )
    for k in range(src.shape[1]):
        acc = acc + src_f32[:, k, :]
    return acc


@cute.jit
def _fp4_e2m1_nibble_to_f32(nibble: Int32) -> Float32:
    value = Float32(0.0)
    if nibble == Int32(1):
        value = Float32(0.5)
    elif nibble == Int32(2):
        value = Float32(1.0)
    elif nibble == Int32(3):
        value = Float32(1.5)
    elif nibble == Int32(4):
        value = Float32(2.0)
    elif nibble == Int32(5):
        value = Float32(3.0)
    elif nibble == Int32(6):
        value = Float32(4.0)
    elif nibble == Int32(7):
        value = Float32(6.0)
    elif nibble == Int32(9):
        value = Float32(-0.5)
    elif nibble == Int32(10):
        value = Float32(-1.0)
    elif nibble == Int32(11):
        value = Float32(-1.5)
    elif nibble == Int32(12):
        value = Float32(-2.0)
    elif nibble == Int32(13):
        value = Float32(-3.0)
    elif nibble == Int32(14):
        value = Float32(-4.0)
    elif nibble == Int32(15):
        value = Float32(-6.0)
    return value


@cute.kernel
def topk_reduce_bf16_vec_kernel(
    combine_output: cute.Tensor,
    topk_score: Optional[cute.Tensor],
    reduced_output: cute.Tensor,
    num_topk: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    store_dtype: cutlass.Constexpr[str],
):
    """BF16 reduce with one thread handling one 8-hidden vector."""

    hidden_vec_block_idx, token_idx, _ = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    block_dim = cute.arch.block_dim()[0]
    vec_idx = hidden_vec_block_idx * block_dim + tid
    base_h = vec_idx * Int32(BF16_HIDDEN_PER_THREAD)

    if base_h < Int32(hidden):
        acc = cute.make_rmem_tensor((BF16_HIDDEN_PER_THREAD, ), cutlass.Float32)
        for i in cutlass.range_constexpr(0, BF16_HIDDEN_PER_THREAD, 1):
            acc[i] = Float32(0.0)

        copy_atom_bf16_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )

        for k in cutlass.range_constexpr(0, num_topk, 1):
            score_value = Float32(1.0)
            if cutlass.const_expr(topk_score is not None):
                score_value = Float32(topk_score[token_idx, Int32(k)])
            score_pair = (score_value, score_value)

            in_regs = cute.make_rmem_tensor(
                (BF16_HIDDEN_PER_THREAD, ),
                cutlass.BFloat16,
            )
            in_row = combine_output[token_idx, Int32(k), None]
            in_tile = cute.local_tile(
                in_row,
                (BF16_HIDDEN_PER_THREAD, ),
                (base_h // Int32(BF16_HIDDEN_PER_THREAD), ),
            )
            in_aligned_iter = cute.make_ptr(
                in_tile.element_type,
                in_tile.iterator.toint(),
                AddressSpace.gmem,
                assumed_align=16,
            )
            in_tile = cute.make_tensor(in_aligned_iter, in_tile.layout)
            cute.copy(
                copy_atom_bf16_vec,
                cute.coalesce(in_tile),
                cute.coalesce(in_regs),
            )

            for pair_i in cutlass.range_constexpr(
                    0,
                    BF16_HIDDEN_PER_THREAD // 2,
                    1,
            ):
                val_pair = (
                    Float32(in_regs[2 * pair_i]),
                    Float32(in_regs[2 * pair_i + 1]),
                )
                old_acc_pair = (acc[2 * pair_i], acc[2 * pair_i + 1])
                if cutlass.const_expr(topk_score is not None):
                    acc_pair = cute.arch.fma_packed_f32x2(
                        val_pair,
                        score_pair,
                        old_acc_pair,
                    )
                else:
                    acc_pair = cute.arch.add_packed_f32x2(
                        old_acc_pair,
                        val_pair,
                    )
                acc[2 * pair_i] = acc_pair[0]
                acc[2 * pair_i + 1] = acc_pair[1]

        out_row = reduced_output[token_idx, None]
        out_tile = cute.local_tile(
            out_row,
            (BF16_HIDDEN_PER_THREAD, ),
            (base_h // Int32(BF16_HIDDEN_PER_THREAD), ),
        )
        if cutlass.const_expr(store_dtype == "bf16"):
            out_regs = cute.make_rmem_tensor(
                (BF16_HIDDEN_PER_THREAD, ),
                cutlass.BFloat16,
            )
            out_regs.store(acc.load().to(cutlass.BFloat16))
            out_aligned_iter = cute.make_ptr(
                out_tile.element_type,
                out_tile.iterator.toint(),
                AddressSpace.gmem,
                assumed_align=16,
            )
            out_tile = cute.make_tensor(out_aligned_iter, out_tile.layout)
            cute.copy(
                copy_atom_bf16_vec,
                cute.coalesce(out_regs),
                cute.coalesce(out_tile),
            )
        else:
            for i in cutlass.range_constexpr(0, BF16_HIDDEN_PER_THREAD, 1):
                out_tile[i] = acc[i]


@cute.kernel
def topk_reduce_mxfp8_vec_kernel(
    combine_output: cute.Tensor,
    topk_score: Optional[cute.Tensor],
    mxfp8_scale: cute.Tensor,
    reduced_output: cute.Tensor,
    num_topk: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    mxfp8_scale_rank: cutlass.Constexpr[int],
):
    """MXFP8 reduce with one thread handling one 16-hidden vector."""

    hidden_vec_block_idx, token_idx, _ = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    block_dim = cute.arch.block_dim()[0]
    vec_idx = hidden_vec_block_idx * block_dim + tid
    base_h = vec_idx * Int32(MXFP8_HIDDEN_PER_THREAD)

    if base_h < Int32(hidden):
        acc = cute.make_rmem_tensor((MXFP8_HIDDEN_PER_THREAD, ),
                                    cutlass.Float32)
        for i in cutlass.range_constexpr(0, MXFP8_HIDDEN_PER_THREAD, 1):
            acc[i] = Float32(0.0)

        copy_atom_ldg_128b = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float8E4M3FN,
            num_bits_per_copy=128,
        )
        copy_atom_stg_256b = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=256,
        )
        scale_col = base_h // Int32(MXFP8_SCALE_BLOCK_SIZE)

        for k in cutlass.range_constexpr(0, num_topk, 1):
            if cutlass.const_expr(mxfp8_scale_rank == 3):
                scale = Float32(mxfp8_scale[token_idx, Int32(k), scale_col])
            else:
                scale = Float32(mxfp8_scale[token_idx, scale_col])
            scale_pair = (scale, scale)
            score_value = Float32(1.0)
            if cutlass.const_expr(topk_score is not None):
                score_value = Float32(topk_score[token_idx, Int32(k)])
            score_pair = (score_value, score_value)

            in_regs = cute.make_rmem_tensor(
                (MXFP8_HIDDEN_PER_THREAD, ),
                cutlass.Float8E4M3FN,
            )
            in_row = combine_output[token_idx, Int32(k), None]
            in_tile = cute.local_tile(
                in_row,
                (MXFP8_HIDDEN_PER_THREAD, ),
                (base_h // Int32(MXFP8_HIDDEN_PER_THREAD), ),
            )
            in_aligned_iter = cute.make_ptr(
                in_tile.element_type,
                in_tile.iterator.toint(),
                AddressSpace.gmem,
                assumed_align=16,
            )
            in_tile = cute.make_tensor(in_aligned_iter, in_tile.layout)
            cute.copy(
                copy_atom_ldg_128b,
                cute.coalesce(in_tile),
                cute.coalesce(in_regs),
            )
            in_vals = cute.make_rmem_tensor(
                (MXFP8_HIDDEN_PER_THREAD, ),
                cutlass.Float32,
            )
            in_vals.store(in_regs.load().to(cutlass.Float32))

            for pair_i in cutlass.range_constexpr(
                    0,
                    MXFP8_HIDDEN_PER_THREAD // 2,
                    1,
            ):
                val_pair = (
                    in_vals[2 * pair_i],
                    in_vals[2 * pair_i + 1],
                )
                old_acc_pair = (acc[2 * pair_i], acc[2 * pair_i + 1])
                if cutlass.const_expr(topk_score is not None):
                    contrib_pair = cute.arch.mul_packed_f32x2(
                        val_pair,
                        scale_pair,
                    )
                    acc_pair = cute.arch.fma_packed_f32x2(
                        contrib_pair,
                        score_pair,
                        old_acc_pair,
                    )
                else:
                    acc_pair = cute.arch.fma_packed_f32x2(
                        val_pair,
                        scale_pair,
                        old_acc_pair,
                    )
                acc[2 * pair_i] = acc_pair[0]
                acc[2 * pair_i + 1] = acc_pair[1]

        out_row = reduced_output[token_idx, None]
        for chunk in cutlass.range_constexpr(
                0,
                MXFP8_HIDDEN_PER_THREAD // BF16_STORE_ELEMENTS_PER_256B,
                1,
        ):
            out_regs = cute.make_rmem_tensor(
                (BF16_STORE_ELEMENTS_PER_256B, ),
                cutlass.BFloat16,
            )
            for i in cutlass.range_constexpr(0, BF16_STORE_ELEMENTS_PER_256B,
                                             1):
                out_regs[i] = acc[chunk * BF16_STORE_ELEMENTS_PER_256B + i].to(
                    cutlass.BFloat16)
            out_h = base_h + Int32(chunk * BF16_STORE_ELEMENTS_PER_256B)
            out_tile = cute.local_tile(
                out_row,
                (BF16_STORE_ELEMENTS_PER_256B, ),
                (out_h // Int32(BF16_STORE_ELEMENTS_PER_256B), ),
            )
            out_aligned_iter = cute.make_ptr(
                out_tile.element_type,
                out_tile.iterator.toint(),
                AddressSpace.gmem,
                assumed_align=32,
            )
            out_tile = cute.make_tensor(out_aligned_iter, out_tile.layout)
            cute.copy(
                copy_atom_stg_256b,
                cute.coalesce(out_regs),
                cute.coalesce(out_tile),
            )


@cute.kernel
def topk_reduce_kernel(
    combine_output: cute.Tensor,
    topk_score: Optional[cute.Tensor],
    mxfp8_scale: Optional[cute.Tensor],
    nvfp4_sfc_scale: Optional[cute.Tensor],
    nvfp4_global_scale: Optional[cute.Tensor],
    reduced_output: cute.Tensor,
    num_topk: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    store_dtype: cutlass.Constexpr[str],
    mxfp8_scale_rank: cutlass.Constexpr[int],
):
    """Reduce ``combine_output[t, :, h]`` into ``reduced_output[t, h]``.

    In the default path, ``combine_output`` is BF16.  In MXFP8 mode,
    ``combine_output`` is FP8 E4M3 and ``mxfp8_scale`` is UE8M0 with either
    shape ``(T, ceil_div(H, 32))`` or ``(T, K, ceil_div(H, 32))``.  Optional
    ``topk_score`` is FP32 with shape ``(T, K)`` and scales each K
    contribution before accumulation.  Shapes and store dtype are supplied as
    constexprs by the launcher so the K loop is fully unrolled and matches the
    host reference order exactly.
    """

    hidden_block_idx, token_idx, _ = cute.arch.block_idx()

    h = hidden_block_idx * cute.arch.block_dim()[0] + cute.arch.thread_idx()[0]

    if h < Int32(hidden):
        acc = Float32(0.0)
        for k in cutlass.range_constexpr(0, num_topk, 1):
            if cutlass.const_expr(nvfp4_sfc_scale is not None):
                byte_col = h // Int32(2)
                shift = (h - byte_col * Int32(2)) * Int32(4)
                packed = Int32(combine_output[token_idx, Int32(k), byte_col])
                nibble = (packed >> shift) & Int32(0x0F)
                contrib = _fp4_e2m1_nibble_to_f32(nibble)
                sfc_col = h // Int32(NVFP4_SFC_SCALE_BLOCK_SIZE)
                global_col = h // Int32(NVFP4_GLOBAL_SCALE_BLOCK_SIZE)
                sfc = Float32(nvfp4_sfc_scale[token_idx, Int32(k), sfc_col])
                global_sf = Float32(nvfp4_global_scale[token_idx,
                                                       Int32(k), global_col])
                contrib = contrib * sfc * global_sf
            else:
                contrib = Float32(combine_output[token_idx, Int32(k), h])
                if cutlass.const_expr(mxfp8_scale is not None):
                    scale_col = h // Int32(MXFP8_SCALE_BLOCK_SIZE)
                    if cutlass.const_expr(mxfp8_scale_rank == 3):
                        scale = Float32(mxfp8_scale[token_idx,
                                                    Int32(k), scale_col])
                    else:
                        scale = Float32(mxfp8_scale[token_idx, scale_col])
                    contrib = contrib * scale
            if cutlass.const_expr(topk_score is not None):
                contrib = contrib * Float32(topk_score[token_idx, Int32(k)])
                acc = acc + contrib
            else:
                acc = acc + contrib
        if cutlass.const_expr(store_dtype == "bf16"):
            reduced_output[token_idx, h] = acc.to(cutlass.BFloat16)
        else:
            reduced_output[token_idx, h] = acc


@cute.kernel
def topk_reduce_nvfp4_vec_kernel(
    combine_output: cute.Tensor,
    topk_score: Optional[cute.Tensor],
    nvfp4_sfc_scale: cute.Tensor,
    nvfp4_global_scale: cute.Tensor,
    reduced_output: cute.Tensor,
    num_topk: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    store_dtype: cutlass.Constexpr[str],
):
    """NVFP4 reduce with one thread handling two per-16 hidden blocks."""

    hidden_vec_block_idx, token_idx, _ = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    block_dim = cute.arch.block_dim()[0]
    vec_idx = hidden_vec_block_idx * block_dim + tid
    base_h = vec_idx * Int32(NVFP4_HIDDEN_PER_THREAD)

    if base_h < Int32(hidden):
        sfc_col_base = base_h // Int32(NVFP4_SFC_SCALE_BLOCK_SIZE)
        global_col = base_h // Int32(NVFP4_GLOBAL_SCALE_BLOCK_SIZE)

        copy_atom_ldg_sfc = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Uint8,
            num_bits_per_copy=NVFP4_SFC_INPUT_BITS_PER_COPY,
        )
        copy_atom_stg_256b = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=256,
        )

        global_regs = cute.make_rmem_tensor((num_topk, ), cutlass.Float32)
        for k in cutlass.range_constexpr(0, num_topk, 1):
            global_regs[k] = Float32(nvfp4_global_scale[token_idx,
                                                        Int32(k), global_col])
        if cutlass.const_expr(topk_score is not None):
            score_regs = cute.make_rmem_tensor((num_topk, ), cutlass.Float32)
            for k in cutlass.range_constexpr(0, num_topk, 1):
                score_regs[k] = Float32(topk_score[token_idx, Int32(k)])

        out_row = reduced_output[token_idx, None]
        for sfc_block_i in cutlass.range_constexpr(
                0,
                NVFP4_HIDDEN_PER_THREAD // NVFP4_SFC_SCALE_BLOCK_SIZE,
                1,
        ):
            acc = cute.make_rmem_tensor(
                (NVFP4_SFC_SCALE_BLOCK_SIZE, ),
                cutlass.Float32,
            )
            for i in cutlass.range_constexpr(0, NVFP4_SFC_SCALE_BLOCK_SIZE, 1):
                acc[i] = Float32(0.0)

            sfc_base_h = base_h + Int32(
                sfc_block_i * NVFP4_SFC_SCALE_BLOCK_SIZE)
            for k in cutlass.range_constexpr(0, num_topk, 1):
                global_sf = global_regs[k]
                global_pair = (global_sf, global_sf)
                score_value = Float32(1.0)
                if cutlass.const_expr(topk_score is not None):
                    score_value = score_regs[k]
                score_pair = (score_value, score_value)

                q_bytes = cute.make_rmem_tensor(
                    (NVFP4_SFC_PACKED_BYTES, ),
                    cutlass.Uint8,
                )
                q_row = combine_output[token_idx, Int32(k), None]
                q_tile = cute.local_tile(
                    q_row,
                    (NVFP4_SFC_PACKED_BYTES, ),
                    (sfc_base_h // Int32(NVFP4_SFC_SCALE_BLOCK_SIZE), ),
                )
                q_aligned_iter = cute.make_ptr(
                    q_tile.element_type,
                    q_tile.iterator.toint(),
                    AddressSpace.gmem,
                    assumed_align=NVFP4_SFC_PACKED_BYTES,
                )
                q_tile = cute.make_tensor(q_aligned_iter, q_tile.layout)
                cute.copy(
                    copy_atom_ldg_sfc,
                    cute.coalesce(q_tile),
                    cute.coalesce(q_bytes),
                )
                q_fp4 = cute.recast_tensor(q_bytes, cutlass.Float4E2M1FN)
                q_vals = q_fp4.load().to(cutlass.Float32)
                sfc = Float32(nvfp4_sfc_scale[
                    token_idx,
                    Int32(k),
                    sfc_col_base + Int32(sfc_block_i),
                ])
                sfc_pair = (sfc, sfc)
                for byte_offset in cutlass.range_constexpr(
                        0,
                        NVFP4_SFC_SCALE_BLOCK_SIZE // 2,
                        1,
                ):
                    val_pair = (
                        q_vals[2 * byte_offset],
                        q_vals[2 * byte_offset + 1],
                    )
                    contrib_pair = cute.arch.mul_packed_f32x2(
                        val_pair, sfc_pair)
                    old_acc_pair = (
                        acc[2 * byte_offset],
                        acc[2 * byte_offset + 1],
                    )
                    if cutlass.const_expr(topk_score is not None):
                        contrib_pair = cute.arch.mul_packed_f32x2(
                            contrib_pair,
                            global_pair,
                        )
                        acc_pair = cute.arch.fma_packed_f32x2(
                            contrib_pair,
                            score_pair,
                            old_acc_pair,
                        )
                    else:
                        acc_pair = cute.arch.fma_packed_f32x2(
                            contrib_pair,
                            global_pair,
                            old_acc_pair,
                        )
                    acc[2 * byte_offset] = acc_pair[0]
                    acc[2 * byte_offset + 1] = acc_pair[1]

            if cutlass.const_expr(store_dtype == "bf16"):
                out_regs = cute.make_rmem_tensor(
                    (BF16_STORE_ELEMENTS_PER_256B, ),
                    cutlass.BFloat16,
                )
                for i in cutlass.range_constexpr(0,
                                                 BF16_STORE_ELEMENTS_PER_256B,
                                                 1):
                    out_regs[i] = acc[i].to(cutlass.BFloat16)
                out_tile = cute.local_tile(
                    out_row,
                    (BF16_STORE_ELEMENTS_PER_256B, ),
                    (sfc_base_h // Int32(BF16_STORE_ELEMENTS_PER_256B), ),
                )
                out_aligned_iter = cute.make_ptr(
                    out_tile.element_type,
                    out_tile.iterator.toint(),
                    AddressSpace.gmem,
                    assumed_align=32,
                )
                out_tile = cute.make_tensor(out_aligned_iter, out_tile.layout)
                cute.copy(
                    copy_atom_stg_256b,
                    cute.coalesce(out_regs),
                    cute.coalesce(out_tile),
                )
            else:
                out_tile = cute.local_tile(
                    out_row,
                    (NVFP4_SFC_SCALE_BLOCK_SIZE, ),
                    (sfc_base_h // Int32(NVFP4_SFC_SCALE_BLOCK_SIZE), ),
                )
                for i in cutlass.range_constexpr(0, NVFP4_SFC_SCALE_BLOCK_SIZE,
                                                 1):
                    out_tile[i] = acc[i]


def _validate_tensors(
    combine_output: torch.Tensor,
    reduced_output: torch.Tensor,
    topk_score: Optional[torch.Tensor] = None,
    mxfp8_scale: Optional[torch.Tensor] = None,
    nvfp4_sfc_scale: Optional[torch.Tensor] = None,
    nvfp4_global_scale: Optional[torch.Tensor] = None,
) -> Tuple[int, int, int, int]:
    if combine_output.dim() != 3:
        raise ValueError(f"combine_output must have shape (T, K, H), got "
                         f"{tuple(combine_output.shape)}.")
    if reduced_output.dim() != 2:
        raise ValueError(f"reduced_output must have shape (T, H), got "
                         f"{tuple(reduced_output.shape)}.")
    if reduced_output.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            f"reduced_output must be torch.float32 or torch.bfloat16, "
            f"got {reduced_output.dtype}.")
    if not combine_output.is_cuda or not reduced_output.is_cuda:
        raise ValueError(
            "combine_output and reduced_output must both be CUDA tensors.")
    if combine_output.device != reduced_output.device:
        raise ValueError(
            f"combine_output and reduced_output must be on the same device, got "
            f"{combine_output.device} and {reduced_output.device}.")

    if mxfp8_scale is not None and (nvfp4_sfc_scale is not None
                                    or nvfp4_global_scale is not None):
        raise ValueError("MXFP8 and NVFP4 modes are mutually exclusive.")
    if (nvfp4_sfc_scale is None) != (nvfp4_global_scale is None):
        raise ValueError(
            "nvfp4_sfc_scale and nvfp4_global_scale must be provided together.")

    T, K, H_storage = combine_output.shape
    if T <= 0 or K <= 0 or H_storage <= 0:
        raise ValueError(
            f"combine_output shape must have positive dimensions, got "
            f"{tuple(combine_output.shape)}.")

    nvfp4_mode = nvfp4_sfc_scale is not None
    H = int(H_storage) * 2 if nvfp4_mode else int(H_storage)
    if reduced_output.shape != (T, H):
        raise ValueError(f"reduced_output shape must be {(T, H)}, got "
                         f"{tuple(reduced_output.shape)}.")

    mxfp8_scale_rank = 0
    if mxfp8_scale is None and not nvfp4_mode:
        if combine_output.dtype != torch.bfloat16:
            raise TypeError(
                f"combine_output must be torch.bfloat16 unless mxfp8_scale is "
                f"or NVFP4 scales are provided, got {combine_output.dtype}.")
    elif mxfp8_scale is not None:
        if not hasattr(torch, "float8_e4m3fn") or not hasattr(
                torch, "float8_e8m0fnu"):
            raise TypeError(
                "MXFP8 mode requires torch float8_e4m3fn and float8_e8m0fnu.")
        if combine_output.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"MXFP8 combine_output must be torch.float8_e4m3fn, got "
                f"{combine_output.dtype}.")
        if mxfp8_scale.dtype != torch.float8_e8m0fnu:
            raise TypeError(f"mxfp8_scale must be torch.float8_e8m0fnu, got "
                            f"{mxfp8_scale.dtype}.")
        if reduced_output.dtype != torch.bfloat16:
            raise TypeError(f"MXFP8 reduced_output must be torch.bfloat16, got "
                            f"{reduced_output.dtype}.")
        if not mxfp8_scale.is_cuda:
            raise ValueError("mxfp8_scale must be a CUDA tensor.")
        if mxfp8_scale.device != combine_output.device:
            raise ValueError(
                f"mxfp8_scale must be on {combine_output.device}, got "
                f"{mxfp8_scale.device}.")
        scale_cols = (H + MXFP8_SCALE_BLOCK_SIZE - 1) // MXFP8_SCALE_BLOCK_SIZE
        if mxfp8_scale.dim() == 2:
            expected_scale_shape = (T, scale_cols)
        elif mxfp8_scale.dim() == 3:
            expected_scale_shape = (T, K, scale_cols)
        else:
            raise ValueError(
                "mxfp8_scale must have shape (T, ceil_div(H, 32)) or "
                f"(T, K, ceil_div(H, 32)), got {tuple(mxfp8_scale.shape)}.")
        if mxfp8_scale.shape != expected_scale_shape:
            raise ValueError(
                f"mxfp8_scale shape must be {expected_scale_shape}, got "
                f"{tuple(mxfp8_scale.shape)}.")
        mxfp8_scale_rank = mxfp8_scale.dim()
    else:
        if not hasattr(torch, "float8_e4m3fn"):
            raise TypeError("NVFP4 mode requires torch float8_e4m3fn.")
        if combine_output.dtype != torch.uint8:
            raise TypeError(
                f"NVFP4 combine_output must be packed torch.uint8, got "
                f"{combine_output.dtype}.")
        if nvfp4_sfc_scale.dtype != torch.float8_e4m3fn:
            raise TypeError(f"nvfp4_sfc_scale must be torch.float8_e4m3fn, got "
                            f"{nvfp4_sfc_scale.dtype}.")
        if nvfp4_global_scale.dtype != torch.float32:
            raise TypeError(f"nvfp4_global_scale must be torch.float32, got "
                            f"{nvfp4_global_scale.dtype}.")
        if not nvfp4_sfc_scale.is_cuda or not nvfp4_global_scale.is_cuda:
            raise ValueError("NVFP4 scales must be CUDA tensors.")
        if nvfp4_sfc_scale.device != combine_output.device:
            raise ValueError(
                f"nvfp4_sfc_scale must be on {combine_output.device}, got "
                f"{nvfp4_sfc_scale.device}.")
        if nvfp4_global_scale.device != combine_output.device:
            raise ValueError(
                f"nvfp4_global_scale must be on {combine_output.device}, got "
                f"{nvfp4_global_scale.device}.")
        sfc_cols = (H + NVFP4_SFC_SCALE_BLOCK_SIZE -
                    1) // NVFP4_SFC_SCALE_BLOCK_SIZE
        global_cols = (H + NVFP4_GLOBAL_SCALE_BLOCK_SIZE -
                       1) // NVFP4_GLOBAL_SCALE_BLOCK_SIZE
        expected_sfc_shape = (T, K, sfc_cols)
        expected_global_shape = (T, K, global_cols)
        if nvfp4_sfc_scale.dim(
        ) != 3 or nvfp4_sfc_scale.shape != expected_sfc_shape:
            raise ValueError(
                f"nvfp4_sfc_scale shape must be {expected_sfc_shape}, got "
                f"{tuple(nvfp4_sfc_scale.shape)}.")
        if (nvfp4_global_scale.dim() != 3
                or nvfp4_global_scale.shape != expected_global_shape):
            raise ValueError(
                f"nvfp4_global_scale shape must be {expected_global_shape}, got "
                f"{tuple(nvfp4_global_scale.shape)}.")
    if topk_score is not None:
        if topk_score.dim() != 2:
            raise ValueError(f"topk_score must have shape (T, K), got "
                             f"{tuple(topk_score.shape)}.")
        if topk_score.dtype != torch.float32:
            raise TypeError(
                f"topk_score must be torch.float32, got {topk_score.dtype}.")
        if not topk_score.is_cuda:
            raise ValueError("topk_score must be a CUDA tensor.")
        if topk_score.device != combine_output.device:
            raise ValueError(
                f"topk_score must be on {combine_output.device}, got "
                f"{topk_score.device}.")
        if topk_score.shape != (T, K):
            raise ValueError(f"topk_score shape must be {(T, K)}, got "
                             f"{tuple(topk_score.shape)}.")
    return int(T), int(K), int(H), int(mxfp8_scale_rank)


def _infer_assumed_align(tensor: torch.Tensor, max_align: int = 16) -> int:
    ptr = int(tensor.data_ptr())
    for align in (16, 8, 4, 2, 1):
        if align <= max_align and ptr % align == 0:
            return align
    return 1


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    assumed_align = _infer_assumed_align(tensor)
    cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    leading_dim = cutlass_torch.get_leading_dim(tensor)
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def compile_topk_reduce(
    combine_output: torch.Tensor,
    reduced_output: torch.Tensor,
    topk_score: Optional[torch.Tensor] = None,
    *,
    mxfp8_scale: Optional[torch.Tensor] = None,
    nvfp4_sfc_scale: Optional[torch.Tensor] = None,
    nvfp4_global_scale: Optional[torch.Tensor] = None,
    threads: Optional[int] = None,
    stream: Optional[cuda.CUstream] = None,
):
    """Compile a shape-specialized topk reduce launcher.

    The returned tuple is always ``(compiled, combine_cute, reduced_cute,
    topk_score_cute, mxfp8_scale_cute, nvfp4_sfc_scale_cute,
    nvfp4_global_scale_cute, stream)``.  Missing optional inputs are represented
    by ``None``.  Callers that only need a one-shot reduce should use
    :func:`run_topk_reduce`.
    """
    T, K, H, mxfp8_scale_rank = _validate_tensors(
        combine_output,
        reduced_output,
        topk_score,
        mxfp8_scale,
        nvfp4_sfc_scale,
        nvfp4_global_scale,
    )
    if threads is not None and threads <= 0:
        raise ValueError(f"threads must be positive, got {threads}.")
    if stream is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    store_dtype = "bf16" if reduced_output.dtype == torch.bfloat16 else "fp32"

    combine_cute = _to_cute_tensor(combine_output)
    reduced_cute = _to_cute_tensor(reduced_output)
    topk_score_cute = _to_cute_tensor(
        topk_score) if topk_score is not None else None
    mxfp8_scale_cute = _to_cute_tensor(
        mxfp8_scale) if mxfp8_scale is not None else None
    nvfp4_sfc_scale_cute = (_to_cute_tensor(nvfp4_sfc_scale)
                            if nvfp4_sfc_scale is not None else None)
    nvfp4_global_scale_cute = (_to_cute_tensor(nvfp4_global_scale)
                               if nvfp4_global_scale is not None else None)
    nvfp4_mode = nvfp4_sfc_scale is not None
    bf16_vectorized = (
        not nvfp4_mode and mxfp8_scale is None
        and combine_output.dtype == torch.bfloat16
        and H % BF16_HIDDEN_PER_THREAD == 0 and combine_output.stride(-1) == 1
        and combine_output.stride(-2) % BF16_HIDDEN_PER_THREAD == 0
        and reduced_output.stride(-1) == 1
        and (reduced_output.dtype != torch.bfloat16
             or reduced_output.stride(0) % BF16_HIDDEN_PER_THREAD == 0))
    mxfp8_vectorized = (
        mxfp8_scale is not None and not nvfp4_mode
        and H % MXFP8_HIDDEN_PER_THREAD == 0 and combine_output.stride(-1) == 1
        and combine_output.stride(-2) % MXFP8_HIDDEN_PER_THREAD == 0
        and reduced_output.dtype == torch.bfloat16
        and reduced_output.stride(-1) == 1
        and reduced_output.stride(0) % MXFP8_HIDDEN_PER_THREAD == 0)
    nvfp4_vectorized = (
        nvfp4_mode and H % NVFP4_HIDDEN_PER_THREAD == 0
        and combine_output.stride(-1) == 1
        and combine_output.stride(-2) % (NVFP4_HIDDEN_PER_THREAD // 2) == 0
        and reduced_output.stride(-1) == 1
        and (reduced_output.dtype != torch.bfloat16
             or reduced_output.stride(0) % NVFP4_HIDDEN_PER_THREAD == 0))
    if bf16_vectorized:
        hidden_per_thread = BF16_HIDDEN_PER_THREAD
    elif mxfp8_vectorized:
        hidden_per_thread = MXFP8_HIDDEN_PER_THREAD
    elif nvfp4_vectorized:
        hidden_per_thread = NVFP4_HIDDEN_PER_THREAD
    else:
        hidden_per_thread = 1
    if threads is None:
        if bf16_vectorized:
            launch_threads = BF16_VECTOR_THREADS
        elif mxfp8_vectorized:
            launch_threads = MXFP8_VECTOR_THREADS
        elif nvfp4_vectorized:
            launch_threads = NVFP4_VECTOR_THREADS
        else:
            launch_threads = DEFAULT_THREADS
    else:
        launch_threads = threads
    hidden_blocks = (H + launch_threads * hidden_per_thread -
                     1) // (launch_threads * hidden_per_thread)
    launch_grid = [hidden_blocks, T, 1]

    @cute.jit
    def _launcher(
        combine_cute: cute.Tensor,
        reduced_cute: cute.Tensor,
        topk_score_cute: Optional[cute.Tensor],
        mxfp8_scale_cute: Optional[cute.Tensor],
        nvfp4_sfc_scale_cute: Optional[cute.Tensor],
        nvfp4_global_scale_cute: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(bf16_vectorized):
            topk_reduce_bf16_vec_kernel(
                combine_cute,
                topk_score_cute,
                reduced_cute,
                num_topk=K,
                hidden=H,
                store_dtype=store_dtype,
            ).launch(
                grid=launch_grid,
                block=[launch_threads, 1, 1],
                stream=stream,
            )
        elif cutlass.const_expr(mxfp8_vectorized):
            topk_reduce_mxfp8_vec_kernel(
                combine_cute,
                topk_score_cute,
                mxfp8_scale_cute,
                reduced_cute,
                num_topk=K,
                hidden=H,
                mxfp8_scale_rank=mxfp8_scale_rank,
            ).launch(
                grid=launch_grid,
                block=[launch_threads, 1, 1],
                stream=stream,
            )
        elif cutlass.const_expr(nvfp4_vectorized):
            topk_reduce_nvfp4_vec_kernel(
                combine_cute,
                topk_score_cute,
                nvfp4_sfc_scale_cute,
                nvfp4_global_scale_cute,
                reduced_cute,
                num_topk=K,
                hidden=H,
                store_dtype=store_dtype,
            ).launch(
                grid=launch_grid,
                block=[launch_threads, 1, 1],
                stream=stream,
            )
        else:
            topk_reduce_kernel(
                combine_cute,
                topk_score_cute,
                mxfp8_scale_cute,
                nvfp4_sfc_scale_cute,
                nvfp4_global_scale_cute,
                reduced_cute,
                num_topk=K,
                hidden=H,
                store_dtype=store_dtype,
                mxfp8_scale_rank=mxfp8_scale_rank,
            ).launch(
                grid=launch_grid,
                block=[launch_threads, 1, 1],
                stream=stream,
            )

    compiled = cute.compile(
        _launcher,
        combine_cute,
        reduced_cute,
        topk_score_cute,
        mxfp8_scale_cute,
        nvfp4_sfc_scale_cute,
        nvfp4_global_scale_cute,
        stream,
    )
    return (
        compiled,
        combine_cute,
        reduced_cute,
        topk_score_cute,
        mxfp8_scale_cute,
        nvfp4_sfc_scale_cute,
        nvfp4_global_scale_cute,
        stream,
    )


def launch_compiled_topk_reduce(
    compiled,
    combine_cute: cute.Tensor,
    reduced_cute: cute.Tensor,
    topk_score_cute: Optional[cute.Tensor],
    mxfp8_scale_cute: Optional[cute.Tensor],
    nvfp4_sfc_scale_cute: Optional[cute.Tensor],
    nvfp4_global_scale_cute: Optional[cute.Tensor],
    stream: cuda.CUstream,
    *,
    synchronize: bool = False,
    return_elapsed_ms: bool = False,
) -> Optional[float]:
    """Launch a topk reduce plan returned by :func:`compile_topk_reduce`."""

    def _launch() -> None:
        compiled(
            combine_cute,
            reduced_cute,
            topk_score_cute,
            mxfp8_scale_cute,
            nvfp4_sfc_scale_cute,
            nvfp4_global_scale_cute,
            stream,
        )

    if return_elapsed_ms:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        _launch()
        stop.record()
        stop.synchronize()
        elapsed_ms = float(start.elapsed_time(stop))
    else:
        _launch()
        elapsed_ms = None

    if synchronize:
        torch.cuda.synchronize()
    return elapsed_ms


def run_topk_reduce(
    combine_output: torch.Tensor,
    reduced_output: torch.Tensor,
    topk_score: Optional[torch.Tensor] = None,
    *,
    mxfp8_scale: Optional[torch.Tensor] = None,
    nvfp4_sfc_scale: Optional[torch.Tensor] = None,
    nvfp4_global_scale: Optional[torch.Tensor] = None,
    threads: Optional[int] = None,
    stream: Optional[cuda.CUstream] = None,
    synchronize: bool = False,
    return_elapsed_ms: bool = False,
) -> Optional[float]:
    """Compile and launch the topk reduce kernel.

    Returns the measured kernel elapsed time in milliseconds when
    ``return_elapsed_ms`` is True, otherwise returns ``None``.
    """
    plan = compile_topk_reduce(
        combine_output,
        reduced_output,
        topk_score,
        mxfp8_scale=mxfp8_scale,
        nvfp4_sfc_scale=nvfp4_sfc_scale,
        nvfp4_global_scale=nvfp4_global_scale,
        threads=threads,
        stream=stream,
    )
    (
        compiled,
        combine_cute,
        reduced_cute,
        topk_score_cute,
        mxfp8_scale_cute,
        nvfp4_sfc_scale_cute,
        nvfp4_global_scale_cute,
        stream,
    ) = plan
    return launch_compiled_topk_reduce(
        compiled,
        combine_cute,
        reduced_cute,
        topk_score_cute,
        mxfp8_scale_cute,
        nvfp4_sfc_scale_cute,
        nvfp4_global_scale_cute,
        stream,
        synchronize=synchronize,
        return_elapsed_ms=return_elapsed_ms,
    )


def benchmark_topk_reduce_vs_torch_sum(
    *,
    tokens: int,
    topk: int,
    hidden: int,
    warmup: int = 5,
    iters: int = 50,
    output_dtype: torch.dtype = torch.float32,
    seed: int = 20260531,
    use_topk_score: bool = False,
    use_mxfp8: bool = False,
    use_nvfp4: bool = False,
    mxfp8_scale_rank: int = 3,
    threads: Optional[int] = None,
    print_result: bool = True,
) -> dict[str, float]:
    """Compare CuTeDSL topk_reduce against torch K-axis sum.

    The torch baseline intentionally uses the runner/reference expression:
    ``combine_output_ref.to(torch.float32).sum(dim=1)`` when ``topk_score``
    is absent, or the weighted equivalent when present.  MXFP8 and NVFP4
    inputs are converted from FP32 into their quantized data plus scale tensors
    before both benchmark paths.  CuTeDSL compile time is excluded from the
    measured kernel time.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for topk_reduce benchmark.")
    if output_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"output_dtype must be FP32 or BF16, got {output_dtype}.")
    if use_mxfp8 and use_nvfp4:
        raise ValueError(
            "MXFP8 and NVFP4 benchmark modes are mutually exclusive.")
    if use_mxfp8 and output_dtype != torch.bfloat16:
        raise ValueError("MXFP8 benchmark requires BF16 output.")
    if threads is None:
        if use_nvfp4:
            threads = NVFP4_VECTOR_THREADS
        elif use_mxfp8:
            threads = MXFP8_VECTOR_THREADS
        else:
            threads = BF16_VECTOR_THREADS

    torch.manual_seed(seed)
    combine_output_fp32 = torch.randn(
        (tokens, topk, hidden),
        device="cuda",
        dtype=torch.float32,
    )
    if use_mxfp8:
        combine_output_ref, mxfp8_scale = make_mxfp8_input(
            combine_output_fp32,
            scale_rank=mxfp8_scale_rank,
        )
        nvfp4_sfc_scale = None
        nvfp4_global_scale = None
        input_dtype_name = "mxfp8"
    elif use_nvfp4:
        combine_output_ref, nvfp4_sfc_scale, nvfp4_global_scale = make_nvfp4_input(
            combine_output_fp32, )
        mxfp8_scale = None
        input_dtype_name = "nvfp4"
    else:
        combine_output_ref = combine_output_fp32.to(torch.bfloat16)
        mxfp8_scale = None
        nvfp4_sfc_scale = None
        nvfp4_global_scale = None
        input_dtype_name = "bf16"
    topk_output = torch.empty(
        (tokens, hidden),
        device="cuda",
        dtype=output_dtype,
    )
    topk_score = None
    if use_topk_score:
        topk_score = torch.rand((tokens, topk),
                                device="cuda",
                                dtype=torch.float32)

    if print_result:
        print(
            "compiling topk_reduce "
            f"shape={(tokens, topk, hidden)} output_dtype={output_dtype} "
            f"input_dtype={input_dtype_name} "
            f"mxfp8_scale_rank={mxfp8_scale_rank if use_mxfp8 else 'none'} "
            f"topk_score={'on' if topk_score is not None else 'off'} "
            f"threads={threads}",
            flush=True,
        )
    (
        compiled,
        combine_cute,
        reduced_cute,
        topk_score_cute,
        mxfp8_scale_cute,
        nvfp4_sfc_scale_cute,
        nvfp4_global_scale_cute,
        stream,
    ) = compile_topk_reduce(
        combine_output_ref,
        topk_output,
        topk_score,
        mxfp8_scale=mxfp8_scale,
        nvfp4_sfc_scale=nvfp4_sfc_scale,
        nvfp4_global_scale=nvfp4_global_scale,
        threads=threads,
    )

    compiled(
        combine_cute,
        reduced_cute,
        topk_score_cute,
        mxfp8_scale_cute,
        nvfp4_sfc_scale_cute,
        nvfp4_global_scale_cute,
        stream,
    )
    torch.cuda.synchronize()

    def reference_result(*, timed_baseline: bool) -> torch.Tensor:
        if use_mxfp8:
            return mxfp8_reference_sum(
                combine_output_ref,
                mxfp8_scale,
                topk_score,
            ).to(output_dtype)
        if use_nvfp4:
            return nvfp4_reference_sum(
                combine_output_ref,
                nvfp4_sfc_scale,
                nvfp4_global_scale,
                topk_score,
            ).to(output_dtype)
        if topk_score is None:
            if output_dtype == torch.bfloat16 and not timed_baseline:
                return ordered_reference_sum(combine_output_ref).to(
                    output_dtype)
            return combine_output_ref.to(
                torch.float32).sum(dim=1).to(output_dtype)
        return weighted_reference_sum(combine_output_ref,
                                      topk_score).to(output_dtype)

    expected_result = reference_result(timed_baseline=False)
    torch.testing.assert_close(topk_output,
                               expected_result,
                               atol=1e-5,
                               rtol=1e-5)

    def measure_cuda_ms(fn) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        return float(start.elapsed_time(stop)) / float(iters)

    def run_compiled_topk_reduce() -> None:
        compiled(
            combine_cute,
            reduced_cute,
            topk_score_cute,
            mxfp8_scale_cute,
            nvfp4_sfc_scale_cute,
            nvfp4_global_scale_cute,
            stream,
        )

    torch_result = None

    def run_torch_sum() -> None:
        nonlocal torch_result
        torch_result = reference_result(timed_baseline=True)

    topk_ms = measure_cuda_ms(run_compiled_topk_reduce)
    torch_ms = measure_cuda_ms(run_torch_sum)
    assert torch_result is not None
    speedup = torch_ms / topk_ms
    read_bytes, write_bytes, total_bytes = logical_io_bytes(
        combine_output_ref,
        topk_output,
        topk_score,
        mxfp8_scale,
        nvfp4_sfc_scale,
        nvfp4_global_scale,
    )
    topk_bw = bandwidth_gbps(total_bytes, topk_ms)
    torch_bw = bandwidth_gbps(total_bytes, torch_ms)

    if print_result:
        print("topk_reduce_vs_torch_sum "
              f"shape={(tokens, topk, hidden)} output_dtype={output_dtype} "
              f"input_dtype={input_dtype_name} "
              f"mxfp8_scale_rank={mxfp8_scale_rank if use_mxfp8 else 'none'} "
              f"topk_score={'on' if topk_score is not None else 'off'} "
              f"threads={threads} "
              f"warmup={warmup} iters={iters} "
              f"topk_reduce_ms={topk_ms:.6f} "
              f"torch_sum_ms={torch_ms:.6f} "
              f"speedup_vs_torch={speedup:.3f}x "
              f"read_gb={read_bytes / 1.0e9:.6f} "
              f"write_gb={write_bytes / 1.0e9:.6f} "
              f"topk_reduce_bw_gbps={topk_bw:.3f} "
              f"torch_sum_bw_gbps={torch_bw:.3f}")

    return {
        "topk_reduce_ms": topk_ms,
        "torch_sum_ms": torch_ms,
        "speedup_vs_torch": speedup,
        "read_bytes": float(read_bytes),
        "write_bytes": float(write_bytes),
        "total_bytes": float(total_bytes),
        "topk_reduce_bw_gbps": topk_bw,
        "torch_sum_bw_gbps": torch_bw,
        "use_mxfp8": float(use_mxfp8),
        "use_nvfp4": float(use_nvfp4),
        "threads": float(threads),
    }


def _parse_bench_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Benchmark CuTeDSL topk_reduce against "
                     "combine_output_ref.to(torch.float32).sum(dim=1)."))
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output_dtype",
                        choices=["fp32", "bf16"],
                        default="bf16")
    parser.add_argument("--use_topk_score", action="store_true")
    parser.add_argument("--use_mxfp8", action="store_true")
    parser.add_argument("--use_nvfp4", action="store_true")
    parser.add_argument("--mxfp8_scale_rank",
                        type=int,
                        choices=[2, 3],
                        default=3)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260531)
    return parser.parse_args()


def main() -> int:
    args = _parse_bench_args()
    output_dtype = torch.bfloat16 if args.output_dtype == "bf16" else torch.float32
    benchmark_topk_reduce_vs_torch_sum(
        tokens=args.tokens,
        topk=args.topk,
        hidden=args.hidden,
        warmup=args.warmup,
        iters=args.iters,
        output_dtype=output_dtype,
        use_topk_score=args.use_topk_score,
        use_mxfp8=args.use_mxfp8,
        use_nvfp4=args.use_nvfp4,
        mxfp8_scale_rank=args.mxfp8_scale_rank,
        threads=args.threads,
        seed=args.seed,
    )
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
