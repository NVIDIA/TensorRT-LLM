# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Active-only NVFP4 weight dequant for MoE on SM<100 (used by
W4A16NVFP4CutlassFusedMoEMethod). Static shapes -> CUDA-graph capturable.

Pipeline: scatter active_mask[E] from routing -> static (E, N/BN, K/BK) grid
-> inactive blocks early-return, active blocks dequant their tile. Downstream
fused_moe only reads rows in token_selected_experts, so leaving inactive
rows uninitialized is safe.
"""

import torch
import triton
import triton.language as tl

from tensorrt_llm.quantization.utils.fp4_utils import pad_up

# E2M1 codebook (signed-magnitude nibble layout). Index 0b1000 nominally
# encodes "-0" and is treated as 0.0. Kept as a Python list so we can build
# the device tensor lazily on first use.
_E2M1_CODEBOOK = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]  # yapf: disable

# Per-device cache so we don't reallocate the table on every call.
_E2M1_CODEBOOK_CACHE: "dict[torch.device, torch.Tensor]" = {}


def _get_e2m1_codebook(device: torch.device) -> torch.Tensor:
    table = _E2M1_CODEBOOK_CACHE.get(device)
    if table is None:
        table = torch.tensor(_E2M1_CODEBOOK, dtype=torch.float32, device=device)
        _E2M1_CODEBOOK_CACHE[device] = table
    return table


def build_active_expert_mask(
    token_selected_experts: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Sync-free, CUDA-graph-safe active-expert mask via scalar ``scatter_``
    (the fancy-index form ``mask[ids] = 1`` is illegal under stream capture).

    Caller must pre-clamp ids into ``[0, num_experts)``.
    """
    mask = torch.zeros(num_experts, dtype=torch.uint8, device=token_selected_experts.device)
    mask.scatter_(0, token_selected_experts.reshape(-1).long(), 1)
    return mask


@triton.jit
def _dequant_nvfp4_active_kernel(
    # Inputs
    packed_weight_ptr,
    scale_ptr,
    weight_scale_2_ptr,
    active_mask_ptr,
    e2m1_table_ptr,  # [16] fp32 codebook
    # Output
    out_ptr,
    # Strides (element counts, not bytes)
    pw_stride_e,
    pw_stride_n,
    sc_stride_e,
    sc_stride_n,
    out_stride_e,
    out_stride_n,
    # Shapes (runtime)
    N,
    K,
    # Compile-time
    SF_VEC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-block dequant of one tile of one expert's weight."""
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # ---- Active-mask early exit ----
    is_active = tl.load(active_mask_ptr + pid_e)
    if is_active == 0:
        return

    # Output element offsets
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]
    n_mask = n_offs < N
    k_mask = k_offs < K
    full_mask = n_mask[:, None] & k_mask[None, :]  # [BLOCK_N, BLOCK_K]

    # ---- Load packed FP4 bytes ----
    # Each k position corresponds to byte k//2 with nibble shift (k%2)*4.
    # Adjacent k positions share a byte: redundant loads, but cache friendly.
    packed_idx = k_offs // 2  # [BLOCK_K]
    nibble_shift = (k_offs % 2) * 4
    w_offs = pid_e * pw_stride_e + n_offs[:, None] * pw_stride_n + packed_idx[None, :]
    packed = tl.load(packed_weight_ptr + w_offs, mask=full_mask, other=0).to(tl.int32)
    nibble = (packed >> nibble_shift[None, :]) & 0xF  # [BLOCK_N, BLOCK_K]

    # ---- E2M1 codebook lookup via gather ----
    # 16-element table lives in L1 / constant cache after first touch.
    # nibble has shape [BLOCK_N, BLOCK_K]; the load fans out per lane and
    # the hardware broadcasts duplicate indices.
    val = tl.load(e2m1_table_ptr + nibble)  # [BLOCK_N, BLOCK_K] fp32

    # ---- Per-block FP8 (e4m3) scale ----
    # Each sf_vec_size consecutive k positions share one scale byte.
    sf_idx = k_offs // SF_VEC  # [BLOCK_K]
    s_offs = pid_e * sc_stride_e + n_offs[:, None] * sc_stride_n + sf_idx[None, :]
    scale_byte = tl.load(scale_ptr + s_offs, mask=full_mask, other=0)
    # Reinterpret uint8 bits as fp8 e4m3, then convert to fp32.
    scale_fp = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    # ---- Per-tensor scale (scalar per expert) ----
    s2 = tl.load(weight_scale_2_ptr + pid_e)

    # ---- Combine and store ----
    out_val = val * scale_fp * s2
    o_offs = pid_e * out_stride_e + n_offs[:, None] * out_stride_n + k_offs[None, :]
    tl.store(out_ptr + o_offs, out_val.to(out_ptr.dtype.element_ty), mask=full_mask)


def dequant_nvfp4_active_triton(
    packed_weight: torch.Tensor,
    scale_linear: torch.Tensor,
    weight_scale_2: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    target_dtype: torch.dtype = torch.bfloat16,
    sf_vec_size: int = 16,
    block_n: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    """Triton-based active-only NVFP4 weight dequant.

    Args:
        packed_weight: ``[E, N, K_packed]`` uint8 -- FP4 nibbles, two per byte.
        scale_linear: ``[E, N_pad, K_sf_pad]`` uint8 -- per-block FP8 (e4m3)
            scale in **linear** (un-swizzled) layout. The kernel uses tensor
            strides directly, so padding on N/K_sf is fine as long as the
            stride math points to the right elements.
        weight_scale_2: ``[E]`` float32 -- per-tensor scale.
        active_mask: ``[E]`` uint8 -- 1 for experts that need dequanting.
        target_dtype: ``torch.bfloat16`` or ``torch.float16``.
        sf_vec_size: NVFP4 per-block scale vector size (fixed at 16).
        block_n, block_k: Triton tile shape. ``block_k`` should be a
            multiple of ``sf_vec_size`` so each tile covers an integer
            number of scale blocks.

    Returns:
        ``[E, N, K]`` (with ``K = K_packed * 2``) in ``target_dtype``.
        Tiles belonging to inactive experts are left uninitialized; they
        are never read by the downstream MoE kernel.
    """
    assert packed_weight.dim() == 3, "packed_weight must be 3D [E, N, K/2]"
    assert sf_vec_size == 16, "NVFP4 fixed at 16-element blocks"
    assert block_k % sf_vec_size == 0, (
        f"block_k={block_k} must be a multiple of sf_vec_size={sf_vec_size}"
    )

    E, N, K_packed = packed_weight.shape
    K = K_packed * 2
    device = packed_weight.device

    assert packed_weight.stride(-1) == 1, (
        "packed_weight innermost stride must be 1 (contiguous K dim)"
    )
    assert scale_linear.dim() == 3, f"scale_linear must be 3D, got {tuple(scale_linear.shape)}"
    assert scale_linear.stride(-1) == 1, (
        "scale_linear innermost stride must be 1 (contiguous K_sf dim)"
    )
    assert scale_linear.shape[0] == E, (
        f"scale_linear E mismatch: {scale_linear.shape[0]} vs packed_weight E={E}"
    )
    assert active_mask.dim() == 1 and active_mask.is_contiguous() and active_mask.shape[0] == E, (
        f"active_mask must be 1D contiguous of length E={E}, got "
        f"shape={tuple(active_mask.shape)} contiguous={active_mask.is_contiguous()}"
    )
    assert (
        weight_scale_2.dim() == 1
        and weight_scale_2.is_contiguous()
        and weight_scale_2.shape[0] == E
    ), (
        f"weight_scale_2 must be 1D contiguous of length E={E}, got "
        f"shape={tuple(weight_scale_2.shape)} contiguous={weight_scale_2.is_contiguous()}"
    )

    if active_mask.dtype != torch.uint8:
        active_mask = active_mask.to(torch.uint8)

    out = torch.empty(E, N, K, dtype=target_dtype, device=device)
    e2m1_table = _get_e2m1_codebook(device)

    grid = (E, triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    _dequant_nvfp4_active_kernel[grid](
        packed_weight,
        scale_linear,
        weight_scale_2,
        active_mask,
        e2m1_table,
        out,
        # strides (in elements)
        packed_weight.stride(0),
        packed_weight.stride(1),
        scale_linear.stride(0),
        scale_linear.stride(1),
        out.stride(0),
        out.stride(1),
        # shapes
        N,
        K,
        # constexpr
        SF_VEC=sf_vec_size,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


@triton.jit
def _dequant_nvfp4_linear_kernel(
    # Inputs
    packed_weight_ptr,
    scale_ptr,
    weight_scale_2_ptr,  # pointer to one fp32 scalar (per-tensor)
    e2m1_table_ptr,
    # Output
    out_ptr,
    # Strides (in elements)
    pw_stride_n,
    sc_stride_n,
    out_stride_n,
    # Shapes (runtime)
    N,
    K,
    # Compile-time
    SF_VEC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-block dequant of one tile of a single 2D weight matrix.

    Dedicated 2D path for ``NVFP4LinearMethod``-style weights (one matrix,
    one per-tensor scale, no expert dim or active mask). Codebook gather and
    FP8 e4m3 -> fp32 conversion match the MoE kernel.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = n_offs < N
    k_mask = k_offs < K
    full_mask = n_mask[:, None] & k_mask[None, :]

    # ---- Load packed FP4 bytes ----
    packed_idx = k_offs // 2
    nibble_shift = (k_offs % 2) * 4
    w_offs = n_offs[:, None] * pw_stride_n + packed_idx[None, :]
    packed = tl.load(packed_weight_ptr + w_offs, mask=full_mask, other=0).to(tl.int32)
    nibble = (packed >> nibble_shift[None, :]) & 0xF

    # ---- E2M1 codebook gather ----
    val = tl.load(e2m1_table_ptr + nibble)

    # ---- Per-block FP8 (e4m3) scale ----
    sf_idx = k_offs // SF_VEC
    s_offs = n_offs[:, None] * sc_stride_n + sf_idx[None, :]
    scale_byte = tl.load(scale_ptr + s_offs, mask=full_mask, other=0)
    scale_fp = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    # ---- Per-tensor scale (single scalar, broadcast to the tile) ----
    s2 = tl.load(weight_scale_2_ptr)

    out_val = val * scale_fp * s2
    o_offs = n_offs[:, None] * out_stride_n + k_offs[None, :]
    tl.store(out_ptr + o_offs, out_val.to(out_ptr.dtype.element_ty), mask=full_mask)


def dequant_nvfp4_2d_triton(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    *,
    target_dtype: torch.dtype = torch.bfloat16,
    sf_vec_size: int = 16,
    block_n: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    """2D (Linear) NVFP4 dequant via a dedicated Triton kernel.

    Distinct from the MoE 3D path: no expert dim, no active mask, the
    per-tensor scale is a single scalar.

    Args:
        packed_weight: ``[N, K_packed]`` uint8 -- FP4 nibbles, two per byte.
        weight_scale: per-block FP8 (e4m3) scale in **linear** (un-swizzled)
            layout. Accepted as either:

            * a flat 1-D buffer of length ``pad_up(N, 128) * pad_up(K/sf, 4)``
              (what ``NVFP4LinearMethod.create_weights`` allocates), or
            * a 2-D buffer of shape ``[pad_rows, pad_cols]``.
        weight_scale_2: per-tensor FP32 scale (any shape with a single
            element; only ``data_ptr()`` is consumed by the kernel).
        target_dtype: BF16 or FP16.
        sf_vec_size: NVFP4 per-block size (16).
        block_n, block_k: Triton tile shape. ``block_k`` must be a multiple
            of ``sf_vec_size``.

    Returns:
        ``[N, K]`` in ``target_dtype``.
    """
    assert packed_weight.dim() == 2, "packed_weight must be 2D [N, K/2]"
    assert sf_vec_size == 16, "NVFP4 fixed at 16-element blocks"
    assert block_k % sf_vec_size == 0, (
        f"block_k={block_k} must be a multiple of sf_vec_size={sf_vec_size}"
    )

    N, K_packed = packed_weight.shape
    K = K_packed * 2
    device = packed_weight.device

    assert packed_weight.stride(-1) == 1, (
        "packed_weight innermost stride must be 1 (contiguous K dim)"
    )

    # Reshape (possibly flat) scale to its 2D [pad_rows, pad_cols] form so
    # the kernel can use ``scale.stride(0)`` directly.
    if weight_scale.dim() == 1:
        pad_rows = pad_up(N, 128)
        pad_cols = pad_up(K // sf_vec_size, 4)
        weight_scale = weight_scale.view(pad_rows, pad_cols)
    elif weight_scale.dim() != 2:
        raise ValueError(f"weight_scale must be 1D or 2D, got shape {tuple(weight_scale.shape)}")
    assert weight_scale.stride(-1) == 1, (
        "weight_scale innermost stride must be 1 (contiguous K_sf dim)"
    )

    out = torch.empty(N, K, dtype=target_dtype, device=device)
    e2m1_table = _get_e2m1_codebook(device)

    # The kernel does ``tl.load(weight_scale_2_ptr)`` -- a single scalar read.
    # Reject multi-element buffers explicitly: any trailing values would be
    # silently dropped and the matrix would dequantize against the wrong scale.
    assert weight_scale_2.numel() == 1, (
        f"weight_scale_2 must have exactly 1 element, got {weight_scale_2.numel()}"
    )
    weight_scale_2 = weight_scale_2.reshape(-1)

    grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    _dequant_nvfp4_linear_kernel[grid](
        packed_weight,
        weight_scale,
        weight_scale_2,
        e2m1_table,
        out,
        packed_weight.stride(0),
        weight_scale.stride(0),
        out.stride(0),
        N,
        K,
        SF_VEC=sf_vec_size,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out
