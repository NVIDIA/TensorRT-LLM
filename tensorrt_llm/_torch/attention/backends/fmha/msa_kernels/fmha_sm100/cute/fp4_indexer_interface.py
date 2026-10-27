# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Public FP4 sparse-attention indexer block-score interface."""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import make_ptr

from src.sm100.fp4_indexer import (
    Fp4FormatSpec,
    Fp4IndexerDecodePackedQSm100,
    Fp4IndexerDecodeQPackSm100,
    Fp4IndexerScaleReorderSm100,
    Fp4IndexerStagedMmaSm100,
    _BLOCK_K,
    _DECODE_K_TILES_PER_CTA,
    _DECODE_PACK_Q_LEN,
    _DECODE_QHEAD_PER_KV,
    _FP4_PACKED_D_BYTES,
    _HEAD_DIM,
    _MMA_TILER_MN,
    _PAGE_SIZE,
    ceil_div,
    k_tiles_per_cta_for,
    normalize_fp4_format,
)


_PUBLIC_SCALE_LAYOUT = "public"
_PREORDERED_MMA_SCALE_LAYOUT = "preordered_mma"
_FP4_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


def _device_arch(device: torch.device) -> tuple[int, int]:
    major, minor = torch.cuda.get_device_capability(device)
    return int(major), int(minor)


def _supports_tmem_load_red(device_arch: tuple[int, int]) -> bool:
    return device_arch >= (10, 3)


def normalize_scale_layout(scale_layout: str) -> str:
    """Normalize and validate FP4 indexer scale layout mode.

    Parameters
    ----------
    scale_layout : str
        Either ``"public"`` for logical scale tensors or ``"preordered_mma"``
        for tensors already laid out with ``fp4_indexer_mma_scale_storage_*``.

    Returns
    -------
    str
        The normalized scale layout string.
    """

    scale_layout = str(scale_layout)
    if scale_layout not in (_PUBLIC_SCALE_LAYOUT, _PREORDERED_MMA_SCALE_LAYOUT):
        raise ValueError(f"scale_layout must be 'public' or 'preordered_mma', got {scale_layout!r}")
    return scale_layout


def _causal_compact_task_count(q_len: int, k_len: int, k_tiles_per_cta: int) -> int:
    if q_len <= 0 or k_len <= 0:
        return 0
    q_tile_count = ceil_div(q_len, _MMA_TILER_MN[0])
    k_group_count = ceil_div(ceil_div(k_len, _PAGE_SIZE), k_tiles_per_cta)
    group_tokens = k_tiles_per_cta * _BLOCK_K
    causal_offset = int(k_len) - int(q_len)
    tasks = 0
    for q_tile_idx in range(q_tile_count):
        q_tile_start = q_tile_idx * _MMA_TILER_MN[0]
        q_tile_last = min(q_tile_start + _MMA_TILER_MN[0] - 1, int(q_len) - 1)
        visible_limit = q_tile_last + causal_offset
        if visible_limit >= 0:
            tasks += min(k_group_count, visible_limit // group_tokens + 1)
    return tasks


def _causal_compact_task_bound(max_q_len: int, max_k_len: int, k_tiles_per_cta: int) -> int:
    """Conservative X-grid bound for per-batch causal prefill compact mapping."""

    if max_q_len <= 0 or max_k_len <= 0:
        return 0
    q_tile_count = ceil_div(max_q_len, _MMA_TILER_MN[0])
    candidates = {int(max_q_len)}
    for q_tile_idx in range(q_tile_count):
        q_len = q_tile_idx * _MMA_TILER_MN[0] + 1
        if q_len <= max_q_len:
            candidates.add(q_len)
    return max(_causal_compact_task_count(q_len, max_k_len, k_tiles_per_cta) for q_len in candidates)


def _require_cuda_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_int32_vector(tensor: torch.Tensor, *, name: str, device: torch.device) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same CUDA device")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank-1")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_fp4_packed_dtype(tensor: torch.Tensor, *, name: str) -> None:
    fp4_x2_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    allowed = {torch.uint8, torch.int8}
    if fp4_x2_dtype is not None:
        allowed.add(fp4_x2_dtype)
    if tensor.dtype not in allowed:
        raise TypeError(f"{name} must use packed FP4 storage dtype, got {tensor.dtype}")


def _as_fp4_thd_bytes(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [total_q, Hq, 64]")
    if int(tensor.shape[-1]) != _FP4_PACKED_D_BYTES:
        raise ValueError(f"{name}.shape[-1] must be 64 packed bytes for D=128")
    _require_fp4_packed_dtype(tensor, name=name)
    if tensor.dtype == torch.uint8:
        return tensor
    return tensor.view(torch.uint8)


def _as_fp4_paged_hnd_bytes(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [total_pages, Hk, 128, 64]")
    if int(tensor.shape[-2]) != _PAGE_SIZE:
        raise ValueError(f"{name}.shape[-2] must be 128")
    if int(tensor.shape[-1]) != _FP4_PACKED_D_BYTES:
        raise ValueError(f"{name}.shape[-1] must be 64 packed bytes for D=128")
    _require_fp4_packed_dtype(tensor, name=name)
    if tensor.dtype == torch.uint8:
        return tensor
    return tensor.view(torch.uint8)


def validate_q_scale_thg(
    scale: torch.Tensor,
    *,
    name: str,
    fmt: Fp4FormatSpec,
    total_q: int,
    heads: int,
) -> None:
    """Validate public Q FP4 scale layout ``[total_q, Hq, G]``.

    Parameters
    ----------
    scale : torch.Tensor
        Logical Q scale tensor.
    name : str
        Name used in validation error messages.
    fmt : Fp4FormatSpec
        FP4 format specification from ``normalize_fp4_format``.
    total_q : int
        Total query token count.
    heads : int
        Number of Q heads.
    """

    expected = (int(total_q), int(heads), fmt.scale_groups)
    if tuple(scale.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(scale.shape)}")
    if scale.dtype != fmt.torch_scale_dtype:
        raise TypeError(f"{name} must have dtype {fmt.torch_scale_dtype}, got {scale.dtype}")
    if not scale.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def validate_k_scale_phsg(
    scale: torch.Tensor,
    *,
    name: str,
    fmt: Fp4FormatSpec,
    page_count: int,
    heads: int,
) -> None:
    """Validate public K FP4 scale layout ``[page_count, Hk, 128, G]``.

    Parameters
    ----------
    scale : torch.Tensor
        Logical K scale tensor.
    name : str
        Name used in validation error messages.
    fmt : Fp4FormatSpec
        FP4 format specification from ``normalize_fp4_format``.
    page_count : int
        Number of physical KV pages.
    heads : int
        Number of KV heads.
    """

    expected = (int(page_count), int(heads), _PAGE_SIZE, fmt.scale_groups)
    if tuple(scale.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(scale.shape)}")
    if scale.dtype != fmt.torch_scale_dtype:
        raise TypeError(f"{name} must have dtype {fmt.torch_scale_dtype}, got {scale.dtype}")
    if not scale.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def fp4_indexer_mma_scale_shape(mn: int, l: int, *, fp4_format: str) -> tuple[int, int, int, int, int, int]:
    """Return semantic MMA scale view shape ``(32,4,restM,4,restG,L)``."""

    spec = normalize_fp4_format(fp4_format)
    return (32, 4, ceil_div(mn, 128), 4, ceil_div(spec.scale_groups, 4), int(l))


def fp4_indexer_mma_scale_stride(mn: int, l: int, *, fp4_format: str) -> tuple[int, int, int, int, int, int]:
    """Return element strides for ``fp4_indexer_mma_scale_shape``."""

    spec = normalize_fp4_format(fp4_format)
    rest_m = ceil_div(mn, 128)
    rest_g = ceil_div(spec.scale_groups, 4)
    return (16, 4, 512 * rest_g, 1, 512, 512 * rest_m * rest_g)


def fp4_indexer_mma_scale_storage_shape(mn: int, l: int, *, fp4_format: str) -> tuple[int, int, int, int, int, int]:
    """Return contiguous storage shape for preordered MMA scales."""

    spec = normalize_fp4_format(fp4_format)
    return (int(l), ceil_div(mn, 128), ceil_div(spec.scale_groups, 4), 32, 4, 4)


def fp4_indexer_mma_scale_storage_stride(mn: int, l: int, *, fp4_format: str) -> tuple[int, int, int, int, int, int]:
    """Return element strides for ``fp4_indexer_mma_scale_storage_shape``."""

    spec = normalize_fp4_format(fp4_format)
    rest_m = ceil_div(mn, 128)
    rest_g = ceil_div(spec.scale_groups, 4)
    return (512 * rest_m * rest_g, 512 * rest_g, 512, 16, 4, 1)


def validate_mma_scale_storage(
    scale: torch.Tensor,
    *,
    name: str,
    fmt: Fp4FormatSpec,
    mn: int,
    l: int,
) -> None:
    """Validate preordered MMA scale storage expected by the FP4 indexer.

    Parameters
    ----------
    scale : torch.Tensor
        Tensor view whose shape/stride should match
        ``fp4_indexer_mma_scale_storage_shape`` and
        ``fp4_indexer_mma_scale_storage_stride``.
    name : str
        Name used in validation error messages.
    fmt : Fp4FormatSpec
        FP4 format specification from ``normalize_fp4_format``.
    mn : int
        Logical M/N extent of the scale domain.
    l : int
        Logical batch/head extent folded into the final layout dimension.
    """

    expected_shape = fp4_indexer_mma_scale_storage_shape(mn, l, fp4_format=fmt.name)
    expected_stride = fp4_indexer_mma_scale_storage_stride(mn, l, fp4_format=fmt.name)
    if tuple(scale.shape) != expected_shape:
        raise ValueError(f"{name} must have MMA storage shape {expected_shape}, got {tuple(scale.shape)}")
    if tuple(scale.stride()) != expected_stride:
        raise ValueError(f"{name} must have MMA storage stride {expected_stride}, got {tuple(scale.stride())}")
    if scale.dtype != fmt.torch_scale_dtype:
        raise TypeError(f"{name} must have dtype {fmt.torch_scale_dtype}, got {scale.dtype}")


def _empty_mma_scale_tensor(
    *,
    mn: int,
    l: int,
    spec: Fp4FormatSpec,
    device: torch.device,
) -> torch.Tensor:
    rest_m = ceil_div(mn, 128)
    rest_g = ceil_div(spec.scale_groups, 4)
    storage = torch.empty(
        (int(l), rest_m, rest_g, 32, 4, 4),
        dtype=spec.torch_scale_dtype,
        device=device,
    )
    return storage.permute(3, 4, 1, 5, 2, 0)


def _compile_fp4_scale_reorder_kernel(
    *,
    fmt: Fp4FormatSpec,
    q_scale_ptr: cute.Pointer,
    k_scale_ptr: cute.Pointer,
    q_scale_mma_ptr: cute.Pointer,
    k_scale_mma_ptr: cute.Pointer,
    problem_size: tuple,
    stream: cuda.CUstream,
):
    key = (
        "fp4_indexer_scale_reorder_sm100_1cta",
        fmt.name,
    )
    if key not in _FP4_COMPILE_CACHE:
        kernel = Fp4IndexerScaleReorderSm100(fmt=fmt.name)
        _FP4_COMPILE_CACHE[key] = cute.compile(
            kernel,
            q_scale_ptr,
            k_scale_ptr,
            q_scale_mma_ptr,
            k_scale_mma_ptr,
            problem_size,
            stream,
        )
    return _FP4_COMPILE_CACHE[key]


def fp4_indexer_reorder_scales_for_mma_cute(
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    *,
    fp4_format: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder public Q/K FP4 scales to MMA-friendly storage.

    Parameters
    ----------
    q_scale : torch.Tensor
        Public Q scale tensor with shape ``[total_q, Hq, G]``.
    k_scale : torch.Tensor
        Public K scale tensor with shape ``[page_count, Hk, 128, G]``.
    fp4_format : str
        ``"mxfp4"`` or ``"nvfp4"``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(q_scale_mma, k_scale_mma)`` views in the storage layout validated by
        ``validate_mma_scale_storage``.  These tensors can be passed to
        ``fp4_indexer_block_scores`` with ``scale_layout="preordered_mma"``.
    """

    spec = normalize_fp4_format(fp4_format)
    if q_scale.device != k_scale.device:
        raise ValueError("q_scale and k_scale must be on the same CUDA device")
    _require_cuda_tensor(q_scale, name="q_scale")
    _require_cuda_tensor(k_scale, name="k_scale")
    if q_scale.ndim != 3:
        raise ValueError(f"q_scale must have shape [total_q, Hq, G], got {tuple(q_scale.shape)}")
    if k_scale.ndim != 4:
        raise ValueError(f"k_scale must have shape [page_count, Hk, 128, G], got {tuple(k_scale.shape)}")
    total_q, heads_q, _ = (int(v) for v in q_scale.shape)
    page_count, heads_k, _, _ = (int(v) for v in k_scale.shape)
    validate_q_scale_thg(q_scale, name="q_scale", fmt=spec, total_q=total_q, heads=heads_q)
    validate_k_scale_phsg(k_scale, name="k_scale", fmt=spec, page_count=page_count, heads=heads_k)

    q_scale_mma = _empty_mma_scale_tensor(
        mn=total_q,
        l=heads_q,
        spec=spec,
        device=q_scale.device,
    )
    k_scale_mma = _empty_mma_scale_tensor(
        mn=_PAGE_SIZE,
        l=page_count * heads_k,
        spec=spec,
        device=k_scale.device,
    )

    q_scale_ptr = make_ptr(
        spec.cutlass_scale_dtype,
        q_scale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    k_scale_ptr = make_ptr(
        spec.cutlass_scale_dtype,
        k_scale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    q_scale_mma_ptr = make_ptr(
        spec.cutlass_scale_dtype,
        q_scale_mma.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    k_scale_mma_ptr = make_ptr(
        spec.cutlass_scale_dtype,
        k_scale_mma.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    problem_size = (
        Int32(total_q),
        Int32(heads_q),
        Int32(page_count),
        Int32(heads_k),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q_scale.device).cuda_stream)
    compiled = _compile_fp4_scale_reorder_kernel(
        fmt=spec,
        q_scale_ptr=q_scale_ptr,
        k_scale_ptr=k_scale_ptr,
        q_scale_mma_ptr=q_scale_mma_ptr,
        k_scale_mma_ptr=k_scale_mma_ptr,
        problem_size=problem_size,
        stream=stream,
    )
    compiled(
        q_scale_ptr,
        k_scale_ptr,
        q_scale_mma_ptr,
        k_scale_mma_ptr,
        problem_size,
        stream,
    )
    return q_scale_mma, k_scale_mma


def _compile_fp4_decode_q_pack_kernel(
    *,
    fmt: Fp4FormatSpec,
    q_ptr: cute.Pointer,
    q_scale_ptr: cute.Pointer,
    q_pack_ptr: cute.Pointer,
    q_scale_pack_ptr: cute.Pointer,
    cu_seqlens_q_ptr: cute.Pointer,
    problem_size: tuple,
    stream: cuda.CUstream,
):
    key = (
        "fp4_indexer_decode_q_pack_sm100",
        fmt.name,
    )
    if key not in _FP4_COMPILE_CACHE:
        kernel = Fp4IndexerDecodeQPackSm100(fmt=fmt.name)
        _FP4_COMPILE_CACHE[key] = cute.compile(
            kernel,
            q_ptr,
            q_scale_ptr,
            q_pack_ptr,
            q_scale_pack_ptr,
            cu_seqlens_q_ptr,
            problem_size,
            stream,
        )
    return _FP4_COMPILE_CACHE[key]


def _pack_decode_q_for_mma(
    q_bytes: torch.Tensor,
    q_scale_storage: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    fmt: Fp4FormatSpec,
    heads_q: int,
    heads_k: int,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_pack = torch.empty(
        (batch * heads_k, _PAGE_SIZE, _FP4_PACKED_D_BYTES),
        dtype=torch.uint8,
        device=q_bytes.device,
    )
    q_scale_pack = torch.empty(
        fp4_indexer_mma_scale_storage_shape(_PAGE_SIZE, batch * heads_k, fp4_format=fmt.name),
        dtype=fmt.torch_scale_dtype,
        device=q_bytes.device,
    )
    if q_pack.data_ptr() % 128 != 0:
        raise ValueError("internal decode q_pack data pointer must be 128B aligned for TMA")
    if q_scale_pack.data_ptr() % 32 != 0:
        raise ValueError("internal decode q_scale_pack data pointer must be 32B aligned")
    q_ptr = make_ptr(cutlass.Uint8, q_bytes.data_ptr(), cute.AddressSpace.gmem, assumed_align=128)
    q_scale_ptr = make_ptr(
        fmt.cutlass_scale_dtype,
        q_scale_storage.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    q_pack_ptr = make_ptr(cutlass.Uint8, q_pack.data_ptr(), cute.AddressSpace.gmem, assumed_align=128)
    q_scale_pack_ptr = make_ptr(
        fmt.cutlass_scale_dtype,
        q_scale_pack.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    cu_seqlens_q_ptr = make_ptr(
        cutlass.Int32,
        cu_seqlens_q.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    problem_size = (
        Int32(q_bytes.shape[0]),
        Int32(heads_q),
        Int32(heads_k),
        Int32(batch),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q_bytes.device).cuda_stream)
    compiled = _compile_fp4_decode_q_pack_kernel(
        fmt=fmt,
        q_ptr=q_ptr,
        q_scale_ptr=q_scale_ptr,
        q_pack_ptr=q_pack_ptr,
        q_scale_pack_ptr=q_scale_pack_ptr,
        cu_seqlens_q_ptr=cu_seqlens_q_ptr,
        problem_size=problem_size,
        stream=stream,
    )
    compiled(
        q_ptr,
        q_scale_ptr,
        q_pack_ptr,
        q_scale_pack_ptr,
        cu_seqlens_q_ptr,
        problem_size,
        stream,
    )
    return q_pack, q_scale_pack


def _compile_fp4_decode_packed_q_kernel(
    *,
    fmt: Fp4FormatSpec,
    causal: bool,
    compact_schedule: bool,
    device_arch: tuple[int, int],
    use_tmem_load_red: bool,
    q_pack_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    q_scale_pack_ptr: cute.Pointer,
    k_scale_ptr: cute.Pointer,
    scores_ptr: cute.Pointer,
    kv_indices_ptr: cute.Pointer,
    cu_seqlens_q_ptr: cute.Pointer,
    cu_seqlens_k_ptr: cute.Pointer,
    cu_page_offsets_ptr: cute.Pointer,
    qo_offset_ptr: cute.Pointer,
    problem_size: tuple,
    stream: cuda.CUstream,
):
    key = (
        "fp4_indexer_decode_packed_q_sm100",
        fmt.name,
        bool(causal),
        bool(compact_schedule),
        device_arch,
    )
    if key not in _FP4_COMPILE_CACHE:
        kernel = Fp4IndexerDecodePackedQSm100(
            fmt=fmt.name,
            causal=causal,
            compact_schedule=compact_schedule,
            use_tmem_load_red=use_tmem_load_red,
        )
        _FP4_COMPILE_CACHE[key] = cute.compile(
            kernel,
            q_pack_ptr,
            k_ptr,
            q_scale_pack_ptr,
            k_scale_ptr,
            scores_ptr,
            kv_indices_ptr,
            cu_seqlens_q_ptr,
            cu_seqlens_k_ptr,
            cu_page_offsets_ptr,
            qo_offset_ptr,
            problem_size,
            stream,
        )
    return _FP4_COMPILE_CACHE[key]


def _run_fp4_decode_packed_q_scores(
    q_pack: torch.Tensor,
    k_bytes: torch.Tensor,
    q_scale_pack: torch.Tensor,
    k_scale_storage: torch.Tensor,
    scores: torch.Tensor,
    kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_page_offsets: torch.Tensor,
    qo_offset_arg: torch.Tensor,
    *,
    fmt: Fp4FormatSpec,
    causal: bool,
    has_qo_offset: int,
    heads_q: int,
    heads_k: int,
    batch: int,
    max_k_tiles: int,
    total_q: int,
    device_arch: tuple[int, int],
    use_tmem_load_red: bool,
) -> None:
    page_count = int(k_bytes.shape[0])
    rectangular_groups = batch * ceil_div(max_k_tiles, _DECODE_K_TILES_PER_CTA)
    compact_groups = ceil_div(page_count + batch * (_DECODE_K_TILES_PER_CTA - 1), _DECODE_K_TILES_PER_CTA)
    compact_schedule = compact_groups < rectangular_groups
    if compact_schedule:
        scores.fill_(float("-inf"))

    q_pack_ptr = make_ptr(cutlass.Uint8, q_pack.data_ptr(), cute.AddressSpace.gmem, assumed_align=128)
    k_ptr = make_ptr(cutlass.Uint8, k_bytes.data_ptr(), cute.AddressSpace.gmem, assumed_align=128)
    q_scale_pack_ptr = make_ptr(
        fmt.cutlass_scale_dtype,
        q_scale_pack.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    k_scale_ptr = make_ptr(
        fmt.cutlass_scale_dtype,
        k_scale_storage.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    scores_ptr = make_ptr(cutlass.Float32, scores.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    kv_indices_ptr = make_ptr(cutlass.Int32, kv_indices.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    cu_seqlens_q_ptr = make_ptr(cutlass.Int32, cu_seqlens_q.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    cu_seqlens_k_ptr = make_ptr(cutlass.Int32, cu_seqlens_k.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    cu_page_offsets_ptr = make_ptr(cutlass.Int32, cu_page_offsets.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    qo_offset_ptr = make_ptr(cutlass.Int32, qo_offset_arg.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    problem_size = (
        Int32(_PAGE_SIZE),
        Int32(max_k_tiles * _PAGE_SIZE),
        Int32(_HEAD_DIM),
        Int32(batch * heads_k),
        Int32(page_count * heads_k),
        Int32(heads_q),
        Int32(heads_k),
        Int32(batch),
        Int32(max_k_tiles),
        Int32(total_q),
        Int32(has_qo_offset),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q_pack.device).cuda_stream)
    compiled = _compile_fp4_decode_packed_q_kernel(
        fmt=fmt,
        causal=causal,
        compact_schedule=compact_schedule,
        device_arch=device_arch,
        use_tmem_load_red=use_tmem_load_red,
        q_pack_ptr=q_pack_ptr,
        k_ptr=k_ptr,
        q_scale_pack_ptr=q_scale_pack_ptr,
        k_scale_ptr=k_scale_ptr,
        scores_ptr=scores_ptr,
        kv_indices_ptr=kv_indices_ptr,
        cu_seqlens_q_ptr=cu_seqlens_q_ptr,
        cu_seqlens_k_ptr=cu_seqlens_k_ptr,
        cu_page_offsets_ptr=cu_page_offsets_ptr,
        qo_offset_ptr=qo_offset_ptr,
        problem_size=problem_size,
        stream=stream,
    )
    compiled(
        q_pack_ptr,
        k_ptr,
        q_scale_pack_ptr,
        k_scale_ptr,
        scores_ptr,
        kv_indices_ptr,
        cu_seqlens_q_ptr,
        cu_seqlens_k_ptr,
        cu_page_offsets_ptr,
        qo_offset_ptr,
        problem_size,
        stream,
    )


def _compile_fp4_qk_kernel(
    *,
    fmt: Fp4FormatSpec,
    causal: bool,
    preordered_q_scale_tma: bool,
    compact_schedule: bool,
    device_arch: tuple[int, int],
    use_tmem_load_red: bool,
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    q_scale_ptr: cute.Pointer,
    k_scale_ptr: cute.Pointer,
    scores_ptr: cute.Pointer,
    kv_indices_ptr: cute.Pointer,
    cu_seqlens_q_ptr: cute.Pointer,
    cu_seqlens_k_ptr: cute.Pointer,
    cu_page_offsets_ptr: cute.Pointer,
    qo_offset_ptr: cute.Pointer,
    problem_size: tuple,
    stream: cuda.CUstream,
):
    key = (
        "fp4_indexer_staged_mma_sm100",
        fmt.name,
        bool(causal),
        bool(preordered_q_scale_tma),
        bool(compact_schedule),
        device_arch,
    )
    if key not in _FP4_COMPILE_CACHE:
        kernel = Fp4IndexerStagedMmaSm100(
            fmt=fmt.name,
            causal=causal,
            preordered_q_scale_tma=preordered_q_scale_tma,
            compact_schedule=compact_schedule,
            use_tmem_load_red=use_tmem_load_red,
        )
        _FP4_COMPILE_CACHE[key] = cute.compile(
            kernel,
            q_ptr,
            k_ptr,
            q_scale_ptr,
            k_scale_ptr,
            scores_ptr,
            kv_indices_ptr,
            cu_seqlens_q_ptr,
            cu_seqlens_k_ptr,
            cu_page_offsets_ptr,
            qo_offset_ptr,
            problem_size,
            stream,
        )
    return _FP4_COMPILE_CACHE[key]


def fp4_indexer_block_scores(
    q_fp4: torch.Tensor,
    k_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_page_offsets: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    kv_indices: torch.Tensor,
    fp4_format: str,
    causal: bool = False,
    qo_offset: Optional[torch.Tensor] = None,
    scale_layout: str = _PREORDERED_MMA_SCALE_LAYOUT,
) -> torch.Tensor:
    """Return FP4 QK max scores per 128-token KV page.

    Parameters
    ----------
    q_fp4 : torch.Tensor
        Packed FP4 Q tensor with shape ``[total_qo_len, Hq, 64]``.  The last
        dimension stores two FP4 values per byte for logical head dimension
        128.
    k_fp4 : torch.Tensor
        Packed paged FP4 K tensor with shape ``[total_pages, Hk, 128, 64]``.
    q_scale : torch.Tensor
        Q scale tensor.  With ``scale_layout="public"``, shape is
        ``[total_qo_len, Hq, G]``.  With ``"preordered_mma"``, use
        ``fp4_indexer_reorder_scales_for_mma_cute`` output layout.
    k_scale : torch.Tensor
        K scale tensor.  With ``scale_layout="public"``, shape is
        ``[total_pages, Hk, 128, G]``.  With ``"preordered_mma"``, use the
        preordered MMA scale layout.
    cu_seqlens_q : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of Q lengths.
    cu_seqlens_k : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of KV lengths.
    cu_page_offsets : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of per-request
        page counts.
    max_seqlen_q : int
        Maximum Q sequence length.
    max_seqlen_k : int
        Maximum KV sequence length.
    kv_indices : torch.Tensor
        Flattened physical page indices with shape ``[sum_pages]`` and dtype
        int32.
    fp4_format : str
        ``"mxfp4"`` or ``"nvfp4"``.
    causal : bool, optional
        Whether to apply causal masking.
    qo_offset : torch.Tensor, optional
        Shape ``[batch_size]``, dtype int32.  Per-request causal offset.  Valid
        only when ``causal=True``.
    scale_layout : str, optional
        ``"public"`` accepts logical public scale tensors and launches a scale
        reorder kernel.  ``"preordered_mma"`` expects preordered MMA scale
        tensors and skips the reorder.

    Returns
    -------
    torch.Tensor
        Shape ``[Hq, ceil(max_seqlen_k / 128), total_qo_len]``, dtype float32.
        Entries beyond the valid KV page range are ``-inf``.
    """

    spec = normalize_fp4_format(fp4_format)
    causal = bool(causal)
    scale_layout = normalize_scale_layout(scale_layout)
    use_preordered_q_scale_tma = int(max_seqlen_q) >= _PAGE_SIZE
    q_bytes = _as_fp4_thd_bytes(q_fp4, name="q_fp4")
    k_bytes = _as_fp4_paged_hnd_bytes(k_fp4, name="k_fp4")
    total_q, heads_q, _ = (int(v) for v in q_bytes.shape)
    page_count, heads_k, page_size, _ = (int(v) for v in k_bytes.shape)
    if page_size != _PAGE_SIZE:
        raise ValueError(f"k_fp4 page_size must be 128, got {page_size}")
    if heads_q % heads_k != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    _require_cuda_tensor(q_fp4, name="q_fp4")
    _require_cuda_tensor(k_fp4, name="k_fp4")
    device_arch = _device_arch(q_fp4.device)
    use_tmem_load_red = _supports_tmem_load_red(device_arch)
    _require_int32_vector(cu_seqlens_q, name="cu_seqlens_q", device=q_fp4.device)
    _require_int32_vector(cu_seqlens_k, name="cu_seqlens_k", device=q_fp4.device)
    _require_int32_vector(cu_page_offsets, name="cu_page_offsets", device=q_fp4.device)
    if q_scale.device != q_fp4.device or k_scale.device != q_fp4.device:
        raise ValueError("q_scale and k_scale must be on the same CUDA device as q_fp4")
    if scale_layout == _PUBLIC_SCALE_LAYOUT:
        validate_q_scale_thg(q_scale, name="q_scale", fmt=spec, total_q=total_q, heads=heads_q)
        validate_k_scale_phsg(k_scale, name="k_scale", fmt=spec, page_count=page_count, heads=heads_k)
    else:
        validate_mma_scale_storage(q_scale, name="q_scale", fmt=spec, mn=total_q, l=heads_q)
        validate_mma_scale_storage(k_scale, name="k_scale", fmt=spec, mn=_PAGE_SIZE, l=page_count * heads_k)
    batch = int(cu_seqlens_q.shape[0]) - 1
    if batch < 0:
        raise ValueError("cu_seqlens_q must have shape [B + 1]")
    if cu_seqlens_q.shape != cu_seqlens_k.shape or cu_seqlens_q.shape != cu_page_offsets.shape:
        raise ValueError("cu_seqlens_q, cu_seqlens_k, and cu_page_offsets must have shape [B + 1]")
    if q_bytes.data_ptr() % 128 != 0:
        raise ValueError("q_fp4 data pointer must be 128B aligned for TMA")
    if k_bytes.data_ptr() % 128 != 0:
        raise ValueError("k_fp4 data pointer must be 128B aligned for TMA")
    if kv_indices is None:
        raise ValueError("kv_indices is required")
    if kv_indices.device != q_fp4.device or kv_indices.dtype != torch.int32 or kv_indices.ndim != 1:
        raise ValueError("kv_indices must have shape [sum_pages], dtype torch.int32, and match q_fp4.device")
    if not kv_indices.is_contiguous():
        raise ValueError("kv_indices must be contiguous")
    if qo_offset is not None:
        if not causal:
            raise ValueError("qo_offset is only valid when causal=True")
        if qo_offset.device != q_fp4.device or qo_offset.dtype != torch.int32 or qo_offset.shape != (batch,):
            raise ValueError("qo_offset must have shape [B], dtype torch.int32, and match q_fp4.device")
        if not qo_offset.is_contiguous():
            raise ValueError("qo_offset must be contiguous")

    m_extent = int(max_seqlen_q)
    max_k_tiles = ceil_div(int(max_seqlen_k), _PAGE_SIZE)
    n_aligned = max_k_tiles * _PAGE_SIZE
    if max_k_tiles == 0:
        return torch.full(
            (heads_q, 0, total_q),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp4.device,
        )

    scores = torch.empty(
        (heads_q, max_k_tiles, total_q),
        dtype=torch.float32,
        device=q_fp4.device,
    )
    if qo_offset is None:
        qo_offset_arg = torch.empty((batch,), dtype=torch.int32, device=q_fp4.device)
        has_qo_offset = 0
    else:
        qo_offset_arg = qo_offset
        has_qo_offset = 1
    if scale_layout == _PUBLIC_SCALE_LAYOUT:
        q_scale_arg, k_scale_arg = fp4_indexer_reorder_scales_for_mma_cute(
            q_scale,
            k_scale,
            fp4_format=spec.name,
        )
    else:
        q_scale_arg = q_scale
        k_scale_arg = k_scale
    scale_assumed_align = 32
    if q_scale_arg.data_ptr() % scale_assumed_align != 0:
        raise ValueError(f"q_scale data pointer must be {scale_assumed_align}B aligned for MMA storage scale")
    if k_scale_arg.data_ptr() % scale_assumed_align != 0:
        raise ValueError(f"k_scale data pointer must be {scale_assumed_align}B aligned for MMA storage scale")
    use_decode_packed_q = int(max_seqlen_q) <= _DECODE_PACK_Q_LEN and heads_q // heads_k == _DECODE_QHEAD_PER_KV
    if use_decode_packed_q:
        q_pack, q_scale_pack = _pack_decode_q_for_mma(
            q_bytes,
            q_scale_arg,
            cu_seqlens_q,
            fmt=spec,
            heads_q=heads_q,
            heads_k=heads_k,
            batch=batch,
        )
        _run_fp4_decode_packed_q_scores(
            q_pack,
            k_bytes,
            q_scale_pack,
            k_scale_arg,
            scores,
            kv_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_page_offsets,
            qo_offset_arg,
            fmt=spec,
            causal=causal,
            has_qo_offset=has_qo_offset,
            heads_q=heads_q,
            heads_k=heads_k,
            batch=batch,
            max_k_tiles=max_k_tiles,
            total_q=total_q,
            device_arch=device_arch,
            use_tmem_load_red=use_tmem_load_red,
        )
        return scores
    prefill_compact_task_count = 0
    prefill_compact_schedule = False
    if causal and has_qo_offset == 0:
        k_tiles_per_cta = k_tiles_per_cta_for(causal)
        q_tile_count = ceil_div(m_extent, _MMA_TILER_MN[0])
        k_group_count = ceil_div(max_k_tiles, k_tiles_per_cta)
        rectangular_task_count = q_tile_count * k_group_count
        prefill_compact_task_count = min(
            rectangular_task_count,
            _causal_compact_task_bound(m_extent, int(max_seqlen_k), k_tiles_per_cta),
        )
        prefill_compact_schedule = prefill_compact_task_count * 20 <= rectangular_task_count * 19
        if prefill_compact_schedule:
            scores.fill_(float("-inf"))
    q_ptr = make_ptr(
        cutlass.Uint8,
        q_bytes.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=128,
    )
    k_ptr = make_ptr(
        cutlass.Uint8,
        k_bytes.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=128,
    )
    q_scale_ptr = make_ptr(
        spec.cutlass_scale_dtype,
        q_scale_arg.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=scale_assumed_align,
    )
    k_scale_ptr = make_ptr(
        spec.cutlass_scale_dtype,
        k_scale_arg.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=scale_assumed_align,
    )
    scores_ptr = make_ptr(
        cutlass.Float32,
        scores.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    kv_indices_ptr = make_ptr(
        cutlass.Int32,
        kv_indices.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    cu_seqlens_q_ptr = make_ptr(
        cutlass.Int32,
        cu_seqlens_q.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    cu_seqlens_k_ptr = make_ptr(
        cutlass.Int32,
        cu_seqlens_k.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    cu_page_offsets_ptr = make_ptr(
        cutlass.Int32,
        cu_page_offsets.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    qo_offset_ptr = make_ptr(
        cutlass.Int32,
        qo_offset_arg.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    problem_size = (
        Int32(m_extent),
        Int32(n_aligned),
        Int32(_HEAD_DIM),
        Int32(batch * heads_q),
        Int32(page_count * heads_k),
        Int32(heads_q),
        Int32(heads_k),
        Int32(batch),
        Int32(max_k_tiles),
        Int32(total_q),
        Int32(has_qo_offset),
        Int32(prefill_compact_task_count),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q_fp4.device).cuda_stream)
    compiled = _compile_fp4_qk_kernel(
        fmt=spec,
        causal=causal,
        preordered_q_scale_tma=use_preordered_q_scale_tma,
        compact_schedule=prefill_compact_schedule,
        device_arch=device_arch,
        use_tmem_load_red=use_tmem_load_red,
        q_ptr=q_ptr,
        k_ptr=k_ptr,
        q_scale_ptr=q_scale_ptr,
        k_scale_ptr=k_scale_ptr,
        scores_ptr=scores_ptr,
        kv_indices_ptr=kv_indices_ptr,
        cu_seqlens_q_ptr=cu_seqlens_q_ptr,
        cu_seqlens_k_ptr=cu_seqlens_k_ptr,
        cu_page_offsets_ptr=cu_page_offsets_ptr,
        qo_offset_ptr=qo_offset_ptr,
        problem_size=problem_size,
        stream=stream,
    )
    compiled(
        q_ptr,
        k_ptr,
        q_scale_ptr,
        k_scale_ptr,
        scores_ptr,
        kv_indices_ptr,
        cu_seqlens_q_ptr,
        cu_seqlens_k_ptr,
        cu_page_offsets_ptr,
        qo_offset_ptr,
        problem_size,
        stream,
    )
    return scores


__all__ = [
    "fp4_indexer_block_scores",
]
