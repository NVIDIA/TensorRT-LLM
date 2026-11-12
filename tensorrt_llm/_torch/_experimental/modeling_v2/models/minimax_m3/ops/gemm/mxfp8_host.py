# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MXFP8 host helpers; each target supplies its own tuning tables and caches."""

from __future__ import annotations

from typing import Callable, Collection, Optional

import torch


def gemv_mma_n(m: int, *, mma_widths: tuple[int, ...]) -> int:
    """Choose the smallest compiled MMA width covering the activation rows."""
    return next((w for w in mma_widths if m <= w))


def gemv_smem_bytes(ring: int, tiles: int, split: int, mma_n: int) -> int:
    """Compute the shared-memory bytes for the supplied launch geometry."""
    return (
        ring * (128 * 128 + 512)
        + tiles * (mma_n * 128 + 512)
        + split * (128 // split) * (mma_n + 4) * 4
        + 1024
    )


def gemv_dummy_timers(
    device: torch.device, *, dummy: dict[torch.device, torch.Tensor]
) -> torch.Tensor:
    """Placeholder for the kernel's timer buffer, which it only writes when compiled with timers on."""
    t = dummy.get(device)
    if t is None:
        t = dummy[device] = torch.zeros(8, dtype=torch.int64, device=device)
    return t


def gemv_arg(t: torch.Tensor, align: int = 16) -> object:
    """Expose a tensor with the launch alignment and dynamic leading layout."""
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=t.dim() - 1)


def gemv_get_config(
    n: int,
    k: int,
    m: Optional[int] = None,
    *,
    configs_by_m: dict[tuple[int, int, int], tuple],
    configs_wide: dict[tuple[int, int], tuple],
    configs: dict[tuple[int, int], tuple],
    strided: bool,
    strided_max_rows: int,
    strided_shapes: Collection[tuple[int, int]],
) -> Optional[tuple]:
    """Select a configuration from the target tables in their original order."""
    cfg = configs_by_m.get((n, k, m)) if m is not None else None
    if cfg is None and m is not None and (m > 8):
        cfg = configs_wide.get((n, k))
    cfg = cfg or configs.get((n, k))
    if (
        cfg is None
        or n % 128 != 0
        or k % 128 != 0
        or (k // 128 % cfg[0] != 0)
        or (128 % cfg[0] != 0)
    ):
        return None
    if (
        strided
        and m is not None
        and (m <= strided_max_rows)
        and ((n, k) in strided_shapes)
        and (len(cfg) == 2)
    ):
        cfg = cfg + (0,)
    return cfg


def gemv_declines(
    n: int,
    k: int,
    m: int,
    x_bf16: bool,
    cfg: tuple,
    *,
    max_tokens: int,
    smem_limit: int,
    select_mma_n: Callable[[int], int],
    smem_bytes: Callable[[int, int, int, int], int],
) -> bool:
    """Whether a (split, ring, ...) config declines an [M, K] x [N, K]^T call on shape grounds."""
    split = cfg[0]
    if not 0 < m <= max_tokens or 128 % split != 0 or k // 128 % split != 0:
        return True
    tiles = k // 128 // split
    mma_n = select_mma_n(m)
    if 16 < m <= 32 and k < 1024 or (x_bf16 and tiles * 4 * mma_n > 8 * 128):
        return True
    return smem_bytes(min(cfg[1], tiles, 2), tiles, split, mma_n) > smem_limit


def gemv_views(
    x: torch.Tensor,
    xsf: Optional[torch.Tensor],
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    y: torch.Tensor,
    x_bf16: bool,
    x_swiglu: bool = False,
    *,
    placeholder_scales: Callable[[int, torch.device], torch.Tensor],
    placeholder_words: Callable[[torch.device], torch.Tensor],
    arg: Callable[..., object],
    dummy_timers: Callable[[torch.device], torch.Tensor],
) -> tuple:
    """Build argument views using the target placeholder and timer caches."""
    k = x.shape[1] // 2 if x_swiglu else x.shape[1]
    if x_bf16:
        x_bytes = x.view(torch.uint8)
        xsf = placeholder_scales(k, x.device)
        x_words = x.view(torch.int32)
    else:
        x_bytes = x.view(torch.uint8)
        x_words = placeholder_words(x.device)
    return (
        arg(weight.view(torch.uint8)),
        arg(weight_scale.view(torch.uint8).reshape(-1)),
        arg(x_bytes),
        arg(xsf.view(torch.uint8).reshape(-1)),
        arg(y.view(-1) if y.numel() else placeholder_words(x.device).view(torch.bfloat16)),
        arg(x_words.reshape(-1)),
        arg(dummy_timers(x.device)),
    )


def gemv_extra_args(
    device: torch.device,
    *,
    placeholder_words: Callable[[torch.device], torch.Tensor],
    arg: Callable[..., object],
) -> tuple:
    """Placeholder arguments of the kernel's draft-input prologue and RoPE / KV epilogue (compiled off)."""
    i32 = placeholder_words(device)
    f32 = i32.view(torch.float32)
    u8 = i32.view(torch.uint8)
    d = (arg(i32), arg(i32), arg(i32), arg(i32), arg(f32), 0, arg(i32), arg(i32), 0)
    r = (arg(f32), arg(i32), arg(i32), arg(u8), arg(u8), arg(f32), arg(i32))
    return d + r


def gemv_placeholder_scales(
    k: int, device: torch.device, *, placeholders: dict[tuple, torch.Tensor]
) -> torch.Tensor:
    """Get the target cache's zeroed scale placeholder for this device and width."""
    key = ("sf", k, device)
    t = placeholders.get(key)
    if t is None:
        t = placeholders[key] = torch.zeros(k // 128 * 512, dtype=torch.uint8, device=device)
    return t


def gemv_placeholder_words(
    device: torch.device, *, placeholders: dict[tuple, torch.Tensor]
) -> torch.Tensor:
    """Get the target cache's zeroed word placeholder for this device."""
    key = ("w", device)
    t = placeholders.get(key)
    if t is None:
        t = placeholders[key] = torch.zeros(16, dtype=torch.int32, device=device)
    return t


def gemv_ensure_compiled(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    use_pdl: bool,
    *,
    get_config: Callable[..., Optional[tuple]],
    ensured: set[tuple],
    mma_widths: tuple[int, ...],
    mx_gemv: Callable[..., Optional[torch.Tensor]],
    mx_gemv_bf16: Callable[..., Optional[torch.Tensor]],
) -> None:
    """Compiles the kernel of this weight's shape outside capture (a one-token dummy call)."""
    if (
        weight.dim() != 2
        or weight.dtype != torch.float8_e4m3fn
        or get_config(*weight.shape) is None
    ):
        return
    n, k = weight.shape
    if (n, k, use_pdl) in ensured:
        return
    ensured.add((n, k, use_pdl))
    xq = torch.zeros(1, k, dtype=torch.float8_e4m3fn, device=weight.device)
    xsf = torch.full((128 * (k // 32),), 127, dtype=torch.uint8, device=weight.device)
    mx_gemv(xq, xsf, weight, weight_scale, use_pdl=use_pdl)
    for width in mma_widths[1:]:
        mx_gemv(
            torch.zeros(width, k, dtype=torch.float8_e4m3fn, device=weight.device),
            xsf,
            weight,
            weight_scale,
            use_pdl=use_pdl,
        )
    mx_gemv_bf16(
        torch.zeros(1, k, dtype=torch.bfloat16, device=weight.device),
        weight,
        weight_scale,
        use_pdl=use_pdl,
    )
