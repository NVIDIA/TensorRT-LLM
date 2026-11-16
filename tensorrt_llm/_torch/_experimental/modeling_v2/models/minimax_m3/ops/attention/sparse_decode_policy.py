# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Vendored from vLLM (Apache-2.0):
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/vllm/models/minimax_m3/common/ops/sparse_attn.py
"""Shared sparse-decode policies and buffer type; target instances remain isolated."""

from typing import Optional

import torch
import triton

_TARGET_GRID = 256


class MergeMxfp8:
    """MXFP8 copy of the split-K merge output for the o_proj GEMM.

    From ``MERGE_MXFP8_MIN_TOKENS`` to ``MAX_ROWS`` rows the merge kernel writes the E4M3 rows and
    swizzled UE8M0 scales next to the BF16 output and publishes them to ``m3_prequant.SLOT``, so the MXFP8 o_proj skips
    its own quantization. One buffer of ``MAX_ROWS`` rows (one 128-row scale tile) per width serves every layer: each
    o_proj consumes it before the next layer's merge rewrites it, in stream order.
    """

    MAX_ROWS = 128  # capacity: fixed: the merge hand-off's rows (decode steps)

    def __init__(self):
        self._buffers = {}

    def buffers(self, width: int, device: torch.device):
        key = (width, device)
        if key not in self._buffers:
            self._buffers[key] = (
                torch.empty((self.MAX_ROWS, width), dtype=torch.float8_e4m3fn, device=device),
                torch.zeros((self.MAX_ROWS * (width // 32),), dtype=torch.uint8, device=device),
            )
        return self._buffers[key]

    @staticmethod
    def handoff(mxfp8, rows: int):
        """The hand-off of a ``rows``-row step: its E4M3 rows and the 128-row scale tiles that cover them. The o_proj
        GEMM checks the swizzled scales against the step's padded rows, so the step gets the tiles that cover its rows,
        never the whole buffer."""
        q8, sf = mxfp8
        width = q8.shape[-1]
        return q8[:rows], sf[: (rows + 127) // 128 * 128 * (width // 32)]


def resolve_num_topk_chunks(total_q: int, num_kv_heads: int, max_topk: int) -> int:
    """Split-K factor over the top-k blocks, as a power of two.

    Depends only on shapes that are fixed for a captured batch size, so the
    launch geometry is frozen inside a CUDA graph.
    """
    target = max(1, min(max_topk, _TARGET_GRID // max(1, total_q * num_kv_heads)))
    return 1 << (target.bit_length() - 1)


_NVFP4_TUNED_CAPABILITIES = ((10, 3),)


def _sm100f_nvfp4_num_topk_chunks(
    *,
    total_q: int,
    num_kv_heads: int,
    gqa_group_size: int,
    max_topk: int,
    decode_query_len: int,
    capability: Optional[tuple[int, int]] = None,
) -> Optional[int]:
    """Select GB300-measured split-K factors for M3's SM100-family decode shape."""
    if capability is None:
        capability = torch.cuda.get_device_capability()
    if (
        capability not in _NVFP4_TUNED_CAPABILITIES
        or num_kv_heads != 4
        or gqa_group_size != 16
        or max_topk != 16
        or decode_query_len != 4
        or total_q % decode_query_len != 0
    ):
        return None
    local_batch = total_q // decode_query_len
    return {
        2: 8,
        3: 8,
        4: 8,
        5: 16,
        6: 4,
        7: 4,
        8: 2,
        9: 2,
        10: 8,
        11: 4,
        12: 2,
        13: 2,
        14: 2,
    }.get(local_batch)


def _sm100f_nvfp4_use_linear_softmax(
    *,
    total_q: int,
    num_kv_heads: int,
    gqa_group_size: int,
    max_topk: int,
    decode_query_len: int,
    capability: Optional[tuple[int, int]] = None,
) -> bool:
    """Use the GB300-measured lower-register recurrence on validated M3 shapes."""
    if capability is None:
        capability = torch.cuda.get_device_capability()
    if (
        capability not in _NVFP4_TUNED_CAPABILITIES
        or num_kv_heads not in (1, 2, 4)
        or gqa_group_size != 16
        or max_topk != 16
        or decode_query_len != 4
        or total_q % decode_query_len != 0
    ):
        return False
    local_batch = total_q // decode_query_len
    # The TP4/DEP32 single-request graph is 8.9% faster with the original
    # log-domain recurrence; every other measured DEP8/16/32 bucket is neutral
    # or faster with the linear denominator.
    return not (num_kv_heads == 1 and local_batch == 1)


def _sm100f_nvfp4_query_group_size(
    *,
    total_q: int,
    num_kv_heads: int,
    gqa_group_size: int,
    max_topk: int,
    decode_query_len: int,
    capability: Optional[tuple[int, int]] = None,
) -> int:
    """Group Eagle queries using the GB300-measured shared-page policy."""
    if capability is None:
        capability = torch.cuda.get_device_capability()
    if (
        capability not in _NVFP4_TUNED_CAPABILITIES
        or num_kv_heads != 4
        or gqa_group_size != 16
        or max_topk != 16
        or decode_query_len != 4
        or total_q % decode_query_len != 0
    ):
        return 1
    # The sorted-union kernel is 6.5%, 1.6%, and 24.2% faster at local batches
    # 11, 12, and 14 with split-K 8. Other measured buckets are neutral or
    # slower and retain the single-query producer.
    return 2 if total_q // decode_query_len in (11, 12, 14) else 1


def _sm100f_nvfp4_launch_options(
    *,
    total_q: int,
    num_kv_heads: int,
    gqa_group_size: int,
    max_topk: int,
    decode_query_len: int,
    capability: Optional[tuple[int, int]] = None,
) -> dict[str, int]:
    """Select native-NVFP4 launch geometry for the SM100 family.

    The FP16 QK/PV path is register-sensitive. GB300 dense replay shows four
    warps/two stages wins through local batch 28, while two warps/one stage
    avoids the next residency cliff at local batches 29--32. Leave every
    unmeasured architecture, model geometry, and larger batch on Triton's
    established defaults.
    """
    if capability is None:
        capability = torch.cuda.get_device_capability()
    if (
        capability not in _NVFP4_TUNED_CAPABILITIES
        or num_kv_heads != 4
        or gqa_group_size != 16
        or max_topk != 16
        or decode_query_len != 4
    ):
        return {}
    local_batch = total_q // decode_query_len
    if local_batch == 1 or local_batch in (2, 3, 5, 6, 7, 10, 11, 12):
        return {"num_warps": 4, "num_stages": 1}
    if local_batch == 4:
        return {"num_warps": 2, "num_stages": 1}
    if 8 <= local_batch <= 28:
        return {"num_warps": 4, "num_stages": 2}
    if 29 <= local_batch <= 32:
        return {"num_warps": 2, "num_stages": 1}
    return {}


def _sm100f_nvfp4_merge_launch_options(
    *,
    total_q: int,
    num_kv_heads: int,
    gqa_group_size: int,
    max_topk: int,
    decode_query_len: int,
    capability: Optional[tuple[int, int]] = None,
) -> dict[str, int]:
    """Select GB300-measured one-warp merges for M3's SM100-family graphs."""
    if capability is None:
        capability = torch.cuda.get_device_capability()
    if (
        capability in _NVFP4_TUNED_CAPABILITIES
        and num_kv_heads == 4
        and gqa_group_size == 16
        and max_topk == 16
        and decode_query_len == 4
        and total_q % decode_query_len == 0
        and total_q // decode_query_len in (2, 4, 5, 6, 7, 10, 11)
    ):
        return {"num_warps": 1}
    return {}


def _check_nvfp4_inputs(
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    nvfp4_args: tuple[Optional[torch.Tensor], ...],
    *,
    head_dim: int,
    scale_cols: int,
) -> None:
    """Validate the packed-NVFP4 cache against what the kernel assumes.

    The kernel indexes block scales by a flat within-page offset and reshapes a
    [., SCALE_COLS] scale tile up to head_dim, so it needs contiguous scale rows
    and head_dim exactly BLOCK_SIZE_D. Both hold for MiniMax-M3; check them
    rather than let a mismatch read neighboring pages.
    """
    k_block_scale, v_block_scale, k_global_scale, v_global_scale = nvfp4_args
    if any(arg is None for arg in nvfp4_args):
        raise ValueError(
            "MiniMax-M3 NVFP4 sparse decode needs all four of k_block_scale, "
            "v_block_scale, k_global_scale and v_global_scale."
        )
    if head_dim != triton.next_power_of_2(head_dim):
        raise ValueError(f"NVFP4 sparse decode requires a power-of-two head_dim; got {head_dim}.")
    if v_paged.shape[:3] != k_paged.shape[:3]:
        raise ValueError("NVFP4 K and V caches must have matching page, head and token extents.")
    for name, data in (("k_paged", k_paged), ("v_paged", v_paged)):
        if int(data.shape[3]) * 2 != head_dim:
            raise ValueError(
                f"NVFP4 {name} must pack two elements per byte: expected "
                f"head_dim/2 ({head_dim // 2}) columns, got {int(data.shape[3])}."
            )
    for name, scale in (("k_block_scale", k_block_scale), ("v_block_scale", v_block_scale)):
        if tuple(scale.shape) != tuple(k_paged.shape[:3]) + (scale_cols,):
            raise ValueError(
                f"NVFP4 {name} must be [num_pages, num_kv_heads, page_size, "
                f"{scale_cols}] to match the cache; got {tuple(scale.shape)}."
            )
        if scale.stride(3) != 1 or scale.stride(2) != scale_cols:
            raise ValueError(
                f"NVFP4 {name} needs contiguous per-page scale bytes; got strides "
                f"{tuple(scale.stride())}."
            )
    for name, gs in (("k_global_scale", k_global_scale), ("v_global_scale", v_global_scale)):
        if gs.dtype != torch.float32 or gs.numel() != 1:
            raise ValueError(
                f"NVFP4 {name} must be a single-element FP32 tensor; got "
                f"{gs.numel()} element(s) of {gs.dtype}."
            )
