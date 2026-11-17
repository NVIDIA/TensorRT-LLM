# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused MXFP8 activation quantization + low-M MXFP8 GEMM for decode (SM100+).

``out[M, N] = mxfp8(x)[M, K] @ mxfp8(W)[N, K]^T`` for ``1 <= M <= 8`` in one kernel
(``torch.ops.trtllm.minimax_m3_mxfp8_low_m_gemm``, reached through the ``ops/gemm`` entry).

* Each CTA owns ``16 * ROW_TILES`` weight rows and ``K / CLUSTER_K`` columns.
  Before ``griddepcontrol.wait`` it issues ``cp.async.bulk`` copies of its weight
  tile and UE8M0 scale chunks (existing swizzled 128x4 layout) into shared memory,
  so the weight stream overlaps the producer kernel under PDL. Configurations
  keep the whole grid resident in one wave so every CTA prefetches.
* Activation quantization reproduces ``trtllm::mxfp8_quantize`` with identical
  intrinsics (block 32, UE8M0 rounded toward +inf, saturating RN E4M3); each
  thread issues all of its loads before converting, one L2 round trip per CTA.
* One ``mma.sync.m16n8k32`` E4M3 per 32-wide scale block and row tile with exact
  power-of-two rescaling and FP32 accumulation. Warps reduce through shared
  memory (reusing the consumed weight tile); split-K peers push their partial
  tiles into the cluster root with ``st.async`` and exit, and the root sums the
  ranks in order.
"""

from typing import Optional

import torch

from tensorrt_llm.logger import logger

from ..ops.gemm.mxfp8_low_m_gemm import mxfp8_low_m_gemm as _low_m_gemm_op

# (N, K) -> (warps, cluster_k, max_m, row_tiles); measured on GB300 TP4 M3 shapes. The fused kernel beats native
# quantize + GEMM only up to max_m rows; larger M, and every other shape, keeps the native path. These four
# configurations are the ones built into ``torch.ops.trtllm.minimax_m3_mxfp8_low_m_gemm``.
_CONFIGS = {
    (2560, 6144): (4, 4, 8, 2),  # QKV (q, k, v, index q / k)
    (6144, 2048): (8, 2, 8, 4),  # o_proj
    (6144, 6144): (4, 4, 1, 1),  # dense MLP gate_up
    (6144, 3072): (4, 2, 8, 2),  # dense MLP down
}
# Opt-in shared memory per CTA on SM100 (227 KiB) less a margin for the static mbarriers.
_MAX_DYNAMIC_SMEM = 227 * 1024 - 1024  # capacity: fixed: the device's shared-memory limit

_logged_shapes = set()


def _smem_bytes(m: int, k: int, warps: int, cluster_k: int, row_tiles: int) -> int:
    """Dynamic shared memory of one CTA, mirroring the host-side size in ``launch``."""
    kc = k // cluster_k
    kc_blocks = kc // 32
    tile = 16 * row_tiles * 8 * (6 if m > 32 else 4 if m > 16 else 2 if m > 8 else 1)
    w_bytes = 16 * row_tiles * (kc + 32)
    red_bytes = warps * tile * 4
    return (
        w_bytes
        + row_tiles * (kc_blocks // 4) * 256
        + m * kc
        + (m * kc_blocks + 15) // 16 * 16
        + (cluster_k - 1) * tile * 4
        + (0 if red_bytes <= w_bytes else red_bytes)
    )


def get_config(n: int, k: int) -> Optional[tuple]:
    """Return (warps, cluster_k, max_m, row_tiles) for an (N, K) weight, or None when no configuration is built."""
    return _CONFIGS.get((n, k))


def mxfp8_low_m_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
    use_pdl: bool = False,
    config: Optional[tuple] = None,
) -> Optional[torch.Tensor]:
    """Fused MXFP8 quantize + GEMM; ``weight_scale`` is the swizzled 128x4 buffer.

    Returns None for unsupported operands so callers keep the native path.
    """
    if (
        output_dtype != torch.bfloat16
        or x.dtype != torch.bfloat16
        or torch.cuda.get_device_capability(x.device)[0] < 10
    ):
        return None
    n, k = weight.shape
    config = config or get_config(n, k)
    if config is None or config not in _CONFIGS.values():
        return None
    warps, cluster_k, max_m, row_tiles = config
    if (
        x.shape[0] > max_m
        or n % (16 * row_tiles) != 0
        or k % (cluster_k * 128) != 0
        or (k // cluster_k // 32) % warps != 0
        or x.data_ptr() % 16 != 0
        or x.stride(0) % 8 != 0
        or weight.data_ptr() % 16 != 0
        or weight_scale.data_ptr() % 16 != 0
        or _smem_bytes(x.shape[0], k, warps, cluster_k, row_tiles) > _MAX_DYNAMIC_SMEM
    ):
        return None
    if (n, k) not in _logged_shapes:
        _logged_shapes.add((n, k))
        logger.info(
            f"[MXFP8 low-M] fused decode GEMM engaged: N={n} K={k} "
            f"warps={warps} cluster_k={cluster_k} row_tiles={row_tiles} pdl={use_pdl}"
        )
    return _low_m_gemm_op(x, weight, weight_scale, warps, cluster_k, row_tiles, use_pdl)
