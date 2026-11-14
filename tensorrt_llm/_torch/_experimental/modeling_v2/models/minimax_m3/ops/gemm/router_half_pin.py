# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact, qualified half-row TF32 router dispatch for the M3.1 TP2 IPC boundary."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.m31_switches import (
    m31_disabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

# Only the full-step row counts qualified bitwise against the full-row cuBLAS reference.
QUALIFIED_ROWS = frozenset(
    (8000, 16000, 16384, 16317, 16314, 16313, 16341, 16308, 16334, 16296, 16259)
)
_DISABLED = m31_disabled("ROUTER_HALF_PIN")
_PREPARED = {}
_LOGGED = set()


def _load():
    """Whether the family library carries the half-row router GEMM (its fake registers with the library)."""
    return hasattr(family_ops(), "m31_router_half_pin")


def try_router_half_pin(x, weight, total_rows):
    """Return None for an unqualified layout/tactic, retaining the stock full-row path."""
    if (
        _DISABLED
        or total_rows not in QUALIFIED_ROWS
        or not x.is_cuda
        or x.dtype != torch.float32
        or weight.dtype != torch.float32
        or x.dim() != 2
        or weight.shape != (128, 6144)
        or x.shape[1] != 6144
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.device != weight.device
        or not torch.backends.cuda.matmul.allow_tf32
    ):
        return None
    split = min(max((total_rows // 2 + 64) // 128 * 128, 128), total_rows - 1)
    if x.shape[0] not in (split, total_rows - split):
        return None
    key = (x.device.index, x.shape[0])
    if key not in _PREPARED:
        if torch.cuda.is_current_stream_capturing():
            return None
        if torch.cuda.get_device_capability(x.device) != (10, 7) or not _load():
            _PREPARED[key] = False
        else:
            _PREPARED[key] = family_ops().m31_router_half_pin_prepare(x.shape[0], x.device.index)
    if not _PREPARED[key]:
        return None
    if key not in _LOGGED:
        from tensorrt_llm.logger import logger

        logger.info(
            f"[M3.1] ROUTER_HALF_PIN taken device={x.device.index} rows={x.shape[0]} total={total_rows}"
        )
        _LOGGED.add(key)
    return family_ops().m31_router_half_pin(x, weight)


_ARMED = False


def arm_router_half_pin() -> None:
    """Install from the TP2 target's constructor (the qualified TP2 model only); the comm source stays intact."""
    global _ARMED
    if _ARMED or _DISABLED:
        return
    from ..comm.tp2ctx_ipc import Tp2ctxIpcWorkspace

    stock = Tp2ctxIpcWorkspace.route_owned_full_shape

    def route_owned(self, x_owned, weight, num_tokens):
        pinned = try_router_half_pin(x_owned, weight, num_tokens)
        if pinned is not None:
            return pinned
        return stock(self, x_owned, weight, num_tokens)

    Tp2ctxIpcWorkspace.route_owned_full_shape = route_owned
    _ARMED = True
