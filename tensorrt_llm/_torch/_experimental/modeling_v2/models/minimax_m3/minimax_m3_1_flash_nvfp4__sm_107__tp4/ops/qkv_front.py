# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QKV_FRONT: a sparse layer's front in one launch, ``minimax_m31_qkv_front``
(csrc/qkvFrontSm107.cu, a port of the reference front kernel).

It fuses the MoE boundary all-reduce (the prepushed 2 / 3 consumer of ``m3_mc_allreduce_norm``'s exchange: residual,
RMSNorm, MXFP8), the QKV MXFP8 GEMV and the NVFP4 QKV + indexer producer. At split 8 every output is bit for bit the
served chain's (the cluster all-reduce at the stock oneshot order -> ``m3_mx_gemv`` (8, 8) -> ``producerPw4``): the
updated residual, the normed rows, q / index-q and every K / V / index-K cache byte. The exchange, its flags and the
MoE's push are unchanged; the front adds its own hand-off workspace (``FrontWorkspace``), one per (device, split).

This target keeps the switch, the contract and the engagement line and calls ``qkv_front``.
"""

from __future__ import annotations

import sys
import threading
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (
    native_ops,
)

HIDDEN = 6144
MAX_TOKENS = 8
#: Bytes of the hand-off: two parities of [16 rows x 6144] E4M3 (fill 0x80) and of 48 x 256-byte scale atoms + 512
#: (fill 0xFF); int32 call counters per CTA (fill 0). csrc/qkvFrontSm107.h.
XQ_BYTES = 2 * 16 * HIDDEN
XSF_BYTES = 2 * ((HIDDEN // 128) * 256 + 512)
MAX_CTAS = 256
#: The split that is bitwise the served chain (m3_mx_gemv's (8, 8) config); 4 is the reference geometry (exact-class).
SPLIT_BITWISE = 8

_lock = threading.Lock()
_workspaces: Dict[Tuple[int, int], "FrontWorkspace"] = {}
_engaged: set = set()


class FrontWorkspace:
    """The front's boundary -> GEMV hand-off on one device for one split: ``xq`` / ``xsf`` (two parities, Lamport
    sentinels) and the per-CTA call counters ``ctr``. Created eagerly, outside CUDA-graph capture."""

    def __init__(self, device: torch.device) -> None:
        self.xq = torch.full((XQ_BYTES,), 0x80, dtype=torch.uint8, device=device)
        self.xsf = torch.full((XSF_BYTES,), 0xFF, dtype=torch.uint8, device=device)
        self.ctr = torch.zeros(MAX_CTAS, dtype=torch.int32, device=device)


def op_available() -> bool:
    """Whether this build registers ``minimax_m31_qkv_front``."""
    return hasattr(native_ops, "minimax_m31_qkv_front") and hasattr(
        native_ops, "minimax_m31_qkv_front_max_clusters"
    )


_fits: Dict[Tuple[int, int, int], bool] = {}


def fits(device: torch.device, heads_q: int, split: int = SPLIT_BITWISE) -> bool:
    """Whether the device holds all heads_q + 4 clusters of the front's grid at once (the launcher refuses a grid that
    it does not: its GEMV clusters wait on the boundary clusters). Queried once per (device, heads, split)."""
    index = torch.device(device).index
    key = (torch.cuda.current_device() if index is None else index, int(heads_q), int(split))
    if key not in _fits:
        with torch.cuda.device(key[0]):
            n = int(native_ops.minimax_m31_qkv_front_max_clusters(int(split)))
        _fits[key] = n >= heads_q + 4
    return _fits[key]


def get_workspace(device: torch.device, split: int = SPLIT_BITWISE) -> Optional[FrontWorkspace]:
    """The device's hand-off workspace for ``split``; None when it does not exist yet during a capture."""
    index = torch.device(device).index
    key = (torch.cuda.current_device() if index is None else index, int(split))
    with _lock:
        ws = _workspaces.get(key)
        if ws is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            ws = _workspaces[key] = FrontWorkspace(torch.device("cuda", key[0]))
        return ws


def log_engaged(target: str, rank: int, device: torch.device, rows: int, split: int) -> None:
    """One engagement line per (target, rank, rows) on stderr: ``[M31_QKV_FRONT] engaged ...``."""
    key = (target, rank, rows)
    if key in _engaged:
        return
    _engaged.add(key)
    print(
        f"[M31_QKV_FRONT] engaged: target {target} rank {rank} device {device} rows {rows} split {split} "
        "(MoE boundary all-reduce + QKV GEMV + producer in one launch)",
        file=sys.stderr,
        flush=True,
    )


def qkv_front(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    out_residual: torch.Tensor,
    out_normed: Optional[torch.Tensor],
    exchange: Tuple[torch.Tensor, torch.Tensor, int, int, int],
    prepushed: int,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: FrontWorkspace,
    kv_data: torch.Tensor,
    kv_scale: torch.Tensor,
    index_k: torch.Tensor,
    out_cache_loc: torch.Tensor,
    kv_quant_scale: torch.Tensor,
    peps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    position_ids: torch.Tensor,
    heads_q: int,
    split: int = SPLIT_BITWISE,
    pdl: bool = True,
    early: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the front; returns (q FP8 [T, heads_q, 128], index_q FP8 [T, 1, 128]) and writes ``out_residual``,
    ``out_normed`` (optional) and the caches. ``exchange`` = (the bucket's unicast int32 view, its cluster-8 flags,
    rank, world, bucket tokens); ``prepushed`` 2 (x in set A) or 3 (x in set A, x2 in set B)."""
    ws_uc, flags, rank, world, maxt = exchange
    q, index_q = native_ops.minimax_m31_qkv_front(
        residual,
        norm_weight,
        float(eps),
        out_residual,
        out_normed,
        ws_uc,
        flags,
        int(rank),
        int(world),
        int(maxt),
        int(prepushed),
        weight,
        weight_scale,
        workspace.xq,
        workspace.xsf,
        workspace.ctr,
        kv_data,
        kv_scale,
        index_k,
        out_cache_loc,
        kv_quant_scale,
        float(peps),
        q_weight,
        k_weight,
        index_q_weight,
        index_k_weight,
        rotary_cos_sin,
        position_ids,
        int(heads_q),
        int(split),
        bool(pdl),
        bool(early),
    )
    return q, index_q
