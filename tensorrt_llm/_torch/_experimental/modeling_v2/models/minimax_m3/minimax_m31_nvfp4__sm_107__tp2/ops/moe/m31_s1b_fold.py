# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S1b fold: the decode router above the tensor-core router's rows, routed from top-k ids.

Stock: ``F.linear(float(h), w)`` (cuBLASLt split-K TF32 GEMM + its splitKreduce), then the trtllm-gen MiniMax2 routing
selects the top-4 from the logits. Here: ONE strided-batched TF32 GEMM over S K-slices (partials), then
``trtllm::minimax_m31_s1b_fold`` sums them in slice order (bitwise cuBLASLt's reduce at the split factor its heuristic
picks for the shape) and runs the routing library's own MiniMax2 selection (the production E128K8 instance), so the
top-4 ids (slot order) and bf16 weights are bitwise the stock routing's; the MoE runner then routes from them
(RoutingFromTopKIds). VR microbench: -1.7 / -2.6 us per layer at 88 / 176 rows.

The split factor and the weights operand's layout (a strided view of the router weight, or a laid-out copy) are pinned
together per row count at prebuild, by a bitwise probe of this device's cuBLASLt; every pin is then re-validated with
its final operand before capture (a failure drops the row count, which keeps the stock path). Kill switch
TRTLLM_M31_DISABLE_S1B_FOLD=1.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.fakes import m31_fake_ops
from tensorrt_llm.logger import logger

from ...m31_switches import m31_disabled

S1B_FOLD = not m31_disabled("S1B_FOLD")
SPLITS = (
    16,
    8,
)  # the fold kernel's built split factors (cuBLASLt's choices at 49..448 rows on sm_107)
HIDDEN, EXPERTS, TOP_K = 6144, 128, 4
PROBES = 3
_STATE: dict = {}
_PINS: Dict[int, int] = {}  # row count -> split factor
_LAYOUT: Dict[
    int, bool
] = {}  # row count -> True: strided view, False: laid-out copy (probed together with its S)
_LAID: Dict[Tuple[int, int], torch.Tensor] = {}
_LOGGED: set = set()


def fold_op():
    """``trtllm::minimax_m31_s1b_fold`` from the separately built library, never loaded during capture."""
    if "op" not in _STATE:
        if torch.cuda.is_current_stream_capturing():
            return None
        op = getattr(torch.ops.trtllm, "minimax_m31_s1b_fold", None)
        if op is not None:
            m31_fake_ops.register()
        _STATE["op"] = op
    return _STATE["op"]


def x_slices(x: torch.Tensor, s: int) -> torch.Tensor:
    """fp32 [rows, 6144] -> [S, rows, 6144 / S] as a view (no copy)."""
    return x.view(x.shape[0], s, HIDDEN // s).transpose(0, 1)


def w_strided(w: torch.Tensor, s: int) -> torch.Tensor:
    """fp32 [128, 6144] -> [S, 6144 / S, 128] as a view (no copy)."""
    return w.view(EXPERTS, s, HIDDEN // s).permute(1, 2, 0)


def _w_operand(w: torch.Tensor, s: int, strided: bool) -> torch.Tensor:
    return w_strided(w, s) if strided else _LAID[(w.data_ptr(), s)]


def _slice_sum(p: torch.Tensor) -> torch.Tensor:
    r = p[0].clone()
    for k in range(1, p.shape[0]):
        r = r + p[k]
    return r


def _bitwise(rows: int, s: int, w: torch.Tensor, w_op: torch.Tensor, gen: torch.Generator) -> bool:
    for _ in range(PROBES):
        x = torch.randn(rows, HIDDEN, generator=gen, device=w.device).to(torch.bfloat16).float()
        got = _slice_sum(torch.bmm(x_slices(x, s), w_op))
        if not torch.equal(got.view(torch.int32), F.linear(x, w).view(torch.int32)):
            return False
    return True


def prebuild(router_weights, rows_range) -> Dict[int, int]:
    """Pin, for every decode row count in ``rows_range`` (outside capture, before warmup), a split factor S AND the
    weights operand's layout whose bmm + slice-order sum is bitwise this device's F.linear on random inputs: the
    strided view first, else a laid-out copy (one per layer and S, built only for the S some row count needs). Then
    every pin is re-validated with exactly the operand it will execute (its own S and layout, the final copies);
    a failure drops that row count. Returns row count -> S (empty: the fold is off or not built)."""
    _PINS.clear()
    _LAYOUT.clear()
    _LAID.clear()
    if not S1B_FOLD or fold_op() is None or not router_weights:
        logger.info(
            f"[M3.1] S1B_FOLD not taken: {'switched off' if not S1B_FOLD else 'library not built'}"
        )
        return dict(_PINS)
    w0 = router_weights[0]
    gen = torch.Generator(device=w0.device).manual_seed(20261102)
    rows_list = list(rows_range)
    for rows in rows_list:
        for s in SPLITS:
            if _bitwise(rows, s, w0, w_strided(w0, s), gen):
                _PINS[rows], _LAYOUT[rows] = s, True
                break
    for rows in rows_list:
        if rows in _PINS:
            continue
        for s in SPLITS:
            if _bitwise(rows, s, w0, w_strided(w0, s).contiguous(), gen):
                _PINS[rows], _LAYOUT[rows] = s, False
                break
    for s in {v for r, v in _PINS.items() if not _LAYOUT[r]}:
        for w in router_weights:
            _LAID[(w.data_ptr(), s)] = w_strided(w, s).contiguous()
    # Final validation with the operand each pin executes (no pin runs a layout it was not probed with).
    dropped = [
        r
        for r in sorted(_PINS)
        if not _bitwise(r, _PINS[r], w0, _w_operand(w0, _PINS[r], _LAYOUT[r]), gen)
    ]
    for r in dropped:
        del _PINS[r], _LAYOUT[r]
    torch.cuda.synchronize()
    groups = {}
    for r, s in _PINS.items():
        groups.setdefault((s, _LAYOUT[r]), []).append(r)
    logger.info(
        "[M3.1] S1B_FOLD pinned "
        + ", ".join(
            f"S{s} {'strided' if st else 'laid out'}: {len(v)} row counts ({min(v)}..{max(v)})"
            for (s, st), v in sorted(groups.items())
        )
        + f"; {len(rows_list) - len(_PINS)} row counts keep the stock router"
        + (f" ({len(dropped)} dropped by the final validation)" if dropped else "")
    )
    return dict(_PINS)


def route(
    x_f32: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, routed_scaling: float, pdl: bool
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(top-k ids int32 [rows, 4], weights bf16 [rows, 4]) for fp32 router input ``x_f32``, or None (no pin, or an
    operand outside the fold's layout): the caller keeps the stock logits path."""
    rows = x_f32.shape[0] if x_f32.dim() == 2 else 0
    s = _PINS.get(rows)
    if (
        s is None
        or x_f32.dtype != torch.float32
        or x_f32.shape[1] != HIDDEN
        or x_f32.stride(1) != 1
        or x_f32.stride(0) != HIDDEN
        or w.dtype != torch.float32
        or tuple(w.shape) != (EXPERTS, HIDDEN)
        or not w.is_contiguous()
        or bias.dtype != torch.float32
        or bias.numel() != EXPERTS
    ):
        return None
    partials = torch.bmm(x_slices(x_f32, s), _w_operand(w, s, _LAYOUT[rows]))
    ids = torch.empty(rows, TOP_K, dtype=torch.int32, device=x_f32.device)
    weights = torch.empty(rows, TOP_K, dtype=torch.bfloat16, device=x_f32.device)
    torch.ops.trtllm.minimax_m31_s1b_fold(
        partials, bias.contiguous(), None, ids, weights, routed_scaling, pdl
    )
    if rows not in _LOGGED:
        _LOGGED.add(rows)
        logger.info(f"[M3.1] S1B_FOLD taken at {rows} rows (S {s})")
    return ids, weights
