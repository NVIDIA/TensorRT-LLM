# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The stage's multi-instance context FMHA, set up at the stage's own head counts.

The TEP target's library (``kernels/fmk_context_fmha/attention``) serves a TEP4 rank: its step
plans, its checks and its warm-up are made at 32 query / 4 KV heads. A stage runs all of the
model's heads (TP1), ``dispatch.FMK_STAGE_HEADS``. This module makes those parts at the
stage's heads and reuses the rest read-only: the TEP library's ``FmkPureContextFmha`` (the
call-time checks, the thop preprocess and the kernel launch) and the op's planner, workspace
and kernel (``ops/attention/context_fmha_fmk``). Nothing here changes the TEP target.

* ``StagePlans``: the plan of the latest lengths at the stage's heads, which the library's
  served call asks for. The planner gets one work unit per group of ``NUM_INST`` query
  heads, so a plan covers all 128 heads: 32 head groups per 128-row block, four times a
  TEP4 rank's units, over the same per-CTA workspace slots.
* ``stage_refusal``: what the kernel cannot serve on a layer; the TEP library's checks of a
  pure context install, with the stage's heads in place of a TEP4 rank's.
* ``stage_prepare``: the reserved partial-result workspace, after a warm-up call at the
  stage's heads (a warm-up at 32 / 4 would leave heads 32..127 of its output unwritten).
* ``prepare``: every check and build of the stage's full-attention layers, without changing
  a layer; ``StageInstall.commit`` then puts the library before each layer's thop fallback.
"""

from __future__ import annotations

import dataclasses
import time
from typing import List, Optional, Tuple

import torch
from torch.utils._python_dispatch import _disable_current_modes

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.quantization.mode import QuantMode

from ... import dispatch as _dispatch
from ..._tep import module as _tep_module

_library = _tep_module("kernels.fmk_context_fmha.attention")
_fmk = _tep_module("ops.attention.context_fmha_fmk")


def _plan(q_lens, kv_lens, num_ctas: int, device) -> _fmk.FmkPlan:
    """The planner's work units of one context call at the stage's heads."""
    num_heads, num_kv_heads = _dispatch.FMK_STAGE_HEADS
    return _fmk.plan_context_fmha(
        q_lens,
        kv_lens,
        num_ctas=num_ctas,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_inst=_library.NUM_INST,
        tile_n=_library.TILE_N,
        device=device,
    )


class StagePlans:
    """The plan of the latest lengths at the stage's heads, shared by the stage's
    full-attention layers of a step (what ``FmkPureContextFmha`` asks its plans for).

    A step's full-attention layers run one after another with the same lengths, so the first
    plans and the rest reuse the plan's device buffer (read in stream order). A step with
    other lengths replaces it; the replaced buffer returns to the caching allocator on the
    same stream.
    """

    def __init__(self, num_ctas: int):
        self.num_ctas = num_ctas
        self._key: Optional[Tuple] = None
        self._plan: Optional[_fmk.FmkPlan] = None
        #: Plans built, and the host time they took.
        self.planned = 0
        self.plan_seconds = 0.0

    def get(self, q_lens: Tuple[int, ...], kv_lens: Tuple[int, ...], device) -> _fmk.FmkPlan:
        """The plan of these lengths at ``dispatch.FMK_STAGE_HEADS``."""
        key = (q_lens, kv_lens, device)
        if key != self._key:
            start = time.perf_counter()
            self._plan = _plan(q_lens, kv_lens, self.num_ctas, device)
            self.plan_seconds += time.perf_counter() - start
            self.planned += 1
            self._key = key
        return self._plan


def stage_refusal(
    attention, sm: Optional[int], heads: Tuple[int, int] = _dispatch.FMK_STAGE_HEADS
) -> Optional[str]:
    """Why the kernel cannot serve ``attention``'s pure context steps at ``heads`` (query, KV)
    per rank, or None: the TEP library's pure context checks, at ``heads``."""
    attn = attention.attn
    if sm != _library.SERVED_SM:
        return f"the kernel is written for sm{_library.SERVED_SM}, not sm{sm}"
    if not any(isinstance(fmha, FallbackFmha) for fmha in attn._fmha_manager.fmha_libs):
        return "no thop fallback among its FMHA libraries"
    if getattr(attn, "is_mla_enable", False):
        return "MLA"
    if attn.head_dim != _fmk.HEAD_DIM:
        return f"head dim {attn.head_dim}"
    if (
        attn.num_heads % attn.num_kv_heads
        or (attn.num_heads // attn.num_kv_heads) % _library.NUM_INST
    ):
        return f"{attn.num_heads} query heads over {attn.num_kv_heads} KV heads"
    if (attn.num_heads, attn.num_kv_heads) != tuple(heads):
        return (
            f"{attn.num_heads} query heads over {attn.num_kv_heads} KV heads per rank (the "
            f"stage install serves {heads[0]} over {heads[1]})"
        )
    if not QuantMode(attn.quant_mode).has_fp8_kv_cache():
        return "a KV cache that is not FP8"
    if not hasattr(thop, "trtllm_gen_context_preprocess"):
        return "the binding lacks trtllm_gen_context_preprocess"
    return None


def stage_warm_up(device: torch.device, num_ctas: int, workspace: _fmk.FmkWorkspace) -> None:
    """Compile the kernel and its merge at the stage's heads on a small synthetic call (one
    request of one row over 12,800 zero keys, which the planner cuts) and check that it ran
    and wrote every head."""
    num_heads, num_kv_heads = _dispatch.FMK_STAGE_HEADS
    pages = 400
    q = torch.zeros(1, num_heads, _fmk.HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    pool = torch.zeros(
        2 * pages, num_kv_heads, _fmk.PAGE, _fmk.HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    ids = torch.arange(pages, dtype=torch.int32, device=device)
    table = torch.stack([2 * ids, 2 * ids + 1]).unsqueeze(0).contiguous()
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([pages * _fmk.PAGE], dtype=torch.int32, device=device)
    scales = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    out = torch.full(
        (1, num_heads, _fmk.HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=device
    )
    plan = _plan((1,), (pages * _fmk.PAGE,), num_ctas, device)
    _fmk.paged_context_fmha_fmk(
        q, pool, table, cu_q, kv_lens, plan, scales[0:1], scales[1:2], out, workspace=workspace
    )
    if plan.merge_meta is None or not bool(torch.isfinite(out).all()):
        raise ValueError(
            f"{_dispatch.TAG} fmk_pure_context_fmha: the warm-up call did not run as planned "
            f"(merges {0 if plan.merge_meta is None else plan.merge_meta.shape[0]})"
        )


def stage_prepare(device: torch.device, num_ctas: int) -> _fmk.FmkWorkspace:
    """The reserved partial-result workspace, after a warm-up call on it at the stage's heads.

    The loader builds the model under ``MetaInitMode``, which turns ``torch.empty`` into meta
    tensors (no storage). The workspace and the warm-up are device-side setup, not model
    construction, so the active dispatch modes are popped while they run.
    """
    with _disable_current_modes():
        workspace = _fmk.FmkWorkspace()
        workspace.reserve(_fmk.max_slots(num_ctas), _library.NUM_INST, device)
        if workspace.part_o.is_meta or workspace.part_ml.is_meta:
            raise ValueError(f"{_dispatch.TAG} fmk_pure_context_fmha: the workspace has no storage")
        stage_warm_up(device, num_ctas, workspace)
    return workspace


@dataclasses.dataclass
class StageInstall:
    """What ``prepare`` checked and built, before any layer changes; ``commit`` makes the
    changes."""

    chosen: List
    plans: StagePlans
    workspace: _fmk.FmkWorkspace
    num_ctas: int

    def commit(self) -> int:
        """Put an ``FmkPureContextFmha`` just before the thop fallback of every chosen layer
        that has none; returns the number of chosen layers."""
        for attention in self.chosen:
            attn = attention.attn
            if _library.pure_context_library(attn) is not None:
                continue
            libs = attn._fmha_manager.fmha_libs
            index = next(i for i, fmha in enumerate(libs) if isinstance(fmha, FallbackFmha))
            libs.insert(
                index,
                _library.FmkPureContextFmha(attn, libs[index], self.plans, self.workspace),
            )
            attn._fmha_manager._cache.clear()
        return len(self.chosen)


def _layer_refusal(attention, sm: Optional[int]) -> Optional[str]:
    reason = stage_refusal(attention, sm)
    if reason is None and _library.context_library(attention.attn) is not None:
        # A stage runs no mixed step, and that library's plans are a TEP4 rank's.
        return "fmk_context_fmha is installed on it"
    return reason


def prepare(layers, compute_capability: Optional[Tuple[int, int]]) -> StageInstall:
    """Check every full-attention layer of ``layers`` and build the workspace and the step
    plans at the stage's heads, without changing any layer.

    Raises:
        ValueError: if a full-attention layer or the device is not one the kernel serves at
            the stage's heads, or the warm-up did not run as planned.
    """
    sm = _dispatch.sm_of(compute_capability)
    chosen = [layer.self_attn for layer in layers if not layer.self_attn.is_sliding]
    refused = [
        f"layer {attention.layer_idx}: {reason}"
        for attention in chosen
        if (reason := _layer_refusal(attention, sm)) is not None
    ]
    if refused:
        raise ValueError(
            f"{_dispatch.TAG} fmk_pure_context_fmha cannot take the stage's pure context "
            "steps: " + "; ".join(refused)
        )
    device = torch.device("cuda", torch.cuda.current_device())
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = stage_prepare(device, num_ctas)
    return StageInstall(chosen, StagePlans(num_ctas), workspace, num_ctas)


__all__ = ["StageInstall", "StagePlans", "prepare", "stage_prepare", "stage_refusal"]
