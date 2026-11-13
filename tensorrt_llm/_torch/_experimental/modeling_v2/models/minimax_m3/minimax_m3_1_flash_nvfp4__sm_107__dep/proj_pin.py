# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_PROJ_PIN: each layer's qkv_proj and o_proj run FlashInfer's CuTe DSL MXFP8 GEMM at the tuned tactic that
``ops/gemm/mxfp8_cute_pinned_tactics`` pins for their shape (non-split-K, so bitwise the default tactic's output),
where the table has one: 33..128 rows at [9856, 6144] and [6144, 8192].

The table is engaged for these two projections' own calls (``engaged_for``), not for the process: the shared
expert's [6144, 6144] entry stays unengaged. It acts where the projection's GEMM runs through the table's ``run``
(the MXFP8 hand-off apply in ``merge_mxfp8``); a GEMM that reaches TRT-LLM's FlashInfer runner keeps the engine's
own tactic choice.

``TRTLLM_M31_DISABLE_DEP_PROJ_PIN=1`` installs nothing; the table's ``TRTLLM_MXFP8_CUTE_PIN=0`` disables every entry.
Read at import.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Iterable, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as _pin,
)
from tensorrt_llm._torch.modules.linear import Linear, MXFP8LinearMethod
from tensorrt_llm.logger import logger

from .m31_switches import m31_disabled

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig

DEP_PROJ_PIN = not m31_disabled("DEP_PROJ_PIN")
PROJECTIONS = ("qkv_proj", "o_proj")

# The (outcome, detail) pairs already logged.
_LOGGED: set = set()


def _log_once(outcome: str, detail: str) -> None:
    if (outcome, detail) not in _LOGGED:
        _LOGGED.add((outcome, detail))
        logger.info(f"[M3.1 DEP] DEP_PROJ_PIN {outcome}: {detail}")


def proj_decline_reason(proj: Optional[torch.nn.Module]) -> Optional[str]:
    """Why the projection ``proj`` keeps its stock forward, or None if its GEMM may run with the table engaged."""
    if not isinstance(proj, Linear):
        return f"{type(proj).__name__} is not a Linear"
    if not isinstance(proj.quant_method, MXFP8LinearMethod):
        return f"quant method {type(proj.quant_method).__name__}"
    n, k = proj.weight.shape
    if not any(key[:2] == (n, k) for key in _pin.TACTICS):
        return f"no pinned tactic for [{n}, {k}]"
    if "forward" in proj.__dict__:
        return "its forward is already replaced"
    return None


def _pinned_forward(self: Linear, *args, **kwargs):
    with _pin.engaged_for(f"DEP_PROJ_PIN {self._dep_proj_pin}"):
        return type(self).forward(self, *args, **kwargs)


def _installed(proj: torch.nn.Module) -> bool:
    return getattr(proj.__dict__.get("forward"), "__func__", None) is _pinned_forward


def install(layers: Iterable[torch.nn.Module], model_config: ModelConfig) -> int:
    """Engage the table for the GEMM of each layer's qkv_proj and o_proj that ``proj_decline_reason`` admits. Returns
    how many projections run it, including any an earlier call installed; 0 with TRTLLM_M31_DISABLE_DEP_PROJ_PIN=1 or
    under torch.compile (the engagement is a Python context around the call). Logs each distinct decline once."""
    if not DEP_PROJ_PIN:
        return 0
    if model_config.extra_attrs.get("torch_compile_enabled", False):
        _log_once("not installed", "torch.compile is on")
        return 0
    count = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        for name in PROJECTIONS:
            proj = getattr(attn, name, None)
            if proj is None:
                continue
            if not _installed(proj):
                reason = proj_decline_reason(proj)
                if reason is not None:
                    _log_once("not installed", f"{name}: {reason}")
                    continue
                proj._dep_proj_pin = name
                proj.forward = types.MethodType(_pinned_forward, proj)
            count += 1
    return count
