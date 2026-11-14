# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE_SWIGLU for GatedMLP instances: the SwiGLU-OAI activation (limit 7, alpha 1.702, beta 1) of the MLP's
gate_up output by ``mv2_minimax_m3::minimax_m31_swiglu_oai_vec`` in its row-grid mode (nv 0, v5: bit for bit the
Triton ``silu_and_mul`` it replaces, with PDL),
installed per MLP instance on ``_apply_activation``.

``install(mlps)`` wraps each GatedMLP that applies a bf16 SwiGLU-OAI through ``_apply_activation`` (SiLU activation,
fused gate_up, constants (7, 1.702, 1), no FP8 down_proj input); others keep the stock activation. An installed MLP
never runs the Triton kernel: an operand the op does not take (not a contiguous 2-D bf16 CUDA tensor, under
torch.compile, the op not built) or ``TRTLLM_M31_DISABLE_NATIVE_SWIGLU=1`` (read at import) runs ``swiglu_oai_torch``,
the same SwiGLU-OAI in torch ops (fp32, not bitwise the Triton kernel). A LoRA call keeps the stock activation.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.m31_switches import (
    m31_disabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm.logger import logger

NATIVE_SWIGLU = not m31_disabled("NATIVE_SWIGLU")
LIMIT, ALPHA, BETA = 7.0, 1.702, 1.0
_OPS: list = []
_LOGGED: set = set()


def swiglu_op():
    """``mv2_minimax_m3::minimax_m31_swiglu_oai_vec`` when this build carries it, else None (looked up once)."""
    if not _OPS:
        _OPS.append(getattr(family_ops(), "minimax_m31_swiglu_oai_vec", None))
    return _OPS[0]


def _once(line: str) -> None:
    if line not in _LOGGED:
        _LOGGED.add(line)
        logger.info(f"[M3.1] NATIVE_SWIGLU {line}")


def swiglu_oai_torch(gu: torch.Tensor) -> torch.Tensor:
    """SwiGLU-OAI of ``gu`` = [gate | up] in torch ops (fp32 math, one rounding to ``gu``'s dtype)."""
    gate, up = gu.float().chunk(2, dim=-1)
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(-LIMIT, LIMIT)
    return (gate * torch.sigmoid(ALPHA * gate) * (up + BETA)).to(gu.dtype)


def outside_contract(gu: torch.Tensor) -> Optional[str]:
    """Why the op does not take ``gu``, or None: a contiguous 2-D bf16 CUDA [rows, 2d] with d even, outside
    torch.compile, on a build that carries the op."""
    if not NATIVE_SWIGLU:
        return "off (TRTLLM_M31_DISABLE_NATIVE_SWIGLU=1)"
    if torch.compiler.is_compiling():
        return "under torch.compile"
    if (
        not gu.is_cuda
        or gu.dtype != torch.bfloat16
        or gu.dim() != 2
        or not gu.is_contiguous()
        or gu.shape[1] % 4
    ):
        return f"gu {gu.dtype} {tuple(gu.shape)} strides {gu.stride()} on {gu.device}"
    if swiglu_op() is None:
        return "mv2_minimax_m3::minimax_m31_swiglu_oai_vec is not built"
    return None


def swiglu_oai(gu: torch.Tensor) -> torch.Tensor:
    """SwiGLU-OAI of ``gu``: the op's row grid with PDL, else ``swiglu_oai_torch``."""
    reason = outside_contract(gu)
    if reason is None:
        return swiglu_op()(gu, True, 0)
    _once(f"not taken ({reason}); the torch SwiGLU-OAI runs")
    return swiglu_oai_torch(gu)


def serves(mlp) -> bool:
    """Whether ``mlp`` is a GatedMLP whose ``_apply_activation`` applies a SwiGLU-OAI with these constants to a bf16
    gate_up output (the stock Triton ``silu_and_mul`` path)."""
    return (
        isinstance(mlp, GatedMLP)
        and mlp.activation is F.silu
        and not mlp.split_gate_up
        and mlp.swiglu_limit == LIMIT
        and mlp.swiglu_alpha == ALPHA
        and mlp.swiglu_beta == BETA
        and not (mlp.down_proj.has_fp8_qdq or mlp.down_proj.has_w4a8_nvfp4_fp8)
    )


def install(mlps: Iterable) -> int:
    """Wrap ``_apply_activation`` of every MLP in ``mlps`` that ``serves`` admits (once per instance); returns how
    many are wrapped."""
    count = 0
    for mlp in mlps:
        if not serves(mlp) or getattr(mlp, "_m31_native_swiglu", False):
            continue
        stock = mlp._apply_activation

        def apply(x, *, has_lora: bool = False, _stock=stock):
            if has_lora:
                return _stock(x, has_lora=True)
            return swiglu_oai(x)

        mlp._apply_activation = apply
        mlp._m31_native_swiglu = True
        count += 1
    return count


def engagement(count: int) -> str:
    """The log line's state for ``count`` wrapped MLPs."""
    if not NATIVE_SWIGLU:
        return f"off (TRTLLM_M31_DISABLE_NATIVE_SWIGLU=1); the torch SwiGLU-OAI on {count} MLPs"
    if swiglu_op() is None:
        return f"op not built; the torch SwiGLU-OAI on {count} MLPs"
    return f"row-grid CUDA SwiGLU-OAI with PDL on {count} MLPs"
