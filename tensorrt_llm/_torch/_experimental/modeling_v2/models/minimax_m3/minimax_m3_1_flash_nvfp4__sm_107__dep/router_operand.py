# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_ROUTER_OPERAND: the MoE gate takes its replicated BF16 operand from the post-attention add + RMSNorm instead of
a ``torch.cat``.

``MiniMaxM3Gate`` computes FP32-grade logits from BF16 rows ``x`` as one cuBLAS GEMM of ``cat([x] * terms, -1)`` and
the BF16 terms of its FP32 weight side by side (``split_router_weight_bf16``); the ``cat`` is a copy kernel of its
own. At a site, the MoE layer's
post-attention norm runs ``add_rmsnorm_replicated`` (kernels/megamoe), which also writes the normalized rows
``terms`` times side by side, and publishes that operand in ``SLOT`` keyed by its output; the gate runs the stock GEMM
on the operand when its input is that output. The norm is bitwise FlashInfer's where
``megamoe_topk_reduce_add_rmsnorm_matches_flashinfer`` holds, and the operand holds the bytes the ``cat`` writes in the
same contiguous [rows, terms * H] layout at the start of an allocation, so ``trtllm::cublas_mm`` picks the same
algorithm and the logits are the stock gate's bit for bit. A norm call the kernel does not serve (no residual, more
than MAX_ROWS rows, an operand that is not contiguous 16-byte-aligned BF16 rows, torch.compile) is the stock norm and
publishes nothing; a gate call whose input is not the published output, or that the stock gate would not run as the
``cat`` GEMM, is the stock gate. Every gate call clears the slot. Each rank logs once that the norm published and,
per step width, whether a gate took the operand or missed a published one. Where ``shared_mxfp8`` gave the norm its
shared expert's gate_up_proj hand-off, the same launch (``add_rmsnorm_replicated_mxfp8``) also writes the normalized
rows' MXFP8 form and publishes it there.

``TRTLLM_M31_DISABLE_DEP_ROUTER_OPERAND=1`` wraps no norm and no gate, so nothing is reserved or published. Read at
import.
"""

from __future__ import annotations

import types
from typing import Iterable, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.models.modeling_minimaxm3 import ROUTER_BF16_SPLIT_TERMS, MiniMaxM3Gate
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import is_torch_compiling
from tensorrt_llm.logger import logger

from . import shared_mxfp8 as _shared
from .m31_switches import m31_disabled
from .merge_mxfp8 import MAX_ROWS
from .tail import flashinfer_norm_decline_reason

DEP_ROUTER_OPERAND = not m31_disabled("DEP_ROUTER_OPERAND")

#: Copies of the normalized rows in the gate's operand.
TERMS = ROUTER_BF16_SPLIT_TERMS

# The (outcome, detail) pairs already logged: one line for the norm publishing and per distinct step width a gate
# takes or misses the operand.
_LOGGED: set = set()


def _log_once(outcome: str, detail: str) -> None:
    if (outcome, detail) not in _LOGGED:
        _LOGGED.add((outcome, detail))
        logger.info(f"[M3.1 DEP] DEP_ROUTER_OPERAND {outcome}: {detail}")


class OperandSlot:
    """The replicated router operand of one normalized output, for the gate whose input is that output."""

    def __init__(self):
        self.pending: Optional[Tuple[int, int, torch.Tensor]] = None

    def publish(self, x: torch.Tensor, operand: torch.Tensor) -> None:
        self.pending = (x.data_ptr(), x.shape[0], operand)

    def clear(self) -> None:
        self.pending = None

    def take(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """The operand if ``x`` is the published output, TERMS of its rows wide, else None; clears the slot."""
        pending, self.pending = self.pending, None
        if pending is None or x.dim() != 2:
            return None
        ptr, rows, operand = pending
        if x.data_ptr() != ptr or x.shape[0] != rows or operand.shape[1] != TERMS * x.shape[1]:
            return None
        return operand


SLOT = OperandSlot()

# Per (device, hidden size): BF16 [MAX_ROWS, TERMS * H], allocated outside graph capture.
_BUFFERS: dict = {}


def kernel():
    """``add_rmsnorm_replicated``: the family CUDA C++ add + RMSNorm (NATIVE_ADD_RMSNORM,
    ``minimax_m3/ops/add_rmsnorm``, bitwise the CuTe DSL kernel it falls back to), imported on first use."""
    from ..ops.add_rmsnorm import add_rmsnorm_replicated

    return add_rmsnorm_replicated


def reserve(device: torch.device, hidden_size: int) -> int:
    """Allocate the operand buffer of ``hidden_size``-wide rows on ``device`` once, outside any graph capture; returns
    its bytes (0 with TRTLLM_M31_DISABLE_DEP_ROUTER_OPERAND=1)."""
    if not DEP_ROUTER_OPERAND:
        return 0
    key = (device, hidden_size)
    if key not in _BUFFERS:
        _BUFFERS[key] = torch.empty(
            (MAX_ROWS, TERMS * hidden_size), dtype=torch.bfloat16, device=device
        )
    return _BUFFERS[key].numel() * _BUFFERS[key].element_size()


def operand(x: torch.Tensor) -> Optional[torch.Tensor]:
    """The [rows, TERMS * H] view the copies of the rows ``x`` go to, or None when none is reserved for them."""
    rows, hidden_size = x.shape
    full = _BUFFERS.get((x.device, hidden_size))
    if full is None or not 0 < rows <= MAX_ROWS:
        return None
    return full[:rows]


def _served_operand(x: torch.Tensor, residual: torch.Tensor) -> Optional[torch.Tensor]:
    """The operand of an add + RMSNorm of ``x`` and ``residual`` the kernel serves, or None."""
    if is_torch_compiling() or x.dim() != 2 or residual.shape != x.shape:
        return None
    for t in (x, residual):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.data_ptr() % 16 != 0:
            return None
    return operand(x)


def _norm_forward(
    self: RMSNorm,
    hidden_states: torch.Tensor,
    residual=RMSNorm._ARGUMENT_NOT_SPECIFIED_SENTINEL,
    *,
    return_norm_out: bool = False,
):
    """``RMSNorm.forward``, the add + RMSNorm by ``add_rmsnorm_replicated`` with the gate's operand where the kernel
    serves it, by ``add_rmsnorm_replicated_mxfp8`` where the norm also carries the shared gate_up_proj hand-off."""
    out = None
    if isinstance(residual, torch.Tensor) and not return_norm_out and self.nvfp4_scale is None:
        out = _served_operand(hidden_states, residual)
    if out is None:
        return RMSNorm.forward(self, hidden_states, residual, return_norm_out=return_norm_out)
    handoff = _shared.gate_up_handoff(self, hidden_states)
    if handoff is None:
        kernel()(
            hidden_states,
            residual,
            self.weight,
            self.variance_epsilon,
            out,
            enable_pdl=get_env_enable_pdl(),
        )
    else:
        _shared.norm_kernel()(
            hidden_states,
            residual,
            self.weight,
            self.variance_epsilon,
            out,
            *handoff,
            enable_pdl=get_env_enable_pdl(),
        )
        _shared.publish_gate_up(hidden_states, handoff)
    _log_once("published", "post-attention add + RMSNorm")
    SLOT.publish(hidden_states, out)
    return hidden_states, residual


def _runs_the_cat_gemm(gate: MiniMaxM3Gate, x: torch.Tensor) -> bool:
    """Whether ``MiniMaxM3Gate.forward`` runs ``x`` as the cuBLAS GEMM of ``cat([x] * TERMS, -1)``."""
    split_weight = gate.split_weight
    return (
        split_weight is not None
        and x.dtype == torch.bfloat16
        and x.shape[1] == gate.hidden_size
        and split_weight.shape[1] == TERMS * gate.hidden_size
    )


def _router_gemm(operand: torch.Tensor, split_weight: torch.Tensor) -> torch.Tensor:
    """``MiniMaxM3Gate.forward``'s GEMM with the ``cat`` replaced by ``operand``."""
    return torch.ops.trtllm.cublas_mm(operand, split_weight.t(), bias=None, out_dtype=torch.float32)


def _gate_forward(self: MiniMaxM3Gate, hidden_states: torch.Tensor) -> torch.Tensor:
    """``MiniMaxM3Gate.forward``, the GEMM on the post-attention norm's operand when the input is its output."""
    pending = SLOT.pending
    taken = SLOT.take(hidden_states) if pending is not None else None
    if pending is not None and taken is None:
        _log_once(
            "missed",
            f"gate input {tuple(hidden_states.shape)} is not the {pending[1]}-row post-attention norm output",
        )
    if taken is None or not _runs_the_cat_gemm(self, hidden_states):
        return MiniMaxM3Gate.forward(self, hidden_states)
    _log_once("taken", f"router GEMM on the norm's copies, {hidden_states.shape[0]} rows")
    return _router_gemm(taken, self.split_weight)


def _wrapped(module: Optional[torch.nn.Module], forward) -> bool:
    return (
        module is not None and getattr(module.__dict__.get("forward"), "__func__", None) is forward
    )


def is_site(layer: torch.nn.Module) -> bool:
    """Whether ``install`` wrapped ``layer``'s post-attention norm and gate."""
    norm = getattr(layer, "post_attention_layernorm", None)
    gate = getattr(getattr(layer, "block_sparse_moe", None), "gate", None)
    return _wrapped(norm, _norm_forward) and _wrapped(gate, _gate_forward)


def site_decline_reason(layer: torch.nn.Module) -> Optional[str]:
    """Why ``layer``'s gate keeps the ``torch.cat`` operand, or None if its post-attention norm may write it."""
    gate = getattr(getattr(layer, "block_sparse_moe", None), "gate", None)
    if type(gate) is not MiniMaxM3Gate:
        return "no MiniMax-M3 MoE gate"
    split_weight = gate.split_weight
    if split_weight is None or split_weight.shape[1] != TERMS * gate.hidden_size:
        return f"the gate's weight is not split into {TERMS} BF16 terms"
    if layer.pre_feed_forward_fusion:
        return "the post-attention norm is fused with an AllReduce"
    norm = getattr(layer, "post_attention_layernorm", None)
    reason = flashinfer_norm_decline_reason(norm)
    if reason is not None:
        return reason
    if norm.weight.numel() != gate.hidden_size:
        return "the post-attention norm and the gate disagree on the hidden size"
    if "forward" in norm.__dict__ and not _wrapped(norm, _norm_forward):
        return "the norm's forward is already replaced"
    if "forward" in gate.__dict__ and not _wrapped(gate, _gate_forward):
        return "the gate's forward is already replaced"
    return None


def install(layers: Iterable[torch.nn.Module]) -> int:
    """Wrap the post-attention norm and the gate of each MoE layer ``site_decline_reason`` admits; returns how many
    layers carry the operand, including any an earlier call installed (0 with TRTLLM_M31_DISABLE_DEP_ROUTER_OPERAND=1).
    Logs each distinct decline once."""
    if not DEP_ROUTER_OPERAND:
        return 0
    count = 0
    for layer in layers:
        reason = site_decline_reason(layer)
        if reason is not None:
            _log_once("not installed", reason)
            continue
        norm = layer.post_attention_layernorm
        gate = layer.block_sparse_moe.gate
        norm.forward = types.MethodType(_norm_forward, norm)
        gate.forward = types.MethodType(_gate_forward, gate)
        count += 1
    return count
