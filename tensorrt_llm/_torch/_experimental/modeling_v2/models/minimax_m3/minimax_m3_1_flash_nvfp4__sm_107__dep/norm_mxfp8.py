# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_NORM_MXFP8: the qkv_proj after a layer boundary takes its MXFP8 activation from the boundary add + RMSNorm
instead of quantizing.

The stock boundary into layer i >= 1 runs FlashInfer's fused add + RMSNorm (layer i's ``input_layernorm``, layer i - 1's
``next_layer_layernorm``), then layer i's MXFP8 qkv_proj quantizes the normalized rows (``trtllm::mxfp8_quantize(x,
True)``). At a site, the norm runs ``add_rmsnorm_mxfp8`` (kernels/megamoe), which also writes the rows' MXFP8 form,
and a boundary DEP_TAIL's form (b) runs hands that form the same buffers (``tail``). Both are bitwise the stock pair
where ``megamoe_topk_reduce_add_rmsnorm_matches_flashinfer`` holds; the MXFP8 rows are bitwise both quantize tactics of
the decode-graph ``mxfp8_quantize_gemm_autotuned`` (``merge_mxfp8``). The norm publishes the pair in ``SLOT`` keyed by
its output; ``DepNormMxfp8LinearMethod`` (``merge_mxfp8.DepMergeMxfp8LinearMethod`` on this slot) takes it when its
input is that output. A call the kernel does not serve (no residual, more than MAX_ROWS rows, an operand that is not
contiguous 16-byte-aligned BF16 rows, torch.compile) is the stock norm and publishes nothing. Each rank logs once
which producers published and, per step width, whether a qkv_proj took the hand-off or missed a published one.

``TRTLLM_M31_DISABLE_DEP_NORM_MXFP8=1`` wraps no norm and swaps no qkv_proj, so nothing is reserved or published.
Read at import.

NORM_L2PF: at a site, the norm's CTAs also pull the first quarter of the next qkv_proj's MXFP8 weight and of
its weight scales into L2 before the PDL wait, at 65..128 rows (``ops/add_rmsnorm.l2pf_ranges``); the outputs are
unchanged. A boundary whose norm a DEP_TAIL form (b) runs does not prefetch (the tail kernel runs that norm).
``TRTLLM_M31_DISABLE_NORM_L2PF=1`` arms no site. Read at import.
"""

from __future__ import annotations

import types
from typing import Iterable, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import is_torch_compiling
from tensorrt_llm.logger import logger

from .m31_switches import m31_disabled
from .merge_mxfp8 import MAX_ROWS, DepMergeMxfp8LinearMethod, HandoffSlot, scale_bytes

DEP_NORM_MXFP8 = not m31_disabled("DEP_NORM_MXFP8")
NORM_L2PF = not m31_disabled("NORM_L2PF")
# The site attribute holding (qkv_proj,) of the next layer when NORM_L2PF armed the site (a tuple: no submodule).
_L2PF_TARGET = "_m31_l2pf_qkv"

# The (outcome, detail) pairs already logged: one line per producer that publishes and per distinct step width a
# qkv_proj takes or misses the hand-off.
_LOGGED: set = set()


def _log_once(outcome: str, detail: str) -> None:
    if (outcome, detail) not in _LOGGED:
        _LOGGED.add((outcome, detail))
        logger.info(f"[M3.1 DEP] DEP_NORM_MXFP8 {outcome}: {detail}")


def _log_l2pf_once(detail: str) -> None:
    if ("NORM_L2PF", detail) not in _LOGGED:
        _LOGGED.add(("NORM_L2PF", detail))
        logger.info(f"[M3.1 DEP] NORM_L2PF: {detail}")


SLOT = HandoffSlot()

# Per (device, hidden size): (E4M3 [MAX_ROWS, H], scales of MAX_ROWS rows), allocated outside graph capture.
_BUFFERS: dict = {}


def kernel():
    """``add_rmsnorm_mxfp8``: the family CUDA C++ add + RMSNorm (NATIVE_ADD_RMSNORM, ``minimax_m3/ops/add_rmsnorm``,
    bitwise the CuTe DSL kernel it falls back to), imported on first use."""
    from ..ops.add_rmsnorm import add_rmsnorm_mxfp8

    return add_rmsnorm_mxfp8


def l2pf_prefetch(norm: torch.nn.Module, rows: int):
    """The ``prefetch`` argument for site ``norm``'s call of ``rows`` rows: the next qkv_proj's weight + scales when
    NORM_L2PF armed the site and ``rows`` is in the band, else None."""
    target = norm.__dict__.get(_L2PF_TARGET)
    if target is None:
        return None
    weight = getattr(target[0], "weight", None)
    weight_scale = getattr(target[0], "weight_scale", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(weight_scale, torch.Tensor):
        return None
    from ..ops.add_rmsnorm import l2pf_ranges

    prefetch = l2pf_ranges(rows, weight, weight_scale)
    if prefetch is not None:
        _log_l2pf_once(f"taken at {rows} rows")
    return prefetch


def reserve(device: torch.device, hidden_size: int) -> int:
    """Allocate the hand-off buffers of ``hidden_size``-wide rows on ``device`` (scales zeroed) once, outside any graph
    capture; returns their bytes (0 with TRTLLM_M31_DISABLE_DEP_NORM_MXFP8=1)."""
    if not DEP_NORM_MXFP8:
        return 0
    key = (device, hidden_size)
    if key not in _BUFFERS:
        _BUFFERS[key] = (
            torch.empty((MAX_ROWS, hidden_size), dtype=torch.float8_e4m3fn, device=device),
            torch.zeros((scale_bytes(MAX_ROWS, hidden_size),), dtype=torch.uint8, device=device),
        )
    q8, sf = _BUFFERS[key]
    return q8.numel() + sf.numel()


def buffers(x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The (q8, sf) views the MXFP8 form of the rows ``x`` goes to, or None when none are reserved for them."""
    rows, hidden_size = x.shape
    full = _BUFFERS.get((x.device, hidden_size))
    if full is None or not 0 < rows <= MAX_ROWS:
        return None
    return full[0][:rows], full[1][: scale_bytes(rows, hidden_size)]


def _handoff(
    x: torch.Tensor, residual: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The buffers of an add + RMSNorm of ``x`` and ``residual`` the kernel serves, or None."""
    if is_torch_compiling() or x.dim() != 2 or residual.shape != x.shape:
        return None
    for t in (x, residual):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.data_ptr() % 16 != 0:
            return None
    return buffers(x)


def publish(x: torch.Tensor, handoff: Tuple[torch.Tensor, torch.Tensor], producer: str) -> None:
    """Publish the MXFP8 form ``handoff`` of the normalized rows ``x`` for the next qkv_proj."""
    _log_once("published", producer)
    SLOT.publish(x, *handoff)


def _norm_forward(
    self: RMSNorm,
    hidden_states: torch.Tensor,
    residual=RMSNorm._ARGUMENT_NOT_SPECIFIED_SENTINEL,
    *,
    return_norm_out: bool = False,
):
    """``RMSNorm.forward``, the add + RMSNorm by ``add_rmsnorm_mxfp8`` with the hand-off where the kernel serves it."""
    handoff = None
    if isinstance(residual, torch.Tensor) and not return_norm_out and self.nvfp4_scale is None:
        handoff = _handoff(hidden_states, residual)
    if handoff is None:
        return RMSNorm.forward(self, hidden_states, residual, return_norm_out=return_norm_out)
    kernel()(
        hidden_states,
        residual,
        self.weight,
        self.variance_epsilon,
        *handoff,
        enable_pdl=get_env_enable_pdl(),
        prefetch=l2pf_prefetch(self, hidden_states.shape[0]),
    )
    publish(hidden_states, handoff, "boundary add + RMSNorm")
    return hidden_states, residual


def is_site(norm: torch.nn.Module) -> bool:
    """Whether ``norm`` is a boundary norm ``install`` wrapped."""
    return getattr(norm.__dict__.get("forward"), "__func__", None) is _norm_forward


def site_handoff(
    norm: torch.nn.Module, x: torch.Tensor, residual: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The buffers another kernel running site ``norm``'s add + RMSNorm of ``x`` and ``residual`` writes the hand-off
    to, or None when ``norm`` is not a site or the kernel would not serve the call."""
    return _handoff(x, residual) if is_site(norm) else None


class DepNormMxfp8LinearMethod(DepMergeMxfp8LinearMethod):
    """MXFP8LinearMethod that runs its GEMM on the boundary norm's MXFP8 hand-off when its input is the published
    normalized rows; every other call is the stock apply."""

    slot = SLOT
    consumer = "qkv_proj"
    producer = "boundary norm"
    _log = staticmethod(_log_once)


def site_decline_reason(layer: torch.nn.Module, next_layer: torch.nn.Module) -> Optional[str]:
    """Why the boundary from ``layer`` into ``next_layer`` carries no hand-off, or None if it may."""
    from .tail import boundary_norm_decline_reason

    norm = getattr(next_layer, "input_layernorm", None)
    if norm is None or getattr(layer, "next_layer_layernorm", None) is not norm:
        return "the boundary norm is not the next layer's input norm"
    reason = boundary_norm_decline_reason(layer)
    if reason is not None:
        return reason
    if norm.weight.numel() % 128 != 0:
        return f"hidden size {norm.weight.numel()} is not a multiple of 128"
    if "forward" in norm.__dict__ and not is_site(norm):
        return "the norm's forward is already replaced"
    qkv_proj = getattr(getattr(next_layer, "self_attn", None), "qkv_proj", None)
    method = getattr(qkv_proj, "quant_method", None)
    if type(method) not in (MXFP8LinearMethod, DepNormMxfp8LinearMethod):
        return f"qkv_proj runs {type(method).__name__}"
    return None


def install(layers: Iterable[torch.nn.Module]) -> int:
    """Wrap each boundary norm ``site_decline_reason`` admits and swap the qkv_proj after it to
    ``DepNormMxfp8LinearMethod``; returns how many boundaries carry the hand-off, including any an earlier call
    installed (0 with TRTLLM_M31_DISABLE_DEP_NORM_MXFP8=1). Logs each distinct decline once."""
    if not DEP_NORM_MXFP8:
        return 0
    layers = list(layers)
    count = 0
    for layer, next_layer in zip(layers, layers[1:]):
        reason = site_decline_reason(layer, next_layer)
        if reason is not None:
            _log_once("not installed", reason)
            continue
        norm = next_layer.input_layernorm
        norm.forward = types.MethodType(_norm_forward, norm)
        next_layer.self_attn.qkv_proj.quant_method.__class__ = DepNormMxfp8LinearMethod
        if NORM_L2PF:
            norm.__dict__[_L2PF_TARGET] = (next_layer.self_attn.qkv_proj,)
        count += 1
    _log_l2pf_engagement(count)
    return count


def _log_l2pf_engagement(sites: int) -> None:
    """The NORM_L2PF engagement line (once per distinct state)."""
    from ..ops.add_rmsnorm import L2PF_DIVISOR, L2PF_MAX_ROWS, L2PF_MIN_ROWS, takes_prefetch

    if not NORM_L2PF:
        _log_l2pf_once("off (TRTLLM_M31_DISABLE_NORM_L2PF=1)")
    elif not takes_prefetch():
        _log_l2pf_once("not taken: the add + RMSNorm op has no prefetch arguments (or is off)")
    else:
        _log_l2pf_once(
            f"on: {sites} qkv boundary norms prefetch 1/{L2PF_DIVISOR} of the qkv weight + scales at "
            f"{L2PF_MIN_ROWS}..{L2PF_MAX_ROWS} rows (DEP_TAIL form (b) boundaries do not)"
        )
