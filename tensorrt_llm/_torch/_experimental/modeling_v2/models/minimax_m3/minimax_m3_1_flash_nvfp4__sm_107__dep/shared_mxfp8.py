# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_SHARED_MXFP8: the MoE shared expert's two MXFP8 GEMMs take their activations from the kernels before them
instead of quantizing.

* gate_up_proj (DEP_SHARED_GU_MXFP8): at a DEP_ROUTER_OPERAND site, the MoE layer's post-attention add + RMSNorm runs
  ``add_rmsnorm_replicated_mxfp8`` (kernels/megamoe), which also writes the normalized rows' MXFP8 form, and publishes
  it in ``GATE_UP_SLOT`` keyed by its output (``router_operand``).
* down_proj (DEP_SHARED_DOWN_MXFP8): the shared expert's SwiGLU runs ``swiglu_mxfp8`` (kernels/megamoe), which also
  writes its output's MXFP8 form, and publishes it in ``DOWN_SLOT`` keyed by that output.

The norm is bitwise FlashInfer's where ``megamoe_topk_reduce_add_rmsnorm_matches_flashinfer`` holds, the SwiGLU is
bitwise ``torch.ops.trtllm.silu_and_mul``, and both MXFP8 forms are bitwise both quantize tactics of the decode-graph
``mxfp8_quantize_gemm_autotuned``. Each consumer is a ``merge_mxfp8.DepMergeMxfp8LinearMethod`` on its slot, which runs
the GEMM the stock apply would run on the same operands when its input is the published output and the stock apply
otherwise. A norm call or a SwiGLU the kernels do not serve (more than MAX_ROWS rows, operands that are not contiguous
16-byte-aligned BF16 rows, torch.compile, LoRA) is stock and publishes nothing. Each rank logs once which producers
published and, per step width, whether a projection took the hand-off or missed a published one.

``TRTLLM_M31_DISABLE_DEP_SHARED_MXFP8=1`` installs neither hand-off; ``TRTLLM_M31_DISABLE_DEP_SHARED_GU_MXFP8=1`` or
``TRTLLM_M31_DISABLE_DEP_SHARED_DOWN_MXFP8=1`` keeps that one stock. Read at import.

DEP_SHARED_PIN: both GEMMs run inside ``mxfp8_cute_pinned_tactics.engaged_for``, so the SM107 pinned CuTe DSL table
serves them at 33..64 rows (65..128 rows stay on DEP_GEMM_PROVIDER's cuBLASLt) through the table's run() in the
hand-off apply. Both take the table's entries: gate_up's own bucket-64 tactic GATE_UP_PIN ((128, 64), (1, 4), no swap,
prefetch, split 1) is the faster tactic for the GEMM alone behind the post-attention norm, not inside the full chain
(norm -> gate_up -> SwiGLU -> down), so gate_up carries no override (GATE_UP_PIN stays defined, unused). Every pinned
tactic is non-split-K, so the outputs are the default tactic's bit for bit. ``TRTLLM_M31_DISABLE_DEP_SHARED_PIN=1``
engages nothing: the GEMMs take the engine's choice. On DEP_SE_SIDE's 28-SM side stream (``se_side``) both GEMMs take
the table's own 2-CTA-cluster entries instead (no gate_up override: a 4-CTA cluster cannot launch on 28 SMs), still
bitwise the default tactic. Read at import."""

from __future__ import annotations

import types
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as _pin,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod
from tensorrt_llm._torch.utils import is_torch_compiling
from tensorrt_llm.logger import logger

from . import se_side as _se_side
from .m31_switches import m31_disabled
from .merge_mxfp8 import MAX_ROWS, DepMergeMxfp8LinearMethod, HandoffSlot, scale_bytes

DEP_SHARED_MXFP8 = not m31_disabled("DEP_SHARED_MXFP8")
DEP_SHARED_GU_MXFP8 = DEP_SHARED_MXFP8 and not m31_disabled("DEP_SHARED_GU_MXFP8")
DEP_SHARED_DOWN_MXFP8 = DEP_SHARED_MXFP8 and not m31_disabled("DEP_SHARED_DOWN_MXFP8")
DEP_SHARED_PIN = not m31_disabled("DEP_SHARED_PIN")
#: gate_up's own bucket-64 tactic, not engaged (it does not help inside the full chain); gate_up takes the table's.
GATE_UP_PIN = {(6144, 6144, 64): ((128, 64), (1, 4), False, True, 1)}

# The (outcome, detail) pairs already logged: one line per producer that publishes and per distinct step width a
# projection takes or misses the hand-off.
_LOGGED: set = set()


def _log_once(outcome: str, detail: str) -> None:
    if (outcome, detail) not in _LOGGED:
        _LOGGED.add((outcome, detail))
        logger.info(f"[M3.1 DEP] DEP_SHARED_MXFP8 {outcome}: {detail}")


GATE_UP_SLOT = HandoffSlot()
DOWN_SLOT = HandoffSlot()

# Marks a post-attention norm whose add + RMSNorm writes its shared expert's gate_up_proj hand-off.
_GATE_UP_NORM = "_m31_shared_gate_up_handoff"

# Per (device, width): (E4M3 [MAX_ROWS, width], scales of MAX_ROWS rows), allocated outside graph capture. The widths
# are the ones ``install`` recorded: the hidden size for gate_up_proj, the shared intermediate size for down_proj.
_BUFFERS: dict = {}
_WIDTHS: set = set()


def norm_kernel():
    """``add_rmsnorm_replicated_mxfp8``: the family CUDA C++ add + RMSNorm (NATIVE_ADD_RMSNORM,
    ``minimax_m3/ops/add_rmsnorm``, bitwise the CuTe DSL kernel it falls back to), imported on first use."""
    from ..ops.add_rmsnorm import add_rmsnorm_replicated_mxfp8

    return add_rmsnorm_replicated_mxfp8


def swiglu_kernel():
    """``swiglu_mxfp8``, imported on first use."""
    from ..kernels.megamoe.swiglu_mxfp8 import swiglu_mxfp8

    return swiglu_mxfp8


def reserve(device: torch.device) -> int:
    """Allocate the hand-off buffers of every width ``install`` recorded on ``device`` (scales zeroed) once, outside
    any graph capture; returns their bytes."""
    total = 0
    for width in sorted(_WIDTHS):
        key = (device, width)
        if key not in _BUFFERS:
            _BUFFERS[key] = (
                torch.empty((MAX_ROWS, width), dtype=torch.float8_e4m3fn, device=device),
                torch.zeros((scale_bytes(MAX_ROWS, width),), dtype=torch.uint8, device=device),
            )
        q8, sf = _BUFFERS[key]
        total += q8.numel() + sf.numel()
    return total


def buffers(
    rows: int, width: int, device: torch.device
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The (q8, sf) views the MXFP8 form of ``rows`` rows of ``width`` goes to, or None when none are reserved."""
    full = _BUFFERS.get((device, width))
    if full is None or not 0 < rows <= MAX_ROWS:
        return None
    return full[0][:rows], full[1][: scale_bytes(rows, width)]


def _served_rows(x: torch.Tensor) -> bool:
    return (
        not is_torch_compiling()
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
    )


def gate_up_handoff(
    norm: torch.nn.Module, x: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The buffers the post-attention norm ``norm`` writes the MXFP8 form of its rows ``x`` to for its shared
    expert's gate_up_proj, or None when ``norm`` carries no hand-off or the rows do not fit it."""
    if not norm.__dict__.get(_GATE_UP_NORM, False) or not _served_rows(x):
        return None
    return buffers(x.shape[0], x.shape[1], x.device)


def publish_gate_up(x: torch.Tensor, handoff: Tuple[torch.Tensor, torch.Tensor]) -> None:
    """Publish the MXFP8 form ``handoff`` of the post-attention norm's rows ``x`` for the shared gate_up_proj."""
    _log_once("published", "post-attention add + RMSNorm")
    GATE_UP_SLOT.publish(x, *handoff)


class _DepSharedPinnedMxfp8LinearMethod(DepMergeMxfp8LinearMethod):
    """The hand-off apply inside DEP_SHARED_PIN's engagement of the pinned table, with ``pin_overrides`` ahead of it."""

    pin_overrides = None

    def apply(self, module, input: torch.Tensor, bias: Optional[torch.Tensor]):
        if not DEP_SHARED_PIN:
            return super().apply(module, input, bias)
        if _se_side.on_side():
            # DEP_SE_SIDE's 28-SM stream: the table's 2-CTA-cluster entries (a 4-CTA cluster cannot launch there).
            with _pin.engaged_for(
                f"DEP_SHARED_PIN {self.consumer} (DEP_SE_SIDE)", _se_side.SIDE_PIN_OVERRIDES
            ):
                return super().apply(module, input, bias)
        with _pin.engaged_for(f"DEP_SHARED_PIN {self.consumer}", self.pin_overrides):
            return super().apply(module, input, bias)


class DepSharedGateUpMxfp8LinearMethod(_DepSharedPinnedMxfp8LinearMethod):
    """MXFP8LinearMethod that runs the shared gate_up_proj's GEMM on the post-attention norm's MXFP8 hand-off when
    its input is the published normalized rows; every other call is the stock apply."""

    slot = GATE_UP_SLOT
    consumer = "shared gate_up_proj"
    producer = "post-attention norm"
    pin_overrides = None  # not GATE_UP_PIN: it does not help inside the full chain
    _log = staticmethod(_log_once)


class DepSharedDownMxfp8LinearMethod(_DepSharedPinnedMxfp8LinearMethod):
    """MXFP8LinearMethod that runs the shared down_proj's GEMM on the SwiGLU's MXFP8 hand-off when its input is the
    published SwiGLU output; every other call is the stock apply."""

    slot = DOWN_SLOT
    consumer = "shared down_proj"
    producer = "SwiGLU"
    _log = staticmethod(_log_once)


def swiglu_constants(mlp: GatedMLP) -> Tuple[float, float, Optional[float]]:
    """(alpha, beta, limit) as ``torch.ops.trtllm.silu_and_mul`` applies the GatedMLP's SwiGLU settings: alpha 1 and
    beta 0 when unset, no clamp unless the limit is above 0."""
    alpha = 1.0 if mlp.swiglu_alpha is None else float(mlp.swiglu_alpha)
    beta = 0.0 if mlp.swiglu_beta is None else float(mlp.swiglu_beta)
    limit = mlp.swiglu_limit
    return alpha, beta, (float(limit) if limit is not None and limit > 0.0 else None)


def _shared_forward(
    self: GatedMLP,
    x: torch.Tensor,
    all_rank_num_tokens=None,
    final_all_reduce_params=None,
    lora_params: Optional[dict] = None,
    **kwargs,
) -> torch.Tensor:
    """``GatedMLP.forward`` on the path ``shared_decline_reason`` admits, the SwiGLU by ``swiglu_mxfp8`` with the
    down_proj hand-off where the kernel serves the rows; LoRA calls and the rows it does not serve are stock."""
    if bool(lora_params):
        return GatedMLP.forward(
            self, x, all_rank_num_tokens, final_all_reduce_params, lora_params, **kwargs
        )
    h1 = self.gate_up_proj(x)
    handoff = (
        buffers(h1.shape[0], h1.shape[1] // 2, h1.device)
        if _served_rows(h1) and h1.shape[1] % 2 == 0
        else None
    )
    if handoff is None:
        h2 = self._apply_activation(h1)
    else:
        alpha, beta, limit = swiglu_constants(self)
        h2 = torch.empty((h1.shape[0], h1.shape[1] // 2), dtype=h1.dtype, device=h1.device)
        swiglu_kernel()(
            h1, h2, *handoff, alpha=alpha, beta=beta, limit=limit, enable_pdl=get_env_enable_pdl()
        )
        _log_once("published", "shared expert SwiGLU")
        DOWN_SLOT.publish(h2, *handoff)
    return self.down_proj(h2, all_reduce_params=final_all_reduce_params, layer_idx=self.layer_idx)


def _wrapped(module: torch.nn.Module, forward) -> bool:
    return getattr(module.__dict__.get("forward"), "__func__", None) is forward


def _mxfp8(linear: torch.nn.Module, method: type) -> bool:
    return type(getattr(linear, "quant_method", None)) in (MXFP8LinearMethod, method)


def shared_decline_reason(mlp: Optional[torch.nn.Module]) -> Optional[str]:
    """Why the shared expert ``mlp`` keeps both stock quantizes, or None if it is the MXFP8 SwiGLU GatedMLP whose
    forward runs gate_up_proj, ``_apply_activation`` and down_proj."""
    if type(mlp) is not GatedMLP:
        return f"shared expert {type(mlp).__name__}"
    if mlp.split_gate_up or mlp.activation is not F.silu:
        return "the shared expert is not one fused gate_up_proj with a SwiGLU"
    if mlp._can_fuse_gate_up_swiglu() or mlp._can_fuse_swiglu_fp8_quant():
        return "the shared expert fuses its SwiGLU elsewhere"
    if mlp.down_proj.has_fp8_qdq or mlp.down_proj.has_w4a8_nvfp4_fp8:
        return "the shared expert's SwiGLU writes FP8"
    if not _mxfp8(mlp.gate_up_proj, DepSharedGateUpMxfp8LinearMethod) or not _mxfp8(
        mlp.down_proj, DepSharedDownMxfp8LinearMethod
    ):
        return "the shared expert's projections are not MXFP8"
    if mlp.down_proj.in_features % 128 != 0 or mlp.gate_up_proj.in_features % 128 != 0:
        return "the shared expert's widths are not multiples of 128"
    return None


def install(
    layers: Iterable[torch.nn.Module], router_site: Callable[[torch.nn.Module], bool]
) -> Tuple[int, int]:
    """Install the hand-offs on each MoE layer whose shared expert ``shared_decline_reason`` admits: the gate_up_proj
    one where ``router_site(layer)`` (its post-attention norm is a DEP_ROUTER_OPERAND site), the down_proj one
    everywhere. Returns how many layers carry each, (gate_up, down), including any an earlier call installed; (0, 0)
    with TRTLLM_M31_DISABLE_DEP_SHARED_MXFP8=1. Logs each distinct decline once."""
    if not DEP_SHARED_MXFP8:
        return 0, 0
    gate_up = down = 0
    for layer in layers:
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        mlp = getattr(moe, "shared_experts", None)
        reason = shared_decline_reason(mlp)
        if reason is not None:
            _log_once("not installed", reason)
            continue
        if DEP_SHARED_GU_MXFP8:
            norm = layer.post_attention_layernorm
            if not router_site(layer):
                _log_once(
                    "gate_up_proj not installed",
                    "the post-attention norm is not a DEP_ROUTER_OPERAND site",
                )
            elif norm.weight.numel() != mlp.gate_up_proj.in_features:
                _log_once(
                    "gate_up_proj not installed",
                    "the post-attention norm and the shared expert disagree on the width",
                )
            else:
                norm.__dict__[_GATE_UP_NORM] = True
                mlp.gate_up_proj.quant_method.__class__ = DepSharedGateUpMxfp8LinearMethod
                _WIDTHS.add(mlp.gate_up_proj.in_features)
                gate_up += 1
        if DEP_SHARED_DOWN_MXFP8:
            if "forward" in mlp.__dict__ and not _wrapped(mlp, _shared_forward):
                _log_once(
                    "down_proj not installed", "the shared expert's forward is already replaced"
                )
            else:
                mlp.forward = types.MethodType(_shared_forward, mlp)
                mlp.down_proj.quant_method.__class__ = DepSharedDownMxfp8LinearMethod
                _WIDTHS.add(mlp.down_proj.in_features)
                down += 1
    return gate_up, down
