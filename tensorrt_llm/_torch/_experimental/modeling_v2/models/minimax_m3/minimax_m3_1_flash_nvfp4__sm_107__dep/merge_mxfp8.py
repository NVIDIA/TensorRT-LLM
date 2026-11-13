# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_MERGE_MXFP8: the o_proj of a layer LC2 serves takes its MXFP8 activation from the decode instead of quantizing.

Given the hand-off buffers, LC2's merge also writes each output row's MXFP8 form (E4M3 rows + UE8M0 scales in the
128x4-swizzled layout of ``trtllm::mxfp8_quantize(x, True)``): bitwise that quantize on every row of the step, and
bitwise FlashInfer's CuTe DSL quantize, the other tactic of the decode-graph ``mxfp8_quantize_gemm_autotuned``.
``lc2_decode.dep_lc2_decode`` publishes the pair in ``SLOT`` keyed by its output; ``DepMergeMxfp8LinearMethod`` takes
it when its input is that output and runs the GEMM the stock apply would run on the same operands
(``mxfp8_gemm_autotuned``, the same op key and tactics, in decode graphs). Every take clears the slot, and so does
every decode call; a miss is the stock apply. Each rank logs once per step width whether its o_proj took the hand-off
or missed a published one.

The step's scales fill whole 128-row tiles; LC2 does not write the rows past the step's (the quantize zero-fills them).
They scale only GEMM rows past the step's, which are not stored.

``TRTLLM_M31_DISABLE_DEP_MERGE_MXFP8=1`` reserves no buffers, so the decode publishes nothing, and swaps no o_proj.
Read at import.

DEP_PROJ_CUTE: where DEP_PROJ_PIN engages the table for the projection (qkv_proj and o_proj, through their hand-offs),
the GEMM runs the pinned CuTe DSL tactic itself rather than the backend the decode-graph AutoTuner picked for the
bucket, which profiles with the weight L2-resident and can pick CUTLASS. Every FlashInfer CUTLASS tactic and the pinned
tactic write the same output bit for bit at these shapes, so only the time changes.
``TRTLLM_M31_DISABLE_DEP_PROJ_CUTE=1`` restores the AutoTuner's choice. Read at import.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as _pin,
)
from tensorrt_llm._torch.custom_ops.torch_custom_ops import mxfp8_gemm_autotuned
from tensorrt_llm._torch.modules import linear as _linear
from tensorrt_llm._torch.modules.linear import Linear, MXFP8LinearMethod
from tensorrt_llm._torch.utils import is_torch_compiling
from tensorrt_llm.logger import logger

from . import gemm_provider as _provider
from .m31_switches import m31_disabled

DEP_MERGE_MXFP8 = not m31_disabled("DEP_MERGE_MXFP8")
DEP_PROJ_CUTE = not m31_disabled("DEP_PROJ_CUTE")

# The (outcome, detail) pairs already logged: one line per distinct step width an o_proj takes or misses the hand-off.
_LOGGED: set = set()


def _log_once(outcome: str, detail: str) -> None:
    if (outcome, detail) not in _LOGGED:
        _LOGGED.add((outcome, detail))
        logger.info(f"[M3.1 DEP] DEP_MERGE_MXFP8 {outcome}: {detail}")


#: LC2's output row at the attention-DP geometry: 4 KV heads x 16 query heads x head_dim 128.
WIDTH = 4 * 16 * 128
#: LC2's widest step (lc2_decode.MAX_ROWS).
MAX_ROWS = 448


def scale_bytes(rows: int, width: int = WIDTH) -> int:
    """The swizzled UE8M0 scales of ``rows`` rows of ``width``: whole 128-row tiles of whole 4-scale columns."""
    return (rows + 127) // 128 * 128 * ((width // 32 + 3) // 4 * 4)


class HandoffSlot:
    """The MXFP8 form of one BF16 output, for the next MXFP8 linear whose input is that output."""

    def __init__(self):
        self.pending: Optional[Tuple[int, int, torch.Tensor, torch.Tensor]] = None

    def publish(self, x: torch.Tensor, q8: torch.Tensor, sf: torch.Tensor) -> None:
        self.pending = (x.data_ptr(), x.shape[0], q8, sf)

    def clear(self) -> None:
        self.pending = None

    def take(self, x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """(E4M3 rows, scales) if ``x`` is the published output, of the E4M3 rows' shape, else None; clears the
        slot."""
        pending, self.pending = self.pending, None
        if pending is None or x.dim() != 2:
            return None
        ptr, rows, q8, sf = pending
        if x.data_ptr() != ptr or x.shape[0] != rows or x.shape[1] != q8.shape[1]:
            return None
        return q8, sf


SLOT = HandoffSlot()

# Per device: (E4M3 [MAX_ROWS, WIDTH], scales of MAX_ROWS rows), allocated outside graph capture.
_BUFFERS: dict = {}


def reserve(device: torch.device) -> int:
    """Allocate the hand-off buffers on ``device`` (scales zeroed) once, outside any graph capture; returns their
    bytes (0 with TRTLLM_M31_DISABLE_DEP_MERGE_MXFP8=1)."""
    if not DEP_MERGE_MXFP8:
        return 0
    if device not in _BUFFERS:
        _BUFFERS[device] = (
            torch.empty((MAX_ROWS, WIDTH), dtype=torch.float8_e4m3fn, device=device),
            torch.zeros((scale_bytes(MAX_ROWS),), dtype=torch.uint8, device=device),
        )
    q8, sf = _BUFFERS[device]
    return q8.numel() + sf.numel()


def buffers(rows: int, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The (q8, sf) views a decode step of ``rows`` rows writes on ``device``, or None when none are reserved there."""
    full = _BUFFERS.get(device)
    if full is None or rows > MAX_ROWS:
        return None
    return full[0][:rows], full[1][: scale_bytes(rows)]


class DepMergeMxfp8LinearMethod(MXFP8LinearMethod):
    """MXFP8LinearMethod that runs its GEMM on the decode's MXFP8 hand-off when its input is the published LC2 output
    (the operands the stock apply would quantize, bit for bit); every other call is the stock apply. The GEMM half
    below is ``MXFP8LinearMethod.apply``'s; a change there must be made here too. A subclass takes another producer's
    hand-off by naming its slot, consumer, producer and log."""

    slot = SLOT
    consumer = "o_proj"
    producer = "decode"
    _log = staticmethod(_log_once)

    def apply(self, module: Linear, input: torch.Tensor, bias: Optional[torch.Tensor]):
        pending = self.slot.pending
        prequant = self.slot.take(input) if pending is not None else None
        if pending is not None and prequant is None:
            self._log(
                "missed",
                f"{self.consumer} input {tuple(input.shape)} is not the {pending[1]}-row "
                f"{self.producer} output",
            )
        if prequant is None or not self.use_cutlass:
            return super().apply(module, input, bias)
        act_e4m3, act_sf = prequant
        # DEP_GEMM_PROVIDER first (o_proj / shared gate_up at the decode rows); None outside its routes.
        output = _provider.run(act_e4m3, act_sf, module.weight, module.weight_scale, module.dtype)
        via = " (DEP_GEMM_PROVIDER)" if output is not None else ""
        # None outside DEP_PROJ_PIN's engagement of this projection, or for a shape / bucket the table does not hold.
        if output is None and DEP_PROJ_CUTE:
            output = _pin.run(act_e4m3, act_sf, module.weight, module.weight_scale, module.dtype)
            via = " (DEP_PROJ_CUTE: the pinned CuTe DSL tactic)" if output is not None else ""
        self._log(
            "taken",
            f"{self.consumer} GEMM on the {self.producer}'s MXFP8 rows, {input.shape[0]} rows"
            + via,
        )
        if output is not None:
            return output if bias is None else output + bias
        if (
            self.tune_decode_graph_backends
            and not is_torch_compiling()
            and _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
        ):
            output = mxfp8_gemm_autotuned(
                act_e4m3,
                act_sf,
                module.weight,
                module.weight_scale,
                module.dtype,
                tune=_linear._FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get(),
            )
        else:
            use_flashinfer = self.backend == "flashinfer" or (
                self.backend == "auto"
                and not is_torch_compiling()
                and (
                    _linear._FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get()
                    or (
                        self._flashinfer_autotuned
                        and _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
                    )
                )
            )
            if use_flashinfer:
                flashinfer_mxfp8 = self._flashinfer_mxfp8
                assert flashinfer_mxfp8 is not None
                output = flashinfer_mxfp8(
                    act_e4m3, act_sf, module.weight, module.weight_scale, module.dtype
                )
            else:
                gemm = (
                    torch.ops.trtllm.mxfp8_mxfp8_gemm_autotuned
                    if self.needs_native_autotune
                    else torch.ops.trtllm.mxfp8_mxfp8_gemm
                )
                output = gemm(
                    act_e4m3,
                    act_sf,
                    module.weight,
                    module.weight_scale,
                    module._mxfp8_alpha,
                    module.dtype,
                )
        if bias is not None:
            output = output + bias
        return output


def install(layers: Iterable[torch.nn.Module], serves) -> int:
    """Swap to ``DepMergeMxfp8LinearMethod`` the MXFP8 o_proj of each layer whose attention ``serves(attn)`` (LC2's
    decode hook is set); returns how many carry it, including any an earlier call swapped (0 with
    TRTLLM_M31_DISABLE_DEP_MERGE_MXFP8=1)."""
    if not DEP_MERGE_MXFP8:
        return 0
    count = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or not serves(attn):
            continue
        method = getattr(getattr(attn, "o_proj", None), "quant_method", None)
        if type(method) is MXFP8LinearMethod:
            method.__class__ = DepMergeMxfp8LinearMethod
        if type(method) is DepMergeMxfp8LinearMethod:
            count += 1
    return count
