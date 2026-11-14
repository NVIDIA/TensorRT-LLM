# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOFTMAX_MC: the spec sampler's softmax, bitwise ``flashinfer.sampling.softmax(logits, temperature)``, through the
sampler's softmax override (``sampler/ops/flashinfer.set_softmax_override``).

Two in-tree SM107 operations, each FlashInfer's arithmetic on the shapes it takes:

* ``mv2_minimax_m3::minimax_m31_softmax_mc`` (kernels/minimaxM31/softmaxMcSm103.cu) at 1..128 rows, FlashInfer's Path A
  (rows <= 128, vocab >= 24576): FlashInfer's Map kernel, then its Reduce prologue in ``write_ctas`` CTAs per row (16 up
  to 12 rows, 8 up to 24, 4 up to 128), each writing its share of the row (FlashInfer: one CTA per row).
* ``mv2_minimax_m3::minimax_m31_softmax_pathb`` (thop/minimaxM31/minimaxM31SoftmaxPathBOp.cpp) at 129..448 rows of vocab
  200064, FlashInfer's Path B: one CTA per row with FlashInfer's ordered denominator recurrence and final CUB sum, warp
  integer maxima for the per-chunk max.

With 7 draft tokens the draft's 19..48 rows take the MC kernel and the verify's 152..384 rows take Path B, both bitwise
FlashInfer's softmax.

``softmax_mc`` returns None, and the sampler keeps the stock FlashInfer call, for any other shape, an input outside
the kernels' contract, a build without the operation, or a first MC call inside a graph capture. Kill switches:
``TRTLLM_M31_DISABLE_SOFTMAX_MC=1`` installs no override (stock FlashInfer softmax);
``TRTLLM_M31_DISABLE_SOFTMAX_PATHB=1`` keeps the 129..448-row calls on FlashInfer. Read at import.

Workspace (MC only): one partials buffer per device (128 rows x ceil(2^20 / 8192) float2, 128 KiB), kept for the life
of the process (captured graphs keep its address). Every launch overwrites what it reads. ``prebuild`` allocates it
outside any capture.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm.logger import logger

from .m31_switches import m31_disabled

SOFTMAX_MC = not m31_disabled("SOFTMAX_MC")
PATHB = SOFTMAX_MC and not m31_disabled("SOFTMAX_PATHB")

# FlashInfer's Path A bounds (OnlineSoftmax: batch <= 128 and d >= 24576), its slice, and the widest vocab the MC
# workspace holds.
MC_MAX_ROWS = 128
PATH_A_MIN_VOCAB = 24576
SLICE = 8192
MAX_VOCAB = 1 << 20
# (largest row count, write CTAs per row).
WRITE_CTAS_BY_ROWS = ((12, 16), (24, 8), (MC_MAX_ROWS, 4))
# The Path-B operation's shapes.
PATHB_MIN_ROWS = 129
PATHB_MAX_ROWS = 448
PATHB_VOCAB = 200064

# The resolved operations, decided once per process.
_STATE: dict = {}
# Per device index: the MC partials workspace.
_WORKSPACES: dict = {}


def mc_op():
    """``mv2_minimax_m3::minimax_m31_softmax_mc`` when this build carries it, else None."""
    if "mc" not in _STATE:
        _STATE["mc"] = getattr(family_ops(), "minimax_m31_softmax_mc", None)
    return _STATE["mc"]


def pathb_op():
    """``mv2_minimax_m3::minimax_m31_softmax_pathb`` when this build carries it, else None; never resolved inside a
    capture."""
    if "pathb" not in _STATE:
        if torch.cuda.is_current_stream_capturing():
            return None
        _STATE["pathb"] = getattr(family_ops(), "minimax_m31_softmax_pathb", None)
    return _STATE["pathb"]


def write_ctas(rows: int) -> int:
    """The MC kernel's write CTAs per row for a call of ``rows`` rows (1..MC_MAX_ROWS)."""
    for top, ctas in WRITE_CTAS_BY_ROWS:
        if rows <= top:
            return ctas
    return WRITE_CTAS_BY_ROWS[-1][1]


def partials_numel(rows: int, d: int) -> int:
    """fp32 elements of the MC partials for rows x d (one float2 per row and 8192-wide slice)."""
    return 2 * rows * ((d + SLICE - 1) // SLICE)


def prebuild(device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """Resolve both operations and allocate (once) the device's MC workspace; must run outside a graph capture.
    Returns the workspace, or None when the MC operation is not built or the stream is capturing."""
    if PATHB:
        pathb_op()
    if mc_op() is None:
        return None
    dev = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if dev.index not in _WORKSPACES:
        if torch.cuda.is_current_stream_capturing():
            return None
        _WORKSPACES[dev.index] = torch.empty(
            partials_numel(MC_MAX_ROWS, MAX_VOCAB), dtype=torch.float32, device=dev
        )
    return _WORKSPACES[dev.index]


def _temperature_args(temperature, logits: torch.Tensor):
    """``(temp_arr, temp_val)`` as both operations take FlashInfer's temperature, or None when it is outside their
    contract (anything but None, a Python number, or a contiguous fp32 [rows] tensor on logits' device)."""
    if temperature is None:
        return None, 1.0
    if isinstance(temperature, torch.Tensor):
        if (
            temperature.dtype != torch.float32
            or temperature.dim() != 1
            or temperature.shape[0] != logits.shape[0]
            or not temperature.is_contiguous()
            or temperature.device != logits.device
        ):
            return None
        return temperature, 0.0
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        return None, float(temperature)
    return None


def route(rows: int, vocab: int) -> Optional[str]:
    """The operation a call of rows x vocab goes to ("pathb" / "mc"), or None for the stock call."""
    if PATHB and PATHB_MIN_ROWS <= rows <= PATHB_MAX_ROWS and vocab == PATHB_VOCAB:
        return "pathb"
    if 1 <= rows <= MC_MAX_ROWS and PATH_A_MIN_VOCAB <= vocab <= MAX_VOCAB:
        return "mc"
    return None


def softmax_mc(
    logits: torch.Tensor, temperature: Union[torch.Tensor, float, None]
) -> Optional[torch.Tensor]:
    """The sampler's softmax override: FlashInfer's softmax of fp32 logits [rows, vocab] at ``temperature`` bit for
    bit, or None (stock FlashInfer)."""
    if (
        not isinstance(logits, torch.Tensor)
        or not logits.is_cuda
        or logits.dtype != torch.float32
        or logits.dim() != 2
        or not logits.is_contiguous()
        or logits.shape[1] % 4 != 0
        or logits.data_ptr() % 16 != 0
    ):
        return None
    rows, vocab = logits.shape
    path = route(rows, vocab)
    temps = None if path is None else _temperature_args(temperature, logits)
    if temps is None:
        return None
    temp_arr, temp_val = temps
    if path == "pathb":
        op = pathb_op()
        if op is None:
            return None
        out = torch.empty_like(logits)
        op(logits, temp_arr, temp_val, out)
        return out
    op = mc_op()
    partials = None if op is None else prebuild(logits.device)
    if partials is None:
        return None
    out = torch.empty_like(logits)
    op(logits, temp_arr, temp_val, partials, out, write_ctas(rows))
    return out


def engagement() -> str:
    """The log line's state: what the override takes in this process."""
    if not SOFTMAX_MC:
        return "off (TRTLLM_M31_DISABLE_SOFTMAX_MC=1); stock FlashInfer softmax"
    if mc_op() is None:
        return "not taken: the op is not built; stock FlashInfer softmax"
    pathb = (
        f"Path B at {PATHB_MIN_ROWS}..{PATHB_MAX_ROWS} rows"
        if PATHB
        else "Path B off (TRTLLM_M31_DISABLE_SOFTMAX_PATHB=1)"
    )
    return f"engaged (MC at 1..{MC_MAX_ROWS} rows, write CTAs 16 / 8 / 4; {pathb})"


def log_engagement() -> None:
    logger.info(f"[M3.1 DEP] SOFTMAX_MC: {engagement()}")
