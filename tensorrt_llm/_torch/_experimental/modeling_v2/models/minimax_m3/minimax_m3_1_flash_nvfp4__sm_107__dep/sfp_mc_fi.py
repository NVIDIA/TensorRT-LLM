# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SFP_MC_FI: the spec sampler's draft draws as one multi-CTA launch each, bitwise FlashInfer's
``sampling_from_probs(deterministic=True)`` with seed / offset tensors (what ``sampling_from_probs_op`` runs for each
DSpark ancestral draft position), through the sampler's sampling-from-probs override
(``sampler/ops/flashinfer.set_sampling_from_probs_override``).

``native_ops.minimax_m31_sampling_from_probs_mc_fi`` (kernels/minimaxM31/samplingFromProbsMcFiSm107.cu) on
``ctas_per_row`` CTAs per row. FlashInfer runs one CTA per row and stops at the chunk that holds the drawn token, so its
time follows the token's position; this op reads every chunk once, split over the row's CTAs, so its time does not
depend on where the drawn token lies.

The override returns None, and the sampler keeps the stock FlashInfer call, on a build without the op, past MAX_ROWS
rows, for an input outside the kernel's contract, or when the workspace is first needed inside a capture. Each distinct
reason logs one line per process: ``[M3.1 DEP] SFP_MC_FI not taken: <reason> (first: <value>)``. Kill switch:
``TRTLLM_M31_DISABLE_DEP_SFP_MC_FI=1`` installs no override. Read at import.

Workspace: one zero-initialized byte buffer per device, MAX_ROWS rows of
``minimax_m31_sampling_from_probs_mc_fi_ws_bytes()``, kept for the life of the process (captured graphs keep its
address); every launch leaves it ready for the next. ``prebuild`` allocates it outside any capture.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (
    native_ops,
)
from tensorrt_llm.logger import logger

from .m31_switches import m31_disabled

DEP_SFP_MC_FI = not m31_disabled("DEP_SFP_MC_FI")

#: The most rows one call takes: one draw per request at DEP8's widest CUDA-graph bucket per rank.
MAX_ROWS = 48  # capacity: fixed: DEP8's top graph bucket
#: The largest CTA count per row (the grid is rows x ctas_per_row 1024-thread CTAs).
MAX_CTAS_PER_ROW = 64
MAX_VOCAB = 64 * 4096

# The op and the workspace bytes per row, decided once per process.
_STATE: dict = {}
# Per device index: the persistent workspace and the SM count.
_WORKSPACES: dict = {}
_SMS: dict = {}
# The ctas_per_row values a served call has launched with, each logged once per process.
_TAKEN: set = set()
# The reasons a call was declined, each logged once per process.
_DECLINED: set = set()


def sfp_mc_fi_op():
    """``native_ops.minimax_m31_sampling_from_probs_mc_fi`` when this build carries it, else None."""
    if "op" not in _STATE:
        op = getattr(native_ops, "minimax_m31_sampling_from_probs_mc_fi", None)
        _STATE["op"] = op
        _STATE["ws_bytes"] = (
            int(native_ops.minimax_m31_sampling_from_probs_mc_fi_ws_bytes())
            if op is not None
            else 0
        )
    return _STATE["op"]


def ctas_per_row(rows: int, sms: int) -> int:
    """The largest power of two <= MAX_CTAS_PER_ROW with ctas x rows <= sms, else 1 (more CTAs than SMs
    only queue)."""
    g = MAX_CTAS_PER_ROW
    while g > 1 and g * rows > sms:
        g //= 2
    return g


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SMS:
        _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SMS[idx]


def prebuild(device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """Allocate (once) the device's zeroed workspace for MAX_ROWS rows; must run outside a graph capture. Returns it, or
    None when the op is not built or the stream is capturing."""
    if sfp_mc_fi_op() is None:
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
        _WORKSPACES[dev.index] = torch.zeros(
            MAX_ROWS * _STATE["ws_bytes"], dtype=torch.uint8, device=dev
        )
    return _WORKSPACES[dev.index]


def _outside_contract(probs, seed, offset) -> Optional[tuple[str, str]]:
    """Why the inputs fall outside the kernel's contract, as (reason, the offending value), or None inside it: float32
    CUDA [rows, vocab] probs with a unit inner stride, a row stride divisible by 4 and 16-byte alignment (1..MAX_ROWS
    rows, vocab divisible by 4 and <= MAX_VOCAB), and contiguous int64 seed / offset with at least one entry on the
    probs' device."""
    for name, t in (("probs", probs), ("seed", seed), ("offset", offset)):
        if not isinstance(t, torch.Tensor):
            return "probs / seed / offset not all tensors", f"{name} is {type(t).__name__}"
    if not probs.is_cuda:
        return "probs not on a CUDA device", str(probs.device)
    if probs.dtype != torch.float32:
        return "probs not float32", str(probs.dtype)
    if probs.dim() != 2:
        return "probs not 2-D", str(tuple(probs.shape))
    if probs.stride(1) != 1 or probs.stride(0) % 4 != 0:
        return "probs strides outside the contract", str(probs.stride())
    if probs.data_ptr() % 16 != 0:
        return "probs not 16-byte aligned", hex(probs.data_ptr() % 16)
    rows, vocab = probs.shape
    if not 1 <= rows <= MAX_ROWS:
        return f"rows outside 1..{MAX_ROWS}", f"{rows} rows"
    if vocab < 4 or vocab % 4 != 0 or vocab > MAX_VOCAB:
        return f"vocab not a multiple of 4 in 4..{MAX_VOCAB}", f"vocab {vocab}"
    for name, t in (("seed", seed), ("offset", offset)):
        if t.dtype != torch.int64:
            return f"{name} not int64", str(t.dtype)
        if t.numel() < 1:
            return f"{name} empty", "0 entries"
        if not t.is_contiguous():
            return f"{name} not contiguous", str(t.stride())
        if t.device != probs.device:
            return f"{name} on another device", str(t.device)
    return None


def _declined(reason: str, value: str) -> None:
    """The not-taken line: once per distinct reason, with the first offending value."""
    if reason in _DECLINED:
        return
    _DECLINED.add(reason)
    logger.info(
        f"[M3.1 DEP] SFP_MC_FI not taken: {reason} (first: {value}); stock FlashInfer sampling_from_probs"
    )


def _taken(rows: int, ctas: int) -> None:
    """The taken line: once per distinct ctas_per_row a served call launches with."""
    if ctas in _TAKEN:
        return
    _TAKEN.add(ctas)
    logger.info(f"[M3.1 DEP] SFP_MC_FI taken (ctas_per_row {ctas} at {rows} rows)")


def sampling_from_probs_mc_fi(
    probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> Optional[torch.Tensor]:
    """The sampler's sampling-from-probs override: FlashInfer's int32 tokens bit for bit, or None (stock)."""
    op = sfp_mc_fi_op()
    if op is None:
        _declined("the op is not built", "minimax_m31_sampling_from_probs_mc_fi")
        return None
    outside = _outside_contract(probs, seed, offset)
    if outside is not None:
        _declined(*outside)
        return None
    workspace = prebuild(probs.device)
    if workspace is None:
        _declined("no workspace", f"{probs.device}, first needed inside a graph capture")
        return None
    rows = probs.shape[0]
    ctas = ctas_per_row(rows, _sm_count(probs.device))
    out = op(probs, seed, offset, workspace, ctas)
    _taken(rows, ctas)
    return out


def engagement() -> str:
    """The log line's state: whether the override is installed in this process."""
    if not DEP_SFP_MC_FI:
        return "off (TRTLLM_M31_DISABLE_DEP_SFP_MC_FI=1); stock FlashInfer sampling_from_probs"
    if sfp_mc_fi_op() is None:
        return "not taken: the op is not built; stock FlashInfer sampling_from_probs"
    return f"engaged (FlashInfer-bitwise, rows <= {MAX_ROWS})"


def log_engagement() -> None:
    logger.info(f"[M3.1 DEP] SFP_MC_FI: {engagement()}")
