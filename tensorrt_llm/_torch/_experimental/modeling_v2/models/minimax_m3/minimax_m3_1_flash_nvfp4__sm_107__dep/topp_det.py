# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TOPP_DET: the spec sampler's top-p renorm as one persistent launch, bitwise FlashInfer's deterministic
``top_p_renorm_probs(probs, top_p, is_deterministic=True)`` on every row, through the sampler's top-p renorm override
(``sampler/ops/flashinfer.set_top_p_renorm_override``).

``mv2_minimax_m3::minimax_m31_topp_renorm_det_v3`` (kernels/minimaxM31/toppRenormDetV3Sm103.cu) with the candidate list,
the one-sync list path and 8 privatized pass histograms (V4K8), on ``ctas_per_row`` CTAs per row, instead of
FlashInfer's five launches with a one-CTA-per-row apply. ``keep_all_ge1=False``: rows with top_p >= 1 (greedy rows and
CUDA-graph padding rows carry 1.0) take FlashInfer's radix passes like every other row, so they are FlashInfer's output
too (the other targets keep every element there). The DEP target keeps FlashInfer's deterministic mode (it does not arm
the shared sm_107 opt-ins), which is what this matches.

``top_p_renorm_det`` returns None, and the sampler keeps the stock FlashInfer call, on a build whose op lacks
``keep_all_ge1`` or V4K8, past MAX_ROWS rows, for an input outside the kernel's contract, or when the workspace is first
needed inside a capture. Kill switch: ``TRTLLM_M31_DISABLE_TOPP_DET=1`` installs no override. Read at import.

TOPP_SPLIT: a call too wide for 2 CTAs per row (rows > SMs / 2, so ``ctas_per_row`` falls to 1) but whose remainder fits
at >= 4 (rows - SMs / 2 <= SMs / 4) runs as two stream-ordered launches: SMs / 2 rows at 2 CTAs, then the rest at their
own ``ctas_per_row``, each on its own rows' slice of the workspace. Rows are independent, so the output is the same
bits. The verify with 7 draft tokens at 16 requests (128 rows) takes it. Past that window the second launch does not
help over one CTA per row, so the single launch stays. Kill switch:
``TRTLLM_M31_DISABLE_TOPP_SPLIT=1``.

Workspace: one zero-initialized byte buffer per device, MAX_ROWS rows of ``minimax_m31_topp_renorm_det_v3_ws_bytes()``
(~90 MB at 448 rows), kept for the life of the process (captured graphs keep its address); the kernel leaves it ready
for the next launch. ``prebuild`` allocates it outside any capture.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm.logger import logger

from .m31_switches import m31_disabled

TOPP_DET = not m31_disabled("TOPP_DET")
TOPP_SPLIT = not m31_disabled("TOPP_SPLIT")
# TOPP_V6: on sm_107 the v3 renorm sums each chain straight from global memory (the same adds in the same order). Kill
# switch TRTLLM_M31_DISABLE_TOPP_V6=1 (the staged loop on every device), passed as the op's chain_prefetch argument.
TOPP_V6 = not m31_disabled("TOPP_V6")

#: The widest call the workspace holds: the verify with 7 draft tokens at 48 requests is 384 rows, the draft 48.
MAX_ROWS = 448  # capacity: fixed: TOPP_DET's widest supported call
#: The largest CTA count per row (the persistent grid is rows x ctas_per_row 1024-thread CTAs, all co-resident).
MAX_CTAS_PER_ROW = 16
#: The op's schema arguments this target needs: V4K8 and the FlashInfer rows switch.
_REQUIRED_ARGS = ("list_path", "wide_hist", "keep_all_ge1")

# The op and its workspace bytes per row, decided once per process.
_STATE: dict = {}
# Per device index: the persistent workspace and the SM count.
_WORKSPACES: dict = {}
_SMS: dict = {}


def topp_det_op():
    """``mv2_minimax_m3::minimax_m31_topp_renorm_det_v3`` when this build carries it with every argument in
    _REQUIRED_ARGS, else None."""
    if "op" not in _STATE:
        op = getattr(family_ops(), "minimax_m31_topp_renorm_det_v3", None)
        if op is not None and not all(arg in str(op.default._schema) for arg in _REQUIRED_ARGS):
            op = None
        _STATE["op"] = op
        _STATE["ws_bytes"] = (
            int(family_ops().minimax_m31_topp_renorm_det_v3_ws_bytes()) if op is not None else 0
        )
    return _STATE["op"]


def ctas_per_row(rows: int, sms: int) -> int:
    """The largest power of two <= MAX_CTAS_PER_ROW with ctas x rows <= sms (1 past that: one CTA per row needs no
    co-residency)."""
    g = MAX_CTAS_PER_ROW
    while g > 1 and g * rows > sms:
        g //= 2
    return g


def launch_plan(rows: int, sms: int, split: bool = True) -> list:
    """The launches of one call as [(first row, end row, ctas)]: one launch at ``ctas_per_row``, or the TOPP_SPLIT
    pair (see the module docstring) when ``split`` and SMs / 2 < rows <= SMs / 2 + SMs / 4."""
    half = sms // 2
    if split and half < rows <= half + sms // 4:
        return [(0, half, 2), (half, rows, ctas_per_row(rows - half, sms))]
    return [(0, rows, ctas_per_row(rows, sms))]


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SMS:
        _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SMS[idx]


def prebuild(device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """Allocate (once) the device's zeroed workspace for MAX_ROWS rows; must run outside a graph capture. Returns it,
    or None when the op is not built or the stream is capturing."""
    if topp_det_op() is None:
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


def outside_contract(probs: torch.Tensor, top_p) -> bool:
    """Whether probs / top_p fall outside the kernel's contract: fp32 contiguous 16-byte aligned [1..MAX_ROWS, d] with
    d divisible by 4, and a contiguous fp32 [rows] top_p on the same device."""
    return (
        not isinstance(top_p, torch.Tensor)
        or probs.dtype != torch.float32
        or probs.dim() != 2
        or not probs.is_contiguous()
        or not 1 <= probs.shape[0] <= MAX_ROWS
        or probs.shape[1] % 4 != 0
        or probs.data_ptr() % 16 != 0
        or top_p.dtype != torch.float32
        or top_p.dim() != 1
        or top_p.shape[0] != probs.shape[0]
        or not top_p.is_contiguous()
        or top_p.device != probs.device
    )


def top_p_renorm_det(probs: torch.Tensor, top_p: torch.Tensor) -> Optional[torch.Tensor]:
    """The sampler's top-p renorm override: FlashInfer's deterministic renorm of probs [rows, d] at per-row top_p,
    bit for bit, or None (stock FlashInfer)."""
    if not isinstance(probs, torch.Tensor) or not probs.is_cuda or outside_contract(probs, top_p):
        return None
    op = topp_det_op()
    workspace = None if op is None else prebuild(probs.device)
    if workspace is None:
        return None
    plan = launch_plan(probs.shape[0], _sm_count(probs.device), TOPP_SPLIT)
    return _launch_plan(op, probs, top_p, workspace, plan, _STATE["ws_bytes"])


def _launch(
    op, probs: torch.Tensor, top_p: torch.Tensor, workspace: torch.Tensor, ctas: int, out=None
) -> torch.Tensor:
    """One renorm launch: PDL, the candidate list, the list path and 8 pass histograms on, keep_all_ge1 off."""
    out = torch.empty_like(probs) if out is None else out
    op(probs, top_p, workspace, out, ctas, True, True, True, True, False, chain_prefetch=TOPP_V6)
    return out


def _launch_plan(
    op, probs: torch.Tensor, top_p: torch.Tensor, workspace: torch.Tensor, plan: list, ws_bytes: int
) -> torch.Tensor:
    """The plan's launches in stream order, each on its rows of probs / top_p / out and of the workspace."""
    if len(plan) == 1:
        return _launch(op, probs, top_p, workspace, plan[0][2])
    out = torch.empty_like(probs)
    for r0, r1, ctas in plan:
        _launch(
            op,
            probs[r0:r1],
            top_p[r0:r1],
            workspace[r0 * ws_bytes : r1 * ws_bytes],
            ctas,
            out[r0:r1],
        )
    return out


def engagement() -> str:
    """The log line's state: whether the override is taken in this process."""
    if not TOPP_DET:
        return "off (TRTLLM_M31_DISABLE_TOPP_DET=1); stock FlashInfer top_p_renorm_probs"
    if topp_det_op() is None:
        return "not taken: the op is not built with V4K8 and keep_all_ge1; stock FlashInfer top_p_renorm_probs"
    split = "on" if TOPP_SPLIT else "off (TRTLLM_M31_DISABLE_TOPP_SPLIT=1)"
    return f"engaged (V4K8, FlashInfer's passes on top_p >= 1 rows, rows <= {MAX_ROWS}; TOPP_SPLIT {split})"


def log_engagement() -> None:
    logger.info(f"[M3.1 DEP] TOPP_DET: {engagement()}")
