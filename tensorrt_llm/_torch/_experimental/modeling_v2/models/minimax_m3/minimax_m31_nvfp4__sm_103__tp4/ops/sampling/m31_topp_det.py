# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ST-01 TOPP_DET: the spec sampler's top-p renorm as one persistent C++ launch, exact vs FlashInfer's deterministic
``top_p_renorm_probs`` (``trtllm::minimax_m31_topp_renorm_det``, kernels/minimaxM31/toppRenormDetSm103.cu).

TOPP_DET v3 (``trtllm::minimax_m31_topp_renorm_det_v3``, kernels/minimaxM31/toppRenormDetV3Sm103.cu) is the same
renorm, bitwise v1.2's output, with the redundant per-CTA selection and a programmatic dependent launch. It is taken
when it is built (in-tree, or overlaid as tensorrt_llm/libs/libm31_topp_det_v3.so) unless
``TRTLLM_M31_DISABLE_TOPPDET_V3`` is set; v1.2 otherwise. ``TRTLLM_M31_DISABLE_TOPPDET_T3`` keeps v3 on its global
radix passes (no candidate list).

The target installs ``top_p_renorm_det`` as the sampler's top-p renorm override
(``sampler/ops/flashinfer.set_top_p_renorm_override``); it returns None, and the sampler keeps the stock FlashInfer
call, whenever the op is not built or the input is outside the kernel's contract.

Workspace: one zero-initialized byte buffer per device, ``MAX_ROWS`` rows of
``minimax_m31_topp_renorm_det_ws_bytes()``, kept for the life of the process (CUDA graphs captured with it keep its
address). The kernel leaves it ready for the next launch, so it is zeroed once. ``prebuild`` allocates it outside any
CUDA-graph capture (the target calls it from ``cache_derived_state``, before the executor's warmup and captures); a
call that finds no workspace allocates one when the stream is not capturing, and inside a capture returns None
(stock FlashInfer for that call) rather than allocating from the graph's private pool.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.fakes import (
    m31_fake_ops as _m31_fake_ops,
)
from tensorrt_llm.logger import logger

from ...m31_switches import m31_disabled

# The largest CTA count per row (the persistent grid is rows x ctas_per_row 1024-thread CTAs, all co-resident).
MAX_CTAS_PER_ROW = 16

# The rows the workspace holds and the override takes: the TP8 / TP4 K7 c1 calls (the draft's 7 rows and the
# verify's 8), where the microbenchmarks measured it faster than stock FlashInfer (v1.2: 54.3 vs 104.2 us per
# call at 7 peaked rows, 54.8 vs 105.1 at 8). At 56 / 64 rows v1 was slower than FlashInfer's default mode (340 vs
# 187 / 199 us) and larger shapes are not re-measured, so larger calls keep the stock kernel.
MAX_ROWS = 8

# The op when it is built (or overlaid as tensorrt_llm/libs/libm31_topp_det.so), decided once per process.
_STATE: dict = {}
# Per device index: the persistent workspace and the SM count.
_WORKSPACES: dict = {}
_SMS: dict = {}


def _load_op(name: str, lib_name: str):
    """``torch.ops.trtllm.<name>`` if registered (in-tree), else after loading tensorrt_llm/libs/<lib_name> when it
    exists, else None. Returns (op, the library path loaded or None)."""
    op = getattr(torch.ops.trtllm, name, None)
    if op is not None:
        return op, None
    import tensorrt_llm

    lib = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs", lib_name)
    if not os.path.exists(lib):
        return None, None
    torch.ops.load_library(lib)
    _m31_fake_ops.register()
    return getattr(torch.ops.trtllm, name, None), lib


def topp_det_op():
    """The renorm op this process uses, decided once: ``trtllm::minimax_m31_topp_renorm_det_v3`` when it is built
    and TOPPDET_V3 is not disabled, else ``trtllm::minimax_m31_topp_renorm_det`` when built, else None."""
    if "op" not in _STATE:
        op, ws, v3 = None, 0, False
        if not m31_disabled("TOPPDET_V3"):
            op, lib = _load_op("minimax_m31_topp_renorm_det_v3", "libm31_topp_det_v3.so")
            if op is not None:
                v3 = True
                ws = int(torch.ops.trtllm.minimax_m31_topp_renorm_det_v3_ws_bytes())
                tag = "in-tree"
                if lib is not None:
                    with open(lib, "rb") as f:
                        digest = hashlib.md5(f.read(), usedforsecurity=False).hexdigest()[:8]
                        tag = f"libm31_topp_det_v3.so md5 {digest}"
                logger.info(f"[M3.1] TOPP_DET v3 private op engaged ({tag})")
        if op is None:
            op, _ = _load_op("minimax_m31_topp_renorm_det", "libm31_topp_det.so")
            ws = (
                int(torch.ops.trtllm.minimax_m31_topp_renorm_det_ws_bytes())
                if op is not None
                else 0
            )
        _STATE["op"], _STATE["ws_bytes"], _STATE["v3"] = op, ws, v3
        # v3's candidate-list radix passes (bitwise the same output) unless TOPPDET_T3 is disabled.
        _STATE["cand_list"] = v3 and not m31_disabled("TOPPDET_T3")
        if v3:
            logger.info(
                "[M3.1] TOPP_DET v3 T3 candidate-list fused passes: "
                f"{'on' if _STATE['cand_list'] else 'off'} (cap 4096)"
            )
    return _STATE["op"]


def is_v3() -> bool:
    """Whether the engaged op is TOPP_DET v3."""
    topp_det_op()
    return _STATE["v3"]


def ws_bytes_per_row() -> int:
    """Workspace bytes per row (0 when the op is not built)."""
    topp_det_op()
    return _STATE["ws_bytes"]


def sm_count(device: torch.device) -> int:
    """The device's SM count, cached per device."""
    idx = torch.device(device).index
    idx = torch.cuda.current_device() if idx is None else idx
    if idx not in _SMS:
        _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SMS[idx]


def ctas_per_row(rows: int, sms: int) -> int:
    """The largest power of two <= MAX_CTAS_PER_ROW with ctas x rows <= sms (1 past that: one CTA per row needs no
    co-residency)."""
    g = MAX_CTAS_PER_ROW
    while g > 1 and g * rows > sms:
        g //= 2
    return g


def prebuild(device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """Allocate (once) the device's zeroed workspace for MAX_ROWS rows; must run outside a CUDA-graph capture.
    Returns it, or None when the op is not built or the stream is capturing."""
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
            MAX_ROWS * ws_bytes_per_row(), dtype=torch.uint8, device=dev
        )
    return _WORKSPACES[dev.index]


class M31ToppDet(OpWrapper):
    """FlashInfer's ``top_p_renorm_probs(probs, top_p, is_deterministic=True)`` of fp32 probs [rows, d] with fp32
    per-row top_p [rows], bit for bit on every row with top_p < 1, in ONE launch of rows x ctas_per_row CTAs
    (FlashInfer's: five launches, apply on one CTA per row, slow on high-entropy rows). Rows with top_p >= 1 keep every
    element (FlashInfer drops tail buckets there in ~43% of rows, in both of its modes); their output is
    ``ftz(ftz(p) * norm)``. ``workspace`` is the persistent zeroed buffer of ``prebuild``. ``__call__`` returns None
    (the caller keeps the stock call) for a non-fp32, non-contiguous or misaligned input, a width not divisible by 4,
    a top_p that is not a contiguous fp32 [rows] tensor on probs' device, more rows than the workspace holds, or when
    the op is not built."""

    ARCHS = frozenset({Arch.SM_103})

    # The c1 K7 calls were checked bitwise on GB300 over 8 row classes x 4 top_p modes x 16 seeds.
    CELLS: tuple[Cell, ...] = (
        Cell(why="the K7 draft's 7 rows at c1 (vocab 200064)", spec=dict(rows=7, d=200064)),
        Cell(why="the K7 verify's 8 rows at c1 (vocab 200064)", spec=dict(rows=8, d=200064)),
        Cell(why="one row (the smallest grid, 16 CTAs)", spec=dict(rows=1, d=200064)),
    )

    note = """
    Exact by construction: the det histograms are integer (any aggregation order is FlashInfer's), the per-pass
    selection is FlashInfer's last-block code on them, and the kept sum is FlashInfer's 1024 per-thread chains into
    the same cub BlockReduce. Built with --use_fast_math, as FlashInfer's JIT builds it.
    """

    def __call__(
        self,
        probs: torch.Tensor,
        top_p: torch.Tensor,
        workspace: Optional[torch.Tensor],
        ctas_per_row: int,
    ) -> Optional[torch.Tensor]:
        if (
            not isinstance(top_p, torch.Tensor)
            or workspace is None
            or not probs.is_cuda
            or probs.dtype != torch.float32
            or probs.dim() != 2
            or not probs.is_contiguous()
            or probs.shape[0] < 1
            or probs.shape[1] % 4 != 0
            or probs.data_ptr() % 16 != 0
            or top_p.dtype != torch.float32
            or top_p.dim() != 1
            or top_p.shape[0] != probs.shape[0]
            or not top_p.is_contiguous()
            or top_p.device != probs.device
            or workspace.device != probs.device
            or ctas_per_row < 1
            or 1024 % ctas_per_row != 0
        ):
            return None
        op = topp_det_op()
        if (
            op is None
            or workspace.numel() * workspace.element_size() < probs.shape[0] * ws_bytes_per_row()
        ):
            return None
        out = torch.empty_like(probs)
        if is_v3():
            op(probs, top_p, workspace, out, int(ctas_per_row), True, _STATE["cand_list"])
        else:
            op(probs, top_p, workspace, out, int(ctas_per_row))
        return out

    def reference(self, probs, top_p, workspace=None, ctas_per_row=None) -> torch.Tensor:
        """FlashInfer's deterministic renorm (rows with top_p < 1; workspace and ctas_per_row are ignored)."""
        import flashinfer.sampling

        return flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=True)

    def is_valid(self, probs, top_p, workspace=None, ctas_per_row=None, **kwargs) -> None:
        assert probs.dtype == torch.float32 and probs.dim() == 2, "probs: fp32 [rows, d]"


m31_topp_det = M31ToppDet()


def top_p_renorm_det(probs: torch.Tensor, top_p: torch.Tensor) -> Optional[torch.Tensor]:
    """The sampler's top-p renorm override: the det renorm of probs [rows, d] at per-row top_p, or None (the sampler
    runs stock FlashInfer) when the op is not built, rows exceed MAX_ROWS, no workspace can be had (first use inside
    a capture) or the input is outside the contract."""
    if not isinstance(top_p, torch.Tensor) or not probs.is_cuda or probs.dim() != 2:
        return None
    rows = probs.shape[0]
    if rows < 1 or rows > MAX_ROWS or topp_det_op() is None:
        return None
    workspace = prebuild(probs.device)
    if workspace is None:
        return None
    return m31_topp_det(probs, top_p, workspace, ctas_per_row(rows, sm_count(probs.device)))
