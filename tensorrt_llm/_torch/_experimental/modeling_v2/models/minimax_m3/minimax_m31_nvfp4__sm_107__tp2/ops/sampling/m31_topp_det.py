# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TOPP_DET: the spec sampler's top-p renorm as one persistent C++ launch, exact vs FlashInfer's deterministic
``top_p_renorm_probs`` (``trtllm::minimax_m31_topp_renorm_det``, kernels/minimaxM31/toppRenormDetSm103.cu).

TOPP_DET v3 (``trtllm::minimax_m31_topp_renorm_det_v3``, kernels/minimaxM31/toppRenormDetV3Sm103.cu) is the same
renorm, bitwise v1.2's output, with the redundant per-CTA selection and a programmatic dependent launch. It is taken
when it is built unless
``TRTLLM_M31_DISABLE_TOPPDET_V3`` is set; v1.2 otherwise. ``TRTLLM_M31_DISABLE_TOPPDET_T3`` keeps v3 on its global
radix passes (no candidate list).

V4K8 (a library whose op takes ``list_path`` / ``wide_hist``, default on): the one-sync list path (pass 0 lists the
elements >= 2^-12; when the selected bucket lies at or above that edge every CTA finishes the selection, FlashInfer's
chain sums and the write from the list after one barrier) and 8 privatized pass histograms, both bitwise the same
output; ``TRTLLM_M31_DISABLE_TOPPDET_V4`` / ``TRTLLM_M31_DISABLE_TOPPDET_K8`` turn them off.

The target installs ``top_p_renorm_det`` as the sampler's top-p renorm override
(``sampler/ops/flashinfer.set_top_p_renorm_override``); it returns None, and the sampler keeps the stock FlashInfer
call, whenever the op is not built or the input is outside the kernel's contract.

Workspace: one zero-initialized byte buffer per device, ``max_rows()`` rows of
``minimax_m31_topp_renorm_det_ws_bytes()``, kept for the life of the process (CUDA graphs captured with it keep its
address). The kernel leaves it ready for the next launch, so it is zeroed once. ``prebuild`` allocates it outside any
CUDA-graph capture (the target calls it from ``cache_derived_state``, before the executor's warmup and captures); a
call that finds no workspace allocates one when the stream is not capturing, and inside a capture returns None
(stock FlashInfer for that call) rather than allocating from the graph's private pool.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger

from ...kernels import m3_report
from ...m31_switches import m31_disabled

# The largest CTA count per row (the persistent grid is rows x ctas_per_row 1024-thread CTAs, all co-resident).
MAX_CTAS_PER_ROW = 16

# The rows the workspace holds and the override takes with v3 / V4K8: up to the agg TP2 K3 c64 calls (256 rows) and
# the K7 c56 calls (448 rows). The VR microbenchmarks (rows 9..192 and 192 / 256 / 320 / 368 / 448;
# x peaked / randn / near_uniform) measured v3 and V4K8 bitwise FlashInfer's det mode on every cell and V4K8 faster
# on every cell (by >= 67 us per call; e.g. 64 rows 208.5 -> 44.5 us peaked, 256 rows 629.1 -> 343.8 randn, 448 rows
# 1047.6 -> 528.5 randn). Past 106 rows each row runs on one CTA (no co-residency). The workspace is ~0.2 MB per row
# (~90 MB per device at 448 rows).
MAX_ROWS = 448
# v1.2 keeps the TP8 / TP4 K7 c1 calls (the draft's 7 rows and the verify's 8), where GB300 microbenchmarks measured it
# faster than stock FlashInfer (54.3 vs 104.2 us per call at 7 peaked rows, 54.8 vs 105.1 at 8). At
# 56 / 64 rows v1 was slower than FlashInfer's default mode (340 vs 187 / 199 us) and v1.2 is not
# re-measured there, so larger calls keep the stock kernel when v3 is not engaged.
V12_MAX_ROWS = 8

# The op when it is built, decided once per process.
_STATE: dict = {}
# Per device index: the persistent workspace and the SM count.
_WORKSPACES: dict = {}
_SMS: dict = {}
# The row counts already logged as taken / stock (one engagement line per distinct row count and outcome).
_LOGGED: set = set()


def _load_op(name: str):
    """``torch.ops.trtllm.<name>`` when this build carries it, else None."""
    return getattr(torch.ops.trtllm, name, None)


def topp_det_op():
    """The renorm op this process uses, decided once: ``trtllm::minimax_m31_topp_renorm_det_v3`` when it is built
    and TOPPDET_V3 is not disabled, else ``trtllm::minimax_m31_topp_renorm_det`` when built, else None."""
    if "op" not in _STATE:
        op, ws, v3 = None, 0, False
        if not m31_disabled("TOPPDET_V3"):
            op = _load_op("minimax_m31_topp_renorm_det_v3")
            if op is not None:
                v3 = True
                ws = int(torch.ops.trtllm.minimax_m31_topp_renorm_det_v3_ws_bytes())
                logger.info("[M3.1] TOPP_DET v3 private op engaged (in-tree)")
        if op is None:
            op = _load_op("minimax_m31_topp_renorm_det")
            ws = (
                int(torch.ops.trtllm.minimax_m31_topp_renorm_det_ws_bytes())
                if op is not None
                else 0
            )
        _STATE["op"], _STATE["ws_bytes"], _STATE["v3"] = op, ws, v3
        # v3's candidate-list radix passes (bitwise the same output) unless TOPPDET_T3 is disabled.
        _STATE["cand_list"] = v3 and not m31_disabled("TOPPDET_T3")
        # V4K8 (a library whose schema carries list_path / wide_hist): the one-sync list path and 8 privatized pass
        # histograms, both bitwise the same output, unless TOPPDET_V4 / TOPPDET_K8 is disabled.
        v4k8 = v3 and "list_path" in str(op.default._schema)
        _STATE["v4k8"] = v4k8
        _STATE["list_path"] = v4k8 and not m31_disabled("TOPPDET_V4")
        _STATE["wide_hist"] = v4k8 and not m31_disabled("TOPPDET_K8")
        if v3:
            logger.info(
                "[M3.1] TOPP_DET v3 T3 candidate-list fused passes: "
                f"{'on' if _STATE['cand_list'] else 'off'} (cap 4096)"
            )
            logger.info(
                f"[M3.1] TOPP_DET V4K8: list path {'on' if _STATE['list_path'] else 'off'}, "
                f"8 pass histograms {'on' if _STATE['wide_hist'] else 'off'}"
                f"{'' if v4k8 else ' (library without V4K8)'}"
            )
        if op is not None:
            logger.info(f"[M3.1] TOPP_DET row cap: {MAX_ROWS if v4k8 else V12_MAX_ROWS} rows")
    return _STATE["op"]


def is_v3() -> bool:
    """Whether the engaged op is TOPP_DET v3."""
    topp_det_op()
    return _STATE["v3"]


def max_rows() -> int:
    """The rows the workspace holds and the override takes: MAX_ROWS with v3 + V4K8, V12_MAX_ROWS otherwise."""
    # TP2: the wide cap needs the V4K8 library; V4K8 (not v3/T3 alone) measured faster than stock at
    # 9..448 rows (VR microbenchmarks), so a v3/T3-only build keeps the 8-row cap.
    topp_det_op()
    return MAX_ROWS if _STATE.get("v4k8") else V12_MAX_ROWS


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
    """Allocate (once) the device's zeroed workspace for max_rows() rows; must run outside a CUDA-graph capture.
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
            max_rows() * ws_bytes_per_row(), dtype=torch.uint8, device=dev
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

    ARCHS = frozenset({Arch.SM_107})

    # The c1 K7 calls the microbenchmark gated bitwise on GB300 (8 row classes x 4 top_p modes x 16 seeds); the
    # sm_107 build of the same source is gated by the same microbenchmark on VR before the target takes it.
    CELLS: tuple[Cell, ...] = (
        Cell(why="the K7 draft's 7 rows at c1 (vocab 200064)", spec=dict(rows=7, d=200064)),
        Cell(why="the K7 verify's 8 rows at c1 (vocab 200064)", spec=dict(rows=8, d=200064)),
        Cell(why="one row (the smallest grid, 16 CTAs)", spec=dict(rows=1, d=200064)),
        Cell(why="TP2 WIDE K7 c8: 64 rows, 2 CTAs per row", spec=dict(rows=64, d=200064)),
        Cell(why="TP2 WIDE K7 c24: 192 rows, 1 CTA per row", spec=dict(rows=192, d=200064)),
        Cell(why="agg TP2 K3 c64: 256 rows, 1 CTA per row", spec=dict(rows=256, d=200064)),
        Cell(why="the 448-row cap (K7 c56), 1 CTA per row", spec=dict(rows=448, d=200064)),
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
            if _STATE["v4k8"]:
                op(
                    probs,
                    top_p,
                    workspace,
                    out,
                    int(ctas_per_row),
                    True,
                    _STATE["cand_list"],
                    _STATE["list_path"],
                    _STATE["wide_hist"],
                )
            else:
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
    runs stock FlashInfer) when the op is not built, rows exceed max_rows(), no workspace can be had (first use inside
    a capture) or the input is outside the contract."""
    if not isinstance(top_p, torch.Tensor) or not probs.is_cuda or probs.dim() != 2:
        return None
    rows = probs.shape[0]
    if rows < 1 or topp_det_op() is None:
        return None
    if rows > max_rows():
        _log_once(rows, False, f"above the {max_rows()}-row cap")
        return None
    workspace = prebuild(probs.device)
    if workspace is None:
        return None
    g = ctas_per_row(rows, sm_count(probs.device))
    out = m31_topp_det(probs, top_p, workspace, g)
    if out is None:
        _log_once(rows, False, "outside the kernel's contract")
    else:
        _log_once(rows, True, f"ctas_per_row {g}")
        m3_report.work("TOPP_DET", f"{rows} rows")
        if _STATE.get("v3"):
            m3_report.work("TOPPDET_V3", f"{rows} rows")
        if _STATE.get("cand_list"):
            m3_report.work("TOPPDET_T3", f"{rows} rows")
    return out


def _log_once(rows: int, taken: bool, detail: str) -> None:
    """One engagement line per distinct (row count, outcome): "TOPP_DET taken at N rows" when the op served the call,
    "TOPP_DET stock at N rows" when the sampler keeps FlashInfer."""
    if (rows, taken) not in _LOGGED:
        _LOGGED.add((rows, taken))
        logger.info(f"[M3.1] TOPP_DET {'taken' if taken else 'stock'} at {rows} rows ({detail})")
