# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOFTMAX_MC: the spec sampler's softmax with its write pass on many CTAs per row, exact vs
``flashinfer.sampling.softmax(logits, temperature)`` (``mv2_minimax_m3::minimax_m31_softmax_mc``,
csrc/m31/softmaxMcSm103.cu).

The target installs ``softmax_mc`` as the sampler's softmax override (``sampler/ops/flashinfer.set_softmax_override``);
it returns None, and the sampler keeps the stock FlashInfer call, whenever the op is not built or the input is outside
the kernel's contract (FlashInfer's Path A shapes: rows <= 128 and vocab >= 24576; here also rows <= ``max_rows()``).

Workspace: one partials buffer per device (``max_rows()`` x ceil(``MAX_VOCAB`` / 8192) float2, 32 KB at 32 rows), kept
for the life of the process (CUDA graphs captured with it keep its address). Every launch overwrites what it reads, so
it needs no initialization. ``prebuild`` allocates it outside any CUDA-graph capture (the target calls it from
``cache_derived_state``, before the executor's warmup and captures); a call that finds no workspace allocates one when
the stream is not capturing, and inside a capture returns None (stock FlashInfer for that call) rather than allocating
from the graph's private pool. Calls that share it must be stream-ordered (the sampler's are).
"""

from __future__ import annotations

from typing import Optional, Union

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm.logger import logger

from ...m31_switches import m31_disabled

# The write pass's CTAs per row (bitwise FlashInfer's softmax at 4 / 8 / 16). The grid (rows x WRITE_CTAS) needs no
# co-residency, so the SM count does not bound it.
WRITE_CTAS = 16

# The rows the override takes: one request at draft length 7 (the draft's 7 rows and the verify's 8). Larger calls keep
# the stock kernel.
MAX_ROWS = 8
NARROW_MAX_ROWS = MAX_ROWS  # capacity: fixed: the c1 cap (TP8WIDE_SOFTMAX off)
# TP8WIDE_SOFTMAX: the override also takes the 2..4-request calls of 9..32 rows at draft length 7 (FlashInfer's Path A
# shapes, where the kernel is FlashInfer's softmax bit for bit at every write-CTA count), with fewer write CTAs per row
# once rows x 16 outgrows the SMs: 16 up to 12 rows (every one-request call keeps 16), 8 up to 24, 4 up to 32. Scoped
# to the TP8 decode batch <= 4. Kill switch
# TRTLLM_M31_DISABLE_TP8WIDE_SOFTMAX=1 (or the group TRTLLM_M31_DISABLE_TP8WIDE=1): the 8-row cap at 16 write CTAs.
TP8WIDE_SOFTMAX = not (m31_disabled("TP8WIDE") or m31_disabled("TP8WIDE_SOFTMAX"))
WIDE_MAX_ROWS = 32  # capacity: fixed: the K7 4-request verify (4 x 8 rows)
# (largest row count, write CTAs per row).
WRITE_CTAS_BY_ROWS = ((12, WRITE_CTAS), (24, 8), (WIDE_MAX_ROWS, 4))
_WIDE_LOGGED: set = set()


def max_rows() -> int:
    """The rows the override takes: WIDE_MAX_ROWS with TP8WIDE_SOFTMAX, NARROW_MAX_ROWS otherwise."""
    return WIDE_MAX_ROWS if TP8WIDE_SOFTMAX else NARROW_MAX_ROWS


def write_ctas(rows: int) -> int:
    """The write CTAs per row for a call of ``rows`` rows (WRITE_CTAS with TP8WIDE_SOFTMAX off)."""
    if TP8WIDE_SOFTMAX:
        for top, ctas in WRITE_CTAS_BY_ROWS:
            if rows <= top:
                return ctas
    return WRITE_CTAS


# FlashInfer's Path A bounds (OnlineSoftmax: batch <= 128 and d >= 24576); outside them FlashInfer takes its
# single-block Path B, which this kernel does not reproduce.
PATH_A_MAX_ROWS = 128
PATH_A_MIN_VOCAB = 24576
# FlashInfer's Path A slice (DEFAULT_SLICE_SIZE) and the widest vocab the workspace holds.
SLICE = 8192
MAX_VOCAB = 1 << 20

# The op when it is built, decided once per process.
_STATE: dict = {}
# Per device index: the persistent partials workspace.
_WORKSPACES: dict = {}


def softmax_mc_op():
    """``mv2_minimax_m3::minimax_m31_softmax_mc`` if this build carries it, else None."""
    if "op" not in _STATE:
        op = getattr(family_ops(), "minimax_m31_softmax_mc", None)
        _STATE["op"] = op
    return _STATE["op"]


def partials_numel(rows: int, d: int) -> int:
    """fp32 elements of the partials workspace for rows x d (one float2 per row and 8192-wide slice)."""
    return 2 * rows * ((d + SLICE - 1) // SLICE)


def prebuild(device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """Allocate (once) the device's partials workspace for max_rows() rows of up to MAX_VOCAB; must run outside a
    CUDA-graph capture. Returns it, or None when the op is not built or the stream is capturing."""
    if softmax_mc_op() is None:
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
            partials_numel(max_rows(), MAX_VOCAB), dtype=torch.float32, device=dev
        )
    return _WORKSPACES[dev.index]


class M31SoftmaxMc(OpWrapper):
    """FlashInfer's ``softmax(logits, temperature)`` of fp32 logits [rows, d], bit for bit, on FlashInfer's Path A
    shapes (rows <= 128, d >= 24576): FlashInfer's Map kernel, then its Reduce kernel's prologue in each of
    ``write_ctas`` CTAs per row, each writing its share of the row (FlashInfer: one CTA writes the whole row).
    ``temperature`` is None (1.0), a Python number (one temperature) or an fp32 [rows] tensor (one per row), as
    FlashInfer takes it. ``partials`` is the persistent workspace of ``prebuild``. ``__call__`` returns None (the caller
    keeps the stock call) for a non-fp32, non-contiguous or misaligned input, a width not divisible by 4, a shape
    outside Path A or the workspace, a temperature tensor that is not a contiguous fp32 [rows] tensor on logits'
    device, or when the op is not built."""

    ARCHS = frozenset({Arch.SM_107})

    # The one-request calls at draft length 7, bitwise FlashInfer's softmax.
    CELLS: tuple[Cell, ...] = (
        Cell(why="the K7 draft's 7 rows at c1 (vocab 200064)", spec=dict(rows=7, d=200064)),
        Cell(why="the K7 verify's 8 rows at c1 (vocab 200064)", spec=dict(rows=8, d=200064)),
        Cell(why="one row (the smallest grid)", spec=dict(rows=1, d=200064)),
        # TP8WIDE_SOFTMAX: the K7 2..4-request calls at the write-CTA counts of WRITE_CTAS_BY_ROWS.
        Cell(
            why="two requests' draft rows (14) at draft length 7, 8 write CTAs",
            spec=dict(rows=14, d=200064),
        ),
        Cell(
            why="two requests' verify rows (16) at draft length 7, 8 write CTAs",
            spec=dict(rows=16, d=200064),
        ),
        Cell(
            why="three requests' verify rows (24) at draft length 7, 8 write CTAs",
            spec=dict(rows=24, d=200064),
        ),
        Cell(
            why="four requests' verify rows (32) at draft length 7, 4 write CTAs",
            spec=dict(rows=32, d=200064),
        ),
    )

    note = """
    Exact by construction: the Map kernel is FlashInfer's, every write CTA reduces the same partials with the same cub
    BlockReduce<float2, 1024> and Float2SoftmaxReduceOp as FlashInfer's single Reduce CTA, and the per-element write
    keeps FlashInfer's source form. Built with --use_fast_math, as FlashInfer's JIT builds it.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        temperature: Union[torch.Tensor, float, None],
        partials: Optional[torch.Tensor],
        write_ctas: int,
    ) -> Optional[torch.Tensor]:
        if (
            partials is None
            or not isinstance(logits, torch.Tensor)
            or not logits.is_cuda
            or logits.dtype != torch.float32
            or logits.dim() != 2
            or not logits.is_contiguous()
            or logits.shape[0] < 1
            or logits.shape[0] > PATH_A_MAX_ROWS
            or logits.shape[1] < PATH_A_MIN_VOCAB
            or logits.shape[1] % 4 != 0
            or logits.data_ptr() % 16 != 0
            or partials.device != logits.device
            or partials.numel() * partials.element_size()
            < 4 * partials_numel(logits.shape[0], logits.shape[1])
            or write_ctas < 1
            or write_ctas > 65535
        ):
            return None
        if temperature is None:
            temp_arr, temp_val = None, 1.0
        elif isinstance(temperature, torch.Tensor):
            if (
                temperature.dtype != torch.float32
                or temperature.dim() != 1
                or temperature.shape[0] != logits.shape[0]
                or not temperature.is_contiguous()
                or temperature.device != logits.device
            ):
                return None
            temp_arr, temp_val = temperature, 0.0
        elif isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
            temp_arr, temp_val = None, float(temperature)
        else:
            return None
        op = softmax_mc_op()
        if op is None:
            return None
        out = torch.empty_like(logits)
        op(logits, temp_arr, temp_val, partials, out, int(write_ctas))
        return out

    def reference(self, logits, temperature, partials=None, write_ctas=None) -> torch.Tensor:
        """FlashInfer's softmax (partials and write_ctas are ignored)."""
        import flashinfer.sampling

        return flashinfer.sampling.softmax(logits, temperature)

    def is_valid(self, logits, temperature, partials=None, write_ctas=None, **kwargs) -> None:
        assert logits.dtype == torch.float32 and logits.dim() == 2, "logits: fp32 [rows, d]"


m31_softmax_mc = M31SoftmaxMc()


def softmax_mc(
    logits: torch.Tensor, temperature: Union[torch.Tensor, float, None]
) -> Optional[torch.Tensor]:
    """The sampler's softmax override: FlashInfer's softmax of logits [rows, d] at temperature, or None (the sampler
    runs stock FlashInfer) when the op is not built, rows exceed max_rows(), d exceeds MAX_VOCAB, no workspace can be
    had (first use inside a capture) or the input is outside the contract."""
    if not isinstance(logits, torch.Tensor) or not logits.is_cuda or logits.dim() != 2:
        return None
    rows = logits.shape[0]
    if rows < 1 or rows > max_rows() or logits.shape[1] > MAX_VOCAB or softmax_mc_op() is None:
        return None
    partials = prebuild(logits.device)
    if partials is None:
        return None
    ctas = write_ctas(rows)
    out = m31_softmax_mc(logits, temperature, partials, ctas)
    if out is not None and rows > NARROW_MAX_ROWS and rows not in _WIDE_LOGGED:
        _WIDE_LOGGED.add(rows)
        logger.info(
            f"[TP8WIDE] SOFTMAX taken at {rows} rows ({ctas} write CTAs per row; "
            "kill TRTLLM_M31_DISABLE_TP8WIDE_SOFTMAX=1)"
        )
    return out
