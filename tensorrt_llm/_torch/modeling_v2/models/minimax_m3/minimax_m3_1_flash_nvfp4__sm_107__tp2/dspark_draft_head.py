# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DHEAD_MX8: the DSpark drafter's head GEMM on an MXFP8 copy of this rank's vocab shard (opt-in).

The drafter borrows the target's BF16 head. Under S4 (the one-launch Markov chain) every served generation step
computes this rank's vocab shard of the block logits, [B x K, 6144] x [100032, 6144]^T, with the head's own BF16 GEMM
(cuBLAS nvjet: 94.29 us at 14 rows on SM107, about 13 TB/s on 1.23 GB). DHEAD_MX8 keeps a separate
MXFP8 copy of the shard (E4M3 codes and one UE8M0 scale per 32 columns, 128x4-swizzled), quantizes the drafter's rows
with FlashInfer's CuTe DSL quantizer and runs FlashInfer's CuTe DSL MXFP8 GEMM at a pinned tactic: 1.65 + 48.80 =
50.87 us at 14 rows, 50.93 at 7 (same job).

- The shard (100,032 rows at TP2) is not a multiple of the GEMM's 128-row tile, so the copy is padded to 100,096 rows
  with zero rows and the logits are sliced back to the shard (a contiguous [rows, 100032] BF16 tensor, as the BF16
  head returns). The target's head is never touched: its weight, its padding_size (0, which S4's
  ``_head_is_sharded`` requires) and the target's own verify logits stay BF16.
- Served row counts: B x K for B in ``BATCHES`` (K7: 7 and 14 rows, the measured ones). Any other row count, and any
  input it does not take, falls back to the BF16 head with one logged reason per row count.
- Not bitwise against the BF16 head (both operands are MXFP8-rounded): drafter-only precision, gated on natural
  acceptance; default off until that gate passes.
- Memory: the copy is allocated at prebuild (model load, before the KV cache is sized), about 634 MB per GPU at TP2
  (615 MB codes + 19 MB scales), so the KV pool is that much smaller at the same free_gpu_memory_fraction.

Compiles (the quantizer per (K, dtype, PDL, layout); the GEMM once per tactic) and one run per served row count happen
at set-up, outside capture, on every rank; a forward never compiles. The tactic and the shapes are the same on every
rank, so the ranks run the same kernels.

DHEAD_NOSLICE (``TRTLLM_M31_DISABLE_DRAFT_HEAD_NO_SLICE=1`` restores the slice copy): on the greedy ("tokens")
S4 steps the worker asks for the logits as the [rows, 100032] window of the padded [rows, 100096] GEMM output (row
stride 100,096) instead of its contiguous copy (``out[:, :n].contiguous()``: an elementwise copy kernel of 2.8 MB per
14-row step, 5.2 us on SM107 TP2 b2). Under MARKOV16's fold the chain reads that window directly (its kernel takes
the row stride and reads only the shard's columns, the rest as -inf: ``ops/spec/m31_markov16.base_stride``); on a step
whose chain reads the S4 padded buffer, the one buffer copy reads the window. The same values in the same order: the
draft tokens are bitwise. Any other step (S4-logits, a call the worker does not mark) keeps the contiguous copy.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Optional, Set, Tuple

import torch

from tensorrt_llm.logger import logger

from .kernels import m3_report
from .m31_switches import m31_disabled

# DHEAD_NOSLICE (see the module docstring): default ON with DHEAD_MX8; kill switch
# TRTLLM_M31_DISABLE_DRAFT_HEAD_NO_SLICE=1, read at import.
DHEAD_NOSLICE = not m31_disabled("DRAFT_HEAD_NO_SLICE")

SF_VEC = 32  # columns per UE8M0 scale
PAD_ROWS = 128  # the GEMM's N tile and the swizzled scale layout's row tile
#: FlashInfer CuTe DSL MXFP8 GEMM tactic (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch, split_k): the fastest
#: deterministic valid tactic of the TP2 micro at M 14 and 7 (N 100,096, K 6144; 128-131 valid, 25 timed). Non-split-K.
TACTIC = ((128, 32), (2, 1), True, False, 1)
BATCHES = (
    1,
    2,
)  # decode batches whose B x K draft rows the copy serves (b2t16 GEN: batches 1 and 2)
CHUNK_ROWS = (
    8192  # weight rows quantized per call at set-up (bounds the transient BF16 copy to ~100 MB)
)


def padded_n(n: int) -> int:
    """The copy's row count: the shard rounded up to the GEMM's 128-row tile."""
    return (int(n) + PAD_ROWS - 1) // PAD_ROWS * PAD_ROWS


def quantize_weight(
    weight: torch.Tensor, chunk_rows: int = CHUNK_ROWS
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The MXFP8 copy of a BF16 [N, K] weight, padded to ``padded_n(N)`` zero rows: E4M3 codes [Np, K] and the
    128x4-swizzled UE8M0 scales (uint8, Np x K / 32 bytes). Quantized ``chunk_rows`` rows at a time (a multiple of 128:
    each 128-row tile of the swizzled layout is one contiguous run of 128 x K / 32 bytes, so the chunks' scales
    concatenate to the whole tensor's); the codes and scales equal one ``mxfp8_quantize`` of the padded weight."""
    n, k = (int(s) for s in weight.shape)
    if chunk_rows % PAD_ROWS or k % (4 * SF_VEC):
        raise ValueError(
            f"chunk_rows {chunk_rows} must be a multiple of {PAD_ROWS}, K {k} of {4 * SF_VEC}"
        )
    rows = padded_n(n)
    cols = k // SF_VEC
    codes = torch.zeros((rows, k), dtype=torch.float8_e4m3fn, device=weight.device)
    sf = torch.empty((rows * cols,), dtype=torch.uint8, device=weight.device)
    for r0 in range(0, rows, chunk_rows):
        r1 = min(r0 + chunk_rows, rows)
        chunk = weight[r0 : min(r1, n)]
        if chunk.shape[0] != r1 - r0:
            padded = torch.zeros((r1 - r0, k), dtype=weight.dtype, device=weight.device)
            padded[: chunk.shape[0]] = chunk
            chunk = padded
        q, s = torch.ops.trtllm.mxfp8_quantize(chunk.contiguous(), True)
        codes[r0:r1] = q.view(r1 - r0, k)
        sf[r0 * cols : r1 * cols] = s.view(torch.uint8).reshape(-1)[: (r1 - r0) * cols]
    return codes, sf


def _quantizer(name: str) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """The activation quantizer (MXFP8, 128x4-swizzled scales). "cute-dsl": FlashInfer's CuTe DSL kernel (1.65 us at
    14 rows); "trtllm": torch.ops.trtllm.mxfp8_quantize (2.38 us). TP2 micro: both give the same codes and scales."""
    if name == "cute-dsl":
        import flashinfer

        def quantize(x):
            return flashinfer.mxfp8_quantize(
                x, is_sf_swizzled_layout=True, alignment=SF_VEC, enable_pdl=None, backend="cute-dsl"
            )

        return quantize
    if name == "trtllm":
        return lambda x: torch.ops.trtllm.mxfp8_quantize(x, True)
    raise ValueError(f"unknown MXFP8 quantizer {name!r}")


def unfit(head, shard: Optional[int], k: int) -> Optional[str]:
    """Why a head cannot take the MXFP8 copy (None: it can). Config and shapes only, the same on every rank."""
    weight = getattr(head, "weight", None)
    if shard is None:
        return (
            "S4 is not set up (the drafter head is the gathered BF16 head, not this rank's shard)"
        )
    if k <= 0:
        return f"draft length {k}"
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return "no 2-D head weight"
    if getattr(head, "bias", None) is not None or getattr(head, "padding_size", 0) != 0:
        return "the head has a bias or padding"
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous() or not weight.is_cuda:
        return f"head weight {weight.dtype} (a contiguous CUDA BF16 weight needed)"
    if int(weight.shape[0]) != int(shard):
        return f"head rows {int(weight.shape[0])} are not the S4 shard {shard}"
    if int(weight.shape[1]) % (4 * SF_VEC):
        return f"hidden size {int(weight.shape[1])} is not a multiple of {4 * SF_VEC}"
    if torch.cuda.get_device_capability(weight.device)[0] != 10:
        return "FlashInfer's CuTe DSL MXFP8 GEMM needs SM10x"
    return None


class DraftHeadMx8:
    """The MXFP8 copy of one rank's head shard and its pinned GEMM; call it with the drafter's block rows."""

    # DHEAD_NOSLICE's contract (the drafter lane's DraftHeadFp4 declares it too): ``__call__(x, contiguous=False)``
    # returns the [rows, n] window of the padded [rows, np] output. The S4 worker asks a draft head copy for the window
    # only when the copy declares this (a copy without it keeps the contiguous slice and is called as ``head8(x)``).
    padded_window = True

    def __init__(self, head, rows: Iterable[int], rank=None, quantizer: str = "cute-dsl") -> None:
        start = time.perf_counter()
        weight = head.weight
        self._head = [head]  # identity only: the target owns its head
        self.rank = rank
        self.n, self.k = (int(s) for s in weight.shape)
        self.np = padded_n(self.n)
        self.rows = tuple(sorted({int(m) for m in rows if int(m) > 0}))
        self.codes, self.sf = quantize_weight(weight)
        self.bytes = self.codes.numel() * self.codes.element_size() + self.sf.numel()
        from flashinfer.gemm import gemm_base as _gb

        major, minor = _gb.get_compute_capability(weight.device)
        self._runner = _gb._cute_dsl_gemm_mxfp8_runner(major, minor, True, torch.bfloat16)
        self._workspace = _gb._get_cache_buf(
            "mm_mxfp8_workspace", _gb.DEFAULT_WORKSPACE_SIZE, weight.device
        )
        self.quantizer = quantizer
        self._quantize = _quantizer(quantizer)
        # Compile (quantizer, GEMM) and run once per served row count, outside capture.
        for m in self.rows:
            self._run(torch.zeros((m, self.k), dtype=torch.bfloat16, device=weight.device))
        torch.cuda.synchronize(weight.device)
        self.setup_s = time.perf_counter() - start
        self._logged: Set[Tuple[str, int]] = set()

    def serves_head(self, head) -> bool:
        return head is self._head[0]

    def _run(self, x: torch.Tensor, contiguous: bool = True) -> torch.Tensor:
        """The shard logits of ``x``: contiguous [m, n] (the slice copy), or with ``contiguous`` False the [m, n]
        window of the padded [m, np] output (row stride np; DHEAD_NOSLICE)."""
        m = int(x.shape[0])
        xq, xsf = self._quantize(x)
        out = torch.empty((m, self.np), dtype=torch.bfloat16, device=x.device)
        self._runner(
            inputs=[
                xq.view(m, self.k),
                self.codes.t(),
                xsf,
                self.sf,
                torch.bfloat16,
                out,
                self._workspace,
            ],
            tactic=TACTIC,
        )
        return out[:, : self.n].contiguous() if contiguous else out[:, : self.n]

    def why_not(self, x) -> Optional[str]:
        """Why a call does not take the copy (None: it does). Shapes and dtypes only: fixed per captured graph."""
        if not isinstance(x, torch.Tensor) or x.dim() != 2:
            return "not a 2-D activation"
        if int(x.shape[0]) not in self.rows:
            return f"{int(x.shape[0])} rows (served: {self.rows})"
        if x.dtype != torch.bfloat16 or int(x.shape[1]) != self.k or not x.is_contiguous():
            return f"activation {x.dtype} {tuple(x.shape)} contiguous {x.is_contiguous()}"
        if x.device != self.codes.device:
            return f"activation on {x.device}, copy on {self.codes.device}"
        return None

    def __call__(self, x: torch.Tensor, contiguous: bool = True) -> Optional[torch.Tensor]:
        """This rank's vocab-shard logits [rows, N] (BF16; contiguous, or with ``contiguous`` False the window of the
        padded output: DHEAD_NOSLICE) of ``x`` from the MXFP8 copy, or None (the BF16 head runs) with one logged reason
        per row count."""
        rows = int(x.shape[0]) if isinstance(x, torch.Tensor) and x.dim() >= 1 else -1
        why = self.why_not(x)
        if why is not None:
            if ("off", rows) not in self._logged:
                self._logged.add(("off", rows))
                logger.info(
                    f"[M3 dispatch] DHEAD_MX8 not taken on rank {self.rank}: {why}; BF16 draft head"
                )
            return None
        if ("on", rows) not in self._logged:
            self._logged.add(("on", rows))
            logger.info(
                f"[M3 dispatch] DHEAD_MX8 taken on rank {self.rank}: MXFP8 draft head at {rows} rows "
                f"([{rows}, {self.k}] x [{self.np}, {self.k}]^T, logits [{rows}, {self.n}])"
            )
        m3_report.work("DHEAD_MX8", f"{rows} draft rows, shard {self.n} padded to {self.np}")
        return self._run(x, contiguous)
