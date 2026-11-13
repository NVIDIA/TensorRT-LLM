# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 draft head (DHEAD_MX4): the DSpark drafter's Markov-chain head GEMM on an MXFP8 copy of this rank's vocab
shard, written straight into the chain's padded logits buffer (this target's kill switch:
TRTLLM_M31_DISABLE_TP8_DRAFT_HEAD_MXFP8=1).

The drafter borrows the target's BF16 head. Under the one-launch Markov chain every generation step computes this
rank's vocab shard of the block logits, [B x K, 6144] x [shard, 6144]^T, with the head's own BF16 GEMM and then copies
it into the padded buffer the chain reads. DHEAD_MX4 keeps a separate MXFP8 copy of the shard (E4M3 codes and one UE8M0
scale per 32 columns, 128x4-swizzled), quantizes the drafter's rows with FlashInfer's CuTe DSL quantizer (the target's
pinned quantize op) and runs FlashInfer's CuTe DSL MXFP8 GEMM at a pinned tactic straight into that buffer
(``kernels/m3_draft_head_gemm``): one quantize + one GEMM in place of the GEMM + copy.

- The -inf tail. The buffer's columns past the shard must hold -inf: the chain takes every column
  as a candidate, so a 0 there would win the argmax whenever every real logit is below 0. The GEMM therefore runs at
  the logical N (the shard): its TMA store is bounded by N and never writes the tail. The copy's storage is padded
  with zero rows to the 128-row tile for the swizzled scales and the last tile's loads only; those rows are
  never stored.
- The returned logits are the [B x K, shard] view of the buffer (row stride = the buffer's padded rows), not a
  contiguous tensor: the worker reshapes it to [B, K, shard] (a view), ``_refine_block_logits`` checks only its last
  dimension, and the chain reads the buffer. Under S4-logits (rejection draft sampling) the biased logits are rebuilt
  from a non-contiguous base by the in-tree lines (TAILBK_SPEC S4L takes contiguous logits only): the same values.
- The target's head is never touched: its weight, its padding_size (0, which S4's ``_head_is_sharded`` requires) and
  the target's own verify logits stay BF16.
- Row counts: B x K for B in ``BATCHES`` (batch 1 at draft length 7: 7 rows). Any other call goes to the
  BF16 head's path (PADFOLD, or the head's GEMM + the copy) or, when it is not a block call, to the drafter's own
  logits processor, with one logged line per (row count, reason).
- Not bitwise against the BF16 head (both operands are MXFP8-rounded): drafter-only precision (it changes draft
  tokens, not the target's distribution).
- Memory: the copy is allocated at prebuild (model load, before the KV cache is sized): padded shard rows x 6,144 B of
  codes + padded shard rows x 192 B of scales per GPU, so the KV pool is that much smaller (times the memory fraction)
  at the same free_gpu_memory_fraction.

Set-up (every rank or none, outside capture): the copy, the GEMM compile, one run per served row count and a
self-check on a scratch buffer of the same row stride (the tail stays -inf, the first and last 128-row tiles match
the fp32 product of the dequantized operands, two runs agree bit for bit); a check that fails declines the copy. A
set-up error that leaves the CUDA context faulted is raised instead (the next CUDA call would fail anyway). The
tactic is the same on every rank: ``DEFAULT_TACTIC`` or ``TRTLLM_M31_DHEAD_MX4_TACTIC``, for a micro bench's sweep
to pin. The variable's value has no spaces or commas, so launch harnesses that split environment lists on those can
carry it: ``<tile_m>x<tile_n>-<cluster_m>x<cluster_n>-swap|noswap-pf|nopf[-k<split_k>]``, e.g.
``128x16-2x1-swap-nopf-k1`` (``tactic_name``; the default). The Python literal ``((tile_m, tile_n), (cluster_m,
cluster_n), swap_ab, use_prefetch, split_k)`` is also read. An explicit value that is malformed, or that the GEMM cannot
run at the served shape, fails the set-up on every rank instead of declining.
"""

from __future__ import annotations

import ast
import os
import re
import time
from typing import Callable, Iterable, Optional, Set, Tuple

import torch

from .kernels import m3_report

PATH = "DHEAD_MX4"
SF_VEC = 32  # columns per UE8M0 scale
PAD_ROWS = 128  # the swizzled scale layout's row tile (and the GEMM's weight tile)
#: FlashInfer CuTe DSL MXFP8 GEMM tactic (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch, split_k) for this
#: head GEMM (M 7, K 6144): with swap_ab the N tile spans the 7 activation rows, so 16 pads less than 32. Deterministic,
#: -inf tail kept; its CUDA-graph replay is checked by the SM10x unit tests. TACTIC_ENV overrides it.
DEFAULT_TACTIC = ((128, 16), (2, 1), True, False, 1)
TACTIC_ENV = "TRTLLM_M31_DHEAD_MX4_TACTIC"
BATCHES = (1,)  # decode batches whose B x K draft rows the copy serves
CHUNK_ROWS = (
    8192  # weight rows quantized per call at set-up (bounds the transient copies to ~100 MB)
)
CHECK_SEED = 4243  # the self-check's activation seed (the same on every rank)
CHECK_REL_ERR = (
    2.0**-6
)  # self-check bound: BF16 output rounding + the FP32 summation order, relative to the max


def padded_n(n: int) -> int:
    """The copy's storage rows: the shard rounded up to the 128-row tile."""
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


def dequantize(codes: torch.Tensor, sf: torch.Tensor, r0: int, r1: int) -> torch.Tensor:
    """FP32 values of rows r0..r1-1 of an MXFP8 [rows, K] tensor (E4M3 ``codes``) with 128x4-swizzled UE8M0 ``sf``."""
    k = int(codes.shape[-1])
    cols = k // SF_VEC
    r = torch.arange(r0, r1, device=codes.device).view(-1, 1)
    c = torch.arange(cols, device=codes.device).view(1, -1)
    index = (
        (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * cols)
    )
    e = sf.view(torch.uint8).reshape(-1)[index.reshape(-1)].view(r1 - r0, cols).float()
    q = codes.view(-1, k)[r0:r1].float().view(r1 - r0, cols, SF_VEC)
    return (q * torch.exp2(e - 127.0).unsqueeze(-1)).view(r1 - r0, k)


_TACTIC_NAME = re.compile(r"(\d+)x(\d+)-(\d+)x(\d+)-(swap|noswap)-(pf|nopf)(?:-k(\d+))?")


def tactic_name(tactic) -> str:
    """The tactic's delimiter-free spelling (``parse_tactic`` reads it back):
    ``<tile_m>x<tile_n>-<cluster_m>x<cluster_n>-swap|noswap-pf|nopf-k<split_k>``."""
    (tm, tn), (cm, cn), swap_ab, prefetch, split_k = tactic
    return (
        f"{int(tm)}x{int(tn)}-{int(cm)}x{int(cn)}-{'swap' if swap_ab else 'noswap'}-"
        f"{'pf' if prefetch else 'nopf'}-k{int(split_k)}"
    )


def parse_tactic(text: str):
    """A tactic from ``tactic_name``'s spelling (``128x32-2x1-swap-nopf-k1``; ``-k<split_k>`` optional, default 1) or
    from its Python literal ``((tile_m, tile_n), (cluster_m, cluster_n), swap_ab, use_prefetch, split_k)``; ValueError
    when the text is neither (whether the GEMM takes it is checked at set-up)."""
    text = text.strip()
    named = _TACTIC_NAME.fullmatch(text)
    if named is not None:
        tm, tn, cm, cn, swap, pf, split_k = named.groups()
        return (
            (int(tm), int(tn)),
            (int(cm), int(cn)),
            swap == "swap",
            pf == "pf",
            int(split_k or 1),
        )
    try:
        value = ast.literal_eval(text)
        (tm, tn), (cm, cn), swap_ab, prefetch, split_k = value
    except (ValueError, SyntaxError, TypeError) as e:
        raise ValueError(
            f"{TACTIC_ENV}={text!r} is not <tile_m>x<tile_n>-<cluster_m>x<cluster_n>-swap|noswap-pf|nopf[-k<split_k>] "
            "(e.g. 128x32-2x1-swap-nopf-k1) or ((tile_m, tile_n), (cluster_m, cluster_n), swap_ab, use_prefetch, "
            "split_k)"
        ) from e
    if not all(
        isinstance(v, int) and not isinstance(v, bool) for v in (tm, tn, cm, cn, split_k)
    ) or not all(isinstance(v, bool) for v in (swap_ab, prefetch)):
        raise ValueError(
            f"{TACTIC_ENV}={text!r}: tile / cluster / split_k ints, swap_ab / use_prefetch bools"
        )
    return ((tm, tn), (cm, cn), swap_ab, prefetch, split_k)


def pinned_tactic(environ=None) -> Tuple[tuple, str]:
    """(tactic, where it came from): TACTIC_ENV when set (a malformed value raises ValueError), else DEFAULT_TACTIC."""
    text = (os.environ if environ is None else environ).get(TACTIC_ENV, "").strip()
    if not text:
        return DEFAULT_TACTIC, "default"
    return parse_tactic(text), TACTIC_ENV


def unfit(head, shard: Optional[int], k: int, row_stride: Optional[int]) -> Optional[str]:
    """Why a head cannot take the MXFP8 copy (None: it can). Config and shapes only, the same on every rank."""
    weight = getattr(head, "weight", None)
    if shard is None or row_stride is None:
        return (
            "S4 is not set up (the drafter head is the gathered BF16 head, not this rank's shard)"
        )
    if k <= 0:
        return f"draft length {k}"
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return "no 2-D head weight"
    if getattr(head, "bias", None) is not None or getattr(head, "padding_size", 0) != 0:
        return "the head has a bias or padding"
    if int(weight.shape[0]) != int(shard):
        return f"head rows {int(weight.shape[0])} are not the S4 shard {shard}"
    if int(weight.shape[1]) % (4 * SF_VEC):
        return f"hidden size {int(weight.shape[1])} is not a multiple of {4 * SF_VEC}"
    if int(row_stride) < int(shard) or (int(row_stride) * 2) % 16:
        return f"S4 buffer rows {row_stride} (>= the shard {shard}, 16-byte rows needed)"
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous() or not weight.is_cuda:
        return f"head weight {weight.dtype} (a contiguous CUDA BF16 weight needed)"
    if torch.cuda.get_device_capability(weight.device)[0] != 10:
        return "FlashInfer's CuTe DSL MXFP8 GEMM needs SM10x"
    return None


def tactic_invalid(tactic, n: int, rows: Iterable[int], k: int) -> Optional[str]:
    """Why the row-strided GEMM cannot run ``tactic`` at [m, k] x [n, k]^T for some m of ``rows`` (None: it can)."""
    from .kernels import m3_draft_head_gemm

    for m in rows:
        why = m3_draft_head_gemm.why_invalid(tactic, int(n), int(m), int(k))
        if why is not None:
            return why
    return None


def raise_if_cuda_fault(error: BaseException) -> None:
    """Raise when a set-up failure left this process's CUDA context faulted (a sticky error, such as an illegal address
    in the kernel): declining the copy would only move the failure to the next CUDA call, under another name. A
    failure that left the context usable returns (the caller declines)."""
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return
    try:
        torch.cuda.synchronize()
    except Exception as probe:  # noqa: BLE001 - any failure of the probe means the context is not usable
        raise RuntimeError(
            f"DHEAD_MX4 set-up left the CUDA context faulted ({type(error).__name__}: {error}; probe: {probe})"
        ) from error


def _cute_dsl_quantizer() -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """The activation quantizer: the target's pinned MXFP8 quantize op at FlashInfer's CuTe DSL tactic (codes and
    128x4-swizzled scales equal ``torch.ops.trtllm.mxfp8_quantize(x, True)``'s)."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8QuantizeRunner

    from .ops.gemm.m3_mxfp8_pinned_quantize import m3_mxfp8_pinned_quantize

    tactic = MXFP8QuantizeRunner.CUTE_DSL
    return lambda x: m3_mxfp8_pinned_quantize(x, tactic)


class DraftHeadMx4:
    """The MXFP8 copy of one rank's head shard and its pinned row-strided GEMM; call it with the drafter's block rows
    and the step's S4 buffer.

    ``row_stride``: the S4 buffer's padded rows (the model passes it; a micro bench may leave it None and write
    through ``into``). ``tactic``: None reads ``pinned_tactic()``. ``gemm``
    (``kernels/m3_draft_head_gemm.StridedMxfp8Gemm`` by default), ``act_quantizer`` and ``weight_quantizer``
    (``quantize_weight``) can be replaced, for tests off SM10x; ``check=False`` skips the set-up self-check."""

    def __init__(
        self,
        head,
        rows: Iterable[int],
        row_stride: Optional[int] = None,
        rank=None,
        tactic=None,
        *,
        gemm=None,
        act_quantizer: Optional[Callable] = None,
        weight_quantizer: Optional[Callable] = None,
        check: bool = True,
    ) -> None:
        start = time.perf_counter()
        weight = head.weight
        self._head = [head]  # identity only: the target owns its head
        self.rank = rank
        self.n, self.k = (int(s) for s in weight.shape)
        self.np = padded_n(self.n)
        self.row_stride = None if row_stride is None else int(row_stride)
        self.rows = tuple(sorted({int(m) for m in rows if int(m) > 0}))
        if not self.rows:
            raise ValueError("no served row counts")
        self.tactic = pinned_tactic()[0] if tactic is None else tactic
        self._gemm = gemm if gemm is not None else self._real_gemm(self.tactic)
        self.pdl = getattr(
            self._gemm, "enable_pdl", None
        )  # the GEMM's programmatic dependent launch
        self.codes, self.sf = (weight_quantizer or quantize_weight)(weight)
        self.bytes = self.codes.numel() * self.codes.element_size() + self.sf.numel()
        self.quantizer = "cute-dsl" if act_quantizer is None else "custom"
        self._quantize = act_quantizer or _cute_dsl_quantizer()
        self._logged: Set[Tuple[str, int, str]] = set()
        self._warm(check)
        self.setup_s = time.perf_counter() - start

    def _real_gemm(self, tactic):
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        from .kernels import m3_draft_head_gemm

        why = tactic_invalid(tactic, self.n, self.rows, self.k)
        if why is not None:
            raise ValueError(why)
        gemm = m3_draft_head_gemm.StridedMxfp8Gemm(tactic, enable_pdl=get_env_enable_pdl())
        gemm.compile()
        return gemm

    def _scratch_stride(self) -> int:
        return self.row_stride if self.row_stride is not None else self.np + PAD_ROWS

    def _warm(self, check: bool) -> None:
        """One run per served row count (compiles the quantizer), then the self-check; outside capture."""
        device = self.codes.device
        scratch = torch.empty(
            (max(self.rows), self._scratch_stride()), dtype=torch.bfloat16, device=device
        )
        for m in self.rows:
            scratch.fill_(float("-inf"))
            self.run_into(
                torch.zeros((m, self.k), dtype=torch.bfloat16, device=device), scratch[:m]
            )
        self.check = self.self_check() if check else "skipped"
        if self.codes.is_cuda:
            torch.cuda.synchronize(device)

    def with_tactic(self, tactic, check: bool = True) -> "DraftHeadMx4":
        """Switch the GEMM to another tactic (a micro bench's sweep; the copy is kept): validate, compile, warm and
        self-check it, outside capture. Returns self."""
        self._gemm = self._real_gemm(tactic)
        self.pdl = getattr(self._gemm, "enable_pdl", None)
        self.tactic = tactic
        self._warm(check)
        return self

    def serves_head(self, head) -> bool:
        return head is self._head[0]

    def run_into(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Quantize ``x`` [M, K] and write its logits into ``out[:M, :N]``; returns that [M, N] view. ``out``: a
        row-major BF16 [M, ldc] buffer (ldc >= N), or the [M, N] view of one (rows ldc apart, unit column stride).
        Nothing past column N of a row is written."""
        m = int(x.shape[0])
        if not out.is_contiguous():
            ldc = int(out.stride(0))
            if out.dim() != 2 or int(out.stride(1)) != 1 or ldc < self.n or int(out.shape[1]) > ldc:
                raise ValueError(
                    f"out {tuple(out.shape)} stride {out.stride()}: rows of a row-major buffer needed"
                )
            out = out.as_strided((int(out.shape[0]), ldc), (ldc, 1))
        xq, xsf = self._quantize(x)
        self._gemm(xq.view(m, self.k), xsf, self.codes, self.sf, self.n, out)
        return out[:m, : self.n]

    def into(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """``run_into`` (the micro bench's hook name)."""
        return self.run_into(x, out)

    def self_check(self) -> str:
        """Raise RuntimeError unless a run into a -inf scratch buffer of the served row stride keeps the tail at -inf,
        matches the fp32 product of the dequantized operands on the first and the last 128-row weight tiles, and
        repeats bit for bit; else 'ok (...)'."""
        m = max(self.rows)
        device = self.codes.device
        g = torch.Generator(device=device).manual_seed(CHECK_SEED)
        x = (torch.randn((m, self.k), generator=g, device=device) * 0.5).to(torch.bfloat16)
        scratch = torch.full(
            (m, self._scratch_stride()), float("-inf"), dtype=torch.bfloat16, device=device
        )
        y = self.run_into(x, scratch).clone()
        tail_ok = (
            bool(torch.isneginf(scratch[:, self.n :]).all()) if scratch.shape[1] > self.n else True
        )
        again = self.run_into(x, scratch)
        same = bool(torch.equal(y, again))
        xq, xsf = self._quantize(x)
        xd = dequantize(xq.view(m, self.k), xsf, 0, m)
        last = (self.n - 1) // PAD_ROWS * PAD_ROWS
        worst = 0.0
        for r0, r1 in ((0, min(PAD_ROWS, self.n)), (last, self.n)):
            ref = xd @ dequantize(self.codes, self.sf, r0, r1).t()
            top = float(ref.abs().max())
            err = float((y[:, r0:r1].float() - ref).abs().max())
            worst = max(worst, err / top if top > 0 else err)
        finite = bool(torch.isfinite(y).all())
        if not (tail_ok and same and finite and worst <= CHECK_REL_ERR):
            raise RuntimeError(
                f"DHEAD_MX4 self-check failed: tail -inf {tail_ok}, deterministic {same}, finite {finite}, "
                f"rel err {worst:.2e} (bound {CHECK_REL_ERR:.2e})"
            )
        return f"ok (rel err {worst:.1e})"

    def why_not(self, x, buf) -> Optional[str]:
        """Why a call does not take the copy (None: it does). Shapes and dtypes only: fixed per captured graph."""
        if not isinstance(x, torch.Tensor) or x.dim() != 2:
            return "not a 2-D activation"
        rows = int(x.shape[0])
        if rows not in self.rows:
            return f"{rows} rows (served: {self.rows})"
        if x.dtype != torch.bfloat16 or int(x.shape[1]) != self.k or not x.is_contiguous():
            return f"activation {x.dtype} {tuple(x.shape)} contiguous {x.is_contiguous()}"
        if x.device != self.codes.device:
            return f"activation on {x.device}, copy on {self.codes.device}"
        if not isinstance(buf, torch.Tensor):
            return "no S4 buffer for this batch"
        if self.row_stride is None:
            return "no S4 buffer row stride set (a bench copy)"
        if (
            buf.dtype != torch.bfloat16
            or not buf.is_contiguous()
            or int(buf.shape[-1]) != self.row_stride
            or buf.numel() != rows * self.row_stride
            or buf.device != x.device
        ):
            return f"S4 buffer {buf.dtype} {tuple(buf.shape)} (a contiguous BF16 [B, K, {self.row_stride}] needed)"
        return None

    def __call__(
        self,
        x: torch.Tensor,
        head,
        buf: Optional[torch.Tensor],
        fallback: str = "BF16 draft head",
    ) -> Optional[torch.Tensor]:
        """This rank's vocab-shard logits of ``x`` from the MXFP8 copy, written into ``buf`` (the step's S4 buffer,
        [B, K, row_stride]) and returned as its [B x K, N] view; None when ``head`` is not the copy's or the call is
        not one it takes, with one logged line per (row count, reason) naming ``fallback``, the path that serves the
        call instead (the S4 processor passes "drafter processor" for a call that is not a block call)."""
        rows = int(x.shape[0]) if isinstance(x, torch.Tensor) and x.dim() >= 1 else -1
        if not self.serves_head(head):
            why, kind = "the call's head is not the head the copy was built from", "head"
        else:
            why, kind = self.why_not(x, buf), "off"
        if why is not None:
            if (kind, rows, why) not in self._logged:
                self._logged.add((kind, rows, why))
                m3_report.uncovered(PATH, f"rank {self.rank}: {why}; {fallback}")
            return None
        if ("on", rows, "") not in self._logged:
            self._logged.add(("on", rows, ""))
            m3_report.taken(
                PATH,
                rows,
                f"MXFP8 draft head into the S4 buffer: [{rows}, {self.k}] x [{self.n}, {self.k}]^T, row stride "
                f"{self.row_stride}, tactic {self.tactic}",
            )
        return self.run_into(x, buf.view(rows, self.row_stride))
