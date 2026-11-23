# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_HEAD_NVFP4 (opt-in): the DSpark drafter head's shard GEMM on an NVFP4 copy of this rank's vocab shard.

The alternative to DRAFT_HEAD_MXFP8 (dspark_draft_head: an MXFP8 copy, about 634 MB per GPU): the same
[B x K, 6144] x [100032, 6144]^T shard GEMM at the draft-length-7 batch-1 / batch-2 rows, on a copy with E2M1 codes and
one E4M3 scale per 16 columns (128x4-swizzled) under a per-shard FP32 global scale, padded to 100,096 zero rows:
307.5 MB of codes + 38.4 MB of scales = 345.9 MB per GPU, about half of the MXFP8 copy's bytes.

- Activation: the drafter's rows are quantized to NVFP4 (``torch.ops.trtllm.fp4_quantize``) under a global scale fixed
  at set-up from the drafter's final RMSNorm, whose output is the head's input. A normed row's squares sum to at most
  K, so no element of ``rms_norm(h) x g`` exceeds sqrt(K) x max|g|: the scale 2688 / (``ACT_SCALE_MARGIN`` x sqrt(K) x
  max|g|) keeps every 16-column block scale at or below E4M3's 448 for any input, and puts a typical block (maximum
  about 2 x |g|) at a block scale of about 11 x |g| / max|g|, far inside E4M3's normal range (2^-6 .. 448). It depends
  on g only through max|g|, so a final norm with small gains keeps full block-scale precision (a fixed scale of 1.0
  turns the scales of blocks whose maximum is below about 0.09 subnormal and below about 0.006 zero). alpha = 1 / (act
  scale x weight scale) stays on the device.
- GEMM: over the copy's padded 100,096 rows for every backend, as DRAFT_HEAD_MXFP8 runs its GEMM: [rows, 100096] BF16.
  The call returns the [rows, 100032] contiguous slice copy, or, with ``contiguous=False``, the [rows, 100032] window
  of the padded output (row stride 100,096): DRAFT_HEAD_NO_SLICE's contract (``padded_window``), the slice copy's bytes
  without the copy. The GEMM is ``DEFAULT_TACTIC`` (TRT-LLM's ``NVFP4GemmUnifiedRunner`` CUTLASS runner at its default
  config) unless the constructor is given another choice per row count (``plan_tactics``): a TRT-LLM NVFP4 runner
  (cutlass / cublaslt / cuda_core) and its sub-tactic, or FlashInfer's CuTe DSL block-scaled FP4 GEMM at its own
  per-shape choice or a given tactic tuple.
- Not bitwise against the BF16 head or DRAFT_HEAD_MXFP8 (4-bit operands): drafter-only precision. Opt-in, default off.
- Precedence: DRAFT_HEAD_NVFP4 and DRAFT_HEAD_MXFP8 serve the same call (``_S4State.head8``). With both opted in,
  DRAFT_HEAD_NVFP4 is set up first and, when every rank set it up, DRAFT_HEAD_MXFP8 is not built (its copy is not
  allocated) and its line names why; when DRAFT_HEAD_NVFP4 declines, DRAFT_HEAD_MXFP8 is set up as before.
- Rows: B x K for B in ``dspark_draft_head.BATCHES`` (7 and 14 at draft length 7); other row counts keep the BF16 head
  with one logged reason per row count.
- Set-up per rank: unfit checks, the drafter final norm's gains, the chunked quantize of the shard, one compile + run
  per row count, then per row count a self-check on rows shaped like the head's input (``rms_norm(randn) x g``) against
  the BF16 head (relative RMS below ``SELF_CHECK_REL_RMS``: catches a wrong layout or scale, not the NVFP4 rounding;
  finite; two runs bitwise; the window bitwise the slice copy at row stride 100,096), the fraction of activation block
  scales that are subnormal or zero (logged), then all-rank agreement before any rank uses it.
- Engagement line: ``[M3 dispatch] DRAFT_HEAD_NVFP4 taken: rank R: ... under CUDA-graph capture``, once per process and
  row count, at the first call recorded into a CUDA graph (the first eager call logs an ``eager`` line). It is not
  evidence that a graph replayed.

Opt-in: ``TRTLLM_M31_ENABLE_DRAFT_HEAD_NVFP4=1``, read at import.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F

from tensorrt_llm.logger import logger

from . import dspark_draft_head as _dhead8
from .kernels import m3_report
from .m31_switches import m31_enabled

DHEAD_FP4 = m31_enabled("DRAFT_HEAD_NVFP4")
SF_VEC = 16  # columns per E4M3 block scale (NVFP4)
FP4_RANGE = 448.0 * 6.0  # E4M3 max x E2M1 max: the global-scale numerator of the NVFP4 recipe
#: Headroom of the activation global scale over the RMSNorm bound sqrt(K) x max|g| (the BF16 rounding of the normed
#: rows and of the gain product, a few 2^-8).
ACT_SCALE_MARGIN = 1.02
#: Set-up self-check bound on the relative RMS against the BF16 head on head-input-shaped rows (NVFP4 W4A4 rounding is
#: about 0.1-0.15 on random data; a wrong layout or scale is about 1.4).
SELF_CHECK_REL_RMS = 0.3
SELF_CHECK_SEED = 20261101
CHUNK_ROWS = 8192  # weight rows quantized per call at set-up (a multiple of 128)

#: A GEMM choice: (backend, value): a TRT-LLM runner backend and its sub-tactic id, or ("flashinfer", None | tuple).
Spec = Tuple[str, object]
#: TRT-LLM's NVFP4 GEMM runners a tactic may name (``NVFP4GemmUnifiedRunner`` backends).
TRTLLM_BACKENDS = ("cutlass", "cublaslt", "cuda_core")
#: The GEMM when no tactic is pinned: the CUTLASS runner at its default config (the runner's own fallback).
DEFAULT_TACTIC: Spec = ("cutlass", -1)


def padded_n(n: int) -> int:
    return _dhead8.padded_n(n)


def quantize_weight(
    weight: torch.Tensor, chunk_rows: int = CHUNK_ROWS
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """The NVFP4 copy of a BF16 [N, K] weight padded to ``padded_n(N)`` zero rows: packed E2M1 codes [Np, K / 2]
    (uint8), the 128x4-swizzled E4M3 block scales (uint8, Np x K / 16 bytes) and the FP32 global scale (448 x 6 / the
    weight's max). Chunks of ``chunk_rows`` rows (a multiple of 128: each 128-row tile of the swizzled layout is one
    contiguous run of 128 x K / 16 bytes) concatenate to one quantize of the padded weight."""
    n, k = (int(s) for s in weight.shape)
    if chunk_rows % 128 or k % (4 * SF_VEC):
        raise ValueError(
            f"chunk_rows {chunk_rows} must be a multiple of 128, K {k} of {4 * SF_VEC}"
        )
    rows = padded_n(n)
    cols = k // SF_VEC
    # max |w| without an |w| copy of the shard (1.2 GB at TP2).
    amax = max(float(weight.max().float().item()), -float(weight.min().float().item()))
    gscale = FP4_RANGE / amax if amax > 0 else 1.0
    gs = torch.tensor([gscale], dtype=torch.float32, device=weight.device)
    codes = torch.zeros((rows, k // 2), dtype=torch.uint8, device=weight.device)
    sf = torch.empty((rows * cols,), dtype=torch.uint8, device=weight.device)
    for r0 in range(0, rows, chunk_rows):
        r1 = min(r0 + chunk_rows, rows)
        chunk = weight[r0 : min(r1, n)]
        if chunk.shape[0] != r1 - r0:
            padded = torch.zeros((r1 - r0, k), dtype=weight.dtype, device=weight.device)
            padded[: chunk.shape[0]] = chunk
            chunk = padded
        q, s = torch.ops.trtllm.fp4_quantize(chunk.contiguous(), gs, SF_VEC, False, True)
        codes[r0:r1] = q.view(torch.uint8).view(r1 - r0, k // 2)
        sf[r0 * cols : r1 * cols] = s.view(torch.uint8).reshape(-1)[: (r1 - r0) * cols]
    return codes, sf, gscale


# ------------------------------------------------------------------------------------------- activation scale


def norm_gains(norm, k: int) -> Optional[Tuple[torch.Tensor, float]]:
    """The drafter final RMSNorm's gains as (g, the bound's max|g|), or None when ``norm`` has no 1-D weight of ``k``
    entries or no positive finite gain. A Gemma-style norm (``use_gemma``: (1 + w) x) has g = 1 + w, and its bound
    takes the larger of max|1 + w| and max|w| (a fused all-reduce + norm kernel may apply either form)."""
    w = getattr(norm, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 1 or int(w.shape[0]) != int(k):
        return None
    w = w.detach().float()
    if getattr(norm, "use_gemma", False):
        g = w + 1.0
        gmax = max(float(g.abs().max().item()), float(w.abs().max().item()))
    else:
        g = w
        gmax = float(w.abs().max().item())
    if not (gmax > 0 and math.isfinite(gmax)):
        return None
    return g, gmax


def act_global_scale(gmax: float, k: int) -> float:
    """The activation's NVFP4 global scale for a head input ``rms_norm(h) x g`` with max|g| = ``gmax`` and K = ``k``:
    2688 / (``ACT_SCALE_MARGIN`` x sqrt(k) x gmax), so the largest possible block maximum (sqrt(k) x gmax, a row with
    all its energy in one column) maps to a block scale of 448 / ``ACT_SCALE_MARGIN``."""
    if not (gmax > 0 and math.isfinite(gmax)) or k <= 0:
        raise ValueError(f"max|g| {gmax} / K {k}: no activation scale")
    return FP4_RANGE / (ACT_SCALE_MARGIN * math.sqrt(k) * gmax)


def head_input_rows(
    m: int, k: int, gamma: Optional[torch.Tensor], generator, device
) -> torch.Tensor:
    """Rows shaped like the head's input: ``rms_norm(randn) x g`` (unit gains when ``gamma`` is None), BF16."""
    z = torch.randn((m, k), device=device, generator=generator)
    x = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True))
    if gamma is not None:
        x = x * gamma.to(device=device, dtype=torch.float32)
    return x.to(torch.bfloat16)


def act_block_scales(sf: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """The E4M3 block-scale bytes [rows, k / 16] of the first ``rows`` rows of a 128x4-swizzled NVFP4 scale buffer."""
    cols = k // SF_VEC
    pad = (cols + 3) // 4
    r = torch.arange(rows, device=sf.device).view(-1, 1)
    j = torch.arange(cols, device=sf.device).view(1, -1)
    index = ((r // 128) * pad + j // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (j % 4)
    return sf.view(torch.uint8).reshape(-1)[index.reshape(-1)].view(rows, cols)


def subnormal_or_zero(scales: torch.Tensor) -> torch.Tensor:
    """Which E4M3 bytes are subnormal or zero (exponent field 0)."""
    return (scales.view(torch.uint8) & 0x78) == 0


# ------------------------------------------------------------------------------------------------- tactics


def _flashinfer_unfit(m: int, value) -> Optional[str]:
    """Why a FlashInfer FP4 tactic tuple cannot run ``m`` rows (None: it can): the sm103 / sm107 kernels' swap_ab
    needs m % 8 == 0 (FlashInfer gemm_base's tactic list: ``m_aligned = m % 8 == 0``, ``if swap_ab and not m_aligned:
    continue``), so neither 7 nor 14 rows can take it; the sm100 kernel's swap_ab can."""
    if value is None:
        return None
    _tile, _cluster, swap_ab, _prefetch, kernel_type, _extra = value
    if kernel_type in ("sm103", "sm107") and swap_ab and m % 8:
        return f"FlashInfer's {kernel_type} FP4 kernel with swap_ab needs rows % 8 == 0, not {m}"
    return None


def plan_tactics(
    rows: Iterable[int], pins: Union[None, Spec, Dict[Optional[int], Spec]]
) -> Dict[int, Tuple[Spec, bool]]:
    """{row count: (choice, pinned)} for ``rows``: a row count's own pin, else the default pin (key None), else
    ``DEFAULT_TACTIC``. ``pins``: None, one choice for every row count, or a {rows or None: choice} map. A choice is
    (backend, value): ``(<cutlass|cublaslt|cuda_core>, <sub-tactic int, -1 the runner's default>)``,
    ``("flashinfer", None)`` (FlashInfer's own per-shape choice) or ``("flashinfer", <6-tuple>)``. Raises when a pin
    names a row count the head does not take or a FlashInfer tactic that cannot run its row count."""
    rows = tuple(sorted({int(m) for m in rows if int(m) > 0}))
    if pins is None:
        pins = {}
    elif isinstance(pins, tuple):
        pins = {None: pins}
    extra = sorted(m for m in pins if m is not None and m not in rows)
    if extra:
        raise ValueError(f"tactic pins rows {extra}, the head takes {rows}")
    plan: Dict[int, Tuple[Spec, bool]] = {}
    for m in rows:
        spec = pins.get(m, pins.get(None))
        pinned = spec is not None
        spec = spec if pinned else DEFAULT_TACTIC
        if spec[0] == "flashinfer":
            why = _flashinfer_unfit(m, spec[1])
            if why is not None:
                raise ValueError(f"tactic: {why}")
        plan[m] = (spec, pinned)
    return plan


def format_tactic(spec: Spec) -> str:
    backend, value = spec
    return backend if value is None else f"{backend}:{value}"


def _default_flashinfer_tactic(m: int, n: int, k: int, device) -> Tuple[tuple, str]:
    """FlashInfer's own untuned choice for this shape and where it came from: on SM107 the sm107 selector when its
    kernel imports, unless it picks an sm107 swap_ab tactic ``m`` rows cannot run (it scores ``m`` in its power-of-two
    bucket, so 7 and 14 rows take the 8- and 16-row picks), then the sm100 selector (whose kernel runs on SM107 too);
    else the sm100 selector. A function of the shape and the SM count: the same on every rank."""
    from flashinfer.gemm.kernels import utils as _u

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 7):
        try:
            from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm107 import (  # noqa: F401
                Sm107BlockScaledPersistentDenseGemmKernel,
            )
        except ImportError:
            pass
        else:
            tactic = _u._select_sm107_mm_fp4_cute_dsl_tactic(m, n, k, sms, SF_VEC)
            if _flashinfer_unfit(m, tactic) is None:
                return tactic, "sm107 selector"
            return (
                _u._select_sm100_mm_fp4_cute_dsl_tactic(m, n, k, sms, SF_VEC),
                "sm100 selector (the sm107 pick needs rows % 8 == 0)",
            )
    return _u._select_sm100_mm_fp4_cute_dsl_tactic(m, n, k, sms, SF_VEC), "sm100 selector"


def _act_quantizer(
    name: str,
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """The activation's NVFP4 quantizer (packed codes [M, K / 2], 128x4-swizzled E4M3 scales). "trtllm":
    torch.ops.trtllm.fp4_quantize; "cute-dsl": FlashInfer's CuTe DSL kernel (experimental there)."""
    if name == "trtllm":
        return lambda x, gs: torch.ops.trtllm.fp4_quantize(x, gs, SF_VEC, False, True)
    if name == "cute-dsl":
        import flashinfer

        return lambda x, gs: flashinfer.fp4_quantize(x, gs, SF_VEC, False, True, backend="cute-dsl")
    raise ValueError(f"unknown NVFP4 activation quantizer {name!r}")


def unfit(head, shard: Optional[int], k: int) -> Optional[str]:
    """Why a head cannot take the NVFP4 copy (None: it can): DRAFT_HEAD_MXFP8's conditions (the same head and shard)."""
    return _dhead8.unfit(head, shard, k)


class DraftHeadFp4:
    """The NVFP4 copy of one rank's head shard and its GEMMs; call it with the drafter's block rows (the
    DRAFT_HEAD_MXFP8 contract: ``serves_head``, ``__call__(x, contiguous=True)`` -> [rows, N] BF16 or None)."""

    #: DRAFT_HEAD_NO_SLICE's contract: ``__call__(x, contiguous=False)`` returns the [rows, n] window of a [rows, np]
    #: buffer (row stride np == padded_n(n)) holding the slice copy's bytes.
    padded_window = True

    def __init__(
        self,
        head,
        rows: Iterable[int],
        rank=None,
        quantizer: str = "trtllm",
        tactic: Union[None, Spec, Dict] = None,
        gains: Optional[Tuple[torch.Tensor, float]] = None,
        act_scale: Optional[float] = None,
    ) -> None:
        """``tactic``: None (``DEFAULT_TACTIC``), one choice, a {rows or None: choice} map or ``plan_tactics``'s plan.
        ``gains``: ``norm_gains`` of the drafter's final norm (None: unit gains). ``act_scale`` overrides the
        activation global scale (tests only)."""
        start = time.perf_counter()
        weight = head.weight
        self._head = [head]  # identity only: the target owns its head
        self.rank = rank
        self.n, self.k = (int(s) for s in weight.shape)
        self.np = padded_n(self.n)
        self.rows = tuple(sorted({int(m) for m in rows if int(m) > 0}))
        if (
            isinstance(tactic, dict)
            and tactic
            and all(
                isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], bool)
                for v in tactic.values()
            )
        ):
            plan = tactic  # plan_tactics's {rows: (choice, pinned)}
        else:
            plan = plan_tactics(self.rows, tactic)
        dev = weight.device
        self._gamma = None if gains is None else gains[0].to(device=dev, dtype=torch.float32)
        self.gmax = 1.0 if gains is None else float(gains[1])
        self.ascale = (
            float(act_scale) if act_scale is not None else act_global_scale(self.gmax, self.k)
        )
        self.codes, self.sf, self.wscale = quantize_weight(weight)
        self.bytes = self.codes.numel() + self.sf.numel()
        self._gs = torch.tensor([self.ascale], dtype=torch.float32, device=dev)
        self._alpha = torch.tensor(
            [1.0 / (self.ascale * self.wscale)], dtype=torch.float32, device=dev
        )
        self.tactics: Dict[int, Spec] = {}
        self.sources: Dict[int, str] = {}
        self._runners: Dict[str, object] = {}
        for m in self.rows:
            (backend, value), pinned = plan[m]
            if backend == "flashinfer":
                if "flashinfer" not in self._runners:
                    from flashinfer.gemm import gemm_base as _gb

                    major, minor = _gb.get_compute_capability(dev)
                    # The CuTe DSL FP4 runner ignores the workspace slot of its inputs (FlashInfer gemm_base: the
                    # forward unpacks it into "_"), so none is allocated.
                    self._runners["flashinfer"] = _gb._cute_dsl_gemm_fp4_runner(
                        major, minor, True, torch.bfloat16, True
                    )
                if value is None:
                    # "flashinfer" without a tuple: its own per-shape choice
                    value, source = _default_flashinfer_tactic(m, self.np, self.k, dev)
                    self.sources[m] = f"pinned flashinfer, {source}" if pinned else source
                else:
                    self.sources[m] = "pinned"
            elif backend in TRTLLM_BACKENDS:
                if backend not in self._runners:
                    from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
                        NVFP4GemmUnifiedRunner,
                    )

                    self._runners[backend] = NVFP4GemmUnifiedRunner(0, torch.bfloat16, [backend])
                value = int(value)
                self.sources[m] = "pinned" if pinned else "default"
            else:
                raise ValueError(f"unknown NVFP4 GEMM backend {backend!r}")
            self.tactics[m] = (backend, value)
        self.pinned = any(pinned for _, pinned in plan.values())
        self.quantizer = quantizer
        self._quantize = _act_quantizer(quantizer)
        # Compile (quantizer, GEMMs) and run once per served row count, outside capture.
        for m in self.rows:
            self._run(torch.zeros((m, self.k), dtype=torch.bfloat16, device=dev))
        torch.cuda.synchronize(dev)
        self.self_check, self.act_subnormal = self._self_check(head)
        self.setup_s = time.perf_counter() - start
        self._logged: Set[tuple] = set()

    @property
    def backend(self) -> str:
        """The GEMM backend(s) of the served row counts ("cutlass", or "cutlass+flashinfer" when they differ)."""
        return "+".join(sorted({b for b, _ in self.tactics.values()}))

    def serves_head(self, head) -> bool:
        return head is self._head[0]

    def _gemm(self, x: torch.Tensor) -> torch.Tensor:
        """The [m, np] BF16 logits of ``x`` over the padded copy, at the row count's tactic."""
        m = int(x.shape[0])
        backend, value = self.tactics[m]
        xq, xsf = self._quantize(x, self._gs)
        if backend == "flashinfer":
            out = torch.empty((m, self.np), dtype=torch.bfloat16, device=x.device)
            self._runners["flashinfer"](
                inputs=[
                    xq.view(torch.uint8).view(m, self.k // 2),
                    self.codes.t(),
                    xsf.view(torch.uint8).reshape(-1),
                    self.sf,
                    self._alpha,
                    torch.bfloat16,
                    out,
                    SF_VEC,
                    True,
                    None,
                ],
                tactic=value,
            )
            return out
        return self._runners[backend](
            [xq, self.codes, xsf, self.sf, self._alpha], tactic=(backend, value)
        )

    def _run(self, x: torch.Tensor, contiguous: bool = True) -> torch.Tensor:
        """The shard logits of ``x``: the contiguous [m, n] slice copy, or with ``contiguous`` False the [m, n] window
        of the padded [m, np] output (row stride np; DRAFT_HEAD_NO_SLICE)."""
        window = self._gemm(x)[:, : self.n]
        return window.contiguous() if contiguous else window

    def _self_check(self, head) -> Tuple[Dict[int, float], float]:
        """Per served row count, on head-input-shaped rows (a private generator): the relative RMS against the BF16
        head (raises above the bound or when not finite), two runs bitwise, the window bitwise the slice copy at row
        stride np. Returns ({rows: relative RMS}, the fraction of activation block scales that are subnormal or
        zero)."""
        dev = self.codes.device
        gen = torch.Generator(device=dev).manual_seed(SELF_CHECK_SEED)
        rel: Dict[int, float] = {}
        weak = blocks = 0
        for m in self.rows:
            x = head_input_rows(m, self.k, self._gamma, gen, dev)
            ref = F.linear(x, head.weight).float()
            got = self._run(x)
            again = self._run(x)
            win = self._run(x, contiguous=False)
            if tuple(got.shape) != (m, self.n) or not torch.equal(got, again):
                raise RuntimeError(
                    f"self-check at {m} rows: logits {tuple(got.shape)} not deterministic"
                )
            if (
                tuple(win.shape) != (m, self.n)
                or tuple(win.stride()) != (self.np, 1)
                or not torch.equal(win, got)
            ):
                raise RuntimeError(
                    f"self-check at {m} rows: the window {tuple(win.shape)} stride {tuple(win.stride())} is not the "
                    f"slice copy at row stride {self.np}"
                )
            r = ((got.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
            if not r < SELF_CHECK_REL_RMS or not torch.isfinite(got.float()).all():
                raise RuntimeError(
                    f"self-check relative RMS {r:.3g} at {m} rows against the BF16 head (bound {SELF_CHECK_REL_RMS})"
                )
            rel[m] = r
            scales = act_block_scales(self._quantize(x, self._gs)[1], m, self.k)
            weak += int(subnormal_or_zero(scales).sum().item())
            blocks += scales.numel()
        return rel, weak / max(blocks, 1)

    def why_not(self, x) -> Optional[str]:
        """Why a call does not take the copy (None: it does). Shapes and dtypes only: fixed per captured graph."""
        if not isinstance(x, torch.Tensor) or x.dim() != 2:
            return "not a 2-D activation"
        if int(x.shape[0]) not in self.rows:
            return f"{int(x.shape[0])} rows (taken: {self.rows})"
        if x.dtype != torch.bfloat16 or int(x.shape[1]) != self.k or not x.is_contiguous():
            return f"activation {x.dtype} {tuple(x.shape)} contiguous {x.is_contiguous()}"
        if x.device != self.codes.device:
            return f"activation on {x.device}, copy on {self.codes.device}"
        return None

    def __call__(self, x: torch.Tensor, contiguous: bool = True) -> Optional[torch.Tensor]:
        """This rank's vocab-shard logits [rows, N] (BF16; contiguous, or with ``contiguous`` False the window of the
        padded output: DRAFT_HEAD_NO_SLICE) of ``x`` from the NVFP4 copy, or None (the BF16 head runs) with one logged
        reason per row count."""
        rows = int(x.shape[0]) if isinstance(x, torch.Tensor) and x.dim() >= 1 else -1
        why = self.why_not(x)
        if why is not None:
            if ("off", rows) not in self._logged:
                self._logged.add(("off", rows))
                logger.info(
                    f"[M3 dispatch] DRAFT_HEAD_NVFP4 not taken: rank {self.rank}: {why}; BF16 draft head"
                )
            return None
        captured = torch.cuda.is_current_stream_capturing()
        if ("on", rows, captured) not in self._logged:
            self._logged.add(("on", rows, captured))
            what = (
                f"NVFP4 draft head at {rows} rows ([{rows}, {self.k}] x [{self.np}, {self.k}]^T, logits [{rows}, "
                f"{self.n}] {'contiguous' if contiguous else f'window, row stride {self.np}'}, GEMM "
                f"{format_tactic(self.tactics[rows])})"
            )
            logger.info(
                f"[M3 dispatch] DRAFT_HEAD_NVFP4 taken: rank {self.rank}: {what} under CUDA-graph capture"
                if captured
                else f"[M3 sm107 tp2] DRAFT_HEAD_NVFP4 eager on rank {self.rank}: {what}"
            )
        m3_report.work("DRAFT_HEAD_NVFP4", f"{rows} draft rows, shard {self.n} padded to {self.np}")
        return self._run(x, contiguous)


def _raise_if_cuda_fault(agree: Optional[Callable[[bool], bool]] = None) -> None:
    """A set-up failure that left the CUDA context faulted is raised (the rank cannot serve), not declined. Before it
    raises, this rank votes no in the set-up agreement (``agree``, a host-side collective) so the other ranks leave it
    and decline instead of waiting for a rank that will not reach it."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except RuntimeError as fault:
        if agree is not None:
            try:
                agree(False)
            except Exception:  # noqa: BLE001 - the CUDA fault below is the error to report
                pass
        raise RuntimeError("DRAFT_HEAD_NVFP4 set-up left a CUDA fault") from fault


def prebuild(
    state, lm_head, k: int, rank=None, tp_size=None, agree=None, norm=None
) -> Tuple[bool, str]:
    """Set DRAFT_HEAD_NVFP4 up on the Markov-chain state (every rank or none): (set up, its set-up line). Off by its
    switch: no collective (the environment is the same on every rank). On: every rank builds the copy (or names why
    not) and the ranks agree (``agree``, the worker's all-ranks check) before ``state.head8`` takes it. ``norm``: the
    drafter's final RMSNorm (the activation scale needs its gains)."""
    if not DHEAD_FP4:
        why = "default off: opt in with TRTLLM_M31_ENABLE_DRAFT_HEAD_NVFP4=1"
        return False, f"M3.1 DSpark DRAFT_HEAD_NVFP4: NVFP4 draft head off on rank {rank} ({why})"
    pins = None
    rows = tuple(b * k for b in _dhead8.BATCHES if b * k > 0)
    why = unfit(lm_head, getattr(state, "shard", None), k)
    gains = plan = None
    if why is None:
        plan = plan_tactics(rows, pins)
        gains = norm_gains(norm, int(lm_head.weight.shape[1]))
        if gains is None:
            why = "no drafter final RMSNorm weight of the head's K entries (the activation scale needs its gains)"
    head = None
    if why is None:
        try:
            head = DraftHeadFp4(lm_head, rows, rank=rank, tactic=plan, gains=gains)
        except Exception as e:  # noqa: BLE001 - any set-up failure (a compile, memory, the self-check) declines on
            # every rank through the agreement below instead of leaving the ranks on different heads
            _raise_if_cuda_fault(agree)
            why = f"set-up failed: {type(e).__name__}: {e}"
    ok = why is None
    if agree is not None:
        ok = agree(ok)
    if not ok:
        return (
            False,
            f"M3.1 DSpark DRAFT_HEAD_NVFP4: NVFP4 draft head off on rank {rank} ({why or 'off on another rank'})",
        )
    state.head8 = head
    gemms = ", ".join(
        f"{m}: {format_tactic(t)} ({head.sources[m]})" for m, t in head.tactics.items()
    )
    checks = ", ".join(f"{m}: {r:.3g}" for m, r in head.self_check.items())
    return True, (
        f"M3.1 DSpark DRAFT_HEAD_NVFP4: NVFP4 draft head set up on rank {rank}/{tp_size} (shard {head.n} rows padded "
        f"to {head.np}, K {head.k}, rows {head.rows}, GEMMs {{{gemms}}}, act global scale {head.ascale:.6g} (2688 / "
        f"({ACT_SCALE_MARGIN:g} x sqrt(K) x max|g| {head.gmax:.4g})), act blocks subnormal or zero "
        f"{100 * head.act_subnormal:.3g}%, weight global scale {head.wscale:.6g}, quantize {head.quantizer}, copy "
        f"{head.bytes / 1e6:.1f} MB, window [rows, {head.n}] of [rows, {head.np}], self-check rel RMS {{{checks}}}, "
        f"set-up {head.setup_s:.2f} s)"
    )


__all__ = [
    "ACT_SCALE_MARGIN",
    "DEFAULT_TACTIC",
    "DHEAD_FP4",
    "DraftHeadFp4",
    "act_block_scales",
    "act_global_scale",
    "format_tactic",
    "head_input_rows",
    "norm_gains",
    "padded_n",
    "plan_tactics",
    "prebuild",
    "quantize_weight",
    "subnormal_or_zero",
    "unfit",
]
