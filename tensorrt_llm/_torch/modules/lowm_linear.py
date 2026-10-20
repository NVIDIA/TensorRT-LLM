# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-sized (small M) projections on one register-prefetch warp-MMA kernel (Qwen3.5 decode).

At decode the per-tensor FP8 (W8A8) projections of Qwen3.5 run M = 7 * batch rows (target verify) or
M = batch rows (MTP draft steps). There they are latency bound: an activation-quantize kernel plus a
cuBLAS FP8 GEMM cost ~8-9 us for a weight read that takes ~1.5 us at HBM speed. ``lowm_gemm`` runs
one CuTe DSL kernel instead (``cute_dsl_kernels/blackwell/lowm_mma_gemm``): weights and scales are
loaded into registers before ``griddepcontrol.wait``, the activations after it, and the dependent
launch is released once the weight loads are in flight.

FP8 weights take FP8 activations (``mma.sync`` E4M3 x E4M3, FP32 accumulation, ``acc * s_in * s_w``):
either the producer's E4M3 output or BF16 activations quantized in registers with the module's static
input scale, with the arithmetic of the standalone quantize kernel (same FP8 values). BF16 weight
segments (``in_proj_ba``, the shared-expert gate, the BF16 MTP layer) take BF16 activations.

Switches (all default off; read on every call so tests and A/Bs can flip them at run time):

* ``TRTLLM_Q35_LOWM_FP8=1``: FP8 projections opted in by the model (``module._q35_lowm = True``).
* ``TRTLLM_Q35_LOWM_FP8_ACT=quant|bf16``: what a BF16 input does on an FP8 weight: ``quant``
  (default, W8A8 numerics) or ``bf16`` (BF16 activations times the converted weights, no
  activation quantization).
* ``TRTLLM_Q35_LOWM_GDN_BA=1``: GDN ``in_proj_ba`` (BF16) in the same launch as ``in_proj_qkvz``.
* ``TRTLLM_Q35_LOWM_SE=1``: shared expert as two launches (gate_up + SwiGLU + gate, down).
* ``TRTLLM_Q35_LOWM_BF16=1``: BF16 projections opted in by the model (``module._q35_lowm_bf16``).
* ``TRTLLM_Q35_LOWM_MAX_M``: largest row count that takes these kernels (default: per shape, the
  measured crossover against the stock path, see ``_TACTICS``).
* ``TRTLLM_Q35_LOWM_SKIP``: projection labels that keep the stock path (A/Bs), separated by ``,`` ``:``
  or ``+``, e.g. ``gdn.in_proj_qkvz+attn.qkv_proj`` (labels: gdn.in_proj_qkvz, gdn.in_proj_ba,
  gdn.out_proj, attn.qkv_proj, attn.o_proj, mtp.fc, se).
* ``TRTLLM_Q35_LOWM_CAPS``: per-label row limits on top of the measured ones, ``<label>:<rows>`` pairs
  separated by ``+`` or ``,`` (e.g. ``se:16+gdn.in_proj_qkvz:8``).
* ``TRTLLM_Q35_LOWM_EARLY_TRIGGER`` (1): release the dependent launch once the weights are in flight.
* ``TRTLLM_Q35_LOWM_TACTIC``: ``<N>x<K>:<k_warps>x<n_tiles>[x<ksplit>[x<swap>[x<min_blocks>]]],...``
  per-shape overrides (tuning; ``ksplit`` = CTAs per cluster splitting K, ``swap`` = 1 for swapped MMA
  operands at M <= 8, ``min_blocks`` = launch-bounds CTAs per SM).
* ``TRTLLM_Q35_LOWM_CHECK=1``: numerics check (eager runs only): every low-M call is also computed on the
  stock path and per-projection error statistics are logged every 200 calls and at exit.
"""

from __future__ import annotations

import functools
import os
from typing import Dict, Optional, Tuple, Union

import torch

_NUM_SMS_DEFAULT = 148


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


@functools.lru_cache(maxsize=None)
def _device_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


@functools.lru_cache(maxsize=None)
def _num_sms() -> int:
    if not torch.cuda.is_available():
        return _NUM_SMS_DEFAULT
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def lowm_fp8_enabled() -> bool:
    return _env_on("TRTLLM_Q35_LOWM_FP8") and _device_supported()


def lowm_bf16_enabled() -> bool:
    return _env_on("TRTLLM_Q35_LOWM_BF16") and _device_supported()


def _bf16_label_enabled(label) -> bool:
    # GDN in_proj_ba also runs on its own low-M launch with TRTLLM_Q35_LOWM_GDN_BA when the qkvz
    # projection gets FP8 activations (no fused launch then).
    return lowm_bf16_enabled() or (label == "gdn.in_proj_ba" and lowm_gdn_ba_enabled())


def lowm_gdn_ba_enabled() -> bool:
    return lowm_fp8_enabled() and _env_on("TRTLLM_Q35_LOWM_GDN_BA")


def lowm_se_enabled() -> bool:
    return _env_on("TRTLLM_Q35_LOWM_SE") and _device_supported()


def lowm_max_m() -> int:
    """Global row cap (TRTLLM_Q35_LOWM_MAX_M, default 32; per-shape limits in ``lowm_rows_ok``)."""
    return int(os.environ.get("TRTLLM_Q35_LOWM_MAX_M", "32"))


@functools.lru_cache(maxsize=None)
def _caps(spec: str) -> Dict[str, int]:
    out = {}
    for item in spec.replace(" ", "").replace("+", ",").split(","):
        if item:
            label, rows = item.rsplit(":", 1)
            out[label] = int(rows)
    return out


def lowm_label_rows_ok(label, m: int) -> bool:
    """TRTLLM_Q35_LOWM_CAPS: whether ``m`` rows are within the per-label limit (no limit: True)."""
    spec = os.environ.get("TRTLLM_Q35_LOWM_CAPS", "")
    if not spec or not isinstance(label, str):
        return True
    cap = _caps(spec).get(label)
    return cap is None or m <= cap


def lowm_skipped(label, m: Optional[int] = None) -> bool:
    """TRTLLM_Q35_LOWM_SKIP: projection labels that keep the stock path (separated by ``,``, ``:`` or
    ``+``; srun --export cannot carry commas)."""
    if m is not None and not lowm_label_rows_ok(label, m):
        return True
    skip = os.environ.get("TRTLLM_Q35_LOWM_SKIP", "")
    if not skip or not isinstance(label, str):
        return False
    return label in skip.replace(" ", "").replace(":", ",").replace("+", ",").split(",")


def lowm_fp8_act() -> str:
    act = os.environ.get("TRTLLM_Q35_LOWM_FP8_ACT", "quant")
    if act not in ("quant", "bf16"):
        raise ValueError(f"TRTLLM_Q35_LOWM_FP8_ACT={act!r} must be quant or bf16")
    return act


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def _early_trigger() -> bool:
    return _env_on("TRTLLM_Q35_LOWM_EARLY_TRIGGER", "1")


Tactic = Tuple[int, int, int, int, int]  # (k_warps, n_tiles, ksplit, swap_ab, min_blocks)


def _tactic3(t) -> Tactic:
    """Normalize a (k_warps, n_tiles[, ksplit[, swap_ab[, min_blocks]]]) tactic to five entries."""
    return (
        t[0],
        t[1],
        t[2] if len(t) > 2 else 1,
        t[3] if len(t) > 3 else 0,
        t[4] if len(t) > 4 else 1,
    )


@functools.lru_cache(maxsize=None)
def _tactic_overrides(spec: str) -> Dict[Tuple[int, int], Tactic]:
    out = {}
    for item in spec.replace(" ", "").replace("+", ",").split(","):
        if not item:
            continue
        try:
            shape, tactic = item.split(":")
            n, k = (int(v) for v in shape.lower().split("x"))
            out[(n, k)] = _tactic3(tuple(int(v) for v in tactic.lower().split("x")))
        except ValueError as e:
            raise ValueError(f"bad TRTLLM_Q35_LOWM_TACTIC entry {item!r}") from e
    return out


# Measured on GB300: chains of up to 192 distinct weight sets in one CUDA graph, and the same with a non-PDL kernel
# after every GEMM ("serial"). Key (N, K, mode) with mode fp8 / quant / bf16act (FP8 weights), bf16 (BF16 weights),
# mixed / mixed8 (GDN in_proj_qkvz FP8 + in_proj_ba BF16 in one launch, BF16 activations quantized in registers /
# E4M3 activations from the producer); value: ((max rows, tactic), ...) in increasing row order, tactic =
# (k_warps, n_tiles[, ksplit[, swap_ab[, min_blocks]]]). The last max rows is the crossover: beyond it the stock path
# (activation quantize + cuBLAS) was as fast or faster. Shapes that are not listed keep the stock path (lowm_rows_ok).
_TACTICS = {
    (2560, 4096, "fp8"): ((8, (16, 4, 1, 1)), (16, (8, 1, 1, 0, 3))),
    (2560, 4096, "quant"): ((8, (16, 4, 1, 1)), (16, (16, 4))),
    (2560, 4096, "mixed"): ((8, (16, 4, 1, 1)), (16, (16, 4))),
    (2560, 4096, "mixed8"): ((8, (16, 2, 1, 1, 2)), (16, (8, 4))),
    (2560, 4096, "bf16act"): ((8, (8, 2)),),
    (4096, 1024, "fp8"): ((8, (8, 2, 1, 1)), (24, (8, 2))),
    (4096, 1024, "quant"): ((8, (8, 2, 1, 1)), (32, (8, 2))),
    (256, 4096, "quant"): ((8, (16, 2, 1, 1)), (16, (16, 2)), (32, (8, 2, 2))),
    (256, 4096, "fp8"): ((8, (16, 2, 1, 1)), (16, (16, 2)), (32, (8, 2, 2))),
    (4096, 128, "quant"): ((8, (2, 2)), (14, (1, 2)), (18, (2, 4)), (32, (1, 4))),
    (4096, 128, "fp8"): ((8, (2, 2)), (14, (1, 2)), (18, (2, 4)), (32, (1, 4))),
    (16, 4096, "bf16"): ((16, (32, 1)), (32, (16, 1))),
    (512, 4096, "bf16"): ((8, (32, 1)), (16, (16, 1))),
    (2560, 4096, "bf16"): ((8, (16, 2, 1, 1)), (16, (8, 2, 1, 0, 2))),
    (4096, 1024, "bf16"): ((8, (8, 1, 1, 1)), (32, (8, 2))),
    (256, 4096, "bf16"): ((8, (16, 1)), (16, (8, 1)), (32, (16, 1))),
    (4096, 128, "bf16"): ((4, (2, 1)), (16, (1, 2))),
    # Per-rank shapes at attention TP16 (TEP16): GDN in_proj_qkvz 1280 (+ in_proj_ba 8), attention qkv
    # 1536, out_proj / o_proj K = 512, shared expert inter 64 (gate_up 128, down K = 64), MTP BF16 qkv / o / fc.
    # Every post-all-reduce projection (in_proj / qkv in all modes, MTP qkv / fc) runs low-M up to M = 32: at M 19-32
    # it is a few percent faster than stock standalone (5.1-5.2 vs 5.3 us with in-register quant), and behind an
    # early-releasing all-reduce its weight prefetch hides ~1 us more.
    (1280, 4096, "fp8"): ((8, (16, 2, 1, 1)), (16, (16, 2)), (32, (8, 2))),
    (1280, 4096, "quant"): ((8, (16, 2, 1, 1)), (18, (16, 2)), (32, (8, 2))),
    (1280, 4096, "mixed"): ((8, (16, 2, 1, 1)), (18, (16, 2)), (32, (8, 2))),
    (1280, 4096, "mixed8"): ((8, (16, 2, 1, 1)), (16, (16, 2)), (32, (8, 2))),
    (1536, 4096, "fp8"): ((8, (16, 2, 1, 1)), (32, (8, 2))),
    (1536, 4096, "quant"): ((8, (16, 2, 1, 1)), (18, (16, 2)), (32, (8, 2))),
    (4096, 512, "fp8"): ((8, (4, 2, 1, 1, 3)), (16, (8, 2))),
    (4096, 512, "quant"): ((16, (4, 4)), (32, (8, 2))),
    (128, 4096, "quant"): ((8, (8, 2, 4, 1)), (16, (8, 2, 4)), (32, (4, 2, 4))),
    (128, 4096, "fp8"): ((8, (8, 2, 4, 1)), (16, (8, 2, 4)), (32, (4, 2, 4))),
    (4096, 64, "fp8"): ((8, (1, 2, 1, 1)), (32, (1, 2))),
    (4096, 64, "quant"): ((8, (1, 2, 1, 1)), (32, (1, 2))),
    (8, 4096, "bf16"): ((8, (32, 1)), (32, (8, 1, 8))),
    (1536, 4096, "bf16"): ((8, (8, 2, 1, 1)), (32, (8, 2))),
    (4096, 512, "bf16"): ((16, (4, 2)), (32, (8, 1))),
}


def lowm_rows_ok(m: int, n: int, k: int, mode: str) -> bool:
    """Whether ``m`` rows of an ``n x k`` projection in ``mode`` take the low-M kernel: up to the measured
    crossover of the shape. Shapes without measurements (e.g. another TP size) keep the stock path unless
    opted in with TRTLLM_Q35_LOWM_MAX_M or a TRTLLM_Q35_LOWM_TACTIC override for the shape."""
    if not 0 < m <= lowm_max_m():
        return False
    if "TRTLLM_Q35_LOWM_MAX_M" in os.environ:
        return True
    entries = _TACTICS.get((n, k, mode))
    if entries is None:
        return (n, k) in _tactic_overrides(os.environ.get("TRTLLM_Q35_LOWM_TACTIC", ""))
    return m <= entries[-1][0]


def lowm_conv_supported(m: int, tactic, conv_channels: int, tokens: int) -> bool:
    """Whether ``lowm_gemm(..., conv=...)`` supports ``m`` rows with the FP8-segment ``tactic``: the conv1d
    epilogue needs ksplit 1, whole FP8 CTAs over the conv channels and one (request, channel) pair per thread
    (the checks of ``LowMMmaGemmKernel``)."""
    k_warps, n_tiles = tactic[0], tactic[1]
    ksplit = tactic[2] if len(tactic) > 2 else 1
    cols = 8 * n_tiles  # N_TILE columns per tile
    return (
        ksplit == 1
        and tokens > 0
        and m % tokens == 0
        and conv_channels % cols == 0
        and (m // tokens) * cols <= 32 * k_warps
    )


def default_tactic(
    m: int, n: int, k: int, fp8: bool = True, even_tiles: bool = False, mode: Optional[str] = None
) -> Tactic:
    """(k_warps, n_tiles, ksplit) for an ``m x n x k`` GEMM (n = weight rows of the segment);
    ``even_tiles`` for a SwiGLU segment; ``mode`` selects the measured table entry."""
    override = _tactic_overrides(os.environ.get("TRTLLM_Q35_LOWM_TACTIC", "")).get((n, k))
    if override is not None:
        return override
    entries = _TACTICS.get((n, k, mode or ("quant" if fp8 else "bf16")))
    if entries:
        for max_rows, tactic in entries:
            if m <= max_rows or max_rows == entries[-1][0]:
                k_warps, n_tiles, ksplit, swap, min_blocks = _tactic3(tactic)
                if even_tiles and n_tiles % 2:
                    n_tiles *= 2
                if n % (8 * n_tiles) == 0 and (k // 64) % (k_warps * ksplit) == 0:
                    return k_warps, n_tiles, ksplit, swap, min_blocks
                break
    chunks = k // 64
    # Weight bytes per lane per 64-wide chunk: 16 (FP8) or 32 (BF16); keep the prefetch of a lane
    # at <= ~256 B (64 registers) and give each warp >= 2 chunks.
    lane_budget = 256
    per_chunk = 16 if fp8 else 32
    sms = _num_sms()
    best = None
    for n_tiles in (1, 2, 4):
        if n % (8 * n_tiles) or (even_tiles and n_tiles % 2):
            continue
        ctas = n // (8 * n_tiles)
        for k_warps in (1, 2, 4, 8, 16):
            if chunks % k_warps:
                continue
            per_warp = chunks // k_warps
            if per_warp * n_tiles * per_chunk > lane_budget:
                continue
            # Prefer one wave of CTAs (2 resident per SM), then fewer chunks per warp, then more warps.
            waves = -(-ctas // (2 * sms))
            key = (waves, per_warp > 8, -min(ctas, 2 * sms), per_warp, -k_warps)
            if best is None or key < best[0]:
                best = (key, (k_warps, n_tiles, 1, 0, 1))
    if best is None:
        raise ValueError(f"no low-M tactic for N={n}, K={k}")
    return best[1]


class _GraphSafeDlpack:
    """DLPack export that does not synchronize streams (CUDA-graph capture safe)."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def _cute(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(_GraphSafeDlpack(t), assumed_align=16)


_DUMMIES: dict = {}


def _dummy(kind: str, device) -> torch.Tensor:
    """Placeholder operands with fixed layouts (unused by the compiled variant)."""
    key = (kind, device)
    if key not in _DUMMIES:
        if kind == "a16":
            t = torch.zeros((1, 64), dtype=torch.bfloat16, device=device)
        elif kind == "a8":
            t = torch.zeros((1, 64), dtype=torch.uint8, device=device)
        elif kind == "b8":
            t = torch.zeros((8, 64), dtype=torch.uint8, device=device)
        elif kind == "b16":
            t = torch.zeros((8, 64), dtype=torch.bfloat16, device=device)
        elif kind == "s":
            t = torch.ones((1,), dtype=torch.float32, device=device)
        elif kind == "c":
            t = torch.zeros((1, 8), dtype=torch.bfloat16, device=device)
        elif kind == "cw":
            t = torch.zeros((8, 4), dtype=torch.bfloat16, device=device)
        elif kind == "cs":
            t = torch.zeros((1, 8, 3), dtype=torch.bfloat16, device=device)
        elif kind == "slot":
            t = torch.zeros((1,), dtype=torch.int32, device=device)
        elif kind == "win":
            t = torch.zeros((1, 1, 8, 3), dtype=torch.bfloat16, device=device)
        else:
            raise KeyError(kind)
        _DUMMIES[key] = t
    return _DUMMIES[key]


def _row_view(rows: int, cols: int, stride: int, dtype, device) -> torch.Tensor:
    return torch.empty((rows, stride), dtype=dtype, device=device)[:, :cols]


@functools.lru_cache(maxsize=None)
def _compiled(
    m: int,
    k: int,
    k_warps: int,
    n_fp8: int,
    n_tiles: int,
    fp8_act: str,
    swiglu_fp8: bool,
    n_bf16: int,
    n_tiles_bf16: int,
    swiglu_bf16: bool,
    split_out: int,
    n_valid: int,
    quant_out: bool,
    n_out: int,
    n_out2: int,
    out_dtype: torch.dtype,
    a16_stride: int,
    a8_stride: int,
    use_pdl: bool,
    early_trigger: bool,
    release_after_wait: bool,
    unsafe_skip_wait: bool = False,
    ksplit: int = 1,
    swap_ab: bool = False,
    min_blocks: int = 1,
    conv_channels: int = 0,
    conv_tokens: int = 1,
    conv_write_state: bool = True,
    conv_shapes: tuple = (),
):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    from ..cute_dsl_kernels.blackwell.lowm_mma_gemm import LowMMmaGemmKernel

    kernel = LowMMmaGemmKernel(
        num_rows=m,
        k_extent=k,
        k_warps=k_warps,
        use_pdl=use_pdl,
        n_fp8=n_fp8,
        n_tiles=n_tiles,
        fp8_act=fp8_act,
        swiglu_fp8=swiglu_fp8,
        n_bf16=n_bf16,
        n_tiles_bf16=n_tiles_bf16,
        swiglu_bf16=swiglu_bf16,
        split_out=split_out,
        n_valid=n_valid,
        quant_out=quant_out,
        early_trigger=early_trigger,
        release_after_wait=release_after_wait,
        unsafe_skip_wait=unsafe_skip_wait,
        ksplit=ksplit,
        swap_ab=swap_ab,
        min_blocks=min_blocks,
        conv_channels=conv_channels,
        conv_tokens=conv_tokens,
        conv_write_state=conv_write_state,
    )
    dev = torch.device("cuda")
    if conv_channels:
        (cw_shape, cw_stride), (cs_shape, cs_stride), n_req, (win_shape, win_stride) = conv_shapes
        cw = torch.empty_strided(cw_shape, cw_stride, dtype=torch.bfloat16, device=dev)
        cs = torch.empty_strided(cs_shape, cs_stride, dtype=torch.bfloat16, device=dev)
        slots = torch.zeros((n_req,), dtype=torch.int32, device=dev)
        win = torch.empty_strided(win_shape, win_stride, dtype=torch.bfloat16, device=dev)
    else:
        cw, cs, slots, win = (
            _dummy("cw", dev),
            _dummy("cs", dev),
            _dummy("slot", dev),
            _dummy("win", dev),
        )
    a16 = _row_view(m, k, a16_stride, torch.bfloat16, dev) if a16_stride else _dummy("a16", dev)
    a8 = _row_view(m, k, a8_stride, torch.uint8, dev) if a8_stride else _dummy("a8", dev)
    b = torch.empty((n_fp8, k), dtype=torch.uint8, device=dev) if n_fp8 else _dummy("b8", dev)
    b2 = (
        torch.empty((n_bf16, k), dtype=torch.bfloat16, device=dev) if n_bf16 else _dummy("b16", dev)
    )
    s = _dummy("s", dev)
    c = torch.empty((m, n_out), dtype=torch.uint8 if quant_out else out_dtype, device=dev)
    c2 = torch.empty((m, n_out2), dtype=out_dtype, device=dev) if n_out2 else _dummy("c", dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel,
        _cute(a16),
        _cute(a8),
        _cute(b),
        _cute(s),
        _cute(s),
        _cute(s),
        _cute(b2),
        _cute(c),
        _cute(c2),
        _cute(cw),
        _cute(cs),
        _cute(slots),
        _cute(win),
        stream,
    )


def lowm_gemm(
    *,
    x16: Optional[torch.Tensor] = None,
    x8: Optional[torch.Tensor] = None,
    w_fp8: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    fp8_act: Optional[str] = None,
    swiglu_fp8: bool = False,
    w_bf16: Optional[torch.Tensor] = None,
    swiglu_bf16: bool = False,
    split_out: int = 0,
    n_valid: int = 0,
    out_scale: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    tactic: Optional[Tuple[int, int]] = None,
    tactic_bf16: Optional[Tuple[int, int]] = None,
    release_after_wait: bool = False,
    _unsafe_skip_wait: bool = False,
    conv: Optional[dict] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """``[x] @ [w_fp8; w_bf16].T`` for small M (see the module doc).

    x16: [M, K] BF16 (rows may be strided; 16-byte aligned); x8: [M, K] E4M3 activations quantized
    with ``input_scale``. w_fp8: [N8, K] E4M3 with the per-tensor ``weight_scale``; w_bf16: [N16, K]
    BF16. ``fp8_act``: ``fp8`` (x8), ``quant`` or ``bf16`` (x16); default ``fp8`` when x8 is given,
    else ``quant``. ``swiglu_*``: the segment's rows are (gate, up) 8-row pairs (``swiglu_pairs``).
    ``split_out``: output columns >= ``split_out`` are returned as a second tensor. ``n_valid``: only
    the first ``n_valid`` logical columns are computed into the outputs (zero-padded weight rows).
    ``out_scale``: the first output is E4M3, quantized with this static scale (``x * (1/s)``).
    ``tactic`` / ``tactic_bf16``: (k_warps, n_tiles[, ksplit]) of the FP8 / BF16 segment; one k_warps and
    one ksplit (CTAs per cluster splitting K) for both.
    ``conv`` (Qwen3.5 GDN in_proj, TRTLLM_Q35_GDN_FUSE=4): the first ``conv["weight"].shape[0]`` output
    columns of the FP8 segment go through the GDN causal conv1d update of the MTP target-verify step
    (kernel module doc). Keys: ``weight`` [C, 4] BF16, ``state`` [slots, C, 3] BF16 (conv states, updated to
    the window after the last token), ``slots`` [M / tokens] int32 (conv-state slot of each request),
    ``windows`` [>= M / tokens, tokens, C, 3] BF16 (window after every token), ``tokens``.
    """
    import cuda.bindings.driver as cuda

    x_any = x16 if x16 is not None else x8
    assert x_any is not None and x_any.dim() == 2
    m, k = x_any.shape
    dev = x_any.device
    n_fp8 = 0 if w_fp8 is None else w_fp8.shape[0]
    n_bf16 = 0 if w_bf16 is None else w_bf16.shape[0]
    if fp8_act is None:
        fp8_act = "fp8" if x8 is not None else "quant"
    if n_fp8:
        assert w_fp8.shape[1] == k and w_fp8.is_contiguous() and weight_scale is not None
        if fp8_act == "fp8":
            assert x8 is not None and input_scale is not None
        elif fp8_act == "quant":
            assert x16 is not None and input_scale is not None
        else:
            assert x16 is not None
    if n_bf16:
        assert w_bf16.shape[1] == k and w_bf16.is_contiguous() and x16 is not None
    if x16 is not None:
        assert x16.dtype == torch.bfloat16 and x16.stride(1) == 1 and x16.stride(0) % 8 == 0
    if x8 is not None:
        assert x8.stride(1) == 1 and x8.stride(0) % 16 == 0
    k_warps, n_tiles, ksplit, swap, min_blocks = _tactic3(
        tactic
        or (default_tactic(m, n_fp8, k, True, swiglu_fp8, fp8_act) if n_fp8 else (0, 1, 1, 0, 1))
    )
    kw_bf16, n_tiles_bf16, ks_bf16, swap_bf16, mb_bf16 = _tactic3(
        tactic_bf16
        or (default_tactic(m, n_bf16, k, False, swiglu_bf16, "bf16") if n_bf16 else (0, 1, 1, 0, 1))
    )
    if not n_fp8:
        k_warps, ksplit, swap, min_blocks = kw_bf16, ks_bf16, swap_bf16, mb_bf16
    elif n_bf16 and (kw_bf16, ks_bf16) != (k_warps, ksplit):
        # One block and cluster shape for both segments: the BF16 segment follows the FP8 segment's
        # split-K, one tile per CTA.
        n_tiles_bf16 = 1
    n_logical = (n_fp8 // 2 if swiglu_fp8 else n_fp8) + (n_bf16 // 2 if swiglu_bf16 else n_bf16)
    if n_valid:
        assert split_out < n_valid <= n_logical
        n_logical = n_valid
    n_out = split_out if split_out else n_logical
    n_out2 = n_logical - split_out if split_out else 0
    uses_a16 = n_bf16 > 0 or (n_fp8 > 0 and fp8_act != "fp8")
    uses_a8 = n_fp8 > 0 and fp8_act == "fp8"
    a16 = x16 if uses_a16 else _dummy("a16", dev)
    a8 = x8.view(torch.uint8) if uses_a8 else _dummy("a8", dev)
    conv_args = ()
    if conv is not None:
        cw, cs, slots, win, tokens = (
            conv["weight"],
            conv["state"],
            conv["slots"],
            conv["windows"],
            int(conv["tokens"]),
        )
        assert cw.dim() == 2 and cw.shape[1] == 4 and cw.dtype == torch.bfloat16
        assert cs.dim() == 3 and cs.shape[1:] == (cw.shape[0], 3) and cs.dtype == torch.bfloat16
        assert (
            win.dim() == 4
            and win.shape[1:] == (tokens, cw.shape[0], 3)
            and win.dtype == torch.bfloat16
        )
        assert slots.dtype == torch.int32 and slots.dim() == 1 and m == slots.shape[0] * tokens
        assert slots.is_contiguous()  # no copy kernel between the producer and this PDL launch
        assert win.shape[0] >= slots.shape[0]
        # Layouts are compile-time constants of the kernel: specialize on the shapes and strides.
        conv_args = (
            cw.shape[0],
            tokens,
            bool(conv.get("write_state", True)),
            (
                (tuple(cw.shape), tuple(cw.stride())),
                (tuple(cs.shape), tuple(cs.stride())),
                slots.shape[0],
                (tuple(win.shape), tuple(win.stride())),
            ),
        )
        conv_tensors = (cw, cs, slots, win)
    else:
        conv_tensors = (
            _dummy("cw", dev),
            _dummy("cs", dev),
            _dummy("slot", dev),
            _dummy("win", dev),
        )
    fn = _compiled(
        m,
        k,
        k_warps,
        n_fp8,
        n_tiles,
        fp8_act,
        swiglu_fp8,
        n_bf16,
        n_tiles_bf16,
        swiglu_bf16,
        split_out,
        n_valid,
        out_scale is not None,
        n_out,
        n_out2,
        out_dtype,
        a16.stride(0) if uses_a16 else 0,
        a8.stride(0) if uses_a8 else 0,
        _use_pdl(),
        _early_trigger(),
        release_after_wait,
        _unsafe_skip_wait,
        ksplit,
        bool(swap) and m <= 8,
        min_blocks,
        *conv_args,
    )
    out = torch.empty(
        (m, n_out), dtype=torch.uint8 if out_scale is not None else out_dtype, device=dev
    )
    out2 = torch.empty((m, n_out2), dtype=out_dtype, device=dev) if n_out2 else _dummy("c", dev)
    ones = _dummy("s", dev)
    w_scale = weight_scale.reshape(1) if n_fp8 else ones
    in_scale = input_scale.reshape(1) if (n_fp8 and fp8_act != "bf16") else ones
    o_scale = out_scale.reshape(1) if out_scale is not None else ones
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        _cute(a16),
        _cute(a8),
        _cute(w_fp8.view(torch.uint8) if n_fp8 else _dummy("b8", dev)),
        _cute(w_scale),
        _cute(in_scale),
        _cute(o_scale),
        _cute(w_bf16 if n_bf16 else _dummy("b16", dev)),
        _cute(out),
        _cute(out2),
        *(_cute(t) for t in conv_tensors),
        stream,
    )
    if out_scale is not None:
        out = out.view(torch.float8_e4m3fn)
    return (out, out2) if n_out2 else out


def lowm_bf16_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    tactic: Optional[Tuple[int, ...]] = None,
) -> Optional[torch.Tensor]:
    """``x @ weight.T`` for a BF16 ``weight [N, K]`` on the low-M kernel (for example the MoE router
    [512, 4096]), output BF16 or FP32; None when it does not apply (device, dtype, shape, or M above
    the measured crossover / TRTLLM_Q35_LOWM_MAX_M). Usable inside CUDA graphs (compile happens on the
    first eager call of a shape)."""
    if not _device_supported() or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return None
    x2 = _flat_rows(x)
    m, k = x2.shape
    n = weight.shape[0]
    if (
        weight.shape[1] != k
        or k % 64
        or n % 8
        or x2.stride(1) != 1
        or x2.stride(0) % 8
        or not weight.is_contiguous()
        or out_dtype not in (torch.bfloat16, torch.float32)
    ):
        return None
    if tactic is None and not lowm_rows_ok(m, n, k, "bf16"):
        return None
    out = lowm_gemm(x16=x2, w_bf16=weight, out_dtype=out_dtype, tactic_bf16=tactic)
    if x.dim() != 2:
        out = out.reshape(*x.shape[:-1], n)
    return out


def swiglu_pairs(gate_up: torch.Tensor) -> torch.Tensor:
    """Rows ``[gate (I); up (I)]`` -> ``[gate 0..7; up 0..7; gate 8..15; up 8..15; ...]`` (the
    kernel's SwiGLU pairing; I must be a multiple of 8)."""
    rows, k = gate_up.shape
    inter = rows // 2
    assert rows == 2 * inter and inter % 8 == 0
    gate = gate_up[:inter].reshape(inter // 8, 8, k)
    up = gate_up[inter:].reshape(inter // 8, 8, k)
    return torch.stack((gate, up), dim=1).reshape(rows, k).contiguous()


class _Check:
    """Per-projection error statistics of the low-M path against the stock path (TRTLLM_Q35_LOWM_CHECK)."""

    def __init__(self):
        self.stats = {}
        self.calls = 0
        import atexit

        atexit.register(self.report)

    def add(self, kind: str, out: torch.Tensor, ref: torch.Tensor) -> None:
        if torch.cuda.is_current_stream_capturing():
            return
        o = out.float() if out.dtype != torch.float8_e4m3fn else out.float()
        r = ref.float()
        diff = (o - r).abs()
        scale = r.abs().amax().clamp_min(1e-30)
        rel_max = (diff.amax() / scale).item()
        rel_fro = (diff.norm() / r.norm().clamp_min(1e-30)).item()
        equal = (o == r).float().mean().item()
        st = self.stats.setdefault(kind, [0, 0.0, 0.0, 0.0, 0.0])
        st[0] += 1
        st[1] = max(st[1], rel_max)
        st[2] += rel_fro
        st[3] = max(st[3], rel_fro)
        st[4] += equal
        self.calls += 1
        if self.calls % 200 == 0:
            self.report()

    def report(self) -> None:
        if not self.stats:
            return
        rank = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "?"))
        for kind, (n, rel_max, fro_sum, fro_max, eq_sum) in sorted(self.stats.items()):
            print(
                f"[LOWM_CHECK rank {rank}] {kind:24s} calls={n:6d} max|d|/max|ref|={rel_max:.3e} "
                f"rel_fro mean={fro_sum / n:.3e} max={fro_max:.3e} bitwise-equal={eq_sum / n:.4f}",
                flush=True,
            )


_CHECK: Optional[_Check] = None


def lowm_check(kind: str, out: torch.Tensor, ref_fn) -> None:
    """Compare ``out`` with ``ref_fn()`` (the stock path) when TRTLLM_Q35_LOWM_CHECK=1 (eager calls)."""
    global _CHECK
    if not _env_on("TRTLLM_Q35_LOWM_CHECK") or torch.cuda.is_current_stream_capturing():
        return
    if _CHECK is None:
        _CHECK = _Check()
    _CHECK.add(kind, out, ref_fn())


def _flat_rows(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() == 2 else x.reshape(-1, x.shape[-1])


def stock_fp8_linear(module: torch.nn.Module, x2: torch.Tensor) -> torch.Tensor:
    """The stock W8A8 path (static activation quantize + cuBLAS FP8 GEMM), for checks."""
    s_in = module.input_scale
    q = (
        x2
        if x2.dtype == torch.float8_e4m3fn
        else torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x2.contiguous(), s_in)[0]
    )
    return torch.ops.trtllm.cublas_scaled_mm(
        q,
        module.weight.t(),
        scale_a=s_in,
        scale_b=module.weight_scale,
        bias=None,
        out_dtype=module.dtype or torch.bfloat16,
    )


def maybe_lowm_fp8_linear(
    module: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Small-M path of an opted-in FP8 per-tensor (W8A8) ``Linear``; None when it does not apply."""
    if not getattr(module, "_q35_lowm", False) or bias is not None or not lowm_fp8_enabled():
        return None
    if x.dtype not in (torch.bfloat16, torch.float8_e4m3fn) or x.dim() < 2:
        return None
    x2 = _flat_rows(x)
    m, k = x2.shape
    if lowm_skipped(module._q35_lowm, m):
        return None
    weight = module.weight
    if weight.dtype != torch.float8_e4m3fn or k % 64 or weight.shape[0] % 8 or x2.stride(1) != 1:
        return None
    mode = "fp8" if x2.dtype == torch.float8_e4m3fn else lowm_fp8_act()
    if not lowm_rows_ok(m, weight.shape[0], k, mode):
        return None
    input_scale = getattr(module, "input_scale", None)
    if getattr(module, "force_dynamic_quantization", False):
        input_scale = None
    if x2.dtype == torch.float8_e4m3fn:
        if input_scale is None:
            return None
        act = "fp8"
        out = lowm_gemm(
            x8=x2,
            w_fp8=weight,
            weight_scale=module.weight_scale,
            input_scale=input_scale,
            fp8_act=act,
            out_dtype=module.dtype or torch.bfloat16,
        )
    else:
        act = lowm_fp8_act()
        if act == "quant" and input_scale is None:
            return None
        out = lowm_gemm(
            x16=x2,
            w_fp8=weight,
            weight_scale=module.weight_scale,
            input_scale=input_scale,
            fp8_act=act,
            out_dtype=module.dtype or torch.bfloat16,
        )
    lowm_check(f"fp8:{module._q35_lowm}:{act}", out, lambda: stock_fp8_linear(module, x2))
    if x.dim() != 2:
        out = out.reshape(*x.shape[:-1], out.shape[-1])
    return out


def maybe_lowm_bf16_linear(
    module: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Small-M path of an opted-in BF16 ``Linear``; None when it does not apply."""
    label = getattr(module, "_q35_lowm_bf16", False)
    if not label or bias is not None or not _bf16_label_enabled(label):
        return None
    if x.dtype != torch.bfloat16 or x.dim() < 2:
        return None
    x2 = _flat_rows(x)
    m, k = x2.shape
    if lowm_skipped(module._q35_lowm_bf16, m):
        return None
    weight = module.weight
    if weight.dtype != torch.bfloat16 or k % 64 or weight.shape[0] % 8:
        return None
    if x2.stride(1) != 1 or x2.stride(0) % 8 or not lowm_rows_ok(m, weight.shape[0], k, "bf16"):
        return None
    out = lowm_gemm(x16=x2, w_bf16=weight, out_dtype=module.dtype or torch.bfloat16)
    lowm_check(f"bf16:{module._q35_lowm_bf16}", out, lambda: torch.nn.functional.linear(x2, weight))
    if x.dim() != 2:
        out = out.reshape(*x.shape[:-1], out.shape[-1])
    return out
