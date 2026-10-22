# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-sized (M <= 32) Qwen3.5 projections on the low-M warp-MMA kernel (``kernels/blackwell/lowm_mma_gemm``).

Replaces activation-quantize + cuBLAS for per-tensor FP8 (W8A8) and BF16 linears where they are latency bound;
FP8 numerics match the stock path. Opt-in per module (``_q35_lowm`` / ``_q35_lowm_bf16``) and gated on
``lowm_enabled()``. Only shapes listed in ``_TACTICS`` take it, up to their crossover row count; anything else
(other TP sizes, larger M) keeps the stock path.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple, Union

import torch

from .. import q35_env
from ..flashinfer_utils import get_env_enable_pdl

_NUM_SMS_DEFAULT = 148

# BF16 input to an FP8-weight projection is quantized in registers with the static input scale.
LOWM_FP8_ACT = "quant"
LOWM_MAX_M = 32


def lowm_enabled() -> bool:
    """Whether the low-M projections are on (the Qwen3.5 SM 10.3 fast paths)."""
    return q35_env.fast_paths_enabled()


@functools.lru_cache(maxsize=None)
def _device_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


@functools.lru_cache(maxsize=None)
def _num_sms() -> int:
    if not torch.cuda.is_available():
        return _NUM_SMS_DEFAULT
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


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


# GB300-tuned, per-rank shapes for TP8 then TP16. Key (N, K, mode): fp8 / quant (FP8 weights, E4M3 / BF16 acts),
# bf16, mixed / mixed8 (GDN in_proj_qkvz FP8 + in_proj_ba BF16 in one launch, BF16 / E4M3 acts). Value: ((max rows,
# tactic), ...) in increasing row order; the last max rows is the crossover beyond which the stock path is as fast.
_TACTICS = {
    (2560, 4096, "fp8"): ((8, (16, 4, 1, 1)), (16, (8, 1, 1, 0, 3))),
    (2560, 4096, "quant"): ((8, (16, 4, 1, 1)), (16, (16, 4))),
    (2560, 4096, "mixed"): ((8, (16, 4, 1, 1)), (16, (16, 4))),
    (2560, 4096, "mixed8"): ((8, (16, 2, 1, 1, 2)), (16, (8, 4))),
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
    # TP16 per-rank shapes.
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
    """Whether ``m`` rows of an ``n x k`` projection in ``mode`` take the low-M kernel (listed shape, ``m`` <=
    its crossover)."""
    if not 0 < m <= LOWM_MAX_M:
        return False
    entries = _TACTICS.get((n, k, mode))
    return entries is not None and m <= entries[-1][0]


def lowm_conv_supported(m: int, tactic, conv_channels: int, tokens: int) -> bool:
    """Whether ``lowm_gemm(..., conv=...)`` supports ``m`` rows with the FP8-segment ``tactic``; mirrors the
    conv-epilogue checks of ``LowMMmaGemmKernel``."""
    k_warps, n_tiles = tactic[0], tactic[1]
    ksplit = tactic[2] if len(tactic) > 2 else 1
    cols = 8 * n_tiles
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
    """Tactic for an ``m x n x k`` segment (n = weight rows): the ``_TACTICS`` entry for ``mode``, else a
    heuristic. ``even_tiles``: SwiGLU segment."""
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

    from ..kernels.blackwell.lowm_mma_gemm import LowMMmaGemmKernel

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
    conv: Optional[dict] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """``[x] @ [w_fp8; w_bf16].T`` for small M.

    Args:
        x16: [M, K] BF16; rows may be strided, row stride a multiple of 8 elements.
        x8: [M, K] E4M3, quantized with ``input_scale``.
        w_fp8: [N8, K] E4M3 with the per-tensor FP32 ``weight_scale``. w_bf16: [N16, K] BF16.
        fp8_act: ``fp8`` (x8), ``quant`` or ``bf16`` (x16); default ``fp8`` when x8 is given, else ``quant``.
        swiglu_fp8, swiglu_bf16: the segment's rows are in ``swiglu_pairs`` order.
        split_out: output columns ``>= split_out`` are returned as a second tensor.
        n_valid: only the first ``n_valid`` output columns are stored (zero-padded weight rows).
        out_scale: the first output is E4M3, quantized with this static scale (``x * (1/s)``).
        tactic, tactic_bf16: (k_warps, n_tiles[, ksplit[, swap_ab[, min_blocks]]]) per segment; k_warps and
            ksplit must agree (else the BF16 segment follows the FP8 one with one tile per CTA).
        conv: GDN causal conv1d update on the first ``C`` FP8 output columns (MTP target verify). Keys:
            ``weight`` [C, 4] BF16, ``state`` [slots, C, 3] BF16 (updated in place), ``slots`` [M / tokens]
            int32 contiguous, ``windows`` [>= M / tokens, tokens, C, 3] BF16 (window after every token),
            ``tokens``, optional ``write_state``.
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
        # One block and cluster shape for both segments.
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
        # Layouts are compile-time constants: specialize on the shapes and strides.
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
        get_env_enable_pdl(),
        True,  # early_trigger
        release_after_wait,
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
    """``x @ weight.T`` for BF16 ``x [..., K]`` and ``weight [N, K]``, output BF16 or FP32; None when it does not
    apply. CUDA-graph safe once the shape has been compiled by an eager call."""
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
    """Rows ``[gate (I); up (I)]`` -> ``[gate 0..7; up 0..7; gate 8..15; up 8..15; ...]``; I a multiple of 8."""
    rows, k = gate_up.shape
    inter = rows // 2
    assert rows == 2 * inter and inter % 8 == 0
    gate = gate_up[:inter].reshape(inter // 8, 8, k)
    up = gate_up[inter:].reshape(inter // 8, 8, k)
    return torch.stack((gate, up), dim=1).reshape(rows, k).contiguous()


def _flat_rows(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() == 2 else x.reshape(-1, x.shape[-1])


def maybe_lowm_fp8_linear(
    module: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Small-M path of an FP8 per-tensor (W8A8) ``Linear``; None when it does not apply.

    The caller checks the opt-in (``module._q35_lowm``) and the gate (``lowm_enabled()``).
    """
    if bias is not None or x.dtype not in (torch.bfloat16, torch.float8_e4m3fn) or x.dim() < 2:
        return None
    x2 = _flat_rows(x)
    m, k = x2.shape
    weight = module.weight
    if weight.dtype != torch.float8_e4m3fn or k % 64 or weight.shape[0] % 8 or x2.stride(1) != 1:
        return None
    act = "fp8" if x2.dtype == torch.float8_e4m3fn else LOWM_FP8_ACT
    if not lowm_rows_ok(m, weight.shape[0], k, act):
        return None
    input_scale = getattr(module, "input_scale", None)
    if getattr(module, "force_dynamic_quantization", False) or input_scale is None:
        return None
    out = lowm_gemm(
        x8=x2 if act == "fp8" else None,
        x16=None if act == "fp8" else x2,
        w_fp8=weight,
        weight_scale=module.weight_scale,
        input_scale=input_scale,
        fp8_act=act,
        out_dtype=module.dtype or torch.bfloat16,
    )
    if x.dim() != 2:
        out = out.reshape(*x.shape[:-1], out.shape[-1])
    return out


def maybe_lowm_bf16_linear(
    module: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Small-M path of a BF16 ``Linear``; None when it does not apply.

    The caller checks the opt-in (``module._q35_lowm_bf16``) and the gate (``lowm_enabled()``).
    """
    if bias is not None or x.dtype != torch.bfloat16 or x.dim() < 2:
        return None
    x2 = _flat_rows(x)
    m, k = x2.shape
    weight = module.weight
    if weight.dtype != torch.bfloat16 or k % 64 or weight.shape[0] % 8:
        return None
    if x2.stride(1) != 1 or x2.stride(0) % 8 or not lowm_rows_ok(m, weight.shape[0], k, "bf16"):
        return None
    out = lowm_gemm(x16=x2, w_bf16=weight, out_dtype=module.dtype or torch.bfloat16)
    if x.dim() != 2:
        out = out.reshape(*x.shape[:-1], out.shape[-1])
    return out


__all__ = [
    "LOWM_FP8_ACT",
    "LOWM_MAX_M",
    "default_tactic",
    "lowm_bf16_gemm",
    "lowm_conv_supported",
    "lowm_enabled",
    "lowm_gemm",
    "lowm_rows_ok",
    "maybe_lowm_bf16_linear",
    "maybe_lowm_fp8_linear",
    "swiglu_pairs",
]
