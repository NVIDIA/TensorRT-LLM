# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launcher of the low-M warp-MMA GEMM (``kernels/blackwell/low_m_mma_gemm``) and its tactic heuristic.

Plain projections reach it through the low-M GEMM race of ``modules/low_m_gemm.py``; fused users (GDN input
projection + conv, shared-expert SwiGLU (``fused_shared_expert``), GEMM + all-reduce) launch it with the
tactic that race picked.
"""

from __future__ import annotations

import functools
from typing import List, Optional, Tuple, Union

import torch

from ..._utils import is_blackwell_datacenter
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE, graph_safe_cute_tensor
from ..flashinfer_utils import get_env_enable_pdl

_NUM_SMS_DEFAULT = 148

# Most rows of one launch.
MAX_M = 32


@functools.lru_cache(maxsize=None)
def warp_mma_supported() -> bool:
    """SM 10.0 / 10.3 with the CuTe DSL."""
    return IS_CUTLASS_DSL_AVAILABLE and torch.cuda.is_available() and is_blackwell_datacenter()


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


def conv_supported(m: int, tactic, conv_channels: int, tokens: int) -> bool:
    """Whether ``low_m_mma_gemm(..., conv=...)`` supports ``m`` rows with the FP8-segment ``tactic``; mirrors the
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


def default_tactic(m: int, n: int, k: int, fp8: bool = True, even_tiles: bool = False) -> Tactic:
    """Heuristic tactic for an ``m x n x k`` segment (n = weight rows). ``even_tiles``: SwiGLU segment."""
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


def candidate_tactics(m: int, n: int, k: int, fp8: bool) -> List[Tactic]:
    """The tactics the low-M GEMM race times for a plain projection, heuristic first: half, the same and twice
    the heuristic's warps over K, times 1, 2 or 4 tiles per CTA and K split over a cluster of 1, 2, 4 or 8 CTAs
    (while the grid fits one CTA per SM), each also with A and B swapped (m <= 8, even tiles); a lane
    prefetches at most 512 B of weights."""
    try:
        base = default_tactic(m, n, k, fp8)
    except ValueError:
        return []
    chunks = k // 64
    per_chunk = 16 if fp8 else 32
    out = {base: None}
    for ksplit in (1, 2, 4, 8):
        for n_tiles in (1, 2, 4):
            ctas = n // (8 * n_tiles)
            if n % (8 * n_tiles) or (ksplit > 1 and ctas * ksplit > _num_sms()):
                continue
            for k_warps in (base[0] // 2, base[0], 2 * base[0]):
                split = k_warps * ksplit
                if (
                    not 1 <= k_warps <= 32
                    or chunks % split
                    or chunks // split * n_tiles * per_chunk > 512
                ):
                    continue
                out[(k_warps, n_tiles, ksplit, 0, 1)] = None
                if m <= 8 and n_tiles % 2 == 0:
                    out[(k_warps, n_tiles, ksplit, 1, 1)] = None
    return list(out)


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

    from ..kernels.blackwell.low_m_mma_gemm import LowMMmaGemmKernel

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
        graph_safe_cute_tensor(a16),
        graph_safe_cute_tensor(a8),
        graph_safe_cute_tensor(b),
        graph_safe_cute_tensor(s),
        graph_safe_cute_tensor(s),
        graph_safe_cute_tensor(s),
        graph_safe_cute_tensor(b2),
        graph_safe_cute_tensor(c),
        graph_safe_cute_tensor(c2),
        graph_safe_cute_tensor(cw),
        graph_safe_cute_tensor(cs),
        graph_safe_cute_tensor(slots),
        graph_safe_cute_tensor(win),
        stream,
    )


def low_m_mma_gemm(
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
        tactic or (default_tactic(m, n_fp8, k, True, swiglu_fp8) if n_fp8 else (0, 1, 1, 0, 1))
    )
    kw_bf16, n_tiles_bf16, ks_bf16, swap_bf16, mb_bf16 = _tactic3(
        tactic_bf16
        or (default_tactic(m, n_bf16, k, False, swiglu_bf16) if n_bf16 else (0, 1, 1, 0, 1))
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
        graph_safe_cute_tensor(a16),
        graph_safe_cute_tensor(a8),
        graph_safe_cute_tensor(w_fp8.view(torch.uint8) if n_fp8 else _dummy("b8", dev)),
        graph_safe_cute_tensor(w_scale),
        graph_safe_cute_tensor(in_scale),
        graph_safe_cute_tensor(o_scale),
        graph_safe_cute_tensor(w_bf16 if n_bf16 else _dummy("b16", dev)),
        graph_safe_cute_tensor(out),
        graph_safe_cute_tensor(out2),
        *(graph_safe_cute_tensor(t) for t in conv_tensors),
        stream,
    )
    if out_scale is not None:
        out = out.view(torch.float8_e4m3fn)
    return (out, out2) if n_out2 else out


def swiglu_pairs(gate_up: torch.Tensor) -> torch.Tensor:
    """Rows ``[gate (I); up (I)]`` -> ``[gate 0..7; up 0..7; gate 8..15; up 8..15; ...]``; I a multiple of 8."""
    rows, k = gate_up.shape
    inter = rows // 2
    assert rows == 2 * inter and inter % 8 == 0
    gate = gate_up[:inter].reshape(inter // 8, 8, k)
    up = gate_up[inter:].reshape(inter // 8, 8, k)
    return torch.stack((gate, up), dim=1).reshape(rows, k).contiguous()


def fused_shared_expert_weights(
    gate_up_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Weights of :func:`fused_shared_expert` for a SwiGLU MLP with a sigmoid gate, or None when the shapes do
    not fit the kernel.

    Args:
        gate_up_weight: [2I, K] E4M3, rows ``[gate; up]``. down_weight: [H, I] E4M3.
        gate_weight: [1, K] BF16 weight of the gate.
        out: the result of an earlier call, rewritten in place (captured graphs hold its addresses).

    Returns:
        (gate_up in ``swiglu_pairs`` order, gate weight zero-padded to one 8-row tile); the originals stay
        for larger batches.
    """
    rows, k = gate_up_weight.shape
    if (
        rows % 16
        or k % 64
        or down_weight.shape[1] % 64
        or gate_weight.dtype != torch.bfloat16
        or tuple(gate_weight.shape) != (1, k)
    ):
        return None
    pairs = swiglu_pairs(gate_up_weight)
    gate_rows = torch.zeros((8, k), dtype=gate_weight.dtype, device=gate_weight.device)
    gate_rows[:1].copy_(gate_weight)
    if out is None:
        return pairs, gate_rows
    out[0].copy_(pairs)
    out[1].copy_(gate_rows)
    return out


def fused_shared_expert(
    dispatcher,
    x: torch.Tensor,
    weights: Tuple[torch.Tensor, torch.Tensor],
    *,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    gate_up_input_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_input_scale: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(MLP output, gate logits [M, 1]) of a SwiGLU MLP with a sigmoid gate in two launches, or None.

    (1) gate_up with the SwiGLU in the epilogue (E4M3 output at ``down_input_scale``) plus the gate as a BF16
    segment; (2) down_proj on that E4M3 output. Each launch takes the warp-MMA tactic ``dispatcher``'s low-M
    race picked for the plain projection (BF16 input for gate_up, E4M3 for down_proj), so it runs only where
    the plain path would run this kernel; None when the race picked another runner or the gate_up tactic has
    no SwiGLU form.

    Args:
        dispatcher: the ``LowMGemmDispatcher`` of the projections.
        x: [M, K] BF16.
        weights: :func:`fused_shared_expert_weights` of the projections.
    """
    gate_up_tactic = dispatcher.warp_mma_tactic(
        x.shape[0], gate_up_weight, gate_up_weight_scale, gate_up_input_scale
    )
    down_tactic = dispatcher.warp_mma_tactic(
        x.shape[0], down_weight, down_weight_scale, down_input_scale, quantize=False
    )
    if gate_up_tactic is None or down_tactic is None:
        return None
    pairs, gate_rows = weights
    k_warps, n_tiles, ksplit, swap, min_blocks = _tactic3(gate_up_tactic)
    if n_tiles % 2:
        # A SwiGLU segment's CTAs hold whole (gate, up) tile pairs.
        n_tiles *= 2
        if pairs.shape[0] % (8 * n_tiles):
            return None
    inter = pairs.shape[0] // 2
    hidden, gate_logits = low_m_mma_gemm(
        x16=x,
        w_fp8=pairs,
        weight_scale=gate_up_weight_scale,
        input_scale=gate_up_input_scale,
        swiglu_fp8=True,
        w_bf16=gate_rows,
        split_out=inter,
        n_valid=inter + 1,
        out_scale=down_input_scale,
        out_dtype=x.dtype,
        tactic=(k_warps, n_tiles, ksplit, swap, min_blocks),
    )
    out = low_m_mma_gemm(
        x8=hidden,
        w_fp8=down_weight,
        weight_scale=down_weight_scale,
        input_scale=down_input_scale,
        fp8_act="fp8",
        out_dtype=x.dtype,
        tactic=down_tactic,
    )
    return out, gate_logits


__all__ = [
    "MAX_M",
    "candidate_tactics",
    "conv_supported",
    "default_tactic",
    "fused_shared_expert",
    "fused_shared_expert_weights",
    "low_m_mma_gemm",
    "swiglu_pairs",
    "warp_mma_supported",
]
