# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drafter MXFP8 projections (DRAFTER_MXFP8): the DSpark drafter's decode projections on this target's MXFP8 GEMV.

The drafter checkpoint ships its linears as MXFP8 (E4M3 codes, one UE8M0 scale per 32 columns) and the stock drafter
dequantizes them to BF16 at load, so every decode pass streams twice the bytes it needs. This module recovers the MXFP8
form of each BF16 weight (exactly: the recovery is checked to dequantize back to the same BF16 bits, else the
projection is left alone) and routes the projection's GEMM to ``m3_mx_gemv_bf16`` for decode inputs it serves; any
other call runs the Linear unchanged. The activation is quantized to MXFP8 inside the GEMV, so the output is not the
stock BF16 GEMM's: drafter-only precision (it changes draft tokens, not the target's distribution). Row-parallel
projections keep their all-reduce (only the Linear's GEMM is replaced).
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from .kernels.mx_gemv import op as _gemv
from .ops.comm.m3_mc_allreduce_norm import swizzle_sf

E4M3_MAX = 448.0
BLOCK = 32


def mxfp8_from_bf16(weight: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The MXFP8 form (E4M3 codes [N, K], UE8M0 scale bytes [N, K / 32]) of a BF16 weight that is exactly an MXFP8
    dequantization, or None. Each block takes the smallest power-of-two scale that fits its amax in E4M3, which is at
    most the shipped scale, so the codes are exact whenever the weight came from MXFP8."""
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or weight.shape[1] % BLOCK != 0:
        return None
    n, k = weight.shape
    blocks = weight.float().view(n, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(-1)
    exp = torch.ceil(torch.log2(amax.clamp_min(2.0**-126) / E4M3_MAX)).clamp(-127, 127)
    scale = torch.exp2(exp).unsqueeze(-1)
    codes = (blocks / scale).to(torch.float8_e4m3fn)
    back = (codes.float() * scale).view(n, k).to(torch.bfloat16)
    if not torch.equal(back, weight):
        return None
    return codes.view(n, k).contiguous(), (exp + 127).to(torch.uint8).contiguous()


def _projections(draft_model) -> Iterable[torch.nn.Module]:
    for layer in getattr(getattr(draft_model, "model", None), "layers", []) or []:
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        for lin in (
            getattr(attn, "qkv_proj", None),
            getattr(attn, "o_proj", None),
            getattr(mlp, "gate_up_proj", None),
            getattr(mlp, "down_proj", None),
        ):
            if lin is not None:
                yield lin


def route_mx_linears(draft_model, rows: Iterable[int]) -> int:
    """Route each drafter projection whose BF16 weight is an exact MXFP8 dequantization to the MXFP8 GEMV for 2-D
    contiguous bf16 inputs of the given row counts (the decode passes'); compiles those kernels now (outside capture).
    Returns how many projections were routed."""
    rows = sorted({int(m) for m in rows if 0 < int(m) <= _gemv.MAX_TOKENS})
    n_routed = 0
    for lin in _projections(draft_model):
        weight = getattr(lin, "weight", None)
        if (
            weight is None
            or getattr(lin, "bias", None) is not None
            or getattr(lin, "_m31_mxdraft", None) is not None
        ):
            continue
        rec = mxfp8_from_bf16(weight.data)
        if rec is None:
            continue
        n, k = weight.shape
        served = tuple(m for m in rows if _gemv.serves(n, k, m, True))
        if not served:
            continue
        w8, sf = rec[0], swizzle_sf(rec[1])
        for m in served:
            _gemv.precompile(w8, sf, m, True)
        original = lin.apply_linear

        def apply_linear(
            input,
            bias,
            lora_params=None,
            layer_idx=None,
            *,
            _o=original,
            _w8=w8,
            _sf=sf,
            _served=served,
        ):
            if (
                bias is None
                and not lora_params
                and isinstance(input, torch.Tensor)
                and input.dtype == torch.bfloat16
                and input.dim() == 2
                and input.shape[0] in _served
                and input.is_contiguous()
            ):
                out = _gemv.m3_mx_gemv_bf16(input, _w8, _sf)
                if out is not None:
                    return out
            return _o(input, bias, lora_params, layer_idx)

        lin._m31_mxdraft = (w8, sf, served)
        lin.apply_linear = apply_linear
        n_routed += 1
    return n_routed


# ----------------------------------------------------------------------------------------------------------------------
# HISLAYOUT (GV_C): the reference drafter GEMV layout
# (M3LL vllm/drafter_gemv.py, M3LL_DEV_DRAFTER_GEMV=mlp, configs drafter_gemv.py:36, 42-45) on THIS target's existing
# MXFP8 GEMV (kernels/mx_gemv: m3_mx_gemv_bf16 and m3_mx_gemv_swiglu, the same kernel lineage; no new kernel):
#   qkv_proj      2304 x 6144: m3_mx_gemv_bf16 at (split 8, ring 6, contig, 1 acc, A from smem, st.async)
#   o_proj        6144 x 2048: m3_mx_gemv_bf16 at (4, 4, 1, 1, 0, 1)
#   gate_up_proj  6144 x 6144: m3_mx_gemv_bf16 at (8, 2, 1, 1, 0, 1)
#   SwiGLU + down 6144 x 3072: m3_mx_gemv_swiglu at (4, 6, 1, 1, 0, 1): the SwiGLU-OAI of the Triton silu_and_mul
#                 (bit for bit) and its MXFP8 quantization in the GEMV's prologue, then the MXFP8 GEMV; the drafter
#                 MLP's
#                 separate activation launch is gone.
# Precision: MXFP8 weights (the exact form of the shipped BF16 drafter weights, mxfp8_from_bf16) x MXFP8 activations
# (quantized in-kernel exactly as mxfp8_quantize), fp32 sums: drafter-only, gated on natural acceptance length. The
# drafter's context-KV GEMM stays on F.linear (its fused weight folds the input-layernorm scale, so it is not an exact
# MXFP8 dequantization). Only the decode rows <= HIS_MAX_ROWS take it (the reference MAX_M); other rows run the Linear.
# MB (GV_C): -34.67 us/step vs the bf16 nvjet split-K path (incl. ctx-KV, not routed here).
HIS_MAX_ROWS = 8  # capacity: fixed: the reference drafter GEMV row bound (MAX_M)
HIS_CFGS = {
    (2304, 6144): (8, 6, 1, 1, 0, 1),
    (6144, 2048): (4, 4, 1, 1, 0, 1),
    (6144, 6144): (8, 2, 1, 1, 0, 1),
}
HIS_DOWN_CFGS = {(6144, 3072): (4, 6, 1, 1, 0, 1)}
# (tag, N, K, M) of the calls already reported by the dispatch line (host-side; eager or at capture).
_HIS_TAKEN: set = set()


def _his_taken(tag: str, n: int, k: int, m: int) -> None:
    key = (tag, n, k, m)
    if key not in _HIS_TAKEN:
        _HIS_TAKEN.add(key)
        logger.info(f"[M3 dispatch] m31_drafter_hislayout taken at {m} rows ({tag} {n} x {k})")


def _compile_now(call) -> bool:
    """Run one eager call (compiling its kernel) even while mx_gemv compiles are frozen; True if it was served."""
    was = _gemv.frozen()
    _gemv.freeze(False)
    try:
        return call() is not None
    finally:
        _gemv.freeze(was)


def _route_his_linear(lin, served_rows) -> bool:
    weight = getattr(lin, "weight", None)
    if (
        weight is None
        or getattr(lin, "bias", None) is not None
        or getattr(lin, "_m31_mxdraft", None) is not None
    ):
        return False
    n, k = weight.shape
    cfg = HIS_CFGS.get((n, k))
    if cfg is None:
        return False
    rec = mxfp8_from_bf16(weight.data)
    if rec is None:
        return False
    w8, sf = rec[0], swizzle_sf(rec[1])
    served = tuple(
        m
        for m in served_rows
        if not _gemv._declines(n, k, m, True, cfg)
        and _compile_now(
            lambda m=m: _gemv.m3_mx_gemv_bf16(
                torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device), w8, sf, True, cfg
            )
        )
    )
    if not served:
        return False
    original = lin.apply_linear

    def apply_linear(
        input,
        bias,
        lora_params=None,
        layer_idx=None,
        *,
        _o=original,
        _w8=w8,
        _sf=sf,
        _served=served,
        _cfg=cfg,
    ):
        if (
            bias is None
            and not lora_params
            and isinstance(input, torch.Tensor)
            and input.dtype == torch.bfloat16
            and input.dim() == 2
            and input.shape[0] in _served
            and input.is_contiguous()
        ):
            out = _gemv.m3_mx_gemv_bf16(input, _w8, _sf, True, _cfg)
            if out is not None:
                _his_taken("gemv", _w8.shape[0], _w8.shape[1], input.shape[0])
                return out
        return _o(input, bias, lora_params, layer_idx)

    lin._m31_mxdraft = (w8, sf, served)
    lin._m31_hislayout = cfg
    lin.apply_linear = apply_linear
    if _row_parallel_reducing(lin):
        _wrap_row_parallel_forward(lin, w8, sf, served, cfg)
    return True


def _row_parallel_reducing(lin) -> bool:
    mode = getattr(lin, "tp_mode", None)
    return (
        mode is not None
        and getattr(mode, "name", str(mode)).endswith("ROW")
        and bool(getattr(lin, "reduce_output", False))
        and int(getattr(lin, "tp_size", 1)) > 1
    )


def _wrap_row_parallel_forward(lin, w8, sf, served, cfg) -> None:
    """A row-parallel Linear's served forward (Linear.forward, tensorrt_llm/_torch/modules/linear.py:4316) does not
    reach ``apply_linear`` on two of its three ROW branches: the fused GEMM + all-reduce (linear.py:4339,
    apply_linear_allreduce) and the NCCL-window zero-copy GEMM (linear.py:4350-4357: ``quant_method.apply`` straight
    into the window buffer when TLLM_NCCL_SYMMETRIC_ZERO_COPY=1, the default, and the projection's all-reduce reports
    ``uses_nccl_symmetric_memory_window`` -- the drafter's DraftOneshotAllReduce (dspark.py:139-140) delegates that to
    its AUTO all-reduce, so it is True on the served MPI path). A hook on ``apply_linear`` alone is therefore a
    registered hook, not an executed GEMV, for o_proj / down_proj. This wrapper serves the decode rows itself: the
    MXFP8 GEMV, then the projection's own all-reduce with the caller's parameters -- the non-window ROW branch
    (linear.py:4359-4361: apply_linear, then self.all_reduce(output, all_reduce_params=...)) with the GEMV in place
    of the stock GEMM. Every other call (other rows, LoRA, a declined GEMV) runs the stock forward unchanged."""
    stock_forward = lin.forward

    def forward(
        input,
        *,
        all_reduce_params=None,
        lora_params=None,
        layer_idx=None,
        _stock=stock_forward,
        _w8=w8,
        _sf=sf,
        _served=served,
        _cfg=cfg,
    ):
        if (
            not lora_params
            and isinstance(input, torch.Tensor)
            and input.dtype == torch.bfloat16
            and input.dim() == 2
            and input.shape[0] in _served
            and input.is_contiguous()
        ):
            out = _gemv.m3_mx_gemv_bf16(input, _w8, _sf, True, _cfg)
            if out is not None:
                _his_taken("gemv", _w8.shape[0], _w8.shape[1], input.shape[0])
                return lin.all_reduce(out, all_reduce_params=all_reduce_params)
        return _stock(
            input, all_reduce_params=all_reduce_params, lora_params=lora_params, layer_idx=layer_idx
        )

    lin.forward = forward
    lin._m31_hislayout_row_forward = True


def _route_his_mlp(mlp, served_rows) -> bool:
    """The drafter MLP's SwiGLU + down projection as one m3_mx_gemv_swiglu launch for the served rows: the MLP's
    forward runs gate_up_proj, then the fused GEMV on its output, then the down projection's own all-reduce (row
    parallel), exactly the path the GatedMLP takes for this drafter (gate_up -> _apply_activation -> down: apply_linear
    then all_reduce); anything else (other rows, LoRA, a declined call) runs the stock forward."""
    down = getattr(mlp, "down_proj", None)
    gu = getattr(mlp, "gate_up_proj", None)
    if down is None or gu is None or getattr(mlp, "_m31_hislayout_mlp", None) is not None:
        return False
    if getattr(mlp, "split_gate_up", False) or getattr(down, "bias", None) is not None:
        return False
    weight = down.weight
    n, k = weight.shape
    cfg = HIS_DOWN_CFGS.get((n, k))
    if cfg is None or gu.weight.shape[0] != 2 * k:
        return False
    rec = mxfp8_from_bf16(weight.data)
    if rec is None:
        return False
    w8, sf = rec[0], swizzle_sf(rec[1])
    served = tuple(
        m
        for m in served_rows
        if _compile_now(
            lambda m=m: _gemv.m3_mx_gemv_swiglu(
                torch.zeros(m, 2 * k, dtype=torch.bfloat16, device=weight.device), w8, sf, True, cfg
            )
        )
    )
    if not served:
        return False
    stock_forward = mlp.forward

    def forward(
        x,
        all_rank_num_tokens=None,
        final_all_reduce_params=None,
        lora_params=None,
        _stock=stock_forward,
        _w8=w8,
        _sf=sf,
        _served=served,
        _cfg=cfg,
        **kwargs,
    ):
        if (
            not lora_params
            and not kwargs
            and isinstance(x, torch.Tensor)
            and x.dtype == torch.bfloat16
            and x.dim() == 2
            and x.shape[0] in _served
            and x.is_contiguous()
        ):
            h1 = mlp.gate_up_proj(x)
            if h1.dim() == 2 and h1.is_contiguous():
                out = _gemv.m3_mx_gemv_swiglu(h1, _w8, _sf, True, _cfg)
                if out is not None:
                    _his_taken("swiglu+down", _w8.shape[0], _w8.shape[1], x.shape[0])
                    if getattr(down, "tp_size", 1) > 1 and getattr(down, "reduce_output", False):
                        out = down.all_reduce(out, all_reduce_params=final_all_reduce_params)
                    return out
                return down(
                    mlp._apply_activation(h1),
                    all_reduce_params=final_all_reduce_params,
                    layer_idx=getattr(mlp, "layer_idx", None),
                )
        return _stock(x, all_rank_num_tokens, final_all_reduce_params, lora_params, **kwargs)

    mlp.forward = forward
    mlp._m31_hislayout_mlp = (w8, sf, served, cfg)
    return True


def route_his_layout(draft_model, rows: Iterable[int]) -> Tuple[int, int]:
    """HISLAYOUT on every drafter layer for the decode row counts in ``rows`` (<= HIS_MAX_ROWS): returns
    (projections routed among qkv / o_proj / gate_up, MLPs whose SwiGLU + down run fused). Compiles now (outside
    capture)."""
    served_rows = sorted({int(m) for m in rows if 0 < int(m) <= HIS_MAX_ROWS})
    n_lin = n_mlp = 0
    if not served_rows:
        return 0, 0
    for layer in getattr(getattr(draft_model, "model", None), "layers", []) or []:
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        for lin in (
            getattr(attn, "qkv_proj", None),
            getattr(attn, "o_proj", None),
            getattr(mlp, "gate_up_proj", None),
        ):
            if lin is not None and _route_his_linear(lin, served_rows):
                n_lin += 1
        if mlp is not None and _route_his_mlp(mlp, served_rows):
            n_mlp += 1
    return n_lin, n_mlp
