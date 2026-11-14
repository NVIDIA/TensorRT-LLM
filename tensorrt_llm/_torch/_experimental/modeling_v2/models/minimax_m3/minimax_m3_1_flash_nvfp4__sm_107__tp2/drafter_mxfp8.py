# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXDRAFT: the DSpark drafter's decode projections on this target's MXFP8 GEMV.

The drafter checkpoint ships its linears as MXFP8 (E4M3 codes, one UE8M0 scale per 32 columns) and the stock drafter
dequantizes them to BF16 at load, so every decode pass streams twice the bytes it needs. MXDRAFT recovers the MXFP8
form of each BF16 weight (exactly: the recovery is checked to dequantize back to the same BF16 bits, else the
projection is left alone) and routes the projection's GEMM to ``m3_mx_gemv_bf16`` for decode inputs it serves; any
other call runs the Linear unchanged. The activation is quantized to MXFP8 inside the GEMV, so the output is not the
stock BF16 GEMM's: drafter-only precision, gated on natural AL. Row-parallel projections keep
their all-reduce (only the Linear's GEMM is replaced).

The hook sits on ``apply_linear``, which a row-parallel Linear that all-reduces into an NCCL window buffer never calls
(it calls ``quant_method.apply`` directly), so MXDRAFT records per projection the rows it serves (``_m31_mxdraft``)
and the rows its hook actually executes (``_m31_mxdraft_exec``: none for such a projection). The drafter's T6 routing
(``dspark_t6``) takes the (projection, rows) MXDRAFT does not execute.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch

from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm.logger import logger

from ..m31_drafter_layout import route_his_layout as _route_his_layout
from .kernels import m3_report
from .kernels.mx_gemv import op as _gemv
from .m31_switches import m31_disabled
from .ops.comm.m3_mc_allreduce_norm import swizzle_sf
from .ops.gemm.m3_mxfp8_pinned_quantize import m3_mxfp8_pinned_quantize

E4M3_MAX = 448.0
BLOCK = 32

# W8 HIS_WIDE (TP2-wide): the K7 c2..c4 drafter passes (14 / 21 / 28 rows). The bf16-input GEMV declines a call whose
# in-kernel quantize exceeds 8 rounds per CTA (mx_gemv._declines: k-tiles x 4 x MMA width > 1024), so on the drafter
# shapes it serves 7 rows on all four projections and 14 only on qkv / o: gate_up / down at 14 and every projection at
# 21 / 28 fell back to the stock BF16 GEMM on the dequantized weights (2x the bytes). With HIS_WIDE those rows run the
# decode-graph MXFP8 quantize (m3_mxfp8_pinned_quantize at the decode tactic, bitwise mxfp8_quantize(x, True) = the
# bf16 GEMV's in-kernel quantize) + the MXFP8-input GEMV on the same config: the drafter precision of the 7-row
# DRAFTER_HISLAYOUT path (not bitwise to the stock BF16 GEMM those rows ran; drafter-only precision).
# Kill: TRTLLM_M31_DISABLE_DRAFTER_HISLAYOUT_WIDE=1 (TRTLLM_M31_DISABLE_DRAFTER_HISLAYOUT=1 turns all of HISLAYOUT
# off). Read once, at import.
W8_HIS_WIDE = not m31_disabled("DRAFTER_HISLAYOUT_WIDE")
HIS_WIDE_MAX_ROWS = 28  # capacity: fixed: the K7 c4 drafter rows (4 x 7)
HIS_QUANT_TACTIC = (
    0  # capacity: fixed: a tactic id (modeling.MXFP8_DECODE_QUANT_TACTIC), not a size
)
_HIS_WIDE_LOGGED: list = []


def _his_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16 [M, K] -> (E4M3 [M, K], swizzled UE8M0 scales): the decode graph's pinned MXFP8 quantize."""
    return m3_mxfp8_pinned_quantize(x, HIS_QUANT_TACTIC)


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


def apply_linear_reached(lin) -> bool:
    """Whether a forward of ``lin`` reaches its ``apply_linear`` (where MXDRAFT's hook sits). Mirrors Linear.forward: a
    row-parallel Linear that reduces its output runs the fused GEMM + all-reduce (``use_fused_gemm_allreduce``) or, when
    its quant method can write the NCCL window buffer its all-reduce uses, calls ``quant_method.apply`` directly; both
    bypass ``apply_linear``. Decided by the module's configuration and the import-time environment only, so it is fixed
    per process and the same on every rank."""
    if getattr(lin, "tp_mode", None) != TensorParallelMode.ROW or not getattr(
        lin, "reduce_output", False
    ):
        return True
    if getattr(getattr(lin, "partition_plan", None), "enabled", False):
        return True  # locality-domain shards: the shard-aware GEMM goes through apply_linear
    if getattr(lin, "use_fused_gemm_allreduce", False):
        return False
    ar = getattr(lin, "all_reduce", None)
    window = getattr(ar, "uses_nccl_symmetric_memory_window", None)
    return not (
        ar is not None
        and getattr(
            type(getattr(lin, "quant_method", None)),
            "supports_nccl_symmetric_memory_window_output",
            False,
        )
        and callable(window)
        and bool(window())
    )


def executed_rows(lin) -> Tuple[int, ...]:
    """The decode rows MXDRAFT's hook executes for ``lin``: its served rows when a forward reaches the hook, else none
    (empty for a projection MXDRAFT did not route)."""
    return tuple(getattr(lin, "_m31_mxdraft_exec", ()) or ())


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
                    m3_report.work(
                        "MXDRAFT", f"{input.shape[0]} rows [{_w8.shape[0]}, {_w8.shape[1]}]"
                    )
                    return out
            return _o(input, bias, lora_params, layer_idx)

        lin._m31_mxdraft = (w8, sf, served)
        lin._m31_mxdraft_exec = served if apply_linear_reached(lin) else ()
        lin.apply_linear = apply_linear
        n_routed += 1
    return n_routed


def route_projection_layout(draft_model, rows: Iterable[int]) -> Tuple[int, int]:
    """Route TP2 decode projections through family ROW wrappers, keeping standalone activation.

    The ring-2 candidate was slower; retain the current target tactics. The drafter shapes' TP2 tactics do not depend
    on the row count (no per-M or wide entry for them), so every row the GEMV serves (up to MAX_TOKENS, the rows
    MXDRAFT served: B x K for B <= MAX_FAST_BATCH) takes the same config as MXDRAFT's apply_linear hook, bit for bit;
    a projection taken here is skipped by route_mx_linears, so bounding at 8 rows would drop B >= 2 off the GEMV.
    """
    configs = {}
    for layer in getattr(getattr(draft_model, "model", None), "layers", []) or []:
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        for name, lin in (
            ("qkv", getattr(attn, "qkv_proj", None)),
            ("o", getattr(attn, "o_proj", None)),
            ("gate_up", getattr(mlp, "gate_up_proj", None)),
            ("down", getattr(mlp, "down_proj", None)),
        ):
            weight = getattr(lin, "weight", None)
            if weight is None or len(weight.shape) != 2:
                continue
            shape = tuple(weight.shape)
            cfg = _gemv.get_config(*shape)
            if cfg is None:
                continue
            configs[shape] = cfg
    routed = _route_his_layout(
        draft_model,
        rows,
        gemv=_gemv,
        recover=mxfp8_from_bf16,
        swizzle=swizzle_sf,
        configs=configs,
        down_configs={},
        fuse_mlp=False,
        max_rows=_gemv.MAX_TOKENS,
        quantize=_his_quantize if W8_HIS_WIDE else None,
        quantize_max_rows=HIS_WIDE_MAX_ROWS if W8_HIS_WIDE else 0,
    )
    wide = sorted(
        {
            m
            for layer in getattr(getattr(draft_model, "model", None), "layers", []) or []
            for lin in (
                getattr(getattr(layer, "self_attn", None), "qkv_proj", None),
                getattr(getattr(layer, "self_attn", None), "o_proj", None),
                getattr(getattr(layer, "mlp", None), "gate_up_proj", None),
                getattr(getattr(layer, "mlp", None), "down_proj", None),
            )
            for m in getattr(lin, "_m31_hislayout_mxq", ())
        }
    )
    if not _HIS_WIDE_LOGGED:
        _HIS_WIDE_LOGGED.append(True)
        logger.info(
            f"[TP2WIDE] W8 HIS rows={'|'.join(map(str, wide)) or 'none'} "
            f"{'on' if W8_HIS_WIDE and wide else 'off'} (quantize + MXFP8-input GEMV, rows <= {HIS_WIDE_MAX_ROWS}; "
            "kill TRTLLM_M31_DISABLE_DRAFTER_HISLAYOUT_WIDE)"
        )
    return routed
