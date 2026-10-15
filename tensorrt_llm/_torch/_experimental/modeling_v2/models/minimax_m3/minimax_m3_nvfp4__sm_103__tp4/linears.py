# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode-graph MXFP8 GEMM dispatch of this target's projections.

Every MXFP8 projection of the text model (QKV, o_proj, the dense MLPs and the
shared experts) runs the built-in ``MXFP8LinearMethod`` outside decode CUDA
graphs. Inside them (``flashinfer_mxfp8_decode_graph_capture_active``) the
projection's (N, K) and the step's rows pick one path (``dispatch.linear_path``,
row ranges per projection in ``dispatch.py``):

1. an activation its producer already quantized to MXFP8 (the boundary
   all-reduce or the attention merge publish it in ``m3_prequant.SLOT``): the
   tcgen05 decode GEMV on the rows it serves, else the decode-graph MXFP8 GEMM
   (AutoTuner-selected backend per token bucket, the built-in method's GEMM);
2. a bf16 activation on the rows the GEMV serves quantizing in its prologue;
3. projections marked for it, on its rows: the fused low-M MXFP8 GEMM;
4. otherwise the built-in method (MXFP8 quantize + GEMM, AutoTuner-selected per bucket).

No path runs FlashInfer's own per-rank config sweep: its tactics are not merged
across ranks and it profiled wherever the projection ran (the MoE aux stream for
the shared expert), one CUDA graph per tactic.

A kernel declining rows its path covers is a miss (``dispatch.missed``); the
chain then continues down the list. The GEMVs push their rows straight into
the boundary all-reduce the decode forward armed for the projection's push
slot (``m3_push``, fixed at construction).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.custom_ops.torch_custom_ops import mxfp8_gemm_autotuned
from tensorrt_llm._torch.cute_dsl_kernels.minimax_m3.mx_gemv import op as mx_gemv
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.linear import (
    Linear,
    MXFP8LinearMethod,
    flashinfer_mxfp8_autotune_active,
    flashinfer_mxfp8_decode_graph_capture_active,
)
from tensorrt_llm._torch.utils import is_torch_compiling

from . import dispatch
from .kernels import m3_mc_allreduce, m3_prequant
from .kernels.mxfp8_low_m_gemm import mxfp8_low_m_gemm
from .ops.gemm.m3_mx_gemv import m3_mx_gemv
from .ops.gemm.m3_mx_gemv_bf16 import m3_mx_gemv_bf16

# The boundary all-reduce a projection's GEMV pushes its rows into, when the decode forward arms one: the attention
# boundary (o_proj) or the MoE boundary's shared-expert set (the shared expert's down projection).
PUSH_ATTN = "attn"
PUSH_MOE = "moe"


def mark_m3_linear(module: Linear, low_m_gemm: bool, push: Optional[str] = None) -> None:
    """Mark a projection for this target's decode dispatch; ``adopt_m3_linears`` applies it after the load.

    ``low_m_gemm``: the projection may take the fused low-M MXFP8 GEMM. ``push``: ``PUSH_ATTN`` / ``PUSH_MOE`` when
    its GEMV pushes into that boundary's all-reduce whenever the decode forward armed it.
    """
    module.m3_low_m_gemm = low_m_gemm
    module.m3_push = push


def adopt_m3_linears(root: nn.Module) -> None:
    """Put every marked projection under ``root`` on the decode dispatch.

    Runs once the weights are real: the quantization method a projection ends
    up with is only final after the model applied its quantization config.
    It also compiles every decode GEMV kernel a projection can reach here,
    before the executor's warmup: on the current (main) stream, the same shapes
    in the same order on every rank, and never inside a forward between
    collectives or on the MoE aux stream (``prebuild_m3_gemvs``).
    """
    use_pdl = get_env_enable_pdl()
    compiled = False
    for name, module in root.named_modules():
        if not hasattr(module, "m3_low_m_gemm"):
            continue
        method = module.quant_method
        if not isinstance(method, MXFP8LinearMethod) or not method.use_cutlass:
            raise ValueError(
                f"{name}: the MiniMax-M3 decode dispatch needs a CUTLASS MXFP8 projection, got {type(method).__name__}"
            )
        if type(method) is MXFP8LinearMethod:
            method.__class__ = M3MXFP8LinearMethod
        mx_gemv.ensure_compiled(module.weight, module.weight_scale, use_pdl)
        prebuild_m3_gemvs(module, use_pdl)
        compiled = True
    if compiled:
        torch.cuda.synchronize()


def _bucket_tokens(rows: int) -> int:
    """The token capacity of the multicast all-reduce bucket a ``rows``-token boundary uses."""
    return next(b for b in m3_mc_allreduce.BUCKETS if rows <= b)


def gemv_variants(module: Linear) -> List[Tuple[int, bool, Optional[Tuple[int, int, int]]]]:
    """Every (rows, bf16 activation, push (world, bucket tokens, mode)) call the decode dispatch can route to this
    projection's GEMV (``M3MXFP8LinearMethod.apply``): an MXFP8 activation on the rows ``MX_GEMV_ROWS`` gives it, a
    bf16 one up to ``MX_GEMV_BF16_MAX_TOKENS``; each pushing when the projection's boundary push covers the rows."""
    n, k = module.weight.shape
    push_rows = {PUSH_ATTN: dispatch.ATTN_PUSH.keys, PUSH_MOE: dispatch.MOE_PUSH.keys}.get(
        module.m3_push, frozenset()
    )
    world = int(module.tp_size)
    variants = []
    for rows in range(1, dispatch.DECODE_MAX_TOKENS + 1):
        for x_bf16, served in (
            (False, rows in dispatch.MX_GEMV_ROWS.get((n, k), ())),
            (True, rows <= dispatch.MX_GEMV_BF16_MAX_TOKENS.get((n, k), 0)),
        ):
            if not served:
                continue
            variants.append((rows, x_bf16, None))
            if rows in push_rows:
                variants.append(
                    (rows, x_bf16, (world, _bucket_tokens(rows), dispatch.AR_PUSH_MODE))
                )
    return variants


def prebuild_m3_gemvs(module: Linear, use_pdl: bool) -> int:
    """Compile every GEMV kernel of ``gemv_variants`` (the push ones included) without launching; returns how many
    distinct kernels the projection takes."""
    keys = set()
    for rows, x_bf16, push in gemv_variants(module):
        key = mx_gemv.precompile(
            module.weight, module.weight_scale, rows, x_bf16, use_pdl=use_pdl, ar=push
        )
        if key is not None:
            keys.add(key)
    return len(keys)


def _low_m_gemm_supports(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> bool:
    """Whether the fused MXFP8 low-M kernel handles these operands."""
    return (
        x.dim() == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.stride(-1) == 1
        and weight.dtype == torch.float8_e4m3fn
        and weight.is_contiguous()
        and weight_scale.dtype == torch.uint8
        and x.shape[1] == weight.shape[1]
        and weight.shape[1] % 256 == 0
    )


def _push_target(module: Linear):
    """The armed boundary all-reduce this projection pushes its output into, if any."""
    if module.m3_push == PUSH_MOE:
        return m3_prequant.MOE_PUSH.take_shared()
    if module.m3_push == PUSH_ATTN:
        return m3_prequant.AR_PUSH.take_target()
    return None


def _uncompiled_note() -> str:
    key = mx_gemv.last_uncompiled
    return "" if key is None else f" (its kernel {key} was not compiled before warmup)"


def _note_pushed(output: Optional[torch.Tensor], module: Linear) -> None:
    if output is None or not mx_gemv.last_pushed:
        return
    if module.m3_push == PUSH_MOE:
        m3_prequant.MOE_PUSH.shared_pushed = output.data_ptr()
    else:
        m3_prequant.AR_PUSH.pushed = output.data_ptr()


class M3MXFP8LinearMethod(MXFP8LinearMethod):
    """``MXFP8LinearMethod`` with the decode-graph dispatch in the module docstring."""

    def apply(self, module: Linear, input: torch.Tensor, bias: Optional[torch.Tensor]):
        if bias is not None or input.dim() != 2:
            return super().apply(module, input, bias)
        input = input.contiguous()
        weight, weight_scale = module.weight, module.weight_scale
        use_pdl = get_env_enable_pdl()
        decode_graph = not is_torch_compiling() and flashinfer_mxfp8_decode_graph_capture_active()
        if not decode_graph:
            return super().apply(module, input, bias)

        rows = input.shape[0]
        n, k = weight.shape
        prequant = m3_prequant.SLOT.take(input)
        if module.dtype != torch.bfloat16 or (prequant is None and input.dtype != torch.bfloat16):
            path = (
                dispatch.PREQUANT_GEMM_PATH if prequant is not None else dispatch.GENERIC_GEMM_PATH
            )
        else:
            path = dispatch.linear_path(n, k, rows, prequant is not None, module.m3_low_m_gemm)
        output = None
        if prequant is not None:
            act, act_scale = prequant
            if path == dispatch.MX_GEMV_PATH:
                output = m3_mx_gemv(
                    act, act_scale, weight, weight_scale, use_pdl=use_pdl, ar=_push_target(module)
                )
                _note_pushed(output, module)
                if output is None:
                    dispatch.missed(
                        path,
                        rows,
                        f"the GEMV declined the [{n}, {k}] projection's MXFP8 activation"
                        + _uncompiled_note(),
                    )
            if output is None:
                if path == dispatch.PREQUANT_GEMM_PATH:
                    dispatch.uncovered(
                        dispatch.MX_GEMV_PATH,
                        f"the [{n}, {k}] projection at rows the GEMV does not serve",
                    )
                output = mxfp8_gemm_autotuned(
                    act,
                    act_scale,
                    weight,
                    weight_scale,
                    module.dtype,
                    tune=flashinfer_mxfp8_autotune_active(),
                )
            return output
        if path == dispatch.MX_GEMV_BF16_PATH:
            output = m3_mx_gemv_bf16(
                input, weight, weight_scale, use_pdl=use_pdl, ar=_push_target(module)
            )
            _note_pushed(output, module)
            if output is None:
                dispatch.missed(
                    path,
                    rows,
                    f"the GEMV declined the [{n}, {k}] projection's input" + _uncompiled_note(),
                )
        if (
            output is None
            and path in (dispatch.MX_GEMV_BF16_PATH, dispatch.LOW_M_GEMM_PATH)
            and module.m3_low_m_gemm
            and rows <= dispatch.LOW_M_GEMM_MAX_TOKENS.get((n, k), 0)
            and _low_m_gemm_supports(input, weight, weight_scale)
        ):
            output = mxfp8_low_m_gemm(input, weight, weight_scale, module.dtype, use_pdl=use_pdl)
            if output is None and path == dispatch.LOW_M_GEMM_PATH:
                dispatch.missed(path, rows, f"the GEMM declined the [{n}, {k}] projection's input")
        if output is None:
            if path == dispatch.GENERIC_GEMM_PATH:
                dispatch.uncovered(
                    dispatch.MX_GEMV_BF16_PATH,
                    f"the [{n}, {k}] projection's bf16 input beyond the decode GEMV's rows",
                )
            # Stock's decode-graph dispatch for this shape (mxfp8_quantize_gemm_autotuned): same tactics, same bits.
            output = super().apply(module, input, bias)
        return output
