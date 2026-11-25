# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared drafter precision-layout routing, including row-parallel forward dispatch.

Providers and projection tactics belong to the target. This module reuses the
packaged GV_C routing and keeps each projection's own all-reduce contract.
Natural acceptance-length gating is required for this precision change.
"""

from __future__ import annotations

import torch

from tensorrt_llm.logger import logger

_HIS_TAKEN: set = set()
MAX_ROWS = 8


def route_his_layout(
    draft_model,
    rows,
    *,
    gemv,
    recover,
    swizzle,
    configs,
    down_configs,
    fuse_mlp=True,
    max_rows=MAX_ROWS,
    quantize=None,
    quantize_max_rows=0,
):
    """Route supported decode rows using target providers; return linear/MLP counts.

    ``max_rows`` bounds the routed row counts (default MAX_ROWS, the TP4/TP8 reference tactics' bound); a target whose
    ``configs`` are its own row-independent GEMV tactics may pass its GEMV's row bound so the projections it takes
    keep every decode row its DRAFTER_MXFP8 routing takes.

    ``quantize`` (optional, default None: TP4 / TP8 unchanged) is the target's bf16 [M, K] -> (E4M3 [M, K], swizzled
    UE8M0 scales) quantize, bitwise ``torch.ops.trtllm.mxfp8_quantize(x, True)``. With it, a linear projection's rows
    up to ``quantize_max_rows`` that the bf16-input GEMV declines (its in-kernel quantize is capped per CTA) but the
    MXFP8-input GEMV serves run that quantize + ``gemv.m3_mx_gemv`` on the same config: the same operands as the
    bf16-input GEMV's in-kernel quantize, so the same drafter precision as the rows it serves.
    """

    def _his_taken(tag: str, n: int, k: int, m: int) -> None:
        key = (tag, n, k, m)
        if key not in _HIS_TAKEN:
            _HIS_TAKEN.add(key)
            logger.info(f"[M3 dispatch] m31_drafter_hislayout taken at {m} rows ({tag} {n} x {k})")

    def _compile_now(call) -> bool:
        """Run one eager call (compiling its kernel) even while mx_gemv compiles are frozen; True if it was served."""
        was = gemv.frozen()
        gemv.freeze(False)
        try:
            return call() is not None
        finally:
            gemv.freeze(was)

    def _route_his_linear(lin, served_rows) -> bool:
        weight = getattr(lin, "weight", None)
        if (
            weight is None
            or getattr(lin, "bias", None) is not None
            or getattr(lin, "_m31_mxdraft", None) is not None
        ):
            return False
        n, k = weight.shape
        cfg = configs.get((n, k))
        if cfg is None:
            return False
        rec = recover(weight.data)
        if rec is None:
            return False
        w8, sf = rec[0], swizzle(rec[1])
        served = tuple(
            m
            for m in served_rows
            if not gemv._declines(n, k, m, True, cfg)
            and _compile_now(
                lambda m=m: gemv.m3_mx_gemv_bf16(
                    torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device),
                    w8,
                    sf,
                    True,
                    cfg,
                )
            )
        )
        served_q = tuple(
            m
            for m in served_rows
            if quantize is not None
            and m <= quantize_max_rows
            and m not in served
            and not gemv._declines(n, k, m, False, cfg)
            and _compile_now(
                lambda m=m: gemv.m3_mx_gemv(
                    *quantize(torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device)),
                    w8,
                    sf,
                    True,
                    cfg,
                )
            )
        )
        if not served and not served_q:
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
            _served_q=served_q,
            _cfg=cfg,
        ):
            if bias is None and not lora_params:
                out = _gemv_rows(input, _w8, _sf, _served, _served_q, _cfg)
                if out is not None:
                    return out
            return _o(input, bias, lora_params, layer_idx)

        lin._m31_mxdraft = (w8, sf, served + served_q)
        lin._m31_hislayout = cfg
        lin._m31_hislayout_mxq = served_q
        lin.apply_linear = apply_linear
        if _row_parallel_reducing(lin):
            _wrap_row_parallel_forward(lin, w8, sf, served, served_q, cfg)
        return True

    def _gemv_rows(input, w8, sf, served, served_q, cfg):
        """The GEMV's output for a 2-D contiguous bf16 decode input of a served row count (bf16-input GEMV), or of a
        ``served_q`` row count (quantize, then the MXFP8-input GEMV); None for any other call or a declined GEMV."""
        if not (
            isinstance(input, torch.Tensor)
            and input.dtype == torch.bfloat16
            and input.dim() == 2
            and input.is_contiguous()
        ):
            return None
        rows = input.shape[0]
        if rows in served:
            out = gemv.m3_mx_gemv_bf16(input, w8, sf, True, cfg)
            if out is not None:
                _his_taken("gemv", w8.shape[0], w8.shape[1], rows)
            return out
        if rows in served_q:
            xq, xsf = quantize(input)
            out = gemv.m3_mx_gemv(xq, xsf, w8, sf, True, cfg)
            if out is not None:
                _his_taken("quantize+gemv", w8.shape[0], w8.shape[1], rows)
            return out
        return None

    def _row_parallel_reducing(lin) -> bool:
        mode = getattr(lin, "tp_mode", None)
        return (
            mode is not None
            and getattr(mode, "name", str(mode)).endswith("ROW")
            and bool(getattr(lin, "reduce_output", False))
            and int(getattr(lin, "tp_size", 1)) > 1
        )

    def _wrap_row_parallel_forward(lin, w8, sf, served, served_q, cfg) -> None:
        """A row-parallel Linear's served forward (Linear.forward, tensorrt_llm/_torch/modules/linear.py:4316) does not
        reach ``apply_linear`` on two of its three ROW branches: the fused GEMM + all-reduce (linear.py:4339,
        apply_linear_allreduce) and the NCCL-window zero-copy GEMM (linear.py:4350-4357: ``quant_method.apply`` straight
        into the window buffer when TLLM_NCCL_SYMMETRIC_ZERO_COPY=1, the default, and the projection's
        all-reduce reports
        ``uses_nccl_symmetric_memory_window`` -- the drafter's DraftOneshotAllReduce (dspark.py:139-140)
        delegates that to
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
            _served_q=served_q,
            _cfg=cfg,
        ):
            if not lora_params:
                out = _gemv_rows(input, _w8, _sf, _served, _served_q, _cfg)
                if out is not None:
                    return lin.all_reduce(out, all_reduce_params=all_reduce_params)
            return _stock(
                input,
                all_reduce_params=all_reduce_params,
                lora_params=lora_params,
                layer_idx=layer_idx,
            )

        lin.forward = forward
        lin._m31_hislayout_row_forward = True

    def _route_his_mlp(mlp, served_rows) -> bool:
        """The drafter MLP's SwiGLU + down projection as one m3_mx_gemv_swiglu launch for the served rows: the MLP's
        forward runs gate_up_proj, then the fused GEMV on its output, then the down projection's own all-reduce (row
        parallel), exactly the path the GatedMLP takes for this drafter (gate_up -> _apply_activation ->
        down: apply_linear
        then all_reduce); anything else (other rows, LoRA, a declined call) runs the stock forward."""
        down = getattr(mlp, "down_proj", None)
        gu = getattr(mlp, "gate_up_proj", None)
        if down is None or gu is None or getattr(mlp, "_m31_hislayout_mlp", None) is not None:
            return False
        if getattr(mlp, "split_gate_up", False) or getattr(down, "bias", None) is not None:
            return False
        weight = down.weight
        n, k = weight.shape
        cfg = down_configs.get((n, k))
        if cfg is None or gu.weight.shape[0] != 2 * k:
            return False
        rec = recover(weight.data)
        if rec is None:
            return False
        w8, sf = rec[0], swizzle(rec[1])
        served = tuple(
            m
            for m in served_rows
            if _compile_now(
                lambda m=m: gemv.m3_mx_gemv_swiglu(
                    torch.zeros(m, 2 * k, dtype=torch.bfloat16, device=weight.device),
                    w8,
                    sf,
                    True,
                    cfg,
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
                    out = gemv.m3_mx_gemv_swiglu(h1, _w8, _sf, True, _cfg)
                    if out is not None:
                        _his_taken("swiglu+down", _w8.shape[0], _w8.shape[1], x.shape[0])
                        if getattr(down, "tp_size", 1) > 1 and getattr(
                            down, "reduce_output", False
                        ):
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

    served_rows = sorted({int(m) for m in rows if 0 < int(m) <= max_rows})
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
        if fuse_mlp:
            if mlp is not None and _route_his_mlp(mlp, served_rows):
                n_mlp += 1
        else:
            down = getattr(mlp, "down_proj", None)
            if down is not None and _route_his_linear(down, served_rows):
                n_lin += 1
    return n_lin, n_mlp
