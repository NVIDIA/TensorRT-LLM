# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Elementwise kernels of the DFlash 2 drafter folded into the split-K GEMV before them (``drafter_folds``).

Each fold removes one launch from the drafter's block decode and computes the same values
(``ops/gemm/dr3_fold_splitk_gemm.py``: the GEMV's owner CTA applies the elementwise kernel's statements to its
reduced, BF16-rounded tile). ``install`` (the ``drafter_folds`` fast path) folds CF, with its spread epilogue,
and SF; no composition installs QF.

* CF: each block convolution's prepare side. The coefficient projection
  (``kernel_projection``) of every DFlash 2 convolution runs ``conv_splitk_gemm``, which also writes the
  convolution of its input; the 8-row ``prepare`` launches no ``dflash2_block_conv``. The projection weight's
  side-0 rows are permuted in place at install (``permute_conv_rows``, per kernel-M tile of the fixed tactic:
  a runtime relayout of the loaded parameter, not a checkpoint change); the side-1 rows the drafter hop reads
  through ``finish`` keep their order. Any other ``prepare`` call runs the GEMM on the permuted weight,
  restores the column order (``unpermute_conv_columns``) and runs the built-in convolution.
* SF: each drafter MLP's ``silu(gate) * up``. ``gate_up_proj``'s weight
  rows are interleaved in place per kernel-M tile (``interleave_gate_up``); the 8-row MLP runs
  ``silu_splitk_gemm`` then ``down_proj``; any other call runs the GEMM on the interleaved weight, restores the
  ``[gate | up]`` columns and runs the built-in activation.
* QF (``install_qk_fold``, on top of ``drafter_attn_pdl``): each drafter layer's QK norm +
  RoPE. The block decode skips ``drafter_attn_pdl``'s launch and hands the window attention a ``QkFold``: its
  split programs normalize and rotate the raw query / key heads they read (``ops/spec/dr3_qk_norm_rope.py``,
  the CUDA kernel's bits) and its append stores the rotated keys. No weight is changed.

Ordering (PDL): the fused GEMV keeps flashinfer's split-K protocol -- its weight streams before its
``griddepcontrol.wait``, its activation after it, its dependents are released after the weight's last load --
and CF's owner warps execute ``griddepcontrol.wait`` before they read the activation rows for the convolution.
The dependent GEMV (``qkv_proj`` / ``gate_up_proj`` after CF, ``down_proj`` after SF) reads its activation after
its own wait, i.e. after the fused GEMV's grid completed. No buffer is shared across launches beyond the
fused GEMV's fresh outputs.

Install order: after ``drafter_block_conv`` / ``drafter_block_decode`` (the convolution class whose ``prepare``
consults CF), ``drafter_fixed_gemms`` (the tactics) and ``drafter_pdl_chain`` (``EntryTriggerGatedMLP``, whose
``forward`` consults SF). Idempotent: a module already folded is left alone (its weight is not permuted twice).
"""

from __future__ import annotations

import collections
from typing import Optional

import torch
from torch import nn

from tensorrt_llm.logger import logger

#: Instance-dict key marking a CF-folded convolution for the spread epilogue: every CF-folded convolution runs
#: ``ops/gemm/dr3_fold_splitk_gemm.py``'s ``spread=True`` epilogue, where each CTA of a side-0 cluster convolves
#: a quarter of the tile's columns instead of the owner all of them (``install_conv_spread``, part of
#: ``drafter_folds``). Value-identical: the same FP32 sum in the same order, the same rounding points
#: (``ConvSplitKDenseGemmKernel``'s docstring). It acts on the CF kernels only (the coefficient GEMVs); the SF
#: gate_up GEMVs and their tactic are independent of it.
CONV_SPREAD_KEY = "_sp_cf2"

#: Whether the folded GEMVs release their dependents (the next projection GEMV) right after their first weight
#: ring (``ops/gemm/dr3_fold_splitk_gemm.py``, ``entry_trigger``) instead of after their last one. Off in every
#: composition; a test may set it. Value-identical: a trigger placement only.
GEMV_ENTRY_TRIGGER = False
#: SF's gate_up tactic ``(mma_m, mma_n, split_k, ab_stages)``: the sm107 table's gate_up tactic
#: (``fixed_low_m_gemm.SM107_SPLITK_TACTICS``) with ``mma_n`` 8 and 12 ring stages. Value-identical: it keeps
#: the table tactic's ``mma_m`` (so the weight interleave is the same) and ``split_k`` (so every output element
#: is the same FP32 sum: the MMA chain over the same K slices in the same order, then the same rank-order DSMEM
#: reduction); only the ring depth and the activation tile width change (``_fold_tactic`` raises otherwise). The
#: unfused gate_up GEMV keeps the table's tactic; CF runs at the table's tactic. On an architecture
#: other than ``SF_TACTIC_SM`` SF runs at the table's tactic.
SF_TACTIC = (64, 8, 1, 12)
SF_TACTIC_SM = (10, 7)

#: The folded block decode's row count: one request of K + 1 = 8 rows (the fixed tactics' M).
ROWS = 8

#: Instance-dict keys holding a folded module's ``SplitKTactic``.
CONV_KEY = "_dr3_cf"
SILU_KEY = "_dr3_sf"
#: The drafter's instance-dict key holding QF's per-layer (eps, base).
QK_KEY = "_dr3_qf"

#: Calls per path (``conv_fused`` / ``conv_plain`` / ``silu_fused`` / ``silu_plain`` / ``qk_fused``), host side:
#: a test reads them to see that a folded forward took the fused kernels.
CALLS = collections.Counter()


def _splitk_tactic(rows: int, in_features: int, out_features: int):
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic

    from ...ops.gemm.fixed_low_m_gemm import tactic_for

    raw = tactic_for(rows, in_features, out_features)
    return None if raw is None else SplitKTactic(*raw)


def _fold_tactic(override, rows: int, in_features: int, out_features: int):
    """The folded GEMV's tactic: the fixed table's, or ``override`` (the same ``mma_m`` / ``split_k``).

    Raises:
        ValueError: on an override that changes ``mma_m`` or ``split_k``.
    """
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic

    served = _splitk_tactic(rows, in_features, out_features)
    if served is None or override is None:
        return served
    tactic = SplitKTactic(*override)
    if (tactic.mma_m, tactic.split_k) != (served.mma_m, served.split_k):
        raise ValueError(
            f"[musespark_open] fold tactic {tuple(override)}: a fold tactic keeps the table tactic's mma_m and split_k "
            f"({served}); only mma_n and ab_stages may change"
        )
    return tactic


def _sf_override(weight: torch.Tensor):
    """``SF_TACTIC`` on an ``SF_TACTIC_SM`` device, else None (the table's tactic)."""
    if weight.is_cuda and tuple(torch.cuda.get_device_capability(weight.device)) == SF_TACTIC_SM:
        return SF_TACTIC
    return None


# ---- CF ----------------------------------------------------------------------------------------------------------
def _conv_modules(draft_model: nn.Module) -> list:
    return list(getattr(draft_model, "attention_convs", None) or ()) + list(
        getattr(draft_model, "mlp_convs", None) or ()
    )


def conv_fold_reason(draft_model: Optional[nn.Module]) -> Optional[str]:
    """Why CF cannot be installed on ``draft_model``, or None."""
    from ...ops.gemm import dr3_fold_splitk_gemm as _fold
    from ...spec_worker import FixedTacticProjection
    from ..drafter_block_decode.block_decode import FusedDFlash2BlockConv

    if draft_model is None:
        return "no drafter"
    convs = _conv_modules(draft_model)
    if not convs:
        return "the drafter has no DFlash 2 block convolutions"
    for index, conv in enumerate(convs):
        where = f"block convolution {index}"
        if type(conv) is not FusedDFlash2BlockConv:
            return f"{where} is {type(conv).__name__}, not the fused convolution (select drafter_block_conv)"
        if conv.taps != _fold.CONV_TAPS or conv.group_size != _fold.CONV_GROUP:
            return f"{where} has {conv.taps} taps / group {conv.group_size}"
        proj = conv.kernel_projection
        if type(proj) is not FixedTacticProjection or proj.bias is not None:
            return f"{where}'s projection is not on the fixed tactic (select drafter_fixed_gemms)"
        if not proj._fixed_gemm_pdl:
            return f"{where}'s projection is launched without programmatic dependent launch"
        weight = proj.weight
        if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
            return f"{where}'s projection weight is not contiguous BF16"
        base = conv.base_kernel
        if base.dtype != torch.bfloat16 or not base.is_contiguous():
            return f"{where}'s base kernel is not contiguous BF16"
        tactic = _fold_tactic(None, ROWS, weight.shape[1], weight.shape[0])
        if tactic is None:
            return f"{where}: no fixed tactic for its projection at {ROWS} rows"
        x = torch.empty(ROWS, weight.shape[1], dtype=torch.bfloat16, device=weight.device)
        with torch.inference_mode():
            if not _fold.conv_takes(x, weight, base[0], ROWS, tactic):
                return f"{where}: the fused projection does not take tactic {tactic}"
    return None


def install_conv_fold(draft_model: nn.Module) -> int:
    """Fold every block convolution's prepare side into its projection; return how many changed.

    Raises:
        ValueError: ``conv_fold_reason``.
    """
    from ...ops.gemm import dr3_fold_splitk_gemm as _fold

    convs = _conv_modules(draft_model) if draft_model is not None else []
    if convs and all(CONV_KEY in conv.__dict__ for conv in convs):
        return 0
    reason = conv_fold_reason(draft_model)
    if reason is not None:
        raise ValueError(f"[musespark_open] drafter conv fold cannot run here: {reason}")
    changed = 0
    for conv in convs:
        if CONV_KEY in conv.__dict__:
            continue
        weight = conv.kernel_projection.weight
        tactic = _fold_tactic(None, ROWS, weight.shape[1], weight.shape[0])
        with torch.no_grad():
            weight.data.copy_(_fold.permute_conv_rows(weight.data, tactic.mma_m))
        conv.__dict__[CONV_KEY] = tactic
        changed += 1
    return changed


def install_conv_spread(draft_model: nn.Module) -> int:
    """Mark every CF-folded convolution for the spread epilogue; return how many changed.

    Raises:
        ValueError: if CF is not installed on every convolution, or a convolution's tactic is not covered.
    """
    from ...ops.gemm import dr3_fold_splitk_gemm as _fold

    convs = _conv_modules(draft_model) if draft_model is not None else []
    if not convs or not all(CONV_KEY in conv.__dict__ for conv in convs):
        raise ValueError(
            "[musespark_open] the conv fold's spread epilogue needs the conv fold on every block convolution"
        )
    changed = 0
    for index, conv in enumerate(convs):
        if conv.__dict__.get(CONV_SPREAD_KEY):
            continue
        tactic = conv.__dict__[CONV_KEY]
        if not _fold.spread_takes(tactic):
            raise ValueError(
                f"[musespark_open] the spread epilogue: block convolution {index}'s tactic {tactic} is not covered"
            )
        conv.__dict__[CONV_SPREAD_KEY] = True
        changed += 1
    return changed


def folded_prepare(conv: nn.Module, tactic, hidden_states: torch.Tensor, block_size: int):
    """``DFlash2BlockConv.prepare`` on a CF-folded convolution: ``(convolved input, finish coefficients)``."""
    from ...ops.gemm import dr3_fold_splitk_gemm as _fold

    proj = conv.kernel_projection
    weight = proj.weight
    rows = hidden_states.shape[0]
    spread = bool(conv.__dict__.get(CONV_SPREAD_KEY))
    if (
        hidden_states.dim() == 2
        and rows == ROWS
        and _fold.conv_takes(hidden_states, weight, conv.base_kernel[0], block_size, tactic, spread)
    ):
        CALLS["conv_fused_v2" if spread else "conv_fused"] += 1
        out, coefficients = _fold.conv_splitk_gemm(
            hidden_states,
            weight,
            conv.base_kernel[0],
            block_size,
            tactic,
            proj._fixed_gemm_pdl,
            spread,
            entry_trigger=GEMV_ENTRY_TRIGGER,
        )
        return out, coefficients.view(rows, 2, conv.taps, conv.num_groups)[:, 1]
    CALLS["conv_plain"] += 1
    coefficients = _fold.unpermute_conv_columns(proj(hidden_states), tactic.mma_m).view(
        rows, 2, conv.taps, conv.num_groups
    )
    return conv._convolve(hidden_states, coefficients[:, 0], 0, block_size), coefficients[:, 1]


# ---- SF ----------------------------------------------------------------------------------------------------------
def silu_fold_reason(draft_model: Optional[nn.Module]) -> Optional[str]:
    """Why SF cannot be installed on ``draft_model``, or None."""
    from ...ops.gemm import dr3_fold_splitk_gemm as _fold
    from ..drafter_pdl_chain.chain import EntryTriggerGatedMLP, _fixed_tactic_reason

    if draft_model is None:
        return "no drafter"
    for index, layer in enumerate(draft_model.model.layers):
        mlp = layer.mlp
        where = f"layer {index} mlp"
        if type(mlp) is not EntryTriggerGatedMLP:
            return f"{where} is {type(mlp).__name__}, not the PDL chain's MLP (select drafter_pdl_chain)"
        linear = mlp.gate_up_proj
        reason = _fixed_tactic_reason(linear, ROWS, f"{where} gate_up_proj")
        if reason is not None:
            return reason
        if linear.bias is not None or not linear.weight.is_contiguous():
            return f"{where} gate_up_proj has a bias or a non-contiguous weight"
        weight = linear.weight
        tactic = _fold_tactic(_sf_override(weight), ROWS, weight.shape[1], weight.shape[0])
        x = torch.empty(ROWS, weight.shape[1], dtype=torch.bfloat16, device=weight.device)
        with torch.inference_mode():
            if tactic is None or not _fold.silu_takes(x, weight, tactic):
                return f"{where}: the fused gate_up does not take tactic {tactic}"
    return None


def install_silu_fold(draft_model: nn.Module) -> int:
    """Fold every drafter MLP's activation into its gate_up GEMV; return how many changed.

    Raises:
        ValueError: ``silu_fold_reason``.
    """
    from ...ops.gemm import dr3_fold_splitk_gemm as _fold

    layers = list(draft_model.model.layers) if draft_model is not None else []
    if layers and all(SILU_KEY in layer.mlp.__dict__ for layer in layers):
        return 0
    reason = silu_fold_reason(draft_model)
    if reason is not None:
        raise ValueError(f"[musespark_open] drafter silu fold cannot run here: {reason}")
    changed = 0
    for layer in layers:
        mlp = layer.mlp
        if SILU_KEY in mlp.__dict__:
            continue
        weight = mlp.gate_up_proj.weight
        tactic = _fold_tactic(_sf_override(weight), ROWS, weight.shape[1], weight.shape[0])
        with torch.no_grad():
            weight.data.copy_(_fold.interleave_gate_up(weight.data, tactic.mma_m))
        mlp.__dict__[SILU_KEY] = tactic
        changed += 1
    return changed


def folded_mlp(mlp: nn.Module, tactic, x, final_all_reduce_params, lora_params):
    """``GatedMLP.forward`` on an SF-folded MLP (interleaved ``gate_up_proj`` weight)."""
    from ...ops.gemm import dr3_fold_splitk_gemm as _fold

    if lora_params:
        raise NotImplementedError(
            "[musespark_open] drafter silu fold: LoRA on an interleaved gate_up_proj"
        )
    weight = mlp.gate_up_proj.weight
    rows = x.reshape(-1, x.shape[-1]) if isinstance(x, torch.Tensor) and x.dim() >= 2 else None
    if rows is not None and rows.shape[0] == ROWS and _fold.silu_takes(rows, weight, tactic):
        CALLS["silu_fused"] += 1
        h2 = _fold.silu_splitk_gemm(
            rows, weight, tactic, mlp.gate_up_proj.quant_method.enable_pdl, GEMV_ENTRY_TRIGGER
        ).view(*x.shape[:-1], weight.shape[0] // 2)
    else:
        CALLS["silu_plain"] += 1
        h2 = mlp._apply_activation(_fold.deinterleave_columns(mlp.gate_up_proj(x), tactic.mma_m))
    return mlp.down_proj(h2, all_reduce_params=final_all_reduce_params, layer_idx=mlp.layer_idx)


# ---- QF ----------------------------------------------------------------------------------------------------------
def qk_fold_reason(draft_model: Optional[nn.Module]) -> Optional[str]:
    """Why QF cannot be installed on ``draft_model``, or None.

    QF replaces ``drafter_attn_pdl``'s QK norm + RoPE launch, so it needs that path and the geometry its Triton
    copy reproduces (NeoX, full rotary 128, no YaRN, QK norm without the Gemma weight, BF16 weights). The split
    kernel's pre-wait loads then follow the qkv GEMV directly: that GEMV releases its dependents from its weight
    producer after the last weight tile, which needs more k-tiles than pipeline stages to imply that the CTA has
    passed its own ``griddepcontrol.wait`` (so the GEMV's predecessors are complete, as with the QK norm kernel
    between them).
    """
    if draft_model is None:
        return "no drafter"
    pdl = draft_model.__dict__.get("_pt_attn_pdl")
    if pdl is None:
        return "needs drafter_attn_pdl, whose launch the fold replaces"
    from ..drafter_pdl_chain.chain import _fixed_tactic_reason

    for index, (layer, args) in enumerate(zip(draft_model.model.layers, pdl.layer_args)):
        where = f"layer {index} attention"
        if not (
            args.is_neox
            and args.head_dim == 128
            and args.rotary_dim == 128
            and args.factor == 1.0
            and args.attention_factor == 1.0
            and args.is_qk_norm
            and not args.use_gemma
        ):
            return f"{where}: {args} is not the NeoX / full-rotary / no-YaRN QK norm the fold reproduces"
        attn = layer.self_attn
        if attn.q_norm.weight.dtype != torch.bfloat16 or attn.k_norm.weight.dtype != torch.bfloat16:
            return f"{where}: the QK norm weights are not BF16"
        proj = attn.qkv_proj
        reason = _fixed_tactic_reason(proj, ROWS, f"{where} qkv_proj")
        if reason is not None:
            return reason
        tactic = _splitk_tactic(ROWS, proj.weight.shape[1], proj.weight.shape[0])
        if tactic is None:
            return f"{where}: no fixed tactic for qkv_proj at {ROWS} rows"
        k_tiles = proj.weight.shape[1] // 128 // tactic.split_k
        if k_tiles <= tactic.ab_stages:
            return f"{where}: qkv_proj has {k_tiles} k-tiles per CTA for {tactic.ab_stages} stages"
    return None


def install_qk_fold(draft_model: nn.Module) -> int:
    """Set ``draft_model``'s QF marker (the per-layer eps / base); return how many layers it covers (0 if set).

    Raises:
        ValueError: ``qk_fold_reason``.
    """
    if QK_KEY in draft_model.__dict__:
        return 0
    reason = qk_fold_reason(draft_model)
    if reason is not None:
        raise ValueError(f"[musespark_open] drafter qk fold cannot run here: {reason}")
    layer_args = draft_model.__dict__["_pt_attn_pdl"].layer_args
    draft_model.__dict__[QK_KEY] = tuple((args.eps, args.base) for args in layer_args)
    return len(layer_args)


def layer_qk_fold(qk_fold, layer_idx: int, attn_mod: nn.Module, positions_i32: torch.Tensor):
    """The ``QkFold`` of one drafter layer's window attention call."""
    from ...ops.spec.dflash_window_attention import QkFold

    CALLS["qk_fused"] += 1
    eps, base = qk_fold[layer_idx]
    return QkFold(attn_mod.q_norm.weight, attn_mod.k_norm.weight, positions_i32, eps, base, 1.0)


# ---- entry point ------------------------------------------------------------------------------------------------
def install(draft_model: Optional[nn.Module]) -> None:
    """The ``drafter_folds`` folds: CF with its spread epilogue, then SF at ``SF_TACTIC``. Idempotent.

    Raises:
        ValueError: as ``install_conv_fold``, ``install_conv_spread`` and ``install_silu_fold``.
    """
    changed = install_conv_fold(draft_model)
    if changed:
        logger.info(
            f"[musespark_open] drafter conv fold: block convolution prepare in the coefficient GEMV ({changed} modules)"
        )
    changed = install_conv_spread(draft_model)
    if changed:
        logger.info(
            f"[musespark_open] drafter conv fold: the epilogue spread over each side-0 cluster ({changed} modules)"
        )
    changed = install_silu_fold(draft_model)
    if changed:
        logger.info(
            f"[musespark_open] drafter silu fold: silu(gate) * up in the gate_up GEMV ({changed} MLPs)"
        )
        tactic = draft_model.model.layers[0].mlp.__dict__[SILU_KEY]
        logger.info(f"[musespark_open] drafter silu fold tactic: {tactic}")
