# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-side residual join of a sequence-parallel layer also writes the router's FP32 input.

On a sequence-parallel context step each rank's MoE routes its own rows: the built-in
router reads the pre-FFN norm output (BF16, ``[shard, 8192]``) as FP32, so the step
runs a standalone BF16 -> FP32 copy of it in front of the router GEMM (``router_logits``
-> ``self.router(hidden_states.to(torch.float32))``). That norm output is written by
the attention-side residual join (``MuseSparkOpenGatedResidual._add_after_rmsnorm_and_norm``,
one Triton kernel), which holds every value in registers when it rounds it to BF16.

``SequenceParallelAttnJoin`` re-classes a layer's ``add_attn`` so that, while a
sequence-parallel forward runs (``SequenceParallelStep.active``), the join is this
module's copy of the built-in kernel with one more store: the BF16-rounded norm output
widened back to FP32, which is ``ffn_input.to(torch.float32)`` bit for bit. The pair is
handed to the layer's MoE (``moe.sp_router_input``), whose sequence-parallel arm
(``moe.ContextSequenceParallelMoE``) feeds the FP32 rows to the router GEMM when it is
called with that very norm output. Every other call is the built-in join.

Numerics: bitwise. The kernel body is the built-in one (same block, warps, loads,
roundings and reduction), and the extra output is an exact widening of the BF16 value
the built-in stores.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.models import modeling_musespark_open as _builtin
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual
from tensorrt_llm.logger import logger

#: Rows at or below which the built-in join may take its cluster boundary kernel instead
#: of the Triton one; a sequence-parallel shard always has more.
_CLUSTER_BOUNDARY_MAX_ROWS = 32


@triton.jit
def _join_with_router_input_kernel(
    out_ptr,  # [T, HIDDEN] fp32, the new residual stream
    residual_ptr,  # [T, HIDDEN] fp32, the old one
    branch_ptr,  # [T, HIDDEN] branch output, activation dtype
    gain_ptr,  # [HIDDEN] (1 + w) when HAS_GAIN, else unread
    alpha_ptr,  # [HIDDEN] fp32
    beta_ptr,  # [HIDDEN] fp32
    eps,
    next_out_ptr,  # [T, HIDDEN] activation dtype, the next norm's output
    next_weight_ptr,
    next_eps,
    router_ptr,  # [T, HIDDEN] fp32, the next norm's output widened back from the activation dtype
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_GAIN: tl.constexpr,
    NORM_FP32_RESIDUAL: tl.constexpr,
):
    """The built-in ``_gated_residual_after_rmsnorm_kernel`` with ``FUSE_NEXT_NORM`` set, plus one store.

    Statement for statement the built-in body up to the next norm's store; that store
    rounds to the activation dtype explicitly (round to nearest even, the conversion the
    built-in store applies) and the rounded value is also stored widened to FP32.
    """
    token = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < HIDDEN
    base = token * HIDDEN + offs

    branch = tl.load(branch_ptr + base, mask=mask, other=0.0)
    scaled = branch.to(tl.float32)
    if HAS_GAIN:
        gain = tl.load(gain_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        scaled = (scaled * gain).to(branch.dtype).to(tl.float32)
    variance = tl.sum(scaled * scaled, axis=0) / HIDDEN
    normed = (scaled * tl.rsqrt(variance + eps)).to(branch.dtype).to(tl.float32)

    residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    alpha = tl.load(alpha_ptr + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    updated = alpha * residual + beta * normed
    tl.store(out_ptr + base, updated, mask=mask)
    if NORM_FP32_RESIDUAL:
        next_input = updated
    else:
        next_input = updated.to(branch.dtype).to(tl.float32)
    next_variance = tl.sum(next_input * next_input, axis=0) / HIDDEN
    next_weight = tl.load(next_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    next_output = next_input * tl.rsqrt(next_variance + next_eps)
    next_output = next_output * (next_weight + 1.0)
    rounded = next_output.to(next_out_ptr.dtype.element_ty)
    tl.store(next_out_ptr + base, rounded, mask=mask)
    tl.store(router_ptr + base, rounded.to(tl.float32), mask=mask)


def join_with_router_input(
    join: MuseSparkOpenGatedResidual,
    residual: torch.Tensor,
    branch: torch.Tensor,
    eps: float,
    next_norm,
    gain: Optional[torch.Tensor] = None,
    out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(updated, next_input, next_input.float())`` from one launch, or None.

    ``out``: contiguous ``(updated, next_input, router_input)`` buffers of the call's
    shape to write instead of fresh ones (row slices of whole-shard buffers; the kernel
    is one program per row, so a slice's rows are the whole-shard call's bit for bit).

    None wherever the built-in ``_add_after_rmsnorm_and_norm`` would not launch its
    Triton kernel with this call's operands (its unfused or cluster-boundary arms); the
    caller then runs the built-in join.
    """
    if not (
        join._fused_enabled
        and residual.is_cuda
        and residual.dtype == torch.float32
        and branch.dtype == torch.bfloat16
        and residual.ndim == 2
        and residual.shape == branch.shape
        and residual.shape[0] > _CLUSTER_BOUNDARY_MAX_ROWS
        and residual.is_contiguous()
        and branch.is_contiguous()
        and next_norm.use_gemma
        and next_norm.weight.dtype == branch.dtype
        and next_norm.weight.is_contiguous()
        and (gain is None or (gain.dtype == branch.dtype and gain.is_contiguous()))
        and not torch.compiler.is_compiling()
    ):
        return None
    alpha, beta = join.coefficients()
    if out is None:
        updated = torch.empty_like(residual)
        next_input = torch.empty_like(branch)
        router_input = torch.empty_like(residual)
    else:
        updated, next_input, router_input = out
        if not (
            updated.shape == residual.shape
            and router_input.shape == residual.shape
            and next_input.shape == branch.shape
            and updated.is_contiguous()
            and next_input.is_contiguous()
            and router_input.is_contiguous()
        ):
            return None
    hidden_size = residual.shape[-1]
    _join_with_router_input_kernel[(residual.shape[0],)](
        updated,
        residual,
        branch,
        gain if gain is not None else alpha,
        alpha,
        beta,
        eps,
        next_input,
        next_norm.weight,
        next_norm.variance_epsilon,
        router_input,
        HIDDEN=hidden_size,
        BLOCK_H=triton.next_power_of_2(hidden_size),
        HAS_GAIN=gain is not None,
        NORM_FP32_RESIDUAL=join._preqkv_norm_fp32,
        num_warps=_builtin._RESIDUAL_NUM_WARPS,
    )
    return updated, next_input, router_input


def tail_pipeline(layer, residual: torch.Tensor, split) -> Tuple[torch.Tensor, torch.Tensor]:
    """The attention-side join and the MoE input preparation, chunk by chunk (``MUSESPARK_OPEN_SP_TAIL_PIPELINE``).

    ``split`` is the overlapped output projection's ``SplitProjection``: chunk ``c``
    holds rows ``[lo, hi)`` of this rank's shard, reduced on the projection stream. For
    each chunk in turn, after its reduce-scatter's event, the main stream runs the join
    (``join_with_router_input`` into row slices of whole-shard buffers) and the pooler
    GEMM + pre-expert NVFP4 quantization on its BF16 rows, while the later chunks are
    still being reduced. The NVFP4 payload and scales are concatenated and handed to the
    layer's MoE (``moe.sp_pre``) with the join's FP32 rows (``moe.sp_router_input``);
    its sequence-parallel arm runs the router GEMM on the whole shard, routes and gathers
    as before.

    Numerics: bitwise. The join and the quantization are per row; the pooler GEMM (BF16,
    8192 -> 4096) is row-split invariant on sm103 (raw bits equal for 1024/1024, 512/1536,
    1023/1025 splits of a 2048-row shard);
    the router GEMM (FP32/TF32) is not, so it is not split.
    """
    from tensorrt_llm._torch.utils import Fp4QuantizedTensor

    moe = layer.feed_forward
    join = layer.add_attn
    eps = layer.post_attn_norm.variance_epsilon
    next_norm = layer.pre_ffn_norm
    main = torch.cuda.current_stream()
    hidden = residual.shape[1]
    updated = torch.empty_like(residual)
    next_input = torch.empty((residual.shape[0], hidden), dtype=split.results[0].dtype, device=residual.device)
    router_input = torch.empty_like(residual)
    payload, scales = [], []
    for (lo, hi), reduced, event in split.chunks():
        main.wait_event(event)
        rows = slice(lo, hi)
        joined = join_with_router_input(
            join, residual[rows], reduced, eps, next_norm, None,
            out=(updated[rows], next_input[rows], router_input[rows]),
        )
        if joined is None:
            raise RuntimeError(
                "[musespark_open] MUSESPARK_OPEN_SP_TAIL_PIPELINE: the attention-side join of a "
                "chunk would not run its fused kernel"
            )
        chunk_in = next_input[rows]
        prepared = moe.prepare_expert_input(moe.pooler(chunk_in))
        if isinstance(prepared, Fp4QuantizedTensor):
            if prepared.is_sf_swizzled:
                raise RuntimeError(
                    "[musespark_open] MUSESPARK_OPEN_SP_TAIL_PIPELINE: the pre-expert quantization of "
                    "a chunk returned swizzled scales"
                )
            payload.append(prepared.fp4_tensor)
            scales.append(prepared.scaling_factor)
        else:
            # BF16 expert layers (and any layer the fused quantization declines): the
            # pre-expert norm output, per row, gathered in BF16 by the MoE as before.
            payload.append(prepared)
            scales.append(None)
    if any(sc is None for sc in scales) and not all(sc is None for sc in scales):
        raise RuntimeError(
            "[musespark_open] MUSESPARK_OPEN_SP_TAIL_PIPELINE: the chunks of one shard were "
            "prepared in different formats"
        )
    split.release(main)
    # The router GEMM is not row-split invariant (FP32/TF32 accumulation differs between
    # row counts on sm103), so it runs once on the whole shard, in the MoE as before, from
    # the join's FP32 rows.
    moe.sp_router_input = (next_input, router_input)
    cat = lambda parts: torch.cat(parts, 0) if len(parts) > 1 else parts[0]  # noqa: E731
    if scales[0] is None:
        prepared = cat(payload)
    else:
        prepared = Fp4QuantizedTensor(cat(payload), cat(scales), is_sf_swizzled=False)
    moe.sp_pre = (next_input, None, prepared)
    logger.info_once(
        f"[musespark_open] MUSESPARK_OPEN_SP_TAIL_PIPELINE: attention-side join and pooled NVFP4 "
        f"expert input prepared in {len(payload)} chunks behind the output projection's reduce-scatters",
        key="musespark_open.sp_tail_pipeline",
    )
    return updated, next_input


class SequenceParallelStep:
    """Whether a sequence-parallel forward is running (set by ``forward.ContextSequenceParallel``)."""

    def __init__(self) -> None:
        self.active = False


class SequenceParallelAttnJoin:
    """``MuseSparkOpenGatedResidual`` mixin for a layer's ``add_attn``.

    While ``_sp_step.active``, a join that the built-in would run as its Triton kernel
    runs ``join_with_router_input`` and hands ``(next_input, its FP32 widening)`` to the
    layer's MoE (``sp_router_input``). Any other call is the built-in join.
    """

    _sp_step: Optional[SequenceParallelStep] = None
    _sp_moe = None

    def _add_after_rmsnorm_and_norm(self, residual, branch, eps, next_norm, gain=None):
        step, moe = self._sp_step, self._sp_moe
        if step is not None and step.active and moe is not None:
            joined = join_with_router_input(self, residual, branch, eps, next_norm, gain)
            if joined is not None:
                updated, next_input, router_input = joined
                moe.sp_router_input = (next_input, router_input)
                return updated, next_input
        return super()._add_after_rmsnorm_and_norm(residual, branch, eps, next_norm, gain)


_CLASSES = {}


def _with_mixin(base: type) -> type:
    """``base`` with ``SequenceParallelAttnJoin`` in front, one class per base."""
    if issubclass(base, SequenceParallelAttnJoin):
        return base
    if base not in _CLASSES:
        _CLASSES[base] = type(
            "SequenceParallelAttnJoinOver" + base.__name__,
            (SequenceParallelAttnJoin, base),
            {"__module__": __name__},
        )
    return _CLASSES[base]


def install(model, step: SequenceParallelStep) -> int:
    """Re-class every layer's ``add_attn`` and point it at ``step`` and the layer's MoE; returns how many.

    Raises:
        ValueError: if a layer's ``add_attn`` is not a built-in ``MuseSparkOpenGatedResidual``.
    """
    joins = [layer.add_attn for layer in model.layers]
    wrong = [
        layer.layer_idx
        for layer, join in zip(model.layers, joins)
        if not isinstance(join, MuseSparkOpenGatedResidual)
    ]
    if wrong:
        raise ValueError(
            "[musespark_open] context_sequence_parallel re-classes the built-in gated residual "
            f"of every layer; layers {wrong} carry another class"
        )
    changed = 0
    for layer, join in zip(model.layers, joins):
        mixed = _with_mixin(type(join))
        if type(join) is not mixed:
            join.__class__ = mixed
            changed += 1
        # Plain attributes, not registered submodules: ``nn.Module.__setattr__`` would file
        # the MoE under the join's children, where attribute lookup finds the class's
        # ``_sp_moe = None`` first, and the module tree would hold the MoE twice.
        object.__setattr__(join, "_sp_step", step)
        object.__setattr__(join, "_sp_moe", layer.feed_forward)
        if os.getenv("MUSESPARK_OPEN_SP_TAIL_PIPELINE", "0") == "1":
            layer._sp_tail_pipeline = tail_pipeline
    return changed
