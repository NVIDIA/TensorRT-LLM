# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE of a sequence-parallel context step: the expert input quantized on the row shard, then gathered.

The built-in MoE's sequence-parallel arm all-gathers the BF16 pooled rows with the FP32
router logits, then runs the pre-expert RMSNorm fused with the NVFP4 quantization
(``prepare_expert_input``) on every row of the step. Both are per row: the norm reduces
over one row, and every 16-element block gets its own E4M3 scale under the layer's
static global scale. So ``ContextSequenceParallelMoE`` runs them on the rank's own rows
and all-gathers the NVFP4 payload and its block scales instead of the BF16 rows: a
quarter of the norm and quantization work, and 4.5 bits per gathered element instead of
16. The experts receive the bytes the built-in arm hands them, so the step's result is
the built-in arm's bit for bit.

When the built-in arm would not quantize (``prepare_expert_input`` returns the BF16 norm
output, or the step is above the experts' ``moe_max_num_tokens``), the call is the
built-in arm's.

Everything after the gather -- the unfinalized expert call, the per-expert post-norm
finalize, the BF16 cast, the reduce-scatter of the partial sums and the unpool on the
shard -- is the built-in arm's sequence for a step above the layer's fused finalize
all-reduce rows (``_finalize_ar_max_tokens``). The mixin always finalizes unfused, so its
partial sums are reduced once, by the reduce-scatter, at any row count. A call it hands to
the class behind runs the built-in arm, which takes the fused finalize all-reduce up to
``_finalize_ar_max_tokens`` rows and then reduce-scatters as well; ``install`` therefore
refuses a layer whose fused finalize serves the path's smallest step.

Where the built-in finalize runs its sparse-slot kernel, that same kernel stores BF16
instead of the FP32 output the built-in arm then casts (``finalize_to``): the kernel's
store rounds each FP32 sum to BF16 to nearest even, as the cast does, so the partials
are the built-in arm's bit for bit and the FP32 partials are never written or read back.
The built-in finalize still decides which kernel runs.

Routing on the shard (``shard_routing``). The built-in arm gathers every rank's FP32
router logits (1 KiB per row) and then routes every row of the step on every rank. The
routing is per row, so where the built-in routing method would run its fused top-8
kernel on the gathered logits, each rank runs that kernel on its own rows -- launched as
the call over all the step's rows would be (``TORCH_ORDER`` and warps chosen from the
step's row count) -- and the top-8 ids and FP32 weights (64 B per row) are gathered with
the NVFP4 payload instead of the logits. The routing method's one-slot memo then holds
the gathered ids and weights, keyed on a placeholder that stands for the logits (no
values); the experts and the finalize read the memo, so both see the ids and weights
the built-in arm computes, bit for bit.
Every consumer of a shard-routed call's ``router_logits`` must read through that memo
(``routing_method.apply``): the placeholder's values are uninitialized.

The router's FP32 input. When the layer's attention-side residual join already wrote
the FP32 widening of this call's input (``router_input.SequenceParallelAttnJoin``,
``sp_router_input``), the router GEMM reads it instead of a fresh BF16 -> FP32 copy:
the same values, the same GEMM.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton

from tensorrt_llm._torch.distributed import allgather, reducescatter
from tensorrt_llm._torch.models import modeling_musespark_open as _builtin
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenMoE,
    MuseSparkOpenRoutingMethod,
    _pad_sequence_parallel_tokens,
    _version_or_none,
)
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger

_CLASSES: Dict[Tuple[type, type], type] = {}

#: The routed experts and the top-k the fused routing kernel serves.
_EXPERTS, _TOP_K = 256, 8


def shard_routing(
    routing, logits: torch.Tensor, rows: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The fused routing kernel's ``(selected, weights)`` for the shard's ``logits``, or None.

    Launched with the arguments ``MuseSparkOpenRoutingMethod.apply`` chooses for a call
    over all ``rows`` rows of the step (``TORCH_ORDER`` above 128 rows when the method
    takes it, one warp from 512 rows), one program per row, so every row's ids and
    weights are that call's. None where ``apply`` on the step's logits would not run the
    fused kernel, or where its memo is off (the experts then could not read the
    gathered routing).
    """
    bias = getattr(routing, "e_score_correction_bias", None)
    if not (
        isinstance(routing, MuseSparkOpenRoutingMethod)
        and routing._memo_enabled
        and rows > 0
        and (rows <= 128 or routing._full_routing_all_batches)
        and routing.top_k == _TOP_K
        and routing.output_dtype == torch.float32
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.shape[0] > 0
        and logits.shape[1] == _EXPERTS
        and logits.is_contiguous()
        and bias is not None
        and bias.is_cuda
        and bias.dtype == torch.float32
        and bias.shape == (_EXPERTS,)
        and bias.is_contiguous()
        and not torch.compiler.is_compiling()
    ):
        return None
    shard = logits.shape[0]
    selected = torch.empty((shard, _TOP_K), device=logits.device, dtype=torch.int32)
    weights = torch.empty_like(selected, dtype=torch.float32)
    _builtin._full_routing_kernel[(shard,)](
        logits,
        bias,
        selected,
        weights,
        routing.renorm_eps,
        EXPERTS=_EXPERTS,
        TORCH_ORDER=routing._full_routing_torch_order and rows > 128,
        num_warps=1 if rows >= 512 else 4,
    )
    return selected, weights


def routed_placeholder(
    routing, selected: torch.Tensor, weights: torch.Tensor, rows: int
) -> torch.Tensor:
    """A ``[rows, 256]`` FP32 stand-in for the step's logits, memoized to ``(selected, weights)``.

    The stand-in is a zero-stride view of one element: it carries no values and is never
    read, because every routing call on it returns the memo.
    """
    placeholder = selected.new_empty((1, 1), dtype=torch.float32).expand(rows, _EXPERTS)
    routing._routing_memo = (placeholder, _version_or_none(placeholder), selected, weights)
    return placeholder


def _fp32_router_input_takes(moe, hidden_states: torch.Tensor) -> bool:
    """Whether the built-in ``router_logits`` on ``hidden_states`` is ``router(hidden.float()) * m``.

    That is its last arm: more than 128 rows (the small-batch kernels end there), no
    BF16x3 router and no armed ``router_gemm`` call.
    """
    call = getattr(moe, "_router_gemm_call", None)
    weight = getattr(moe.router, "weight", None)
    return (
        hidden_states.ndim == 2
        and hidden_states.shape[0] > 128
        and not getattr(moe, "_bf16x3_router_enabled", False)
        and (call is None or not call.is_armed)
        and weight is not None
        and weight.dtype == torch.float32
    )


def _rows_of(tensor, rows: int, shape_tail: Tuple[int, ...]):
    """The first ``rows`` rows of a gathered ``[*, width]`` tensor, in the shard's own layout."""
    return tensor[:rows].reshape((rows, *shape_tail))


class _SparseFinalizeTo:
    """The built-in sparse-slot finalize launch, storing ``dtype`` instead of FP32.

    Stands in for the built-in ``_sparse_post_expert_norm_combine`` (same signature,
    same kernel object, same launch arguments and warps) and keeps the tensor it returns.
    """

    def __init__(self, dtype) -> None:
        self.dtype = dtype
        self.out: Optional[torch.Tensor] = None

    def __call__(
        self,
        gemm2_output: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        routing_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        *,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
    ) -> torch.Tensor:
        out = torch.empty((num_tokens, hidden_size), dtype=self.dtype, device=gemm2_output.device)
        _builtin._post_expert_norm_combine_sparse_kernel[(num_tokens,)](
            out,
            gemm2_output,
            expanded_idx_to_permuted_idx.reshape(-1),
            routing_weights.reshape(-1),
            norm_weight,
            gemm2_output.stride(0),
            gemm2_output.shape[0],
            eps,
            TOP_K=top_k,
            HIDDEN=hidden_size,
            BLOCK_H=triton.next_power_of_2(hidden_size),
            num_warps=_builtin._SPARSE_FINALIZE_NUM_WARPS,
        )
        self.out = out
        return out


def finalize_to(
    moe,
    gemm2_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    routing_weights: torch.Tensor,
    rows: int,
    dtype,
) -> torch.Tensor:
    """``moe.finalize(...).to(dtype)``, with the sparse-slot kernel storing BF16 itself.

    ``moe.finalize`` runs as it is, with the built-in sparse-slot launch swapped for this
    call only (``_SparseFinalizeTo``), so the built-in finalize's own dispatch decides
    whether that kernel runs. Triton's store converts the kernel's FP32 sum to BF16 with
    round-to-nearest-even, the conversion ``Tensor.to(torch.bfloat16)`` applies, so when
    ``finalize`` returns the launch's tensor itself the result is the cast of its FP32
    output bit for bit. Any other kernel returns FP32, cast here as before. If
    ``finalize`` returned something computed from the BF16 sums, the call is repeated as
    built (FP32, then the cast) and the swap is reported once.
    """
    if dtype != torch.bfloat16:
        return moe.finalize(gemm2_output, expanded_idx_to_permuted_idx, routing_weights, rows).to(
            dtype
        )
    launch = _SparseFinalizeTo(dtype)
    builtin_launch = _builtin._sparse_post_expert_norm_combine
    _builtin._sparse_post_expert_norm_combine = launch
    try:
        combined = moe.finalize(gemm2_output, expanded_idx_to_permuted_idx, routing_weights, rows)
    finally:
        _builtin._sparse_post_expert_norm_combine = builtin_launch
    if launch.out is None or combined is launch.out:
        return combined.to(dtype)
    logger.warning_once(
        "[musespark_open] context_sequence_parallel: the MoE finalize computed further on the "
        "sparse-slot kernel's output; finalizing in FP32 as built",
        key="musespark_open.context_sp.finalize_to.rerun",
    )
    return moe.finalize(gemm2_output, expanded_idx_to_permuted_idx, routing_weights, rows).to(dtype)


class ContextSequenceParallelMoE:
    """MoE mixin: a sequence-parallel call gathers its expert input as NVFP4.

    Acts on calls with ``sp_num_tokens`` only; every other call is the class behind it.
    """

    #: ``(norm output, its FP32 widening)`` left by the layer's attention-side join for
    #: the next sequence-parallel call (``router_input``); consumed or dropped by it.
    sp_router_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    #: Whether a sequence-parallel call routes the shard's rows and gathers their ids and
    #: weights (``shard_routing``); otherwise it gathers the router logits (``install``).
    _sp_shard_routing: bool = True

    def _sp_router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The router logits of the shard's rows, from the join's FP32 rows when it left them."""
        handed = self.sp_router_input
        self.sp_router_input = None
        if (
            handed is not None
            and handed[0] is hidden_states
            and _fp32_router_input_takes(self, hidden_states)
        ):
            return self.router(handed[1]) * self.metap_multiplier
        return self.router_logits(hidden_states)

    def forward(self, hidden_states, attn_metadata, *args, **kwargs):
        rows: Optional[int] = kwargs.get("sp_num_tokens")
        experts = self.experts
        if (
            rows is None
            or args
            or getattr(experts, "comm", None) is not None
            or rows > getattr(experts, "moe_max_num_tokens", rows)
        ):
            self.sp_router_input = None
            return super().forward(hidden_states, attn_metadata, *args, **kwargs)
        mapping = self.mapping
        shard = hidden_states.shape[0]
        routing_method = experts.routing_method
        with self.nvtx("router"):
            router_logits = self._sp_router_logits(hidden_states)
        with self.nvtx("pool"):
            prepared = self.prepare_expert_input(self.pooler(hidden_states))
            if isinstance(prepared, Fp4QuantizedTensor):
                fp4, scale = prepared.fp4_tensor, prepared.scaling_factor
                if prepared.is_sf_swizzled or fp4.shape[0] != shard or scale.numel() % shard:
                    raise RuntimeError(
                        "context_sequence_parallel: the pre-expert quantization returned a "
                        f"layout that is not per row ({tuple(fp4.shape)} payload, "
                        f"{tuple(scale.shape)} scales, swizzled {prepared.is_sf_swizzled}) "
                        f"for {shard} rows"
                    )
                routed = (
                    shard_routing(routing_method, router_logits, rows)
                    if self._sp_shard_routing
                    else None
                )
                head = [router_logits] if routed is None else list(routed)
                gathered = allgather(
                    [*head, fp4.reshape(shard, -1), scale.reshape(shard, -1)], mapping, dim=0
                )
                fp4_rows, scale_rows = gathered[-2:]
                scale_tail = tuple(scale.shape[1:]) if scale.dim() > 1 else ()
                pooled = Fp4QuantizedTensor(
                    _rows_of(fp4_rows, rows, tuple(fp4.shape[1:])),
                    scale_rows[:rows].reshape((rows, *scale_tail) if scale_tail else (-1,)),
                    is_sf_swizzled=False,
                )
                if routed is None:
                    router_logits = gathered[0][:rows]
                else:
                    router_logits = routed_placeholder(
                        routing_method, gathered[0][:rows], gathered[1][:rows], rows
                    )
            else:
                logits, pooled = allgather([router_logits, prepared], mapping, dim=0)
                pooled = pooled[:rows]
                router_logits = logits[:rows]
        with self.nvtx("experts"):
            outputs = experts(
                pooled,
                router_logits,
                do_finalize=False,
                output_dtype=hidden_states.dtype,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens
                if attn_metadata is not None
                else None,
                use_dp_padding=False,
            )
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
            raise RuntimeError(
                "MuseSparkOpen MoE expects the unfinalized (gemm2_output, expert_weights, "
                f"expanded_idx_to_permuted_idx) triple, got {type(outputs).__name__}"
            )
        gemm2_output, _, expanded_idx_to_permuted_idx = outputs
        with self.nvtx("router"):
            _, token_final_scales = routing_method.apply(router_logits)
        routing_method.clear_routing_memo()
        with self.nvtx("finalize"):
            combined = finalize_to(
                self,
                gemm2_output,
                expanded_idx_to_permuted_idx,
                token_final_scales,
                rows,
                hidden_states.dtype,
            )
        with self.nvtx("all_reduce"):
            combined = reducescatter(
                _pad_sequence_parallel_tokens(combined, mapping.tp_size), mapping, dim=0
            )
        with self.nvtx("unpool"):
            return self._replicated_projection(
                self.unpooler, combined, self._unpool_gather_workspace
            )


def _with_mixin(base: type) -> type:
    """``base`` with ``ContextSequenceParallelMoE`` in front, one class per base."""
    if issubclass(base, ContextSequenceParallelMoE):
        return base
    key = (ContextSequenceParallelMoE, base)
    if key not in _CLASSES:
        name = "ContextSequenceParallelMoEOver" + base.__name__
        _CLASSES[key] = type(name, (ContextSequenceParallelMoE, base), {"__module__": __name__})
    return _CLASSES[key]


def install(model, min_rows: int, shard_routing: bool = True) -> int:
    """Re-class every layer's MoE with the mixin in front of its class; returns how many changed.

    ``min_rows`` is the fewest rows of a sequence-parallel step. ``shard_routing`` has a
    sequence-parallel call route the shard's rows and gather their top-8 ids and weights
    (``shard_routing``); without it the call gathers the router logits and routes every
    row of the step, as the built-in arm does. Runs after every
    construct-time re-class of the MoE (``PrefillTarget`` is built last in
    ``MuseSparkOpenTepModelingV2.adopt``), so no later fixed re-class drops the mixin.

    Raises:
        ValueError: if a layer's MoE is not a built-in ``MuseSparkOpenMoE``, or its fused
            finalize all-reduce serves ``min_rows`` rows (a sequence-parallel call the
            mixin hands to the class behind would then reduce its partial sums twice).
    """
    changed = 0
    for layer in model.layers:
        moe = layer.feed_forward
        if not isinstance(moe, MuseSparkOpenMoE):
            raise ValueError(
                "[musespark_open] context_sequence_parallel re-classes the built-in MoE, got "
                f"{type(moe).__name__} in layer {layer.layer_idx}"
            )
        fused_rows = getattr(moe, "_finalize_ar_max_tokens", 128)
        if fused_rows >= min_rows:
            raise ValueError(
                "[musespark_open] context_sequence_parallel needs every layer's fused finalize "
                f"all-reduce below {min_rows} rows, layer {layer.layer_idx} serves {fused_rows}"
            )
        mixed = _with_mixin(type(moe))
        if type(moe) is not mixed:
            moe.__class__ = mixed
            changed += 1
        moe._sp_shard_routing = bool(shard_routing)
    return changed
