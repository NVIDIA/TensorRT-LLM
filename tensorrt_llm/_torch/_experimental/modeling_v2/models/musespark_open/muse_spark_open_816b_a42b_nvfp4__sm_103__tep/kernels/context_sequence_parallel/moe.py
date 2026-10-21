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
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton

from tensorrt_llm._torch.distributed import allgather, reducescatter
from tensorrt_llm._torch.models import modeling_musespark_open as _builtin
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenMoE,
    _pad_sequence_parallel_tokens,
)
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger

_CLASSES: Dict[Tuple[type, type], type] = {}


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

    def forward(self, hidden_states, attn_metadata, *args, **kwargs):
        rows: Optional[int] = kwargs.get("sp_num_tokens")
        experts = self.experts
        if (
            rows is None
            or args
            or getattr(experts, "comm", None) is not None
            or rows > getattr(experts, "moe_max_num_tokens", rows)
        ):
            return super().forward(hidden_states, attn_metadata, *args, **kwargs)
        mapping = self.mapping
        shard = hidden_states.shape[0]
        with self.nvtx("router"):
            router_logits = self.router_logits(hidden_states)
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
                logits, fp4_rows, scale_rows = allgather(
                    [router_logits, fp4.reshape(shard, -1), scale.reshape(shard, -1)],
                    mapping,
                    dim=0,
                )
                scale_tail = tuple(scale.shape[1:]) if scale.dim() > 1 else ()
                pooled = Fp4QuantizedTensor(
                    _rows_of(fp4_rows, rows, tuple(fp4.shape[1:])),
                    scale_rows[:rows].reshape((rows, *scale_tail) if scale_tail else (-1,)),
                    is_sf_swizzled=False,
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
        routing_method = experts.routing_method
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


def install(model, min_rows: int) -> int:
    """Re-class every layer's MoE with the mixin in front of its class; returns how many changed.

    ``min_rows`` is the fewest rows of a sequence-parallel step. Runs after every
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
    return changed
