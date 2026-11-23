# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stage's context MoE call on the TEP target's sequence-parallel MoE kernels, without its collectives.

On a TEP4 context step the TEP target's ``ContextSequenceParallelMoE`` (``kernels/
context_sequence_parallel/moe.py``) and ``SequenceParallelAttnJoin`` (``…/router_input.py``)
take two per-row costs off the built-in MoE call:

* the router's FP32 input: the attention-side residual join, which holds the pre-FFN norm
  output in registers when it rounds it to BF16, also stores that value widened to FP32, and
  the router GEMM reads it in place of the built-in ``hidden_states.to(torch.float32)`` copy
  (``context_router_input``);
* the finalize: where the built-in finalize runs its sparse-slot kernel, that kernel stores
  BF16 itself (``finalize_to``) in place of an FP32 output and a separate BF16 cast
  (``context_finalize_bf16``).

Both act only inside the TEP target's sequence-parallel forward. A pipeline stage runs no
sequence parallelism (TP1: every row of the step is on the stage), so this module wires the
same two kernels into the stage's own MoE call: ``StageContextMoE`` re-classes a layer's MoE
and, on a call of at least ``MIN_ROWS`` rows while ``StageStep.active`` (set by
``modeling.PrefillTarget`` for its forward), runs the built-in MoE call of a TP1 layer without
an external MoE communication, statement for statement, with those two swaps. Every other
call is the class behind it (decode steps, smaller calls, a layer with a communication
strategy or a fused finalize all-reduce workspace).

Numerics: bitwise. The join kernel is the built-in one with one more store, an exact
widening of the BF16 value the built-in stores; the router GEMM then reads the same FP32
values the copy would have written. Triton's BF16 store rounds to nearest even, as
``Tensor.to(torch.bfloat16)`` does, so the finalize's BF16 output is the built-in cast of its
FP32 output bit for bit (and where the built-in finalize does not run the sparse-slot
kernel, ``finalize_to`` casts as the built-in does).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .._tep import module as _tep_module

#: The fewest rows of a call the two swaps act on: context calls (the TEP target's
#: sequence-parallel ``MIN_ROWS``). Above the built-in router's small-batch kernels (128
#: rows) and above its router / pool stream overlap (up to 512 rows), so every call taken
#: runs the built-in router as ``router(hidden.float()) * metap`` and the pool after it.
MIN_ROWS = 1024


class StageStep:
    """Whether a stage's context forward is running (``modeling.PrefillTarget`` sets it)."""

    def __init__(self) -> None:
        self.active = False


def _tep_moe():
    return _tep_module("kernels.context_sequence_parallel.moe")


def fp32_router_input_takes(moe, hidden_states: torch.Tensor) -> bool:
    """Whether ``moe``'s built-in router on ``hidden_states`` is ``router(hidden.float()) * metap``.

    The TEP target's ``_fp32_router_input_takes``: more than 128 rows, no BF16x3 router, no
    armed router GEMM, an FP32 router weight.
    """
    return bool(_tep_moe()._fp32_router_input_takes(moe, hidden_states))


class StageContextMoE:
    """``MuseSparkOpenMoE`` mixin: a stage's context call with the router input and finalize swaps.

    Acts on a call of at least ``MIN_ROWS`` rows while ``_stage_step.active``, on a TP1 layer
    without an external MoE communication or a fused finalize workspace; every other call is
    the class behind it.
    """

    #: Set by ``adopt``: the step flag and which of the two swaps are on.
    _stage_step: Optional[StageStep] = None
    _stage_router_input: bool = False
    _stage_finalize_bf16: bool = False
    #: ``(norm output, its FP32 widening)`` the attention-side join left for the next call
    #: (``context_router_input``); consumed or dropped by that call.
    stage_router_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def stage_takes(self, hidden_states: torch.Tensor) -> bool:
        """Whether a call on ``hidden_states`` (no extra arguments) runs the swapped call."""
        step = self._stage_step
        experts = self.experts
        mapping = self.mapping
        return (
            step is not None
            and step.active
            and not torch.compiler.is_compiling()
            and hidden_states.ndim == 2
            and MIN_ROWS
            <= hidden_states.shape[0]
            <= getattr(experts, "moe_max_num_tokens", hidden_states.shape[0])
            and getattr(experts, "comm", None) is None
            and getattr(self, "_finalize_ar_workspace", None) is None
            and int(mapping.tp_size) == 1
            and int(mapping.moe_tp_size) == 1
        )

    def _stage_router_logits(self, hidden_states, handed) -> Optional[torch.Tensor]:
        """The router logits from the join's FP32 rows, or None (the built-in router runs)."""
        if (
            not self._stage_router_input
            or handed is None
            or handed[0] is not hidden_states
            or not fp32_router_input_takes(self, hidden_states)
        ):
            return None
        return self.router(handed[1]) * self.metap_multiplier

    def forward(self, hidden_states, attn_metadata, *args, **kwargs):
        handed = self.stage_router_input
        self.stage_router_input = None
        if args or kwargs.get("sp_num_tokens") is not None or not self.stage_takes(hidden_states):
            return super().forward(hidden_states, attn_metadata, *args, **kwargs)
        num_tokens = hidden_states.shape[0]
        experts = self.experts
        routing_method = experts.routing_method
        # The built-in ``prepare_moe_inputs`` above its overlap rows: the router, then the pool.
        with self.nvtx("router"):
            router_logits = self._stage_router_logits(hidden_states, handed)
            if router_logits is None:
                router_logits = self.router_logits(hidden_states)
        # Release the join's FP32 rows ([rows, hidden] FP32) once the router GEMM has read them,
        # as the built-in copy and the TEP handed rows are, so they do not stay alive through the
        # experts, the finalize and the unpool.
        handed = None
        with self.nvtx("pool"):
            pooled = self.prepare_expert_input(
                self._replicated_projection(self.pooler, hidden_states, self._pool_gather_workspace)
            )
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
        gemm2_output, _kernel_weights, expanded_idx_to_permuted_idx = outputs
        with self.nvtx("router"):
            _, token_final_scales = routing_method.apply(router_logits)
        routing_method.clear_routing_memo()
        with self.nvtx("finalize"):
            if self._stage_finalize_bf16:
                combined = _tep_moe().finalize_to(
                    self,
                    gemm2_output,
                    expanded_idx_to_permuted_idx,
                    token_final_scales,
                    num_tokens,
                    hidden_states.dtype,
                )
            else:
                combined = self.finalize(
                    gemm2_output, expanded_idx_to_permuted_idx, token_final_scales, num_tokens
                ).to(hidden_states.dtype)
        # TP1: no all-reduce (the built-in's ``has_tp()`` gate).
        with self.nvtx("unpool"):
            return self._replicated_projection(
                self.unpooler, combined, self._unpool_gather_workspace
            )


_CLASSES: Dict[type, type] = {}


def _with_mixin(base: type) -> type:
    """``base`` with ``StageContextMoE`` in front, one class per base."""
    if issubclass(base, StageContextMoE):
        return base
    if base not in _CLASSES:
        _CLASSES[base] = type(
            "StageContextMoEOver" + base.__name__, (StageContextMoE, base), {"__module__": __name__}
        )
    return _CLASSES[base]


def check(layers) -> None:
    """Raise ``ValueError`` when a layer's MoE is not the built-in one. Changes nothing."""
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE

    wrong = [
        layer.layer_idx for layer in layers if not isinstance(layer.feed_forward, MuseSparkOpenMoE)
    ]
    if wrong:
        raise ValueError(f"the stage MoE swaps re-class the built-in MoE; layers {wrong} are not")


def adopt(layers, step: StageStep, **flags: bool) -> int:
    """Re-class every layer's MoE with ``StageContextMoE`` and set ``flags``; returns how many changed.

    ``flags`` are ``router_input`` / ``finalize_bf16``; a flag not given keeps its value, so
    the two levers' installers compose in either order.
    """
    unknown = sorted(set(flags) - {"router_input", "finalize_bf16"})
    if unknown:
        raise KeyError(f"not a stage MoE swap: {unknown}")
    changed = 0
    for layer in layers:
        moe = layer.feed_forward
        mixed = _with_mixin(type(moe))
        if type(moe) is not mixed:
            moe.__class__ = mixed
            changed += 1
        moe._stage_step = step
        for name, value in flags.items():
            setattr(moe, f"_stage_{name}", bool(value))
    return changed
