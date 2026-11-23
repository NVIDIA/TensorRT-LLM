# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``context_router_input``: a stage's attention-side join also writes the router's FP32 input.

The TP1 wiring of the TEP target's ``context_sp_router_input``. The built-in MoE router of a
call above 128 rows reads the pre-FFN norm output (BF16, ``[rows, 8192]``) as FP32, so every
context layer runs a standalone BF16 -> FP32 copy of it (``router(hidden.to(torch.float32))``)
in front of the router GEMM. The attention-side residual join that writes that norm output
(one Triton kernel) holds every value in registers when it rounds it to BF16.

Each stage layer's ``add_attn`` is re-classed with ``StageAttnJoin``: while a stage context
forward runs (``stage_moe.StageStep.active``) and the layer's MoE will read an FP32 router
input for the call's rows (``stage_moe.MIN_ROWS`` rows or more, the built-in FP32 router
arm), the join is the TEP target's ``join_with_router_input`` -- the built-in kernel with one
more store, the BF16-rounded norm output widened back to FP32 -- and hands the pair to the
layer's MoE (``StageContextMoE.stage_router_input``), whose router GEMM reads the FP32 rows
when it is called with that very norm output. Every other call is the built-in join and the
built-in router.

Numerics: bitwise; see ``stage_moe``.

Construction-time; agreed over the stages (``dispatch.agreed_install``): every stage checks
its layers, the outcome is agreed, and only then are the layers re-classed.
"""

from __future__ import annotations

from typing import Dict

from tensorrt_llm.logger import logger

from ... import dispatch as _dispatch
from ..._tep import module as _tep_module
from .. import stage_moe as _stage_moe


class StageAttnJoin:
    """``MuseSparkOpenGatedResidual`` mixin for a stage layer's ``add_attn``.

    While ``_stage_step.active``, a join whose layer MoE takes the call
    (``StageContextMoE.stage_takes``) and reads an FP32 router input for it runs the TEP
    target's ``join_with_router_input`` and hands ``(next_input, its FP32 widening)`` to the
    MoE. Any other call, and a call the TEP kernel declines (``None``), is the built-in join.
    """

    _stage_step = None
    _stage_moe = None

    def _add_after_rmsnorm_and_norm(self, residual, branch, eps, next_norm, gain=None):
        step, moe = self._stage_step, self._stage_moe
        if (
            step is not None
            and step.active
            and moe is not None
            and getattr(moe, "_stage_router_input", False)
            and moe.stage_takes(branch)
            and _stage_moe.fp32_router_input_takes(moe, branch)
        ):
            router_input = _tep_module("kernels.context_sequence_parallel.router_input")
            joined = router_input.join_with_router_input(
                self, residual, branch, eps, next_norm, gain
            )
            if joined is not None:
                updated, next_input, fp32_rows = joined
                moe.stage_router_input = (next_input, fp32_rows)
                return updated, next_input
        return super()._add_after_rmsnorm_and_norm(residual, branch, eps, next_norm, gain)


_CLASSES: Dict[type, type] = {}


def _with_mixin(base: type) -> type:
    """``base`` with ``StageAttnJoin`` in front, one class per base."""
    if issubclass(base, StageAttnJoin):
        return base
    if base not in _CLASSES:
        _CLASSES[base] = type(
            "StageAttnJoinOver" + base.__name__, (StageAttnJoin, base), {"__module__": __name__}
        )
    return _CLASSES[base]


def install(text_model, mapping, step: "_stage_moe.StageStep") -> int:
    """Re-class every stage layer's ``add_attn`` and MoE for the router input; returns the layers.

    Raises ``ValueError`` on every stage alike when some stage refuses (a layer whose
    ``add_attn`` or MoE is not the built-in one).
    """
    layers = _dispatch.owned_layers(text_model)

    def check() -> None:
        from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual

        wrong = [
            layer.layer_idx
            for layer in layers
            if not isinstance(layer.add_attn, MuseSparkOpenGatedResidual)
        ]
        if wrong:
            raise ValueError(
                f"{_dispatch.TAG} context_router_input re-classes the built-in gated residual; "
                f"layers {wrong} carry another class"
            )
        _stage_moe.check(layers)

    _dispatch.agreed_install(mapping, check, (ValueError,))
    _stage_moe.adopt(layers, step, router_input=True)
    for layer in layers:
        join = layer.add_attn
        mixed = _with_mixin(type(join))
        if type(join) is not mixed:
            join.__class__ = mixed
        # Plain attributes, not registered submodules: ``nn.Module.__setattr__`` would file the
        # MoE under the join's children (the module tree would hold it twice).
        object.__setattr__(join, "_stage_step", step)
        object.__setattr__(join, "_stage_moe", layer.feed_forward)
    logger.info(
        f"{_dispatch.TAG} context_router_input: the attention-side join of the {len(layers)} "
        f"layers of stage {text_model._pp_stage} writes the router's FP32 input on context "
        f"calls of {_stage_moe.MIN_ROWS} rows or more (no BF16 -> FP32 copy before the router)"
    )
    return len(layers)
