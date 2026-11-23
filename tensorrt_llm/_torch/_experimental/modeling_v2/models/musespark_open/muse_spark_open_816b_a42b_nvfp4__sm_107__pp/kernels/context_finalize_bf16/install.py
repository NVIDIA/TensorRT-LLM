# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``context_finalize_bf16``: a stage's context MoE finalize stores BF16 from its sparse-slot kernel.

The TP1 wiring of the finalize of the TEP target's ``ContextSequenceParallelMoE``. The
built-in MoE call of a context step finalizes in FP32 (the per-expert post-norm combine;
with ``MUSESPARK_OPEN_LARGE_FINALIZE_SPARSE`` the sparse-slot Triton kernel) and then casts
the ``[rows, 4096]`` result to BF16 for the unpool projection: an FP32 write, a read back
and a BF16 write. Each stage layer's MoE is re-classed with ``stage_moe.StageContextMoE``;
on a stage context call of ``stage_moe.MIN_ROWS`` rows or more it finalizes with the TEP
target's ``finalize_to``: the built-in finalize's own dispatch, with its sparse-slot launch
storing BF16 itself. Where the built-in finalize runs another kernel the result is cast as
before. Every other call is the built-in MoE call.

Numerics: bitwise; see ``stage_moe``.

Construction-time; agreed over the stages (``dispatch.agreed_install``).
"""

from __future__ import annotations

from tensorrt_llm.logger import logger

from ... import dispatch as _dispatch
from .. import stage_moe as _stage_moe


def install(text_model, mapping, step: "_stage_moe.StageStep") -> int:
    """Re-class every stage layer's MoE for the BF16 finalize; returns the layers.

    Raises ``ValueError`` on every stage alike when some stage refuses (a layer whose MoE is
    not the built-in one).
    """
    layers = _dispatch.owned_layers(text_model)
    _dispatch.agreed_install(mapping, lambda: _stage_moe.check(layers), (ValueError,))
    _stage_moe.adopt(layers, step, finalize_bf16=True)
    sparse = all(getattr(layer.feed_forward, "_large_finalize_sparse", False) for layer in layers)
    logger.info(
        f"{_dispatch.TAG} context_finalize_bf16: the MoE finalize of the {len(layers)} layers of "
        f"stage {text_model._pp_stage} stores BF16 on context calls of {_stage_moe.MIN_ROWS} "
        "rows or more"
        + (
            ""
            if sparse
            else " (no effect: MUSESPARK_OPEN_LARGE_FINALIZE_SPARSE is off, so the finalize "
            "runs the dense kernel and is cast as before)"
        )
    )
    return len(layers)
