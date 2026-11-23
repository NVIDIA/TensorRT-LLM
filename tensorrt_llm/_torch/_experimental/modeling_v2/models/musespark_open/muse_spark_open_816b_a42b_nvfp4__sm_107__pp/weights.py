# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The weights a stage expects: its slice of the Muse Spark Open NVFP4 checkpoint, as the built-in loads it.

The stage's module tree is the built-in model's after ``MuseSparkOpenModel.__pp_init__``:
the layers of the other stages are stripped (``skip_forward``), the embedding is kept on the
first stage and the final norm and LM head on the last. The built-in loader
(``MuseSparkOpenOmniForConditionalGeneration.load_weights`` and the weight mapper's pipeline
filter) maps every checkpoint tensor of the stage's layers -- the open-format renames, the
stacked NVFP4 experts and their scales, the head-interleaved query and output-gate
projection -- and skips the others. Without speculative decoding no drafter is loaded. With
a DFlash drafter the built-in loader loads it (its own checkpoint, ``load_draft_weights``)
on the stages that keep it, and the last stage then reduces it to the tensors the drafter
context write reads (``kernels/dflash_context/slim.py``, from the shell's
``post_load_weights``); the checkpoint files are only read.

``install_loaded_features`` then installs the plan's weight-dependent fast paths on the
stage's layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)

if TYPE_CHECKING:
    from . import dispatch as _dispatch
    from .modeling import ModelingV2MuseSparkOpen816bA42bNvfp4Sm107Pp


def load(
    model: "ModelingV2MuseSparkOpen816bA42bNvfp4Sm107Pp",
    weights: Any,
    weight_mapper: Optional[Any] = None,
) -> Dict[str, int]:
    """Load the stage's slice of the checkpoint ``weights`` into ``model`` with the built-in loader.

    Returns the built-in loader's key-accounting summary.
    """
    return MuseSparkOpenOmniForConditionalGeneration.load_weights(model, weights, weight_mapper)


def install_loaded_features(
    text_model: MuseSparkOpenModel,
    mapping,
    plan: "_dispatch.Plan",
    compute_capability: Optional[Tuple[int, int]] = None,
) -> None:
    """Install the plan's weight-dependent fast paths on the stage's layers, in order.

    ``prefill_cutedsl_experts`` first (it wraps the expert forwards), then the lossy
    ``context_fp8_dense`` (MXFP8 copies of the dense projections at the lever's
    ``MIN_ROWS`` / ``PROJECTIONS``). Each import happens only when its path is planned, and
    each install is collective over the stages. Last, ``kernels/bf16_expert_tile`` (a fixed
    FlashInfer tactic for the stage's BF16 expert layers; bitwise; not a lever of the plan).
    Under ``NO_PATHS`` this installs nothing.
    """
    if plan.enabled("prefill_cutedsl_experts"):
        from .kernels.prefill_cutedsl_experts import install as _prefill_cutedsl

        _prefill_cutedsl.install(text_model, mapping, compute_capability)
        plan.installed("prefill_cutedsl_experts", True)
    if plan.enabled("context_fp8_dense"):
        from .kernels.context_fp8_dense import install as _context_fp8

        _context_fp8.install(
            text_model,
            mapping,
            compute_capability,
            min_rows=int(plan.param("context_fp8_dense", "MIN_ROWS")),
            projections=tuple(plan.param("context_fp8_dense", "PROJECTIONS")),
        )
        plan.installed("context_fp8_dense", True)
    from .dispatch import NO_PATHS

    if plan is NO_PATHS:
        return
    from .kernels.bf16_expert_tile import install as _bf16_expert_tile

    _bf16_expert_tile.install(text_model, mapping)
