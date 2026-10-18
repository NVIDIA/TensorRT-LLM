# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The weights this target expects: the Muse Spark Open NVFP4 checkpoint as the built-in model loads it.

The target's module tree is the built-in model's, so the built-in loader maps
every checkpoint tensor -- the open-format renames, the stacked NVFP4 experts and
their scales, the head-interleaved query and output-gate projection, and the
perception towers when they are built. The drafter's weights come from the
``speculative_model`` checkpoint through the built-in draft loader, not from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)

from . import dispatch as _tp_agreement

if TYPE_CHECKING:
    from . import dispatch as _dispatch
    from .modeling import ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep


def load(
    model: "ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep",
    weights: Any,
    weight_mapper: Optional[Any] = None,
) -> Dict[str, int]:
    """Load the checkpoint ``weights`` into ``model`` with the built-in loader.

    Returns the built-in loader's key-accounting summary.
    """
    return MuseSparkOpenOmniForConditionalGeneration.load_weights(model, weights, weight_mapper)


def install_loaded_features(
    text_model: MuseSparkOpenModel, mapping, plan: "_dispatch.Plan"
) -> None:
    """Install the plan's expert paths on a text model whose weights are final.

    ``replicated_experts``: ``ring_buddy.install`` (collective over the TP
    group), balanced. Imports ``ring_buddy`` only when it is planned; without it
    this installs no expert path. With ``persistent_experts`` the install
    routes the NVFP4 layers' stacked calls to the persistent expert kernel
    (``persistent=True``; the keyword is passed only then), with
    ``three_window_placement`` it stacks a third ring window on those
    layers (``windows=3``, likewise passed only then), and with
    ``bf16_routed_experts`` the BF16 layers' stacked calls take unpacked
    routing (``bf16_unpacked=True``; likewise passed only then).

    Then ``early_expert_ids`` (``early_ids.install``, on the layers just installed),
    and last ``pool_into_route`` (``DirectPool.install_sites``, whose routed kernel
    variants carry the placement and the early expert ids), each collective over the
    TP group.
    """
    if plan.enabled("replicated_experts"):
        _install_ring_buddy(text_model, mapping, plan)
    if plan.enabled("pool_into_route"):
        pool_direct = getattr(text_model, "_pool_direct", None)
        if pool_direct is None:
            raise ValueError(
                "[musespark_open] pool_into_route is planned but the model shell built no direct pool "
                "transport"
            )
        pool_direct.install_sites(text_model, mapping)


def _install_ring_buddy(text_model, mapping, plan: "_dispatch.Plan") -> None:
    """``install_loaded_features``'s ``replicated_experts`` install and, after it, ``early_expert_ids``."""
    from .kernels.replicated_experts import ring_buddy as _ring_buddy

    extra = {"persistent": True} if plan.enabled("persistent_experts") else {}
    if plan.enabled("three_window_placement"):
        extra["windows"] = 3
    if plan.enabled("bf16_routed_experts"):
        extra["bf16_unpacked"] = True
    _ring_buddy.install(
        text_model,
        mapping,
        balance=True,
        static_from=_ring_buddy.BLOCK,
        **extra,
    )
    if plan.enabled("early_expert_ids"):
        # The persistent expert kernel's early start, on the layers just installed
        # (needs their stacks and the persistent kernel's workspace).
        from .kernels.early_expert_ids import early_ids as _early_experts

        def install_early_ids():
            if _early_experts.install(text_model) == 0:
                raise ValueError(
                    "[musespark_open] early_expert_ids installed the early expert ids on no layer "
                    "(see its warning above)"
                )

        # Agreed over the TP group (replicated_embedding's collectives follow): a
        # refusal, none installed (no layer runs the prologue's routed kernel into the
        # persistent expert kernel with PDL on), or a failed compile of the persistent
        # kernel's early-ids variant.
        _tp_agreement.agreed_install(mapping, install_early_ids, (ValueError, RuntimeError))


def install_embedding_features(
    text_model: MuseSparkOpenModel, mapping, plan: "_dispatch.Plan"
) -> None:
    """Install ``replicated_embedding`` on a text model whose weights are final.

    ``replicated_embedding.install`` is collective over the TP group. Imports
    ``replicated_embedding`` only when it is planned; otherwise this does nothing.
    """
    if plan.enabled("replicated_embedding"):
        from .kernels.replicated_embedding import embedding as _replicated_embedding

        _replicated_embedding.install(text_model, mapping)
