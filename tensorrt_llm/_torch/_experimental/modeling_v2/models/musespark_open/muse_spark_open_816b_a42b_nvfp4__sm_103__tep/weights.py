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

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

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
    text_model: MuseSparkOpenModel,
    mapping,
    plan: "_dispatch.Plan",
    compute_capability: Optional[Tuple[int, int]] = None,
) -> None:
    """Install the plan's expert paths on a text model whose weights are final.

    ``replicated_experts``: ``ring_buddy.install`` (collective over the TP
    group), balanced. Imports ``ring_buddy`` only when an expert path is planned;
    without one this installs no expert path. With ``persistent_experts`` the install
    routes the NVFP4 layers' stacked calls to the persistent expert kernel
    (``persistent=True``; the keyword is passed only then), with
    ``three_window_placement`` it stacks a third ring window on those
    layers (``windows=3``, likewise passed only then), and with
    ``bf16_routed_experts`` the BF16 layers' stacked calls take unpacked
    routing (``bf16_unpacked=True``; likewise passed only then).

    Without ``replicated_experts``, ``persistent_experts`` and ``bf16_routed_experts``
    are installed on the owner-only window (``_install_owner_window``: no
    replication, every expert on its owner), at tp 4 or tp 8; with
    ``half_window_experts`` (tp 4) the NVFP4 layers also stack the ring predecessor's
    upper half, which the prologue's routed kernel places onto (``half=True``).

    Then ``hot_replicas`` (``hot_replicas.install``: hooks only, at tp 4; the copies are
    gathered on the first forward of a deployment that serves 33..512-row decode steps),
    then, on Rubin (sm107) at tp 4 (``prefill_cutedsl_experts.serves`` of
    ``compute_capability``, the device's; None installs it nowhere),
    ``prefill_cutedsl_experts`` (``prefill_cutedsl_experts.install``: the
    4096..8192-row expert calls of the NVFP4 layers on the Rubin CuTe DSL GEMMs),
    then ``early_expert_ids`` (``early_ids.install``, on the layers just installed),
    then ``pool_into_route`` (``DirectPool.install_sites``, whose routed kernel
    variants carry the placement and the early expert ids), each collective over the
    TP group, and last ``next_qkv_prefetch``'s wrapper around every hooked layer's
    expert forward (``prefetch.wrap_expert_forwards``, outermost, so it sees
    whichever expert call the paths above installed).
    """
    if plan.enabled("replicated_experts"):
        _install_ring_buddy(text_model, mapping, plan)
    elif plan.enabled("persistent_experts") or plan.enabled("bf16_routed_experts"):
        _install_owner_window(text_model, mapping, plan)
    if plan.enabled("hot_replicas"):
        from .kernels.replicated_experts import hot_replicas as _hot_replicas

        # After the owner-window install, whose expert and routing hooks it wraps.
        _hot_replicas.install(text_model, mapping)
    if compute_capability is not None:
        from .kernels.prefill_cutedsl_experts import install as _prefill_cutedsl

        if _prefill_cutedsl.serves(compute_capability, mapping.tp_size):
            # After the expert paths above, whose expert forwards it wraps for the large
            # prefill calls only.
            _prefill_cutedsl.install(text_model, mapping)
    if plan.enabled("early_expert_ids"):
        _install_early_expert_ids(text_model, mapping)
    if plan.enabled("pool_into_route"):
        pool_direct = getattr(text_model, "_pool_direct", None)
        if pool_direct is None:
            raise ValueError(
                "[musespark_open] pool_into_route is planned but the model shell built no direct pool "
                "transport"
            )
        pool_direct.install_sites(text_model, mapping)
    if plan.enabled("next_qkv_prefetch"):
        from .kernels.next_qkv_prefetch import prefetch as _qkv_prefetch

        _qkv_prefetch.wrap_expert_forwards(text_model)


def _install_ring_buddy(text_model, mapping, plan: "_dispatch.Plan") -> None:
    """``install_loaded_features``'s ``replicated_experts`` install."""
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


def _install_owner_window(text_model, mapping, plan: "_dispatch.Plan") -> None:
    """``persistent_experts`` / ``bf16_routed_experts`` without ``replicated_experts``.

    ``ring_buddy.install(windows=1)``: every MoE layer's stacked calls run over the
    rank's own expert block (64 experts at tp 4, 32 at tp 8) with owner-only ids;
    nothing is replicated. Collective over the TP group (its refusals are agreed).

    Raises:
        ValueError: at a tensor-parallel size other than 4 or 8, or for any
            refusal of the install (on every rank alike).
    """
    tp = int(getattr(mapping, "tp_size", 0) or 0)
    if tp == 4:
        # One tray: the owner-only window is the served expert path (a second ring
        # window would cost about 113 GiB per rank).
        pass
    elif tp == 8:
        # The tp 8 composition serves replicated_experts; this window serves plans
        # that leave it out.
        pass
    else:
        raise ValueError(
            "[musespark_open] persistent_experts / bf16_routed_experts on the owner-only expert window "
            f"are written for tp 4 and tp 8, not tp {tp}"
        )
    from .kernels.replicated_experts import ring_buddy as _ring_buddy

    extra = {"persistent": True} if plan.enabled("persistent_experts") else {}
    if plan.enabled("wide_persistent_experts"):
        from .ops.moe import persistent_experts as _persistent

        extra["persistent_rows"] = _persistent.WIDE_MAX_ROWS
    if plan.enabled("bf16_routed_experts"):
        extra["bf16_unpacked"] = True
    if plan.enabled("half_window_experts"):
        extra["half"] = True
    _ring_buddy.install(text_model, mapping, balance=False, windows=1, **extra)


def _install_early_expert_ids(text_model, mapping) -> None:
    """``early_expert_ids``: the persistent expert kernel's early start, on the layers the
    expert install just patched (needs their stacks and the persistent kernel's workspace)."""
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
    if plan.enabled("model_boundary"):
        # After replicated_embedding, whose table it gathers from. Collective (agreed refusal).
        from .kernels.boundary_early_release import triggers as _pdl_triggers
        from .kernels.model_boundary import boundary as _model_boundary

        _model_boundary.install(text_model, _pdl_triggers.capture_layers(text_model), mapping)
