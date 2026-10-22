# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``early_expert_ids``: the persistent expert kernel starts its weight stream before its inputs exist.

With ``moe_prologue`` a decode MoE layer of at most 8 rows prepares its expert input
in one kernel (``ops/comm/pool_gather_route``: the pool gather, the NVFP4
quantization, the routing and the replicated experts' placement), and the persistent
kernel (``ops/moe/persistent_experts``) follows it on the same stream. The
expert weights it streams depend only on the placed ids, which exist inside
the routed kernel before that kernel ends.

This install binds one ``ops/moe/early_expert_ids`` buffer per device to
the persistent experts' workspace and gives it to every NVFP4 layer that runs both kernels:

* the routed kernel's first CTA stores the placed ids as tagged words as soon
  as its routing warps have them, and triggers its dependents after its logit
  exchange (``early_expert_ids.TRIGGER``);
* the persistent kernel, launched behind it, polls those words before its
  dependency wait, routes and starts streaming weights; it waits only before
  it reads the activations or writes anything.

Values are unchanged bit for bit (the same ids reach the same kernel).
Installed after ``ring_buddy.install`` (the expert stacks and the persistent
experts' workspace exist then), before any graph capture. With PDL off
(``TRTLLM_ENABLE_PDL``) it installs on no layer (the early start needs the
programmatic edge), and ``modeling.install_loaded_features`` refuses the selection.
A layer whose placement the routed kernel cannot publish (not one of
``pool_gather_route.EARLY_PLACES``) is refused here. The polls that fell back
are logged at process exit.
"""

from __future__ import annotations

import atexit

import torch

from tensorrt_llm.logger import logger

_EXIT_LOG = {"registered": False}


def _eligible(moe) -> bool:
    """Whether this layer runs the prologue's routed kernel followed by the persistent expert kernel.

    A class built on the prologue MoE qualifies: the one the target makes is
    ``router_gemm``'s, which changes the router logits of 9..512-row calls only.
    """
    from ...ops.moe import persistent_experts as _persistent
    from ..moe_prologue.prologue import MuseSparkOpenTepPrologueMoE
    from ..replicated_experts import ring_buddy as _ring_buddy

    state = getattr(moe, "_replicated_experts", None)
    if not isinstance(moe, MuseSparkOpenTepPrologueMoE) or state is None or state.kind != "nvfp4":
        return False
    if not getattr(moe, "_prologue_enable_pdl", False):
        return False
    backend = _ring_buddy._backend(moe)
    return (
        _ring_buddy._is_plain_swiglu(backend)
        and backend.w3_w1_weight.shape[-2] // 2 == _persistent.INTER
        and backend.w3_w1_weight.shape[-1] * 2 == _persistent.HIDDEN
    )


def unpublishable(places: dict) -> dict:
    """The layers (index -> placement kind) whose placement ``early_expert_ids`` cannot publish.

    ``places``: layer index -> the placement kind the routed kernel runs there
    (``moe_prologue._placement``). The routed kernel publishes the ids that
    ``pool_gather_route._placed_expert`` computes, i.e. the kinds in
    ``EARLY_PLACES`` (and none without a placement).
    """
    from ...ops.comm import pool_gather_route as _route

    allowed = (_route.PLACE_NONE,) + tuple(_route.EARLY_PLACES)
    return {index: kind for index, kind in places.items() if kind not in allowed}


def _log_fallbacks_at_exit() -> None:
    from ...ops.moe import early_expert_ids as _early_ids

    seen = _early_ids.fallbacks_seen()
    if seen is not None:
        logger.info(
            f"[musespark_open] early_expert_ids: {seen} early-id polls fell back (deadline) in this process"
        )


def install(text_model) -> int:
    """Give every eligible NVFP4 MoE layer the device's early-ids buffer; returns their count.

    Compiles the persistent expert kernel's early-ids variant for each layer's stack now (no graph
    capture may compile). Every rank takes the same branch: the decision reads
    the model's structure, the placement mode and the PDL switch only.

    Raises:
        ValueError: if an eligible layer's placement is not one the routed kernel
            publishes, or with the persistent expert kernel's fast route.
    """
    from ...ops.moe import persistent_experts as _persistent
    from ..moe_prologue.prologue import _placement

    layers = [layer for layer in text_model.layers if _eligible(layer.feed_forward)]
    if not layers:
        logger.warning(
            "[musespark_open] early_expert_ids: no layer runs the routed prologue into the persistent experts "
            "with PDL on; nothing installed (the selection is refused)"
        )
        return 0
    refused = unpublishable(
        {layer.layer_idx: _placement(layer.feed_forward)[0] for layer in layers}
    )
    if refused:
        raise ValueError(
            "[musespark_open] early_expert_ids publishes the placed ids that pool_gather_route._placed_expert "
            f"computes (placement kinds {sorted(set(refused.values()))} on layers "
            f"{sorted(refused)} are not among them); route that placement through the "
            "helper and EARLY_PLACES, or deselect early_expert_ids"
        )
    early = None
    for layer in layers:
        moe = layer.feed_forward
        stacks = moe._replicated_experts.stacks
        w13 = stacks["w3_w1_weight"]
        if early is None:
            early = _persistent.early_expert_ids(w13.device)
        _persistent.persistent_experts.prepare(
            w13,
            stacks["w3_w1_weight_scale"].view(torch.float8_e4m3fn),
            stacks["w2_weight"],
            stacks["w2_weight_scale"].view(torch.float8_e4m3fn),
            early=early,
        )
        moe._early_ids_state = early
    if not _EXIT_LOG["registered"]:
        atexit.register(_log_fallbacks_at_exit)
        _EXIT_LOG["registered"] = True
    trigger = {0: "exit", 1: "exchange", 2: "entry"}[early.trigger]
    logger.info(
        f"[musespark_open] early_expert_ids: early expert ids on {len(layers)} MoE layers (routed kernel triggers at "
        f"{trigger}; poll back-off {early.poll_ns} ns, deadline {early.deadline_ns} ns)"
    )
    return len(layers)
