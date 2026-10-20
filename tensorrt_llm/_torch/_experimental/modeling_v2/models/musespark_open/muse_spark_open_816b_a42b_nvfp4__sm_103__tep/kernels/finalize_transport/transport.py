# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``finalize_transport``: the fused finalize + all-reduce on the hop transport.

Where the op is called: the built-in MoE's forward hands its unfinalized
expert rows to ``self._finalize_ar_workspace.finalize(gemm2_output, indices,
routing_weights, norm_weight, eps, num_tokens)``, the ``FinalizeIpcWorkspace``
the built-in text model builds -- at TP8 over two nodes when
``MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL=1`` (fabric), at TP4 on one node
always (IPC) -- and gives every layer. Every path that reaches the
finalize -- decode and prefill, the built-in expert call and the replicated experts'
stacked one -- goes through that attribute.

So ``install`` builds one ``FinalizeTransportWorkspace`` in front of the model's
built-in workspace (collective: a multicast fabric allocation and the kernel's
compilation on every rank) and puts it in every layer's attribute. Its
``finalize`` has the built-in signature: calls of 1..16 tokens run
``ops/moe/finalize_allreduce_transport``, bit for bit the built-in kernel's
result; larger calls run the built-in kernel. No module is re-classed, so the
fast paths that re-class the MoE (``projection_gathers``) or patch its experts (``replicated_experts``)
compose with it in any order.
"""

from __future__ import annotations

from typing import Optional

from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel

from ...ops.moe import finalize_allreduce_transport as _fin

#: The attribute the built-in MoE reads its fused finalize workspace from.
ATTRIBUTE = "_finalize_ar_workspace"

#: The fast path that installs the workspace, and the transport it builds.
FEATURES = {"finalize_transport": _fin.DEFAULT}


def refusal(model: MuseSparkOpenModel) -> Optional[str]:
    """Why ``install`` cannot replace this model's fused finalize, or None. Changes nothing."""
    legacy = getattr(model, ATTRIBUTE, None)
    if legacy is None:
        return (
            "the model runs no fused finalize all-reduce (it needs "
            "MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL=1 at TP8 over two four-GPU nodes, or "
            "TP4 on one four-GPU node)"
        )
    others = [
        layer.layer_idx
        for layer in model.layers
        if getattr(layer.feed_forward, ATTRIBUTE, None) is not legacy
    ]
    if others:
        return f"layers {others} do not call the model's fused finalize workspace"
    if not _fin.HAS_CUTE_DSL:
        return "the CuTe DSL or the built-in finalize module is unavailable"
    return None


def agreed_refusal(model: MuseSparkOpenModel) -> Optional[str]:
    """Why ``install`` refuses, as every TP rank agrees, or None (collective over the TP group).

    Each rank's ``refusal`` is gathered, so that a reason only some ranks see
    (ranks running different images or environments) makes every rank refuse
    before the workspace's collective construction. Changes nothing.
    """
    reason = refusal(model)
    local = None if reason is None else f"[musespark_open] finalize_transport cannot run: {reason}"
    everyone = Distributed.get(model.layers[0].mapping).tp_allgather(local, small_payload=True)
    refused = [(rank, why) for rank, why in enumerate(everyone) if why is not None]
    if not refused:
        return None
    reasons = sorted({why for _, why in refused})
    if len(refused) == len(everyone) and len(reasons) == 1:
        return reasons[0]
    ranks = [rank for rank, _ in refused]
    return f"{'; '.join(reasons)} (on TP ranks {ranks} of {len(everyone)})"


def install(model: MuseSparkOpenModel):
    """Put a ``FinalizeTransportWorkspace`` in every layer's finalize attribute (collective).

    Raises:
        ValueError: if ``refusal`` names a reason on some rank (on every rank
            alike, from ``agreed_refusal``, before the workspace is built).
        RuntimeError: if the workspace could not be built on some rank (on every
            rank).
    """
    reason = agreed_refusal(model)
    if reason is not None:
        raise ValueError(reason)
    legacy = getattr(model, ATTRIBUTE)
    transport = FEATURES["finalize_transport"]
    workspace = _fin.FinalizeTransportWorkspace(legacy, legacy._mapping, transport)
    for layer in model.layers:
        setattr(layer.feed_forward, ATTRIBUTE, workspace)
    _fin.log_once_enabled(len(model.layers), transport)
    return workspace
