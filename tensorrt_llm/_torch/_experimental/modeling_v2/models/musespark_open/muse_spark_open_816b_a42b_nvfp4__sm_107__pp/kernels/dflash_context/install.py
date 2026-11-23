# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the DFlash drafter context on a pipeline stage of a context server.

``install`` runs when the shell is constructed, on every stage, after the composition (with
its ``dispatch.DrafterContext``) is agreed: it plans the capture route of the stage split
(``route.capture_route``), checks the last stage's DFlash worker (``worker.refusal``), agrees
the outcome over the stages, and only then builds the stage's ``carry.CaptureCarry`` and
re-classes the last stage's worker, which reads the carry's drafter context window. It warns
when the buffered PP ring cannot carry the enlarged context-step boundary (``ring_refusal``):
such steps then go through NCCL. ``slim_last_stage`` reduces the last stage's drafter once its
weights are loaded.
"""

from __future__ import annotations

from typing import Any, Optional

from tensorrt_llm.logger import logger

from ... import dispatch as _dispatch
from ... import stages as _stages
from . import carry as _carry
from . import route as _route
from . import slim as _slim
from . import worker as _worker


def capture_route(model_config: Any, drafter: "_dispatch.DrafterContext") -> _route.CaptureRoute:
    """The capture route of the deployment's split and drafter target layers."""
    mapping = model_config.mapping
    pretrained = model_config.pretrained_config
    num_layers = int(pretrained.num_hidden_layers)
    ranges = _stages.stage_ranges(
        num_layers, int(mapping.pp_size), getattr(mapping, "pp_partition", None)
    )
    element = _element_size(getattr(model_config, "torch_dtype", None) or pretrained.torch_dtype)
    return _route.capture_route(
        drafter.target_layers,
        ranges,
        int(pretrained.hidden_size),
        activation_bytes=element,
        capture_bytes=element,
    )


def _element_size(dtype: Any) -> int:
    size = getattr(dtype, "itemsize", None)
    return int(size) if size else 2


def _max_rows(model_config: Any) -> Optional[int]:
    rows = getattr(model_config, "max_num_tokens", None)
    return int(rows) if rows else None


def ring_refusal(route: _route.CaptureRoute, stage: int, max_rows: Optional[int]) -> Optional[str]:
    """Why this stage's buffered PP ring cannot carry the context steps, or None.

    Reads the PP transport the executor initialized before the model (the communicator's
    ``pp_comm_buffered`` and its ring): None without the buffered transport, without a
    known ``max_num_tokens``, or when every edge of this stage the ring carries fits a slot.
    """
    from tensorrt_llm._torch.distributed import communicator, pp_buffered

    if max_rows is None or not communicator.pp_comm_buffered():
        return None
    ring = getattr(communicator, "_pp_comm", None)
    slot_bytes = getattr(ring, "slot_bytes", None)
    if slot_bytes is None:
        return None
    edges = []
    if getattr(ring, "recv_on_ring", False) and stage > 0:
        edges.append(stage - 1)
    if getattr(ring, "send_on_ring", False) and stage < route.pp_size - 1:
        edges.append(stage)
    return _route.ring_refusal(
        route,
        max_rows,
        int(slot_bytes),
        edges,
        align=pp_buffered.TENSOR_ALIGN,
    )


def install(
    shell: Any, model_config: Any, drafter: "_dispatch.DrafterContext"
) -> _carry.CaptureCarry:
    """Plan, check and agree the stage's drafter context, then build it. Collective.

    Every stage calls it at the same point. A refusal on any stage raises on every stage.

    Raises:
        ValueError: when the stage split does not match the built stage, or the last
            stage's worker is not a DFlash worker (on every stage alike).
    """
    mapping = model_config.mapping
    stage = int(getattr(mapping, "pp_rank", 0) or 0)
    last = stage == int(mapping.pp_size) - 1
    text_model = shell.llm.model

    def check() -> _route.CaptureRoute:
        route = capture_route(model_config, drafter)
        built = tuple(getattr(text_model, "_pp_stage", None) or ())
        if built != route.stages[stage].layers:
            raise ValueError(
                f"{_dispatch.TAG} stage {stage} holds layers {list(built)}; the capture route of "
                f"the split puts {list(route.stages[stage].layers)} on it"
            )
        if last:
            reason = _worker.refusal(getattr(shell.llm, "spec_worker", None))
            if reason is not None:
                raise ValueError(f"{_dispatch.TAG} {reason}")
        return route

    route = _dispatch.agreed_install(mapping, check)
    reason = ring_refusal(route, stage, _max_rows(model_config))
    if reason is not None:
        logger.warning(f"{_dispatch.TAG} {reason}")
    carry = _carry.CaptureCarry(route, stage)
    if last:
        _worker.install(shell.llm.spec_worker, carry)
    return carry


def describe(
    route: _route.CaptureRoute,
    stage: int,
    rows: Optional[int],
    carry: Optional[_carry.CaptureCarry] = None,
) -> str:
    """One log line: this stage's captures and the largest boundary at ``rows`` tokens.

    ``carry`` is accepted for the callers' convenience and adds nothing to the line."""
    info = route.stages[stage]
    own = list(route.layers_of(info.own))
    received = [layer for span in info.received for layer in route.layers_of(span)]
    sent = list(route.edge_layers(stage)) if stage < route.pp_size - 1 else []
    line = f"DFlash captures on stage {stage}: taps {own}, receives {received}, sends {sent}"
    if rows is not None and route.pp_size > 1:
        edge, size = route.largest_edge(rows)
        line += (
            f"; largest context-step boundary {edge} -> {edge + 1}: "
            f"{-(-size // (1 << 20))} MiB at {rows} tokens"
        )
    return line


def slim_last_stage(shell: Any) -> Optional[int]:
    """Reduce the last stage's loaded drafter (``slim.slim``); the bytes released, or None on
    a stage without a drafter."""
    draft_model = getattr(shell.llm, "draft_model", None)
    if draft_model is None:
        return None
    return _slim.slim(draft_model)
