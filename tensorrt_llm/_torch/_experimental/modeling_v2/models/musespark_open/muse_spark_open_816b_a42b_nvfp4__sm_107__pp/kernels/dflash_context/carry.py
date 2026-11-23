# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The stage forward of a context step that carries the DFlash captures to the last stage.

``CaptureCarry.forward_stage`` is the built-in stage forward
(``MuseSparkOpenModel._forward_pipeline_stage``: the same embedding, the same per-layer
calls, the same boundary tensors) with the captures of ``route.CaptureRoute`` appended to
the boundary message. ``PrefillTarget`` runs it while ``CaptureCarry.active`` is set;
every other step (``DecodeTarget``) runs the built-in stage forward and its two-tensor
boundary. Both ends of an edge decide alike: the target is picked from the step's context
rows, which every stage of a micro-batch shares.

Where the captures live:

* On a stage that sends (every stage but the last), the step's capture buffer
  (``DFlashSpecMetadata.captured_hidden_states``, ``[max_rows, n_layers * hidden]``) is
  unused otherwise, so the groups are carved out of its storage, layer-major: the group of
  capture indices ``[a, b)`` takes ``rows * (b - a) * hidden`` elements from element
  ``a * max_rows * hidden``. The received groups are received into those views, and the
  stage's own taps write into its group: during the forward the metadata's buffer and its
  layer-to-column map are swapped for the group (``max_rows`` rows of the group's width),
  so the built-in tap fills it in place. Nothing is allocated for the captures there.
* The last stage receives the groups into fresh tensors and copies each into its column
  blocks of the capture buffer after its layers; its own layers tap the buffer directly.

A step without a capture buffer (no speculative metadata) sends groups of the same shapes
whose contents nobody reads, so the message shapes never depend on it. Nobody reads them
because every stage of the step decides alike whether it has speculative metadata: the
executor decides it from the active requests and the speculative config, which all stages
share. The message carries no flag for it.

The buffers stay at fixed addresses for a captured breakable prefill graph: the views are
of the shared capture buffer, and the last stage's receive tensors come from the graph's
pool; the send and the receive are the graph's eager breaks.

**A drafter context window.** When the KV cache manager keeps only a window of the drafter
context (``window``: the step's ``SpecMetadata.draft_context_keep_from``), only the kept rows'
captures cross the boundaries; in this module the kept rows are the plan's carried rows (each
request's kept rows and the lead rows its GEMMs run over, ``window.WindowPlan``). ``plan_step``
plans the step's kept rows before the stage forward runs (the shell's forward, which also runs
on every replay of a captured graph), and the boundary is sent and received by eager breaks of
their own that read the plan of the step being run: a sending stage sends the boundary, the kept
rows of every received group and the kept rows of its own group (a slice of its group's rows
when the kept rows are one run of step rows, else an ``index_select``); a receiving stage
receives the groups into tensors of the kept rows and keeps them (``held``) for its send, or,
on the last stage, for the drafter context write, which assembles the kept rows' captures
(``window_capture``) and stores the requests' kept rows among them (``worker``). A step with
no kept row sends the boundary alone. Both ends plan from the same schedule, so they agree on
every message. A stage's prefill graphs are all captured with the window or all without it. A
graph captured with it raises on a replayed step without a plan instead of sending a message the
other end does not expect; a graph captured without it carries the whole step, and the last stage
then writes the whole drafter context (``window_capture`` returns None), which is never short of
a window.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import torch

from .route import CaptureRoute, Span, StageRoute
from .window import WindowPlan, step_plan


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _raw_transport() -> Tuple[
    Callable[[List[torch.Tensor]], None], Callable[[List[torch.Tensor]], None]
]:
    """The PP transport's send and receive of one message, outside any graph break: the
    buffered ring's own calls, else one NCCL send / receive per tensor."""
    from tensorrt_llm._torch.distributed import communicator

    if communicator.pp_comm_buffered():
        ring = communicator._pp_comm
        return ring.send_tensors, ring.recv_tensors

    def send(tensors: List[torch.Tensor]) -> None:
        for tensor in tensors:
            communicator.pp_send(tensor)

    def recv(tensors: List[torch.Tensor]) -> None:
        for tensor in tensors:
            communicator.pp_recv(tensor)

    return send, recv


class _Staging:
    """The capture groups of one step on one stage."""

    def __init__(
        self,
        buffer: Optional[torch.Tensor],
        hidden: int,
        rows: int,
        dtype: torch.dtype,
        device: torch.device,
        carve: bool,
    ) -> None:
        self.buffer = buffer
        self.hidden = hidden
        self.rows = rows
        self.dtype = dtype
        self.device = device
        self.carve = carve and buffer is not None
        if self.carve and rows > buffer.shape[0]:
            raise ValueError(
                f"a {rows}-token step exceeds the {buffer.shape[0]}-row DFlash capture buffer"
            )

    def _region(self, span: Span, rows: int) -> torch.Tensor:
        first, end = span
        width = (end - first) * self.hidden
        start = first * self.buffer.shape[0] * self.hidden
        return self.buffer.view(-1)[start : start + rows * width].view(rows, width)

    def group(self, span: Span) -> torch.Tensor:
        """The ``[rows, n * hidden]`` tensor of the capture group ``span``."""
        if self.carve:
            return self._region(span, self.rows)
        width = (span[1] - span[0]) * self.hidden
        return torch.empty((self.rows, width), dtype=self.dtype, device=self.device)

    @contextlib.contextmanager
    def own_taps(self, spec_metadata: Any, route: CaptureRoute, stage: StageRoute) -> Iterator:
        """While the layers run: the stage's taps write into its own group (a sending stage)."""
        if not (self.carve and stage.sent and stage.taps):
            yield
            return
        first, _ = stage.own
        buffer = spec_metadata.captured_hidden_states
        index = spec_metadata._layer_to_idx
        spec_metadata.captured_hidden_states = self._region(stage.own, buffer.shape[0])
        spec_metadata._layer_to_idx = {
            layer: position for position, layer in enumerate(route.layers_of(stage.own))
        }
        try:
            yield
        finally:
            spec_metadata.captured_hidden_states = buffer
            spec_metadata._layer_to_idx = index

    def assemble(self, groups: Sequence[torch.Tensor], spans: Sequence[Span]) -> None:
        """Copy the received groups into their column blocks of the capture buffer (last stage)."""
        if self.buffer is None:
            return
        for group, (first, end) in zip(groups, spans):
            self.buffer[: self.rows, first * self.hidden : end * self.hidden].copy_(group)


class CaptureCarry:
    """The context-step stage forward of one stage, with the DFlash captures carried.

    ``route`` is the split's ``CaptureRoute`` and ``stage`` this stage's index;
    ``send`` / ``recv`` the stage-boundary transport (the communicator's
    ``pp_send_tensors`` / ``pp_recv_tensors`` when None). With a drafter context window,
    ``raw_send`` / ``raw_recv`` carry one message outside any graph break (``_raw_transport``
    when None) and ``eager_break`` makes a call an eager break of a breakable prefill graph
    (``eager_on_graph`` when None); all three are resolved at the first window step.
    """

    def __init__(
        self,
        route: CaptureRoute,
        stage: int,
        send: Optional[Callable[[List[torch.Tensor]], None]] = None,
        recv: Optional[Callable[[List[torch.Tensor]], None]] = None,
        raw_send: Optional[Callable[[List[torch.Tensor]], None]] = None,
        raw_recv: Optional[Callable[[List[torch.Tensor]], None]] = None,
        eager_break: Optional[Callable[[Callable], Callable]] = None,
    ) -> None:
        self.route = route
        self.stage = route.stages[stage]
        self.last = stage == route.pp_size - 1
        self.first = stage == 0
        self.active = False
        self._raw_send = raw_send
        self._raw_recv = raw_recv
        self._eager_break = eager_break
        self._send_window: Optional[Callable[[List[torch.Tensor]], None]] = None
        self._recv_window: Optional[Callable[[List[torch.Tensor]], None]] = None
        #: The kept rows of the step being run (``plan_step``); None writes every row.
        self.plan: Optional[WindowPlan] = None
        self._index: Optional[torch.Tensor] = None
        #: The received groups' kept rows, and the plan they were received for.
        self._held: List[torch.Tensor] = []
        self._held_plan: Optional[WindowPlan] = None
        self._group_dtype = torch.bfloat16
        #: Whether this stage's prefill graphs were captured with the window (None: none yet).
        self._captured_window: Optional[bool] = None
        if send is None or recv is None:
            from tensorrt_llm._torch.distributed.communicator import (
                pp_recv_tensors,
                pp_send_tensors,
            )

            send = pp_send_tensors if send is None else send
            recv = pp_recv_tensors if recv is None else recv
        self._send = send
        self._recv = recv

    def plan_step(self, attn_metadata: Any, spec_metadata: Any) -> Optional[WindowPlan]:
        """Plan the kept rows of the step about to run (``window.step_plan``); eager, before
        the stage forward or its graph replay."""
        self.plan = step_plan(attn_metadata, spec_metadata)
        self._index = None
        self._held = []
        self._held_plan = None
        return self.plan

    def _rows(self, device: Any) -> torch.Tensor:
        if self._index is None:
            self._index = self.plan.index(device)
        return self._index

    def _window_ops(self) -> None:
        if self._send_window is not None:
            return
        if self._raw_send is None or self._raw_recv is None:
            send, recv = _raw_transport()
            self._raw_send = self._raw_send or send
            self._raw_recv = self._raw_recv or recv
        if self._eager_break is None:
            from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph

            self._eager_break = eager_on_graph
        self._send_window = self._eager_break(self._send_kept)
        self._recv_window = self._eager_break(self._recv_kept)

    def _step_plan(self) -> WindowPlan:
        if self.plan is None:
            raise RuntimeError(
                "a prefill graph captured with the drafter context window replayed a step "
                "without one; every context step of this server must keep the window"
            )
        return self.plan

    def _recv_kept(self, boundary: List[torch.Tensor]) -> None:
        """Receive the boundary and the kept rows of every received group (eager)."""
        plan = self._step_plan()
        groups = []
        if plan.count:
            device = boundary[0].device
            groups = [
                torch.empty(
                    (plan.count, (end - first) * self.route.hidden),
                    dtype=self._group_dtype,
                    device=device,
                )
                for first, end in self.stage.received
            ]
        self._raw_recv([*boundary, *groups])
        self._held = groups
        self._held_plan = plan

    def _send_kept(self, tensors: List[torch.Tensor]) -> None:
        """Send the boundary, the held groups and the kept rows of the own group (eager).

        ``tensors`` is the boundary, then the own group's full rows on a tapping stage.
        """
        plan = self._step_plan()
        if not self.first and self._held_plan is not plan:
            raise RuntimeError("the captures held on this stage were received for another step")
        boundary, own = tensors[:2], tensors[2:]
        groups = []
        if plan.count:
            groups = [*self._held, *(self._kept_rows(group) for group in own)]
        self._raw_send([*boundary, *groups])
        self._held = []

    def _kept_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """The plan's carried rows of a per-step-row tensor, in order: a slice when they are
        one run of rows (``WindowPlan.span``), else an ``index_select`` of the plan's index."""
        span = self.plan.span()
        if span is not None:
            return rows.narrow(0, *span)
        return rows.index_select(0, self._rows(rows.device))

    def window_capture(
        self, buffer: torch.Tensor
    ) -> Optional[Tuple[WindowPlan, Optional[torch.Tensor]]]:
        """``(plan, captures)`` of the kept rows on the last stage, or None.

        ``captures`` is ``[count, n_layers * hidden]`` in the capture buffer's column order:
        the held groups' kept rows and the kept rows of the stage's own taps in ``buffer``;
        None when the step keeps no row. None when this step was not received with the window
        (no plan, or a graph captured without the window wrote the whole capture buffer).
        """
        plan = self.plan
        if not self.last or plan is None or self._held_plan is not plan:
            return None
        if not plan.count:
            return plan, None
        hidden = self.route.hidden
        width = hidden * len(self.route.capture_layers)
        captures = torch.empty((plan.count, width), dtype=buffer.dtype, device=buffer.device)
        for group, (first, end) in zip(self._held, self.stage.received):
            captures[:, first * hidden : end * hidden].copy_(group)
        if self.stage.taps:
            first, end = self.stage.own
            columns = buffer[:, first * hidden : end * hidden]
            captures[:, first * hidden : end * hidden].copy_(self._kept_rows(columns))
        self._held = []
        return plan, captures

    def _note_capture(self, window: bool) -> None:
        if not _capturing():
            return
        if self._captured_window is None:
            self._captured_window = window
        elif self._captured_window != window:
            raise RuntimeError(
                "prefill graphs of one stage were captured with and without the drafter "
                "context window"
            )

    @contextlib.contextmanager
    def activated(self) -> Iterator[None]:
        """``active`` for the duration: ``PrefillTarget`` runs its forward inside."""
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def _staging(self, spec_metadata: Any, rows: int, device: torch.device) -> _Staging:
        buffer = getattr(spec_metadata, "captured_hidden_states", None)
        dtype = buffer.dtype if buffer is not None else torch.bfloat16
        return _Staging(buffer, self.route.hidden, rows, dtype, device, carve=not self.last)

    def forward_stage(
        self,
        core: Any,
        attn_metadata: Any,
        input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """The stage forward of ``core`` (the built-in stage text model) on a context step.

        The same calls as ``MuseSparkOpenModel._forward_pipeline_stage``; the boundary
        message is ``[residual, attention input, *received groups, own group]``.
        """
        first, end = core._pp_stage
        if (first, end) != self.stage.layers:
            raise ValueError(
                f"the stage holds layers [{first}, {end}); the capture route was planned for "
                f"[{self.stage.layers[0]}, {self.stage.layers[1]})"
            )
        spec_metadata = kwargs.get("spec_metadata")
        rows_of = input_ids if input_ids is not None else inputs_embeds
        rows = rows_of.shape[0]
        staging = self._staging(spec_metadata, rows, rows_of.device)
        window = self.plan is not None and staging.buffer is not None
        self._note_capture(window)
        if window:
            self._window_ops()
            self._group_dtype = staging.dtype
            received = []
        else:
            received = [staging.group(span) for span in self.stage.received]
        if self.first:
            if inputs_embeds is None:
                inputs_embeds = core.embed_tokens(input_ids)
            embeds = core.embed_norm(inputs_embeds.to(core.dtype))
            if core.m_emb != 1.0:
                embeds = embeds * core.m_emb
            hidden_states = embeds.to(torch.float32)
            precomputed_attn_input = None
        else:
            shape = (rows, core.model_config.pretrained_config.hidden_size)
            hidden_states = torch.empty(shape, dtype=torch.float32, device=rows_of.device)
            precomputed_attn_input = torch.empty(shape, dtype=core.dtype, device=rows_of.device)
            if window:
                self._recv_window([hidden_states, precomputed_attn_input])
            else:
                self._recv([hidden_states, precomputed_attn_input, *received])
        with staging.own_taps(spec_metadata, self.route, self.stage):
            for index in range(first, end):
                next_norm = (
                    core.layers[index + 1].pre_qkv_proj_norm
                    if index + 1 < len(core.layers)
                    else None
                )
                result = core.layers[index](
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    attn_metadata=attn_metadata,
                    precomputed_attn_input=precomputed_attn_input,
                    next_layer_norm=next_norm,
                    **kwargs,
                )
                if next_norm is None:
                    hidden_states = result
                else:
                    hidden_states, precomputed_attn_input = result
        if not self.last:
            own = [staging.group(self.stage.own)] if self.stage.taps else []
            if window:
                self._send_window([hidden_states, precomputed_attn_input, *own])
            else:
                self._send([hidden_states, precomputed_attn_input, *received, *own])
            return hidden_states
        if not window:
            staging.assemble(received, self.stage.received)
        return core.norm(hidden_states.to(core.dtype))
