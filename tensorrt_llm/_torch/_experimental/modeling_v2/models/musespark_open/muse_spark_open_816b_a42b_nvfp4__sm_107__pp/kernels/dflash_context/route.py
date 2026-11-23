# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the DFlash captures of a context step travel between the pipeline stages. Pure.

The DFlash drafter reads the target's residual stream after each of its target layers
(``target_layer_ids``), each as one BF16 ``[rows, hidden]`` column block of the step's
capture buffer, in layer order; the drafter context is written on the last stage, which
holds the draft KV layers. Every target layer runs on one stage, so the captures of the
earlier stages are carried forward along the chain: on a context step a stage sends the
next one, after the stage boundary (the FP32 residual and the BF16 attention input of the
next layer), every capture group it received and then its own group. A group is one stage's
captures side by side in layer order, ``[rows, n * hidden]``, contiguous. The last stage
copies each received group into its column blocks of the capture buffer; its own layers tap
into that buffer directly. The drafter then reads the same BF16 captures a one-stage
forward taps, so its fc GEMM sees the same inputs (one GEMM over all of them).

``capture_route`` computes, for a pipeline split, which captures each stage taps, receives
and sends, and ``CaptureRoute.edge_bytes`` the bytes of each stage-to-stage message as the
buffered PP transport packs it (``ring_refusal``).
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, List, Optional, Sequence, Tuple

#: Bytes per element of the FP32 residual of the stage boundary.
RESIDUAL_BYTES = 4
#: The alignment of each tensor inside a message of the buffered PP transport
#: (``distributed.pp_buffered.TENSOR_ALIGN``).
TENSOR_ALIGN = 256

#: A half-open range ``[first, end)`` of capture indices (positions in the sorted target
#: layers), one stage's group.
Span = Tuple[int, int]


def _align(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


@dataclasses.dataclass(frozen=True)
class StageRoute:
    """What one stage does with the captures of a context step.

    ``layers`` is the stage's ``[first, end)`` decoder layers; ``own`` the capture indices
    its layers tap (empty when ``own[0] == own[1]``); ``received`` the groups it receives
    from the previous stage and ``sent`` the groups it sends to the next one, in message
    order (empty on the first and on the last stage respectively).
    """

    stage: int
    layers: Tuple[int, int]
    own: Span
    received: Tuple[Span, ...]
    sent: Tuple[Span, ...]

    @property
    def taps(self) -> bool:
        """Whether a layer of this stage is a target layer of the drafter."""
        return self.own[1] > self.own[0]


@dataclasses.dataclass(frozen=True)
class CaptureRoute:
    """The route of every capture through the stages of one pipeline split.

    ``capture_layers`` are the drafter's target layers, ascending; ``hidden`` the residual
    width; ``activation_bytes`` and ``capture_bytes`` the bytes per element of the
    boundary's attention input and of a capture.
    """

    capture_layers: Tuple[int, ...]
    hidden: int
    stages: Tuple[StageRoute, ...]
    activation_bytes: int = 2
    capture_bytes: int = 2

    @property
    def pp_size(self) -> int:
        return len(self.stages)

    def layers_of(self, span: Span) -> Tuple[int, ...]:
        """The target layers of the capture indices ``span``."""
        return self.capture_layers[span[0] : span[1]]

    def edge_layers(self, stage: int) -> Tuple[int, ...]:
        """The target layers whose captures cross the edge ``stage -> stage + 1``, in order."""
        return tuple(layer for span in self.stages[stage].sent for layer in self.layers_of(span))

    def edge_tensors(self, stage: int, rows: int) -> List[Tuple[Tuple[int, int], int]]:
        """``(shape, bytes per element)`` of each tensor of the message ``stage -> stage + 1``
        of a ``rows``-token step: the residual, the attention input, then each sent group.

        Raises:
            ValueError: for the last stage, which sends nothing.
        """
        if stage >= self.pp_size - 1:
            raise ValueError(f"stage {stage} is the last of {self.pp_size}; it sends nothing")
        tensors = [
            ((rows, self.hidden), RESIDUAL_BYTES),
            ((rows, self.hidden), self.activation_bytes),
        ]
        for first, end in self.stages[stage].sent:
            tensors.append(((rows, (end - first) * self.hidden), self.capture_bytes))
        return tensors

    def edge_bytes(self, stage: int, rows: int, align: int = TENSOR_ALIGN) -> int:
        """Bytes of the message ``stage -> stage + 1``, each tensor at an ``align`` boundary."""
        return sum(
            _align(shape[0] * shape[1] * size, align)
            for shape, size in self.edge_tensors(stage, rows)
        )

    def boundary_bytes(self, rows: int, align: int = TENSOR_ALIGN) -> int:
        """Bytes of the stage boundary alone (residual and attention input)."""
        return _align(rows * self.hidden * RESIDUAL_BYTES, align) + _align(
            rows * self.hidden * self.activation_bytes, align
        )

    def largest_edge(self, rows: int, align: int = TENSOR_ALIGN) -> Tuple[int, int]:
        """``(stage, bytes)`` of the largest message: the edge ``stage -> stage + 1``."""
        edges = [(self.edge_bytes(s, rows, align), s) for s in range(self.pp_size - 1)]
        size, stage = max(edges, key=lambda item: (item[0], -item[1]))
        return stage, size


def capture_route(
    capture_layers: Iterable[int],
    stage_ranges: Sequence[Tuple[int, int]],
    hidden: int,
    activation_bytes: int = 2,
    capture_bytes: int = 2,
) -> CaptureRoute:
    """The ``CaptureRoute`` of the target layers ``capture_layers`` over ``stage_ranges``.

    ``stage_ranges`` are the stages' contiguous ``[first, end)`` layers in stage order
    (``stages.stage_ranges``).

    Raises:
        ValueError: for repeated target layers, or one no stage holds.
    """
    layers = tuple(sorted(int(layer) for layer in capture_layers))
    if len(set(layers)) != len(layers):
        raise ValueError(f"the target layers {list(layers)} repeat a layer")
    ranges = tuple((int(first), int(end)) for first, end in stage_ranges)
    held = [layer for layer in layers if any(f <= layer < e for f, e in ranges)]
    if held != list(layers):
        missing = sorted(set(layers) - set(held))
        raise ValueError(f"no stage of {list(ranges)} holds the target layers {missing}")
    owns: List[Span] = []
    for first, end in ranges:
        inside = [index for index, layer in enumerate(layers) if first <= layer < end]
        owns.append((inside[0], inside[-1] + 1) if inside else (0, 0))
    routes = []
    carried: Tuple[Span, ...] = ()
    last = len(ranges) - 1
    for stage, (span, own) in enumerate(zip(ranges, owns)):
        received = carried
        sent: Tuple[Span, ...] = ()
        if stage < last:
            sent = received + ((own,) if own[1] > own[0] else ())
        routes.append(StageRoute(stage, span, own, received, sent))
        carried = sent
    return CaptureRoute(
        layers, int(hidden), tuple(routes), int(activation_bytes), int(capture_bytes)
    )


def ring_refusal(
    route: CaptureRoute,
    rows: int,
    slot_bytes: int,
    edges: Iterable[int],
    align: int = TENSOR_ALIGN,
) -> Optional[str]:
    """Why a buffered PP ring of ``slot_bytes`` slots cannot carry the context steps, or None.

    ``edges`` are the edges ``stage -> stage + 1`` the ring carries; ``rows`` the most
    tokens of one step (``max_num_tokens``). A message larger than a slot leaves the ring
    for NCCL; the reason names the slot size it would need.
    """
    worst = None
    for stage in sorted(set(int(edge) for edge in edges)):
        need = route.edge_bytes(stage, rows, align)
        if need > slot_bytes and (worst is None or need > worst[1]):
            worst = (stage, need)
    if worst is None:
        return None
    stage, need = worst
    need_mb = -(-need // (1 << 20))
    layers = list(route.edge_layers(stage))
    return (
        f"the context step's stage boundary {stage} -> {stage + 1} carries the DFlash captures "
        f"of target layers {layers} and takes up to {need_mb} MiB at {rows} tokens, more than "
        f"the {slot_bytes >> 20} MiB slot of the buffered PP transport, so such steps go "
        "through NCCL (a lower max_num_tokens keeps them on the ring)"
    )
