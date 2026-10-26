# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One break of a breakable CUDA graph: a callable run eagerly between captured segments.

Wraps the engine's `eager_on_graph` (`pyexecutor/breakable_cuda_graph`). It launches no
kernel of its own; what it certifies is the control flow around the callable it is handed.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph as _eager_on_graph

from .._op import Arch, Cell, OpWrapper


class EagerOnGraph(OpWrapper):
    """`inner(*args)`, eagerly; under breakable CUDA-graph capture, a break.

    Outside a `BreakableCUDAGraphCapture` the call is a passthrough. Inside one it ends the
    captured segment, runs `inner(*args)` once, records a replay function holding `inner`
    and the arguments, and opens the next segment. `BreakableCUDAGraph.replay` then runs
    each segment and, between two of them, re-runs `inner` with the **capture-time**
    arguments: a CUDA tensor by its capture-time address, a CPU tensor as the same object,
    an `int` / `float` / `bool` / `str` / `None` frozen at its capture-time value. Whatever
    varies per step therefore reaches `inner` through state it reads itself, which the
    caller refreshes before each replay, never through an argument.

    Fusion boundary: none. The break runs exactly what `inner` runs, on the current stream.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Spec axes: `capture` (inside a breakable capture, or plain eager), `breaks` (breaks in
    # one body), `tokens` (the bucket `T` the body is captured at), `width` (row width),
    # `live_rows` (the rows `n` each replay carries, set together with a fresh per-step scale
    # and fresh input rows). Each break's `inner` reads `n` and the scale from live state,
    # mixes rows `[:n]` across the batch, writes them into a fixed in-place output, zeroes
    # rows `[n:]` and returns None: the shape of every break a prefill body makes.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="outside capture: a passthrough, bitwise the inner call",
            spec=dict(capture=False, breaks=2, tokens=256, width=128, live_rows=(256, 17)),
        ),
        Cell(
            why="one break, bucket 256: replays at new live rows equal eager",
            spec=dict(capture=True, breaks=1, tokens=256, width=128, live_rows=(17, 256, 1)),
        ),
        Cell(
            why="50 breaks in one body, bucket 1024",
            spec=dict(capture=True, breaks=50, tokens=1024, width=128, live_rows=(1000, 255, 1024)),
        ),
        Cell(
            why="bucket 8192, live rows across the 2048 boundary and the full bucket",
            spec=dict(capture=True, breaks=2, tokens=8192, width=64, live_rows=(6144, 8192, 2047)),
        ),
    )

    note = """
    Preconditions and hazards, each silent unless stated otherwise:

    * **Scalars are frozen.** An `int`, `float`, `bool`, `str` or `None` argument replays at
      its capture-time value. A static per-site constant (a layer index) is right; a
      per-step count is wrong at every replay.
    * **CUDA tensors are re-read by address.** An argument must live at a stable address for
      the graph's life: allocated inside the captured body (the graph pool) or an
      engine-owned persistent buffer. A tensor allocated per step outside the graph replays
      the capture-time memory.
    * **Any other argument raises.** A dataclass, a namespace or a bound object raises
      TypeError at capture, after `inner` already ran once. Loud, but it is a break raising
      under capture, so the next hazard follows it.
    * **A break that raises under capture leaves a CUDA error pending.** The raise comes
      after the break ended the captured segment and before it opened the next, and the
      capture's exit ends the ended segment again: it raises RuntimeError over the break's
      own error and leaves cudaErrorIllegalState pending in the CUDA runtime, so the
      process's next CUDA launch fails once with "the operation cannot be performed in the
      present state". A caller that recovers must take the pending error
      (`cudaGetLastError`) before launching anything else.
    * **Side-stream waits are dropped.** While a capture is open, including inside `inner`'s
      own capture-time run, a wait of the capture stream on a stream that is not capturing
      is dropped with a warning, and replay does not preserve it.
    * **The return value is copied each replay** onto the object `inner` returned at
      capture. `inner` writing a fixed output in place returns None.
    * **A fixed output must be fully written.** Rows `inner` leaves alone keep whatever an
      earlier replay or another bucket left there.

    Hard limits
      none: no index-typed argument
    """

    def __call__(self, *args: Any, inner: Callable[..., Any]) -> Any:
        return _eager_on_graph(inner)(*args)

    def reference(self, *args: Any, inner: Callable[..., Any]) -> Any:
        """The inner call, eagerly: what every replay of the break must reproduce."""
        return inner(*args)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Bitwise: a break adds no arithmetic, so any difference is a frozen or stale
        operand, not rounding."""
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        assert torch.equal(out, ref), (
            f"{int((out != ref).sum())} of {out.numel()} elements differ from the eager body"
        )


eager_on_graph = EagerOnGraph()
