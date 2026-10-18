# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared by the real-weight composition tests: per-layer output copies and graph edges.

The tests record every decoder layer's outputs with a forward hook, inside the
CUDA graph capture too. A copy made on the capturing stream right after a layer
returns sits between that layer's fused post-FFN boundary (the fused unpool gather) and
the next layer's QKV GEMM: the GEMM becomes a dependent of a plain copy kernel, a full
dependency, and the early release's early launch of it never happens in the test.
``LayerOutputCopies`` makes the copies on a side stream forked after each layer
instead (the copied tensors stay referenced until the join), and the test joins
it after the model call, before anything reads the copies.

``programmatic_edges`` reads a captured graph's programmatic edges (the edges into
kernels launched with programmatic dependent launch), so a test can check that
each fused boundary is the programmatic predecessor of the next QKV GEMM.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import torch

#: Where the hook copies: ``side`` (default) or ``main`` (the capturing stream: the
#: negative control of the edge check, no fused boundary keeps a programmatic edge).
HOOK_STREAM_ENV = "MODELING_V2_HOOK_STREAM"
#: The fused unpool gather + post-FFN boundary kernel (``ops/comm/unpool_gather_boundary``).
BOUNDARY_KERNEL = "unpool"

Edge = Tuple[str, str, int]


def hook_on_side_stream() -> bool:
    """Whether the per-layer copies go to a side stream (``MODELING_V2_HOOK_STREAM``)."""
    value = os.environ.get(HOOK_STREAM_ENV, "side")
    if value not in ("side", "main"):
        raise ValueError(f"{HOOK_STREAM_ENV} is 'side' or 'main', got {value!r}")
    return value == "side"


def dedicated_stream() -> "torch.cuda.ExternalStream":
    """A non-blocking CUDA stream of its own on the current device.

    ``torch.cuda.Stream()`` hands out the streams of a small per-device pool in
    turn, so after enough other allocations it returns a stream already in use --
    possibly the one a later ``torch.cuda.graph`` captures on. A side stream that
    is the capturing stream puts its work on the forward's own stream.
    """
    from cuda.bindings import runtime

    device = torch.cuda.current_device()
    (status,) = runtime.cudaSetDevice(device)
    if status != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaSetDevice({device}) failed: {status}")
    status, handle = runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
    if status != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamCreateWithFlags failed: {status}")
    return torch.cuda.ExternalStream(int(handle), device=torch.device("cuda", device))


class LayerOutputCopies:
    """A forward hook that appends copies of a layer's outputs to ``sink``.

    With ``side`` the copies are made on a side stream of its own (never a
    pooled stream, see ``dedicated_stream``) forked from the current stream
    after the layer; the layer's own output tensors are kept referenced until
    ``join``, so the current stream cannot reuse their memory while the side
    stream still reads them. ``join`` makes the current stream wait for the
    copies; a test calls it after every model call (inside the capture for a
    captured forward, which must end with every forked stream joined).
    """

    def __init__(self, sink: List[list], side: bool) -> None:
        self.sink = sink
        self.side = side
        self.stream = dedicated_stream() if side else None
        self._sources: List[Sequence[torch.Tensor]] = []

    def __call__(self, module, args, output) -> None:
        outputs = output if isinstance(output, tuple) else (output,)
        if not self.side:
            self.sink.append([t.clone() for t in outputs])
            return
        if self.stream.cuda_stream == torch.cuda.current_stream().cuda_stream:
            raise RuntimeError("the layer-output side stream is the stream the forward runs on")
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.sink.append([t.clone() for t in outputs])
        self._sources.append(outputs)

    def join(self) -> None:
        """Make the current stream wait for every copy made since the last join."""
        if self._sources:
            torch.cuda.current_stream().wait_stream(self.stream)
            self._sources.clear()


def _check(result) -> tuple:
    from cuda.bindings import driver

    status, *values = result
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {status}")
    return tuple(values)


def _kernel_name(node, names: Dict[int, str]) -> str:
    from cuda.bindings import driver

    key = int(node)
    if key not in names:
        (kind,) = _check(driver.cuGraphNodeGetType(node))
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            names[key] = f"<{kind.name}>"
        else:
            (params,) = _check(driver.cuGraphKernelNodeGetParams(node))
            status, raw = driver.cuFuncGetName(params.func)
            if status != driver.CUresult.CUDA_SUCCESS:
                status, raw = driver.cuKernelGetName(params.kern)
            if status != driver.CUresult.CUDA_SUCCESS:
                raw = "<unnamed kernel>"
            names[key] = raw.decode() if isinstance(raw, bytes) else str(raw)
    return names[key]


def programmatic_edges(graph: "torch.cuda.CUDAGraph") -> Dict[Edge, int]:
    """``{(from kernel, to kernel, from port): count}`` over a captured graph's programmatic edges.

    ``graph`` must have been made with ``keep_graph=True`` and not yet replayed.
    """
    from cuda.bindings import driver

    handle = driver.CUgraph(graph.raw_cuda_graph())
    get = getattr(driver, "cuGraphGetEdges_v2", None) or driver.cuGraphGetEdges
    total = int(_check(get(handle, 0))[-1])
    sources, targets, data, _ = _check(get(handle, total))
    programmatic = int(driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC)
    names: Dict[int, str] = {}
    out: Dict[Edge, int] = {}
    for source, target, edge in zip(sources, targets, data):
        if int(edge.type) == programmatic:
            key = (_kernel_name(source, names), _kernel_name(target, names), int(edge.from_port))
            out[key] = out.get(key, 0) + 1
    return out


def boundary_edges(edges: Dict[Edge, int]) -> int:
    """How many programmatic edges leave a fused unpool boundary kernel."""
    return sum(
        count for (source, _, _), count in edges.items() if BOUNDARY_KERNEL in source.lower()
    )


def edge_table(edges: Dict[Edge, int], limit: int = 40) -> List[list]:
    """The most frequent programmatic edges, for a result table."""
    ordered = sorted(edges.items(), key=lambda item: -item[1])[:limit]
    return [[source[:200], target[:200], port, count] for (source, target, port), count in ordered]
