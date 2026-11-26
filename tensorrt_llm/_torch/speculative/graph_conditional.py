# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One CUDA graph for greedy and sampling batches, via CUDA graph conditional nodes.

The one-engine speculative workers capture a greedy graph and a sampling graph per
batch size, because the two record different kernels. With the merged mode the sampling
sites record both variants under conditional (IF) nodes, and a device-side "all requests
greedy" bool picks the executed one at replay, so a batch size needs a single graph and
no host synchronization.

``TRTLLM_MERGE_SAMPLING_GRAPHS=1`` turns the mode on; unset or ``0`` keeps the two-graph
behavior. It also needs torch conditional-node capture, and a worker that declares
``supports_merged_sampling_graphs``.
"""

import contextlib
import os
from typing import Callable, Iterator, TypeVar

import torch
from torch.utils import _pytree as pytree

MERGE_SAMPLING_GRAPHS_ENV = "TRTLLM_MERGE_SAMPLING_GRAPHS"

T = TypeVar("T")

_warming_up_both_branches = False


def merge_sampling_graphs_requested() -> bool:
    """Whether the environment asks for merged greedy/sampling graphs."""
    return os.environ.get(MERGE_SAMPLING_GRAPHS_ENV, "0") == "1"


def conditional_nodes_available() -> bool:
    """Whether this torch captures conditional (IF) nodes into CUDA graphs."""
    graph_cls = torch.cuda.CUDAGraph
    return all(
        hasattr(graph_cls, name)
        for name in (
            "begin_capture_to_if_node",
            "end_capture_to_conditional_node",
            "get_currently_capturing_graph",
        )
    )


@contextlib.contextmanager
def warm_up_both_branches() -> Iterator[None]:
    """Run both branches of every ``graph_if_else`` inside the block.

    The eager warmup that precedes a capture initializes lazily built state (kernel
    compilation, library handles). With a single merged capture, neither branch may meet
    that state for the first time while capturing, so the warmup executes both.
    """
    global _warming_up_both_branches
    previous = _warming_up_both_branches
    _warming_up_both_branches = True
    try:
        yield
    finally:
        _warming_up_both_branches = previous


@contextlib.contextmanager
def sampling_variant_override(spec_metadata, all_greedy: bool) -> Iterator[None]:
    """Make ``spec_metadata`` describe an all-greedy or a sampling batch while tracing a branch.

    The sampling code picks its kernels from these host flags, so each branch is traced under
    the flags it stands for. The temperature-only variant is covered by the sampling branch.
    """
    saved = (
        spec_metadata.is_all_greedy_sample,
        spec_metadata.is_all_temperature_only,
        spec_metadata.temperature_only,
    )
    spec_metadata.is_all_greedy_sample = all_greedy
    spec_metadata.is_all_temperature_only = False
    spec_metadata.temperature_only = False
    try:
        yield
    finally:
        (
            spec_metadata.is_all_greedy_sample,
            spec_metadata.is_all_temperature_only,
            spec_metadata.temperature_only,
        ) = saved


def merged_sampling_active(spec_metadata) -> bool:
    """Whether the sampling sites should go through ``graph_if_else`` for this metadata."""
    return bool(getattr(spec_metadata, "merged_sampling_graph", False))


def graph_if_else(
    pred: torch.Tensor,
    if_fn: Callable[[], T],
    else_fn: Callable[[], T],
    *,
    host_pred: bool,
) -> T:
    """Run ``if_fn()`` where ``pred`` is true and ``else_fn()`` where it is false.

    ``pred`` is a scalar bool CUDA tensor. While a CUDA graph is being captured, both
    functions are traced into conditional nodes and ``pred`` is read on the device at every
    replay. Outside capture, ``host_pred`` (the same decision, known on the host) picks the
    one function to run, except while ``warm_up_both_branches`` is active, where both run.

    Both functions must return the same pytree of tensors with equal shapes and dtypes; the
    result is a pytree of tensors the executed branch has copied its outputs into.
    """
    if not torch.cuda.is_current_stream_capturing():
        if _warming_up_both_branches:
            if_result = if_fn()
            else_result = else_fn()
            return if_result if host_pred else else_result
        return if_fn() if host_pred else else_fn()

    if os.environ.get("TRTLLM_COND_IMPL") == "torch_cond":  # EXPERIMENT, not for commit
        from torch._higher_order_ops.cudagraph_conditional_nodes import (
            CUDAGraphCaptureControlFlowOpDispatchMode)
        with CUDAGraphCaptureControlFlowOpDispatchMode():
            return torch.cond(pred, if_fn, else_fn, ())

    graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()

    # A body is always closed, so a failing branch surfaces its own error rather than torch's
    # complaint about an unfinished conditional capture.
    graph.begin_capture_to_if_node(pred)
    try:
        if_flat, spec = pytree.tree_flatten(if_fn())
        # Allocated inside the first body, so the buffers come from the capture's own pool.
        outputs = [torch.empty_like(tensor) for tensor in if_flat]
        for output, tensor in zip(outputs, if_flat):
            output.copy_(tensor)
    finally:
        graph.end_capture_to_conditional_node()

    not_pred = torch.logical_not(pred)
    graph.begin_capture_to_if_node(not_pred)
    try:
        else_flat, else_spec = pytree.tree_flatten(else_fn())
        if else_spec != spec:
            raise RuntimeError(
                f"graph_if_else branches return different structures: {spec} vs {else_spec}"
            )
        for output, tensor in zip(outputs, else_flat):
            if output.shape != tensor.shape or output.dtype != tensor.dtype:
                raise RuntimeError(
                    "graph_if_else branches return different tensors: "
                    f"{tuple(output.shape)} {output.dtype} vs "
                    f"{tuple(tensor.shape)} {tensor.dtype}"
                )
            output.copy_(tensor)
    finally:
        graph.end_capture_to_conditional_node()
    return pytree.tree_unflatten(outputs, spec)
