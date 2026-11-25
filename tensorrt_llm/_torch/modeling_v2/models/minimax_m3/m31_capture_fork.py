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
"""FRONT_CAPFORK: the Eagle3 hidden-state capture of a decode layer, off the layer's PDL chain (family level:
the sm_107 tp2, tp4 and tp8 targets import it).

On a capture layer whose capture norm is not armed, the in-tree capture (``maybe_capture_hidden_states`` ->
``trtllm::inplace_slice_copy``, a cudaMemcpy2DAsync of the boundary's updated residual into the capture buffer's column
block) runs between the feed-forward boundary all-reduce and the next layer's QKV GEMV. In a CUDA graph that copy is a
memcpy node: the GEMV's dependency becomes a non-kernel node, so it loses its programmatic (PDL) edge, cannot launch
early and streams its weights after the copy.

``fork_capture`` runs the same copy on a capture stream that waits for the boundary (event record / wait), so the
GEMV keeps its programmatic edge to the boundary; the source tensor is held alive until ``join``, which makes the
current stream wait for the capture stream. The target's forward joins right after the decoder stack returns, before
the final-hidden capture, the head and the drafter read the capture buffer. Same bytes, same buffer: bit for bit the
in-line capture.
"""

import os
import sys
from typing import Dict, List, Optional

import torch

_STREAMS: Dict[int, torch.cuda.Stream] = {}
# Sources of the forked copies not yet joined, per device index (held so the caching allocator cannot recycle them).
_PENDING: Dict[int, List[torch.Tensor]] = {}
_LOGGED: set = set()


def _rank() -> str:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return str(torch.distributed.get_rank())
    except Exception:  # noqa: BLE001
        pass
    for name in ("OMPI_COMM_WORLD_RANK", "RANK", "SLURM_PROCID", "PMI_RANK"):
        if name in os.environ:
            return os.environ[name]
    return "?"


def capture_stream(device: torch.device) -> torch.cuda.Stream:
    """The device's capture stream (created on first use; the first use is the eager warmup forward)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _STREAMS.get(index)
    if stream is None:
        stream = _STREAMS[index] = torch.cuda.Stream(device=index)
    return stream


def fork_capture(spec_metadata, layer_id: int, hidden_states: torch.Tensor, target: str) -> None:
    """``spec_metadata.maybe_capture_hidden_states(layer_id, hidden_states, None)`` on the capture stream, after
    everything already issued on the current stream (the boundary that wrote ``hidden_states``)."""
    device = hidden_states.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    main = torch.cuda.current_stream(device)
    aux = capture_stream(device)
    aux.wait_stream(main)
    with torch.cuda.stream(aux):
        spec_metadata.maybe_capture_hidden_states(layer_id, hidden_states, None)
    _PENDING.setdefault(index, []).append(hidden_states)
    if target not in _LOGGED:
        _LOGGED.add(target)
        print(
            f"[M31_FRONT_CAPFORK] engaged: target {target} rank {_rank()} device {index}: Eagle3 capture copy "
            f"(layer {layer_id}) forked off the boundary -> QKV PDL edge",
            file=sys.stderr,
            flush=True,
        )


def pending(device=None) -> int:
    """How many forked copies the current stream has not joined yet (all devices when ``device`` is None)."""
    if device is None:
        return sum(len(v) for v in _PENDING.values())
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return len(_PENDING.get(index, ()))


def join() -> int:
    """Make each device's current stream wait for its capture stream when a forked copy is pending; returns how many
    copies it joined. A no-op (no stream operation, nothing recorded into a graph) when none is pending."""
    joined = 0
    for index, sources in list(_PENDING.items()):
        if not sources:
            continue
        torch.cuda.current_stream(index).wait_stream(_STREAMS[index])
        joined += len(sources)
        sources.clear()
    return joined


def capture_or_inline(
    spec_metadata,
    layer_id: int,
    hidden_states: torch.Tensor,
    residual,
    target: str,
    enabled: bool,
    max_rows: Optional[int] = None,
) -> None:
    """Shared standalone-DSpark capture selection; the caller joins before consuming captures.

    ``max_rows`` restricts a newly ported shape without changing the established targets.
    Final hidden state (-1), residual addends, CPU tensors and disabled paths keep the original callback.
    """
    if (
        enabled
        and residual is None
        and layer_id != -1
        and hidden_states.is_cuda
        and (max_rows is None or 1 <= hidden_states.shape[0] <= max_rows)
    ):
        fork_capture(spec_metadata, layer_id, hidden_states, target)
        return
    spec_metadata.maybe_capture_hidden_states(layer_id, hidden_states, residual)
