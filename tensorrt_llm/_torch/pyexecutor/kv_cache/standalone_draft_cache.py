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
"""History contract of a standalone drafter whose context KV the target manager owns."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from tensorrt_llm._utils import prefer_pinned

if TYPE_CHECKING:
    from .kv_cache_manager_v2 import KVCacheManagerV2


@dataclass(frozen=True)
class StandaloneDraftHistory:
    """Committed draft context length and the next absolute position.

    ``valid_length`` tokens ending just before ``position`` hold drafter
    context KV. Both are independent of the target's history and of any
    scratch capacity reserved for the next draft forward.
    """

    valid_length: int
    position: int

    def __post_init__(self) -> None:
        if type(self.valid_length) is not int or type(self.position) is not int:
            raise ValueError("Standalone draft history requires integer length and position")
        if self.valid_length < 0 or self.position < self.valid_length:
            raise ValueError("Invalid standalone draft history length or position")


@dataclass(frozen=True)
class DraftHistoryUpdate:
    """One iteration's draft history, published after its sampling event completes."""

    manager: "KVCacheManagerV2"
    request_ids: tuple[int, ...]
    cache_instances: tuple[object, ...]
    values_host: torch.Tensor
    # Recorded after the device-to-host copy of a CUDA ``values``.
    copied: Optional[torch.cuda.Event] = None

    @classmethod
    def capture(
        cls,
        manager: "KVCacheManagerV2",
        request_ids: tuple[int, ...],
        values: torch.Tensor,
    ) -> "DraftHistoryUpdate":
        """Queue a length/position readback on the current stream.

        Args:
            manager: The unified cache manager owning the draft layers.
            request_ids: Requests of the rows of ``values``.
            values: ``[len(request_ids), 2]`` integer tensor of
                ``(valid_length, position)`` per request.
        """
        if tuple(values.shape) != (len(request_ids), 2):
            raise ValueError("Draft history update requires one length/position pair per request")
        values_host = torch.empty(
            values.shape,
            dtype=values.dtype,
            device="cpu",
            pin_memory=values.is_cuda and prefer_pinned(),
        )
        values_host.copy_(values, non_blocking=True)
        copied = None
        if values.is_cuda:
            copied = torch.cuda.Event()
            copied.record()
        return cls(
            manager,
            tuple(request_ids),
            tuple(manager.kv_cache_map[request_id] for request_id in request_ids),
            values_host,
            copied,
        )

    def publish(self) -> None:
        """Publish the readback without reviving released or replaced requests.

        Call once the iteration's sampling has completed: the readback is then
        done and waiting on it does not block.
        """
        if self.copied is not None:
            self.copied.synchronize()
        for request_id, cache, (length, position) in zip(
            self.request_ids, self.cache_instances, self.values_host.tolist()
        ):
            current = self.manager.kv_cache_map.get(request_id)
            if current is cache and current.is_active:
                self.manager.set_draft_history(request_id, length, position)

    def issue(self) -> "DraftHistoryUpdate":
        """The readback is already queued (``DeferredDraftHistory.issue`` returns one of these)."""
        return self


_readback_streams: dict[int, torch.cuda.Stream] = {}


def _readback_stream(device: torch.device) -> torch.cuda.Stream:
    """The side stream of a device's draft-history device-to-host copies."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _readback_streams.get(index)
    if stream is None:
        stream = _readback_streams[index] = torch.cuda.Stream(device=index)
    return stream


class DeferredDraftHistory:
    """An iteration's draft-history readback, recorded on the host and queued later.

    ``read`` queues the gathers of the ``[len(request_ids), 2]`` (length, position) values on
    the current stream and returns them. The executor calls :meth:`issue` once the iteration's
    sampling is queued, so nothing is queued between the forward and the sampler. The worker
    that made it issues it before its own next write to the state ``read`` gathers, so the
    gathers always run before that write in stream order. The device-to-host copy runs on a
    side stream after the gathers; :meth:`DraftHistoryUpdate.publish` waits for it. Issuing it
    while the current stream captures a CUDA graph raises.
    """

    def __init__(
        self,
        manager: "KVCacheManagerV2",
        request_ids: tuple[int, ...],
        read: Callable[[], torch.Tensor],
    ):
        self.manager = manager
        self.request_ids = tuple(request_ids)
        # The cache of each request at this iteration, as DraftHistoryUpdate.capture takes it.
        self.cache_instances = tuple(manager.kv_cache_map[r] for r in self.request_ids)
        self._read: Optional[Callable[[], torch.Tensor]] = read
        self._update: Optional[DraftHistoryUpdate] = None

    @property
    def issued(self) -> bool:
        return self._update is not None

    def issue(self) -> DraftHistoryUpdate:
        """Queue the gathers on the current stream and the copy on the side stream (once)."""
        if self._update is not None:
            return self._update
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # The gathers, the side-stream wait and the copy would be recorded into the graph.
            raise RuntimeError(
                "A draft-history readback must not be issued under CUDA graph capture"
            )
        values = self._read()
        self._read = None
        if tuple(values.shape) != (len(self.request_ids), 2):
            raise ValueError("Draft history update requires one length/position pair per request")
        if not values.is_cuda:
            values_host = values.clone()
            copied = None
        else:
            ready = torch.cuda.Event()
            ready.record()
            side = _readback_stream(values.device)
            side.wait_event(ready)
            values_host = torch.empty(
                values.shape, dtype=values.dtype, device="cpu", pin_memory=prefer_pinned()
            )
            with torch.cuda.stream(side):
                values_host.copy_(values, non_blocking=True)
                copied = torch.cuda.Event()
                copied.record()
            # The caching allocator keeps the values until the side stream has read them.
            values.record_stream(side)
        self._update = DraftHistoryUpdate(
            self.manager, self.request_ids, self.cache_instances, values_host, copied
        )
        return self._update

    def publish(self) -> None:
        """Issue (if nothing has yet) and publish, as :meth:`DraftHistoryUpdate.publish`."""
        self.issue().publish()
