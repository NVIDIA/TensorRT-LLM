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
"""Layer-wise host-KV onboarding.

Context resumes defer their host->GPU copies; ``prepare_resources`` issues them
in layer order on the core's dedicated high-priority stream, recording an event
per layer. The execution stream no longer waits for the whole migration (nor for
the offloads that freed its GPU slots): attention for layer ``i`` waits only for
layer ``i``'s copies, so the forward overlaps the transfer.

Readers must cooperate, so deferral is automatic only when an attention backend
that calls :meth:`LayerwiseOnboard.wait_layer` before touching a layer's cache
has registered itself with :func:`register_layerwise_consumer`; otherwise
resumes wait for their copies as before. The last target layer also retires
every manager's outstanding events (including a separate draft manager's, whose
attention does not wait per layer), so a draft forward that runs after the
target forward sees complete pages.

Enabled by ``KvCacheConfig.layerwise_host_onboard`` (default off).
"""

from __future__ import annotations

import weakref
from typing import Callable, Optional

import torch

_consumer_registered = False
# Managers whose flushed copies still have per-layer events to retire.
_armed: "weakref.WeakSet[LayerwiseOnboard]" = weakref.WeakSet()


# One event per layer: the first layer's readers wait for the least data.
LAYERWISE_ONBOARD_GROUPS = 1 << 20


def register_layerwise_consumer() -> None:
    """Declare that the model's attention calls :meth:`LayerwiseOnboard.wait_layer` per layer."""
    global _consumer_registered
    _consumer_registered = True


def layerwise_consumer_registered() -> bool:
    return _consumer_registered


def wait_all_armed(stream: int) -> None:
    """Make ``stream`` wait for every manager's outstanding onboarding copies."""
    for onboard in list(_armed):
        onboard.wait_all(stream)


class LayerwiseOnboard:
    """Per-manager driver of the core's deferred, layer-ordered onboarding."""

    def __init__(
        self,
        impl,
        layer_offset: Callable[[int], Optional[int]],
        last_target_layer: Callable[[], Optional[int]],
    ):
        self._impl = impl
        self._layer_offset = layer_offset
        self._last_target_layer_fn = last_target_layer
        self._last_target_layer: Optional[int] = None
        self._last_target_layer_known = False
        self.pending = False
        impl.set_layerwise_onboard(True, LAYERWISE_ONBOARD_GROUPS)

    @property
    def armed(self) -> bool:
        """Cheap check for the per-layer hook: anything to wait for, here or elsewhere."""
        return self.pending or bool(_armed)

    def _is_last_target_layer(self, layer_idx: int) -> bool:
        if not self._last_target_layer_known:
            self._last_target_layer = self._last_target_layer_fn()
            self._last_target_layer_known = True
        return self._last_target_layer is not None and layer_idx == self._last_target_layer

    def flush(self, stream: int) -> None:
        """Issue this step's deferred copies; ``stream`` waits only for partial-block copies."""
        if self._impl.flush_onboard(stream) or self._impl.has_onboard_events:
            self.pending = True
            _armed.add(self)
            if not _consumer_registered:
                self.wait_all(stream)

    def wait_layer(self, layer_idx: int, stream: Optional[int] = None) -> None:
        """Order ``stream`` (default: current) after the onboarding copies of ``layer_idx``."""
        if torch.cuda.is_current_stream_capturing():
            return
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        if self.pending:
            offset = self._layer_offset(layer_idx)
            if offset is not None:
                self._impl.wait_layer_onboard(offset, stream)
        if self._is_last_target_layer(layer_idx) and _armed:
            wait_all_armed(stream)

    def wait_all(self, stream: int) -> None:
        self._impl.wait_all_onboard(stream)
        self.pending = False
        _armed.discard(self)
