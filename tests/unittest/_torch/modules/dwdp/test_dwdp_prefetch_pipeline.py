# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""CPU-mocked tests for the DWDP prefetch event protocol."""

from contextlib import nullcontext
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.dwdp.weight_manager import DWDPWeightManager

pytestmark = pytest.mark.cpu_only

LAYERS = [3, 5, 7]


class _Event:
    def __init__(self, name: str, log: List[Tuple[str, ...]]) -> None:
        self.name = name
        self._log = log

    def record(self, stream) -> None:
        self._log.append(("record", self.name, stream.name))


class _Stream:
    def __init__(self, name: str, log: List[Tuple[str, ...]]) -> None:
        self.name = name
        self._log = log

    def wait_event(self, event: _Event) -> None:
        self._log.append(("wait", self.name, event.name))


class TestDWDPPrefetchPipeline:
    def setup_method(self):
        self.log: List[Tuple[str, ...]] = []
        self.weight_buffer = MagicMock()
        self.weight_buffer.device_id = 0
        self.weight_buffer.local_start = 0
        self.weight_buffer.local_end = 2
        self.weight_buffer.buffer_index_for_layer.side_effect = lambda layer: (
            LAYERS.index(layer) % 2
        )
        self.weight_buffer.get_full_tensor.side_effect = lambda layer, name: torch.tensor(
            [float(layer)]
        )

        self.copy_stream = _Stream("copy", self.log)
        self.compute_stream = _Stream("compute", self.log)
        events = iter(
            [_Event(name, self.log) for name in ("prefetch0", "prefetch1", "consume0", "consume1")]
        )
        self.patches = [
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.Stream",
                return_value=self.copy_stream,
            ),
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.Event",
                side_effect=lambda: next(events),
            ),
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.current_stream",
                return_value=self.compute_stream,
            ),
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.stream",
                return_value=nullcontext(),
            ),
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.synchronize",
                side_effect=lambda device: self.log.append(("synchronize",)),
            ),
            patch.object(
                DWDPWeightManager,
                "_prefetch_layer_per_slice",
                lambda _self, layer: self.log.append(("copy", layer)),
            ),
        ]
        for cuda_patch in self.patches:
            cuda_patch.start()

        self.manager = DWDPWeightManager(
            weight_buffer=self.weight_buffer,
            peer_views={},
            peer_ranges=[(0, 2), (2, 4)],
            moe_layer_indices=LAYERS,
            weight_names=["gate_up_proj", "down_proj"],
            dwdp_rank=0,
            dwdp_size=2,
        )
        self.backend = nn.Module()
        self.backend.gate_up_proj = nn.Parameter(torch.empty(1), requires_grad=False)
        self.backend.down_proj = nn.Parameter(torch.empty(1), requires_grad=False)
        self.log.clear()

    def teardown_method(self):
        self.manager._released = True
        for cuda_patch in reversed(self.patches):
            cuda_patch.stop()

    @staticmethod
    def _copy(layer: int, slot: int) -> List[Tuple[str, ...]]:
        return [
            ("wait", "copy", f"consume{slot}"),
            ("copy", layer),
            ("record", f"prefetch{slot}", "copy"),
        ]

    def _bind(self, layer: int) -> None:
        self.manager.wait_and_bind(self.backend, layer)
        assert self.backend.gate_up_proj.data.item() == float(layer)
        assert self.backend.down_proj.data.item() == float(layer)

    def test_event_order_across_passes(self):
        self.manager.start_forward()
        assert self.log == self._copy(3, 0) + self._copy(5, 1)

        # Binding waits for the copy and does not release the slot.
        self.log.clear()
        self._bind(3)
        assert self.log == [("wait", "compute", "prefetch0")]

        # The release is recorded after the MoE was enqueued, then the copy two layers ahead
        # waits for it.
        self.log.clear()
        self.manager.finish_layer(3)
        assert self.log == [("record", "consume0", "compute")] + self._copy(7, 0)

        self.log.clear()
        self._bind(5)
        self.manager.finish_layer(5)
        # Layer 3 of the next pass needs slot 0, still held by layer 7.
        assert self.log == [
            ("wait", "compute", "prefetch1"),
            ("record", "consume1", "compute"),
        ]

        # The last MoE layer of the pass prefetches the first layers of the next one.
        self.log.clear()
        self._bind(7)
        self.manager.finish_layer(7)
        assert self.log == [
            ("wait", "compute", "prefetch0"),
            ("record", "consume0", "compute"),
        ] + self._copy(3, 0) + self._copy(5, 1)

        self.log.clear()
        self.manager.start_forward()
        assert self.log == []
        self._bind(3)
        assert self.log == [("wait", "compute", "prefetch0")]

    def test_bind_without_start_forward_issues_its_copy_first(self):
        self._bind(3)
        assert self.log == self._copy(3, 0) + self._copy(5, 1) + [("wait", "compute", "prefetch0")]

    def test_abandoned_pass_synchronizes_before_reusing_slots(self):
        self.manager.start_forward()
        self._bind(3)
        self.log.clear()
        self.manager.start_forward()
        assert self.log == [("synchronize",)] + self._copy(3, 0) + self._copy(5, 1)

    def test_invalidate_regathers_the_first_layers(self):
        self.manager.start_forward()
        for layer in LAYERS:
            self._bind(layer)
            self.manager.finish_layer(layer)
        self.log.clear()
        self.manager.invalidate_prefetch()
        assert self.log == []
        self.manager.start_forward()
        assert self.log == self._copy(3, 0) + self._copy(5, 1)

    def test_unknown_layer_raises(self):
        with pytest.raises(KeyError):
            self.manager.wait_and_bind(self.backend, 4)
        with pytest.raises(KeyError):
            self.manager.finish_layer(4)
