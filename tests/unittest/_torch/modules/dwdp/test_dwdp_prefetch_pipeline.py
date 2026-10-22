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
"""CPU-mocked tests for the DWDP double-buffer event protocol."""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.dwdp.weight_manager import DWDPWeightManager


class TestDWDPPrefetchPipeline:
    def setup_method(self):
        self.weight_buffer = MagicMock()
        self.weight_buffer.device_id = 0
        self.weight_buffer.local_start = 0
        self.weight_buffer.local_end = 2
        self.weight_buffer.buffer_index_for_layer.side_effect = {
            3: 0,
            5: 1,
            7: 0,
        }.__getitem__

        self.copy_stream = MagicMock(name="copy_stream")
        self.compute_stream = MagicMock(name="compute_stream")
        self.events = [
            MagicMock(name="prefetch_0"),
            MagicMock(name="prefetch_1"),
            MagicMock(name="consume_0"),
            MagicMock(name="consume_1"),
        ]
        self.patches = [
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.Stream",
                return_value=self.copy_stream,
            ),
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.Event",
                side_effect=self.events,
            ),
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.current_stream",
                return_value=self.compute_stream,
            ),
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.stream",
                return_value=nullcontext(),
            ),
        ]
        for cuda_patch in self.patches:
            cuda_patch.start()

        self.manager = DWDPWeightManager(
            weight_buffer=self.weight_buffer,
            peer_views={},
            peer_ranges=[(0, 2), (2, 4)],
            moe_layer_indices=[3, 5, 7],
            weight_names=["gate_up_proj", "down_proj"],
            dwdp_rank=0,
            dwdp_size=2,
        )
        for event in self.events:
            event.reset_mock()

    def teardown_method(self):
        self.manager._released = True
        for cuda_patch in reversed(self.patches):
            cuda_patch.stop()

    def test_wait_and_bind_waits_for_copy_without_signalling_consumption(self):
        full_tensors = {
            "gate_up_proj": torch.tensor([1.0]),
            "down_proj": torch.tensor([2.0]),
        }
        self.weight_buffer.get_full_tensor.side_effect = lambda _layer_idx, name: full_tensors[name]
        backend = nn.Module()
        backend.gate_up_proj = nn.Parameter(torch.empty(1), requires_grad=False)
        backend.down_proj = nn.Parameter(torch.empty(1), requires_grad=False)

        self.manager.wait_and_bind(backend, 3)

        self.compute_stream.wait_event.assert_called_once_with(self.events[0])
        self.events[2].record.assert_not_called()
        assert backend.gate_up_proj.data.data_ptr() == full_tensors["gate_up_proj"].data_ptr()
        assert backend.down_proj.data.data_ptr() == full_tensors["down_proj"].data_ptr()

    def test_record_compute_signals_the_current_slot(self):
        self.manager.record_compute(3)

        self.events[2].record.assert_called_once_with(self.compute_stream)
        self.events[3].record.assert_not_called()
