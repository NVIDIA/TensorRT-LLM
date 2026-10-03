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
"""``DFlashSpecMetadata.capture_view``: a captured layer's slot of the capture buffer, which a kernel can write the tap
into (host-side). The view shares the buffer's storage, so what is written through it is what ``get_hidden_states``
hands the drafter, in that layer's columns only; a layer that is not captured, or metadata without a capture buffer,
has no view."""

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

pytestmark = pytest.mark.cpu_only

HIDDEN, MAX_TOKENS = 16, 32


@pytest.fixture(autouse=True)
def cpu_buffers(monkeypatch):
    """The metadata's buffers on the host."""
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)


def _metadata(layers=(3, 1)):
    return DFlashSpecMetadata(
        max_num_requests=8,
        max_draft_len=3,
        max_total_draft_tokens=3,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
        layers_to_capture=None if layers is None else list(layers),
        hidden_size=HIDDEN,
        max_num_tokens=MAX_TOKENS,
        dtype=torch.bfloat16,
    )


def _zero(t):
    return torch.equal(t, torch.zeros_like(t))


@pytest.mark.parametrize("layer,slot", [(1, 0), (3, 1)])
def test_capture_view_is_the_layers_slot(layer, slot):
    metadata = _metadata()  # layers 3 and 1: the buffer holds layer 1's slot first
    buffer = metadata.captured_hidden_states
    buffer.zero_()
    num_tokens = 5

    view = metadata.capture_view(layer, num_tokens)

    assert view.shape == (num_tokens, HIDDEN) and view.dtype == buffer.dtype
    assert view.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
    assert view.data_ptr() == buffer[0, slot * HIDDEN :].data_ptr()
    assert view.stride() == buffer.stride()
    tap = torch.randn(num_tokens, HIDDEN).to(torch.bfloat16)
    view.copy_(tap)
    captured = metadata.get_hidden_states(num_tokens)
    other = 1 - slot
    assert torch.equal(captured[:, slot * HIDDEN : (slot + 1) * HIDDEN], tap)
    assert _zero(captured[:, other * HIDDEN : (other + 1) * HIDDEN])
    assert _zero(buffer[num_tokens:])


def test_cuda_graph_metadata_views_the_shared_buffer():
    metadata = _metadata()
    graph = metadata.create_cuda_graph_metadata(4)
    assert graph.capture_view(3, 4).data_ptr() == metadata.capture_view(3, 4).data_ptr()


def test_no_view_outside_the_captured_layers():
    assert _metadata().capture_view(2, 4) is None
    uncaptured = _metadata(layers=None)
    assert uncaptured.captured_hidden_states is None
    assert uncaptured.capture_view(1, 4) is None
