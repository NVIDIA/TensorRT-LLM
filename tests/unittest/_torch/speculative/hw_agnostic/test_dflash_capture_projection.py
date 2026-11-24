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
"""DFlash capture projection and the unified draft cache's metadata contract.

With ``project_captures`` the capture accumulates the drafter's fc over the
captured layers instead of keeping each layer's hidden states. A worker whose
context lives in the KV cache manager's draft layers assigns its own slots, so
``DFlashSpecMetadata.prepare`` leaves them alone.

These tests require CUDA: ``DFlashSpecMetadata`` allocates its buffers on cuda.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="DFlashSpecMetadata allocates its buffers on cuda",
)

HIDDEN_SIZE = 16
MAX_TOKENS = 8
OUT_FEATURES = 12
LAYERS = [1, 3]


def _make_metadata(project_captures: bool) -> DFlashSpecMetadata:
    return DFlashSpecMetadata(
        max_draft_len=4,
        max_total_draft_tokens=4,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
        max_num_requests=2,
        max_num_tokens=MAX_TOKENS,
        hidden_size=HIDDEN_SIZE,
        layers_to_capture=LAYERS,
        dtype=torch.bfloat16,
        project_captures=project_captures,
    )


def _fc_blocks():
    weight = torch.randn(
        OUT_FEATURES, HIDDEN_SIZE * len(LAYERS), dtype=torch.bfloat16, device="cuda"
    )
    blocks = [weight[:, i * HIDDEN_SIZE : (i + 1) * HIDDEN_SIZE] for i in range(len(LAYERS))]
    return weight, blocks


def test_projected_captures_equal_fc_of_the_concatenated_captures():
    md = _make_metadata(project_captures=True)
    assert md.captured_hidden_states is None
    weight, blocks = _fc_blocks()
    md.bind_capture_projection(blocks)

    hidden = [
        torch.randn(MAX_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") for _ in LAYERS
    ]
    for layer_id, h in zip(LAYERS, hidden):
        md.maybe_capture_hidden_states(layer_id, h, None)
    md.maybe_capture_hidden_states(2, hidden[0], None)  # not captured

    projected = md.get_projected_captures(MAX_TOKENS)
    assert projected.dtype == torch.float32
    reference = torch.cat(hidden, dim=1).float() @ weight.float().t()
    torch.testing.assert_close(projected, reference, rtol=1e-4, atol=1e-4)
    assert md.get_hidden_states(MAX_TOKENS) is None


def test_projection_restarts_at_the_first_captured_layer():
    md = _make_metadata(project_captures=True)
    weight, blocks = _fc_blocks()
    md.bind_capture_projection(blocks)
    for _ in range(2):
        hidden = [
            torch.randn(MAX_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
            for _ in LAYERS
        ]
        for layer_id, h in zip(LAYERS, hidden):
            md.maybe_capture_hidden_states(layer_id, h, None)
    reference = torch.cat(hidden, dim=1).float() @ weight.float().t()
    torch.testing.assert_close(
        md.get_projected_captures(MAX_TOKENS), reference, rtol=1e-4, atol=1e-4
    )


def test_projection_without_blocks_keeps_the_capture_buffer():
    md = _make_metadata(project_captures=True)
    md.bind_capture_projection(None)
    assert md.get_projected_captures(MAX_TOKENS) is None
    assert md.captured_hidden_states.shape == (MAX_TOKENS, HIDDEN_SIZE * len(LAYERS))


def test_projection_must_be_bound_before_capturing():
    md = _make_metadata(project_captures=True)
    h = torch.zeros(MAX_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="bind_capture_projection"):
        md.maybe_capture_hidden_states(LAYERS[0], h, None)
    with pytest.raises(ValueError, match="capture projection"):
        md.bind_capture_projection(_fc_blocks()[1][:1])


def test_projection_is_a_no_op_without_project_captures():
    md = _make_metadata(project_captures=False)
    md.bind_capture_projection(_fc_blocks()[1])
    assert md.get_projected_captures(MAX_TOKENS) is None
    assert md.projected_captures is None


@pytest.mark.parametrize("bind_base_first", [True, False])
def test_graph_metadata_shares_the_projected_buffer(bind_base_first):
    md = _make_metadata(project_captures=True)
    graph_md = md.create_cuda_graph_metadata(max_batch_size=2)
    _, blocks = _fc_blocks()
    first, second = (md, graph_md) if bind_base_first else (graph_md, md)
    first.bind_capture_projection(blocks)
    second.bind_capture_projection(blocks)
    assert graph_md.projected_captures is md.projected_captures


def _prepare(worker, request_ids):
    meta = SimpleNamespace(
        request_ids=request_ids,
        num_generations=len(request_ids),
        batch_indices_cuda=torch.zeros(8, dtype=torch.int, device="cuda"),
        _dflash_worker=worker,
    )
    meta._batch_indices_buffer = DeviceCopyBuffer(meta.batch_indices_cuda)
    DFlashSpecMetadata.prepare(meta)
    torch.cuda.synchronize()
    return meta


def test_prepare_leaves_unified_draft_cache_slots_to_the_worker():
    # Main's slot mapping would evict request 5 and assign request 6.
    worker = SimpleNamespace(
        _ctx_buf_inited=True,
        _has_unified_draft_cache=lambda: True,
        _req_to_slot={5: 0},
        _free_slots=[1],
    )
    meta = _prepare(worker, [6])
    assert worker._req_to_slot == {5: 0}
    assert worker._free_slots == [1]
    assert meta.batch_indices_cuda[:1].tolist() == [0]
