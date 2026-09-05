# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate the shared sequence-length staging buffer and token gather."""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata

MAX_NUM_SEQUENCES = 8


class _Staging:
    """Bind only the production setup needed by the sequence-length test."""

    _post_init_with_buffers = TrtllmAttentionMetadata._post_init_with_buffers
    _invalidate_mla_scheduler_buffers = TrtllmAttentionMetadata._invalidate_mla_scheduler_buffers
    _snapshot_seq_lens_for_copy = TrtllmAttentionMetadata._snapshot_seq_lens_for_copy
    get_empty = staticmethod(TrtllmAttentionMetadata.get_empty)
    get_empty_like = staticmethod(TrtllmAttentionMetadata.get_empty_like)

    def __init__(self, max_num_sequences=MAX_NUM_SEQUENCES):
        self.max_num_sequences = max_num_sequences
        self.max_num_requests = max_num_sequences
        self.max_num_tokens = max_num_sequences
        self.is_cuda_graph = False
        self.workspace = None
        self.cuda_graph_workspace = None
        # These unrelated features need additional managers and buffers.
        self.kv_cache_manager = None
        self.draft_kv_cache_manager = None
        self.enable_flash_mla = False
        self.enable_helix = False
        self.enable_context_mla_with_cached_kv = False
        self._post_init_with_buffers(None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSeqLensStaging:
    def test_one_copy_transfers_both_rows(self):
        meta = _Staging()
        prompt = torch.arange(MAX_NUM_SEQUENCES, dtype=torch.int32) + 100
        kv = torch.arange(MAX_NUM_SEQUENCES, dtype=torch.int32) + 200

        meta.prompt_lens_cpu.copy_(prompt)
        meta._kv_lens_stage_cpu.copy_(kv)
        snapshot = meta._snapshot_seq_lens_for_copy()
        meta._seq_lens_stage_cuda.copy_(snapshot, non_blocking=True)
        torch.cuda.synchronize()

        torch.testing.assert_close(meta.prompt_lens_cuda.cpu(), prompt)
        torch.testing.assert_close(meta.kv_lens_cuda.cpu(), kv)

    def test_copy_source_is_an_immutable_snapshot(self):
        meta = _Staging()
        meta.prompt_lens_cpu.fill_(11)
        meta._kv_lens_stage_cpu.fill_(22)

        snapshot = meta._snapshot_seq_lens_for_copy()

        meta._seq_lens_stage_cuda.copy_(snapshot, non_blocking=True)
        meta._seq_lens_stage_cpu.zero_()
        torch.cuda.synchronize()
        torch.testing.assert_close(snapshot[0], torch.full_like(snapshot[0], 11))
        torch.testing.assert_close(snapshot[1], torch.full_like(snapshot[1], 22))
        torch.testing.assert_close(
            meta.prompt_lens_cuda, torch.full_like(meta.prompt_lens_cuda, 11)
        )
        torch.testing.assert_close(meta.kv_lens_cuda, torch.full_like(meta.kv_lens_cuda, 22))


@pytest.mark.parametrize("beam_width", [1, 2])
@pytest.mark.parametrize("slots", [[0], [3, 1], [2, 2, 0]])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
        ),
    ],
)
def test_input_id_gather_matches_advanced_indexing(slots, beam_width, device):
    """Direct ``index_select`` must preserve request-major token ordering."""
    num_slots, num_requests = 4, len(slots)
    new_tokens = torch.arange(1 * num_slots * beam_width, dtype=torch.int32, device=device).reshape(
        1, num_slots, beam_width
    )
    previous_slots = torch.tensor(slots, dtype=torch.int32, device=device)

    expected = new_tokens[:1, previous_slots, :beam_width].flatten()

    input_ids = torch.zeros(num_slots * beam_width, dtype=torch.int32, device=device)
    torch.index_select(
        new_tokens[0, :, :beam_width],
        0,
        previous_slots,
        out=input_ids[: num_requests * beam_width].view(num_requests, beam_width),
    )

    torch.testing.assert_close(input_ids[: num_requests * beam_width], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_input_id_gather_replays_with_stable_addresses():
    num_slots, beam_width = 4, 2
    new_tokens = torch.zeros((1, num_slots, beam_width), dtype=torch.int32, device="cuda")
    previous_slots = torch.tensor([3, 1, 2], dtype=torch.int32, device="cuda")
    input_ids = torch.empty(previous_slots.numel() * beam_width, dtype=torch.int32, device="cuda")
    input_address = input_ids.data_ptr()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.index_select(
            new_tokens[0],
            0,
            previous_slots,
            out=input_ids.view(previous_slots.numel(), beam_width),
        )

    for offset in (10, 100):
        new_tokens.copy_(
            torch.arange(num_slots * beam_width, dtype=torch.int32, device="cuda").view_as(
                new_tokens
            )
            + offset
        )
        graph.replay()
        torch.cuda.synchronize()
        expected = new_tokens[0, previous_slots.to(torch.int64)].flatten()
        assert input_ids.data_ptr() == input_address
        torch.testing.assert_close(input_ids, expected)
