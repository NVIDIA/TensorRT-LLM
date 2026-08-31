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
"""The per-step staging that keeps one transfer where there used to be two.

``TrtllmAttentionMetadata`` refreshes the prompt lens and the KV lens once per
``prepare()``. Both are int32 vectors over the batch, so they share one device
buffer and one pinned host buffer and go up in a single copy. These tests pin
the aliasing that makes the single copy cover both, and the equivalence of the
gather that fills the model's input-id buffer.
"""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata

MAX_NUM_SEQUENCES = 8


class _Staging:
    """The state ``_post_init_with_buffers`` touches, with the real method bound.

    Deliberately not a full ``TrtllmAttentionMetadata``: that needs a KV cache
    manager and a resource pool, neither of which the staging setup reads.
    Binding the production function keeps this a test of the shipped code
    rather than of a reimplementation.
    """

    _post_init_with_buffers = TrtllmAttentionMetadata._post_init_with_buffers
    _invalidate_mla_scheduler_buffers = TrtllmAttentionMetadata._invalidate_mla_scheduler_buffers
    get_empty = staticmethod(TrtllmAttentionMetadata.get_empty)
    get_empty_like = staticmethod(TrtllmAttentionMetadata.get_empty_like)

    def __init__(self, max_num_sequences=MAX_NUM_SEQUENCES):
        self.max_num_sequences = max_num_sequences
        self.max_num_requests = max_num_sequences
        self.max_num_tokens = max_num_sequences
        self.is_cuda_graph = False
        self.workspace = None
        self.cuda_graph_workspace = None
        # No KV cache manager, no MLA and no helix: those branches allocate
        # unrelated buffers and read state this stand-in does not model.
        self.kv_cache_manager = None
        self.draft_kv_cache_manager = None
        self.enable_flash_mla = False
        self.enable_helix = False
        self.enable_context_mla_with_cached_kv = False
        # buffers=None makes get_empty allocate directly instead of pooling.
        self._post_init_with_buffers(None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSeqLensStaging:
    def test_device_lens_share_one_buffer(self):
        meta = _Staging()
        stage = meta._seq_lens_stage_cuda

        assert stage.shape == (2, MAX_NUM_SEQUENCES)
        assert stage.dtype == torch.int32
        # Both exposed views must be rows of the staging buffer, so that one
        # copy of the parent lands both of them.
        assert meta.prompt_lens_cuda.data_ptr() == stage[0].data_ptr()
        assert meta.kv_lens_cuda.data_ptr() == stage[1].data_ptr()
        assert meta.prompt_lens_cuda.is_contiguous()
        assert meta.kv_lens_cuda.is_contiguous()
        assert meta.prompt_lens_cuda.shape == (MAX_NUM_SEQUENCES,)
        assert meta.kv_lens_cuda.shape == (MAX_NUM_SEQUENCES,)

    def test_host_stages_share_one_pinned_buffer(self):
        meta = _Staging()
        stage = meta._seq_lens_stage_cpu

        assert stage.shape == (2, MAX_NUM_SEQUENCES)
        assert stage.dtype == torch.int32
        assert not stage.is_cuda
        assert meta.prompt_lens_cpu.data_ptr() == stage[0].data_ptr()
        assert meta._kv_lens_stage_cpu.data_ptr() == stage[1].data_ptr()
        # self.kv_lens carries num_extra_kv_tokens, which the device-side lens
        # exclude, so it must NOT be the staging row.
        assert meta.kv_lens.data_ptr() != stage[1].data_ptr()

    def test_one_copy_transfers_both_rows(self):
        meta = _Staging()
        prompt = torch.arange(MAX_NUM_SEQUENCES, dtype=torch.int32) + 100
        kv = torch.arange(MAX_NUM_SEQUENCES, dtype=torch.int32) + 200

        meta.prompt_lens_cpu.copy_(prompt)
        meta._kv_lens_stage_cpu.copy_(kv)
        meta._seq_lens_stage_cuda.copy_(meta._seq_lens_stage_cpu, non_blocking=True)
        torch.cuda.synchronize()

        torch.testing.assert_close(meta.prompt_lens_cuda.cpu(), prompt)
        torch.testing.assert_close(meta.kv_lens_cuda.cpu(), kv)


@pytest.mark.parametrize("beam_width", [1, 2])
@pytest.mark.parametrize("slots", [[0], [3, 1], [2, 2, 0]])
def test_input_id_gather_matches_advanced_indexing(slots, beam_width):
    """``index_select(out=...)`` must reproduce the indexing it replaced.

    The steady-state generation path gathers one token per request out of the
    sampler's ``[step, slot, beam]`` buffer and writes it into the flat input-id
    buffer. It used to index into a temporary and copy that over; the gather now
    writes the destination directly, and the two must agree element for element,
    including the request-major / beam-minor order of the flattened result.
    """
    num_slots, num_requests = 4, len(slots)
    new_tokens = torch.arange(1 * num_slots * beam_width, dtype=torch.int32).reshape(
        1, num_slots, beam_width
    )
    previous_slots = torch.tensor(slots, dtype=torch.int32)

    expected = new_tokens[:1, previous_slots, :beam_width].flatten()

    input_ids = torch.zeros(num_requests * beam_width, dtype=torch.int32)
    torch.index_select(
        new_tokens[0, :, :beam_width],
        0,
        previous_slots,
        out=input_ids.view(num_requests, beam_width),
    )

    torch.testing.assert_close(input_ids, expected)
