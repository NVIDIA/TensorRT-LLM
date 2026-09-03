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
"""Unit tests for Mamba2 metadata preparation optimizations."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.mamba import mamba2_metadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
    REPLAY_WORK_CACHE_BUF_IDX,
    REPLAY_WORK_CACHE_SLOT,
    REPLAY_WORK_PNAT,
    REPLAY_WORK_POSITION_IN_DECODE_BATCH,
    Mamba2Metadata,
    _build_replay_work_items_torch,
    _build_replay_work_items_triton,
    cu_seqlens_to_chunk_indices_offsets,
    cu_seqlens_to_chunk_indices_offsets_triton,
)
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    MIN_REPLAY_HISTORY_SIZE,
    ReplayStateUpdateMetadata,
)

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for triton kernels",
)


class _GdnReplayCacheManager:
    use_replay_state_update = True
    use_gdn_cached_replay_all_layer_commit = True

    def __init__(self, prev_num_accepted_tokens, cache_buf_idx):
        self.prev_num_accepted_tokens = prev_num_accepted_tokens
        self.cache_buf_idx = cache_buf_idx

    def get_replay_state_update_metadata(self):
        return ReplayStateUpdateMetadata(
            prev_num_accepted_tokens=self.prev_num_accepted_tokens,
            cache_buf_idx=self.cache_buf_idx,
            replay_step_width=6,
            replay_history_size=MIN_REPLAY_HISTORY_SIZE,
        )


def _torch_reference_work_items(state_indices, prev_num_accepted_tokens, cache_buf_idx):
    """Run the production ATen path into fresh buffers.

    The Triton kernel's contract is that it reproduces this path exactly, so
    this path -- not a third hand-written copy -- is what it is compared to.
    """
    num_decodes = state_indices.numel()
    work_items = torch.zeros(num_decodes, 4, dtype=torch.int32, device="cuda")
    n_writes = torch.zeros(1, dtype=torch.int32, device="cuda")
    _build_replay_work_items_torch(
        state_indices,
        prev_num_accepted_tokens,
        cache_buf_idx,
        work_items,
        n_writes,
        6,
        MIN_REPLAY_HISTORY_SIZE,
    )
    return work_items, n_writes


@skip_no_cuda
class TestCuSeqlensToChunkIndicesOffsets:
    """Tests for cu_seqlens_to_chunk_indices_offsets_triton function."""

    def test_empty_sequence(self):
        """Test with empty cu_seqlens (no sequences)."""
        cu_seqlens = torch.tensor([0], dtype=torch.int, device="cuda")
        chunk_size = 8

        indices_triton, offsets_triton = cu_seqlens_to_chunk_indices_offsets_triton(
            cu_seqlens, chunk_size
        )

        assert indices_triton.numel() == 0
        assert offsets_triton.numel() == 0

    def test_single_sequence_aligned(self):
        """Test with a single sequence that aligns with chunk size."""
        cu_seqlens = torch.tensor([0, 16], dtype=torch.int, device="cuda")
        chunk_size = 8

        indices_ref, offsets_ref = cu_seqlens_to_chunk_indices_offsets(cu_seqlens, chunk_size)
        indices_triton, offsets_triton = cu_seqlens_to_chunk_indices_offsets_triton(
            cu_seqlens, chunk_size
        )

        torch.testing.assert_close(indices_triton, indices_ref)
        torch.testing.assert_close(offsets_triton, offsets_ref)


@skip_no_cuda
class TestMamba2Metadata:
    def test_prepare_handles_tensor_cached_tokens(self):
        metadata = Mamba2Metadata(max_batch_size=4, chunk_size=8)
        seq_lens = torch.tensor([4, 3], dtype=torch.int)
        attn_metadata = SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cuda=seq_lens.cuda(),
            num_contexts=2,
            num_ctx_tokens=7,
            kv_cache_manager=None,
            request_ids=None,
            kv_cache_params=SimpleNamespace(
                num_cached_tokens_per_seq=torch.tensor([0, 5], dtype=torch.int),
            ),
        )

        metadata.prepare(attn_metadata)

        assert metadata.has_initial_states_cpu.is_pinned()
        torch.testing.assert_close(metadata.has_initial_states_cpu[:2], torch.tensor([False, True]))
        torch.testing.assert_close(
            metadata.has_initial_states[:2].cpu(), torch.tensor([False, True])
        )
        assert metadata.use_initial_states is True
        assert metadata.chunk_indices is not None
        assert metadata.chunk_offsets is not None

    def test_prepare_replay_work_items_write_first(self):
        class ReplayCacheManager:
            use_replay_state_update = True

            def __init__(self):
                self.state_indices = [0, 3, 1, 4, 2]
                self.prev_num_accepted_tokens = torch.tensor(
                    [0, 4, 10, 11, 20], dtype=torch.int32, device="cuda"
                )
                self.cache_buf_idx = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32, device="cuda")

            def get_state_indices(self, request_ids, is_padding):
                return self.state_indices[: len(request_ids)]

            def get_replay_state_update_metadata(self):
                return ReplayStateUpdateMetadata(
                    prev_num_accepted_tokens=self.prev_num_accepted_tokens,
                    cache_buf_idx=self.cache_buf_idx,
                    replay_step_width=6,
                    replay_history_size=MIN_REPLAY_HISTORY_SIZE,
                )

        metadata = Mamba2Metadata(max_batch_size=5, chunk_size=8)
        seq_lens = torch.tensor([2, 7, 7, 7, 7], dtype=torch.int)
        attn_metadata = SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cuda=seq_lens.cuda(),
            num_contexts=1,
            num_ctx_tokens=2,
            kv_cache_manager=ReplayCacheManager(),
            request_ids=[10, 11, 12, 13, 14],
            kv_cache_params=SimpleNamespace(
                num_cached_tokens_per_seq=torch.tensor([0], dtype=torch.int),
            ),
        )

        metadata.prepare(attn_metadata)

        expected = torch.tensor(
            [
                [0, 3, 11, 1],
                [2, 4, 20, 0],
                [1, 1, 4, 1],
                [3, 2, 10, 0],
            ],
            dtype=torch.int32,
            device="cuda",
        )
        actual = metadata.replay_work_items[:4]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            metadata.replay_n_writes.cpu(), torch.tensor([2], dtype=torch.int32)
        )
        assert actual[0, REPLAY_WORK_POSITION_IN_DECODE_BATCH] == 0
        assert actual[0, REPLAY_WORK_CACHE_SLOT] == 3
        assert actual[0, REPLAY_WORK_PNAT] == 11
        assert actual[0, REPLAY_WORK_CACHE_BUF_IDX] == 1

    @pytest.mark.parametrize("num_decodes", [16, 17, 40, 255, 256])
    def test_replay_work_items_triton_matches_torch(self, num_decodes):
        """The two builders must agree everywhere the dispatch may pick either.

        num_decodes 17 and 255 are not powers of two, so the kernel runs with
        masked-off lanes -- those must contribute nothing to the write-first
        prefix sum.
        """
        num_slots = num_decodes + 7
        prev_num_accepted_tokens = torch.arange(num_slots, dtype=torch.int32, device="cuda") % 21
        cache_buf_idx = torch.arange(num_slots, dtype=torch.int32, device="cuda") % 2
        state_indices = torch.randperm(num_slots, device="cuda")[:num_decodes].to(torch.int32)

        triton_items = torch.zeros(num_decodes, 4, dtype=torch.int32, device="cuda")
        triton_n_writes = torch.zeros(1, dtype=torch.int32, device="cuda")
        _build_replay_work_items_triton(
            state_indices,
            prev_num_accepted_tokens,
            cache_buf_idx,
            triton_items,
            triton_n_writes,
            6,
            MIN_REPLAY_HISTORY_SIZE,
        )
        torch_items, torch_n_writes = _torch_reference_work_items(
            state_indices, prev_num_accepted_tokens, cache_buf_idx
        )

        torch.testing.assert_close(triton_items, torch_items)
        torch.testing.assert_close(triton_n_writes, torch_n_writes)

    def test_prepare_replay_work_items_uses_the_selected_builder(self):
        """The entry point must feed the builder the decode slice, not the whole batch."""
        num_contexts, num_decodes = 3, 40
        num_slots = num_contexts + num_decodes + 7
        prev_num_accepted_tokens = torch.arange(num_slots, dtype=torch.int32, device="cuda") % 21
        cache_buf_idx = torch.arange(num_slots, dtype=torch.int32, device="cuda") % 2
        state_indices = torch.randperm(num_slots, device="cuda")[: num_contexts + num_decodes].to(
            torch.int32
        )
        manager = _GdnReplayCacheManager(prev_num_accepted_tokens, cache_buf_idx)
        metadata = Mamba2Metadata(max_batch_size=num_contexts + num_decodes, chunk_size=8)
        metadata.state_indices[: num_contexts + num_decodes].copy_(state_indices)

        metadata._prepare_replay_work_items(manager, num_contexts + num_decodes, num_contexts)
        expected_items, expected_n_writes = _torch_reference_work_items(
            state_indices[num_contexts:], prev_num_accepted_tokens, cache_buf_idx
        )

        torch.testing.assert_close(metadata.replay_work_items[:num_decodes], expected_items)
        torch.testing.assert_close(metadata.replay_n_writes, expected_n_writes)

    @pytest.mark.parametrize(
        ("num_decodes", "expect_fused"),
        [
            pytest.param(8, False, id="below-partition-min"),
            pytest.param(16, True, id="partition-min"),
            pytest.param(256, True, id="fused-max"),
            pytest.param(257, False, id="above-fused-max"),
        ],
    )
    def test_prepare_gdn_replay_work_items_dispatch_boundary(
        self, monkeypatch, num_decodes, expect_fused
    ):
        """Pin which batch sizes reach the single-launch kernel.

        The equivalence tests above pass on either side of this boundary, so
        without this the fused launch could silently stop being used.
        """
        num_slots = num_decodes + 7
        prev_num_accepted_tokens = torch.arange(num_slots, dtype=torch.int32, device="cuda") % 21
        cache_buf_idx = torch.arange(num_slots, dtype=torch.int32, device="cuda") % 2
        manager = _GdnReplayCacheManager(prev_num_accepted_tokens, cache_buf_idx)
        metadata = Mamba2Metadata(max_batch_size=num_decodes, chunk_size=8)
        metadata.state_indices.copy_(torch.arange(num_decodes, dtype=torch.int32, device="cuda"))

        launches = []
        original_kernel = mamba2_metadata._prepare_gdn_replay_work_items_kernel

        class _CountingKernel:
            def __getitem__(self, grid):
                launches.append(grid)
                return original_kernel[grid]

        monkeypatch.setattr(
            mamba2_metadata, "_prepare_gdn_replay_work_items_kernel", _CountingKernel()
        )

        metadata._prepare_replay_work_items(manager, num_decodes, 0)

        assert bool(launches) is expect_fused

    def test_prepare_gdn_replay_work_items_cuda_graph_replay(self):
        num_decodes = 40
        num_slots = num_decodes + 7
        prev_num_accepted_tokens = torch.arange(num_slots, dtype=torch.int32, device="cuda") % 21
        cache_buf_idx = torch.arange(num_slots, dtype=torch.int32, device="cuda") % 2
        manager = _GdnReplayCacheManager(prev_num_accepted_tokens, cache_buf_idx)
        metadata = Mamba2Metadata(max_batch_size=num_decodes, chunk_size=8)
        metadata.state_indices.copy_(torch.arange(num_decodes, dtype=torch.int32, device="cuda"))

        metadata._prepare_replay_work_items(manager, num_decodes, 0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            metadata._prepare_replay_work_items(manager, num_decodes, 0)

        updated_state_indices = torch.arange(
            num_slots - 1,
            num_slots - num_decodes - 1,
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        metadata.state_indices.copy_(updated_state_indices)
        prev_num_accepted_tokens.copy_(
            (torch.arange(num_slots, dtype=torch.int32, device="cuda") * 7) % 21
        )
        cache_buf_idx.bitwise_xor_(1)
        graph.replay()
        torch.cuda.synchronize()

        expected_items, expected_n_writes = _torch_reference_work_items(
            updated_state_indices, prev_num_accepted_tokens, cache_buf_idx
        )
        torch.testing.assert_close(metadata.replay_work_items[:num_decodes], expected_items)
        torch.testing.assert_close(metadata.replay_n_writes, expected_n_writes)

    def test_single_sequence_unaligned(self):
        """Test with a single sequence that doesn't align with chunk size."""
        cu_seqlens = torch.tensor([0, 10], dtype=torch.int, device="cuda")
        chunk_size = 8

        indices_ref, offsets_ref = cu_seqlens_to_chunk_indices_offsets(cu_seqlens, chunk_size)
        indices_triton, offsets_triton = cu_seqlens_to_chunk_indices_offsets_triton(
            cu_seqlens, chunk_size
        )

        torch.testing.assert_close(indices_triton, indices_ref)
        torch.testing.assert_close(offsets_triton, offsets_ref)

    def test_two_sequences_aligned(self):
        """Test with two sequences, both aligned with chunk boundaries."""
        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int, device="cuda")
        chunk_size = 8

        indices_ref, offsets_ref = cu_seqlens_to_chunk_indices_offsets(cu_seqlens, chunk_size)
        indices_triton, offsets_triton = cu_seqlens_to_chunk_indices_offsets_triton(
            cu_seqlens, chunk_size
        )

        torch.testing.assert_close(indices_triton, indices_ref)
        torch.testing.assert_close(offsets_triton, offsets_ref)

    def test_two_sequences_misaligned(self):
        """Test with two sequences where second starts at misaligned position."""
        # Example from docstring: cu_seqlens = [0, 5, 10], chunk_size = 8
        # -> chunk_indices = [0, 0, 1], chunk_offsets = [0, 5, 0]
        cu_seqlens = torch.tensor([0, 5, 10], dtype=torch.int, device="cuda")
        chunk_size = 8

        indices_ref, offsets_ref = cu_seqlens_to_chunk_indices_offsets(cu_seqlens, chunk_size)
        indices_triton, offsets_triton = cu_seqlens_to_chunk_indices_offsets_triton(
            cu_seqlens, chunk_size
        )

        # Verify against expected values from docstring
        expected_indices = torch.tensor([0, 0, 1], dtype=torch.int, device="cuda")
        expected_offsets = torch.tensor([0, 5, 0], dtype=torch.int, device="cuda")

        torch.testing.assert_close(indices_ref, expected_indices)
        torch.testing.assert_close(offsets_ref, expected_offsets)

        torch.testing.assert_close(indices_triton, indices_ref)
        torch.testing.assert_close(offsets_triton, offsets_ref)

    @pytest.mark.parametrize("chunk_size", [8, 16, 32, 64, 128])
    def test_multiple_sequences_various_chunk_sizes(self, chunk_size):
        """Test with multiple sequences and various chunk sizes."""
        # Create sequences with varying lengths
        cu_seqlens = torch.tensor([0, 10, 25, 40, 60, 75], dtype=torch.int, device="cuda")

        indices_ref, offsets_ref = cu_seqlens_to_chunk_indices_offsets(cu_seqlens, chunk_size)
        indices_triton, offsets_triton = cu_seqlens_to_chunk_indices_offsets_triton(
            cu_seqlens, chunk_size
        )

        torch.testing.assert_close(indices_triton, indices_ref)
        torch.testing.assert_close(offsets_triton, offsets_ref)

    def test_all_sequences_within_one_chunk(self):
        """Test when all sequences fit within a single chunk."""
        cu_seqlens = torch.tensor([0, 2, 4, 6], dtype=torch.int, device="cuda")
        chunk_size = 64  # Large chunk size

        indices_ref, offsets_ref = cu_seqlens_to_chunk_indices_offsets(cu_seqlens, chunk_size)
        indices_triton, offsets_triton = cu_seqlens_to_chunk_indices_offsets_triton(
            cu_seqlens, chunk_size
        )

        torch.testing.assert_close(indices_triton, indices_ref)
        torch.testing.assert_close(offsets_triton, offsets_ref)
