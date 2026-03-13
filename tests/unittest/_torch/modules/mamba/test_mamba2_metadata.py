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

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
    cu_seqlens_to_chunk_indices_offsets,
    cu_seqlens_to_chunk_indices_offsets_triton,
)

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for triton kernels",
)


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
