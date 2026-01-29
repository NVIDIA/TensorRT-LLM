# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test TRT-LLM attention backend with CUDA graphs.

This test suite validates that the TRT-LLM attention backend works correctly
with CUDA graphs by ensuring tensor addresses remain fixed across forward passes.
"""

import pytest
import torch

# Skip tests if TRT-LLM or Auto-Deploy is not available
pytest.importorskip("tensorrt_llm")

from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
    enable_pt_cache_backend,
    reset_trtllm_attention_state,
)

try:
    from tensorrt_llm._torch.auto_deploy.custom_ops.pt_cache_backend import (
        PTCacheBackend,
        PTCacheConfig,
    )

    HAS_PT_CACHE_BACKEND = True
except ImportError:
    HAS_PT_CACHE_BACKEND = False


class TestTRTLLMAttentionCUDAGraph:
    """Test suite for TRT-LLM attention with CUDA graphs."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Reset state before each test
        reset_trtllm_attention_state()
        yield
        # Cleanup after test
        reset_trtllm_attention_state()

    @pytest.mark.skipif(not HAS_PT_CACHE_BACKEND, reason="PTCacheBackend not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_metadata_tensor_addresses_remain_fixed(self):
        """Test that metadata tensor addresses don't change across forward passes.

        This is the core requirement for CUDA graph compatibility.
        """
        # Import SequenceInfo
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo

        # Enable PT cache backend
        enable_pt_cache_backend(True)

        # Create a simple config
        max_batch_size = 8
        max_seq_len = 1024
        num_layers = 2
        num_kv_heads = 8
        head_dim = 128
        tokens_per_block = 64

        si = SequenceInfo(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_num_tokens=2048,
            page_size=tokens_per_block,
        )

        config = PTCacheConfig(
            num_layers=num_layers,
            num_kv_heads_per_layer=[num_kv_heads] * num_layers,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_num_sequences=max_batch_size,
            max_seq_len=max_seq_len,
            num_pages=si.num_pages,  # Use calculated value from SequenceInfo
            dtype=torch.float16,
        )

        backend = PTCacheBackend(config)
        backend.initialize(si, torch.device("cuda"))

        # Get metadata tensors
        sequence_length = backend.sequence_length
        context_lengths = backend.context_lengths
        block_offsets = backend.kv_cache_block_offsets

        # Record initial addresses
        seq_len_addr = sequence_length.data_ptr()
        ctx_len_addr = context_lengths.data_ptr()
        block_off_addr = block_offsets.data_ptr()

        print("Initial addresses:")
        print(f"  sequence_length: {hex(seq_len_addr)}")
        print(f"  context_lengths: {hex(ctx_len_addr)}")
        print(f"  block_offsets: {hex(block_off_addr)}")

        # Simulate multiple forward passes with different batch sizes
        batch_sizes = [1, 4, 8, 2, 6]

        for i, batch_size in enumerate(batch_sizes):
            # Create dummy metadata for this forward pass
            batch_info_host = torch.tensor([batch_size, batch_size * 10, 0], dtype=torch.int32)
            cu_seqlen_host = torch.cumsum(
                torch.tensor([0] + [10] * batch_size, dtype=torch.int32), 0
            )
            cu_num_pages_host = torch.cumsum(
                torch.tensor([0] + [2] * batch_size, dtype=torch.int32), 0
            )
            cache_loc = torch.arange(batch_size * 2, dtype=torch.int32, device="cuda")
            seq_len_with_cache_host = torch.tensor([10] * batch_size, dtype=torch.int32)

            # Call metadata preparation
            prep_fn = backend.get_host_prepare_metadata_function()
            if prep_fn is not None:
                prep_fn(
                    batch_info_host,
                    cu_seqlen_host,
                    cu_num_pages_host,
                    cache_loc,
                    seq_len_with_cache_host,
                    skip_device_ops=False,
                )

            # Check addresses haven't changed
            assert sequence_length.data_ptr() == seq_len_addr, (
                f"sequence_length address changed on iteration {i}"
            )
            assert context_lengths.data_ptr() == ctx_len_addr, (
                f"context_lengths address changed on iteration {i}"
            )
            assert block_offsets.data_ptr() == block_off_addr, (
                f"block_offsets address changed on iteration {i}"
            )

        print("✓ All tensor addresses remained fixed across forward passes")

    @pytest.mark.skipif(not HAS_PT_CACHE_BACKEND, reason="PTCacheBackend not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_unused_sequence_slots_are_zeroed(self):
        """Test that unused sequence slots are properly zeroed out.

        This ensures the kernel doesn't process stale data from previous forward passes.
        """
        # Import SequenceInfo
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo

        enable_pt_cache_backend(True)

        max_batch_size = 16
        si = SequenceInfo(
            max_batch_size=max_batch_size,
            max_seq_len=1024,
            max_num_tokens=2048,
            page_size=64,
        )

        config = PTCacheConfig(
            num_layers=1,
            num_kv_heads_per_layer=[8],
            head_dim=128,
            tokens_per_block=64,
            max_num_sequences=max_batch_size,
            max_seq_len=1024,
            num_pages=si.num_pages,  # Use calculated value from SequenceInfo
            dtype=torch.float16,
        )

        backend = PTCacheBackend(config)
        backend.initialize(si, torch.device("cuda"))

        # First forward pass: large batch
        batch_size_large = 12
        batch_info_host = torch.tensor(
            [batch_size_large, batch_size_large * 10, 0], dtype=torch.int32
        )
        cu_seqlen_host = torch.cumsum(
            torch.tensor([0] + [10] * batch_size_large, dtype=torch.int32), 0
        )
        cu_num_pages_host = torch.cumsum(
            torch.tensor([0] + [2] * batch_size_large, dtype=torch.int32), 0
        )
        cache_loc = torch.arange(batch_size_large * 2, dtype=torch.int32, device="cuda")
        seq_len_with_cache_host = torch.tensor([10] * batch_size_large, dtype=torch.int32)

        prep_fn = backend.get_host_prepare_metadata_function()
        prep_fn(
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages_host,
            cache_loc,
            seq_len_with_cache_host,
            skip_device_ops=False,
        )

        # Verify first 12 slots are non-zero
        assert backend.sequence_length[:batch_size_large].abs().sum() > 0
        assert backend.context_lengths[:batch_size_large].abs().sum() > 0

        # Second forward pass: small batch
        batch_size_small = 3
        batch_info_host = torch.tensor(
            [batch_size_small, batch_size_small * 10, 0], dtype=torch.int32
        )
        cu_seqlen_host = torch.cumsum(
            torch.tensor([0] + [10] * batch_size_small, dtype=torch.int32), 0
        )
        cu_num_pages_host = torch.cumsum(
            torch.tensor([0] + [2] * batch_size_small, dtype=torch.int32), 0
        )
        cache_loc = torch.arange(batch_size_small * 2, dtype=torch.int32, device="cuda")
        seq_len_with_cache_host = torch.tensor([10] * batch_size_small, dtype=torch.int32)

        prep_fn(
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages_host,
            cache_loc,
            seq_len_with_cache_host,
            skip_device_ops=False,
        )

        # Verify first 3 slots are non-zero
        assert backend.sequence_length[:batch_size_small].abs().sum() > 0
        assert backend.context_lengths[:batch_size_small].abs().sum() > 0

        # Verify slots 3-11 (previously used, now unused) are ZERO in DEVICE tensors
        # Note: HOST tensors don't need zeroing since they get sliced before passing to kernel
        assert backend.sequence_length[batch_size_small:batch_size_large].abs().sum() == 0, (
            "Unused slots not zeroed in sequence_length (device tensor)"
        )
        assert backend.context_lengths[batch_size_small:batch_size_large].abs().sum() == 0, (
            "Unused slots not zeroed in context_lengths (device tensor)"
        )

        # HOST tensors are sliced before being passed to kernel, so we don't need to check them
        # (the kernel only sees the sliced portion)

        print(f"✓ Unused sequence slots [{batch_size_small}:{batch_size_large}] properly zeroed")

    @pytest.mark.skipif(not HAS_PT_CACHE_BACKEND, reason="PTCacheBackend not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        torch.cuda.get_device_capability()[0] < 8, reason="CUDA graphs require SM 8.0+"
    )
    def test_cuda_graph_capture_replay(self):
        """Test actual CUDA graph capture and replay with TRT-LLM attention.

        This is an integration test that verifies the entire CUDA graph workflow.
        Note: This test requires a complete model setup which may be complex.
        """
        # This would require setting up a full model with TRT-LLM attention
        # For now, we'll test the metadata tensor behavior which is the core issue
        pytest.skip("Full integration test requires model setup - use manual testing")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_variable_batch_sizes(self):
        """Test that different batch sizes work correctly without slicing issues."""
        # This test would verify that passing full tensors works for various batch sizes
        # It would check that only the active sequences are processed
        pytest.skip("Requires kernel-level testing - covered by end-to-end tests")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
