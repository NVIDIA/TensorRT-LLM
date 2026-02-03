"""
Unit tests for suffix automaton speculative decoding.

Tests both the native CUDA kernel and Python fallback implementations.
"""

import pytest
import torch

# Import the suffix automaton module
from tensorrt_llm._torch.speculative import suffix_automaton as sa


class TestSuffixAutomatonBasic:
    """Basic functionality tests."""

    def test_init(self):
        """Test initialization."""
        sa.init(max_num_requests=16, max_seq_len=1024)
        assert sa._global_sa_manager is not None
        sa.shutdown()

    def test_add_request(self):
        """Test adding a request with context tokens."""
        sa.init(max_num_requests=16, max_seq_len=1024)
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]  # Has repeating pattern
        sa.add_request(0, context_tokens)
        sa.shutdown()

    def test_prepare_and_extend(self):
        """Test prepare and extend operations."""
        sa.init(max_num_requests=16, max_seq_len=1024)

        # Add request with context that has a repeating pattern
        # [1, 2, 3, 4, 5, 1, 2, 3] - the suffix [1, 2, 3] appears earlier
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        sa.add_request(0, context_tokens)

        # Prepare for batch processing
        request_ids = [0]
        max_draft_len = 4
        sa.prepare(request_ids, max_draft_len)

        # Create input tensors
        batch_size = 1
        accepted_tokens = torch.tensor([[6, 0, 0, 0, 0]], dtype=torch.int32, device='cuda')
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device='cuda')

        # Create output tensors
        match_len_out = torch.zeros((batch_size,), dtype=torch.int32, device='cuda')
        draft_tokens_out = torch.zeros((batch_size, max_draft_len), dtype=torch.int32, device='cuda')

        # Extend
        sa.extend(match_len_out, draft_tokens_out, accepted_tokens, num_accepted_tokens)

        # Note: Results depend on whether native kernel or Python fallback is used
        # The important thing is that it doesn't crash
        print(f"match_len: {match_len_out}")
        print(f"draft_tokens: {draft_tokens_out}")

        sa.shutdown()


class TestSuffixAutomatonManager:
    """Tests for SuffixAutomatonManager class."""

    def test_manager_creation(self):
        """Test manager creation."""
        config = sa.SAConfig(max_seq_len=1024, max_slots=16)
        manager = sa.SuffixAutomatonManager(config, max_num_requests=16)
        assert manager is not None

        # Check if using native or fallback
        print(f"Using native kernel: {manager._use_native}")
        manager.shutdown()

    def test_manager_add_remove(self):
        """Test adding and removing requests."""
        config = sa.SAConfig(max_seq_len=1024, max_slots=16)
        manager = sa.SuffixAutomatonManager(config, max_num_requests=16)

        # Add requests
        manager.add_request(0, [1, 2, 3, 4, 5])
        manager.add_request(1, [10, 20, 30, 40, 50])

        assert 0 in manager._request_to_slot
        assert 1 in manager._request_to_slot

        # Remove a request
        manager.remove_request(0)
        assert 0 not in manager._request_to_slot
        assert 1 in manager._request_to_slot

        manager.shutdown()

    def test_manager_extend(self):
        """Test extend operation."""
        config = sa.SAConfig(max_seq_len=1024, max_slots=16)
        manager = sa.SuffixAutomatonManager(config, max_num_requests=16)

        # Add request with repeating pattern
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Extend
        accepted_tokens = torch.tensor([[6, 0, 0, 0, 0]], dtype=torch.int32, device='cuda')
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device='cuda')

        match_len, draft_tokens = manager.extend(
            request_ids, accepted_tokens, num_accepted_tokens, max_draft_len
        )

        print(f"match_len: {match_len}")
        print(f"draft_tokens: {draft_tokens}")

        manager.shutdown()


class TestCUDAGraphCompatibility:
    """Tests for CUDA graph compatibility."""

    @pytest.mark.skipif(
        not sa.is_native_available(),
        reason="Native kernel not available"
    )
    def test_cuda_graph_capture_native(self):
        """Test that native extend works during CUDA graph capture."""
        config = sa.SAConfig(max_seq_len=1024, max_slots=16)
        manager = sa.SuffixAutomatonManager(config, max_num_requests=16)

        # Add request
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare (must be done before capture)
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Create input/output tensors
        accepted_tokens = torch.tensor([[6, 0, 0, 0, 0]], dtype=torch.int32, device='cuda')
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device='cuda')

        # Warmup
        for _ in range(3):
            manager.extend(request_ids, accepted_tokens, num_accepted_tokens, max_draft_len)

        # Capture CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            match_len, draft_tokens = manager.extend(
                request_ids, accepted_tokens, num_accepted_tokens, max_draft_len
            )

        # Replay
        g.replay()

        print(f"CUDA graph capture succeeded with native kernel")
        print(f"match_len: {match_len}")
        print(f"draft_tokens: {draft_tokens}")

        manager.shutdown()

    def test_cuda_graph_capture_fallback(self):
        """Test that Python fallback returns zeros during CUDA graph capture."""
        # Temporarily force fallback mode
        original_native = sa._sa_native
        sa._sa_native = None

        try:
            config = sa.SAConfig(max_seq_len=1024, max_slots=16)
            manager = sa.SuffixAutomatonManager(config, max_num_requests=16)
            assert not manager._use_native, "Should be using fallback"

            # Add request
            context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
            manager.add_request(0, context_tokens)

            # Prepare
            request_ids = [0]
            max_draft_len = 4
            manager.prepare(request_ids, max_draft_len)

            # Create input tensors
            accepted_tokens = torch.tensor([[6, 0, 0, 0, 0]], dtype=torch.int32, device='cuda')
            num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device='cuda')

            # Warmup
            for _ in range(3):
                manager.extend(request_ids, accepted_tokens, num_accepted_tokens, max_draft_len)

            # Capture CUDA graph - should not crash
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                match_len, draft_tokens = manager.extend(
                    request_ids, accepted_tokens, num_accepted_tokens, max_draft_len
                )

            # Check that zeros were returned (fallback behavior during capture)
            assert match_len.sum().item() == 0, "Fallback should return zeros during capture"

            print("CUDA graph capture succeeded with Python fallback (returned zeros)")

            manager.shutdown()
        finally:
            # Restore native
            sa._sa_native = original_native


class TestNativeKernelAvailability:
    """Tests for native kernel availability."""

    def test_is_native_available(self):
        """Test native availability check."""
        available = sa.is_native_available()
        print(f"Native kernel available: {available}")

        if available:
            # Verify we can access the native module
            from tensorrt_llm._C._internal import suffix_automaton as native
            print(f"MAX_SEQUENCE_LENGTH: {native.MAX_SEQUENCE_LENGTH}")
            print(f"MAX_SLOTS: {native.MAX_SLOTS}")
            print(f"STATE_SIZE_BYTES: {native.STATE_SIZE_BYTES}")


if __name__ == "__main__":
    # Run basic tests
    print("=" * 60)
    print("Testing suffix automaton module")
    print("=" * 60)

    print("\n--- Native kernel availability ---")
    test = TestNativeKernelAvailability()
    test.test_is_native_available()

    print("\n--- Basic tests ---")
    test = TestSuffixAutomatonBasic()
    test.test_init()
    test.test_add_request()
    test.test_prepare_and_extend()

    print("\n--- Manager tests ---")
    test = TestSuffixAutomatonManager()
    test.test_manager_creation()
    test.test_manager_add_remove()
    test.test_manager_extend()

    print("\n--- CUDA graph compatibility tests ---")
    test = TestCUDAGraphCompatibility()
    if sa.is_native_available():
        test.test_cuda_graph_capture_native()
    test.test_cuda_graph_capture_fallback()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
