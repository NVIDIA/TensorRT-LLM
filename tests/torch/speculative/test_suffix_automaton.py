"""Unit tests for suffix automaton speculative decoding.

Tests the native CUDA kernel implementation.
"""

import pytest
import torch

from tensorrt_llm._torch.speculative import suffix_automaton as sa
from tensorrt_llm._torch.speculative.suffix_automaton import (
    SAConfig,
    SuffixAutomatonManager,
)  # noqa: I001

# Skip all tests if native kernel is not available
pytestmark = pytest.mark.skipif(
    not sa.is_native_available(), reason="Native suffix automaton kernel not available"
)


class TestSuffixAutomatonManager:
    """Tests for SuffixAutomatonManager class."""

    def test_manager_creation(self):
        """Test manager creation."""
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)
        assert manager is not None
        manager.shutdown()

    def test_manager_add_remove(self):
        """Test adding and removing requests."""
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

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
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        # Add request with repeating pattern
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Extend
        accepted_tokens = torch.tensor([[6, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend(
            request_ids, accepted_tokens, num_accepted_tokens, max_draft_len
        )

        print(f"match_len: {match_len}")
        print(f"draft_tokens: {draft_tokens}")

        manager.shutdown()


class TestCUDAGraphCompatibility:
    """Tests for CUDA graph compatibility."""

    def test_cuda_graph_capture(self):
        """Test that native extend works during CUDA graph capture."""
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        # Add request
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare (must be done before capture)
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Create input/output tensors
        accepted_tokens = torch.tensor([[6, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

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

        print("CUDA graph capture succeeded with native kernel")
        print(f"match_len: {match_len}")
        print(f"draft_tokens: {draft_tokens}")

        manager.shutdown()


class TestExtendNgram:
    """Tests for extend_ngram() batched method - CUDA graph compatible."""

    def test_extend_ngram_longest_match(self):
        """Test extend_ngram with longest match mode (max_ngram_size=-1)."""
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        # Add request with repeating pattern
        # [0, 2, 3, 4, 10, 1, 2, 3, 4, 5, 1, 2, 3] - pattern [1, 2, 3] appears twice
        # Pattern [1, 2, 3, 4] appears at positions 5-8
        context_tokens = [0, 2, 3, 4, 10, 1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Extend with token 4 - this continues the pattern!
        # New sequence: [0, 2, 3, 4, 10, 1, 2, 3, 4, 5, 1, 2, 3, 4]
        # Suffix [1, 2, 3, 4] matches at position 5-8
        # Continuation starts at position 9, draft tokens are [5, 1, 2, 3]
        accepted_tokens = torch.tensor([[4, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        # Use longest match mode
        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=-1,  # Longest match
        )

        print(f"extend_ngram (longest): match_len={match_len}, draft_tokens={draft_tokens}")

        # Verify results
        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        # The longest match should be [1, 2, 3, 4] (length 4)
        assert match_len_val == 4, f"Expected longest match of 4, got {match_len_val}"

        # Draft tokens should be [5, 1, 2, 3] (tokens after the match at pos 5-8)
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [5, 1, 2, 3], f"Expected [5, 1, 2, 3], got {draft_list}"

        manager.shutdown()

    def test_extend_ngram_fixed_size(self):
        """Test extend_ngram with fixed-size ngram matching."""
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        # Add request with repeating pattern
        # [0, 2, 3, 4, 10, 1, 2, 3, 4, 5, 1, 2, 3]
        # Pattern [2, 3, 4] appears at positions 1-3 and 6-8
        context_tokens = [0, 2, 3, 4, 10, 1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Extend with token 4 - continues the pattern
        # New sequence: [0, 2, 3, 4, 10, 1, 2, 3, 4, 5, 1, 2, 3, 4]
        accepted_tokens = torch.tensor([[4, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        # Use fixed-size ngram mode (max_ngram_size=3 means try 3, 2, 1)
        # Should find 3-gram [2, 3, 4] matching at position 1-3 (leftmost match)
        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=3,  # Try 3-gram, 2-gram, 1-gram
        )

        print(f"extend_ngram (fixed 3): match_len={match_len}, draft_tokens={draft_tokens}")

        # Verify results
        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        # With max_ngram_size=3, should find 3-gram match
        assert match_len_val == 3, f"Expected 3-gram match, got {match_len_val}"

        # Draft tokens after 3-gram [2, 3, 4] at pos 1-3 are [10, 1, 2, 3]
        # (leftmost match returns tokens starting at position 4)
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [10, 1, 2, 3], f"Expected [10, 1, 2, 3], got {draft_list}"

        manager.shutdown()

    def test_extend_ngram_no_match(self):
        """Test extend_ngram when no match exists."""
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        # Add request with repeating pattern
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Extend with unique token 99 - no pattern will match
        # New sequence: [1, 2, 3, 4, 5, 1, 2, 3, 99]
        accepted_tokens = torch.tensor([[99, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids, accepted_tokens, num_accepted_tokens, max_draft_len, max_ngram_size=-1
        )

        print(f"extend_ngram (no match): match_len={match_len}, draft_tokens={draft_tokens}")

        # Verify no match found
        match_len_val = match_len[0].item()
        assert match_len_val == 0, f"Expected no match, got match_len={match_len_val}"

        manager.shutdown()

    def test_extend_ngram_batch(self):
        """Test extend_ngram with multiple requests in batch."""
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        # Add multiple requests with different patterns
        # Request 0: [1, 2, 3, 4, 5, 1, 2, 3] + [4] -> match [1,2,3,4] at pos 0
        manager.add_request(0, [1, 2, 3, 4, 5, 1, 2, 3])
        # Request 1: [10, 20, 30, 10, 20] + [30] -> match [10,20,30] at pos 0
        manager.add_request(1, [10, 20, 30, 10, 20])
        # Request 2: [100, 200, 300, 400] + [500] -> no match (unique token)
        manager.add_request(2, [100, 200, 300, 400])

        # Prepare
        request_ids = [0, 1, 2]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Extend with tokens that create patterns for requests 0 and 1
        accepted_tokens = torch.tensor(
            [
                [4, 0, 0, 0, 0],  # Continues [1,2,3,4] pattern
                [30, 0, 0, 0, 0],  # Continues [10,20,30] pattern
                [500, 0, 0, 0, 0],  # Unique, no match
            ],
            dtype=torch.int32,
            device="cuda",
        )
        num_accepted_tokens = torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids, accepted_tokens, num_accepted_tokens, max_draft_len, max_ngram_size=-1
        )

        print(f"extend_ngram batch: match_len={match_len}")
        print(f"extend_ngram batch: draft_tokens={draft_tokens}")

        # Verify results
        match_lens = match_len.cpu().tolist()

        # Request 0: should find match (longest is 4: [1,2,3,4])
        assert match_lens[0] == 4, f"Request 0: expected match_len=4, got {match_lens[0]}"
        # Draft tokens should be [5, 1, 2, 3]
        draft_0 = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_0 == [5, 1, 2, 3], f"Request 0: expected [5,1,2,3], got {draft_0}"

        # Request 1: should find match (longest is 3: [10,20,30])
        assert match_lens[1] == 3, f"Request 1: expected match_len=3, got {match_lens[1]}"
        # Draft tokens should be [10, 20, 30, 0] (only 3 tokens available after pos 3)
        draft_1 = draft_tokens[1, :3].cpu().tolist()
        assert draft_1 == [10, 20, 30], f"Request 1: expected [10,20,30], got {draft_1}"

        # Request 2: no match
        assert match_lens[2] == 0, f"Request 2: expected match_len=0, got {match_lens[2]}"

        manager.shutdown()

    def test_extend_ngram_cuda_graph(self):
        """Test that extend_ngram works with CUDA graph capture."""
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        # Add request with repeating pattern
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare (must be done before capture)
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Create input/output tensors - use token 4 to create pattern match
        accepted_tokens = torch.tensor([[4, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        # Warmup - each call extends SA state, so we'll reset after
        for _ in range(3):
            manager.extend_ngram(
                request_ids, accepted_tokens, num_accepted_tokens, max_draft_len, max_ngram_size=3
            )

        # Reset the request state after warmup (warmup extends SA multiple times)
        manager.remove_request(0)
        manager.add_request(0, context_tokens)
        manager.prepare(request_ids, max_draft_len)

        # Capture CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            match_len, draft_tokens = manager.extend_ngram(
                request_ids, accepted_tokens, num_accepted_tokens, max_draft_len, max_ngram_size=3
            )

        # Replay
        g.replay()

        print("CUDA graph capture succeeded with extend_ngram!")
        print(f"match_len: {match_len}")
        print(f"draft_tokens: {draft_tokens}")

        # Verify results (with max_ngram_size=3, should find 3-gram match)
        match_len_val = match_len[0].item()
        assert match_len_val == 3, f"Expected 3-gram match, got {match_len_val}"

        # Draft tokens should be [5, 1, 2, 3]
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [5, 1, 2, 3], f"Expected [5, 1, 2, 3], got {draft_list}"

        manager.shutdown()


class TestNativeKernelAvailability:
    """Tests for native kernel availability."""

    def test_is_native_available(self):
        """Test native availability check."""
        available = sa.is_native_available()
        print(f"Native kernel available: {available}")
        assert available, "Native kernel should be available for these tests"

        # Verify we can access the native module
        from tensorrt_llm.bindings.internal import suffix_automaton as native

        # Test dynamic state size API (replaces static constants)
        max_seq_len = 1024
        state_size = native.get_state_size(max_seq_len)
        print(f"get_state_size({max_seq_len}): {state_size} bytes")
        assert state_size > 0, "State size should be positive"

        # Verify state size scales with max_seq_len
        state_size_large = native.get_state_size(max_seq_len * 2)
        print(f"get_state_size({max_seq_len * 2}): {state_size_large} bytes")
        assert state_size_large > state_size, "Larger max_seq_len should have larger state size"


if __name__ == "__main__":
    # Run basic tests
    print("=" * 60)
    print("Testing suffix automaton module (native kernel only)")
    print("=" * 60)

    if not sa.is_native_available():
        print("ERROR: Native kernel not available. Tests cannot run.")
        exit(1)

    print("\n--- Native kernel availability ---")
    test = TestNativeKernelAvailability()
    test.test_is_native_available()

    print("\n--- Manager tests ---")
    test = TestSuffixAutomatonManager()
    test.test_manager_creation()
    test.test_manager_add_remove()
    test.test_manager_extend()

    print("\n--- extend_ngram tests ---")
    test = TestExtendNgram()
    test.test_extend_ngram_longest_match()
    test.test_extend_ngram_fixed_size()
    test.test_extend_ngram_no_match()
    test.test_extend_ngram_batch()
    test.test_extend_ngram_cuda_graph()

    print("\n--- CUDA graph compatibility tests ---")
    test = TestCUDAGraphCompatibility()
    test.test_cuda_graph_capture()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
