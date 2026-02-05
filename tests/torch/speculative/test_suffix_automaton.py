"""Unit tests for suffix automaton speculative decoding.

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

        print("CUDA graph capture succeeded with native kernel")
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


class TestLookupFixed:
    """Tests for lookup_fixed() method."""

    def test_lookup_fixed_basic(self):
        """Test basic fixed-length lookup."""
        state = sa.SuffixAutomatonState(max_seq_len=1024)

        # Build SA with tokens that have a repeating pattern
        # [1, 2, 3, 4, 5, 1, 2, 3]
        # The suffix [1, 2, 3] appears earlier starting at position 0
        tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        state.extend_batch(tokens)

        # Look for a 3-gram match
        result = state.lookup_fixed(3)
        assert result is not None, "Should find a 3-gram match"

        continuation_pos, match_len = result
        assert match_len == 3, f"Expected match length 3, got {match_len}"
        # continuation_pos should be right after the first [1,2,3], which is position 3
        assert continuation_pos == 3, f"Expected continuation at pos 3, got {continuation_pos}"

        # Get draft tokens from continuation position
        draft_tokens = state.get_draft_tokens(continuation_pos, 2)
        # Should be [4, 5] - the tokens after the first [1, 2, 3]
        assert draft_tokens == [4, 5], f"Expected [4, 5], got {draft_tokens}"

    def test_lookup_fixed_2gram(self):
        """Test 2-gram lookup."""
        state = sa.SuffixAutomatonState(max_seq_len=1024)

        # [1, 2, 3, 4, 1, 2]
        # The suffix [1, 2] appears earlier at position 0
        tokens = [1, 2, 3, 4, 1, 2]
        state.extend_batch(tokens)

        result = state.lookup_fixed(2)
        assert result is not None

        continuation_pos, match_len = result
        assert match_len == 2
        assert continuation_pos == 2  # After first [1, 2]

        draft_tokens = state.get_draft_tokens(continuation_pos, 2)
        assert draft_tokens == [3, 4]

    def test_lookup_fixed_no_match(self):
        """Test lookup when no match exists."""
        state = sa.SuffixAutomatonState(max_seq_len=1024)

        # All unique tokens - no repeating pattern
        tokens = [1, 2, 3, 4, 5]
        state.extend_batch(tokens)

        result = state.lookup_fixed(2)
        assert result is None, "Should not find a match with all unique tokens"

    def test_lookup_fixed_too_short(self):
        """Test lookup with sequence too short for target length."""
        state = sa.SuffixAutomatonState(max_seq_len=1024)

        tokens = [1, 2]  # Only 2 tokens
        state.extend_batch(tokens)

        # Need at least target_len + 1 tokens for a non-self match
        result = state.lookup_fixed(2)
        assert result is None, "Sequence too short for 2-gram match"

        result = state.lookup_fixed(3)
        assert result is None, "Sequence too short for 3-gram match"

    def test_lookup_fixed_1gram(self):
        """Test 1-gram (single token) lookup."""
        state = sa.SuffixAutomatonState(max_seq_len=1024)

        # [1, 2, 3, 1]
        # The suffix [1] appears earlier at position 0
        tokens = [1, 2, 3, 1]
        state.extend_batch(tokens)

        result = state.lookup_fixed(1)
        assert result is not None

        continuation_pos, match_len = result
        assert match_len == 1
        assert continuation_pos == 1  # After first [1]

        draft_tokens = state.get_draft_tokens(continuation_pos, 2)
        assert draft_tokens == [2, 3]

    def test_lookup_fixed_multiple_matches(self):
        """Test that lookup_fixed returns a valid match when multiple exist."""
        state = sa.SuffixAutomatonState(max_seq_len=1024)

        # [1, 2, 3, 1, 2, 3, 1, 2]
        # The suffix [1, 2] appears at positions 0 and 3
        tokens = [1, 2, 3, 1, 2, 3, 1, 2]
        state.extend_batch(tokens)

        result = state.lookup_fixed(2)
        assert result is not None

        continuation_pos, match_len = result
        assert match_len == 2
        # Could be position 2 (after first [1,2]) or 5 (after second [1,2])
        # Current implementation returns first match
        assert continuation_pos == 2, f"Expected first match at pos 2, got {continuation_pos}"

    def test_lookup_fixed_via_manager(self):
        """Test lookup_fixed through SuffixAutomatonManager.

        Note: Manager's lookup_fixed(), lookup(), and get_draft_tokens() methods
        now only use Python fallback. For CUDA graph compatible operations,
        use extend_ngram() instead.
        """
        config = sa.SAConfig(max_seq_len=1024, max_slots=16)
        manager = sa.SuffixAutomatonManager(config, max_num_requests=16)

        # Add request with repeating pattern
        # Note: These methods now always use Python fallback (no native host-side functions)
        tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, tokens)

        # Test lookup_fixed (Python fallback only)
        result = manager.lookup_fixed(0, 3)
        assert result is not None

        continuation_pos, match_len = result
        assert match_len == 3
        assert continuation_pos == 3

        # Get draft tokens (Python fallback only)
        draft_tokens = manager.get_draft_tokens(0, continuation_pos, 2)
        assert draft_tokens == [4, 5]

        manager.shutdown()

    def test_lookup_fixed_incremental_extend(self):
        """Test lookup_fixed after incremental token extensions."""
        state = sa.SuffixAutomatonState(max_seq_len=1024)

        # Start with initial tokens
        state.extend_batch([1, 2, 3])

        # No match yet (need pattern to repeat)
        result = state.lookup_fixed(2)
        assert result is None

        # Add more tokens that create a repeating pattern
        state.extend_batch([4, 1, 2])

        # Now [1, 2] should match
        result = state.lookup_fixed(2)
        assert result is not None

        continuation_pos, match_len = result
        assert match_len == 2
        assert continuation_pos == 2  # After first [1, 2]


class TestExtendNgram:
    """Tests for extend_ngram() batched method - CUDA graph compatible."""

    def test_extend_ngram_longest_match(self):
        """Test extend_ngram with longest match mode (max_ngram_size=-1)."""
        config = sa.SAConfig(max_seq_len=1024, max_slots=16)
        manager = sa.SuffixAutomatonManager(config, max_num_requests=16)

        # Add request with repeating pattern
        # [1, 2, 3, 4, 5, 1, 2, 3] - the pattern [1, 2, 3] appears twice
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Extend with token 4 - this continues the pattern!
        # New sequence: [1, 2, 3, 4, 5, 1, 2, 3, 4]
        # Suffix [1, 2, 3, 4] matches at position 0-3
        # Continuation starts at position 4, draft tokens are [5, 1, 2, 3]
        accepted_tokens = torch.tensor([[4, 0, 0, 0, 0]], dtype=torch.int32, device='cuda')
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device='cuda')

        # Use longest match mode
        match_len, draft_tokens = manager.extend_ngram(
            request_ids, accepted_tokens, num_accepted_tokens, max_draft_len,
            max_ngram_size=-1  # Longest match
        )

        print(f"extend_ngram (longest): match_len={match_len}, draft_tokens={draft_tokens}")

        # Verify results
        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        # The longest match should be [1, 2, 3, 4] (length 4)
        assert match_len_val == 4, f"Expected longest match of 4, got {match_len_val}"

        # Draft tokens should be [5, 1, 2, 3] (tokens after the match at pos 0-3)
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [5, 1, 2, 3], f"Expected [5, 1, 2, 3], got {draft_list}"

        manager.shutdown()

    def test_extend_ngram_fixed_size(self):
        """Test extend_ngram with fixed-size ngram matching."""
        config = sa.SAConfig(max_seq_len=1024, max_slots=16)
        manager = sa.SuffixAutomatonManager(config, max_num_requests=16)

        # Add request with repeating pattern
        # [1, 2, 3, 4, 5, 1, 2, 3]
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Extend with token 4 - continues the pattern
        # New sequence: [1, 2, 3, 4, 5, 1, 2, 3, 4]
        accepted_tokens = torch.tensor([[4, 0, 0, 0, 0]], dtype=torch.int32, device='cuda')
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device='cuda')

        # Use fixed-size ngram mode (max_ngram_size=3 means try 3, 2, 1)
        # Should find 3-gram [2, 3, 4] matching at position 1-3
        match_len, draft_tokens = manager.extend_ngram(
            request_ids, accepted_tokens, num_accepted_tokens, max_draft_len,
            max_ngram_size=3  # Try 3-gram, 2-gram, 1-gram
        )

        print(f"extend_ngram (fixed 3): match_len={match_len}, draft_tokens={draft_tokens}")

        # Verify results
        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        # With max_ngram_size=3, should find 3-gram match
        assert match_len_val == 3, f"Expected 3-gram match, got {match_len_val}"

        # Draft tokens after 3-gram [2, 3, 4] at pos 1-3 are [5, 1, 2, 3]
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [5, 1, 2, 3], f"Expected [5, 1, 2, 3], got {draft_list}"

        manager.shutdown()

    def test_extend_ngram_no_match(self):
        """Test extend_ngram when no match exists."""
        config = sa.SAConfig(max_seq_len=1024, max_slots=16)
        manager = sa.SuffixAutomatonManager(config, max_num_requests=16)

        # Add request with repeating pattern
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Extend with unique token 99 - no pattern will match
        # New sequence: [1, 2, 3, 4, 5, 1, 2, 3, 99]
        accepted_tokens = torch.tensor([[99, 0, 0, 0, 0]], dtype=torch.int32, device='cuda')
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device='cuda')

        match_len, draft_tokens = manager.extend_ngram(
            request_ids, accepted_tokens, num_accepted_tokens, max_draft_len,
            max_ngram_size=-1
        )

        print(f"extend_ngram (no match): match_len={match_len}, draft_tokens={draft_tokens}")

        # Verify no match found
        match_len_val = match_len[0].item()
        assert match_len_val == 0, f"Expected no match, got match_len={match_len_val}"

        manager.shutdown()

    def test_extend_ngram_batch(self):
        """Test extend_ngram with multiple requests in batch."""
        config = sa.SAConfig(max_seq_len=1024, max_slots=16)
        manager = sa.SuffixAutomatonManager(config, max_num_requests=16)

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
        accepted_tokens = torch.tensor([
            [4, 0, 0, 0, 0],    # Continues [1,2,3,4] pattern
            [30, 0, 0, 0, 0],   # Continues [10,20,30] pattern
            [500, 0, 0, 0, 0]   # Unique, no match
        ], dtype=torch.int32, device='cuda')
        num_accepted_tokens = torch.tensor([1, 1, 1], dtype=torch.int32, device='cuda')

        match_len, draft_tokens = manager.extend_ngram(
            request_ids, accepted_tokens, num_accepted_tokens, max_draft_len,
            max_ngram_size=-1
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

    @pytest.mark.skipif(
        not sa.is_native_available(),
        reason="Native kernel not available"
    )
    def test_extend_ngram_cuda_graph(self):
        """Test that extend_ngram works with CUDA graph capture (native only)."""
        config = sa.SAConfig(max_seq_len=1024, max_slots=16)
        manager = sa.SuffixAutomatonManager(config, max_num_requests=16)

        # Add request with repeating pattern
        context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        manager.add_request(0, context_tokens)

        # Prepare (must be done before capture)
        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Create input/output tensors - use token 4 to create pattern match
        accepted_tokens = torch.tensor([[4, 0, 0, 0, 0]], dtype=torch.int32, device='cuda')
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device='cuda')

        # Warmup - each call extends SA state, so we'll reset after
        for _ in range(3):
            manager.extend_ngram(request_ids, accepted_tokens, num_accepted_tokens,
                                max_draft_len, max_ngram_size=3)

        # Reset the request state after warmup (warmup extends SA multiple times)
        manager.remove_request(0)
        manager.add_request(0, context_tokens)
        manager.prepare(request_ids, max_draft_len)

        # Capture CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            match_len, draft_tokens = manager.extend_ngram(
                request_ids, accepted_tokens, num_accepted_tokens,
                max_draft_len, max_ngram_size=3
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

    def test_extend_ngram_fallback(self):
        """Test extend_ngram Python fallback when native is not available."""
        # Temporarily force fallback mode
        original_native = sa._sa_native
        sa._sa_native = None

        try:
            config = sa.SAConfig(max_seq_len=1024, max_slots=16)
            manager = sa.SuffixAutomatonManager(config, max_num_requests=16)
            assert not manager._use_native, "Should be using fallback"

            # Add request with repeating pattern
            context_tokens = [1, 2, 3, 4, 5, 1, 2, 3]
            manager.add_request(0, context_tokens)

            # Prepare
            request_ids = [0]
            max_draft_len = 4
            manager.prepare(request_ids, max_draft_len)

            # Extend with token 4 to create pattern match
            # New sequence: [1, 2, 3, 4, 5, 1, 2, 3, 4]
            accepted_tokens = torch.tensor([[4, 0, 0, 0, 0]], dtype=torch.int32, device='cuda')
            num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device='cuda')

            # Test with longest match - should find [1, 2, 3, 4] match (length 4)
            match_len, draft_tokens = manager.extend_ngram(
                request_ids, accepted_tokens, num_accepted_tokens, max_draft_len,
                max_ngram_size=-1
            )
            print(f"Fallback longest: match_len={match_len}, draft_tokens={draft_tokens}")

            # Verify longest match result
            match_len_val = match_len[0].item()
            assert match_len_val == 4, f"Fallback longest: expected match_len=4, got {match_len_val}"
            draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
            assert draft_list == [5, 1, 2, 3], f"Fallback longest: expected [5,1,2,3], got {draft_list}"

            # Need to re-add request since state was modified
            manager.remove_request(0)
            manager.add_request(0, context_tokens)
            manager.prepare(request_ids, max_draft_len)

            # Test with fixed-size ngram - should find 3-gram match
            match_len, draft_tokens = manager.extend_ngram(
                request_ids, accepted_tokens, num_accepted_tokens, max_draft_len,
                max_ngram_size=3
            )
            print(f"Fallback fixed 3: match_len={match_len}, draft_tokens={draft_tokens}")

            # Verify fixed-size result
            match_len_val = match_len[0].item()
            assert match_len_val == 3, f"Fallback fixed: expected match_len=3, got {match_len_val}"
            draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
            assert draft_list == [5, 1, 2, 3], f"Fallback fixed: expected [5,1,2,3], got {draft_list}"

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
            from tensorrt_llm.bindings.internal import suffix_automaton as native
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

    print("\n--- lookup_fixed tests (Python fallback) ---")
    test = TestLookupFixed()
    test.test_lookup_fixed_basic()
    test.test_lookup_fixed_2gram()
    test.test_lookup_fixed_no_match()
    test.test_lookup_fixed_too_short()
    test.test_lookup_fixed_1gram()
    test.test_lookup_fixed_multiple_matches()
    test.test_lookup_fixed_via_manager()
    test.test_lookup_fixed_incremental_extend()

    print("\n--- extend_ngram tests (CUDA graph compatible) ---")
    test = TestExtendNgram()
    test.test_extend_ngram_longest_match()
    test.test_extend_ngram_fixed_size()
    test.test_extend_ngram_batch()
    test.test_extend_ngram_fallback()
    if sa.is_native_available():
        test.test_extend_ngram_cuda_graph()

    print("\n--- CUDA graph compatibility tests ---")
    test = TestCUDAGraphCompatibility()
    if sa.is_native_available():
        test.test_cuda_graph_capture_native()
    test.test_cuda_graph_capture_fallback()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
