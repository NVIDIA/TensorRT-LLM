"""
Unit tests for the suffix automaton speculative decoding implementation.

Tests cover:
1. SuffixAutomatonState - basic operations (extend, lookup, get_draft_tokens)
2. SuffixAutomatonManager - request lifecycle and batch operations
3. SAResourceManager - integration with TRT-LLM resource management
4. Module-level interface compatibility with external sa_spec
"""

import pytest
import torch
from typing import List

from tensorrt_llm._torch.speculative.suffix_automaton import (
    SAConfig,
    SAResourceManager,
    SuffixAutomatonManager,
    SuffixAutomatonState,
    add_request,
    extend,
    init,
    prepare,
    remove_request,
    shutdown,
)


class TestSuffixAutomatonState:
    """Tests for SuffixAutomatonState class."""

    def test_init(self):
        """Test basic initialization."""
        state = SuffixAutomatonState(max_seq_len=1024)
        assert state.max_seq_len == 1024
        assert len(state.tokens) == 0

    def test_extend_single_token(self):
        """Test extending with a single token."""
        state = SuffixAutomatonState(max_seq_len=1024)
        state.extend(42)
        assert state.tokens == [42]

    def test_extend_batch(self):
        """Test extending with multiple tokens."""
        state = SuffixAutomatonState(max_seq_len=1024)
        state.extend_batch([1, 2, 3, 4, 5])
        assert state.tokens == [1, 2, 3, 4, 5]

    def test_clear(self):
        """Test clearing the state."""
        state = SuffixAutomatonState(max_seq_len=1024)
        state.extend_batch([1, 2, 3])
        state.clear()
        assert len(state.tokens) == 0

    def test_lookup_no_match(self):
        """Test lookup with no repeating pattern."""
        state = SuffixAutomatonState(max_seq_len=1024)
        state.extend_batch([1, 2, 3, 4, 5])
        result = state.lookup()
        # No suffix appears earlier in unique sequence
        assert result is None

    def test_lookup_simple_match(self):
        """Test lookup with a simple repeating pattern."""
        state = SuffixAutomatonState(max_seq_len=1024)
        # Sequence: A B C A B
        # The suffix "AB" at end matches "AB" at beginning
        state.extend_batch([10, 20, 30, 10, 20])
        result = state.lookup()
        assert result is not None
        pos, length = result
        assert length == 2  # "AB" matches
        assert pos == 2  # Position after the match (where "C" is)

    def test_lookup_long_match(self):
        """Test lookup with a longer repeating pattern."""
        state = SuffixAutomatonState(max_seq_len=1024)
        # Sequence: A B C D A B C
        # The suffix "ABC" at end matches "ABC" at beginning
        state.extend_batch([10, 20, 30, 40, 10, 20, 30])
        result = state.lookup()
        assert result is not None
        pos, length = result
        assert length == 3  # "ABC" matches
        assert pos == 3  # Position after the match (where "D" is)

    def test_get_draft_tokens(self):
        """Test getting draft tokens from a position."""
        state = SuffixAutomatonState(max_seq_len=1024)
        state.extend_batch([1, 2, 3, 4, 5, 6, 7, 8])
        
        # Get 3 tokens starting at position 2
        drafts = state.get_draft_tokens(start_pos=2, num_tokens=3)
        assert drafts == [3, 4, 5]

    def test_get_draft_tokens_at_end(self):
        """Test getting draft tokens near the end of sequence."""
        state = SuffixAutomatonState(max_seq_len=1024)
        state.extend_batch([1, 2, 3, 4, 5])
        
        # Request more tokens than available
        drafts = state.get_draft_tokens(start_pos=3, num_tokens=5)
        assert drafts == [4, 5]  # Only 2 tokens available


class TestSuffixAutomatonManager:
    """Tests for SuffixAutomatonManager class."""

    @pytest.fixture
    def manager(self):
        """Create a manager for testing."""
        config = SAConfig(max_seq_len=1024, max_slots=8, threshold=4)
        return SuffixAutomatonManager(config, max_num_requests=8)

    def test_add_request(self, manager):
        """Test adding a new request."""
        manager.add_request(request_id=1, context_tokens=[1, 2, 3, 4, 5])
        assert 1 in manager._request_to_slot
        assert 1 in manager._host_states
        assert manager._host_states[1].tokens == [1, 2, 3, 4, 5]

    def test_add_multiple_requests(self, manager):
        """Test adding multiple requests."""
        manager.add_request(request_id=1, context_tokens=[1, 2, 3])
        manager.add_request(request_id=2, context_tokens=[4, 5, 6])
        manager.add_request(request_id=3, context_tokens=[7, 8, 9])
        
        assert len(manager._request_to_slot) == 3
        assert len(manager._host_states) == 3

    def test_remove_request(self, manager):
        """Test removing a request."""
        manager.add_request(request_id=1, context_tokens=[1, 2, 3])
        manager.remove_request(request_id=1)
        
        assert 1 not in manager._request_to_slot
        assert 1 not in manager._host_states

    def test_slot_reuse(self, manager):
        """Test that slots are reused after removal."""
        initial_free_slots = len(manager._free_slots)
        
        manager.add_request(request_id=1, context_tokens=[1, 2, 3])
        assert len(manager._free_slots) == initial_free_slots - 1
        
        manager.remove_request(request_id=1)
        assert len(manager._free_slots) == initial_free_slots

    def test_extend(self, manager):
        """Test extending SA states with accepted tokens."""
        manager.add_request(request_id=0, context_tokens=[1, 2, 3, 1, 2])  # Has "12" repeat
        
        accepted_tokens = torch.tensor([[4, 0, 0, 0]], dtype=torch.int32)
        num_accepted = torch.tensor([1], dtype=torch.int32)
        
        match_len, draft_tokens = manager.extend(
            request_ids=[0],
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted,
            max_draft_len=3
        )
        
        # After adding token 4, the state is [1,2,3,1,2,4]
        # Check that something was returned
        assert match_len.shape[0] == 1
        assert draft_tokens.shape == (1, 3)

    def test_shutdown(self, manager):
        """Test shutdown cleans up resources."""
        manager.add_request(request_id=1, context_tokens=[1, 2, 3])
        manager.add_request(request_id=2, context_tokens=[4, 5, 6])
        
        manager.shutdown()
        
        assert len(manager._host_states) == 0
        assert len(manager._request_to_slot) == 0
        assert len(manager._free_slots) == manager.max_num_requests


class TestModuleLevelInterface:
    """Tests for module-level interface compatibility."""

    def setup_method(self):
        """Initialize before each test."""
        shutdown()  # Clean up any previous state

    def teardown_method(self):
        """Clean up after each test."""
        shutdown()

    def test_init_and_add_request(self):
        """Test initializing and adding a request."""
        init(max_num_requests=4)
        add_request(request_id=1, context_tokens=[1, 2, 3, 4, 5])
        # Should not raise

    def test_prepare(self):
        """Test preparing batch indices."""
        init(max_num_requests=4)
        add_request(request_id=1, context_tokens=[1, 2, 3])
        prepare(request_ids=[1], max_draft_len=4)
        # Should not raise

    def test_extend_interface(self):
        """Test the extend interface matches external sa_spec."""
        init(max_num_requests=4)
        
        batch_size = 2
        max_draft_len = 4
        
        match_len_out = torch.zeros((batch_size,), dtype=torch.int32, device='cuda')
        draft_tokens_out = torch.zeros((batch_size, max_draft_len), dtype=torch.int32, device='cuda')
        accepted_tokens = torch.ones((batch_size, max_draft_len + 1), dtype=torch.int32, device='cuda')
        num_accepted = torch.ones((batch_size,), dtype=torch.int32, device='cuda')
        
        extend(match_len_out, draft_tokens_out, accepted_tokens, num_accepted)
        # Should not raise

    def test_remove_request_interface(self):
        """Test removing a request through module interface."""
        init(max_num_requests=4)
        add_request(request_id=1, context_tokens=[1, 2, 3])
        remove_request(request_id=1)
        # Should not raise


class TestSAResourceManager:
    """Tests for SAResourceManager integration."""

    def test_init(self):
        """Test initialization."""
        class MockConfig:
            sa_spec_threshold = 4
        
        manager = SAResourceManager(config=MockConfig(), max_num_requests=8)
        assert manager.max_num_requests == 8

    def test_get_max_resource_count(self):
        """Test resource count reporting."""
        class MockConfig:
            sa_spec_threshold = 4
        
        manager = SAResourceManager(config=MockConfig(), max_num_requests=16)
        assert manager.get_max_resource_count() == 16

    def test_get_needed_resource_to_completion(self):
        """Test resource needs calculation (always 0 for SA)."""
        class MockConfig:
            sa_spec_threshold = 4
        
        manager = SAResourceManager(config=MockConfig(), max_num_requests=8)
        # SA doesn't need additional resources for completion
        assert manager.get_needed_resource_to_completion(None) == 0


class TestSuffixAutomatonAlgorithm:
    """Tests for the suffix automaton algorithm correctness."""

    def test_example_from_spec(self):
        """Test the example from the plan: SABCDBDAB finding AB match."""
        state = SuffixAutomatonState(max_seq_len=1024)
        # S=83, A=65, B=66, C=67, D=68
        # Sequence: S A B C D B D A B
        state.extend_batch([83, 65, 66, 67, 68, 66, 68, 65, 66])
        
        result = state.lookup()
        assert result is not None
        pos, length = result
        # The suffix "AB" (at end) matches "AB" (at positions 1-2)
        # So we expect length=2 and pos=3 (position after "AB", where "C" is)
        assert length >= 2  # At least AB matches
        
        # Get draft tokens from the match position
        drafts = state.get_draft_tokens(pos, 4)
        # Should return tokens after the first "AB" occurrence

    def test_no_match_unique_sequence(self):
        """Test that unique sequences return no match."""
        state = SuffixAutomatonState(max_seq_len=1024)
        # Unique tokens with no repeating suffix
        state.extend_batch([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        result = state.lookup()
        # Should be None since no suffix repeats
        assert result is None

    def test_single_token_no_match(self):
        """Test that single token doesn't match."""
        state = SuffixAutomatonState(max_seq_len=1024)
        state.extend(42)
        
        result = state.lookup()
        assert result is None

    def test_empty_state_no_match(self):
        """Test that empty state returns no match."""
        state = SuffixAutomatonState(max_seq_len=1024)
        
        result = state.lookup()
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
