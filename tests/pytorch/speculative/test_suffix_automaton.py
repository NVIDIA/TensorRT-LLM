"""Unit tests for suffix automaton speculative decoding.

Tests the native CUDA kernel implementation.
"""

import torch

from tensorrt_llm._torch.speculative.suffix_automaton import (
    SAConfig,
    SuffixAutomatonManager,
)  # noqa: I001


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

        # Case 1: context_tokens=[0,1,2,1,2], extend with token 1
        # New sequence: [0, 1, 2, 1, 2, 1]
        # Longest suffix match in context: [1, 2, 1] at positions 1-3 → match_len=3
        # Continuation after match: token at position 4 is 2 → draft starts with [2, ...]
        context_tokens = [0, 1, 2, 1, 2]
        manager.add_request(0, context_tokens)

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=-1,  # Longest match
        )

        print(f"extend_ngram (longest): match_len={match_len}, draft_tokens={draft_tokens}")

        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        # Longest match is [1, 2, 1] (length 3)
        assert match_len_val == 3, f"Expected longest match of 3, got {match_len_val}"
        # Draft tokens: continuation after match at 1-3 is token 2 at index 4
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list[0] == 2, f"Expected first draft token 2, got {draft_list}"

        manager.shutdown()

        # Case 2: context_tokens=[0, 1, 2, 3, 1, 2, 4, 1], extend with token 2
        # New sequence: [0, 1, 2, 3, 1, 2, 4, 1, 2]
        # Longest suffix match in context: [1, 2] at positions 1-2 or 4-5 → match_len=2
        # Kernel uses leftmost match (1-2): continuation is token at position 3 → draft [3, 1, 2, 4]
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)
        context_tokens = [0, 1, 2, 3, 1, 2, 4, 1]
        manager.add_request(0, context_tokens)

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[2, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=-1,  # Longest match
        )

        print(f"extend_ngram (longest) case 2: match_len={match_len}, draft_tokens={draft_tokens}")

        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        assert match_len_val == 2, f"Expected longest match of 2, got {match_len_val}"
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [3, 1, 2, 4], f"Expected draft [3, 1, 2, 4], got {draft_list}"

        manager.shutdown()

        # Case 3: context_tokens=[1, 1, 1, 1], extend with token 1
        # New sequence: [1, 1, 1, 1, 1]
        # Longest suffix match in context: [1, 1, 1, 1] at positions 0-3 → match_len=4
        # Only one draft token (kernel yields token at position after match; rest zero-padded)
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)
        context_tokens = [1, 1, 1, 1]
        manager.add_request(0, context_tokens)

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=-1,  # Longest match
        )

        print(f"extend_ngram (longest) case 3: match_len={match_len}, draft_tokens={draft_tokens}")

        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        assert match_len_val == 4, f"Expected longest match of 4, got {match_len_val}"
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [1, 0, 0, 0], f"Expected draft [1, 0, 0, 0], got {draft_list}"

        manager.shutdown()

        # Case 4: context_tokens=[0, 1, 2, 3], extend with token 2
        # New sequence: [0, 1, 2, 3, 2]
        # Longest suffix match in context: [2] at position 2 → match_len=1
        # Continuation after match: tokens at 3 and from extended seq → draft [3, 2, 0, 0]
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)
        context_tokens = [0, 1, 2, 3]
        manager.add_request(0, context_tokens)

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[2, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=-1,  # Longest match
        )

        print(f"extend_ngram (longest) case 4: match_len={match_len}, draft_tokens={draft_tokens}")

        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        assert match_len_val == 1, f"Expected longest match of 1, got {match_len_val}"
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [3, 2, 0, 0], f"Expected draft [3, 2, 0, 0], got {draft_list}"

        manager.shutdown()

    def test_extend_ngram_fixed_size(self):
        """Test extend_ngram with fixed-size ngram matching."""
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        # Case 1: context_tokens=[0, 1, 2, 3, 1, 2], extend with token 3
        # New sequence: [0, 1, 2, 3, 1, 2, 3]
        # With max_ngram_size=3: try 3-gram [1, 2, 3] → matches at positions 1-3 (leftmost)
        # match_len=3, continuation → draft [1, 2, 3, 0]
        context_tokens = [0, 1, 2, 3, 1, 2]
        manager.add_request(0, context_tokens)

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[3, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=3,  # Try 3-gram, 2-gram, 1-gram
        )

        print(f"extend_ngram (fixed) case 1: match_len={match_len}, draft_tokens={draft_tokens}")

        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        assert match_len_val == 3, f"Expected 3-gram match, got {match_len_val}"
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [1, 2, 3, 0], f"Expected draft [1, 2, 3, 0], got {draft_list}"

        manager.shutdown()

        # Case 2: context_tokens=[0, 1, 2, 3, 1, 2, 4, 1], extend with token 2
        # New sequence: [0, 1, 2, 3, 1, 2, 4, 1, 2]
        # With max_ngram_size=3: 3-gram [4, 1, 2] not in context; 2-gram [1, 2] matches at 1-2 (leftmost)
        # match_len=2, continuation → draft [3, 1, 2, 4]
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)
        context_tokens = [0, 1, 2, 3, 1, 2, 4, 1]
        manager.add_request(0, context_tokens)

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[2, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=3,  # Try 3-gram, 2-gram, 1-gram
        )

        print(f"extend_ngram (fixed) case 2: match_len={match_len}, draft_tokens={draft_tokens}")

        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        assert match_len_val == 2, f"Expected 2-gram match, got {match_len_val}"
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [3, 1, 2, 4], f"Expected draft [3, 1, 2, 4], got {draft_list}"

        manager.shutdown()

        # Case 3: context_tokens=[0, 2, 3, 4, 5, 1, 2, 3, 4, 6, 1, 2, 3], extend with token 4
        # New sequence: [0, 2, 3, 4, 5, 1, 2, 3, 4, 6, 1, 2, 3, 4]
        # With max_ngram_size=3: 3-gram [2, 3, 4] matches at positions 1-3 (leftmost)
        # match_len=3, continuation after 1-3 → draft [5, 1, 2, 3]
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)
        context_tokens = [0, 2, 3, 4, 5, 1, 2, 3, 4, 6, 1, 2, 3]
        manager.add_request(0, context_tokens)

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[4, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=3,  # Try 3-gram, 2-gram, 1-gram
        )

        print(f"extend_ngram (fixed) case 3: match_len={match_len}, draft_tokens={draft_tokens}")

        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        assert match_len_val == 3, f"Expected 3-gram match, got {match_len_val}"
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [5, 1, 2, 3], f"Expected draft [5, 1, 2, 3], got {draft_list}"

        manager.shutdown()

        # Case 4: context_tokens=[1, 2, 1, 2], extend with token 1
        # New sequence: [1, 2, 1, 2, 1]
        # With max_ngram_size=3: 3-gram [2, 1, 2] matches at positions 1-3 (leftmost)
        # match_len=3; continuation from match start when no token after match → draft [2, 1, 0, 0]
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)
        context_tokens = [1, 2, 1, 2]
        manager.add_request(0, context_tokens)

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=3,  # Try 3-gram, 2-gram, 1-gram
        )

        print(f"extend_ngram (fixed) case 4: match_len={match_len}, draft_tokens={draft_tokens}")

        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match, got match_len={match_len_val}"
        assert match_len_val == 3, f"Expected 3-gram match, got {match_len_val}"
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [2, 1, 0, 0], f"Expected draft [2, 1, 0, 0], got {draft_list}"

        manager.shutdown()

    def test_extend_ngram_no_match(self):
        """Test extend_ngram when no match exists (moved from longest_match case 5)."""
        # context_tokens=[0, 1, 2, 3], extend with token 4
        # New sequence: [0, 1, 2, 3, 4]
        # Token 4 not in context → no suffix match, match_len=0, draft is zero-padded
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)
        context_tokens = [0, 1, 2, 3]
        manager.add_request(0, context_tokens)

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[4, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=-1,  # Longest match
        )

        print(f"extend_ngram (no match): match_len={match_len}, draft_tokens={draft_tokens}")

        match_len_val = match_len[0].item()
        assert match_len_val == 0, f"Expected no match (0), got {match_len_val}"
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [0, 0, 0, 0], f"Expected draft [0, 0, 0, 0], got {draft_list}"

        manager.shutdown()

        # Same no-match scenario with max_ngram_size=3: context [0, 1, 2, 3], extend token 4
        config = SAConfig(max_seq_len=1024, max_slots=16)
        manager = SuffixAutomatonManager(config, max_num_requests=16)
        context_tokens = [0, 1, 2, 3]
        manager.add_request(0, context_tokens)

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[4, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=3,
        )

        print(
            f"extend_ngram (no match, max_ngram_size=3): match_len={match_len}, draft_tokens={draft_tokens}"
        )

        match_len_val = match_len[0].item()
        assert match_len_val == 0, f"Expected no match (0), got {match_len_val}"
        draft_list = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_list == [0, 0, 0, 0], f"Expected draft [0, 0, 0, 0], got {draft_list}"

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


class TestExtendGlobal:
    """Tests for extend_global() — cross-request pattern sharing."""

    def test_extend_global_cross_request_match(self):
        """Request B finds a pattern from Request A's context."""
        config = SAConfig(max_seq_len=1024, max_slots=16, enable_global_pool=True)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        # Request 0: context has [1, 2, 3, 4, 5]
        manager.add_request(0, [1, 2, 3, 4, 5])
        # Request 1: context has [10, 20, 1, 2, 3] — ends with same [1, 2, 3]
        manager.add_request(1, [10, 20, 1, 2, 3])

        request_ids = [0, 1]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Extend with token 6 for req 0, token 4 for req 1 (so req 1 ends [.., 3, 4])
        accepted_tokens = torch.tensor(
            [[6, 0, 0, 0, 0], [4, 0, 0, 0, 0]],
            dtype=torch.int32,
            device="cuda",
        )
        num_accepted_tokens = torch.tensor([1, 1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_global(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=-1,
        )

        print(f"global cross-request: match_len={match_len}, draft_tokens={draft_tokens}")

        # Request 1's SA has [10, 20, 1, 2, 3, 4]. lookupWithSuffix on
        # Request 0's SA [1, 2, 3, 4, 5, 6] processes tokens:
        #   10 → no match, 20 → no match, 1 → match(1), 2 → match(2),
        #   3 → match(3), 4 → match(4)
        # Match [1, 2, 3, 4] (len=4) from req 0 → continuation is [5, 6]
        match_len_1 = match_len[1].item()
        assert match_len_1 == 4, f"Request 1 should match [1,2,3,4] (len=4), got {match_len_1}"
        draft_1 = draft_tokens[1, :max_draft_len].cpu().tolist()
        assert draft_1[0] == 5, f"Expected continuation starting with 5, got {draft_1}"
        assert draft_1[1] == 6, f"Expected second draft token 6, got {draft_1}"

        manager.shutdown()

    def test_extend_global_prefers_own_slot(self):
        """When match lengths are equal, prefer the requesting SA's own slot."""
        config = SAConfig(max_seq_len=1024, max_slots=16, enable_global_pool=True)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        # Request 0: [1, 2, 3, 100, 1, 2] — has [1, 2] with continuation [3, 100, ...]
        manager.add_request(0, [1, 2, 3, 100, 1, 2])
        # Request 1: [1, 2, 3, 200, 1, 2] — has [1, 2] with continuation [3, 200, ...]
        manager.add_request(1, [1, 2, 3, 200, 1, 2])

        request_ids = [0, 1]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Extend both with token 3
        accepted_tokens = torch.tensor(
            [[3, 0, 0, 0, 0], [3, 0, 0, 0, 0]],
            dtype=torch.int32,
            device="cuda",
        )
        num_accepted_tokens = torch.tensor([1, 1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_global(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=-1,
        )

        print(f"global prefer-own: match_len={match_len}, draft_tokens={draft_tokens}")

        # Request 0 should prefer its own SA (continuation 100) over request 1's (200)
        draft_0 = draft_tokens[0].cpu().tolist()
        assert draft_0[0] == 100, f"Request 0 should use own slot continuation (100), got {draft_0}"

        # Request 1 should prefer its own SA (continuation 200) over request 0's (100)
        draft_1 = draft_tokens[1].cpu().tolist()
        assert draft_1[0] == 200, f"Request 1 should use own slot continuation (200), got {draft_1}"

        manager.shutdown()

    def test_extend_global_no_match(self):
        """No match across any SA returns match_len=0."""
        config = SAConfig(max_seq_len=1024, max_slots=16, enable_global_pool=True)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        manager.add_request(0, [1, 2, 3])
        manager.add_request(1, [4, 5, 6])

        request_ids = [0, 1]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Token 99 doesn't exist in any SA
        accepted_tokens = torch.tensor(
            [[99, 0, 0, 0, 0], [99, 0, 0, 0, 0]],
            dtype=torch.int32,
            device="cuda",
        )
        num_accepted_tokens = torch.tensor([1, 1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_global(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=-1,
        )

        print(f"global no-match: match_len={match_len}, draft_tokens={draft_tokens}")

        assert match_len[0].item() == 0, "Request 0 should have no match"
        assert match_len[1].item() == 0, "Request 1 should have no match"
        draft_0 = draft_tokens[0, :max_draft_len].cpu().tolist()
        assert draft_0 == [0] * max_draft_len, f"Expected zeroed draft for request 0, got {draft_0}"
        draft_1 = draft_tokens[1, :max_draft_len].cpu().tolist()
        assert draft_1 == [0] * max_draft_len, f"Expected zeroed draft for request 1, got {draft_1}"

        manager.shutdown()

    def test_extend_global_active_slot_mask(self):
        """Removed requests should not be searchable via the active slot mask."""
        config = SAConfig(max_seq_len=1024, max_slots=16, enable_global_pool=True)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        # Request 0 has pattern [1, 2, 3, 4, 5]
        manager.add_request(0, [1, 2, 3, 4, 5])
        # Request 1 has [10, 20, 1, 2]
        manager.add_request(1, [10, 20, 1, 2])

        # Remove request 0 — its slot mask should be cleared
        manager.remove_request(0)

        request_ids = [1]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        # Request 1 extends with token 3: suffix [1, 2, 3] should NOT match
        # against removed request 0's SA
        accepted_tokens = torch.tensor([[3, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_global(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=-1,
        )

        print(f"global mask test: match_len={match_len}")

        # Req 1's SA is [10, 20, 1, 2, 3] (all unique tokens). lookupWithSuffix
        # on its own SA matches the entire sequence (len=5) but pos=4 has no
        # continuation (pos+1 == mTokens.size()), so it returns empty.
        # Req 0's slot is masked out, so it's never searched.
        assert match_len[0].item() == 0, (
            f"Expected match_len=0 (no continuation in own SA, removed slot masked), "
            f"got {match_len[0].item()}"
        )

        manager.shutdown()

    def test_extend_global_single_request(self):
        """Global search with a single request behaves like local search."""
        config = SAConfig(max_seq_len=1024, max_slots=16, enable_global_pool=True)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        context_tokens = [0, 1, 2, 1, 2]
        manager.add_request(0, context_tokens)

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_global(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=-1,
        )

        print(f"global single request: match_len={match_len}, draft_tokens={draft_tokens}")

        # Same as local: [0, 1, 2, 1, 2, 1] → longest suffix [1, 2, 1] matches at pos 1-3
        match_len_val = match_len[0].item()
        assert match_len_val == 3, f"Expected match_len=3, got {match_len_val}"
        assert draft_tokens[0, 0].item() == 2, (
            f"Expected continuation 2, got {draft_tokens[0, 0].item()}"
        )

        manager.shutdown()

    def test_extend_global_cuda_graph(self):
        """Test that extend_global works with CUDA graph capture."""
        config = SAConfig(max_seq_len=1024, max_slots=16, enable_global_pool=True)
        manager = SuffixAutomatonManager(config, max_num_requests=16)

        manager.add_request(0, [1, 2, 3, 4, 5, 1, 2, 3])

        request_ids = [0]
        max_draft_len = 4
        manager.prepare(request_ids, max_draft_len)

        accepted_tokens = torch.tensor([[4, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")

        # Warmup
        for _ in range(3):
            manager.extend_global(
                request_ids,
                accepted_tokens,
                num_accepted_tokens,
                max_draft_len,
                max_ngram_size=-1,
            )

        # Reset state after warmup
        manager.remove_request(0)
        manager.add_request(0, [1, 2, 3, 4, 5, 1, 2, 3])
        manager.prepare(request_ids, max_draft_len)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            match_len, draft_tokens = manager.extend_global(
                request_ids,
                accepted_tokens,
                num_accepted_tokens,
                max_draft_len,
                max_ngram_size=-1,
            )

        g.replay()

        print(f"global CUDA graph: match_len={match_len}, draft_tokens={draft_tokens}")

        match_len_val = match_len[0].item()
        assert match_len_val >= 1, f"Expected match after CUDA graph replay, got {match_len_val}"

        manager.shutdown()


class TestRetainedPool:
    """Tests for retained slot pool (completed requests stay searchable)."""

    def test_retained_slot_is_searchable(self):
        """Completed request's SA stays searchable by active requests."""
        # pool_size=4 > max_num_requests=2 → retention capacity of 2
        config = SAConfig(
            max_seq_len=1024,
            max_slots=2,
            enable_global_pool=True,
            global_pool_size=4,
        )
        manager = SuffixAutomatonManager(config, max_num_requests=2)

        # Request A: context has [1, 2, 3, 4, 5]
        manager.add_request(0, [1, 2, 3, 4, 5])
        # Request B: context has [10, 20, 30]
        manager.add_request(1, [10, 20, 30])

        # Flush A's state to GPU
        manager.prepare([0, 1], max_draft_len=4)

        # Complete request A — should be retained, not freed
        manager.remove_request(0)
        assert 0 not in manager._request_to_slot
        assert len(manager._retained_slots) == 1

        # Request C arrives, ends with [1, 2, 3]
        manager.add_request(2, [50, 60, 1, 2, 3])

        request_ids = [1, 2]
        manager.prepare(request_ids, max_draft_len=4)

        # Extend request C with token 4 — should match retained A's [1,2,3,4]
        accepted_tokens = torch.tensor(
            [[99, 0, 0, 0, 0], [4, 0, 0, 0, 0]],
            dtype=torch.int32,
            device="cuda",
        )
        num_accepted_tokens = torch.tensor([1, 1], dtype=torch.int32, device="cuda")

        match_len, draft_tokens = manager.extend_global(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len=4,
            max_ngram_size=-1,
        )

        # Request C (index 1) should find a match from retained A
        match_len_c = match_len[1].item()
        assert match_len_c >= 3, (
            f"Request C should match retained A's pattern (len>=3), got {match_len_c}"
        )
        draft_c = draft_tokens[1].cpu().tolist()
        assert draft_c[0] == 5, f"Expected continuation token 5 from A, got {draft_c}"

        manager.shutdown()

    def test_eviction_fifo_order(self):
        """Oldest retained slot is evicted first when pool is full."""
        # pool_size=4, max_batch=2 → 2 retained slot capacity
        config = SAConfig(
            max_seq_len=1024,
            max_slots=2,
            enable_global_pool=True,
            global_pool_size=4,
        )
        manager = SuffixAutomatonManager(config, max_num_requests=2)
        # Initial: free=[0,1,2,3], active={}, retained={}

        manager.add_request(0, [1, 2, 3])
        manager.add_request(1, [4, 5, 6])
        manager.prepare([0, 1], max_draft_len=4)
        # free=[0,1], active={2,3}, retained={}  (slots allocated from end)

        # Complete A → retained
        manager.remove_request(0)
        assert len(manager._retained_slots) == 1
        assert len(manager._active_slots) == 1

        # Complete B → retained
        manager.remove_request(1)
        assert len(manager._retained_slots) == 2  # A and B both retained
        assert len(manager._active_slots) == 0

        # Requests C and D fill both free slots
        manager.add_request(2, [7, 8, 9])
        manager.add_request(3, [10, 11, 12])
        # free=[], active={slot_c, slot_d}, retained={slot_a: 0, slot_b: 1}
        assert len(manager._free_slots) == 0
        assert len(manager._active_slots) == 2
        assert len(manager._retained_slots) == 2

        # Request E arrives — pool full, must evict oldest retained (A)
        manager.add_request(4, [13, 14, 15])
        assert len(manager._retained_slots) == 1
        retained_rids = list(manager._retained_slots.values())
        assert retained_rids == [1], f"Expected B (rid=1) retained, got {retained_rids}"

        manager.shutdown()

    def test_active_never_evicted(self):
        """Active (in-flight) requests must never be evicted."""
        # pool_size=2 = max_batch=2 → 0 retained capacity → no retention
        config = SAConfig(
            max_seq_len=1024,
            max_slots=2,
            enable_global_pool=True,
            global_pool_size=2,
        )
        manager = SuffixAutomatonManager(config, max_num_requests=2)

        manager.add_request(0, [1, 2, 3])
        manager.add_request(1, [4, 5, 6])
        manager.prepare([0, 1], max_draft_len=4)

        # Complete A — pool_size == max_num_requests, so no retention
        manager.remove_request(0)
        assert len(manager._retained_slots) == 0
        assert len(manager._free_slots) == 1

        manager.shutdown()

    def test_no_retention_when_global_pool_disabled(self):
        """With global pool off, remove_request always frees immediately."""
        config = SAConfig(
            max_seq_len=1024,
            max_slots=4,
            enable_global_pool=False,
        )
        manager = SuffixAutomatonManager(config, max_num_requests=4)

        manager.add_request(0, [1, 2, 3])
        manager.prepare([0], max_draft_len=4)
        manager.remove_request(0)

        assert len(manager._retained_slots) == 0
        assert len(manager._free_slots) == 4

        manager.shutdown()

    def test_stale_request_not_retained(self):
        """Request removed before GPU copy is flushed should not be retained."""
        config = SAConfig(
            max_seq_len=1024,
            max_slots=2,
            enable_global_pool=True,
            global_pool_size=4,
        )
        manager = SuffixAutomatonManager(config, max_num_requests=2)

        # Add but don't prepare (GPU copy still pending)
        manager.add_request(0, [1, 2, 3])
        assert 0 in manager._pending_copies

        # Remove before prepare — should NOT be retained (stale GPU data)
        manager.remove_request(0)
        assert len(manager._retained_slots) == 0
        assert len(manager._free_slots) == 4  # slot returned to free list

        manager.shutdown()


class TestNativeKernel:
    """Tests for native kernel."""

    def test_native_kernel(self):
        """Test native kernel can be accessed."""
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

    print("\n--- Native kernel tests ---")
    test = TestNativeKernel()
    test.test_native_kernel()

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

    print("\n--- extend_global tests ---")
    test = TestExtendGlobal()
    test.test_extend_global_cross_request_match()
    test.test_extend_global_prefers_own_slot()
    test.test_extend_global_no_match()
    test.test_extend_global_active_slot_mask()
    test.test_extend_global_single_request()
    test.test_extend_global_cuda_graph()

    print("\n--- CUDA graph compatibility tests ---")
    test = TestCUDAGraphCompatibility()
    test.test_cuda_graph_capture()

    print("\n--- Retained pool tests ---")
    test = TestRetainedPool()
    test.test_retained_slot_is_searchable()
    test.test_eviction_fifo_order()
    test.test_active_never_evicted()
    test.test_no_retention_when_global_pool_disabled()
    test.test_stale_request_not_retained()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
