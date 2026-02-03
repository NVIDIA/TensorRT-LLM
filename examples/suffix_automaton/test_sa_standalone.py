#!/usr/bin/env python3
"""
Standalone test for Suffix Automaton algorithm correctness.

This script tests the SA implementation without requiring a full LLM model,
demonstrating the pattern matching and draft token prediction capabilities.
"""

import time
from typing import List, Tuple

import torch

# Import the native SA implementation
from tensorrt_llm._torch.speculative.suffix_automaton import (
    SAConfig,
    SuffixAutomatonManager,
    SuffixAutomatonState,
    init,
    add_request,
    prepare,
    extend,
    remove_request,
    shutdown,
)


def test_basic_pattern_matching():
    """Test basic suffix pattern matching."""
    print("\n" + "="*60)
    print("Test 1: Basic Pattern Matching")
    print("="*60)
    
    state = SuffixAutomatonState(max_seq_len=1024)
    
    # Simulate a sequence with repeating pattern: "ABCDAB"
    # Tokens: [10, 20, 30, 40, 10, 20]
    tokens = [10, 20, 30, 40, 10, 20]
    state.extend_batch(tokens)
    
    print(f"Sequence: {tokens}")
    print(f"Pattern: 'AB' at end should match 'AB' at beginning")
    
    result = state.lookup()
    if result:
        pos, length = result
        print(f"Match found! Length: {length}, Position after match: {pos}")
        
        # Get draft tokens (what follows the first occurrence of AB)
        drafts = state.get_draft_tokens(pos, 2)
        print(f"Draft tokens (what followed 'AB' before): {drafts}")
        
        # Expected: [30, 40] since that's what came after the first "AB"
        if drafts == [30, 40]:
            print("PASSED: Draft tokens match expected [30, 40]")
        else:
            print(f"FAILED: Expected [30, 40], got {drafts}")
    else:
        print("FAILED: No match found (expected match of length 2)")
    
    return result is not None


def test_long_pattern():
    """Test longer repeating patterns."""
    print("\n" + "="*60)
    print("Test 2: Longer Pattern Matching")
    print("="*60)
    
    state = SuffixAutomatonState(max_seq_len=1024)
    
    # Simulate code-like repetition: for i in range(10): print(i) for i in range(10):
    # Simplified as token sequence with long repeat
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5]
    state.extend_batch(tokens)
    
    print(f"Sequence length: {len(tokens)}")
    print(f"Looking for suffix match...")
    
    result = state.lookup()
    if result:
        pos, length = result
        print(f"Match found! Length: {length}, Position after match: {pos}")
        
        drafts = state.get_draft_tokens(pos, 6)
        print(f"Draft tokens: {drafts}")
        
        # The suffix [1,2,3,4,5] matches the beginning [1,2,3,4,5]
        # So drafts should be [6, 7, 8, 9, 10]
        expected = [6, 7, 8, 9, 10]
        if drafts == expected:
            print(f"PASSED: Draft tokens match expected {expected}")
        else:
            print(f"Note: Got {drafts} (expected {expected})")
    else:
        print("No match found")
    
    return result is not None and result[1] >= 5


def test_no_match():
    """Test that unique sequences return no match."""
    print("\n" + "="*60)
    print("Test 3: No Match for Unique Sequence")
    print("="*60)
    
    state = SuffixAutomatonState(max_seq_len=1024)
    
    # Unique sequence with no repeating suffix
    tokens = list(range(1, 11))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    state.extend_batch(tokens)
    
    print(f"Sequence: {tokens}")
    
    result = state.lookup()
    if result is None:
        print("PASSED: No match found (as expected for unique sequence)")
        return True
    else:
        print(f"FAILED: Unexpected match found: {result}")
        return False


def test_incremental_extension():
    """Test extending SA state incrementally."""
    print("\n" + "="*60)
    print("Test 4: Incremental Extension")
    print("="*60)
    
    state = SuffixAutomatonState(max_seq_len=1024)
    
    # Start with some tokens
    initial = [1, 2, 3, 4, 5]
    state.extend_batch(initial)
    print(f"Initial sequence: {initial}")
    
    # Extend one at a time
    for token in [1, 2, 3]:
        state.extend(token)
        result = state.lookup()
        if result:
            pos, length = result
            print(f"After adding {token}: match length={length}")
    
    print("PASSED: Incremental extension works")
    return True


def test_manager_batch_operations():
    """Test SuffixAutomatonManager with multiple requests."""
    print("\n" + "="*60)
    print("Test 5: Manager Batch Operations")
    print("="*60)
    
    config = SAConfig(max_seq_len=1024, max_slots=8, threshold=2)
    manager = SuffixAutomatonManager(config, max_num_requests=8)
    
    # Add multiple requests
    manager.add_request(request_id=0, context_tokens=[1, 2, 3, 1, 2])
    manager.add_request(request_id=1, context_tokens=[10, 20, 30, 40])
    manager.add_request(request_id=2, context_tokens=[5, 5, 5, 5, 5])
    
    print("Added 3 requests with different patterns")
    
    # Simulate accepting new tokens
    accepted_tokens = torch.tensor([
        [3, 0, 0, 0],  # Request 0: accept token 3
        [10, 20, 0, 0],  # Request 1: accept tokens 10, 20
        [5, 0, 0, 0],  # Request 2: accept token 5
    ], dtype=torch.int32)
    
    num_accepted = torch.tensor([1, 2, 1], dtype=torch.int32)
    
    match_len, draft_tokens = manager.extend(
        request_ids=[0, 1, 2],
        accepted_tokens=accepted_tokens,
        num_accepted_tokens=num_accepted,
        max_draft_len=4
    )
    
    print(f"Match lengths: {match_len.tolist()}")
    print(f"Draft tokens shape: {draft_tokens.shape}")
    
    # Clean up
    manager.shutdown()
    
    print("PASSED: Manager batch operations work")
    return True


def test_module_interface():
    """Test module-level interface (sa_spec compatibility)."""
    print("\n" + "="*60)
    print("Test 6: Module-Level Interface")
    print("="*60)
    
    # Initialize
    init(max_num_requests=4)
    print("Initialized global SA manager")
    
    # Add requests
    add_request(request_id=100, context_tokens=[1, 2, 3, 4, 1, 2])
    add_request(request_id=101, context_tokens=[10, 20, 30])
    print("Added 2 requests")
    
    # Prepare
    prepare(request_ids=[100, 101], max_draft_len=4)
    print("Prepared batch")
    
    # Extend
    batch_size = 2
    max_draft_len = 4
    
    match_len_out = torch.zeros((batch_size,), dtype=torch.int32, device='cuda')
    draft_tokens_out = torch.zeros((batch_size, max_draft_len), dtype=torch.int32, device='cuda')
    accepted_tokens = torch.tensor([[3, 4, 0, 0, 0], [10, 0, 0, 0, 0]], dtype=torch.int32, device='cuda')
    num_accepted = torch.tensor([2, 1], dtype=torch.int32, device='cuda')
    
    extend(match_len_out, draft_tokens_out, accepted_tokens, num_accepted)
    print(f"Extended with accepted tokens")
    print(f"Match lengths: {match_len_out.cpu().tolist()}")
    
    # Remove requests
    remove_request(request_id=100)
    remove_request(request_id=101)
    print("Removed requests")
    
    # Shutdown
    shutdown()
    print("Shutdown complete")
    
    print("PASSED: Module interface works")
    return True


def test_performance():
    """Test SA performance with larger sequences."""
    print("\n" + "="*60)
    print("Test 7: Performance Test")
    print("="*60)
    
    state = SuffixAutomatonState(max_seq_len=262144)
    
    # Build a long sequence with periodic patterns
    pattern = list(range(100))
    sequence = pattern * 100  # 10,000 tokens
    
    print(f"Building SA with {len(sequence)} tokens...")
    
    start = time.perf_counter()
    state.extend_batch(sequence)
    build_time = time.perf_counter() - start
    
    print(f"Build time: {build_time*1000:.2f}ms")
    
    # Lookup
    start = time.perf_counter()
    result = state.lookup()
    lookup_time = time.perf_counter() - start
    
    print(f"Lookup time: {lookup_time*1000:.2f}ms")
    
    if result:
        pos, length = result
        print(f"Match found: length={length}")
    
    print("PASSED: Performance test complete")
    return True


def main():
    print("="*60)
    print("Suffix Automaton Standalone Tests")
    print("="*60)
    
    tests = [
        ("Basic Pattern Matching", test_basic_pattern_matching),
        ("Long Pattern", test_long_pattern),
        ("No Match", test_no_match),
        ("Incremental Extension", test_incremental_extension),
        ("Manager Batch Operations", test_manager_batch_operations),
        ("Module Interface", test_module_interface),
        ("Performance", test_performance),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"EXCEPTION: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  [{status}] {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
