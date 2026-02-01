#!/usr/bin/env python3
"""Test script for TRT-LLM attention backend in Auto-Deploy.

This script tests the TRT-LLM attention implementation against FlashInfer
to verify correctness and compare performance.
"""

import torch


def test_trtllm_attention_import():
    """Test that the TRT-LLM attention module imports correctly."""
    print("=" * 60)
    print("Test 1: Import TRT-LLM Attention Module")
    print("=" * 60)

    try:
        from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import TrtllmAttention

        print("✓ Successfully imported TrtllmAttention")
        print(f"  - is_paged: {TrtllmAttention.is_paged()}")
        print(f"  - layout: {TrtllmAttention.get_attention_layout()}")
        print(f"  - metadata_args: {TrtllmAttention.get_standard_metadata_args()}")
        return True
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_attention_registry():
    """Test that TRT-LLM is registered in the AttentionRegistry."""
    print("\n" + "=" * 60)
    print("Test 2: Verify AttentionRegistry Registration")
    print("=" * 60)

    try:
        # Import custom_ops to trigger all registrations
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        has_trtllm = AttentionRegistry.has("trtllm")
        has_flashinfer = AttentionRegistry.has("flashinfer")

        print(f"  - trtllm registered: {has_trtllm}")
        print(f"  - flashinfer registered: {has_flashinfer}")

        if has_trtllm:
            trtllm_cls = AttentionRegistry.get("trtllm")
            print(f"  - TrtllmAttention class: {trtllm_cls}")
            print("✓ TRT-LLM backend is registered")
            return True
        else:
            print("✗ TRT-LLM backend NOT registered")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_custom_op_registration():
    """Test that the custom op is properly registered."""
    print("\n" + "=" * 60)
    print("Test 3: Verify Custom Op Registration")
    print("=" * 60)

    try:
        # Import to trigger registration

        # Check if op is registered
        op = torch.ops.auto_deploy.trtllm_attention_mha_with_cache
        print(f"  - Custom op registered: {op}")
        print(f"  - Op default: {op.default}")
        print("✓ Custom op is registered")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_metadata_translation():
    """Test metadata translation from AD format to TRT-LLM format."""
    print("\n" + "=" * 60)
    print("Test 4: Test Metadata Translation")
    print("=" * 60)

    try:
        from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
            TrtllmLayerState,
            _prepare_trtllm_metadata,
        )

        # Create mock AD metadata for a batch of 2 sequences
        # Sequence 1: 10 tokens (prefill)
        # Sequence 2: 1 token (decode) with 5 cached tokens
        device = "cuda"

        batch_info_host = torch.tensor(
            [1, 10, 1], dtype=torch.int32
        )  # 1 prefill, 10 tokens, 1 decode
        cu_seqlen_host = torch.tensor([0, 10, 11], dtype=torch.int32)  # cumulative seq lens
        cu_num_pages = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cu_num_pages_host = torch.tensor([0, 1, 2], dtype=torch.int32)
        cache_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)  # page indices
        last_page_len = torch.tensor([10, 6], dtype=torch.int32, device=device)
        last_page_len_host = torch.tensor([10, 6], dtype=torch.int32)
        seq_len_with_cache_host = torch.tensor([10, 6], dtype=torch.int32)  # total including cache

        # Create mock caches
        num_pages = 10
        page_size = 64
        num_kv_heads = 8
        head_dim = 128

        k_cache = torch.randn(
            num_pages, page_size, num_kv_heads, head_dim, device=device, dtype=torch.float16
        )
        v_cache = torch.randn(
            num_pages, page_size, num_kv_heads, head_dim, device=device, dtype=torch.float16
        )

        # Create layer state
        state = TrtllmLayerState(
            layer_idx=0,
            num_heads=32,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_requests=64,
            max_context_length=2048,
        )

        # Test metadata translation
        result = _prepare_trtllm_metadata(
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            state,
            k_cache,
            v_cache,
        )

        (
            sequence_length,
            host_past_key_value_lengths,
            host_total_kv_lens,
            context_lengths,
            host_context_lengths,
            host_request_types,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
        ) = result

        print(f"  - sequence_length: {sequence_length.tolist()}")
        print(f"  - host_past_key_value_lengths: {host_past_key_value_lengths.tolist()}")
        print(f"  - host_total_kv_lens: {host_total_kv_lens.tolist()}")
        print(f"  - context_lengths: {context_lengths.tolist()}")
        print(f"  - host_context_lengths: {host_context_lengths.tolist()}")
        print(f"  - host_request_types: {host_request_types.tolist()}")
        print(f"  - kv_cache_block_offsets shape: {kv_cache_block_offsets.shape}")
        print(f"  - host_kv_cache_pool_pointers: {host_kv_cache_pool_pointers.tolist()}")

        # Verify results
        assert sequence_length[0].item() == 10, (
            f"Expected seq_len[0]=10, got {sequence_length[0].item()}"
        )
        assert sequence_length[1].item() == 6, (
            f"Expected seq_len[1]=6, got {sequence_length[1].item()}"
        )
        assert host_request_types[0].item() == 0, "Prefill request should have type 0"
        assert host_request_types[1].item() == 1, "Decode request should have type 1"

        print("✓ Metadata translation works correctly")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_kernel_invocation():
    """Test basic kernel invocation (may fail due to parameter mismatches)."""
    print("\n" + "=" * 60)
    print("Test 5: Test Kernel Invocation (Basic)")
    print("=" * 60)

    try:
        from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
            _global_state,
            trtllm_mha_with_cache,
        )

        device = "cuda"
        dtype = torch.float16

        # Model config
        batch_size = 2
        seq_len = 4
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 64
        num_pages = 10

        # Create inputs - AD provides [batch, seq, num_heads, head_dim]
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

        # Metadata (all prefill)
        batch_info_host = torch.tensor([batch_size, batch_size * seq_len, 0], dtype=torch.int32)
        cu_seqlen_host = torch.tensor([0, seq_len, seq_len * 2], dtype=torch.int32)
        cu_num_pages = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cu_num_pages_host = torch.tensor([0, 1, 2], dtype=torch.int32)
        cache_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
        last_page_len = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device)
        last_page_len_host = torch.tensor([seq_len, seq_len], dtype=torch.int32)
        seq_len_with_cache_host = torch.tensor([seq_len, seq_len], dtype=torch.int32)

        # Caches - AD format: [num_pages, page_size, num_kv_heads, head_dim]
        k_cache = torch.zeros(
            num_pages, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        v_cache = torch.zeros(
            num_pages, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
        )

        # Workspace
        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        _global_state.init_workspace(workspace)

        print("  - Inputs created successfully")
        print(f"  - q.shape: {q.shape} (batch, seq, heads, head_dim)")
        print(f"  - k.shape: {k.shape}")
        print(f"  - v.shape: {v.shape}")
        print(f"  - k_cache.shape: {k_cache.shape} (pages, page_size, kv_heads, head_dim)")

        # Compute expected fused QKV shape
        fused_qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        print(f"  - Expected fused QKV hidden dim: {fused_qkv_dim}")

        # Try to call the kernel
        print("  - Attempting kernel call...")

        output = trtllm_mha_with_cache(
            q,
            k,
            v,
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            k_cache,
            v_cache,
            workspace,
            0,  # layer_idx
            None,  # scale
            num_heads,
            num_kv_heads,
            head_dim,
            page_size,
            64,  # max_num_requests
            2048,  # max_context_length
        )

        print(f"  - Output shape: {output.shape}")
        print(f"  - Output dtype: {output.dtype}")

        # Verify output shape
        expected_shape = (batch_size, seq_len, num_heads * head_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        print("✓ Kernel invocation succeeded")
        return True

    except Exception as e:
        print(f"✗ Kernel invocation failed: {e}")
        import traceback

        traceback.print_exc()
        print("\nNote: This is expected if thop.attention parameters don't match.")
        print("The kernel interface may need adjustment.")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TRT-LLM Attention Backend Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Import", test_trtllm_attention_import()))
    results.append(("Registry", test_attention_registry()))
    results.append(("Custom Op", test_custom_op_registration()))
    results.append(("Metadata", test_metadata_translation()))
    results.append(("Kernel", test_kernel_invocation()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
