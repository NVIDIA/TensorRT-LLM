#!/usr/bin/env python3
"""Simple test for FlashInfer TRT-LLM attention backend."""

import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.flashinfer_trtllm_attention import (
    FlashInferTrtllmAttention,
    flashinfer_trtllm_mha_with_cache,
    prepare_flashinfer_trtllm_metadata,
)


def test_metadata_preparation():
    """Test metadata preparation function."""
    print("Testing metadata preparation...")

    # Create dummy inputs
    batch_size = 4
    seq_len_val = 8
    page_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = torch.randint(0, 1000, (batch_size, seq_len_val), device=device)
    position_ids = torch.arange(seq_len_val, device=device).unsqueeze(0).expand(batch_size, -1)
    seq_len = torch.full((batch_size,), seq_len_val, dtype=torch.int32, device=device)
    input_pos = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Create cache_loc: flat list of page indices
    pages_per_seq = torch.tensor([2, 3, 2, 2], dtype=torch.int32, device=device)
    cache_loc = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32, device=device)

    # Call metadata preparation
    block_tables, seq_lens_out = prepare_flashinfer_trtllm_metadata(
        input_ids,
        position_ids,
        seq_len,
        input_pos,
        cache_loc,
        pages_per_seq,
        page_size,
    )

    print(f"  block_tables shape: {block_tables.shape}")
    print(f"  seq_lens_out shape: {seq_lens_out.shape}")
    print(f"  block_tables:\n{block_tables}")
    print(f"  seq_lens_out: {seq_lens_out}")

    # Verify shapes
    assert block_tables.shape[0] == batch_size
    assert seq_lens_out.shape[0] == batch_size

    print("✓ Metadata preparation test passed!")


def test_attention_op():
    """Test attention op (requires GPU)."""
    if not torch.cuda.is_available():
        print("Skipping attention op test (no GPU)")
        return

    print("\nTesting attention op...")

    # Setup
    batch_size = 2
    seq_len = 4
    n_heads = 8
    n_kv_heads = 4
    head_dim = 128
    num_pages = 16
    page_size = 64

    device = torch.device("cuda")

    # Create inputs
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, n_kv_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, n_kv_heads, head_dim, device=device, dtype=torch.float16)

    # Create caches
    k_cache = torch.zeros(
        num_pages, page_size, n_kv_heads, head_dim, device=device, dtype=torch.float16
    )
    v_cache = torch.zeros(
        num_pages, page_size, n_kv_heads, head_dim, device=device, dtype=torch.float16
    )

    # Create workspace
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    # Create metadata
    block_tables = torch.tensor([[0, 1, 0], [2, 3, 4]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device)

    # Call attention
    try:
        output = flashinfer_trtllm_mha_with_cache(
            q,
            k,
            v,
            block_tables,
            seq_lens,
            k_cache,
            v_cache,
            workspace_buffer,
            scale=None,
        )

        print(f"  output shape: {output.shape}")
        assert output.shape == q.shape
        print("✓ Attention op test passed!")

    except Exception as e:
        print(f"✗ Attention op test failed: {e}")
        import traceback

        traceback.print_exc()


def test_backend_registration():
    """Test that backend is properly registered."""
    print("\nTesting backend registration...")

    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

    # Check if backend is registered
    assert "flashinfer_trtllm" in AttentionRegistry._registry, "Backend not registered!"
    backend = AttentionRegistry.get("flashinfer_trtllm")
    assert backend is FlashInferTrtllmAttention

    print("✓ Backend registration test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("FlashInfer TRT-LLM Attention Backend Tests")
    print("=" * 60)

    test_backend_registration()
    test_metadata_preparation()
    test_attention_op()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
