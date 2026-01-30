"""Attention operations.

This module provides various attention implementations and backends:
- torch_attention: PyTorch reference implementations
- torch_backend_attention: PyTorch-based attention backend
- flashinfer_attention: FlashInfer-based optimized attention
- triton_attention: Triton-based attention implementations
- triton_attention_with_kv_cache: Triton attention with KV cache support
- triton_attention_with_paged_kv_cache: Triton attention with paged KV cache
"""

__all__ = [
    "torch_attention",
    "torch_backend_attention",
    "flashinfer_attention",
    "triton_attention",
    "triton_attention_with_kv_cache",
    "triton_attention_with_paged_kv_cache",
]
