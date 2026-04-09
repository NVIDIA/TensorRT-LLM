"""MLA (Multi-head Latent Attention) custom ops.

Exports:
- TorchBackendMLAAttention: Attention descriptor for MLA (registered as "torch_mla")
- FlashInferMLAAttention: Attention descriptor for FlashInfer MLA (registered as "flashinfer_mla")
- TrtllmMLAAttention: Attention descriptor for TRT-LLM MLA (registered as "trtllm_mla")
- torch_mla: Source op for MLA attention
- torch_backend_mla_with_cache: Cached backend op with FlashInfer-compatible cache
- flashinfer_mla_with_cache: Cached backend op using FlashInfer MLA kernels
- trtllm_mla_with_cache: Cached backend op using TRT-LLM thop.attention with MLA
- trtllm_mla_fused_rope_with_cache: Fused RoPE + cached TRT-LLM MLA op
"""

from .flashinfer_mla import FlashInferMLAAttention, flashinfer_mla_with_cache
from .torch_backend_mla import TorchBackendMLAAttention, torch_backend_mla_with_cache
from .torch_mla import torch_mla
from .trtllm_mla import (
    TrtllmMLAAttention,
    prepare_trtllm_mla_metadata,
    trtllm_mla_fused_rope_with_cache,
    trtllm_mla_with_cache,
)

__all__ = [
    "TorchBackendMLAAttention",
    "FlashInferMLAAttention",
    "TrtllmMLAAttention",
    "torch_mla",
    "torch_backend_mla_with_cache",
    "flashinfer_mla_with_cache",
    "trtllm_mla_with_cache",
    "trtllm_mla_fused_rope_with_cache",
    "prepare_trtllm_mla_metadata",
]
