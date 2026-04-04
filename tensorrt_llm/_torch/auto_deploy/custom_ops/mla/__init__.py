"""MLA (Multi-head Latent Attention) custom ops.

Exports:
- TorchBackendMLAAttention: Attention descriptor for MLA (registered as "torch_mla")
- FlashInferMLAAttention: Attention descriptor for FlashInfer MLA (registered as "flashinfer_mla")
- FlashInferTrtllmMLAAttention: Attention descriptor for FlashInfer TRTLLM-gen MLA
- torch_mla: Source op for MLA attention
- torch_backend_mla_with_cache: Cached backend op with FlashInfer-compatible cache
- flashinfer_mla_with_cache: Cached backend op using FlashInfer MLA kernels
- flashinfer_trtllm_mla_with_cache: Cached backend op using FlashInfer Path 2 kernels
"""

from .flashinfer_mla import FlashInferMLAAttention, flashinfer_mla_with_cache
from .flashinfer_trtllm_mla import FlashInferTrtllmMLAAttention, flashinfer_trtllm_mla_with_cache
from .torch_backend_mla import TorchBackendMLAAttention, torch_backend_mla_with_cache
from .torch_mla import torch_mla

__all__ = [
    "TorchBackendMLAAttention",
    "FlashInferMLAAttention",
    "FlashInferTrtllmMLAAttention",
    "torch_mla",
    "torch_backend_mla_with_cache",
    "flashinfer_mla_with_cache",
    "flashinfer_trtllm_mla_with_cache",
]
