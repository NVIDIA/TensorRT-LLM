"""MLA (Multi-head Latent Attention) custom ops.

Exports:
- TorchBackendMLAAttention: Attention descriptor for MLA (registered as "torch_mla")
- FlashInferMLAAttention: Attention descriptor for FlashInfer MLA (registered as "flashinfer_mla")
- torch_mla: Source op for MLA attention
- torch_backend_mla_with_cache: Cached backend op with FlashInfer-compatible cache
- flashinfer_mla_with_cache: Cached backend op using FlashInfer MLA kernels
"""

from .flashinfer_mla import FlashInferMLAAttention, flashinfer_mla_with_cache
from .torch_backend_mla import TorchBackendMLAAttention, torch_backend_mla_with_cache
from .torch_mla import torch_mla

__all__ = [
    "TorchBackendMLAAttention",
    "FlashInferMLAAttention",
    "torch_mla",
    "torch_backend_mla_with_cache",
    "flashinfer_mla_with_cache",
]
