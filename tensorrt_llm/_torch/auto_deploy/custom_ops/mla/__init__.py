"""MLA (Multi-head Latent Attention) custom ops.

Exports:
- MultiHeadLatentAttention: Attention descriptor for MLA (registered as "torch_mla")
- torch_mla: Source op for MLA attention
- torch_backend_mla_with_cache: Cached backend op with FlashInfer-compatible cache
"""

from .torch_backend_mla import MultiHeadLatentAttention, torch_backend_mla_with_cache
from .torch_mla import torch_mla

__all__ = ["MultiHeadLatentAttention", "torch_mla", "torch_backend_mla_with_cache"]
