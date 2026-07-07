# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse attention integration facade for the shared MLA module."""

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._utils import get_sm_version

if TYPE_CHECKING:
    from tensorrt_llm._torch.distributed import AllReduceParams
    from tensorrt_llm._torch.modules.mla import MLA


def _get_sparse_algorithm(mla: "MLA") -> str:
    algorithm = getattr(mla.sparse_params, "algorithm", None)
    if algorithm is None:
        raise ValueError("Sparse MLA requires sparse_params.algorithm")
    return algorithm


def forward_sparse_mla(
    mla: "MLA",
    position_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    dsv4_epilogue_output: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> None:
    """Dispatch the shared MLA module to an algorithm-specific sparse implementation."""
    algorithm = _get_sparse_algorithm(mla)
    if algorithm == "dsa":
        if dsv4_epilogue_output is not None:
            raise RuntimeError("DSA does not support MLA epilogue fusion buffers")
        from .dsa.mla_module import forward_sparse_mla as forward_dsa_mla

        forward_dsa_mla(mla, position_ids, hidden_states, attn_metadata, output)
        return
    if algorithm == "deepseek_v4":
        from .deepseek_v4.mla_module import forward_sparse_mla as forward_deepseek_v4_mla

        forward_deepseek_v4_mla(
            mla,
            position_ids,
            hidden_states,
            attn_metadata,
            output,
            dsv4_epilogue_output=dsv4_epilogue_output,
        )
        return
    raise NotImplementedError(f"Sparse MLA algorithm {algorithm!r} is not supported")


def forward_sparse_mla_custom_op(
    mla: "MLA",
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    output: torch.Tensor,
    latent_cache_gen: Optional[torch.Tensor],
    dsv4_epilogue_output: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> None:
    """Run the custom-op entry point selected by the sparse MLA algorithm."""
    algorithm = _get_sparse_algorithm(mla)
    if algorithm == "dsa":
        if dsv4_epilogue_output is not None:
            raise RuntimeError("DSA does not support MLA epilogue fusion buffers")
        from .dsa.mla_module import forward_sparse_mla_custom_op as forward_dsa_custom_op

        forward_dsa_custom_op(mla, hidden_states, position_ids, output)
        return
    if algorithm == "deepseek_v4":
        dsv4_output = None
        dsv4_output_sf = None
        if dsv4_epilogue_output is not None:
            dsv4_output, dsv4_output_sf = dsv4_epilogue_output
        torch.ops.trtllm.mla_custom_op_inplace(
            hidden_states,
            position_ids,
            mla.layer_idx_str,
            output,
            latent_cache_gen,
            dsv4_output,
            dsv4_output_sf,
            dsv4_epilogue_output is not None,
        )
        return
    raise NotImplementedError(f"Sparse MLA algorithm {algorithm!r} is not supported")


def project_sparse_mla_output(
    mla: "MLA",
    attn_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    position_ids: Optional[torch.Tensor],
    attn_metadata: AttentionMetadata,
    all_reduce_params: Optional["AllReduceParams"],
) -> torch.Tensor:
    """Apply the output projection selected by the sparse MLA algorithm."""
    algorithm = _get_sparse_algorithm(mla)
    if algorithm == "deepseek_v4":
        from .deepseek_v4.mla_module import project_sparse_mla_output as project_deepseek_v4

        if position_ids is None and not isinstance(attn_output, tuple):
            raise ValueError("DeepSeek-V4 sparse MLA requires position_ids")
        return project_deepseek_v4(mla, attn_output, position_ids)
    if algorithm == "dsa":
        if isinstance(attn_output, tuple):
            raise RuntimeError("DSA does not support MLA epilogue fusion buffers")
        return mla._project_output_impl(attn_output, position_ids, attn_metadata, all_reduce_params)
    raise NotImplementedError(f"Sparse MLA algorithm {algorithm!r} is not supported")


def should_use_sparse_mla_epilogue_fusion(
    mla: "MLA", num_contexts: int, num_generations: int
) -> bool:
    """Return whether the selected sparse MLA algorithm can fuse its epilogue."""
    if mla.sparse_params is None:
        return False
    algorithm = _get_sparse_algorithm(mla)
    if algorithm == "deepseek_v4":
        from .deepseek_v4.mla_module import should_use_dsv4_epilogue_fusion

        return should_use_dsv4_epilogue_fusion(mla, num_contexts, num_generations)
    if algorithm == "dsa":
        return False
    raise NotImplementedError(f"Sparse MLA algorithm {algorithm!r} is not supported")


def create_sparse_mla_epilogue_buffers(
    mla: "MLA", q: torch.Tensor, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate epilogue buffers for the selected sparse MLA algorithm."""
    algorithm = _get_sparse_algorithm(mla)
    if algorithm == "deepseek_v4":
        from .deepseek_v4.mla_module import create_dsv4_epilogue_buffers

        return create_dsv4_epilogue_buffers(mla, q, num_tokens)
    raise RuntimeError(f"Sparse MLA algorithm {algorithm!r} has no fused epilogue")


def validate_sparse_mla_epilogue_buffers(
    mla: "MLA",
    num_tokens: int,
    epilogue_output: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate epilogue buffers for the selected sparse MLA algorithm."""
    algorithm = _get_sparse_algorithm(mla)
    if algorithm == "deepseek_v4":
        from .deepseek_v4.mla_module import validate_dsv4_epilogue_buffers

        return validate_dsv4_epilogue_buffers(mla, num_tokens, epilogue_output)
    raise RuntimeError(f"Sparse MLA algorithm {algorithm!r} has no fused epilogue")


def forward_context_sparse_mla(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    enable_dsv4_epilogue_fusion: bool = False,
    dsv4_epilogue_output: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run context-phase attention for DSA models.

    Dispatches to the short-seq MHA path (forward_context) when the max
    per-sequence KV length (including cached tokens) is within the
    threshold, or falls through to the absorption/sparse MLA path
    otherwise.  forward_context() further dispatches to the appropriate
    handler (forward_context_default, forward_context_with_cached_kv, or
    forward_context_with_chunked_prefill) based on cached-KV state.

    Args:
        q: Query tensor, shape [num_ctx_tokens, num_heads * qk_head_dim].
        compressed_kv: Latent KV, shape [num_ctx_tokens, kv_lora_rank].
        k_pe: RoPE key portion, shape [num_ctx_tokens, qk_rope_head_dim].
        attn_metadata: Attention metadata for the current batch.
        output: Pre-allocated output tensor, written in-place.
        latent_cache: Concatenated [compressed_kv, k_pe] for KV cache.
        topk_indices: Sparse routing indices from the indexer (None when
            the short-seq MHA path is used).
        position_ids: Token position IDs (required for short-seq MHA).
    """
    from .dsa.mla_module import forward_sparse_mla_kvcache_bf16, should_use_short_mha

    # Short-sequence MHA: bypass absorption path for short prefills,
    # using kv_b_proj expansion + standard attention instead.
    # See __init__ comment for rationale. topk_indices is not used
    # because dense attention is faster than sparse routing at this scale.
    # forward_context() handles cached tokens by dispatching to
    # forward_context_with_cached_kv or forward_context_with_chunked_prefill.
    if not enable_dsv4_epilogue_fusion and should_use_short_mha(self, attn_metadata, position_ids):
        return self.forward_context(
            q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
        )

    if get_sm_version() >= 100:
        return self.forward_absorption_context(
            q,
            compressed_kv,
            k_pe,
            attn_metadata,
            output,
            position_ids=position_ids,
            latent_cache=latent_cache,
            topk_indices=topk_indices,
            enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
            dsv4_epilogue_output=dsv4_epilogue_output,
        )
    else:
        assert not self.is_deepseek_v4, "DeepSeek-V4 is not supported on pre-blackwell GPUs."
        return forward_sparse_mla_kvcache_bf16(
            self, q, latent_cache, attn_metadata, output, topk_indices, is_generation=False
        )


def forward_generation_sparse_mla(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    latent_cache: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    enable_dsv4_epilogue_fusion: bool = False,
    dsv4_epilogue_output: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    from .dsa.mla_module import forward_sparse_mla_kvcache_bf16

    if get_sm_version() >= 100:
        return self.forward_absorption_generation(
            q,
            compressed_kv,
            k_pe,
            attn_metadata,
            output,
            position_ids=position_ids,
            latent_cache=latent_cache,
            topk_indices=topk_indices,
            enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
            dsv4_epilogue_output=dsv4_epilogue_output,
        )
    else:
        assert not self.is_deepseek_v4, "DeepSeek-V4 is not supported on pre-blackwell GPUs."
        return forward_sparse_mla_kvcache_bf16(
            self, q, latent_cache, attn_metadata, output, topk_indices, is_generation=True
        )
