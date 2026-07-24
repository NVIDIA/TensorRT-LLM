# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSA custom ops used by piecewise CUDA graph capture."""

from typing import List, Optional

import torch

from tensorrt_llm._torch.utils import Fp4QuantizedTensor

from .module import _forward_dsa_attn, forward_dsa_proj


def _extract_extra_attrs(layer_idx: str):
    from tensorrt_llm._torch.modules.attention import extract_extra_attrs

    return extract_extra_attrs(layer_idx, "mla")


@torch.library.custom_op("trtllm::mla_dsa_proj", mutates_args=())
def mla_dsa_proj(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    hidden_states_fp4: Optional[torch.Tensor] = None,
    hidden_states_sf: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Token-wise projections for DSA MLA (CUDA-graph-capturable).

    Runs kv_a_proj, layernorms, q_b_proj, and conditionally
    indexer.pre_indexer_proj (FP8/FP4 quantize, weight scaling).  Does NOT
    update the indexer k cache — that happens in Op 2 (mla_dsa_attn_inplace)
    because the scatter kernel accesses batch-specific metadata.

    Returns [q, compressed_kv, k_pe, latent_cache] when the short-MHA path
    handles all tokens, or [q, compressed_kv, k_pe, latent_cache, q_fp8,
    k_fp8, k_scale, weights, q_scale] when the indexer runs.  Under torch
    compile, _should_use_short_mha returns False so the result is always
    length 9, keeping control flow straight-line for CUDA graph capture.
    The trailing q_scale is only consumed by the FP4 dispatch; the FP8
    path ignores it in _forward_dsa_attn.
    """
    metadata, mla_layer = _extract_extra_attrs(layer_idx)
    if hidden_states_fp4 is not None or hidden_states_sf is not None:
        assert hidden_states_fp4 is not None and hidden_states_sf is not None, (
            "hidden_states_fp4 and hidden_states_sf must be passed together"
        )
        hidden_states = Fp4QuantizedTensor(
            fp4_tensor=hidden_states_fp4,
            scaling_factor=hidden_states_sf,
            unquantized_hidden_states=hidden_states,
        )
    return forward_dsa_proj(mla_layer, position_ids, hidden_states, metadata)


@mla_dsa_proj.register_fake
def _mla_dsa_proj_fake(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    hidden_states_fp4: Optional[torch.Tensor] = None,
    hidden_states_sf: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    # Under torch compile _should_use_short_mha is False, so the result is
    # always 9 tensors (4 attention inputs + 5 indexer intermediates, with
    # q_scale as the 9th carried for the FP4 dispatch).
    metadata, mla_layer = _extract_extra_attrs(layer_idx)
    num_tokens = hidden_states.shape[0]
    indexer = mla_layer.mqa.indexer
    q = hidden_states.new_empty([num_tokens, mla_layer.num_heads_tp * mla_layer.qk_head_dim])
    compressed_kv = hidden_states.new_empty([num_tokens, mla_layer.kv_lora_rank])
    k_pe = hidden_states.new_empty([num_tokens, mla_layer.qk_rope_head_dim])
    latent_cache = hidden_states.new_empty(
        [num_tokens, mla_layer.kv_lora_rank + mla_layer.qk_rope_head_dim]
    )
    if indexer is None:
        # DSA "shared" layer: no indexer, mirror forward_dsa_proj's early
        # return of only the 4 base tensors (no indexer intermediates).
        return [q, compressed_kv, k_pe, latent_cache]
    # Indexer intermediates: q_fp8, k_fp8, k_scale, weights, q_scale.
    # Under FP4 q_fp8's trailing dim is head_dim // 2 (two E2M1 codes per
    # byte) and q_scale carries one int32 per (token, head) packing four
    # UE8M0 exponents; under FP8 q_fp8's trailing dim is head_dim and
    # q_scale carries one float32 per (token, head).
    if indexer.use_fp4:
        q_fp8 = hidden_states.new_empty(
            [num_tokens, indexer.n_heads, indexer.head_dim // 2], dtype=torch.int8
        )
        k_fp8 = hidden_states.new_empty([num_tokens, indexer.head_dim // 2], dtype=torch.int8)
        k_scale = hidden_states.new_empty([num_tokens, 1], dtype=torch.int32)
        q_scale = hidden_states.new_empty([num_tokens, indexer.n_heads, 1], dtype=torch.int32)
    else:
        q_fp8 = hidden_states.new_empty(
            [num_tokens, indexer.n_heads, indexer.head_dim], dtype=torch.float8_e4m3fn
        )
        k_fp8 = hidden_states.new_empty([num_tokens, indexer.head_dim], dtype=torch.float8_e4m3fn)
        k_scale = hidden_states.new_empty([num_tokens, 1], dtype=torch.float32)
        q_scale = hidden_states.new_empty([num_tokens, indexer.n_heads, 1], dtype=torch.float32)
    weights = hidden_states.new_empty([num_tokens, indexer.n_heads], dtype=torch.float32)
    return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights, q_scale]


@torch.library.custom_op("trtllm::mla_dsa_attn_inplace", mutates_args=("output",))
def mla_dsa_attn_inplace(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Batch-structure-dependent attention dispatch for DSA MLA.

    indexer_intermediates is [q_fp8, k_fp8, k_scale, weights, q_scale] when
    the indexer ran in Op 1, or [] when short-MHA handled all tokens. The
    trailing q_scale is only consumed by the FP4 dispatch; the FP8 path
    ignores it. Runs sparse_attn_indexer then dispatches context/generation
    attention. This op is excluded from CUDA graph capture.
    """
    metadata, mla_layer = _extract_extra_attrs(layer_idx)
    _forward_dsa_attn(
        mla_layer,
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        indexer_intermediates,
        position_ids,
        metadata,
        output,
    )


@mla_dsa_attn_inplace.register_fake
def _mla_dsa_attn_inplace_fake(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Model the in-place output mutation during fake-tensor propagation."""
