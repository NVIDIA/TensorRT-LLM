# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-owned DSA attention dispatch for eager and breakable graphs."""

from typing import List, Optional

import torch

from tensorrt_llm._torch.attention.attention import extract_extra_attrs
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph

from ...._native import namespace_of
from ._dsa import _forward_dsa_attn


def _extract_extra_attrs(layer_idx: str):
    return extract_extra_attrs(layer_idx, "mla")


@torch.library.custom_op(
    f"{namespace_of(__package__)}::mla_dsa_attn_inplace", mutates_args=("output",)
)
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


maybe_bcg_mla_dsa_attn_inplace = eager_on_graph(mla_dsa_attn_inplace)
