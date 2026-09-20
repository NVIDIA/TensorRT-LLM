# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QSA custom operations for breakable CUDA graph capture."""

from typing import Optional

import torch

from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph

from ...interface import PredefinedAttentionMask
from .metadata import QSAAttentionMetadata
from .module import QSASparseHooks


def _extract_qsa_extra_attrs(layer_idx: str):
    from tensorrt_llm._torch.attention.attention import Attention, extract_extra_attrs

    metadata, attention = extract_extra_attrs(layer_idx, "attn")
    if not isinstance(metadata, QSAAttentionMetadata):
        raise TypeError("QSA breakable CUDA graph received incompatible metadata")
    if not isinstance(attention, Attention):
        raise TypeError("QSA breakable CUDA graph requires an Attention layer")
    if not isinstance(attention.sparse_attn_hooks, QSASparseHooks):
        raise TypeError("QSA breakable CUDA graph requires QSA sparse hooks")
    return metadata, attention


@torch.library.custom_op("trtllm::qsa_attn_inplace", mutates_args=("output",))
def qsa_attn_inplace(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    q_index: torch.Tensor,
    token_k: torch.Tensor,
    position_coordinates: torch.Tensor,
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    output_gate: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Run live QSA dispatch and write the fixed physical output in place."""
    metadata, attention = _extract_qsa_extra_attrs(layer_idx)
    output.zero_()
    sparse_output = attention.sparse_attn_hooks.forward(
        attention,
        q,
        k,
        v,
        metadata,
        PredefinedAttentionMask.CAUSAL,
        None,
        None,
        None,
        None,
        None,
        0,
        False,
        output_gate,
        qsa_index_projection=(q_index, token_k, position_coordinates),
    )
    if sparse_output is not None:
        output[: sparse_output.shape[0]].copy_(sparse_output)
        return

    dense_output, output_sf = attention._attn_impl(
        q,
        k,
        v,
        metadata,
        PredefinedAttentionMask.CAUSAL,
        mrope_rotary_cos_sin,
        mrope_position_deltas,
        None,
        None,
        output=output,
    )
    if output_sf is not None:
        raise RuntimeError("QSA output gating requires an unquantized attention output")
    if output_gate is not None:
        dense_output = attention.apply_output_gate(
            dense_output, output_gate[: dense_output.shape[0]]
        )
    output[: dense_output.shape[0]].copy_(dense_output)


@qsa_attn_inplace.register_fake
def _qsa_attn_inplace_fake(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    q_index: torch.Tensor,
    token_k: torch.Tensor,
    position_coordinates: torch.Tensor,
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    output_gate: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Model the in-place output mutation during fake-tensor propagation."""


maybe_bcg_qsa_attn_inplace = eager_on_graph(qsa_attn_inplace)


__all__ = ["maybe_bcg_qsa_attn_inplace", "qsa_attn_inplace"]
