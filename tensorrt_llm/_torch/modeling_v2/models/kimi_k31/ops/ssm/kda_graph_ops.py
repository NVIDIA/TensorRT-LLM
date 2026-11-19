# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 KDA core as one break of a breakable CUDA graph.

Under breakable CUDA-graph capture the KDA layer runs its metadata-dependent core (prefill, decode and verify
dispatch) eagerly between captured segments. The break receives only the hidden states, the layer's registry key and
the graph-owned core buffer; it finds the step's attention metadata and the layer through the model's extra
attributes, which the engine refreshes before every replay.
"""

from __future__ import annotations

import weakref

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph
from tensorrt_llm._torch.utils import get_model_extra_attrs

#: The model extra attribute that maps a KDA layer's registry key to a weak reference to the layer.
KDA_LAYERS = "kda_layers"


def kda_layer_and_metadata(layer_idx: str):
    """``(attention metadata, KDA layer)`` of the running step for the layer registered under ``layer_idx``."""
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"
    metadata_ref = extra_attrs.get("attention_metadata")
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, AttentionMetadata), "Metadata must be AttentionMetadata"
    kda_layers = extra_attrs.get(KDA_LAYERS)
    assert kda_layers is not None, "KDA layer registry is not set"
    kda_layer_ref = kda_layers.get(layer_idx)
    assert kda_layer_ref is not None, f"Cannot find KDA layer for layer {layer_idx}"
    kda_layer = kda_layer_ref()
    assert kda_layer is not None, f"The KDA layer registered for layer {layer_idx} no longer exists"
    return metadata, kda_layer


def kda_core_inplace(hidden_states: torch.Tensor, layer_idx: str, output: torch.Tensor) -> None:
    """Run the metadata-dependent KDA core and write it into ``output``.

    Rows past the live batch (prefill-graph padding) are zeroed: the buffer is graph-pool memory another
    capture may have left non-finite, and the rows reach the MoE router after ``o_proj``.
    """
    metadata, kda_layer = kda_layer_and_metadata(layer_idx)
    kda_layer._forward_impl(hidden_states, metadata, output=output)
    if output.shape[0] > metadata.num_tokens:
        output[metadata.num_tokens :].zero_()


maybe_bcg_kda_core_inplace = eager_on_graph(kda_core_inplace)


def register_kda_layer(extra_attrs: dict, layer, layer_idx: int) -> str:
    """Register ``layer`` in the model extra attributes' KDA layer registry; returns its key (``"<layer_idx>"``,
    suffixed ``"_<n>"`` when another stack already registered that index)."""
    kda_layers = extra_attrs.setdefault(KDA_LAYERS, {})
    key = str(layer_idx)
    suffix = 0
    while key in kda_layers:
        key = f"{layer_idx}_{suffix}"
        suffix += 1
    kda_layers[key] = weakref.ref(layer)
    return key
