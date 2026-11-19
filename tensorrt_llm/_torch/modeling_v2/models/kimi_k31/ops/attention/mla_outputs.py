# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 MLA layer registry and the MLA output buffer as a custom op.

Each MLA module registers a weak reference under ``model_config.extra_attrs["mla_layers"][layer_idx_str]``; the
engine publishes the step's attention metadata as ``extra_attrs["attention_metadata"]``. The DSA custom ops look both
up by the layer's string index.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.utils import get_model_extra_attrs


def extract_mla_extra_attrs(layer_idx: str):
    """The step's attention metadata and the MLA module registered as ``layer_idx``."""
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"
    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, TrtllmAttentionMetadata), (
        "Metadata must be a TrtllmAttentionMetadata"
    )
    mla_layers = extra_attrs.get("mla_layers", None)
    assert mla_layers is not None, "Attention layer is not registered"
    mla_layer_ref = mla_layers.get(layer_idx, None)
    assert mla_layer_ref is not None, f"Cannot find attention layer for layer {layer_idx}"
    return metadata, mla_layer_ref()


def _create_mla_outputs_impl(hidden_states: torch.Tensor, layer_idx: str) -> torch.Tensor:
    metadata, mla_layer = extract_mla_extra_attrs(layer_idx)
    outputs = mla_layer._create_outputs(hidden_states, metadata)
    if len(outputs) != 1:
        raise RuntimeError("MLA custom ops require exactly one output tensor.")
    return outputs[0]


@torch.library.custom_op("trtllm::k31_create_mla_outputs", mutates_args=())
def _create_mla_outputs(hidden_states: torch.Tensor, layer_idx: str) -> torch.Tensor:
    return _create_mla_outputs_impl(hidden_states, layer_idx)


@_create_mla_outputs.register_fake
def _create_mla_outputs_fake(hidden_states: torch.Tensor, layer_idx: str) -> torch.Tensor:
    return _create_mla_outputs_impl(hidden_states, layer_idx)


def create_mla_outputs(hidden_states: torch.Tensor, layer_idx: str) -> torch.Tensor:
    """The attention output buffer of the MLA module registered as ``layer_idx`` for ``hidden_states``'s rows
    (``K31MLA._create_outputs``), as a custom op: opaque to graph tracing, whose fake gives the buffer's shape."""
    return torch.ops.trtllm.k31_create_mla_outputs(hidden_states, layer_idx)
