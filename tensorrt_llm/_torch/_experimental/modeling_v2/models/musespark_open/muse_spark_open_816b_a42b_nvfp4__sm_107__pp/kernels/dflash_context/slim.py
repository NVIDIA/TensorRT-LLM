# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The last stage's drafter, reduced to what writing the drafter context reads. In memory only.

A context server never runs the drafter forward (``worker``), so of the loaded DFlash drafter
the context write reads only:

* ``fc`` and ``hidden_norm`` (``project_target_hidden``);
* the fused K/V projection of every drafter layer, its stacked k_norm weights and its bias,
  which ``precompute_context_kv`` builds once from the layers' QKV weights
  (``_build_fused_kv_buffers``) into tensors of their own;
* the RoPE cos/sin table, which the drafter takes from layer 0's rotary embedding (the RoPE
  owner) and caches as FP32 (``_get_cos_sin_cache``).

``slim`` builds the lazily built parts first, then removes the weights of every drafter
layer (keeping layer 0's rotary embedding), of the drafter's final norm and of the DFlash 2
block convolutions and candidate selector: the attention, MLP and selector weights a draft
step reads. The target's embedding and LM head, which the drafter borrows, are not drafter
modules and stay. The checkpoint files are not touched.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torch import nn

#: The drafter modules ``slim`` keeps, as attribute names of the drafter.
KEPT = ("fc", "hidden_norm")
#: Drafter modules a draft step alone reads, besides the layers and the final norm.
DRAFT_ONLY = ("attention_convs", "mlp_convs", "candidate_selector")


def _bytes(modules: Iterable[nn.Module]) -> int:
    total = 0
    seen = set()
    for module in modules:
        for tensor in list(module._parameters.values()) + list(module._buffers.values()):
            if tensor is None or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
    return total


def removable(draft_model: nn.Module) -> List[Tuple[nn.Module, Optional[List[nn.Module]]]]:
    """``(module, kept submodules)`` of every drafter module ``slim`` removes the weights of."""
    layers = draft_model.model.layers
    rope_owner = getattr(layers[0].self_attn, "rotary_emb", None) if len(layers) else None
    out: List[Tuple[nn.Module, Optional[List[nn.Module]]]] = []
    for index, layer in enumerate(layers):
        keep = [rope_owner] if index == 0 and isinstance(rope_owner, nn.Module) else None
        out.append((layer, keep))
    norm = getattr(draft_model.model, "norm", None)
    if isinstance(norm, nn.Module):
        out.append((norm, None))
    for name in DRAFT_ONLY:
        module = getattr(draft_model, name, None)
        if isinstance(module, nn.Module):
            out.append((module, None))
    return out


def slim(
    draft_model: nn.Module,
    remove_weights: Optional[Callable[..., None]] = None,
    iter_modules: Optional[Callable[..., Iterable[nn.Module]]] = None,
) -> int:
    """Reduce the loaded drafter to the context write's tensors; returns the bytes released.

    ``remove_weights`` / ``iter_modules`` are the model helpers of the same names
    (``models.modeling_utils``, imported when None).

    Raises:
        ValueError: for a drafter without ``fc``, ``hidden_norm`` or the fused K/V build.
    """
    if remove_weights is None or iter_modules is None:
        from tensorrt_llm._torch.models import modeling_utils

        remove_weights = remove_weights or modeling_utils.remove_weights
        iter_modules = iter_modules or modeling_utils.iter_modules
    missing = [name for name in KEPT if not isinstance(getattr(draft_model, name, None), nn.Module)]
    if missing or not hasattr(draft_model, "_build_fused_kv_buffers"):
        raise ValueError(
            f"the drafter {type(draft_model).__name__} has no "
            f"{', '.join(missing) or '_build_fused_kv_buffers'}; its context write cannot be kept"
        )
    # The lazily built parts read the layer weights: build them before the weights go.
    draft_model._build_fused_kv_buffers()
    draft_model._get_cos_sin_cache()
    released = 0
    for module, keep in removable(draft_model):
        released += _bytes(iter_modules(module, keep))
        remove_weights(module, keep)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return released
