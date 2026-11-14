# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Which post-MoE boundaries the deferred-norm hop serves.

The scopes (the deferred-norm numerical variant):

* ``all``: every boundary ``L -> L+1`` whose next layer is a target attention
  layer (the hop's unrestricted scope).
* ``shared``: only boundaries whose next layer is a shared-indexer layer (it
  reuses the previous full layer's top-k and has no indexer of its own). On a
  boundary into a full-indexer layer the hop is not engaged, so the layer
  keeps the post-MoE hop and the next layer receives its normalized tensor and
  runs its native or absorbed-query projection path; the indexer
  ``wk | weights_proj`` inputs of full layers then carry no deferred-norm
  rounding change.

The served scope is ``shared`` (``selected_scope``).

A layer's indexer type is the one its attention module is built with:
``sparse_attention_config.to_sparse_params(pretrained_config,
layer_idx).is_full_indexer_layer`` (``MLA`` builds an ``Indexer`` only on
full layers).
"""

import sys
from typing import Optional, Tuple

#: How the decline reasons name the scope (``<label>=<scope>: ...``).
SCOPE_LABEL = "TRTLLM_STAIRCASE_GLM_LL_DEFERRED_NORM_SCOPE"
# Scope names.
SCOPES = ("all", "shared")
# The scope that serves every boundary, without looking at the next layer.
UNRESTRICTED_SCOPE = SCOPES[0]
# The scope of this target.
DEFAULT_SCOPE = "shared"
# Calling frames searched for the decoder layer that constructs a hop.
_MAX_FRAMES = 8


def selected_scope() -> str:
    """The served scope, ``DEFAULT_SCOPE``."""
    return DEFAULT_SCOPE


def is_full_indexer_layer(model_config, layer_idx: int) -> Optional[bool]:
    """Whether target layer ``layer_idx`` owns an indexer (None: unknown).

    Same call as ``MLA.__init__`` makes to decide whether the layer builds its
    ``Indexer``.
    """
    sparse = getattr(model_config, "sparse_attention_config", None)
    if sparse is None or not hasattr(sparse, "to_sparse_params"):
        return None
    params = sparse.to_sparse_params(
        pretrained_config=getattr(model_config, "pretrained_config", None), layer_idx=layer_idx
    )
    full = getattr(params, "is_full_indexer_layer", None)
    return None if full is None else bool(full)


def constructing_layer(post_moe_hop):
    """The decoder layer whose constructor is building a deferred_norm hop for ``post_moe_hop``.

    Searches the calling frames for a ``self`` that holds ``post_moe_hop`` as its
    ``post_moe_hop`` and has an integer ``layer_idx`` and a ``model_config``:
    the layer assigns its post_moe_hop before it asks the deferred_norm factory for the
    deferred hop. Returns ``None`` if no such frame is found within
    ``_MAX_FRAMES`` callers.
    """
    frame = sys._getframe(1)
    try:
        for _ in range(_MAX_FRAMES):
            if frame is None:
                return None
            owner = frame.f_locals.get("self")
            if (
                owner is not None
                and getattr(owner, "post_moe_hop", None) is post_moe_hop
                and isinstance(getattr(owner, "layer_idx", None), int)
                and getattr(owner, "model_config", None) is not None
            ):
                return owner
            frame = frame.f_back
        return None
    finally:
        del frame


def next_layer_kind(
    post_moe_hop, next_layer_full_indexer: Optional[bool] = None
) -> Tuple[Optional[bool], Optional[int], str]:
    """``(full, next_layer_idx, how)`` of the layer after the hop's MoE layer.

    ``next_layer_full_indexer`` given: that value (``how = "argument"``).
    Otherwise it is resolved from the constructing decoder layer
    (:func:`constructing_layer`). ``full`` is ``None`` when it cannot be
    resolved; ``how`` then says why.
    """
    if next_layer_full_indexer is not None:
        return bool(next_layer_full_indexer), None, "argument"
    layer = constructing_layer(post_moe_hop)
    if layer is None:
        return None, None, "no constructing decoder layer found"
    nxt = layer.layer_idx + 1
    full = is_full_indexer_layer(layer.model_config, nxt)
    if full is None:
        return None, nxt, "the model config has no sparse-attention indexer layout"
    return full, nxt, "model config"


def decline_reason(
    scope: str, full: Optional[bool], next_layer_idx: Optional[int], how: str
) -> Optional[str]:
    """Why ``scope`` keeps post_moe_hop's hop on this boundary (None: deferred_norm may serve it)."""
    if scope == "all":
        return None
    where = f"layer {next_layer_idx}" if next_layer_idx is not None else "the next layer"
    if full is None:
        return f"{SCOPE_LABEL}={scope}: indexer type of {where} unknown ({how}); post_moe_hop hop"
    if full:
        return f"{SCOPE_LABEL}={scope}: {where} is a full-indexer layer; post_moe_hop hop"
    return None
