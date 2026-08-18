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
"""Inkling's selection key, its per-forward backend inputs, and the architecture
config that carries the former.

**Inkling is not sparse attention** (see the package docstring). It reuses three
``sparse/`` mechanisms because none of them is actually about sparsity -- they are
the framework's only registered seams for "select a model-specific backend" and
"pass a model-specific per-forward input":

* :class:`SparseParams` is a plain class whose sole contract is ``algorithm:
  str``. It is the key ``sparse/registry.py`` dispatches on, which is how every
  other model-specific backend and cache manager is selected.
* :class:`SparseBackendForwardArgs` is the registered slot on
  ``AttentionForwardArgs.sparse_backend_args``. Without it a model-specific
  forward input has nowhere to go: ``merge_attention_forward_args`` rejects
  unknown kwargs, and widening the shared dataclass for one model is what that
  slot exists to avoid.
* :class:`InklingSparseAttentionConfig` is what makes the first bullet reachable
  from outside the ``Attention`` module -- see below.

Why the config exists at all
----------------------------
``sparse/registry.py`` dispatches on ``SparseParams``, which the module layer
derives from ``ModelConfig.sparse_attention_config``. Consumers that hold no
``ModelConfig`` -- ``PyTorchModelEngine``, the KV-cache creator, a dozen vision
encoders -- instead ask ``get_attention_backend(name)`` and read ``.Metadata``
off the result. With the config left ``None``, those consumers resolved
``TrtllmAttentionMetadata``: the decode seq lens and page table were never
published and the short-conv runtime never built, so warmup's first mixed batch
died inside the backend. Populating the config is what makes the registry the
single selection path for *all* consumers, and it is why
``get_attention_backend`` needs no model-specific branch.

Why it is NOT a member of the user-facing union
-----------------------------------------------
``ModelConfig`` is a plain dataclass; ``sparse_attention_config`` is only
*annotated* ``Optional[SparseAttentionConfig]`` and never validated against the
discriminated union in ``llmapi/llm_args.py``. So this is a standalone
``BaseModel``: it satisfies every consumer (including the ``.model_dump()`` in
``modules/attention.py``) while adding nothing to the user-facing schema -- no
api-stability snapshot change, no golden-manifest regeneration, no telemetry
review. Inkling has exactly one correct backend and nothing for a user to
configure, so a user-settable knob would be a knob with one legal value.

It is injected by ``ModelConfig.from_pretrained`` keyed on the checkpoint
architecture, before the instance is frozen -- the same seam DeepSeek-V4 uses.

The cost, stated plainly
------------------------
``sparse_attention_config`` becomes non-``None`` for a model that is not sparse,
so everything reading that field as "this model is sparse" now has to know about
Inkling. Two such readers exist today:

* ``get_sparse_attn_kv_cache_manager`` -- **handled**; without an entry there
  ``get_kv_cache_manager_cls`` raises ``Unsupported sparse attention algorithm``
  at engine startup.
* the ``Using sparse attention: ...`` line in ``modules/attention.py`` --
  **tolerated, not handled**. It now prints for a dense model. Silencing it would
  mean a model-specific branch in a file shared by every attention module, which
  is the kind of edit this arrangement is supposed to remove, so the misleading
  line is accepted as the visible cost of the field saying something untrue.

A reader added later needs the same audit, and its failure mode is a wrong
branch rather than an error. That is the standing liability of this approach; the
alternative is a single ``elif`` in ``get_attention_backend``, which cannot be
reached by accident.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from pydantic import BaseModel

from ..params import SparseBackendForwardArgs, SparseParams


class InklingSparseParams(SparseParams):
    """Selection key for Inkling's backend and cache manager in ``sparse/registry.py``.

    ``algorithm`` is the only field the registry reads. Inkling has no tunables
    at this layer -- the three per-layer scalars the kernels need (``sm_scale``,
    ``rel_extent``, ``window_left``) differ per layer and are set on the backend
    instance by ``InklingAttention.__init__``.
    """

    algorithm = "inkling"


class InklingSparseAttentionConfig(BaseModel):
    """Architecture-derived config that carries :class:`InklingSparseParams`.

    Deliberately a standalone ``BaseModel`` rather than a member of the
    ``SparseAttentionConfig`` union in ``llmapi/llm_args.py``: it must never
    appear in the user-facing schema (see the module docstring). ``BaseModel`` is
    the base rather than a plain class because ``modules/attention.py`` calls
    ``.model_dump()`` on whatever sits in that field.

    ``to_sparse_params`` mirrors the union members' signature -- the module layer
    calls it with ``pretrained_config`` and ``layer_idx`` -- and ignores both,
    since the selection key is the same for every layer.
    """

    algorithm: Literal["inkling"] = "inkling"

    def to_sparse_params(self, pretrained_config=None, layer_idx=None) -> InklingSparseParams:
        return InklingSparseParams()


@dataclass(kw_only=True, slots=True)
class InklingBackendForwardArgs(SparseBackendForwardArgs):
    """Per-forward inputs ``InklingAttention`` hands to its backend.

    Rides ``AttentionForwardArgs.sparse_backend_args`` so the backend keeps the
    standard :meth:`AttentionBackend.forward` signature. Both fields are inputs
    no other model has, which is why they cannot live on the shared
    ``AttentionForwardArgs``:

    ``rel_logits``
        ``[num_query_tokens, local_heads, rel_extent]`` fp32 additive bias,
        gathered and added inside the Triton kernels. Note this is *not* the
        shared ``AttentionForwardArgs.relative_attention_bias``: that field is
        T5's, shaped ``[num_heads, num_buckets]`` and broadcast across the batch
        (see ``cpp/tensorrt_llm/thop/attentionOp.cpp``), i.e. a static
        position-indexed table. Inkling's bias is *content*-dependent -- it is
        projected from the hidden states per query token -- so it carries an
        extra leading axis that field has no room for. Passing it there would
        also force the unfused C++ attention path, which materializes the full
        ``[batch, heads, q_len, k_len]`` score matrix.

    ``allow_mixed``
        Certifies that the per-request short-conv state pool is active for this
        forward, which is what makes a mixed context+generation batch legal: the
        stateless short-conv path convolves across the context/generation
        boundary of the packed batch and would silently corrupt the generation
        rows. Defaults to ``False`` so a caller must opt in explicitly.
    """

    rel_logits: torch.Tensor
    allow_mixed: bool = False


def inkling_forward_args(
    rel_logits: torch.Tensor,
    *,
    allow_mixed: bool,
    output: Optional[torch.Tensor] = None,
):
    """Build the ``AttentionForwardArgs`` carrying Inkling's backend inputs."""
    from ...interface import AttentionForwardArgs

    return AttentionForwardArgs(
        output=output,
        sparse_backend_args=InklingBackendForwardArgs(
            rel_logits=rel_logits, allow_mixed=allow_mixed
        ),
    )
