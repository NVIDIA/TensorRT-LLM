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

**Inkling is not sparse attention** (see the package docstring). It reuses two
``sparse/`` seams because neither is about sparsity: :class:`SparseParams`, the
key ``sparse/registry.py`` dispatches on, and :class:`SparseBackendForwardArgs`,
the slot on ``AttentionForwardArgs`` that exists so a model-specific forward
input does not require widening the shared dataclass.

**Why the config exists.** The registry dispatches on ``SparseParams``, derived
from ``ModelConfig.sparse_attention_config``. Consumers holding no
``ModelConfig`` -- ``PyTorchModelEngine``, the KV-cache creator, the vision
encoders -- instead call ``get_attention_backend(name)`` and read ``.Metadata``.
Left ``None`` they resolved ``TrtllmAttentionMetadata``, and warmup's first mixed
batch died inside the backend. Populating it makes the registry the single
selection path.

**Why it is not in the user-facing union.** ``ModelConfig`` never validates that
field against the discriminated union, so this subclasses
``BaseSparseAttentionConfig`` without joining it: no api-stability snapshot,
golden manifest or telemetry review, and no user knob with one legal value.
Injected by ``ModelConfig.from_pretrained`` keyed on architecture, the seam
DeepSeek-V4 uses.

**The cost.** The field is non-``None`` for a dense model, so every reader
treating it as "is sparse" must know about Inkling.
``get_sparse_attn_kv_cache_manager`` is handled; the ``Using sparse attention:``
log line is tolerated, since silencing it needs a model-specific branch in a
shared file. A reader added later fails by taking a wrong branch rather than
erroring -- the standing liability of this route versus one ``elif`` in
``get_attention_backend``.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from tensorrt_llm.llmapi.llm_args import BaseSparseAttentionConfig

from ..params import SparseBackendForwardArgs, SparseParams


class InklingSparseParams(SparseParams):
    """Selection key for Inkling's backend and cache manager in
    ``sparse/registry.py``. ``algorithm`` is the only field the registry reads;
    the kernels' per-layer scalars (``sm_scale``, ``rel_extent``,
    ``window_left``) are set on the backend instance instead.
    """

    algorithm = "inkling"


class InklingSparseAttentionConfig(BaseSparseAttentionConfig):
    """Architecture-derived config that carries :class:`InklingSparseParams`.

    Subclasses the interface but is **not** in the ``SparseAttentionConfig``
    union -- separate properties, which an earlier version conflated by
    duck-typing ``BaseSparseAttentionConfig`` and asserting it was *not* a
    subclass. That cost an end-to-end run when the engine called
    ``to_sparse_metadata_params``, a base method the imitation lacked.
    Inheriting makes that class of failure impossible; the union is explicitly
    enumerated, so only "not in the union" needs asserting.

    ``to_sparse_params`` mirrors the union members' signature and ignores both
    arguments: the selection key is the same for every layer.
    """

    algorithm: Literal["inkling"] = "inkling"

    def to_sparse_params(self, pretrained_config=None, layer_idx=None) -> InklingSparseParams:
        return InklingSparseParams()


@dataclass(kw_only=True, slots=True)
class InklingBackendForwardArgs(SparseBackendForwardArgs):
    """Per-forward inputs ``InklingAttention`` hands to its backend, riding
    ``AttentionForwardArgs.sparse_backend_args`` so the backend keeps the
    standard :meth:`AttentionBackend.forward` signature.

    ``rel_logits``
        ``[num_query_tokens, local_heads, rel_extent]`` fp32 additive bias, added
        inside the Triton kernels. *Not* the shared
        ``AttentionForwardArgs.relative_attention_bias``: that one is T5's,
        ``[num_heads, num_buckets]`` broadcast across the batch, i.e. a
        position-indexed table with no per-query-token axis. Inkling's is
        projected from the hidden states, so it is content-dependent. Using that
        field would also disable the fused C++ path.

    ``allow_mixed``
        Certifies the short-conv state pool is active, which is what makes a
        mixed context+generation batch legal -- the stateless path convolves
        across the boundary of the packed batch and corrupts generation rows.
        Defaults to ``False`` so callers opt in explicitly.
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
