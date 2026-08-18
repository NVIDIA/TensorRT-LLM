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
"""Inkling's per-forward backend inputs.

**Inkling is not sparse attention** (see the package docstring). It reuses one
``sparse/`` carrier type because that type is not actually about sparsity:
:class:`SparseBackendForwardArgs` is the registered slot on
``AttentionForwardArgs.sparse_backend_args``, and it is the framework's only
home for a model-specific per-forward input. Without it such an input has
nowhere to go -- ``merge_attention_forward_args`` rejects unknown kwargs
outright, and widening the shared ``AttentionForwardArgs`` dataclass for one
model is exactly what that slot exists to avoid.

Note what is deliberately NOT reused: ``SparseParams`` and
``sparse/registry.py`` for selecting the backend class. That was tried and
reverted. The registry is keyed on ``SparseParams``, which only the
``Attention`` module has; every other consumer -- ``PyTorchModelEngine``, the
KV-cache creator, a dozen vision encoders -- calls
``get_attention_backend(name)`` with a bare string and reads ``.Metadata`` off
the result. Routing selection through the registry left those callers resolving
``TrtllmAttentionMetadata``, so the decode seq lens and page table were never
published and the short-conv runtime was never built; warmup's mixed
context+generation batch then died inside
:meth:`InklingTritonAttention.forward`. The backend NAME is Inkling's identity
outside the module layer, and ``attention_backend/utils.py`` is where that name
is resolved.

The other model-specific backends can live on the registry alone because they
have a user-facing ``SparseAttentionConfig``, from which the engine derives
``SparseParams`` itself. Giving Inkling one would mean a new member of that
discriminated union in ``llmapi/llm_args.py`` -- a public API change requiring
an api-stability review, a regenerated golden manifest and telemetry sign-off,
for a value no user would ever set. That is a separate decision, not a detail
of this file.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from ..params import SparseBackendForwardArgs


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
